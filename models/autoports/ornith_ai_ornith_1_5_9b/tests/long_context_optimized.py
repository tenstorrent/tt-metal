# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Advertised-context probe: complete TT prefill, HF windows and final traced token.

Full attention uses HF-projected K/V for ALL preceding tokens. HF decoder.forward
then evaluates selected query windows against that exact history. Linear attention
runs the actual HF decoder over every token with its recurrent cache.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
)

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import MODEL, REVISION, ROOT, pcc, weights
from models.autoports.ornith_ai_ornith_1_5_9b.tests.validate_optimized import (
    buffers,
    clear,
    device_only,
    restore,
    upload,
)
from models.autoports.ornith_ai_ornith_1_5_9b.tt.optimized_decoder import OptimizedDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--length", type=int, default=262144)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(91)
    c = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True).text_config
    c._attn_implementation = "sdpa"
    n = args.length
    assert n <= c.max_position_embeddings
    sd = weights(args.layer)
    ref = Qwen3_5DecoderLayer(c, args.layer).bfloat16().eval()
    ref.load_state_dict(sd)
    rope = Qwen3_5TextRotaryEmbedding(c)
    full = c.layer_types[args.layer] == "full_attention"
    x = torch.randn(1, n, 4096, dtype=torch.bfloat16) * 0.1
    spans = [(0, 32), (n // 2, 32), (n - 64, 32), (n - 32, 27), (n - 5, 4), (n - 1, 1)]
    expected = {}
    begin = time.time()
    with torch.no_grad():
        if full:
            keys = torch.empty((1, 4, n, 256), dtype=torch.bfloat16)
            vals = torch.empty_like(keys)
            for start in range(0, n, 1024):
                end = min(start + 1024, n)
                z = ref.input_layernorm(x[:, start:end])
                k = ref.self_attn.k_norm(ref.self_attn.k_proj(z).view(1, end - start, 4, 256)).transpose(1, 2)
                v = ref.self_attn.v_proj(z).view(1, end - start, 4, 256).transpose(1, 2)
                cos, sin = rope(z, torch.arange(start, end)[None])
                _, k = apply_rotary_pos_emb(k, k, cos, sin)
                keys[:, :, start:end] = k
                vals[:, :, start:end] = v
            for start, count in spans:
                cache = DynamicCache(config=c)
                if start:
                    cache.update(keys[:, :, :start], vals[:, :, :start], args.layer)
                pos = torch.arange(start, start + count)[None]
                mask = torch.where(torch.arange(start + count)[None, :] <= pos.T, 0.0, float("-inf"))[None, None]
                xx = x[:, start : start + count]
                expected[start] = ref(xx, position_embeddings=rope(xx, pos), attention_mask=mask, past_key_values=cache)
        else:
            cache = DynamicCache(config=c)
            for start in range(0, n, 1024):
                end = min(start + 1024, n)
                xx = x[:, start:end]
                out = ref(xx, position_embeddings=rope(xx, torch.arange(start, end)[None]), past_key_values=cache)
                for a, count in spans:
                    if start <= a and a + count <= end:
                        expected[a] = out[:, a - start : a - start + count].clone()
                if start % 32768 == 0:
                    print("HF_PREFIX", end, flush=True)
    print("HF_REFERENCE_READY", time.time() - begin, flush=True)
    records = []
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        layer = OptimizedDecoder.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh)
        state = layer.allocate_state(context=n)
        table = upload(torch.randperm(n // 32, dtype=torch.int32)[None], mesh, True)
        pos = upload(torch.tensor([n - 1], dtype=torch.int32), mesh, True)
        dec = upload(x[:, n - 1 :][None], mesh)
        for length in [n - 5, n - 1, n]:
            clear(state)
            tx = upload(x[:, :length][None], mesh)
            with device_only():
                out = layer.prefill_forward(tx, state=state, page_table=table)
            ttnn.synchronize_device(mesh)
            for start, count in spans:
                if start + count > length:
                    continue
                actual = ttnn.to_torch(out[:, :, start : start + count, :]).reshape(1, count, 4096)
                score = pcc(expected[start], actual)
                delta = pcc(expected[start] - x[:, start : start + count], actual - x[:, start : start + count])
                record = {
                    "prefill_length": length,
                    "window_start": start,
                    "window_length": count,
                    "pcc": score,
                    "contribution_pcc": delta,
                }
                print(json.dumps(record), flush=True)
                records.append(record)
                assert score >= 0.995 and delta >= 0.995
            del out, tx
            if length == n - 1:
                snapshots = [(b, ttnn.clone(b)) for b in buffers(state)]
                warm = layer.decode_forward(dec, state=state, page_table=table, current_position=pos)
                restore(snapshots)
                del warm
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                with device_only():
                    outd = layer.decode_forward(dec, state=state, page_table=table, current_position=pos)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                from ttnn.tools import trace_allocation_tracker

                trace_allocation_tracker.acknowledge_corruptible(outd)
                previous = None
                for _ in range(3):
                    restore(snapshots)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    actual = ttnn.to_torch(outd).reshape(1, 1, 4096)
                    if previous is not None:
                        assert torch.equal(previous, actual)
                    previous = actual
                score = pcc(expected[n - 1], actual)
                delta = pcc(expected[n - 1] - x[:, n - 1 :], actual - x[:, n - 1 :])
                assert score >= 0.995 and delta >= 0.995
                records.append(
                    {
                        "decode_context": n,
                        "position": n - 1,
                        "traced_pcc": score,
                        "contribution_pcc": delta,
                        "replays": 3,
                        "deterministic": True,
                    }
                )
                print("LONG_DECODE_PCC", score, flush=True)
                ttnn.release_trace(mesh, trace)
                del outd, snapshots
        Path(args.output).write_text(
            json.dumps(
                {
                    "model": MODEL,
                    "revision": REVISION,
                    "layer": args.layer,
                    "kind": c.layer_types[args.layer],
                    "advertised_context": c.max_position_embeddings,
                    "tested_prefill": n,
                    "tested_decode": n,
                    "records": records,
                    "reference": "HF complete linear decoder; HF complete K/V history plus sampled full-attention decoder windows",
                    "implementation_sha256": hashlib.sha256(
                        (ROOT / "tt/optimized_decoder.py").read_bytes()
                    ).hexdigest(),
                },
                indent=2,
            )
            + "\n"
        )
        print("LONG_CONTEXT_PASS", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
