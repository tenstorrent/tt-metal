# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real-weight 128-input/128-step teacher-forced decoder state/trace stress."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.evidence import decoder_class, decoder_kwargs, provenance
from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import MODEL, REVISION, load_layer, pcc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--compact-io", action="store_true")
    args = ap.parse_args()
    prov = provenance()
    torch.set_num_threads(8)
    torch.manual_seed(1729)
    c = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True).text_config
    c._attn_implementation = "sdpa"
    weights = load_layer(args.layer)
    ref = Qwen3_5DecoderLayer(c, args.layer).eval().to(torch.bfloat16)
    ref.load_state_dict(weights)
    rot = Qwen3_5TextRotaryEmbedding(c)
    cache = DynamicCache(config=c)
    prefix = torch.randn(args.batch, 128, 5120, dtype=torch.bfloat16) * 0.1
    tokens = torch.randn(args.steps, args.batch, 1, 5120, dtype=torch.bfloat16) * 0.1
    with torch.no_grad():
        expected_pre = ref(
            prefix, position_embeddings=rot(prefix, torch.arange(128).expand(args.batch, -1)), past_key_values=cache
        )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    tid = None
    try:
        layer = decoder_class().from_state_dict(
            weights, hf_config=c, layer_idx=args.layer, mesh_device=mesh, **decoder_kwargs()
        )
        read = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        linear = layer.kind == "linear_attention"
        pages = (128 + args.steps + 31) // 32
        state = (
            layer.create_recurrent_state(args.batch)
            if linear
            else layer.create_paged_state(torch.randperm(args.batch * pages).reshape(args.batch, pages))
        )
        prekw = (
            {}
            if linear
            else {
                "rope": tuple(
                    layer._upload(t.unsqueeze(1)) for t in rot(prefix, torch.arange(128).expand(args.batch, -1))
                )
            }
        )
        pre = read(layer.prefill_forward(layer._upload(prefix), state=state, **prekw))
        buffers = [state.conv, state.recurrent] if linear else [state.key, state.value]
        saved = [ttnn.from_torch(read(t), dtype=t.dtype, layout=t.layout) for t in buffers]
        decode_tokens = tokens.reshape(args.steps, 1, args.batch, 5120) if args.compact_io else tokens
        td = layer._upload(decode_tokens[0])
        hosts = [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for t in decode_tokens]
        kw = {}
        positions = []
        ropes = []
        if not linear:
            for i in range(args.steps):
                pos = torch.full((args.batch, 1), 128 + i, dtype=torch.long)
                positions.append(ttnn.from_torch(pos.reshape(-1).int(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT))
                ropes.append(
                    tuple(
                        ttnn.from_torch(t.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                        for t in rot(tokens[i], pos)
                    )
                )
            kw = {
                "current_pos": ttnn.from_torch(
                    torch.full((args.batch,), 128, dtype=torch.int32),
                    device=mesh,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                "rope": tuple(
                    layer._upload(t.unsqueeze(1))
                    for t in rot(tokens[0], torch.full((args.batch, 1), 128, dtype=torch.long))
                ),
            }
        layer.decode_forward(td, state=state, **kw)
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = layer.decode_forward(td, state=state, **kw)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        for h, d in zip(saved, buffers):
            ttnn.copy_host_to_device_tensor(h, d)
        rows = []
        for i in range(args.steps):
            ttnn.copy_host_to_device_tensor(hosts[i], td)
            if not linear:
                ttnn.copy_host_to_device_tensor(positions[i], kw["current_pos"])
                for h, d in zip(ropes[i], kw["rope"]):
                    ttnn.copy_host_to_device_tensor(h, d)
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            elapsed = (time.perf_counter() - start) * 1000
            actual = read(out).reshape(args.batch, 1, 5120)
            with torch.no_grad():
                expected = ref(
                    tokens[i],
                    position_embeddings=rot(tokens[i], torch.full((args.batch, 1), 128 + i, dtype=torch.long)),
                    past_key_values=cache,
                )
            per_user = [pcc(expected[b], actual[b]) for b in range(args.batch)]
            rows.append({"step": i, "position": 128 + i, "per_user_pcc": per_user, "trace_wall_ms": elapsed})
        result = {
            "provenance": prov,
            "layer": args.layer,
            "batch": args.batch,
            "prefill_length": 128,
            "steps": args.steps,
            "prefill_pcc": pcc(expected_pre, pre),
            "decode_traced": True,
            "compact_io": args.compact_io,
            "minimum_per_user_pcc": min(v for r in rows for v in r["per_user_pcc"]),
            "trace_wall_median_ms": statistics.median(r["trace_wall_ms"] for r in rows),
            "rows": rows,
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        assert result["prefill_pcc"] >= 0.995 and result["minimum_per_user_pcc"] >= 0.995, result[
            "minimum_per_user_pcc"
        ]
        print("SEQUENCE_PASS", result["minimum_per_user_pcc"], result["trace_wall_median_ms"], flush=True)
    finally:
        if tid is not None:
            ttnn.release_trace(mesh, tid)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
