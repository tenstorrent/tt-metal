# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prepare eager prefill variants first; retain one decode trace across requests."""

import argparse
import json
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
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(72)
    c = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True).text_config
    c._attn_implementation = "sdpa"
    weights = load_layer(args.layer)
    hf = Qwen3_5DecoderLayer(c, args.layer).eval().to(torch.bfloat16)
    hf.load_state_dict(weights)
    rotary = Qwen3_5TextRotaryEmbedding(c)
    requests = []
    for length, start in [(31, 0), (32, 0), (33, 0), (513, 32), (513, 33), (33, 0)]:
        x = torch.randn(1, length, 5120, dtype=torch.bfloat16) * 0.1
        d = torch.randn(1, 1, 5120, dtype=torch.bfloat16) * 0.1
        with torch.no_grad():
            cache = DynamicCache(config=c)
            hf(x, position_embeddings=rotary(x, torch.arange(length).reshape(1, -1)), past_key_values=cache)
            expected = hf(d, position_embeddings=rotary(d, torch.tensor([[length]])), past_key_values=cache)
        requests.append(
            {
                "x": x,
                "d": d,
                "expected": expected,
                "start": start,
                "length": length,
                "pages": torch.randperm(64, dtype=torch.int32).reshape(1, 64),
            }
        )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        layer = decoder_class().from_state_dict(
            weights, **decoder_kwargs(), hf_config=c, layer_idx=args.layer, mesh_device=mesh
        )
        linear = layer.kind == "linear_attention"
        state = layer.create_recurrent_state() if linear else layer.create_paged_state(requests[0]["pages"])
        buffers = [state.conv, state.recurrent] if linear else [state.key, state.value]
        zeros = [ttnn.from_torch(torch.zeros(tuple(t.shape)), dtype=t.dtype, layout=t.layout) for t in buffers]
        td = layer._upload(requests[0]["d"])
        dp = (
            None
            if linear
            else ttnn.from_torch(
                torch.tensor([31], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
            )
        )
        dr = (
            None
            if linear
            else tuple(layer._upload(t.unsqueeze(1)) for t in rotary(requests[0]["d"], torch.tensor([[31]])))
        )
        dk = {} if linear else {"current_pos": dp, "rope": dr}
        for r in requests:
            r["tx"] = layer._upload(r["x"])
            r["rope"] = (
                None
                if linear
                else tuple(
                    layer._upload(t.unsqueeze(1)) for t in rotary(r["x"], torch.arange(r["length"]).reshape(1, -1))
                )
            )
            r["hd"] = ttnn.from_torch(r["d"], dtype=td.dtype, layout=td.layout)
            if not linear:
                r["hp"] = ttnn.from_torch(
                    torch.tensor([r["length"]], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                r["hr"] = tuple(
                    ttnn.from_torch(t.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                    for t in rotary(r["d"], torch.tensor([[r["length"]]]))
                )
                r["hpages"] = ttnn.from_torch(r["pages"], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

        def prefill(r):
            for z, t in zip(zeros, buffers):
                ttnn.copy_host_to_device_tensor(z, t)
            if not linear:
                ttnn.copy_host_to_device_tensor(r["hpages"], state.page_table)
            start = r["start"]
            if start:
                prefix = r["tx"][:, :start]
                pr = None if linear else tuple(t[:, :, :start, :] for t in r["rope"])
                out = layer.prefill_forward(prefix, state=state, rope=pr)
                del out, prefix, pr
                suffix = r["tx"][:, start:]
                sr = None if linear else tuple(t[:, :, start:, :] for t in r["rope"])
                out = layer.prefill_forward(suffix, state=state, start_pos=start, rope=sr)
                del out, suffix, sr
            else:
                out = layer.prefill_forward(r["tx"], state=state, rope=r["rope"])
                del out
            ttnn.copy_host_to_device_tensor(r["hd"], td)
            if not linear:
                ttnn.copy_host_to_device_tensor(r["hp"], dp)
                for h, t in zip(r["hr"], dr):
                    ttnn.copy_host_to_device_tensor(h, t)

        for r in requests:
            prefill(r)
            warm = layer.decode_forward(td, state=state, **dk)
            del warm
        prefill(requests[0])
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = layer.decode_forward(td, state=state, **dk)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        results = []
        for r in requests:
            prefill(r)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            entry = {
                "length": r["length"],
                "start": r["start"],
                "trace_id": str(trace),
                "pcc": pcc(r["expected"], actual),
            }
            results.append(entry)
            print(entry, flush=True)
            assert entry["pcc"] >= 0.995
        args.output.write_text(
            json.dumps(
                {
                    "layer": args.layer,
                    "provenance": provenance(),
                    "captures": 1,
                    "request_invalidations": 0,
                    "request_releases": 0,
                    "results": results,
                    "program_cache_entries": mesh.num_program_cache_entries(),
                },
                indent=2,
            )
            + "\n"
        )
        ttnn.release_trace(mesh, trace)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
