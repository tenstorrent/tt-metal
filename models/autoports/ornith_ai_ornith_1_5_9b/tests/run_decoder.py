# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded standalone layer harness; deliberately bypasses global device fixtures."""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.functional_decoder import FunctionalDecoder

MODEL = "ornith-ai/Ornith-1.5-9B"
REVISION = "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"
ROOT = Path(__file__).resolve().parents[1]


def config():
    return Qwen3_5TextConfig(**json.loads((ROOT / "tests/target_config.json").read_text())["text_config"])


def weights(layer, synthetic=False):
    if synthetic:
        stats = json.loads((ROOT / f"doc/functional_decoder/weight_stats_layer_{layer}.json").read_text())
        generator = torch.Generator().manual_seed(42)
        return {
            row["name"]: (torch.randn(row["shape"], generator=generator) * row["std"] + row["mean"]).bfloat16()
            for row in stats
        }

    path = Path(os.environ["HF_HOME"]) / f"hub/models--ornith-ai--Ornith-1.5-9B/snapshots/{REVISION}"
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}."
    out = {}
    for shard in sorted({v for k, v in index.items() if k.startswith(prefix)}):
        with safe_open(path / shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    out[key[len(prefix) :]] = f.get_tensor(key)
    assert out
    stats = [
        {
            "name": k,
            "shape": list(v.shape),
            "dtype": str(v.dtype),
            "mean": v.float().mean().item(),
            "std": v.float().std().item(),
        }
        for k, v in out.items()
    ]
    (ROOT / f"doc/functional_decoder/weight_stats_layer_{layer}.json").write_text(json.dumps(stats, indent=2) + "\n")
    return out


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--length", type=int, default=32)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--context", type=int, default=1024)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(7)
    c = config()
    c._attn_implementation = "sdpa"
    sd = weights(args.layer)
    ref = Qwen3_5DecoderLayer(c, args.layer).bfloat16().eval()
    ref.load_state_dict(sd, strict=True)
    rope = Qwen3_5TextRotaryEmbedding(c)
    cache = DynamicCache(config=c)
    x = (torch.randn(1, args.length, 4096) * 0.1).bfloat16()
    xd = (torch.randn(1, 1, 4096) * 0.1).bfloat16()
    with torch.no_grad():
        pos = torch.arange(args.length)[None, :]
        mask = torch.full((args.length, args.length), float("-inf")).triu(1)[None, None]
        y = ref(
            x,
            position_embeddings=rope(x, pos),
            attention_mask=mask if c.layer_types[args.layer] == "full_attention" else None,
            past_key_values=cache,
        )
        yd = ref(xd, position_embeddings=rope(xd, torch.tensor([[args.length]])), past_key_values=cache)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        tt = FunctionalDecoder.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh)
        state = tt.allocate_state(context=args.context)
        table = ttnn.from_torch(
            torch.randperm(args.context // 32, dtype=torch.int32)[None, :],
            device=mesh,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        position = ttnn.from_torch(
            torch.tensor([args.length], dtype=torch.int32), device=mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        tx = ttnn.from_torch(x[None], device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        td = ttnn.from_torch(xd[None], device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        out = tt.prefill_forward(tx, state=state, page_table=table)
        actual = ttnn.to_torch(out).reshape_as(y)
        pc = pcc(y, actual)
        print("PREFILL", args.layer, args.length, pc, "DELTA", pcc(y - x, actual - x), flush=True)
        snapshots = [
            (buf, ttnn.clone(buf)) for buf in (state.conv, state.recurrent, state.key, state.value) if buf is not None
        ]

        def restore():
            for dst, src in snapshots:
                ttnn.copy(src, dst)

        outd = tt.decode_forward(td, state=state, page_table=table, current_position=position)
        restore()
        if args.trace:
            ttnn.synchronize_device(mesh)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            outd = tt.decode_forward(td, state=state, page_table=table, current_position=position)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            from ttnn.tools import trace_allocation_tracker

            # This output is consumed before the next replay; no other trace is live.
            trace_allocation_tracker.acknowledge_corruptible(outd)
            previous = None
            for replay in range(3):
                restore()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                observed = ttnn.to_torch(outd)
                if previous is not None:
                    assert torch.equal(previous, observed)
                previous = observed
            print("TRACE_REPLAYS", 3, "DETERMINISTIC", True, flush=True)
        else:
            outd = tt.decode_forward(td, state=state, page_table=table, current_position=position)
        actuald = ttnn.to_torch(outd).reshape_as(yd)
        dc = pcc(yd, actuald)
        print("DECODE", dc, "DELTA", pcc(yd - xd, actuald - xd), flush=True)
        assert pc >= 0.995 and dc >= 0.995
        if args.trace:
            ttnn.release_trace(mesh, trace)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
