# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy harness for TtTextMLP at the PRODUCTION operating points.

Production context (tt/ocr_model.py / tt/decoder_layer.py):

- DECODE (hot path, traced token step): 28 layers consume a REPLICATED
  bf16 [1, 1, 1, 1536] residual row every token; MLP weights are stored
  bfloat8_b (mlp_dtype=mlp_gate_up_dtype=bfloat8_b in ocr_model). Per-chip
  TP slices: gate/up [1536, 2240] col-parallel, down [2240, 1536]
  row-parallel + reduce_scatter/all_gather all-reduce.
- PREFILL (runs once): fp32 [1, 1, P32, 1536] replicated rows (e.g. 128
  golden bucket, 2336 long sample), same bfloat8_b weights.

Run under tracy:
    python -m tracy -p -v -r --op-support-count 20000 \
        models/demos/rednote_hilab_dots.ocr/tt/profile_text_mlp.py --traced

Flags: --seq 1 (decode row, default) | N (prefill rows); --act-dtype
bf16|fp32 (production: bf16 decode, fp32 prefill); --weight-dtype
bfp8|bf16|fp32 (production: bfp8); --iters replay count.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

import ttnn

_TT_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _TT_DIR.parent

_spec = importlib.util.spec_from_file_location("dots_ocr_tt_text_mlp", _TT_DIR / "text_mlp.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TtTextMLP = _mod.TtTextMLP

REPO = "rednote-hilab/dots.ocr"
HIDDEN = 1536


def _load_weights(prefix, keys):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = Path(snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for k in keys:
        full = f"{prefix}.{k}"
        with safe_open(snap / idx[full], framework="pt") as f:
            out[k] = f.get_tensor(full).float()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    parser.add_argument("--iters", type=int, default=1, help="profiled replay count")
    parser.add_argument("--seq", type=int, default=1, help="logical rows: 1 = decode token row (default)")
    parser.add_argument("--act-dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--weight-dtype", choices=["bfp8", "bf16", "fp32"], default="bfp8")
    args = parser.parse_args()
    act_dtype = ttnn.bfloat16 if args.act_dtype == "bf16" else ttnn.float32
    w_dtype = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16, "fp32": ttnn.float32}[args.weight_dtype]

    golden = torch.load(_MODEL_DIR / "reference" / "golden" / "text_mlp.pt")
    seq, dim = args.seq, HIDDEN
    # Production-distribution rows from the golden activation, tiled/cropped
    # to the requested logical seq (decode = 1 row).
    g = golden["input"].reshape(1, -1, HIDDEN)
    reps = (seq + g.shape[1] - 1) // g.shape[1]
    x = g.repeat(1, reps, 1)[:, :seq, :]

    sd = _load_weights("model.layers.0.mlp", ["gate_proj.weight", "up_proj.weight", "down_proj.weight"])

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=32768,
        trace_region_size=50_000_000,
    )
    try:
        grid = mesh_device.compute_with_storage_grid_size()
        print(f"queried compute grid: {grid.x}x{grid.y} = {grid.x * grid.y} cores")
        # Production weight dtype: ocr_model stores all MLP weights bfloat8_b.
        block = TtTextMLP(mesh_device, sd, dtype=w_dtype)

        # Persistent input buffer: stable address for trace replay.
        x_tt = ttnn.from_torch(
            x.reshape(1, 1, seq, dim).float(),
            dtype=act_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Warmup: compile every kernel into the program cache.
        for _ in range(3):
            out = block.forward(x_tt)
            ttnn.deallocate(out)
        ttnn.synchronize_device(mesh_device)

        if args.traced:
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            out = block.forward(x_tt)
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            # One warmup replay, then the profiled replays.
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            for _ in range(max(1, args.iters)):
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            ttnn.release_trace(mesh_device, tid)
        else:
            out = block.forward(x_tt)
            ttnn.synchronize_device(mesh_device)
        print(
            "profiled iteration complete (traced=%s, seq=%d, act=%s, w=%s)"
            % (args.traced, seq, args.act_dtype, args.weight_dtype)
        )
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
