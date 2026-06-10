# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy harness for TtPatchMerger at the PRODUCTION operating point.

Production context (tt/ocr_model.py): the bf16 vision tower (tp=4 on the
1x4 QB mesh) calls the merger ONCE per image on the post_trunk_norm output
— a replicated [1, 1, padded_seq, 1536] bf16 DRAM-interleaved activation.
At DOCUMENT scale (/tmp/demo_image1_cropped.jpg => ~11224 vision tokens)
padded_seq = 11264. Inside the block: LayerNorm -> ROW_MAJOR reshape
[11264,1536]->[2816,6144] -> Linear 6144->6144 -> exact GELU ->
Linear 6144->1536. Merger weights are replicated (run-once encoder,
placement=replicate per ARCHITECTURE.md).

Operating points:

- default: bf16 + seq=11264 (PRODUCTION, occupancy REDO);
- ``--seq 896`` selects the golden 784-patch gate shape;
- ``--dtype fp32`` selects the legacy high-precision path used by
  tests/test_vision_transformer.py (fp32 weights + HiFi4).

Run under tracy:
    python -m tracy -p -v -r --op-support-count 20000 \
        models/demos/rednote_hilab_dots.ocr/tt/profile_patch_merger.py --traced
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

import ttnn

_TT_DIR = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("dots_ocr_tt_patch_merger", _TT_DIR / "patch_merger.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TtPatchMerger = _mod.TtPatchMerger

PROD_SEQ = 11264  # ~11224 document vision tokens padded to a multiple of 128
HIDDEN = 1536


def _load_weights():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = Path(snapshot_download("rednote-hilab/dots.ocr", allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for short in ("ln_q.weight", "ln_q.bias", "mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias"):
        full = f"vision_tower.merger.{short}"
        with safe_open(snap / idx[full], framework="pt") as f:
            out[short] = f.get_tensor(full).float()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--weight-dtype", choices=["auto", "match", "bf8b"], default="auto")
    parser.add_argument("--seq", type=int, default=PROD_SEQ, help="padded sequence rows (896=gate shape)")
    parser.add_argument("--iters", type=int, default=5, help="timed untraced iterations")
    args = parser.parse_args()
    dtype = ttnn.bfloat16 if args.dtype == "bf16" else ttnn.float32
    print(f"input: seq={args.seq} dtype={args.dtype} weight_dtype={args.weight_dtype}")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=32768,
        trace_region_size=120_000_000,
    )
    try:
        kwargs = {}
        if args.weight_dtype == "bf8b":
            kwargs["weight_dtype"] = ttnn.bfloat8_b
        elif args.weight_dtype == "match":
            kwargs["weight_dtype"] = dtype
        block = TtPatchMerger(mesh_device, _load_weights(), dtype=dtype, **kwargs)

        torch.manual_seed(0)
        x = torch.randn(1, 1, args.seq, HIDDEN)
        # Persistent input buffer: stable address for trace replay.
        x_tt = ttnn.from_torch(
            x,
            dtype=dtype,
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

        # Untraced wall-clock (the merger runs once per image in production).
        times = []
        for _ in range(max(1, args.iters)):
            t0 = time.perf_counter()
            out = block.forward(x_tt)
            ttnn.synchronize_device(mesh_device)
            times.append((time.perf_counter() - t0) * 1000)
            ttnn.deallocate(out)
        times.sort()
        print(f"untraced wall ms: min={times[0]:.2f} median={times[len(times)//2]:.2f} max={times[-1]:.2f}")

        if args.traced:
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            out = block.forward(x_tt)
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            # One warmup replay, then the timed/profiled replay.
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            print(f"traced wall ms: {(time.perf_counter() - t0) * 1000:.2f}")
            ttnn.release_trace(mesh_device, tid)
        print("profiled iteration complete (traced=%s)" % args.traced)
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
