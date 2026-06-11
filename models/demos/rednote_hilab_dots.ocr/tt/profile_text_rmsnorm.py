# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy harness for TtTextRMSNorm at the PRODUCTION operating points.

Production context (tt/ocr_model.py / tt/decoder_layer.py):

- DECODE (hot path, traced token step): 28 layers x 2 norms + final norm
  consume a REPLICATED bf16 [1, 1, 1, 1536] residual row (perf REDO 4
  precision budget) with an fp32 TILE [1,1,1,dim] gamma, every token.
  This is the shape that matters — the contract's decode-path posture.
- PREFILL (runs once): fp32 [1, 1, P32, 1536] replicated rows (the
  fp32-mandatory attention chain), e.g. 2336 for the long sample.

Run under tracy:
    python -m tracy -p -v -r --op-support-count 20000 \
        models/demos/rednote_hilab_dots.ocr/tt/profile_text_rmsnorm.py --traced

Flags: --seq 1 (decode row, default) | N (prefill rows); --act-dtype
bf16|fp32 (production: bf16 at decode, fp32 at prefill); --variant
interleaved (production path) | sharded (occupancy A/B: i2s WIDTH-shard ->
LayerNormShardedMultiCoreProgramConfig -> s2i DRAM, honoring the
DRAM-in/DRAM-out block contract); --grid X,Y for the sharded sweep.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

import ttnn

_TT_DIR = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("dots_ocr_tt_text_rmsnorm", _TT_DIR / "text_rmsnorm.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TtTextRMSNorm = _mod.TtTextRMSNorm

HIDDEN = 1536
TILE = 32


def _load_weight():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = Path(snapshot_download("rednote-hilab/dots.ocr", allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    full = "model.layers.0.input_layernorm.weight"
    with safe_open(snap / idx[full], framework="pt") as f:
        return f.get_tensor(full).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    parser.add_argument("--seq", type=int, default=1, help="logical rows: 1 = decode token row (default)")
    parser.add_argument("--act-dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument(
        "--variant",
        choices=["interleaved", "sharded"],
        default="interleaved",
        help="interleaved: production block path; sharded: occupancy A/B with i2s/s2i bounce",
    )
    parser.add_argument("--grid", default="8,6", help="sharded-variant grid 'x,y' (width-shard cores)")
    args = parser.parse_args()
    act_dtype = ttnn.bfloat16 if args.act_dtype == "bf16" else ttnn.float32

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=32768,
        trace_region_size=25_000_000,
    )
    try:
        grid = mesh_device.compute_with_storage_grid_size()
        print(f"queried compute grid: {grid.x}x{grid.y} = {grid.x * grid.y} cores")
        # Production gamma dtype: fp32 TILE [1,1,1,dim] (ocr_model builds
        # every text norm with dtype=ttnn.float32; bf16 RM gammas measured
        # WORSE on the bf16 decode rows, REDO 4 A/B — see ocr_model.py).
        block = TtTextRMSNorm(mesh_device, {"weight": _load_weight()}, dtype=ttnn.float32, eps=1e-6)

        torch.manual_seed(0)
        padded_seq = ((args.seq + TILE - 1) // TILE) * TILE
        # LOGICAL row count, exactly as production hands it over (decode is
        # [1,1,1,H]; TILE layout pads 1 -> 32 physically). The logical shape
        # drives the block's path gate, so it must not be pre-padded here.
        x_host = torch.randn(1, 1, args.seq, HIDDEN)
        # Persistent input buffer: stable address for trace replay.
        x_tt = ttnn.from_torch(
            x_host,
            dtype=act_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        if args.variant == "sharded":
            gx, gy = (int(v) for v in args.grid.split(","))
            num_cores = gx * gy
            dim_tiles = HIDDEN // TILE
            assert dim_tiles % num_cores == 0, f"{dim_tiles} tiles !% {num_cores} cores"
            block_w = dim_tiles // num_cores
            subblock_w = min(4, block_w)
            while block_w % subblock_w:
                subblock_w -= 1
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                subblock_w=subblock_w,
                block_h=padded_seq // TILE,
                block_w=block_w,
                inplace=False,
            )
            shard_mc = ttnn.create_sharded_memory_config(
                (1, 1, padded_seq, HIDDEN),
                ttnn.CoreGrid(y=gy, x=gx),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
            )

            def run_once():
                x_sh = ttnn.interleaved_to_sharded(x_tt, shard_mc)
                y_sh = ttnn.rms_norm(
                    x_sh,
                    epsilon=block.eps,
                    weight=block.weight,
                    program_config=pc,
                    memory_config=shard_mc,
                    compute_kernel_config=block.compute_kernel_config,
                )
                ttnn.deallocate(x_sh)
                y = ttnn.sharded_to_interleaved(y_sh, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(y_sh)
                return y

        else:

            def run_once():
                return block.forward(x_tt)

        # Warmup: compile every kernel into the program cache.
        for _ in range(3):
            out = run_once()
            ttnn.deallocate(out)
        ttnn.synchronize_device(mesh_device)

        if args.traced:
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            out = run_once()
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            # One warmup replay, then the profiled replay.
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            ttnn.release_trace(mesh_device, tid)
        else:
            out = run_once()
            ttnn.synchronize_device(mesh_device)
        print(
            "profiled iteration complete (traced=%s, seq=%d, act=%s, variant=%s)"
            % (args.traced, args.seq, args.act_dtype, args.variant)
        )
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
