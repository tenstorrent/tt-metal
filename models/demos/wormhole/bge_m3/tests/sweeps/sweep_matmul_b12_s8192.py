# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone matmul (MinimalMatmul) block-size sweep for BGE-M3 B12/S8192 on N300.

The 4 encoder matmuls run on DEFAULT ttnn routing at S8192 (all SLOW, in0_block_w=1,
DRAM-bandwidth-bound at 12-36 GB/s). The 2D mcast program config can't be used
here: per_core_M = M_tiles/grid_y = 3072/8 = 384 tiles blows L1 (needs 20MB).
minimal_matmul streams M in blocks, so it fits and is the same primitive the
Blackhole B32 path uses. This sweep finds the best (M_block, K_block, N_block,
subblock_h, subblock_w) per shape via device profiler + trace capture.

Shapes (M = 12*8192 = 98304 rows = 3072 tiles):
  wi:  K=1024 N=4096  +GELU   (slowest, 42.7ms default)
  qkv: K=1024 N=3072          (22.8ms)
  wo:  K=4096 N=1024          (13.9ms)
  out: K=1024 N=1024          (7.6ms)

Run (device profiler, from tt-metal root):
    TT_VISIBLE_DEVICES=0 TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r \\
      --no-runtime-analysis -v -m pytest \\
      models/demos/wormhole/bge_m3/tests/sweeps/sweep_matmul_b12_s8192.py -k sweep -sv \\
      > /tmp/mm_sweep.log 2>&1
Then read SWEEP_ORDER from the log + parse the ops CSV by index.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

try:
    from tracy import signpost
except ImportError:

    def signpost(*_args, **_kwargs):
        return None


M = 98304  # 12 * 8192 = 3072 tiles
GRID = ttnn.CoreCoord(8, 8)

# (label, K, N, has_gelu)
SHAPES = [
    ("wi", 1024, 4096, True),
    ("qkv", 1024, 3072, False),
    ("wo", 4096, 1024, False),
    ("out", 1024, 1024, False),
]

# Block-size candidates (in tiles). subblock h*w <= 8 (fp32_dest_acc_en=False).
M_BLOCKS = [4, 8, 16]
K_BLOCKS = [4, 8, 16, 32]
N_BLOCKS = [4, 8]
SUBBLOCKS = [(4, 2), (2, 4), (4, 1), (2, 2), (1, 4), (8, 1), (1, 8)]


def _pick_subblock(mb, nb):
    for h, w in SUBBLOCKS:
        if mb % h == 0 and nb % w == 0 and h * w <= 8:
            return h, w
    return 1, 1


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 90_000_000, "num_command_queues": 1}],
    indirect=True,
)
def test_matmul_sweep(mesh_device):
    if os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1":
        pytest.fail("Set TT_METAL_DEVICE_PROFILER=1 and run under python -m tracy.")

    ck = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    order = []
    runners = []
    for label, K, N, gelu in SHAPES:
        k_tiles = K // 32
        n_tiles = N // 32
        act = ttnn.from_torch(
            torch.randn(1, 1, M, K, dtype=torch.bfloat16),
            dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
            device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w = ttnn.from_torch(
            torch.randn(1, 1, K, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
            device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        act_fmt = (ttnn.UnaryOpType.GELU, True) if gelu else None
        for mb in M_BLOCKS:
            for kb in K_BLOCKS:
                if kb > k_tiles:
                    continue
                for nb in N_BLOCKS:
                    if nb > n_tiles:
                        continue
                    sbh, sbw = _pick_subblock(mb, nb)
                    cfg = ttnn.MinimalMatmulConfig(
                        M_block_size=mb, K_block_size=kb, N_block_size=nb,
                        subblock_h=sbh, subblock_w=sbw,
                        compute_with_storage_grid_size=GRID,
                    )

                    def run(a=act, weight=w, c=cfg, fa=act_fmt):
                        out = ttnn.experimental.minimal_matmul(
                            input_tensor=a, weight_tensor=weight, bias_tensor=None,
                            fused_activation=fa, config=c,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            dtype=ttnn.bfloat8_b, compute_kernel_config=ck,
                        )
                        ttnn.deallocate(out)

                    order.append(f"{label}_m{mb}_k{kb}_n{nb}_sb{sbh}x{sbw}")
                    runners.append(run)

    logger.info(f"matmul sweep: {len(runners)} combos")
    valid_order, valid_runners = [], []
    for lbl, run in zip(order, runners):
        try:
            run()
            ttnn.synchronize_device(mesh_device)
            valid_order.append(lbl)
            valid_runners.append(run)
        except Exception as e:
            logger.warning(f"skip {lbl}: {str(e)[:70]}")
    ttnn.synchronize_device(mesh_device)

    logger.info("SWEEP_ORDER: " + " ".join(valid_order))
    signpost("start")
    for run in valid_runners:
        run()
        ttnn.synchronize_device(mesh_device)
    signpost("stop")
    logger.info(f"matmul sweep done: {len(valid_runners)}/{len(order)} valid")
