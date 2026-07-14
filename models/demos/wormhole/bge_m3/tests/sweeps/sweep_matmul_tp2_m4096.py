# SPDX-License-Identifier: Apache-2.0
"""Matmul (MinimalMatmul) block-size sweep for the SEQ-PARALLEL TP2 shapes.

In sequence-parallel TP2 each chip holds S/2 = 4096 tokens, so the per-chip
matmul M-dim is 12*4096 = 49152 rows (1536 tiles), HALF the single-chip S8192
value (98304). The in-model minimal_matmul configs were tuned for M=98304, so
they may be suboptimal at M=49152. This sweep re-tunes (M_block, K_block,
N_block, subblock) per shape and ranks by wall-clock time (no CSV parse needed).

Shapes (per-chip, M = 12*4096 = 49152 = 1536 tiles):
  wi:  K=1024 N=4096  +GELU
  qkv: K=1024 N=3072
  wo:  K=4096 N=1024
  out: K=1024 N=1024

Run (single device is enough — matmul is token-local, no CCL):
  TT_VISIBLE_DEVICES=0 pytest \\
    models/demos/wormhole/bge_m3/tests/sweeps/sweep_matmul_tp2_m4096.py -s -q
"""
import time

import pytest
import torch
from loguru import logger

import ttnn

M = 49152  # 12 * 4096 = 1536 tiles
GRID = ttnn.CoreCoord(8, 8)

SHAPES = [
    ("wi", 1024, 4096, True),
    ("qkv", 1024, 3072, False),
    ("wo", 4096, 1024, False),
    ("out", 1024, 1024, False),
]

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
def test_matmul_sweep_m4096(mesh_device):
    ck = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    results = {}  # label -> {cfg_str: us}
    for label, K, N, gelu in SHAPES:
        k_tiles, n_tiles = K // 32, N // 32
        act = ttnn.from_torch(
            torch.randn(1, 1, M, K, dtype=torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w = ttnn.from_torch(
            torch.randn(1, 1, K, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        act_fmt = (ttnn.UnaryOpType.GELU, True) if gelu else None
        results[label] = {}
        for mb in M_BLOCKS:
            for kb in K_BLOCKS:
                if kb > k_tiles:
                    continue
                for nb in N_BLOCKS:
                    if nb > n_tiles:
                        continue
                    sbh, sbw = _pick_subblock(mb, nb)
                    cfg = ttnn.MinimalMatmulConfig(
                        M_block_size=mb,
                        K_block_size=kb,
                        N_block_size=nb,
                        subblock_h=sbh,
                        subblock_w=sbw,
                        compute_with_storage_grid_size=GRID,
                    )
                    cfg_str = f"m{mb}_k{kb}_n{nb}_sb{sbh}x{sbw}"

                    def run(a=act, weight=w, c=cfg, fa=act_fmt):
                        out = ttnn.experimental.minimal_matmul(
                            input_tensor=a,
                            weight_tensor=weight,
                            bias_tensor=None,
                            fused_activation=fa,
                            config=c,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            dtype=ttnn.bfloat8_b,
                            compute_kernel_config=ck,
                        )
                        ttnn.deallocate(out)

                    try:
                        for _ in range(2):
                            run()
                        ttnn.synchronize_device(mesh_device)
                        N_IT = 10
                        t0 = time.perf_counter()
                        for _ in range(N_IT):
                            run()
                        ttnn.synchronize_device(mesh_device)
                        us = (time.perf_counter() - t0) / N_IT * 1e6
                        results[label][cfg_str] = us
                    except Exception as e:
                        logger.warning(f"skip {label} {cfg_str}: {str(e)[:50]}")
        ttnn.deallocate(act)
        ttnn.deallocate(w)

    print("\n" + "=" * 70)
    print("MATMUL SWEEP RESULTS (seq-parallel M=49152, wall-clock us)")
    print("=" * 70)
    for label, _, _, _ in SHAPES:
        res = results[label]
        if not res:
            continue
        ranked = sorted(res.items(), key=lambda x: x[1])
        print(f"\n{label}: best 5 of {len(res)}")
        for cfg_str, us in ranked[:5]:
            print(f"  {us:8.1f} us  {cfg_str}")
