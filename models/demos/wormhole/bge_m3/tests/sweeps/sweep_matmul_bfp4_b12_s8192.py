# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Dtype sweep (bfloat8_b vs bfloat4_b) for the 4 BGE-M3 B12/S8192 matmuls on N300.

Uses the CURRENT in-model-winning block configs per shape and only varies the
weight / activation / output dtype. bf4 halves weight+activation read bandwidth
vs bf8; the matmuls are ~16% of runtime and compute-bound at LoFi, so if the
kernel accepts bf4 and it's faster, it's a candidate (PCC-gated by checks.sh at
e2e). This does NOT decide correctness — it only screens device speed + whether
bf4 is a valid format for minimal_matmul. Always confirm the winner with one e2e
+ the PCC gate before keeping.

Run (device profiler, from tt-metal root):
    TT_VISIBLE_DEVICES=0 TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r \\
      --no-runtime-analysis -v -m pytest \\
      models/demos/wormhole/bge_m3/tests/sweeps/sweep_matmul_bfp4_b12_s8192.py -k sweep -sv \\
      > /tmp/mm_bfp4_sweep.log 2>&1
Then read SWEEP_ORDER from the log + parse the ops CSV (2 device ops per combo).
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
        pass


M = 98304  # 12 * 8192 = 3072 tiles
GRID = ttnn.CoreCoord(8, 8)

# (label, K, N, has_gelu, current winning block config m/k/n/sbh/sbw)
SHAPES = [
    ("wi", 1024, 4096, True, (16, 16, 4, 4, 2)),
    ("qkv", 1024, 3072, False, (16, 4, 4, 4, 2)),
    ("wo", 4096, 1024, False, (16, 16, 4, 4, 2)),
    ("out", 1024, 1024, False, (16, 8, 4, 4, 2)),
]

# (weight_dtype, act_dtype, out_dtype)
DTYPE_COMBOS = [
    ("wbf8_abf8_obf8", ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b),  # current baseline
    ("wbf4_abf8_obf8", ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat8_b),  # bf4 weights only
    ("wbf4_abf4_obf8", ttnn.bfloat4_b, ttnn.bfloat4_b, ttnn.bfloat8_b),  # bf4 weights+act
]


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 90_000_000, "num_command_queues": 1}],
    indirect=True,
)
def test_matmul_bfp4_sweep(mesh_device):
    if os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1":
        pytest.fail("Set TT_METAL_DEVICE_PROFILER=1 and run under python -m tracy.")

    # LoFi + fp32_dest_acc_en=False matches the model's S8192 matmul kernel.
    ck = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    order, runners = [], []
    for label, K, N, gelu, blk in SHAPES:
        mb, kb, nb, sbh, sbw = blk
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb,
            subblock_h=sbh,
            subblock_w=sbw,
            compute_with_storage_grid_size=GRID,
        )
        act_fmt = (ttnn.UnaryOpType.GELU, True) if gelu else None
        for dlabel, wdt, adt, odt in DTYPE_COMBOS:
            act = ttnn.from_torch(
                torch.randn(1, 1, M, K, dtype=torch.bfloat16),
                dtype=adt,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            w = ttnn.from_torch(
                torch.randn(1, 1, K, N, dtype=torch.bfloat16),
                dtype=wdt,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            def run(a=act, weight=w, c=cfg, fa=act_fmt, od=odt):
                out = ttnn.experimental.minimal_matmul(
                    input_tensor=a,
                    weight_tensor=weight,
                    bias_tensor=None,
                    fused_activation=fa,
                    config=c,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=od,
                    compute_kernel_config=ck,
                )
                ttnn.deallocate(out)

            order.append(f"{label}_{dlabel}")
            runners.append(run)

    logger.info(f"matmul bfp4 sweep: {len(runners)} combos")
    valid_order, valid_runners = [], []
    for lbl, run in zip(order, runners):
        try:
            run()
            ttnn.synchronize_device(mesh_device)
            valid_order.append(lbl)
            valid_runners.append(run)
        except Exception as e:
            logger.warning(f"skip {lbl}: {str(e)[:80]}")
    ttnn.synchronize_device(mesh_device)

    logger.info("SWEEP_ORDER: " + " ".join(valid_order))
    signpost("start")
    for run in valid_runners:
        run()
        ttnn.synchronize_device(mesh_device)
    signpost("stop")
    logger.info(f"matmul bfp4 sweep done: {len(valid_runners)}/{len(order)} valid")
