# SPDX-License-Identifier: Apache-2.0
#
# llk_analysis, ops/blackhole/sfpu_unary round 3: ttnn.linear with a fused activation (core_grid), the caller of the pack-thread
# entry points (bmm_fused_activation.hpp, apply_activation_from_pack). R3_MM_MODE=dump saves the output bits to
# SFPU_FIX_DUMP_DIR (main against the branch, compare_r2_dumps.py); R3_MM_MODE=perf runs five launches after a warm-up
# for the device profiler.
import os
from pathlib import Path

import pytest
import torch
import ttnn
import sys

if not os.environ.get("HWLOCK_HELD") and not os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
    sys.exit("not under hwlock")

pytestmark = pytest.mark.use_module_device

U = ttnn.UnaryOpType
ACTS = {
    "gelu": ttnn.UnaryWithParam(U.GELU, 0.0),
    "gelu_fast": ttnn.UnaryWithParam(U.GELU, 1.0),
    "gelu_tanh": ttnn.UnaryWithParam(U.GELU_TANH),
    "tanh": ttnn.UnaryWithParam(U.TANH, 0.0),
    "tanh_fast": ttnn.UnaryWithParam(U.TANH, 1.0),
    "silu": ttnn.UnaryWithParam(U.SILU),
    "relu6": ttnn.UnaryWithParam(U.RELU6),
    "sigmoid": ttnn.UnaryWithParam(U.SIGMOID),
    "sigmoid_fast": ttnn.UnaryWithParam(U.SIGMOID, 4.0, 1.0),
    "hardsigmoid": ttnn.UnaryWithParam(U.HARDSIGMOID),
    "hardtanh": ttnn.UnaryWithParam(U.HARDTANH, -1.0, 1.0),
    "selu": ttnn.UnaryWithParam(U.SELU, 1.0507009, 1.6732632),
    "softplus": ttnn.UnaryWithParam(U.SOFTPLUS, 1.0, 20.0),
    # the control: the same matmul kernel without an activation
    "none": None,
}
# (m, k, n): a short-k shape where the activation is a large part of the op, and an MLP-like shape
SHAPES = {"k64": (1024, 64, 2048), "mlp": (512, 2048, 2048), "s1": (256, 64, 512)}
# R3_MM_GRID=1x1: a single-core twin of the kernel loop (all of the output on one core, inputs in L1)
MODE = os.environ.get("R3_MM_MODE", "dump")
SEL = [a for a in os.environ.get("R3_MM_ACTS", ",".join(k for k in ACTS if k != "none")).split(",") if a]
GRID = os.environ.get("R3_MM_GRID")
SSEL = [s for s in os.environ.get("R3_MM_SHAPES", "k64" if MODE == "dump" else "k64,mlp").split(",") if s]


@pytest.mark.parametrize("fp32_acc", [False, True], ids=["dest16", "dest32"])
@pytest.mark.parametrize("shape", SSEL)
@pytest.mark.parametrize("act", SEL)
def test_r3_mm(device, act, shape, fp32_acc):
    m, k, n = SHAPES[shape]
    g = torch.Generator().manual_seed(0)
    a = torch.randn((1, 1, m, k), generator=g, dtype=torch.float32).to(torch.bfloat16)
    b = (torch.randn((k, n), generator=g, dtype=torch.float32) * (4.0 / k**0.5)).to(torch.bfloat16)
    mem = ttnn.L1_MEMORY_CONFIG if GRID else ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    grid = ttnn.CoreGrid(y=int(GRID.split("x")[0]), x=int(GRID.split("x")[1])) if GRID else device.core_grid
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32_acc, packer_l1_acc=False
    )
    # core_grid makes ttnn.linear build a program config that fuses the activation into the matmul kernel
    # (apply_activation_from_pack, the pack-thread entry points); without it the activation runs as a separate unary op
    run = lambda: ttnn.linear(
        ta, tb, bias=None, activation=ACTS[act], compute_kernel_config=cfg, core_grid=grid, memory_config=mem
    )
    out = run()
    if MODE == "perf":
        ttnn.synchronize_device(device)
        for _ in range(int(os.environ.get("R3_MM_LAUNCHES", "5"))):
            out = run()
        ttnn.synchronize_device(device)
    y = ttnn.to_torch(out)
    assert y.shape == (1, 1, m, n)
    assert torch.isfinite(y.float()).all()
    if MODE == "dump":
        d = Path(os.environ["SFPU_FIX_DUMP_DIR"])
        d.mkdir(parents=True, exist_ok=True)
        torch.save(y.contiguous().view(torch.int16), d / f"mm_{act}_{shape}_{'d32' if fp32_acc else 'd16'}.pt")
