# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Reduce-scatter fold + SwiGLU cost for a K-split gate/up (fp32 DEST): C contributions (bf16 or bfp8) of each gate /
up tile folded in DEST (moe_fused_swiglu's add_tiles fold), SwiGLU on the PACK thread, h packed. Cycles per h tile
and h error against fp32 partials (what the partial format costs)."""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import (
    aiclk_mhz,
    cb,
    crs,
    kpath,
    pcc,
    quantize,
    read_stamps,
    stamps_tensor,
    to_dev,
)

DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}
NT = 8  # outputs a pass (tiles of each of gate / up)


def tiles_to_dev(parts, dtype, device):
    """parts: [C, NT, 32, 32] -> a [C * NT * 32, 32] tensor (tile i = row block i)."""
    return to_dev(parts.reshape(-1, 32), dtype, device)


def run_fold(device, C, p_dt, out_dt, mode, passes):
    core = (0, 0)
    grid = crs([core])
    torch.manual_seed(0)
    # partial sums over K / C of a gate / up dot product: the total has std ~2.5 (SiLU's nonlinear range)
    gp = torch.randn(C, NT, 32, 32) * (2.5 / C**0.5)
    up = torch.randn(C, NT, 32, 32) * (1.0 / C**0.5)
    pd, od = DT[p_dt], DT[out_dt]
    g_t, u_t = tiles_to_dev(gp, pd, device), tiles_to_dev(up, pd, device)
    o_t = to_dev(torch.zeros(NT * 32, 32), od, device)
    st = stamps_tensor(device, [core])
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rd[0][0] = [g_t.buffer_address(), u_t.buffer_address(), C * NT, C * NT]
    wr[0][0] = [o_t.buffer_address(), passes * NT, NT, 2]
    cp[0][0] = [st.buffer_address()]
    prog = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=kpath("bench_reader.cpp"),
                core_ranges=grid,
                compile_time_args=list(ttnn.TensorAccessorArgs(g_t).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(u_t).get_compile_time_args()),
                runtime_args=rd,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=kpath("bench_writer.cpp"),
                core_ranges=grid,
                compile_time_args=list(ttnn.TensorAccessorArgs(o_t).get_compile_time_args()),
                runtime_args=wr,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=kpath("fold_bench.cpp"),
                core_ranges=grid,
                compile_time_args=[C, NT, passes],
                named_compile_time_args=[("expert_activation", 0)],
                defines=[
                    ("FB_MODE", str(mode)),
                    ("EXPERT_ACTIVATION_BETA", "1.0f"),
                    ("EXPERT_ACTIVATION_LINEAR_BETA", "1.0f"),
                    ("EXPERT_ACTIVATION_LIMIT", "7.0f"),
                    ("EXPERT_ACTIVATION_ALPHA", "1.702f"),
                ],
                runtime_args=cp,
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True),
            ),
        ],
        semaphores=[],
        cbs=[cb(0, pd, C * NT, grid), cb(1, pd, C * NT, grid), cb(16, od, 4, grid)],
    )
    ttnn.generic_op([g_t, u_t, o_t, st], prog)
    ttnn.synchronize_device(device)
    s = read_stamps(st, 1)[0]
    cyc = int((s[5] - s[0]) & 0xFFFFFFFF)
    o = ttnn.to_torch(o_t).float().reshape(NT, 32, 32)
    g32, u32 = gp.sum(0), up.sum(0)  # fp32 partials: the reference
    if mode == 1:
        ref = g32
    elif mode == 2:
        ref = torch.nn.functional.silu(gp[0]) * up[0]
    else:
        ref = torch.nn.functional.silu(g32) * u32
    gq = quantize(gp.reshape(-1, 32), pd).reshape(gp.shape).sum(0)
    uq = quantize(up.reshape(-1, 32), pd).reshape(up.shape).sum(0)
    ref_q = torch.nn.functional.silu(gq) * uq  # what exact fp32 math on the quantized partials gives
    rel = float((o - ref).norm() / ref.norm())
    rel_q = float((ref_q - ref).norm() / ref.norm()) if mode == 0 else float("nan")
    return cyc, pcc(o, ref), rel, rel_q


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("p_dt", ["bf16", "bfp8"])
@pytest.mark.parametrize("C", [2, 4, 5, 10])
def test_fold_bench(device, C, p_dt, mode):
    out_dt = "bf16"
    c1 = run_fold(device, C, p_dt, out_dt, mode, 4)
    c2 = run_fold(device, C, p_dt, out_dt, mode, 68)
    per = (c2[0] - c1[0]) / (64 * NT)
    logger.info(
        f"FOLD C {C} partials {p_dt} {'fold+swiglu' if mode == 0 else 'fold only'}: {per:.0f} cyc per h tile "
        f"({per / C:.0f} a contribution pair); h pcc {c2[1]:.6f} rel err {c2[2]:.2e} (partial quantization alone "
        f"{c2[3]:.2e}); aiclk {aiclk_mhz(0)}"
    )
    assert c2[1] > 0.999


def test_fold_floor(device):
    """C = 1, copy gate / up + SwiGLU: the per-h-tile floor of the epilogue."""
    c1 = run_fold(device, 1, "bf16", "bf16", 2, 4)
    c2 = run_fold(device, 1, "bf16", "bf16", 2, 68)
    logger.info(f"FOLD floor (copy + swiglu): {(c2[0] - c1[0]) / (64 * NT):.0f} cyc per h tile, pcc {c2[1]:.6f}")


ROW_NAMES = {
    0: "1x20 today (K120 1x2 + SwiGLU)",
    3: "1x20 no SwiGLU",
    1: "10x2 (K12 5x 1x4 + fold + SwiGLU)",
    2: "10x2 matmuls only",
}


def run_row(device, mode, C, p_dt, passes):
    core = (0, 0)
    grid = crs([core])
    st = stamps_tensor(device, [core])
    pd = DT[p_dt]
    xn = 120 if mode in (0, 3) else 12
    cbs = [
        cb(0, ttnn.bfloat8_b, xn, grid),
        cb(1, ttnn.bfloat4_b, 240, grid),
        cb(2, ttnn.bfloat8_b, 20, grid),
        cb(3, ttnn.bfloat4_b, 120, grid),
        cb(4, pd, 2 * C, grid),
        cb(16, pd, 20, grid),
        cb(17, ttnn.bfloat8_b, 1, grid),
        cb(18, ttnn.bfloat16, 6, grid),
    ]
    push = [5, 0, xn, 1, 240, 2, 20, 3, 120, 4, 2 * C]
    cp = ttnn.RuntimeArgs()
    cp[0][0] = [st.buffer_address()]
    prog = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=kpath("push_only.cpp"),
                core_ranges=grid,
                compile_time_args=push,
                runtime_args=ttnn.RuntimeArgs(),
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=kpath("row_bench.cpp"),
                core_ranges=grid,
                compile_time_args=[passes, C],
                named_compile_time_args=[("expert_activation", 0)],
                defines=[
                    ("RB_MODE", str(mode)),
                    ("EXPERT_ACTIVATION_BETA", "1.0f"),
                    ("EXPERT_ACTIVATION_LINEAR_BETA", "1.0f"),
                    ("EXPERT_ACTIVATION_LIMIT", "7.0f"),
                    ("EXPERT_ACTIVATION_ALPHA", "1.702f"),
                ],
                runtime_args=cp,
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True),
            ),
        ],
        semaphores=[],
        cbs=cbs,
    )
    ttnn.generic_op([st, st], prog)
    ttnn.synchronize_device(device)
    s = read_stamps(st, 1)[0]
    return int((s[5] - s[0]) & 0xFFFFFFFF)


@pytest.mark.parametrize(
    "mode,C,p_dt",
    [(0, 1, "bf16"), (3, 1, "bf16"), (2, 10, "bf16"), (1, 10, "bf16"), (1, 10, "bfp8"), (1, 5, "bf16"), (1, 4, "bf16")],
)
def test_row_bench(device, mode, C, p_dt):
    """Cycles a row tile on one core (gate/up + epilogue + down; 360 tile products, 5760 cycles at 100% LoFi)."""
    a = run_row(device, mode, C, p_dt, 4)
    b = run_row(device, mode, C, p_dt, 68)
    per = (b - a) / 64
    logger.info(f"ROW {ROW_NAMES[mode]} C {C} partials {p_dt}: {per:.0f} cyc a row ({100 * 5760 / per:.1f}% LoFi)")
