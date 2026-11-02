# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Wormhole approximate exp built with DISABLE_SFPLOADMACRO (the hand-written SFPMAD fallback loops).

One core runs copy_tile -> exp_tile<approx=true>(x) -> pack_tile through ttnn.generic_op with the stock
eltwise_sfpu.cpp test kernel, once per InputClamping mode. DISABLE_SFPLOADMACRO is passed as a kernel define
(it feeds the kernel build hash), so the cases run in-process on the shared device fixture. The env var
TT_METAL_DISABLE_SFPLOADMACRO=1 compiles every kernel the same way and a define cannot undo it, so the
sfploadmacro_on cases skip when it is set (the ttsim lane exports it).

Wormhole has no SFPU scoreboard: an instruction that reads an LReg on the cycle after SFPMAD wrote it
does not see the SFPMAD result. The fallback loops in ckernel_sfpu_exp.h issue SFPMAD and then
SFP_STOCH_RND on the same LReg back to back, which needs an SFPNOP between them.
  - InputClamping::None (the SDPA streaming path): every output is 0.
  - InputClamping::ClampToNegative: lanes whose input is below the clamp threshold return garbage.
"""
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import is_wormhole_b0

KERNEL = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpu.cpp"
UNARY_DATAFLOW = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow"
CB_IN = 0  # tt::CBIndex::c_0 in eltwise_sfpu.cpp
CB_OUT = 16  # tt::CBIndex::c_16 in eltwise_sfpu.cpp
NUM_TILES = 4
TILE_PAGE_BYTES = 32 * 32 * 2  # one bf16 tile
CLAMP_THRESHOLD = -88.5  # ClampToNegative maps inputs below this to exp(CLAMP_THRESHOLD) ~ 0
PCC = 0.999
ATOL = 0.1  # approx exp is within ~3.5 % of exp(x) <= exp(0.5)


def make_input(clamp):
    torch.manual_seed(0)
    x = torch.empty(1, NUM_TILES, 32, 32).uniform_(-10.0, 0.5)  # inside approx exp's valid range
    if clamp:
        # Rows 16-31 of every tile sit below the clamp threshold, where the clamped lanes are produced.
        x[:, :, 16:, :].uniform_(-200.0, CLAMP_THRESHOLD)
    return x.bfloat16()


def run_exp_approx(device, x, clamp, disable_sfploadmacro):
    mode = "ClampToNegative" if clamp else "None"
    defines = [
        ("SFPU_OP_EXP_INCLUDE", "1"),
        (
            "SFPU_OP_CHAIN_0",
            f"exp_tile_init<true, 0x3F800000, InputClamping::{mode}>(); exp_tile<true, false, InputClamping::{mode}>(0);",
        ),
    ]
    if disable_sfploadmacro:
        defines.append(("DISABLE_SFPLOADMACRO", "1"))

    inp = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(x.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cbs = [
        ttnn.CBDescriptor(
            total_size=pages * TILE_PAGE_BYTES,
            core_ranges=core,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb, data_format=ttnn.bfloat16, page_size=TILE_PAGE_BYTES)
            ],
        )
        for cb, pages in ((CB_IN, 2), (CB_OUT, NUM_TILES))  # the kernel reserves the whole output block at once
    ]
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [inp.buffer_address(), NUM_TILES, 0]
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[0][0] = [out.buffer_address(), NUM_TILES, 0]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{UNARY_DATAFLOW}/reader_unary_interleaved_start_id.cpp",
            core_ranges=core,
            compile_time_args=ttnn.TensorAccessorArgs(inp).get_compile_time_args(),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{UNARY_DATAFLOW}/writer_unary_interleaved_start_id.cpp",
            core_ranges=core,
            compile_time_args=[CB_OUT] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KERNEL,
            core_ranges=core,
            compile_time_args=[1, NUM_TILES],  # per_core_block_cnt, per_core_block_dim,
            defines=defines,
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(math_approx_mode=True),
        ),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return ttnn.to_torch(ttnn.generic_op([inp, out], program)).float()


@pytest.mark.skipif(
    not is_wormhole_b0(),
    reason="Wormhole-only: the Blackhole SFPU stalls an SFPMAD read-after-write in hardware (Blackhole ISA, SFPMAD scheduling)",
)
@pytest.mark.parametrize("clamp", [False, True], ids=["clamp_none", "clamp_negative"])
@pytest.mark.parametrize("disable_sfploadmacro", [True, False], ids=["sfploadmacro_off", "sfploadmacro_on"])
def test_exp_approx_disable_sfploadmacro(device, disable_sfploadmacro, clamp):
    if not disable_sfploadmacro and os.environ.get("TT_METAL_DISABLE_SFPLOADMACRO") == "1":
        pytest.skip("TT_METAL_DISABLE_SFPLOADMACRO=1 builds every kernel without SFPLOADMACRO; same as the off case")
    x = make_input(clamp)
    y = run_exp_approx(device, x, clamp, disable_sfploadmacro)
    ref = torch.exp(x.float())
    pcc = torch.corrcoef(torch.stack([y.flatten(), ref.flatten()]))[0, 1].nan_to_num(0.0).item()
    max_abs = (y - ref).abs().max().item()
    head = [f"{v:.4g}" for v in y.flatten()[:3].tolist()]
    assert pcc >= PCC and max_abs <= ATOL, f"pcc={pcc:.5f} max_abs_err={max_abs:.3g} first outputs={head}"
