# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Wormhole approximate exp with TT_METAL_DISABLE_SFPLOADMACRO=1 (the hand-written SFPMAD fallback loops).

One core runs copy_tile -> exp_tile<approx=true>(x) -> pack_tile through ttnn.generic_op
(kernel: tests/tt_metal/tt_metal/test_kernels/compute/exp_approx_tile.cpp), once per InputClamping mode.

Wormhole has no SFPU scoreboard: an instruction that reads an LReg on the cycle after SFPMAD wrote it
does not see the SFPMAD result. The fallback loops in ckernel_sfpu_exp.h issue SFPMAD and then
SFP_STOCH_RND on the same LReg back to back, which needs an SFPNOP between them.
  - InputClamping::None (the SDPA streaming path): every output is 0.
  - InputClamping::ClampToNegative: lanes whose input is below the -88.5 clamp threshold return garbage.

The env var is read once at device startup, so every case runs in a fresh child process.
"""
import os
import subprocess
import sys

import pytest
import torch

from models.common.utility_functions import is_wormhole_b0

KERNEL = "tests/tt_metal/tt_metal/test_kernels/compute/exp_approx_tile.cpp"
UNARY_DATAFLOW = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow"
NUM_TILES = 4
PCC = 0.999
ATOL = 0.1  # approx exp is within ~3.5 % of exp(x) <= exp(0.5)


def make_input(clamp):
    torch.manual_seed(0)
    x = torch.empty(1, NUM_TILES, 32, 32).uniform_(-10.0, 0.5)  # inside approx exp's valid range
    if clamp:
        # ClampToNegative must map inputs below -88.5 to exp(-88.5) ~ 0; rows 16-31 of every tile sit below it.
        x[:, :, 16:, :].uniform_(-200.0, -88.5)
    return x.bfloat16()


def run_case(clamp):
    """Child process: run the kernel once and print 'RESULT <pcc> <max_abs_err> <first 3 outputs>'."""
    import ttnn

    x = make_input(clamp)
    device = ttnn.open_device(device_id=0)
    try:
        inp = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(x.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        page = 32 * 32 * 2
        cbs = [
            ttnn.CBDescriptor(
                total_size=2 * page,
                core_ranges=core,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)],
            )
            for i in (0, 16)
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
                compile_time_args=[16] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                core_ranges=core,
                compile_time_args=[NUM_TILES, int(clamp)],
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(math_approx_mode=True),
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        y = ttnn.to_torch(ttnn.generic_op([inp, out], program)).float()
    finally:
        ttnn.close_device(device)
    ref = torch.exp(x.float())
    pcc = torch.corrcoef(torch.stack([y.flatten(), ref.flatten()]))[0, 1].nan_to_num(0.0).item()
    max_abs = (y - ref).abs().max().item()
    print(f"RESULT {pcc:.5f} {max_abs:.3g} {' '.join(f'{v:.4g}' for v in y.flatten()[:3].tolist())}")


def spawn(disable_sfploadmacro, clamp):
    env = {**os.environ, "TT_METAL_DISABLE_SFPLOADMACRO": "1" if disable_sfploadmacro else "0"}
    p = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "clamp" if clamp else "none"],
        env=env,
        capture_output=True,
        text=True,
    )
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, f"child failed (rc={p.returncode}):\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}"
    pcc, max_abs, *head = line.split()[1:]
    return float(pcc), float(max_abs), head


@pytest.mark.skipif(not is_wormhole_b0(), reason="Wormhole-only: Blackhole stalls on SFPMAD read-after-write")
@pytest.mark.parametrize("clamp", [False, True], ids=["clamp_none", "clamp_negative"])
@pytest.mark.parametrize("disable_sfploadmacro", [True, False], ids=["sfploadmacro_off", "sfploadmacro_on"])
def test_exp_approx_disable_sfploadmacro(disable_sfploadmacro, clamp):
    if not disable_sfploadmacro and os.environ.get("TT_METAL_SIMULATOR"):
        pytest.skip("ttsim does not implement SFPLOADMACRO")
    pcc, max_abs, head = spawn(disable_sfploadmacro, clamp)
    assert pcc >= PCC and max_abs <= ATOL, f"pcc={pcc} max_abs_err={max_abs} first outputs={head}"


if __name__ == "__main__":
    if len(sys.argv) > 1:  # child process
        run_case(clamp=sys.argv[1] == "clamp")
