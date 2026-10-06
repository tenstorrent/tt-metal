# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Repro: approximate exp on Wormhole with TT_METAL_DISABLE_SFPLOADMACRO=1.

One core runs read -> exp_tile<approx=true>(x) -> write through ttnn.generic_op
(kernel: tests/tt_metal/tt_metal/test_kernels/compute/exp_approx_tile.cpp).

With SFPLOADMACRO disabled, the WH InputClamping::None fallback in ckernel_sfpu_exp.h issues SFPMAD and
then SFP_STOCH_RND on the MAD result in the next cycle. WH does not stall for that, so on silicon every
output is 0. ttsim does not model the hazard and passes. SDPA's streaming compute (bf16 dest acc) uses
this path, so attention outputs are all zero (tt-metal#59499).

The flag is read once at device startup, so every case runs in a fresh process.

    pytest tests/ttnn/unit_tests/operations/eltwise/test_exp_approx_disable_sfploadmacro.py
    python tests/ttnn/unit_tests/operations/eltwise/test_exp_approx_disable_sfploadmacro.py   # summary table
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


def run_case(clamp):
    """Child process: run the kernel once and print 'RESULT <pcc> <max_rel_err> <first 3 outputs>'."""
    import ttnn

    torch.manual_seed(0)
    x = torch.empty(1, NUM_TILES, 32, 32).uniform_(-10.0, 0.5).bfloat16()  # inside approx exp's valid range
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
    rel = ((y - ref).abs() / ref).max().item()
    print(f"RESULT {pcc:.5f} {rel:.3g} {' '.join(f'{v:.4g}' for v in y.flatten()[:3].tolist())}")


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
    pcc, rel, *head = line.split()[1:]
    return float(pcc), float(rel), head


@pytest.mark.skipif(not is_wormhole_b0(), reason="Wormhole-only: Blackhole stalls on SFPMAD read-after-write")
@pytest.mark.parametrize("clamp", [False, True], ids=["clamp_none", "clamp_negative"])
@pytest.mark.parametrize("disable_sfploadmacro", [True, False], ids=["sfploadmacro_off", "sfploadmacro_on"])
def test_exp_approx_disable_sfploadmacro(disable_sfploadmacro, clamp):
    if not disable_sfploadmacro and os.environ.get("TT_METAL_SIMULATOR"):
        pytest.skip("ttsim does not implement SFPLOADMACRO")
    pcc, rel, head = spawn(disable_sfploadmacro, clamp)
    assert pcc >= PCC, f"pcc={pcc} max_rel_err={rel} first outputs={head}"


if __name__ == "__main__":
    if len(sys.argv) > 1:  # child process
        run_case(clamp=sys.argv[1] == "clamp")
    else:
        print(f"{'DISABLE_SFPLOADMACRO':>20} {'clamping':>16}  pcc      max_rel_err  first outputs")
        for disable in (False, True):
            for clamp in (False, True):
                pcc, rel, head = spawn(disable, clamp)
                name = "ClampToNegative" if clamp else "None"
                print(f"{int(disable):>20} {name:>16}  {pcc:<8.5f} {rel:<12.3g} {' '.join(head)}")
