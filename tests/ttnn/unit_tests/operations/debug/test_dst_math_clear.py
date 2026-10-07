# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import ttnn


@pytest.fixture
def enabled_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        yield
    finally:
        device.disable_and_clear_program_cache()


COMPUTE = r"""
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu.h"
#endif

void kernel_main() {
    constexpr auto iterations = get_compile_time_arg_val(0);
    constexpr auto delay = get_compile_time_arg_val(1);
    CircularBuffer out(16);
    compute_kernel_hw_startup(0, 16);
    for (uint32_t i = 0; i < iterations; ++i) {
        tile_regs_acquire_math_clear();
#ifdef TRISC_MATH
        sfpu::_init_sfpu_config_reg();
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_3);
        math::reset_counters(p_setrwc::SET_ABD_F);
        for (uint32_t d = 0; d < delay; ++d) {
            TTI_SFPNOP;
        }
        TT_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, 0x3f80 + i % 64);
        TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0x4280);
        _llk_math_eltwise_sfpu_start_(1);
        // Skip a write on alternate visits to each half. PACK must see zero,
        // not the value from the previous visit, proving acquisition clears DST.
        if (i % 4 < 2) {
            TTI_SFPSTORE(p_sfpu::LREG4, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_3, 0);
            TTI_SFPSTORE(p_sfpu::LREG4, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_3, 2);
        }
        TTI_SFPSTORE(p_sfpu::LREG5, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_3, 64);
        TTI_SFPSTORE(p_sfpu::LREG5, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_3, 66);
        _llk_math_eltwise_sfpu_done_();
#endif
        tile_regs_commit();
        out.reserve_back(2);
        tile_regs_wait();
        pack_block(1, 16, 2);
        tile_regs_release_math_clear();
        out.push_back(2);
    }
}
"""

WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const auto output = TensorAccessor(TensorAccessorArgs<0>(), get_arg_val<uint32_t>(0));
    const auto iterations = get_arg_val<uint32_t>(1);
    const auto delay = get_arg_val<uint32_t>(2);
    Noc noc;
    DataflowBuffer partial(16);
    for (uint32_t i = 0; i < iterations * 2; ++i) {
        partial.wait_front(1);
        for (uint32_t d = 0; d < delay; ++d) {
            asm volatile("nop");
        }
        noc.async_write(partial, output, 4096, {}, {.page_id = i});
        noc.async_writes_flushed();
        partial.pop_front(1);
    }
    noc.async_write_barrier();
}
"""


@pytest.mark.parametrize("full_sync", [False, True], ids=["double_buffered", "single_buffered"])
def test_dst_math_clear(device, enabled_program_cache, full_sync):
    # Revisit each DST half after both a write and a skipped write, then replay
    # the cached program. Timing stress is kept out of routine simulator runs.
    _run_dst_math_clear(device, full_sync, math_delay=0, writer_delay=0, iterations=8, repeats=2)


@pytest.mark.skipif(
    os.getenv("TT_METAL_DST_MATH_CLEAR_STRESS") != "1",
    reason="set TT_METAL_DST_MATH_CLEAR_STRESS=1 to run the hardware timing stress test",
)
@pytest.mark.skipif(bool(os.getenv("TT_METAL_SIMULATOR")), reason="timing stress requires hardware")
@pytest.mark.parametrize("full_sync", [False, True], ids=["double_buffered", "single_buffered"])
@pytest.mark.parametrize("math_delay,writer_delay", [(0, 0), (64, 0), (0, 256), (64, 256)])
def test_dst_math_clear_stress(device, enabled_program_cache, full_sync, math_delay, writer_delay):
    _run_dst_math_clear(device, full_sync, math_delay, writer_delay, iterations=64, repeats=128)


def _run_dst_math_clear(device, full_sync, math_delay, writer_delay, *, iterations, repeats):
    if device.arch() != ttnn.device.Arch.BLACKHOLE:
        pytest.skip("MATH-owned DST clearing addresses Blackhole's cross-half ZEROACC race")
    core = ttnn.CoreCoord(0, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    # generic_op requires an input, but this kernel produces its own values.
    unused_input = ttnn.from_torch(
        torch.zeros((1, 1, 32, 32)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    output = ttnn.empty((iterations * 2, 1, 32, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    cbs = [
        ttnn.CBDescriptor(
            total_size=pages * 4096,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32, page_size=4096)],
        )
        for index, pages in ((0, 2), (16, 4))
    ]
    runtime = ttnn.RuntimeArgs()
    runtime[0][0] = [output.buffer_address(), iterations, writer_delay]
    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=WRITER,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=ttnn.TensorAccessorArgs(output).get_compile_time_args(),
                runtime_args=runtime,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=COMPUTE,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=[iterations, math_delay],
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    dst_full_sync_en=full_sync,
                ),
            ),
        ],
        semaphores=[],
        cbs=cbs,
    )
    indices = torch.arange(iterations)
    expected_first = torch.where(indices % 4 < 2, 1 + indices.remainder(64).float() / 128, 0)
    for _ in range(repeats):
        ttnn.generic_op([unused_input, output], program)
        actual = ttnn.to_torch(output)[:, 0]
        torch.testing.assert_close(actual[0::2, 0, 0], expected_first, rtol=0, atol=0)
        torch.testing.assert_close(actual[1::2, 0, 0], torch.full((iterations,), 64.0), rtol=0, atol=0)
        assert torch.count_nonzero(actual[:, 8:, :]) == 0
