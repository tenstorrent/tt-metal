# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# The tt-metal half of the emule-only CB cap above silicon (docs/cb-fantasy-mode.md in
# tt-emule). Skipped unless TT_EMULE_NUM_CBS raised the cap, so this is inert in normal runs.
#
# Three separately-compiled places size a CB array on this side: NUM_CIRCULAR_BUFFERS (the
# std::arrays in CircularBufferConfig), the emule headers' __EMULE_CTX_MAX_CBS (per-core and
# per-fiber CB state), and the same define reaching JIT kernels through the runner. Agreement is
# "by construction" — one build-wide CMake define plus the runner's emitted -D — but that is a
# claim about build wiring, and a disagreement corrupts memory rather than failing. Only a
# program that actually drives a CB index above the silicon cap proves it.
#
# Required env: TT_EMULE_NUM_CBS=<n above the arch cap>, on a build configured with
# -DTT_METAL_USE_EMULE=ON -DEMULE_CB_CEILING=<n or more>.

import os

import pytest
import torch

import ttnn

TILE_BYTES = 2 * 32 * 32  # bfloat16


def _silicon_cap() -> int:
    """What the hardware really has. Distinct from get_arch_num_circular_buffers, which returns the
    raised cap when the mode is on — comparing against that would make the test vacuous."""
    return ttnn._ttnn.device.get_silicon_num_circular_buffers()


def _num_pairs() -> int:
    """Fill the whole budget: each pair is one input CB and one output CB, so this spans indices
    0 .. 2*n-1. Derived rather than fixed so the top slot is always exercised, whatever the
    budget, and so a budget too small to clear the silicon cap skips instead of overflowing."""
    return ttnn._ttnn.device.get_arch_num_circular_buffers() // 2


def _skip_unless_cap_raised() -> tuple[int, int]:
    """Setting TT_EMULE_NUM_CBS at or below the arch's own cap is legal and changes nothing, so it
    is a skip rather than a failure. Needs a device, hence a call rather than a decorator."""
    silicon_cap = _silicon_cap()
    reported = ttnn._ttnn.device.get_arch_num_circular_buffers()
    if reported <= silicon_cap:
        pytest.skip(f"the reported cap ({reported}) is not above the arch's ({silicon_cap})")
    return silicon_cap, reported


_READER_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pairs = get_arg_val<uint32_t>(1);

    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(src_args, src_addr);

    Noc noc;
    for (uint32_t i = 0; i < num_pairs; ++i) {
        DataflowBuffer dfb(static_cast<uint16_t>(i));
        dfb.reserve_back(1);
        noc.async_read(s, dfb, dfb.get_entry_size(), {.page_id = i}, {});
        noc.async_read_barrier();
        dfb.push_back(1);
    }
}
"""

_WRITER_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pairs = get_arg_val<uint32_t>(1);

    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    for (uint32_t i = 0; i < num_pairs; ++i) {
        DataflowBuffer dfb(static_cast<uint16_t>(num_pairs + i));
        dfb.wait_front(1);
        noc.async_write(dfb, s, dfb.get_entry_size(), {}, {.page_id = i});
        noc.async_writes_flushed();
        dfb.pop_front(1);
    }
    noc.async_write_barrier();
}
"""

# copy_tile from in-CB i to out-CB num_pairs+i. Every CB shares one data format, so a single
# init covers all of them; what varies is only the index, which is the point.
_COMPUTE_SOURCE = """
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t num_pairs = get_compile_time_arg_val(0);

    init_sfpu(0, num_pairs);
    for (uint32_t i = 0; i < num_pairs; ++i) {
        const uint32_t in_id = i;
        const uint32_t out_id = num_pairs + i;
        CircularBuffer buff_in(in_id);
        CircularBuffer buff_out(out_id);
        buff_in.wait_front(1);
        buff_out.reserve_back(1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(in_id);
        copy_tile(in_id, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_id);
        tile_regs_release();
        buff_out.push_back(1);
        buff_in.pop_front(1);
    }
}
"""


def _make_cb(index: int, core_ranges) -> "ttnn.CBDescriptor":
    """One page, so each CB is the smallest object that still has its own address."""
    return ttnn.CBDescriptor(
        total_size=TILE_BYTES,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=index,
                data_format=ttnn.bfloat16,
                page_size=TILE_BYTES,
            )
        ],
    )


@pytest.mark.skipif(
    not os.environ.get("TT_EMULE_NUM_CBS"),
    reason="needs TT_EMULE_NUM_CBS to raise the CB cap above silicon",
)
def test_cb_indices_above_the_silicon_cap_round_trip(device):
    """One core, a CB per slot in the whole budget, so the top indices are above the arch's cap.

    Each tile is distinct, so a CB whose address, page size, or page count came out wrong at a
    high index shows up as a mismatched tile rather than as a silent pass.
    Both dataflow kernels take their tensor address from a runtime arg, which lives past the CB
    region in L1 — if the CB region were sized from a truncated 64-bit firmware mask instead of
    the kernel group's exact extent, those args would be overlapped and the read would fault or
    return garbage. So the round-trip also covers the L1 layout.
    """
    silicon_cap, _ = _skip_unless_cap_raised()
    num_pairs = _num_pairs()
    num_cbs = 2 * num_pairs
    if num_cbs <= silicon_cap:
        pytest.skip(f"a budget of {num_cbs} CBs does not exceed the arch cap of {silicon_cap}")

    torch.manual_seed(1234)
    shape = [1, num_pairs, 32, 32]
    data = torch.randn(shape, dtype=torch.bfloat16)

    src = ttnn.from_torch(
        data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    dst = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cbs = [_make_cb(i, core_ranges) for i in range(num_cbs)]

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[0][0] = [src.buffer_address(), num_pairs]
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[0][0] = [dst.buffer_address(), num_pairs]

    reader = ttnn.KernelDescriptor(
        kernel_source=_READER_SOURCE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=ttnn.TensorAccessorArgs(src).get_compile_time_args(),
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=_WRITER_SOURCE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=ttnn.TensorAccessorArgs(dst).get_compile_time_args(),
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_SOURCE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=[num_pairs],
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    out = ttnn.generic_op([src, dst], program)
    ttnn.synchronize_device(device)

    got = ttnn.to_torch(out).to(torch.bfloat16)
    assert torch.equal(got, data), (
        f"tile round-trip through {num_cbs} CBs (indices 0..{num_cbs - 1}, arch cap {silicon_cap}) "
        f"did not match; a CB-count disagreement between the host, the emule runtime, and the JIT "
        f"kernels puts a high CB's geometry at the wrong words"
    )


@pytest.mark.skipif(
    not os.environ.get("TT_EMULE_NUM_CBS"),
    reason="needs TT_EMULE_NUM_CBS to raise the CB cap above silicon",
)
def test_reported_cap_honours_the_request(device):
    """The env var is the single enforced-cap decision point; nothing else may reinterpret it."""
    _, reported = _skip_unless_cap_raised()
    requested = int(os.environ["TT_EMULE_NUM_CBS"])
    assert reported >= requested, (
        f"TT_EMULE_NUM_CBS={requested} but the device reports {reported}; this build has no "
        f"EMULE_CB_CEILING, or a lower one than requested"
    )
