# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Test: cb_descriptor_from_sharded_tensor address_offset preserved on cache hit.

Regression test for https://github.com/tenstorrent/tt-metal/issues/42072.
Verifies that generic_op correctly applies CBDescriptor.address_offset when
rebinding CB buffers on cached program reuse.
"""

import pytest
import torch
from loguru import logger

import ttnn

KERNEL_SOURCE = """
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t output_cb = get_compile_time_arg_val(1);
    constexpr uint32_t num_pages = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);

    // Input CB is tensor-backed (pre-filled via setup_sharded_buffer)
    cb_reserve_back(input_cb, num_pages);
    cb_push_back(input_cb, num_pages);

    // Copy input -> output via local NOC read
    cb_wait_front(input_cb, num_pages);
    uint32_t src_addr = get_read_ptr(input_cb);

    cb_reserve_back(output_cb, num_pages);
    uint32_t dst_addr = get_write_ptr(output_cb);

    noc_async_read(get_noc_addr(src_addr), dst_addr, num_pages * page_size);
    noc_async_read_barrier();

    cb_push_back(output_cb, num_pages);
    cb_pop_front(input_cb, num_pages);
}
"""

K = 1024
PAGE_SIZE = 64  # 1x32 bfloat16 tile
NUM_PAGES = K * 2 // PAGE_SIZE  # 32
REGION_BYTES = NUM_PAGES * PAGE_SIZE  # 2048


@pytest.mark.parametrize("device", [(1,)], indirect=True)
def test_cb_address_offset_preserved_on_cache_hit(device):
    """generic_op must preserve CBDescriptor.address_offset across cached invocations."""
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    # Fused buffer: region A (offset 0) = 1.0, region B (offset REGION_BYTES) = 2.0
    region_a = torch.ones(1, K, dtype=torch.bfloat16)
    region_b = torch.full((1, K), 2.0, dtype=torch.bfloat16)
    raw = torch.cat([region_a.view(-1), region_b.view(-1)]).view(torch.uint8)
    buf = raw.view(torch.int32).unsqueeze(0)
    n_uint32 = buf.shape[1]

    fused = ttnn.from_torch(
        buf,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core, (1, n_uint32), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )

    out_tensor = ttnn.from_torch(
        torch.zeros(1, n_uint32 // 2, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core, (1, n_uint32 // 2), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )

    def build_pd(offset):
        in_cb = ttnn.cb_descriptor_from_sharded_tensor(
            0, fused, address_offset=offset, total_size=REGION_BYTES, core_ranges=core
        )
        in_cb.format_descriptors = [ttnn.CBFormatDescriptor(0, ttnn.bfloat16, PAGE_SIZE)]

        out_cb = ttnn.cb_descriptor_from_sharded_tensor(1, out_tensor, total_size=REGION_BYTES, core_ranges=core)
        out_cb.format_descriptors = [ttnn.CBFormatDescriptor(1, ttnn.bfloat16, PAGE_SIZE)]

        kernel = ttnn.KernelDescriptor(
            kernel_source=KERNEL_SOURCE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=core,
            compile_time_args=[0, 1, NUM_PAGES, PAGE_SIZE],
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default
            ),
        )
        return ttnn.ProgramDescriptor(cbs=[in_cb, out_cb], kernels=[kernel])

    def run(offset, label):
        pd = build_pd(offset)
        mesh_pd = ttnn.MeshProgramDescriptor()
        coord = ttnn.MeshCoordinate(0, 0)
        mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = pd
        ttnn.generic_op([fused, out_tensor], mesh_pd)
        ttnn.synchronize_device(device)
        result = ttnn.to_torch(out_tensor).view(torch.uint8).view(torch.bfloat16).float()
        val = result.mean().item()
        logger.info(f"{label}: mean={val:.4f}")
        return val

    # First call: offset=0, reads region A (1.0). Creates cached program.
    val_a = run(0, "offset=0 (expect 1.0)")
    assert abs(val_a - 1.0) < 0.01, f"First call wrong: {val_a}"

    # Second call: offset=REGION_BYTES, reads region B (2.0). Cache hit.
    # Bug: cached program's address_offset_ is stale (0), so it reads region A again.
    val_b = run(REGION_BYTES, f"offset={REGION_BYTES} (expect 2.0)")
    assert abs(val_b - 2.0) < 0.01, (
        f"address_offset lost on cache hit! "
        f"val_b={val_b:.4f} (expected 2.0, got {val_a:.4f}). "
        f"Cached program uses stale address_offset from first call."
    )
