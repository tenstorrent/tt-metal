# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Standalone SwiGLU SFPU accuracy test.

Tests the SwiGLU activation kernel in isolation, without matmul or weight
quantization, by feeding bf16 gate/up tiles directly through the SFPU and
comparing against a PyTorch reference.

Formula:
    gate_clamped = clamp(gate, max=7.0)
    up_clamped   = clamp(up, min=-7.0, max=7.0)
    result       = (up_clamped + 1) * gate_clamped * sigmoid(1.702 * gate_clamped)
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------
def swiglu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    gate = torch.clamp(gate, max=clamp_limit)
    up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
    return (up + 1.0) * gate * torch.sigmoid(alpha * gate)


# ---------------------------------------------------------------------------
# Standalone SFPU test via generic_op
# ---------------------------------------------------------------------------
# Inline reader kernel: reads gate tiles into CB0 and up tiles into CB1
# from two DRAM-interleaved tensors.
BINARY_READER_SOURCE = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t gate_addr = get_arg_val<uint32_t>(0);
    const uint32_t up_addr   = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t start_id  = get_arg_val<uint32_t>(3);

    constexpr auto gate_args = TensorAccessorArgs<0>();
    constexpr auto up_args   = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_gate = 0;
    constexpr uint32_t cb_up   = 1;

    const uint32_t gate_page_bytes = get_tile_size(cb_gate);
    const uint32_t up_page_bytes   = get_tile_size(cb_up);

    const auto gate_s = TensorAccessor(gate_args, gate_addr, gate_page_bytes);
    const auto up_s   = TensorAccessor(up_args,   up_addr,   up_page_bytes);

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(cb_gate, 1);
        noc_async_read_page(i, gate_s, get_write_ptr(cb_gate));
        noc_async_read_barrier();
        cb_push_back(cb_gate, 1);

        cb_reserve_back(cb_up, 1);
        noc_async_read_page(i, up_s, get_write_ptr(cb_up));
        noc_async_read_barrier();
        cb_push_back(cb_up, 1);
    }
}
"""

# Path to the compute kernel (lives next to swiglu_sfpu.h so the include resolves)
COMPUTE_KERNEL_PATH = "ttnn/cpp/ttnn/operations/experimental/moe/device/kernels/test_swiglu_compute.cpp"

# Reuse the standard unary writer
WRITER_KERNEL_PATH = (
    "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"
)

# bf16 tile page size
BF16_TILE_PAGE_SIZE = 2 * 1024  # 32 x 32 x 2 bytes


def run_swiglu_sfpu_test(device, num_tiles):
    """
    Run the SwiGLU SFPU in isolation on device and compare against PyTorch.

    Creates bf16 gate and up tensors, feeds them through the standalone
    SwiGLU compute kernel (no matmul, no weight quantization), and checks
    PCC against the reference.
    """
    shape = [1, num_tiles, 32, 32]

    # Create random bf16 inputs in a range that exercises clamping
    torch.manual_seed(42)
    torch_gate = (torch.randn(shape) * 4.0).to(torch.bfloat16)  # some values > 7
    torch_up = (torch.randn(shape) * 4.0).to(torch.bfloat16)  # some values outside [-7, 7]

    # PyTorch reference (in float32 for precision)
    torch_ref = swiglu_reference(torch_gate.float(), torch_up.float())

    # Send to device
    dram_config = ttnn.DRAM_MEMORY_CONFIG
    gate_tensor = ttnn.from_torch(
        torch_gate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram_config
    )
    up_tensor = ttnn.from_torch(
        torch_up, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram_config
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram_config
    )

    # ---- Core grid: single core for simplicity ----
    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ---- Circular buffers ----
    cb_total_size = 2 * BF16_TILE_PAGE_SIZE  # double buffer

    gate_cb_format = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=BF16_TILE_PAGE_SIZE)
    up_cb_format = ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat16, page_size=BF16_TILE_PAGE_SIZE)
    out_cb_format = ttnn.CBFormatDescriptor(buffer_index=16, data_format=ttnn.bfloat16, page_size=BF16_TILE_PAGE_SIZE)

    gate_cb = ttnn.CBDescriptor(total_size=cb_total_size, core_ranges=core_range, format_descriptors=[gate_cb_format])
    up_cb = ttnn.CBDescriptor(total_size=cb_total_size, core_ranges=core_range, format_descriptors=[up_cb_format])
    out_cb = ttnn.CBDescriptor(total_size=cb_total_size, core_ranges=core_range, format_descriptors=[out_cb_format])

    # ---- Reader kernel (inline source, reads two tensors) ----
    gate_cta = list(ttnn.TensorAccessorArgs(gate_tensor).get_compile_time_args())
    up_cta = list(ttnn.TensorAccessorArgs(up_tensor).get_compile_time_args())
    reader_compile_time_args = gate_cta + up_cta

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[0][0] = [gate_tensor.buffer_address(), up_tensor.buffer_address(), num_tiles, 0]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=BINARY_READER_SOURCE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_range,
        compile_time_args=reader_compile_time_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- Writer kernel (reuse standard unary writer) ----
    writer_compile_time_args = [16]  # output CB index
    writer_compile_time_args.extend(list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()))

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[0][0] = [output_tensor.buffer_address(), num_tiles, 0]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=WRITER_KERNEL_PATH,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=writer_compile_time_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---- Compute kernel (SwiGLU SFPU) ----
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=COMPUTE_KERNEL_PATH,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=[num_tiles],
        defines=[],
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(math_approx_mode=True),
    )

    # ---- Program descriptor ----
    program = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[gate_cb, up_cb, out_cb],
    )

    # ---- Execute ----
    io_tensors = [gate_tensor, up_tensor, output_tensor]
    result = ttnn.generic_op(io_tensors, program)

    # ---- Compare ----
    torch_output = ttnn.to_torch(result)

    passing, pcc_value = comp_pcc(torch_ref, torch_output)

    return passing, pcc_value


PCC_THRESHOLD = 0.999  # High threshold: no weight quantization error


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW},
            id="dispatch_row",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_tiles", [1, 4, 16], ids=["1tile", "4tiles", "16tiles"])
def test_swiglu_sfpu(device, num_tiles):
    """
    Test SwiGLU SFPU in isolation.

    Feeds bf16 gate/up tiles directly through the SFPU kernel (no matmul,
    no weight quantization) and compares against PyTorch reference.
    Expects PCC > 0.999 since the only error source is bf16 SFPU precision.
    """
    passing, pcc_value = run_swiglu_sfpu_test(device, num_tiles)
    logger.info(f"SwiGLU SFPU standalone ({num_tiles} tiles): PCC = {pcc_value}")
    assert pcc_value >= PCC_THRESHOLD, f"PCC {pcc_value} < {PCC_THRESHOLD}"
