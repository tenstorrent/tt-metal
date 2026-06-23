# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared harness for eltwise_chain helper tests.

Factored out of test_chain_reconfig.py so every helper-test suite (operand-kind,
DEST/block, out-of-bounds, lifecycle, ...) builds its generic_op programs the same
way instead of copy-pasting the builders.

The reader/writer dataflow kernels live under tests/chain_reconfig/ (they are generic
N-input / N-output streamers, reused unchanged). Each suite supplies its own compute
kernel path.
"""

import torch
import ttnn

# Tile size in bytes per dtype (tt_metal/api/tt-metalium/tt_backend_api_types.hpp).
DTYPE_TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.float32: 4096,
    ttnn.bfloat8_b: 1088,
}

# Reusable generic dataflow kernels (live alongside the reconfig suite).
DATAFLOW_DIR = "ttnn/cpp/ttnn/kernel_lib/tests/chain_reconfig"
READER = {n: f"{DATAFLOW_DIR}/reader_{n}_input{'s' if n > 1 else ''}.cpp" for n in (1, 2, 3, 4)}
WRITER_1OUT = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"
WRITER_2OUT = f"{DATAFLOW_DIR}/writer_2_outputs.cpp"


def torch_dtype_for(ttnn_dtype):
    if ttnn_dtype == ttnn.float32:
        return torch.float32
    if ttnn_dtype == ttnn.bfloat16:
        return torch.bfloat16
    # bfp8_b host-side stays float32 so quantization happens during ttnn.from_torch.
    return torch.float32


def make_input(shape, ttnn_dtype, device, seed, scale=0.5, bias=0.25):
    torch.manual_seed(seed)
    torch_t = torch.randn(shape, dtype=torch.float32) * scale + bias
    torch_t = torch_t.to(torch_dtype_for(ttnn_dtype))
    tt_t = ttnn.from_torch(
        torch_t,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return torch_t, tt_t


def pcc_threshold(dtypes):
    if any(d == ttnn.bfloat8_b for d in dtypes):
        return 0.99
    if any(d == ttnn.float32 for d in dtypes):
        return 0.999
    return 0.9999


def single_core_grid():
    core = ttnn.CoreCoord(0, 0)
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def cb_descriptor(cb_id, dtype, num_pages, core_grid):
    page_size = DTYPE_TILE_BYTES[dtype]
    fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page_size)
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=core_grid,
        format_descriptors=[fmt],
    )


def build_reader_kernel(input_tensors, num_tiles, core_grid):
    """N-input streamer; N inferred from len(input_tensors). Pushes num_tiles to each CB c_0..c_{N-1}."""
    n = len(input_tensors)
    cta = []
    for t in input_tensors:
        cta.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [t.buffer_address() for t in input_tensors] + [num_tiles, 0]
    return ttnn.KernelDescriptor(
        kernel_source=READER[n],
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=cta,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )


def build_reader_asym_kernel(input_tensors, counts, core_grid):
    """2-input reader pushing counts[0] tiles to c_0 and counts[1] to c_1 (asymmetric)."""
    assert len(input_tensors) == 2 and len(counts) == 2
    cta = []
    for t in input_tensors:
        cta.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [t.buffer_address() for t in input_tensors] + [counts[0], counts[1]]
    return ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/kernel_lib/tests/axes/reader_2_asym.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=cta,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )


def build_compute_kernel_rt(
    kernel_source, compile_time_args, runtime_args, core_grid, fp32_dest_acc_en=False, dst_full_sync_en=False
):
    """Compute kernel with both compile-time AND runtime args (e.g. a TileOffset base)."""
    rt = ttnn.RuntimeArgs()
    rt[0][0] = list(runtime_args)
    return ttnn.KernelDescriptor(
        kernel_source=kernel_source,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=list(compile_time_args),
        runtime_args=rt,
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en),
    )


def build_writer_1out_kernel(output_tensor, num_tiles, core_grid):
    cta = [16] + ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [output_tensor.buffer_address(), num_tiles, 0]
    return ttnn.KernelDescriptor(
        kernel_source=WRITER_1OUT,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=cta,
        runtime_args=rt,
        config=ttnn.WriterConfigDescriptor(),
    )


def build_writer_2out_kernel(output_tensors, num_tiles, core_grid):
    cta = []
    for t in output_tensors:
        cta.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [t.buffer_address() for t in output_tensors] + [num_tiles, 0]
    return ttnn.KernelDescriptor(
        kernel_source=WRITER_2OUT,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=cta,
        runtime_args=rt,
        config=ttnn.WriterConfigDescriptor(),
    )


def build_compute_kernel(
    kernel_source,
    compile_time_args,
    core_grid,
    fp32_dest_acc_en=False,
    dst_full_sync_en=False,
    math_fidelity=None,
):
    """kernel_source is a full repo-relative path. compile_time_args is a list of uint32.
    math_fidelity is an optional ttnn.MathFidelity (defaults to the descriptor default when None)."""
    cfg_kwargs = dict(fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en)
    if math_fidelity is not None:
        cfg_kwargs["math_fidelity"] = math_fidelity
    return ttnn.KernelDescriptor(
        kernel_source=kernel_source,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=list(compile_time_args),
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(**cfg_kwargs),
    )
