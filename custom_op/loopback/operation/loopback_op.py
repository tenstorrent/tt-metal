# SPDX-License-Identifier: Apache-2.0
"""
Loopback operation: copies an fp32 tensor through a compute kernel using copy_tile.

Purpose: test the effect of unpack_to_dest on fp32 data fidelity.
- unpack_to_dest=True  → unpacker sends data directly to the 32-bit DST register.
                          fp32 values are preserved bit-exactly. Expected: output == input.
- unpack_to_dest=False → unpacker routes data through 19-bit SrcA/SrcB registers.
                          Mantissa bits are truncated. Expected: output != input on real HW.
                          The simulator may raise UndefinedBehavior instead.
"""

from pathlib import Path
import ttnn

_HERE = Path(__file__).parent
_READER_SRC = (_HERE / "kernels" / "dataflow" / "loopback_reader.cpp").read_text()
_WRITER_SRC = (_HERE / "kernels" / "dataflow" / "loopback_writer.cpp").read_text()
_COMPUTE_SRC = (_HERE / "kernels" / "compute" / "loopback_compute.cpp").read_text()

# fp32 tile: 32 × 32 × 4 bytes = 4096 bytes
_TILE_BYTES = 32 * 32 * 4
_TILES_PER_CB = 2  # double-buffer

_CB_IN = 0
_CB_OUT = 16


def loopback(input_tensor: ttnn.Tensor, unpack_to_dest: bool = True) -> ttnn.Tensor:
    """Copy input_tensor to a new output tensor through the compute kernel.

    Args:
        input_tensor:    fp32 tensor in TILE_LAYOUT on device.
        unpack_to_dest:  When True, use UnpackToDestFp32 for cb_in so fp32
                         data bypasses SrcA/SrcB and lands directly in DST.

    Returns:
        A new fp32 tensor on the same device.
    """
    assert input_tensor.dtype == ttnn.float32, "loopback expects float32 input"
    assert input_tensor.layout == ttnn.TILE_LAYOUT, "loopback expects TILE_LAYOUT"

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(input_tensor.shape),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
    )

    # --- number of tiles ---
    shape = input_tensor.shape
    num_tiles = 1
    for d in shape:
        num_tiles *= d
    num_tiles //= 32 * 32
    num_tiles = max(1, num_tiles)

    # --- single core ---
    core = ttnn.CoreCoord(0, 0)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # --- circular buffers ---
    cb_total = _TILES_PER_CB * _TILE_BYTES

    in_fmt = ttnn.CBFormatDescriptor(buffer_index=_CB_IN, data_format=ttnn.float32, page_size=_TILE_BYTES)
    out_fmt = ttnn.CBFormatDescriptor(buffer_index=_CB_OUT, data_format=ttnn.float32, page_size=_TILE_BYTES)

    in_cb_desc = ttnn.CBDescriptor(total_size=cb_total, core_ranges=grid, format_descriptors=[in_fmt])
    out_cb_desc = ttnn.CBDescriptor(total_size=cb_total, core_ranges=grid, format_descriptors=[out_fmt])

    # --- compile-time args ---
    reader_ct = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    writer_ct = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    compute_ct = [num_tiles]

    # --- runtime args: ttnn.RuntimeArgs indexed by [x][y] core coordinate ---
    x, y = core.x, core.y

    reader_rt = ttnn.RuntimeArgs()
    reader_rt[x][y] = [input_tensor.buffer_address(), num_tiles]

    writer_rt = ttnn.RuntimeArgs()
    writer_rt[x][y] = [output_tensor.buffer_address(), num_tiles]

    compute_rt = ttnn.RuntimeArgs()
    compute_rt[x][y] = []

    # --- compute config ---
    # NUM_CIRCULAR_BUFFERS = 32 on Wormhole/Blackhole
    modes = [ttnn.UnpackToDestMode.Default] * 32
    if unpack_to_dest:
        modes[_CB_IN] = ttnn.UnpackToDestMode.UnpackToDestFp32

    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.fp32_dest_acc_en = True
    compute_config.math_fidelity = ttnn.MathFidelity.HiFi4
    compute_config.math_approx_mode = False
    compute_config.unpack_to_dest_mode = modes

    # --- kernel descriptors ---
    reader_k = ttnn.KernelDescriptor(
        kernel_source=_READER_SRC,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=grid,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_k = ttnn.KernelDescriptor(
        kernel_source=_WRITER_SRC,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=grid,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_k = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_SRC,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=grid,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_config,
    )

    prog = ttnn.ProgramDescriptor(
        kernels=[reader_k, compute_k, writer_k],
        semaphores=[],
        cbs=[in_cb_desc, out_cb_desc],
    )

    return ttnn.generic_op([input_tensor, output_tensor], prog)
