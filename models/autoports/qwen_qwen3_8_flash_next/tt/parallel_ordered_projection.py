# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Parallel independent K-blocks of the exact ordered QSA indexer projection.

Q1 uses five 512-wide blocks (100 cores); Q2..32 use 1024/1024/512
(60 cores). Each worker computes a single [32,32] FP32 partial. A second
20-core program combines partials in the original order using direct FP32
unpack-to-DST and SFPU add, followed by explicit device BF16 typecast.
Reader broadcasts only valid rows and zeroes ping-pong padding once. The default
compute uses the existing fused indexer SFPU face masks (<=4, <=8, <=16, full)
and initializes identical BF16 input copying once. There is no host tensor
conversion, runtime persistent cache, or mesh CCL.
The public input/output and setup weight contracts match OrderedIndexerProjection.

The extra DRAM partials are 400 KiB (Q1) or 240 KiB (Q2..32) per device.
Scratch is at most 140 KiB/core. All program allocations are invocation-local;
prepare all supported signatures before capturing a trace, as for the original.
This candidate needs real-device exact-output and timing validation.
"""

from pathlib import Path

_KERNEL_ROOT = Path(__file__).parent / "kernels"


def parallel_ordered_indexer_projection(
    hidden_states,
    transposed_weight,
    *,
    output=None,
    block_size=None,
    fp32_output=False,
    direct_bf16_pack=False,
    valid_row_compute=True,
    copy_once=True,
):
    import ttnn as t

    if not isinstance(hidden_states, t.Tensor) or not isinstance(transposed_weight, t.Tensor):
        raise ValueError("Expected device tensors")
    count = hidden_states.shape[-2]
    if tuple(hidden_states.shape) != (1, 1, count, 2560) or not 1 <= count <= 32:
        raise ValueError("Expected BF16 input [1,1,1..32,2560]")
    if tuple(transposed_weight.shape) != (1, 1, 2560, 640):
        raise ValueError("Expected pretransposed BF16 weight [1,1,2560,640]")
    block_size = (512 if count == 1 else 1024) if block_size is None else block_size
    if type(block_size) is not int or block_size not in (512, 1024):
        raise ValueError("block_size must be 512 or 1024")
    if type(fp32_output) is not bool or type(direct_bf16_pack) is not bool:
        raise ValueError("Diagnostic mode flags must be bool")
    if type(valid_row_compute) is not bool or type(copy_once) is not bool:
        raise ValueError("Row-compute and copy-once flags must be bool")
    if direct_bf16_pack:
        raise ValueError("This path requires explicit FP32-to-BF16 typecast")
    device = hidden_states.device()
    grid = device.compute_with_storage_grid_size()
    blocks = (2560 + block_size - 1) // block_size
    if grid.x * grid.y < blocks * 20:
        raise ValueError(f"Parallel ordered projection requires {blocks * 20} compute cores")

    def validate(value, padded, dtype):
        if (
            not t.is_tensor_storage_on_device(value)
            or not value.is_allocated()
            or value.dtype != dtype
            or value.layout != t.TILE_LAYOUT
            or value.memory_config() != t.DRAM_MEMORY_CONFIG
            or tuple(value.padded_shape) != padded
            or value.device() != device
        ):
            raise ValueError("Expected exact interleaved DRAM tile geometry on the input device")

    validate(hidden_states, (1, 1, 32, 2560), t.bfloat16)
    validate(transposed_weight, (1, 1, 2560, 640), t.bfloat16)

    def empty(shape, dtype):
        return t.empty(shape, dtype=dtype, layout=t.TILE_LAYOUT, device=device, memory_config=t.DRAM_MEMORY_CONFIG)

    output_dtype = t.float32 if fp32_output else t.bfloat16
    if output is None:
        output = empty((1, 1, count, 640), output_dtype)
    if not isinstance(output, t.Tensor) or tuple(output.shape) != (1, 1, count, 640):
        raise ValueError("Output shape must be [1,1,Q,640]")
    validate(output, (1, 1, 32, 640), output_dtype)
    if output.buffer_address() in (hidden_states.buffer_address(), transposed_weight.buffer_address()):
        raise ValueError("Output may not alias an input")
    partials = empty((1, blocks, 32, 640), t.float32)
    fp32_result = output if fp32_output else empty((1, 1, count, 640), t.float32)

    def core_set(size):
        cores = [t.CoreCoord(i % grid.x, i // grid.x) for i in range(size)]
        return cores, t.CoreRangeSet([t.CoreRange(core, core) for core in cores])

    def config(exact_input=False):
        cfg = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
            math_approx_mode=False,
        )
        if exact_input:
            modes = [t.UnpackToDestMode.Default] * 64
            modes[0] = t.UnpackToDestMode.UnpackToDestFp32
            cfg.unpack_to_dest_mode = modes
        return cfg

    def cb(ranges, index, tiles, dtype):
        tile_bytes = 4096 if dtype == t.float32 else 2048
        return t.CBDescriptor(
            total_size=tiles * tile_bytes,
            core_ranges=ranges,
            format_descriptors=[t.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=tile_bytes)],
        )

    def launch(tensors, ranges, specs, cbs):
        kernels = [
            t.KernelDescriptor(
                kernel_source=str(_KERNEL_ROOT / filename),
                source_type=t.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=ranges,
                compile_time_args=ct,
                runtime_args=rt,
                config=cfg,
            )
            for filename, ct, rt, cfg in specs
        ]
        t.generic_op(tensors, t.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))

    cores, ranges = core_set(blocks * 20)
    reader, writer, compute = t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs()
    for index, core in enumerate(cores):
        block, column = divmod(index, 20)
        begin = block * block_size
        width = min(block_size, 2560 - begin)
        reader[core.x][core.y] = [
            hidden_states.buffer_address(),
            transposed_weight.buffer_address(),
            column,
            count,
            begin // 32,
            width // 32,
        ]
        writer[core.x][core.y] = [partials.buffer_address(), index]
        compute[core.x][core.y] = [width, count]
    read_ct = []
    for value in (hidden_states, transposed_weight):
        read_ct.extend(t.TensorAccessorArgs(value).get_compile_time_args())
    write_ct = [4096, *t.TensorAccessorArgs(partials).get_compile_time_args()]
    launch(
        [hidden_states, transposed_weight, partials],
        ranges,
        [
            ("parallel_ordered_projection_reader.cpp", read_ct, reader, t.ReaderConfigDescriptor()),
            ("parallel_ordered_projection_writer.cpp", write_ct, writer, t.WriterConfigDescriptor()),
            ("parallel_ordered_projection_compute.cpp", [int(copy_once), int(valid_row_compute)], compute, config()),
        ],
        [cb(ranges, i, n, t.bfloat16) for i, n in ((0, block_size // 32), (1, block_size // 32), (2, 2), (3, 2))]
        + [cb(ranges, 4, 1, t.float32)],
    )

    cores, ranges = core_set(20)
    reader, writer, compute = t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs()
    for column, core in enumerate(cores):
        reader[core.x][core.y] = [partials.buffer_address(), column, blocks]
        writer[core.x][core.y] = [fp32_result.buffer_address(), column]
        compute[core.x][core.y] = [blocks]
    launch(
        [partials, fp32_result],
        ranges,
        [
            (
                "parallel_ordered_projection_combine_reader.cpp",
                t.TensorAccessorArgs(partials).get_compile_time_args(),
                reader,
                t.ReaderConfigDescriptor(),
            ),
            (
                "parallel_ordered_projection_writer.cpp",
                [4096, *t.TensorAccessorArgs(fp32_result).get_compile_time_args()],
                writer,
                t.WriterConfigDescriptor(),
            ),
            ("parallel_ordered_projection_combine_compute.cpp", [], compute, config(exact_input=True)),
        ],
        [cb(ranges, 0, 2, t.float32), cb(ranges, 4, 1, t.float32)],
    )
    if not fp32_output:
        t.typecast(fp32_result, t.bfloat16, output_tensor=output)
    return output


class ParallelOrderedIndexerProjection:
    """Setup-only weight preparation; no mutable invocation state."""

    def __init__(self, loader, name="self_attn.indexer.index_qk_proj.weight"):
        import ttnn

        weight = loader.state_dict[name]
        if tuple(weight.shape) != (640, 2560):
            raise ValueError("Expected target indexer weight [640,2560]")
        self.weight = loader.tensor(weight.transpose(0, 1).contiguous().reshape(1, 1, 2560, 640), ttnn.bfloat16)

    def __call__(self, hidden_states):
        return parallel_ordered_indexer_projection(hidden_states, self.weight)
