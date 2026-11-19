# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact Q1 SFPU projection for selected GR-up and GDN-output roles.

Unchanged O-by-K resident weights, R2/R4/R8/R16 output parallelism, separate FP32
multiply, tile0 seed, sequential K-tile addition, standard SFPU row reduction.
Two native programs use invocation-local compact scratch and fully overwrite
output padding. No persistent helper cache, CCL, host conversion, or K split.
Prepare each signature before capture; other roles retain their selected path.
"""

from pathlib import Path

_KERNEL_ROOT = Path(__file__).parent / "kernels"


def exact_projection_q1(
    value, weight, *, rows_per_worker=8, num_cores=None, raw_fp32=False, output=None, zero_once=False
):
    """BF16/FP32 [1,1,1,K] @ unchanged [1,1,O,K]; explicit BF16 output cast.

    raw_fp32=True exposes pre-cast bits for comparison with selected multiply
    plus accurate sum. Q>1 is deliberately rejected; a caller must retain its
    existing prefill implementation. O and K must be positive multiples of32.
    zero_once=True opts into initializing both physical input-CB pages once
    per invocation. The default conservatively clears each page on every use.
    R2 uses the existing full four-row SFPU band (two iterations), with two
    zero-padded weight rows; it does not use an invalid one-iteration mask.
    Exactness acceptance covers finite model operands; nonfinite payload
    propagation is not established by output-row regrouping.
    """
    import ttnn as t

    if not isinstance(value, t.Tensor) or not isinstance(weight, t.Tensor):
        raise ValueError("Expected device activation and resident weight tensors")
    if len(weight.shape) != 4 or tuple(weight.shape)[:2] != (1, 1):
        raise ValueError("Expected weight [1,1,O,K]")
    o, k = tuple(weight.shape)[-2:]
    if o < 32 or k < 32 or o % 32 or k % 32 or tuple(value.shape) != (1, 1, 1, k):
        raise ValueError("Q1 only, with tile-aligned positive O/K")
    if type(rows_per_worker) is not int or rows_per_worker not in (2, 4, 8, 16) or type(raw_fp32) is not bool:
        raise ValueError("rows_per_worker must be2/4/8/16 and raw_fp32 must be bool")
    if type(zero_once) is not bool:
        raise ValueError("zero_once must be bool")
    device = value.device()
    if tuple(device.shape) != (1, 4):
        raise ValueError("Experimental exact projection requires mesh1x4")
    if not any(name in str(device.arch()).lower() for name in ("wormhole", "blackhole")):
        raise ValueError("Exact FP32 unpack candidate supports Wormhole/Blackhole only")

    def validate(x, shape, dtype=None):
        if (
            not isinstance(x, t.Tensor)
            or not t.is_tensor_storage_on_device(x)
            or not x.is_allocated()
            or x.device() != device
            or x.layout != t.TILE_LAYOUT
            or x.memory_config() != t.DRAM_MEMORY_CONFIG
            or tuple(x.padded_shape) != shape
            or x.dtype not in (t.bfloat16, t.float32)
            or (dtype is not None and x.dtype != dtype)
        ):
            raise ValueError("Expected allocated BF16/FP32 interleaved DRAM TILE tensor on the same mesh")
        tile = x.get_tile()
        if tuple(tile.tile_shape) != (32, 32) or tile.transpose_within_face or tile.transpose_of_faces:
            raise ValueError("Expected standard untransposed 32x32 tiles")

    validate(value, (1, 1, 32, k))
    validate(weight, (1, 1, o, k))
    groups, kt = o // rows_per_worker, k // 32
    grid = device.compute_with_storage_grid_size()
    available = min(groups, grid.x * grid.y)
    num_cores = available if num_cores is None else num_cores
    if type(num_cores) is not int or not 1 <= num_cores <= available:
        raise ValueError(f"num_cores must be in1..{available}")
    dtype = t.float32 if raw_fp32 else t.bfloat16
    if output is None:
        output = t.empty(
            (1, 1, 1, o), dtype=dtype, layout=t.TILE_LAYOUT, device=device, memory_config=t.DRAM_MEMORY_CONFIG
        )
    validate(output, (1, 1, 32, o), dtype)
    if tuple(output.shape) != (1, 1, 1, o) or output.buffer_address() in (
        value.buffer_address(),
        weight.buffer_address(),
    ):
        raise ValueError("Output must have logical shape[1,1,1,O] and may not alias inputs")
    raw = (
        output
        if raw_fp32
        else t.empty(
            (1, 1, 1, o), dtype=t.float32, layout=t.TILE_LAYOUT, device=device, memory_config=t.DRAM_MEMORY_CONFIG
        )
    )
    scratch = t.empty(
        (1, 1, groups, 32),
        dtype=t.float32,
        layout=t.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=t.DRAM_MEMORY_CONFIG,
    )

    def core_set(n):
        cores = [t.CoreCoord(i % grid.x, i // grid.x) for i in range(n)]
        return cores, t.CoreRangeSet([t.CoreRange(c, c) for c in cores])

    def cb(ranges, index, count, dtype, page=None):
        page = page or (4096 if dtype == t.float32 else 2048)
        return t.CBDescriptor(
            total_size=count * page,
            core_ranges=ranges,
            format_descriptors=[t.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )

    def launch(tensors, ranges, specs, cbs):
        t.generic_op(
            tensors,
            t.ProgramDescriptor(
                kernels=[
                    t.KernelDescriptor(
                        kernel_source=str(_KERNEL_ROOT / filename),
                        source_type=t.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=ranges,
                        compile_time_args=ct,
                        runtime_args=rt,
                        config=cfg,
                    )
                    for filename, ct, rt, cfg in specs
                ],
                cbs=cbs,
                semaphores=[],
            ),
        )

    config = t.ComputeConfigDescriptor(
        math_fidelity=t.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=False, math_approx_mode=False
    )
    modes = [t.UnpackToDestMode.Default] * 64
    for index, tensor in ((2, value), (3, weight)):
        if tensor.dtype == t.float32:
            modes[index] = t.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    cores, ranges = core_set(num_cores)
    reader, writer, compute = t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs()
    for i, c in enumerate(cores):
        reader[c.x][c.y] = [value.buffer_address(), weight.buffer_address(), i, num_cores, groups, kt]
        writer[c.x][c.y] = [scratch.buffer_address(), i, num_cores, groups]
        compute[c.x][c.y] = [i, num_cores, groups, kt]
    read_ct = [
        rows_per_worker,
        4 if value.dtype == t.float32 else 2,
        4 if weight.dtype == t.float32 else 2,
        int(zero_once),
    ]
    for x in (value, weight):
        read_ct.extend(t.TensorAccessorArgs(x).get_compile_time_args())
    launch(
        [value, weight, scratch],
        ranges,
        [
            ("exact_projection_q1_reader.cpp", read_ct, reader, t.ReaderConfigDescriptor()),
            (
                "exact_projection_q1_writer.cpp",
                [rows_per_worker, *t.TensorAccessorArgs(scratch).get_compile_time_args()],
                writer,
                t.WriterConfigDescriptor(),
            ),
            ("exact_projection_q1_compute.cpp", [rows_per_worker], compute, config),
        ],
        [
            cb(ranges, 2, 2, value.dtype),
            cb(ranges, 3, 2, weight.dtype),
            cb(ranges, 4, 2, t.float32),
            cb(ranges, 5, 1, t.float32, 128),
        ],
    )
    assemble_cores = min(o // 32, grid.x * grid.y)
    cores, ranges = core_set(assemble_cores)
    assemble = t.RuntimeArgs()
    for i, c in enumerate(cores):
        assemble[c.x][c.y] = [scratch.buffer_address(), raw.buffer_address(), i, assemble_cores, o // 32]
    ct = [
        rows_per_worker,
        *t.TensorAccessorArgs(scratch).get_compile_time_args(),
        *t.TensorAccessorArgs(raw).get_compile_time_args(),
    ]
    launch(
        [scratch, raw],
        ranges,
        [
            ("exact_projection_q1_assemble.cpp", ct, assemble, t.ReaderConfigDescriptor()),
        ],
        [cb(ranges, 0, 1, t.float32, 128), cb(ranges, 1, 1, t.float32)],
    )
    if not raw_fp32:
        t.typecast(raw, t.bfloat16, output_tensor=output)
    return output
