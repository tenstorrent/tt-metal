# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``silu(a) * b`` as one model-local ``ttnn.generic_op`` (the SwiGLU product).

The stock path is ``ttnn.mul(a, b, input_tensor_a_activations=[SILU])``, whose SiLU costs
+33 us at bs1 (27 -> 60 us) and +336 us at bs32 (1254 -> 1590 us) on top of a multiply that
is already at DRAM roofline. This op streams tiles of a and b, computes SiLU on the DST tile
and multiplies b in with a dest-reuse FPU multiply, so the SFPU work overlaps the reads.
``mode`` (compile-time): 0 = ``silu_tile`` (approximation per ``math_approx_mode``),
1 = ``x * sigmoid_fast(x)``, 2 = Blackhole ``clamped_silu_glu`` (DeepSeek-V4 semantics:
clamps gate to <= 10 and up to [-10, 10] -- numerics differ for |x| > 10).
3 = ``minimal_matmul``'s single-pass SwiGLU (one SFPU pass over the a / b tile pair, its sigmoid sized for a bfp8
output: Schraudolph exp, bare SFPARECIP), about a third of ``silu_tile``'s SFPU time.
"""
import os

import ttnn
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads.op import (
    _TILE_BYTES,
    _core_ranges,
    _split_work_to_cores,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
READER_KERNEL = os.path.join(_HERE, "kernels", "reader_silu_mul.cpp")
COMPUTE_KERNEL = os.path.join(_HERE, "kernels", "compute_silu_mul.cpp")
WRITER_KERNEL = os.path.join(_HERE, "kernels", "writer_silu_mul.cpp")
READER_SHARDED_KERNEL = os.path.join(_HERE, "kernels", "reader_silu_mul_sharded.cpp")
WRITER_SHARDED_KERNEL = os.path.join(_HERE, "kernels", "writer_silu_mul_sharded.cpp")


def supported(a: ttnn.Tensor, b: ttnn.Tensor) -> bool:
    return (
        a.layout == ttnn.TILE_LAYOUT
        and b.layout == ttnn.TILE_LAYOUT
        and a.dtype in _TILE_BYTES
        and b.dtype in _TILE_BYTES
        and list(a.padded_shape) == list(b.padded_shape)
        and not a.is_sharded()
        and not b.is_sharded()
    )


def supported_sharded(a: ttnn.Tensor, b: ttnn.Tensor) -> bool:
    """a and b block-sharded alike in L1 (e.g. bs1's FF1 / FF3 outputs on their matmul grid): each core multiplies its
    own shards, no NoC reads."""
    return (
        a.layout == ttnn.TILE_LAYOUT
        and b.layout == ttnn.TILE_LAYOUT
        and a.dtype in _TILE_BYTES
        and b.dtype in _TILE_BYTES
        and list(a.padded_shape) == list(b.padded_shape)
        and a.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
        and a.memory_config() == b.memory_config()
        and a.memory_config().buffer_type == ttnn.BufferType.L1
        and a.memory_config().shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    )


def silu_mul(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    out_dtype: ttnn.DataType | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
    mode: int | None = None,
    approx: bool | None = None,
) -> ttnn.Tensor:
    if memory_config is None:
        memory_config = a.memory_config()
    if mode is None:
        mode = int(os.getenv("QWEN_SILU_MUL_MODE", "0"))
    if approx is None:
        approx = os.getenv("QWEN_SILU_MUL_APPROX", "1") == "1"
    if a.is_sharded():
        return _silu_mul_sharded(a, b, out_dtype=out_dtype, memory_config=memory_config, mode=mode, approx=approx)
    device = a.device()
    shape = list(a.padded_shape)
    out_dtype = out_dtype or a.dtype
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), out_dtype, ttnn.TILE_LAYOUT, device, memory_config)
    n_tiles = 1
    for d in shape:
        n_tiles *= int(d)
    n_tiles //= 32 * 32
    CH = 8 if mode == 0 else 4  # DST tiles per session (8 at bf16 half-sync; modes 1/2 need scratch slots)
    n_units = -(-n_tiles // CH)

    grid = device.compute_with_storage_grid_size()
    num_cores, per_core = _split_work_to_cores(n_units, int(grid.x), int(grid.y))
    used_cores = _core_ranges(per_core)

    def cb(index, tiles, dtype):
        ts = _TILE_BYTES[dtype]
        return ttnn.CBDescriptor(
            total_size=tiles * ts,
            core_ranges=used_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=ts)],
        )

    cbs = [cb(0, CH * 2, a.dtype), cb(1, CH * 2, b.dtype), cb(16, CH * 2, out_dtype)]
    reader_ct = [CH, n_tiles]
    for t in (a, b):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    compute_ct = [CH, mode]
    writer_ct = [CH, n_tiles]
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    reader_rt, compute_rt, writer_rt = [], [], []
    cursor = 0
    for cx, cy, n in per_core:
        reader_rt.append(((cx, cy), [a.buffer_address(), b.buffer_address(), n, cursor]))
        compute_rt.append(((cx, cy), [n, cursor]))
        writer_rt.append(((cx, cy), [out.buffer_address(), n, cursor]))
        cursor += n
    pd = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=READER_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=used_cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=COMPUTE_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=used_cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    math_approx_mode=bool(approx),
                    fp32_dest_acc_en=False,
                    dst_full_sync_en=False,
                    bfp8_pack_precise=True,
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=WRITER_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=used_cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ],
        cbs=cbs,
    )
    ttnn.generic_op([a, b, out], pd)
    return out


def _compute_config(approx):
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=bool(approx),
        fp32_dest_acc_en=False,
        dst_full_sync_en=False,
        bfp8_pack_precise=True,
    )


def _silu_mul_sharded(a, b, *, out_dtype, memory_config, mode, approx):
    """Block-sharded a / b (``supported_sharded``) into an interleaved output: CBs 0 / 1 alias each core's shards, the
    writer scatters the result tiles to their pages."""
    assert supported_sharded(a, b), "silu_mul: sharded a / b must be block-sharded alike in L1"
    assert not memory_config.is_sharded(), "silu_mul: the sharded path writes an interleaved output"
    device = a.device()
    shape = list(a.padded_shape)
    out_dtype = out_dtype or a.dtype
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), out_dtype, ttnn.TILE_LAYOUT, device, memory_config)
    spec = a.memory_config().shard_spec
    sh, sw = spec.shape[0] // 32, spec.shape[1] // 32
    Ht, Wt = shape[-2] // 32, shape[-1] // 32
    for d in shape[:-2]:
        Ht *= int(d)
    n_tiles = sh * sw
    CH = 8 if mode == 0 else 4
    assert n_tiles % CH == 0, f"silu_mul: shard of {n_tiles} tiles is not a multiple of {CH}"
    cores = spec.grid
    ts = _TILE_BYTES[out_dtype]
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(0, a),
        ttnn.cb_descriptor_from_sharded_tensor(1, b),
        ttnn.CBDescriptor(
            total_size=CH * 2 * ts,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=out_dtype, page_size=ts)],
        ),
    ]
    writer_ct = [CH, n_tiles // CH, sw, Wt]
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    writer_rt, compute_rt = [], []
    for cr in cores.ranges():
        for y in range(cr.start.y, cr.end.y + 1):
            for x in range(cr.start.x, cr.end.x + 1):
                # ROW_MAJOR block sharding: grid x walks the width shards, grid y the height shards
                gx, gy = x - cores.bounding_box().start.x, y - cores.bounding_box().start.y
                if gy * sh >= Ht:
                    continue
                valid_w = max(0, min(sw, Wt - gx * sw))
                writer_rt.append(((x, y), [out.buffer_address(), gy * sh, gx * sw, valid_w]))
                compute_rt.append(((x, y), [n_tiles // CH]))
    pd = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=READER_SHARDED_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores,
                compile_time_args=[n_tiles],
                runtime_args=[],
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=COMPUTE_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores,
                compile_time_args=[CH, mode],
                runtime_args=compute_rt,
                config=_compute_config(approx),
            ),
            ttnn.KernelDescriptor(
                kernel_source=WRITER_SHARDED_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ],
        cbs=cbs,
    )
    ttnn.generic_op([a, b, out], pd)
    return out
