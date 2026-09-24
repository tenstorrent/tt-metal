# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Row-split variant of ``fused_add_rmsnorm`` for few-row inputs (bs1: 16 tile-rows).

Each tile-row is handled by ``R`` cores, each owning ``Wt/R`` column tiles. Every core adds
its slice, squares it and reduces it to a partial mean-square tile; the partials are
exchanged inside the row group with NoC writes + one semaphore per core; then each core
finishes rsqrt and applies inv·gamma to its own slice. One launch replaces the stock
add → interleaved-to-sharded → block-sharded LayerNorm → sharded-to-interleaved chain.
Exactly one row slice per core (rows_t * R <= cores).
"""
import os

import ttnn
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads.op import _TILE_BYTES, _core_ranges

_HERE = os.path.dirname(os.path.abspath(__file__))
READER_KERNEL = os.path.join(_HERE, "kernels", "reader_add_rmsnorm_split.cpp")
COMPUTE_KERNEL = os.path.join(_HERE, "kernels", "compute_add_rmsnorm_split.cpp")
WRITER_KERNEL = os.path.join(_HERE, "kernels", "writer_add_rmsnorm_split.cpp")
_BF16_TILE = _TILE_BYTES[ttnn.bfloat16]
_CHUNK = 4


def pick_split(rows_t: int, w_tiles: int, num_cores: int) -> int:
    """Largest R with w_tiles % R == 0, (w_tiles/R) % 4 == 0 and rows_t * R <= num_cores; 0 if none >= 2."""
    best = 0
    for r in range(2, w_tiles + 1):
        if w_tiles % r == 0 and (w_tiles // r) % _CHUNK == 0 and rows_t * r <= num_cores:
            best = r
    return best


def fused_add_rmsnorm_split(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    gamma_tiles: ttnn.Tensor,
    scaler_tile: ttnn.Tensor,
    eps_tile: ttnn.Tensor,
    *,
    R: int,
    sum_dtype: ttnn.DataType | None = None,
    out_dtype: ttnn.DataType | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    if memory_config is None:
        memory_config = a.memory_config()
    device = a.device()
    shape = list(a.padded_shape)
    W = shape[-1]
    Wt = W // 32
    rows = 1
    for d in shape[:-1]:
        rows *= int(d)
    rows_t = rows // 32
    if Wt % R != 0 or (Wt // R) % _CHUNK != 0:
        raise ValueError(f"R={R} must divide {Wt} tiles into multiples of {_CHUNK}")
    Wc = Wt // R
    sum_dtype = sum_dtype or a.dtype
    out_dtype = out_dtype or sum_dtype
    b8 = ttnn.bfloat8_b

    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    n_cores = rows_t * R
    if n_cores > gx * gy:
        raise ValueError(f"rows_t*R = {n_cores} > {gx * gy} cores")
    cores = [divmod(i, gy) for i in range(n_cores)]  # (cx, cy), y-major like _split_work_to_cores
    used_cores = _core_ranges([(cx, cy, 1) for cx, cy in cores])
    virt = []
    for cx, cy in cores:
        v = device.worker_core_from_logical_core(ttnn.CoreCoord(cx, cy))
        virt.append((int(v.x), int(v.y)))

    sum_tensor = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), sum_dtype, ttnn.TILE_LAYOUT, device, memory_config)
    out_tensor = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), out_dtype, ttnn.TILE_LAYOUT, device, memory_config)

    def cb(index, tiles, dtype):
        ts = _TILE_BYTES[dtype]
        return ttnn.CBDescriptor(
            total_size=tiles * ts,
            core_ranges=used_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=ts)],
        )

    cbs = [
        cb(0, Wc, a.dtype),  # a slice
        cb(1, Wc, b.dtype),  # b slice
        cb(2, Wc, ttnn.bfloat16),  # gamma slice (resident)
        cb(3, 1, ttnn.bfloat16),  # 1/W scaler
        cb(4, 1, ttnn.bfloat16),  # eps
        cb(5, Wc, b8),  # sum, bfp8 working copy
        cb(6, Wc, ttnn.bfloat16),  # sum^2
        cb(7, 1, ttnn.bfloat16),  # this core's partial mean-square (compute -> writer)
        cb(8, R, ttnn.bfloat16),  # all R partials of the row (peers write here)
        cb(9, 1, ttnn.bfloat16),  # rsqrt
        cb(16, Wc, sum_dtype),  # sum out slice
        cb(17, Wc, out_dtype),  # normalised out slice
    ]
    SEM_ID = 0
    semaphores = [ttnn.SemaphoreDescriptor(SEM_ID, ttnn.CoreType.WORKER, used_cores, 0)]

    reader_ct = [Wt, Wc]
    for t in (a, b, gamma_tiles, scaler_tile, eps_tile):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    compute_ct = [Wc, R, _CHUNK]
    writer_ct = [Wt, Wc, R, SEM_ID]
    for t in (sum_tensor, out_tensor):
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    reader_rt, compute_rt, writer_rt = [], [], []
    for i, (cx, cy) in enumerate(cores):
        row, k = divmod(i, R)
        core = (cx, cy)
        reader_rt.append(
            (
                core,
                [
                    a.buffer_address(),
                    b.buffer_address(),
                    gamma_tiles.buffer_address(),
                    scaler_tile.buffer_address(),
                    eps_tile.buffer_address(),
                    row,
                    k,
                ],
            )
        )
        compute_rt.append((core, [0]))
        peers = []
        for j in range(R):
            vx, vy = virt[row * R + j]
            peers += [vx, vy]
        writer_rt.append((core, [sum_tensor.buffer_address(), out_tensor.buffer_address(), row, k] + peers))

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
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
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
        semaphores=semaphores,
        cbs=cbs,
    )
    ttnn.generic_op([a, b, gamma_tiles, scaler_tile, eps_tile, sum_tensor, out_tensor], pd)
    return sum_tensor, out_tensor
