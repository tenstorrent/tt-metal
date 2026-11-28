# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Head split + RoPE on Q and K in one ``ttnn.generic_op``.

Replaces ``nlp_create_qkv_heads`` followed by two ``rotary_embedding_llama`` calls. Those
read the fused QKV activation and write Q/K/V heads, then read Q and K again and write
them rotated: three DRAM round trips over Q and K. Here each core reads a work unit of the
fused QKV row (one 32-token seq tile x one GQA group: q_per_kv Q heads, one K, one V head),
rotates the Q and K heads in compute and writes all three straight to the head layout.

    qkv [B, 1, S, (n_heads + 2 * n_kv) * D]  ->  q [B, n_heads, S, D], k, v [B, n_kv, S, D]
    q, k = x * cos + (x @ T) * sin   (interleaved-pair rotation, as rotary_embedding_llama)
"""

import os

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops._common import (
    TILE_BYTES,
    core_ranges,
    split_work_to_cores,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
READER_KERNEL = os.path.join(_HERE, "kernels", "reader_heads_rope.cpp")
COMPUTE_KERNEL = os.path.join(_HERE, "kernels", "compute_heads_rope.cpp")
WRITER_KERNEL = os.path.join(_HERE, "kernels", "writer_heads_rope.cpp")
TILE = 32


def fused_heads_rope(qkv, cos, sin, trans_mat, *, n_heads, n_kv_heads, batch, memory_config=None, out_dtype=None):
    """qkv: [1, 1, B * S, (n_heads + 2 * n_kv) * D] (or [B, 1, S, ...]) bf16 TILE interleaved.
    cos / sin: [1, 1, S, D] bf16 TILE (rotary_embedding_llama layout). trans_mat: one 32x32 tile.
    Returns (q, k, v). ``out_dtype`` other than qkv's dtype is not supported yet (bfp8 K / V come
    out wrong; open issue)."""
    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    if out_dtype is not None and out_dtype != qkv.dtype:
        raise NotImplementedError("fused_heads_rope: output dtype must match the input dtype")
    device = qkv.device()
    rows = int(qkv.padded_shape[-2]) * int(qkv.padded_shape[-3]) * int(qkv.padded_shape[0])
    width = int(qkv.padded_shape[-1])
    D = width // (n_heads + 2 * n_kv_heads)
    S = rows // batch
    assert D % TILE == 0 and S % TILE == 0 and n_heads % n_kv_heads == 0
    Wt, St = D // TILE, S // TILE
    q_per_kv = n_heads // n_kv_heads
    unit_tiles = (q_per_kv + 2) * Wt

    q = ttnn.allocate_tensor_on_device(
        ttnn.Shape([batch, n_heads, S, D]), qkv.dtype, ttnn.TILE_LAYOUT, device, memory_config
    )
    k = ttnn.allocate_tensor_on_device(
        ttnn.Shape([batch, n_kv_heads, S, D]), qkv.dtype, ttnn.TILE_LAYOUT, device, memory_config
    )
    v = ttnn.allocate_tensor_on_device(
        ttnn.Shape([batch, n_kv_heads, S, D]), qkv.dtype, ttnn.TILE_LAYOUT, device, memory_config
    )

    grid = device.compute_with_storage_grid_size()
    _, per_core = split_work_to_cores(batch * St * n_kv_heads, int(grid.x), int(grid.y))
    used = core_ranges(per_core)
    nbuf = 2 if max(n for _, _, n in per_core) > 1 else 1

    def cb(index, tiles, dtype):
        tsize = TILE_BYTES[dtype]
        return ttnn.CBDescriptor(
            total_size=tiles * tsize,
            core_ranges=used,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=tsize)],
        )

    bf = ttnn.bfloat16
    cbs = [
        cb(0, unit_tiles * nbuf, qkv.dtype),  # unit input: Q heads, K, V
        cb(9, Wt * nbuf, bf),  # cos of the unit's seq tile
        cb(10, Wt * nbuf, bf),  # sin
        cb(11, 1, bf),  # rotation tile
        cb(13, Wt, bf),  # (x @ T) * sin
        cb(16, unit_tiles * nbuf, qkv.dtype),  # unit output
    ]

    reader_ct = [n_heads, n_kv_heads, Wt, St]
    for t in (qkv, cos, sin, trans_mat):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct = [n_heads, n_kv_heads, Wt, St]
    for t in (q, k, v):
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    reader_rt, compute_rt, writer_rt = [], [], []
    cursor = 0
    for cx, cy, n in per_core:
        core = (cx, cy)
        reader_rt.append(
            (
                core,
                [
                    qkv.buffer_address(),
                    cos.buffer_address(),
                    sin.buffer_address(),
                    trans_mat.buffer_address(),
                    n,
                    cursor,
                ],
            )
        )
        compute_rt.append((core, [n, cursor]))
        writer_rt.append((core, [q.buffer_address(), k.buffer_address(), v.buffer_address(), n, cursor]))
        cursor += n

    def kernel(src, ct, rt, config):
        return ttnn.KernelDescriptor(
            kernel_source=src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=used,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    program = ttnn.ProgramDescriptor(
        kernels=[
            kernel(READER_KERNEL, reader_ct, reader_rt, ttnn.ReaderConfigDescriptor()),
            kernel(
                COMPUTE_KERNEL,
                [q_per_kv, Wt, n_kv_heads, St],
                compute_rt,
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    dst_full_sync_en=False,
                ),
            ),
            kernel(WRITER_KERNEL, writer_ct, writer_rt, ttnn.WriterConfigDescriptor()),
        ],
        cbs=cbs,
    )
    ttnn.generic_op([qkv, cos, sin, trans_mat, q, k, v], program)
    return q, k, v
