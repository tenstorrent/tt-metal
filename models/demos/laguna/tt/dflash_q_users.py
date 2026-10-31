# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batched DFlash draft: query rows between the head-split layout and a per-user attention layout (Laguna;
kernels/dflash_q_users.cpp).

The batched draft runs Q rows per user, 32 / Q users per 32-row tile. Its attention must stay per user (each user's
queries see only that user's context ring), so the G = NH / NKV query heads of a KV head and the user's Q rows are
gathered into one [RP, 128] block per (user, KV head) -- [U, NKV, RP, 128], rows (h % G) * Q + i, zero padded to RP
-- and the attention output is scattered back. Both are row copies (Q * 32 bytes each), one multi-core program."""

from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


def _run(src, out, mode, U, NH, NKV, Q, RP):
    device = src.device()
    gs = device.compute_with_storage_grid_size()
    n_out = (U * NKV * (RP // TILE) * 4) if mode == 0 else (NH * (U * Q // TILE) * 4)
    cores = min(gs.x * gs.y, n_out)
    per = -(-n_out // cores)
    cores = -(-n_out // per)
    grid = ttnn.num_cores_to_corerangeset(cores, gs, True)
    k = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "dflash_q_users.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[gs.x, mode, U, NH, NKV, Q, RP, per]
        + list(ttnn.TensorAccessorArgs(src).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
        common_runtime_args=[src.buffer_address(), out.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    cb = ttnn.CBDescriptor(
        total_size=2048,
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)],
    )
    ttnn.generic_op([src, out], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb]))
    return out


def to_users(q, U, Q, NKV, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """q [1, NH, U * Q, 128] bf16 TILE -> [U, NKV, RP, 128] (RP = G * Q rounded up to 32)."""
    NH = int(q.shape[1])
    G = NH // NKV
    RP = -(-G * Q // TILE) * TILE
    assert q.dtype == ttnn.bfloat16 and q.layout == ttnn.TILE_LAYOUT and int(q.shape[-1]) == 128 and 16 % Q == 0
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([U, NKV, RP, 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, q.device(),
                                         memory_config)  # fmt: skip
    return _run(q, out, 0, U, NH, NKV, Q, RP)


def from_users(a, U, Q, NH, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """a [U, NKV, RP, 128] bf16 TILE (to_users' layout) -> [1, NH, U * Q, 128]."""
    NKV, RP = int(a.shape[1]), int(a.shape[2])
    assert a.dtype == ttnn.bfloat16 and a.layout == ttnn.TILE_LAYOUT and int(a.shape[-1]) == 128
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, NH, U * Q, 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, a.device(),
                                         memory_config)  # fmt: skip
    return _run(a, out, 1, U, NH, NKV, Q, RP)
