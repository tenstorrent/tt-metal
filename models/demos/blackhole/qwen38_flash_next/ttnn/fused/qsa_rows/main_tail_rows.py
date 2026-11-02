# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""qsa_rows program 2, ``main_tail_rows``: the verify tile's main tail as one program.

The decode ``qsa_block.main_tail`` program with its KV stage replaced (``qsa_block.KVStage``); the norm, RoPE and
query kernels are per tile row, so they run unchanged on the tile's 32 rows. The KV core reads the current block back,
scatters row j < R of the packed tiles into slot (P + j) & 31 of that block or of the next block's zero rows, zeroes
the current block's rows past them, and writes both blocks, the same rows the op chain rewrites. A placed -0 is
canonicalized to +0 as in the chain; a kept -0 in the resident block stays -0 here where the chain reads +0
(numerically inert). Rows past R are stale and never read for the KV write.
"""

from __future__ import annotations

import ttnn

from .. import program as fp
from .. import qsa_block
from ..qsa_block import BF16, HEAD_DIM, KV_WIDTH, LOCAL_HEADS, ROPE_DIM, SPARSE_HEADS

NAME = "qsa_rows"
ROWS_KV_READER = fp.kernel_source(NAME, "rows_kv_reader.cpp")
ROWS_KV_WRITER = fp.kernel_source(NAME, "rows_kv_writer.cpp")
CB_BLOCK, CB_NEXT, CB_KV_POS = 29, 30, 31
ROW_BYTES = KV_WIDTH * 2  # one row-major cache row, 512 bf16
POS_PAGE = (
    128  # the position page at +0 (with the writer's block word at +4), the row-count page at +64 (rows_kv_cbs.h)
)


def _expect_rows_tile(tensor, width: int, label: str) -> None:
    shape = tuple(tensor.shape)
    if (
        len(shape) != 4
        or shape[:2] != (1, 1)
        or shape[2] != fp.TILE
        or shape[3] < width
        or tensor.layout != ttnn.TILE_LAYOUT
    ):
        raise ValueError(f"{label} must be a TILE row tile [1, 1, 32, >= {width}], got {tensor.layout} {shape}")


def kv_stage(position, rows, kv_cache, *, single_row: bool, v_ws, v_first: int, core=None) -> qsa_block.KVStage:
    """The verify rows' KV stage on one core: writes the current and (unless ``single_row``) the next KV block.

    ``position`` / ``rows`` are uint32 ROW_MAJOR [1, 1, 1, 1] device scalars P (the tile's first position) and R (read
    on the core, clamped to 32); ``kv_cache`` is the packed KV cache [1, 1, C, 512] bf16 ROW_MAJOR."""

    for label, scalar in (("position", position), ("rows", rows)):
        shape = tuple(scalar.shape)
        if shape != (1, 1, 1, 1) or scalar.dtype != ttnn.uint32 or scalar.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(f"{label} must be uint32 ROW_MAJOR [1, 1, 1, 1], got {scalar.layout} {shape}")
    cache_shape = tuple(kv_cache.shape)
    if (
        len(cache_shape) != 4
        or cache_shape[:2] != (1, 1)
        or cache_shape[3] != KV_WIDTH
        or cache_shape[2] % fp.TILE
        or kv_cache.dtype != BF16
        or kv_cache.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise ValueError(f"KV cache must be ROW_MAJOR bf16 [1, 1, C, 512] with C a multiple of 32, got {cache_shape}")
    core = ttnn.CoreCoord(0, 2) if core is None else core  # the decode form's first staging core
    acc = fp.accessor_args

    def cb_descriptors(all_set):
        return [
            fp.cb_descriptor(CB_BLOCK, BF16, ROW_BYTES, fp.TILE, all_set),
            fp.cb_descriptor(CB_NEXT, BF16, ROW_BYTES, fp.TILE, all_set),
            fp.cb_descriptor(CB_KV_POS, ttnn.uint32, POS_PAGE, 1, all_set),
        ]

    def kernels(ctx):
        core_set = ctx["staging_set"]
        return [
            fp.reader_kernel(
                ROWS_KV_READER,
                core_set,
                [*acc(kv_cache), *acc(v_ws), *acc(position), *acc(rows)],
                [
                    (
                        core,
                        [
                            kv_cache.buffer_address(),
                            v_ws.buffer_address(),
                            position.buffer_address(),
                            rows.buffer_address(),
                            v_first,
                            int(single_row),
                        ],
                    )
                ],
            ),
            fp.writer_kernel(
                ROWS_KV_WRITER, core_set, acc(kv_cache), [(core, [kv_cache.buffer_address(), int(single_row)])]
            ),
        ]

    return qsa_block.KVStage(
        cores=[core], cb_descriptors=cb_descriptors, kernels=kernels, io=[position, rows, kv_cache]
    )


def main_tail_rows(
    qg_ws,
    k_ws,
    v_ws,
    position,
    q_norm,
    k_norm,
    cos,
    sin,
    kv_cache,
    *,
    rows,
    single_row: bool = False,
    eps: float = qsa_block.EPS,
    qg_first: int = 0,
    k_first: int = 0,
    v_first: int = 0,
):
    """The verify tile's main tail; returns the sparse query [1, 32, 32, 512] bf16 ROW_MAJOR.

    ``qg_ws`` / ``k_ws`` / ``v_ws`` are 32-row TILE row tiles (rows past R stale), ``position`` / ``rows`` device
    scalars P / R, the RoPE tables [1, 1, 32, 64] hold row j at P + j; ``kv_cache`` is updated in place."""

    for tensor, width, label in (
        (qg_ws, 2 * LOCAL_HEADS * HEAD_DIM, "qg"),
        (k_ws, HEAD_DIM, "k"),
        (v_ws, HEAD_DIM, "v"),
    ):
        _expect_rows_tile(tensor, width, label)
    stage = kv_stage(position, rows, kv_cache, single_row=single_row, v_ws=v_ws, v_first=v_first)
    return qsa_block.main_tail(
        qg_ws,
        k_ws,
        v_ws,
        None,  # position inputs: the stage reads P itself
        q_norm,
        k_norm,
        cos,
        sin,
        None,  # no staging tensor
        kv_cache,
        eps=eps,
        qg_first=qg_first,
        k_first=k_first,
        v_first=v_first,
        kv_stage=stage,
    )


def main_tail_rows_composed(
    qg_ws,
    k_ws,
    v_ws,
    q_norm,
    k_norm,
    cos,
    sin,
    kv_cache,
    *,
    kv_block_start,
    kv_block_start_next,
    kv_read_indices,
    stage_keep,
    stage_a_select,
    stage_b_select,
    slot_zero,
    single_row: bool = False,
    eps: float = qsa_block.EPS,
    zero_value_half=None,
):
    """The unfused op chain on the 32-row tile; returns (sparse query [1, 32, 32, 512] ROW_MAJOR, gate [1, 6, 32, 256]).

    ``kv_read_indices`` uint32 [1, 32] are the current block's rows, ``stage_keep`` bf16 TILE [1, 1, 32, 1] is 0 at
    this pass's slots, ``stage_a_select`` / ``stage_b_select`` bf16 TILE [1, 1, 32, 32] place packed row j at block
    row r. ``zero_value_half`` is uploaded here when None, which is not traceable."""

    import torch

    from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_prefill

    mesh = qg_ws.device()
    config = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    dram = ttnn.DRAM_MEMORY_CONFIG
    rows = fp.TILE
    qg = ttnn.to_memory_config(qg_ws, dram)
    k = ttnn.to_memory_config(k_ws, dram)
    v = ttnn.to_memory_config(v_ws, dram)
    q_heads, gate_heads = [], []
    for head in range(LOCAL_HEADS):
        start = head * 2 * HEAD_DIM
        q_heads.append(ttnn.slice(qg, (0, 0, 0, start), (1, 1, rows, start + HEAD_DIM), memory_config=dram))
        gate_heads.append(
            ttnn.slice(qg, (0, 0, 0, start + HEAD_DIM), (1, 1, rows, start + 2 * HEAD_DIM), memory_config=dram)
        )
    q = ttnn.concat(q_heads, dim=1, memory_config=dram)
    gate = ttnn.concat(gate_heads, dim=1, memory_config=dram)
    q_normed = ttnn.rms_norm(q, epsilon=eps, weight=q_norm, memory_config=dram, compute_kernel_config=config)
    k_normed = ttnn.rms_norm(k, epsilon=eps, weight=k_norm, memory_config=dram, compute_kernel_config=config)
    q_rotated = apply_partial_rope_prefill(q_normed, cos, sin, LOCAL_HEADS, ROPE_DIM)
    k_rotated = apply_partial_rope_prefill(k_normed, cos, sin, 1, ROPE_DIM)
    # _write_packed_kv_verify
    packed = ttnn.concat([v, k_rotated], dim=3, memory_config=dram)
    looked_up = ttnn.embedding(kv_read_indices, kv_cache, layout=ttnn.TILE_LAYOUT, dtype=BF16, memory_config=dram)
    resident = ttnn.unsqueeze_to_4D(looked_up) if len(looked_up.shape) == 3 else looked_up
    kept = ttnn.multiply(resident, stage_keep, memory_config=dram)
    placed = ttnn.matmul(stage_a_select, packed, memory_config=dram, compute_kernel_config=config)
    staged = ttnn.add(kept, placed, memory_config=dram, fast_and_approximate_mode=False)
    staged_rm = ttnn.to_layout(staged, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(kv_cache, staged_rm, slot_zero, kv_block_start, 0, 1, 0)
    # the interleaved copies are temporaries only when they are copies (an interleaved input comes back as a new
    # handle on the same buffer, which must stay the caller's)
    temporaries = [
        t for t, source in ((qg, qg_ws), (k, k_ws), (v, v_ws)) if t.buffer_address() != source.buffer_address()
    ]
    temporaries += [
        *q_heads,
        *gate_heads,
        q,
        q_normed,
        k_normed,
        k_rotated,
        looked_up,
        kept,
        placed,
        staged,
        staged_rm,
    ]
    if not single_row:
        placed_next = ttnn.matmul(stage_b_select, packed, memory_config=dram, compute_kernel_config=config)
        next_rm = ttnn.to_layout(placed_next, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            kv_cache, next_rm, slot_zero, kv_block_start_next, 0, 1, 0
        )
        temporaries += [placed_next, next_rm]
    temporaries.append(packed)
    # the sparse query build of _sparse_value_attention_rows
    zero_half = zero_value_half
    if zero_half is None:
        zero_half = ttnn.from_torch(
            torch.zeros(1, LOCAL_HEADS, rows, HEAD_DIM, dtype=torch.bfloat16),
            dtype=BF16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=dram,
        )
    sparse_tiled = ttnn.concat([zero_half, q_rotated], dim=3, memory_config=dram)
    sparse_rm = ttnn.to_layout(sparse_tiled, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    sparse_query = ttnn.pad(
        sparse_rm, [(0, 0), (0, SPARSE_HEADS - LOCAL_HEADS), (0, 0), (0, 0)], 0.0, memory_config=dram
    )
    temporaries += [q_rotated, sparse_tiled, sparse_rm] + ([zero_half] if zero_value_half is None else [])
    for t in temporaries:
        ttnn.deallocate(t)
    return sparse_query, gate


__all__ = ["kv_stage", "main_tail_rows", "main_tail_rows_composed", "ROWS_KV_READER", "ROWS_KV_WRITER"]
