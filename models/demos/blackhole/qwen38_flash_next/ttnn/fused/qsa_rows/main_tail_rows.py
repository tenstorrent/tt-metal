# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""qsa_rows program 2, ``main_tail_rows``: the verify tile's main tail as ONE program.

The chain (``Qwen38TTNNQSA._main_projection_rows`` after its linears, ``_write_packed_kv_verify``, the query build of
``_sparse_value_attention_rows``): the head split of qg (12 slices, 2 concats), rms_norm of q and k, their partial
RoPE, the packed row [v | k], the current KV block read back (the kv row lookup), its rows from P % 32 on zeroed
(stage_keep) and the new rows placed (stage_a_select, a one-hot matmul), the sum written back as the block; the next
block written as the one-hot placement of the rows crossing the block edge (stage_b_select) into zero rows -- every
pass, unless ``single_row`` -- and the sparse query [zeros(256) | q] per head padded to 32 heads, ROW_MAJOR.

Here: the decode ``qsa_block.main_tail`` program with its KV stage replaced (``qsa_block.KVStage``).  Its norm, RoPE,
head-split and query kernels run unchanged on the tile's 32 rows (they are per tile row; the chain's rms_norm and
RoPE also run over the whole tile), so the sparse query is the decode program's proven output at rows = 32.  The new
KV core (``kernels/rows_kv_reader.cpp`` / ``rows_kv_writer.cpp``) reads the current block's 32 row-major rows back,
scatters row j < R of the packed tiles into slot (P + j) & 31 of that block or of the next block's zero rows, zeroes
the current block's rows past them as the chain leaves them, and writes both blocks -- the same rows the chain rewrites.  Bitwise by construction where the chain's arithmetic is exact
(a one-hot bf16 placement in fp32 accumulate copies a row, 1.0 * x keeps it); a placed or kept -0 reads +0 in the chain
(its zero terms add +0) and the kernel canonicalizes a placed -0 to +0 the same way; a kept -0 in the resident block
would read +0 in the chain and -0 here (numerically inert for the attention it feeds; the device test's random rows
carry no exact zeros).  Rows past R are the tile's stale rows and are never read for the KV write.
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
    """The verify rows' KV stage on one core: reads the current block back, scatters the R packed rows, writes the
    current and (unless ``single_row``) the next block.  ``position`` is the pass's uint32 ROW_MAJOR [1, 1, 1, 1] scalar
    P (the tile's first position; the block start P & ~31 is derived on the core), ``rows`` the same form's row count R
    (the verify form's constant k + 1, the commit form's accepted count a*: one program for both; read on the core,
    clamped to 32); ``kv_cache`` the packed KV cache [1, 1, C, 512] bf16 ROW_MAJOR."""

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
    core = ttnn.CoreCoord(0, 2) if core is None else core  # the first lane's staging core position of the decode form
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

    blocks = 1 if single_row else 2
    return qsa_block.KVStage(
        cores=[core],
        cb_descriptors=cb_descriptors,
        kernels=kernels,
        io=[position, rows, kv_cache],
        reads=(position, rows),
        writes=(),
        partial=((kv_cache, blocks * fp.TILE * ROW_BYTES),),
        flops=0,
        name="main_tail_rows",
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
    """The verify tile's main tail: ``qg_ws`` / ``k_ws`` / ``v_ws`` the projections as 32-row TILE row tiles (rows
    past R stale), ``position`` / ``rows`` uint32 ROW_MAJOR [1, 1, 1, 1] = P / R (device scalars), the norm weights [1, 1, 1, 256], the RoPE tables
    [1, 1, 32, 64] (row j at P + j), the packed KV cache [1, 1, C, 512] ROW_MAJOR updated in place (the block at
    P & ~31 and, unless ``single_row``, the next block).  Returns the sparse query [1, 32, 32, 512] bf16 ROW_MAJOR."""

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
    """The chain on the 32-row tile (ttnn/qsa.py ``_main_projection_rows`` after its linears, ``_write_packed_kv_verify``
    and the query build of ``_sparse_value_attention_rows``), with the verify inputs as the chain derives them:
    ``kv_read_indices`` uint32 [1, 32] (the current block's rows), ``stage_keep`` bf16 TILE [1, 1, 32, 1] (0 at this
    pass's slots), ``stage_a_select`` / ``stage_b_select`` bf16 TILE [1, 1, 32, 32] (row r of the block <- packed row j),
    ``slot_zero`` / ``kv_block_start`` / ``kv_block_start_next`` uint32 ROW_MAJOR [1, 1, 1, 1]; ``zero_value_half`` the
    chain's constant zero tile [1, 6, 32, 256] bf16 TILE (uploaded here when None: then not traceable).  Returns the
    sparse query [1, 32, 32, 512] ROW_MAJOR and the gate [1, 6, 32, 256] TILE."""

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
