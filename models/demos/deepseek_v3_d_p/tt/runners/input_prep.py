# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""MLA-family (DeepSeek-V3 / Kimi) prefill input prep.

The model-agnostic engine helpers (mesh open, H2D service, trace loading) live in
the common package at ``models.demos.common.prefill.runners.runner_utils``. What
remains here is the one piece of model-specific glue the runtime needs:
``prepare_prefill_input_tensor`` (the SP-sharded chunk input), which backs
``TtPrefillRuntime.make_chunk_input``, plus the MTP lookahead upload built on top of it.

KV-cache PCC validation + golden loaders live in
``models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation``; the host-pull KV
diagnostics used only by tests live in ``tests/test_runner_utils.py``.
"""

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import (
    create_balanced_chunk_order,
    reorder_tensor_chunks,
    rotated_chip_positions,
    rotated_rows_are_contiguous,
)


def prepare_prefill_input_tensor(
    token_ids: list[int],
    mesh_device: ttnn.MeshDevice,
    sp_factor: int,
    is_balanced: bool,
    mesh_shape: tuple,
    sp_axis: int,
    *,
    chunk_start: int = 0,
) -> ttnn.Tensor:
    """Shard and upload one chunk's token IDs as a prefill input tensor.

    An SP-sharded uint32 ROW_MAJOR DRAM tensor, ``[sp_factor, 1, len(token_ids) // sp_factor]``, from
    the chunk in natural position order at ``chunk_start``. Chip ``c``'s row carries the positions
    chip ``c`` OWNS: a chunk resuming mid-slab is rotated across the chips, and the device derives
    each row's rope angle and KV-cache slot the same way (``rotated_chip_positions``). A slab-aligned
    ``chunk_start`` -- every caller but an MTP mid-slab resume -- takes the plain reshape the rotation
    degenerates to.
    """
    isl_per_chip = len(token_ids) // sp_factor
    assert (
        len(token_ids) == sp_factor * isl_per_chip
    ), f"got {len(token_ids)} ids, not divisible by sp_factor={sp_factor}"
    flat = torch.tensor(token_ids, dtype=torch.int64)
    if is_balanced:
        assert chunk_start % len(token_ids) == 0, (
            f"is_balanced cannot express a rotated chunk; chunk_start={chunk_start} must be a multiple "
            f"of the chunk size {len(token_ids)}"
        )
        t = reorder_tensor_chunks(
            flat.unsqueeze(0).unsqueeze(0).unsqueeze(-1), create_balanced_chunk_order(sp_factor), seq_dim=2
        )
        token_ids_sharded = t.squeeze(0).squeeze(-1).reshape(sp_factor, 1, isl_per_chip)
    elif chunk_start % len(token_ids) == 0:
        token_ids_sharded = flat.reshape(sp_factor, 1, isl_per_chip)
    else:
        token_ids_sharded = flat[_rotation_index(chunk_start, sp_factor, isl_per_chip)].reshape(
            sp_factor, 1, isl_per_chip
        )
    return _upload_ids(token_ids_sharded, mesh_device, mesh_shape, sp_axis)


def _rotation_index(chunk_start: int, sp_factor: int, isl_per_chip: int) -> torch.Tensor:
    """Chip-major device row -> offset into the chunk's id list; ``arange`` when slab-aligned."""
    return torch.tensor(
        [p - chunk_start for row in rotated_chip_positions(chunk_start, sp_factor, isl_per_chip) for p in row],
        dtype=torch.long,
    )


def prepare_prefill_mtp_tokens(
    token_ids: list[int],
    mesh_device: ttnn.MeshDevice,
    sp_factor: int,
    mesh_shape: tuple,
    sp_axis: int,
    *,
    num_mtp_tokens: int,
    chunk_start: int = 0,
) -> ttnn.Tensor:
    """Upload the MTP lookahead ids: the ``num_mtp_tokens`` ids that follow each chip's trunk shard.

    Chip ``c`` takes the ids past the LAST POSITION IT CARRIES, so concatenated onto its trunk row
    every MTP level reads the same local slice. Block-cyclic only.
    """
    assert num_mtp_tokens > 0, f"num_mtp_tokens must be positive, got {num_mtp_tokens}"
    isl_per_chip = (len(token_ids) - num_mtp_tokens) // sp_factor
    assert len(token_ids) == sp_factor * isl_per_chip + num_mtp_tokens, (
        f"got {len(token_ids)} ids, expected sp_factor*L + num_mtp_tokens = "
        f"{sp_factor}*{isl_per_chip} + {num_mtp_tokens}"
    )
    assert rotated_rows_are_contiguous(chunk_start, isl_per_chip), (
        f"chunk_start={chunk_start} must be a multiple of the per-chip shard {isl_per_chip}: off that "
        "boundary a chip's positions are discontiguous, so 'the ids following its shard' is not a run"
    )
    trunk_ends = [row[-1] for row in rotated_chip_positions(chunk_start, sp_factor, isl_per_chip)]
    index = torch.tensor(
        [end + 1 + k - chunk_start for end in trunk_ends for k in range(num_mtp_tokens)], dtype=torch.long
    )
    rows = torch.tensor(token_ids, dtype=torch.int64)[index].reshape(sp_factor, 1, num_mtp_tokens)
    return _upload_ids(rows, mesh_device, mesh_shape, sp_axis)


def _upload_ids(rows: torch.Tensor, mesh_device: ttnn.MeshDevice, mesh_shape: tuple, sp_axis: int) -> ttnn.Tensor:
    """Upload a host ``[sp_factor, 1, row_len]`` id block, one row per SP chip."""
    return ttnn.from_torch(
        rows.contiguous(),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(sp_axis, None)),
    )


def mtp_generation_union_rows(
    sp_factor: int,
    chunk_size: int,
    *,
    num_mtp_tokens: int,
    chunk_start: int,
    actual_end: int,
    level: int,
) -> list:
    """Where global position ``actual_end + level`` sits in each chip's union, or None.

    The geometry of last-chunk generation, stated once for both mask builders below. Adjacent chips'
    unions overlap, so a position can land on two chips and both get patched. Block-cyclic only, and
    keyed off ``rotated_chip_positions``: a chunk resuming off a slab boundary is rotated, so chip
    c's rows are NOT ``[chunk_start + c*isl_per_chip, ...)``.
    """
    assert num_mtp_tokens > 0, f"num_mtp_tokens must be positive, got {num_mtp_tokens}"
    assert chunk_size % sp_factor == 0, f"chunk {chunk_size} not divisible by sp_factor {sp_factor}"
    isl_per_chip = chunk_size // sp_factor
    global_pos = actual_end + level
    rows = []
    for trunk in rotated_chip_positions(chunk_start, sp_factor, isl_per_chip):
        if global_pos in trunk:
            rows.append(trunk.index(global_pos))
            continue
        past_shard = global_pos - trunk[-1] - 1
        rows.append(isl_per_chip + past_shard if 0 <= past_shard < num_mtp_tokens else None)
    assert any(r is not None for r in rows), (
        f"no chip holds global position {actual_end + level}: chunk_start={chunk_start} "
        f"chunk_size={chunk_size} num_mtp_tokens={num_mtp_tokens} level={level}. MTP levels must be <= num_mtp_tokens."
    )
    return rows


def build_mtp_generation_keep_mask(
    mesh_device: ttnn.MeshDevice,
    sp_factor: int,
    chunk_size: int,
    mesh_shape: tuple,
    sp_axis: int,
    *,
    emb_dim_per_chip: int,
    num_mtp_tokens: int,
    chunk_start: int,
    actual_end: int,
    levels,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    """``[sp, 1, U, H/tp]`` of ones, zero on every row generation will write.

    Applied once before the first generated level, so each level's patch is an add onto a cleared row.
    ``levels`` is the GENERATED range: clearing a provided level's row would lose a real embedding.
    """
    isl_per_chip = chunk_size // sp_factor
    union_len = isl_per_chip + num_mtp_tokens
    keep = torch.ones(sp_factor, 1, union_len, 1, dtype=torch.float32)
    levels = list(levels)
    assert levels, "keep mask asked for an empty generated range; build no generation at all instead"
    for level in levels:
        for c, u in enumerate(
            mtp_generation_union_rows(
                sp_factor,
                chunk_size,
                num_mtp_tokens=num_mtp_tokens,
                chunk_start=chunk_start,
                actual_end=actual_end,
                level=level,
            )
        ):
            if u is not None:
                keep[c, 0, u, 0] = 0.0
    mask = keep.expand(sp_factor, 1, union_len, int(emb_dim_per_chip)).contiguous()
    return _upload_sp_sharded(mask, mesh_device, mesh_shape, sp_axis, dtype)


def build_mtp_generation_select(
    mesh_device: ttnn.MeshDevice,
    sp_factor: int,
    chunk_size: int,
    mesh_shape: tuple,
    sp_axis: int,
    *,
    num_mtp_tokens: int,
    chunk_start: int,
    actual_end: int,
    level: int,
    source_row: int,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    """``[sp, 1, U, 32*sp]`` one-hot selector: ``select @ gathered`` broadcasts the generated embedding
    onto exactly the union rows holding global position ``actual_end + level``.
    """
    isl_per_chip = chunk_size // sp_factor
    union_len = isl_per_chip + num_mtp_tokens
    width = ttnn.TILE_SIZE * sp_factor
    assert 0 <= source_row < width, f"source_row {source_row} out of range [0, {width})"
    select = torch.zeros(sp_factor, 1, union_len, width, dtype=torch.float32)
    for c, u in enumerate(
        mtp_generation_union_rows(
            sp_factor,
            chunk_size,
            num_mtp_tokens=num_mtp_tokens,
            chunk_start=chunk_start,
            actual_end=actual_end,
            level=level,
        )
    ):
        if u is not None:
            select[c, 0, u, source_row] = 1.0
    return _upload_sp_sharded(select, mesh_device, mesh_shape, sp_axis, dtype)


def _upload_sp_sharded(
    t: torch.Tensor, mesh_device: ttnn.MeshDevice, mesh_shape: tuple, sp_axis: int, dtype: ttnn.DataType
) -> ttnn.Tensor:
    """Upload a host ``[sp_factor, 1, rows, cols]`` block as TILE DRAM, one row block per SP chip."""
    return ttnn.from_torch(
        t.contiguous(),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(sp_axis, None)),
    )
