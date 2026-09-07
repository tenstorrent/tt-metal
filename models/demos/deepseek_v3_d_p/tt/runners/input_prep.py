# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""MLA-family (DeepSeek-V3 / Kimi) prefill input prep.

The model-agnostic engine helpers (mesh open, H2D service, trace loading) live in
the common package at ``models.demos.common.prefill.runners.runner_utils``. What
remains here is the one piece of model-specific glue the runtime needs:
``prepare_prefill_input_tensor`` (the SP-sharded chunk input), which backs
``TtPrefillRuntime.make_chunk_input``, plus the MTP lookahead upload built on top of it.

The union geometry those MTP ids feed lives one layer down in
``models.demos.deepseek_v3_d_p.tt.mtp_prefill.device_windows``; what is here is only the upload.

KV-cache PCC validation + golden loaders live in
``models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation``; the host-pull KV
diagnostics used only by tests live in ``tests/test_runner_utils.py``.
"""

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import create_balanced_chunk_order, reorder_tensor_chunks


def prepare_prefill_input_tensor(
    token_ids: list[int],
    mesh_device: ttnn.MeshDevice,
    sp_factor: int,
    is_balanced: bool,
    mesh_shape: tuple,
    sp_axis: int,
) -> ttnn.Tensor:
    """Shard and upload one chunk's token IDs as a prefill input tensor.

    Produces an SP-sharded uint32 ROW_MAJOR DRAM tensor of shape
    ``[sp_factor, 1, len(token_ids) // sp_factor]`` -- the format
    ``TtPrefillTransformer.forward`` expects, and the same one
    ``prefill_producer._h2d_rows`` builds for the H2D socket.

    MTP's lookahead ids are NOT folded in here; they ride their own tensor
    (:func:`prepare_prefill_mtp_tokens`), so an MTP run and a plain run upload identical trunk
    chunks.
    """
    isl_per_chip = len(token_ids) // sp_factor
    assert (
        len(token_ids) == sp_factor * isl_per_chip
    ), f"got {len(token_ids)} ids, not divisible by sp_factor={sp_factor}"
    flat = torch.tensor(token_ids, dtype=torch.int64)
    if is_balanced:
        t = reorder_tensor_chunks(
            flat.unsqueeze(0).unsqueeze(0).unsqueeze(-1), create_balanced_chunk_order(sp_factor), seq_dim=2
        )
        token_ids_sharded = t.squeeze(0).squeeze(-1).reshape(sp_factor, 1, isl_per_chip)
    else:
        token_ids_sharded = flat.reshape(sp_factor, 1, isl_per_chip)
    return _upload_ids(token_ids_sharded, mesh_device, mesh_shape, sp_axis)


def prepare_prefill_mtp_tokens(
    token_ids: list[int],
    mesh_device: ttnn.MeshDevice,
    sp_factor: int,
    mesh_shape: tuple,
    sp_axis: int,
    *,
    num_mtp_tokens: int,
) -> ttnn.Tensor:
    """Upload the MTP lookahead ids: the ``num_mtp_tokens`` ids that follow each chip's trunk shard.

    ``token_ids`` is this chunk's ``C`` tokens followed by the next ``num_mtp_tokens`` from the stream
    (``C + num_mtp_tokens`` in all), and chip ``c`` gets ``token_ids[(c+1)*L : (c+1)*L + num_mtp_tokens]``.
    Concatenated onto that chip's trunk row it forms the contiguous ``token_ids[c*L : c*L+L+num_mtp_tokens]``,
    so MTP level ``k`` reads the same local slice ``[k, k+L)`` on every chip.

    Only the LAST chip's row reaches past the chunk; the other ``sp-1`` take theirs from inside it.
    Block-cyclic only: under ``is_balanced`` the row -> position map is a permutation, so "the next
    ids after this row" is not a contiguous slice of the stream.
    """
    assert num_mtp_tokens > 0, f"num_mtp_tokens must be positive, got {num_mtp_tokens}"
    isl_per_chip = (len(token_ids) - num_mtp_tokens) // sp_factor
    assert len(token_ids) == sp_factor * isl_per_chip + num_mtp_tokens, (
        f"got {len(token_ids)} ids, expected sp_factor*L + num_mtp_tokens = "
        f"{sp_factor}*{isl_per_chip} + {num_mtp_tokens}"
    )
    # unfold gives the sp windows of length `num_mtp_tokens` at stride L, as a view; dropping the first L
    # ids is what shifts window c from chip c's own rows to the ids just past them.
    rows = (
        torch.tensor(token_ids, dtype=torch.int64)[isl_per_chip:].unfold(0, num_mtp_tokens, isl_per_chip).unsqueeze(1)
    )
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


def build_position_zero_mask(
    mesh_device: ttnn.MeshDevice,
    sp_factor: int,
    chunk_size: int,
    is_balanced: bool,
    mesh_shape: tuple,
    sp_axis: int,
    *,
    emb_dim_per_chip: int,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    """Multiplicative mask that zeroes the embedding at ABSOLUTE position 0, for every MTP level.

    vLLM zeroes it on every level (``torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)``
    in ``deepseek_mtp.py``), not just level 1, and ``fused_mtp_reference`` mirrors that.

    It cannot be done from the token side: zeroing a token *id* gives ``embed(0)``, not ``0``. And it
    cannot be done inside ``TtFusedMTP``, because under SP the row index is not the absolute
    position -- only the caller knows which row is position 0.

    The mask is built by pushing a position indicator through the SAME sharding path the tokens
    take, rather than asserting that position 0 lands on chip 0 row 0. It does land there under both
    layouts today (``create_balanced_chunk_order`` starts at chunk 0), but deriving it costs one
    host reshape and stops that from being a silent assumption.

    Only the chunk with ``actual_start == 0`` needs this; every later chunk's rows are all past
    position 0.

    Args:
        chunk_size: ``C``, the padded chunk length (NOT the per-chip length).
        emb_dim_per_chip: ``H / tp``. Materialized at full width so the multiply is plain
            elementwise -- a width-1 broadcast in TILE_LAYOUT is the kind of thing that works until
            it does not, and the full mask is 1.875 MiB/chip at L=640, H/tp=1536, built once.
    """
    # Right padding only: row 0 of the chunk is absolute position 0 exactly when the chunk's real
    # tokens start at row 0. TtPrefillTransformer asserts padding_side == "right" before MTP runs,
    # so this is a restatement of that contract at the place it is actually relied on.
    assert chunk_size % sp_factor == 0, f"chunk {chunk_size} not divisible by sp_factor {sp_factor}"
    isl_per_chip = chunk_size // sp_factor
    keep = torch.ones(chunk_size, dtype=torch.float32)
    keep[0] = 0.0
    if is_balanced:
        t = keep.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        t = reorder_tensor_chunks(t, create_balanced_chunk_order(sp_factor), seq_dim=2)
        sharded = t.squeeze(0).squeeze(-1).reshape(sp_factor, 1, isl_per_chip)
    else:
        sharded = keep.reshape(sp_factor, 1, isl_per_chip)
    mask = sharded.unsqueeze(-1).expand(sp_factor, 1, isl_per_chip, int(emb_dim_per_chip)).contiguous()
    return ttnn.from_torch(
        mask,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
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

    THE geometry of last-chunk generation, stated once and shared by both mask builders below.

    Chip ``c``'s union covers global positions ``[chunk_start + c*L, chunk_start + c*L + U)`` with
    ``L = chunk_size / sp`` and ``U = L + num_mtp_tokens``. So the row holding position ``p`` is
    ``u = p - chunk_start - c*L``, present iff ``0 <= u < U``. Adjacent chips' unions OVERLAP by
    ``num_mtp_tokens`` rows (that overlap is what makes every level's window the same local slice), so two
    chips can hold the same position -- both entries are returned and both get patched, which is what
    keeps the windows consistent across the seam.

    Level ``k``'s window at trunk row ``r`` reads global ``chunk_start + c*L + r + k + 1``; the last
    real row is ``actual_end - 1``, so level ``k`` reads ``actual_end + k`` there. Writing the
    generated embedding at that ONE global position therefore feeds every level that needs it, with
    no per-level splice.

    Block-cyclic only: under ``is_balanced`` the row -> position map is a permutation and a chip's
    union is not a contiguous position range at all.
    """
    assert num_mtp_tokens > 0, f"num_mtp_tokens must be positive, got {num_mtp_tokens}"
    assert chunk_size % sp_factor == 0, f"chunk {chunk_size} not divisible by sp_factor {sp_factor}"
    isl_per_chip = chunk_size // sp_factor
    union_len = isl_per_chip + num_mtp_tokens
    target = actual_end + level - chunk_start
    rows = []
    for c in range(sp_factor):
        u = target - c * isl_per_chip
        rows.append(int(u) if 0 <= u < union_len else None)
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
    num_levels: int,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    """``[sp, 1, U, H/tp]`` of ones, zero on every row generation will write.

    Applied to the union ONCE before level 0, so each level's patch is a plain add onto a cleared
    row rather than a read-modify-write. What it clears is whatever the producer put past the
    request's real length -- its pad id, or, if the prompt pool outlives the request, the next ids
    of the pool, which would be the answer leaking into its own draft.

    Rows past the last real one are pad for MTP exactly as they are for the trunk, so clearing all
    ``K`` positions up front (rather than one per level) costs nothing: at level ``k`` the rows that
    read a not-yet-generated position are pad rows.

    Full width, like :func:`build_position_zero_mask`, so the multiply is plain elementwise.
    """
    isl_per_chip = chunk_size // sp_factor
    union_len = isl_per_chip + num_mtp_tokens
    keep = torch.ones(sp_factor, 1, union_len, 1, dtype=torch.float32)
    for level in range(num_levels):
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
    """``[sp, 1, U, 32*sp]`` one-hot selector: ``select @ gathered`` is the generated embedding,
    broadcast to exactly the union rows that hold global position ``actual_end + level``.

    ``gathered`` is the SP-all-gathered ``[1, 1, 32*sp, H/tp]`` block of the level's LM-head tile
    embeddings, so ``source_row = device_id * 32 + token_offset`` picks the one row that is the
    generated token (``TtLMHead.forward`` returns that pair). A chip that does not hold the position
    gets an all-zero row block and adds nothing.

    A matmul rather than a scatter because there is no device-side scatter that takes a runtime row
    index, and because one-hot bf16 is bit-exact: ``1.0 * x == x``, so the patched row equals the
    embedding row it came from with no PCC blur. The contracted dim is ``32*sp`` -- tile-aligned by
    construction, no padding subtleties.
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
