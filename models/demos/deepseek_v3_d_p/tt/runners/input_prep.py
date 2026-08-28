# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""MLA-family (DeepSeek-V3 / Kimi) prefill input prep.

The model-agnostic engine helpers (mesh open, H2D service, trace loading) live in
the common package at ``models.demos.common.prefill.runners.runner_utils``. What
remains here is the one piece of model-specific glue the runtime needs:
``prepare_prefill_input_tensor`` (the SP-sharded chunk input), which backs
``TtPrefillRuntime.make_chunk_input``, plus the MTP shift-window uploads built on top of it.

The MTP index algebra itself lives one layer down in
``models.demos.deepseek_v3_d_p.tt.mtp_prefill.token_windows``, which has no ``ttnn`` import so it
can be proven without a mesh; what is here is only the upload.

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
    """Shard and upload token IDs to device as a prefill input tensor.

    Produces an SP-sharded uint32 ROW_MAJOR DRAM tensor of shape
    [sp_factor, 1, len(token_ids) // sp_factor] — the format expected by
    TtPrefillTransformer.forward.
    """
    isl_per_chip = len(token_ids) // sp_factor
    if is_balanced:
        chunk_order = create_balanced_chunk_order(sp_factor)
        t = torch.tensor(token_ids, dtype=torch.int64).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        t = reorder_tensor_chunks(t, chunk_order, seq_dim=2)
        token_ids_sharded = t.squeeze(0).squeeze(-1).reshape(sp_factor, 1, isl_per_chip)
    else:
        token_ids_sharded = torch.tensor(token_ids, dtype=torch.int64).reshape(sp_factor, 1, isl_per_chip)
    return ttnn.from_torch(
        token_ids_sharded,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(sp_axis, None)),
    )


def prepare_prefill_mtp_window(
    token_ids: list[int],
    mesh_device: ttnn.MeshDevice,
    sp_factor: int,
    is_balanced: bool,
    mesh_shape: tuple,
    sp_axis: int,
) -> ttnn.Tensor:
    """Upload one MTP shift-window, sharded exactly like this chunk's trunk input.

    ``is_balanced`` MUST equal the value used for the trunk input of the same chunk. Sharding is a
    fixed row -> position permutation applied to the window's contents, so applying the *same* one
    to a shifted window puts ``t_{p+k}`` on the row whose hidden is at ``p`` -- under either layout.
    Applying a *different* one pairs every row with the wrong hidden, silently and with no shape
    error. Chunked prefill is non-balanced throughout (``TtPrefillRuntime.make_chunk_input`` passes
    ``is_balanced=False``), so in practice this is False; the parameter exists so the coupling is
    stated rather than assumed.
    """
    return prepare_prefill_input_tensor(token_ids, mesh_device, sp_factor, is_balanced, mesh_shape, sp_axis)


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
