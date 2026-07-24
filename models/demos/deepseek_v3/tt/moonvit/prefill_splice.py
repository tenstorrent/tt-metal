# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Host-roundtrip vision-text splice for DeepSeek-V3 multimodal prefill.

Mirrors HF's `inputs_embeds[input_ids == image_token_index] = image_features`
from KimiVLForConditionalGeneration, but expressed for the tt-metal flow:

  1. Text tokens go through `Embedding2D.forward_prefill` on device, producing
     a row-sharded `[1, 1, seq_len/num_rows, hidden]` tensor.
  2. We gather that to host as a `[1, 1, seq_len, hidden]` torch tensor.
  3. The vision-token embeddings (output of MoonViT) are scattered into
     positions where the original tokens were `media_placeholder_token_id`.
  4. The fused tensor is pushed back to device in the same row-sharded
     layout the embedding step produced. The caller then hands it to
     `RowBatchedModel.forward_prefill_from_embeddings`.

The host roundtrip is the v1 choice (Gemma3 pattern); device-side splice
is a deferred optimization — see plan Deferred #1.

For the single-device prototype on Blackhole, "row-sharded" reduces to
"replicated on the one device", so the layout question is trivial. For
mesh deployment (DUAL Galaxy 8×8 etc.), the splice still works because
we round-trip through host — the mesh layout is preserved by symmetry.
"""
from __future__ import annotations

from typing import Optional

import torch

import ttnn


def _is_mesh_device(device) -> bool:
    return type(device).__name__ == "MeshDevice"


def splice_vision_into_text_embeddings(
    text_embedded: torch.Tensor,
    tokens: torch.Tensor,
    vision_tokens: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Host-side `masked_scatter` of vision-token embeddings into text embeddings.

    Args:
        text_embedded: shape ``[*, seq_len, hidden]`` (the prefix dims are
            preserved). dtype matches the text embedding output.
        tokens: shape ``[*, seq_len]`` uint/int. Must broadcast with
            text_embedded over its non-final dim. Positions where
            ``tokens == image_token_id`` are the splice targets.
        vision_tokens: shape ``[L_vision, hidden]`` — the vision-token
            embeddings to splice in. ``L_vision`` MUST equal the count of
            ``image_token_id`` occurrences in ``tokens``.
        image_token_id: the placeholder id to splice on.

    Returns:
        new tensor with the same shape and dtype as ``text_embedded`` but
        with vision tokens substituted at image-token positions.
    """
    if text_embedded.shape[:-1] != tokens.shape:
        raise ValueError(
            f"text_embedded shape {tuple(text_embedded.shape)} and tokens shape "
            f"{tuple(tokens.shape)} must agree on all dims except the last (hidden)"
        )
    if vision_tokens.ndim != 2:
        raise ValueError(f"vision_tokens must be 2D (L_vision, hidden); got shape {tuple(vision_tokens.shape)}")
    if vision_tokens.shape[-1] != text_embedded.shape[-1]:
        raise ValueError(
            f"vision_tokens hidden {vision_tokens.shape[-1]} != " f"text_embedded hidden {text_embedded.shape[-1]}"
        )

    image_mask = tokens == image_token_id
    n_image_positions = int(image_mask.sum().item())
    if n_image_positions != vision_tokens.shape[0]:
        raise ValueError(
            f"vision_tokens has {vision_tokens.shape[0]} rows but tokens contains "
            f"{n_image_positions} occurrences of image_token_id={image_token_id}"
        )

    # Cast vision tokens to the text-embedded dtype so masked_scatter type-matches.
    vision_cast = vision_tokens.to(text_embedded.dtype)

    # Expand the mask over the hidden dim for masked_scatter.
    expanded_mask = image_mask.unsqueeze(-1).expand_as(text_embedded)

    # HF reference uses `inputs_embeds[input_ids == image_token_index] = image_features`
    # which is equivalent to masked_scatter with the flattened source — both pull
    # rows from vision_cast in row-major order, landing them at the True positions
    # in row-major order across the masked tensor. Use masked_scatter (returns a new
    # tensor, doesn't mutate text_embedded).
    return text_embedded.masked_scatter(expanded_mask, vision_cast)


def splice_vision_via_host(
    mesh_device,
    text_embedded_tt: ttnn.Tensor,
    tokens: torch.Tensor,
    vision_tokens: torch.Tensor,
    image_token_id: int,
    dtype=ttnn.bfloat16,
    mesh_composer: Optional["ttnn.MeshComposer"] = None,
    mesh_mapper: Optional["ttnn.MeshMapper"] = None,
) -> ttnn.Tensor:
    """Splice vision tokens into device-resident text embeddings via a host roundtrip.

    Device → host → splice (on host, reusing ``splice_vision_into_text_embeddings``)
    → device. This is NOT an on-device scatter — the splice itself runs on host;
    a true on-device splice is deferred (see module docstring). The function only
    handles the gather/push so callers holding device tensors don't have to.

    Args:
        mesh_device: the ttnn mesh device (or single device) owning ``text_embedded_tt``.
        text_embedded_tt: device tensor produced by ``Embedding2D.forward_prefill``.
            Shape on device is ``[1, 1, seq_len_local, hidden]`` where ``seq_len_local``
            is ``seq_len / num_rows`` on a multi-row mesh, or ``seq_len`` on a
            single device.
        tokens: host torch tensor (the original tokens), shape ``[1, 1, seq_len]``.
            Used to locate image positions; the row-sharded device tensor doesn't
            know which positions are image tokens, so the host needs to keep them.
        vision_tokens: host torch tensor of vision-token embeddings, shape
            ``(L_vision, hidden)``.
        image_token_id: placeholder token id used in ``tokens``.
        dtype: target ttnn dtype for the pushed-back tensor.
        mesh_composer: optional ``ttnn.ConcatMeshToTensor`` (or similar) used to
            gather the device tensor to host. If None, picks a sensible default
            based on whether ``mesh_device`` is a MeshDevice.
        mesh_mapper: optional ``ttnn.ReplicateTensorToMesh`` (or similar) used to
            push back. If None, picks a sensible default.

    Returns:
        ttnn tensor in the same layout as ``text_embedded_tt``.
    """
    is_mesh = _is_mesh_device(mesh_device)

    # Default composer: replicate path is just to_torch with no composer for
    # single-device, ConcatMeshToTensor along seq dim for mesh.
    if mesh_composer is None and is_mesh:
        mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=2)
    if mesh_mapper is None:
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    # 1. Gather to host. For multi-row meshes, ConcatMeshToTensor along seq dim
    #    reconstructs the full-seq tensor; for single device this is a no-op.
    text_pt = ttnn.to_torch(text_embedded_tt, mesh_composer=mesh_composer)

    # If a mesh composer was used and the result has a stacked-batch dim from
    # replication artifacts, drop to the first row (caller can override via
    # explicit mesh_composer if more nuance is needed).
    if is_mesh and text_pt.shape[0] != 1:
        text_pt = text_pt[:1]

    # 2. Splice on host.
    fused_pt = splice_vision_into_text_embeddings(text_pt, tokens, vision_tokens, image_token_id)

    # 3. Push fused tensor back to device in the same layout.
    return ttnn.from_torch(
        fused_pt.contiguous(),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
