# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN denoise-forward helpers for DiffusionGemma canvas attention.

This is the short-prompt W2 path: canvas queries attend to the frozen prompt
prefix plus the current canvas through an explicit all-attend mask. Sampling,
self-conditioning, and the multi-layer generation loop live in later W3/W4
helpers; this module owns the real masked attention wiring.
"""

from __future__ import annotations

import torch
import ttnn

from models.demos.gemma4.tt.attention.kv_phase import KVCachePhase
from models.experimental.diffusion_gemma.reference.attention_mask import build_canvas_denoise_mask

NEG = -1.0e9


def _replicate_mapper(mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None


def build_device_canvas_denoise_mask(
    mesh_device,
    *,
    prompt_len: int,
    canvas_len: int,
    dtype=ttnn.bfloat16,
):
    """Build the canonical all-attend `[1, 1, C, P+C]` denoise mask on device."""
    mask = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        local_window=False,
        neg_inf=NEG,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, prompt_len + canvas_len)
    return ttnn.from_torch(
        mask,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=_replicate_mapper(mesh_device),
    )


def denoise_attention_forward(
    tt_model,
    *,
    layer_idx: int,
    prompt_hidden,
    canvas_hidden,
    attn_mask=None,
):
    """Run one DiffusionGemma denoise attention layer on canvas hidden states.

    Args:
        tt_model: `Gemma4Model` carrying the reused DiffusionGemma decoder weights.
        layer_idx: decoder layer to run.
        prompt_hidden: frozen prompt hidden states `[1, 1, P, H]` on device.
        canvas_hidden: current canvas hidden states `[1, 1, C, H]` on device.
        attn_mask: optional prebuilt `[1, 1, C, P+C]` additive mask on device.

    Returns:
        The attention output for the canvas positions `[1, 1, C, H]`.
    """
    prompt_len = prompt_hidden.shape[-2]
    canvas_len = canvas_hidden.shape[-2]
    kv_hidden = ttnn.concat([prompt_hidden, canvas_hidden], dim=2)
    created_mask = attn_mask is None
    if created_mask:
        attn_mask = build_device_canvas_denoise_mask(
            tt_model.mesh_device,
            prompt_len=prompt_len,
            canvas_len=canvas_len,
        )

    out = tt_model.layers[layer_idx].self_attn(
        canvas_hidden,
        rope_mats=tt_model._get_rope_mats(layer_idx, seq_len=prompt_len + canvas_len),
        is_decode=False,
        kv_phase=KVCachePhase.DENOISE_READONLY,
        attn_mask=attn_mask,
        kv_hidden_states=kv_hidden,
        q_rope_offset=prompt_len,
    )
    kv_hidden.deallocate(True)
    if created_mask:
        attn_mask.deallocate(True)
    return out
