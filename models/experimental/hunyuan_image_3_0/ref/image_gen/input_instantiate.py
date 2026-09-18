# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Scatter VAE latents and timestep embeddings into the token sequence (host PyTorch).
#
# Mirrors upstream:
#   HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
#     instantiate_vae_image_tokens()  (~1806)
#     instantiate_vit_image_tokens()  (~1903)
#     instantiate_continuous_tokens()   (~1964)

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from models.experimental.hunyuan_image_3_0.ref.vision.siglip2 import LightProjector, Siglip2VisionTransformer

    from .patch_embed import UNetDown
    from .timestep_embedder import TimestepEmbedder


def instantiate_vae_image_tokens(
    hidden_states: Tensor,
    timesteps: Tensor | list[Tensor],
    images: Tensor | list[Tensor] | list[list[Tensor]],
    image_mask: Tensor,
    patch_embed: UNetDown,
    time_embed: TimestepEmbedder,
) -> Tensor:
    """Patch-embed VAE latents and scatter into ``hidden_states`` at ``image_mask``."""
    if hidden_states is None:
        raise ValueError("hidden_states is required for I2I cond path")

    bsz, seqlen, n_embd = hidden_states.shape

    if isinstance(images, Tensor):
        assert images.ndim == 4, f"images must be 4-D, got {images.ndim}-D"
        assert isinstance(timesteps, Tensor), f"timesteps must be 1-D tensor, got {type(timesteps)}"
        index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
        t_emb = time_embed(timesteps)
        image_seq, _, _ = patch_embed(images, t_emb)
        image_scatter_index = index.masked_select(image_mask.bool()).reshape(bsz, -1)
        hidden_states.scatter_(
            dim=1,
            index=image_scatter_index.unsqueeze(-1).expand(-1, -1, n_embd),
            src=image_seq,
        )
        return hidden_states

    index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
    for i, (image_i, t_i) in enumerate(zip(images, timesteps)):
        t_i_emb = time_embed(t_i)
        if isinstance(image_i, Tensor):
            image_i_seq, _, _ = patch_embed(image_i, t_i_emb)
        elif isinstance(image_i, list):
            image_i_seq_list = []
            for j in range(len(image_i)):
                image_ij = image_i[j].unsqueeze(0)
                image_i_seq_j = patch_embed(image_ij, t_i_emb[j : j + 1])[0]
                image_i_seq_list.append(image_i_seq_j)
            image_i_seq = torch.cat(image_i_seq_list, dim=1)
        else:
            raise TypeError(f"image_i must be Tensor or list, got {type(image_i)}")

        image_i_index = index[i : i + 1].masked_select(image_mask[i : i + 1].bool()).reshape(1, -1)
        hidden_states[i : i + 1].scatter_(
            dim=1,
            index=image_i_index.unsqueeze(-1).expand(-1, -1, n_embd),
            src=image_i_seq.reshape(1, -1, n_embd),
        )
    return hidden_states


def _forward_vision_encoder(
    vision_model: Siglip2VisionTransformer,
    aligner: LightProjector,
    images: Tensor,
    image_kwargs: dict[str, Tensor] | None,
) -> Tensor:
    from models.experimental.hunyuan_image_3_0.ref.vision.siglip2 import forward_vision_with_aligner

    image_kwargs = image_kwargs or {}
    return forward_vision_with_aligner(
        vision_model,
        aligner,
        images,
        spatial_shapes=image_kwargs["spatial_shapes"],
        attention_mask=image_kwargs["attention_mask"],
    )


def instantiate_vit_image_tokens(
    hidden_states: Tensor,
    images: Tensor | list[Tensor],
    image_masks: Tensor,
    image_kwargs: dict[str, list[Tensor]] | None,
    vision_model: Siglip2VisionTransformer,
    aligner: LightProjector,
) -> Tensor:
    """Encode with SigLIP2 + aligner and scatter into ``hidden_states`` at ``image_masks``."""
    bsz, seqlen, n_embd = hidden_states.shape
    index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

    if isinstance(images, Tensor):
        assert images.ndim in (3, 4, 5), f"images must be 3-D/4-D/5-D, got {images.ndim}-D"
        kwargs = dict(image_kwargs) if image_kwargs is not None else {}
        if images.ndim in (4, 5):
            batch, num_images = images.shape[:2]
            images = images.view(batch * num_images, *images.shape[2:])
            for key, value in list(kwargs.items()):
                kwargs[key] = value.reshape(batch * num_images, *value.shape[2:])
        else:
            num_images = 1

        image_embeds = _forward_vision_encoder(vision_model, aligner, images, kwargs)
        image_seqlen = image_embeds.size(1)
        image_scatter_index = index.masked_select(image_masks.bool()).reshape(bsz, -1)
        hidden_states.scatter_(
            dim=1,
            index=image_scatter_index.unsqueeze(-1).expand(-1, -1, n_embd),
            src=image_embeds.reshape(bsz, num_images * image_seqlen, n_embd),
        )
        return hidden_states

    if not isinstance(images, list):
        raise ValueError(f"images must be Tensor or list, got {type(images)}")

    for i, image in enumerate(images):
        cur_kwargs = {k: v[i] for k, v in image_kwargs.items()} if image_kwargs is not None else {}
        image_embed = _forward_vision_encoder(vision_model, aligner, image, cur_kwargs)
        num_images, image_seqlen, _ = image_embed.shape
        image_embed = image_embed.reshape(num_images * image_seqlen, n_embd)
        image_scatter_index = index[i : i + 1].masked_select(image_masks[i : i + 1].bool()).reshape(1, -1)
        hidden_states[i : i + 1].scatter_(
            dim=1,
            index=image_scatter_index.unsqueeze(-1).expand(-1, -1, n_embd),
            src=image_embed.reshape(1, -1, n_embd),
        )
    return hidden_states


def instantiate_continuous_tokens(
    hidden_states: Tensor,
    timesteps: Tensor | list[Tensor] | None,
    timesteps_index: Tensor | list[Tensor] | None,
    timestep_emb: TimestepEmbedder,
) -> Tensor:
    """Scatter ``timestep_emb`` outputs at ``timesteps_index`` (cond/gen timestep placeholders)."""
    if timesteps is None or timesteps_index is None:
        return hidden_states

    bsz, _, n_embd = hidden_states.shape

    if isinstance(timesteps, list):
        for i, timestep in enumerate(timesteps):
            timestep_src = timestep_emb(timestep)
            hidden_states[i : i + 1].scatter_(
                dim=1,
                index=timesteps_index[i].unsqueeze(0).unsqueeze(-1).expand(1, -1, n_embd),
                src=timestep_src.reshape(1, -1, n_embd),
            )
        return hidden_states

    timesteps_src = timestep_emb(timesteps.reshape(-1))
    hidden_states.scatter_(
        dim=1,
        index=timesteps_index.unsqueeze(-1).expand(-1, -1, n_embd),
        src=timesteps_src.reshape(bsz, -1, n_embd),
    )
    return hidden_states
