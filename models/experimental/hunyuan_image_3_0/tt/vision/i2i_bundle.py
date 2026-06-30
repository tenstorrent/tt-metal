# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# On-device I2I cond encode: VAEEncoderTTNN + HunyuanTtSiglip2Vision + aligner.
# Tokenizer/template stay on host; tensor math runs on TTNN (VAE encoder pattern).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import ttnn

from models.experimental.hunyuan_image_3_0.ref.cond_vae_encode import prepare_vae_encode_input
from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import (
    GenImageHostInputs,
    enrich_bundle_attention,
    prepare_i2i_inputs,
    prepare_recaption_inputs,
)
from models.experimental.hunyuan_image_3_0.ref.vae.encoder import IN_CHANNELS, PIXEL_H, PIXEL_W
from models.experimental.hunyuan_image_3_0.tt.image_gen.cond_instantiate import (
    instantiate_continuous_tokens_tt,
    instantiate_vae_image_tokens_tt,
)
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder
from models.experimental.hunyuan_image_3_0.tt.vision.i2i import encode_cond_vision, inject_cond_vision
from models.experimental.hunyuan_image_3_0.tt.vision.preprocess import find_image_token_spans, to_vision_inputs
from models.experimental.hunyuan_image_3_0.tt.vision.siglip2 import (
    HunyuanTtLightProjector,
    HunyuanTtSiglip2Vision,
)
from models.experimental.hunyuan_image_3_0.tt.vae.cond_posterior import (
    apply_vae_latent_scaling_tt,
    diagonal_gaussian_sample_tt,
    squeeze_temporal_tt,
)
from models.experimental.hunyuan_image_3_0.tt.vae.decoder import bcthw_to_bthwc
from models.experimental.hunyuan_image_3_0.tt.vae.encoder import VAEEncoderTTNN
from models.experimental.hunyuan_image_3_0.tt.wte import HunyuanTtWte
from models.tt_dit.utils.conv3d import aligned_channels


@dataclass
class CondVaeEncodeTT:
    """On-device cond VAE latents (``BTHWC``) and zero timesteps."""

    cond_vae_images: ttnn.Tensor | list[ttnn.Tensor] | list[list[ttnn.Tensor]] | None
    cond_timesteps: torch.Tensor | list[torch.Tensor] | None


def _is_dropped_vit_layer(key: str, num_layers: int) -> bool:
    prefix = "encoder.layers."
    if not key.startswith(prefix):
        return False
    return int(key[len(prefix) :].split(".", 1)[0]) >= num_layers


def _filter_vision_state_dict(sd: dict, num_layers: int) -> dict:
    return {k: v for k, v in sd.items() if not k.startswith("head.") and not _is_dropped_vit_layer(k, num_layers)}


def load_tt_vision_stack(
    mesh_device,
    vision_state_dict: dict,
    aligner_state_dict: dict,
    *,
    num_layers: int = 27,
    weight_dtype=ttnn.bfloat16,
) -> tuple[HunyuanTtSiglip2Vision, HunyuanTtLightProjector]:
    vision_sd = _filter_vision_state_dict(vision_state_dict, num_layers)
    vision = HunyuanTtSiglip2Vision(mesh_device, vision_sd, num_layers=num_layers, weight_dtype=weight_dtype)
    aligner = HunyuanTtLightProjector(mesh_device, aligner_state_dict, weight_dtype=weight_dtype)
    return vision, aligner


def load_tt_vae_encoder(mesh_device, *, dtype=ttnn.bfloat16) -> VAEEncoderTTNN:
    return VAEEncoderTTNN(mesh_device, dtype=dtype)


def load_tt_cond_patch_embed(
    mesh_device,
    patch_embed_sd: dict,
    *,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    dtype=ttnn.bfloat16,
) -> HunyuanTtUNetDown:
    return HunyuanTtUNetDown(
        mesh_device,
        {f"patch_embed.{k}": v for k, v in patch_embed_sd.items()},
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dtype=dtype,
    )


def load_tt_cond_timestep_embedders(
    mesh_device,
    *,
    hidden_size: int,
    time_embed_sd: dict,
    timestep_emb_sd: dict,
    dtype=ttnn.bfloat16,
) -> tuple[HunyuanTtTimestepEmbedder, HunyuanTtTimestepEmbedder]:
    time_embed = HunyuanTtTimestepEmbedder(
        mesh_device,
        hidden_size,
        {f"time_embed.{k}": v for k, v in time_embed_sd.items()},
        "time_embed",
        weight_dtype=dtype,
    )
    timestep_emb = HunyuanTtTimestepEmbedder(
        mesh_device,
        hidden_size,
        {f"timestep_emb.{k}": v for k, v in timestep_emb_sd.items()},
        "timestep_emb",
        weight_dtype=dtype,
    )
    return time_embed, timestep_emb


def _resize_vae_cond_to_encoder(image):
    """TT VAE encoder is fixed at ``PIXEL_H x PIXEL_W``; resize cond tensors on host."""
    import torch
    import torch.nn.functional as F

    x = image.detach().clone() if isinstance(image, torch.Tensor) else torch.as_tensor(image)
    if x.ndim == 3:
        _, h, w = x.shape
        if (h, w) != (PIXEL_H, PIXEL_W):
            x = F.interpolate(x.unsqueeze(0), size=(PIXEL_H, PIXEL_W), mode="bilinear", align_corners=False).squeeze(0)
    elif x.ndim == 4:
        _, _, h, w = x.shape
        if (h, w) != (PIXEL_H, PIXEL_W):
            x = F.interpolate(x, size=(PIXEL_H, PIXEL_W), mode="bilinear", align_corners=False)
    else:
        raise ValueError(f"cond VAE image must be [C,H,W] or [B,C,H,W], got shape {tuple(x.shape)}")
    return x


def _pad_and_upload_bcthw(mesh_device, image, *, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Host cond image tensor -> padded BCTHW on device."""
    import torch.nn.functional as F

    x = prepare_vae_encode_input(_resize_vae_cond_to_encoder(image)).float()
    padded_c = aligned_channels(IN_CHANNELS)
    if x.shape[1] < padded_c:
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, padded_c - x.shape[1]))
    host = x.bfloat16() if dtype == ttnn.bfloat16 else x.float()
    x_bcthw = ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return bcthw_to_bthwc(x_bcthw)


def _zero_timesteps(batch: int) -> torch.Tensor:
    return torch.zeros(batch, dtype=torch.float32)


def vae_encode_image_tt(
    mesh_device,
    vae_encoder: VAEEncoderTTNN,
    image,
    *,
    scaling_factor: float | None = None,
    shift_factor: float | None = None,
    seed: int | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Encode one cond image on device -> ``(timesteps, latents_bthwc)`` on device."""
    x_tt = _pad_and_upload_bcthw(mesh_device, image)
    h_tt = vae_encoder(x_tt)
    ttnn.deallocate(x_tt, force=False)
    z_tt = diagonal_gaussian_sample_tt(h_tt, mesh_device=mesh_device, seed=seed)
    z_tt = squeeze_temporal_tt(z_tt)
    if scaling_factor is not None or shift_factor is not None:
        z_tt = apply_vae_latent_scaling_tt(z_tt, scaling_factor=scaling_factor, shift_factor=shift_factor)
    bsz = int(z_tt.shape[0])
    t_tt = _zero_timesteps(bsz)
    return t_tt, z_tt


def encode_cond_images_tt(
    mesh_device,
    vae_encoder: VAEEncoderTTNN,
    batch_cond_images: list[list[Any]] | None,
    *,
    cfg_factor: int = 1,
    scaling_factor: float | None = None,
    shift_factor: float | None = None,
    seed: int | None = None,
) -> CondVaeEncodeTT:
    if batch_cond_images is None or len(batch_cond_images) == 0 or len(batch_cond_images[0]) == 0:
        return CondVaeEncodeTT(cond_vae_images=None, cond_timesteps=None)

    from models.experimental.hunyuan_image_3_0.ref.cond_vae_encode import _first_section_type

    section_type = _first_section_type(batch_cond_images)
    if section_type not in ("cond_vae_image", "cond_joint_image"):
        return CondVaeEncodeTT(cond_vae_images=None, cond_timesteps=None)

    if scaling_factor is None:
        scaling_factor = 0.562679178327931

    batch_latents: list[list[ttnn.Tensor]] = []
    batch_t: list[list[torch.Tensor]] = []
    for cond_images in batch_cond_images:
        lat_list: list[ttnn.Tensor] = []
        t_list: list[ttnn.Tensor] = []
        for cond_image in cond_images:
            from models.experimental.hunyuan_image_3_0.ref.tokenizer.image_info import CondImage

            vae_img = cond_image.vae_image if isinstance(cond_image, CondImage) else cond_image
            t_i, z_i = vae_encode_image_tt(
                mesh_device,
                vae_encoder,
                vae_img,
                scaling_factor=scaling_factor,
                shift_factor=shift_factor,
                seed=seed,
            )
            lat_list.append(z_i)
            t_list.append(t_i)
        batch_latents.append(lat_list)
        batch_t.append(t_list)

    if all(len(items) == 1 for items in batch_latents):
        cond_vae_images = ttnn.concat([items[0] for items in batch_latents], dim=0)
        cond_t = torch.cat([items[0] for items in batch_t], dim=0)
        if cfg_factor > 1:
            cond_vae_images = ttnn.repeat(cond_vae_images, (cfg_factor, 1, 1, 1, 1))
            cond_t = cond_t.repeat(cfg_factor)
    else:
        cond_vae_images = batch_latents
        cond_t = [torch.cat(row, dim=0) for row in batch_t]
        if cfg_factor > 1:
            cond_vae_images = cond_vae_images * cfg_factor
            cond_t = cond_t * cfg_factor

    return CondVaeEncodeTT(cond_vae_images=cond_vae_images, cond_timesteps=cond_t)


def _vit_tensor_to_vision_inputs(mesh_device, vit_tensor) -> Any:
    kwargs = vit_tensor.vision_encoder_kwargs
    pv = vit_tensor.unsqueeze(0) if vit_tensor.ndim == 2 else vit_tensor
    spatial = kwargs["spatial_shapes"]
    mask = kwargs["pixel_attention_mask"]
    if spatial.ndim == 1:
        hw = ((int(spatial[0]), int(spatial[1])),)
    else:
        hw = tuple((int(spatial[i][0]), int(spatial[i][1])) for i in range(spatial.shape[0]))
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    return to_vision_inputs(mesh_device, pv.float(), hw, mask)


def _encode_vit_batch_row_tt(
    mesh_device,
    vision: HunyuanTtSiglip2Vision,
    aligner: HunyuanTtLightProjector,
    cond_images: list[Any],
) -> ttnn.Tensor:
    chunks: list[ttnn.Tensor] = []
    for cond_image in cond_images:
        from models.experimental.hunyuan_image_3_0.ref.tokenizer.image_info import CondImage

        vit_t = cond_image.vit_image if isinstance(cond_image, CondImage) else cond_image
        vi = _vit_tensor_to_vision_inputs(mesh_device, vit_t)
        emb_tt = encode_cond_vision(vision, aligner, vi)
        chunks.append(emb_tt)
    if len(chunks) == 1:
        return chunks[0]
    out = ttnn.concat(chunks, dim=1)
    for c in chunks[1:]:
        ttnn.deallocate(c, force=False)
    return out


def _hidden_tt_to_bundle(hidden_tt: ttnn.Tensor, wte_tt: HunyuanTtWte, bundle: GenImageHostInputs):
    """Single host download for ``GenImageHostInputs.inputs_embeds`` (denoise_loop boundary)."""
    bsz = int(bundle.input_ids.shape[0])
    seq = int(bundle.input_ids.shape[1])
    bundle.inputs_embeds = wte_tt.to_torch(hidden_tt, batch=bsz, seq=seq)
    ttnn.deallocate(hidden_tt, force=False)


def _vit_slices_for_row(bundle: GenImageHostInputs, row: int) -> list:
    """ViT inject spans only — not VAE ``<img>`` runs from ``find_image_token_spans``."""
    slices = bundle.vit_image_slices
    if slices is None:
        return find_image_token_spans(bundle.input_ids[row])
    if slices and isinstance(slices[0], slice):
        return list(slices)
    row_slices = slices[row]
    if row_slices is None:
        return []
    if isinstance(row_slices, slice):
        return [row_slices]
    return list(row_slices)


def build_i2i_inputs_embeds_tt(
    bundle: GenImageHostInputs,
    wte_tt: HunyuanTtWte,
    mesh_device,
    *,
    vae_encoder: VAEEncoderTTNN,
    vision: HunyuanTtSiglip2Vision,
    aligner: HunyuanTtLightProjector,
    cond_patch_embed: HunyuanTtUNetDown,
    cond_time_embed: HunyuanTtTimestepEmbedder,
    cond_timestep_emb: HunyuanTtTimestepEmbedder,
    seed: int | None = None,
    vae_scaling_factor: float | None = None,
    vae_shift_factor: float | None = None,
) -> GenImageHostInputs:
    """TT cond VAE + ViT encode and inject — stays on device until bundle export."""
    if bundle.batch_cond_images is None or len(bundle.batch_cond_images[0]) == 0:
        raise ValueError("build_i2i_inputs_embeds_tt requires batch_cond_images on the bundle")
    if bundle.vae_image_mask is None:
        raise ValueError("build_i2i_inputs_embeds_tt requires vae_image_mask on the bundle")

    if vae_scaling_factor is None:
        vae_scaling_factor = 0.562679178327931

    hidden_tt = wte_tt.embed(bundle.input_ids)

    enc = encode_cond_images_tt(
        mesh_device,
        vae_encoder,
        bundle.batch_cond_images,
        cfg_factor=bundle.cfg_factor,
        scaling_factor=vae_scaling_factor,
        shift_factor=vae_shift_factor,
        seed=seed,
    )

    hidden_tt = instantiate_vae_image_tokens_tt(
        hidden_tt,
        enc.cond_vae_images,
        bundle.vae_image_mask,
        cond_patch_embed,
        cond_time_embed,
        mesh_device,
    )

    if bundle.cond_timestep_scatter_index is not None:
        hidden_tt = instantiate_continuous_tokens_tt(
            hidden_tt,
            enc.cond_timesteps,
            bundle.cond_timestep_scatter_index,
            cond_timestep_emb,
            mesh_device,
        )

    if bundle.vit_image_mask is not None and bool(bundle.vit_image_mask.any()):
        bsz = int(hidden_tt.shape[0])
        n_cond_rows = len(bundle.batch_cond_images)
        for row in range(bsz):
            # CFG duplicates token rows (cond + uncond) but cond images are stored once per prompt.
            cond_images = bundle.batch_cond_images[row % n_cond_rows]
            vit_row = _encode_vit_batch_row_tt(mesh_device, vision, aligner, cond_images)
            img_slices = _vit_slices_for_row(bundle, row)
            if not img_slices:
                continue
            row_hidden = ttnn.slice(hidden_tt, [row, 0, 0], [row + 1, hidden_tt.shape[1], hidden_tt.shape[2]])
            row_out = inject_cond_vision(row_hidden, vit_row, img_slices=img_slices)
            ttnn.deallocate(row_hidden, force=False)
            ttnn.deallocate(vit_row, force=False)
            if bsz == 1:
                ttnn.deallocate(hidden_tt, force=False)
                hidden_tt = row_out
            else:
                pre = (
                    ttnn.slice(hidden_tt, [0, 0, 0], [row, hidden_tt.shape[1], hidden_tt.shape[2]]) if row > 0 else None
                )
                post = (
                    ttnn.slice(hidden_tt, [row + 1, 0, 0], [bsz, hidden_tt.shape[1], hidden_tt.shape[2]])
                    if row + 1 < bsz
                    else None
                )
                pieces = [p for p in (pre, row_out, post) if p is not None]
                new_hidden = ttnn.concat(pieces, dim=0)
                if pre is not None:
                    ttnn.deallocate(pre, force=False)
                if post is not None:
                    ttnn.deallocate(post, force=False)
                ttnn.deallocate(row_out, force=False)
                ttnn.deallocate(hidden_tt, force=False)
                hidden_tt = new_hidden

    bundle.cond_vae_images = None
    bundle.cond_timesteps = None
    _hidden_tt_to_bundle(hidden_tt, wte_tt, bundle)
    return bundle


def prepare_recaption_ar_bundle_tt(
    mesh_device,
    tok,
    prompt: str,
    processor,
    wte_tt: HunyuanTtWte,
    *,
    cond_images=None,
    bot_task: str = "recaption",
    system_prompt: str | None = None,
    sequence_template: str | None = None,
    vae_encoder: VAEEncoderTTNN,
    vision: HunyuanTtSiglip2Vision,
    aligner: HunyuanTtLightProjector,
    cond_patch_embed: HunyuanTtUNetDown,
    cond_time_embed: HunyuanTtTimestepEmbedder,
    cond_timestep_emb: HunyuanTtTimestepEmbedder,
    seed: int | None = None,
) -> GenImageHostInputs:
    bundle = prepare_recaption_inputs(
        tok,
        prompt,
        cond_images=cond_images,
        bot_task=bot_task.split("_")[0] if bot_task == "think_recaption" else bot_task,
        system_prompt=system_prompt,
        sequence_template=sequence_template,
    )
    has_cond = bundle.batch_cond_images is not None and len(bundle.batch_cond_images[0]) > 0
    if has_cond:
        bundle = build_i2i_inputs_embeds_tt(
            bundle,
            wte_tt,
            mesh_device,
            vae_encoder=vae_encoder,
            vision=vision,
            aligner=aligner,
            cond_patch_embed=cond_patch_embed,
            cond_time_embed=cond_time_embed,
            cond_timestep_emb=cond_timestep_emb,
            seed=seed,
        )
    else:
        hidden_tt = wte_tt.embed(bundle.input_ids)
        _hidden_tt_to_bundle(hidden_tt, wte_tt, bundle)
    bundle.bot_task = bot_task
    return enrich_bundle_attention(bundle, processor)


def prepare_i2i_denoise_bundle_tt(
    mesh_device,
    tok,
    prompt: str,
    cond_image,
    processor,
    wte_tt: HunyuanTtWte,
    *,
    vae_encoder: VAEEncoderTTNN,
    vision: HunyuanTtSiglip2Vision,
    aligner: HunyuanTtLightProjector,
    cond_patch_embed: HunyuanTtUNetDown,
    cond_time_embed: HunyuanTtTimestepEmbedder,
    cond_timestep_emb: HunyuanTtTimestepEmbedder,
    image_size: str | int | tuple[int, int] | list[int] = 1024,
    cfg_factor: int | None = None,
    sequence_template: str | None = None,
    system_prompt: str | None = None,
    cot_text: str | None = None,
    seed: int | None = None,
) -> GenImageHostInputs:
    bundle = prepare_i2i_inputs(
        tok,
        prompt,
        cond_image,
        image_size=image_size,
        cfg_factor=cfg_factor,
        sequence_template=sequence_template,
        system_prompt=system_prompt,
        cot_text=cot_text,
    )
    bundle = build_i2i_inputs_embeds_tt(
        bundle,
        wte_tt,
        mesh_device,
        vae_encoder=vae_encoder,
        vision=vision,
        aligner=aligner,
        cond_patch_embed=cond_patch_embed,
        cond_time_embed=cond_time_embed,
        cond_timestep_emb=cond_timestep_emb,
        seed=seed,
    )
    return enrich_bundle_attention(bundle, processor)
