# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# On-device I2I cond encode: resolution-aware VAEEncoderTTNN + TT patch/ViT inject.
# Tokenizer/template stay on host; VAE encode / patch scatter / ViT run on TTNN.

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
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
from models.experimental.hunyuan_image_3_0.ref.vae.encoder import IN_CHANNELS, PIXEL_H, PIXEL_T, PIXEL_W
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


def load_tt_vae_encoder(
    mesh_device,
    *,
    pixel_t: int = PIXEL_T,
    pixel_h: int = PIXEL_H,
    pixel_w: int = PIXEL_W,
    dtype=ttnn.bfloat16,
) -> VAEEncoderTTNN:
    return VAEEncoderTTNN(
        mesh_device,
        dtype=dtype,
        pixel_t=pixel_t,
        pixel_h=pixel_h,
        pixel_w=pixel_w,
    )


_TT_VAE_ENCODER_CACHE: dict[tuple[int, int, int, int], VAEEncoderTTNN] = {}


def get_tt_vae_encoder(
    mesh_device,
    pixel_h: int,
    pixel_w: int,
    *,
    pixel_t: int = PIXEL_T,
    dtype=ttnn.bfloat16,
) -> VAEEncoderTTNN:
    """Return a cached ``VAEEncoderTTNN`` built for the cond image spatial size."""
    key = (id(mesh_device), pixel_t, pixel_h, pixel_w)
    enc = _TT_VAE_ENCODER_CACHE.get(key)
    if enc is None:
        enc = load_tt_vae_encoder(
            mesh_device,
            pixel_t=pixel_t,
            pixel_h=pixel_h,
            pixel_w=pixel_w,
            dtype=dtype,
        )
        _TT_VAE_ENCODER_CACHE[key] = enc
    return enc


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


def _vae_cond_as_tensor(image) -> torch.Tensor:
    """Return preprocessed cond VAE tensor unchanged (matches host ``vae_encode_image``).

    ``CondImage.vae_image`` is already ``resize_and_crop``'d by ``HunyuanImage3ImageProcessor``
    to the resolution-group target (typically non-square). Do **not** re-stretch to 1024² —
    that breaks aspect ratio and ``vae_image_mask`` token counts.
    """
    x = image.detach().clone() if isinstance(image, torch.Tensor) else torch.as_tensor(image)
    if x.ndim not in (3, 4):
        raise ValueError(f"cond VAE image must be [C,H,W] or [B,C,H,W], got shape {tuple(x.shape)}")
    return x.float()


def _vae_cond_spatial_hw(image) -> tuple[int, int]:
    x = _vae_cond_as_tensor(image)
    if x.ndim == 3:
        return int(x.shape[1]), int(x.shape[2])
    return int(x.shape[2]), int(x.shape[3])


def _resolve_tt_vae_encoder(
    mesh_device,
    image,
    vae_encoder: VAEEncoderTTNN | None,
) -> VAEEncoderTTNN:
    pixel_h, pixel_w = _vae_cond_spatial_hw(image)
    if vae_encoder is not None and vae_encoder.pixel_h == pixel_h and vae_encoder.pixel_w == pixel_w:
        return vae_encoder
    return get_tt_vae_encoder(mesh_device, pixel_h, pixel_w)


def _zero_timesteps(batch: int) -> torch.Tensor:
    return torch.zeros(batch, dtype=torch.float32)


def _host_bcthw_for_vae_encode(image, *, dtype=ttnn.bfloat16) -> torch.Tensor:
    """Host cond image tensor -> padded BCTHW (no device upload)."""
    import torch.nn.functional as F

    x = prepare_vae_encode_input(_vae_cond_as_tensor(image)).float()
    padded_c = aligned_channels(IN_CHANNELS)
    if x.shape[1] < padded_c:
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, padded_c - x.shape[1]))
    return x.bfloat16() if dtype == ttnn.bfloat16 else x.float()


def _host_bthwc_for_vae_encode(image, *, dtype=ttnn.bfloat16) -> torch.Tensor:
    return _host_bcthw_for_vae_encode(image, dtype=dtype).permute(0, 2, 3, 4, 1).contiguous()


def _pad_and_upload_bcthw(mesh_device, image, *, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Host cond image tensor -> padded BCTHW on device."""
    host = _host_bcthw_for_vae_encode(image, dtype=dtype)
    x_bcthw = ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return bcthw_to_bthwc(x_bcthw)


def vae_encode_image_tt(
    mesh_device,
    vae_encoder: VAEEncoderTTNN | None,
    image,
    *,
    scaling_factor: float | None = None,
    shift_factor: float | None = None,
    seed: int | None = None,
    cond_encode_trace: bool = True,
) -> tuple[torch.Tensor, ttnn.Tensor]:
    """Encode one cond image on device -> ``(timesteps, latents_bthwc)`` on device."""
    from models.experimental.hunyuan_image_3_0.tt.trace_config import cond_encode_trace_enabled

    t0 = time.perf_counter()
    vae_encoder = _resolve_tt_vae_encoder(mesh_device, image, vae_encoder)
    pixel_h, pixel_w = _vae_cond_spatial_hw(image)
    traced = cond_encode_trace and cond_encode_trace_enabled()
    if traced:
        from models.experimental.hunyuan_image_3_0.tt.cond_encode_trace import run_vae_encode_traced

        host_bthwc = _host_bthwc_for_vae_encode(image)
        h_tt = run_vae_encode_traced(
            mesh_device,
            vae_encoder,
            host_bthwc,
            pixel_h=pixel_h,
            pixel_w=pixel_w,
        )
    else:
        x_tt = _pad_and_upload_bcthw(mesh_device, image)
        h_tt = vae_encoder(x_tt)
        ttnn.deallocate(x_tt, force=False)
    z_tt = diagonal_gaussian_sample_tt(h_tt, mesh_device=mesh_device, seed=seed)
    z_tt = squeeze_temporal_tt(z_tt)
    if scaling_factor is not None or shift_factor is not None:
        z_tt = apply_vae_latent_scaling_tt(z_tt, scaling_factor=scaling_factor, shift_factor=shift_factor)
    bsz = int(z_tt.shape[0])
    t_tt = _zero_timesteps(bsz)
    ttnn.synchronize_device(mesh_device)
    mode = "trace" if traced else "eager"
    print(
        f"[cond_encode] VAE encode ({pixel_h}x{pixel_w}) {mode} total={(time.perf_counter() - t0) * 1000:.1f} ms",
        flush=True,
    )
    return t_tt, z_tt


def encode_cond_images_tt(
    mesh_device,
    batch_cond_images: list[list[Any]] | None,
    *,
    model_dir: Path | None = None,
    vae_encoder: VAEEncoderTTNN | None = None,
    cfg_factor: int = 1,
    scaling_factor: float | None = None,
    shift_factor: float | None = None,
    generator: torch.Generator | None = None,
    seed: int | None = None,
    cond_encode_trace: bool = True,
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
                cond_encode_trace=cond_encode_trace,
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


def _vit_host_tensors(vit_tensor) -> tuple[torch.Tensor, tuple[tuple[int, int], ...], torch.Tensor]:
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
    return pv.float(), hw, mask


def _vit_tensor_to_vision_inputs(mesh_device, vit_tensor) -> Any:
    pv, hw, mask = _vit_host_tensors(vit_tensor)
    return to_vision_inputs(mesh_device, pv, hw, mask)


def _encode_vit_cond_image_tt(
    mesh_device,
    vision: HunyuanTtSiglip2Vision,
    aligner: HunyuanTtLightProjector,
    vit_tensor,
    *,
    cond_encode_trace: bool = True,
) -> ttnn.Tensor:
    from models.experimental.hunyuan_image_3_0.tt.trace_config import cond_encode_trace_enabled

    pv, hw, mask = _vit_host_tensors(vit_tensor)
    if cond_encode_trace and cond_encode_trace_enabled():
        from models.experimental.hunyuan_image_3_0.tt.cond_encode_trace import run_vit_encode_traced

        return run_vit_encode_traced(mesh_device, vision, aligner, pv, mask, spatial_shapes_hw=hw)
    vi = to_vision_inputs(mesh_device, pv, hw, mask)
    return encode_cond_vision(vision, aligner, vi)


def _encode_vit_batch_row_tt(
    mesh_device,
    vision: HunyuanTtSiglip2Vision,
    aligner: HunyuanTtLightProjector,
    cond_images: list[Any],
    *,
    cond_encode_trace: bool = True,
) -> ttnn.Tensor:
    from models.experimental.hunyuan_image_3_0.tt.trace_config import cond_encode_trace_enabled

    t0 = time.perf_counter()
    traced = cond_encode_trace and cond_encode_trace_enabled()
    chunks: list[ttnn.Tensor] = []
    for cond_image in cond_images:
        from models.experimental.hunyuan_image_3_0.ref.tokenizer.image_info import CondImage

        vit_t = cond_image.vit_image if isinstance(cond_image, CondImage) else cond_image
        emb_tt = _encode_vit_cond_image_tt(mesh_device, vision, aligner, vit_t, cond_encode_trace=cond_encode_trace)
        chunks.append(emb_tt)
    if len(chunks) == 1:
        out = chunks[0]
    else:
        out = ttnn.concat(chunks, dim=1)
        for c in chunks[1:]:
            ttnn.deallocate(c, force=False)
    ttnn.synchronize_device(mesh_device)
    mode = "trace" if traced else "eager"
    print(
        f"[cond_encode] ViT encode ({len(cond_images)} image(s)) {mode} "
        f"total={(time.perf_counter() - t0) * 1000:.1f} ms",
        flush=True,
    )
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
    model_dir: Path | None = None,
    vae_encoder: VAEEncoderTTNN | None = None,
    vision: HunyuanTtSiglip2Vision,
    aligner: HunyuanTtLightProjector,
    cond_patch_embed: HunyuanTtUNetDown,
    cond_time_embed: HunyuanTtTimestepEmbedder,
    cond_timestep_emb: HunyuanTtTimestepEmbedder,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    vae_scaling_factor: float | None = None,
    vae_shift_factor: float | None = None,
    cond_encode_trace: bool = True,
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
        bundle.batch_cond_images,
        model_dir=model_dir,
        vae_encoder=vae_encoder,
        cfg_factor=bundle.cfg_factor,
        scaling_factor=vae_scaling_factor,
        shift_factor=vae_shift_factor,
        generator=generator,
        seed=seed,
        cond_encode_trace=cond_encode_trace,
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
        vit_emb_cache: dict[int, ttnn.Tensor] = {}
        for row in range(bsz):
            # CFG duplicates token rows (cond + uncond) but cond images are stored once per prompt.
            cond_idx = row % n_cond_rows
            cond_images = bundle.batch_cond_images[cond_idx]
            if cond_idx not in vit_emb_cache:
                vit_emb_cache[cond_idx] = _encode_vit_batch_row_tt(
                    mesh_device, vision, aligner, cond_images, cond_encode_trace=cond_encode_trace
                )
            cached = vit_emb_cache[cond_idx]
            vit_row = ttnn.allocate_tensor_on_device(cached.spec, cached.device())
            ttnn.copy(cached, vit_row)
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
        for emb in vit_emb_cache.values():
            ttnn.deallocate(emb, force=False)

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
    model_dir: Path | None = None,
    vae_encoder: VAEEncoderTTNN | None = None,
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
            model_dir=model_dir,
            vae_encoder=vae_encoder,
            vision=vision,
            aligner=aligner,
            cond_patch_embed=cond_patch_embed,
            cond_time_embed=cond_time_embed,
            cond_timestep_emb=cond_timestep_emb,
            seed=seed,
            cond_encode_trace=False,
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
    model_dir: Path | None = None,
    vae_encoder: VAEEncoderTTNN | None = None,
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
        model_dir=model_dir,
        vae_encoder=vae_encoder,
        vision=vision,
        aligner=aligner,
        cond_patch_embed=cond_patch_embed,
        cond_time_embed=cond_time_embed,
        cond_timestep_emb=cond_timestep_emb,
        seed=seed,
    )
    return enrich_bundle_attention(bundle, processor)
