# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Host-side T2I / I2I input bundle for device upload (tokenizer → model.wte).
#
# Mapping to HF upstream (modeling_hunyuan_image_3.py forward / gen_image path)
# -----------------------------------------------------------------------------
#  Step | HF                                           | This port (host)
#  -----+----------------------------------------------+----------------------------------
#   1   | apply_chat_template                          | tok.apply_chat_template
#   2   | input_ids / gen_image_mask / scatter indices | GenImageHostInputs fields
#   3   | position_ids = arange(seq_len)               | torch.arange → position_ids
#   4   | build_batch_rope_image_info                  | build_rope_image_info
#   5   | CFG section duplication                      | _expand_sections_for_cfg
#   6   | _encode_cond_image + instantiate_*           | build_i2i_inputs_embeds
#   7   | prepare_full_attn_slices + attn mask         | build_attention_mask_for_bundle
#
# Output is consumed by HunyuanTtModel.embed() and the denoise pipeline on device.
#
# References
# ----------
#   ref/tokenizer/hunyuan_tokenizer.py — HunyuanTokenizer.apply_chat_template
#   ref/tokenizer/chat_template.py     — TokenizerEncodeOutput tensors
#   tt/model.py                        — wte + backbone entry point
#   tt/pipeline.py                     — denoise step consumes latent + conditioning

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from .image_info import CondImage

if TYPE_CHECKING:
    from models.experimental.hunyuan_image_3_0.ref.image_processor import HunyuanImage3ImageProcessor

    from .chat_template import TokenizerEncodeOutput
    from .hunyuan_tokenizer import HunyuanTokenizer


@dataclass
class GenImageHostInputs:
    """Model inputs assembled on host before TTNN upload."""

    input_ids: torch.Tensor
    gen_image_mask: torch.Tensor | None
    vae_image_mask: torch.Tensor | None
    vit_image_mask: torch.Tensor | None
    gen_timestep_scatter_index: torch.Tensor | None
    gen_timestep_r_scatter_index: torch.Tensor | None
    cond_timestep_scatter_index: torch.Tensor | None
    guidance_scatter_index: torch.Tensor | None
    position_ids: torch.Tensor
    rope_image_info: list | None
    batch_size: int
    cfg_factor: int
    seq_len: int
    sections: list[list[dict[str, Any]]]
    vae_image_slices: list | None = None
    vit_image_slices: list | None = None
    joint_image_slices: list | None = None
    gen_image_slices: list | None = None
    full_attn_slices: list[list[slice]] | None = None
    attention_mask: torch.Tensor | None = None
    batch_cond_images: list[list[Any]] | None = None
    cond_vae_images: torch.Tensor | list | None = None
    cond_timesteps: torch.Tensor | list | None = None
    cond_vit_images: torch.Tensor | list | None = None
    cond_vit_image_kwargs: dict[str, list[torch.Tensor]] | None = None
    inputs_embeds: torch.Tensor | None = None


def _expand_sections_for_cfg(
    sections: list[dict[str, Any]] | list[list[dict[str, Any]]],
    *,
    batch_size: int,
    cfg_factor: int,
) -> list[list[dict[str, Any]]]:
    if batch_size == 1 and not isinstance(sections[0], list):
        base = [sections]
    elif isinstance(sections[0], dict):
        base = [sections]
    else:
        base = sections

    rows: list[list[dict[str, Any]]] = []
    for section_row in base:
        rows.append(section_row)
        if cfg_factor > 1:
            rows.append(section_row)
    return rows


def _normalize_all_image_slices(
    all_image_slices,
    batch_rows: int,
) -> list[list[slice]]:
    if all_image_slices is None:
        return [[] for _ in range(batch_rows)]
    if not all_image_slices:
        return [[] for _ in range(batch_rows)]
    if isinstance(all_image_slices[0], slice):
        return [all_image_slices for _ in range(batch_rows)]
    return list(all_image_slices)


def build_rope_image_info(
    sections_list: list[list[dict[str, Any]]],
    all_image_slices: list[list[slice]],
    *,
    rope_type: str = "2d",
    cond_token_attn_type: str = "joint_full",
) -> list | None:
    """Mirror HF ``HunyuanImage3Model.build_batch_rope_image_info`` for T2I and I2I."""
    if rope_type == "default":
        return None
    if rope_type != "2d":
        raise ValueError(f"Unsupported rope_type={rope_type!r}")

    rope_image_info = []
    for sections_i, image_slices in zip(sections_list, all_image_slices):
        rope_2d_image_slices: list[slice] = []
        rope_2d_image_shapes: list[tuple[int, int]] = []
        image_idx = 0
        for section in sections_i:
            if section["type"] in ("gen_image", "cond_vae_image", "cond_vit_image"):
                if image_idx >= len(image_slices):
                    raise ValueError(f"Image index {image_idx} out of range for {len(image_slices)} slice(s)")
                rope_2d_image_slices.append(image_slices[image_idx])
                rope_2d_image_shapes.append((section["token_height"], section["token_width"]))
                image_idx += 1
            elif section["type"] == "cond_joint_image":
                if image_idx + 1 >= len(image_slices):
                    raise ValueError(f"Image index {image_idx + 1} out of range for {len(image_slices)} slice(s)")
                if cond_token_attn_type in ("full", "joint_full"):
                    rope_2d_image_slices.extend([image_slices[image_idx], image_slices[image_idx + 1]])
                    rope_2d_image_shapes.extend(list(zip(section["token_height"], section["token_width"])))
                elif cond_token_attn_type == "full_causal":
                    rope_2d_image_slices.append(image_slices[image_idx])
                    rope_2d_image_shapes.append((section["token_height"][0], section["token_width"][0]))
                elif cond_token_attn_type == "causal":
                    pass
                else:
                    raise NotImplementedError(
                        f"cond_token_attn_type {cond_token_attn_type!r} not supported by build_rope_image_info"
                    )
                image_idx += 2
        rope_image_info.append(list(zip(rope_2d_image_slices, rope_2d_image_shapes)))
    return rope_image_info


def _build_host_inputs_from_template(
    tok: HunyuanTokenizer,
    *,
    output,
    sections,
    batch_size: int,
    cfg_factor: int,
    batch_cond_images: list[list[Any]] | None = None,
) -> GenImageHostInputs:
    batch_rows = output.tokens.shape[0]
    sections_list = _expand_sections_for_cfg(sections, batch_size=batch_size, cfg_factor=cfg_factor)
    all_slices = _normalize_all_image_slices(output.all_image_slices, batch_rows)
    rope_type = tok.config.raw.get("rope_type", "2d")
    cond_token_attn_type = tok.config.raw.get("cond_token_attn_type", "joint_full")
    rope_image_info = build_rope_image_info(
        sections_list,
        all_slices,
        rope_type=rope_type,
        cond_token_attn_type=cond_token_attn_type,
    )

    seq_len = int(output.tokens.shape[1])
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_rows, -1)

    return GenImageHostInputs(
        input_ids=output.tokens,
        gen_image_mask=output.gen_image_mask,
        vae_image_mask=output.vae_image_mask,
        vit_image_mask=output.vit_image_mask,
        gen_timestep_scatter_index=output.gen_timestep_scatter_index,
        gen_timestep_r_scatter_index=output.gen_timestep_r_scatter_index,
        cond_timestep_scatter_index=output.cond_timestep_scatter_index,
        guidance_scatter_index=output.guidance_scatter_index,
        position_ids=position_ids,
        rope_image_info=rope_image_info,
        batch_size=batch_size,
        cfg_factor=cfg_factor,
        seq_len=seq_len,
        sections=sections_list,
        vae_image_slices=output.vae_image_slices,
        vit_image_slices=output.vit_image_slices,
        joint_image_slices=output.joint_image_slices,
        gen_image_slices=output.gen_image_slices,
        batch_cond_images=batch_cond_images,
    )


def tokenizer_output_from_bundle(bundle: GenImageHostInputs) -> TokenizerEncodeOutput:
    """Reconstruct ``TokenizerEncodeOutput`` for ``prepare_full_attn_slices``."""
    from .chat_template import TokenizerEncodeOutput

    return TokenizerEncodeOutput(
        tokens=bundle.input_ids,
        gen_image_mask=bundle.gen_image_mask,
        vae_image_mask=bundle.vae_image_mask,
        vit_image_mask=bundle.vit_image_mask,
        gen_timestep_scatter_index=bundle.gen_timestep_scatter_index,
        gen_timestep_r_scatter_index=bundle.gen_timestep_r_scatter_index,
        cond_timestep_scatter_index=bundle.cond_timestep_scatter_index,
        guidance_scatter_index=bundle.guidance_scatter_index,
        gen_image_slices=bundle.gen_image_slices,
        vae_image_slices=bundle.vae_image_slices,
        vit_image_slices=bundle.vit_image_slices,
        joint_image_slices=bundle.joint_image_slices,
    )


def build_full_attn_slices(
    bundle: GenImageHostInputs,
    processor: HunyuanImage3ImageProcessor,
) -> list[list[slice]]:
    """Mirror upstream ``prepare_full_attn_slices`` per batch row (cond + gen spans)."""
    output = tokenizer_output_from_bundle(bundle)
    batch_rows = int(bundle.input_ids.shape[0])
    return [processor.prepare_full_attn_slices(output, batch_idx=i) for i in range(batch_rows)]


def build_attention_mask_for_bundle(
    bundle: GenImageHostInputs,
    processor: HunyuanImage3ImageProcessor,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build bool attention mask ``[bsz, 1, S, S]`` via ``prepare_full_attn_slices``."""
    from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask

    spans = build_full_attn_slices(bundle, processor)
    return build_attention_mask(bundle.seq_len, spans, bsz=int(bundle.input_ids.shape[0]), device=device)


def enrich_bundle_attention(
    bundle: GenImageHostInputs,
    processor: HunyuanImage3ImageProcessor,
) -> GenImageHostInputs:
    """Populate ``full_attn_slices`` and ``attention_mask`` on the bundle."""
    bundle.full_attn_slices = build_full_attn_slices(bundle, processor)
    bundle.attention_mask = build_attention_mask_for_bundle(bundle, processor)
    return bundle


def get_gen_image_slice(bundle: GenImageHostInputs, row: int = 0) -> slice:
    """Return the contiguous gen-image token span for denoise scatter."""
    if bundle.gen_image_slices is not None:
        row_slices = bundle.gen_image_slices[row]
        if isinstance(row_slices, slice):
            return row_slices
        return row_slices[0]
    if bundle.rope_image_info is None:
        raise ValueError("bundle has no gen_image_slices or rope_image_info")
    return bundle.rope_image_info[row][-1][0]


def get_gen_image_grid(bundle: GenImageHostInputs, row: int = 0) -> tuple[int, int]:
    """Return ``(token_h, token_w)`` for the gen-image span."""
    gen_slice = get_gen_image_slice(bundle, row)
    if bundle.rope_image_info is not None:
        for sl, hw in bundle.rope_image_info[row]:
            if sl == gen_slice:
                return hw
    raise ValueError(f"gen-image grid not found for slice {gen_slice!r}")


def bundle_to_denoise_cond(
    bundle: GenImageHostInputs,
    wte_weight: torch.Tensor,
    processor: HunyuanImage3ImageProcessor,
    *,
    row: int = 0,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Build static conditioning for ``HunyuanTtDenoiseStep`` / ``denoise_loop``.

    T2I: returns ``text_pre`` / ``text_post`` wte slices around the gen span.
    I2I: when ``bundle.inputs_embeds`` is set, returns host ``base_embeds`` with
    cond tokens already scattered (gen span still wte placeholders).
    Always includes multi-span ``attention_mask`` (additive float) and full
    ``image_infos`` for multi-span 2D RoPE.
    """
    from models.experimental.hunyuan_image_3_0.ref.attention.mask import to_additive

    if bundle.full_attn_slices is None or bundle.attention_mask is None:
        enrich_bundle_attention(bundle, processor)

    mask_add = to_additive(bundle.attention_mask[row : row + 1], dtype=torch.float32)
    image_infos = [bundle.rope_image_info[row]]

    if bundle.inputs_embeds is not None:
        return dict(
            base_embeds_host=bundle.inputs_embeds[row : row + 1].to(dtype=dtype),
            base_embeds=bundle.inputs_embeds[row : row + 1].to(dtype=dtype),
            gen_timestep_scatter_index=(
                bundle.gen_timestep_scatter_index[row : row + 1]
                if bundle.gen_timestep_scatter_index is not None
                else None
            ),
            image_infos=image_infos,
            attention_mask=mask_add,
            batch=1,
            gen_slice=get_gen_image_slice(bundle, row),
            gen_hw=get_gen_image_grid(bundle, row),
        )

    emb = F.embedding(bundle.input_ids[row : row + 1], wte_weight.to(dtype=dtype))
    gen_slice = get_gen_image_slice(bundle, row)
    text_post = emb[:, gen_slice.stop :, :] if gen_slice.stop < bundle.seq_len else None
    return dict(
        text_pre=emb[:, : gen_slice.start, :],
        text_post=text_post,
        gen_timestep_scatter_index=(
            bundle.gen_timestep_scatter_index[row : row + 1] if bundle.gen_timestep_scatter_index is not None else None
        ),
        image_infos=image_infos,
        attention_mask=mask_add,
        batch=1,
        gen_slice=gen_slice,
        gen_hw=get_gen_image_grid(bundle, row),
    )


def scatter_gen_timestep_embeds(
    embeds: torch.Tensor,
    timesteps: torch.Tensor,
    scatter_index: torch.Tensor | None,
    timestep_emb,
) -> torch.Tensor:
    """Scatter diffusion timestep token(s) at ``gen_timestep_scatter_index`` (per denoise step)."""
    from models.experimental.hunyuan_image_3_0.ref.image_gen.input_instantiate import instantiate_continuous_tokens

    if scatter_index is None:
        return embeds
    return instantiate_continuous_tokens(embeds.clone(), timesteps, scatter_index, timestep_emb)


def build_i2i_cfg_conds(
    bundle: GenImageHostInputs,
    wte_weight: torch.Tensor,
    processor: HunyuanImage3ImageProcessor,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return ``(cond, uncond)`` denoise dicts for CFG (rows 0/1)."""
    cond = bundle_to_denoise_cond(bundle, wte_weight, processor, row=0, dtype=dtype)
    if bundle.input_ids.shape[0] < 2:
        return cond, None
    uncond = bundle_to_denoise_cond(bundle, wte_weight, processor, row=1, dtype=dtype)
    return cond, uncond


def prepare_i2i_denoise_bundle(
    tok: HunyuanTokenizer,
    prompt: str,
    cond_image: CondImage,
    processor: HunyuanImage3ImageProcessor,
    wte_weight: torch.Tensor,
    *,
    patch_embed,
    time_embed,
    timestep_emb,
    vision_model=None,
    aligner=None,
    image_size: str | int | tuple[int, int] | list[int] = 1024,
    cfg_factor: int | None = None,
    sequence_template: str | None = None,
    system_prompt: str | None = None,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> GenImageHostInputs:
    """Full host I2I path: template → cond encode → attention mask."""
    bundle = prepare_i2i_inputs(
        tok,
        prompt,
        cond_image,
        image_size=image_size,
        cfg_factor=cfg_factor,
        sequence_template=sequence_template,
        system_prompt=system_prompt,
    )
    bundle = build_i2i_inputs_embeds(
        bundle,
        wte_weight,
        patch_embed=patch_embed,
        time_embed=time_embed,
        timestep_emb=timestep_emb,
        vision_model=vision_model,
        aligner=aligner,
        generator=generator,
        dtype=dtype,
    )
    return enrich_bundle_attention(bundle, processor)


def prepare_gen_image_inputs(
    tok: HunyuanTokenizer,
    prompt: str | list[str],
    *,
    image_size: str | int | tuple[int, int] | list[int] = 1024,
    cfg_factor: int | None = None,
    max_length: int | None = None,
) -> GenImageHostInputs:
    """Run chat template and build position IDs + 2D RoPE metadata for the transformer."""
    if cfg_factor is None:
        cfg_factor = 1 if tok.config.cfg_distilled else 2

    result = tok.apply_chat_template(
        prompt,
        image_size=image_size,
        cfg_factor=cfg_factor,
        max_length=max_length,
    )
    output = result["output"]
    sections = result["sections"]
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    return _build_host_inputs_from_template(
        tok,
        output=output,
        sections=sections,
        batch_size=batch_size,
        cfg_factor=cfg_factor,
    )


def prepare_i2i_inputs(
    tok: HunyuanTokenizer,
    prompt: str | list[str],
    cond_images: CondImage | list[CondImage] | list[list[CondImage]],
    *,
    image_size: str | int | tuple[int, int] | list[int] = 1024,
    cfg_factor: int | None = None,
    max_length: int | None = None,
    sequence_template: str | None = None,
    system_prompt: str | None = None,
) -> GenImageHostInputs:
    """Run I2I chat template and build host bundle (cond images + gen_image section)."""
    if cfg_factor is None:
        cfg_factor = 1 if tok.config.cfg_distilled else 2
    if sequence_template is None:
        sequence_template = tok.sequence_template

    batch_prompt = [prompt] if isinstance(prompt, str) else list(prompt)
    batch_size = len(batch_prompt)

    if isinstance(cond_images, CondImage):
        batch_cond_images: list[list[CondImage]] = [[cond_images] for _ in range(batch_size)]
    elif cond_images and isinstance(cond_images[0], CondImage):
        if len(cond_images) != batch_size:
            batch_cond_images = [list(cond_images) for _ in range(batch_size)]
        else:
            batch_cond_images = [[c] for c in cond_images]
    else:
        batch_cond_images = list(cond_images)
        if len(batch_cond_images) != batch_size:
            raise ValueError(
                f"cond_images batch length ({len(batch_cond_images)}) must match prompt batch ({batch_size})"
            )

    result = tok.apply_chat_template(
        batch_prompt,
        image_size=image_size,
        cond_images=batch_cond_images,
        cfg_factor=cfg_factor,
        max_length=max_length,
        sequence_template=sequence_template,
        system_prompt=system_prompt,
    )
    output = result["output"]
    sections = result["sections"]
    return _build_host_inputs_from_template(
        tok,
        output=output,
        sections=sections,
        batch_size=batch_size,
        cfg_factor=cfg_factor,
        batch_cond_images=batch_cond_images,
    )


def build_i2i_inputs_embeds(
    bundle: GenImageHostInputs,
    wte_weight: torch.Tensor,
    *,
    patch_embed,
    time_embed,
    timestep_emb,
    vae_encoder: Any | None = None,
    vision_model: Any | None = None,
    aligner: Any | None = None,
    model_dir: Path | None = None,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
    vae_scaling_factor: float | None = None,
    vae_shift_factor: float | None = None,
    vit_num_layers: int | None = None,
) -> GenImageHostInputs:
    """Encode cond VAE/ViT and scatter into ``inputs_embeds`` (upstream forward steps 4+8)."""
    from models.experimental.hunyuan_image_3_0.ref.cond_vae_encode import encode_cond_images
    from models.experimental.hunyuan_image_3_0.ref.cond_vit_encode import encode_cond_vit_images
    from models.experimental.hunyuan_image_3_0.ref.image_gen.input_instantiate import (
        instantiate_continuous_tokens,
        instantiate_vae_image_tokens,
        instantiate_vit_image_tokens,
    )
    from models.experimental.hunyuan_image_3_0.ref.vae.encoder import load_encoder
    from models.experimental.hunyuan_image_3_0.ref.weights import MODEL_DIR

    if bundle.batch_cond_images is None or len(bundle.batch_cond_images[0]) == 0:
        raise ValueError("build_i2i_inputs_embeds requires batch_cond_images on the bundle")
    if bundle.vae_image_mask is None:
        raise ValueError("build_i2i_inputs_embeds requires vae_image_mask on the bundle")

    if vae_encoder is None:
        vae_encoder = load_encoder(model_dir or MODEL_DIR, dtype=dtype)

    if vae_scaling_factor is None:
        vae_scaling_factor = 0.562679178327931

    hidden = F.embedding(bundle.input_ids, wte_weight.to(dtype=dtype))
    enc = encode_cond_images(
        bundle.batch_cond_images,
        vae_encoder,
        cfg_factor=bundle.cfg_factor,
        scaling_factor=vae_scaling_factor,
        shift_factor=vae_shift_factor,
        generator=generator,
        dtype=dtype,
    )

    hidden = instantiate_vae_image_tokens(
        hidden,
        enc.cond_timesteps,
        enc.cond_vae_images,
        bundle.vae_image_mask,
        patch_embed,
        time_embed,
    )
    if bundle.cond_timestep_scatter_index is not None:
        hidden = instantiate_continuous_tokens(
            hidden,
            enc.cond_timesteps,
            bundle.cond_timestep_scatter_index,
            timestep_emb,
        )

    needs_vit = bundle.vit_image_mask is not None and bool(bundle.vit_image_mask.any())
    if needs_vit:
        if vision_model is None or aligner is None:
            from models.experimental.hunyuan_image_3_0.ref.vision.siglip2 import load_aligner, load_siglip2_vision

            _dir = model_dir or MODEL_DIR
            if vision_model is None:
                vision_model = load_siglip2_vision(_dir, num_layers=vit_num_layers, dtype=dtype)
            if aligner is None:
                aligner = load_aligner(_dir, dtype=dtype)

        vit_enc = encode_cond_vit_images(
            bundle.batch_cond_images,
            cfg_factor=bundle.cfg_factor,
            dtype=dtype,
        )
        if vit_enc.cond_vit_images is None:
            raise ValueError("vit_image_mask is set but batch_cond_images has no ViT tensors")

        hidden = instantiate_vit_image_tokens(
            hidden,
            vit_enc.cond_vit_images,
            bundle.vit_image_mask,
            vit_enc.cond_vit_image_kwargs,
            vision_model,
            aligner,
        )
        bundle.cond_vit_images = vit_enc.cond_vit_images
        bundle.cond_vit_image_kwargs = vit_enc.cond_vit_image_kwargs

    bundle.cond_vae_images = enc.cond_vae_images
    bundle.cond_timesteps = enc.cond_timesteps
    bundle.inputs_embeds = hidden
    return bundle
