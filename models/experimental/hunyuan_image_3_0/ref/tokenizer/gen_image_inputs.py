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
from typing import TYPE_CHECKING, Any

import torch

from .image_info import CondImage

if TYPE_CHECKING:
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
    batch_cond_images: list[list[Any]] | None = None


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
        batch_cond_images=batch_cond_images,
    )


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
) -> GenImageHostInputs:
    """Run I2I chat template and build host bundle (cond images + gen_image section)."""
    if cfg_factor is None:
        cfg_factor = 1 if tok.config.cfg_distilled else 2

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
