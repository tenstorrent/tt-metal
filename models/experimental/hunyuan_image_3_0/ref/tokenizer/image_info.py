# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# ImageInfo dataclass + gen-image layout metadata for T2I token sequences.
#
# Mapping to HF upstream (hunyuan_image_3/image_processor.py)
# ------------------------------------------------------------
#  Step | HF                                           | This port (host)
#  -----+----------------------------------------------+----------------------------------
#   1   | parse image_size (int/str/ratio token)       | _parse_image_size
#   2   | ResolutionGroup target size lookup           | ResolutionGroup.get_target_size
#   3   | base_size + ratio_index                      | get_base_size_and_ratio_index
#   4   | token_h/w from VAE downsample + patch_size   | build_gen_image_info
#   5   | ImageProcessor.build_gen_image_info          | build_gen_image_info → ImageInfo
#
# ImageInfo.meta_info feeds the gen_image section encoder in chat_template.py.
#
# References
# ----------
#   ref/tokenizer/resolution.py          — Resolution / ResolutionGroup helpers
#   ref/tokenizer/chat_template.py       — _encode_gen_image_section
#   ref/tokenizer/hunyuan_tokenizer.py   — HunyuanTokenizer.build_gen_image_info

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .resolution import Resolution, ResolutionGroup


def default(value, default_value):
    return value if value is not None else default_value


@dataclass
class ImageInfo:
    """Latent patch grid metadata for ``gen_image`` sections."""

    image_type: str = "gen_image"
    image_width: int | None = None
    image_height: int | None = None
    token_width: int | None = None
    token_height: int | None = None
    image_token_length: int | None = None
    base_size: int | None = None
    ratio_index: int | None = None
    add_timestep_token: bool = True
    add_guidance_token: bool = False
    add_timestep_r_token: bool = False
    use_front_boi_token: bool = True
    add_image_shape_token: bool = True

    def __post_init__(self) -> None:
        if self.image_token_length is None and self.token_width is not None and self.token_height is not None:
            self.image_token_length = self.token_width * self.token_height

    @property
    def meta_info(self) -> dict[str, Any]:
        if self.image_type not in ("vae", "gen_image"):
            raise ValueError(f"Unsupported image_type for meta_info: {self.image_type!r}")
        return dict(
            token_length=self.image_token_length,
            add_timestep_token=self.add_timestep_token,
            add_guidance_token=self.add_guidance_token,
            add_timestep_r_token=self.add_timestep_r_token,
            use_front_boi_token=self.use_front_boi_token,
            add_image_shape_token=self.add_image_shape_token,
            base_size=self.base_size,
            ratio_idx=self.ratio_index,
            token_height=self.token_height,
            token_width=self.token_width,
            image_height=self.image_height,
            image_width=self.image_width,
        )


def _parse_image_size(image_size: str | int | tuple[int, int] | list[int], vae_reso_group: ResolutionGroup):
    if isinstance(image_size, int):
        return image_size, image_size
    if isinstance(image_size, str):
        if image_size.startswith("<img_ratio_"):
            ratio_index = int(image_size.split("_")[-1].rstrip(">"))
            reso = vae_reso_group[ratio_index]
            return reso.height, reso.width
        if "x" in image_size:
            h, w = (int(s) for s in image_size.split("x"))
            return h, w
        if ":" in image_size:
            w, h = (int(s) for s in image_size.split(":"))
            return h, w
        raise ValueError(f"Unsupported image_size string: {image_size!r}")
    if isinstance(image_size, (list, tuple)):
        if len(image_size) != 2:
            raise ValueError(f"image_size must be (H, W), got {image_size}")
        return int(image_size[0]), int(image_size[1])
    raise ValueError(f"Unsupported image_size type: {type(image_size)}")


def build_gen_image_info(
    *,
    image_size: str | int | tuple[int, int] | list[int],
    image_base_size: int,
    vae_downsample_factor: tuple[int, int],
    vae_patch_size: int = 1,
    add_guidance_token: bool = False,
    add_timestep_r_token: bool = False,
) -> ImageInfo:
    """Build ``ImageInfo`` for text-to-image (mirrors HF ``ImageProcessor.build_gen_image_info``)."""
    vae_reso_group = ResolutionGroup(
        base_size=image_base_size,
        extra_resolutions=[
            Resolution("1024x768"),
            Resolution("1280x720"),
            Resolution("768x1024"),
            Resolution("720x1280"),
        ],
    )
    height, width = _parse_image_size(image_size, vae_reso_group)
    image_width, image_height = vae_reso_group.get_target_size(width, height)
    h_factor = int(vae_downsample_factor[0]) * int(vae_patch_size)
    w_factor = int(vae_downsample_factor[1]) * int(vae_patch_size)
    token_height = image_height // h_factor
    token_width = image_width // w_factor
    base_size, ratio_idx = vae_reso_group.get_base_size_and_ratio_index(width, height)
    return ImageInfo(
        image_type="gen_image",
        image_width=image_width,
        image_height=image_height,
        token_width=token_width,
        token_height=token_height,
        base_size=base_size,
        ratio_index=ratio_idx,
        add_guidance_token=add_guidance_token,
        add_timestep_r_token=add_timestep_r_token,
    )
