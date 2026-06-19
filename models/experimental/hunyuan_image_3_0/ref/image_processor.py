# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Phase A — Host input stack for I2I / Instruct (before device upload).
#
# PyTorch/host reference mirroring upstream:
#   HunyuanImage-3.0/hunyuan_image_3/image_processor.py
#
# Scope (I2I conditioning path, cond_image_type="vae_vit")
# ------------------------------------------------------------
#   build_cond_images()         — dual VAE + ViT preprocessing per input image
#   vae_process_image()         — resize/crop/normalize for VAE encoder
#   vit_process_image()         — Siglip2ImageProcessorFast patchify + masks
#   prepare_full_attn_slices()  — multi-span bidirectional attn (cond + gen)
#   postprocess_outputs()       — crop/align generated image to input aspect
#
# References
# ----------
#   README.md Phase 4 item 12
#   ref/tokenizer/gen_image_inputs.py — T2I host bundle (extend for I2I cond)

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Union

import torch
from PIL import Image
from torchvision import transforms
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import load_image
from transformers.models.siglip2.image_processing_siglip2_fast import Siglip2ImageProcessorFast

from models.experimental.hunyuan_image_3_0.ref.tokenizer.image_info import (
    CondImage,
    ImageInfo,
    ImageTensor,
)
from models.experimental.hunyuan_image_3_0.ref.tokenizer.resolution import Resolution, ResolutionGroup

InputImage = Union[Image.Image, str]


class SliceVocabLogitsProcessor(LogitsProcessor):
    """Restrict logits to a vocabulary slice (used for image-ratio token sampling)."""

    def __init__(self, vocab_start: int | None = None, vocab_end: int | None = None, **kwargs):
        if vocab_start is not None and vocab_end is not None:
            assert vocab_start < vocab_end, f"Ensure vocab_start {vocab_start} < vocab_end {vocab_end}"
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end
        self.other_slices = kwargs.get("other_slices", [])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = scores[:, self.vocab_start : self.vocab_end]
        for other_slice in self.other_slices:
            scores_processed = torch.cat(
                [scores_processed, scores[:, other_slice[0] : other_slice[1]]],
                dim=-1,
            )
        return scores_processed

    def __repr__(self) -> str:
        return (
            f"SliceVocabLogitsWarper(vocab_start={self.vocab_start}, "
            f"vocab_end={self.vocab_end}, other_slices={self.other_slices})"
        )


def resize_and_crop(
    image: Image.Image,
    target_size: tuple[int, int],
    resample=Image.Resampling.LANCZOS,
    crop_type: str = "center",
    crop_coords=None,
) -> Image.Image:
    """Resize and crop a PIL image to ``target_size`` (width, height)."""
    tw, th = target_size
    w, h = image.size

    tr = th / tw
    r = h / w

    if crop_type == "resize":
        image = image.resize((tw, th), resample=resample)
    else:
        if r < tr:
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))

        if crop_type == "center":
            crop_top = int(round((resize_height - th) / 2.0))
            crop_left = int(round((resize_width - tw) / 2.0))
        elif crop_type == "random":
            crop_top = random.randint(0, resize_height - th)
            crop_left = random.randint(0, resize_width - tw)
        elif crop_type == "fixed":
            assert crop_coords is not None, "crop_coords should be provided when crop_type is fixed."
            crop_left, crop_top = crop_coords
        else:
            raise ValueError(f"crop_type must be center, random or fixed, but got {crop_type}")

        image = image.resize((resize_width, resize_height), resample=resample)
        image = image.crop((crop_left, crop_top, crop_left + tw, crop_top + th))

    return image


@dataclass
class ResolutionGroupConfig:
    base_size: int | None = None
    step: int | None = None
    align: int = 16

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VAEInfo:
    encoder_type: str
    down_h_factor: int = -1
    down_w_factor: int = -1
    patch_size: int = 1
    h_factor: int = -1
    w_factor: int = -1
    image_type: str | None = None

    def __post_init__(self) -> None:
        self.h_factor = self.down_h_factor * self.patch_size
        self.w_factor = self.down_w_factor * self.patch_size
        if self.image_type is None:
            self.image_type = "vae"


@dataclass
class ViTInfo:
    encoder_type: str
    h_factor: int = -1
    w_factor: int = -1
    max_token_length: int = 0
    processor: Callable[..., Any] = field(default_factory=BaseImageProcessor)
    image_type: str | None = None

    def __post_init__(self) -> None:
        if self.image_type is None:
            self.image_type = self.encoder_type.split("-")[0]


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _tensor_scalar(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


class HunyuanImage3ImageProcessor:
    """Host-side image processor for HunyuanImage-3.0 I2I / Instruct inputs."""

    def __init__(self, config: dict[str, Any] | Any) -> None:
        self.config = config

        image_base_size = _config_get(config, "image_base_size", 1024)
        self.reso_group_config = ResolutionGroupConfig(base_size=image_base_size)
        self.vae_reso_group = ResolutionGroup(
            **self.reso_group_config.to_dict(),
            extra_resolutions=[
                Resolution("1024x768"),
                Resolution("1280x720"),
                Resolution("768x1024"),
                Resolution("720x1280"),
            ],
        )

        self.img_ratio_slice_logits_processor = None
        self.pil_image_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        vae_downsample = _config_get(config, "vae_downsample_factor", [16, 16])
        if isinstance(vae_downsample, int):
            vae_downsample = (vae_downsample, vae_downsample)
        self.vae_info = VAEInfo(
            encoder_type=_config_get(config, "vae_type", "hunyuan-image-vae-v1"),
            down_h_factor=int(vae_downsample[0]),
            down_w_factor=int(vae_downsample[0]),
            patch_size=int(_config_get(config, "patch_size", 1)),
        )

        vit_type = _config_get(config, "vit_type")
        if vit_type == "siglip2-so400m-patch16-naflex":
            vit_processor_cfg = _config_get(config, "vit_processor") or {}
            self.vit_processor = Siglip2ImageProcessorFast.from_dict(vit_processor_cfg)
        else:
            raise ValueError(f"Unsupported vit_type: {vit_type!r}")

        self.vit_info = ViTInfo(
            encoder_type=vit_type,
            h_factor=self.vit_processor.patch_size,
            w_factor=self.vit_processor.patch_size,
            max_token_length=self.vit_processor.max_num_patches,
            processor=self.vit_processor,
        )

        self.cond_token_attn_type = _config_get(config, "cond_token_attn_type", "joint_full")
        self.cond_image_type = _config_get(config, "cond_image_type", "vae_vit")

    def build_gen_image_info(
        self,
        image_size,
        add_guidance_token: bool = False,
        add_timestep_r_token: bool = False,
    ) -> ImageInfo:
        if isinstance(image_size, str):
            if image_size.startswith("<img_ratio_"):
                ratio_index = int(image_size.split("_")[-1].rstrip(">"))
                reso = self.vae_reso_group[ratio_index]
                image_size = reso.height, reso.width
            elif "x" in image_size:
                image_size = [int(s) for s in image_size.split("x")]
            elif ":" in image_size:
                image_size = [int(s) for s in image_size.split(":")]
                assert len(image_size) == 2, f"`image_size` should be in the format of 'W:H', got {image_size}."
                image_size = [image_size[1], image_size[0]]
            else:
                raise ValueError(
                    f"`image_size` should be in the format of 'HxW', 'W:H' or <img_ratio_i>, got {image_size}."
                )
            assert len(image_size) == 2, f"`image_size` should be in the format of 'HxW', got {image_size}."
        elif isinstance(image_size, (list, tuple)):
            assert len(image_size) == 2 and all(isinstance(s, int) for s in image_size), (
                f"`image_size` should be a tuple of two integers or a string in the format of 'HxW', "
                f"got {image_size}."
            )
        else:
            raise ValueError(
                f"`image_size` should be a tuple of two integers or a string in the format of 'WxH', "
                f"got {image_size}."
            )

        image_width, image_height = self.vae_reso_group.get_target_size(image_size[1], image_size[0])
        token_height = image_height // self.vae_info.h_factor
        token_width = image_width // self.vae_info.w_factor
        base_size, ratio_idx = self.vae_reso_group.get_base_size_and_ratio_index(image_size[1], image_size[0])
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

    def as_image_tensor(self, image, image_type: str, **kwargs) -> ImageTensor:
        if isinstance(image, Image.Image):
            tensor = self.pil_image_to_tensor(image)
        else:
            tensor = image

        origin_size = kwargs["origin_size"]
        ori_image_width = origin_size[0]
        ori_image_height = origin_size[1]

        if image_type == "vae":
            assert tensor.ndim == 3 or tensor.ndim == 4
            h, w = tensor.shape[-2], tensor.shape[-1]
            assert h % self.vae_info.h_factor == 0 and w % self.vae_info.w_factor == 0, (
                f"Image size should be divisible by ({self.vae_info.h_factor}, {self.vae_info.w_factor}), "
                f"but got ({h} x {w})."
            )
            tk_height = h // self.vae_info.h_factor
            tk_width = w // self.vae_info.w_factor
            base_size, ratio_idx = self.vae_reso_group.get_base_size_and_ratio_index(w, h)
            tensor.i = ImageInfo(
                image_type=image_type,
                image_width=w,
                image_height=h,
                token_width=tk_width,
                token_height=tk_height,
                base_size=base_size,
                ratio_index=ratio_idx,
                ori_image_width=ori_image_width,
                ori_image_height=ori_image_height,
            )
            tensor.section_type = "cond_vae_image"
        elif image_type == "siglip2":
            spatial_shapes = kwargs["spatial_shapes"]
            pixel_attention_mask = kwargs["pixel_attention_mask"]
            token_h = _tensor_scalar(spatial_shapes[0])
            token_w = _tensor_scalar(spatial_shapes[1])
            tensor.i = ImageInfo(
                image_type=image_type,
                image_width=token_w * self.vit_info.w_factor,
                image_height=token_h * self.vit_info.h_factor,
                token_width=token_w,
                token_height=token_h,
                image_token_length=self.vit_info.max_token_length,
                ori_image_width=ori_image_width,
                ori_image_height=ori_image_height,
            )
            tensor.section_type = "cond_vit_image"
            tensor.vision_encoder_kwargs = {
                "spatial_shapes": spatial_shapes,
                "pixel_attention_mask": pixel_attention_mask,
            }
        elif image_type == "anyres":
            token_width = kwargs["resized_image_width"] // self.vit_info.w_factor
            token_height = kwargs["resized_image_height"] // self.vit_info.h_factor
            tensor.i = ImageInfo(
                image_type=image_type,
                image_width=kwargs["resized_image_width"],
                image_height=kwargs["resized_image_height"],
                token_width=token_width,
                token_height=token_height,
                image_token_length=token_height * (token_width + 1) + 2,
            )
            tensor.section_type = "cond_vit_image"
        else:
            raise ValueError(f"Unknown image type: {image_type}")

        return tensor

    def vae_process_image(
        self,
        image: Image.Image,
        target_size: tuple[int, int],
        random_crop: bool | str = False,
    ) -> ImageTensor:
        origin_size = image.size
        crop_type = random_crop if isinstance(random_crop, str) else ("random" if random_crop else "center")
        resized_image = resize_and_crop(image, target_size, crop_type=crop_type)
        return self.as_image_tensor(
            resized_image,
            image_type=self.vae_info.image_type,
            origin_size=origin_size,
        )

    def vit_process_image(self, image: Image.Image) -> ImageTensor:
        origin_size = image.size
        inputs = self.vit_info.processor(image)
        pixel_values = inputs["pixel_values"].squeeze(0)

        remain_keys = set(inputs.keys()) - {"pixel_values"}
        remain_kwargs = {}
        for key in remain_keys:
            if isinstance(inputs[key], torch.Tensor):
                remain_kwargs[key] = inputs[key].squeeze(0)
            else:
                remain_kwargs[key] = inputs[key]

        return self.as_image_tensor(
            pixel_values,
            image_type=self.vit_info.image_type,
            origin_size=origin_size,
            **remain_kwargs,
        )

    def get_image_with_size(
        self,
        src: InputImage,
        random_crop: bool | str = False,
        return_type: str = "vae",
    ) -> tuple[ImageTensor | CondImage, bool]:
        image = load_image(src)
        img_success = True
        origin_size = image.size

        vae_image_tensor = None
        vit_image_tensor = None

        if "vae" in return_type:
            target_size = self.vae_reso_group.get_target_size(*origin_size)
            vae_image_tensor = self.vae_process_image(image, target_size, random_crop=random_crop)

        if "vit" in return_type:
            vit_image_tensor = self.vit_process_image(image)

        if return_type == "vae":
            image_tensor = vae_image_tensor
        elif return_type == "vit":
            image_tensor = vit_image_tensor
        elif return_type == "vae_vit":
            image_tensor = CondImage(
                image_type=return_type,
                vae_image=vae_image_tensor,
                vit_image=vit_image_tensor,
            )
        else:
            raise ValueError(f"Unknown return_type: {return_type}")

        return image_tensor, img_success

    def build_cond_images(
        self,
        *,
        image_list: list[InputImage] | None = None,
        message_list: list[dict[str, Any]] | None = None,
        infer_align_image_size: bool = False,
    ) -> list[CondImage] | None:
        if image_list is not None and message_list is not None:
            raise ValueError("`image_list` and `message_list` cannot be provided at the same time.")

        if message_list is not None:
            image_list = []
            for message in message_list:
                visuals = [
                    content
                    for content in message["content"]
                    if isinstance(content, dict) and content["type"] in ["image"]
                ]
                image_list.extend(
                    [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["image", "url", "path", "base64"]
                        if key in vision_info and vision_info["type"] == "image"
                    ]
                )

        if image_list is None:
            return None

        random_crop = "resize" if infer_align_image_size else "center"
        return [
            self.get_image_with_size(src, return_type=self.cond_image_type, random_crop=random_crop)[0]
            for src in image_list
        ]

    def prepare_full_attn_slices(self, output, *, batch_idx: int | None = None, with_gen: bool = True):
        if self.cond_image_type == "vae":
            cond_choices = {
                "causal": [],
                "full": output.vae_image_slices[batch_idx] if batch_idx is not None else output.vae_image_slices,
            }
        elif self.cond_image_type == "vit":
            cond_choices = {
                "causal": [],
                "full": output.vit_image_slices[batch_idx] if batch_idx is not None else output.vit_image_slices,
            }
        elif self.cond_image_type == "vae_vit":
            cond_choices = {
                "causal": [],
                "full": (
                    output.vae_image_slices[batch_idx] + output.vit_image_slices[batch_idx]
                    if batch_idx is not None
                    else output.vae_image_slices + output.vit_image_slices
                ),
                "joint_full": (
                    output.joint_image_slices[batch_idx] if batch_idx is not None else output.joint_image_slices
                ),
                "full_causal": (
                    output.vae_image_slices[batch_idx] if batch_idx is not None else output.vae_image_slices
                ),
            }
        else:
            raise ValueError(f"Unknown cond_image_type: {self.cond_image_type}")

        slices = cond_choices[self.cond_token_attn_type]

        if with_gen:
            gen_image_slices = output.gen_image_slices[batch_idx] if batch_idx is not None else output.gen_image_slices
            slices = slices + gen_image_slices
        return slices

    def build_img_ratio_slice_logits_proc(self, tokenizer) -> None:
        if self.img_ratio_slice_logits_processor is None:
            self.img_ratio_slice_logits_processor = LogitsProcessorList()
            self.img_ratio_slice_logits_processor.append(
                SliceVocabLogitsProcessor(
                    vocab_start=tokenizer.start_ratio_token_id,
                    vocab_end=tokenizer.end_ratio_token_id + 1,
                    other_slices=getattr(tokenizer, "ratio_token_other_slices", []),
                )
            )

    def postprocess_outputs(
        self,
        outputs: list[Image.Image],
        batch_cond_images,
        infer_align_image_size: bool = False,
    ) -> list[Image.Image]:
        if infer_align_image_size:
            target_area = self.vae_reso_group.base_size**2

            for batch_index, (output_image, cond_images) in enumerate(zip(outputs, batch_cond_images)):
                output_image_ratio_index = self.vae_reso_group.get_base_size_and_ratio_index(
                    width=output_image.width,
                    height=output_image.height,
                )[1]

                cond_images_ratio_index_list = []
                cond_images_ori_width_list = []
                cond_images_ori_height_list = []
                for cond_image in cond_images:
                    if isinstance(cond_image, torch.Tensor):
                        cond_images_ratio_index_list.append(cond_image.i.ratio_index)
                        cond_images_ori_width_list.append(cond_image.i.ori_image_width)
                        cond_images_ori_height_list.append(cond_image.i.ori_image_height)
                    else:
                        cond_images_ratio_index_list.append(cond_image.vae_image.i.ratio_index)
                        cond_images_ori_width_list.append(cond_image.vae_image.i.ori_image_width)
                        cond_images_ori_height_list.append(cond_image.vae_image.i.ori_image_height)

                if len(cond_images) == 0:
                    continue
                if len(cond_images) == 1:
                    if output_image_ratio_index == cond_images_ratio_index_list[0]:
                        ratio_diff = abs(
                            cond_images_ori_height_list[0] / cond_images_ori_width_list[0]
                            - self.vae_reso_group[output_image_ratio_index].ratio
                        )
                        if ratio_diff >= 0.01:
                            scale = math.sqrt(
                                target_area / (cond_images_ori_width_list[0] * cond_images_ori_height_list[0])
                            )
                            new_w = round(cond_images_ori_width_list[0] * scale)
                            new_h = round(cond_images_ori_height_list[0] * scale)
                            outputs[batch_index] = output_image.resize(
                                (new_w, new_h),
                                resample=Image.Resampling.LANCZOS,
                            )
                else:
                    for cond_image_ratio_index, cond_image_ori_width, cond_image_ori_height in zip(
                        cond_images_ratio_index_list,
                        cond_images_ori_width_list,
                        cond_images_ori_height_list,
                    ):
                        if output_image_ratio_index == cond_image_ratio_index:
                            ratio_diff = abs(
                                cond_image_ori_height / cond_image_ori_width
                                - self.vae_reso_group[output_image_ratio_index].ratio
                            )
                            if ratio_diff >= 0.01:
                                scale = math.sqrt(target_area / (cond_image_ori_width * cond_image_ori_height))
                                new_w = round(cond_image_ori_width * scale)
                                new_h = round(cond_image_ori_height * scale)
                                outputs[batch_index] = output_image.resize(
                                    (new_w, new_h),
                                    resample=Image.Resampling.LANCZOS,
                                )
                            break

        return outputs


__all__ = [
    "CondImage",
    "HunyuanImage3ImageProcessor",
    "ImageTensor",
    "ResolutionGroupConfig",
    "SliceVocabLogitsProcessor",
    "VAEInfo",
    "ViTInfo",
    "resize_and_crop",
]
