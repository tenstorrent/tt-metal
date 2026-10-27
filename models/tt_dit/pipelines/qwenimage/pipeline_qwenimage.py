# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import diffusers
import numpy as np
import torch
import tqdm
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.models.transformers.transformer_qwenimage import QwenImageCheckpoint
from models.tt_dit.models.vae.vae_wan_2d import WanVaeDecoder2DAdapter, WanVaeEncoder2DAdapter
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.cfg import CFGCombiner, create_submeshes, submesh_shape
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin
from models.tt_dit.pipelines.qwenimage.text_encoder import (
    EDIT_FORMAT,
    EDIT_PLUS_FORMAT,
    TEXT_TO_IMAGE_FORMAT,
    PromptFormat,
    TextEncoder,
)
from models.tt_dit.solvers import EulerSolver, calculate_shift
from models.tt_dit.utils import tensor
from models.tt_dit.utils.padding import torch_pad
from models.tt_dit.utils.tracing import Tracer

_VAE_SCALE_FACTOR = 8
_LATENT_CHANNELS = 16
_DEFAULT_CHECKPOINT = "Qwen/Qwen-Image"

# The prompt lengths the text encoder and the transformer run at, counted after the template prefix
# is dropped.
_SEQUENCE_LENGTH_BUCKETS = (128, 256, 512)

# The supported image sizes, (width, height): the sizes recommended for Qwen-Image and
# Qwen-Image-2512, moved to multiples of 32 px so that the VAE can split them evenly, plus 1024x1024.
_RESOLUTIONS = (
    (1024, 1024),  # 1:1
    (1344, 1344),  # 1:1, reference: 1328x1328
    (1664, 928),  # 16:9, reference: 1664x928
    (928, 1664),  # 9:16, reference: 928x1664
    (1472, 1120),  # 4:3, reference: 1472x1104
    (1120, 1472),  # 3:4, reference: 1104x1472
    (1600, 1056),  # 3:2, reference: 1584x1056
    (1056, 1600),  # 2:3, reference: 1056x1584
)

# The spatial lengths the transformer runs at. Each image is padded to the smallest that fits it.
_SPATIAL_SEQUENCE_LENGTH_BUCKETS = (4096, 6144, 6528, 6656, 7168)

# The edit models resize each input image, and by default size the output, to about 1024² pixels at
# the image's aspect ratio. Edit-2509 and 2511 show the input images to the text encoder at about
# 384² instead.
_EDIT_AREA = 1024 * 1024
_EDIT_PLUS_VISION_AREA = 384 * 384


def edit_size(area: int, aspect_ratio: float) -> tuple[int, int]:
    """The (width, height) diffusers' edit pipelines give an image of ``aspect_ratio``.

    That is about ``area`` pixels, in multiples of 32 px.
    """
    width = math.sqrt(area * aspect_ratio)
    height = width / aspect_ratio
    return round(width / 32) * 32, round(height / 32) * 32


# The edit resolutions: diffusers' sizes for the aspect ratios of _RESOLUTIONS. An input image is
# resized to the one closest to it in aspect ratio, which is diffusers' size if its aspect ratio is
# one of these.
_EDIT_RESOLUTIONS = tuple(edit_size(_EDIT_AREA, r) for r in (1, 16 / 9, 9 / 16, 4 / 3, 3 / 4, 3 / 2, 2 / 3))

# Prompt lengths with the input images, counted like _SEQUENCE_LENGTH_BUCKETS. A 1024² image takes
# 1369 tokens in Qwen-Image-Edit, a 384² one 196 in Edit-2509 and 2511.
_EDIT_SEQUENCE_LENGTH_BUCKETS = (1408, 1536, 1920)
_EDIT_PLUS_SEQUENCE_LENGTH_BUCKETS = (256, 512, 1024)


class _Mode(NamedTuple):
    prompt_format: PromptFormat
    # The most input images; 0 for text-to-image.
    max_images: int
    # The area the text encoder sees each input image at; None for the size of its latents.
    vision_area: int | None
    sequence_length_buckets: tuple[int, ...]
    resolutions: tuple[tuple[int, int], ...]


# By the diffusers pipeline class of the checkpoint.
_MODES = {
    "QwenImagePipeline": _Mode(TEXT_TO_IMAGE_FORMAT, 0, None, _SEQUENCE_LENGTH_BUCKETS, _RESOLUTIONS),
    "QwenImageEditPipeline": _Mode(EDIT_FORMAT, 1, None, _EDIT_SEQUENCE_LENGTH_BUCKETS, _EDIT_RESOLUTIONS),
    "QwenImageEditPlusPipeline": _Mode(
        EDIT_PLUS_FORMAT, 3, _EDIT_PLUS_VISION_AREA, _EDIT_PLUS_SEQUENCE_LENGTH_BUCKETS, _EDIT_RESOLUTIONS
    ),
}


def _mode(checkpoint_name: str) -> _Mode:
    class_name = diffusers.DiffusionPipeline.load_config(checkpoint_name)["_class_name"]
    if class_name not in _MODES:
        msg = f"unsupported checkpoint {checkpoint_name} of pipeline class {class_name}"
        raise ValueError(msg)
    return _MODES[class_name]


def _edit_spatial_buckets(resolutions: Sequence[tuple[int, int]], max_images: int) -> tuple[int, ...]:
    """One spatial bucket per number of input images, in multiples of 128.

    Each fits the output and that many input images at any of ``resolutions``.
    """
    tokens = max((w // (_VAE_SCALE_FACTOR * 2)) * (h // (_VAE_SCALE_FACTOR * 2)) for w, h in resolutions)
    return tuple(-(-tokens * (k + 1) // 128) * 128 for k in range(1, max_images + 1))


_PRESETS_WH: dict[tuple[int, ...], dict] = {
    (2, 4): {
        "cfg": (1, 0),
        "sp": (2, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "encoder_fsdp": (2, 0),
        "num_links": 1,
        "resolutions": ((1024, 1024),),
        "offload_vision_tower": True,
    },
    (4, 8): {
        "cfg": (2, 1),
        "sp": (4, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "encoder_fsdp": (4, 0),
        "num_links": 4,
    },
}

_PRESETS_BH: dict[tuple[int, ...], dict] = {
    (2, 2): {
        "cfg": (2, 0),
        "sp": (1, 0),
        "tp": (2, 1),
        "encoder_tp": (2, 1),
        "encoder_fsdp": None,
        "num_links": 2,
    },
    (2, 4): {
        "cfg": (2, 0),
        "sp": (1, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "encoder_fsdp": None,
        "num_links": 2,
    },
    (4, 8): {
        "cfg": (2, 1),
        "sp": (4, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "encoder_fsdp": None,
        "num_links": 2,
    },
}


@dataclass(frozen=True, kw_only=True)
class QwenImagePipelineConfig:
    topology: ttnn.Topology
    num_links: int

    dit_parallel_config: DiTParallelConfig
    encoder_parallel_config: EncoderParallelConfig
    vae_parallel_config: VaeHWParallelConfig

    use_torch_text_encoder: bool
    use_torch_vae_decoder: bool
    use_torch_vae_encoder: bool
    # Load the text encoder's device weights only while encoding.
    offload_text_encoder: bool
    # Load the vision tower's device weights only while encoding the input images.
    offload_vision_tower: bool

    resolutions: tuple[tuple[int, int], ...]
    spatial_sequence_length_buckets: tuple[int, ...]
    cfg_enabled: bool
    sequence_length_buckets: tuple[int, ...]
    # The most input images of an edit checkpoint; 0 for text-to-image.
    max_images: int

    checkpoint_name: str

    @classmethod
    def default(
        cls,
        *,
        mesh_shape: ttnn.MeshShape,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        num_links: int | None = None,
        dit_parallel_config: DiTParallelConfig | None = None,
        encoder_parallel_config: EncoderParallelConfig | None = None,
        vae_parallel_config: VaeHWParallelConfig | None = None,
        use_torch_text_encoder: bool = False,
        use_torch_vae_decoder: bool = False,
        use_torch_vae_encoder: bool = False,
        offload_text_encoder: bool | None = None,
        offload_vision_tower: bool | None = None,
        resolutions: Sequence[tuple[int, int]] | None = None,
        spatial_sequence_length_buckets: Sequence[int] | None = None,
        cfg_enabled: bool = True,
        sequence_length_buckets: Sequence[int] | None = None,
        max_images: int | None = None,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> QwenImagePipelineConfig:
        """The preset for ``mesh_shape``.

        Resolutions, buckets and ``max_images`` default to the ones of the checkpoint's kind:
        text-to-image, Qwen-Image-Edit, or Edit-2509 and 2511.
        """
        preset_dict = _PRESETS_BH if ttnn.device.is_blackhole() else _PRESETS_WH
        preset = preset_dict.get(tuple(mesh_shape), {})
        mode = _mode(checkpoint_name)

        if dit_parallel_config is None:
            dit_parallel_config = DiTParallelConfig.from_tuples(cfg=preset["cfg"], sp=preset["sp"], tp=preset["tp"])

        if encoder_parallel_config is None:
            encoder_parallel_config = EncoderParallelConfig.from_tuples(
                tp=preset["encoder_tp"], sp=None, fsdp=preset["encoder_fsdp"]
            )

        if vae_parallel_config is None:
            vae_parallel_config = VaeHWParallelConfig.from_axes(submesh_shape(dit_parallel_config), h_axis=1, w_axis=0)

        if resolutions is None:
            resolutions = preset.get("resolutions", mode.resolutions)

        if max_images is None:
            max_images = mode.max_images
        if max_images > mode.max_images:
            msg = f"{checkpoint_name} takes at most {mode.max_images} input images"
            raise ValueError(msg)
        if (max_images > 0) != (mode.max_images > 0):
            msg = f"max_images must be {'positive' if mode.max_images else '0'} for {checkpoint_name}"
            raise ValueError(msg)

        if spatial_sequence_length_buckets is None:
            spatial_sequence_length_buckets = (
                _edit_spatial_buckets(resolutions, max_images) if max_images else _SPATIAL_SEQUENCE_LENGTH_BUCKETS
            )
        if sequence_length_buckets is None:
            sequence_length_buckets = mode.sequence_length_buckets
        if offload_text_encoder is None:
            offload_text_encoder = max_images > 0 and preset.get("edit_offload_text_encoder", False)
        if offload_vision_tower is None:
            offload_vision_tower = preset.get("offload_vision_tower", False)

        return cls(
            topology=topology,
            num_links=num_links if num_links is not None else preset["num_links"],
            dit_parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel_config=vae_parallel_config,
            use_torch_text_encoder=use_torch_text_encoder,
            use_torch_vae_decoder=use_torch_vae_decoder,
            use_torch_vae_encoder=use_torch_vae_encoder,
            offload_text_encoder=offload_text_encoder,
            offload_vision_tower=offload_vision_tower,
            resolutions=tuple(resolutions),
            spatial_sequence_length_buckets=tuple(spatial_sequence_length_buckets),
            cfg_enabled=cfg_enabled,
            sequence_length_buckets=tuple(sequence_length_buckets),
            max_images=max_images,
            checkpoint_name=checkpoint_name,
        )


class _BlendInputs(NamedTuple):
    """What the latents are blended with after each step, in the layout of the spatial sequence.

    Where ``mask`` is 0, the latents are replaced by ``noise`` and ``image_latents`` mixed to the
    noise level of the next step, plus ``condition``: the inpainting image outside the mask, and
    the input images of an edit in their rows.
    """

    image_latents: ttnn.Tensor
    noise: ttnn.Tensor
    mask: ttnn.Tensor
    condition: ttnn.Tensor


class _ImageSize(NamedTuple):
    width: int
    height: int
    latents_width: int
    latents_height: int
    sequence_length: int


class _Layout(NamedTuple):
    """The spatial sequence: the output's latents, then each input image's, then padding."""

    output: _ImageSize
    conditions: tuple[_ImageSize, ...]
    sequence_length: int
    padded_sequence_length: int


class QwenImagePipeline(PipelineAPIMixin):
    """QwenImagePipeline is a pipeline for generating images from text prompts.

    It uses a transformer to encode the text prompts and a VAE to decode the latent space.
    Dynamic loading is controlled by the initialization state. During inference, modules
    will be loaded/offloaded as needed.
    """

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        cfg_enabled: bool = True,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> QwenImagePipeline:
        config = QwenImagePipelineConfig.default(
            mesh_shape=mesh_device.shape,
            cfg_enabled=cfg_enabled,
            checkpoint_name=checkpoint_name,
        )
        return cls(device=mesh_device, config=config)

    def __init__(self, *, device: ttnn.MeshDevice, config: QwenImagePipelineConfig) -> None:
        self._parallel_config = config.dit_parallel_config
        self._sp_axis = config.dit_parallel_config.sequence_parallel.mesh_axis
        self._cfg_parallel = config.dit_parallel_config.cfg_parallel.factor != 1
        self._cfg_enabled = config.cfg_enabled
        self._sequence_length_buckets = config.sequence_length_buckets
        self._vae_parallel_config = config.vae_parallel_config
        self._spatial_sequence_length_buckets = sorted(set(config.spatial_sequence_length_buckets))

        sp_factor = config.dit_parallel_config.sequence_parallel.factor
        for bucket in self._spatial_sequence_length_buckets:
            # Ring attention needs whole tiles on each device.
            if bucket % (ttnn.TILE_SIZE * sp_factor) != 0:
                msg = f"spatial bucket {bucket} must be a multiple of {ttnn.TILE_SIZE * sp_factor}"
                raise ValueError(msg)

        self._mode = _mode(config.checkpoint_name)
        self._max_images = config.max_images
        self._checkpoint = QwenImageCheckpoint(config.checkpoint_name)
        self._image_sizes = {(w, h): self._image_size(w, h) for w, h in config.resolutions}

        # Only the spatial buckets that some resolution, or with edits some number of input images
        # at one resolution, is padded to are traced and warmed up.
        self._spatial_sequence_length_buckets = sorted(
            {
                self._layout(s, [s] * k).padded_sequence_length
                for s in self._image_sizes.values()
                for k in (range(1, self._max_images + 1) if self._max_images else [0])
            }
        )

        logger.info(f"Parallel config: {config.dit_parallel_config}")
        logger.info(f"Original mesh shape: {device.shape}")
        self._devices = create_submeshes(device, config.dit_parallel_config)
        logger.info(f"Created submeshes with shape {self._devices[0].shape}")

        self._ccl_managers = [
            CCLManager(d, num_links=config.num_links, topology=config.topology) for d in self._devices
        ]
        self._combiner = CFGCombiner(self._devices, rescale=True)

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.checkpoint_name, subfolder="scheduler")
        self._solvers = [EulerSolver(scheduler=scheduler) for _ in self._devices]
        # One tracer per submesh, prompt bucket and spatial bucket.
        self._tracers = {
            (idx, bucket, spatial_bucket): Tracer(self._traced_step, device=d, prep_run=False)
            for idx, d in enumerate(self._devices)
            for bucket in config.sequence_length_buckets
            for spatial_bucket in self._spatial_sequence_length_buckets
        }
        self._image_processor = VaeImageProcessor(vae_scale_factor=_VAE_SCALE_FACTOR * 2)
        self._mask_processor = VaeImageProcessor(
            vae_scale_factor=_VAE_SCALE_FACTOR * 2,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

        # The encoder is loaded before the transformers, for memory efficiency.
        logger.info("creating text encoder...")
        self._text_encoder = TextEncoder(
            checkpoint_name=config.checkpoint_name,
            device=self._devices[0],
            ccl_manager=self._ccl_managers[0],
            parallel_config=config.encoder_parallel_config,
            sequence_length_buckets=config.sequence_length_buckets,
            use_torch=config.use_torch_text_encoder,
            prompt_format=self._mode.prompt_format,
            offload=config.offload_text_encoder,
            offload_vision=config.offload_vision_tower,
            vision_image_sizes=[self._vision_size(s.width / s.height, s) for s in self._image_sizes.values()],
            max_images=self._max_images,
        )

        logger.info("creating transformers...")
        self._transformers = [
            self._checkpoint.build(ccl_manager=m, parallel_config=config.dit_parallel_config, is_fsdp=False)
            for m in self._ccl_managers
        ]

        logger.info("creating VAE decoder...")
        self._vae = WanVaeDecoder2DAdapter(
            checkpoint_name=config.checkpoint_name,
            parallel_config=config.vae_parallel_config,
            ccl_manager=self._ccl_managers[-1],
            use_torch=config.use_torch_vae_decoder,
        )

        logger.info("creating VAE encoder...")
        self._vae_encoder = WanVaeEncoder2DAdapter(
            checkpoint_name=config.checkpoint_name,
            parallel_config=config.vae_parallel_config,
            ccl_manager=self._ccl_managers[-1],
            use_torch=config.use_torch_vae_encoder,
        )

        # Allocated before the traces: the blend inputs of each spatial bucket and submesh,
        # patchified and padded like the latents, for zero_cond_t checkpoints the rows of the input
        # images, and the spatial length without padding of each submesh.
        p = self._checkpoint.patch_size
        mesh_axes = [None, self._sp_axis, None]
        self._blend_inputs: dict[int, list[_BlendInputs]] = {}
        self._condition_masks: dict[int, list[ttnn.Tensor]] = {}
        for bucket in self._spatial_sequence_length_buckets:
            shape = [1, bucket, _LATENT_CHANNELS * p * p]
            self._blend_inputs[bucket] = [
                _BlendInputs(
                    image_latents=tensor.zeros(shape, device=d, mesh_axes=mesh_axes),
                    noise=tensor.zeros(shape, device=d, mesh_axes=mesh_axes),
                    mask=tensor.zeros(shape, device=d, mesh_axes=mesh_axes),
                    condition=tensor.zeros(shape, device=d, mesh_axes=mesh_axes),
                )
                for d in self._devices
            ]
            if self._checkpoint.zero_cond_t:
                self._condition_masks[bucket] = [
                    tensor.zeros([1, bucket, 1], device=d, mesh_axes=mesh_axes) for d in self._devices
                ]
        self._sequence_lengths = [self._sequence_length_tensor(0, device=d) for d in self._devices]

        logger.info("pipeline allocation run...")
        self._warm_up(traced=False)

        logger.info("pipeline capture run...")
        self._warm_up(traced=True)

    def _warm_up(self, *, traced: bool) -> None:
        """Runs every prompt bucket with every spatial bucket, and the VAE at every resolution."""
        if self._max_images:
            self._warm_up_edit(traced=traced)
        else:
            # The transformer needs each spatial bucket, so the whole pipeline runs at one
            # resolution per bucket.
            sizes_per_bucket = {self._layout(size).padded_sequence_length: size for size in self._image_sizes.values()}
            for size in sizes_per_bucket.values():
                image = Image.new("RGB", (size.width, size.height))
                mask_image = Image.new("L", (size.width, size.height), 255)

                for bucket in sorted(self._sequence_length_buckets, reverse=True):
                    # Each "a " is one token, and the trailing space and the template suffix add
                    # six more, so this comes to two tokens short of the bucket, which the 32-token
                    # spacing keeps above the next-smaller one.
                    prompt = "a " * (bucket - 8)
                    self(
                        prompts=[prompt],
                        negative_prompts=[prompt],
                        num_inference_steps=2,
                        cfg_scale=2 if self._cfg_enabled else 1,
                        width=size.width,
                        height=size.height,
                        traced=traced,
                        encoder_traced=traced,
                        vae_traced=traced,
                        image=image,
                        mask_image=mask_image,
                        strength=1.0,
                    )

        # The VAE needs each resolution.
        for size in self._image_sizes.values():
            self._vae.decode(torch.zeros(1, size.latents_height, size.latents_width, _LATENT_CHANNELS), traced=traced)
            self._vae_encoder.encode(torch.zeros(1, 3, size.height, size.width), traced=traced)

    def _warm_up_edit(self, *, traced: bool) -> None:
        """Runs every prompt bucket with each number of input images.

        The vision tower pads its input to one length per image count, so any resolution warms it
        up. How many prompt tokens the images take depends on their resolution, though, so each
        prompt bucket runs at the largest resolution that reaches it.
        """
        sizes = sorted(self._image_sizes.values(), key=lambda s: s.sequence_length, reverse=True)

        for image_count in range(1, self._max_images + 1):
            done_buckets = set()
            for size in sizes:
                image = Image.new("RGB", (size.width, size.height))
                images = [image] * image_count
                vision_images = self._vision_images(images, [size] * image_count)
                # The prompt bucket of the image announcements alone, and the ones above it.
                empty_length = self._text_encoder.prompt_length("", images=vision_images)
                buckets = sorted(b for b in self._sequence_length_buckets if b >= empty_length)

                for bucket in reversed(buckets):
                    if bucket in done_buckets:
                        continue
                    done_buckets.add(bucket)

                    # Fill the bucket up to two tokens, staying above the next-smaller bucket.
                    word_count = bucket - empty_length - 2
                    prompt = "a " * word_count
                    while word_count > 0 and self._text_encoder.prompt_length(prompt, images=vision_images) > bucket:
                        word_count -= 1
                        prompt = "a " * word_count

                    self(
                        prompts=[prompt],
                        negative_prompts=[prompt],
                        num_inference_steps=2,
                        cfg_scale=2 if self._cfg_enabled else 1,
                        width=size.width,
                        height=size.height,
                        traced=traced,
                        encoder_traced=traced,
                        vae_traced=traced,
                        image=images,
                        # Edit inpainting takes a single image.
                        mask_image=Image.new("L", (size.width, size.height), 255) if image_count == 1 else None,
                        strength=1.0,
                    )

    def __call__(
        self,
        *,
        prompts: Sequence[str],
        negative_prompts: Sequence[str] | None = None,
        num_inference_steps: int,
        seed: int = 0,
        num_images_per_prompt: int = 1,
        cfg_scale: float = 4.0,
        width: int | None = None,
        height: int | None = None,
        traced: bool = True,
        vae_traced: bool | None = None,
        encoder_traced: bool | None = None,
        image: Image.Image | Sequence[Image.Image] | None = None,
        mask_image: Image.Image | None = None,
        strength: float = 0.6,
        on_event: PipelineEventCallback | None = None,
    ) -> list[Image.Image]:
        """Generates an image from the prompt.

        ``width`` and ``height`` must be one of the configured resolutions. They default to
        1024 x 1024, when inpainting to the configured resolution closest to the image in aspect
        ratio, and when editing to the size of the last input image.

        Inpainting: pass ``image`` and ``mask_image``. The white part of the mask is regenerated,
        the rest is kept. ``strength`` sets how much the white part may change: 1 ignores what was
        there, lower values stay closer to it.

        Editing, with the Qwen-Image-Edit checkpoints: pass the input ``image``, or up to
        ``max_images`` of them for Edit-2509 and 2511. Each is resized to the configured resolution
        closest to it in aspect ratio. With ``mask_image`` and a single input image, only the white
        part of the mask is edited, as by diffusers' ``QwenImageEditInpaintPipeline``; the output
        then has the input image's size.
        """
        prompt_count = len(prompts)

        if cfg_scale > 1 and not self._cfg_enabled:
            msg = "cfg_scale > 1 requires CFG to be enabled"
            raise ValueError(msg)

        images = [image] if isinstance(image, Image.Image) else list(image) if image is not None else []
        edit = self._max_images > 0
        if edit:
            if not images or len(images) > self._max_images:
                msg = f"editing needs 1 to {self._max_images} input images, got {len(images)}"
                raise ValueError(msg)
            inpaint = mask_image is not None
            if inpaint and len(images) != 1:
                msg = "edit inpainting takes a single input image"
                raise ValueError(msg)
            conditions = [self._nearest_size(img) for img in images]
        else:
            if (image is None) != (mask_image is None):
                msg = "inpainting needs both image and mask_image"
                raise ValueError(msg)
            if len(images) > 1:
                msg = "inpainting takes a single image"
                raise ValueError(msg)
            inpaint = image is not None
            conditions = []

        if (width is None) != (height is None):
            msg = "pass both width and height, or neither"
            raise ValueError(msg)
        if width is None or height is None:
            if conditions:
                size = conditions[-1]
            elif inpaint:
                size = self._nearest_size(images[0])
            else:
                size = self._image_sizes.get((1024, 1024))
        else:
            size = self._image_sizes.get((width, height))
        if size is None:
            msg = f"unsupported size {width}x{height}, supported are {list(self._image_sizes)}"
            raise ValueError(msg)
        if edit and inpaint and size != conditions[0]:
            msg = f"edit inpainting generates at the input image's size {conditions[0].width}x{conditions[0].height}"
            raise ValueError(msg)
        layout = self._layout(size, conditions)

        vae_traced = vae_traced if vae_traced is not None else traced
        encoder_traced = encoder_traced if encoder_traced is not None else traced
        on_event = on_event if on_event is not None else null_callback
        negative_prompts = negative_prompts if negative_prompts is not None else [""] * prompt_count

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

        on_event(SectionStart("total"))

        logger.info("encoding prompts...")
        on_event(SectionStart("encoder"))
        # One entry per submesh: the CFG pass it runs, or both passes batched on the only one.
        torch_contexts = self._text_encoder.encode_cfg(
            prompts,
            negative_prompts,
            num_images_per_prompt=num_images_per_prompt,
            cfg_enabled=self._cfg_enabled,
            batch_passes=not self._cfg_parallel,
            on_event=on_event,
            traced=encoder_traced,
            images=self._vision_images(images, conditions) if edit else (),
        )
        on_event(SectionEnd("encoder"))
        prompt_sequence_lengths = [c.shape[1] for c in torch_contexts]

        logger.info("preparing timesteps...")
        mu = calculate_shift(size.sequence_length, self._solvers[0].scheduler)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        for solver in self._solvers:
            solver.set_schedule(sigmas=sigmas, mu=mu)
        timesteps = self._solvers[0].timesteps

        # Inpainting skips the first steps of the schedule, by the same rounding as diffusers.
        first_step = int(max(num_inference_steps - min(num_inference_steps * strength, num_inference_steps), 0))
        if inpaint and first_step >= num_inference_steps:
            msg = f"strength {strength} leaves no denoising steps"
            raise ValueError(msg)
        first_step = first_step if inpaint else 0

        logger.info("preparing inputs...")
        latents = self._prepare_latents(
            layout,
            images=images,
            mask_image=mask_image,
            batch_size=prompt_count * num_images_per_prompt,
            seed=seed,
            first_step=first_step if inpaint else None,
            traced=vae_traced,
        )
        context = [tensor.from_torch(c, device=d) for c, d in zip(torch_contexts, self._devices, strict=True)]
        latents = tensor.from_torch_to_devices(latents, devices=self._devices, mesh_axes=[None, self._sp_axis, None])
        ropes = [
            self._checkpoint.rope_tables(
                latents_height=size.latents_height,
                latents_width=size.latents_width,
                image_latents_sizes=[(c.latents_height, c.latents_width) for c in layout.conditions],
                prompt_sequence_length=length,
                padded_spatial_sequence_length=layout.padded_sequence_length,
                device=device,
                sp_axis=self._sp_axis,
            )
            for length, device in zip(prompt_sequence_lengths, self._devices, strict=True)
        ]
        for d, sequence_length in zip(self._devices, self._sequence_lengths, strict=True):
            host = self._sequence_length_tensor(layout.sequence_length, device=d, on_host=True)
            ttnn.copy_host_to_device_tensor(host, sequence_length)
        condition_masks = self._condition_masks.get(layout.padded_sequence_length, [None] * len(self._devices))
        tracers = [
            self._tracers[idx, length, layout.padded_sequence_length]
            for idx, length in enumerate(prompt_sequence_lengths)
        ]

        logger.info("denoising...")
        on_event(SectionStart("denoising"))

        for step, t in enumerate(tqdm.tqdm(timesteps[first_step:]), start=first_step):
            on_event(SectionStart(f"denoising_step_{step}"))

            velocity_preds = []
            for idx, tracer in enumerate(tracers):
                timestep = ttnn.full(
                    [1, 1],
                    fill_value=t,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.float32,
                    device=self._devices[idx],
                )
                spatial_rope, prompt_rope = ropes[idx]

                velocity_preds.append(
                    tracer(
                        submesh_idx=idx,
                        latents=latents[idx],
                        prompt=context[idx] if step == first_step else tracer.inputs["prompt"],
                        timestep=timestep,
                        spatial_rope=spatial_rope if step == first_step else tracer.inputs["spatial_rope"],
                        prompt_rope=prompt_rope if step == first_step else tracer.inputs["prompt_rope"],
                        spatial_sequence_length=self._sequence_lengths[idx],
                        condition_mask=condition_masks[idx],
                        traced=traced,
                        tracer_blocking_execution=False,
                    )
                )

                # latents can be overwritten by trace execution, use the captured input instead,
                # which is safe.
                latents[idx] = tracer.inputs["latents"]

            if self._cfg_enabled:
                velocity_preds = self._combiner.combine(velocity_preds, cfg_scale)

            latents = [
                solver.step(step=step, latent=latents[idx], velocity_pred=velocity_preds[idx])
                for idx, solver in enumerate(self._solvers)
            ]

            if inpaint or edit:
                # Outside the mask, the latents are replaced by the image latents, noised to the
                # level of the next step, and the input images' rows by their latents.
                sigma_next = self._solvers[0].sigmas[step + 1]
                for idx, inputs in enumerate(self._blend_inputs[layout.padded_sequence_length]):
                    known = inputs.noise * sigma_next + inputs.image_latents * (1 - sigma_next)
                    if edit:
                        known = known + inputs.condition
                    latents[idx] = known + inputs.mask * (latents[idx] - known)

            self.synchronize_devices()  # for time profiling
            on_event(SectionEnd(f"denoising_step_{step}"))

        on_event(SectionEnd("denoising"))

        logger.info("decoding image...")
        on_event(SectionStart("vae"))
        images = self._decode_latents(latents[-1], size=size, traced=vae_traced)
        on_event(SectionEnd("vae"))

        on_event(SectionEnd("total"))
        return images

    def _traced_step(self, *, submesh_idx: int, latents: ttnn.Tensor, **kwargs: Any) -> ttnn.Tensor:
        if self._cfg_enabled and not self._cfg_parallel:
            latents = ttnn.concat([latents, latents])

        return self._transformers[submesh_idx].forward(spatial=latents, **kwargs)

    def synchronize_devices(self) -> None:
        for d in self._devices:
            ttnn.synchronize_device(d)

    def _image_size(self, width: int, height: int) -> _ImageSize:
        p = self._checkpoint.patch_size
        latents_width = width // _VAE_SCALE_FACTOR
        latents_height = height // _VAE_SCALE_FACTOR

        # The VAE splits the latents evenly across the devices.
        h_factor = self._vae_parallel_config.height_parallel.factor
        w_factor = self._vae_parallel_config.width_parallel.factor
        if (
            width % (_VAE_SCALE_FACTOR * p) != 0
            or height % (_VAE_SCALE_FACTOR * p) != 0
            or latents_height % h_factor != 0
            or latents_width % w_factor != 0
        ):
            msg = (
                f"size {width}x{height} must be a multiple of {_VAE_SCALE_FACTOR * p}, with latents divisible "
                f"by {h_factor} in height and {w_factor} in width"
            )
            raise ValueError(msg)

        return _ImageSize(
            width=width,
            height=height,
            latents_width=latents_width,
            latents_height=latents_height,
            sequence_length=(latents_width // p) * (latents_height // p),
        )

    def _layout(self, output: _ImageSize, conditions: Sequence[_ImageSize] = ()) -> _Layout:
        length = output.sequence_length + sum(c.sequence_length for c in conditions)
        buckets = [b for b in self._spatial_sequence_length_buckets if b >= length]
        if not buckets:
            msg = (
                f"size {output.width}x{output.height} with {len(conditions)} input images exceeds the largest "
                "spatial bucket"
            )
            raise ValueError(msg)
        return _Layout(
            output=output,
            conditions=tuple(conditions),
            sequence_length=length,
            padded_sequence_length=buckets[0],
        )

    def _nearest_size(self, image: Image.Image) -> _ImageSize:
        """The configured resolution closest to ``image`` in aspect ratio."""
        aspect_ratio = image.width / image.height
        return min(self._image_sizes.values(), key=lambda s: abs(math.log(s.width / s.height / aspect_ratio)))

    def _vision_images(self, images: Sequence[Image.Image], sizes: Sequence[_ImageSize]) -> list[Image.Image]:
        """The input images as the text encoder sees them.

        That is at the size of their latents, or for Edit-2509 and 2511 at about 384² pixels.
        """
        resized = []
        for image, size in zip(images, sizes, strict=True):
            width, height = self._vision_size(image.width / image.height, size)
            resized.append(self._image_processor.resize(image, height, width))
        return resized

    def _vision_size(self, aspect_ratio: float, size: _ImageSize) -> tuple[int, int]:
        """The (width, height) the text encoder sees an input image of ``aspect_ratio`` in, which is
        resized to ``size`` for the transformer."""
        area = self._mode.vision_area
        return (size.width, size.height) if area is None else edit_size(area, aspect_ratio)

    @staticmethod
    def _sequence_length_tensor(length: int, *, device: ttnn.MeshDevice, on_host: bool = False) -> ttnn.Tensor:
        return tensor.from_torch(
            torch.tensor([length]),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            on_host=on_host,
        )

    def _random_latents(self, size: _ImageSize, *, batch_size: int, seed: int, inpaint: bool) -> torch.Tensor:
        torch.manual_seed(seed)

        shape = [batch_size, _LATENT_CHANNELS, size.latents_height, size.latents_width]

        if inpaint:
            # The reference samples the VAE latent distribution of the image before drawing the
            # noise. Its spread is negligible, so the image is encoded to the mean, but the draw is
            # repeated so that the noise matches.
            torch.randn(shape, dtype=torch.bfloat16)

        # Noise is drawn in bfloat16 and channels first like in the reference implementation.
        noise = torch.randn(shape, dtype=torch.bfloat16)
        return self._transformers[0].patchify(noise.permute(0, 2, 3, 1).float())

    def _prepare_latents(
        self,
        layout: _Layout,
        *,
        images: Sequence[Image.Image],
        mask_image: Image.Image | None,
        batch_size: int,
        seed: int,
        first_step: int | None,
        traced: bool,
    ) -> torch.Tensor:
        """Returns the initial latents in the layout of the spatial sequence.

        Also fills the blend inputs, and the condition mask of zero_cond_t checkpoints. Inpainting,
        with ``first_step`` the first denoising step: the output starts as the image latents noised
        to that step. Editing: the input images' rows hold their latents.
        """
        output = layout.output
        inpaint = first_step is not None
        noise = self._random_latents(output, batch_size=batch_size, seed=seed, inpaint=inpaint)
        # Only an edit conditions on its input images; text-to-image inpainting has none.
        condition_images = images if self._max_images else []
        conditions = [
            self._encode_image(image, size=size, traced=traced)
            for image, size in zip(condition_images, layout.conditions, strict=True)
        ]
        blank = [torch.zeros_like(c) for c in conditions]

        if inpaint:
            assert mask_image is not None
            # When editing, the image to inpaint is the input image, at the same size.
            image_latents = conditions[0] if conditions else self._encode_image(images[0], size=output, traced=traced)
            sigma = self._solvers[0].sigmas[first_step]
            latents = self._joint(layout, sigma * noise + (1 - sigma) * image_latents, conditions)
            blend = _BlendInputs(
                image_latents=self._joint(layout, image_latents, blank),
                noise=self._joint(layout, noise, blank),
                mask=self._joint(layout, self._latent_mask(mask_image, size=output), blank),
                condition=self._joint(layout, torch.zeros_like(noise), conditions),
            )
        else:
            latents = self._joint(layout, noise, conditions)
            # Without inpainting, only an edit blends, keeping the input images' rows.
            blend = _BlendInputs(
                image_latents=self._joint(layout, torch.zeros_like(noise), blank),
                noise=self._joint(layout, torch.zeros_like(noise), blank),
                mask=self._joint(layout, torch.ones_like(noise), blank),
                condition=self._joint(layout, torch.zeros_like(noise), conditions),
            )

        mesh_axes = [None, self._sp_axis, None]
        if inpaint or conditions:
            for d, buffers in zip(self._devices, self._blend_inputs[layout.padded_sequence_length], strict=True):
                for source, buffer in zip(blend, buffers, strict=True):
                    host = tensor.from_torch(source, device=d, mesh_axes=mesh_axes, on_host=True)
                    ttnn.copy_host_to_device_tensor(host, buffer)

        if self._checkpoint.zero_cond_t:
            condition_mask = torch.zeros([1, layout.padded_sequence_length, 1])
            condition_mask[:, output.sequence_length : layout.sequence_length] = 1
            for d, buffer in zip(self._devices, self._condition_masks[layout.padded_sequence_length], strict=True):
                host = tensor.from_torch(condition_mask, device=d, mesh_axes=mesh_axes, on_host=True)
                ttnn.copy_host_to_device_tensor(host, buffer)

        return latents

    @staticmethod
    def _joint(layout: _Layout, output: torch.Tensor, conditions: Sequence[torch.Tensor] = ()) -> torch.Tensor:
        """Concatenates patchified tensors along the sequence and pads them to ``layout``'s bucket."""
        joint = torch.cat([output, *conditions], dim=1)
        return torch_pad(joint, layout.padded_sequence_length - joint.shape[1], dim=1)

    def _encode_image(self, image: Image.Image, *, size: _ImageSize, traced: bool) -> torch.Tensor:
        pixels = self._image_processor.preprocess(image, height=size.height, width=size.width)
        latents = self._vae_encoder.encode(pixels.to(torch.float32), traced=traced)
        return self._transformers[0].patchify(latents)

    def _latent_mask(self, mask_image: Image.Image, *, size: _ImageSize) -> torch.Tensor:
        mask = self._mask_processor.preprocess(mask_image, height=size.height, width=size.width)
        mask = torch.nn.functional.interpolate(mask, size=(size.latents_height, size.latents_width))
        return self._transformers[0].patchify(mask.repeat(1, _LATENT_CHANNELS, 1, 1).permute(0, 2, 3, 1))

    def _decode_latents(self, tt_latents: ttnn.Tensor, *, size: _ImageSize, traced: bool) -> list[Image.Image]:
        tt_latents = self._ccl_managers[-1].all_gather_persistent_buffer(
            tt_latents, dim=1, mesh_axis=self._sp_axis, use_hyperparams=True
        )

        torch_latents = tensor.to_torch(tt_latents)[:, : size.sequence_length]
        torch_latents = self._transformers[0].unpatchify(
            torch_latents, height=size.latents_height, width=size.latents_width
        )

        decoded_output = self._vae.decode(torch_latents, traced=traced)

        image = self._image_processor.postprocess(decoded_output, output_type="pt")
        assert isinstance(image, torch.Tensor)

        return self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))
