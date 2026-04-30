# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Adapted from:
#   https://huggingface.co/spaces/Qwen/Qwen-Image-Edit-2509/blob/main/qwenimage/pipeline_qwenimage_edit_plus.py
#   models/tt_dit/pipelines/qwenimage/pipeline_qwenimage.py

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import diffusers
import numpy as np
import torch
import tqdm
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler

from ...encoders.qwen25vl.encoder_pair import Qwen25VlTokenizerEncoderPair
from ...models.transformers.transformer_qwenimage import QwenImageTransformer
from ...models.vae.vae_qwenimage import QwenImageVaeDecoder, QwenImageVaeEncoder
from ...parallel.config import (
    DiTParallelConfig,
    EncoderParallelConfig,
    ParallelFactor,
    VaeHWParallelConfig,
    VAEParallelConfig,
)
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from ...utils.padding import PaddingConfig
from ..qwenimage.qwen_pos_embed_tt import QwenPosEmbedTT

if TYPE_CHECKING:
    pass

    from PIL import Image

CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024

PROMPT_TEMPLATE_EDIT = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
PROMPT_DROP_IDX_EDIT = 64

# Fixed sequence length the image-conditioned prompt path (``_encode_prompts_with_images``)
# pads its output to. The text-only path already produces a fixed 512 tokens (via
# ``sequence_length=512 + PROMPT_DROP_IDX_EDIT`` then dropping the first 64), so using
# the same constant here keeps both paths shape-stable across calls. That matters for
# traced multi-prompt interactive runs: the trace's prompt-input placeholder is bound
# once, and every later call must match it. A 384x384 condition image contributes
# ~188 tokens plus ~50 template tokens; 512 leaves ~275 tokens for the user text,
# which is plenty for typical prompts.
IMAGE_PROMPT_PAD_SEQ_LEN = 512


@dataclass
class PipelineTrace:
    tid: int
    spatial_input: ttnn.Tensor
    prompt_input: ttnn.Tensor
    timestep_input: ttnn.Tensor
    sigma_difference_input: ttnn.Tensor
    latents_output: ttnn.Tensor
    spatial_rope_cos: ttnn.Tensor
    spatial_rope_sin: ttnn.Tensor
    prompt_rope_cos: ttnn.Tensor
    prompt_rope_sin: ttnn.Tensor


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int,
    max_seq_len: int,
    base_shift: float,
    max_shift: float,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _schedule(
    scheduler: FlowMatchEulerDiscreteScheduler,
    *,
    step_count: int,
    spatial_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scheduler.set_timesteps(
        sigmas=np.linspace(1.0, 1 / step_count, step_count),
        mu=_calculate_shift(
            spatial_sequence_length,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        ),
    )

    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas

    assert isinstance(timesteps, torch.Tensor)
    assert isinstance(sigmas, torch.Tensor)

    return timesteps, sigmas


def _calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


def _pack_latents(latents: torch.Tensor, batch_size: int, num_channels: int, height: int, width: int) -> torch.Tensor:
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
    return latents


def _unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
    batch_size, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
    return latents


class QwenImageEditPipeline:
    """
    Pipeline for editing images using Qwen-Image-Edit-2511.

    Uses the same QwenImageTransformer2DModel, AutoencoderKLQwenImage VAE, and
    Qwen2.5VL text encoder as the QwenImage generation pipeline. The key differences:
      - Takes an input image that is encoded through the VAE
      - Images are also passed to the VL encoder for cross-modal conditioning
      - Noise latents are concatenated with image latents along the sequence dimension
      - Supports true classifier-free guidance
    """

    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        checkpoint_name: str = "Qwen/Qwen-Image-Edit-2511",
        use_torch_text_encoder: bool = False,
        use_torch_vae_decoder: bool = False,
        parallel_config: DiTParallelConfig,
        encoder_parallel_config: EncoderParallelConfig | None = None,
        vae_parallel_config: VAEParallelConfig | None = None,
        topology: ttnn.Topology,
        num_links: int,
        height: int = 1024,
        width: int = 1024,
        is_fsdp: bool = False,
        dynamic_load_encoder: bool = True,
        dynamic_load_vae: bool = False,
    ) -> None:
        if dynamic_load_encoder or dynamic_load_vae:
            assert (
                cache.cache_dir_is_set()
            ), "Dynamic loading of encoder or vae is enabled but the cache directory (env variable TT_DIT_CACHE_DIR) is not set."

        self._mesh_device = mesh_device
        self._parallel_config = parallel_config
        self._height = height
        self._width = width
        self._is_fsdp = is_fsdp
        self._checkpoint_name = checkpoint_name

        submesh_shape = list(mesh_device.shape)
        submesh_shape[parallel_config.sequence_parallel.mesh_axis] = parallel_config.sequence_parallel.factor
        submesh_shape[parallel_config.tensor_parallel.mesh_axis] = parallel_config.tensor_parallel.factor
        logger.info(f"Parallel config: {parallel_config}")
        logger.info(f"Original mesh shape: {mesh_device.shape}")
        logger.info(f"Creating submeshes with shape {submesh_shape}")

        self._submesh_devices = self._mesh_device.create_submeshes(ttnn.MeshShape(*submesh_shape))[
            0 : parallel_config.cfg_parallel.factor
        ]
        self._ccl_managers = [
            CCLManager(submesh_device, num_links=num_links, topology=topology)
            for submesh_device in self._submesh_devices
        ]

        self.encoder_submesh_idx = 0
        self.vae_submesh_idx = len(self._submesh_devices) - self.encoder_submesh_idx - 1

        self.encoder_device = self._submesh_devices[self.encoder_submesh_idx] if not use_torch_text_encoder else None
        self.vae_device = self._submesh_devices[self.vae_submesh_idx] if not use_torch_vae_decoder else None

        self._encoder_parallel_config = encoder_parallel_config
        self._vae_parallel_config = vae_parallel_config
        self._wan_vae_parallel_config = self.get_wan_vae_parallel_config()

        self.encoder_mesh_shape = self.get_mesh_shape(
            self.encoder_device, self._encoder_parallel_config.tensor_parallel
        )
        self.vae_mesh_shape = self.get_mesh_shape(self.vae_device, self._vae_parallel_config.tensor_parallel)

        logger.info("loading models...")

        torch_transformer = diffusers.QwenImageTransformer2DModel.from_pretrained(
            checkpoint_name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        torch_transformer.eval()

        self._torch_vae = AutoencoderKLQwenImage.from_pretrained(checkpoint_name, subfolder="vae")
        assert isinstance(self._torch_vae, AutoencoderKLQwenImage)
        self._vae_state_dict = self._torch_vae.state_dict()

        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_name, subfolder="scheduler")

        self._num_channels_latents = 16
        self._patch_size = torch_transformer.config.patch_size
        self._vae_scale_factor = 8
        self._zero_cond_t = getattr(torch_transformer.config, "zero_cond_t", False)

        if torch_transformer.config.num_attention_heads % parallel_config.tensor_parallel.factor != 0:
            padding_config = PaddingConfig.from_tensor_parallel_factor(
                torch_transformer.config.num_attention_heads,
                torch_transformer.config.attention_head_dim,
                parallel_config.tensor_parallel.factor,
            )
        else:
            padding_config = None

        self._transformer_state_dict = torch_transformer.state_dict()
        self._padding_config = padding_config
        self._pos_embed = torch_transformer.pos_embed
        # Device-side RoPE builder; used when sp_factor == 1 (output is replicated).
        # Falls back to the host builder above for sp_factor > 1 because we do not
        # yet have a ttnn primitive for replicated->SP-sharded resharding along the
        # sequence axis.
        self._pos_embed_tt = QwenPosEmbedTT(
            torch_pos_embed=torch_transformer.pos_embed,
            submesh_devices=self._submesh_devices,
        )

        self.transformers = []
        for i, submesh_device in enumerate(self._submesh_devices):
            self.transformers.append(
                QwenImageTransformer(
                    patch_size=torch_transformer.config.patch_size,
                    in_channels=torch_transformer.config.in_channels,
                    num_layers=torch_transformer.config.num_layers,
                    attention_head_dim=torch_transformer.config.attention_head_dim,
                    num_attention_heads=torch_transformer.config.num_attention_heads,
                    joint_attention_dim=torch_transformer.config.joint_attention_dim,
                    out_channels=torch_transformer.config.out_channels,
                    device=submesh_device,
                    ccl_manager=self._ccl_managers[i],
                    parallel_config=self._parallel_config,
                    padding_config=self._padding_config,
                    is_fsdp=self._is_fsdp,
                )
            )
        self._transformers_loaded = False

        self._use_torch_text_encoder = use_torch_text_encoder
        with self.mesh_reshape(self.encoder_device, self.encoder_mesh_shape, synchronize=True):
            logger.info("creating TT-NN text encoder (loading before transformers for memory efficiency)...")
            self._text_encoder = Qwen25VlTokenizerEncoderPair(
                self._checkpoint_name,
                tokenizer_subfolder="tokenizer",
                encoder_subfolder="text_encoder",
                device=self._submesh_devices[self.encoder_submesh_idx],
                ccl_manager=self._ccl_managers[self.encoder_submesh_idx],
                parallel_config=self._encoder_parallel_config,
                use_torch=use_torch_text_encoder,
                is_fsdp=True,
                build_vision_encoder=not use_torch_text_encoder,
            )

        # Image-conditioned VL preprocessing (token expansion + ``pixel_values``) is handled
        # inside ``Qwen25VlTokenizerEncoderPair`` via ``Qwen25VlMultimodalPreprocessor``.

        if not dynamic_load_encoder or use_torch_text_encoder:
            self._load_transformers(self.encoder_submesh_idx)

        self._load_transformers(self.vae_submesh_idx)

        self._latents_scaling = 1.0 / torch.tensor(self._torch_vae.config.latents_std)
        self._latents_shift = torch.tensor(self._torch_vae.config.latents_mean)
        self._latents_mean = torch.tensor(self._torch_vae.config.latents_mean).view(
            1, self._torch_vae.config.z_dim, 1, 1, 1
        )
        self._latents_std = torch.tensor(self._torch_vae.config.latents_std).view(
            1, self._torch_vae.config.z_dim, 1, 1, 1
        )

        self._image_processor = VaeImageProcessor(vae_scale_factor=2 * self._vae_scale_factor)

        self._use_torch_vae_decoder = use_torch_vae_decoder

        if use_torch_vae_decoder:
            self._vae_decoder = None
        else:
            with self.mesh_reshape(self.vae_device, self.vae_mesh_shape, synchronize=True):
                logger.info("creating TT-NN VAE decoder...")
                self._vae_decoder = QwenImageVaeDecoder(
                    base_dim=self._torch_vae.config.base_dim,
                    z_dim=self._torch_vae.config.z_dim,
                    dim_mult=self._torch_vae.config.dim_mult,
                    num_res_blocks=self._torch_vae.config.num_res_blocks,
                    temperal_downsample=self._torch_vae.config.temperal_downsample,
                    device=self.vae_device,
                    parallel_config=self._wan_vae_parallel_config,
                    ccl_manager=self._ccl_managers[self.vae_submesh_idx],
                )

            if not dynamic_load_vae:
                self._vae_decoder.load_torch_state_dict(self._vae_state_dict)

        # On-device VAE encoder for image conditioning
        if not use_torch_vae_decoder:
            with self.mesh_reshape(self.vae_device, self.vae_mesh_shape, synchronize=True):
                logger.info("creating TT-NN VAE encoder...")
                self._vae_encoder = QwenImageVaeEncoder(
                    base_dim=self._torch_vae.config.base_dim,
                    z_dim=self._torch_vae.config.z_dim,
                    dim_mult=self._torch_vae.config.dim_mult,
                    num_res_blocks=self._torch_vae.config.num_res_blocks,
                    attn_scales=self._torch_vae.config.attn_scales,
                    temperal_downsample=self._torch_vae.config.temperal_downsample,
                    is_residual=getattr(self._torch_vae.config, "is_residual", False),
                    device=self.vae_device,
                    parallel_config=self._wan_vae_parallel_config,
                    ccl_manager=self._ccl_managers[self.vae_submesh_idx],
                )
                self._vae_encoder.load_torch_state_dict(self._vae_state_dict)
        else:
            self._vae_encoder = None

        self._traces = None

        logger.info("warming up for tracing...")
        self.run_single_edit(prompt="test edit", image=None, num_inference_steps=1, seed=0, traced=False, skip_vae=True)

    def _load_transformers(self, idx) -> None:
        if self.transformers[idx].is_loaded():
            return

        cache.load_model(
            tt_model=self.transformers[idx],
            get_torch_state_dict=lambda: self._transformer_state_dict,
            model_name=self._checkpoint_name,
            subfolder="transformer",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self._submesh_devices[idx].shape),
            is_fsdp=self._is_fsdp,
        )

        ttnn.synchronize_device(self._submesh_devices[idx])

    def _deallocate_transformers(self, idx) -> None:
        if not self.transformers[idx].is_loaded():
            return

        logger.info("deallocating transformer weights to free memory...")
        self.transformers[idx].deallocate_weights()
        ttnn.synchronize_device(self._submesh_devices[idx])

    def _deallocate_vae(self) -> None:
        if self._use_torch_vae_decoder or not self._vae_decoder.is_loaded():
            return

        logger.info("deallocating VAE decoder weights to free memory...")
        self._vae_decoder.deallocate_weights()
        ttnn.synchronize_device(self.vae_device)

    def _reload_vae(self) -> None:
        if self._use_torch_vae_decoder or self._vae_decoder.is_loaded():
            return

        with self.mesh_reshape(self.vae_device, self.vae_mesh_shape, synchronize=True):
            logger.info("loading VAE decoder weights to device...")
            self._vae_decoder.load_torch_state_dict(self._vae_state_dict)

    @contextmanager
    def mesh_reshape(
        self, device: ttnn.MeshDevice | None, mesh_shape: ttnn.MeshShape | None, synchronize: bool = False
    ):
        if device is None:
            yield
        else:
            original_mesh_shape = ttnn.MeshShape(tuple(device.shape))
            assert (
                original_mesh_shape.mesh_size() == mesh_shape.mesh_size()
            ), f"Device cannot be reshaped device shape: {device.shape} mesh shape: {mesh_shape}"
            if original_mesh_shape != mesh_shape:
                device.reshape(mesh_shape)
            yield
            if original_mesh_shape != device.shape:
                device.reshape(original_mesh_shape)
            if synchronize:
                ttnn.synchronize_device(device)

    @staticmethod
    def get_mesh_shape(mesh_device, parallel_factor):
        if mesh_device is None:
            return None
        mesh_shape = list(mesh_device.shape)
        mesh_shape[parallel_factor.mesh_axis] = parallel_factor.factor
        mesh_shape[1 - parallel_factor.mesh_axis] = mesh_device.shape.mesh_size() // parallel_factor.factor
        return ttnn.MeshShape(tuple(mesh_shape))

    def get_wan_vae_parallel_config(self):
        return VaeHWParallelConfig(
            height_parallel=ParallelFactor(
                factor=self.vae_device.shape[self._vae_parallel_config.tensor_parallel.mesh_axis],
                mesh_axis=self._vae_parallel_config.tensor_parallel.mesh_axis,
            ),
            width_parallel=ParallelFactor(
                factor=self.vae_device.shape[1 - self._vae_parallel_config.tensor_parallel.mesh_axis],
                mesh_axis=1 - self._vae_parallel_config.tensor_parallel.mesh_axis,
            ),
        )

    @staticmethod
    def create_pipeline(
        *,
        checkpoint_name: str = "Qwen/Qwen-Image-Edit-2511",
        mesh_device: ttnn.MeshDevice,
        dit_cfg: tuple[int, int] | None = None,
        dit_sp: tuple[int, int] | None = None,
        dit_tp: tuple[int, int] | None = None,
        encoder_tp: tuple[int, int] | None = None,
        vae_tp: tuple[int, int] | None = None,
        use_torch_text_encoder: bool = False,
        use_torch_vae_decoder: bool = False,
        num_links: int | None = None,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        width: int = 1024,
        height: int = 1024,
        is_fsdp: bool | None = None,
        dynamic_load_encoder: bool | None = None,
        dynamic_load_vae: bool | None = None,
    ) -> QwenImageEditPipeline:
        default_config = {
            (2, 4): {
                "cfg_config": (2, 0),
                "sp": (1, 0),
                "tp": (4, 1),
                "encoder_tp": (4, 1),
                "vae_tp": (4, 1),
                "num_links": 1,
                "is_fsdp": True,
                "dynamic_load_encoder": True,
                "dynamic_load_vae": not ttnn.device.is_blackhole(),
            },
            (4, 8): {
                "cfg_config": (2, 1),
                "sp": (4, 0),
                "tp": (4, 1),
                "encoder_tp": (4, 1),
                "vae_tp": (4, 1),
                "num_links": 4,
                "is_fsdp": False,
                "dynamic_load_encoder": False,
                "dynamic_load_vae": False,
            },
        }
        cfg_factor, cfg_axis = dit_cfg or default_config[tuple(mesh_device.shape)]["cfg_config"]
        sp_factor, sp_axis = dit_sp or default_config[tuple(mesh_device.shape)]["sp"]
        tp_factor, tp_axis = dit_tp or default_config[tuple(mesh_device.shape)]["tp"]
        encoder_tp_factor, encoder_tp_axis = encoder_tp or default_config[tuple(mesh_device.shape)]["encoder_tp"]
        vae_tp_factor, vae_tp_axis = vae_tp or default_config[tuple(mesh_device.shape)]["vae_tp"]
        num_links = num_links or default_config[tuple(mesh_device.shape)]["num_links"]
        is_fsdp = is_fsdp if is_fsdp is not None else default_config[tuple(mesh_device.shape)]["is_fsdp"]
        dynamic_load_encoder = (
            dynamic_load_encoder
            if dynamic_load_encoder is not None
            else default_config[tuple(mesh_device.shape)]["dynamic_load_encoder"]
        )
        dynamic_load_vae = (
            dynamic_load_vae
            if dynamic_load_vae is not None
            else default_config[tuple(mesh_device.shape)]["dynamic_load_vae"]
        )

        dit_parallel_config = DiTParallelConfig(
            cfg_parallel=ParallelFactor(factor=cfg_factor, mesh_axis=cfg_axis),
            tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
            sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
        )
        encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=encoder_tp_factor, mesh_axis=encoder_tp_axis)
        )
        vae_parallel_config = VAEParallelConfig(
            tensor_parallel=ParallelFactor(factor=vae_tp_factor, mesh_axis=vae_tp_axis)
        )

        logger.info(f"Mesh device shape: {mesh_device.shape}")
        logger.info(f"Parallel config: {dit_parallel_config}")
        logger.info(f"Encoder parallel config: {encoder_parallel_config}")
        logger.info(f"VAE parallel config: {vae_parallel_config}")

        return QwenImageEditPipeline(
            mesh_device=mesh_device,
            checkpoint_name=checkpoint_name,
            use_torch_text_encoder=use_torch_text_encoder,
            use_torch_vae_decoder=use_torch_vae_decoder,
            parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel_config=vae_parallel_config,
            topology=topology,
            num_links=num_links,
            width=width,
            height=height,
            is_fsdp=is_fsdp,
            dynamic_load_encoder=dynamic_load_encoder,
            dynamic_load_vae=dynamic_load_vae,
        )

    def prepare_encoder(self) -> None:
        if not self._text_encoder.encoder_loaded():
            self._deallocate_transformers(self.encoder_submesh_idx)
            with self.mesh_reshape(self.encoder_device, self.encoder_mesh_shape):
                self._text_encoder.reload_encoder_weights()

    def prepare_transformers(self) -> None:
        if not self.transformers[self.encoder_submesh_idx].is_loaded():
            self._text_encoder.deallocate_encoder_weights()
            self._load_transformers(self.encoder_submesh_idx)

        if not self.transformers[self.vae_submesh_idx].is_loaded():
            self._deallocate_vae()
            self._load_transformers(self.vae_submesh_idx)

    def prepare_vae(self) -> None:
        if not self._vae_decoder.is_loaded():
            self._deallocate_transformers(self.vae_submesh_idx)
            with self.mesh_reshape(self.vae_device, self.vae_mesh_shape):
                self._reload_vae()

    def _get_tt_vae_latents_normalization(self, shard_shape: tuple[int, ...]):
        key = tuple(shard_shape)
        cached = getattr(self, "_tt_vae_norm_cache", None)
        if cached is None:
            cached = {}
            self._tt_vae_norm_cache = cached
        if key in cached:
            return cached[key]
        mean_BCHW = self._latents_mean.squeeze(2).to(torch.bfloat16).contiguous()
        inv_std_BCHW = (1.0 / self._latents_std).squeeze(2).to(torch.bfloat16).contiguous()
        tt_mean = tensor.from_torch(mean_BCHW, device=self.vae_device, layout=ttnn.TILE_LAYOUT)
        tt_inv_std = tensor.from_torch(inv_std_BCHW, device=self.vae_device, layout=ttnn.TILE_LAYOUT)
        cached[key] = (tt_mean, tt_inv_std)
        return tt_mean, tt_inv_std

    def _encode_vae_image(self, image_BCHW: torch.Tensor) -> torch.Tensor:
        """Encode an image through the VAE to get latent representation ``(B, C, H, W)``."""
        if self._vae_encoder is not None:
            height_par = self._wan_vae_parallel_config.height_parallel.factor * self._vae_scale_factor
            with self.mesh_reshape(self.vae_device, self.vae_mesh_shape):
                tt_image, logical_h = self._vae_encoder.prepare_input(image_BCHW, height_par)
                tt_latents_BCTHW, new_logical_h = self._vae_encoder.forward(tt_image, logical_h)

                B, C, T, H_sh, W_sh = tt_latents_BCTHW.shape
                tt_mean, tt_inv_std = self._get_tt_vae_latents_normalization((B, C, T, H_sh, W_sh))
                tt_latents_BCHW = ttnn.reshape(tt_latents_BCTHW, (B, C, T * H_sh, W_sh))
                tt_latents_BCHW = ttnn.to_layout(tt_latents_BCHW, ttnn.TILE_LAYOUT)
                tt_latents_BCHW = ttnn.subtract(tt_latents_BCHW, tt_mean)
                tt_latents_BCHW = ttnn.multiply(tt_latents_BCHW, tt_inv_std)
                tt_latents_BCHW = ttnn.to_layout(tt_latents_BCHW, ttnn.ROW_MAJOR_LAYOUT)
                tt_latents_BCTHW = ttnn.reshape(tt_latents_BCHW, (B, C, T, H_sh, W_sh))

                image_latents = self._vae_encoder.postprocess_output(tt_latents_BCTHW, new_logical_h)
            image_latents = image_latents.to(dtype=image_BCHW.dtype)
        else:
            with torch.no_grad():
                encoder_output = self._torch_vae.encode(image_BCHW.unsqueeze(2))
                if hasattr(encoder_output, "latent_dist"):
                    image_latents = encoder_output.latent_dist.mode()
                elif hasattr(encoder_output, "latents"):
                    image_latents = encoder_output.latents
                else:
                    raise AttributeError("Could not access latents from VAE encoder output")
            image_latents = (image_latents - self._latents_mean.to(image_latents.device, image_latents.dtype)) / (
                self._latents_std.to(image_latents.device, image_latents.dtype)
            )

        # Drop the temporal singleton dim; image VAE always runs with T=1.
        return image_latents.squeeze(2)

    def _prepare_image_latents(
        self,
        images: list[torch.Tensor],
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> list[torch.Tensor] | None:
        """Encode images through VAE and return one host ``[B, C, H, W]`` tensor per
        reference image, batch-replicated to ``batch_size``.

        Latent packing and the cross-image concat both run on device in ``__call__``
        (via ``tensor.pack_latents_device`` + ``ttnn.concat``) so the equivalent host
        compute here is skipped. Returning a list of unpacked tensors also lets the
        transformer loop upload each image once per submesh and pack in-place,
        consistent with how ``noise_latents`` is already handled for all
        ``sp_factor`` configurations.
        """
        if not images:
            return None

        all_image_latents: list[torch.Tensor] = []
        for img in images:
            img = img.to(dtype=torch.float32)
            image_latents = self._encode_vae_image(img)

            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                additional = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional, dim=0)

            all_image_latents.append(image_latents.to(dtype=dtype))

        return all_image_latents

    def run_single_edit(
        self,
        *,
        prompt: str,
        image: Image.Image | list[Image.Image] | None = None,
        negative_prompt: str | None = None,
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: int = 0,
        traced: bool = True,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
        skip_vae: bool = False,
    ) -> list[Image.Image]:
        return self(
            prompts=[prompt],
            images=image,
            negative_prompts=[negative_prompt],
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            seed=seed,
            traced=traced,
            profiler=profiler,
            profiler_iteration=profiler_iteration,
            skip_vae=skip_vae,
        )

    def __call__(
        self,
        *,
        num_images_per_prompt: int = 1,
        true_cfg_scale: float = 4.0,
        prompts: list[str],
        images: Image.Image | list[Image.Image] | None = None,
        negative_prompts: list[str | None],
        num_inference_steps: int,
        seed: int | None = None,
        traced: bool = False,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
        skip_vae: bool = False,
    ) -> list[Image.Image]:
        prompt_count = len(prompts)

        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        cfg_factor = self._parallel_config.cfg_parallel.factor

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple prompts is not supported"

        latents_height = self._height // self._vae_scale_factor
        latents_width = self._width // self._vae_scale_factor
        transformer_batch_size = prompt_count * num_images_per_prompt
        spatial_sequence_length = (latents_height // self._patch_size) * (latents_width // self._patch_size)

        with profiler("total", profiler_iteration) if profiler else nullcontext():
            has_neg_prompt = negative_prompts[0] is not None
            do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

            # Preprocess input images for conditioning
            condition_images = []
            vae_images = []
            vae_image_sizes = []
            if images is not None:
                if not isinstance(images, list):
                    images = [images]
                for img in images:
                    image_width, image_height = img.size
                    cond_w, cond_h = _calculate_dimensions(CONDITION_IMAGE_SIZE, image_width / image_height)
                    vae_w, vae_h = _calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
                    condition_images.append(self._image_processor.resize(img, cond_h, cond_w))
                    vae_images.append(self._image_processor.preprocess(img, vae_h, vae_w))
                    vae_image_sizes.append((vae_w, vae_h))

            logger.info("encoding prompts...")
            self.prepare_encoder()

            with profiler("encoder", profiler_iteration) if profiler else nullcontext():
                with self.mesh_reshape(self.encoder_device, self.encoder_mesh_shape):
                    prompt_embeds, prompt_mask = self._encode_prompts(
                        prompts=prompts,
                        negative_prompts=negative_prompts,
                        images=condition_images if condition_images else None,
                        num_images_per_prompt=num_images_per_prompt,
                        do_true_cfg=do_true_cfg,
                        profiler=profiler,
                        profiler_iteration=profiler_iteration,
                    )
            _, prompt_sequence_length, _ = prompt_embeds.shape

            self.prepare_transformers()

            logger.info("preparing timesteps...")
            timesteps, sigmas = _schedule(
                self._scheduler,
                step_count=num_inference_steps,
                spatial_sequence_length=spatial_sequence_length,
            )

            logger.info("preparing latents...")

            if seed is not None:
                torch.manual_seed(seed)

            shape = [
                transformer_batch_size,
                self._num_channels_latents,
                1,
                self._height // self._vae_scale_factor,
                self._width // self._vae_scale_factor,
            ]
            noise_latents = torch.randn(shape, dtype=torch.float32)

            # Encode source images through VAE for latent-space conditioning
            image_latents_packed = None
            if vae_images:
                image_latents_packed = self._prepare_image_latents(
                    vae_images,
                    batch_size=transformer_batch_size,
                    height=self._height,
                    width=self._width,
                    dtype=torch.float32,
                )

            # Latent packing and RoPE assembly both run on device. When sp_factor > 1 the
            # sequence axis of the transformer input and the spatial RoPE tables must be
            # sharded across the sp mesh axis; we build each as a replicated device tensor
            # first, then convert it to sp-sharded via ``ttnn.mesh_partition`` (inverse of
            # all-gather). Prompt RoPE stays replicated because the text sequence is not
            # sp-sharded.
            noise_h, noise_w = noise_latents.shape[3], noise_latents.shape[4]
            sp_factor = self._parallel_config.sequence_parallel.factor
            noise_seq_len = (noise_h // 2) * (noise_w // 2)
            image_seq_len = 0
            if image_latents_packed is not None:
                # image_latents_packed is now a list of [B, C, H, W] host tensors (one
                # per reference image). Each contributes (H/2)*(W/2) packed spatial
                # tokens once packed on device in the loop below.
                for img_lat in image_latents_packed:
                    _, _, h_lat, w_lat = img_lat.shape
                    image_seq_len += (h_lat // 2) * (w_lat // 2)
            total_spatial_seq = noise_seq_len + image_seq_len

            # RoPE: build img_shapes accounting for noise + image patches
            p = self._patch_size
            img_shapes_per_batch = [(1, latents_height // p, latents_width // p)]
            for vae_w, vae_h in vae_image_sizes:
                img_shapes_per_batch.append(
                    (1, vae_h // self._vae_scale_factor // p, vae_w // self._vae_scale_factor // p)
                )
            img_shapes = [img_shapes_per_batch] * transformer_batch_size
            txt_seq_lens = [prompt_sequence_length] * transformer_batch_size

            # Transfer to devices
            tt_prompt_embeds_device_list = []
            tt_prompt_embeds_list = []
            tt_latents_step_list = []
            tt_spatial_rope_cos_list = []
            tt_spatial_rope_sin_list = []
            tt_prompt_rope_cos_list = []
            tt_prompt_rope_sin_list = []

            for i, submesh_device in enumerate(self._submesh_devices):
                if do_true_cfg and cfg_factor == 2 and prompt_embeds.shape[0] >= 2:
                    prompt_slice = prompt_embeds[i : i + 1]
                else:
                    prompt_slice = prompt_embeds

                tt_prompt_embeds_device = tensor.from_torch(
                    prompt_slice,
                    device=submesh_device,
                    on_host=traced,
                )
                tt_prompt_embeds = tensor.from_torch(
                    prompt_slice,
                    device=submesh_device,
                    on_host=True,
                )

                tt_noise = tensor.from_torch(noise_latents, device=submesh_device, on_host=False)
                tt_noise_packed = tensor.pack_latents_device(
                    tt_noise,
                    transformer_batch_size,
                    self._num_channels_latents,
                    noise_h,
                    noise_w,
                )
                if image_latents_packed is not None:
                    # Upload each VAE image tensor [B, C, H, W] and pack on device so
                    # the host-side ``_pack_latents`` permute/reshape is skipped.
                    tt_image_packed_list = []
                    for img_lat in image_latents_packed:
                        tt_img_bchw = tensor.from_torch(img_lat, device=submesh_device, on_host=False)
                        _, _, h_lat, w_lat = img_lat.shape
                        tt_image_packed_list.append(
                            tensor.pack_latents_device(
                                tt_img_bchw,
                                transformer_batch_size,
                                self._num_channels_latents,
                                h_lat,
                                w_lat,
                            )
                        )
                    if len(tt_image_packed_list) == 1:
                        tt_image_packed = tt_image_packed_list[0]
                    else:
                        tt_image_packed = ttnn.concat(tt_image_packed_list, dim=1)
                    tt_initial_latents = ttnn.concat([tt_noise_packed, tt_image_packed], dim=1)
                else:
                    tt_initial_latents = tt_noise_packed
                if sp_factor > 1:
                    tt_initial_latents = tensor.mesh_partition_with_padding(
                        tt_initial_latents, dim=1, cluster_axis=sp_axis, cluster_size=sp_factor
                    )

                cos_half, sin_half, txt_cos_half, txt_sin_half = self._pos_embed_tt.build(
                    submesh_index=i,
                    img_shapes_per_batch=img_shapes[0],
                    max_txt_seq_len=txt_seq_lens[0],
                )
                tt_spatial_rope_cos = tensor.rope_double_last_dim_device(cos_half)
                tt_spatial_rope_sin = tensor.rope_double_last_dim_device(sin_half)
                tt_prompt_rope_cos = tensor.rope_double_last_dim_device(txt_cos_half)
                tt_prompt_rope_sin = tensor.rope_double_last_dim_device(txt_sin_half)
                if sp_factor > 1:
                    tt_spatial_rope_cos = tensor.mesh_partition_with_padding(
                        tt_spatial_rope_cos, dim=0, cluster_axis=sp_axis, cluster_size=sp_factor
                    )
                    tt_spatial_rope_sin = tensor.mesh_partition_with_padding(
                        tt_spatial_rope_sin, dim=0, cluster_axis=sp_axis, cluster_size=sp_factor
                    )

                if traced:
                    if self._traces is None:
                        tt_prompt_embeds_device = tt_prompt_embeds_device.to(submesh_device)
                    else:
                        ttnn.copy(tt_initial_latents, self._traces[i].spatial_input)
                        ttnn.copy_host_to_device_tensor(tt_prompt_embeds_device, self._traces[i].prompt_input)
                        ttnn.copy(tt_spatial_rope_cos, self._traces[i].spatial_rope_cos)
                        ttnn.copy(tt_spatial_rope_sin, self._traces[i].spatial_rope_sin)
                        ttnn.copy(tt_prompt_rope_cos, self._traces[i].prompt_rope_cos)
                        ttnn.copy(tt_prompt_rope_sin, self._traces[i].prompt_rope_sin)

                        tt_initial_latents = self._traces[i].spatial_input
                        tt_prompt_embeds_device = self._traces[i].prompt_input
                        tt_spatial_rope_cos = self._traces[i].spatial_rope_cos
                        tt_spatial_rope_sin = self._traces[i].spatial_rope_sin
                        tt_prompt_rope_cos = self._traces[i].prompt_rope_cos
                        tt_prompt_rope_sin = self._traces[i].prompt_rope_sin

                tt_prompt_embeds_device_list.append(tt_prompt_embeds_device)
                tt_prompt_embeds_list.append(tt_prompt_embeds)
                tt_latents_step_list.append(tt_initial_latents)
                tt_spatial_rope_cos_list.append(tt_spatial_rope_cos)
                tt_spatial_rope_sin_list.append(tt_spatial_rope_sin)
                tt_prompt_rope_cos_list.append(tt_prompt_rope_cos)
                tt_prompt_rope_sin_list.append(tt_prompt_rope_sin)

            logger.info("denoising...")

            for submesh_nr in range(len(self._submesh_devices)):
                ttnn.copy_host_to_device_tensor(
                    tt_prompt_embeds_list[submesh_nr],
                    tt_prompt_embeds_device_list[submesh_nr],
                )

            with profiler("denoising", profiler_iteration) if profiler else nullcontext():
                for i, t in enumerate(tqdm.tqdm(timesteps)):
                    with profiler(f"denoising_step_{i}", profiler_iteration) if profiler else nullcontext():
                        sigma_difference = sigmas[i + 1] - sigmas[i]

                        tt_timestep_list = []
                        tt_sigma_difference_list = []
                        for submesh_device in self._submesh_devices:
                            tt_timestep = ttnn.full(
                                [1, 1],
                                fill_value=t,
                                layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.float32,
                                device=submesh_device if not traced else None,
                            )
                            tt_timestep_list.append(tt_timestep)

                            tt_sigma_difference = ttnn.full(
                                [1, 1],
                                fill_value=sigma_difference,
                                layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.bfloat16,
                                device=submesh_device if not traced else None,
                            )
                            tt_sigma_difference_list.append(tt_sigma_difference)

                        tt_latents_step_list = self._step(
                            timestep=tt_timestep_list,
                            latents=tt_latents_step_list,
                            do_true_cfg=do_true_cfg,
                            true_cfg_scale=true_cfg_scale,
                            prompt_embeds=tt_prompt_embeds_device_list,
                            sigma_difference=tt_sigma_difference_list,
                            spatial_rope_cos=tt_spatial_rope_cos_list,
                            spatial_rope_sin=tt_spatial_rope_sin_list,
                            prompt_rope_cos=tt_prompt_rope_cos_list,
                            prompt_rope_sin=tt_prompt_rope_sin_list,
                            spatial_sequence_length=total_spatial_seq,
                            prompt_sequence_length=prompt_sequence_length,
                            noise_seq_len=noise_seq_len,
                            traced=traced,
                            profiler=profiler,
                            profiler_iteration=profiler_iteration,
                        )

            if skip_vae:
                logger.info("skipping VAE decode (warmup mode)")
                output = []
            else:
                logger.info("decoding image...")

                with profiler("vae", profiler_iteration) if profiler else nullcontext():
                    ttnn.synchronize_device(self.vae_device)

                    tt_latents = self._ccl_managers[self.vae_submesh_idx].all_gather_persistent_buffer(
                        tt_latents_step_list[self.vae_submesh_idx],
                        dim=1,
                        mesh_axis=sp_axis,
                        use_hyperparams=True,
                    )

                    torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])

                    # Slice to noise-only tokens (exclude image condition tokens)
                    torch_latents = torch_latents[:, :noise_seq_len]

                    latents_height = self._height // self._vae_scale_factor
                    latents_width = self._width // self._vae_scale_factor

                    # Unpatchify: (B, seq, patch*patch*C) -> (B, H, W, C=16) channel-last
                    torch_latents = self.transformers[0].unpatchify(
                        torch_latents,
                        height=latents_height,
                        width=latents_width,
                    )

                    # Denormalize latents
                    torch_latents = torch_latents / self._latents_scaling + self._latents_shift

                    if self._vae_decoder is None:
                        torch_latents = torch_latents.permute(0, 3, 1, 2).unsqueeze(2)
                        with torch.no_grad():
                            decoded_output = self._torch_vae.decode(torch_latents).sample[:, :, 0]
                    else:
                        self.prepare_vae()

                        with self.mesh_reshape(self.vae_device, self.vae_mesh_shape):
                            tt_latents, logical_h = self._vae_decoder.prepare_input(torch_latents)
                            tt_decoded_output, logical_h = self._vae_decoder.forward(tt_latents, logical_h)
                            decoded_output = self._vae_decoder.postprocess_output(tt_decoded_output, logical_h)

                    image = self._image_processor.postprocess(decoded_output, output_type="pt")
                    assert isinstance(image, torch.Tensor)

                    output = self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))

        return output

    def _step_inner(
        self,
        *,
        latent: ttnn.Tensor,
        prompt: ttnn.Tensor,
        timestep: ttnn.Tensor,
        submesh_index: int,
        spatial_rope_cos: ttnn.Tensor,
        spatial_rope_sin: ttnn.Tensor,
        prompt_rope_cos: ttnn.Tensor,
        prompt_rope_sin: ttnn.Tensor,
        spatial_sequence_length: int,
        prompt_sequence_length: int,
    ) -> ttnn.Tensor:
        return self.transformers[submesh_index].forward(
            spatial=latent,
            prompt=prompt,
            timestep=timestep,
            spatial_rope=(spatial_rope_cos, spatial_rope_sin),
            prompt_rope=(prompt_rope_cos, prompt_rope_sin),
            spatial_sequence_length=spatial_sequence_length,
            prompt_sequence_length=prompt_sequence_length,
        )

    def _step(
        self,
        *,
        do_true_cfg: bool,
        true_cfg_scale: float,
        latents: list[ttnn.Tensor],
        timestep: list[ttnn.Tensor],
        prompt_embeds: list[ttnn.Tensor],
        sigma_difference: list[ttnn.Tensor],
        spatial_rope_cos: list[ttnn.Tensor],
        spatial_rope_sin: list[ttnn.Tensor],
        prompt_rope_cos: list[ttnn.Tensor],
        prompt_rope_sin: list[ttnn.Tensor],
        spatial_sequence_length: int,
        prompt_sequence_length: int,
        noise_seq_len: int,
        traced: bool,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
    ) -> list[ttnn.Tensor]:
        sp_axis = self._parallel_config.sequence_parallel.mesh_axis

        if traced and self._traces is None:
            self._traces = []
            for submesh_id, submesh_device in enumerate(self._submesh_devices):
                timestep_device = timestep[submesh_id].to(submesh_device)
                sigma_difference_device = sigma_difference[submesh_id].to(submesh_device)

                pred = self._step_inner(
                    latent=latents[submesh_id],
                    prompt=prompt_embeds[submesh_id],
                    timestep=timestep_device,
                    spatial_rope_cos=spatial_rope_cos[submesh_id],
                    spatial_rope_sin=spatial_rope_sin[submesh_id],
                    prompt_rope_cos=prompt_rope_cos[submesh_id],
                    prompt_rope_sin=prompt_rope_sin[submesh_id],
                    spatial_sequence_length=spatial_sequence_length,
                    prompt_sequence_length=prompt_sequence_length,
                    submesh_index=submesh_id,
                )

                trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
                pred = self._step_inner(
                    latent=latents[submesh_id],
                    prompt=prompt_embeds[submesh_id],
                    timestep=timestep_device,
                    spatial_rope_cos=spatial_rope_cos[submesh_id],
                    spatial_rope_sin=spatial_rope_sin[submesh_id],
                    prompt_rope_cos=prompt_rope_cos[submesh_id],
                    prompt_rope_sin=prompt_rope_sin[submesh_id],
                    spatial_sequence_length=spatial_sequence_length,
                    prompt_sequence_length=prompt_sequence_length,
                    submesh_index=submesh_id,
                )
                ttnn.end_trace_capture(submesh_device, trace_id, cq_id=0)

                for device in self._submesh_devices:
                    ttnn.synchronize_device(device)

                self._traces.append(
                    PipelineTrace(
                        spatial_input=latents[submesh_id],
                        prompt_input=prompt_embeds[submesh_id],
                        timestep_input=timestep_device,
                        spatial_rope_cos=spatial_rope_cos[submesh_id],
                        spatial_rope_sin=spatial_rope_sin[submesh_id],
                        prompt_rope_cos=prompt_rope_cos[submesh_id],
                        prompt_rope_sin=prompt_rope_sin[submesh_id],
                        latents_output=pred,
                        sigma_difference_input=sigma_difference_device,
                        tid=trace_id,
                    )
                )

        noise_pred_list = []
        if traced:
            for submesh_id, submesh_device in enumerate(self._submesh_devices):
                ttnn.copy_host_to_device_tensor(timestep[submesh_id], self._traces[submesh_id].timestep_input)
                ttnn.copy_host_to_device_tensor(
                    sigma_difference[submesh_id], self._traces[submesh_id].sigma_difference_input
                )
                ttnn.execute_trace(submesh_device, self._traces[submesh_id].tid, cq_id=0, blocking=False)
                noise_pred_list.append(self._traces[submesh_id].latents_output)

            sigma_difference_device = [trace.sigma_difference_input for trace in self._traces]
        else:
            for submesh_id in range(len(self._submesh_devices)):
                noise_pred = self._step_inner(
                    latent=latents[submesh_id],
                    prompt=prompt_embeds[submesh_id],
                    timestep=timestep[submesh_id],
                    spatial_rope_cos=spatial_rope_cos[submesh_id],
                    spatial_rope_sin=spatial_rope_sin[submesh_id],
                    prompt_rope_cos=prompt_rope_cos[submesh_id],
                    prompt_rope_sin=prompt_rope_sin[submesh_id],
                    spatial_sequence_length=spatial_sequence_length,
                    prompt_sequence_length=prompt_sequence_length,
                    submesh_index=submesh_id,
                )
                noise_pred_list.append(noise_pred)

            sigma_difference_device = sigma_difference

        # True CFG: uncond + scale * (cond - uncond), done on-device with ttnn.lerp
        if do_true_cfg and self._parallel_config.cfg_parallel.factor > 1:
            # With cfg_parallel > 1 and SP > 1, gather full sequence for CFG combine
            uncond = tensor.to_torch(
                noise_pred_list[0].cpu(blocking=True),
                mesh_axes=[None, sp_axis, None],
                composer_device=self._submesh_devices[0],
            ).to(torch.float32)
            cond = tensor.to_torch(
                noise_pred_list[1].cpu(blocking=True),
                mesh_axes=[None, sp_axis, None],
                composer_device=self._submesh_devices[1],
            ).to(torch.float32)

            torch_noise_pred = uncond + true_cfg_scale * (cond - uncond)

            for submesh_id, submesh_device in enumerate(self._submesh_devices):
                noise_pred_list[submesh_id] = tensor.from_torch(
                    torch_noise_pred, device=submesh_device, mesh_axes=[None, sp_axis, None]
                )

        for submesh_id, submesh_device in enumerate(self._submesh_devices):
            ttnn.multiply_(noise_pred_list[submesh_id], sigma_difference_device[submesh_id])
            ttnn.add_(latents[submesh_id], noise_pred_list[submesh_id])

        return latents

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1).long()
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    def _encode_prompts(
        self,
        *,
        prompts: list[str],
        negative_prompts: list[str | None],
        images: list | None = None,
        num_images_per_prompt: int,
        do_true_cfg: bool,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(prompts) == len(negative_prompts), "prompts and negative_prompts must have the same length"

        negative_prompts = [x if x is not None else "" for x in negative_prompts]

        if do_true_cfg and self._parallel_config.cfg_parallel.factor > 1:
            all_prompts = negative_prompts + prompts
        else:
            all_prompts = prompts

        img_prompt_prefix = ""
        if images:
            img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
            for idx in range(len(images)):
                img_prompt_prefix += img_prompt_template.format(idx + 1)

        all_prompts = [PROMPT_TEMPLATE_EDIT.format(img_prompt_prefix + e) for e in all_prompts]

        if images:
            return self._encode_prompts_with_images(all_prompts, images, num_images_per_prompt)

        embeds, mask = self._text_encoder.encode(
            all_prompts,
            num_images_per_prompt=num_images_per_prompt,
            sequence_length=512 + PROMPT_DROP_IDX_EDIT,
        )

        embeds[torch.logical_not(mask)] = 0.0
        return embeds[:, PROMPT_DROP_IDX_EDIT:], mask[:, PROMPT_DROP_IDX_EDIT:]

    def _encode_prompts_with_images(
        self,
        formatted_prompts: list[str],
        images: list,
        num_images_per_prompt: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts with image conditioning using the on-device Qwen2.5-VL encoder."""
        result_hidden, result_mask = self._text_encoder.encode_with_images(
            formatted_prompts,
            list(images) * len(formatted_prompts),
            num_images_per_prompt=num_images_per_prompt,
            omit_final_host_gather=True,
        )

        if isinstance(result_hidden, ttnn.Tensor):
            prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(result_hidden)[0])
            encoder_mask = result_mask
            if (
                prompt_embeds.shape[1] == IMAGE_PROMPT_PAD_SEQ_LEN
                and encoder_mask.shape[1] == IMAGE_PROMPT_PAD_SEQ_LEN
                and num_images_per_prompt == 1
            ):
                return prompt_embeds, encoder_mask

            hidden_states = prompt_embeds
            attention_mask = encoder_mask
        else:
            hidden_states = result_hidden
            attention_mask = result_mask

        split_hidden = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden = [e[PROMPT_DROP_IDX_EDIT:] for e in split_hidden]
        attn_masks = [torch.ones(e.size(0), dtype=torch.long) for e in split_hidden]

        natural_max = max(e.size(0) for e in split_hidden)
        if natural_max > IMAGE_PROMPT_PAD_SEQ_LEN:
            msg = (
                f"Image-conditioned prompt encoded to {natural_max} tokens, which exceeds "
                f"IMAGE_PROMPT_PAD_SEQ_LEN={IMAGE_PROMPT_PAD_SEQ_LEN}. Shorten the prompt or "
                f"raise the constant."
            )
            raise ValueError(msg)
        max_seq = IMAGE_PROMPT_PAD_SEQ_LEN
        prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq - u.size(0), u.size(1))]) for u in split_hidden])
        encoder_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq - u.size(0))]) for u in attn_masks])

        prompt_embeds[torch.logical_not(encoder_mask.bool())] = 0.0

        return prompt_embeds, encoder_mask

    def synchronize_devices(self):
        for device in self._submesh_devices:
            ttnn.synchronize_device(device)
