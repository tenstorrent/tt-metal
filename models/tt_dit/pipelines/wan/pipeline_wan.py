# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py

import html
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import List, Optional, Union

import ftfy
import regex as re
import torch
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan
from diffusers.models import WanTransformer3DModel as TorchWanTransformer3DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor
from loguru import logger
from transformers import AutoTokenizer, UMT5EncoderModel

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler

from ...encoders.umt5.model_umt5 import UMT5Config, UMT5Encoder
from ...models.transformers.wan2_2.transformer_wan import WanTransformer3DModel
from ...models.vae.vae_wan2_1 import WanDecoder
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor, VaeHWParallelConfig
from ...parallel.manager import CCLManager
from ...solvers import UniPCSolver
from ...utils import cache, tensor
from ...utils.conv3d import conv3d_blocking_hash
from ...utils.tensor import (
    fast_device_to_host,
    float32_tensor,
    float_to_uint8,
    float_to_unit_range,
    local_device_to_torch,
    typed_tensor_2dshard,
)

_UNSET = object()  # sentinel for "use config default" in create_pipeline

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers.utils import export_to_video
        >>> from diffusers import AutoencoderKLWan, WanPipeline
        >>> from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        >>> # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        >>> model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        >>> flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
        >>> pipe.to("cuda")

        >>> prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        >>> output = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=720,
        ...     width=1280,
        ...     num_frames=81,
        ...     guidance_scale=5.0,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=16)
        ```
"""


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


@dataclass
class TransformerState:
    model: WanTransformer3DModel
    subfolder: str
    torch_model: TorchWanTransformer3DModel
    guidance_scale: float
    prompt_buffer: object = field(default=None)
    negative_prompt_buffer: object = field(default=None)


class WanPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        mesh_device (`ttnn.MeshDevice`):
            The TT mesh device to run inference on.
        parallel_config (`DiTParallelConfig`):
            Parallelism configuration for the transformer.
        vae_parallel_config (`VaeHWParallelConfig`):
            Parallelism configuration for the VAE decoder.
        encoder_parallel_config (`EncoderParallelConfig`):
            Parallelism configuration for the text encoder.
        num_links (`int`):
            Number of links to use for CCL operations.
        checkpoint_name (`str`, *optional*, defaults to `"Wan-AI/Wan2.2-T2V-A14B-Diffusers"`):
            HuggingFace Hub repo ID to load model weights from.
        scheduler (`UniPCMultistepScheduler`, *optional*):
            Scheduler to use for denoising. Defaults to `UniPCMultistepScheduler` loaded from the checkpoint.
        boundary_ratio (`float`, *optional*, defaults to `0.875`):
            Ratio of total timesteps used as the boundary for switching between the two transformers in two-stage
            denoising. `transformer` handles timesteps >= boundary_timestep and `transformer_2` handles timesteps <
            boundary_timestep. If `None`, only `transformer` is used for the entire denoising process.
        expand_timesteps (`bool`, *optional*, defaults to `False`):
            Whether to expand timesteps per-token for image-to-video (Wan2.2 TI2V) conditioning.
        dynamic_load (`bool`, *optional*, defaults to `False`):
            If `True`, model components are loaded/offloaded to device dynamically during inference.
        topology (`ttnn.Topology`, *optional*, defaults to `ttnn.Topology.Linear`):
            Fabric topology to use for CCL operations across devices.
        is_fsdp (`bool`, *optional*, defaults to `True`):
            Whether to use fully-sharded data parallelism for transformer weights.
        model_type (`str`, *optional*, defaults to `"t2v"`):
            Model variant identifier (e.g. `"t2v"` for text-to-video).
        vae_dtype (`ttnn.DataType`, *optional*, defaults to `ttnn.bfloat16`):
            Data type to use for VAE inference.
        vae_use_cache (`bool`, *optional*, defaults to `True`):
            Whether to cache VAE convolution programs across calls.
        sdpa_t_fracture_w_only (`bool`, *optional*, defaults to `False`):
            Whether to fracture SDPA only along the width dimension for temporal attention.
    """

    def __init__(
        self,
        mesh_device,
        parallel_config,
        vae_parallel_config,
        encoder_parallel_config,
        num_links,
        *,
        checkpoint_name: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        scheduler: UniPCMultistepScheduler = None,
        boundary_ratio: Optional[float] = 0.875,
        expand_timesteps: bool = False,  # Wan2.2 ti2v
        dynamic_load=False,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        is_fsdp: bool = True,
        model_type: str = "t2v",
        vae_dtype: ttnn.DataType = ttnn.bfloat16,
        vae_t_chunk_size: int | None = 1,
        sdpa_t_fracture_w_only: bool = False,
        target_height: int = 0,
        target_width: int = 0,
        t_chunk_size: int = 0,
        run_warmup: bool = True,
    ):
        super().__init__()

        self.checkpoint_name = checkpoint_name
        self.model_type = model_type
        self.vae_t_chunk_size = vae_t_chunk_size

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer", trust_remote_code=True)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            checkpoint_name, subfolder="text_encoder", trust_remote_code=True
        )
        self.vae = AutoencoderKLWan.from_pretrained(checkpoint_name, subfolder="vae", trust_remote_code=True)
        self.torch_transformer = TorchWanTransformer3DModel.from_pretrained(
            checkpoint_name, subfolder="transformer", trust_remote_code=True
        )
        self.torch_transformer_2 = TorchWanTransformer3DModel.from_pretrained(
            checkpoint_name, subfolder="transformer_2", trust_remote_code=True
        )

        self.dit_ccl_manager = CCLManager(
            mesh_device=mesh_device,
            num_links=num_links,
            topology=topology,
        )
        self.vae_ccl_manager = CCLManager(
            mesh_device=mesh_device,
            num_links=num_links,
            topology=ttnn.Topology.Linear,  # NOTE: VAE always uses Linear topology. TODO: enable ring if given.
        )

        # See what options we have for topology. We should consider reusing CCL managers
        self.encoder_ccl_manager = self.vae_ccl_manager

        self.is_fsdp = is_fsdp
        self.parallel_config = parallel_config
        self.vae_parallel_config = vae_parallel_config
        self.encoder_parallel_config = encoder_parallel_config
        self.mesh_device = mesh_device
        self.dynamic_load = dynamic_load

        # Load TT text encoder
        umt5_config = UMT5Config(
            vocab_size=self.text_encoder.config.vocab_size,
            embed_dim=self.text_encoder.config.d_model,
            ff_dim=self.text_encoder.config.d_ff,
            kv_dim=self.text_encoder.config.d_kv,
            num_heads=self.text_encoder.config.num_heads,
            num_hidden_layers=self.text_encoder.config.num_layers,
            max_prompt_length=512,  # TODO: Consider removing
            layer_norm_eps=self.text_encoder.config.layer_norm_epsilon,
            relative_attention_num_buckets=self.text_encoder.config.relative_attention_num_buckets,
            relative_attention_max_distance=self.text_encoder.config.relative_attention_max_distance,
        )

        self.tt_umt5_encoder = UMT5Encoder(
            config=umt5_config,
            mesh_device=self.mesh_device,
            ccl_manager=self.encoder_ccl_manager,
            parallel_config=self.encoder_parallel_config,
        )

        self.transformer = WanTransformer3DModel(
            patch_size=self.torch_transformer.config.patch_size,
            num_heads=self.torch_transformer.config.num_attention_heads,
            dim=self.torch_transformer.config.num_attention_heads * self.torch_transformer.config.attention_head_dim,
            in_channels=self.torch_transformer.config.in_channels,
            out_channels=self.torch_transformer.config.out_channels,
            text_dim=self.torch_transformer.config.text_dim,
            freq_dim=self.torch_transformer.config.freq_dim,
            ffn_dim=self.torch_transformer.config.ffn_dim,
            cross_attn_norm=self.torch_transformer.config.cross_attn_norm,
            eps=self.torch_transformer.config.eps,
            rope_max_seq_len=self.torch_transformer.config.rope_max_seq_len,
            mesh_device=self.mesh_device,
            ccl_manager=self.dit_ccl_manager,
            parallel_config=self.parallel_config,
            is_fsdp=self.is_fsdp,
            model_type=self.model_type,
        )

        self.transformer_2 = WanTransformer3DModel(
            patch_size=self.torch_transformer_2.config.patch_size,
            num_heads=self.torch_transformer_2.config.num_attention_heads,
            dim=self.torch_transformer_2.config.num_attention_heads
            * self.torch_transformer_2.config.attention_head_dim,
            in_channels=self.torch_transformer_2.config.in_channels,
            out_channels=self.torch_transformer_2.config.out_channels,
            text_dim=self.torch_transformer_2.config.text_dim,
            freq_dim=self.torch_transformer_2.config.freq_dim,
            ffn_dim=self.torch_transformer_2.config.ffn_dim,
            cross_attn_norm=self.torch_transformer_2.config.cross_attn_norm,
            eps=self.torch_transformer_2.config.eps,
            rope_max_seq_len=self.torch_transformer_2.config.rope_max_seq_len,
            mesh_device=self.mesh_device,
            ccl_manager=self.dit_ccl_manager,
            parallel_config=self.parallel_config,
            is_fsdp=self.is_fsdp,
            model_type=self.model_type,
        )

        self.tt_vae = WanDecoder(
            base_dim=self.vae.config.base_dim,
            z_dim=self.vae.config.z_dim,
            dim_mult=self.vae.config.dim_mult,
            num_res_blocks=self.vae.config.num_res_blocks,
            attn_scales=self.vae.config.attn_scales,
            temperal_downsample=self.vae.config.temperal_downsample,
            out_channels=self.vae.config.out_channels,
            is_residual=self.vae.config.is_residual,
            mesh_device=self.mesh_device,
            ccl_manager=self.vae_ccl_manager,
            parallel_config=self.vae_parallel_config,
            dtype=vae_dtype,
            sdpa_t_fracture_w_only=sdpa_t_fracture_w_only,
            target_height=target_height,
            target_width=target_width,
            t_chunk_size=t_chunk_size,
            cached=(vae_t_chunk_size is not None),
        )

        self.transformer_states = [
            TransformerState(self.transformer, "transformer", self.torch_transformer, guidance_scale=4.0),
            TransformerState(self.transformer_2, "transformer_2", self.torch_transformer_2, guidance_scale=3.0),
        ]

        scheduler = scheduler or UniPCMultistepScheduler.from_pretrained(
            checkpoint_name, subfolder="scheduler", flow_shift=12.0
        )
        self._solver = UniPCSolver(scheduler=scheduler)

        if self.dynamic_load:
            # setup models that cannot be loaded together with the corresponding model.
            # The module loading utility will take care of the necessary unloading.
            if ttnn.device.is_blackhole():
                self.transformer.register_coresident_exclusions(self.transformer_2)
                self.transformer_2.register_coresident_exclusions(self.transformer)
            else:
                # WH T3K has tighter DRAM — include VAE in the unload chain so
                # transformers and VAE never coexist in DRAM across pipeline runs.
                self.transformer.register_coresident_exclusions(self.transformer_2, self.tt_vae)
                self.transformer_2.register_coresident_exclusions(self.transformer, self.tt_vae)
                self.tt_vae.register_coresident_exclusions(self.transformer, self.transformer_2)

        # Cache warmup: Load in reverse order of use to ensure the earliest required models stay loaded before call.
        self._prepare_transformer(1)
        self._prepare_transformer(0)
        self._prepare_text_encoder()
        self._prepare_vae()

        self.register_to_config(boundary_ratio=boundary_ratio)
        self.register_to_config(expand_timesteps=expand_timesteps)
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # Precompute VAE latent normalization constants (avoids recreating every call)
        self._vae_latents_mean = torch.tensor(self.vae.config.latents_mean, dtype=self.vae.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        self._vae_latents_std = torch.tensor(self.vae.config.latents_std, dtype=self.vae.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )

        # persistent latent buffers to enable safe tracing.
        self.latent_buffer = None
        self.condition_buffer = None

        # TODO: Reset buffers for change in resolution. Also reinitialize trace
        if run_warmup:
            self.warmup_buffers(height=target_height, width=target_width)

    def prepare_text_conditioning(self, tt_model, prompt_embeds, buffer, traced=False):
        prompt_1BLP = tt_model.prepare_text_conditioning(prompt_embeds)
        if buffer is None or not traced:
            buffer = prompt_1BLP
        else:
            ttnn.copy(prompt_1BLP, buffer)
        return buffer

    def warmup_buffers(self, height, width, image_prompt=None):
        self.run_single_prompt(
            prompt="warmup",
            image_prompt=image_prompt,
            height=height,
            width=width,
            num_frames=81,
            num_inference_steps=2,
        )

    @staticmethod
    def create_pipeline(
        mesh_device,
        *,
        checkpoint_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        scheduler=None,
        sp_axis=None,
        tp_axis=None,
        num_links=None,
        dynamic_load=None,
        topology=None,
        is_fsdp=None,
        pipeline_class=None,
        vae_t_chunk_size=_UNSET,
        sdpa_t_fracture_w_only=None,
        target_height: int = 0,
        target_width: int = 0,
        num_frames: int = 81,
    ):
        device_configs = {}
        if ttnn.device.is_blackhole():
            device_configs[(1, 4)] = {
                "sp_axis": 0,
                "tp_axis": 1,
                "num_links": 2,
                "dynamic_load": False,
                "topology": ttnn.Topology.Linear,
                "is_fsdp": True,
            }
            device_configs[(2, 2)] = device_configs[(1, 4)]
            device_configs[(2, 4)] = {
                "sp_axis": 1,
                "tp_axis": 0,
                "num_links": 2,
                "dynamic_load": True,
                "topology": ttnn.Topology.Linear,
                "is_fsdp": False,
                "vae_t_chunk_size": 7,
            }
            device_configs[(4, 8)] = {
                "sp_axis": 1,
                "tp_axis": 0,
                "num_links": 2,
                "dynamic_load": False,
                "topology": ttnn.Topology.Ring,
                "is_fsdp": False,
                "vae_t_chunk_size": None,  # full-T
            }
            device_configs[(4, 32)] = {
                "sp_axis": 1,
                "tp_axis": 0,
                "num_links": 2,
                "dynamic_load": False,
                "topology": ttnn.Topology.Ring,
                "is_fsdp": False,
                "vae_t_chunk_size": None,
                "sdpa_t_fracture_w_only": True,
            }
            config = device_configs[tuple(mesh_device.shape)]
        else:
            device_configs[(2, 4)] = {
                "sp_axis": 0,
                "tp_axis": 1,
                "num_links": 1,
                "dynamic_load": True,
                "topology": ttnn.Topology.Linear,
                "is_fsdp": True,
            }
            device_configs[(4, 8)] = {
                "sp_axis": 1,
                "tp_axis": 0,
                "num_links": 4,
                "dynamic_load": False,
                "topology": ttnn.Topology.Ring,
                "is_fsdp": True,
            }

            config = device_configs[tuple(mesh_device.shape)]

        sp_axis = config["sp_axis"] if sp_axis is None else sp_axis
        tp_axis = config["tp_axis"] if tp_axis is None else tp_axis
        if vae_t_chunk_size is _UNSET:
            vae_t_chunk_size = config.get("vae_t_chunk_size", 1)
        full_latent_T = (num_frames - 1) // 4 + 1
        decoder_t_chunk_size = full_latent_T if vae_t_chunk_size is None else vae_t_chunk_size

        h_factor = tuple(mesh_device.shape)[tp_axis]
        w_factor = tuple(mesh_device.shape)[sp_axis]

        parallel_config = DiTParallelConfig(
            tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=h_factor),
            sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=w_factor),
            cfg_parallel=None,
        )
        vae_parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(
                factor=h_factor,
                mesh_axis=tp_axis,
            ),
            width_parallel=ParallelFactor(
                factor=w_factor,
                mesh_axis=sp_axis,
            ),
        )
        encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=h_factor, mesh_axis=tp_axis)
        )
        pipeline_class_ = pipeline_class or WanPipeline
        return pipeline_class_(
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            vae_parallel_config=vae_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            num_links=num_links or config["num_links"],
            boundary_ratio=0.875,
            scheduler=scheduler,
            dynamic_load=dynamic_load if dynamic_load is not None else config["dynamic_load"],
            topology=topology or config["topology"],
            is_fsdp=is_fsdp if is_fsdp is not None else config["is_fsdp"],
            checkpoint_name=checkpoint_name,
            vae_t_chunk_size=vae_t_chunk_size,
            sdpa_t_fracture_w_only=sdpa_t_fracture_w_only
            if sdpa_t_fracture_w_only is not None
            else config.get("sdpa_t_fracture_w_only", False),
            target_height=target_height,
            target_width=target_width,
            t_chunk_size=decoder_t_chunk_size,
        )

    def _prepare_text_encoder(self):
        cache.load_model(
            self.tt_umt5_encoder,
            model_name=os.path.basename(self.checkpoint_name),
            subfolder="text_encoder",
            parallel_config=self.encoder_parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            get_torch_state_dict=lambda: self.text_encoder.state_dict(),
        )

    def _prepare_transformer(self, idx: int):
        state = self.transformer_states[idx]
        cache.load_model(
            state.model,
            model_name=os.path.basename(self.checkpoint_name),
            subfolder=state.subfolder,
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            is_fsdp=self.is_fsdp,
            get_torch_state_dict=lambda: state.torch_model.state_dict(),
        )

    def _prepare_vae(self):
        blocking_key = conv3d_blocking_hash(self.tt_vae)
        subfolder = f"vae_{blocking_key}" if blocking_key else "vae"
        cache.load_model(
            self.tt_vae,
            model_name=os.path.basename(self.checkpoint_name),
            subfolder=subfolder,
            parallel_config=self.vae_parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            get_torch_state_dict=lambda: self.vae.state_dict(),
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        traced: bool = False,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        # NOTE: while the reference impl does not pad to max_sequence_length, for some reason this seems to be necessary for correctness in this pipeline.
        # TODO: investigate
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        # Shard on batch dimension. On non TP axis
        dims = [None, None]
        DP_axis = 1 - self.parallel_config.tensor_parallel.mesh_axis
        dims[DP_axis] = 0
        mesh_mapper = ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=dims)
        tt_prompt = ttnn.from_torch(
            text_input_ids,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device if not traced else None,
            mesh_mapper=mesh_mapper,
        )

        tt_mask = ttnn.from_torch(
            mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device if not traced else None,
            mesh_mapper=mesh_mapper,
        )

        prompt_embeds = self.tt_umt5_encoder(tt_prompt, attention_mask=tt_mask, zero_masking=True, traced=traced)[-1]

        prompt_embeds = self.encoder_ccl_manager.all_gather(
            prompt_embeds, dim=0, mesh_axis=DP_axis, use_hyperparams=True
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = ttnn.repeat(prompt_embeds, (1, num_videos_per_prompt, 1))
        prompt_embeds_1BLP = ttnn.view(prompt_embeds, (1, batch_size * num_videos_per_prompt, seq_len, -1))
        return prompt_embeds_1BLP

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        traced: bool = True,
    ):
        r"""
        Batch encodes the prompt and negative prompt into text encoder hidden states..

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Setup batching variables
        all_input_prompts = []
        pos_prompt_end_idx = 0
        neg_prompt_end_idx = 0

        if prompt_embeds is None:
            all_input_prompts += prompt
            pos_prompt_end_idx = batch_size * num_videos_per_prompt

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            all_input_prompts += negative_prompt
            neg_prompt_end_idx = pos_prompt_end_idx + batch_size * num_videos_per_prompt

        # Add data to pad for size of device on mesh axis to ensure proper shadding on batch dimension.
        total_prompts = len(all_input_prompts)
        num_devices = self.mesh_device.shape[1 - self.parallel_config.tensor_parallel.mesh_axis]

        # Pad batch list of prompts to ensure proper sharding on batch dimension.
        all_input_prompts += [" "] * ((num_devices - (total_prompts % num_devices)) % num_devices)
        all_prompt_embeds = self._get_t5_prompt_embeds(
            prompt=all_input_prompts,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            traced=traced,
        )

        # When CFG is enabled, we should be able to leave the shards on device.
        prompt_embeds = all_prompt_embeds[:, :pos_prompt_end_idx] if pos_prompt_end_idx > 0 else prompt_embeds
        negative_prompt_embeds = (
            all_prompt_embeds[:, pos_prompt_end_idx:neg_prompt_end_idx]
            if neg_prompt_end_idx > 0
            else negative_prompt_embeds
        )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        guidance_scale_2=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if self.config.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None.")

    def get_model_input(self, latents, cond_latents):
        """
        Adapter function to enable I2V. For base T2V, just return the latents.
        """
        if latents.dtype == ttnn.float32:
            latents = ttnn.typecast(latents, ttnn.bfloat16)
        return latents

    def prepare_latents(
        self,
        batch_size: int,
        image_prompt=None,  # unused in T2V
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype), None

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )

        latents = torch.randn(shape, dtype=torch.float32, device=torch.device(device))
        return latents, None

    @property
    def do_classifier_free_guidance(self):
        return self.transformer_states[0].guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[
            str, List[str]
        ] = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        image_prompt=None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.0,
        guidance_scale_2: Optional[float] = 3.0,
        num_videos_per_prompt: Optional[int] = 1,
        seed: Optional[int] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "uint8",
        return_dict: bool = True,
        max_sequence_length: int = 512,
        traced: bool = False,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during image generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            seed (`int`, *optional*):
                A random generator seed to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `seed`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self.transformer_states[0].guidance_scale = guidance_scale
        self.transformer_states[1].guidance_scale = guidance_scale_2

        # device = self._execution_device
        device = "cpu"

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        with profiler("encoder", profiler_iteration) if profiler else nullcontext():
            self._prepare_text_encoder()
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                traced=traced,
            )

        # 4. Prepare schedule
        self._solver.set_schedule(num_inference_steps, device=device)
        timesteps = self._solver.timesteps

        # 5. Prepare latent variables
        if seed is not None:
            torch.manual_seed(seed)

        with profiler("prepare_latents", profiler_iteration) if profiler else nullcontext():
            latents, cond_latents = self.prepare_latents(
                batch_size=batch_size * num_videos_per_prompt,
                image_prompt=image_prompt,
                num_channels_latents=self.vae.config.z_dim,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                latents=latents,
            )

        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # 6. Denoising loop
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self._solver.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = -1  # Always use transformer (no transformer_2)

        if profiler:
            profiler.start("denoising", profiler_iteration)

        permuted_latent_tt = None
        rope_args = None

        latent_frames, latent_height, latent_width = latents.shape[2], latents.shape[3], latents.shape[4]
        prepared_prompts = [False, False]

        sp_axis = self.transformer_states[0].model.parallel_config.sequence_parallel.mesh_axis
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                warmup_t2 = i == 1 and len(timesteps) == 2  # Ensure transformer_2 is also warmed up

                # 0=> wan2.1 or high-noise stage in wan2.2 (transformer) | 1=> low-noise stage in wan2.2 (transformer_2)
                transformer_idx = 0 if (t >= boundary_timestep) and not warmup_t2 else 1
                self._prepare_transformer(transformer_idx)
                ts = self.transformer_states[transformer_idx]
                if not prepared_prompts[transformer_idx]:
                    # Prepare the text conditioning in an optional persistent buffer depending on traced
                    ts.prompt_buffer = self.prepare_text_conditioning(ts.model, prompt_embeds, ts.prompt_buffer, traced)
                    ts.negative_prompt_buffer = self.prepare_text_conditioning(
                        ts.model, negative_prompt_embeds, ts.negative_prompt_buffer, traced
                    )
                    prepared_prompts[transformer_idx] = True

                if permuted_latent_tt is None:
                    # First iteration, preprocess spatial input and prepare rope features
                    permuted_latent, patchified_seqlen = ts.model.preprocess_spatial_input_host(latents)

                    if cond_latents is not None:
                        cond_latents, _ = ts.model.preprocess_spatial_input_host(cond_latents)
                        cond_latents = tensor.from_torch(
                            cond_latents,
                            device=self.mesh_device,
                            mesh_axes=[None, None, sp_axis, None],
                            dtype=ttnn.bfloat16,
                        )
                        if self.condition_buffer is None:
                            self.condition_buffer = cond_latents
                        else:
                            ttnn.copy(cond_latents, self.condition_buffer)

                    rope_cos_1HND, rope_sin_1HND, trans_mat = ts.model.get_rope_features(latents)
                    rope_args = {
                        "rope_cos_1HND": rope_cos_1HND,
                        "rope_sin_1HND": rope_sin_1HND,
                        "trans_mat": trans_mat,
                    }

                    permuted_latent_tt = tensor.from_torch(
                        permuted_latent,
                        device=self.mesh_device,
                        mesh_axes=[None, None, sp_axis, None],
                        dtype=ts.model.output_dtype,
                    )

                # setup/update latent and condition buffers
                if self.latent_buffer is None:
                    self.latent_buffer = permuted_latent_tt
                else:
                    ttnn.copy(permuted_latent_tt, self.latent_buffer)

                if self.config.expand_timesteps:
                    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                permuted_model_input = self.get_model_input(self.latent_buffer, self.condition_buffer)

                assert timestep.ndim == 1, "Wan2.2-T2V/I2V requires a 1D timestep tensor"
                timestep = float32_tensor(
                    timestep.unsqueeze(1).unsqueeze(1).unsqueeze(1), device=(None if traced else self.mesh_device)
                )

                permuted_noise_pred_tt = ts.model.combined_step(
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    spatial_1BNI=permuted_model_input,
                    prompt_1BLP=ts.prompt_buffer,
                    negative_prompt_1BLP=ts.negative_prompt_buffer,
                    N=patchified_seqlen,
                    timestep=timestep,
                    **rope_args,
                    guidance_scale=ts.guidance_scale,
                    traced=traced,
                    gather_output=False,
                )

                permuted_latent_tt = self._solver.step(
                    step=i,
                    latent=self.latent_buffer,
                    velocity_pred=permuted_noise_pred_tt,
                )

                progress_bar.update()

        self._current_timestep = None

        permuted_latent_tt = ts.model.ccl_manager.all_gather_persistent_buffer(
            permuted_latent_tt, dim=2, mesh_axis=sp_axis
        )
        permuted_latent = local_device_to_torch(permuted_latent_tt)

        # Postprocess spatial output
        latents = ts.model.postprocess_spatial_output_host(
            permuted_latent, F=latent_frames, H=latent_height, W=latent_width, N=patchified_seqlen
        )

        if profiler:
            profiler.end("denoising", profiler_iteration)
            profiler.start("vae", profiler_iteration)

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents = latents * self._vae_latents_std + self._vae_latents_mean

            tt_latents_BTHWC, logical_h = self.tt_vae.prepare_input(latents)

            tt_latents_BTHWC = typed_tensor_2dshard(
                tt_latents_BTHWC,
                self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                shard_mapping={
                    self.vae_parallel_config.height_parallel.mesh_axis: 2,
                    self.vae_parallel_config.width_parallel.mesh_axis: 3,
                },
                dtype=self.tt_vae.dtype,
            )
            self._prepare_vae()
            tt_video_BCTHW, new_logical_h = self.tt_vae(tt_latents_BTHWC, logical_h, t_chunk_size=self.vae_t_chunk_size)

            concat_dims = [None, None]
            concat_dims[self.vae_parallel_config.height_parallel.mesh_axis] = 3
            concat_dims[self.vae_parallel_config.width_parallel.mesh_axis] = 4
            d2h_permute = (0, 2, 3, 4, 1) if output_type in ("np", "uint8") else None

            if output_type == "uint8":
                pre_fn = float_to_uint8
            elif output_type == "np":
                pre_fn = float_to_unit_range
            else:
                pre_fn = None

            video_torch = fast_device_to_host(
                tt_video_BCTHW,
                self.mesh_device,
                concat_dims,
                ccl_manager=self.vae_ccl_manager,
                pre_transfer_fn=pre_fn,
                permute=d2h_permute,
            )

            if d2h_permute is not None:
                # Output is (B, T, H, W, C) — trim height in dim 2.
                video_torch = video_torch[:, :, :new_logical_h, :, :]
            else:
                # Output is (B, C, T, H, W) — trim height in dim 3.
                video_torch = video_torch[:, :, :, :new_logical_h, :]

            if output_type == "uint8":
                video = video_torch.numpy()
            elif output_type == "np":
                video = video_torch.float().numpy()
            else:
                video = self.video_processor.postprocess_video(video_torch, output_type=output_type)
        else:
            video = latents

        if profiler:
            profiler.end("vae", profiler_iteration)

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)

    def run_single_prompt(self, *args, **kwargs):
        return self.__call__(*args, **kwargs).frames

    def synchronize_devices(self):
        ttnn.synchronize_device(self.mesh_device)

    def release_traces(self):
        for model in (self.transformer, self.transformer_2):
            tracer = WanTransformer3DModel.combined_step._tracers.get(model)
            if tracer is not None:
                tracer.release_trace()
