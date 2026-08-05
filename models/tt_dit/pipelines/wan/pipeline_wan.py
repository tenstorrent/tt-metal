# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import tqdm
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.wan2_2.transformer_wan import WanCheckpoint, WanTransformer3DModel
from models.tt_dit.models.vae.vae_wan2_1 import WanVAEDecoderAdapter
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.events import DenoiseStep, PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin
from models.tt_dit.pipelines.wan.text_encoder import TextEncoder
from models.tt_dit.solvers import UniPCSolver, UniPCVariant
from models.tt_dit.utils import tensor
from models.tt_dit.utils.tensor import float32_tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

_UNSET = object()  # sentinel for "use config default" in WanPipelineConfig.default

_DEFAULT_CHECKPOINT = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

_PRESETS_WH: dict[tuple[int, ...], dict] = {
    (2, 4): {
        "sp_axis": 0,
        "tp_axis": 1,
        "num_links": 1,
        "dynamic_load": True,
        "topology": ttnn.Topology.Linear,
        "is_fsdp": True,
    },
    (4, 8): {
        "sp_axis": 1,
        "tp_axis": 0,
        "num_links": 4,
        "dynamic_load": False,
        "topology": ttnn.Topology.Ring,
        "is_fsdp": True,
    },
}

_PRESETS_BH: dict[tuple[int, ...], dict] = {
    (1, 4): {
        "sp_axis": 0,
        "tp_axis": 1,
        "num_links": 2,
        "dynamic_load": False,
        "topology": ttnn.Topology.Linear,
        "is_fsdp": True,
    },
    (2, 2): {
        "sp_axis": 0,
        "tp_axis": 1,
        "num_links": 2,
        "dynamic_load": False,
        "topology": ttnn.Topology.Linear,
        "is_fsdp": True,
    },
    (2, 4): {
        "sp_axis": 1,
        "tp_axis": 0,
        "num_links": 2,
        "dynamic_load": True,
        "topology": ttnn.Topology.Linear,
        "is_fsdp": False,
        "vae_t_chunk_size": 7,
    },
    (4, 8): {
        "sp_axis": 1,
        "tp_axis": 0,
        "num_links": 2,
        "dynamic_load": False,
        "topology": ttnn.Topology.Ring,
        "is_fsdp": False,
        "vae_t_chunk_size": None,  # full-T
    },
    (4, 32): {
        "sp_axis": 1,
        "tp_axis": 0,
        "num_links": 2,
        "dynamic_load": False,
        "topology": ttnn.Topology.Ring,
        "is_fsdp": False,
        "vae_t_chunk_size": None,
        "sdpa_t_fracture_w_only": True,
    },
}


@dataclass(frozen=True, kw_only=True)
class WanPipelineConfig:
    topology: ttnn.Topology
    num_links: int

    dit_parallel_config: DiTParallelConfig
    encoder_parallel_config: EncoderParallelConfig
    vae_parallel_config: VaeHWParallelConfig

    boundary_ratio: float | None
    expand_timesteps: bool
    dynamic_load: bool
    is_fsdp: bool
    model_type: str
    vae_dtype: ttnn.DataType
    vae_t_chunk_size: int | None
    sdpa_t_fracture_w_only: bool

    height: int
    width: int
    num_frames: int
    cfg_enabled: bool
    max_sequence_length: int

    checkpoint_name: str

    @classmethod
    def default(
        cls,
        *,
        mesh_shape: ttnn.MeshShape,
        topology: ttnn.Topology | None = None,
        num_links: int | None = None,
        dit_parallel_config: DiTParallelConfig | None = None,
        encoder_parallel_config: EncoderParallelConfig | None = None,
        vae_parallel_config: VaeHWParallelConfig | None = None,
        boundary_ratio: float | None = 0.875,
        expand_timesteps: bool = False,
        dynamic_load: bool | None = None,
        is_fsdp: bool | None = None,
        model_type: str = "t2v",
        vae_dtype: ttnn.DataType = ttnn.bfloat16,
        vae_t_chunk_size: object = _UNSET,
        sdpa_t_fracture_w_only: bool | None = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        cfg_enabled: bool = True,
        max_sequence_length: int = 512,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> WanPipelineConfig:
        preset_dict = _PRESETS_BH if ttnn.device.is_blackhole() else _PRESETS_WH
        preset = preset_dict.get(tuple(mesh_shape), {})

        if dit_parallel_config is None or vae_parallel_config is None or encoder_parallel_config is None:
            sp_axis = preset["sp_axis"]
            tp_axis = preset["tp_axis"]
            h_factor = tuple(mesh_shape)[tp_axis]
            w_factor = tuple(mesh_shape)[sp_axis]
            if dit_parallel_config is None:
                dit_parallel_config = DiTParallelConfig.from_tuples(
                    cfg=(1, 0), sp=(w_factor, sp_axis), tp=(h_factor, tp_axis)
                )
            if vae_parallel_config is None:
                vae_parallel_config = VaeHWParallelConfig.from_tuples(
                    height=(h_factor, tp_axis), width=(w_factor, sp_axis)
                )
            if encoder_parallel_config is None:
                encoder_parallel_config = EncoderParallelConfig.from_tuple((h_factor, tp_axis))

        if vae_t_chunk_size is _UNSET:
            vae_t_chunk_size = preset.get("vae_t_chunk_size", 1)

        return cls(
            topology=topology if topology is not None else preset["topology"],
            num_links=num_links if num_links is not None else preset["num_links"],
            dit_parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel_config=vae_parallel_config,
            boundary_ratio=boundary_ratio,
            expand_timesteps=expand_timesteps,
            dynamic_load=dynamic_load if dynamic_load is not None else preset["dynamic_load"],
            is_fsdp=is_fsdp if is_fsdp is not None else preset["is_fsdp"],
            model_type=model_type,
            vae_dtype=vae_dtype,
            vae_t_chunk_size=vae_t_chunk_size,
            sdpa_t_fracture_w_only=(
                sdpa_t_fracture_w_only
                if sdpa_t_fracture_w_only is not None
                else preset.get("sdpa_t_fracture_w_only", False)
            ),
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_enabled=cfg_enabled,
            max_sequence_length=max_sequence_length,
            checkpoint_name=checkpoint_name,
        )


@dataclass
class TransformerState:
    model: WanTransformer3DModel
    checkpoint: WanCheckpoint
    guidance_scale: float
    prompt_buffer: object = field(default=None)
    negative_prompt_buffer: object = field(default=None)


class WanPipeline(PipelineAPIMixin):
    r"""Pipeline for text-to-video generation using Wan.

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

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        cfg_enabled: bool = True,
        max_sequence_length: int = 512,
        pipeline_class: type[WanPipeline] | None = None,
    ) -> WanPipeline:
        config = WanPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            checkpoint_name=checkpoint_name,
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_enabled=cfg_enabled,
            max_sequence_length=max_sequence_length,
        )
        pipeline_class_ = pipeline_class or cls
        return pipeline_class_(device=mesh_device, config=config)

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        config: WanPipelineConfig,
        run_warmup: bool = True,
        lora_enabled: bool = False,
    ) -> None:
        super().__init__()

        self.checkpoint_name = config.checkpoint_name
        self.model_type = config.model_type
        self._cfg_enabled = config.cfg_enabled
        self._height = config.height
        self._width = config.width
        self._num_frames = config.num_frames

        self._checkpoint = WanCheckpoint(config.checkpoint_name, subfolder="transformer")
        self._checkpoint_2 = WanCheckpoint(config.checkpoint_name, subfolder="transformer_2")

        self.dit_ccl_manager = CCLManager(
            mesh_device=device,
            num_links=config.num_links,
            topology=config.topology,
        )
        self.vae_ccl_manager = CCLManager(
            mesh_device=device,
            num_links=config.num_links,
            topology=ttnn.Topology.Linear,  # NOTE: VAE always uses Linear topology. TODO: enable ring if given.
        )

        # See what options we have for topology. We should consider reusing CCL managers
        self.encoder_ccl_manager = self.vae_ccl_manager

        self.is_fsdp = config.is_fsdp
        self.parallel_config = config.dit_parallel_config
        self.vae_parallel_config = config.vae_parallel_config
        self.encoder_parallel_config = config.encoder_parallel_config
        self.mesh_device = device
        self.dynamic_load = config.dynamic_load

        self._text_encoder = TextEncoder(
            checkpoint_name=config.checkpoint_name,
            device=self.mesh_device,
            ccl_manager=self.encoder_ccl_manager,
            encoder_parallel_config=self.encoder_parallel_config,
            dit_parallel_config=self.parallel_config,
            max_sequence_length=config.max_sequence_length,
        )

        self.lora_enabled = lora_enabled

        self.transformer = self._checkpoint.build(
            ccl_manager=self.dit_ccl_manager,
            parallel_config=self.parallel_config,
            is_fsdp=self.is_fsdp,
            model_type=self.model_type,
            lora_enabled=lora_enabled,
        )

        self.transformer_2 = self._checkpoint_2.build(
            ccl_manager=self.dit_ccl_manager,
            parallel_config=self.parallel_config,
            is_fsdp=self.is_fsdp,
            model_type=self.model_type,
            lora_enabled=lora_enabled,
        )

        self._vae = WanVAEDecoderAdapter(
            checkpoint_name=config.checkpoint_name,
            parallel_config=self.vae_parallel_config,
            ccl_manager=self.vae_ccl_manager,
            height=config.height,
            width=config.width,
            num_frames=config.num_frames,
            vae_t_chunk_size=config.vae_t_chunk_size,
            vae_dtype=config.vae_dtype,
            sdpa_t_fracture_w_only=config.sdpa_t_fracture_w_only,
        )

        self.transformer_states = [
            TransformerState(self.transformer, self._checkpoint, guidance_scale=4.0),
            TransformerState(self.transformer_2, self._checkpoint_2, guidance_scale=3.0),
        ]

        self._scheduler = UniPCMultistepScheduler.from_pretrained(
            self.checkpoint_name, subfolder="scheduler", flow_shift=12.0
        )
        # Construction-time default, restored whenever a call omits flow_shift so a
        # per-request value never persists into a later request (see __call__).
        self._default_flow_shift = self._scheduler.config.flow_shift
        self._solver = UniPCSolver(
            order=self._scheduler.config.solver_order,
            variant=UniPCVariant(self._scheduler.config.solver_type),
        )

        # persistent latent buffers to enable safe tracing.
        self.latent_buffer = None
        self.condition_buffer = None

        if self.dynamic_load:
            # setup models that cannot be loaded together with the corresponding model.
            # The module loading utility will take care of the necessary unloading.
            if ttnn.device.is_blackhole():
                self.transformer.register_coresident_exclusions(self.transformer_2)
                self.transformer_2.register_coresident_exclusions(self.transformer)
            else:
                # WH T3K has tighter DRAM — include VAE in the unload chain so
                # transformers and VAE never coexist in DRAM across pipeline runs.
                self.transformer.register_coresident_exclusions(self.transformer_2, self._vae.decoder)
                self.transformer_2.register_coresident_exclusions(self.transformer, self._vae.decoder)
                self._vae.decoder.register_coresident_exclusions(self.transformer, self.transformer_2)

        # Cache warmup: Load in reverse order of use to ensure the earliest required models stay loaded before call.
        self._prepare_transformer(1)
        self._prepare_transformer(0)
        self._prepare_text_encoder()
        self._vae.reload_weights()

        self._boundary_ratio = config.boundary_ratio
        self._expand_timesteps = config.expand_timesteps
        self.vae_scale_factor_temporal = self._vae.config.scale_factor_temporal
        self.vae_scale_factor_spatial = self._vae.config.scale_factor_spatial
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # TODO: Reset buffers for change in resolution. Also reinitialize trace
        if run_warmup:
            logger.info("Pipeline allocation run...")
            self(
                prompts=["warmup"],
                num_inference_steps=2,
                guidance_scale=2 if config.cfg_enabled else 1,
                guidance_scale_2=2 if config.cfg_enabled else 1,
            )

    def prepare_text_conditioning(self, tt_model, prompt_embeds, buffer, traced=False):
        prompt_1BLP = tt_model.prepare_text_conditioning(prompt_embeds)
        if buffer is None or not traced:
            buffer = prompt_1BLP
        else:
            ttnn.copy(prompt_1BLP, buffer)
        return buffer

    def _prepare_text_encoder(self) -> None:
        self._text_encoder.prepare()

    def _prepare_transformer(self, idx: int) -> None:
        state = self.transformer_states[idx]
        state.checkpoint.load(
            state.model,
            mesh_device=self.mesh_device,
            parallel_config=self.parallel_config,
            is_fsdp=self.is_fsdp,
        )

    def _step(
        self,
        *,
        step: int,
        t: torch.Tensor,
        ts: TransformerState,
        permuted_latent_tt: ttnn.Tensor,
        mask: torch.Tensor,
        cond_latents: ttnn.Tensor | None,
        rope_args: dict,
        latents_sequence_length: int,
        latents_batch_size: int,
        cfg_enabled: bool,
        traced: bool,
    ) -> ttnn.Tensor:
        if self._expand_timesteps:
            # seq_len: num_latent_frames * latent_height//2 * latent_width//2
            temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
            # batch_size, seq_len
            timestep = temp_ts.unsqueeze(0).expand(latents_batch_size, -1)
        else:
            timestep = t.expand(latents_batch_size)

        permuted_model_input = self.get_model_input(permuted_latent_tt, cond_latents)

        assert timestep.ndim == 1, "Wan2.2-T2V/I2V requires a 1D timestep tensor"
        timestep = float32_tensor(
            timestep.unsqueeze(1).unsqueeze(1).unsqueeze(1), device=(None if traced else self.mesh_device)
        )

        # guidance_scale is passed as a 1-element device tensor (broadcast via ttnn.lerp's
        # tensor-weight overload) so it can be updated in place between traced executions,
        # mirroring how `timestep` above is threaded through the captured trace.
        guidance_scale_tt = float32_tensor(
            torch.tensor(ts.guidance_scale, dtype=torch.float32).reshape(1, 1, 1, 1),
            device=(None if traced else self.mesh_device),
        )

        permuted_velocity_pred_tt = ts.model.combined_step(
            do_classifier_free_guidance=cfg_enabled,
            spatial_1BNI=permuted_model_input,
            prompt_1BLP=ts.prompt_buffer,
            negative_prompt_1BLP=ts.negative_prompt_buffer,
            N=latents_sequence_length,
            timestep=timestep,
            **rope_args,
            guidance_scale=guidance_scale_tt,
            traced=traced,
            gather_output=False,
        )

        return self._solver.step(
            step=step,
            latent=permuted_latent_tt,
            velocity_pred=permuted_velocity_pred_tt,
        )

    def get_model_input(self, latents: ttnn.Tensor, cond_latents: ttnn.Tensor | None) -> ttnn.Tensor:
        """Adapter function to enable I2V. For base T2V, just return the latents (cast to bf16)."""
        del cond_latents
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
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        del image_prompt, dtype

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

    DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"  # noqa: E501, RUF001

    @torch.no_grad()
    def __call__(
        self,
        *,
        prompts: Sequence[str],
        negative_prompts: Sequence[str] | None = None,
        image_prompt=None,
        num_inference_steps: int,
        guidance_scale: float = 4.0,
        guidance_scale_2: float | None = 3.0,
        flow_shift: float | None = None,
        boundary_ratio: float | None = None,
        num_videos_per_prompt: int | None = 1,
        seed: int = 0,
        output_type: str | None = "uint8",
        traced: bool = False,
        on_event: PipelineEventCallback | None = None,
    ):
        on_event = on_event if on_event is not None else null_callback

        negative_prompts = (
            negative_prompts if negative_prompts is not None else [self.DEFAULT_NEGATIVE_PROMPT] * len(prompts)
        )
        height = self._height
        width = self._width
        num_frames = self._num_frames

        # Per-request boundary_ratio overrides the construction-time default. It only affects
        # host-side expert selection (no captured trace depends on it).
        effective_boundary_ratio = boundary_ratio if boundary_ratio is not None else self._boundary_ratio

        if guidance_scale > 1 and not self._cfg_enabled:
            msg = "guidance_scale > 1 requires CFG to be enabled"
            raise ValueError(msg)
        if guidance_scale_2 is not None and guidance_scale_2 > 1 and not self._cfg_enabled:
            msg = "guidance_scale_2 > 1 requires CFG to be enabled"
            raise ValueError(msg)

        if height % 16 != 0 or width % 16 != 0:
            msg = f"`height` and `width` have to be divisible by 16 but are {height} and {width}."
            raise ValueError(msg)

        if effective_boundary_ratio is None and guidance_scale_2 is not None:
            msg = (
                "`guidance_scale_2` is only supported when `boundary_ratio` is not None. "
                "Set it per-request (pass `boundary_ratio=...` to this call) or at construction time."
            )
            raise ValueError(msg)

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. "
                "Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if effective_boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self.transformer_states[0].guidance_scale = guidance_scale
        self.transformer_states[1].guidance_scale = guidance_scale_2

        device = "cpu"

        batch_size = len(prompts)

        # 3. Encode input prompt
        on_event(SectionStart("encoder"))
        self._prepare_text_encoder()
        prompt_embeds, negative_prompt_embeds = self._text_encoder.encode_cfg(
            prompts,
            negative_prompts,
            cfg_enabled=self._cfg_enabled,
            num_videos_per_prompt=num_videos_per_prompt,
            on_event=on_event,
        )
        on_event(SectionEnd("encoder"))

        # 4. Prepare schedule
        # flow_shift is host-side only (it reshapes the sigma schedule); set it on the
        # scheduler config before set_timesteps so the new schedule is recomputed. No
        # captured trace depends on it. Always assign so a per-request value never
        # persists into a later request — fall back to the construction-time default
        # when omitted, mirroring effective_boundary_ratio above.
        self._scheduler.config.flow_shift = flow_shift if flow_shift is not None else self._default_flow_shift
        self._scheduler.set_timesteps(num_inference_steps)
        self._solver.set_schedule(self._scheduler.sigmas.tolist())
        timesteps = self._scheduler.timesteps

        # 5. Prepare latent variables
        torch.manual_seed(seed)

        on_event(SectionStart("prepare_latents"))
        latents, cond_latents = self.prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            image_prompt=image_prompt,
            num_channels_latents=self._vae.config.z_dim,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
        )
        on_event(SectionEnd("prepare_latents"))

        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # 6. Denoising loop
        if effective_boundary_ratio is not None:
            boundary_timestep = effective_boundary_ratio * 1000
        else:
            boundary_timestep = -1  # Always use transformer (no transformer_2)

        on_event(SectionStart("denoising"))

        permuted_latent_tt = None
        rope_args = None

        latent_frames, latent_height, latent_width = latents.shape[2], latents.shape[3], latents.shape[4]
        prepared_prompts = [False, False]

        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
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
                    permuted_latent, latents_sequence_length = ts.model.preprocess_spatial_input_host(latents)

                    sp_axis = ts.model.parallel_config.sequence_parallel.mesh_axis

                    if cond_latents is not None:
                        cond_latents, _ = ts.model.preprocess_spatial_input_host(cond_latents)
                        cond_latents_tt = tensor.from_torch(
                            cond_latents,
                            device=self.mesh_device,
                            mesh_axes=[None, None, sp_axis, None],
                            dtype=ttnn.bfloat16,
                        )
                        if self.condition_buffer is None:
                            self.condition_buffer = cond_latents_tt
                        else:
                            ttnn.copy(cond_latents_tt, self.condition_buffer)

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
                        dtype=ttnn.float32,
                    )

                # setup/update latent buffer
                if self.latent_buffer is None:
                    self.latent_buffer = permuted_latent_tt
                else:
                    ttnn.copy(permuted_latent_tt, self.latent_buffer)

                permuted_latent_tt = self._step(
                    step=i,
                    t=t,
                    ts=ts,
                    permuted_latent_tt=self.latent_buffer,
                    mask=mask,
                    cond_latents=self.condition_buffer,
                    rope_args=rope_args,
                    latents_sequence_length=latents_sequence_length,
                    latents_batch_size=latents.shape[0],
                    cfg_enabled=self._cfg_enabled,
                    traced=traced,
                )

                progress_bar.update()
                on_event(DenoiseStep(step=i + 1, total=num_inference_steps, sigma=float(self._scheduler.sigmas[i])))

        self._current_timestep = None

        sp_axis = ts.model.parallel_config.sequence_parallel.mesh_axis
        permuted_latent_tt = ts.model.ccl_manager.all_gather_persistent_buffer(
            permuted_latent_tt, dim=2, mesh_axis=sp_axis
        )
        permuted_latent = tensor.local_device_to_torch(permuted_latent_tt)

        # Postprocess spatial output
        latents = ts.model.postprocess_spatial_output_host(
            permuted_latent, F=latent_frames, H=latent_height, W=latent_width, N=latents_sequence_length
        )

        on_event(SectionEnd("denoising"))

        if output_type == "latent":
            return latents

        include_last_latent = output_type == "pt_with_last_latent"
        if include_last_latent:
            output_type = "pt"
            last_latent_out = latents.detach().clone()

        video = self._decode_latents(latents, output_type=output_type, on_event=on_event)

        if include_last_latent:
            return video, last_latent_out
        return video

    def _decode_latents(
        self,
        latents: torch.Tensor,
        *,
        output_type: str,
        on_event: PipelineEventCallback,
    ) -> torch.Tensor:
        on_event(SectionStart("vae"))
        video_torch = self._vae.decode(latents, output_type=output_type)

        if output_type == "uint8":
            video = video_torch.numpy()
        elif output_type == "np":
            video = video_torch.float().numpy()
        else:
            video = self.video_processor.postprocess_video(video_torch, output_type=output_type)

        on_event(SectionEnd("vae"))
        return video

    def synchronize_devices(self) -> None:
        ttnn.synchronize_device(self.mesh_device)

    def release_traces(self) -> None:
        for model in (self.transformer, self.transformer_2):
            tracer = WanTransformer3DModel.combined_step._tracers.get(model)
            if tracer is not None:
                tracer.release_trace()
