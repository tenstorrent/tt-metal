# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import tqdm
from diffusers.pipelines.mochi.pipeline_mochi import linear_quadratic_schedule
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.video_processor import VideoProcessor
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.transformer_mochi import MochiCheckpoint
from models.tt_dit.models.vae.vae_mochi import MochiVAEDecoderAdapter
from models.tt_dit.parallel.config import DiTParallelConfig, MochiVAEParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.pipelines.mochi.text_encoder import TextEncoder
from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin
from models.tt_dit.solvers import EulerSolver
from models.tt_dit.utils import cache
from models.tt_dit.utils.mesh import reshape_device
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractContextManager

_DEFAULT_CHECKPOINT = "genmo/mochi-1-preview"

_PRESETS_WH: dict[tuple[int, ...], dict] = {
    (2, 4): {
        "sp": (2, 0),
        "tp": (4, 1),
        "vae_mesh_shape": (1, 8),
        "vae_sp_axis": 0,
        "vae_tp_axis": 1,
        "num_links": 1,
        "reload_dit_model": True,
    },
    (4, 8): {
        "sp": (8, 1),
        "tp": (4, 0),
        "vae_mesh_shape": (4, 8),
        "vae_sp_axis": 0,
        "vae_tp_axis": 1,
        "num_links": 4,
        "reload_dit_model": False,
    },
}

_PRESETS_BH: dict[tuple[int, ...], dict] = {
    (2, 2): {
        "sp": (2, 0),
        "tp": (2, 1),
        "vae_mesh_shape": (1, 4),
        "vae_sp_axis": 0,
        "vae_tp_axis": 1,
        "num_links": 2,
        "reload_dit_model": True,
    },
    (2, 4): {
        "sp": (2, 0),
        "tp": (4, 1),
        "vae_mesh_shape": (2, 4),
        "vae_sp_axis": 0,
        "vae_tp_axis": 1,
        "num_links": 2,
        "reload_dit_model": False,
    },
    (4, 8): {
        "sp": (8, 1),
        "tp": (4, 0),
        "vae_mesh_shape": (4, 8),
        "vae_sp_axis": 0,
        "vae_tp_axis": 1,
        "num_links": 2,
        "reload_dit_model": False,
    },
}


@dataclass(frozen=True, kw_only=True)
class MochiPipelineConfig:
    topology: ttnn.Topology
    num_links: int

    dit_parallel_config: DiTParallelConfig
    vae_parallel_config: MochiVAEParallelConfig
    vae_mesh_shape: tuple[int, ...]

    use_reference_vae: bool
    force_zeros_for_empty_prompt: bool
    reload_dit_model: bool

    height: int
    width: int
    num_frames: int
    max_sequence_length: int

    checkpoint_name: str

    @classmethod
    def default(
        cls,
        *,
        mesh_shape: ttnn.MeshShape,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        num_links: int | None = None,
        dit_parallel_config: DiTParallelConfig | None = None,
        vae_parallel_config: MochiVAEParallelConfig | None = None,
        vae_mesh_shape: tuple[int, ...] | None = None,
        use_reference_vae: bool = False,
        force_zeros_for_empty_prompt: bool = False,
        reload_dit_model: bool | None = None,
        height: int = 480,
        width: int = 848,
        num_frames: int = 168,
        max_sequence_length: int = 256,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> MochiPipelineConfig:
        preset_dict = _PRESETS_BH if ttnn.device.is_blackhole() else _PRESETS_WH
        preset = preset_dict.get(tuple(mesh_shape), {})

        if dit_parallel_config is None:
            dit_parallel_config = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=preset["sp"], tp=preset["tp"])

        if vae_mesh_shape is None:
            vae_mesh_shape = preset["vae_mesh_shape"]

        if vae_parallel_config is None:
            vae_sp_axis = preset["vae_sp_axis"]
            vae_tp_axis = preset["vae_tp_axis"]
            if vae_mesh_shape[0] > 1 and vae_mesh_shape[1] > 1:
                # 2D mesh (e.g. Galaxy): separate H/W on different axes
                vae_parallel_config = MochiVAEParallelConfig.from_tuples(
                    time=(1, vae_tp_axis),
                    h=(vae_mesh_shape[vae_sp_axis], vae_sp_axis),
                    w=(vae_mesh_shape[vae_tp_axis], vae_tp_axis),
                )
            else:
                # 1D mesh (e.g. T3K, N300): use time parallelism, no spatial
                t_axis = 1 if vae_mesh_shape[1] > 1 else 0
                vae_parallel_config = MochiVAEParallelConfig.from_tuples(
                    time=(vae_mesh_shape[t_axis], t_axis),
                    h=(1, 0),
                    w=(1, 1),
                )

        return cls(
            topology=topology,
            num_links=num_links if num_links is not None else preset["num_links"],
            dit_parallel_config=dit_parallel_config,
            vae_parallel_config=vae_parallel_config,
            vae_mesh_shape=tuple(vae_mesh_shape),
            use_reference_vae=use_reference_vae,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            reload_dit_model=reload_dit_model if reload_dit_model is not None else preset["reload_dit_model"],
            height=height,
            width=width,
            num_frames=num_frames,
            max_sequence_length=max_sequence_length,
            checkpoint_name=checkpoint_name,
        )


class MochiPipeline(PipelineAPIMixin):
    r"""The mochi pipeline for text-to-video generation.

    Reference: https://github.com/genmoai/models
    """

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        height: int = 480,
        width: int = 848,
        num_frames: int = 168,
        max_sequence_length: int = 256,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> MochiPipeline:
        config = MochiPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            height=height,
            width=width,
            num_frames=num_frames,
            max_sequence_length=max_sequence_length,
            checkpoint_name=checkpoint_name,
        )
        return cls(device=mesh_device, config=config)

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        config: MochiPipelineConfig,
    ) -> None:
        # TODO: determine these scaling factors from model parameters
        self.vae_spatial_scale_factor = 8
        self.vae_temporal_scale_factor = 6
        self.patch_size = 2

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_scale_factor)

        self._device = device
        self._vae_mesh_shape = config.vae_mesh_shape
        self.parallel_config = config.dit_parallel_config
        self.vae_parallel_config = config.vae_parallel_config
        self.reload_dit_model = config.reload_dit_model  # Only required if VAE is memory-constrained.
        self._height = config.height
        self._width = config.width
        self._num_frames = config.num_frames

        if self._height % self.vae_spatial_scale_factor != 0 or self._width % self.vae_spatial_scale_factor != 0:
            msg = (
                f"`height` and `width` must be divisible by {self.vae_spatial_scale_factor} "
                f"but are {self._height} and {self._width}."
            )
            raise ValueError(msg)

        if self.reload_dit_model and not cache.cache_dir_is_set():
            msg = (
                "Cache must be enabled when DiT model reloading is enabled (reload_dit_model=True). "
                "Please set TT_DIT_CACHE_DIR environment variable to enable caching."
            )
            raise RuntimeError(msg)

        # Create CCL manager
        self._ccl_manager = CCLManager(
            mesh_device=device,
            num_links=config.num_links,
            topology=config.topology,
        )

        # Create VAE CCL manager using the VAE mesh shape.
        with self._reshape_vae():
            self._vae_ccl_manager = CCLManager(
                mesh_device=device,
                num_links=config.num_links,
                topology=ttnn.Topology.Linear,
            )

        checkpoint_name = config.checkpoint_name
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_name, subfolder="scheduler")
        self._solver = EulerSolver()

        # Load pretrained T5 text encoder and tokenizer (Torch)
        self._text_encoder = TextEncoder(
            checkpoint_name=checkpoint_name,
            force_zeros_for_empty_prompt=config.force_zeros_for_empty_prompt,
            max_sequence_length=config.max_sequence_length,
        )

        # Load pretrained Mochi Transformer (TT)
        self._checkpoint = MochiCheckpoint(checkpoint_name)

        self._transformer = self._checkpoint.build(
            ccl_manager=self._ccl_manager,
            parallel_config=self.parallel_config,
            is_fsdp=True,
        )
        self._tracer = Tracer(self._transformer.forward, device=device, prep_run=False)

        self._checkpoint.load(
            self._transformer,
            mesh_device=self._device,
            parallel_config=self.parallel_config,
        )

        with self._reshape_vae():
            self._vae = MochiVAEDecoderAdapter(
                checkpoint_name=checkpoint_name,
                parallel_config=self.vae_parallel_config,
                ccl_manager=self._vae_ccl_manager,
                use_torch=config.use_reference_vae,
            )
            if not config.use_reference_vae:
                self._vae.reload_weights()

        logger.info("Pipeline allocation run...")
        self(prompts=[""], num_inference_steps=2, guidance_scale=2, traced=False)

    def _reshape_vae(self) -> AbstractContextManager[None]:
        return reshape_device(self._device, self._vae_mesh_shape)

    def prepare_latents(self, batch_size: int, num_channels_latents: int, dtype: torch.dtype) -> torch.Tensor:
        height = self._height // self.vae_spatial_scale_factor
        width = self._width // self.vae_spatial_scale_factor
        num_frames = (self._num_frames - 1) // self.vae_temporal_scale_factor + 1

        shape = (batch_size, num_channels_latents, num_frames, height, width)
        return torch.randn(shape, dtype=torch.float32).to(dtype)

    @torch.no_grad()
    def __call__(
        self,
        *,
        prompts: Sequence[str],
        negative_prompts: Sequence[str] | None = None,
        num_inference_steps: int,
        guidance_scale: float = 4.5,
        num_videos_per_prompt: int | None = 1,
        seed: int = 0,
        traced: bool = False,
        vae_traced: bool | None = None,
        on_event: PipelineEventCallback | None = None,
    ) -> torch.Tensor:
        on_event = on_event if on_event is not None else null_callback
        negative_prompts = negative_prompts if negative_prompts is not None else [""] * len(prompts)

        vae_traced = vae_traced if vae_traced is not None else traced

        batch_size = len(prompts)

        # 3. Prepare text embeddings
        on_event(SectionStart("encoder"))
        torch_context, torch_context_masks = self._text_encoder.encode_cfg(
            prompts,
            negative_prompts,
            num_videos_per_prompt=num_videos_per_prompt,
            disable_attention_mask=traced,
            on_event=on_event,
        )
        on_event(SectionEnd("encoder"))

        # 3b. If the transformer was destroyed, recreate it.
        if self._transformer is None:
            logger.info("Recreating MochiTransformer3DModel")
            self._transformer = self._checkpoint.build(
                ccl_manager=self._ccl_manager,
                parallel_config=self.parallel_config,
                is_fsdp=True,
            )
            self._tracer = Tracer(
                self._transformer.forward, device=self._device, prep_run=True, clone_prep_inputs=False
            )

            logger.info("Loading MochiTransformer3DModel state_dict")
            self._checkpoint.load(
                self._transformer,
                mesh_device=self._device,
                parallel_config=self.parallel_config,
            )

        assert self._tracer is not None

        # 4. Prepare latent variables
        torch.manual_seed(seed)

        num_channels_latents = self._checkpoint.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            torch_context[1].dtype,
        )

        # 5. Prepare timestep
        # from https://github.com/genmoai/models/blob/075b6e36db58f1242921deff83a1066887b9c9e1/src/mochi_preview/infer.py#L77
        sigmas = np.array(linear_quadratic_schedule(num_inference_steps, threshold_noise=0.025))
        self._scheduler.set_timesteps(sigmas=sigmas)
        self._solver.set_schedule(self._scheduler.sigmas.tolist())
        timesteps = self._scheduler.timesteps

        # Upload spatial latents and pre-compute rope features once before the loop.
        _, _, f, h, w = latents.shape
        latents, latents_sequence_length = self._transformer.preprocess_spatial_input(latents)
        rope_cos, rope_sin, trans_mat = self._transformer.prepare_rope_features(f, h, w)

        # 6. Denoising loop
        on_event(SectionStart("denoising"))
        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                timestep = torch.tensor([t], dtype=torch.float32)

                temb, context = self._transformer.prepare_timestep_text_features(
                    timestep, torch_context[0], torch_context_masks[0]
                )
                velocity_pred_uncond = self._tracer(
                    spatial_1BNI=latents,
                    temb_11BD=temb,
                    prompt_1BLP=context,
                    rope_cos_1HND=rope_cos if i == 0 else self._tracer.inputs["rope_cos_1HND"],
                    rope_sin_1HND=rope_sin if i == 0 else self._tracer.inputs["rope_sin_1HND"],
                    trans_mat=trans_mat if i == 0 else self._tracer.inputs["trans_mat"],
                    N=latents_sequence_length,
                    traced=traced,
                )

                # latents can be overwritten by trace execution, use the captured input instead,
                # which is safe.
                latents = self._tracer.inputs["spatial_1BNI"]

                # Move to CPU memory since the output tensor will be overwritten by the next trace
                # execution.
                velocity_pred_uncond = velocity_pred_uncond.cpu()

                temb, context = self._transformer.prepare_timestep_text_features(
                    timestep, torch_context[1], torch_context_masks[1]
                )
                velocity_pred_cond = self._tracer(
                    spatial_1BNI=latents,
                    temb_11BD=temb,
                    prompt_1BLP=context,
                    rope_cos_1HND=self._tracer.inputs["rope_cos_1HND"],
                    rope_sin_1HND=self._tracer.inputs["rope_sin_1HND"],
                    trans_mat=self._tracer.inputs["trans_mat"],
                    N=latents_sequence_length,
                    traced=traced,
                )

                velocity_pred_uncond = velocity_pred_uncond.to(velocity_pred_cond.device())

                velocity_pred = velocity_pred_uncond + guidance_scale * (velocity_pred_cond - velocity_pred_uncond)
                latents = self._solver.step(step=i, latent=latents, velocity_pred=velocity_pred)

                progress_bar.update()
        on_event(SectionEnd("denoising"))

        latents = self._transformer.postprocess_spatial_output(latents, f, h, w, latents_sequence_length)

        return self._decode_latents(
            latents,
            vae_traced=vae_traced,
            on_event=on_event,
        )

    def _decode_latents(
        self,
        latents: torch.Tensor,
        *,
        vae_traced: bool,
        on_event: PipelineEventCallback,
    ) -> torch.Tensor:
        # If the VAE is memory-constrained, free the transformer.
        if self.reload_dit_model:
            logger.info("Freeing MochiTransformer3DModel")
            self._transformer = None
            self._tracer.release_trace()
            self._tracer = None

        on_event(SectionStart("vae"))
        with self._reshape_vae():
            video = self._vae.decode(latents, traced=vae_traced)
        on_event(SectionEnd("vae"))

        return self.video_processor.postprocess_video(video, output_type="pil")
