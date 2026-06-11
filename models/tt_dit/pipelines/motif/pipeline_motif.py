# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import tqdm
from diffusers.image_processor import VaeImageProcessor
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.transformer_motif import MotifCheckpoint
from models.tt_dit.models.vae.vae_sd35 import VAEDecoderAdapter
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VAEParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.cfg import CFGCombiner, create_submeshes, distribute_cfg
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin
from models.tt_dit.solvers import EulerSolver
from models.tt_dit.utils.mesh import reshape_device
from models.tt_dit.utils.tensor import from_torch_to_devices
from models.tt_dit.utils.tracing import Tracer

from .text_encoder import TextEncoder

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractContextManager

    from PIL import Image

_VAE_SCALE_FACTOR = 8
_LATENT_CHANNELS = 16
_DEFAULT_CHECKPOINT = "Motif-Technologies/Motif-Image-6B-Preview"

_PRESETS: dict[tuple[int, ...], dict] = {
    (2, 4): {
        "cfg": (2, 0),
        "sp": (1, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "vae_tp": (4, 1),
        "num_links": 1,
    },
    (4, 8): {
        "cfg": (2, 1),
        "sp": (4, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "vae_tp": (4, 1),
        "num_links": 4,
    },
}


@dataclass(frozen=True, kw_only=True)
class MotifPipelineConfig:
    topology: ttnn.Topology
    num_links: int

    dit_parallel_config: DiTParallelConfig
    encoder_parallel_config: EncoderParallelConfig
    vae_parallel_config: VAEParallelConfig

    enable_t5_text_encoder: bool
    use_torch_t5_text_encoder: bool
    use_torch_clip_text_encoder: bool
    use_torch_vae: bool

    height: int
    width: int
    cfg_enabled: bool

    checkpoint_name: str

    @classmethod
    def default(
        cls,
        *,
        mesh_shape: ttnn.MeshShape | None = None,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        num_links: int | None = None,
        dit_parallel_config: DiTParallelConfig | None = None,
        encoder_parallel_config: EncoderParallelConfig | None = None,
        vae_parallel_config: VAEParallelConfig | None = None,
        enable_t5_text_encoder: bool = True,
        use_torch_t5_text_encoder: bool = False,
        use_torch_clip_text_encoder: bool = False,
        use_torch_vae: bool = False,
        height: int = 1024,
        width: int = 1024,
        cfg_enabled: bool = True,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> MotifPipelineConfig:
        """Build a fully populated config, picking parallelism defaults from ``mesh_shape``."""
        preset = _PRESETS.get(tuple(mesh_shape), {}) if mesh_shape is not None else {}

        return cls(
            topology=topology,
            num_links=num_links or preset["num_links"],
            dit_parallel_config=dit_parallel_config
            or DiTParallelConfig.from_tuples(cfg=preset["cfg"], sp=preset["sp"], tp=preset["tp"]),
            encoder_parallel_config=encoder_parallel_config or EncoderParallelConfig.from_tuple(preset["encoder_tp"]),
            vae_parallel_config=vae_parallel_config or VAEParallelConfig.from_tuple(preset["vae_tp"]),
            enable_t5_text_encoder=enable_t5_text_encoder,
            use_torch_t5_text_encoder=use_torch_t5_text_encoder,
            use_torch_clip_text_encoder=use_torch_clip_text_encoder,
            use_torch_vae=use_torch_vae,
            height=height,
            width=width,
            cfg_enabled=cfg_enabled,
            checkpoint_name=checkpoint_name,
        )


class MotifPipeline(PipelineAPIMixin):
    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        width: int = 1024,
        height: int = 1024,
        cfg_enabled: bool = True,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> MotifPipeline:
        config = MotifPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            checkpoint_name=checkpoint_name,
            width=width,
            height=height,
            cfg_enabled=cfg_enabled,
        )
        return cls(device=mesh_device, config=config)

    def __init__(self, *, device: ttnn.MeshDevice, config: MotifPipelineConfig) -> None:
        self._cfg_parallel = config.dit_parallel_config.cfg_parallel.factor != 1
        self._sp_axis = config.dit_parallel_config.sequence_parallel.mesh_axis
        self._encoder_tp = config.encoder_parallel_config.tensor_parallel
        self._height = config.height
        self._width = config.width
        self._cfg_enabled = config.cfg_enabled

        logger.info(f"Parallel config: {config.dit_parallel_config}")
        logger.info(f"Original mesh shape: {device.shape}")
        self._devices = create_submeshes(device, config.dit_parallel_config)
        logger.info(f"Created submeshes with shape {self._devices[0].shape}")

        self._tracers = [Tracer(self._traced_step, device=d, prep_run=False) for d in self._devices]

        self._ccl_managers = [
            CCLManager(d, num_links=config.num_links, topology=config.topology) for d in self._devices
        ]

        self._combiner = CFGCombiner(self._devices)
        self._solvers = (EulerSolver(), EulerSolver()) if self._cfg_parallel else (EulerSolver(),)
        self._image_processor = VaeImageProcessor(vae_scale_factor=_VAE_SCALE_FACTOR)

        logger.info("creating transformer...")
        checkpoint = MotifCheckpoint(config.checkpoint_name)
        self._transformers = [
            checkpoint.build(
                latents_height=config.height // _VAE_SCALE_FACTOR,
                latents_width=config.width // _VAE_SCALE_FACTOR,
                parallel_config=config.dit_parallel_config,
                ccl_manager=m,
            )
            for m in self._ccl_managers
        ]

        with self._reshape_encoder():
            logger.info("creating encoder...")
            self._text_encoder = TextEncoder(
                parallel_config=config.encoder_parallel_config,
                enable_t5=config.enable_t5_text_encoder,
                use_torch_clip_encoder=config.use_torch_clip_text_encoder,
                use_torch_t5_encoder=config.use_torch_t5_text_encoder,
                ccl_manager=self._ccl_managers[0],
            )

            logger.info("creating VAE decoder...")
            self._vae = VAEDecoderAdapter(
                checkpoint_name="stabilityai/stable-diffusion-3.5-large",
                parallel_config=config.vae_parallel_config,
                skip_shift=True,  # Motif omits the VAE shift.
                use_torch=config.use_torch_vae,
                ccl_manager=self._ccl_managers[0],
            )

        self.synchronize_devices()

        logger.info("pipeline allocation run...")
        self(prompts=[""], num_inference_steps=2, traced=False, cfg_scale=2 if config.cfg_enabled else 1)

    def _reshape_encoder(self) -> AbstractContextManager[None]:
        device = self._devices[0]
        tp = self._encoder_tp

        shape = list(device.shape)
        shape[tp.mesh_axis] = tp.factor
        shape[1 - tp.mesh_axis] = device.shape.mesh_size() // tp.factor

        return reshape_device(self._devices[0], ttnn.MeshShape(*shape))

    def __call__(
        self,
        *,
        prompts: Sequence[str],
        prompts_2: Sequence[str] | None = None,
        prompts_3: Sequence[str] | None = None,
        negative_prompts: Sequence[str | None] | None = None,
        negative_prompts_2: Sequence[str | None] | None = None,
        negative_prompts_3: Sequence[str | None] | None = None,
        num_inference_steps: int,
        seed: int = 0,
        num_images_per_prompt: int = 1,
        cfg_scale: float = 5.0,
        linear_quadratic_emulating_steps: int = 100,
        negative_strategy_switch_time: float = 0.85,
        traced: bool = False,
        # currently defaults to off due to ttnn.synchronize_device inside vae_all_gather
        vae_traced: bool | None = False,
        encoder_traced: bool | None = None,
        on_event: PipelineEventCallback | None = None,
    ) -> list[Image.Image]:
        prompt_count = len(prompts)

        if cfg_scale > 1 and not self._cfg_enabled:
            msg = "cfg_scale > 1 requires CFG to be enabled"
            raise ValueError(msg)

        vae_traced = vae_traced if vae_traced is not None else traced
        encoder_traced = encoder_traced if encoder_traced is not None else traced
        on_event = on_event if on_event is not None else null_callback
        negative_prompts = negative_prompts or [None] * prompt_count

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

        logger.info("encoding prompts...")
        on_event(SectionStart("total"))

        on_event(SectionStart("encoder"))
        with self._reshape_encoder():
            (
                torch_early_context,
                torch_early_pooled,
                torch_late_context,
                torch_late_pooled,
            ) = self._text_encoder.encode_cfg(
                (prompts, prompts_2 or prompts, prompts_3 or prompts),
                (negative_prompts, negative_prompts_2 or negative_prompts, negative_prompts_3 or negative_prompts),
                num_images_per_prompt=num_images_per_prompt,
                cfg_enabled=self._cfg_enabled,
                traced=encoder_traced,
                on_event=on_event,
            )
        on_event(SectionEnd("encoder"))

        logger.info("preparing timesteps...")
        sigmas = _schedule(
            step_count=num_inference_steps,
            linear_quadratic_emulating_steps=linear_quadratic_emulating_steps,
        )
        for solver in self._solvers:
            solver.set_schedule(sigmas)

        logger.info("preparing inputs...")
        latents = self._random_latents(batch_size=prompt_count * num_images_per_prompt, seed=seed)
        early_context = distribute_cfg(torch_early_context, devices=self._devices, on_host=traced)
        early_pooled = distribute_cfg(torch_early_pooled, devices=self._devices, on_host=traced)
        late_context = distribute_cfg(torch_late_context, devices=self._devices, on_host=traced)
        late_pooled = distribute_cfg(torch_late_pooled, devices=self._devices, on_host=traced)

        logger.info("denoising...")
        on_event(SectionStart("denoising"))

        for step, t in enumerate(tqdm.tqdm(sigmas[:-1])):
            on_event(SectionStart(f"denoising_step_{step}"))

            context = early_context if t >= negative_strategy_switch_time else late_context
            pooled = early_pooled if t >= negative_strategy_switch_time else late_pooled

            velocity_preds = []
            for idx, tracer in enumerate(self._tracers):
                timestep = ttnn.full(
                    [1, 1],
                    fill_value=t * 1000,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.float32,
                    device=self._devices[idx],
                )

                velocity_preds.append(
                    tracer(
                        submesh_idx=idx,
                        latents=latents[idx],
                        prompt=context[idx],
                        pooled=pooled[idx],
                        timestep=timestep,
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

            # Helps with accurate time profiling.
            self.synchronize_devices()

            on_event(SectionEnd(f"denoising_step_{step}"))

        on_event(SectionEnd("denoising"))

        logger.info("decoding image...")
        on_event(SectionStart("vae"))
        images = self._decode_latents(latents[0], traced=vae_traced)
        on_event(SectionEnd("vae"))

        on_event(SectionEnd("total"))
        return images

    def synchronize_devices(self) -> None:
        for device in self._devices:
            ttnn.synchronize_device(device)

    def _traced_step(self, *, submesh_idx: int, latents: ttnn.Tensor, **kwargs: Any) -> ttnn.Tensor:
        if self._cfg_enabled and not self._cfg_parallel:
            latents = ttnn.concat([latents, latents])

        return self._transformers[submesh_idx].forward(spatial=latents, **kwargs)

    def _random_latents(self, batch_size: int, seed: int) -> list[ttnn.Tensor]:
        torch.manual_seed(seed)

        shape = [batch_size, _LATENT_CHANNELS, self._height // _VAE_SCALE_FACTOR, self._width // _VAE_SCALE_FACTOR]

        # We let randn generate a permuted latent tensor in float32, so that the generated noise
        # matches the reference implementation.
        latents = torch.randn(shape, dtype=torch.float32).to(dtype=torch.bfloat16).permute(0, 2, 3, 1)
        latents = self._transformers[0].patchify(latents)

        return from_torch_to_devices(latents, devices=self._devices, mesh_axes=[None, self._sp_axis, None])

    def _decode_latents(self, tt_latents: ttnn.Tensor, *, traced: bool) -> list[Image.Image]:
        # Sync because we don't pass a persistent buffer or a barrier semaphore.
        ttnn.synchronize_device(self._devices[0])

        tt_latents = self._ccl_managers[0].all_gather_persistent_buffer(
            tt_latents, dim=1, mesh_axis=self._sp_axis, use_hyperparams=True
        )

        torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])
        torch_latents = self._transformers[0].unpatchify(
            torch_latents,
            height=self._height // _VAE_SCALE_FACTOR,
            width=self._width // _VAE_SCALE_FACTOR,
        )

        with self._reshape_encoder():
            decoded_output = self._vae.decode(torch_latents, traced=traced)

        image = self._image_processor.postprocess(decoded_output, output_type="pt")
        assert isinstance(image, torch.Tensor)

        return self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))


def _schedule(*, step_count: int, linear_quadratic_emulating_steps: int) -> list[float]:
    """A slight variation of ``schedules.linear_quadratic``."""
    assert step_count % 2 == 0

    s = step_count
    n = linear_quadratic_emulating_steps
    a = s // 2 / n - 1

    sigmas1 = torch.linspace(1, 0, n + 1)[: s // 2]
    sigmas2 = torch.linspace(0, 1, s // 2 + 1).pow(2) * a - a

    sigmas = torch.concat([sigmas1, sigmas2])

    return sigmas.tolist()
