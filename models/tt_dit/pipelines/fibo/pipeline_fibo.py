# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import tqdm
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.transformer_fibo import FiboCheckpoint
from models.tt_dit.models.vae.vae_fibo import FiboVAEDecoderAdapter
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, Flux2VaeParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.cfg import CFGCombiner, create_submeshes, distribute_cfg, submesh_shape
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.pipelines.fibo.text_encoder import TextEncoder
from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin
from models.tt_dit.solvers import EulerSolver
from models.tt_dit.utils.mesh import reshape_device
from models.tt_dit.utils.tensor import from_torch_to_devices
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractContextManager

    from PIL import Image

# Wan 2.2 VAE compresses 16x in each spatial dimension. FIBO's diffusers pipeline defaults to
# ``do_patching=False``, so the VAE's z-dim channels go straight to the transformer without any
# pipeline-level 2x2 packing; the transformer's own ``patch_size=1`` just flattens H×W into a
# sequence dimension.
_VAE_SCALE_FACTOR = 16
_DEFAULT_CHECKPOINT = "briaai/FIBO"
# Prompt lengths the text encoder is traced at; must be divisible by sp * 128
_DEFAULT_SEQUENCE_LENGTHS = (1024, 1536, 3072)

# Rules of thumb for the encoder parallelism below, from sweeping the text encoder over every
# mesh shape from one to eight devices on a T3K:
#
# - At equal device count, sequence parallelism beats tensor parallelism at every sequence
#   length and by a wide margin. Tensor parallelism all-reduces activations at every linear
#   layer, while sequence parallelism only gathers keys and values inside attention.
# - Splitting the devices between the two still beats pouring them all into either one. At four
#   devices, two by two won over both pure sequence parallelism and pure tensor parallelism,
#   and by the widest margin at the shortest sequence length.
# - Pure tensor parallelism scales negatively: four devices came out slower than one at the
#   longest sequence length.
# - Do not give the encoder more than four devices. Every eight-device shape lost to the best
#   four-device one.
# - Which physical chips a shape lands on is worth a substantial fraction of the runtime, so
#   timings taken on different mesh shapes are not directly comparable.
#
# And for the VAE decoder, from the same sweep:
#
# - Split the image spatially, over height or width. Tensor parallelism lost at every device
#   count, and mixing it into a spatial split cost more than it saved.
# - Height or width makes little difference. Pick whichever the mesh shape makes convenient.
# - Unlike the encoder, the decoder keeps gaining out to eight devices, where it was fastest.
#   The gains are sublinear but had not flattened out.
_PRESETS_WH: dict[tuple[int, ...], dict] = {
    (2, 4): {
        "cfg": (2, 1),
        "sp": (2, 1),
        "tp": (2, 0),
        "encoder_tp": (2, 0),
        "encoder_sp": (2, 1),
        "vae_tp_axis": None,
        "vae_h_axis": 0,
        "vae_w_axis": 1,
        "num_links": 1,
    },
}

_PRESETS_BH: dict[tuple[int, ...], dict] = {
    (2, 2): {
        "cfg": (2, 1),
        "sp": (1, 1),
        "tp": (2, 0),
        "encoder_tp": (2, 0),
        "encoder_sp": None,
        "vae_tp_axis": None,
        "vae_h_axis": 0,
        "vae_w_axis": None,
        "num_links": 2,
    },
}


@dataclass(frozen=True, kw_only=True)
class FiboPipelineConfig:
    topology: ttnn.Topology
    num_links: int

    dit_parallel_config: DiTParallelConfig
    encoder_parallel_config: EncoderParallelConfig
    vae_parallel_config: Flux2VaeParallelConfig

    use_torch_text_encoder: bool
    use_torch_vae_decoder: bool

    height: int
    width: int
    cfg_enabled: bool
    sequence_lengths: tuple[int, ...]

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
        vae_parallel_config: Flux2VaeParallelConfig | None = None,
        use_torch_text_encoder: bool = False,
        use_torch_vae_decoder: bool = False,
        height: int = 1024,
        width: int = 1024,
        cfg_enabled: bool = True,
        sequence_lengths: tuple[int, ...] = _DEFAULT_SEQUENCE_LENGTHS,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> FiboPipelineConfig:
        """Build a fully populated config, picking parallelism defaults from ``mesh_shape``."""
        preset_dict = _PRESETS_BH if ttnn.device.is_blackhole() else _PRESETS_WH
        preset = preset_dict.get(tuple(mesh_shape), {}) if mesh_shape is not None else {}

        dit_parallel_config = dit_parallel_config or DiTParallelConfig.from_tuples(
            cfg=preset["cfg"], sp=preset["sp"], tp=preset["tp"]
        )

        return cls(
            topology=topology,
            num_links=num_links or preset["num_links"],
            dit_parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config
            or EncoderParallelConfig.from_tuples(tp=preset["encoder_tp"], sp=preset["encoder_sp"]),
            vae_parallel_config=vae_parallel_config
            or Flux2VaeParallelConfig.from_axes(
                submesh_shape(dit_parallel_config),
                tp_axis=preset["vae_tp_axis"],
                h_axis=preset["vae_h_axis"],
                w_axis=preset["vae_w_axis"],
            ),
            use_torch_text_encoder=use_torch_text_encoder,
            use_torch_vae_decoder=use_torch_vae_decoder,
            height=height,
            width=width,
            cfg_enabled=cfg_enabled,
            sequence_lengths=sequence_lengths,
            checkpoint_name=checkpoint_name,
        )


class FiboPipeline(PipelineAPIMixin):
    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        width: int = 1024,
        height: int = 1024,
        cfg_enabled: bool = True,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> FiboPipeline:
        config = FiboPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            checkpoint_name=checkpoint_name,
            width=width,
            height=height,
            cfg_enabled=cfg_enabled,
        )
        return cls(device=mesh_device, config=config)

    def __init__(self, *, device: ttnn.MeshDevice, config: FiboPipelineConfig) -> None:
        self._cfg_parallel = config.dit_parallel_config.cfg_parallel.factor != 1
        self._sp_axis = config.dit_parallel_config.sequence_parallel.mesh_axis
        self._encoder_tp = config.encoder_parallel_config.tensor_parallel
        self._encoder_sp = config.encoder_parallel_config.sequence_parallel
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
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.checkpoint_name, subfolder="scheduler")
        self._solvers = (EulerSolver(), EulerSolver()) if self._cfg_parallel else (EulerSolver(),)
        # The ``* 2`` rounds image dims to a multiple of 32 to keep the optional ``do_patching=True``
        # path (a 2x2 pre-pack) usable, even though we never take it.
        self._image_processor = VaeImageProcessor(vae_scale_factor=_VAE_SCALE_FACTOR * 2)

        logger.info("creating transformer...")
        self._checkpoint = FiboCheckpoint(config.checkpoint_name)
        self._transformers = [
            self._checkpoint.build(parallel_config=config.dit_parallel_config, ccl_manager=m)
            for m in self._ccl_managers
        ]

        with self._reshape_encoder():
            logger.info("creating text encoder...")
            self._text_encoder = TextEncoder(
                checkpoint_name=config.checkpoint_name,
                device=self._devices[0],
                ccl_manager=self._ccl_managers[0],
                parallel_config=config.encoder_parallel_config,
                sequence_lengths=config.sequence_lengths,
                use_torch=config.use_torch_text_encoder,
            )

        logger.info("creating VAE decoder...")
        self._vae = FiboVAEDecoderAdapter(
            checkpoint_name=config.checkpoint_name,
            parallel_config=config.vae_parallel_config,
            use_torch=config.use_torch_vae_decoder,
            ccl_manager=self._ccl_managers[0],
        )
        self._vae.reload_weights()

        self._synchronize_devices()

        logger.info("pipeline allocation run...")
        with self._reshape_encoder():
            self._text_encoder.warmup()
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
        negative_prompts: Sequence[str] | None = None,
        num_inference_steps: int,
        seed: int = 0,
        num_images_per_prompt: int = 1,
        cfg_scale: float = 5.0,
        traced: bool = True,
        vae_traced: bool | None = None,
        encoder_traced: bool | None = False,
        on_event: PipelineEventCallback | None = None,
    ) -> list[Image.Image]:
        prompt_count = len(prompts)

        if cfg_scale > 1 and not self._cfg_enabled:
            msg = "cfg_scale > 1 requires CFG to be enabled"
            raise ValueError(msg)

        vae_traced = vae_traced if vae_traced is not None else traced
        encoder_traced = encoder_traced if encoder_traced is not None else traced
        on_event = on_event if on_event is not None else null_callback
        negative_prompts = negative_prompts if negative_prompts is not None else [""] * prompt_count

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

        latents_height = self._height // _VAE_SCALE_FACTOR
        latents_width = self._width // _VAE_SCALE_FACTOR
        latents_sequence_length = latents_height * latents_width

        on_event(SectionStart("total"))

        logger.info("encoding prompts...")
        on_event(SectionStart("encoder"))
        with self._reshape_encoder():
            torch_context, torch_layers, _ = self._text_encoder.encode_cfg(
                prompts,
                negative_prompts,
                num_images_per_prompt=num_images_per_prompt,
                cfg_enabled=self._cfg_enabled,
                traced=encoder_traced,
                on_event=on_event,
            )
        self._synchronize_devices()  # for time profiling
        on_event(SectionEnd("encoder"))

        # Pad/trim SmolLM3's per-layer hidden states to the transformer's block count. Diffusers'
        # FIBO pipeline does the same: when there are MORE encoder layers than DiT blocks, drop the
        # earliest ones (keep the latest); when there are FEWER, duplicate the last layer to fill.
        target_layers = self._checkpoint.num_blocks
        if len(torch_layers) >= target_layers:
            torch_layers = torch_layers[len(torch_layers) - target_layers :]
        else:
            torch_layers = torch_layers + [torch_layers[-1]] * (target_layers - len(torch_layers))

        logger.info("preparing timesteps...")
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = _calculate_shift(latents_sequence_length, self._scheduler)
        self._scheduler.set_timesteps(sigmas=sigmas, mu=mu)
        sigmas = self._scheduler.sigmas.tolist()
        for solver in self._solvers:
            solver.set_schedule(sigmas=sigmas)
        timesteps = self._scheduler.timesteps

        logger.info("preparing inputs...")
        latents = self._random_latents(batch_size=prompt_count * num_images_per_prompt, seed=seed)
        context = distribute_cfg(torch_context, devices=self._devices, on_host=traced)
        layers = [distribute_cfg(layer, devices=self._devices, on_host=traced) for layer in torch_layers]

        prompt_sequence_length = torch_context.shape[1]
        spatial_rope_cos, spatial_rope_sin, prompt_rope_cos, prompt_rope_sin = self._prepare_rope(
            latents_height=latents_height,
            latents_width=latents_width,
            prompt_sequence_length=prompt_sequence_length,
        )

        logger.info("denoising...")
        on_event(SectionStart("denoising"))

        for step, t in enumerate(tqdm.tqdm(timesteps)):
            on_event(SectionStart(f"denoising_step_{step}"))

            velocity_preds = []
            for idx, tracer in enumerate(self._tracers):
                timestep = ttnn.full(
                    [1, 1],
                    fill_value=t,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.float32,
                    device=self._devices[idx],
                )

                velocity_preds.append(
                    tracer(
                        submesh_idx=idx,
                        latents=latents[idx],
                        prompt=context[idx] if step == 0 else tracer.inputs["prompt"],
                        text_encoder_layers=[layer[idx] for layer in layers]
                        if step == 0
                        else tracer.inputs["text_encoder_layers"],
                        timestep=timestep,
                        spatial_rope=(spatial_rope_cos[idx], spatial_rope_sin[idx])
                        if step == 0
                        else tracer.inputs["spatial_rope"],
                        prompt_rope=(prompt_rope_cos[idx], prompt_rope_sin[idx])
                        if step == 0
                        else tracer.inputs["prompt_rope"],
                        spatial_sequence_length=latents_sequence_length,
                        prompt_sequence_length=prompt_sequence_length,
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

            self._synchronize_devices()  # for time profiling
            on_event(SectionEnd(f"denoising_step_{step}"))

        on_event(SectionEnd("denoising"))

        logger.info("decoding image...")
        on_event(SectionStart("vae"))
        images = self._decode_latents(latents[0], traced=vae_traced)
        on_event(SectionEnd("vae"))

        on_event(SectionEnd("total"))
        return images

    def _traced_step(self, *, submesh_idx: int, latents: ttnn.Tensor, **kwargs: Any) -> ttnn.Tensor:
        if self._cfg_enabled and not self._cfg_parallel:
            latents = ttnn.concat([latents, latents])

        return self._transformers[submesh_idx].forward(spatial=latents, **kwargs)

    def _prepare_rope(
        self, *, latents_height: int, latents_width: int, prompt_sequence_length: int
    ) -> tuple[list[ttnn.Tensor], list[ttnn.Tensor], list[ttnn.Tensor], list[ttnn.Tensor]]:
        """Build FIBO's 3-axis RoPE on CPU and distribute spatial/prompt cos/sin to each submesh.

        Mirrors the diffusers FIBO transformer: ``ids = cat([text_ids (zeros), img_ids], dim=0)``
        is fed through ``checkpoint.pos_embed`` to get ``(freqs_cos, freqs_sin)`` of shape
        ``[prompt_seq + img_seq, head_dim]``; we split at ``prompt_sequence_length`` and ship the
        spatial half SP-sharded, the prompt half replicated.
        """
        h = latents_height
        w = latents_width

        img_ids = torch.zeros(h, w, 3)
        img_ids[..., 1] = torch.arange(h)[:, None]
        img_ids[..., 2] = torch.arange(w)[None, :]
        img_ids = img_ids.reshape(h * w, 3)
        text_ids = torch.zeros(prompt_sequence_length, 3)
        ids = torch.cat([text_ids, img_ids], dim=0)

        torch_rope_cos, torch_rope_sin = self._checkpoint.pos_embed(ids)

        torch_prompt_rope_cos = torch_rope_cos[:prompt_sequence_length]
        torch_prompt_rope_sin = torch_rope_sin[:prompt_sequence_length]
        torch_spatial_rope_cos = torch_rope_cos[prompt_sequence_length:]
        torch_spatial_rope_sin = torch_rope_sin[prompt_sequence_length:]

        spatial_rope_cos = from_torch_to_devices(
            torch_spatial_rope_cos, devices=self._devices, mesh_axes=[self._sp_axis, None]
        )
        spatial_rope_sin = from_torch_to_devices(
            torch_spatial_rope_sin, devices=self._devices, mesh_axes=[self._sp_axis, None]
        )
        prompt_rope_cos = from_torch_to_devices(torch_prompt_rope_cos, devices=self._devices)
        prompt_rope_sin = from_torch_to_devices(torch_prompt_rope_sin, devices=self._devices)
        return spatial_rope_cos, spatial_rope_sin, prompt_rope_cos, prompt_rope_sin

    def _random_latents(self, batch_size: int, seed: int) -> list[ttnn.Tensor]:
        torch.manual_seed(seed)

        shape = [
            batch_size,
            self._checkpoint.latent_channels,
            self._height // _VAE_SCALE_FACTOR,
            self._width // _VAE_SCALE_FACTOR,
        ]

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

        decoded_output = self._vae.decode(torch_latents, traced=traced)

        image = self._image_processor.postprocess(decoded_output, output_type="pt")
        assert isinstance(image, torch.Tensor)

        return self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))

    def _synchronize_devices(self) -> None:
        for d in self._devices:
            ttnn.synchronize_device(d)


def _calculate_shift(image_seq_len: int, scheduler: FlowMatchEulerDiscreteScheduler) -> float:
    """Resolution-dependent mu used by FlowMatchEulerDiscreteScheduler's dynamic shifting."""
    base_seq_len = scheduler.config.get("base_image_seq_len", 256)
    max_seq_len = scheduler.config.get("max_image_seq_len", 4096)
    base_shift = scheduler.config.get("base_shift", 0.5)
    max_shift = scheduler.config.get("max_shift", 1.15)

    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b
