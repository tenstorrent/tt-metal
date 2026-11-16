# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import math
import os
from typing import TYPE_CHECKING, TypeVar

import diffusers
import numpy as np
import torch
import tqdm
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, event_section, null_callback

from ...models.transformers.transformer_flux2 import Flux2Transformer
from ...models.vae.vae_flux2 import Flux2VaeDecoder, Flux2VaeEncoder
from ...parallel.config import DiTGParallelConfigNoCFG, EncoderParallelConfig, Flux2VaeParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...utils import cache
from ...utils.padding import PaddingConfig
from ...utils.tensor import fast_device_to_host, float_to_uint8, from_torch, to_torch
from ...utils.tracing import StateTensor, Tracer
from .prompt_encoder import PromptEncoder

if TYPE_CHECKING:
    from collections.abc import Sequence


_T = TypeVar("_T")

# The image sizes (width, height) a pipeline supports by default: the square sizes the inference
# server offers. Other aspect ratios would need padding the latents to whole tiles per device.
_DEFAULT_RESOLUTIONS = ((1024, 1024), (2048, 2048))

# The parallel layout per mesh shape: the DiT's SP and TP axes, the text encoder's TP axis, the VAE's
# channel (TP) and row (H) axes, and the CCL links and topology.
_PRESETS_WH: dict[tuple[int, ...], dict] = {
    (1, 8): {
        "sp_axis": 0,
        "tp_axis": 1,
        "encoder_tp_axis": 1,
        "vae_tp_axis": 1,
        "vae_h_axis": 0,
        "num_links": 1,
        "topology": ttnn.Topology.Linear,
    },
    # SP4/TP2 denoised Klein 1% faster than SP2/TP4 at 1024². TP2 + H4 decoded faster than H2xW4 or H4xW2
    # without TP (0.47 s against 0.78 and 0.56).
    (2, 4): {
        "sp_axis": 1,
        "tp_axis": 0,
        "encoder_tp_axis": 1,
        "vae_tp_axis": 0,
        "vae_h_axis": 1,
        "num_links": 1,
        "topology": ttnn.Topology.Linear,
    },
}

_PRESETS_BH: dict[tuple[int, ...], dict] = {
    (2, 2): {
        "sp_axis": 0,
        "tp_axis": 1,
        "encoder_tp_axis": 1,
        "vae_tp_axis": 1,
        "vae_h_axis": 0,
        "num_links": 2,
        "topology": ttnn.Topology.Linear,
    },
    (2, 4): {
        "sp_axis": 0,
        "tp_axis": 1,
        "encoder_tp_axis": 1,
        "vae_tp_axis": 1,
        "vae_h_axis": 0,
        "num_links": 2,
        "topology": ttnn.Topology.Linear,
    },
    (4, 8): {
        "sp_axis": 0,
        "tp_axis": 1,
        "encoder_tp_axis": 1,
        "vae_tp_axis": 0,
        "vae_h_axis": 1,
        "num_links": 2,
        "topology": ttnn.Topology.Linear,
    },
}


def _release_module_trace(module: object | None) -> None:
    """Release the trace captured on ``module.forward``, if it has one."""
    forward = getattr(type(module), "forward", None)
    tracers = getattr(forward, "_tracers", None)
    if tracers is None:
        return
    tracer = tracers.get(module)
    if tracer is not None:
        tracer.release_trace()


class Flux2TransformerState:
    def __init__(self) -> None:
        self._tt_embedded_prompt = StateTensor()
        self._tt_guidance = StateTensor()
        self._tt_prompt_rope_cos = StateTensor()
        self._tt_prompt_rope_sin = StateTensor()
        self._tt_timestep = StateTensor()
        self._tt_sigma_difference = StateTensor()

    def __getattr__(self, name: str) -> ttnn.Tensor | None:
        return object.__getattribute__(self, f"_{name}").data


class _DenoiseState:
    """The denoising state of one image size, with or without a reference image: the latents, the
    reference tokens, their RoPE and the trace."""

    def __init__(self, *, step_tracer: Tracer) -> None:
        self.step_tracer = step_tracer
        self.latents = StateTensor()
        self.reference = StateTensor()
        self.rope_cos = StateTensor()
        self.rope_sin = StateTensor()
        self.combined_rope_cos = StateTensor()
        self.combined_rope_sin = StateTensor()


class Flux2Pipeline:
    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        checkpoint_name: str = "black-forest-labs/FLUX.2-dev",
        parallel_config: DiTGParallelConfigNoCFG,
        encoder_parallel_config: EncoderParallelConfig,
        vae_parallel: Flux2VaeParallelConfig,
        topology: ttnn.Topology,
        num_links: int,
        height: int = 1024,
        width: int = 1024,
        resolutions: Sequence[tuple[int, int]] = _DEFAULT_RESOLUTIONS,
        is_fsdp: bool = False,
        dynamic_load: bool = False,
        trace_warmup: bool = False,
        vae_use_conv3d: bool = False,
        shard_prompt: bool = True,
    ) -> None:
        """Build the pipeline.

        ``resolutions`` are the image sizes (width, height) that ``__call__`` accepts, all warmed
        up here; ``width`` and ``height`` give the default one.
        """
        self._resolutions = tuple(resolutions)
        if (width, height) not in self._resolutions:
            msg = f"the default size {width}x{height} is not one of the resolutions {self._resolutions}"
            raise ValueError(msg)

        self._mesh_device = mesh_device
        self._parallel_config = parallel_config
        self._encoder_parallel_config = encoder_parallel_config
        self._shard_prompt = shard_prompt

        self._vae_parallel = vae_parallel
        self._vae_use_conv3d = vae_use_conv3d

        self._height = height
        self._width = width
        self.is_fsdp = is_fsdp
        self.dynamic_load = dynamic_load

        # The denoising states by image size and whether there is a reference image, and the VAE
        # traces by image size.
        self._denoise_states: dict[tuple[int, int, bool], _DenoiseState] = {}
        self._vae_decoder_tracers: dict[tuple[int, int], Tracer] = {}
        self._vae_encoder_tracers: dict[tuple[int, int], Tracer] = {}

        logger.info(f"Parallel config: {parallel_config}")
        logger.info(f"Encoder parallel config: {encoder_parallel_config}")
        logger.info(f"VAE parallel: {vae_parallel}")

        self._ccl_manager = CCLManager(self._mesh_device, num_links=num_links, topology=topology)

        logger.info("loading models...")

        self.checkpoint_name = checkpoint_name

        self._torch_transformer = diffusers.Flux2Transformer2DModel.from_pretrained(
            checkpoint_name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,  # bfloat16 is the native datatype of the model
        )
        self._torch_transformer.eval()

        self._torch_vae = AutoencoderKLFlux2.from_pretrained(checkpoint_name, subfolder="vae")
        assert isinstance(self._torch_vae, AutoencoderKLFlux2)

        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_name, subfolder="scheduler")

        logger.info("creating TT-NN transformer...")

        transformer_config = self._torch_transformer.config
        head_dim = transformer_config.attention_head_dim
        num_heads = transformer_config.num_attention_heads
        self._guidance_embeds = transformer_config.guidance_embeds
        self._num_channels_latents = 32
        self._patch_size = 2
        self._vae_scale_factor = 8

        if num_heads % parallel_config.tensor_parallel.factor != 0:
            padding_config = PaddingConfig.from_tensor_parallel_factor(
                num_heads,
                head_dim,
                parallel_config.tensor_parallel.factor,
            )
        else:
            padding_config = None

        self.transformer = Flux2Transformer(
            in_channels=transformer_config.in_channels,
            num_layers=transformer_config.num_layers,
            num_single_layers=transformer_config.num_single_layers,
            attention_head_dim=head_dim,
            num_attention_heads=num_heads,
            joint_attention_dim=transformer_config.joint_attention_dim,
            out_channels=self._torch_transformer.out_channels,
            device=mesh_device,
            ccl_manager=self._ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
            is_fsdp=self.is_fsdp,
            shard_prompt=self._shard_prompt,
            guidance_embeds=self._guidance_embeds,
        )

        self._pos_embed = self._torch_transformer.pos_embed

        self._image_processor = VaeImageProcessor()

        logger.info("creating TT-NN text encoder...")
        self._encoder_ccl_manager = CCLManager(self._mesh_device, num_links=num_links, topology=topology)
        self._prompt_encoder = PromptEncoder(
            checkpoint_name=checkpoint_name,
            use_torch_encoder=False,
            device=self._mesh_device,
            parallel_config=self._encoder_parallel_config,
            ccl_manager=self._encoder_ccl_manager,
        )

        logger.info("creating TT-NN VAE decoder...")
        self._vae_ccl_manager = CCLManager(self._mesh_device, num_links=num_links, topology=ttnn.Topology.Linear)
        self._vae_decoder = Flux2VaeDecoder(
            out_channels=3,
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            z_channels=32,
            device=self._mesh_device,
            parallel_config=self._vae_parallel,
            ccl_manager=self._vae_ccl_manager,
            use_conv3d=self._vae_use_conv3d,
        )

        # Editing is not supported with dynamic loading, which would have to evict the encoder too.
        if self.dynamic_load:
            self._vae_encoder = None
        else:
            logger.info("creating TT-NN VAE encoder...")
            self._vae_encoder = Flux2VaeEncoder(
                in_channels=3,
                block_out_channels=[128, 256, 512, 512],
                layers_per_block=2,
                z_channels=32,
                device=self._mesh_device,
                parallel_config=self._vae_parallel,
                ccl_manager=self._vae_ccl_manager,
            )
            cache.load_model(
                tt_model=self._vae_encoder,
                get_torch_state_dict=self._torch_vae.state_dict,
                model_name=os.path.basename(self.checkpoint_name),
                subfolder="vae_encoder",
                parallel_config=self._vae_parallel,
                mesh_shape=tuple(self._mesh_device.shape),
                mesh_device=self._mesh_device,
            )
        self._latents_bn_mean = self._torch_vae.bn.running_mean.float()
        self._latents_bn_std = torch.sqrt(
            self._torch_vae.bn.running_var.float() + self._torch_vae.config.batch_norm_eps
        )

        if self.dynamic_load:
            self.transformer.register_coresident_exclusions(self._prompt_encoder._encoder, self._vae_decoder)
            self._prompt_encoder._encoder.register_coresident_exclusions(self.transformer)
            self._vae_decoder.register_coresident_exclusions(self.transformer)

        # Load in reverse order of use so the first-used model (encoder) stays loaded before __call__.
        self._prepare_vae()
        self._prepare_transformer()
        self._prepare_prompt_encoder()

        self.ts = Flux2TransformerState()

        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        text_ids = _prepare_ids(text_sequence_length=512)  # matches sequence_length=512 in encode
        prompt_rope_cos, prompt_rope_sin = self._pos_embed.forward(text_ids)
        # Store rope tensors pre-shaped as (1, 1, seq, head_dim) so attention blocks never
        # need to reshape per-call (2800 ops per inference). SP sharding is on dim 2 (seq).
        # When sharding the prompt across SP, the prompt rope must be sharded the same way (on the
        # sequence dim) so per-rank tokens get their matching rope positions before the attention
        # gathers q/k/v back to full length.
        prompt_rope_mesh_axes = [None, None, sp_axis, None] if self._shard_prompt else None
        self.ts._tt_prompt_rope_cos.update(
            prompt_rope_cos.unsqueeze(0).unsqueeze(0), False, mesh_axes=prompt_rope_mesh_axes, device=self._mesh_device
        )
        self.ts._tt_prompt_rope_sin.update(
            prompt_rope_sin.unsqueeze(0).unsqueeze(0), False, mesh_axes=prompt_rope_mesh_axes, device=self._mesh_device
        )

        # Every size runs untraced before the first capture. The buffers that the first run at a size
        # allocates and keeps, such as conv weights and CCL buffers, would otherwise be allocated
        # after an earlier trace was captured, and that trace's replay would overwrite them.
        edits = (False, True) if self._vae_encoder is not None else (False,)
        for width, height in self._resolutions:
            for edit in edits:
                self.warmup(width=width, height=height, edit=edit, warmup_info="e2e warmup")
        if trace_warmup:
            for width, height in self._resolutions:
                for edit in edits:
                    self.warmup(width=width, height=height, edit=edit, traced=True, warmup_info="trace warmup")
                    # TODO: INVESTIGATE. For some reason, the VAE time still incures some overhead without this extra trace
                    self.warmup(
                        width=width,
                        height=height,
                        edit=edit,
                        traced=True,
                        warmup_info="steady state trace warmup for VAE",
                    )

    @property
    def resolutions(self) -> tuple[tuple[int, int], ...]:
        """The supported image sizes as (width, height)."""
        return self._resolutions

    def warmup(
        self, *, width: int, height: int, edit: bool = False, traced: bool = False, warmup_info: str = "warmup"
    ) -> None:
        logger.info(
            f"Warming up pipeline at {width}x{height}{' with a reference image' if edit else ''}: {warmup_info}..."
        )
        self.__call__(
            prompts=["warmup"],
            num_inference_steps=2,
            seed=0,
            width=width,
            height=height,
            images=[Image.new("RGB", (width, height), (128, 128, 128))] if edit else None,
            traced=traced,
        )

    def _release_denoise_trace(self) -> None:
        """Release the captured denoise traces, if any."""
        for state in self._denoise_states.values():
            state.step_tracer.release_trace()

    def _release_vae_traces(self) -> None:
        for tracer in itertools.chain(self._vae_decoder_tracers.values(), self._vae_encoder_tracers.values()):
            tracer.release_trace()

    def release_traces(self) -> None:
        """Release every trace captured by this pipeline or its submodules."""
        self._release_denoise_trace()
        self._release_vae_traces()
        _release_module_trace(getattr(self._prompt_encoder, "_encoder", None))

    def _prepare_transformer(self) -> None:
        # A trace bakes its weights' addresses, so drop the traces this load will evict.
        if self.dynamic_load and not self.transformer.is_loaded():
            self._release_vae_traces()
            _release_module_trace(getattr(self._prompt_encoder, "_encoder", None))
        cache.load_model(
            tt_model=self.transformer,
            get_torch_state_dict=self._torch_transformer.state_dict,
            model_name=os.path.basename(self.checkpoint_name),
            subfolder="transformer",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self._mesh_device.shape),
            mesh_device=self._mesh_device,
            is_fsdp=self.is_fsdp,
        )
        ttnn.synchronize_device(self._mesh_device)

    def _prepare_prompt_encoder(self) -> None:
        encoder = getattr(self._prompt_encoder, "_encoder", None)
        encoder_is_loaded = getattr(encoder, "is_loaded", None)
        if self.dynamic_load and encoder_is_loaded is not None and not encoder_is_loaded():
            self._release_denoise_trace()
        self._prompt_encoder.load_weights()
        ttnn.synchronize_device(self._mesh_device)

    def _prepare_vae(self) -> None:
        if self.dynamic_load and not self._vae_decoder.is_loaded():
            self._release_denoise_trace()
        cache.load_model(
            tt_model=self._vae_decoder,
            get_torch_state_dict=self._torch_vae.state_dict,
            model_name=os.path.basename(self.checkpoint_name),
            subfolder="vae",
            parallel_config=self._vae_parallel,
            mesh_shape=tuple(self._mesh_device.shape),
            mesh_device=self._mesh_device,
        )
        ttnn.synchronize_device(self._mesh_device)

    @staticmethod
    def create_pipeline(
        *,
        mesh_device: ttnn.MeshDevice,
        sp_axis: int | None = None,
        tp_axis: int | None = None,
        encoder_tp_axis: int | None = None,
        vae_tp_axis: int | None = None,
        vae_h_axis: int | None = None,
        vae_w_axis: int | None = None,
        num_links: int | None = None,
        topology: ttnn.Topology | None = None,
        width: int = 1024,
        height: int = 1024,
        resolutions: Sequence[tuple[int, int]] = _DEFAULT_RESOLUTIONS,
        is_fsdp: bool = False,
        dynamic_load: bool = False,
        trace_warmup: bool = True,
        checkpoint_name: str = "black-forest-labs/FLUX.2-dev",
        vae_use_conv3d: bool = False,
        shard_prompt: bool = True,
    ) -> Flux2Pipeline:
        """Build a pipeline; the parallel layout arguments left as ``None`` come from the mesh's preset.

        The VAE axes come from the preset only when none of them is given. The presets do not set
        ``is_fsdp`` or ``dynamic_load``: FLUX.2-dev needs both on 2x2, Klein neither.

        With ``dynamic_load``, loading a component releases the traces that depend on the weights it
        evicts, and the next traced call captures them again, so serving should run without it.
        """
        mesh_shape = tuple(mesh_device.shape)
        preset = (_PRESETS_BH if ttnn.device.is_blackhole() else _PRESETS_WH).get(mesh_shape)

        def from_preset(name: str, value: _T | None) -> _T:
            if value is not None:
                return value
            if preset is None:
                msg = f"{name} is required: there is no FLUX.2 preset for mesh shape {mesh_shape}"
                raise ValueError(msg)
            return preset[name]

        sp_axis = from_preset("sp_axis", sp_axis)
        tp_axis = from_preset("tp_axis", tp_axis)
        encoder_tp_axis = from_preset("encoder_tp_axis", encoder_tp_axis)
        num_links = from_preset("num_links", num_links)
        topology = from_preset("topology", topology)
        if vae_tp_axis is None and vae_h_axis is None and vae_w_axis is None:
            vae_tp_axis = from_preset("vae_tp_axis", vae_tp_axis)
            vae_h_axis = from_preset("vae_h_axis", vae_h_axis)

        dit_parallel_config = DiTGParallelConfigNoCFG(
            tensor_parallel=ParallelFactor(factor=int(mesh_device.shape[tp_axis]), mesh_axis=tp_axis),
            sequence_parallel=ParallelFactor(factor=int(mesh_device.shape[sp_axis]), mesh_axis=sp_axis),
        )
        encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=int(mesh_device.shape[encoder_tp_axis]), mesh_axis=encoder_tp_axis)
        )
        vae_parallel = Flux2VaeParallelConfig.from_axes(
            mesh_device, tp_axis=vae_tp_axis, h_axis=vae_h_axis, w_axis=vae_w_axis
        )

        return Flux2Pipeline(
            mesh_device=mesh_device,
            checkpoint_name=checkpoint_name,
            parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel=vae_parallel,
            topology=topology,
            num_links=num_links,
            width=width,
            height=height,
            resolutions=resolutions,
            is_fsdp=is_fsdp,
            dynamic_load=dynamic_load,
            trace_warmup=trace_warmup,
            vae_use_conv3d=vae_use_conv3d,
            shard_prompt=shard_prompt,
        )

    def __call__(
        self,
        *,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 4.0,
        prompts: Sequence[str],
        num_inference_steps: int,
        prompt_upsample_temperature: float | None = None,  # prompt upsampling is currently very slow
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        images: Sequence[Image.Image] | None = None,
        traced: bool = False,
        on_event: PipelineEventCallback | None = None,
    ) -> list[Image.Image]:
        """Generate an image, or edit one if ``images`` holds a reference image.

        ``width`` and ``height`` default to the size the pipeline was created with and must be one
        of ``resolutions``. The reference image is center-cropped to the output's aspect ratio and
        resized to the output size, at most one megapixel. Each size gets its own
        traces, with and without a reference, captured on the first traced call.
        """
        if images and len(images) > 1:
            msg = "editing with more than one reference image is not supported"
            raise NotImplementedError(msg)
        if images and self._vae_encoder is None:
            msg = "editing is not supported with dynamic loading"
            raise NotImplementedError(msg)
        reference_image = images[0] if images else None

        width = width if width is not None else self._width
        height = height if height is not None else self._height
        if (width, height) not in self._resolutions:
            msg = f"{width}x{height} is not one of the resolutions {self._resolutions}"
            raise ValueError(msg)

        on_event = on_event if on_event is not None else null_callback
        on_event(SectionStart("total"))

        prompt_count = len(prompts)

        sp_axis = self._parallel_config.sequence_parallel.mesh_axis

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

        latents_height = height // self._vae_scale_factor
        latents_width = width // self._vae_scale_factor
        transformer_batch_size = prompt_count * num_images_per_prompt
        spatial_sequence_length = (latents_height // self._patch_size) * (latents_width // self._patch_size)

        logger.info("encoding prompts...")

        with event_section(on_event, "encoder"):
            self._prepare_prompt_encoder()
            if prompt_upsample_temperature is not None:
                prompts = self._prompt_encoder.upsample(
                    prompts,
                    max_length=224,  # TODO: this should be higher
                    temperature=prompt_upsample_temperature,
                    traced=traced,
                )

            prompt_embeds, _mask = self._prompt_encoder.encode(
                prompts,
                num_images_per_prompt=num_images_per_prompt,
                sequence_length=512,
                traced=traced,
            )
            _, prompt_sequence_length, _ = prompt_embeds.shape

        if reference_image is not None:
            logger.info("encoding the reference image...")
            with event_section(on_event, "image_encoder"):
                reference, reference_ids = self._encode_reference(
                    reference_image, width=width, height=height, traced=traced
                )
            reference_sequence_length = reference.shape[1]
        else:
            reference = reference_ids = None
            reference_sequence_length = 0

        logger.info("preparing timesteps...")
        timesteps, sigmas = _schedule(
            self._scheduler,
            step_count=num_inference_steps,
            spatial_sequence_length=spatial_sequence_length,
        )

        guidance = torch.full([transformer_batch_size], fill_value=guidance_scale)

        logger.info("preparing latents...")

        # The noise of diffusers' FLUX.2 pipelines in bfloat16 for the same seed: drawn on the CPU in
        # the patchified layout, one token per row.
        generator = torch.Generator("cpu").manual_seed(seed) if seed is not None else None
        p = self._patch_size
        shape = [transformer_batch_size, self._num_channels_latents * p * p, latents_height // p, latents_width // p]
        latents = torch.randn(shape, generator=generator, dtype=torch.bfloat16).flatten(2).transpose(1, 2).float()

        state = self._denoise_state(width, height, edit=reference is not None)
        image_ids = _prepare_ids(height=latents_height // self._patch_size, width=latents_width // self._patch_size)
        rope_cos, rope_sin = self._pos_embed.forward(image_ids)

        state.latents.update(latents, traced, mesh_axes=[None, sp_axis, None], device=self._mesh_device)

        # Each device runs the transformer on its shard of the latents followed by its shard of the
        # reference tokens, so the RoPE is arranged the same way. The attention does not depend on
        # the order of the tokens, only on their positions.
        if reference is not None:
            sp_factor = self._parallel_config.sequence_parallel.factor
            reference_rope_cos, reference_rope_sin = self._pos_embed.forward(reference_ids)
            rope_cos = _interleave_shards(rope_cos, reference_rope_cos, sp_factor)
            rope_sin = _interleave_shards(rope_sin, reference_rope_sin, sp_factor)
            state.reference.update(reference, traced, mesh_axes=[None, sp_axis, None], device=self._mesh_device)

        # Store rope tensors pre-shaped as (1, 1, seq, head_dim) so attention blocks never
        # need to reshape per-call. SP sharding is on dim 2 (seq).
        rope_mesh_axes = [None, None, sp_axis, None]
        state.rope_cos.update(rope_cos[None, None], traced, mesh_axes=rope_mesh_axes, device=self._mesh_device)
        state.rope_sin.update(rope_sin[None, None], traced, mesh_axes=rope_mesh_axes, device=self._mesh_device)
        # The single blocks run on the concatenated [spatial | prompt] sequence and need the matching
        # rope. It is constant across the denoising steps, so concat once here.
        state.combined_rope_cos.update(ttnn.concat([state.rope_cos.data, self.ts.tt_prompt_rope_cos], dim=2), traced)
        state.combined_rope_sin.update(ttnn.concat([state.rope_sin.data, self.ts.tt_prompt_rope_sin], dim=2), traced)

        if self._guidance_embeds:
            self.ts._tt_guidance.update(guidance.unsqueeze(1), traced, device=self._mesh_device)

        logger.info("denoising...")
        self._prepare_transformer()

        # The prompt is fixed for the whole generation, so run its input projection once here
        # instead of once per denoising step.
        prompt_embeds_mesh_axes = [None, sp_axis, None] if self._shard_prompt else None
        tt_prompt_embeds = from_torch(prompt_embeds, device=self._mesh_device, mesh_axes=prompt_embeds_mesh_axes)
        self.ts._tt_embedded_prompt.update(self.transformer.context_embedder(tt_prompt_embeds), traced)
        ttnn.deallocate(tt_prompt_embeds)

        with event_section(on_event, "denoising"):
            for i, t in enumerate(tqdm.tqdm(timesteps)):
                with event_section(on_event, f"denoising_step_{i}"):
                    sigma_difference = (sigmas[i + 1] - sigmas[i]).item()

                    self.ts._tt_timestep.update(
                        torch.full([1, 1], fill_value=float(t)),
                        device=self._mesh_device,
                        dtype=ttnn.float32,
                        traced=traced,
                    )

                    self.ts._tt_sigma_difference.update(
                        torch.full([1, 1], fill_value=sigma_difference),
                        device=self._mesh_device,
                        dtype=ttnn.float32,
                        traced=traced,
                    )

                    step_inputs = {
                        "spatial": state.latents.data,
                        "reference": state.reference.data if reference is not None else None,
                        "embedded_prompt": self.ts.tt_embedded_prompt,
                        "timestep": self.ts.tt_timestep,
                        "guidance": self.ts.tt_guidance,
                        "sigma_difference": self.ts.tt_sigma_difference,
                        "spatial_rope": (state.rope_cos.data, state.rope_sin.data),
                        "prompt_rope": (self.ts.tt_prompt_rope_cos, self.ts.tt_prompt_rope_sin),
                        "combined_rope": (state.combined_rope_cos.data, state.combined_rope_sin.data),
                        "spatial_sequence_length": spatial_sequence_length + reference_sequence_length,
                        "prompt_sequence_length": prompt_sequence_length,
                    }
                    state.step_tracer(**step_inputs, traced=traced)

        logger.info("decoding image...")
        with event_section(on_event, "vae"):
            self._prepare_vae()

            # Latents arrive SP-sharded on dim=1 (patchified token dim). Redistribute to
            # whatever spatial sharding the VAE conv pyramid expects before unpatchifying.
            #
            # H (token dim=1): gather SP, then re-shard to H axis if on a different axis.
            # W (spatial dim=2): can only be applied after unpatchify since H/W are interleaved
            #   in patchified tokens; mesh_partition is applied right after the reshape below.
            if self._vae_parallel.h_parallel is not None and self._vae_parallel.h_parallel.mesh_axis == sp_axis:
                tt_latents = state.latents.data
            else:
                tt_latents = self._ccl_manager.all_gather(
                    state.latents.data,
                    dim=1,
                    mesh_axis=sp_axis,
                    use_hyperparams=True,
                )
                if self._vae_parallel.h_parallel is not None:
                    tt_latents = ttnn.mesh_partition(
                        tt_latents,
                        dim=1,
                        cluster_axis=self._vae_parallel.h_parallel.mesh_axis,
                        memory_config=tt_latents.memory_config(),
                    )

            tt_latents = self._vae_decoder.preprocess_and_unpatchify(
                tt_latents, height=latents_height, width=latents_width
            )

            if self._vae_parallel.w_parallel is not None:
                tt_latents = ttnn.mesh_partition(
                    tt_latents,
                    dim=2,
                    cluster_axis=self._vae_parallel.w_parallel.mesh_axis,
                    memory_config=tt_latents.memory_config(),
                )

            tracer = self._vae_decoder_tracers.get((width, height))
            if tracer is None:
                tracer = Tracer(self._vae_decoder.forward, device=self._mesh_device, clone_prep_inputs=False)
                self._vae_decoder_tracers[width, height] = tracer
            tt_decoded_output = tracer(tt_latents, traced=traced)
            decoded_output = fast_device_to_host(
                tt_decoded_output,
                self._mesh_device,
                [None, None],
                ccl_manager=self._vae_ccl_manager,
                pre_transfer_fn=float_to_uint8,
            )
            output = [Image.fromarray(image.numpy()) for image in decoded_output]

        on_event(SectionEnd("total"))
        return output

    def _denoise_state(self, width: int, height: int, *, edit: bool) -> _DenoiseState:
        state = self._denoise_states.get((width, height, edit))
        if state is None:
            # The step updates the latents in place, so the run before the capture works on copies.
            state = _DenoiseState(step_tracer=Tracer(self._step, device=self._mesh_device, clone_prep_inputs=True))
            self._denoise_states[width, height, edit] = state
        return state

    def _encode_reference(
        self, image: Image.Image, *, width: int, height: int, traced: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a reference image into normalized latent tokens and their position ids.

        The image is center-cropped to the aspect ratio of the output and resized to the output
        size, scaled down to at most one megapixel.
        """
        assert self._vae_encoder is not None

        scale = min(1.0, math.sqrt(1024 * 1024 / (width * height)))
        multiple = self._vae_scale_factor * self._patch_size
        reference_width = int(width * scale) // multiple * multiple
        reference_height = int(height * scale) // multiple * multiple

        image = image.convert("RGB")
        crop_width = min(image.width, round(image.height * width / height))
        crop_height = min(image.height, round(image.width * height / width))
        left = (image.width - crop_width) // 2
        top = (image.height - crop_height) // 2
        image = image.crop((left, top, left + crop_width, top + crop_height))
        image = image.resize((reference_width, reference_height), Image.Resampling.LANCZOS)

        pixels = torch.from_numpy(np.asarray(image)).float()[None] / 127.5 - 1.0
        h_axis = self._vae_parallel.h_parallel.mesh_axis if self._vae_parallel.h_parallel is not None else None
        w_axis = self._vae_parallel.w_parallel.mesh_axis if self._vae_parallel.w_parallel is not None else None
        tt_pixels = from_torch(
            pixels, device=self._mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=[None, h_axis, w_axis, None]
        )

        tracer = self._vae_encoder_tracers.get((reference_width, reference_height))
        if tracer is None:
            tracer = Tracer(self._vae_encoder.forward, device=self._mesh_device, clone_prep_inputs=False)
            self._vae_encoder_tracers[reference_width, reference_height] = tracer
        tt_latents = tracer(tt_pixels, traced=traced)

        latents = to_torch(tt_latents).float()
        tokens = (self._patchify(latents) - self._latents_bn_mean) / self._latents_bn_std
        # As diffusers, the first reference image is at time 10.
        ids = _prepare_ids(
            height=reference_height // multiple,
            width=reference_width // multiple,
            time=10,
        )
        return tokens, ids

    def _step(
        self,
        *,
        spatial: ttnn.Tensor,
        reference: ttnn.Tensor | None,
        embedded_prompt: ttnn.Tensor,
        timestep: ttnn.Tensor,
        guidance: ttnn.Tensor | None,
        sigma_difference: ttnn.Tensor,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        combined_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        spatial_sequence_length: int,
        prompt_sequence_length: int,
    ) -> None:
        noise_pred = self.transformer.forward(
            spatial=ttnn.concat([spatial, reference], dim=1) if reference is not None else spatial,
            embedded_prompt=embedded_prompt,
            timestep=timestep,
            guidance=guidance,
            spatial_rope=spatial_rope,
            prompt_rope=prompt_rope,
            combined_rope=combined_rope,
            spatial_sequence_length=spatial_sequence_length,
            prompt_sequence_length=prompt_sequence_length,
        )
        if reference is not None:
            noise_pred = noise_pred[:, : spatial.shape[1], :]

        ttnn.multiply_(noise_pred, sigma_difference)
        ttnn.add_(spatial, noise_pred)

    def _patchify(self, latents: torch.Tensor) -> torch.Tensor:
        # N, H, W, C -> N, (H / P) * (W / P), C * P * P
        batch_size, height, width, channels = latents.shape
        p = self._patch_size

        if height % p != 0 or width % p != 0:
            msg = f"height ({height}) and width ({width}) must be divisible by patch_size ({p})"
            raise ValueError(msg)

        latents = latents.reshape([batch_size, height // p, p, width // p, p, channels])
        return latents.permute(0, 1, 3, 5, 2, 4).flatten(3, 5).flatten(1, 2)


def _schedule(
    scheduler: FlowMatchEulerDiscreteScheduler,
    *,
    step_count: int,
    spatial_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scheduler.set_timesteps(
        sigmas=np.linspace(1.0, 1 / step_count, step_count),
        mu=compute_empirical_mu(image_seq_len=spatial_sequence_length, num_steps=step_count),
    )

    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas

    assert isinstance(timesteps, torch.Tensor)
    assert isinstance(sigmas, torch.Tensor)

    return timesteps, sigmas


# From https://github.com/black-forest-labs/flux2/blob/ab7cca68018ad3ceadcace9d6ecb1bc1f6f46b4e/src/flux2/sampling.py#L251C1-L266C21
def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def _interleave_shards(a: torch.Tensor, b: torch.Tensor, num_shards: int) -> torch.Tensor:
    """Concatenate ``a`` and ``b`` on the first dim so that each of ``num_shards`` equal shards of the
    result is the matching shard of ``a`` followed by that of ``b``."""
    shards = zip(a.chunk(num_shards), b.chunk(num_shards), strict=True)
    return torch.cat([t for pair in shards for t in pair])


def _prepare_ids(*, height: int = 1, width: int = 1, text_sequence_length: int = 1, time: int = 0) -> torch.Tensor:
    t = torch.tensor([time])
    h = torch.arange(height)
    w = torch.arange(width)
    s = torch.arange(text_sequence_length)

    return torch.cartesian_prod(t, h, w, s)
