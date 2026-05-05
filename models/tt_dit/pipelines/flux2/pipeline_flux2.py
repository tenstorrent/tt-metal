# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import TYPE_CHECKING

import diffusers
import numpy as np
import torch
import tqdm
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from loguru import logger

import ttnn

from ...models.transformers.transformer_flux2 import Flux2Transformer
from ...models.vae.vae_flux2 import Flux2VaeDecoder
from ...parallel.config import DiTGParallelConfigNoCFG, EncoderParallelConfig, ParallelFactor, VAEParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from ...utils.padding import PaddingConfig
from .prompt_encoder import PromptEncoder

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL import Image


class StateTensor:
    def __init__(self) -> None:
        self._value: ttnn.Tensor | None = None

    def update(
        self,
        value: torch.Tensor | ttnn.Tensor,
        traced: bool,
        dtype: ttnn.DataType = ttnn.bfloat16,
        mesh_axes: list[int] | None = None,
        device: ttnn.Device | None = None,
    ) -> None:
        if torch.is_tensor(value):
            assert device is not None, "device must be provided if using torch tensor"
            value = tensor.from_torch(value, device=device, mesh_axes=mesh_axes, dtype=dtype)
        if self._value is None or not traced:
            self._value = value
        else:
            ttnn.copy(value, self._value)


class Flux2TransformerState:
    def __init__(self) -> None:
        self._tt_prompt_embeds = StateTensor()
        self._tt_latents_step = StateTensor()
        self._tt_guidance = StateTensor()
        self._tt_spatial_rope_cos = StateTensor()
        self._tt_spatial_rope_sin = StateTensor()
        self._tt_prompt_rope_cos = StateTensor()
        self._tt_prompt_rope_sin = StateTensor()
        self._tt_timestep = StateTensor()

    def __getattr__(self, name: str) -> ttnn.Tensor | None:
        return object.__getattribute__(self, f"_{name}")._value


class Flux2Pipeline:
    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        checkpoint_name: str = "black-forest-labs/FLUX.2-dev",
        parallel_config: DiTGParallelConfigNoCFG,
        encoder_parallel_config: EncoderParallelConfig | None = None,
        vae_parallel_config: VAEParallelConfig | None = None,
        topology: ttnn.Topology,
        num_links: int,
        height: int = 1024,
        width: int = 1024,
        is_fsdp: bool = False,
        dynamic_load: bool = False,
        trace_warmup: bool = False,
    ) -> None:
        self._mesh_device = mesh_device
        self._parallel_config = parallel_config
        self._height = height
        self._width = width
        self.is_fsdp = is_fsdp
        self.dynamic_load = dynamic_load

        # setup encoder and vae parallel configs.
        if encoder_parallel_config is None:
            encoder_parallel_config = EncoderParallelConfig(
                tensor_parallel=(
                    parallel_config.tensor_parallel
                    if parallel_config.tensor_parallel.mesh_axis == 4
                    else parallel_config.sequence_parallel
                )
            )

        if vae_parallel_config is None:
            vae_parallel_config = VAEParallelConfig(
                tensor_parallel=(
                    parallel_config.tensor_parallel
                    if parallel_config.tensor_parallel.mesh_axis == 4
                    else parallel_config.sequence_parallel
                )
            )

        self._encoder_parallel_config = encoder_parallel_config
        self._vae_parallel_config = vae_parallel_config

        logger.info(f"Parallel config: {parallel_config}")

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

        head_dim = 128
        num_heads = 48
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
            in_channels=128,
            num_layers=8,
            num_single_layers=48,
            attention_head_dim=head_dim,
            num_attention_heads=num_heads,
            joint_attention_dim=15360,
            out_channels=128,
            device=mesh_device,
            ccl_manager=self._ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
            is_fsdp=self.is_fsdp,
        )

        self._pos_embed = self._torch_transformer.pos_embed

        self._image_processor = VaeImageProcessor()

        logger.info("creating TT-NN text encoder...")
        self._prompt_encoder = PromptEncoder(
            checkpoint_name=checkpoint_name,
            use_torch_encoder=False,
            device=self._mesh_device,
            parallel_config=self._encoder_parallel_config,
            ccl_manager=self._ccl_manager,
        )

        logger.info("creating TT-NN VAE decoder...")
        self._vae_decoder = Flux2VaeDecoder(
            out_channels=3,
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            z_channels=32,
            device=self._mesh_device,
            parallel_config=self._vae_parallel_config,
            ccl_manager=self._ccl_manager,
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
        image_ids = _prepare_ids(height=self._height // 16, width=self._width // 16)
        prompt_rope_cos, prompt_rope_sin = self._pos_embed.forward(text_ids)
        spatial_rope_cos, spatial_rope_sin = self._pos_embed.forward(image_ids)
        self.ts._tt_spatial_rope_cos.update(
            spatial_rope_cos, False, mesh_axes=[sp_axis, None], device=self._mesh_device
        )
        self.ts._tt_spatial_rope_sin.update(
            spatial_rope_sin, False, mesh_axes=[sp_axis, None], device=self._mesh_device
        )
        self.ts._tt_prompt_rope_cos.update(prompt_rope_cos, False, device=self._mesh_device)
        self.ts._tt_prompt_rope_sin.update(prompt_rope_sin, False, device=self._mesh_device)

        self.warmup()  # E2E warmup
        if trace_warmup:  # warmup for trace capture
            self.warmup(traced=True)

    def warmup(self, traced: bool = False) -> None:
        logger.info(f"Warming up pipeline with trace {'enabled' if traced else 'disabled'}...")
        self.__call__(
            prompts=["warmup"],
            num_inference_steps=2,
            seed=0,
            traced=traced,
        )

    def _prepare_transformer(self) -> None:
        cache.load_model(
            tt_model=self.transformer,
            get_torch_state_dict=self._torch_transformer.state_dict,
            model_name=os.path.basename(self.checkpoint_name),
            subfolder="transformer",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self._mesh_device.shape),
            is_fsdp=self.is_fsdp,
        )
        ttnn.synchronize_device(self._mesh_device)

    def _prepare_prompt_encoder(self) -> None:
        self._prompt_encoder.load_weights()
        ttnn.synchronize_device(self._mesh_device)

    def _prepare_vae(self) -> None:
        cache.load_model(
            tt_model=self._vae_decoder,
            get_torch_state_dict=self._torch_vae.state_dict,
            model_name=os.path.basename(self.checkpoint_name),
            subfolder="vae",
            parallel_config=self._vae_parallel_config,
            mesh_shape=tuple(self._mesh_device.shape),
        )
        ttnn.synchronize_device(self._mesh_device)

    @staticmethod
    def create_pipeline(
        *,
        mesh_device: ttnn.MeshDevice,
        dit_sp: tuple[int, int],
        dit_tp: tuple[int, int],
        encoder_tp: tuple[int, int],
        vae_tp: tuple[int, int],
        num_links: int,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        width: int = 1024,
        height: int = 1024,
        is_fsdp: bool = False,
        dynamic_load: bool = False,
        trace_warmup: bool = False,
    ) -> Flux2Pipeline:
        sp_factor, sp_axis = dit_sp
        tp_factor, tp_axis = dit_tp
        encoder_tp_factor, encoder_tp_axis = encoder_tp
        vae_tp_factor, vae_tp_axis = vae_tp

        dit_parallel_config = DiTGParallelConfigNoCFG(
            tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
            sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
        )
        encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=encoder_tp_factor, mesh_axis=encoder_tp_axis)
        )
        vae_parallel_config = VAEParallelConfig(
            tensor_parallel=ParallelFactor(factor=vae_tp_factor, mesh_axis=vae_tp_axis)
        )

        # logger.info(f"Mesh device shape: {mesh_device.shape}")
        logger.info(f"Parallel config: {dit_parallel_config}")
        logger.info(f"Encoder parallel config: {encoder_parallel_config}")
        logger.info(f"VAE parallel config: {vae_parallel_config}")

        return Flux2Pipeline(
            mesh_device=mesh_device,
            parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel_config=vae_parallel_config,
            topology=topology,
            num_links=num_links,
            width=width,
            height=height,
            is_fsdp=is_fsdp,
            dynamic_load=dynamic_load,
            trace_warmup=trace_warmup,
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
        traced: bool = False,
        profiler=None,
        profiler_iteration: int = 0,
    ) -> list[Image.Image]:
        prompt_count = len(prompts)

        sp_axis = self._parallel_config.sequence_parallel.mesh_axis

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

        latents_height = self._height // self._vae_scale_factor
        latents_width = self._width // self._vae_scale_factor
        transformer_batch_size = prompt_count * num_images_per_prompt
        spatial_sequence_length = (latents_height // self._patch_size) * (latents_width // self._patch_size)

        logger.info("encoding prompts...")

        with profiler("encoder", profiler_iteration) if profiler else nullcontext():
            self._prepare_prompt_encoder()

            if prompt_upsample_temperature is not None:
                prompts = self._prompt_encoder.upsample(
                    prompts,
                    max_length=224,  # TODO: this should be higher
                    temperature=prompt_upsample_temperature,
                )

            prompt_embeds, _mask = self._prompt_encoder.encode(
                prompts, num_images_per_prompt=num_images_per_prompt, sequence_length=512
            )
            _, prompt_sequence_length, _ = prompt_embeds.shape

        logger.info("preparing timesteps...")
        timesteps, sigmas = _schedule(
            self._scheduler,
            step_count=num_inference_steps,
            spatial_sequence_length=spatial_sequence_length,
        )

        guidance = torch.full([transformer_batch_size], fill_value=guidance_scale)

        logger.info("preparing latents...")

        if seed is not None:
            torch.manual_seed(seed)

        shape = [transformer_batch_size, self._num_channels_latents, latents_height, latents_width]
        latents = self._patchify(torch.randn(shape).permute(0, 2, 3, 1))

        self.ts._tt_prompt_embeds.update(prompt_embeds, traced, device=self._mesh_device)
        self.ts._tt_latents_step.update(latents, traced, mesh_axes=[None, sp_axis, None], device=self._mesh_device)
        self.ts._tt_guidance.update(guidance.unsqueeze(1), traced, device=self._mesh_device)

        logger.info("denoising...")
        self._prepare_transformer()

        with profiler("denoising", profiler_iteration) if profiler else nullcontext():
            for i, t in enumerate(tqdm.tqdm(timesteps)):
                with profiler(f"denoising_step_{i}", profiler_iteration) if profiler else nullcontext():
                    sigma_difference = (sigmas[i + 1] - sigmas[i]).item()

                    self.ts._tt_timestep.update(
                        torch.full([1, 1], fill_value=float(t)),
                        device=self._mesh_device,
                        dtype=ttnn.float32,
                        traced=traced,
                    )

                    self._step(
                        ts=self.ts,
                        sigma_difference=sigma_difference,
                        spatial_sequence_length=spatial_sequence_length,
                        prompt_sequence_length=prompt_sequence_length,
                        traced=traced,
                    )

        logger.info("decoding image...")

        with profiler("vae", profiler_iteration) if profiler else nullcontext():
            self._prepare_vae()
            tt_latents = self._ccl_manager.all_gather_persistent_buffer(
                self.ts.tt_latents_step,
                dim=1,
                mesh_axis=sp_axis,
                use_hyperparams=True,
            )

            torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])

            tt_latents = tensor.from_torch(torch_latents, device=self._mesh_device)
            tt_latents = self._vae_decoder.preprocess_and_unpatchify(
                tt_latents,
                height=self._height // self._vae_scale_factor,
                width=self._width // self._vae_scale_factor,
            )
            tt_decoded_output = self._vae_decoder.forward(tt_latents)
            decoded_output = ttnn.to_torch(ttnn.get_device_tensors(tt_decoded_output)[0]).permute(0, 3, 1, 2)

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
        guidance: ttnn.Tensor,
        spatial_rope_cos: ttnn.Tensor,
        spatial_rope_sin: ttnn.Tensor,
        prompt_rope_cos: ttnn.Tensor,
        prompt_rope_sin: ttnn.Tensor,
        spatial_sequence_length: int,
        prompt_sequence_length: int,
        traced: bool,
    ) -> ttnn.Tensor:
        return self.transformer.forward(
            spatial=latent,
            prompt=prompt,
            timestep=timestep,
            guidance=guidance,
            spatial_rope=(spatial_rope_cos, spatial_rope_sin),
            prompt_rope=(prompt_rope_cos, prompt_rope_sin),
            spatial_sequence_length=spatial_sequence_length,
            prompt_sequence_length=prompt_sequence_length,
            traced=traced,
        )

    def _step(
        self,
        *,
        ts: Flux2TransformerState,
        sigma_difference: float,
        spatial_sequence_length: int,
        prompt_sequence_length: int,
        traced: bool,
    ) -> None:
        noise_pred = self._step_inner(
            latent=ts.tt_latents_step,
            prompt=ts.tt_prompt_embeds,
            timestep=ts.tt_timestep,
            guidance=ts.tt_guidance,
            spatial_rope_cos=ts.tt_spatial_rope_cos,
            spatial_rope_sin=ts.tt_spatial_rope_sin,
            prompt_rope_cos=ts.tt_prompt_rope_cos,
            prompt_rope_sin=ts.tt_prompt_rope_sin,
            spatial_sequence_length=spatial_sequence_length,
            prompt_sequence_length=prompt_sequence_length,
            traced=traced,
        )

        ttnn.synchronize_device(self._mesh_device)  # Helps with accurate time profiling.
        ttnn.multiply_(noise_pred, sigma_difference)
        ttnn.add_(ts.tt_latents_step, noise_pred)

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


def _prepare_ids(*, height: int = 1, width: int = 1, text_sequence_length: int = 1) -> torch.Tensor:
    t = torch.arange(1)
    h = torch.arange(height)
    w = torch.arange(width)
    s = torch.arange(text_sequence_length)

    return torch.cartesian_prod(t, h, w, s)
