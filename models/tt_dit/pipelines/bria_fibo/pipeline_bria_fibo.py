# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""End-to-end Bria FIBO text->image pipeline on the 2x2 Blackhole mesh.

Wires the three sp1/sp2/sp3 components into one denoise loop, mirroring
``models/tt_dit/pipelines/flux1/pipeline_flux1.py``:

* SmolLM3 text encoder (``SmolLM3TextEncoderWrapper``, replicated on the submesh),
* ``BriaFiboTransformer`` denoiser (sp=2, tp=2) + ``EulerSolver`` flow-match step,
* Wan 2.2 residual VAE decoder (``WanVAEDecoderAdapter``).

FIBO deltas vs flux1 (see the sub-project 4 design spec):

* Latents are **not** 2x2-packed (``in_channels == VAE z_dim == 48``); the flux-style pack/unpack
  is replaced by a plain permute/reshape (``_pack_latents_no_patch`` / ``_unpack_latents_no_patch``).
* CFG runs as two **unpadded per-branch** forwards (positive / negative at their true token lengths),
  combined with ``noise = uncond + guidance_scale * (cond - uncond)``. This avoids the reference's
  padding attention-mask (the tt transformer has none) without touching the validated transformer.
* Per-block caption conditioning: SmolLM3's 37 hidden states are stretched to the transformer's 46
  blocks via ``build_text_encoder_layers``.

This first correctness pass runs UNTRACED (tracing is a documented follow-up).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import tqdm
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from loguru import logger

import ttnn
from models.tt_dit.layers.module import LoadingError
from models.tt_dit.models.transformers.transformer_bria_fibo import BriaFiboCheckpoint
from models.tt_dit.models.vae.vae_wan2_1 import WanVAEDecoderAdapter
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.bria_fibo.text_encoder import SmolLM3TextEncoderWrapper, build_text_encoder_layers
from models.tt_dit.pipelines.cfg import create_submeshes
from models.tt_dit.solvers import EulerSolver
from models.tt_dit.utils import tensor as tt_tensor

if TYPE_CHECKING:
    from PIL import Image

_VAE_SCALE_FACTOR = 16


@dataclass(frozen=True, kw_only=True)
class BriaFiboPipelineConfig:
    topology: ttnn.Topology
    num_links: int

    dit_parallel_config: DiTParallelConfig
    encoder_parallel_config: EncoderParallelConfig
    vae_parallel_config: VaeHWParallelConfig

    height: int
    width: int
    checkpoint_name: str

    @classmethod
    def default(
        cls,
        *,
        mesh_shape: ttnn.MeshShape,
        checkpoint_name: str,
        height: int = 1024,
        width: int = 1024,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        num_links: int = 1,
    ) -> BriaFiboPipelineConfig:
        mesh = tuple(mesh_shape)
        if len(mesh) != 2:
            msg = f"BriaFiboPipeline expects a 2D mesh, got {mesh}"
            raise ValueError(msg)

        sp_axis, tp_axis = 0, 1
        sp_factor = mesh[sp_axis]
        tp_factor = mesh[tp_axis]

        # Transformer: sequence-parallel on axis 0, tensor-parallel on axis 1, single cfg submesh.
        dit_parallel_config = DiTParallelConfig.from_tuples(
            cfg=(1, 0), sp=(sp_factor, sp_axis), tp=(tp_factor, tp_axis)
        )

        # Encoder: tp factor 1 -> fully replicated across the submesh (no CCL); mesh_axis is unused
        # when the factor is 1 (SmolLM3Context sets tp_axis=None), so replication is exact.
        encoder_parallel_config = EncoderParallelConfig.from_tuple((1, tp_axis))

        # VAE: height/width parallel matched to the physical mesh (mirrors wan's (2,2) BH preset:
        # height on the tp axis, width on the sp axis). The Wan 2.2 residual decoder decodes on the full
        # 2x2 submesh (halo/CCL exchange across devices), which distributes the decode's activations over
        # all 4 devices. Verified to run on-device and produce a correct-range, non-degenerate 1024x1024
        # image (test_fibo_pipeline_smoke, force_device_decode=True); the golden PCC-vs-host-reference is
        # gated by test_fibo_pipeline_vae_decode_on_device (native res, run on-demand). Requires the
        # ``decoder_base_dim`` weight-prep fix in ``vae_wan2_1.py`` (without it conv_in loaded (1728,640)
        # vs (1728,1024) and fell back to host).
        vae_parallel_config = VaeHWParallelConfig.from_tuples(height=(tp_factor, tp_axis), width=(sp_factor, sp_axis))

        return cls(
            topology=topology,
            num_links=num_links,
            dit_parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel_config=vae_parallel_config,
            height=height,
            width=width,
            checkpoint_name=checkpoint_name,
        )


class BriaFiboPipeline:
    def __init__(self, *, device: ttnn.MeshDevice, config: BriaFiboPipelineConfig) -> None:
        self._mesh_device = device
        self._config = config
        self._parallel_config = config.dit_parallel_config
        self._height = config.height
        self._width = config.width

        logger.info(f"FIBO parallel config: {config.dit_parallel_config}")
        logger.info(f"Original mesh shape: {tuple(device.shape)}")

        # cfg factor 1 -> a single submesh spanning the whole mesh (sp x tp).
        self._submesh = create_submeshes(device, config.dit_parallel_config)[0]
        logger.info(f"Created submesh with shape {tuple(self._submesh.shape)}")

        self._ccl_manager = CCLManager(self._submesh, num_links=config.num_links, topology=config.topology)

        logger.info("creating TT-NN transformer...")
        checkpoint = BriaFiboCheckpoint(config.checkpoint_name)
        self._checkpoint = checkpoint
        self._transformer = checkpoint.build(ccl_manager=self._ccl_manager, parallel_config=config.dit_parallel_config)
        self._pos_embed = checkpoint.pos_embed
        self._in_channels = checkpoint.in_channels
        self._num_blocks = checkpoint._config.num_layers + checkpoint._config.num_single_layers
        ttnn.synchronize_device(self._submesh)

        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.checkpoint_name, subfolder="scheduler")
        self._solver = EulerSolver()

        logger.info("creating text encoder...")
        self._text_encoder = SmolLM3TextEncoderWrapper(
            config.checkpoint_name,
            device=self._submesh,
            ccl_manager=self._ccl_manager,
            parallel_config=config.encoder_parallel_config,
        )
        ttnn.synchronize_device(self._submesh)

        logger.info("creating VAE decoder...")
        # Decode the Wan 2.2 residual VAE on the same 2x2 submesh as the transformer, reusing its
        # CCLManager for the decoder's halo/all-gather exchange. Sharing the submesh (rather than a
        # dedicated 1-device submesh) spreads the decode's activations across all 4 devices, which
        # coexists with the resident transformer/encoder shards.
        self._vae = WanVAEDecoderAdapter(
            checkpoint_name=config.checkpoint_name,
            parallel_config=config.vae_parallel_config,
            ccl_manager=self._ccl_manager,
            height=config.height,
            width=config.width,
            num_frames=1,
            vae_t_chunk_size=None,  # full-T single pass (T=1)
        )

        self._image_processor = VaeImageProcessor(vae_scale_factor=_VAE_SCALE_FACTOR)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int = 0,
        latents: torch.Tensor | None = None,
        output_type: str = "pil",
        force_device_decode: bool = False,
    ) -> list[Image.Image] | torch.Tensor:
        height = height if height is not None else self._height
        width = width if width is not None else self._width
        submesh = self._submesh
        sp_axis = self._parallel_config.sequence_parallel.mesh_axis

        assert height % (_VAE_SCALE_FACTOR) == 0 and width % (_VAE_SCALE_FACTOR) == 0

        latent_h = height // _VAE_SCALE_FACTOR
        latent_w = width // _VAE_SCALE_FACTOR
        spatial_sequence_length = latent_h * latent_w

        # 1. Encode positive and negative prompts SEPARATELY (per-branch, true token lengths).
        logger.info("encoding prompts...")
        cond_embeds, cond_hidden_states = self._text_encoder.encode_prompt(prompt)
        uncond_embeds, uncond_hidden_states = self._text_encoder.encode_prompt(negative_prompt)
        cond_layers = build_text_encoder_layers(cond_hidden_states, self._num_blocks)
        uncond_layers = build_text_encoder_layers(uncond_hidden_states, self._num_blocks)

        cond_branch = self._prepare_branch(cond_embeds, cond_layers, latent_h, latent_w, sp_axis)
        uncond_branch = self._prepare_branch(uncond_embeds, uncond_layers, latent_h, latent_w, sp_axis)

        # 2. Timesteps + solver schedule (dynamic shift on the image sequence length).
        logger.info("preparing timesteps...")
        self._scheduler.set_timesteps(
            sigmas=np.linspace(1.0, 1 / num_inference_steps, num_inference_steps),
            mu=_calculate_shift(spatial_sequence_length, self._scheduler),
        )
        self._solver.set_schedule(self._scheduler.sigmas.tolist())
        timesteps = self._scheduler.timesteps

        # 3. Latents (no 2x2 pack): (1, 48, h, w) -> (1, h*w, 48), sequence-sharded on sp.
        #    ``latents`` (if given) injects a fixed initial noise in the same packed layout, so a
        #    reference comparison can feed BOTH pipelines identical noise (host/device RNG differ).
        logger.info("preparing latents...")
        latent = self._random_latents(height=height, width=width, seed=seed, latents=latents)

        # 4. Denoise loop: two per-branch forwards per step, combined via CFG.
        logger.info("denoising...")
        for i, t in enumerate(tqdm.tqdm(timesteps)):
            timestep = tt_tensor.from_torch(
                torch.full((1, 1), float(t), dtype=torch.bfloat16), device=submesh, dtype=ttnn.bfloat16
            )

            v_cond = self._run_transformer(latent, cond_branch, timestep, spatial_sequence_length)
            v_uncond = self._run_transformer(latent, uncond_branch, timestep, spatial_sequence_length)

            # noise = uncond + guidance_scale * (cond - uncond)
            velocity = ttnn.lerp(v_uncond, v_cond, guidance_scale)
            ttnn.deallocate(v_cond)
            ttnn.deallocate(v_uncond)
            ttnn.deallocate(timestep)

            new_latent = self._solver.step(step=i, latent=latent, velocity_pred=velocity)
            ttnn.deallocate(velocity)
            ttnn.deallocate(latent)
            latent = new_latent

            ttnn.synchronize_device(submesh)

        # 5. Return the pre-VAE latent (PCC gate) or decode to an image.
        if output_type == "latent":
            logger.info("returning pre-VAE latent...")
            return self._gather_latent(latent)
        logger.info("decoding image...")
        return self._decode_latents(
            latent, height=height, width=width, output_type=output_type, force_device_decode=force_device_decode
        )

    def _prepare_branch(
        self,
        prompt_embeds: torch.Tensor,
        text_encoder_layers: list[torch.Tensor],
        latent_h: int,
        latent_w: int,
        sp_axis: int,
    ) -> dict:
        """Move one CFG branch's conditioning + RoPE to the submesh (reused across all steps)."""
        submesh = self._submesh
        prompt_sequence_length = prompt_embeds.shape[1]

        prompt = tt_tensor.from_torch(prompt_embeds.to(torch.bfloat16), device=submesh)
        layers = [tt_tensor.from_torch(layer.to(torch.bfloat16), device=submesh) for layer in text_encoder_layers]

        # RoPE: flux-style ids (txt = zeros, img = pixel grid), split into prompt / spatial parts.
        # EmbedND is per-row, so the spatial part is identical across branches; the txt part depends
        # only on the branch's token count. Computed per-branch for clarity.
        text_ids = torch.zeros(prompt_sequence_length, 3)
        image_ids = _latent_image_ids(height=latent_h, width=latent_w)
        ids = torch.cat((text_ids, image_ids), dim=0)
        rope_cos, rope_sin = self._pos_embed.forward(ids)

        spatial_rope = (
            tt_tensor.from_torch(rope_cos[prompt_sequence_length:], device=submesh, mesh_axes=[sp_axis, None]),
            tt_tensor.from_torch(rope_sin[prompt_sequence_length:], device=submesh, mesh_axes=[sp_axis, None]),
        )
        prompt_rope = (
            tt_tensor.from_torch(rope_cos[:prompt_sequence_length], device=submesh),
            tt_tensor.from_torch(rope_sin[:prompt_sequence_length], device=submesh),
        )

        return {
            "prompt": prompt,
            "layers": layers,
            "prompt_sequence_length": prompt_sequence_length,
            "spatial_rope": spatial_rope,
            "prompt_rope": prompt_rope,
        }

    def _run_transformer(
        self, latent: ttnn.Tensor, branch: dict, timestep: ttnn.Tensor, spatial_sequence_length: int
    ) -> ttnn.Tensor:
        return self._transformer.forward(
            spatial=latent,
            prompt=branch["prompt"],
            timestep=timestep,
            text_encoder_layers=branch["layers"],
            spatial_rope=branch["spatial_rope"],
            prompt_rope=branch["prompt_rope"],
            spatial_sequence_length=spatial_sequence_length,
            prompt_sequence_length=branch["prompt_sequence_length"],
        )

    def _random_latents(
        self, *, height: int, width: int, seed: int, latents: torch.Tensor | None = None
    ) -> ttnn.Tensor:
        latent_h = height // _VAE_SCALE_FACTOR
        latent_w = width // _VAE_SCALE_FACTOR
        packed_shape = (1, latent_h * latent_w, self._in_channels)

        if latents is None:
            torch.manual_seed(seed)
            latents = torch.randn(1, self._in_channels, latent_h, latent_w, dtype=torch.float32)
            # No 2x2 pack: (1, C, h, w) -> (1, h*w, C).
            latents = latents.permute(0, 2, 3, 1).reshape(*packed_shape)
        elif tuple(latents.shape) != packed_shape:
            # Injected noise must already be in the reference's packed ``_pack_latents_no_patch`` layout.
            msg = (
                f"injected `latents` must be packed {packed_shape} (1, h*w, in_channels) to match the "
                f"reference latent layout, got {tuple(latents.shape)}"
            )
            raise ValueError(msg)

        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        return tt_tensor.from_torch(latents.to(torch.bfloat16), device=self._submesh, mesh_axes=[None, sp_axis, None])

    def _gather_latent(self, latent: ttnn.Tensor) -> torch.Tensor:
        """All-gather the sp-sharded pre-VAE latent to a host ``(1, h*w, 48)`` float32 tensor.

        Matches the reference ``BriaFiboPipeline.__call__(output_type="latent")`` layout: the reference
        returns ``latents`` *without* unpacking (the packed ``(1, h*w, 48)`` form), so we do the same.
        """
        submesh = self._submesh
        sp_axis = self._parallel_config.sequence_parallel.mesh_axis

        ttnn.synchronize_device(submesh)
        latent = self._ccl_manager.all_gather_persistent_buffer(latent, dim=1, mesh_axis=sp_axis, use_hyperparams=True)
        torch_latents = ttnn.to_torch(ttnn.get_device_tensors(latent)[0])  # (1, h*w, 48)
        return torch_latents.to(torch.float32)

    def _decode_latents(
        self,
        latent: ttnn.Tensor,
        *,
        height: int,
        width: int,
        output_type: str,
        force_device_decode: bool = False,
    ) -> list[Image.Image]:
        submesh = self._submesh
        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        latent_h = height // _VAE_SCALE_FACTOR
        latent_w = width // _VAE_SCALE_FACTOR

        # Gather the sequence-sharded latent, then rebuild BCTHW (T=1) for the VAE.
        ttnn.synchronize_device(submesh)
        latent = self._ccl_manager.all_gather_persistent_buffer(latent, dim=1, mesh_axis=sp_axis, use_hyperparams=True)
        torch_latents = ttnn.to_torch(ttnn.get_device_tensors(latent)[0])

        # (1, h*w, 48) -> (1, 48, h, w) (inverse of the no-patch pack) -> (1, 48, 1, h, w).
        torch_latents = torch_latents.reshape(1, latent_h, latent_w, self._in_channels).permute(0, 3, 1, 2)
        torch_latents = torch_latents.unsqueeze(2).to(torch.float32)

        decoded = self._decode_vae(torch_latents, force_device_decode=force_device_decode)  # (1, 3, 1, H, W) in [-1, 1]
        decoded = decoded.squeeze(2)  # (1, 3, H, W)

        image = self._image_processor.postprocess(decoded.float(), output_type=output_type)
        return image

    def _decode_vae(self, latents_bcthw: torch.Tensor, *, force_device_decode: bool = False) -> torch.Tensor:
        """Decode the (denormalized internally) BCTHW latent to RGB in [-1, 1].

        Primary path: the on-device ``WanVAEDecoderAdapter`` on the 2x2 submesh (hw-parallel residual
        decode; returns raw 12-ch patchified pixels, so we ``unpatchify(patch_size=2)`` + ``clamp`` to
        match sp3's ``test_vae`` post-processing). Verified to run + produce a correct-range image; the
        golden PCC-vs-host-reference is gated by test_fibo_pipeline_vae_decode_on_device.

        The ``LoadingError`` fallback to the host reference ``AutoencoderKLWan.decode`` is now only a
        defensive net (the historical failure -- ``decoder.conv_in.weight`` (1728, 640) vs (1728, 1024)
        -- was the adapter omitting ``decoder_base_dim``, fixed in ``vae_wan2_1.py``). Pass
        ``force_device_decode=True`` to re-raise instead of falling back, proving the on-device path.
        Any non-``LoadingError`` failure (OOM, real device/shape bug) always propagates.
        """
        try:
            out = self._vae.decode(latents_bcthw, output_type="pt")  # (1, C>=12, 1, H/2, W/2)
            out = out[:, : self._vae.config.out_channels]  # trim any conv channel padding to 12
            out = unpatchify(out, patch_size=self._vae.config.patch_size)  # (1, 3, 1, H, W)
            return torch.clamp(out, min=-1.0, max=1.0)
        except LoadingError as e:
            if force_device_decode:
                raise
            logger.warning(
                f"on-device VAE weight load failed ({type(e).__name__}: {e}); falling back to host torch VAE"
            )
            return self._host_decode_vae(latents_bcthw)

    def _host_decode_vae(self, latents_bcthw: torch.Tensor) -> torch.Tensor:
        vae = self._vae._torch_vae
        z_dim = vae.config.z_dim
        latents = latents_bcthw.to(vae.dtype)
        mean = torch.tensor(vae.config.latents_mean, dtype=vae.dtype).view(1, z_dim, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, dtype=vae.dtype).view(1, z_dim, 1, 1, 1)
        latents = latents * std + mean  # matches reference: latent / (1/std) + mean
        out = vae.decode(latents, return_dict=False)[0]  # applies unpatchify + clamp internally
        return out.to(torch.float32)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/flux/pipeline_flux.py
def _latent_image_ids(*, height: int, width: int) -> torch.Tensor:
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    h, w, c = latent_image_ids.shape
    return latent_image_ids.reshape(h * w, c)


def _calculate_shift(image_seq_len: int, scheduler: FlowMatchEulerDiscreteScheduler) -> float:
    base_seq_len = scheduler.config.get("base_image_seq_len", 256)
    max_seq_len = scheduler.config.get("max_image_seq_len", 4096)
    base_shift = scheduler.config.get("base_shift", 0.5)
    max_shift = scheduler.config.get("max_shift", 1.15)

    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b
