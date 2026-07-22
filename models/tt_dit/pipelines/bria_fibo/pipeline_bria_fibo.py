# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""End-to-end Bria FIBO text->image pipeline on the 2x2 Blackhole mesh.

Wires the three sp1/sp2/sp3 components into one denoise loop, mirroring
``models/tt_dit/pipelines/flux1/pipeline_flux1.py``:

* SmolLM3 text encoder (``SmolLM3TextEncoderWrapper``, tensor-parallel on tp_axis of the submesh),
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
from models.tt_dit.utils.tracing import Tracer

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
        num_links: int | None = None,
    ) -> BriaFiboPipelineConfig:
        mesh = tuple(mesh_shape)
        if len(mesh) != 2:
            msg = f"BriaFiboPipeline expects a 2D mesh, got {mesh}"
            raise ValueError(msg)

        # num_links is bounded by the ethernet channels physically available between adjacent
        # devices on a mesh axis. The 4x8 Blackhole Galaxy exposes only 2 channels per hop
        # (num_links=4 -> "Requested link index 2 out of bounds" fabric fatal); the 2x2 BH dev
        # mesh supports 4. Shape-driven default (the only hardware-dependent preset FIBO carries);
        # an explicit num_links= overrides. Unknown shapes fall back to the safe minimum of 1.
        if num_links is None:
            num_links = {(2, 2): 4, (4, 8): 2}.get(mesh, 1)

        sp_axis, tp_axis = 0, 1
        sp_factor = mesh[sp_axis]
        tp_factor = mesh[tp_axis]

        # Transformer: sequence-parallel on axis 0, tensor-parallel on axis 1, single cfg submesh.
        dit_parallel_config = DiTParallelConfig.from_tuples(
            cfg=(1, 0), sp=(sp_factor, sp_axis), tp=(tp_factor, tp_axis)
        )

        # Encoder: sequence-parallel across tokens on axis 1 (SP = mesh[1] = 8 on 4x8) x tensor-parallel
        # on axis 0 (TP = mesh[0] = 4), on the whole mesh (same submesh as the DiT). The token sequence is
        # padded to a fixed 1024 bucket and sharded over the SP axis; SmolLM3Attention all-gathers K/V over
        # the SP axis and Q/K/V/O over the TP axis. PCC-validated by
        # tests/encoders/smollm3/test_smollm3.py::test_smollm3_encoder_sp. On the 4x8 Galaxy this SP=8 x
        # TP=4 layout measured ~12.5 s/encode vs ~23.8 s for SP=4 x TP=8 (test_fibo_encode_perf).
        enc_sp_axis, enc_tp_axis = 1, 0
        encoder_parallel_config = EncoderParallelConfig.from_tuples(
            tp=(mesh[enc_tp_axis], enc_tp_axis),
            sp=(mesh[enc_sp_axis], enc_sp_axis),
        )

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
    def __init__(
        self, *, device: ttnn.MeshDevice, config: BriaFiboPipelineConfig, run_allocation_pass: bool = True
    ) -> None:
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

        # Denoise trace: ONE trace per step that runs both CFG forwards + the guidance combine. A
        # single trace per device is the pattern every tt_dit pipeline uses; two separate traces
        # sharing this submesh + CCLManager corrupt each other on replay (verified: two-tracer CFG
        # gave ~0 PCC, one trace is bit-exact). prep_run=True compiles the whole step at the real
        # prompt shape before capture. See docs/superpowers/specs/2026-07-09-fibo-denoise-trace-design.md.
        self._tracer = Tracer(self._traced_step, device=self._submesh, prep_run=True, clone_prep_inputs=False)
        self._captured_key: tuple | None = None  # (cfg_on, cond_prompt_len, uncond_prompt_len) last captured

        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.checkpoint_name, subfolder="scheduler")
        self._solver = EulerSolver()

        logger.info("creating text encoder...")
        # Encoder on the whole mesh (same submesh as the DiT): SP across tokens x TP. It gets its OWN
        # CCLManager (like the VAE, not the transformer's): the transformer's CCLManager carries the
        # resident denoise trace, and an untraced encoder all-gather on it between traced generations
        # would desync the baked ping-pong state. A second CCLManager on the same whole-mesh submesh is
        # the established pattern (see the VAE below); it does NOT hit the overlapping-submesh CCL hang
        # that separate cfg-parallel encoder submeshes did.
        self._encoder_ccl_manager = CCLManager(self._submesh, num_links=config.num_links, topology=config.topology)
        self._text_encoder = SmolLM3TextEncoderWrapper(
            config.checkpoint_name,
            device=self._submesh,
            ccl_manager=self._encoder_ccl_manager,
            parallel_config=config.encoder_parallel_config,
            pad_buckets=(1024,),
        )
        ttnn.synchronize_device(self._submesh)

        logger.info("creating VAE decoder...")
        # Decode the Wan 2.2 residual VAE on the same 2x2 submesh as the transformer. It gets its OWN
        # CCLManager (not the transformer's): the transformer's CCLManager carries the resident denoise
        # trace, and CCLManager ping-pong buffers/semaphores are stateful (a Python index flips each
        # call). An untraced VAE/latent-gather all-gather on that same manager between traced
        # generations would desync the state the trace baked, corrupting replay (verified: shared
        # manager -> gen 2+ diverged to ~0.13 PCC). Separate manager mirrors wan/ltx
        # (dit_ccl_manager vs vae_ccl_manager). Sharing the submesh still spreads the decode's
        # activations across all 4 devices.
        self._vae_ccl_manager = CCLManager(self._submesh, num_links=config.num_links, topology=config.topology)
        self._vae = WanVAEDecoderAdapter(
            checkpoint_name=config.checkpoint_name,
            parallel_config=config.vae_parallel_config,
            ccl_manager=self._vae_ccl_manager,
            height=config.height,
            width=config.width,
            num_frames=1,
            vae_t_chunk_size=None,  # full-T single pass (T=1)
        )

        self._image_processor = VaeImageProcessor(vae_scale_factor=_VAE_SCALE_FACTOR)

        # Allocation/warmup run (mirrors pipeline_flux1.py's __init__): run the full pipeline once
        # UNTRACED so every device buffer -- crucially the on-device VAE decode's -- is allocated BEFORE
        # any denoise trace is captured. Without it, the first traced generation captures the trace and
        # only then allocates the VAE buffers *during an active trace*, which corrupts the decoded image
        # (verified: no alloc-run -> traced image std 0.85/degenerate; with it -> bit-identical to
        # untraced). Cheap 2-step run; op compilation is amortized here too.
        #
        # run_allocation_pass=False skips it -- ONLY safe when the caller never captures a denoise trace
        # (e.g. an encode-only harness that just calls _encode). It avoids the full 2-step generation and
        # its on-device VAE decode (conv3d). The VAE/transformer are still constructed either way.
        if run_allocation_pass:
            logger.info("pipeline allocation run (untraced)...")
            self(prompt="", num_inference_steps=2, guidance_scale=2.0, traced=False, force_device_decode=True)
            ttnn.synchronize_device(self._submesh)

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
        traced: bool = False,
    ) -> list[Image.Image] | torch.Tensor:
        height = height if height is not None else self._height
        width = width if width is not None else self._width

        assert height % (_VAE_SCALE_FACTOR) == 0 and width % (_VAE_SCALE_FACTOR) == 0

        # CFG is active only when guidance_scale > 1 (matches the diffusers reference).
        do_cfg = guidance_scale > 1

        # 1-3. Encode, then build per-branch conditioning + schedule + latents.
        encoded = self._encode(prompt, negative_prompt, do_cfg=do_cfg)
        cond_branch, uncond_branch, timesteps, latent, spatial_sequence_length = self._prepare(
            encoded,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            seed=seed,
            latents=latents,
        )

        # 4. Denoise loop (CFG per step).
        latent = self._denoise(
            cond_branch, uncond_branch, timesteps, latent, spatial_sequence_length, guidance_scale, traced=traced
        )

        # 5. Return the pre-VAE latent (PCC gate) or decode to an image.
        if output_type == "latent":
            logger.info("returning pre-VAE latent...")
            return self._gather_latent(latent)
        logger.info("decoding image...")
        return self._decode_latents(
            latent, height=height, width=width, output_type=output_type, force_device_decode=force_device_decode
        )

    def _encode(self, prompt: str, negative_prompt: str, *, do_cfg: bool = True) -> tuple:
        """Encode the positive prompt (and, when CFG is active, the negative prompt) SEPARATELY.

        When ``do_cfg`` is False (``guidance_scale <= 1``) the negative branch is unused, so it is not
        encoded and the uncond entries are returned as ``None``.
        """
        logger.info("encoding prompts...")
        # SP x TP encoder on the whole mesh; positive and negative prompts are encoded sequentially,
        # each padded to the 1024 bucket and sequence-parallel across tokens.
        cond_embeds, cond_hidden_states = self._text_encoder.encode_prompt(prompt)
        if do_cfg:
            uncond_embeds, uncond_hidden_states = self._text_encoder.encode_prompt(negative_prompt)
        else:
            uncond_embeds, uncond_hidden_states = None, None
        return cond_embeds, cond_hidden_states, uncond_embeds, uncond_hidden_states

    def _prepare(
        self,
        encoded: tuple,
        *,
        height: int,
        width: int,
        num_inference_steps: int,
        seed: int,
        latents: torch.Tensor | None,
    ) -> tuple:
        """Build per-branch conditioning (RoPE + host->device uploads), schedule, and initial latents.

        Per-``__call__`` work (recurs on every call): rebuilds the 37->46 layer list, recomputes RoPE
        (``_pos_embed.forward`` in ``_prepare_branch``), uploads ``prompt`` + 2x46 layer tensors + RoPE +
        latents to device, recomputes the schedule. Only weights/submesh/``_pos_embed`` are built once
        (``__init__``), so this is real per-image cost -- a warmup run amortizes op compile, not this.
        """
        cond_embeds, cond_hidden_states, uncond_embeds, uncond_hidden_states = encoded
        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        latent_h = height // _VAE_SCALE_FACTOR
        latent_w = width // _VAE_SCALE_FACTOR
        spatial_sequence_length = latent_h * latent_w

        cond_layers = build_text_encoder_layers(cond_hidden_states, self._num_blocks)
        cond_branch = self._prepare_branch(cond_embeds, cond_layers, latent_h, latent_w, sp_axis)

        # uncond entries are None when CFG is off (guidance_scale <= 1); skip building that branch.
        uncond_branch = None
        if uncond_embeds is not None:
            uncond_layers = build_text_encoder_layers(uncond_hidden_states, self._num_blocks)
            uncond_branch = self._prepare_branch(uncond_embeds, uncond_layers, latent_h, latent_w, sp_axis)

        # Timesteps + solver schedule (dynamic shift on the image sequence length).
        logger.info("preparing timesteps...")
        self._scheduler.set_timesteps(
            sigmas=np.linspace(1.0, 1 / num_inference_steps, num_inference_steps),
            mu=_calculate_shift(spatial_sequence_length, self._scheduler),
        )
        self._solver.set_schedule(self._scheduler.sigmas.tolist())
        timesteps = self._scheduler.timesteps

        # Latents (no 2x2 pack): (1, 48, h, w) -> (1, h*w, 48), sequence-sharded on sp.
        logger.info("preparing latents...")
        latent = self._random_latents(height=height, width=width, seed=seed, latents=latents)

        return cond_branch, uncond_branch, timesteps, latent, spatial_sequence_length

    def _denoise(
        self,
        cond_branch: dict,
        uncond_branch: dict | None,
        timesteps,
        latent: ttnn.Tensor,
        spatial_sequence_length: int,
        guidance_scale: float,
        *,
        traced: bool = False,
    ) -> ttnn.Tensor:
        """Denoise loop, one Euler step per timestep. Syncs per step.

        With CFG (``uncond_branch is not None``): two per-branch forwards per step, combined as
        ``noise = uncond + guidance_scale * (cond - uncond)``. Without CFG (``guidance_scale <= 1`` ->
        ``uncond_branch is None``): a single cond forward per step (``noise = cond``), skipping the dead
        uncond forward + lerp.

        When ``traced`` is set, the transformer forward is captured/replayed via ``_denoise_traced``
        (the untraced body below is left byte-for-byte unchanged so it keeps backing the PCC gates).
        """
        submesh = self._submesh

        if traced:
            return self._denoise_traced(
                cond_branch, uncond_branch, timesteps, latent, spatial_sequence_length, guidance_scale
            )

        logger.info("denoising...")
        for i, t in enumerate(tqdm.tqdm(timesteps)):
            timestep = tt_tensor.from_torch(
                torch.full((1, 1), float(t), dtype=torch.bfloat16), device=submesh, dtype=ttnn.bfloat16
            )

            v_cond = self._run_transformer(latent, cond_branch, timestep, spatial_sequence_length)
            if uncond_branch is not None:
                v_uncond = self._run_transformer(latent, uncond_branch, timestep, spatial_sequence_length)
                # noise = uncond + guidance_scale * (cond - uncond)
                velocity = ttnn.lerp(v_uncond, v_cond, guidance_scale)
                ttnn.deallocate(v_cond)
                ttnn.deallocate(v_uncond)
            else:
                velocity = v_cond  # gs <= 1: noise = cond directly (uncond term cancels)
            ttnn.deallocate(timestep)

            new_latent = self._solver.step(step=i, latent=latent, velocity_pred=velocity)
            ttnn.deallocate(velocity)
            ttnn.deallocate(latent)
            latent = new_latent

            ttnn.synchronize_device(submesh)
        return latent

    def _denoise_traced(
        self,
        cond_branch: dict,
        uncond_branch: dict | None,
        timesteps,
        latent: ttnn.Tensor,
        spatial_sequence_length: int,
        guidance_scale: float,
    ) -> ttnn.Tensor:
        """Traced denoise loop: one captured step (both forwards + guidance combine) replayed per step.

        The step is captured on the first call (``Tracer.prep_run`` compiles it first) and replayed
        thereafter; only the changing ``latent`` and ``timestep`` are copied into the captured input
        buffers each step -- the constant conditioning dicts keep the same tensors, so the tracer skips
        them. The Euler ``solver.step`` (per-step scalar) stays untraced. One sync at the end (not per
        step) so the steps pipeline.
        """
        logger.info("denoising (traced)...")
        submesh = self._submesh
        self._ensure_trace(cond_branch, uncond_branch)

        # Pass the fresh conditioning ONLY on the call that captures; on every replay pass the captured
        # buffers (self._tracer.inputs[...]) so the tracer skips re-copying them. This mirrors flux1 and
        # is REQUIRED: ttnn.copy of the (replicated + sharded) conditioning into the captured buffers on
        # replay corrupts them (verified -- reusing captured buffers holds PCC 1.0 across generations,
        # re-copying fresh conditioning gives ~0 PCC). Only latent/timestep genuinely change and those
        # copy correctly each step. A fixed prompt keeps the trace (see _ensure_trace); a changed prompt
        # releases + recaptures, so `capturing` is True again and the fresh conditioning is captured.
        capturing = not self._tracer.trace_captured
        for i, t in enumerate(tqdm.tqdm(timesteps)):
            timestep = tt_tensor.from_torch(
                torch.full((1, 1), float(t), dtype=torch.bfloat16), device=submesh, dtype=ttnn.bfloat16
            )
            fresh = capturing and i == 0
            velocity = self._tracer(
                latent=latent,
                timestep=timestep,
                cond=cond_branch if fresh else self._tracer.inputs["cond"],
                uncond=uncond_branch if fresh else self._tracer.inputs["uncond"],
                spatial_sequence_length=spatial_sequence_length,
                guidance_scale=guidance_scale,
                traced=True,
                tracer_blocking_execution=False,
            )
            # Trace replay can clobber the tensor object passed as `latent`; the captured input buffer
            # is the safe handle for the solver step.
            latent = self._tracer.inputs["latent"]
            latent = self._solver.step(step=i, latent=latent, velocity_pred=velocity)

        ttnn.synchronize_device(submesh)
        return latent

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

    def _traced_step(
        self,
        *,
        latent: ttnn.Tensor,
        timestep: ttnn.Tensor,
        cond: dict,
        uncond: dict | None,
        spatial_sequence_length: int,
        guidance_scale: float,
    ) -> ttnn.Tensor:
        """The unit captured by the Tracer: both CFG forwards + the guidance combine -> velocity.

        Kept as ONE trace (not one per branch): a single trace per device is the working tt_dit
        pattern; two interleaved traces on this submesh corrupt each other on replay. All inputs are
        Tracer-valid -- tensors, dicts/lists/tuples of tensors, and int/float scalars; the per-branch
        ``prompt_sequence_length`` (in the dicts), ``spatial_sequence_length`` and ``guidance_scale``
        are constant for a fixed prompt+resolution, satisfying the tracer's scalar-equality check.
        """
        v_cond = self._run_transformer(latent, cond, timestep, spatial_sequence_length)
        if uncond is None:
            return v_cond  # gs <= 1: noise = cond directly (uncond term cancels)
        v_uncond = self._run_transformer(latent, uncond, timestep, spatial_sequence_length)
        # noise = uncond + guidance_scale * (cond - uncond)
        return ttnn.lerp(v_uncond, v_cond, guidance_scale)

    def _ensure_trace(self, cond_branch: dict, uncond_branch: dict | None) -> None:
        """Release + recapture the trace when the CFG mode or a prompt token length changed.

        A trace bakes tensor shapes and the cfg-on/off structure, so it is reusable only while those
        are fixed (``spatial_sequence_length`` is fixed by resolution). Same prompt+gs across runs ->
        reuse; a change -> release+recapture (amortized over that generation's steps).
        """
        key = (
            uncond_branch is not None,
            cond_branch["prompt_sequence_length"],
            uncond_branch["prompt_sequence_length"] if uncond_branch is not None else None,
        )
        if self._captured_key is not None and self._captured_key != key:
            self.release_traces()
        self._captured_key = key

    def release_traces(self) -> None:
        """Release the denoise trace (call at teardown; mirrors pipeline_wan.py's release_traces)."""
        self._tracer.release_trace()
        self._captured_key = None

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
        # VAE CCLManager, not the transformer's: this untraced all-gather must not touch the ping-pong
        # state carrying the resident denoise trace (see the VAE CCLManager note in __init__).
        latent = self._vae_ccl_manager.all_gather_persistent_buffer(
            latent, dim=1, mesh_axis=sp_axis, use_hyperparams=True
        )
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

        # Gather the sequence-sharded latent, then rebuild BCTHW (T=1) for the VAE. Uses the VAE
        # CCLManager (not the transformer's) so it doesn't disturb the resident denoise trace.
        ttnn.synchronize_device(submesh)
        latent = self._vae_ccl_manager.all_gather_persistent_buffer(
            latent, dim=1, mesh_axis=sp_axis, use_hyperparams=True
        )
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
