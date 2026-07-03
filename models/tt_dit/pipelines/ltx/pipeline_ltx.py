# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2 audio-video generation pipeline for tt_dit: on-device Gemma text encoding,
LTX-2 sigma schedule, joint AV denoise (Euler + multi-modal guidance), and on-device
VAE video + audio decode.

Reference: LTX-2/packages/ltx-pipelines/ + Wan pipeline_wan.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file

import ttnn

from ...encoders.gemma.encoder_pair import GemmaTokenizerEncoderPair
from ...models.transformers.ltx.rope_ltx import LTXRopeType, precompute_freqs_cis, reshape_interleaved_to_bhnd
from ...models.transformers.ltx.transformer_ltx import LTXTransformerModel
from ...models.upsampler.latent_upsampler_ltx import LTXLatentUpsampler
from ...models.vae.vae_ltx import LTXVideoDecoder
from ...parallel.config import (
    AudioTCParallelConfig,
    DiTParallelConfig,
    EncoderParallelConfig,
    ParallelFactor,
    VaeHWParallelConfig,
)
from ...parallel.manager import CCLManager
from ...utils import cache as cache_module
from ...utils.conv3d import conv3d_blocking_hash
from ...utils.fuse_loras import LoraSpec, fuse_loras_into
from ...utils.mochi import get_rot_transformation_mat
from ...utils.patchifiers import (
    AudioLatentShape,
    VideoLatentShape,
    VideoPixelShape,
    audio_get_patch_grid_bounds,
    get_pixel_coords,
    video_get_patch_grid_bounds,
)
from ...utils.tensor import bf16_tensor, bf16_tensor_2dshard
from ...utils.tracing import Tracer
from ...utils.video import Audio, export_video_audio

LTX_UPSAMPLER_HF_REF = "Lightricks/LTX-2.3:ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

# LTX-2 VAE compression ratios (pixel -> latent). Used throughout to map pixel
# dims to the latent token grid. NOTE: the TILE size used for SP padding (also 32)
# is a separate concept — do NOT replace `32 * sp_factor` padding math with these.
TEMPORAL_COMPRESSION = 8
SPATIAL_COMPRESSION = 32

# Default negative prompt (inlined from the LTX-2 reference
# ``ltx_pipelines.utils.constants.DEFAULT_NEGATIVE_PROMPT`` so the pipeline has no
# runtime dependency on the reference package).
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)


def latent_grid(num_frames: int, height: int, width: int) -> tuple[int, int, int]:
    """Map pixel dims to the LTX latent token grid ``(latent_frames, latent_h, latent_w)``."""
    latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
    return latent_frames, height // SPATIAL_COMPRESSION, width // SPATIAL_COMPRESSION


@dataclass
class TransformerState:
    model: LTXTransformerModel
    cache_name: str
    lora_specs: list[LoraSpec] = field(default_factory=list)
    state_dict_provider: Callable[[], dict[str, torch.Tensor]] | None = None


# =============================================================================
# Scheduler
# =============================================================================

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


def compute_sigmas(
    steps: int,
    num_tokens: int = MAX_SHIFT_ANCHOR,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> torch.FloatTensor:
    """Compute the LTX-2 sigma schedule: noise levels from ~1.0 down to ~``terminal``
    with a token-count-dependent shift. Returns a ``(steps + 1,)`` tensor."""
    sigmas = torch.linspace(1.0, 0.0, steps + 1)

    # Adaptive shift based on token count
    mm = (max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR)
    b = base_shift - mm * BASE_SHIFT_ANCHOR
    sigma_shift = num_tokens * mm + b

    # Exponential shift
    sigmas = torch.where(
        sigmas != 0,
        math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1)),
        0,
    )

    # Stretch to terminal
    if stretch:
        non_zero_mask = sigmas != 0
        non_zero_sigmas = sigmas[non_zero_mask]
        one_minus_z = 1.0 - non_zero_sigmas
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        stretched = 1.0 - (one_minus_z / scale_factor)
        sigmas[non_zero_mask] = stretched

    return sigmas.to(torch.float32)


# =============================================================================
# Diffusion step
# =============================================================================


def euler_step(
    sample: torch.Tensor,
    denoised: torch.Tensor,
    sigma: float,
    sigma_next: float,
) -> torch.Tensor:
    """First-order Euler diffusion step:
    ``x_next = x + (x - denoised) / sigma * (sigma_next - sigma)``."""
    dt = sigma_next - sigma
    velocity = (sample.float() - denoised.float()) / sigma
    return (sample.float() + velocity * dt).to(sample.dtype)


# =============================================================================
# Pipeline
# =============================================================================


class LTXPipeline:
    """
    LTX-2 text-to-audio-video generation pipeline.

    Usage:
        pipeline = LTXPipeline.create_pipeline(
            mesh_device,
            checkpoint_name="Lightricks/LTX-2.3:ltx-2.3-22b-dev.safetensors",
            gemma_path="google/gemma-3-12b-it-qat-q4_0-unquantized",
            num_frames=33,
            height=480,
            width=832,
        )
        output_path = pipeline.generate(prompt="A cat playing piano", output_path="out.mp4")
    """

    HAS_UPSAMPLER: bool = False

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        parallel_config: DiTParallelConfig,
        ccl_manager: CCLManager,
        *,
        checkpoint_name: str | None = None,
        gemma_path: str | None = None,
        vae_parallel_config: VaeHWParallelConfig | None = None,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        mode: str = "av",
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: float = 1000.0,
        is_fsdp: bool = False,
        dynamic_load: bool = False,
        num_frames: int = 0,
        height: int = 0,
        width: int = 0,
        run_warmup: bool = False,
        traced: bool = False,
        extra_transformer_variants: list[tuple[str, list[LoraSpec]]] | None = None,
    ):
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        # Text-encoder TP spans the full width of mesh axis 1 (T5 encoder pattern):
        # TP=1 on 1x1, 4 on 2x4, 8 on 4x8. Shared by the Gemma encoder and the
        # embeddings connectors so both shard identically.
        self.encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=self.mesh_device.shape[1], mesh_axis=1),
        )
        # Capture the DiT forward as a ttnn trace on the first step, replay after.
        self._traced = traced
        # One Tracer per fixed shape ("s1"/"s2"); resident across generate(), freed by release_traces().
        self._tracers: dict[str, Tracer] = {}
        # Per-trace device constants (rope/masks/cross-PE). A ttnn trace bakes absolute tensor
        # addresses, so every held input is allocated up front and never rebuilt.
        self._trace_consts: dict[str, tuple] = {}
        # Per-trace SP-sharded latent buffers + padding masks, held for the traced loop's
        # on-device Euler (nothing freed/reallocated per step).
        self._trace_latents: dict[str, tuple] = {}
        # Shared prompt buffer, built on the first traced step rather than pre-allocated:
        # a low-address pre-alloc would overlap a video activation.
        self._trace_prompt: dict[str, tuple] = {}
        if ccl_manager.topology == ttnn.Topology.Linear:
            self.vae_ccl_manager = ccl_manager
        else:
            self.vae_ccl_manager = CCLManager(
                mesh_device, num_links=ccl_manager.num_links, topology=ttnn.Topology.Linear
            )
        if vae_parallel_config is None:
            vae_parallel_config = VaeHWParallelConfig(
                height_parallel=parallel_config.tensor_parallel,
                width_parallel=parallel_config.sequence_parallel,
            )
        self.vae_parallel_config = vae_parallel_config
        self.mode = mode

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.cross_attention_dim = cross_attention_dim
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos or [20, 2048, 2048]
        self.timestep_scale_multiplier = timestep_scale_multiplier

        self.is_fsdp = is_fsdp
        self.dynamic_load = dynamic_load
        self._init_num_frames = num_frames
        self._init_height = height
        self._init_width = width

        self.checkpoint_name: str | None = (
            LTXPipeline._resolve_checkpoint_file(checkpoint_name) if checkpoint_name else None
        )
        # On-device Gemma text encoder + embeddings connectors. The encoder resolves its own
        # reference and owns its cache name (T5 pattern); reuse its resolved dir for the host
        # reference path. Lazy: no modules are built until the first ensure_loaded().
        self.gemma_encoder_pair = GemmaTokenizerEncoderPair(
            gemma_path,
            mesh_device=self.mesh_device,
            ccl_manager=self.vae_ccl_manager,
            parallel_config=self.encoder_parallel_config,
            checkpoint_name=self.checkpoint_name,
            mode=self.mode,
            dynamic_load=self.dynamic_load,
        )
        self.gemma_path: str | None = self.gemma_encoder_pair.gemma_path

        # ``self.transformer`` always points at the active variant, set via
        # ``_prepare_transformer(idx)``.
        self.transformer: LTXTransformerModel | None = None
        self.transformer_states: list[TransformerState] = []
        self.vae_decoder = None
        self.upsampler: LTXLatentUpsampler | None = None
        # On-device audio decode chain (Stage A mel-VAE + Stage B/C vocoder+BWE).
        # Shells are built at `_instantiate_modules`; weights are loaded via
        # `_prepare_audio_decoder` (disk-cached, fast on warm hits).
        self.tt_audio_decoder = None
        self.tt_vocoder_with_bwe = None

        if self.checkpoint_name is not None:
            self._load_config_from_checkpoint()
            self._instantiate_modules(extra_transformer_variants or [])
            self._register_coresident_exclusions()
            self._prime_caches()
            if run_warmup and num_frames > 0 and height > 0 and width > 0:
                self.warmup_buffers(num_frames=num_frames, height=height, width=width)

    def _traced_step(self, trace_key: str, fn: Callable, *, capture_inputs: dict, replay_inputs: dict):
        """Capture ``fn`` as a ttnn trace on the first call for ``trace_key``; replay after.

        Capture passes the full input set; replay passes only the inputs that change
        (per step or per generate). Omitted kwargs reuse the captured device buffers —
        passing a value-identical constant would re-copy it into the buffer every step.
        Every value in both dicts must be trace-valid: ttnn.Tensor | int | float | str | bool | None.
        """
        tracer = self._tracers.get(trace_key)
        if tracer is None:
            tracer = Tracer(fn, device=self.mesh_device, prep_run=False, clone_prep_inputs=False)
            self._tracers[trace_key] = tracer
            return tracer(**capture_inputs)
        return tracer(**replay_inputs)

    def release_traces(self) -> None:
        """Release captured denoise traces and free their device trace memory."""
        for tracer in self._tracers.values():
            tracer.release_trace()
        self._tracers.clear()
        self._trace_consts.clear()
        self._trace_latents.clear()
        self._trace_prompt.clear()

    @staticmethod
    def _resolve_checkpoint_file(checkpoint: str, default_filename: str = "ltx-2.3-22b-dev.safetensors") -> str:
        """Resolve a checkpoint reference to a local file path."""
        if os.path.exists(checkpoint):
            return checkpoint
        if ":" in checkpoint:
            repo_id, filename = checkpoint.split(":", 1)
        else:
            repo_id, filename = checkpoint, default_filename
        logger.info(f"Resolving HuggingFace checkpoint {repo_id}:{filename} (auto-download if missing)")
        return hf_hub_download(repo_id=repo_id, filename=filename)

    @staticmethod
    def create_pipeline(
        mesh_device: ttnn.MeshDevice,
        *,
        checkpoint_name: str | None = None,
        gemma_path: str | None = None,
        sp_axis: int | None = None,
        tp_axis: int | None = None,
        num_links: int | None = None,
        dynamic_load: bool | None = None,
        topology: ttnn.Topology | None = None,
        is_fsdp: bool | None = None,
        mode: str = "av",
        pipeline_class: type["LTXPipeline"] | None = None,
        run_warmup: bool = False,
        traced: bool = False,
        num_frames: int = 0,
        height: int = 0,
        width: int = 0,
        **extra_pipeline_kwargs,
    ) -> "LTXPipeline":
        """Auto-configure mesh-shape defaults and forward into ``__init__``.

        ``checkpoint_name`` / ``gemma_path`` accept local paths or HuggingFace
        repo strings (auto-downloaded on first use). Subclass-specific kwargs
        (e.g. ``distilled_lora_path``) flow through ``extra_pipeline_kwargs``.
        """
        mesh_shape = tuple(mesh_device.shape)
        device_configs: dict[tuple[int, int], dict] = {}
        if ttnn.device.is_blackhole():
            device_configs[(2, 4)] = {
                "sp_axis": 1,
                "tp_axis": 0,
                "num_links": 2,
                "dynamic_load": True,
                "topology": ttnn.Topology.Linear,
                "is_fsdp": False,
            }
            device_configs[(4, 8)] = {
                "sp_axis": 1,
                "tp_axis": 0,
                "num_links": 2,
                "dynamic_load": False,
                "topology": ttnn.Topology.Ring,
                "is_fsdp": False,
            }
            defaults = device_configs.get(mesh_shape, device_configs[(2, 4)])
        else:
            device_configs[(1, 8)] = {
                "sp_axis": 1,
                "tp_axis": 0,
                "num_links": 1,
                "dynamic_load": False,
                "topology": ttnn.Topology.Linear,
                "is_fsdp": False,
            }
            device_configs[(2, 4)] = {
                "sp_axis": 0,
                "tp_axis": 1,
                "num_links": 2,
                "dynamic_load": False,
                "topology": ttnn.Topology.Linear,
                "is_fsdp": False,
            }
            device_configs[(4, 8)] = {
                "sp_axis": 1,
                "tp_axis": 0,
                "num_links": 4,
                "dynamic_load": False,
                "topology": ttnn.Topology.Ring,
                "is_fsdp": False,
            }
            defaults = device_configs.get(mesh_shape, device_configs[(2, 4)])
        sp_axis = sp_axis if sp_axis is not None else defaults["sp_axis"]
        tp_axis = tp_axis if tp_axis is not None else defaults["tp_axis"]
        num_links = num_links if num_links is not None else defaults["num_links"]
        dynamic_load = dynamic_load if dynamic_load is not None else defaults["dynamic_load"]
        topology = topology if topology is not None else defaults["topology"]
        is_fsdp = is_fsdp if is_fsdp is not None else defaults["is_fsdp"]

        parallel_config = DiTParallelConfig(
            cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
            sequence_parallel=ParallelFactor(factor=mesh_shape[sp_axis], mesh_axis=sp_axis),
            tensor_parallel=ParallelFactor(factor=mesh_shape[tp_axis], mesh_axis=tp_axis),
        )
        # VAE shards H on tp axis, W on sp axis (mirrors Wan role assignment).
        vae_parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=mesh_shape[tp_axis], mesh_axis=tp_axis),
            width_parallel=ParallelFactor(factor=mesh_shape[sp_axis], mesh_axis=sp_axis),
        )
        ccl_manager = CCLManager(mesh_device, num_links=num_links, topology=topology)

        pipeline_cls = pipeline_class or LTXPipeline
        if run_warmup and not (num_frames > 0 and height > 0 and width > 0):
            logger.warning(f"run_warmup=True but invalid shape ({num_frames=}, {height=}, {width=}); skipping warmup")

        return pipeline_cls(
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            vae_parallel_config=vae_parallel_config,
            mode=mode,
            checkpoint_name=checkpoint_name,
            gemma_path=gemma_path,
            is_fsdp=is_fsdp,
            dynamic_load=dynamic_load,
            num_frames=num_frames,
            height=height,
            width=width,
            run_warmup=run_warmup,
            traced=traced,
            **extra_pipeline_kwargs,
        )

    def _load_config_from_checkpoint(self) -> None:
        """Parse the safetensors header to detect model variant + VAE config. No tensor loads."""
        assert self.checkpoint_name is not None
        checkpoint_path = self.checkpoint_name

        # Transformer + connector detection (key scan + one tensor-shape read).
        with safe_open(checkpoint_path, framework="pt") as f:
            keys = list(f.keys())
            adaln_key = "model.diffusion_model.adaln_single.linear.weight"
            if adaln_key in keys:
                self._cross_attention_adaln = f.get_tensor(adaln_key).shape[0] > 6 * self.inner_dim
            else:
                self._cross_attention_adaln = True
            self._has_gate = any("to_gate_logits" in k for k in keys)
        logger.info(f"Detected: has_gate={self._has_gate}, cross_attention_adaln={self._cross_attention_adaln}")

        # VAE config from JSON metadata header.
        with open(checkpoint_path, "rb") as f:
            header_size = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(header_size))
        vae_cfg = json.loads(header.get("__metadata__", {}).get("config", "{}")).get("vae", {})
        self._vae_checkpoint_path = checkpoint_path
        self._vae_decoder_blocks = vae_cfg.get("decoder_blocks", [])
        self._vae_causal = vae_cfg.get("causal_decoder", False)
        self._vae_base_channels = vae_cfg.get("decoder_base_channels", 128)
        if self._vae_decoder_blocks:
            logger.info(f"VAE config: {len(self._vae_decoder_blocks)} blocks, causal={self._vae_causal}")

    @staticmethod
    def _build_transformer_state_dict(checkpoint_path: str, lora_specs: list[LoraSpec]) -> dict[str, torch.Tensor]:
        """Load + LoRA-fuse the transformer state dict from safetensors. Only
        invoked on cache miss by ``cache_module.load_model``."""
        logger.info(f"Transformer cache miss — loading safetensors: {checkpoint_path}")
        raw = load_file(checkpoint_path)
        prefix = "model.diffusion_model."
        sd = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
        if lora_specs:
            sd = fuse_loras_into(sd, lora_specs)
        return sd

    @staticmethod
    def _build_transformer_cache_name(checkpoint_path: str, lora_specs: list[LoraSpec]) -> str:
        """Cache key for ``cache_module.load_model``. LoRA-tagged so fused and
        base weights don't alias in ``TT_DIT_CACHE_DIR``."""
        base = os.path.basename(checkpoint_path).removesuffix(".safetensors")
        if not lora_specs:
            return base
        tag = "+".join(f"{os.path.basename(s.path).removesuffix('.safetensors')}@{s.strength}" for s in lora_specs)
        return f"{base}.lora-{tag}"

    def _new_transformer(self) -> LTXTransformerModel:
        return LTXTransformerModel(
            num_attention_heads=self.num_attention_heads,
            attention_head_dim=self.attention_head_dim,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            cross_attention_dim=self.cross_attention_dim,
            mesh_device=self.mesh_device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            is_fsdp=self.is_fsdp,
            has_audio=self.mode == "av",
            apply_gated_attention=self._has_gate,
            cross_attention_adaln=self._cross_attention_adaln,
        )

    def _new_upsampler(self) -> LTXLatentUpsampler:
        """Spatial-2x latent upsampler at stage-1 shape: input H/W = full // 64,
        latent_frames = (num_frames - 1) // 8 + 1. Config read from checkpoint header."""
        with safe_open(self._upsampler_path, framework="pt") as f:
            cfg = json.loads(f.metadata()["config"])
        latent_frames = (self._init_num_frames - 1) // TEMPORAL_COMPRESSION + 1
        input_hw = (self._init_height // 64, self._init_width // 64)
        return LTXLatentUpsampler(
            input_hw=input_hw,
            in_channels=cfg["in_channels"],
            mid_channels=cfg["mid_channels"],
            num_blocks_per_stage=cfg["num_blocks_per_stage"],
            spatial_upsample=cfg["spatial_upsample"],
            temporal_upsample=cfg["temporal_upsample"],
            spatial_scale=cfg["spatial_scale"],
            rational_resampler=cfg["rational_resampler"],
            mesh_device=self.mesh_device,
            parallel_config=self.vae_parallel_config,
            ccl_manager=self.vae_ccl_manager,
            num_frames=latent_frames,
        )

    def _instantiate_modules(self, extra_variants: list[tuple[str, list[LoraSpec]]]) -> None:
        """Build every TT Module the pipeline will use. No DRAM weights yet —
        ``_prime_caches`` (next) attaches them."""
        self.transformer = self._new_transformer()
        self.transformer_states.append(
            TransformerState(
                model=self.transformer,
                cache_name=self._build_transformer_cache_name(self.checkpoint_name, []),
                state_dict_provider=lambda: self._build_transformer_state_dict(self.checkpoint_name, []),
            )
        )
        for tag, lora_specs in extra_variants:
            specs = list(lora_specs)
            self.transformer_states.append(
                TransformerState(
                    model=self._new_transformer(),
                    cache_name=self._build_transformer_cache_name(self.checkpoint_name, specs),
                    lora_specs=specs,
                    state_dict_provider=lambda s=specs: self._build_transformer_state_dict(self.checkpoint_name, s),
                )
            )
            logger.info(f"Registered transformer variant {tag} with {len(specs)} LoRA(s)")

        if self._vae_decoder_blocks:
            self.vae_decoder = LTXVideoDecoder(
                decoder_blocks=self._vae_decoder_blocks,
                causal=self._vae_causal,
                base_channels=self._vae_base_channels,
                mesh_device=self.mesh_device,
                parallel_config=self.vae_parallel_config,
                ccl_manager=self.vae_ccl_manager,
                num_frames=self._init_num_frames or None,
                height=self._init_height or None,
                width=self._init_width or None,
            )

        if self.HAS_UPSAMPLER:
            assert (
                self._init_height > 0 and self._init_width > 0 and self._init_num_frames > 0
            ), f"{type(self).__name__} requires num_frames/height/width at create_pipeline."
            self._upsampler_path = LTXPipeline._resolve_checkpoint_file(LTX_UPSAMPLER_HF_REF)
            self.upsampler = self._new_upsampler()

        if self.checkpoint_name is not None:
            self._new_audio_decoder()

    def _register_coresident_exclusions(self) -> None:
        """All transformer variants exclude each other AND the VAE. LTX-22B +
        LTX-VAE don't both fit on BH LB; VAE must evict the active transformer
        before decode. LTX needs the eviction on both WH and BH."""
        if not self.dynamic_load:
            return
        models = [s.model for s in self.transformer_states]
        for i, m in enumerate(models):
            peers = [p for j, p in enumerate(models) if j != i]
            if peers:
                m.register_coresident_exclusions(*peers)

        if self.vae_decoder is not None:
            for m in models:
                m.register_coresident_exclusions(self.vae_decoder)
            self.vae_decoder.register_coresident_exclusions(*models)

        # Upsampler is tiny (~120 MB/chip) and stays resident alongside the transformer.
        # Only the full-res VAE decode activations need it evicted.
        if self.upsampler is not None and self.vae_decoder is not None:
            self.vae_decoder.register_coresident_exclusions(self.upsampler)
            self.upsampler.register_coresident_exclusions(self.vae_decoder)

        # The on-device Gemma encoder modules are bidirectionally excluded with the DiT
        # variants + VAE; the pair wires the exclusions on each module at its first build.
        encoder_peers = [*models] + ([self.vae_decoder] if self.vae_decoder is not None else [])
        self.gemma_encoder_pair.register_coresident_peers(encoder_peers)

        # Audio decoder/vocoder are intentionally NOT excluded against the VAE:
        # measured coresident on BH-LB 2x4 (the tightest BH config) with no OOM,
        # even with the fp32 vocoder's conv3d activations live. Excluding them only
        # forces a redundant audio reload at decode (~6s warm) for no memory gain.

    def _prime_caches(self) -> None:
        """Load every module in reverse use order so variant 0 is resident in
        DRAM after ``__init__`` (matches Wan's reverse-use-order priming)."""
        for idx in range(len(self.transformer_states) - 1, 0, -1):
            self._prepare_transformer(idx)
        # Each _prepare_* no-ops when its module isn't present (None), so no call-site guards.
        self._prepare_vae()
        self._prepare_upsampler()
        self._prepare_audio_decoder()
        self._prepare_transformer(0)

    def _prepare_transformer(self, idx: int = 0) -> None:
        state = self.transformer_states[idx]
        cache_module.load_model(
            state.model,
            model_name=state.cache_name,
            subfolder="transformer",
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            is_fsdp=self.is_fsdp,
            get_torch_state_dict=state.state_dict_provider,
        )
        self.transformer = state.model

    def _device_embed_cache_path(self, prompts: list[str]) -> str:
        """Disk-cache path for on-device prompt embeddings. Separate namespace from the
        reference cache (different format) — lets a repeated prompt skip the encoder."""
        cache_dir = os.environ.get("TT_DIT_CACHE_DIR") or os.path.expanduser("~/.cache/tt-dit")
        embed_cache_dir = os.path.join(cache_dir, "ltx-embeddings")
        os.makedirs(embed_cache_dir, exist_ok=True)
        key = hashlib.md5(("device||" + "||".join(prompts)).encode()).hexdigest()
        return os.path.join(embed_cache_dir, f"{key}.device.pt")

    def encode_prompts(
        self, prompts: list[str], *, use_cache: bool = True
    ) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
        """Encode prompts on device via the Gemma encoder pair, with a prompt-embedding disk
        cache (orchestration kept here). A cache hit returns saved embeddings without running
        the encoder; ``use_cache=False`` forces a real encode (used by warmup)."""
        cache_path = self._device_embed_cache_path(prompts)
        if use_cache and os.path.exists(cache_path):
            logger.info(f"Loading cached device embeddings from {cache_path}")
            return torch.load(cache_path, weights_only=False)

        results = self.gemma_encoder_pair.encode(prompts)

        if use_cache:
            torch.save(results, cache_path)
            logger.info(f"Cached device embeddings to {cache_path}")
        return results

    def encode_prompts_reference(self, prompts: list[str]) -> list:
        """Encode prompts using the official LTX-2 reference pipeline (recommended for AV mode)."""
        assert self.checkpoint_name is not None, "checkpoint_name must be set before encode_prompts_reference"
        assert self.gemma_path is not None, "gemma_path must be set before encode_prompts_reference"
        try:
            _ensure_ltx_reference_on_path()
            from ltx_pipelines.utils.blocks import PromptEncoder
        except ImportError as e:
            raise ImportError(
                "encode_prompts_reference() requires the LTX-2 reference package. "
                "Use load_text_encoder() + __call__() for standalone text encoding."
            ) from e

        cache_dir = os.environ.get("TT_DIT_CACHE_DIR") or os.path.expanduser("~/.cache/tt-dit")
        embed_cache_dir = os.path.join(cache_dir, "ltx-embeddings")
        os.makedirs(embed_cache_dir, exist_ok=True)

        cache_key = hashlib.md5("||".join(prompts).encode()).hexdigest()
        cache_path = os.path.join(embed_cache_dir, f"{cache_key}.pt")

        if os.path.exists(cache_path):
            logger.info(f"Loading cached embeddings from {cache_path}")
            return torch.load(cache_path, weights_only=False)

        # PromptEncoder owns Gemma text encoder + embeddings processor lifecycle:
        # builds Gemma, encodes, frees, then builds the embeddings processor.
        prompt_encoder = PromptEncoder(
            checkpoint_path=self.checkpoint_name,
            gemma_root=self.gemma_path,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        results = prompt_encoder(prompts)
        del prompt_encoder

        torch.save(results, cache_path)
        logger.info(f"Cached embeddings to {cache_path}")
        return results

    def _prepare_vae(self) -> None:
        """Push VAE decoder weights onto the mesh. Module was constructed in
        ``__init__``; blocking-hash subfolder forces re-load when conv3d
        ``C_in_block`` changes (mirrors Wan)."""
        if self.vae_decoder is None:
            return

        def _vae_state_provider() -> dict[str, torch.Tensor]:
            logger.info(f"VAE cache miss — loading safetensors: {self._vae_checkpoint_path}")
            raw = load_file(self._vae_checkpoint_path)
            vae_state = {}
            for k, v in raw.items():
                if k.startswith("vae.decoder."):
                    vae_state[k.removeprefix("vae.decoder.")] = v
                elif k.startswith("vae.per_channel_statistics."):
                    short_key = k.removeprefix("vae.")
                    if short_key in ("per_channel_statistics.mean-of-means", "per_channel_statistics.std-of-means"):
                        vae_state[short_key] = v
            return vae_state

        blocking_key = conv3d_blocking_hash(self.vae_decoder)
        subfolder = f"vae_{blocking_key}" if blocking_key else "vae"
        cache_module.load_model(
            self.vae_decoder,
            model_name=os.path.basename(self.checkpoint_name).removesuffix(".safetensors"),
            subfolder=subfolder,
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            get_torch_state_dict=_vae_state_provider,
        )
        logger.info(f"Loaded TTNN VAE decoder ({len(self._vae_decoder_blocks)} blocks)")

    def decode_latents(self, latent: torch.Tensor, latent_frames: int, latent_h: int, latent_w: int) -> torch.Tensor:
        """Decode latent tensor to video pixels.

        Args:
            latent: (B, num_tokens, C) flat latent from denoising loop
            latent_frames, latent_h, latent_w: Spatial dimensions

        Returns:
            (B, 3, F, H, W) decoded video
        """
        if self.vae_decoder is None:
            logger.warning("No VAE decoder loaded, returning raw latent")
            return latent

        B = latent.shape[0]
        # Reshape flat tokens to spatial: (B, num_tokens, C) -> (B, C, F', H', W')
        latent_spatial = latent.reshape(B, latent_frames, latent_h, latent_w, self.in_channels)
        latent_spatial = latent_spatial.permute(0, 4, 1, 2, 3)  # BCTHW

        return self.vae_decoder(latent_spatial)

    def _vae_per_channel_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached ``(mean-of-means, std-of-means)`` reshaped for ``(B, C, F, H, W)``
        broadcast — the un_normalize/normalize bookends matching ``ltx_core.upsample_video``."""
        if getattr(self, "_pcs_cache", None) is None:
            with safe_open(self.checkpoint_name, framework="pt") as f:
                mean = f.get_tensor("vae.per_channel_statistics.mean-of-means").float()
                std = f.get_tensor("vae.per_channel_statistics.std-of-means").float()
            self._pcs_cache = (mean.view(1, -1, 1, 1, 1), std.view(1, -1, 1, 1, 1))
        return self._pcs_cache

    def _prepare_upsampler(self) -> None:
        """Push upsampler weights onto the mesh via the disk cache. Blocking-hash
        subfolder invalidates the cache when conv3d ``C_in_block`` changes."""
        if self.upsampler is None or self.upsampler.is_loaded():
            return

        def _upsampler_state_provider() -> dict[str, torch.Tensor]:
            logger.info(f"Upsampler cache miss — loading safetensors: {self._upsampler_path}")
            return load_file(self._upsampler_path)

        blocking_key = conv3d_blocking_hash(self.upsampler)
        subfolder = f"upsampler_{blocking_key}" if blocking_key else "upsampler"
        cache_module.load_model(
            self.upsampler,
            model_name=os.path.basename(self._upsampler_path).removesuffix(".safetensors"),
            subfolder=subfolder,
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            get_torch_state_dict=_upsampler_state_provider,
        )
        logger.info("Loaded TTNN latent upsampler")

    def _upsample_latent(self, video_latent: torch.Tensor) -> torch.Tensor:
        """Spatial-2x latent upsample. Mirrors ``ltx_core.upsample_video``: un_normalize
        → bare upsampler → re_normalize. ``(B, C, F, H, W)`` host in/out."""
        assert self.upsampler is not None, "upsampler not constructed (HAS_UPSAMPLER=False?)"
        self._prepare_upsampler()
        mean, std = self._vae_per_channel_stats()
        x = video_latent.float() * std + mean
        x = self.upsampler(x)
        return (x.float() - mean) / std

    def _warmup_upsample(self, num_frames: int, height: int, width: int) -> None:
        """JIT-compile the upsampler at the stage-1 half-res shape."""
        if self.upsampler is None:
            return
        latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
        s1_lh, s1_lw = height // 64, width // 64
        dummy = torch.zeros(1, self.in_channels, latent_frames, s1_lh, s1_lw)
        self._upsample_latent(dummy)

    def _warmup_decode(self, num_frames: int, height: int, width: int) -> None:
        """Load VAE + JIT-compile decode kernels with a zero dummy latent at
        the target shape. No-op if no VAE is configured."""
        if self.vae_decoder is None:
            return
        latent_frames, latent_h, latent_w = latent_grid(num_frames, height, width)
        dummy = torch.zeros(1, latent_frames * latent_h * latent_w, self.in_channels)
        self._prepare_vae()
        self.decode_latents(dummy, latent_frames, latent_h, latent_w)

    def _prepare_rope(
        self, num_frames: int, latent_height: int, latent_width: int, fps: float = 24.0
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Compute video RoPE in INTERLEAVED layout for ttnn.experimental.rotary_embedding_llama."""
        v_shape = VideoLatentShape(batch=1, channels=128, frames=num_frames, height=latent_height, width=latent_width)
        v_coords = video_get_patch_grid_bounds(v_shape)
        v_positions = get_pixel_coords(v_coords, scale_factors=(8, 32, 32), causal_fix=True).float()
        v_positions[:, 0, ...] = v_positions[:, 0, ...] / fps

        # Positions MUST stay fp32 — bf16 introduced catastrophic phase error in
        # high-frequency RoPE channels (1700x worse than fp32, randomizing the top
        # half of head_dim).
        cos_freq, sin_freq = precompute_freqs_cis(
            v_positions,
            dim=self.inner_dim,
            out_dtype=torch.float32,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            use_middle_indices_grid=True,
            num_attention_heads=self.num_attention_heads,
            rope_type=LTXRopeType.INTERLEAVED,
        )  # (1, N, dim)

        cos_freq = reshape_interleaved_to_bhnd(cos_freq, self.num_attention_heads)
        sin_freq = reshape_interleaved_to_bhnd(sin_freq, self.num_attention_heads)

        # Pad seq dim to ttnn.TILE_SIZE * sp_factor; padded slots use cos=1, sin=0
        # (identity rotation) — SDPA still masks them via logical_n.
        cos_freq, sin_freq = self._pad_video_rope_sp(cos_freq, sin_freq)

        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        tt_cos = bf16_tensor_2dshard(cos_freq, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
        tt_sin = bf16_tensor_2dshard(sin_freq, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
        return tt_cos, tt_sin

    def _prepare_trans_mat(self) -> ttnn.Tensor:
        """Cached per-tile rotation matrix for rotary_embedding_llama (shared builder)."""
        if getattr(self, "_cached_trans_mat", None) is None:
            self._cached_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=self.mesh_device)
        return self._cached_trans_mat

    def _prepare_prompt(self, prompt_embeds: torch.Tensor) -> ttnn.Tensor:
        """Push prompt embeddings to device, padding/truncating to cross_attention_dim."""
        prompt = prompt_embeds.unsqueeze(0)
        if prompt.shape[-1] < self.cross_attention_dim:
            pad_size = self.cross_attention_dim - prompt.shape[-1]
            prompt = torch.nn.functional.pad(prompt, (0, pad_size))
        elif prompt.shape[-1] > self.cross_attention_dim:
            prompt = prompt[..., : self.cross_attention_dim]
        return bf16_tensor(prompt, device=self.mesh_device)

    def warmup_buffers(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
        num_inference_steps: int = 2,
    ) -> None:
        """Compile every device program full-guidance ``call_av`` will exercise
        (4 transformer passes/step: cond/uncond/ptb/iso) plus VAE decode.
        ``ge_gamma=0`` skips the GE branch (pure host math)."""
        t0 = time.time()
        logger.info(f"warmup (AV): {num_frames}f@{height}x{width}, {num_inference_steps} steps")

        # Dummy zero embeddings at the real shapes — warmup only needs to compile the
        # (shape-driven) call_av kernels, not real prompt content. This avoids loading
        # the encoder here, which would coresident-evict the DiT; the encoder kernels
        # compile on the first generate().
        v_p = torch.zeros(1, self.gemma_encoder_pair.sequence_length, self.gemma_encoder_pair.video_dim)
        a_p = torch.zeros(1, self.gemma_encoder_pair.sequence_length, self.gemma_encoder_pair.audio_dim)
        v_n, a_n = v_p, a_p

        self.call_av(
            video_prompt_embeds=v_p,
            audio_prompt_embeds=a_p,
            neg_video_prompt_embeds=v_n,
            neg_audio_prompt_embeds=a_n,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            seed=0,
            ge_gamma=0.0,
        )

        self._warmup_decode(num_frames, height, width)
        self._prepare_transformer(0)
        logger.info(f"warmup (AV) done in {time.time() - t0:.1f}s")

    def generate(
        self,
        prompt: str,
        *,
        output_path: str,
        negative_prompt: str | None = None,
        num_frames: int = 121,
        height: int = 512,
        width: int = 768,
        num_inference_steps: int = 30,
        video_cfg_scale: float = 3.0,
        audio_cfg_scale: float = 7.0,
        video_stg_scale: float = 1.0,
        audio_stg_scale: float = 1.0,
        video_modality_scale: float = 3.0,
        audio_modality_scale: float = 3.0,
        rescale_scale: float = 0.7,
        stg_block: int = 28,
        seed: int = 10,
        ge_gamma: float = 0.0,
        fps: int = 24,
    ) -> str:
        """Run the full LTX-2.3 Pro AV generation pipeline and write an MP4.

        Guidance defaults match the reference ``LTX_2_3_PARAMS``. ``ge_gamma`` enables
        gradient-estimation sampling (off by default, as in the reference); pass a
        non-zero value (the reference uses 2.0) to turn it on.
        """
        neg = negative_prompt if negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT

        total_t0 = time.time()

        t0 = time.time()
        # On-device Gemma encode; coresident-excluded with the DiT/VAE, so loading it auto-evicts
        # them and _prepare_transformer(0) evicts the encoder back. Only load on a cache miss —
        # a cached prompt skips the encoder entirely.
        cached = os.path.exists(self._device_embed_cache_path([prompt, neg]))
        if not cached:
            self.gemma_encoder_pair.ensure_loaded()
        enc = self.encode_prompts([prompt, neg])
        v_embeds, a_embeds = enc[0][0].float(), enc[0][1].float()
        neg_v, neg_a = enc[1][0].float(), enc[1][1].float()
        logger.info(f"Encoding ({'cache' if cached else 'device'}): {time.time() - t0:.1f}s")

        self._prepare_transformer(0)

        t0 = time.time()
        video_latent, audio_latent = self.call_av(
            video_prompt_embeds=v_embeds,
            audio_prompt_embeds=a_embeds,
            neg_video_prompt_embeds=neg_v,
            neg_audio_prompt_embeds=neg_a,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            video_cfg_scale=video_cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            video_stg_scale=video_stg_scale,
            audio_stg_scale=audio_stg_scale,
            video_modality_scale=video_modality_scale,
            audio_modality_scale=audio_modality_scale,
            rescale_scale=rescale_scale,
            stg_block=stg_block,
            seed=seed,
            ge_gamma=ge_gamma,
        )
        denoise_time = time.time() - t0
        logger.info(f"Denoising: {denoise_time:.1f}s ({denoise_time / num_inference_steps:.1f}s/step)")

        t0 = time.time()
        self._prepare_vae()
        logger.info(f"VAE loaded in {time.time() - t0:.0f}s")

        latent_frames, latent_h, latent_w = latent_grid(num_frames, height, width)

        t0 = time.time()
        video_pixels = self.decode_latents(video_latent, latent_frames, latent_h, latent_w)
        logger.info(f"VAE decode: {time.time() - t0:.1f}s — {video_pixels.shape}")

        audio_obj = self.decode_audio(audio_latent, num_frames, fps=fps)
        export_video_audio(video_pixels, output_path, fps=fps, audio=audio_obj)

        total_time = time.time() - total_t0
        logger.info(f"Total: {total_time:.1f}s | Output: {output_path}")
        return output_path

    def _prepare_audio_rope(self, audio_N: int, audio_N_real: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Compute audio RoPE in INTERLEAVED layout for ttnn.experimental.rotary_embedding_llama."""
        a_shape = AudioLatentShape(batch=1, channels=8, frames=audio_N_real, mel_bins=16)
        a_positions = audio_get_patch_grid_bounds(a_shape).float()  # (1, 1, N, 2)

        # Positions MUST stay fp32 — bf16 randomizes the top half of head_dim.
        a_cos, a_sin = precompute_freqs_cis(
            a_positions,
            dim=2048,
            out_dtype=torch.float32,
            theta=self.positional_embedding_theta,
            max_pos=[20],
            use_middle_indices_grid=True,
            num_attention_heads=32,
            rope_type=LTXRopeType.INTERLEAVED,
        )  # (1, N, 2048)
        a_cos = reshape_interleaved_to_bhnd(a_cos, num_heads=32)  # (1, 32, N, 64)
        a_sin = reshape_interleaved_to_bhnd(a_sin, num_heads=32)

        if audio_N > audio_N_real:
            head_dim = a_cos.shape[-1]
            a_cos_padded = torch.ones(1, 32, audio_N, head_dim)
            a_cos_padded[:, :, :audio_N_real, :] = a_cos
            a_sin_padded = torch.zeros(1, 32, audio_N, head_dim)
            a_sin_padded[:, :, :audio_N_real, :] = a_sin
            a_cos, a_sin = a_cos_padded, a_sin_padded

        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        return (
            bf16_tensor_2dshard(a_cos, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1}),
            bf16_tensor_2dshard(a_sin, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1}),
        )

    def _prepare_av_cross_pe(
        self,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
        audio_N: int,
        audio_N_real: int,
        fps: float = 24.0,
        cross_pe_max_pos: int = 20,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Compute temporal-only cross positional embeddings for A↔V cross-attention.

        Reference: ``MultiModalTransformerArgsPreprocessor.prepare`` builds ``cross_pe`` from
        the temporal slice ``modality.positions[:, 0:1, :]`` at ``dim=audio_cross_attention_dim``
        with ``max_pos=[cross_pe_max_pos]``. Video and audio share the scheme so a video token
        and an audio token at the same time get the same rotary phase (AV temporal sync).

        Returns 8 device tensors used by inner_step:
            (v_q_cos, v_q_sin)         — video Q in A→V cross-attn (SP×TP sharded).
            (a_q_cos, a_q_sin)         — audio Q in V→A cross-attn (SP×TP sharded).
            (v_k_cos, v_k_sin)         — video K in V→A cross-attn (TP-only; K side after AllGather).
            (a_k_cos, a_k_sin)         — audio K in A→V cross-attn (TP-only; K side after AllGather).
        """
        v_shape = VideoLatentShape(
            batch=1, channels=128, frames=latent_frames, height=latent_height, width=latent_width
        )
        v_coords = video_get_patch_grid_bounds(v_shape)
        v_positions = get_pixel_coords(v_coords, scale_factors=(8, 32, 32), causal_fix=True).float()
        v_positions[:, 0, ...] = v_positions[:, 0, ...] / fps  # temporal axis → seconds
        v_temporal = v_positions[:, 0:1, :]  # (1, 1, video_N, 2)

        a_shape = AudioLatentShape(batch=1, channels=8, frames=audio_N_real, mel_bins=16)
        a_positions = audio_get_patch_grid_bounds(a_shape).float()  # (1, 1, audio_N_real, 2)

        rope_kwargs = dict(
            dim=2048,  # audio_cross_attention_dim — both sides share this
            out_dtype=torch.float32,
            theta=self.positional_embedding_theta,
            max_pos=[cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=32,
            rope_type=LTXRopeType.INTERLEAVED,
        )

        # Positions MUST stay fp32 — bf16 randomizes the top half of head_dim.
        v_cos, v_sin = precompute_freqs_cis(v_temporal, **rope_kwargs)  # (1, video_N, 2048)
        a_cos, a_sin = precompute_freqs_cis(a_positions, **rope_kwargs)  # (1, audio_N_real, 2048)
        v_cos = reshape_interleaved_to_bhnd(v_cos, num_heads=32)  # (1, 32, video_N, 64)
        v_sin = reshape_interleaved_to_bhnd(v_sin, num_heads=32)
        a_cos = reshape_interleaved_to_bhnd(a_cos, num_heads=32)
        a_sin = reshape_interleaved_to_bhnd(a_sin, num_heads=32)

        v_cos, v_sin = self._pad_video_rope_sp(v_cos, v_sin)

        if audio_N > audio_N_real:
            head_dim = a_cos.shape[-1]
            a_cos_padded = torch.ones(1, 32, audio_N, head_dim)
            a_cos_padded[:, :, :audio_N_real, :] = a_cos
            a_sin_padded = torch.zeros(1, 32, audio_N, head_dim)
            a_sin_padded[:, :, :audio_N_real, :] = a_sin
            a_cos, a_sin = a_cos_padded, a_sin_padded

        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis

        # Q-side: SP×TP sharded (matches the Q tensor layout post-attention QKV split).
        v_q_cos = bf16_tensor_2dshard(v_cos, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
        v_q_sin = bf16_tensor_2dshard(v_sin, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
        a_q_cos = bf16_tensor_2dshard(a_cos, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
        a_q_sin = bf16_tensor_2dshard(a_sin, device=self.mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})

        # K-side: TP-only on heads (sequence is replicated after AllGather on K).
        v_k_cos = bf16_tensor(v_cos, device=self.mesh_device, mesh_axis=tp_axis, shard_dim=1)
        v_k_sin = bf16_tensor(v_sin, device=self.mesh_device, mesh_axis=tp_axis, shard_dim=1)
        a_k_cos = bf16_tensor(a_cos, device=self.mesh_device, mesh_axis=tp_axis, shard_dim=1)
        a_k_sin = bf16_tensor(a_sin, device=self.mesh_device, mesh_axis=tp_axis, shard_dim=1)

        return (v_q_cos, v_q_sin, a_q_cos, a_q_sin, v_k_cos, v_k_sin, a_k_cos, a_k_sin)

    @staticmethod
    def _zero_sp_padding(t: torch.Tensor, n_real: int) -> torch.Tensor:
        """Zero SP-padded token slots (video or audio) so they do not affect guidance or GE."""
        if t.shape[1] <= n_real:
            return t
        out = t.clone()
        out[:, n_real:, :] = 0.0
        return out

    def _sp_pad_len(self, n_real: int) -> int:
        """Round a seq len up to ttnn.TILE_SIZE * sp_factor (ring-SDPA / SP requirement).
        Shared by the video and audio token grids."""
        sp_factor = self.parallel_config.sequence_parallel.factor
        divisor = ttnn.TILE_SIZE * sp_factor
        return ((n_real + divisor - 1) // divisor) * divisor

    def _pad_video_rope_sp(self, cos_freq: torch.Tensor, sin_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad video RoPE cos/sin on dim=2 to the SP boundary.

        Padded slots use cos=1, sin=0 (identity rotation). Same convention as the audio
        RoPE padding in ``_prepare_audio_rope`` / ``_prepare_av_cross_pe``.
        """
        video_N_real = cos_freq.shape[2]
        video_N = self._sp_pad_len(video_N_real)
        if video_N == video_N_real:
            return cos_freq, sin_freq
        pad = video_N - video_N_real
        H = cos_freq.shape[1]
        d_half = cos_freq.shape[-1]
        cos_pad = torch.ones(1, H, pad, d_half, dtype=cos_freq.dtype)
        sin_pad = torch.zeros(1, H, pad, d_half, dtype=sin_freq.dtype)
        cos_freq = torch.cat([cos_freq, cos_pad], dim=2)
        sin_freq = torch.cat([sin_freq, sin_pad], dim=2)
        return cos_freq, sin_freq

    @staticmethod
    def _apply_modal_guidance(
        den: torch.Tensor,
        unc,
        ptb,
        iso,
        *,
        cfg_scale: float,
        stg_scale: float,
        modality_scale: float,
        rescale_scale: float,
        do_cfg: bool,
        do_stg: bool,
        do_mod: bool,
        real_token_count: int | None = None,
    ) -> torch.Tensor:
        """CFG + STG + modality guidance with optional per-modality token slice (audio pad)."""
        if real_token_count is not None:
            den_s = den[:, :real_token_count, :]
        else:
            den_s = den

        pred = den_s.float()
        c = den_s.float()
        if do_cfg and isinstance(unc, torch.Tensor):
            unc_s = unc[:, :real_token_count, :] if real_token_count is not None else unc
            pred = pred + (cfg_scale - 1) * (c - unc_s.float())
        if do_stg and isinstance(ptb, torch.Tensor):
            ptb_s = ptb[:, :real_token_count, :] if real_token_count is not None else ptb
            pred = pred + stg_scale * (c - ptb_s.float())
        if do_mod and isinstance(iso, torch.Tensor):
            iso_s = iso[:, :real_token_count, :] if real_token_count is not None else iso
            pred = pred + (modality_scale - 1) * (c - iso_s.float())
        if rescale_scale != 0:
            pred = pred * (rescale_scale * (c.std() / pred.std()) + (1 - rescale_scale))

        if real_token_count is not None:
            out = den.clone()
            out[:, :real_token_count, :] = pred.bfloat16()
            return out
        return pred.bfloat16()

    def _prepare_audio_masks(self, audio_N: int, audio_N_real: int) -> tuple:
        """Create SDPA attn mask and padding masks for SP-sharded vs gathered audio.

        Returns (attn_mask, pad_mask_sp, pad_mask_full). pad_mask_sp is sharded on the
        sequence dimension for multiply with local audio activations; pad_mask_full is
        replicated for multiply after all_gather on A-to-V keys.
        """
        if audio_N <= audio_N_real:
            return None, None, None

        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        # Column mask only: real/padded queries are barred from attending TO padded keys.
        # Do NOT mask padded-query rows to -inf — that makes all attention scores in those
        # rows -inf → softmax NaN → NaN propagates via padded-token outputs (which we then
        # multiply by 0; IEEE 0*NaN = NaN, not 0). audio_padding_mask already zeros the
        # padded-query outputs after attention, so column-only masking is sufficient and
        # numerically safer at high σ where activations have largest magnitude.
        mask = torch.zeros(1, 1, audio_N, audio_N)
        mask[:, :, :, audio_N_real:] = float("-inf")
        mask = mask.to(torch.bfloat16)
        tt_attn_mask = bf16_tensor(
            mask,
            device=self.mesh_device,
            mesh_axis=sp_axis,
            shard_dim=2,
        )

        pad_mask = torch.ones(1, 1, audio_N, 1, dtype=torch.bfloat16)
        pad_mask[:, :, audio_N_real:, :] = 0.0
        tt_pad_mask_sp = bf16_tensor(
            pad_mask,
            device=self.mesh_device,
            mesh_axis=sp_axis,
            shard_dim=2,
        )
        tt_pad_mask_full = bf16_tensor(pad_mask, device=self.mesh_device)
        return tt_attn_mask, tt_pad_mask_sp, tt_pad_mask_full

    def _prepare_video_masks(self, video_N: int, video_N_real: int) -> ttnn.Tensor | None:
        """Create the SP-sharded video padding mask, shape (1, 1, video_N, 1).

        Multiply the local (sharded) video activations by this to zero padded slots
        before they propagate downstream (self-attn residual / cross-attn K / FF).
        Returns ``None`` when no padding is needed.

        No SDPA attn_mask is returned (unlike audio) — video self-attention uses
        ring SDPA which masks padded keys via the ``logical_n=video_N_real`` arg.
        """
        if video_N <= video_N_real:
            return None

        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        pad_mask = torch.ones(1, 1, video_N, 1, dtype=torch.bfloat16)
        pad_mask[:, :, video_N_real:, :] = 0.0
        return bf16_tensor(
            pad_mask,
            device=self.mesh_device,
            mesh_axis=sp_axis,
            shard_dim=2,
        )

    @torch.no_grad()
    def call_av(
        self,
        video_prompt_embeds: torch.Tensor,
        audio_prompt_embeds: torch.Tensor,
        neg_video_prompt_embeds: torch.Tensor | None = None,
        neg_audio_prompt_embeds: torch.Tensor | None = None,
        num_frames: int = 33,
        height: int = 512,
        width: int = 768,
        num_inference_steps: int = 30,
        video_cfg_scale: float = 3.0,
        audio_cfg_scale: float = 7.0,
        video_stg_scale: float = 1.0,
        audio_stg_scale: float = 1.0,
        video_modality_scale: float = 3.0,
        audio_modality_scale: float = 3.0,
        rescale_scale: float = 0.7,
        stg_block: int = 28,
        seed: int | None = None,
        ge_gamma: float = 0.0,
        profiler=None,
        profiler_iteration: int = 0,
        sigmas: torch.Tensor | None = None,
        initial_video_latent: torch.Tensor | None = None,
        initial_audio_latent: torch.Tensor | None = None,
        noise_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run AV denoising with full MultiModalGuider guidance. Returns (video_latent, audio_latent).

        Stage-2 / refine usage: pass ``sigmas`` (the explicit schedule), the
        upsampled ``initial_video_latent`` plus the stage-1 ``initial_audio_latent``,
        and ``noise_scale = sigmas[0]`` to renoise. Set all guidance scales to
        their neutral values (``cfg=1.0, stg=0.0, mod=1.0, ge_gamma=0.0``).
        """
        B = 1
        latent_frames, latent_h, latent_w = latent_grid(num_frames, height, width)
        video_N_real = latent_frames * latent_h * latent_w
        # SP padding: round seq dim up to TILE_SIZE * sp_factor so ring SDPA's
        # N_local % TILE_HEIGHT == 0 and N_global == N_local * ring_size checks pass.
        video_N = self._sp_pad_len(video_N_real)

        vps = VideoPixelShape(batch=B, frames=num_frames, height=height, width=width, fps=24)
        als = AudioLatentShape.from_video_pixel_shape(vps)
        audio_N_real = als.frames
        audio_N = self._sp_pad_len(audio_N_real)

        logger.info(
            f"AV: {num_frames}f@{height}x{width}, "
            f"vN={video_N}(real={video_N_real}), aN={audio_N}(real={audio_N_real})"
        )

        v_cos, v_sin = self._prepare_rope(latent_frames, latent_h, latent_w)
        a_cos, a_sin = self._prepare_audio_rope(audio_N, audio_N_real)
        # Cross-modal positional embeddings for A↔V cross-attention: without them audio queries
        # attend to video keys with uniform phase, destroying temporal sync. Reference:
        # MultiModalTransformerArgsPreprocessor (temporal-only column, dim=audio_cross_attention_dim).
        (
            v_xpe_cos,
            v_xpe_sin,
            a_xpe_cos,
            a_xpe_sin,
            v_xpe_cos_full,
            v_xpe_sin_full,
            a_xpe_cos_full,
            a_xpe_sin_full,
        ) = self._prepare_av_cross_pe(latent_frames, latent_h, latent_w, audio_N, audio_N_real)
        trans_mat = self._prepare_trans_mat()
        tt_attn_mask, tt_pad_mask_sp, tt_pad_mask_full = self._prepare_audio_masks(audio_N, audio_N_real)
        tt_v_pad_mask_sp = self._prepare_video_masks(video_N, video_N_real)

        tt_vp = self._prepare_prompt(video_prompt_embeds)
        tt_ap = bf16_tensor(audio_prompt_embeds.unsqueeze(0), device=self.mesh_device)
        tt_nv = self._prepare_prompt(neg_video_prompt_embeds) if neg_video_prompt_embeds is not None else None
        tt_na = (
            bf16_tensor(neg_audio_prompt_embeds.unsqueeze(0), device=self.mesh_device)
            if neg_audio_prompt_embeds is not None
            else None
        )

        if sigmas is None:
            # Match the LTX-2 reference: the scheduler shift uses the fixed
            # MAX_SHIFT_ANCHOR (default num_tokens), not the real token count
            # (ltx_pipelines.ti2vid_two_stages / ti2vid_one_stage).
            sigmas = compute_sigmas(steps=num_inference_steps)
        else:
            assert (
                len(sigmas) == num_inference_steps + 1
            ), f"sigmas length {len(sigmas)} must equal num_inference_steps+1 ({num_inference_steps+1})"

        if initial_video_latent is not None:
            assert noise_scale is not None, "noise_scale required when initial_video_latent is provided"
            if seed is not None:
                torch.manual_seed(seed)
            init_v = initial_video_latent.float()
            if init_v.dim() == 2:
                init_v = init_v.unsqueeze(0)
            noise_v = torch.randn_like(init_v)
            video_lat_real = init_v * (1.0 - noise_scale) + noise_v * noise_scale
        else:
            if seed is not None:
                torch.manual_seed(seed)
            video_lat_real = torch.randn(B, video_N_real, self.in_channels, dtype=torch.bfloat16).float() * sigmas[0]

        # Zero-pad video latent on dim=1 to video_N for SP sharding.
        if video_N > video_N_real:
            video_lat = torch.zeros(B, video_N, self.in_channels)
            video_lat[:, :video_N_real, :] = video_lat_real
        else:
            video_lat = video_lat_real

        if initial_audio_latent is not None:
            assert noise_scale is not None, "noise_scale required when initial_audio_latent is provided"
            if seed is not None:
                torch.manual_seed(seed + 1)
            init_a = initial_audio_latent.float()
            if init_a.dim() == 2:
                init_a = init_a.unsqueeze(0)
            audio_lat = torch.zeros(B, audio_N, self.in_channels)
            audio_lat[:, :audio_N_real, :] = init_a[:, :audio_N_real, :]
            noise_a = torch.randn_like(audio_lat)
            audio_lat = audio_lat * (1.0 - noise_scale) + noise_a * noise_scale
        else:
            audio_lat_real = torch.randn(B, audio_N_real, self.in_channels, dtype=torch.bfloat16).float() * sigmas[0]
            audio_lat = torch.zeros(B, audio_N, self.in_channels)
            audio_lat[:, :audio_N_real, :] = audio_lat_real

        do_cfg = video_cfg_scale > 1.0 or audio_cfg_scale > 1.0
        do_stg = video_stg_scale != 0.0 or audio_stg_scale != 0.0
        do_mod = video_modality_scale != 1.0 or audio_modality_scale != 1.0

        # Gradient estimation state (tracks velocity for velocity correction)
        prev_v_vel = None
        prev_a_vel = None

        for step_idx in range(num_inference_steps):
            sigma = sigmas[step_idx].item()
            sigma_next = sigmas[step_idx + 1].item()

            def _run(vp, ap, skip_ca=False, skip_sa_blocks=None):
                # video_N / audio_N here are LOGICAL (unpadded) — passed as ``logical_n``
                # to ring SDPA so padded K positions get masked. Tensor shapes themselves
                # carry the padded counts.
                v, a = self.transformer.forward(
                    video_1BNI_torch=video_lat.unsqueeze(0),
                    video_prompt_1BLP=vp,
                    video_rope_cos=v_cos,
                    video_rope_sin=v_sin,
                    video_N=video_N_real,
                    audio_1BNI_torch=audio_lat.unsqueeze(0),
                    audio_prompt_1BLP=ap,
                    audio_rope_cos=a_cos,
                    audio_rope_sin=a_sin,
                    audio_N=audio_N,
                    trans_mat=trans_mat,
                    timestep_torch=torch.tensor([sigma]),
                    video_cross_pe_cos=v_xpe_cos,
                    video_cross_pe_sin=v_xpe_sin,
                    audio_cross_pe_cos=a_xpe_cos,
                    audio_cross_pe_sin=a_xpe_sin,
                    video_cross_pe_cos_full=v_xpe_cos_full,
                    video_cross_pe_sin_full=v_xpe_sin_full,
                    audio_cross_pe_cos_full=a_xpe_cos_full,
                    audio_cross_pe_sin_full=a_xpe_sin_full,
                    skip_cross_attn=skip_ca,
                    skip_self_attn_blocks=skip_sa_blocks,
                    audio_attn_mask=tt_attn_mask,
                    audio_padding_mask=tt_pad_mask_sp,
                    audio_padding_mask_full=tt_pad_mask_full,
                    video_padding_mask=tt_v_pad_mask_sp,
                )
                vv = LTXTransformerModel.device_to_host(
                    v,
                    ccl_manager=self.ccl_manager,
                    parallel_config=self.parallel_config,
                    sp_already_gathered=True,
                    tp_already_gathered=True,
                ).squeeze(0)
                av = LTXTransformerModel.device_to_host(
                    a,
                    ccl_manager=self.ccl_manager,
                    parallel_config=self.parallel_config,
                    sp_already_gathered=True,
                    tp_already_gathered=True,
                ).squeeze(0)
                vd = (video_lat.bfloat16().float() - vv.float() * sigma).bfloat16()
                ad = (audio_lat.bfloat16().float() - av.float() * sigma).bfloat16()
                return self._zero_sp_padding(vd, video_N_real), self._zero_sp_padding(ad, audio_N_real)

            v_den, a_den = _run(tt_vp, tt_ap)

            v_unc = a_unc = v_ptb = a_ptb = v_iso = a_iso = 0.0
            if do_cfg:
                v_unc, a_unc = _run(tt_nv, tt_na)
            if do_stg:
                v_ptb, a_ptb = _run(tt_vp, tt_ap, skip_sa_blocks=[stg_block])
            if do_mod:
                v_iso, a_iso = _run(tt_vp, tt_ap, skip_ca=True)

            if do_cfg or do_stg or do_mod:
                v_den = self._apply_modal_guidance(
                    v_den,
                    v_unc,
                    v_ptb,
                    v_iso,
                    cfg_scale=video_cfg_scale,
                    stg_scale=video_stg_scale,
                    modality_scale=video_modality_scale,
                    rescale_scale=rescale_scale,
                    do_cfg=do_cfg,
                    do_stg=do_stg,
                    do_mod=do_mod,
                    real_token_count=video_N_real if video_N > video_N_real else None,
                )
                a_den = self._apply_modal_guidance(
                    a_den,
                    a_unc,
                    a_ptb,
                    a_iso,
                    cfg_scale=audio_cfg_scale,
                    stg_scale=audio_stg_scale,
                    modality_scale=audio_modality_scale,
                    rescale_scale=rescale_scale,
                    do_cfg=do_cfg,
                    do_stg=do_stg,
                    do_mod=do_mod,
                    real_token_count=audio_N_real,
                )

            # Gradient estimation: correct velocity using previous step's velocity.
            # GE math operates on real (unpadded) slices only — padded slots are noise-free
            # placeholders, and including them here would leak garbage into the GE state.
            if ge_gamma != 0.0 and sigma_next != 0.0:
                v_lat_real = video_lat[:, :video_N_real, :]
                v_den_real = v_den[:, :video_N_real, :]
                v_velocity = (v_lat_real.float() - v_den_real.float()) / sigma
                a_lat_real = audio_lat[:, :audio_N_real, :]
                a_den_real = a_den[:, :audio_N_real, :]
                a_velocity = (a_lat_real.float() - a_den_real.float()) / sigma
                if prev_v_vel is not None:
                    v_total = ge_gamma * (v_velocity - prev_v_vel) + prev_v_vel
                    a_total = ge_gamma * (a_velocity - prev_a_vel) + prev_a_vel
                    v_den_real = (v_lat_real.float() - v_total * sigma).bfloat16()
                    a_den_real = (a_lat_real.float() - a_total * sigma).bfloat16()
                    # v_den/a_den came pre-zeroed at padded slots from `_run` — only
                    # the real slice needs updating.
                    v_den[:, :video_N_real, :] = v_den_real
                    a_den[:, :audio_N_real, :] = a_den_real
                prev_v_vel, prev_a_vel = v_velocity, a_velocity

            # Last step: return denoised directly (sigma_next == 0)
            if sigma_next == 0.0:
                video_lat_new = v_den.float()
                audio_lat_new = a_den.float()
            else:
                video_lat_new = euler_step(video_lat, v_den.float(), sigma, sigma_next).bfloat16().float()
                audio_lat_new = euler_step(audio_lat, a_den.float(), sigma, sigma_next).bfloat16().float()
            # Re-zero padded slots after each step to keep the latent clean.
            video_lat = self._zero_sp_padding(video_lat_new, video_N_real)
            audio_lat = self._zero_sp_padding(audio_lat_new, audio_N_real)

            if (step_idx + 1) % 5 == 0 or step_idx == 0:
                logger.info(f"Step {step_idx+1}/{num_inference_steps}: σ {sigma:.4f}→{sigma_next:.4f}")

        logger.info(
            f"AV done. video: ({B},{video_N_real},{self.in_channels}), "
            f"audio: ({B},{audio_N_real},{self.in_channels})"
        )
        return video_lat[:, :video_N_real, :], audio_lat[:, :audio_N_real, :]

    def _new_audio_decoder(self) -> None:
        """Construct audio decoder module shells from checkpoint config.

        No weights are loaded — ``_prepare_audio_decoder`` handles that via the
        disk cache the same way ``_prepare_vae`` does for the video VAE.
        """
        from ...models.audio_vae.audio_decoder_ltx import AudioDecoder
        from ...models.audio_vae.bwe_ltx import MelSTFT, VocoderWithBWE
        from ...models.audio_vae.vocoder_ltx import Vocoder

        with safe_open(self.checkpoint_name, framework="pt") as f:
            config = json.loads(f.metadata()["config"])

        ad = config["audio_vae"]["model"]["params"]
        ddconfig = ad["ddconfig"]
        stft_cfg = config["audio_vae"].get("preprocessing", {}).get("stft", {})
        mel_cfg = config["audio_vae"].get("preprocessing", {}).get("mel", {})
        mel_bins = ddconfig.get("mel_bins") or mel_cfg.get("n_mel_channels")
        if ddconfig.get("norm_type", "pixel") != "pixel" or ddconfig.get("causality_axis", "height") != "height":
            logger.warning("Audio decoder: checkpoint requires unsupported norm/causality; skipping construction")
            return

        voc_cfg = config["vocoder"]["vocoder"]
        bwe_cfg = config["vocoder"]["bwe"]

        mesh_shape = tuple(self.mesh_device.shape)
        t_axis = 0 if mesh_shape[0] >= mesh_shape[1] else 1
        t_factor = mesh_shape[t_axis]
        c_axis = 1 - t_axis
        c_factor = mesh_shape[c_axis]
        # Opt-in channel-TP: T-halo on the larger axis + channel tensor-parallel on
        # the other (sound — channels have no sequence boundary). Off by default
        # (single-axis is the production path); enable with LTX_AUDIO_CHANNEL_TP=1.
        channel_tp_on = os.environ.get("LTX_AUDIO_CHANNEL_TP", "0") == "1"
        if t_factor > 1 and c_factor > 1 and channel_tp_on:
            audio_parallel_config = AudioTCParallelConfig(
                time_parallel=ParallelFactor(factor=t_factor, mesh_axis=t_axis),
                channel_parallel=ParallelFactor(factor=c_factor, mesh_axis=c_axis),
            )
        elif t_factor > 1:
            audio_parallel_config = ParallelFactor(factor=t_factor, mesh_axis=t_axis)
        else:
            audio_parallel_config = None
        audio_ccl = self.vae_ccl_manager if audio_parallel_config is not None else None

        self.tt_audio_decoder = AudioDecoder(
            ch=ddconfig.get("ch", 128),
            out_ch=ddconfig.get("out_ch", 2),
            ch_mult=tuple(ddconfig.get("ch_mult", (1, 2, 4))),
            num_res_blocks=ddconfig.get("num_res_blocks", 2),
            attn_resolutions=tuple(ddconfig.get("attn_resolutions", ())),
            resolution=ddconfig.get("resolution", 256),
            z_channels=ddconfig.get("z_channels", 8),
            mid_block_add_attention=ddconfig.get("mid_block_add_attention", False),
            sample_rate=ad.get("sampling_rate", 16000),
            mel_hop_length=stft_cfg.get("hop_length", 160),
            is_causal=stft_cfg.get("causal", True),
            mel_bins=mel_bins,
            mesh_device=self.mesh_device,
            dtype=ttnn.bfloat16,
        )

        # Architecture params come from the checkpoint config; Vocoder owns the
        # defaults for anything it omits (its defaults are the LTX-2 main-vocoder values).
        voc_keys = (
            "resblock_kernel_sizes",
            "upsample_rates",
            "upsample_kernel_sizes",
            "resblock_dilation_sizes",
            "upsample_initial_channel",
            "resblock",
            "activation",
            "use_tanh_at_final",
            "use_bias_at_final",
        )

        def _tt_vocoder(cfg: dict, *, apply_final_activation: bool, parallel_config) -> Vocoder:
            return Vocoder(
                **{k: cfg[k] for k in voc_keys if k in cfg},
                apply_final_activation=apply_final_activation,
                mesh_device=self.mesh_device,
                dtype=ttnn.float32,
                parallel_config=parallel_config,
                ccl_manager=audio_ccl,
            )

        # BWE stays single-axis: its channel-TP diverges in the full pipeline (the
        # main vocoder's channel-TP is exact). Root cause open.
        bwe_pc = (
            audio_parallel_config.time_parallel
            if isinstance(audio_parallel_config, AudioTCParallelConfig)
            else audio_parallel_config
        )
        main_voc = _tt_vocoder(voc_cfg, apply_final_activation=True, parallel_config=audio_parallel_config)
        bwe_voc = _tt_vocoder(bwe_cfg, apply_final_activation=False, parallel_config=bwe_pc)
        mel_stft = MelSTFT(
            filter_length=bwe_cfg["n_fft"],
            hop_length=bwe_cfg["hop_length"],
            win_length=bwe_cfg["n_fft"],
            n_mel_channels=bwe_cfg["num_mels"],
            mesh_device=self.mesh_device,
            dtype=ttnn.float32,
        )
        self.tt_vocoder_with_bwe = VocoderWithBWE(
            vocoder=main_voc,
            bwe_generator=bwe_voc,
            mel_stft=mel_stft,
            input_sampling_rate=bwe_cfg["input_sampling_rate"],
            output_sampling_rate=bwe_cfg["output_sampling_rate"],
            hop_length=bwe_cfg["hop_length"],
            mesh_device=self.mesh_device,
            dtype=ttnn.float32,
        )
        if isinstance(audio_parallel_config, AudioTCParallelConfig):
            cfg_desc = f"T-shard={t_factor} axis{t_axis} + channel-TP={c_factor} axis{c_axis}"
        elif audio_parallel_config is not None:
            cfg_desc = f"T-shard={t_factor} axis{t_axis} (single-axis)"
        else:
            cfg_desc = "replicated"
        logger.info(f"Constructed audio decoder shells (mesh {mesh_shape}, vocoder {cfg_desc})")

    def _audio_decoder_state_provider(self) -> dict[str, torch.Tensor]:
        """Audio mel-VAE decoder weights from the checkpoint, prefix-stripped to the module keys."""
        logger.info("Audio decoder cache miss — loading from checkpoint")
        state: dict[str, torch.Tensor] = {}
        with safe_open(self.checkpoint_name, framework="pt") as f:
            for k in f.keys():
                if k.startswith("audio_vae.decoder."):
                    state[k.removeprefix("audio_vae.decoder.")] = f.get_tensor(k).float()
                elif k.startswith("audio_vae.per_channel_statistics."):
                    state[k.removeprefix("audio_vae.")] = f.get_tensor(k).float()
        return state

    def _vocoder_state_provider(self) -> dict[str, torch.Tensor]:
        """Vocoder (+ BWE) weights from the checkpoint, prefix-stripped to the module keys."""
        logger.info("Audio vocoder cache miss — loading from checkpoint")
        state: dict[str, torch.Tensor] = {}
        with safe_open(self.checkpoint_name, framework="pt") as f:
            for k in f.keys():
                if k.startswith("vocoder."):
                    state[k.removeprefix("vocoder.")] = f.get_tensor(k).float()
        return state

    def _prepare_audio_decoder(self) -> None:
        """Load audio decoder + vocoder weights via the disk cache (mirrors ``_prepare_vae``).

        The per-channel denormalize stats are re-injected separately after load: they are
        non-Parameter buffers, so the binary cache path doesn't carry them and would
        otherwise leave garbage stats (and a garbage audio track).
        """
        if self.tt_audio_decoder is None or self.tt_vocoder_with_bwe is None:
            return
        if self.tt_audio_decoder.is_loaded() and self.tt_vocoder_with_bwe.is_loaded():
            return

        model_name = os.path.basename(self.checkpoint_name).removesuffix(".safetensors")
        blocking_key = conv3d_blocking_hash(self.tt_vocoder_with_bwe)
        dec_subfolder = f"audio_dec_{blocking_key}" if blocking_key else "audio_dec"
        voc_subfolder = f"audio_voc_{blocking_key}" if blocking_key else "audio_voc"

        if not self.tt_audio_decoder.is_loaded():
            cache_module.load_model(
                self.tt_audio_decoder,
                model_name=model_name,
                subfolder=dec_subfolder,
                parallel_config=self.parallel_config,
                mesh_shape=tuple(self.mesh_device.shape),
                get_torch_state_dict=self._audio_decoder_state_provider,
            )

        if not self.tt_vocoder_with_bwe.is_loaded():
            # The resampler's Hann filter is a non-persistent reference buffer, so
            # it is held as a plain device tensor (not a Parameter) and recomputed
            # at init — the strict state-dict load neither expects nor receives it.
            cache_module.load_model(
                self.tt_vocoder_with_bwe,
                model_name=model_name,
                subfolder=voc_subfolder,
                parallel_config=self.parallel_config,
                mesh_shape=tuple(self.mesh_device.shape),
                get_torch_state_dict=self._vocoder_state_provider,
            )

        # Re-inject the mel-VAE per-channel stats from the checkpoint — see docstring.
        with safe_open(self.checkpoint_name, framework="pt") as f:
            self.tt_audio_decoder.set_per_channel_stats(
                f.get_tensor("audio_vae.per_channel_statistics.std-of-means"),
                f.get_tensor("audio_vae.per_channel_statistics.mean-of-means"),
            )

        logger.info("Loaded TTNN audio decoder + vocoder")

    def decode_audio(self, audio_latent: torch.Tensor, num_frames: int, fps: float = 24.0) -> Audio:
        """Decode an audio latent ``(1, audio_N, 128)`` to a waveform, fully on device.

        The single audio-decode entry point for every LTX pipeline: routes the latent
        through ``AudioDecoder`` (mel-VAE) then ``VocoderWithBWE`` (vocoder + BWE).
        Both take and return torch tensors, handling device upload/download internally.
        ``num_frames``/``fps`` trim the output to the clip duration.
        """
        assert self.checkpoint_name is not None, "checkpoint_name must be set before decode_audio"

        # VAE and audio modules can't be L1-coresident on BH-LB. With
        # dynamic_load the audio (re)load below evicts the VAE via the
        # coresident exclusions registered in `_register_coresident_exclusions`;
        # without dynamic_load nothing evicts, so free the VAE explicitly.
        if not self.dynamic_load and self.vae_decoder is not None and self.vae_decoder.is_loaded():
            self.vae_decoder.deallocate_weights()

        # Shells are built by `_new_audio_decoder` at instantiation time; this
        # method only loads weights into them.
        assert self.tt_audio_decoder is not None and self.tt_vocoder_with_bwe is not None, (
            "audio decoder shells not built — _new_audio_decoder() must run first "
            "(it does, via _instantiate_modules, when checkpoint_name is set at construction)"
        )
        self._prepare_audio_decoder()

        # Unpatchify: (1, audio_N, 128) -> (1, z, audio_N, 128 // z)
        # (z=z_channels, 128 // z is the patchify freq).
        audio_N = audio_latent.shape[1]
        z = self.tt_audio_decoder.z_channels
        audio_spatial = audio_latent.reshape(1, audio_N, z, audio_latent.shape[2] // z).permute(0, 2, 1, 3).float()

        mel = self.tt_audio_decoder(audio_spatial)
        waveform = self.tt_vocoder_with_bwe(mel).squeeze(0).float()
        sampling_rate = self.tt_vocoder_with_bwe.output_sampling_rate

        # Trim to video duration.
        video_duration = num_frames / fps
        target_samples = int(video_duration * sampling_rate)
        if waveform.shape[-1] > target_samples:
            waveform = waveform[..., :target_samples]

        logger.info(
            f"Audio decoded on device: {tuple(waveform.shape)} "
            f"({waveform.shape[-1] / sampling_rate:.2f}s @ {sampling_rate}Hz)"
        )
        return Audio(waveform=waveform, sampling_rate=sampling_rate)
