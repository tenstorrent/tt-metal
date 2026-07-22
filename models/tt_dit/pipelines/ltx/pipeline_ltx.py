# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2 audio-video generation pipeline for tt_dit: on-device Gemma text encoding,
LTX-2 sigma schedule, joint AV denoise (Euler + multi-modal guidance), and on-device
VAE video + audio decode.

Reference: LTX-2/packages/ltx-pipelines/ + Wan pipeline_wan.py
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field

import torch
from huggingface_hub import hf_hub_download
from loguru import logger

import ttnn

from ...encoders.gemma.encoder_pair import GemmaTokenizerEncoderPair
from ...models.audio_vae.audio_decoder_ltx import LTXAudioDecoderAdapter
from ...models.transformers.ltx.rope_ltx import prepare_audio_rope, prepare_av_cross_pe, prepare_video_rope
from ...models.transformers.ltx.transformer_ltx import (
    LTXTransformerCheckpoint,
    LTXTransformerModel,
    build_audio_masks,
    build_video_pad_mask,
)
from ...models.upsampler.latent_upsampler_ltx import LTXLatentUpsampler
from ...models.vae.vae_ltx import LTXVideoVAEAdapter, upsample_latent
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor, VaeHWParallelConfig
from ...parallel.manager import CCLManager
from ...utils.fuse_loras import LoraSpec
from ...utils.ltx import SPATIAL_COMPRESSION, TEMPORAL_COMPRESSION, ceil_to, latent_grid
from ...utils.mochi import get_rot_transformation_mat
from ...utils.patchifiers import AudioLatentShape, VideoPixelShape
from ...utils.tensor import bf16_tensor
from ...utils.tracing import StateTensor
from ...utils.video import Audio

LTX_UPSAMPLER_HF_REF = "Lightricks/LTX-2.3:ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

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


@dataclass
class TransformerState:
    model: LTXTransformerModel
    checkpoint: LTXTransformerCheckpoint
    lora_specs: list[LoraSpec] = field(default_factory=list)


class LTXTransformerState:
    """Per-stage (s1/s2) persistent trace I/O.

    Static per-shape inputs (rope/cross-PE/masks/trans_mat) are bound once; the latent buffers and
    timestep are refreshed each step — in place when traced (a ttnn trace bakes their addresses),
    rebound otherwise. ``__getattr__`` returns the underlying tensor: update via
    ``state._tt_x.update(...)``, read via ``state.tt_x``.
    """

    def __init__(self) -> None:
        self._tt_video_lat = StateTensor()
        self._tt_audio_lat = StateTensor()
        self._tt_timestep = StateTensor()
        self._tt_video_timestep = StateTensor()
        self._tt_video_ts_pair = StateTensor()
        self._tt_video_pin_mask = StateTensor()
        self._tt_i2v_mask = StateTensor()
        self._tt_i2v_clean = StateTensor()
        self._tt_video_pad_mask = StateTensor()
        self._tt_audio_pad_mask = StateTensor()
        self._tt_video_rope_cos = StateTensor()
        self._tt_video_rope_sin = StateTensor()
        self._tt_audio_rope_cos = StateTensor()
        self._tt_audio_rope_sin = StateTensor()
        self._tt_trans_mat = StateTensor()
        self._tt_video_cross_pe_cos = StateTensor()
        self._tt_video_cross_pe_sin = StateTensor()
        self._tt_audio_cross_pe_cos = StateTensor()
        self._tt_audio_cross_pe_sin = StateTensor()
        self._tt_audio_cross_pe_cos_full = StateTensor()
        self._tt_audio_cross_pe_sin_full = StateTensor()
        self._tt_audio_attn_mask = StateTensor()
        self._tt_audio_padding_mask = StateTensor()
        self._tt_audio_padding_mask_full = StateTensor()
        self._tt_video_padding_mask = StateTensor()

    def __getattr__(self, name: str) -> ttnn.Tensor | None:
        return object.__getattribute__(self, f"_{name}")._value


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

    Shared base for the concrete LTX variants (one-stage / distilled / two-stages); it owns
    the device machinery (loaders, ``call_av``, encode/decode) but not ``generate`` /
    ``warmup_buffers`` — those live on the concrete subclasses.

    Usage:
        pipeline = LTXOneStagePipeline.create_pipeline(
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
        is_fsdp: bool = False,
        dynamic_load: bool = False,
        num_frames: int = 0,
        height: int = 0,
        width: int = 0,
        run_warmup: bool = False,
        traced: bool = False,
        extra_transformer_variants: list[tuple[str, list[LoraSpec]]] | None = None,
        image_conditioning: bool | None = None,
    ):
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=self.mesh_device.shape[1], mesh_axis=1),
        )
        self._traced = traced
        self._trace_state: dict[str, LTXTransformerState] = {}
        self._prompt_v = StateTensor()
        self._prompt_a = StateTensor()
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

        self.is_fsdp = is_fsdp
        self.dynamic_load = dynamic_load
        self._init_num_frames = num_frames
        self._init_height = height
        self._init_width = width

        self.checkpoint_name: str | None = (
            LTXPipeline._resolve_checkpoint_file(checkpoint_name) if checkpoint_name else None
        )
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

        self.transformer: LTXTransformerModel | None = None
        self.transformer_states: list[TransformerState] = []
        self.vae: LTXVideoVAEAdapter | None = None
        self._i2v_cond_cache: dict[tuple[str, int, int], torch.Tensor] = {}
        self.upsampler: LTXLatentUpsampler | None = None
        self._audio_adapter: LTXAudioDecoderAdapter | None = None

        # Whether the transformer is built for the per-token video-timestep (I2V) modulation path.
        # Driven purely by whether the caller has a conditioning image: it passes
        # ``image_conditioning=bool(image_path)`` so the presence of a path turns on the dense-
        # modulation path, while pure T2V (no image) keeps the fast scalar-AdaLN path. Defaults to
        # T2V when unset. Resolved before ``_instantiate_modules``.
        self._image_conditioning: bool = bool(image_conditioning)

        if self.checkpoint_name is not None:
            self._instantiate_modules(extra_transformer_variants or [])
            self._register_coresident_exclusions()
            self._prime_caches()
            valid_shape = num_frames > 0 and height > 0 and width > 0
            if (run_warmup or traced) and valid_shape:
                if traced and not run_warmup:
                    logger.info("traced=True: forcing warmup (trace capture requires precompiled kernels)")
                self.warmup_buffers(num_frames=num_frames, height=height, width=width)
            elif traced:
                logger.warning(
                    f"traced=True but invalid shape ({num_frames=}, {height=}, {width=}); "
                    "skipping forced warmup — trace capture will likely fail"
                )

    def release_traces(self) -> None:
        """Release captured denoise traces and free their device trace memory."""
        if self.transformer is not None:
            for tracer in LTXTransformerModel.inner_step._tracers_keyed.get(self.transformer, {}).values():
                tracer.release_trace()
        if self.tt_vocoder_with_bwe is not None:
            self.tt_vocoder_with_bwe.release_trace()
        if self.tt_mel_decoder is not None:
            self.tt_mel_decoder.release_trace()
        self._trace_state.clear()
        self._prompt_v = StateTensor()
        self._prompt_a = StateTensor()

    @property
    def vae_decoder(self):
        """Underlying VAE decoder ``Module`` (or ``None``) — used by coresident exclusions,
        warmup, and ``decode_latents``."""
        return self.vae.decoder if self.vae is not None else None

    @property
    def vae_encoder(self):
        """Underlying VAE encoder ``Module`` (or ``None``) — used by coresident exclusions,
        warmup, and ``encode_image``."""
        return self.vae.encoder if self.vae is not None else None

    @property
    def tt_mel_decoder(self):
        """Underlying mel-VAE decoder ``Module`` (or ``None``) — used by ``release_traces``,
        ``_decode_mel``, and ``decode_audio``."""
        return self._audio_adapter.mel_decoder if self._audio_adapter is not None else None

    @property
    def tt_vocoder_with_bwe(self):
        """Underlying vocoder+BWE ``Module`` (or ``None``) — used by ``release_traces`` and
        ``decode_audio``."""
        return self._audio_adapter.vocoder_with_bwe if self._audio_adapter is not None else None

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

    @classmethod
    def create_pipeline(
        cls,
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

        pipeline_cls = pipeline_class or cls
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

    def _build_transformer(self, checkpoint: LTXTransformerCheckpoint) -> LTXTransformerModel:
        """Build a transformer for ``checkpoint``, passing the pipeline-level flags
        (``has_audio`` / ``image_conditioning``) in explicitly — the checkpoint never
        reads pipeline state. ``image_conditioning`` needs the VAE adapter's encoder blocks,
        so the adapter must be constructed before this is called."""
        return checkpoint.build(
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
            # I2V-capable requires the VAE encoder, but only *activate* the heavier per-token
            # modulation when a conditioning image is actually passed in (``self._image_conditioning``,
            # auto-detected from the conditioning-image path or forced by the caller). Pure T2V keeps
            # the fast scalar-AdaLN path — no separate on/off flag.
            image_conditioning=bool(self.vae is not None and self.vae.encoder_blocks and self._image_conditioning),
        )

    def _instantiate_modules(self, extra_variants: list[tuple[str, list[LoraSpec]]]) -> None:
        """Build every TT Module the pipeline will use. No DRAM weights yet —
        ``_prime_caches`` (next) attaches them."""
        # VAE adapter first: it parses the VAE config that the transformer's ``image_conditioning``
        # flag depends on (whether encoder blocks exist).
        self.vae = LTXVideoVAEAdapter(
            self.checkpoint_name,
            mesh_device=self.mesh_device,
            vae_parallel_config=self.vae_parallel_config,
            vae_ccl_manager=self.vae_ccl_manager,
            dit_parallel_config=self.parallel_config,
            num_frames=self._init_num_frames,
            height=self._init_height,
            width=self._init_width,
        )

        self.transformer_checkpoint = LTXTransformerCheckpoint(self.checkpoint_name, inner_dim=self.inner_dim)
        self.transformer = self._build_transformer(self.transformer_checkpoint)
        self.transformer_states.append(TransformerState(model=self.transformer, checkpoint=self.transformer_checkpoint))
        for tag, lora_specs in extra_variants:
            specs = list(lora_specs)
            self.transformer_states.append(
                TransformerState(
                    model=self._build_transformer(self.transformer_checkpoint),
                    checkpoint=self.transformer_checkpoint,
                    lora_specs=specs,
                )
            )
            logger.info(f"Registered transformer variant {tag} with {len(specs)} LoRA(s)")

        if self.HAS_UPSAMPLER:
            assert (
                self._init_height > 0 and self._init_width > 0 and self._init_num_frames > 0
            ), f"{type(self).__name__} requires num_frames/height/width at create_pipeline."
            self._upsampler_path = LTXPipeline._resolve_checkpoint_file(LTX_UPSAMPLER_HF_REF)
            # Spatial-2x latent upsampler at stage-1 shape: input H/W = full // (SPATIAL_COMPRESSION*2)
            # (stage 1 runs at half-res, then the VAE compresses by SPATIAL_COMPRESSION),
            # latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1. The pipeline computes the
            # stage-1 shape and passes it in; the model reads its own JSON config from the checkpoint.
            upsampler_latent_frames = (self._init_num_frames - 1) // TEMPORAL_COMPRESSION + 1
            hf = self.vae_parallel_config.height_parallel.factor
            wf = self.vae_parallel_config.width_parallel.factor
            upsampler_input_hw = (
                ceil_to(self._init_height // (SPATIAL_COMPRESSION * 2), hf),
                ceil_to(self._init_width // (SPATIAL_COMPRESSION * 2), wf),
            )
            self.upsampler = LTXLatentUpsampler.from_checkpoint(
                self._upsampler_path,
                input_hw=upsampler_input_hw,
                latent_frames=upsampler_latent_frames,
                mesh_device=self.mesh_device,
                parallel_config=self.vae_parallel_config,
                ccl_manager=self.vae_ccl_manager,
                dit_parallel_config=self.parallel_config,
            )

        if self.checkpoint_name is not None:
            # Audio decode stack (mel-VAE decoder + vocoder/BWE). No weights are loaded here —
            # ``_prepare_audio_decoder`` attaches them via the disk cache, like ``_prepare_vae``.
            self._audio_adapter = LTXAudioDecoderAdapter(
                self.checkpoint_name,
                mesh_device=self.mesh_device,
                vae_ccl_manager=self.vae_ccl_manager,
                dit_parallel_config=self.parallel_config,
                traced=self._traced,
            )

    def _register_coresident_exclusions(self) -> None:
        """All transformer variants exclude each other AND the VAE. LTX-22B +
        LTX-VAE don't both fit on BH LB; VAE must evict the active transformer
        before decode. LTX needs the eviction on both WH and BH."""
        if not self.dynamic_load:
            return
        models = [s.model for s in self.transformer_states]

        for i, m in enumerate(models):
            m._coresident_peers = [*models[:i], *models[i + 1 :], self.vae_decoder, self.vae_encoder]
        if self.vae_decoder is not None:
            self.vae_decoder._coresident_peers = [*models, self.upsampler, self.vae_encoder]
        if self.upsampler is not None:
            self.upsampler._coresident_peers = [self.vae_decoder, self.vae_encoder]
        if self.vae_encoder is not None:
            self.vae_encoder._coresident_peers = [*models, self.vae_decoder, self.upsampler]

        for m in [*models, self.vae_decoder, self.upsampler, self.vae_encoder]:
            if m is not None:
                for peer in m._coresident_peers:
                    if peer is not None:
                        m.register_coresident_exclusions(peer)

        encoder_peers = [*models] + ([self.vae_decoder] if self.vae_decoder is not None else [])
        self.gemma_encoder_pair.register_coresident_peers(encoder_peers)

    def _prime_caches(self) -> None:
        """Load every module in reverse use order so variant 0 is resident in
        DRAM after ``__init__`` (matches Wan's reverse-use-order priming)."""
        for idx in range(len(self.transformer_states) - 1, 0, -1):
            self._prepare_transformer(idx)
        self._prepare_vae()
        self._prepare_upsampler()
        self._prepare_audio_decoder()
        self._prepare_transformer(0)

    def _prepare_transformer(self, idx: int = 0) -> None:
        state = self.transformer_states[idx]
        state.checkpoint.load(
            state.model,
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            is_fsdp=self.is_fsdp,
            lora_specs=state.lora_specs,
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

    def _prepare_vae(self) -> None:
        """Delegate: push VAE decoder weights onto the mesh (see ``LTXVideoVAEAdapter``)."""
        if self.vae is not None:
            self.vae.reload_decoder()

    def _prepare_vae_encoder(self) -> None:
        """Delegate: push VAE encoder weights onto the mesh (see ``LTXVideoVAEAdapter``)."""
        if self.vae is not None:
            self.vae.reload_encoder()

    def encode_image(self, image_BCFHW: torch.Tensor) -> torch.Tensor:
        """Encode a conditioning image/clip ``(B, 3, F, H, W)`` in [-1, 1] to a normalized
        latent ``(B, 128, F', H', W')``. Loads the encoder (evicting coresident peers).

        Device-vs-reference parity for this encoder is covered by ``test_ltx_video_encoder_prod``
        in ``tests/models/ltx/test_vae_ltx.py`` (diffusers reference, sharded like this path).
        """
        assert self.vae_encoder is not None, "VAE encoder not constructed (no encoder_blocks in checkpoint?)"
        self._prepare_vae_encoder()
        return self.vae_encoder(image_BCFHW)

    def decode_latents(
        self, latent: torch.Tensor, latent_frames: int, latent_h: int, latent_w: int, *, output_type: str = "float"
    ) -> torch.Tensor:
        """Decode latent tensor to video pixels.

        Args:
            latent: (B, num_tokens, C) flat latent from denoising loop
            latent_frames, latent_h, latent_w: Spatial dimensions
            output_type: "float" → (B, 3, F, H, W) torch in [-1, 1] (for in-pipeline export);
                         "rgb"   → (B, 3, F, H, W) uint8 numpy, RGB planar

        Returns:
            decoded video in the requested format
        """
        if self.vae_decoder is None:
            logger.warning("No VAE decoder loaded, returning raw latent")
            return latent

        B = latent.shape[0]
        # Reshape flat tokens to spatial: (B, num_tokens, C) -> (B, C, F', H', W')
        latent_spatial = latent.reshape(B, latent_frames, latent_h, latent_w, self.in_channels)
        latent_spatial = latent_spatial.permute(0, 4, 1, 2, 3)  # BCTHW

        video = self.vae_decoder(latent_spatial, output_type=output_type)
        if output_type != "float":
            return video.numpy()
        return video

    def _vae_per_channel_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Delegate: cached ``(mean-of-means, std-of-means)`` from the VAE adapter."""
        return self.vae.per_channel_stats()

    def _prepare_upsampler(self) -> None:
        """Delegate: push upsampler weights onto the mesh (see ``LTXLatentUpsampler.reload_weights``)."""
        if self.upsampler is not None:
            self.upsampler.reload_weights()

    def _warmup_upsample(self, num_frames: int, height: int, width: int) -> None:
        """JIT-compile the upsampler at the stage-1 half-res shape."""
        if self.upsampler is None:
            return
        latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
        s1_lh, s1_lw = height // (SPATIAL_COMPRESSION * 2), width // (SPATIAL_COMPRESSION * 2)
        dummy = torch.zeros(1, self.in_channels, latent_frames, s1_lh, s1_lw)
        self._prepare_upsampler()
        upsample_latent(self.upsampler, dummy, *self._vae_per_channel_stats())

    def _warmup_encode(self, height: int, width: int) -> None:
        """Load the VAE encoder + JIT-compile encode kernels with a zero single-frame image
        at the target resolution. No-op when no encoder is configured (non-I2V checkpoints).

        Always device-only (bypasses ``encode_image``'s host parity path): a zeros input gives a
        degenerate PCC, and the meaningful device-vs-host comparison is logged on the real
        conditioning image during generate()."""
        if self.vae_encoder is None:
            return
        self._prepare_vae_encoder()
        self.vae_encoder(torch.zeros(1, 3, 1, height, width))

    def _warmup_decode(self, num_frames: int, height: int, width: int) -> None:
        """Load VAE + JIT-compile decode kernels with a zero dummy latent at
        the target shape. No-op if no VAE is configured."""
        if self.vae_decoder is None:
            return
        latent_frames, latent_h, latent_w = latent_grid(num_frames, height, width)
        dummy = torch.zeros(1, latent_frames * latent_h * latent_w, self.in_channels)
        self._prepare_vae()
        self.decode_latents(dummy, latent_frames, latent_h, latent_w)

    def _warmup_audio_decode(self, audio_latent: torch.Tensor, num_frames: int, fps: float = 24.0) -> None:
        """Eager (untraced) audio decode at the real shape: compiles kernels, warms lazy device
        state, and frees back to a deterministic allocator free-list so a later traced decode
        captures cleanly. No-op if the audio decoder is not configured."""
        if self.tt_mel_decoder is None or self.tt_vocoder_with_bwe is None:
            return
        mel_d, voc = self.tt_mel_decoder, self.tt_vocoder_with_bwe
        saved = (mel_d.use_trace, voc.use_trace, voc.use_trace_bwe)
        mel_d.use_trace = voc.use_trace = voc.use_trace_bwe = False
        try:
            self.decode_audio(audio_latent, num_frames, fps=fps)
        finally:
            mel_d.use_trace, voc.use_trace, voc.use_trace_bwe = saved

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
        image_cond_latent: torch.Tensor | None = None,
        image_cond_strength: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run AV denoising with full MultiModalGuider guidance. Returns (video_latent, audio_latent).

        I2V: pass ``image_cond_latent`` ``(B, 128, F', lh, lw)`` + ``image_cond_strength``. Frame-0
        tokens are pinned to the clean latent via a per-token denoise mask + per-token timesteps +
        a per-step blend (mirrors ``VideoConditionByLatentIndex`` + ``GaussianNoiser``).

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

        # needs_video_ts: transformer expects a per-token timestep (all-ones mask for pure T2V).
        image_cond = image_cond_latent is not None
        needs_video_ts = getattr(self.transformer, "image_conditioning", False)
        denoise_mask = None  # (B, video_N_real, 1)
        clean_latent = None  # (B, video_N_real, C)
        n_cond = 0
        if image_cond or needs_video_ts:
            denoise_mask = torch.ones(B, video_N_real, 1)
            if image_cond:
                cond = image_cond_latent.float()  # (B, C, F', lh, lw)
                # Patchify to token order f*lh*lw + h*lw + w (matches decode_latents reshape).
                cond_tokens = cond.permute(0, 2, 3, 4, 1).reshape(B, -1, self.in_channels)
                n_cond = cond_tokens.shape[1]
                assert n_cond <= video_N_real, f"image cond tokens {n_cond} exceed video tokens {video_N_real}"
                clean_latent = torch.zeros(B, video_N_real, self.in_channels)
                clean_latent[:, :n_cond, :] = cond_tokens
                denoise_mask[:, :n_cond, :] = 1.0 - image_cond_strength
                logger.info(f"I2V: pinning {n_cond} frame-0 tokens (strength={image_cond_strength})")

        vps = VideoPixelShape(batch=B, frames=num_frames, height=height, width=width, fps=24)
        als = AudioLatentShape.from_video_pixel_shape(vps)
        audio_N_real = als.frames
        audio_N = self._sp_pad_len(audio_N_real)

        logger.info(
            f"AV: {num_frames}f@{height}x{width}, "
            f"vN={video_N}(real={video_N_real}), aN={audio_N}(real={audio_N_real})"
        )

        v_cos, v_sin = prepare_video_rope(
            latent_frames,
            latent_h,
            latent_w,
            inner_dim=self.inner_dim,
            num_attention_heads=self.num_attention_heads,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            mesh_device=self.mesh_device,
            parallel_config=self.parallel_config,
        )
        a_cos, a_sin = prepare_audio_rope(
            audio_N,
            audio_N_real,
            theta=self.positional_embedding_theta,
            mesh_device=self.mesh_device,
            parallel_config=self.parallel_config,
        )
        (
            v_xpe_cos,
            v_xpe_sin,
            a_xpe_cos,
            a_xpe_sin,
            a_xpe_cos_full,
            a_xpe_sin_full,
        ) = prepare_av_cross_pe(
            latent_frames,
            latent_h,
            latent_w,
            audio_N,
            audio_N_real,
            theta=self.positional_embedding_theta,
            mesh_device=self.mesh_device,
            parallel_config=self.parallel_config,
        )
        trans_mat = self._prepare_trans_mat()
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tt_attn_mask, tt_pad_mask_sp, tt_pad_mask_full = build_audio_masks(
            audio_N, audio_N_real, mesh_device=self.mesh_device, sp_axis=sp_axis
        )
        tt_v_pad_mask_sp = build_video_pad_mask(video_N, video_N_real, mesh_device=self.mesh_device, sp_axis=sp_axis)

        tt_vp = self._prepare_prompt(video_prompt_embeds)
        tt_ap = bf16_tensor(audio_prompt_embeds.unsqueeze(0), device=self.mesh_device)
        tt_nv = self._prepare_prompt(neg_video_prompt_embeds) if neg_video_prompt_embeds is not None else None
        tt_na = (
            bf16_tensor(neg_audio_prompt_embeds.unsqueeze(0), device=self.mesh_device)
            if neg_audio_prompt_embeds is not None
            else None
        )

        if sigmas is None:
            sigmas = compute_sigmas(steps=num_inference_steps)
        else:
            assert (
                len(sigmas) == num_inference_steps + 1
            ), f"sigmas length {len(sigmas)} must equal num_inference_steps+1 ({num_inference_steps+1})"

        if image_cond:
            ns = noise_scale if initial_video_latent is not None else sigmas[0].item()
            if seed is not None:
                torch.manual_seed(seed)
            if initial_video_latent is not None:
                base_v = initial_video_latent.float()
                if base_v.dim() == 2:
                    base_v = base_v.unsqueeze(0)
                base_v = base_v.clone()
            else:
                base_v = torch.zeros(B, video_N_real, self.in_channels)
            base_v[:, :n_cond, :] = clean_latent[:, :n_cond, :]
            noise_v = torch.randn_like(base_v)
            scaled_mask = denoise_mask * ns
            video_lat_real = noise_v * scaled_mask + base_v * (1.0 - scaled_mask)
        elif initial_video_latent is not None:
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

            # Per-token video timestep (B, video_N): mask * sigma on real tokens, sigma on SP padding.
            video_ts = None
            if needs_video_ts:
                video_ts = torch.full((B, video_N), sigma, dtype=torch.float32)
                video_ts[:, :video_N_real] = denoise_mask[:, :, 0] * sigma

            def _run(vp, ap, skip_ca=False, skip_sa_blocks=None, video_ts=video_ts):
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
                    video_timestep_torch=video_ts,
                    video_cross_pe_cos=v_xpe_cos,
                    video_cross_pe_sin=v_xpe_sin,
                    audio_cross_pe_cos=a_xpe_cos,
                    audio_cross_pe_sin=a_xpe_sin,
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

            # I2V: pin frame-0 to the clean latent (denoised * mask + clean * (1 - mask)) pre-Euler.
            if image_cond:
                v_den_real = v_den[:, :video_N_real, :].float()
                blended = v_den_real * denoise_mask + clean_latent * (1.0 - denoise_mask)
                v_den[:, :video_N_real, :] = blended.to(v_den.dtype)

            # Gradient estimation: correct velocity using previous step's velocity.
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

    def _prepare_audio_decoder(self) -> None:
        """Delegate: load audio decoder + vocoder weights (see ``LTXAudioDecoderAdapter``)."""
        if self._audio_adapter is not None:
            self._audio_adapter.reload_weights()

    def _decode_mel(self, audio_spatial: torch.Tensor) -> torch.Tensor:
        """Run the mel-VAE decoder, traced (capture-once/replay) when the pipeline is traced."""
        if self.tt_mel_decoder.use_trace:
            return self.tt_mel_decoder.forward_traced(audio_spatial)
        return self.tt_mel_decoder(audio_spatial)

    def decode_audio(self, audio_latent: torch.Tensor, num_frames: int, fps: float = 24.0) -> Audio:
        """Decode an audio latent ``(1, audio_N, 128)`` to a waveform, fully on device.

        The single audio-decode entry point for every LTX pipeline: routes the latent
        through ``MelDecoder`` (mel-VAE) then ``VocoderWithBWE`` (vocoder + BWE).
        Both take and return torch tensors, handling device upload/download internally.
        ``num_frames``/``fps`` trim the output to the clip duration.
        """
        assert self.checkpoint_name is not None, "checkpoint_name must be set before decode_audio"

        _dump = os.environ.get("LTX_DUMP_AUDIO_LATENT")
        if _dump and float(audio_latent.abs().max()) > 0:  # skip the all-zero warmup latent
            torch.save(audio_latent.cpu(), _dump)
            logger.info(f"dumped audio latent {tuple(audio_latent.shape)} -> {_dump}")

        assert self.tt_mel_decoder is not None and self.tt_vocoder_with_bwe is not None, (
            "audio decoder shells not built — the LTXAudioDecoderAdapter must be constructed first "
            "(it is, via _instantiate_modules, when checkpoint_name is set at construction)"
        )
        self._prepare_audio_decoder()

        # Unpatchify: (1, audio_N, 128) -> (1, z, audio_N, 128 // z)
        # (z=z_channels, 128 // z is the patchify freq).
        audio_N = audio_latent.shape[1]
        z = self.tt_mel_decoder.z_channels
        audio_spatial = audio_latent.reshape(1, audio_N, z, audio_latent.shape[2] // z).permute(0, 2, 1, 3).float()

        _time_stages = os.environ.get("LTX_TIME_STAGES") in ("1", "true", "True")
        if _time_stages:
            import time as _t

            ttnn.synchronize_device(self.mesh_device)
            _t0 = _t.perf_counter()
            mel = self._decode_mel(audio_spatial)
            ttnn.synchronize_device(self.mesh_device)
            _t_vae = _t.perf_counter()
            waveform = self.tt_vocoder_with_bwe(mel).squeeze(0).float()
            ttnn.synchronize_device(self.mesh_device)
            _t_voc = _t.perf_counter()
            logger.info(
                f"STAGE_SPLIT mel_vae={(_t_vae - _t0) * 1000:.1f}ms " f"vocoder+bwe={(_t_voc - _t_vae) * 1000:.1f}ms"
            )
        else:
            mel = self._decode_mel(audio_spatial)
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
