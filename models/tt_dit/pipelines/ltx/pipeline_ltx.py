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
import sys
from dataclasses import dataclass, field
from io import BytesIO
from typing import Callable

import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file

import ttnn

from ...encoders.gemma.encoder_pair import GemmaTokenizerEncoderPair
from ...models.audio_vae.audio_decoder_ltx import LTXAudioDecoderAdapter
from ...models.transformers.ltx.rope_ltx import prepare_audio_rope, prepare_av_cross_pe, prepare_video_rope
from ...models.transformers.ltx.transformer_ltx import LTXTransformerModel, build_audio_masks, build_video_pad_mask
from ...models.upsampler.latent_upsampler_ltx import LTXLatentUpsampler
from ...models.vae.vae_ltx import LTXVideoDecoder, LTXVideoEncoder, read_vae_per_channel_stats, upsample_latent
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor, VaeHWParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache as cache_module
from ...utils import walltime
from ...utils.conv3d import conv3d_blocking_hash
from ...utils.fuse_loras import LoraSpec, fuse_loras_into
from ...utils.ltx import SPATIAL_COMPRESSION, TEMPORAL_COMPRESSION, ceil_to, latent_grid
from ...utils.mochi import get_rot_transformation_mat
from ...utils.patchifiers import AudioLatentShape, VideoPixelShape
from ...utils.progress import Watchdog
from ...utils.tensor import bf16_tensor
from ...utils.tracing import StateTensor, set_kernel_prewarm_capturing
from ...utils.video import Audio

LTX_UPSAMPLER_HF_REF = "Lightricks/LTX-2.3:ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

# I2V conditioning-image H.264 CRF: round-trip through the codec the VAE/DiT were trained on
# before encoding (a pristine image gives OOD latents). Mirrors ltx_pipelines DEFAULT_IMAGE_CRF.
DEFAULT_IMAGE_CRF = 33

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


def _ensure_ltx_reference_on_path() -> None:
    """Put the LTX-2 reference package (``ltx_core`` / ``ltx_pipelines``) on ``sys.path``.

    Used by the host VAE encoder (I2V device-vs-host parity) and ``encode_prompts_reference``.
    Honors ``LTX_REFERENCE_ROOT`` (a directory containing ``ltx_core``); otherwise falls back to
    ``<repo>/LTX-2/packages/{ltx-core,ltx-pipelines}/src`` — the layout the LTX unit tests assume
    (``git clone https://github.com/Lightricks/LTX-2`` at the repo root). No-op if already importable.
    """
    import importlib.util

    if importlib.util.find_spec("ltx_core") is not None:
        return
    repo_root = os.environ.get("TT_METAL_HOME") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    candidates = []
    if os.environ.get("LTX_REFERENCE_ROOT"):
        candidates.append(os.environ["LTX_REFERENCE_ROOT"])
    candidates.append(os.path.join(repo_root, "LTX-2", "packages", "ltx-core", "src"))
    candidates.append(os.path.join(repo_root, "LTX-2", "packages", "ltx-pipelines", "src"))
    for path in candidates:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


@dataclass
class TransformerState:
    model: LTXTransformerModel
    cache_name: str
    lora_specs: list[LoraSpec] = field(default_factory=list)
    state_dict_provider: Callable[[], dict[str, torch.Tensor]] | None = None


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
        # I2V frame-0 pin buffers: held across every denoise step and read by the (eager) pin, so
        # they MUST be pre-allocated before trace capture — otherwise they land in the trace's
        # activation region and get clobbered on replay, pinning garbage (intermittent static).
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
    # When True, the transformer uses per-token video AdaLN modulation (I2V image conditioning).
    SUPPORTS_IMAGE_CONDITIONING: bool = False

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
        audio_only: bool = False,
        extra_transformer_variants: list[tuple[str, list[LoraSpec]]] | None = None,
    ):
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=self.mesh_device.shape[1], mesh_axis=1),
        )
        # Capture the DiT forward as a ttnn trace on the first step, replay after. The Tracers
        # (one per fixed shape "s1"/"s2") live in the @traced_function cache on
        # LTXTransformerModel.inner_step, keyed per (transformer, trace_key); release_traces frees them.
        self._traced = traced
        # Per-stage (s1/s2) persistent trace I/O. A ttnn trace bakes absolute tensor addresses,
        # so static inputs are bound once and the latent/timestep buffers refreshed in place.
        self._trace_state: dict[str, LTXTransformerState] = {}
        # Prompt buffers shared by both stages (the embedding is identical), built on the first
        # traced step — before s1's capture — so they sit below both traces' activations and
        # neither replay overwrites them.
        self._prompt_v = StateTensor()
        self._prompt_a = StateTensor()
        if ccl_manager.topology == ttnn.Topology.Linear:
            self.vae_ccl_manager = ccl_manager
        else:
            self.vae_ccl_manager = CCLManager(
                mesh_device, num_links=ccl_manager.num_links, topology=ttnn.Topology.Linear
            )

        # Audio decode is torch-in/torch-out and self-contained on its own modules, so it can
        # run on a SUBMESH of the full device while the video pipeline keeps the whole mesh.
        # Fewer T-shards = fewer cross-chip causal-halo barriers (the audio vocoder's dominant
        # device-side cost): T-shard=4 on a 1x4/2x4 slice beats T-shard=8 on the full 4x8.
        # Defaults to the full mesh; LTX_AUDIO_SUBMESH=RxC slices an RxC submesh for audio.
        self.audio_mesh_device = mesh_device
        self.audio_ccl_manager = self.vae_ccl_manager
        self._owned_audio_submesh = None
        _audio_submesh = os.environ.get("LTX_AUDIO_SUBMESH")
        if _audio_submesh:
            r, c = (int(v) for v in _audio_submesh.lower().split("x"))
            full = tuple(mesh_device.shape)
            if r <= full[0] and c <= full[1] and (r, c) != full:
                self.audio_mesh_device = mesh_device.create_submesh(ttnn.MeshShape(r, c))
                self._owned_audio_submesh = self.audio_mesh_device
                self.audio_ccl_manager = CCLManager(
                    self.audio_mesh_device, num_links=ccl_manager.num_links, topology=ttnn.Topology.Linear
                )
                logger.info(f"Audio decode routed onto {r}x{c} submesh of {full[0]}x{full[1]}")
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
        self.vae_decoder = None
        self.vae_encoder = None
        # IC-LoRA reference-video encoders (multi-frame, one per stage res). Built on demand by
        # _build_ref_vae_encoders since ref_pixel_frames isn't known until warmup/generate; the
        # single-frame self.vae_encoder above can't encode a looped reference clip.
        self.vae_ref_encoder_s1 = None
        self.vae_ref_encoder_full = None
        # Memoized I2V conditioning latents, keyed by (image_path, height, width). The VAE
        # encoder is eager (not part of the denoise trace), so re-running it per generation is
        # wasteful and — under a traced replay pass — has hung the device; the encode is
        # deterministic in (path, resolution), so cache the host result and reuse it.
        self._i2v_cond_cache: dict[tuple[str, int, int], torch.Tensor] = {}
        # Same memoization for IC-LoRA reference sheets: (s1_latent, full_latent) per stage res.
        # Keyed by the sheet's CONTENT, not its path — a server decrypts each upload to a fresh temp
        # file per job, so a path key would miss on exactly the repeat-sheet case this exists for.
        self._ref_latent_cache: dict[tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.upsampler: LTXLatentUpsampler | None = None
        # Audio decode stack lives behind the adopted LTXAudioDecoderAdapter (ltx-perf refactor).
        # tt_mel_decoder / tt_vocoder_with_bwe are read-only properties delegating to it.
        self._audio_adapter: LTXAudioDecoderAdapter | None = None

        # decode_audio only touches the audio decoder + vocoder, so audio_only skips building
        # AND priming the 22B transformer / video VAE / upsampler — that prime is the bulk of a
        # cold run (~100s of 22B weight push) and is pure waste for the audio test.
        self.audio_only = audio_only

        if self.checkpoint_name is not None:
            self._load_config_from_checkpoint()
            self._instantiate_modules(extra_transformer_variants or [], audio_only=audio_only)
            self._register_coresident_exclusions()
            # LTX_QUANT selects a DiT-linear quant preset (e.g. all_bf8_lofi). Installed before
            # _prime_caches so the patched _prepare_transformer typecasts weights as they load and
            # re-typecasts after every dynamic_load reload. Off by default (baseline bf16/HiFi2).
            if not audio_only:
                self._maybe_apply_quant_config()
            self._prime_caches(audio_only=audio_only)
            # Tracing (prep_run=False) requires precompiled kernels + pre-allocated trace I/O,
            # so warmup is mandatory when traced. audio_only skips the (video) warmup entirely:
            # decode_audio compiles + captures its own trace lazily on the first call, so the
            # ~10-min video stage1/upsample/stage2/gemma warmup is pure waste for audio decode.
            valid_shape = num_frames > 0 and height > 0 and width > 0
            if (run_warmup or traced) and valid_shape and not audio_only:
                if traced and not run_warmup:
                    logger.info("traced=True: forcing warmup (trace capture requires precompiled kernels)")
                # In-process cold-start: on a cold build_key, run one capture-only warmup to record the
                # kernel manifest (genfiles only, no gcc, no dispatch), batch-compile it off-device
                # in-process, then let the real warmup below hit a warm cache -- so the reserved window
                # never holds the device for the ~500s op-by-op compile. Warm build_keys (in-run prewarm
                # already handling them) skip straight to the real warmup. Tracing is suppressed for the
                # capture pass: capture-only skips dispatch, so trace capture must wait for the real pass.
                _kp = ttnn._ttnn.device
                _cold = _kp.kernel_prewarm_cold_start_needed()
                # The capture pass runs the pipeline once capture-only (no dispatch) to record the
                # kernel manifest, then batch-compiles it off-device. A weight-evicting tier
                # (dynamic_load / TT_DIT_HOST_WEIGHT_CACHE) frees the DiT weights during that pass, but
                # capture-only skips the dispatch that would reset the reload bookkeeping, so the real
                # warmup below reads an evicted weight (`parameter has no data`). Skip the capture for
                # those tiers and let the real warmup compile in-window. The default 3-stage prewarm
                # compiles off-device before the reservation, so it never reaches here (_cold is False).
                if _cold and not self.dynamic_load:
                    logger.info("cold build_key: capturing kernel manifest for off-device batch compile")
                    _kp.kernel_prewarm_set_capture_only(True)
                    # Capture at the run's OWN traced setting: a traced run's real kernels are the
                    # trace-VARIANT programs (persistent-buffer configs), which an eager capture never
                    # builds -- so under traced we run the capture denoise traced too, with the tracer
                    # skipping the device trace-capture (set_kernel_prewarm_capturing) since capture-only
                    # skips dispatch. capture_all records every kernel the real run uses; in_capture_pass
                    # skips the audio decode (its mel-VAE / BWE / vocoder lazily warm persistent device
                    # state on dispatch, which capture-only corrupts -- capturing it produces wrong
                    # audio). The audio path compiles op-by-op in the warm real warmup; the DiT/VAE bulk
                    # is what the off-device batch buys back.
                    set_kernel_prewarm_capturing(self._traced)
                    try:
                        self.warmup_buffers(
                            num_frames=num_frames,
                            height=height,
                            width=width,
                            capture_all=True,
                            in_capture_pass=False,
                            capture_traced=self._traced,
                        )
                    finally:
                        set_kernel_prewarm_capturing(False)
                        _kp.kernel_prewarm_set_capture_only(False)
                    _built = _kp.kernel_prewarm_offline_compile()
                    logger.info(f"cold build_key: batch-compiled {_built} kernels off-device in-process")
                    # The capture pass ran the pipeline under capture-only (no gcc, no dispatch), leaving
                    # in-memory state the real run must NOT reuse: binary-less programs in the program
                    # cache (Kernel::binaries() would assert empty) and per-component trace/tracer state
                    # left half-initialized because dispatch was skipped. Reset both so the real warmup
                    # rebuilds cleanly from the freshly batch-compiled on-disk binaries. Separate
                    # processes never shared this state; one process must reset it.
                    self.release_traces()
                    self.mesh_device.clear_program_cache()
                elif _cold:
                    logger.info(
                        "cold build_key + dynamic_load: skipping in-process off-device capture "
                        "(weight-evicting tier); the real warmup compiles in-window"
                    )
                # On cold-start the real warmup runs capture_all so it dispatches AND trace-captures every
                # component (denoise/VAE/vocoder) while warm -- gen then replays instead of doing a
                # first-time capture that would write device statics an iter_fast warmup never initialized
                # (vocoder prep_run=False). Warm runs keep the normal iter_fast warmup.
                self.warmup_buffers(num_frames=num_frames, height=height, width=width, capture_all=_cold)
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

    def release_audio_submesh(self) -> None:
        """Drop the pipeline's references to the audio decode submesh (LTX_AUDIO_SUBMESH).

        The submesh shares the parent mesh's command queue. ttnn forbids closing a
        cq-sharing child while the parent is alive (close hangs) and forbids closing
        the parent while the child is alive ("cq in use by child submesh"), so the
        submesh's lifetime is bound to the parent: it is reclaimed when the parent mesh
        closes at process teardown. This only frees the audio device tensors. No-op when
        audio runs on the full mesh.
        """
        if self._owned_audio_submesh is not None:
            ttnn.synchronize_device(self._owned_audio_submesh)
            # The adapter owns both mel-decoder + vocoder (exposed via the tt_mel_decoder /
            # tt_vocoder_with_bwe properties); dropping it frees the submesh-resident audio tensors.
            self._audio_adapter = None
            self.audio_ccl_manager = self.vae_ccl_manager
            self.audio_mesh_device = self.mesh_device

    @property
    def tt_mel_decoder(self):
        """Underlying mel-VAE decoder ``Module`` (or ``None``) — used by ``release_traces``,
        ``_decode_mel``, and ``decode_audio``. Backed by ``LTXAudioDecoderAdapter``."""
        return self._audio_adapter.mel_decoder if self._audio_adapter is not None else None

    @property
    def tt_vocoder_with_bwe(self):
        """Underlying vocoder+BWE ``Module`` (or ``None``) — used by ``release_traces`` and
        ``decode_audio``. Backed by ``LTXAudioDecoderAdapter``."""
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
        # LTX_ITER_DIT_RESIDENT=1 (opt-in, dev iteration): force models resident (disable dynamic
        # eviction) so the DiT is loaded once and reused across passes instead of being evicted by
        # the VAE/audio loads and reloaded (~12s each). Trades device memory for reloads; may OOM on
        # small meshes (the 2x4 BH default is dynamic_load=True precisely because memory is tight),
        # so default OFF => default behavior unchanged. Validated by md5 before being kept.
        if os.environ.get("LTX_ITER_DIT_RESIDENT", "0") in ("1", "true", "True"):
            logger.info("LTX_ITER_DIT_RESIDENT=1: forcing dynamic_load=False (models stay resident)")
            dynamic_load = False
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
        self._vae_encoder_blocks = vae_cfg.get("encoder_blocks", [])
        self._vae_causal = vae_cfg.get("causal_decoder", False)
        self._vae_base_channels = vae_cfg.get("decoder_base_channels", 128)
        self._vae_patch_size = vae_cfg.get("patch_size", 4)
        # Extra fields needed to reconstruct the reference (host) VAE encoder for I2V parity checks.
        self._vae_in_channels = vae_cfg.get("in_channels", 3)
        self._vae_latent_channels = vae_cfg.get("latent_channels", 128)
        self._vae_norm_layer = vae_cfg.get("norm_layer", "pixel_norm")
        self._vae_latent_log_var = vae_cfg.get("latent_log_var", "uniform")
        self._vae_spatial_padding_mode = vae_cfg.get("spatial_padding_mode", "zeros")
        if self._vae_decoder_blocks:
            logger.info(f"VAE config: {len(self._vae_decoder_blocks)} blocks, causal={self._vae_causal}")
        if self._vae_encoder_blocks:
            logger.info(f"VAE encoder config: {len(self._vae_encoder_blocks)} blocks")

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
        """Cache key for ``cache_module.load_model``. LoRA-tagged so fused and base
        weights don't alias in ``TT_DIT_CACHE_DIR``; quant-tagged because cached
        tensorbins carry their dtype — a bf8 quant run and the bf16 baseline must use
        separate dirs or the loader dtype-clashes on a stale-precision cache hit."""
        base = os.path.basename(checkpoint_path).removesuffix(".safetensors")
        if lora_specs:
            tag = "+".join(f"{os.path.basename(s.path).removesuffix('.safetensors')}@{s.strength}" for s in lora_specs)
            base = f"{base}.lora-{tag}"
        preset = os.environ.get("LTX_QUANT", "").strip()
        if preset:
            from .quant_config import QuantConfig

            if callable(getattr(QuantConfig, preset, None)):
                base = f"{base}.q-{preset}"
        return base

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
            image_conditioning=(
                self.SUPPORTS_IMAGE_CONDITIONING
                and bool(getattr(self, "_vae_encoder_blocks", []))
                # RUN_I2V=0 forces the fast scalar-AdaLN path (no per-token video timesteps).
                # Opting out here also disables passing images= to generate(); the transformer asserts on it.
                and os.environ.get("RUN_I2V", "1") != "0"
            ),
        )

    def _new_vae_encoder(self, num_frames: int, height: int, width: int) -> LTXVideoEncoder:
        """Image-conditioning VAE encoder (I2V): pixels -> latent. Shares the decoder's
        per-channel statistics and the VAE H/W parallel config."""
        return LTXVideoEncoder(
            encoder_blocks=self._vae_encoder_blocks,
            patch_size=self._vae_patch_size,
            mesh_device=self.mesh_device,
            parallel_config=self.vae_parallel_config,
            ccl_manager=self.vae_ccl_manager,
            num_frames=num_frames or None,
            height=height or None,
            width=width or None,
        )

    def _build_ref_vae_encoders(self, ref_pixel_frames: int, height: int, width: int) -> None:
        """Construct the IC-LoRA reference-video encoders (multi-frame, one per stage res) and wire
        their coresident exclusions. Idempotent. ``ref_pixel_frames`` is the looped-still length
        (>=121, capped to the output frame count); the reference is encoded at the SAME res as the
        target (downscale_factor=1 for Ingredients). The existing ``self.vae_encoder`` is single-frame
        (image i2v), so the reference needs its own shape-baked instances. Exclusion wiring is
        centralized in ``_register_coresident_exclusions`` (which picks the ref encoders up once they
        exist and re-wires them if the decoder/upsampler is later rebuilt)."""
        if self.vae_encoder is None or self.vae_ref_encoder_full is not None:
            return
        self.vae_ref_encoder_s1 = self._new_vae_encoder(
            num_frames=ref_pixel_frames, height=height // 2, width=width // 2
        )
        self.vae_ref_encoder_full = self._new_vae_encoder(num_frames=ref_pixel_frames, height=height, width=width)
        self._register_coresident_exclusions()

    @staticmethod
    def _sheet_digest(path: str) -> str:
        """Content digest of a reference sheet, for keying the encode memoize. A sheet is a few MB,
        so hashing it costs nothing next to the encode it saves."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _load_reference_video(
        image_path: str, height: int, width: int, ref_pixel_frames: int, crf: int = DEFAULT_IMAGE_CRF
    ) -> torch.Tensor:
        """Load an IC-LoRA reference SHEET (still image) and loop it into a static reference VIDEO of
        ``ref_pixel_frames`` at the target res — the model card's "loop the still to the output length"
        preprocessing, done as an in-memory repeat (functionally identical to decoding a looped .mp4,
        no video round-trip needed for a still). Returns ``(1, 3, ref_pixel_frames, H, W)`` in [-1, 1]."""
        frame = LTXPipeline._load_conditioning_image(image_path, height, width, crf=crf)  # (1, 3, 1, H, W)
        return frame.expand(1, 3, ref_pixel_frames, height, width).contiguous()

    def _new_upsampler(self, num_frames: int | None = None) -> LTXLatentUpsampler:
        """Spatial-2x latent upsampler at stage-1 shape: input H/W = full // (SPATIAL_COMPRESSION*2)
        (stage 1 runs at half-res, then the VAE compresses by SPATIAL_COMPRESSION),
        latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1. Config read from checkpoint header.
        ``num_frames`` overrides the pinned frame count (append-token tail-pad decodes one extra latent
        frame); default is the create-time ``_init_num_frames``."""
        with safe_open(self._upsampler_path, framework="pt") as f:
            cfg = json.loads(f.metadata()["config"])
        latent_frames = ((num_frames or self._init_num_frames) - 1) // TEMPORAL_COMPRESSION + 1
        # Round the s1 input H/W up to the mesh factors so the upsampler's pinned GroupNorm
        # and conv dims are built for even shards; upsample_latent replicate-pads runtime
        # input to match and crops the output. Even shards skip the halo crop-masking that
        # seams uneven dims (e.g. 17x30) at the 2x4 boundaries.
        hf = self.vae_parallel_config.height_parallel.factor
        wf = self.vae_parallel_config.width_parallel.factor
        input_hw = (
            ceil_to(self._init_height // (SPATIAL_COMPRESSION * 2), hf),
            ceil_to(self._init_width // (SPATIAL_COMPRESSION * 2), wf),
        )
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

    def _ensure_upsampler_frames(self, num_frames: int) -> None:
        """Rebuild the latent upsampler for a new pinned frame count. The upsampler bakes its
        GroupNorm3D shapes (latent T) at construction, so the append-token tail-pad — which decodes
        one extra latent frame so a last-frame keyframe becomes interior — needs a fresh module.
        Cheap: construction pushes no device weights (the next ``_prepare_upsampler`` loads them,
        under a frame-count-distinct blocking cache key). Dealloc the old weights first, then
        re-wire the coresident exclusions so the rebuilt module evicts / gets evicted correctly
        (idempotent dealloc makes any lingering stale refs harmless)."""
        if self.upsampler is None or getattr(self, "_upsampler_nf", None) == num_frames:
            return
        if self.upsampler.is_loaded():
            self.upsampler.deallocate_weights()
        self.upsampler = self._new_upsampler(num_frames)
        self._upsampler_nf = num_frames
        self._register_coresident_exclusions()

    def _ensure_vae_decoder_frames(self, num_frames: int) -> None:
        """Rebuild the VAE decoder for a new pinned frame count — same GroupNorm3D frame-count baking
        as the upsampler (see ``_ensure_upsampler_frames``); the tail-pad decodes one extra latent
        frame, so the decoder built for the create-time count must be replaced."""
        if self.vae_decoder is None or getattr(self, "_vae_decoder_nf", None) == num_frames:
            return
        if self.vae_decoder.is_loaded():
            self.vae_decoder.deallocate_weights()
        self.vae_decoder = LTXVideoDecoder(
            decoder_blocks=self._vae_decoder_blocks,
            causal=self._vae_causal,
            base_channels=self._vae_base_channels,
            mesh_device=self.mesh_device,
            parallel_config=self.vae_parallel_config,
            ccl_manager=self.vae_ccl_manager,
            num_frames=num_frames or None,
            height=self._init_height or None,
            width=self._init_width or None,
        )
        self._vae_decoder_nf = num_frames
        self._register_coresident_exclusions()

    def _instantiate_modules(
        self, extra_variants: list[tuple[str, list[LoraSpec]]], *, audio_only: bool = False
    ) -> None:
        """Build every TT Module the pipeline will use. No DRAM weights yet —
        ``_prime_caches`` (next) attaches them. ``audio_only`` builds just the
        audio decoder shell; the transformer/VAE/upsampler are never used by
        ``decode_audio``."""
        if audio_only:
            if self.checkpoint_name is not None:
                self._new_audio_decoder()
            return
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
            self._vae_decoder_nf = self._init_num_frames

        if self._vae_encoder_blocks and self._init_height > 0 and self._init_width > 0:
            # Single-frame image conditioning; blocking falls back to channel-keyed entries so the
            # one module can encode at both stage resolutions (conv3d adapts to the runtime shape).
            self.vae_encoder = self._new_vae_encoder(num_frames=1, height=self._init_height, width=self._init_width)

        if self.HAS_UPSAMPLER:
            assert (
                self._init_height > 0 and self._init_width > 0 and self._init_num_frames > 0
            ), f"{type(self).__name__} requires num_frames/height/width at create_pipeline."
            self._upsampler_path = LTXPipeline._resolve_checkpoint_file(LTX_UPSAMPLER_HF_REF)
            self.upsampler = self._new_upsampler()
            self._upsampler_nf = self._init_num_frames

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

        # VAE encoder runs once at the start of generate (before stage 1), then is evicted.
        # Exclude it against the DiT variants, the decoder, and the upsampler.
        if self.vae_encoder is not None:
            enc_peers = [*models]
            if self.vae_decoder is not None:
                enc_peers.append(self.vae_decoder)
            if self.upsampler is not None:
                enc_peers.append(self.upsampler)
            for m in enc_peers:
                m.register_coresident_exclusions(self.vae_encoder)
            self.vae_encoder.register_coresident_exclusions(*enc_peers)

        # IC-LoRA reference encoders (multi-frame, built lazily by _build_ref_vae_encoders) get the
        # SAME wiring as the single-frame image encoder above: evict/be-evicted against every DiT
        # variant, the decoder, the upsampler, and the image encoder; the two ref encoders (distinct
        # shapes) also exclude each other so neither co-fits with an active DiT. Inert until the ref
        # encoders exist; re-runs here (decoder/upsampler rebuilds) re-wire them against current peers.
        ref_encoders = [e for e in (self.vae_ref_encoder_s1, self.vae_ref_encoder_full) if e is not None]
        for ref_enc in ref_encoders:
            ref_peers = [*models]
            for extra in (self.vae_decoder, self.upsampler, self.vae_encoder):
                if extra is not None:
                    ref_peers.append(extra)
            ref_peers += [e for e in ref_encoders if e is not ref_enc]
            for m in ref_peers:
                m.register_coresident_exclusions(ref_enc)
            ref_enc.register_coresident_exclusions(*ref_peers)

        # The on-device Gemma encoder modules are bidirectionally excluded with the DiT
        # variants + VAE; the pair wires the exclusions on each module at its first build.
        encoder_peers = [*models] + ([self.vae_decoder] if self.vae_decoder is not None else [])
        self.gemma_encoder_pair.register_coresident_peers(encoder_peers)

        # Audio decoder/vocoder are intentionally NOT excluded against the VAE:
        # measured coresident on BH-LB 2x4 (the tightest BH config) with no OOM,
        # even with the fp32 vocoder's conv3d activations live. Excluding them only
        # forces a redundant audio reload at decode (~6s warm) for no memory gain.

    def _prime_caches(self, *, audio_only: bool = False) -> None:
        """Load every module in reverse use order so variant 0 is resident in
        DRAM after ``__init__`` (matches Wan's reverse-use-order priming).
        ``audio_only`` primes just the audio decoder — the 22B transformer push
        dominates a cold run and decode_audio never touches it."""
        if audio_only:
            self._prepare_audio_decoder()
            return
        for idx in range(len(self.transformer_states) - 1, 0, -1):
            self._prepare_transformer(idx)
        self._prepare_vae()
        self._prepare_upsampler()
        self._prepare_audio_decoder()
        self._prepare_transformer(0)

    def _maybe_apply_quant_config(self) -> None:
        """Install a DiT-linear quant preset when LTX_QUANT names one. No-op (baseline) otherwise."""
        preset = os.environ.get("LTX_QUANT", "").strip()
        if not preset:
            return
        from .quant_config import QuantConfig, set_quant_config

        factory = getattr(QuantConfig, preset, None)
        if factory is None or not callable(factory):
            logger.warning(f"LTX_QUANT='{preset}' is not a QuantConfig preset; running baseline (bf16/HiFi2)")
            return
        logger.info(f"LTX_QUANT='{preset}': applying DiT-linear quant config")
        set_quant_config(self, factory())

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
            post_load_hook=getattr(self, "_transformer_post_load_hook", None),
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

        with Watchdog("gemma text-encode"):
            results = self.gemma_encoder_pair.encode(prompts)

        if use_cache:
            torch.save(results, cache_path)
            logger.info(f"Cached device embeddings to {cache_path}")
        return results

    def _prepare_vae(self) -> None:
        """Push VAE decoder weights onto the mesh. Module was constructed in
        ``__init__``; blocking-hash subfolder forces re-load when conv3d
        ``C_in_block`` changes (mirrors Wan)."""
        if self.vae_decoder is None:
            return
        # Static load keeps the VAE resident across the audio decode — skip the per-request reload.
        if self.vae_decoder.is_loaded():
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

    def _prepare_vae_encoder(self, encoder=None) -> None:
        """Push VAE encoder weights onto the mesh (I2V image conditioning). Mirrors
        ``_prepare_vae``; shares the decoder's ``vae.per_channel_statistics.*``. ``encoder``
        overrides ``self.vae_encoder`` (the IC-LoRA reference encoders share the same weights)."""
        enc = encoder or self.vae_encoder
        if enc is None:
            return

        def _vae_encoder_state_provider() -> dict[str, torch.Tensor]:
            logger.info(f"VAE encoder cache miss — loading safetensors: {self._vae_checkpoint_path}")
            raw = load_file(self._vae_checkpoint_path)
            enc_state = {}
            for k, v in raw.items():
                if k.startswith("vae.encoder."):
                    enc_state[k.removeprefix("vae.encoder.")] = v
                elif k.startswith("vae.per_channel_statistics."):
                    short_key = k.removeprefix("vae.")
                    if short_key in ("per_channel_statistics.mean-of-means", "per_channel_statistics.std-of-means"):
                        enc_state[short_key] = v
            return enc_state

        blocking_key = conv3d_blocking_hash(enc)
        subfolder = f"vae_enc_{blocking_key}" if blocking_key else "vae_enc"
        cache_module.load_model(
            enc,
            model_name=os.path.basename(self.checkpoint_name).removesuffix(".safetensors"),
            subfolder=subfolder,
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            get_torch_state_dict=_vae_encoder_state_provider,
        )
        logger.info(f"Loaded TTNN VAE encoder ({len(self._vae_encoder_blocks)} blocks)")

    def encode_image(self, image_BCFHW: torch.Tensor, encoder=None) -> torch.Tensor:
        """Encode a conditioning image/clip ``(B, 3, F, H, W)`` in [-1, 1] to a normalized
        latent ``(B, 128, F', H', W')``. Loads the encoder (evicting coresident peers).

        ``encoder`` overrides ``self.vae_encoder`` — used for the multi-frame IC-LoRA reference
        encoders (``vae_ref_encoder_s1``/``vae_ref_encoder_full``), which share the same weights.

        When ``LTX_VAE_ENCODER_HOST=1`` (off by default), also runs the reference CPU/torch
        encoder, logs device-vs-host parity (PCC + abs diff), and returns the HOST latent for
        the run — a self-checking I2V path. Default (``0``) is the device-only fast path.
        """
        enc = encoder or self.vae_encoder
        assert enc is not None, "VAE encoder not constructed (no encoder_blocks in checkpoint?)"
        self._prepare_vae_encoder(enc)
        device_latent = enc(image_BCFHW)
        if os.environ.get("LTX_VAE_ENCODER_HOST", "0") != "1":
            return device_latent
        host_latent = self._host_encode_image(image_BCFHW)
        self._log_encoder_parity(device_latent, host_latent)
        return host_latent

    def _build_host_vae_encoder(self):
        """Reference (CPU/torch) LTX-2 VAE encoder from ``ltx_core``, built lazily from the same
        checkpoint ``vae.encoder.*`` weights + per-channel statistics the device encoder loads.
        Cached across calls. Raises a clear error if the LTX-2 reference package is unavailable."""
        if getattr(self, "_host_vae_encoder_mod", None) is not None:
            return self._host_vae_encoder_mod
        _ensure_ltx_reference_on_path()
        try:
            from ltx_core.model.video_vae.enums import LogVarianceType, NormLayerType, PaddingModeType
            from ltx_core.model.video_vae.video_vae import VideoEncoder
        except ImportError as e:
            raise ImportError(
                "LTX_VAE_ENCODER_HOST=1 needs the LTX-2 reference package (ltx_core), which was not "
                "importable. Clone it at the repo root (git clone https://github.com/Lightricks/LTX-2 "
                "into <repo>/LTX-2) or set LTX_REFERENCE_ROOT to a checkout's "
                "packages/ltx-core/src directory."
            ) from e

        encoder = VideoEncoder(
            in_channels=self._vae_in_channels,
            out_channels=self._vae_latent_channels,
            encoder_blocks=self._vae_encoder_blocks,
            patch_size=self._vae_patch_size,
            norm_layer=NormLayerType(self._vae_norm_layer),
            latent_log_var=LogVarianceType(self._vae_latent_log_var),
            encoder_spatial_padding_mode=PaddingModeType(self._vae_spatial_padding_mode),
        )
        # Pull only the encoder + per-channel-stats tensors (not the full 22B checkpoint) via safe_open.
        state: dict[str, torch.Tensor] = {}
        with safe_open(self._vae_checkpoint_path, framework="pt") as f:
            for k in f.keys():
                if k.startswith("vae.encoder."):
                    state[k.removeprefix("vae.encoder.")] = f.get_tensor(k)
                elif k in (
                    "vae.per_channel_statistics.mean-of-means",
                    "vae.per_channel_statistics.std-of-means",
                ):
                    state[k.removeprefix("vae.")] = f.get_tensor(k)
        encoder.load_state_dict(state, strict=True)
        encoder = encoder.to(torch.float32).eval()
        logger.info(f"Built host VAE encoder (ltx_core reference, {len(self._vae_encoder_blocks)} blocks)")
        self._host_vae_encoder_mod = encoder
        return encoder

    def _host_encode_image(self, image_BCFHW: torch.Tensor) -> torch.Tensor:
        """Encode the conditioning image on the host (reference encoder), matching the device
        encoder's contract: ``(B, 3, F, H, W)`` in [-1, 1] -> normalized ``(B, 128, F', H', W')``."""
        encoder = self._build_host_vae_encoder()
        with torch.no_grad():
            return encoder(image_BCFHW.detach().float())

    @staticmethod
    def _log_encoder_parity(device_latent: torch.Tensor, host_latent: torch.Tensor) -> None:
        """Log device-vs-host conditioning-latent agreement (PCC + abs/rel diff)."""
        d = device_latent.detach().float()
        h = host_latent.detach().float()
        if d.shape != h.shape:
            logger.warning(
                f"I2V VAE encoder parity: SHAPE MISMATCH device {tuple(d.shape)} vs host {tuple(h.shape)} "
                "(using HOST latent for the run)"
            )
            return
        df, hf = d.flatten(), h.flatten()
        dc, hc = df - df.mean(), hf - hf.mean()
        denom = float(dc.norm() * hc.norm())
        pcc = float(torch.dot(dc, hc)) / denom if denom > 0 else float("nan")
        max_abs = float((df - hf).abs().max())
        mean_abs = float((df - hf).abs().mean())
        rel = max_abs / (float(hf.abs().max()) + 1e-8)
        logger.info(
            f"I2V VAE encoder parity (device vs host) shape={tuple(d.shape)}: "
            f"PCC={pcc:.6f}  max|Δ|={max_abs:.6f}  mean|Δ|={mean_abs:.6f}  rel={rel:.4%}  "
            "(using HOST latent for the run)"
        )

    @staticmethod
    def _crf_codec_roundtrip(arr, crf: int):
        """Encode/decode an RGB ``(H,W,3)`` uint8 image through libx264 at the given CRF, cropped
        to even dims. Port of ``ltx_pipelines.utils.media_io`` encode/decode_single_frame."""
        import av  # lazy import (matches utils/video.py); only needed for I2V conditioning
        import numpy as np

        # libx264 requires even dimensions; crop to a multiple of 2 like the reference.
        height = arr.shape[0] // 2 * 2
        width = arr.shape[1] // 2 * 2
        arr = np.ascontiguousarray(arr[:height, :width])

        with BytesIO() as buf:
            container = av.open(buf, mode="w", format="mp4")
            try:
                stream = container.add_stream("libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"})
                stream.height = height
                stream.width = width
                av_frame = av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(format="yuv420p")
                container.mux(stream.encode(av_frame))
                container.mux(stream.encode())
            finally:
                container.close()
            video_bytes = buf.getvalue()

        with BytesIO(video_bytes) as buf:
            container = av.open(buf)
            try:
                vstream = next(s for s in container.streams if s.type == "video")
                frame = next(container.decode(vstream))
            finally:
                container.close()
        return frame.to_ndarray(format="rgb24")

    @staticmethod
    def _load_conditioning_image(
        image_path: str, height: int, width: int, crf: int = DEFAULT_IMAGE_CRF
    ) -> torch.Tensor:
        """Decode -> CRF round-trip -> resize+center-crop -> normalize to [-1,1]. Returns
        ``(1,3,1,H,W)`` float32. Port of ``load_image_and_preprocess``; ``crf=0`` skips the codec."""
        import math

        import numpy as np
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        arr = np.asarray(img)  # (H, W, 3) uint8
        if crf and crf > 0:
            arr = LTXPipeline._crf_codec_roundtrip(arr, crf)
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).float()  # (H, W, 3)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        _, _, src_h, src_w = tensor.shape
        scale = max(height / src_h, width / src_w)
        new_h = math.ceil(src_h * scale)
        new_w = math.ceil(src_w * scale)
        tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
        crop_top = (new_h - height) // 2
        crop_left = (new_w - width) // 2
        tensor = tensor[:, :, crop_top : crop_top + height, crop_left : crop_left + width]

        tensor = tensor.unsqueeze(2)  # (1, 3, 1, H, W)
        return tensor / 127.5 - 1.0

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

        with Watchdog("vae decode"):
            video = self.vae_decoder(latent_spatial, output_type=output_type)
        if output_type != "float":
            return video.numpy()
        return video

    def _vae_per_channel_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached ``(mean-of-means, std-of-means)`` reshaped for ``(B, C, F, H, W)``
        broadcast — the un_normalize/normalize bookends matching ``ltx_core.upsample_video``."""
        if getattr(self, "_pcs_cache", None) is None:
            self._pcs_cache = read_vae_per_channel_stats(self.checkpoint_name)
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

    def _warmup_ref_encode(self, ref_pixel_frames: int, height: int, width: int) -> None:
        """Build + JIT-compile the IC-LoRA reference encoders at both stage resolutions with a
        zero looped clip. No-op when no encoder is configured (non-I2V checkpoints)."""
        if self.vae_encoder is None:
            return
        self._build_ref_vae_encoders(ref_pixel_frames, height, width)
        self.encode_image(torch.zeros(1, 3, ref_pixel_frames, height // 2, width // 2), encoder=self.vae_ref_encoder_s1)
        self.encode_image(torch.zeros(1, 3, ref_pixel_frames, height, width), encoder=self.vae_ref_encoder_full)

    def _warmup_decode(self, num_frames: int, height: int, width: int) -> None:
        """Load VAE + JIT-compile decode kernels with a zero dummy latent at
        the target shape. No-op if no VAE is configured."""
        if self.vae_decoder is None:
            return
        latent_frames, latent_h, latent_w = latent_grid(num_frames, height, width)
        dummy = torch.zeros(1, latent_frames * latent_h * latent_w, self.in_channels)
        self._prepare_vae()
        self.decode_latents(dummy, latent_frames, latent_h, latent_w)

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
            # GaussianNoiser on a conditioned init: frame-0 = clean latent, noise scaled by mask * ns.
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

    def _new_audio_decoder(self) -> None:
        """Construct the audio decode stack via ``LTXAudioDecoderAdapter`` (ltx-perf refactor).

        The adapter parses the checkpoint's ``audio_vae`` / ``vocoder`` config, builds the
        ``MelDecoder`` (mel_decoder_ltx.py) + ``VocoderWithBWE``, selects the audio parallel
        config, and sets the trace flags. It is built on the audio submesh (LTX_AUDIO_SUBMESH)
        so the whole stack lives on ``audio_mesh_device`` / ``audio_ccl_manager`` — passing those
        as the adapter's ``mesh_device`` / ``vae_ccl_manager`` reproduces ltx-rt's submesh
        placement (the adapter derives its parallel config from that mesh's shape). No weights are
        loaded here — ``_prepare_audio_decoder`` handles that via the adapter's disk cache.
        """
        self._audio_adapter = LTXAudioDecoderAdapter(
            self.checkpoint_name,
            mesh_device=self.audio_mesh_device,
            vae_ccl_manager=self.audio_ccl_manager,
            dit_parallel_config=self.parallel_config,
            traced=self._traced,
        )

    def _prepare_audio_decoder(self) -> None:
        """Delegate: load audio decoder + vocoder weights (see ``LTXAudioDecoderAdapter``).

        Mirrors ``_prepare_vae``; the adapter loads both modules via the disk cache and re-injects
        the mel-VAE per-channel denormalize stats after load (non-Parameter buffers the binary
        cache does not carry).
        """
        if self._audio_adapter is not None:
            self._audio_adapter.reload_weights()

    def _warmup_audio_decode(self, audio_latent: torch.Tensor, num_frames: int, fps: float = 24.0) -> None:
        """Eager (untraced) audio decode at the real shape: compiles kernels, warms lazy device
        state, and frees back to a deterministic allocator free-list so a later traced decode
        captures cleanly. Required by the adopted ltx-perf audio trace path (the vocoder / mel
        decoder are ``@traced_function(prep_run=False)``, so they do NOT self-warm before capture —
        capture must run on already-warm state). No-op if the audio decoder is not configured."""
        if self.tt_mel_decoder is None or self.tt_vocoder_with_bwe is None:
            return
        mel_d, voc = self.tt_mel_decoder, self.tt_vocoder_with_bwe
        saved = (mel_d.use_trace, voc.use_trace, voc.use_trace_bwe)
        mel_d.use_trace = voc.use_trace = voc.use_trace_bwe = False
        try:
            self.decode_audio(audio_latent, num_frames, fps=fps)
        finally:
            mel_d.use_trace, voc.use_trace, voc.use_trace_bwe = saved

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

        # VAE and audio modules can't be L1-coresident on BH-LB. With
        # dynamic_load the audio (re)load below evicts the VAE via the
        # coresident exclusions registered in `_register_coresident_exclusions`;
        # without dynamic_load nothing evicts, so free the VAE explicitly.
        if not self.dynamic_load and self.vae_decoder is not None and self.vae_decoder.is_loaded():
            self.vae_decoder.deallocate_weights()

        assert self.tt_mel_decoder is not None and self.tt_vocoder_with_bwe is not None, (
            "audio decoder shells not built — _new_audio_decoder() must run first "
            "(it does, via _instantiate_modules, when checkpoint_name is set at construction)"
        )
        self._prepare_audio_decoder()

        # Unpatchify: (1, audio_N, 128) -> (1, z, audio_N, 128 // z)
        # (z=z_channels, 128 // z is the patchify freq).
        audio_N = audio_latent.shape[1]
        z = self.tt_mel_decoder.z_channels
        audio_spatial = audio_latent.reshape(1, audio_N, z, audio_latent.shape[2] // z).permute(0, 2, 1, 3).float()

        # Optional stage-wall split (LTX_TIME_STAGES=1): mel-VAE vs vocoder+BWE, to find
        # the dominant decode stage. Syncs are host-wall, so this is a coarse stage timer,
        # not a per-op device profile.
        _time_stages = os.environ.get("LTX_TIME_STAGES") in ("1", "true", "True")
        if _time_stages:
            import time as _t

            ttnn.synchronize_device(self.audio_mesh_device)
            _t0 = _t.perf_counter()
            mel = self._decode_mel(audio_spatial)
            ttnn.synchronize_device(self.audio_mesh_device)
            _t_vae = _t.perf_counter()
            waveform = self.tt_vocoder_with_bwe(mel).squeeze(0).float()
            ttnn.synchronize_device(self.audio_mesh_device)
            _t_voc = _t.perf_counter()
            logger.info(
                f"STAGE_SPLIT mel_vae={(_t_vae - _t0) * 1000:.1f}ms " f"vocoder+bwe={(_t_voc - _t_vae) * 1000:.1f}ms"
            )
            # The vocoder+BWE first touch is a cold build (~3 min); record both audio stages
            # so this large, otherwise-untracked cost is visible in the wall-time ledger.
            walltime.record("audio", "mel_vae", _t_vae - _t0)
            walltime.record("audio", "vocoder+bwe", _t_voc - _t_vae)
        else:
            with Watchdog("audio decode (mel-VAE + vocoder)"):
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
