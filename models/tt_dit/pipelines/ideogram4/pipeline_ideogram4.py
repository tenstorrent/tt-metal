# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Ideogram 4.0 text-to-image inference pipeline.

Faithful port of the reference ``pipeline_ideogram4.py`` orchestration, composing
the already-verified tt_dit components:
  * text encoder  — encoders.qwen3vl.Qwen3VlTextEncoder (13-layer tap)
  * denoiser      — models.transformers.transformer_ideogram4.Ideogram4Transformer
                    (a conditional and an unconditional instance — asymmetric CFG)
  * VAE decoder   — models.vae.vae_ideogram4.Ideogram4VAEDecoder
  * sampler       — pipelines.ideogram4.sampler.Ideogram4Sampler

The host-side sequence packing, the per-step CFG blend (v = gw*v_cond +
(1-gw)*v_uncond), the Euler update and the decode (per-channel latent denorm ->
2x2 unpatch -> VAE) mirror the reference exactly. The image latent is sharded on
sequence for SP and the wrapper is per-token, so the denoise loop is SP/TP-ready.
"""

from __future__ import annotations

import os

import torch

import ttnn

from ...models.vae.vae_ideogram4 import Ideogram4VAEDecoder
from ...reference.ideogram4.latent_norm import get_latent_norm
from ...utils import tensor
from ...utils.padding import get_padded_vision_seq_len
from ...utils.tensor import bf16_tensor
from ...utils.tracing import Tracer
from .sampler import Ideogram4Sampler


def unpatchify_latent(z: torch.Tensor, *, grid_h: int, grid_w: int, patch: int) -> torch.Tensor:
    """[B, grid_h*grid_w, patch*patch*ae_ch] -> [B, ae_ch, grid_h*patch, grid_w*patch] (NCHW).

    Matches the reference Ideogram4Pipeline._decode unpatch.
    """
    b = z.shape[0]
    ae_channels = z.shape[-1] // (patch * patch)
    z = z.view(b, grid_h, grid_w, patch, patch, ae_channels)
    z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
    return z.view(b, ae_channels, grid_h * patch, grid_w * patch)


def interleave_layer_taps(taps: list[torch.Tensor]) -> torch.Tensor:
    """Assemble the L Qwen3-VL per-layer taps (each ``[B, n, D]``) into ``[B, n, D*L]`` in
    FEATURE-MAJOR order: ``out[..., f*L + l] = taps[l][..., f]`` (each feature's L layer-values
    contiguous). This is the vendor Ideogram 4.0 layout the ``llm_cond_proj``/``llm_cond_norm``
    weights were trained for -- confirmed by coherent, prompt-accurate generation with the real
    fp8 checkpoint (a layer-major ``l*D + f`` order scrambles the trained projection into noise).
    Kept as a standalone helper so a host unit test can lock the order against regression.
    """
    b, n = taps[0].shape[0], taps[0].shape[1]
    return torch.stack(taps, dim=0).permute(1, 2, 3, 0).reshape(b, n, -1)


class Ideogram4DecodeStage:
    """The pipeline's decode tail: per-channel latent denorm -> 2x2 unpatch -> VAE decode.

    Split out because it is the pipeline-specific NEW device computation (the denoise
    loop reuses the verified sampler + transformer, the encode reuses the verified
    Qwen3-VL encoder). ``latent_shift``/``latent_scale`` are the per-channel
    (128 = 32 ae-channels x 2x2 patch) constants from the reference latent_norm.
    """

    def __init__(self, vae_decoder: Ideogram4VAEDecoder, *, mesh_device: ttnn.MeshDevice, patch: int = 2) -> None:
        self.vae_decoder = vae_decoder
        self.mesh_device = mesh_device
        self.patch = patch
        shift, scale = get_latent_norm()  # each [128]
        self.latent_shift = bf16_tensor(shift.view(1, 1, -1), device=mesh_device)
        self.latent_scale = bf16_tensor(scale.view(1, 1, -1), device=mesh_device)

    def decode(self, z: ttnn.Tensor, *, grid_h: int, grid_w: int) -> torch.Tensor:
        """z: [B, grid_h*grid_w, 128] denoised latent (device). Returns [B,3,H,W] in [-1,1] (torch)."""
        z = z * self.latent_scale + self.latent_shift  # per-channel denorm (device)

        z_torch = tensor.to_torch(z, mesh_axes=[None, None, None])
        z_nchw = unpatchify_latent(z_torch, grid_h=grid_h, grid_w=grid_w, patch=self.patch)

        # Feed replicated NHWC: post_quant_conv is replicated; the decoder conv_in shards
        # channels via out_mesh_axis. Works for TP=1 and TP>1. Output is replicated
        # (VAEDecoder vae_all_gathers before conv_out), so take device 0's tensor.
        tt_z = bf16_tensor(z_nchw.permute(0, 2, 3, 1), device=self.mesh_device)  # NCHW -> NHWC, replicated
        decoded = self.vae_decoder(tt_z)
        return ttnn.to_torch(ttnn.get_device_tensors(decoded)[0]).permute(0, 3, 1, 2)  # NHWC -> NCHW


def cfg_blend(v_cond: ttnn.Tensor, v_uncond: ttnn.Tensor, guidance_weight) -> ttnn.Tensor:
    """Asymmetric-CFG velocity blend v = gw*v_cond + (1-gw)*v_uncond, via ttnn.lerp
    (matches pipelines/cfg.py and Wan2.2). guidance_weight is a float or 1-element tensor."""
    return ttnn.lerp(v_uncond, v_cond, guidance_weight)


__all__ = ["Ideogram4DecodeStage", "Ideogram4Sampler", "cfg_blend", "interleave_layer_taps", "unpatchify_latent"]


# =============================================================================
# Full pipeline class. Follows the tt_dit pipeline mold (PipelineAPIMixin +
# Ideogram4PipelineConfig + create_pipeline), like the Flux1/QwenImage pipelines.
#
# CFG is run SEQUENTIALLY on the full submesh (cfg-parallel factor 1): the
# conditional pass (text+image) and unconditional pass (image-only) are distinct
# distilled nets, so giving each pass the whole mesh beats splitting it. Everything
# is device-resident (encoder + cond + uncond transformers + VAE); no dynamic offload.
# =============================================================================

import gc as _gc
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import transformers as _tf
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL as _AutoencoderKL
from huggingface_hub import snapshot_download
from loguru import logger as _dq_log
from PIL import Image
from safetensors.torch import load_file as _load_file
from safetensors.torch import save_file as _save_file

from ...encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder, create_rope_tensors
from ...models.transformers.transformer_ideogram4 import Ideogram4Transformer, rope_halfsplit_to_interleaved
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, VAEParallelConfig
from ...parallel.manager import CCLManager
from ...reference.ideogram4 import modeling_ideogram4
from ...reference.ideogram4.constants import IMAGE_POSITION_OFFSET, QWEN3_VL_ACTIVATION_LAYERS
from ...reference.ideogram4.dequant import dequant_fp8_state_dict
from ..cfg import create_submeshes
from ..events import DenoiseStep, PipelineEventCallback, SectionEnd, SectionStart, null_callback
from ..pipeline_api import PipelineAPIMixin

# Opt-out switch for the bf16 dequant cache (default on). Dequantizing the fp8
# checkpoints (encoder + 2 transformers) dominates pipeline build time; caching the
# bf16 result makes subsequent builds skip both the fp8 load and the dequant. Set
# IDEOGRAM4_DEQUANT_CACHE=0 to force a fresh dequant.
_DEQUANT_CACHE = os.environ.get("IDEOGRAM4_DEQUANT_CACHE", "1") != "0"


def _dequant_cache_path(path: str) -> str:
    """Where the bf16 copy of ``path`` is cached: under TT_DIT_CACHE_DIR (default ~/.cache/tt-dit),
    never beside the source, which may be a shared or read-only HuggingFace cache. Keyed by the
    source's resolved path, so files with the same name in different checkpoints do not collide."""
    root = os.environ.get("TT_DIT_CACHE_DIR") or os.path.expanduser("~/.cache/tt-dit")
    source = os.path.realpath(path)
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    return os.path.join(root, "ideogram4", "dequant", f"{digest}-{os.path.basename(source)}")


def _dq(path):
    """Load + dequantize an fp8 checkpoint to bf16, caching the bf16 result on disk.

    The cache file (see ``_dequant_cache_path``) is validated against the source's size+mtime
    (stamped into the safetensors metadata), so it is transparently rebuilt if the source
    weights change. Cache write failures fall back to in-memory dequant without erroring.
    """
    if not _DEQUANT_CACHE:
        return dequant_fp8_state_dict(_load_file(path))

    cache_path = _dequant_cache_path(path)
    try:
        st = os.stat(path)
        stamp = f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return dequant_fp8_state_dict(_load_file(path))

    if os.path.exists(cache_path):
        try:
            with __import__("safetensors").safe_open(cache_path, framework="pt") as f:
                meta = f.metadata() or {}
            if meta.get("src_stamp") == stamp:
                _dq_log.info(f"dequant cache HIT: {cache_path}")
                return _load_file(cache_path)
            _dq_log.info(f"dequant cache STALE (src changed), rebuilding: {cache_path}")
        except Exception as e:  # noqa: BLE001 - corrupt/old cache -> rebuild
            _dq_log.warning(f"dequant cache unreadable ({e}); rebuilding: {cache_path}")

    sd = dequant_fp8_state_dict(_load_file(path))
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = cache_path + ".tmp"
        _save_file(sd, tmp, metadata={"src_stamp": stamp})
        os.replace(tmp, cache_path)
        _dq_log.info(f"dequant cache WROTE: {cache_path}")
    except Exception as e:  # noqa: BLE001 - non-fatal: keep going with in-memory sd
        _dq_log.warning(f"dequant cache write failed ({e}); using in-memory dequant")
    return sd


# Max prompt length. Mirrors the reference Ideogram4PipelineConfig.max_text_tokens (2048):
# the reference rejects (does NOT truncate) prompts longer than this, so we do the same.
MAX_TEXT_TOKENS = 2048

# The encoder readback slices the text rows rounded up to a multiple of this, which bounds the
# number of slice programs to MAX_TEXT_TOKENS / _TEXT_READBACK_BUCKET.
_TEXT_READBACK_BUCKET = 256

# The four 2K serving sizes, as (width, height): the resolutions a BH Galaxy pipeline warms by default.
SERVING_SIZES: tuple[tuple[int, int], ...] = ((2048, 2048), (2048, 1152), (1152, 2048), (1360, 2048))

# SP shards the sequence across the ring; each per-device shard must be tile-aligned, i.e. the
# padded length is a multiple of tile_size * sp_factor. This is exactly the shared tt_dit helper
# get_padded_vision_seq_len(N, num_devices) (ring SDPA pads+masks any partial tail internally).

_DEFAULT_WEIGHTS_DIR = os.environ.get("IDEOGRAM4_WEIGHTS")
_DEFAULT_QWEN_REPO = "Qwen/Qwen3-VL-8B-Instruct"
_WEIGHTS_SUBFOLDERS = ("text_encoder", "tokenizer", "transformer", "unconditional_transformer", "vae")


def _resolve_weights_dir(weights: str) -> str:
    """A local directory as is, or a HuggingFace repo id snapshot-downloaded (cached)."""
    if os.path.isdir(weights):
        return weights
    _dq_log.info(f"Resolving HuggingFace repo {weights} (auto-download if missing)")
    return snapshot_download(repo_id=weights, allow_patterns=[f"{sub}/*" for sub in _WEIGHTS_SUBFOLDERS])


# Per-mesh-shape parallelism presets (Blackhole). Ideogram is always cfg-parallel factor 1
# (both distilled nets resident on the full mesh, sequential CFG); tp/sp are (factor, mesh_axis).
_PRESETS: dict[tuple[int, ...], dict] = {
    # BH loudbox 2x4 (mesh-shape convention shared with the other BH pipelines): SP4 (axis 1) x
    # TP2 (axis 0). SP takes the size-4 axis since Ideogram is SP-heavy (full-mesh sequential CFG).
    # No default resolutions: call warmup(sizes) or run untraced.
    (2, 4): {"tp": (2, 0), "sp": (4, 1), "num_links": 2, "topology": ttnn.Topology.Linear},
    # BH Galaxy 4x8, 2D torus Ring: TP4 (axis 0) x SP8 (axis 1). The pipeline warms the four 2K
    # serving sizes at load. Their four kept step traces allocate 10.9 MB in each of the 8 DRAM banks
    # (87 MB), which filled 87% of a 100 MB region (12.5 MB per bank; jobs 179862, 180015).
    # trace_region_size is what the mesh must be opened with for them: 128 MB (16 MB per bank, 68% used).
    (4, 8): {
        "tp": (4, 0),
        "sp": (8, 1),
        "num_links": 2,
        "topology": ttnn.Topology.Ring,
        "resolutions": SERVING_SIZES,
        "trace_region_size": 128_000_000,
    },
}


def _preset(mesh_shape) -> dict:
    preset = _PRESETS.get(tuple(mesh_shape))
    if preset is None:
        raise ValueError(f"no Ideogram4 parallelism preset for mesh shape {tuple(mesh_shape)}; known: {list(_PRESETS)}")
    return preset


def required_trace_region_size(mesh_shape) -> int:
    """The ``trace_region_size`` (bytes) to open a mesh of this shape with, so that the step traces of
    its default resolutions fit; 0 when it has none."""
    return _preset(mesh_shape).get("trace_region_size", 0)


def _check_trace_region(mesh_device: ttnn.MeshDevice, mesh_shape, sizes: Sequence[tuple[int, int]]) -> None:
    """Raise before any device work if warming ``sizes`` would overflow the mesh's trace region.

    Only the mesh's default resolutions have a measured requirement, so the check applies when
    ``sizes`` includes one of them; other sizes are left to ttnn's own check at capture. A trace
    region of 0 (ttnn's dynamic trace allocation in DRAM) is not checked.
    """
    defaults = _preset(mesh_shape).get("resolutions", ())
    if not any(tuple(size) in defaults for size in sizes):
        return
    need = required_trace_region_size(mesh_shape)
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.TRACE)
    # Each bank reserves trace_region_size / num_banks, rounded up to whole trace pages.
    reserved = view.total_bytes_per_bank * view.num_banks
    if 0 < reserved < need:
        msg = (
            f"Ideogram4: the mesh reserves {reserved / 1e6:.0f} MB of trace region, but warming {list(sizes)} "
            f"needs {need / 1e6:.0f} MB; open the mesh with trace_region_size >= {need} "
            f"(required_trace_region_size({tuple(mesh_shape)}))"
        )
        raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class Ideogram4PipelineConfig:
    """Ideogram 4.0 pipeline configuration, mirroring the Flux1/QwenImage config dataclasses."""

    topology: ttnn.Topology
    num_links: int
    dit_parallel_config: DiTParallelConfig
    encoder_parallel_config: EncoderParallelConfig
    vae_parallel_config: VAEParallelConfig
    height: int
    width: int
    resolutions: tuple[tuple[int, int], ...]
    weights_dir: str
    qwen_repo: str

    @classmethod
    def default(
        cls,
        *,
        mesh_shape: ttnn.MeshShape,
        topology: ttnn.Topology | None = None,
        num_links: int | None = None,
        height: int | None = None,
        width: int | None = None,
        resolutions: Sequence[tuple[int, int]] | None = None,
        weights_dir: str = _DEFAULT_WEIGHTS_DIR,
        qwen_repo: str = _DEFAULT_QWEN_REPO,
    ) -> Ideogram4PipelineConfig:
        """The preset for ``mesh_shape``. ``resolutions`` are the (width, height) sizes the pipeline
        warms at load; by default the preset's (the four 2K ``SERVING_SIZES`` on 4x8, none on 2x4).
        ``height`` and ``width`` are the size of a call that names none: by default the first
        resolution, as in Wan, or 512x512 when there are no resolutions."""
        preset = _preset(mesh_shape)
        if resolutions is None:
            resolutions = preset.get("resolutions", ())
        if height is None and width is None and resolutions:
            width, height = resolutions[0]
        height = 512 if height is None else height
        width = 512 if width is None else width
        tp, sp = preset["tp"], preset["sp"]
        return cls(
            topology=topology if topology is not None else preset["topology"],
            num_links=num_links if num_links is not None else preset["num_links"],
            # cfg-parallel factor 1: never split cond/uncond across submeshes (asymmetric nets).
            dit_parallel_config=DiTParallelConfig.from_tuples(cfg=(1, 0), sp=sp, tp=tp),
            encoder_parallel_config=EncoderParallelConfig.from_tuple(tp),
            vae_parallel_config=VAEParallelConfig.from_tuple(tp),
            height=height,
            width=width,
            resolutions=tuple((int(w), int(h)) for w, h in resolutions),
            weights_dir=weights_dir,
            qwen_repo=qwen_repo,
        )


@dataclass(kw_only=True)
class _SizeState:
    """Device state of one warmed (width, height), allocated before any trace capture."""

    grid_h: int
    grid_w: int
    num_img: int
    padded_len: int
    z: ttnn.Tensor  # [1, num_img, in_ch] replicated latent, updated in place by every step
    cos: ttnn.Tensor  # [1, 1, L, head_dim] SP-sharded, prompt-independent
    sin: ttnn.Tensor
    img_mask: ttnn.Tensor  # [1, L, 1] SP-sharded
    idx: ttnn.Tensor  # [1, L] uint32 SP-sharded image indicator
    uncond_logical_n: ttnn.Tensor
    tracer: Tracer | None = None


@dataclass(kw_only=True)
class _SeqBuffers:
    """Per-request conditioning of one padded length, rewritten in place by every request.
    Sizes with the same padded length share it."""

    llm: ttnn.Tensor  # [1, L, llm_features_dim] SP-sharded
    cond_llm_mask: ttnn.Tensor  # [1, L, 1] SP-sharded
    uncond_llm_mask: ttnn.Tensor  # all zeros


class Ideogram4Pipeline(PipelineAPIMixin):
    """Ideogram 4.0 text-to-image pipeline (device-resident, sequential asymmetric CFG).

    Fits the tt_dit pipeline mold (PipelineAPIMixin, Ideogram4PipelineConfig, create_pipeline,
    event callbacks, PIL output). Ideogram's cond/uncond are distinct distilled nets, so — unlike
    the cfg-parallel Flux/QwenImage pipelines — both stay resident on the full mesh and run
    sequentially, and CFG is blended on-device (cfg_blend) rather than across submeshes.
    """

    def __init__(self, *, device: ttnn.MeshDevice, config: Ideogram4PipelineConfig, run_warmup: bool = True) -> None:
        if run_warmup and config.resolutions:
            # Before loading the weights, so a server with too small a trace region fails fast.
            _check_trace_region(device, device.shape, config.resolutions)
        self._config = config
        dit_pc = config.dit_parallel_config
        # Single full-mesh submesh (cfg-parallel factor 1): both the conditional and
        # unconditional distilled nets are resident on the whole mesh and run sequentially,
        # so cfg is never split across submeshes. create_submeshes sizes the submesh to
        # sp_factor x tp_factor, which for cfg=1 is the whole device.
        (self.mesh_device,) = create_submeshes(device, dit_pc)
        mesh_device = self.mesh_device
        tp_axis = dit_pc.tensor_parallel.mesh_axis
        tp_factor = dit_pc.tensor_parallel.factor
        self.tp_axis = tp_axis
        self.tp_factor = tp_factor
        # SP shards the sequence (ring/all-gather attention) across the non-TP axis; it is the
        # hi-res win (4k/16k tok) where activations dominate. Derived from the mesh-shape preset.
        self.sp_axis = dit_pc.sequence_parallel.mesh_axis
        self.sp_factor = dit_pc.sequence_parallel.factor
        # The padding rows of the constant-length sequence are masked only by ring SDPA's logical_n.
        assert self.sp_factor > 1, "Ideogram4Pipeline requires sequence parallelism (sp_factor > 1)"
        self.topology = config.topology
        if not config.weights_dir:
            raise ValueError(
                "Ideogram4 weights not set: pass weights_dir=... to create_pipeline or set the "
                "IDEOGRAM4_WEIGHTS environment variable to the fp8 checkpoint dir or HuggingFace repo id."
            )
        weights_dir = _resolve_weights_dir(config.weights_dir)
        self.weights_dir = weights_dir
        qwen_repo = config.qwen_repo
        self.model_config = modeling_ideogram4.Ideogram4Config()
        self.patch, self.ae = 2, 8
        cfg = self.model_config

        # The encoder and VAE run eagerly between step-trace replays, so they get their own CCL
        # manager: the traced transformers' semaphores and ping-pong buffers are never touched
        # outside their traces, and stay the same at every replay.
        ccl = CCLManager(mesh_device, num_links=config.num_links, topology=config.topology)
        self.ccl = ccl
        self.dit_ccl = CCLManager(mesh_device, num_links=config.num_links, topology=config.topology)
        self._image_processor = VaeImageProcessor(vae_scale_factor=self.patch * self.ae)
        from ...utils.padding import PaddingConfig

        padding_config = (
            PaddingConfig.from_tensor_parallel_factor(cfg.num_heads, cfg.emb_dim // cfg.num_heads, tp_factor)
            if cfg.num_heads % tp_factor != 0
            else None
        )

        # --- text encoder (TP=4) ---
        # Config only — the text-decoder fields live under Qwen3VLConfig.text_config. The real
        # encoder weights come from the Ideogram fp8 checkpoint below, so there is no need to
        # instantiate the 8B HF model just to read its config (which also random-inits the
        # language_model weights and spams a "newly initialized" warning).
        qcfg = _tf.AutoConfig.from_pretrained(qwen_repo).text_config
        enc_sd = _dq(f"{weights_dir}/text_encoder/model.safetensors")
        enc_sd = {k[len("language_model.") :]: v for k, v in enc_sd.items() if k.startswith("language_model.")}
        self.tokenizer = _tf.AutoTokenizer.from_pretrained(weights_dir, subfolder="tokenizer")
        self._enc_head_dim = qcfg.hidden_size // qcfg.num_attention_heads
        self._enc_mrope = qcfg.rope_scaling["mrope_section"]
        # transformers >=4.57 moved rope_theta to the top-level config; older versions
        # kept it inside rope_scaling. Accept both.
        self._enc_rope_theta = qcfg.rope_scaling.get("rope_theta") or qcfg.rope_theta
        self.encoder = Qwen3VlTextEncoder(
            vocab_size=qcfg.vocab_size,
            hidden_size=qcfg.hidden_size,
            intermediate_size=qcfg.intermediate_size,
            hidden_act="silu",
            num_hidden_layers=qcfg.num_hidden_layers,
            num_attention_heads=qcfg.num_attention_heads,
            num_key_value_heads=qcfg.num_key_value_heads,
            rms_norm_eps=qcfg.rms_norm_eps,
            rope_theta=self._enc_rope_theta,
            mrope_section=self._enc_mrope,
            activation_layers=QWEN3_VL_ACTIVATION_LAYERS,
            device=mesh_device,
            parallel_config=self._config.encoder_parallel_config,
            ccl_manager=ccl,
            # FSDP-shard encoder weights across the SP (non-TP) axis instead of replicating
            # them. The encoder runs once (outside the denoise loop), so the per-layer weight
            # all-gather overhead is negligible, while it frees ~(sp_factor-1)/sp_factor of the
            # encoder's resident DRAM during denoise (needed to fit 2048px at SP4xTP2).
            # Auto-disables when the non-TP axis is size 1 (e.g. the TP=4 (1,4) submesh).
            is_fsdp=True,
        )
        self.encoder.load_torch_state_dict(enc_sd)
        del enc_sd
        # The encoder always runs at MAX_TEXT_TOKENS, so its RoPE tables are a constant.
        enc_cos, enc_sin = create_rope_tensors(
            1, MAX_TEXT_TOKENS, None, self._enc_head_dim, self._enc_rope_theta, self._enc_mrope
        )
        self._enc_rope = (bf16_tensor(enc_cos, device=mesh_device), bf16_tensor(enc_sin, device=mesh_device))
        _gc.collect()

        # --- conditional + unconditional denoisers (TP=4, both resident) ---
        def _build_dit(sub):
            sd = _dq(f"{weights_dir}/{sub}/diffusion_pytorch_model.safetensors")
            m = Ideogram4Transformer(
                emb_dim=cfg.emb_dim,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                intermediate_size=cfg.intermediate_size,
                adaln_dim=cfg.adanln_dim,
                in_channels=cfg.in_channels,
                llm_features_dim=cfg.llm_features_dim,
                norm_eps=cfg.norm_eps,
                mesh_device=mesh_device,
                ccl_manager=self.dit_ccl,
                parallel_config=dit_pc,
                padding_config=padding_config,
            )
            m.load_torch_state_dict(sd)
            del sd
            _gc.collect()
            return m

        self.cond = _build_dit("transformer")
        self.uncond = _build_dit("unconditional_transformer")

        # Serving state, filled by warmup(): per-size constants and step trace (keyed by
        # (width, height)), and the per-request buffers of each padded length. The conditional
        # logical_n is written by every request; allocated here, before any capture.
        self._sizes: dict[tuple[int, int], _SizeState] = {}
        self._seq_buffers: dict[int, _SeqBuffers] = {}
        self._cond_logical_n = self._logical_n(0, on_host=False)
        self.num_trace_captures = 0

        # --- VAE decoder (TP=4) ---
        vae_sd = _load_file(f"{weights_dir}/vae/diffusion_pytorch_model.safetensors")
        akl = _AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=cfg.in_channels // (self.patch * self.patch),
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            block_out_channels=(128, 256, 512, 512),
            layers_per_block=2,
            norm_num_groups=32,
        )
        # strict=False so the checkpoint-only bn.* keys don't error; but a real key-map drift that
        # leaves akl params at random init must NOT pass silently -> assert nothing else is missing.
        incompat = akl.load_state_dict({k: v for k, v in vae_sd.items() if not k.startswith("bn.")}, strict=False)
        leftover_missing = [k for k in incompat.missing_keys if not k.startswith("bn.")]
        assert not leftover_missing, f"VAE weights not loaded (key-map drift?): {leftover_missing[:8]}"
        akl = akl.to(torch.bfloat16).eval()
        vae = Ideogram4VAEDecoder.from_torch(
            akl,
            mesh_device=mesh_device,
            parallel_config=self._config.vae_parallel_config,
            ccl_manager=ccl,
        )
        self.decode_stage = Ideogram4DecodeStage(vae, mesh_device=mesh_device, patch=self.patch)
        del akl, vae_sd
        _gc.collect()

        if run_warmup and config.resolutions:
            self.warmup(sizes=config.resolutions)

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        height: int | None = None,
        width: int | None = None,
        weights_dir: str = _DEFAULT_WEIGHTS_DIR,
        qwen_repo: str = _DEFAULT_QWEN_REPO,
        topology: ttnn.Topology | None = None,
        num_links: int | None = None,
        resolutions: Sequence[tuple[int, int]] | None = None,
        run_warmup: bool = True,
    ) -> Ideogram4Pipeline:
        """Build the pipeline and, by default, warm it up to serve.

        ``resolutions`` ((width, height) pairs; default the mesh preset's) are warmed at load with
        ``warmup``, which compiles every program of each size and captures and keeps its step trace,
        so no request at those sizes compiles or captures after this returns, and a request at any
        other size raises. On a BH Galaxy (4x8) the default is the four 2K ``SERVING_SIZES``, which
        need a mesh opened with ``trace_region_size`` >= ``required_trace_region_size(mesh_shape)``
        (128 MB); other meshes have none. A call that names no size runs at ``height`` x ``width``,
        by default the first resolution (512x512 without resolutions). ``run_warmup=False`` skips the warmup: then call ``warmup``
        yourself before traced requests, or run untraced at any size.
        """
        config = Ideogram4PipelineConfig.default(
            mesh_shape=mesh_device.shape,
            height=height,
            width=width,
            resolutions=resolutions,
            weights_dir=weights_dir,
            qwen_repo=qwen_repo,
            topology=topology,
            num_links=num_links,
        )
        return cls(device=mesh_device, config=config, run_warmup=run_warmup)

    @classmethod
    def from_pretrained(
        cls,
        mesh_device,
        weights_dir=_DEFAULT_WEIGHTS_DIR,
        *,
        tp_axis: int | None = None,  # accepted for back-compat; the mesh-shape preset is authoritative
        num_links: int | None = None,
        topology: ttnn.Topology | None = None,
        **kw,
    ):
        """Back-compat factory. Prefer create_pipeline; the tp_axis/num_links/topology now come
        from the mesh-shape preset (tp_axis is ignored, preset is authoritative). Like
        create_pipeline, it warms the configured resolutions unless ``run_warmup=False``."""
        return cls.create_pipeline(
            mesh_device=mesh_device, weights_dir=weights_dir, num_links=num_links, topology=topology, **kw
        )

    def _tokenize(self, prompt: str):
        """Apply the chat template + tokenize -> [1, n_text] input ids. Single source of truth
        for token counting so ``count_text_tokens`` (pre-flight validation) and ``_encode``
        (the MAX_TEXT_TOKENS guard) always agree."""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]

    def count_text_tokens(self, prompt: str) -> int:
        """Number of text tokens the prompt produces, for pre-flight validation BEFORE a job is
        enqueued (so a server can reject an over-length prompt with an error code instead of
        letting _encode raise mid-generation). Same tokenization _encode uses."""
        return int(self._tokenize(prompt).shape[1])

    def _encode(self, prompt: str, *, warm_readback: bool = False):
        """Tokenize + run the device encoder -> real interleaved llm_features (host) + n_text.

        CONSTANT-SHAPE: the token ids are padded to exactly ``MAX_TEXT_TOKENS`` and the
        encoder rope is built for that fixed length, so the encoder always sees the same
        shape regardless of prompt length (no per-length recompile / shape-keyed POBs).
        The Qwen3-VL encoder is CAUSAL (model_qwen3vl: is_causal=(attention_bias is None),
        and we pass attention_mask=None), so the real rows [0:n_text] are unaffected by the
        trailing pad. We slice the taps back to [:, :n_text] and return the real-length feats.

        ``warm_readback`` compiles the readback slice of every row bucket (see below).
        """
        ids = self._tokenize(prompt)
        n_text = ids.shape[1]
        if n_text > MAX_TEXT_TOKENS:
            raise ValueError(
                f"prompt tokenizes to {n_text} text tokens; Ideogram 4 supports at most "
                f"{MAX_TEXT_TOKENS}. Matches the reference pipeline, which raises rather than truncating."
            )
        # Pad ids to the FIXED MAX_TEXT_TOKENS length (causal -> trailing pad is inert).
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        if n_text < MAX_TEXT_TOKENS:
            ids = torch.nn.functional.pad(ids, (0, MAX_TEXT_TOKENS - n_text), value=int(pad_id))
        tt_ids = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device)
        taps = self.encoder.forward(tt_ids, attention_mask=None, pos_embeds=self._enc_rope)
        ttnn.deallocate(tt_ids)
        # taps: 13 x [1, MAX_TEXT_TOKENS, 4096], REPLICATED across the whole mesh.
        # Read back cheaply: slice to the real text rows ON DEVICE (causal => pad rows never
        # influence the real rows, and the model masks them anyway), then read a SINGLE replica
        # via get_device_tensors(t)[0]. Reading every replica pulls all N device copies to host
        # (~1.4 s/chip => ~45 s on a 32-chip galaxy). The slice is rounded up to a whole bucket of
        # rows so that only MAX_TEXT_TOKENS / _TEXT_READBACK_BUCKET slice programs exist, all
        # compiled by warmup, instead of one per prompt length; the host drops the extra rows.
        # Interleave to [1, n_text, 53248] (element = f*13 + l) on host.
        rows = min(MAX_TEXT_TOKENS, -(-n_text // _TEXT_READBACK_BUCKET) * _TEXT_READBACK_BUCKET)
        taps_t = []
        for t in taps:
            part = ttnn.slice(t, [0, 0, 0], [1, rows, t.shape[-1]])
            taps_t.append(ttnn.to_torch(ttnn.get_device_tensors(part)[0])[:, :n_text])
            ttnn.deallocate(part)
        if warm_readback:
            for bucket in range(_TEXT_READBACK_BUCKET, MAX_TEXT_TOKENS + 1, _TEXT_READBACK_BUCKET):
                ttnn.deallocate(ttnn.slice(taps[0], [0, 0, 0], [1, bucket, taps[0].shape[-1]]))
        for t in taps:
            ttnn.deallocate(t)
        return interleave_layer_taps(taps_t).to(torch.bfloat16), n_text

    def _seq_host(self, t: torch.Tensor, seq_dim: int) -> ttnn.Tensor:
        """bf16 host tensor sharded on the SP axis along ``seq_dim``, for an in-place device write."""
        return bf16_tensor(t, device=self.mesh_device, mesh_axis=self.sp_axis, shard_dim=seq_dim, on_host=True)

    def _seq_dev(self, t: torch.Tensor, seq_dim: int) -> ttnn.Tensor:
        """bf16 device tensor sharded on the SP axis along ``seq_dim``."""
        return bf16_tensor(t, device=self.mesh_device, mesh_axis=self.sp_axis, shard_dim=seq_dim)

    def _idx_dev(self, t: torch.Tensor, seq_dim: int) -> ttnn.Tensor:
        """uint32 device tensor (embedding index) sharded on the SP axis."""
        mesh_axes = [self.sp_axis if d == seq_dim else None for d in range(t.ndim)]
        return tensor.from_torch(t, device=self.mesh_device, dtype=ttnn.uint32, mesh_axes=mesh_axes)

    def _logical_n(self, n: int, *, on_host: bool) -> ttnn.Tensor:
        """Ring-SDPA ``logical_n`` as a [1,1,1,1] uint32 ROW_MAJOR replicated tensor. The kernels read
        it on device each dispatch, so they compile once per size for any prompt length, and a trace
        reads whatever value was last written into the buffer."""
        return tensor.from_torch(
            torch.tensor([n], dtype=torch.int64).reshape(1, 1, 1, 1),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_axes=[None, None, None, None],
            on_host=on_host,
        )

    def _host_repl(self, t: torch.Tensor) -> ttnn.Tensor:
        """bf16 HOST tensor replicated across the mesh. A tracer copies it into its captured slot."""
        return tensor.from_torch(
            t.to(torch.bfloat16),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_axes=[None] * t.ndim,
            on_host=True,
        )

    def _step_inputs(self, z: ttnn.Tensor, t_val: float, gw: float, smt: float) -> tuple:
        """The per-step arguments of a size's step function: the resident latent (updated in place,
        so its address never changes and the tracer skips the copy), then t_sin and the two scalars
        guidance weight and step size ``s - t`` as host tensors."""
        t_sin = Ideogram4Transformer.sinusoidal_embedding(torch.tensor([t_val]), self.model_config.emb_dim)
        return (
            z,
            self._host_repl(t_sin.unsqueeze(1)),
            self._host_repl(torch.tensor([[[float(gw)]]])),
            self._host_repl(torch.tensor([[[float(smt)]]])),
        )

    # ------------------------------------------------------------------------------------------
    # Per-size state. A request's sequence has the constant layout [image | text | pad] of length
    # L = get_padded_vision_seq_len(num_img + MAX_TEXT_TOKENS, sp_factor), the same for any prompt.
    # Everything that depends only on the size (positions -> cos/sin, image mask, embedding index,
    # the unconditional logical_n, the latent buffer) is built once per size; the prompt only
    # changes the text features, the text mask and the conditional logical_n, which every request
    # writes in place into buffers allocated before any trace capture.
    # ------------------------------------------------------------------------------------------

    def _build_size_state(self, width: int, height: int) -> _SizeState:
        cfg = self.model_config
        grid_h, grid_w = height // (self.patch * self.ae), width // (self.patch * self.ae)
        num_img = grid_h * grid_w
        L = get_padded_vision_seq_len(num_img + MAX_TEXT_TOKENS, self.sp_factor)

        pos = torch.zeros(1, L, 3, dtype=torch.long)
        hh = torch.arange(grid_h).repeat_interleave(grid_w)
        ww = torch.arange(grid_w).repeat(grid_h)
        pos[:, :num_img, 0] = IMAGE_POSITION_OFFSET
        pos[:, :num_img, 1] = IMAGE_POSITION_OFFSET + hh
        pos[:, :num_img, 2] = IMAGE_POSITION_OFFSET + ww
        # Every row after the image takes text position 0, 1, 2, ... whether it holds text or
        # padding. The real text rows keep the positions the reference gives them, and the padding
        # rows, which logical_n masks out of attention and whose outputs are never read, no longer
        # depend on the prompt length, so cos/sin are a per-size constant.
        pos[:, num_img:] = torch.arange(L - num_img).unsqueeze(-1).expand(-1, 3)

        head_dim = cfg.emb_dim // cfg.num_heads
        rope = modeling_ideogram4.Ideogram4MRoPE(
            head_dim=head_dim, base=cfg.rope_theta, mrope_section=cfg.mrope_section
        )
        cos, sin = rope(pos)
        # Denoiser uses interleaved RoPE (rotary_embedding_llama); permute cos/sin to match.
        cos4, sin4 = rope_halfsplit_to_interleaved(
            cos.unsqueeze(1).to(torch.bfloat16), sin.unsqueeze(1).to(torch.bfloat16), head_dim
        )  # [1, 1, L, head_dim]
        is_img = torch.arange(L).unsqueeze(0) < num_img  # [1, L]
        return _SizeState(
            grid_h=grid_h,
            grid_w=grid_w,
            num_img=num_img,
            padded_len=L,
            z=tensor.from_torch(
                torch.zeros(1, num_img, cfg.in_channels), device=self.mesh_device, mesh_axes=[None, None, None]
            ),
            cos=self._seq_dev(cos4, 2),
            sin=self._seq_dev(sin4, 2),
            img_mask=self._seq_dev(is_img.float().unsqueeze(-1), 1),
            idx=self._idx_dev(is_img.to(torch.int32), 1),
            # The unconditional sequence is the image alone.
            uncond_logical_n=self._logical_n(num_img, on_host=False),
        )

    def _build_seq_buffers(self, L: int) -> _SeqBuffers:
        cfg = self.model_config
        return _SeqBuffers(
            llm=self._seq_dev(torch.zeros(1, L, cfg.llm_features_dim, dtype=torch.bfloat16), 1),
            cond_llm_mask=self._seq_dev(torch.zeros(1, L, 1), 1),
            uncond_llm_mask=self._seq_dev(torch.zeros(1, L, 1), 1),
        )

    def _write_prompt(self, st: _SizeState, sb: _SeqBuffers, llm_real: torch.Tensor, n_text: int) -> None:
        """Write one prompt's conditioning into the buffers the step trace reads, in place."""
        L, num_img = st.padded_len, st.num_img
        llm = torch.zeros(1, L, self.model_config.llm_features_dim, dtype=torch.bfloat16)
        llm[:, num_img : num_img + n_text] = llm_real
        llm_mask = torch.zeros(1, L, 1)
        llm_mask[:, num_img : num_img + n_text] = 1
        ttnn.copy_host_to_device_tensor(self._seq_host(llm, 1), sb.llm)
        ttnn.copy_host_to_device_tensor(self._seq_host(llm_mask, 1), sb.cond_llm_mask)
        ttnn.copy_host_to_device_tensor(self._logical_n(num_img + n_text, on_host=True), self._cond_logical_n)

    def _velocity(self, model, st: _SizeState, sb: _SeqBuffers, x_sp, t_sin, llm_mask, logical_n) -> ttnn.Tensor:
        """Transformer forward + SP-axis gather + image-token slice -> [1, num_img, in_ch] replicated."""
        out = model(
            x=x_sp,
            # The unconditional branch reads the conditional features too; its all-zero text mask
            # zeroes them before they are used (llm_cond_norm(llm_features * llm_token_mask)).
            llm_features=sb.llm,
            t_sin=t_sin,
            cos=st.cos,
            sin=st.sin,
            image_indicator_index=st.idx,
            llm_token_mask=llm_mask,
            output_image_mask=st.img_mask,
            spatial_sequence_length=logical_n,
        )  # [1, L/sp, in_ch] sequence-sharded
        out = self.dit_ccl.all_gather_persistent_buffer(out, dim=1, mesh_axis=self.sp_axis)
        # Image tokens are at the FRONT [0:num_img].
        return ttnn.slice(out, [0, 0, 0], [1, st.num_img, self.model_config.in_channels])

    def _step_fn(self, st: _SizeState, sb: _SeqBuffers):
        """The whole denoising step of one size, captured as one trace: build x from z on device,
        both transformer forwards and their velocity gathers, the CFG blend and the Euler update,
        written back into z in place."""
        pad_rows = st.padded_len - st.num_img

        def step_fn(z_dev, t_sin, gw, smt):
            # The model masks every non-image row of x to 0, so padding z with zeros to L and
            # SP-sharding it on device gives the transformer input; cond and uncond share it.
            x_sp = ttnn.pad(z_dev, [(0, 0), (0, pad_rows), (0, 0)], value=0.0)
            x_sp = ttnn.mesh_partition(x_sp, dim=1, cluster_axis=self.sp_axis)  # [1, L/sp, in_ch]
            v_cond = self._velocity(self.cond, st, sb, x_sp, t_sin, sb.cond_llm_mask, self._cond_logical_n)
            v_uncond = self._velocity(self.uncond, st, sb, x_sp, t_sin, sb.uncond_llm_mask, st.uncond_logical_n)
            v = cfg_blend(v_cond, v_uncond, gw)
            # Euler update z <- z + v * (s - t), as Ideogram4Sampler.step, with the step size a device
            # tensor so one trace serves every step and preset.
            ttnn.copy(z_dev + v * smt, z_dev)
            return z_dev

        return step_fn

    def _free_size_state(self, st: _SizeState, sb: _SeqBuffers | None) -> None:
        for t in (st.z, st.cos, st.sin, st.img_mask, st.idx, st.uncond_logical_n):
            ttnn.deallocate(t)
        if sb is not None:
            for t in (sb.llm, sb.cond_llm_mask, sb.uncond_llm_mask):
                ttnn.deallocate(t)

    def _check_size(self, width: int, height: int) -> None:
        # The latent grid is (height, width) // (patch*ae); a non-divisible size would silently
        # floor (e.g. 513 -> a 512px image), so reject it up front with a clear error.
        vae_patch = self.patch * self.ae
        if height % vae_patch != 0 or width % vae_patch != 0:
            raise ValueError(f"height and width must be multiples of {vae_patch} (patch*ae); got {height}x{width}")
        # Before any device work, rather than after the denoising loop.
        self.decode_stage.vae_decoder.check_output_size(width, height)

    @property
    def resolutions(self) -> tuple[tuple[int, int], ...]:
        """The (width, height) sizes a traced call may ask for: the warmed ones (``warmed_sizes``).

        Unlike Wan's ``resolutions`` (the configured sizes), these are the sizes actually warmed: empty
        after ``run_warmup=False`` until ``warmup`` runs, and they grow with each ``warmup`` call.
        """
        return self.warmed_sizes

    @property
    def warmed_sizes(self) -> tuple[tuple[int, int], ...]:
        """The (width, height) sizes ``warmup`` prepared, in warmup order."""
        return tuple(self._sizes)

    @torch.no_grad()
    def warmup(self, sizes: Sequence[tuple[int, int]] | None = None) -> None:
        """Prepare the pipeline to serve ``sizes`` ((width, height) pairs; default the configured size).

        Runs in two phases so that no buffer that must outlive a trace is allocated after one is
        captured (a trace replay may overwrite memory allocated after its capture):

        1. Eager, at every size: compiles every program a request runs (encoder, text readback,
           both transformers, the step, the VAE decode), prepares the Conv2d weights of each VAE
           size, allocates every CCL buffer, and builds the per-size constants and the per-request
           buffers the step reads.
        2. Captures one denoising-step trace per size, kept until ``release_traces``.

        Afterwards a request at a warmed size compiles nothing, captures nothing and keeps no new
        device allocation; a request at any other size raises. Calling it again re-captures every
        trace, adding the new sizes.
        """
        if sizes is None:
            sizes = [(self._config.width, self._config.height)]
        sizes = [(int(w), int(h)) for w, h in sizes]
        for w, h in sizes:
            self._check_size(w, h)
        # Every warmed size's trace is re-captured, not only the new ones.
        _check_trace_region(self.mesh_device, self.mesh_device.shape, [*self._sizes, *sizes])

        # No trace may be live while phase 1 allocates.
        self.release_traces()

        llm_real, n_text = self._encode("warmup", warm_readback=True)
        for size in sizes:
            if size in self._sizes:
                continue
            _dq_log.info(f"Ideogram4 warmup: eager run at {size[0]}x{size[1]}")
            st = self._build_size_state(*size)
            self._sizes[size] = st
            if st.padded_len not in self._seq_buffers:
                self._seq_buffers[st.padded_len] = self._build_seq_buffers(st.padded_len)
            sb = self._seq_buffers[st.padded_len]
            self._write_prompt(st, sb, llm_real, n_text)
            self._step_fn(st, sb)(
                *(a if a is st.z else a.to(self.mesh_device) for a in self._step_inputs(st.z, 0.5, 3.0, 0.0))
            )
            self.decode_stage.decode(st.z, grid_h=st.grid_h, grid_w=st.grid_w)
            ttnn.synchronize_device(self.mesh_device)

        for size, st in self._sizes.items():
            _dq_log.info(f"Ideogram4 warmup: capturing the step trace at {size[0]}x{size[1]}")
            sb = self._seq_buffers[st.padded_len]
            st.tracer = Tracer(self._step_fn(st, sb), device=self.mesh_device, prep_run=False, clone_prep_inputs=False)
            st.tracer(*self._step_inputs(st.z, 0.5, 3.0, 0.0), tracer_execute_on_capture=False)
            self.num_trace_captures += 1
        ttnn.synchronize_device(self.mesh_device)

    def release_traces(self) -> None:
        """Release every kept step trace (the per-size buffers stay allocated)."""
        for st in self._sizes.values():
            if st.tracer is not None:
                st.tracer.release_trace()
                st.tracer = None

    @torch.no_grad()
    def __call__(
        self,
        *,
        prompts: Sequence[str],
        negative_prompts: Sequence[str] | None = None,
        num_inference_steps: int | None = None,
        seed: int = 0,
        traced: bool = True,
        height: int | None = None,
        width: int | None = None,
        preset: str = "V4_TURBO_12",
        guidance_scale: float | None = None,
        on_event: PipelineEventCallback | None = None,
        latents: torch.Tensor | None = None,
    ) -> list[Image.Image]:
        """Generate one image.

        ``traced=True`` replays the step trace ``warmup`` captured for the size and raises for a size
        that was not warmed up. ``traced=False`` runs the step eagerly: at a warmed size it uses the
        warmed buffers; before any warmup it works at any size, with buffers freed afterwards.
        """
        on_event = on_event if on_event is not None else null_callback
        assert len(prompts) == 1, "Ideogram4 pipeline generates one image per call"
        prompt = prompts[0]
        # Ideogram's unconditional pass is an intrinsically-unconditional distilled net, so a
        # user-supplied negative prompt has no effect; accept it for API parity but ignore it.
        if negative_prompts and any(negative_prompts):
            _dq_log.warning("Ideogram4 ignores negative_prompts (its unconditional net takes no text)")
        height = height if height is not None else self._config.height
        width = width if width is not None else self._config.width
        self._check_size(width, height)

        cfg = self.model_config
        dev = self.mesh_device
        st = self._sizes.get((width, height))
        if st is None and (traced or self._sizes):
            # After warmup, building a size's buffers or compiling its programs would allocate while
            # the kept traces are live, which their replays may overwrite.
            raise ValueError(
                f"size {width}x{height} was not warmed up; warmed sizes are {list(self._sizes)}. "
                f"Call warmup(sizes=[...]) with every size to serve before the first request."
            )
        temporary = st is None
        if temporary:
            st = self._build_size_state(width, height)
            sb = self._build_seq_buffers(st.padded_len)
        else:
            sb = self._seq_buffers[st.padded_len]

        sampler = Ideogram4Sampler.from_preset(preset, height=height, width=width)
        if num_inference_steps is not None and num_inference_steps != sampler.num_steps:
            _dq_log.warning(
                f"num_inference_steps={num_inference_steps} ignored; preset {preset} fixes "
                f"{sampler.num_steps} steps (distilled guidance schedule)"
            )
        if guidance_scale is not None:
            _dq_log.warning(
                f"guidance_scale={guidance_scale} OVERRIDES preset {preset}'s distilled per-step "
                f"guidance schedule at every step (the model was distilled for that schedule; a "
                f"constant override may degrade quality)"
            )
        # Initial noise: caller-supplied latents (e.g. one CPU-drawn tensor shared across runs, so
        # only the prompt varies), else drawn from the per-call seed via an explicit generator so z
        # is reproducible regardless of any other RNG use between here and now.
        if latents is not None:
            expected = (1, st.num_img, cfg.in_channels)
            if tuple(latents.shape) != expected:
                raise ValueError(f"latents shape {tuple(latents.shape)} != {expected} for {height}x{width}")
            z = latents.detach().to(device="cpu", dtype=torch.float32).clone()
        else:
            z = torch.randn(
                1, st.num_img, cfg.in_channels, dtype=torch.float32, generator=torch.Generator().manual_seed(seed)
            )

        on_event(SectionStart("total"))
        on_event(SectionStart("encoder"))
        llm_real, n_text = self._encode(prompt)
        self._write_prompt(st, sb, llm_real, n_text)
        ttnn.synchronize_device(dev)
        on_event(SectionEnd("encoder"))

        # The latent stays resident in st.z for the whole loop: written once here, then updated in
        # place by every step (step_fn ends with copy(z_next -> z)), and read once for decode.
        ttnn.copy_host_to_device_tensor(self._host_repl(z), st.z)
        step_fn = None if traced else self._step_fn(st, sb)
        self.step_trace = []
        _verbose = getattr(self, "_verbose_steps", False)
        on_event(SectionStart("denoising"))
        for n, i in enumerate(reversed(range(sampler.num_steps))):
            on_event(SectionStart(f"denoising_step_{n}"))
            t_val, s_val = sampler.times_for_step(i)
            gw = sampler.guidance_weight(i) if guidance_scale is None else guidance_scale
            args = self._step_inputs(st.z, t_val, gw, s_val - t_val)
            if traced:
                st.tracer(*args)
            else:
                step_fn(*(a if a is st.z else a.to(dev) for a in args))

            z_std = float(tensor.to_torch(st.z, mesh_axes=[None, None, None]).float().std()) if _verbose else 0.0
            self.step_trace.append((t_val, s_val, gw, 0.0, z_std, 0.0))
            if _verbose:
                _dq_log.info(
                    f"  step {n + 1}/{sampler.num_steps}: t={t_val:.4f} s={s_val:.4f} gw={gw:.0f} z_std={z_std:.4f}"
                )
            on_event(DenoiseStep(step=n, total=sampler.num_steps, sigma=t_val))
            on_event(SectionEnd(f"denoising_step_{n}"))
        ttnn.synchronize_device(dev)
        on_event(SectionEnd("denoising"))

        on_event(SectionStart("vae"))
        decoded = self.decode_stage.decode(st.z, grid_h=st.grid_h, grid_w=st.grid_w)
        if temporary:
            self._free_size_state(st, sb)
        # [-1,1] float [B,3,H,W] -> list[PIL.Image] (VaeImageProcessor denorm+clamp), matching
        # the Flux1/QwenImage pipelines' return contract.
        images = self._image_processor.postprocess(decoded.float(), output_type="pil")
        on_event(SectionEnd("vae"))
        on_event(SectionEnd("total"))
        return images
