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
from collections.abc import Sequence
from dataclasses import dataclass

import transformers as _tf
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL as _AutoencoderKL
from loguru import logger as _dq_log
from PIL import Image
from safetensors.torch import load_file as _load_file
from safetensors.torch import save_file as _save_file

from ...encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder, create_rope_tensors
from ...models.transformers.transformer_ideogram4 import Ideogram4Transformer, rope_halfsplit_to_interleaved
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, VAEParallelConfig
from ...parallel.manager import CCLManager
from ...reference.ideogram4 import modeling_ideogram4
from ...reference.ideogram4.constants import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    QWEN3_VL_ACTIVATION_LAYERS,
    SEQUENCE_PADDING_INDICATOR,
)
from ...reference.ideogram4.dequant import dequant_fp8_state_dict
from ..cfg import create_submeshes
from ..events import DenoiseStep, PipelineEventCallback, SectionEnd, SectionStart, null_callback
from ..pipeline_api import PipelineAPIMixin

# Opt-out switch for the bf16 dequant cache (default on). Dequantizing the fp8
# checkpoints (encoder + 2 transformers) dominates pipeline build time; caching the
# bf16 result next to the source makes subsequent builds skip both the fp8 load and
# the dequant. Set IDEOGRAM4_DEQUANT_CACHE=0 to force a fresh dequant.
_DEQUANT_CACHE = os.environ.get("IDEOGRAM4_DEQUANT_CACHE", "1") != "0"


def _dq(path):
    """Load + dequantize an fp8 checkpoint to bf16, caching the bf16 result on disk.

    The cache file lives beside the source (``<path>.dequant-bf16.safetensors``) and is
    validated against the source's size+mtime (stamped into the safetensors metadata), so
    it is transparently rebuilt if the source weights change. Cache write failures (e.g.
    read-only weights dir) fall back to in-memory dequant without erroring.
    """
    if not _DEQUANT_CACHE:
        return dequant_fp8_state_dict(_load_file(path))

    cache_path = path + ".dequant-bf16.safetensors"
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

# SP shards the sequence across the ring; each per-device shard must be tile-aligned, i.e. the
# padded length is a multiple of tile_size * sp_factor. This is exactly the shared tt_dit helper
# get_padded_vision_seq_len(N, num_devices) (ring SDPA pads+masks any partial tail internally).

_DEFAULT_WEIGHTS_DIR = os.environ.get("IDEOGRAM4_WEIGHTS")
_DEFAULT_QWEN_REPO = "Qwen/Qwen3-VL-8B-Instruct"

# Per-mesh-shape parallelism presets (Blackhole). Ideogram is always cfg-parallel factor 1
# (both distilled nets resident on the full mesh, sequential CFG); tp/sp are (factor, mesh_axis).
_PRESETS: dict[tuple[int, ...], dict] = {
    # BH loudbox: SP4 x TP2 (the default full-mesh config).
    (4, 2): {"tp": (2, 1), "sp": (4, 0), "num_links": 2, "topology": ttnn.Topology.Linear},
    # BH Galaxy 4x8, 2D torus Ring: TP4 (axis 0) x SP8 (axis 1).
    (4, 8): {"tp": (4, 0), "sp": (8, 1), "num_links": 2, "topology": ttnn.Topology.Ring},
}


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
    weights_dir: str
    qwen_repo: str

    @classmethod
    def default(
        cls,
        *,
        mesh_shape: ttnn.MeshShape,
        topology: ttnn.Topology | None = None,
        num_links: int | None = None,
        height: int = 512,
        width: int = 512,
        weights_dir: str = _DEFAULT_WEIGHTS_DIR,
        qwen_repo: str = _DEFAULT_QWEN_REPO,
    ) -> Ideogram4PipelineConfig:
        preset = _PRESETS.get(tuple(mesh_shape))
        if preset is None:
            raise ValueError(
                f"no Ideogram4 parallelism preset for mesh shape {tuple(mesh_shape)}; known: {list(_PRESETS)}"
            )
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
            weights_dir=weights_dir,
            qwen_repo=qwen_repo,
        )


class Ideogram4Pipeline(PipelineAPIMixin):
    """Ideogram 4.0 text-to-image pipeline (device-resident, sequential asymmetric CFG).

    Fits the tt_dit pipeline mold (PipelineAPIMixin, Ideogram4PipelineConfig, create_pipeline,
    event callbacks, PIL output). Ideogram's cond/uncond are distinct distilled nets, so — unlike
    the cfg-parallel Flux/QwenImage pipelines — both stay resident on the full mesh and run
    sequentially, and CFG is blended on-device (cfg_blend) rather than across submeshes.
    """

    def __init__(self, *, device: ttnn.MeshDevice, config: Ideogram4PipelineConfig) -> None:
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
        self.topology = config.topology
        weights_dir = config.weights_dir
        if not weights_dir:
            raise ValueError(
                "Ideogram4 weights directory not set: pass weights_dir=... to create_pipeline "
                "or set the IDEOGRAM4_WEIGHTS environment variable to the fp8 checkpoint dir."
            )
        self.weights_dir = weights_dir
        qwen_repo = config.qwen_repo
        self.config = modeling_ideogram4.Ideogram4Config()
        self.patch, self.ae = 2, 8
        cfg = self.config

        ccl = CCLManager(mesh_device, num_links=config.num_links, topology=config.topology)
        self.ccl = ccl
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
        self._enc_rope_theta = qcfg.rope_scaling.get("rope_theta", qcfg.rope_theta)
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
                ccl_manager=ccl,
                parallel_config=dit_pc,
                padding_config=padding_config,
            )
            m.load_torch_state_dict(sd)
            del sd
            _gc.collect()
            return m

        self.cond = _build_dit("transformer")
        self.uncond = _build_dit("unconditional_transformer")

        # Combined-step trace (lazily captured on the first denoise step), covering both
        # transformer forwards + velocity gathers + CFG blend + Euler update in ONE trace.
        # Tracing eliminates the 34-layer x 2-branch op-dispatch host overhead per step; a
        # single trace keeps all persistent-buffer allocation BEFORE capture (no corruption).
        self._step_tracer = None

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

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        height: int = 512,
        width: int = 512,
        weights_dir: str = _DEFAULT_WEIGHTS_DIR,
        qwen_repo: str = _DEFAULT_QWEN_REPO,
        topology: ttnn.Topology | None = None,
        num_links: int | None = None,
    ) -> Ideogram4Pipeline:
        config = Ideogram4PipelineConfig.default(
            mesh_shape=mesh_device.shape,
            height=height,
            width=width,
            weights_dir=weights_dir,
            qwen_repo=qwen_repo,
            topology=topology,
            num_links=num_links,
        )
        return cls(device=mesh_device, config=config)

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
        from the mesh-shape preset (tp_axis is ignored, preset is authoritative)."""
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

    def _encode(self, prompt: str):
        """Tokenize + run the device encoder -> real interleaved llm_features (host) + n_text.

        CONSTANT-SHAPE: the token ids are padded to exactly ``MAX_TEXT_TOKENS`` and the
        encoder rope is built for that fixed length, so the encoder always sees the same
        shape regardless of prompt length (no per-length recompile / shape-keyed POBs).
        The Qwen3-VL encoder is CAUSAL (model_qwen3vl: is_causal=(attention_bias is None),
        and we pass attention_mask=None), so the real rows [0:n_text] are unaffected by the
        trailing pad. We slice the taps back to [:, :n_text] and return the real-length feats.
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
        cos, sin = create_rope_tensors(
            1, MAX_TEXT_TOKENS, None, self._enc_head_dim, self._enc_rope_theta, self._enc_mrope
        )
        tt_ids = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device)
        taps = self.encoder.forward(
            tt_ids,
            attention_mask=None,
            pos_embeds=(bf16_tensor(cos, device=self.mesh_device), bf16_tensor(sin, device=self.mesh_device)),
        )
        # taps: 13 x [1, MAX_TEXT_TOKENS, 4096], REPLICATED across the whole mesh.
        # Read back cheaply: slice to the real text rows [0:n_text] ON DEVICE (causal => pad rows
        # never influence the real rows, and the model masks them anyway), then read a SINGLE
        # replica via get_device_tensors(t)[0]. The old `to_torch(mesh_axes=[None,None,None])`
        # built a composer over every mesh axis, so ttnn.to_torch pulled ALL N device copies to
        # host and discarded N-1 (~1.4 s/chip => ~45 s on a 32-chip galaxy). Slicing to n_text
        # first and reading one replica moves ~n_text/2048 x 1/N the bytes; values are identical
        # (same device-0 data). Interleave to [1, n_text, 53248] (element = f*13 + l) on host.
        taps_t = [
            ttnn.to_torch(ttnn.get_device_tensors(ttnn.slice(t, [0, 0, 0], [1, n_text, t.shape[-1]]))[0]) for t in taps
        ]
        return interleave_layer_taps(taps_t).to(torch.bfloat16), n_text

    def _seq_dev(self, t: torch.Tensor, seq_dim: int) -> ttnn.Tensor:
        """bf16 device tensor: shard on the SP axis along ``seq_dim`` when SP>1, else replicate."""
        if self.sp_factor > 1:
            return bf16_tensor(t, device=self.mesh_device, mesh_axis=self.sp_axis, shard_dim=seq_dim)
        return bf16_tensor(t, device=self.mesh_device)

    def _idx_dev(self, t: torch.Tensor, seq_dim: int) -> ttnn.Tensor:
        """uint32 device tensor (embedding index): shard on the SP axis when SP>1, else replicate."""
        if self.sp_factor > 1:
            mesh_axes = [self.sp_axis if d == seq_dim else None for d in range(t.ndim)]
            return tensor.from_torch(t, device=self.mesh_device, dtype=ttnn.uint32, mesh_axes=mesh_axes)
        return tensor.from_torch(t, device=self.mesh_device, dtype=ttnn.uint32)

    def _branch(self, n_pre, num_img, grid_h, grid_w, llm_real):
        """Build the per-branch device tensors with a CONSTANT [image | text | pad] layout.

        The total length ``L = get_padded_vision_seq_len(num_img + MAX_TEXT_TOKENS, sp_factor)`` is the
        same for every prompt (cond/uncond, any n_text), so the per-token tensors are allocated
        once and reused -> no shape-keyed persistent-output-buffer leak / DRAM fragmentation.

        Layout (image-FIRST):
          [0 : num_img]              -> image tokens   (OUTPUT_IMAGE_INDICATOR)
          [num_img : num_img+n_pre]  -> real text      (LLM_TOKEN_INDICATOR)
          [num_img+n_pre : L]        -> pad            (SEQUENCE_PADDING_INDICATOR -> both masks 0)

        This reorder (text-AFTER-image instead of text-before) is LOSSLESS: the denoiser uses
        FULL attention + per-token MRoPE (permutation-equivariant), so each token's output is
        identical regardless of position in the sequence as long as it keeps its own position_ids.
        The REAL length ``seq = num_img + n_pre`` is returned as ``seq`` and passed to the model
        as ``spatial_sequence_length`` (ring-SDPA ``logical_n``), masking the trailing pad. We do
        NOT pass L as spatial_sequence_length.
        """
        cfg = self.config
        seq = n_pre + num_img  # REAL length (logical_n); image + real text, no pad
        # Constant total length across all prompts (image + MAX text, SP-aligned).
        L = get_padded_vision_seq_len(num_img + MAX_TEXT_TOKENS, self.sp_factor)

        # Indicator: PAD sentinel everywhere, then stamp image-front + real-text region.
        ind = torch.full((1, L), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
        ind[:, :num_img] = OUTPUT_IMAGE_INDICATOR
        if n_pre:
            ind[:, num_img : num_img + n_pre] = LLM_TOKEN_INDICATOR

        llm = torch.zeros(1, L, cfg.llm_features_dim, dtype=torch.bfloat16)
        if n_pre and llm_real is not None:
            llm[:, num_img : num_img + n_pre] = llm_real

        pos = torch.zeros(1, L, 3, dtype=torch.long)
        # Image grid-positions occupy the FRONT [0:num_img].
        hh = torch.arange(grid_h).repeat_interleave(grid_w)
        ww = torch.arange(grid_w).repeat(grid_h)
        pos[:, :num_img, 0] = IMAGE_POSITION_OFFSET
        pos[:, :num_img, 1] = IMAGE_POSITION_OFFSET + hh
        pos[:, :num_img, 2] = IMAGE_POSITION_OFFSET + ww
        # Text positions occupy [num_img : num_img+n_pre]; pad rows keep position 0 (inert).
        if n_pre:
            tp = torch.arange(n_pre)
            pos[:, num_img : num_img + n_pre] = torch.stack([tp, tp, tp], dim=1)

        rope = modeling_ideogram4.Ideogram4MRoPE(
            head_dim=cfg.emb_dim // cfg.num_heads, base=cfg.rope_theta, mrope_section=cfg.mrope_section
        )
        cos, sin = rope(pos)
        cos4 = cos.unsqueeze(1).to(torch.bfloat16)  # [1, 1, L, head_dim]
        sin4 = sin.unsqueeze(1).to(torch.bfloat16)
        # Denoiser uses interleaved RoPE (rotary_embedding_llama); permute cos/sin to match.
        cos4, sin4 = rope_halfsplit_to_interleaved(cos4, sin4, cfg.emb_dim // cfg.num_heads)
        llm_mask = (ind == LLM_TOKEN_INDICATOR).float().unsqueeze(-1)  # [1, L, 1]
        img_mask = (ind == OUTPUT_IMAGE_INDICATOR).float().unsqueeze(-1)
        idx = (ind == OUTPUT_IMAGE_INDICATOR).to(torch.int32)  # [1, L]
        return dict(
            llm=self._seq_dev(llm, 1),
            cos=self._seq_dev(cos4, 2),
            sin=self._seq_dev(sin4, 2),
            llm_mask=self._seq_dev(llm_mask, 1),
            img_mask=self._seq_dev(img_mask, 1),
            idx=self._idx_dev(idx, 1),
            seq=seq,
            padded_len=L,
            n_pre=n_pre,
        )

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
    ) -> list[Image.Image]:
        on_event = on_event if on_event is not None else null_callback
        assert len(prompts) == 1, "Ideogram4 pipeline generates one image per call"
        prompt = prompts[0]
        # Ideogram's unconditional pass is an intrinsically-unconditional distilled net, so a
        # user-supplied negative prompt has no effect; accept it for API parity but ignore it.
        if negative_prompts and any(negative_prompts):
            _dq_log.warning("Ideogram4 ignores negative_prompts (its unconditional net takes no text)")
        height = height if height is not None else self._config.height
        width = width if width is not None else self._config.width

        # The latent grid is (height, width) // (patch*ae); a non-divisible size would silently
        # floor (e.g. 513 -> a 512px image), so reject it up front with a clear error.
        vae_patch = self.patch * self.ae  # 16
        if height % vae_patch != 0 or width % vae_patch != 0:
            raise ValueError(f"height and width must be multiples of {vae_patch} (patch*ae); got {height}x{width}")

        cfg = self.config
        dev = self.mesh_device
        grid_h, grid_w = height // (self.patch * self.ae), width // (self.patch * self.ae)
        num_img = grid_h * grid_w

        on_event(SectionStart("total"))
        on_event(SectionStart("encoder"))
        llm_real, n_text = self._encode(prompt)
        ttnn.synchronize_device(dev)
        on_event(SectionEnd("encoder"))
        cond_b = self._branch(n_text, num_img, grid_h, grid_w, llm_real)
        uncond_b = self._branch(0, num_img, grid_h, grid_w, None)

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
        # Seed the initial noise from the per-call seed via an explicit generator, so z is
        # reproducible regardless of any other RNG use between here and now.
        z = torch.randn(1, num_img, cfg.in_channels, dtype=torch.float32, generator=torch.Generator().manual_seed(seed))

        def _branch_velocity(model, br, x_full):
            """Transformer forward + on-device SP-axis gather + image-token slice ->
            [1, num_img, in_ch] (device, replicated). All CCL ops are inside the trace so the
            ping-pong semaphore sequence the trace records stays self-consistent on replay."""
            out = model(
                x=x_full,
                llm_features=br["llm"],
                t_sin=br["t_sin"],
                cos=br["cos"],
                sin=br["sin"],
                image_indicator_index=br["idx"],
                llm_token_mask=br["llm_mask"],
                output_image_mask=br["img_mask"],
                spatial_sequence_length=br["seq"],
            )  # [1, padded_len/sp, in_ch] sequence-sharded
            if self.sp_factor > 1:
                out = self.ccl.all_gather_persistent_buffer(out, dim=1, mesh_axis=self.sp_axis)
            # Image tokens are at the FRONT [0:num_img] (constant [image | text | pad] layout).
            return ttnn.slice(out, [0, 0, 0], [1, num_img, cfg.in_channels])

        # The ENTIRE per-step computation runs in a SINGLE trace: build x from z on device,
        # cond forward, uncond forward, the two SP-axis velocity gathers, the asymmetric-CFG
        # blend and the Euler update. Capturing all of it in one trace is what makes tracing
        # CORRECT here: the Tracer runs a prep pass that allocates every persistent buffer
        # (the x pad/partition output, both branches' ring-SDPA / all-gather buffers, the
        # blend/Euler temporaries) BEFORE capture, so no device buffer is allocated while a
        # trace is active ("Allocating device buffers is unsafe ... may be corrupted once a
        # trace is executed"). Per-step inputs: the resident device latent z (the previous
        # step's output, copied into the captured z slot in place -> the latent NEVER round-
        # trips to host mid-loop), a HOST t_sin, and 3 scalar HOST tensors (gw, 1-gw, s-t).
        # The trace RETURNS the next z (device), fed straight back next step; z is read back
        # to host exactly ONCE, after the loop, for decode.
        # Constant total length L (same for cond/uncond now that the layout is [image|text|pad]).
        L = cond_b["padded_len"]
        assert uncond_b["padded_len"] == L

        # Number of trailing (text|pad) rows to append to z to reach the constant length L.
        # Image tokens are at the FRONT [0:num_img]; the model masks all non-image rows to 0
        # (forward: x = input_proj(x * output_image_mask) * output_image_mask), so appending
        # zeros here is numerically identical to the host-built [image|zeros...] tensor.
        _x_pad_rows = L - num_img

        def _x_full_from_z(z_dev):
            """Build the SP-sharded transformer x input ON DEVICE from the replicated latent z.

            z_dev: [1, num_img, in_ch] replicated. Pads trailing rows to the constant length L
            (image-first layout), then SP-shards dim 1 across the SP axis via mesh_partition
            (the inverse of the SP all-gather -> each device gets its [1, L/sp, in_ch] shard).
            Replaces the per-step HOST rebuild + host->device copy of x_cond/x_uncond. cond and
            uncond consume IDENTICAL x (conditioning lives only in llm/masks/idx/logical_n), and
            the block never mutates x in place, so a single shared tensor feeds both branches.
            """
            x_full = ttnn.pad(z_dev, [(0, 0), (0, _x_pad_rows), (0, 0)], value=0.0)  # [1, L, in_ch] repl
            if self.sp_factor > 1:
                x_full = ttnn.mesh_partition(x_full, dim=1, cluster_axis=self.sp_axis)  # [1, L/sp, in_ch]
            return x_full

        def step_fn(z_dev, t_sin, gw, smt):
            # t_sin feeds both branches' adaln; share the same per-step embedding.
            cond_b["t_sin"] = t_sin
            uncond_b["t_sin"] = t_sin
            # Build the (padded, SP-sharded) x input on device from the resident z (no host round-trip).
            x_sp = _x_full_from_z(z_dev)
            v_cond = _branch_velocity(self.cond, cond_b, x_sp)
            v_uncond = _branch_velocity(self.uncond, uncond_b, x_sp)
            # Asymmetric CFG via lerp: v = gw*v_cond + (1-gw)*v_uncond. gw is a 1-element
            # device tensor so it can be updated per step inside the trace.
            v = cfg_blend(v_cond, v_uncond, gw)
            # Euler update via the sampler's step() (production path), with the per-step size s-t
            # passed as a device tensor so it re-patches inside the trace -> next z [1, num_img, in_ch].
            z_next = sampler.step(z_dev, v, scale=smt)
            # Write the update back INTO the resident z buffer in place, so the SAME buffer is
            # both this step's input (pad reads it at the top) and the running state the next
            # replay reads. The trace records read(z_dev) -> ... -> copy(z_next -> z_dev); on
            # replay this executes as an in-place recurrence with NO host feedback and NO
            # post-capture device allocation (the caller reuses the identical z buffer each
            # step, so the Tracer's _update_input is a no-op -> no ttnn.copy from the host side).
            ttnn.copy(z_next, z_dev)
            return z_dev

        def _scalar_host(val):
            # 1-element bf16 HOST tensor, replicated across the mesh; the tracer moves it into
            # its captured device slot each step (no new device allocation on replay).
            return tensor.from_torch(
                torch.tensor([[[float(val)]]], dtype=torch.bfloat16),
                device=dev,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_axes=[None, None, None],
                on_host=True,
            )

        def _host_repl(t: torch.Tensor):
            # bf16 HOST tensor replicated across the mesh (for t_sin, which the model consumes
            # replicated, not SP-sharded). The tracer moves it onto its captured device slot.
            return tensor.from_torch(
                t.to(torch.bfloat16),
                device=dev,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_axes=[None] * t.ndim,
                on_host=True,
            )

        self.step_trace = []
        # The latent z stays RESIDENT ON DEVICE across the whole loop in a SINGLE persistent
        # buffer. It is uploaded once here (replicated) and thereafter the trace updates it
        # IN PLACE each step (step_fn: read z -> ... -> ttnn.copy(z_next, z_dev)). No per-step
        # device->host readback, no per-step host x rebuild, and no host-side feedback of the
        # output -> the caller passes the identical z buffer every step, so the Tracer's
        # _update_input sees a matching buffer_address and is a no-op, so nothing is allocated on
        # device once the trace is live. The x transformer input is built on device from z each step
        # (pad -> SP mesh_partition), inside the trace.
        z_dev = _host_repl(z).to(dev)  # [1, num_img, in_ch] replicated, device-resident persistent
        z_out = z_dev
        steps = list(reversed(range(sampler.num_steps)))
        _verbose = getattr(self, "_verbose_steps", False)
        on_event(SectionStart("denoising"))
        for n, i in enumerate(steps):
            on_event(SectionStart(f"denoising_step_{n}"))
            t_val, s_val = sampler.times_for_step(i)
            gw = sampler.guidance_weight(i) if guidance_scale is None else guidance_scale
            smt = s_val - t_val
            t_sin = Ideogram4Transformer.sinusoidal_embedding(torch.tensor([t_val]), cfg.emb_dim).unsqueeze(1)
            # Always pass the persistent z buffer. step_fn updates it in place and returns the
            # same buffer, so its address never changes across steps; on replay the Tracer sees a
            # matching buffer_address and skips the input copy (no host->device copy, no device
            # allocation once the trace is live) -- the same reuse the removed None-path gave, but
            # via the stock Tracer. t_sin + 3 scalars are HOST tensors.
            z_arg = z_dev
            args = (
                z_arg,
                _host_repl(t_sin),
                _scalar_host(gw),
                _scalar_host(smt),
            )
            if traced:
                if self._step_tracer is None:
                    # step_fn writes its result back into z_dev in place (ttnn.copy(z_next, z_dev)),
                    # so the prep pass MUST run on a clone; otherwise it applies step 0's Euler
                    # update to the real z_dev and the executed trace applies it a second time,
                    # double-counting the first step. Cloning only affects the one prep input; the
                    # per-step buffer reuse keys on z_dev's address at _execute time, not on the
                    # prep clone, so it is unaffected.
                    self._step_tracer = Tracer(step_fn, device=dev, clone_prep_inputs=True)
                z_out = self._step_tracer(*args)
            else:
                # Untraced: move the host inputs onto device and run the step eagerly (no
                # capture/replay). Same device computation, just per-op host dispatch each step.
                z_out = step_fn(*(a.to(dev) if isinstance(a, ttnn.Tensor) and a.device() is None else a for a in args))

            # z stays on device; only read it back for the (optional) per-step std log.
            z_std = float(tensor.to_torch(z_out, mesh_axes=[None, None, None]).float().std()) if _verbose else 0.0
            self.step_trace.append((t_val, s_val, gw, 0.0, z_std, 0.0))
            if _verbose:
                _dq_log.info(
                    f"  step {n + 1}/{sampler.num_steps}: t={t_val:.4f} s={s_val:.4f} gw={gw:.0f} z_std={z_std:.4f}"
                )
            on_event(DenoiseStep(step=n, total=sampler.num_steps, sigma=t_val))
            on_event(SectionEnd(f"denoising_step_{n}"))
        on_event(SectionEnd("denoising"))
        # One readback after the loop for decode (z never touched host inside the loop).
        z_host = tensor.to_torch(z_out, mesh_axes=[None, None, None]).float()
        ttnn.synchronize_device(dev)

        # Release the combined-step trace BEFORE decode. The trace bakes in this call's
        # CCLManager ping-pong semaphores/buffers and the closed-over per-branch fixed tensors;
        # the next __call__ re-runs the (untraced) encoder and rebuilds those branch tensors,
        # advancing the shared CCL state out from under a persisted trace. Releasing here also
        # means decode's device allocations happen with no active trace (the "allocating
        # buffers while a trace is active is unsafe" hazard). Re-capturing per call keeps each
        # loop self-consistent and frees the trace region for the next run.
        if self._step_tracer is not None:
            self._step_tracer.release_trace()
            self._step_tracer = None

        # Free the large per-call device tensors now that the denoise loop + trace are done (the
        # [1, L, 53248] llm tensors dominate): a long-lived server would otherwise depend on GC
        # timing and fragment DRAM. Safe here — z was read back above and the trace is released.
        for br in (cond_b, uncond_b):
            for key in ("llm", "cos", "sin", "llm_mask", "img_mask", "idx"):
                ttnn.deallocate(br[key])
        ttnn.deallocate(z_out)

        on_event(SectionStart("vae"))
        decoded = self.decode_stage.decode(bf16_tensor(z_host, device=dev), grid_h=grid_h, grid_w=grid_w)
        # [-1,1] float [B,3,H,W] -> list[PIL.Image] (VaeImageProcessor denorm+clamp), matching
        # the Flux1/QwenImage pipelines' return contract.
        images = self._image_processor.postprocess(decoded.float(), output_type="pil")
        on_event(SectionEnd("vae"))
        on_event(SectionEnd("total"))
        return images
