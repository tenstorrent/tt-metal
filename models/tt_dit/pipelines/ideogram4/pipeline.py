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

    @staticmethod
    def to_images(decoded: torch.Tensor) -> torch.Tensor:
        """[-1,1] float -> uint8 [B,H,W,3], matching the reference final step."""
        decoded = decoded.float().clamp(-1.0, 1.0)
        return ((decoded + 1.0) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1)


def cfg_blend(v_cond: ttnn.Tensor, v_uncond: ttnn.Tensor, guidance_weight: float) -> ttnn.Tensor:
    """Asymmetric-CFG velocity blend: v = gw*v_cond + (1-gw)*v_uncond (device)."""
    return v_cond * guidance_weight + v_uncond * (1.0 - guidance_weight)


import contextlib as _contextlib


@_contextlib.contextmanager
def _patch_zeros_for_trace(device):
    """Scope a wrapper around ``ttnn.zeros`` that makes 0-ELEMENT allocations trace-safe.

    The Ideogram4 attention block creates an empty "joint" tensor for ring SDPA via
    ``ttnn.zeros([B, n_local_heads, 0, head_dim], ...)``. ttnn.zeros host-writes for
    zero-element tensors (no device memset path), which is illegal during trace capture
    ("Writes are not supported during trace capture"). We can't edit the transformer, so
    during capture we hand back a single pre-allocated persistent all-zeros tensor per
    unique (shape, dtype, layout). This is numerically lossless — the empty joint is a
    fixed constant — and only zero-element requests are intercepted; all other ttnn.zeros
    calls fall through to the real op unchanged.
    """
    real_zeros = ttnn.zeros
    cache: dict = {}

    def _zeros(shape, *args, **kwargs):
        if 0 in tuple(shape):
            key = (tuple(shape), kwargs.get("dtype"), kwargs.get("layout"))
            t = cache.get(key)
            if t is None:
                # Allocate once with the real op (outside the recorded trace path).
                t = real_zeros(shape, *args, **kwargs)
                cache[key] = t
            return t
        return real_zeros(shape, *args, **kwargs)

    ttnn.zeros = _zeros
    try:
        yield
    finally:
        ttnn.zeros = real_zeros


__all__ = ["Ideogram4DecodeStage", "Ideogram4Sampler", "cfg_blend", "unpatchify_latent"]


# =============================================================================
# Full pipeline class. Mirrors the QwenImage/SD3.5 pipeline shape: a factory
# (from_pretrained) that builds device-resident models, and __call__ for T2I.
#
# CFG is run SEQUENTIALLY on the full submesh (see the CFG decision doc): the
# conditional pass (text+image) and unconditional pass (image-only) are asymmetric,
# so giving each pass the whole mesh beats splitting it. Everything is device-
# resident at TP=4 (encoder + cond + uncond transformers + VAE); no dynamic offload.
# =============================================================================

import gc as _gc
import os as _os

import transformers as _tf
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL as _AutoencoderKL
from loguru import logger as _dq_log
from safetensors.torch import load_file as _load_file
from safetensors.torch import save_file as _save_file

from ...encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder, create_rope_tensors
from ...models.transformers.transformer_ideogram4 import Ideogram4Transformer
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor, VAEParallelConfig
from ...parallel.manager import CCLManager
from ...reference.ideogram4 import modeling_ideogram4
from ...reference.ideogram4.constants import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    QWEN3_VL_ACTIVATION_LAYERS,
)
from ...reference.ideogram4.dequant import dequant_fp8_state_dict

# Opt-out switch for the bf16 dequant cache (default on). Dequantizing the fp8
# checkpoints (encoder + 2 transformers) dominates pipeline build time; caching the
# bf16 result next to the source makes subsequent builds skip both the fp8 load and
# the dequant. Set IDEOGRAM4_DEQUANT_CACHE=0 to force a fresh dequant.
_DEQUANT_CACHE = _os.environ.get("IDEOGRAM4_DEQUANT_CACHE", "1") != "0"


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
        st = _os.stat(path)
        stamp = f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return dequant_fp8_state_dict(_load_file(path))

    if _os.path.exists(cache_path):
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
        _os.replace(tmp, cache_path)
        _dq_log.info(f"dequant cache WROTE: {cache_path}")
    except Exception as e:  # noqa: BLE001 - non-fatal: keep going with in-memory sd
        _dq_log.warning(f"dequant cache write failed ({e}); using in-memory dequant")
    return sd


# SP shards the sequence; the padded length must be a multiple of
# Ideogram4TransformerBlock.sdpa_k_chunk_size * sp_factor (matches _sp_padded_len in
# test_transformer_ideogram4.py). k_chunk is 256 on Blackhole.
_SP_K_CHUNK = 256

# Max prompt length. Mirrors the reference Ideogram4PipelineConfig.max_text_tokens (2048):
# the reference rejects (does NOT truncate) prompts longer than this, so we do the same.
MAX_TEXT_TOKENS = 2048


def _sp_padded_len(seq_len: int, sp_factor: int) -> int:
    """Pad the sequence so each SP shard is k_chunk- and tile-aligned (ring SDPA).

    Mirrors _sp_padded_len in test_transformer_ideogram4.py: the padded length must
    be a multiple of sdpa_k_chunk_size * sp_factor so each per-device shard is a whole
    number of k_chunks (and therefore tile-aligned). No-op when SP is disabled.
    """
    if sp_factor <= 1:
        return seq_len
    divisor = _SP_K_CHUNK * sp_factor
    return ((seq_len + divisor - 1) // divisor) * divisor


class Ideogram4Pipeline:
    """Ideogram 4.0 text-to-image pipeline (device-resident, TP=4, sequential CFG)."""

    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        weights_dir: str,
        qwen_repo: str = "Qwen/Qwen3-VL-8B-Instruct",
        tp_axis: int = 1,
        num_links: int = 1,
    ) -> None:
        self.mesh_device = mesh_device
        self.weights_dir = weights_dir
        self.tp_axis = tp_axis
        self.config = modeling_ideogram4.Ideogram4Config()
        self.patch, self.ae = 2, 8
        tp_factor = tuple(mesh_device.shape)[tp_axis]
        self.tp_factor = tp_factor
        # Sequence-parallel (SP) over the other mesh axis. Auto-enabled from the mesh shape:
        # a (1,4) submesh -> sp=1 (TP=4 only, unchanged); the full (2,4) mesh -> sp=2 x tp=4.
        # SP shards the sequence (ring/all-gather attention) and is the hi-res win (4k/16k tok),
        # where activations dominate. The denoiser block path is validated in
        # test_transformer_ideogram4.py (sp2tp4); pipeline wiring below is UNVALIDATED end-to-end.
        self.sp_axis = 1 - tp_axis
        self.sp_factor = tuple(mesh_device.shape)[self.sp_axis]
        cfg = self.config

        ccl = CCLManager(mesh_device, num_links=num_links, topology=ttnn.Topology.Linear)
        self.ccl = ccl
        dit_pc = DiTParallelConfig(
            cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
            tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
            sequence_parallel=ParallelFactor(factor=self.sp_factor, mesh_axis=self.sp_axis),
        )
        from ...utils.padding import PaddingConfig

        padding_config = (
            PaddingConfig.from_tensor_parallel_factor(cfg.num_heads, cfg.emb_dim // cfg.num_heads, tp_factor)
            if cfg.num_heads % tp_factor != 0
            else None
        )

        # --- text encoder (TP=4) ---
        hf = _tf.AutoModel.from_pretrained(qwen_repo, torch_dtype=torch.bfloat16)
        lm = hf.language_model if hasattr(hf, "language_model") else hf.model.language_model
        enc_sd = _dq(f"{weights_dir}/text_encoder/model.safetensors")
        enc_sd = {k[len("language_model.") :]: v for k, v in enc_sd.items() if k.startswith("language_model.")}
        qcfg = lm.config
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
            parallel_config=EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis)),
            ccl_manager=ccl,
            # FSDP-shard encoder weights across the SP (non-TP) axis instead of replicating
            # them. The encoder runs once (outside the denoise loop), so the per-layer weight
            # all-gather overhead is negligible, while it frees ~(sp_factor-1)/sp_factor of the
            # encoder's resident DRAM during denoise (needed to fit 2048px at SP4xTP2).
            # Auto-disables when the non-TP axis is size 1 (e.g. the TP=4 (1,4) submesh).
            is_fsdp=True,
        )
        self.encoder.load_torch_state_dict(enc_sd)
        del hf, lm, enc_sd
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
        akl.load_state_dict({k: v for k, v in vae_sd.items() if not k.startswith("bn.")}, strict=False)
        akl = akl.to(torch.bfloat16).eval()
        vae = Ideogram4VAEDecoder.from_torch(
            akl,
            mesh_device=mesh_device,
            parallel_config=VAEParallelConfig(tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis)),
            ccl_manager=ccl,
        )
        self.decode_stage = Ideogram4DecodeStage(vae, mesh_device=mesh_device, patch=self.patch)
        del akl, vae_sd
        _gc.collect()

    @classmethod
    def from_pretrained(
        cls,
        mesh_device,
        weights_dir=os.environ.get("IDEOGRAM4_WEIGHTS", "/data/cglagovich/ideogram-4-fp8"),
        **kw,
    ):
        return cls(mesh_device=mesh_device, weights_dir=weights_dir, **kw)

    def _encode(self, prompt: str):
        """Tokenize + run the device encoder -> real interleaved llm_features (host) + n_text."""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        n_text = ids.shape[1]
        if n_text > MAX_TEXT_TOKENS:
            raise ValueError(
                f"prompt tokenizes to {n_text} text tokens; Ideogram 4 supports at most "
                f"{MAX_TEXT_TOKENS}. Matches the reference pipeline, which raises rather than truncating."
            )
        cos, sin = create_rope_tensors(1, n_text, None, self._enc_head_dim, self._enc_rope_theta, self._enc_mrope)
        tt_ids = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device)
        taps = self.encoder.forward(
            tt_ids,
            attention_mask=None,
            pos_embeds=(bf16_tensor(cos, device=self.mesh_device), bf16_tensor(sin, device=self.mesh_device)),
        )
        # taps: 13 x [1, n_text, 4096] replicated on TP; interleave to [1, n_text, 53248]
        taps_t = [tensor.to_torch(t, mesh_axes=[None, None, None]) for t in taps]
        feats = torch.stack(taps_t, dim=0).permute(1, 2, 3, 0).reshape(1, n_text, -1).to(torch.bfloat16)
        return feats, n_text

    def _seq_dev(self, t: torch.Tensor, seq_dim: int) -> ttnn.Tensor:
        """bf16 device tensor: shard on the SP axis along ``seq_dim`` when SP>1, else replicate."""
        if self.sp_factor > 1:
            return bf16_tensor(t, device=self.mesh_device, mesh_axis=self.sp_axis, shard_dim=seq_dim)
        return bf16_tensor(t, device=self.mesh_device)

    def _host_seq(self, t: torch.Tensor, seq_dim: int) -> ttnn.Tensor:
        """bf16 HOST ttnn tensor with the SP shard/replicate mapping baked in (the tracer
        moves it onto its captured device slots without allocating new device buffers).
        Mirrors _seq_dev's placement so the model sees an identical layout on replay."""
        mesh_axes = [None] * t.ndim
        if self.sp_factor > 1:
            mesh_axes[seq_dim] = self.sp_axis
        return tensor.from_torch(
            t.to(torch.bfloat16),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_axes=mesh_axes,
            on_host=True,
        )

    def _idx_dev(self, t: torch.Tensor, seq_dim: int) -> ttnn.Tensor:
        """uint32 device tensor (embedding index): shard on the SP axis when SP>1, else replicate."""
        if self.sp_factor > 1:
            mesh_axes = [self.sp_axis if d == seq_dim else None for d in range(t.ndim)]
            return tensor.from_torch(t, device=self.mesh_device, dtype=ttnn.uint32, mesh_axes=mesh_axes)
        return tensor.from_torch(t, device=self.mesh_device, dtype=ttnn.uint32)

    def _branch(self, n_pre, num_img, grid_h, grid_w, llm_real):
        cfg = self.config
        seq = n_pre + num_img
        # SP shards the sequence: pad to a k_chunk*sp_factor multiple so each per-device
        # shard is tile-aligned (ring SDPA requirement), then shard every per-token tensor
        # on the SP axis. Padding is appended after the image tokens (mask/indicator 0), so
        # it contributes nothing and is sliced off the velocity output. No-op when SP==1.
        padded_len = _sp_padded_len(seq, self.sp_factor)
        pad = padded_len - seq
        ind = torch.full((1, seq), OUTPUT_IMAGE_INDICATOR, dtype=torch.long)
        if n_pre:
            ind[:, :n_pre] = LLM_TOKEN_INDICATOR
        llm = torch.zeros(1, seq, cfg.llm_features_dim, dtype=torch.bfloat16)
        if n_pre and llm_real is not None:
            llm[:, :n_pre] = llm_real
        pos = torch.zeros(1, seq, 3, dtype=torch.long)
        if n_pre:
            tp = torch.arange(n_pre)
            pos[:, :n_pre] = torch.stack([tp, tp, tp], dim=1)
        hh = torch.arange(grid_h).repeat_interleave(grid_w)
        ww = torch.arange(grid_w).repeat(grid_h)
        pos[:, n_pre:, 0] = IMAGE_POSITION_OFFSET
        pos[:, n_pre:, 1] = IMAGE_POSITION_OFFSET + hh
        pos[:, n_pre:, 2] = IMAGE_POSITION_OFFSET + ww
        rope = modeling_ideogram4.Ideogram4MRoPE(
            head_dim=cfg.emb_dim // cfg.num_heads, base=cfg.rope_theta, mrope_section=cfg.mrope_section
        )
        cos, sin = rope(pos)
        llm_mask = (ind == LLM_TOKEN_INDICATOR).float().unsqueeze(-1)  # [1, seq, 1]
        img_mask = (ind == OUTPUT_IMAGE_INDICATOR).float().unsqueeze(-1)
        idx = (ind == OUTPUT_IMAGE_INDICATOR).to(torch.int32)  # [1, seq]
        cos4 = cos.unsqueeze(1).to(torch.bfloat16)  # [1, 1, seq, head_dim]
        sin4 = sin.unsqueeze(1).to(torch.bfloat16)
        if pad:
            llm = torch.nn.functional.pad(llm, (0, 0, 0, pad))
            cos4 = torch.nn.functional.pad(cos4, (0, 0, 0, pad))
            sin4 = torch.nn.functional.pad(sin4, (0, 0, 0, pad))
            llm_mask = torch.nn.functional.pad(llm_mask, (0, 0, 0, pad))
            img_mask = torch.nn.functional.pad(img_mask, (0, 0, 0, pad))
            idx = torch.nn.functional.pad(idx, (0, pad))
        return dict(
            llm=self._seq_dev(llm, 1),
            cos=self._seq_dev(cos4, 2),
            sin=self._seq_dev(sin4, 2),
            llm_mask=self._seq_dev(llm_mask, 1),
            img_mask=self._seq_dev(img_mask, 1),
            idx=self._idx_dev(idx, 1),
            seq=seq,
            padded_len=padded_len,
            n_pre=n_pre,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        *,
        height: int = 512,
        width: int = 512,
        preset: str = "V4_TURBO_12",
        seed: int = 1234,
        guidance_scale: float | None = None,
    ):
        import time as _time

        from loguru import logger as _lg

        cfg = self.config
        dev = self.mesh_device
        grid_h, grid_w = height // (self.patch * self.ae), width // (self.patch * self.ae)
        num_img = grid_h * grid_w
        torch.manual_seed(seed)
        self.timings = {}

        _t0 = _time.perf_counter()
        llm_real, n_text = self._encode(prompt)
        ttnn.synchronize_device(dev)
        self.timings["encode"] = _time.perf_counter() - _t0
        cond_b = self._branch(n_text, num_img, grid_h, grid_w, llm_real)
        uncond_b = self._branch(0, num_img, grid_h, grid_w, None)

        sampler = Ideogram4Sampler.from_preset(preset, height=height, width=width)
        z = torch.randn(1, num_img, cfg.in_channels, dtype=torch.float32)

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
            start = br["seq"] - num_img  # image-token slice start (drops text prefix + SP pad)
            return ttnn.slice(out, [0, start, 0], [1, br["seq"], cfg.in_channels])

        # The ENTIRE per-step computation runs in a SINGLE trace: cond forward, uncond
        # forward, the two SP-axis velocity gathers, the asymmetric-CFG blend and the Euler
        # update. Capturing all of it in one trace is what makes tracing CORRECT here: the
        # Tracer runs a prep pass that allocates every persistent buffer (both branches'
        # ring-SDPA / all-gather buffers, the blend/Euler temporaries) BEFORE capture, so no
        # device buffer is allocated while a trace is active ("Allocating device buffers is
        # unsafe ... may be corrupted once a trace is executed"). The only per-step inputs are
        # HOST tensors (x_cond, x_uncond, t_sin) plus two 1-element device scalars (gw, s-t);
        # the Tracer copies host inputs into the captured slots, allocating nothing new.
        # x_cond/x_uncond already carry the running latent z (built on host from the previous
        # z), and the trace returns the next z (device) which is read back to host each step.
        cond_pad = cond_b["padded_len"] - cond_b["seq"]
        uncond_pad = uncond_b["padded_len"] - uncond_b["seq"]

        def step_fn(x_cond, x_uncond, t_sin, z_in, gw, one_minus_gw, smt):
            # t_sin feeds both branches' adaln; share the same per-step embedding.
            cond_b["t_sin"] = t_sin
            uncond_b["t_sin"] = t_sin
            v_cond = _branch_velocity(self.cond, cond_b, x_cond)
            v_uncond = _branch_velocity(self.uncond, uncond_b, x_uncond)
            # Asymmetric CFG: v = gw*v_cond + (1-gw)*v_uncond, scalars as 1-element device
            # tensors (broadcast multiply) so they can be updated per step inside the trace.
            v = v_cond * gw + v_uncond * one_minus_gw
            return z_in + v * smt  # Euler update -> next z [1, num_img, in_ch]

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
            # bf16 HOST tensor replicated across the mesh (for t_sin / z_in, which the model
            # consumes replicated, not SP-sharded).
            return tensor.from_torch(
                t.to(torch.bfloat16),
                device=dev,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_axes=[None] * t.ndim,
                on_host=True,
            )

        def _x_inputs(z_host):
            """Build the cond/uncond image-latent inputs on HOST from the current z."""
            pos_x = torch.zeros(1, n_text + num_img, cfg.in_channels, dtype=torch.bfloat16)
            pos_x[:, n_text:] = z_host.to(torch.bfloat16)
            if cond_pad:
                pos_x = torch.nn.functional.pad(pos_x, (0, 0, 0, cond_pad))
            unc_x = z_host.to(torch.bfloat16)
            if uncond_pad:
                unc_x = torch.nn.functional.pad(unc_x, (0, 0, 0, uncond_pad))
            # Shard on the SP axis (host tensors; the tracer moves them onto its captured
            # device slots). _seq_dev shards-or-replicates exactly as the model expects.
            return self._host_seq(pos_x, 1), self._host_seq(unc_x, 1)

        _td = _time.perf_counter()
        self.step_trace = []
        z_host = z.clone()
        steps = list(reversed(range(sampler.num_steps)))
        for n, i in enumerate(steps):
            t_val, s_val = sampler.times_for_step(i)
            gw = sampler.guidance_weight(i) if guidance_scale is None else guidance_scale
            smt = s_val - t_val
            t_sin = Ideogram4Transformer.sinusoidal_embedding(torch.tensor([t_val]), cfg.emb_dim).unsqueeze(1)
            # All per-step inputs are HOST tensors: on capture the Tracer moves them onto the
            # captured device slots; on replay it copies host->slot in place. Nothing is
            # allocated on device after capture, so the active trace cannot corrupt them.
            x_cond_h, x_uncond_h = _x_inputs(z_host)
            args = (
                x_cond_h,
                x_uncond_h,
                _host_repl(t_sin),
                _host_repl(z_host),
                _scalar_host(gw),
                _scalar_host(1.0 - gw),
                _scalar_host(smt),
            )
            if self._step_tracer is None:
                self._step_tracer = Tracer(step_fn, device=dev, clone_prep_inputs=False)
                # ttnn.zeros host-writes for the 0-element ring-SDPA "empty joint" tensor,
                # which is illegal during capture; substitute a persistent zeros constant.
                with _patch_zeros_for_trace(dev):
                    z_out = self._step_tracer(*args)
            else:
                z_out = self._step_tracer(*args)

            z_host = tensor.to_torch(z_out, mesh_axes=[None, None, None]).float()
            self.step_trace.append((t_val, s_val, gw, 0.0, float(z_host.std()), 0.0))
            if getattr(self, "_verbose_steps", False):
                _lg.info(
                    f"  step {n + 1}/{sampler.num_steps}: t={t_val:.4f} s={s_val:.4f} gw={gw:.0f} "
                    f"z_std={z_host.std():.4f}"
                )
        ttnn.synchronize_device(dev)
        self.timings["denoise"] = _time.perf_counter() - _td
        self.timings["denoise_per_step"] = self.timings["denoise"] / sampler.num_steps

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

        _tv = _time.perf_counter()
        decoded = self.decode_stage.decode(bf16_tensor(z_host, device=dev), grid_h=grid_h, grid_w=grid_w)
        img = Ideogram4DecodeStage.to_images(decoded)[0].cpu().numpy()
        self.timings["decode"] = _time.perf_counter() - _tv
        self.timings["total"] = _time.perf_counter() - _t0
        _lg.info(
            f"[latency {height}px {preset}] total={self.timings['total']:.2f}s | encode={self.timings['encode']:.2f}s "
            f"| denoise={self.timings['denoise']:.2f}s ({self.timings['denoise_per_step']*1000:.0f}ms/step x{sampler.num_steps}) "
            f"| decode={self.timings['decode']:.2f}s"
        )
        return img
