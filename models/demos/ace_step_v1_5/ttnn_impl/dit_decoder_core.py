from __future__ import annotations

import math
import os
import traceback
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


def _ace_step_env_truthy(name: str) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on", "y")


def _ace_step_logical_seq_len_dim2(t) -> int:
    """
    Sequence length on dim=2 for tensors shaped [B, 1, S, ...] (e.g. cross encoder states).

    Prefer TTNN ``logical_shape`` so we do not treat TILE storage padding as extra keys.
    """
    fn = getattr(t, "logical_shape", None)
    if callable(fn):
        try:
            s = fn()
            if s is not None and len(s) >= 3:
                return int(s[2])
        except Exception:
            pass
    try:
        ls = getattr(t, "logical_shape", None)
        if ls is not None and not callable(ls) and len(ls) >= 3:
            return int(ls[2])
    except Exception:
        pass
    return int(t.shape[2])


def _ace_step_attn_trace_print(prefix: Optional[str]) -> bool:
    """
    When ACE_STEP_DEBUG_ATTN_TRACE=1, print per-stage tensor stats from attention.
    Optional ACE_STEP_DEBUG_ATTN_TRACE_LAYER=0 restricts to that layer index (substring layer{N}. in prefix).
    """
    if not _ace_step_env_truthy("ACE_STEP_DEBUG_ATTN_TRACE"):
        return False
    layer_filt = (os.environ.get("ACE_STEP_DEBUG_ATTN_TRACE_LAYER") or "").strip()
    if not layer_filt:
        return True
    needle = f"layer{layer_filt}."
    return prefix is not None and needle in prefix


def _ace_step_log_ttnn_tensor(tag: str, t, *, ttnn) -> None:
    """Host readback: use only when ACE_STEP_DEBUG_ATTN_TRACE is enabled (expensive)."""
    try:
        import torch

        x = ttnn.to_torch(t).detach().float().reshape(-1)
        m = torch.isfinite(x)
        if not bool(m.any()):
            print(f"[ace_step_v1_5][attn_trace][ttnn] {tag} shape={tuple(t.shape)} NO_FINITE", flush=True)
            return
        xf = x[m]
        print(
            f"[ace_step_v1_5][attn_trace][ttnn] {tag} shape={tuple(t.shape)} "
            f"min={float(xf.min()):.6g} max={float(xf.max()):.6g} "
            f"mean={float(xf.mean()):.6g} std={float(xf.std(unbiased=False)):.6g} n={int(xf.numel())}",
            flush=True,
        )
    except Exception as ex:
        print(f"[ace_step_v1_5][attn_trace][ttnn] {tag} log_failed={ex!r}", flush=True)


from ._ttnn import get_ttnn


def _require_ttnn():
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required for ace_step_v1_5.ttnn_impl")
    return ttnn


def _to_numpy_host_array(x):
    """
    Normalize a host tensor/array to a numpy array.

    Supports:
    - numpy arrays (returned as-is)
    - CPU torch tensors (converted to numpy; BF16 is upcast to FP32)
    """
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch  # optional dependency for some call sites

        if isinstance(x, torch.Tensor):
            if x.dtype == torch.bfloat16:
                x = x.to(torch.float32)
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


@dataclass(frozen=True)
class AceStepDecoderConfigTTNN:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    sliding_window: Optional[int]
    # Hugging Face config fields needed for parity features.
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    # Optional per-layer attention type selector used by HF ("full_attention" vs "sliding_attention").
    # When absent, all layers default to full attention.
    layer_types: Optional[Sequence[str]] = None


def _maybe_get(state_dict: dict, key: str) -> np.ndarray:
    if key not in state_dict:
        raise KeyError(f"Missing key in state_dict: {key}")
    return state_dict[key]


class TtTimestepEmbedding:
    """
    TTNN version of `TimestepEmbedding`, implemented as a precomputed lookup.

    We precompute (on host) the exact same sinusoidal `t_freq` for a fixed set
    of timesteps and keep the result device-resident. This avoids requiring
    sin/cos/exp kernels for bring-up.
    """

    def __init__(
        self,
        *,
        cfg: AceStepDecoderConfigTTNN,
        state_dict: dict,
        base_address: str,  # e.g. "time_embed" or "time_embed_r"
        mesh_device,
        timesteps_host: np.ndarray,  # shape [N]
        scale: float = 1000.0,
        dtype=None,
    ) -> None:
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.mesh_device = mesh_device
        self.hidden_size = int(cfg.hidden_size)
        self.scale = float(scale)

        self.in_channels = 256
        self.time_embed_dim = int(cfg.hidden_size)
        if timesteps_host.ndim != 1:
            raise ValueError("timesteps_host must be rank-1 [N]")
        self.num_steps = int(timesteps_host.shape[0])

        self.dtype = dtype or getattr(ttnn, "bfloat16", None) or getattr(ttnn, "float16", None)
        if self.dtype is None:
            raise RuntimeError("TTNN build missing a usable dtype (bfloat16/float16)")

        # Build sinusoidal embeddings on host (float32).
        t = timesteps_host.astype(np.float32) * self.scale  # [N]
        half = self.in_channels // 2
        freqs = np.exp((-math.log(10000.0)) * (np.arange(0, half, dtype=np.float32) / float(half)))  # [half]
        args = t[:, None] * freqs[None, :]  # [N, half]
        emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)  # [N, in_channels]

        # Load weights (host arrays) and transfer once.
        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None

        def as_w(key: str):
            return ttnn.as_tensor(
                _maybe_get(state_dict, f"{base_address}.{key}.weight"),
                device=mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        def as_b(key: str):
            b = _maybe_get(state_dict, f"{base_address}.{key}.bias")
            return ttnn.as_tensor(
                b.reshape(1, 1, 1, -1),
                device=mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        self.w1, self.b1 = as_w("linear_1"), as_b("linear_1")
        self.w2, self.b2 = as_w("linear_2"), as_b("linear_2")
        self.wt, self.bt = as_w("time_proj"), as_b("time_proj")

        self.t_freq_table = ttnn.as_tensor(
            emb.reshape(self.num_steps, 1, 1, self.in_channels),
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )

    def __call__(self, timestep_index: int):
        ttnn = self.ttnn
        if not (0 <= int(timestep_index) < self.num_steps):
            raise ValueError(f"timestep_index out of range: {timestep_index} not in [0,{self.num_steps})")

        # Slice out [1,1,1,in_channels]
        t_freq = ttnn.slice(self.t_freq_table, (timestep_index, 0, 0, 0), (timestep_index + 1, 1, 1, self.in_channels))
        temb = ttnn.linear(t_freq, self.w1, bias=self.b1, transpose_b=True)
        temb = ttnn.silu(temb) if hasattr(ttnn, "silu") else ttnn.gelu(temb)
        temb = ttnn.linear(temb, self.w2, bias=self.b2, transpose_b=True)

        # time_proj(act2(temb)) -> [1,1,1,6*D] -> reshape to [1,6,1,D] then [1,6,D]
        h = ttnn.silu(temb) if hasattr(ttnn, "silu") else ttnn.gelu(temb)
        tp = ttnn.linear(h, self.wt, bias=self.bt, transpose_b=True)
        d = self.time_embed_dim
        tp = ttnn.reshape(tp, (1, 6, 1, d))
        tp = ttnn.reshape(tp, (1, 6, d))
        temb = ttnn.reshape(temb, (1, d))
        return temb, tp

    def from_timestep_value(self, timestep: float):
        """
        HF-parity path: compute sinusoidal embedding for an arbitrary timestep value.

        This matches HF's `TimestepEmbedding` contract of generating sin/cos at runtime, but
        it uploads a tiny `[1,1,1,256]` tensor each call (host -> device).
        """
        ttnn = self.ttnn
        try:
            import torch
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("from_timestep_value() requires torch to be installed") from e

        # Build the exact same t_freq as in __init__ precompute, but for a single timestep.
        t = torch.tensor([float(timestep) * self.scale], dtype=torch.float32)  # [1]
        half = self.in_channels // 2
        freqs = torch.exp((-math.log(10000.0)) * (torch.arange(0, half, dtype=torch.float32) / float(half)))  # [half]
        args = t[:, None] * freqs[None, :]  # [1, half]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [1, in_channels]
        emb = emb.reshape(1, 1, 1, self.in_channels)

        # Upload and run the same MLP projection as the lookup-table path.
        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        t_freq = ttnn.from_torch(emb, device=self.mesh_device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        temb = ttnn.linear(t_freq, self.w1, bias=self.b1, transpose_b=True)
        temb = ttnn.silu(temb) if hasattr(ttnn, "silu") else ttnn.gelu(temb)
        temb = ttnn.linear(temb, self.w2, bias=self.b2, transpose_b=True)
        h = ttnn.silu(temb) if hasattr(ttnn, "silu") else ttnn.gelu(temb)
        tp = ttnn.linear(h, self.wt, bias=self.bt, transpose_b=True)
        d = self.time_embed_dim
        tp = ttnn.reshape(tp, (1, 6, 1, d))
        tp = ttnn.reshape(tp, (1, 6, d))
        temb = ttnn.reshape(temb, (1, d))
        # Keep t_freq allocated only for the duration of this call.
        if hasattr(ttnn, "deallocate"):
            try:
                ttnn.deallocate(t_freq)
            except Exception:
                pass
        return temb, tp


class TtHfRotaryEmbedding:
    """
    HF/Qwen3-parity RoPE cache for `ttnn.experimental.rotary_embedding` on [B,H,S,Dh] tensors.

    Uses the same `transformers.Qwen3RotaryEmbedding` path as `TorchAceStepDiTCoreRef._rope`
    (not `get_rot_mats_hf` / generic Llama-style precompute), so PCC vs the torch reference
    does not diverge on rope frequencies.
    """

    def __init__(
        self,
        *,
        mesh_device,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        dtype=None,
    ):
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.mesh_device = mesh_device
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        self.rope_theta = float(rope_theta)
        self.dtype = dtype or getattr(ttnn, "bfloat16", None) or getattr(ttnn, "float16", None)
        if self.dtype is None:
            raise RuntimeError("TTNN build missing a usable dtype (bfloat16/float16)")

        if not hasattr(ttnn, "experimental") or not hasattr(ttnn.experimental, "rotary_embedding"):
            raise RuntimeError("TTNN build missing ttnn.experimental.rotary_embedding (HF-style RoPE).")

        # Build cos/sin on host with HF Qwen3 (matches torch_ref/dit_decoder_core.py `_rope`).
        self._rope_source = "qwen3_hf"
        try:
            import torch
            from transformers import Qwen3Config
            from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

            d_model = int(hidden_size)
            h = int(num_attention_heads)
            kv = int(num_key_value_heads)
            dh = int(head_dim)
            m = int(max_seq_len)
            qc = Qwen3Config(
                vocab_size=8192,
                hidden_size=d_model,
                intermediate_size=max(d_model, 512),
                num_hidden_layers=1,
                num_attention_heads=h,
                num_key_value_heads=kv,
                head_dim=dh,
                max_position_embeddings=m,
                rope_theta=float(rope_theta),
            )
            rope = Qwen3RotaryEmbedding(qc)
            dummy = torch.zeros(1, m, d_model, dtype=torch.float32)
            pos = torch.arange(m, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                cos_t, sin_t = rope(dummy, pos)
            # [1, S, Dh] -> [1, 1, S, Dh] for ttnn.experimental.rotary_embedding / broadcast to [B,H,S,Dh]
            cos_hf = cos_t.unsqueeze(1).contiguous()
            sin_hf = sin_t.unsqueeze(1).contiguous()
        except Exception as e:
            # Fallback: legacy path (may not match Qwen3 exactly). Do not swallow errors silently.
            self._rope_source = "get_rot_mats_hf_fallback"
            if _ace_step_env_truthy("ACE_STEP_DEBUG_ROPE") or _ace_step_env_truthy("ACE_STEP_DEBUG_ATTN_TRACE"):
                print(
                    "[ace_step_v1_5][rope] Qwen3RotaryEmbedding cache build failed; using get_rot_mats_hf. Error:",
                    repr(e),
                    flush=True,
                )
                traceback.print_exc()
            from models.tt_transformers.tt.rope import get_rot_mats_hf

            cos_hf, sin_hf = get_rot_mats_hf(
                head_dim=self.head_dim,
                device=self.mesh_device,
                seq_len=self.max_seq_len,
                theta=self.rope_theta,
                rope_scaling=None,
                datatype=self.dtype,
            )
            self.cos_cached = cos_hf
            self.sin_cached = sin_hf
            return

        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        self.cos_cached = ttnn.from_torch(
            cos_hf,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.dtype,
            mesh_mapper=mapper,
            memory_config=mem,
        )
        self.sin_cached = ttnn.from_torch(
            sin_hf,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.dtype,
            mesh_mapper=mapper,
            memory_config=mem,
        )

    def __call__(self, x, *, token_idx: Optional[int] = None):
        ttnn = self.ttnn
        # `rotary_embedding` uses `x.shape[2]` as seq_len and slices cos/sin caches internally.
        return ttnn.experimental.rotary_embedding(
            x,
            self.cos_cached,
            self.sin_cached,
            token_idx,
            memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None),
        )


def _ace_step_cross_attention_decomposed(
    ttnn,
    *,
    q,
    k,
    v,
    b: int,
    h: int,
    s_q: int,
    scale: float,
    additive_mask_b1qk,
    activations_dtype,
):
    """
    Cross-attention without fused SDPA: matches torch_ref `dit_decoder_core._sdpa`
    (QK^T * scale, softmax on keys, @ V) while allowing additive padding / encoder masks.

    torch_ref runs attention matmuls and softmax in FP32, then casts context back to BF16.
    When `ttnn.float32` is available, this path does the same; otherwise it stays in
    `activations_dtype` for the full decomposed op (tile tail-mask still applied when needed).

    Shapes:
      q,k,v: [B, H, S_q or S_k, Dh] TILE
      additive_mask_b1qk: optional [B, 1, S_q, S_k] added to pre-softmax logits (broadcast over heads).
    """
    s_k = int(k.shape[2])
    dh = int(q.shape[3])
    bh = int(b * h)
    mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    mem_kw = dict(memory_config=mem) if mem is not None else {}
    fp32 = getattr(ttnn, "float32", None)
    use_fp32 = fp32 is not None

    qf = ttnn.reshape(q, (bh, s_q, dh))
    kf = ttnn.reshape(k, (bh, s_k, dh))
    vf = ttnn.reshape(v, (bh, s_k, dh))
    if use_fp32:
        qf = ttnn.typecast(qf, dtype=fp32)
        kf = ttnn.typecast(kf, dtype=fp32)
        vf = ttnn.typecast(vf, dtype=fp32)

    kt = ttnn.permute(kf, (0, 2, 1))
    scores = ttnn.matmul(qf, kt, **mem_kw)
    scores = ttnn.multiply(scores, float(scale))
    if additive_mask_b1qk is not None:
        m = ttnn.repeat(additive_mask_b1qk, (1, int(h), 1, 1))
        m = ttnn.reshape(m, (bh, s_q, s_k))
        if use_fp32:
            m = ttnn.typecast(m, dtype=fp32)
        scores = ttnn.add(scores, m, **mem_kw)
    attn = ttnn.softmax(scores, dim=-1, **mem_kw)
    ctx = ttnn.matmul(attn, vf, **mem_kw)
    if use_fp32 and activations_dtype is not None:
        ctx = ttnn.typecast(ctx, dtype=activations_dtype)
    return ttnn.reshape(ctx, (b, h, s_q, dh))


class TtAceStepAttentionSDPA:
    """
    AceStep DiT attention on TTNN.

    Self-attention uses ``ttnn.transformer.scaled_dot_product_attention``. Cross-attention uses a
    decomposed matmul + softmax + matmul path on device (no CPU PyTorch fallback).

    Shapes:
      - hidden_states: [B, 1, S, D]
      - encoder_hidden_states (cross): [B, 1, S_enc, D]
    """

    def __init__(
        self,
        *,
        cfg: AceStepDecoderConfigTTNN,
        state_dict: dict,
        base_address: str,
        mesh_device,
        dtype=None,
        rotary_embedding: Optional[TtHfRotaryEmbedding] = None,
    ):
        ttnn = _require_ttnn()
        transformer = getattr(ttnn, "transformer", None)
        sdpa = getattr(transformer, "scaled_dot_product_attention", None) if transformer is not None else None
        if sdpa is None:
            raise RuntimeError("TTNN build missing ttnn.transformer.scaled_dot_product_attention")

        self.ttnn = ttnn
        self._sdpa = sdpa
        self.mesh_device = mesh_device
        self.dtype = dtype or getattr(ttnn, "bfloat16", None) or getattr(ttnn, "float16", None)
        if self.dtype is None:
            raise RuntimeError("TTNN build missing a usable dtype (bfloat16/float16)")

        self.d_model = int(cfg.hidden_size)
        self.n_heads = int(cfg.num_attention_heads)
        self.n_kv = int(cfg.num_key_value_heads)
        self.d_head = int(cfg.head_dim)
        self.scale = 1.0 / math.sqrt(float(self.d_head))
        self._rotary = rotary_embedding

        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None

        def as_w(suffix: str):
            return ttnn.as_tensor(
                _maybe_get(state_dict, f"{base_address}.{suffix}.weight"),
                device=mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        def as_b(suffix: str):
            # Biases are optional (AceStepConfig.attention_bias defaults to False).
            key = f"{base_address}.{suffix}.bias"
            b = state_dict.get(key, None)
            if b is None:
                return None
            return ttnn.as_tensor(
                b.reshape(1, 1, 1, -1),
                device=mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        self.wq, self.bq = as_w("q_proj"), as_b("q_proj")
        # Build a fused KV projection to guarantee K and V have identical (logical and padded) sequence length.
        # Some TTNN builds can produce mismatched seq lengths when running separate K and V linears.
        wk_host = _to_numpy_host_array(_maybe_get(state_dict, f"{base_address}.k_proj.weight"))
        wv_host = _to_numpy_host_array(_maybe_get(state_dict, f"{base_address}.v_proj.weight"))
        wkv_host = np.concatenate([wk_host, wv_host], axis=0)
        self.wkv = ttnn.as_tensor(
            wkv_host,
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )

        bk_host = state_dict.get(f"{base_address}.k_proj.bias", None)
        bv_host = state_dict.get(f"{base_address}.v_proj.bias", None)
        if bk_host is not None and bv_host is not None:
            bk_np = _to_numpy_host_array(bk_host)
            bv_np = _to_numpy_host_array(bv_host)
            bkv_host = np.concatenate([bk_np, bv_np], axis=0).reshape(1, 1, 1, -1)
            self.bkv = ttnn.as_tensor(
                bkv_host,
                device=mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
            )
        else:
            self.bkv = None

        self.wo, self.bo = as_w("o_proj"), as_b("o_proj")

        # Per-head RMSNorm weights (shape [Dh]).
        qn = _maybe_get(state_dict, f"{base_address}.q_norm.weight")
        kn = _maybe_get(state_dict, f"{base_address}.k_norm.weight")
        self.q_norm_w = ttnn.as_tensor(
            qn,
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        self.k_norm_w = ttnn.as_tensor(
            kn,
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        self.eps = float(cfg.rms_norm_eps)

    def __call__(
        self,
        hidden_states,
        *,
        encoder_hidden_states=None,
        is_causal: bool = False,
        sliding_window_size: Optional[int] = None,
        attn_mask=None,
        debug: Optional[dict] = None,
        debug_prefix: str = "",
    ):
        ttnn = self.ttnn
        _trace = _ace_step_attn_trace_print(debug_prefix)
        x = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)  # [B,1,S,D]
        q = ttnn.linear(x, self.wq, bias=self.bq, transpose_b=True)
        if encoder_hidden_states is None:
            kv = ttnn.linear(x, self.wkv, bias=self.bkv, transpose_b=True)
        else:
            enc = ttnn.to_layout(encoder_hidden_states, ttnn.TILE_LAYOUT)
            kv = ttnn.linear(enc, self.wkv, bias=self.bkv, transpose_b=True)

        B = int(q.shape[0])
        S = int(q.shape[2])
        H = self.n_heads
        Dh = self.d_head
        if _trace:
            print(
                f"[ace_step_v1_5][attn_trace][ttnn] {debug_prefix}enter "
                f"B={B} S_q_linear={S} H={H} Dh={Dh} cross={encoder_hidden_states is not None} "
                f"scale={self.scale:.6g} attn_mask={'set' if attn_mask is not None else 'None'}",
                flush=True,
            )
            _ace_step_log_ttnn_tensor(f"{debug_prefix}q_after_linear_B1SD", q, ttnn=ttnn)

        # Split fused KV: kv is [B,1,S,2*(n_kv*Dh)]
        kv_dim = int(self.n_kv * Dh)
        k = ttnn.slice(kv, (0, 0, 0, 0), (B, 1, int(kv.shape[2]), kv_dim))
        v = ttnn.slice(kv, (0, 0, 0, kv_dim), (B, 1, int(kv.shape[2]), 2 * kv_dim))
        if hasattr(ttnn, "deallocate"):
            try:
                ttnn.deallocate(kv)
            except Exception:
                pass

        def _seq_len_padded(x) -> int:
            # SDPA validation uses padded shapes. Prefer padded_shape() when available.
            ps = getattr(x, "padded_shape", None)
            try:
                if callable(ps):
                    return int(ps()[2])
                if ps is not None:
                    return int(ps[2])
            except Exception:
                pass
            return int(x.shape[2])

        def _seq_len_logical(x) -> int:
            return int(x.shape[2])

        def _ceil_tile(x: int) -> int:
            return ((int(x) + 31) // 32) * 32

        def _pad_seq_to_rank4(x, target_s: int):
            logical_s = _seq_len_logical(x)
            padded_s = _seq_len_padded(x)
            if logical_s == target_s and padded_s == target_s:
                return x
            if logical_s > target_s:
                return ttnn.slice(x, (0, 0, 0, 0), (int(x.shape[0]), int(x.shape[1]), target_s, int(x.shape[3])))

            pad = target_s - logical_s
            x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x_rm = ttnn.pad(x_rm, padding=((0, 0), (0, 0), (0, pad), (0, 0)), value=0.0)
            # `pad` may only update storage padding. Reshape makes the new sequence length logical too.
            x_rm = ttnn.reshape(x_rm, (int(x.shape[0]), int(x.shape[1]), target_s, int(x.shape[3])))
            return ttnn.to_layout(x_rm, ttnn.TILE_LAYOUT)

        def _pad_seq_dim2_bh_sd(x, target_s: int):
            """Pad/slice sequence dim (index 2) for tensors shaped [B, H, S, Dh]."""
            b0, h0, s0, d0 = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
            if s0 == target_s:
                return x
            if s0 > target_s:
                return ttnn.slice(x, (0, 0, 0, 0), (b0, h0, target_s, d0))
            pad = target_s - s0
            x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x_rm = ttnn.pad(x_rm, padding=((0, 0), (0, 0), (0, pad), (0, 0)), value=0.0)
            x_rm = ttnn.reshape(x_rm, (b0, h0, target_s, d0))
            return ttnn.to_layout(x_rm, ttnn.TILE_LAYOUT)

        is_self_attn = encoder_hidden_states is None

        # Safety: TTNN SDPA requires K/V to have the same logical and padded sequence length.
        # Some builds can produce mismatched seq lengths between K and V.
        S_k_raw = max(_seq_len_logical(k), _seq_len_padded(k))
        S_v_raw = max(_seq_len_logical(v), _seq_len_padded(v))
        if S_k_raw != S_v_raw:
            target = _ceil_tile(max(S_k_raw, S_v_raw))

            k = _pad_seq_to_rank4(k, target)
            v = _pad_seq_to_rank4(v, target)

        # Q: [B,1,S,H*Dh] -> [B,H,S,Dh]
        q = ttnn.reshape(q, (B, 1, S, H, Dh))
        q = ttnn.permute(q, (0, 3, 2, 4, 1))
        q = ttnn.reshape(q, (B, H, S, Dh))
        if _trace:
            _ace_step_log_ttnn_tensor(f"{debug_prefix}q_after_reshape_BHSD", q, ttnn=ttnn)

        if debug is not None and debug.get("enabled", False):
            debug[f"{debug_prefix}q_lin"] = q

        # K/V use num_key_value_heads; we expand to num_attention_heads by repeat if needed.
        S_k = int(k.shape[2])
        kv_h = self.n_kv
        k = ttnn.reshape(k, (B, 1, S_k, kv_h, Dh))
        v = ttnn.reshape(v, (B, 1, S_k, kv_h, Dh))
        k = ttnn.permute(k, (0, 3, 2, 4, 1))
        v = ttnn.permute(v, (0, 3, 2, 4, 1))
        k = ttnn.reshape(k, (B, kv_h, S_k, Dh))
        v = ttnn.reshape(v, (B, kv_h, S_k, Dh))
        if debug is not None and debug.get("enabled", False):
            debug[f"{debug_prefix}k_lin"] = k
            debug[f"{debug_prefix}v_lin"] = v

        if kv_h != H:
            # Grouped-query attention: repeat kv heads to match q heads.
            if H % kv_h != 0:
                raise ValueError(f"num_attention_heads {H} not divisible by num_key_value_heads {kv_h}")
            rep = H // kv_h
            # HF GQA semantics are `repeat_interleave` on the kv-head dimension:
            # kv0 -> q0,q1 ; kv1 -> q2,q3 ; ...
            if hasattr(ttnn, "repeat_interleave"):
                k = ttnn.repeat_interleave(k, rep, dim=1)
                v = ttnn.repeat_interleave(v, rep, dim=1)
            else:
                # Fallback: concat each head `rep` times (interleaved).
                ks = []
                vs = []
                for hi in range(int(kv_h)):
                    k_h = ttnn.slice(k, (0, hi, 0, 0), (B, hi + 1, S_k, Dh))
                    v_h = ttnn.slice(v, (0, hi, 0, 0), (B, hi + 1, S_k, Dh))
                    for _ in range(int(rep)):
                        ks.append(k_h)
                        vs.append(v_h)
                k = ttnn.concat(ks, dim=1) if hasattr(ttnn, "concat") else ttnn.concatenate(ks, dim=1)
                v = ttnn.concat(vs, dim=1) if hasattr(ttnn, "concat") else ttnn.concatenate(vs, dim=1)

        # Cross-attn: **do not** tile-pad K/V before head-dim RMSNorm. HF norms the true encoder
        # length (e.g. 258); padding to 288 first corrupts K statistics and attention. Self-attn:
        # padding K/V before RoPE would extend the sequence seen by RoPE — keep norm+RoPE at true S,
        # then pad for SDPA only.
        tile_target_pre_norm = 0

        if _trace:
            _cross_kv_note = "n/a(self)" if is_self_attn else "n/a(cross, SDPA pad only after norm)"
            print(
                f"[ace_step_v1_5][attn_trace][ttnn] {debug_prefix}kv_pre_norm "
                f"S_q={S} S_k_logical={_seq_len_logical(k)} S_k_padded={_seq_len_padded(k)} "
                f"tile_target_pre_norm={_cross_kv_note} "
                f"k_shape={tuple(k.shape)} v_shape={tuple(v.shape)}",
                flush=True,
            )
            _ace_step_log_ttnn_tensor(f"{debug_prefix}k_pre_rmsnorm", k, ttnn=ttnn)
            _ace_step_log_ttnn_tensor(f"{debug_prefix}v_pre_rmsnorm", v, ttnn=ttnn)

        # Optional debug: export ACE_STEP_DEBUG_SDPA=1 to print padded seq lens
        if os.environ.get("ACE_STEP_DEBUG_SDPA"):
            try:
                print(
                    "[ace_step_v1_5][sdpa] "
                    f"k_shape={tuple(k.shape)} v_shape={tuple(v.shape)} "
                    f"k_padded={_seq_len_padded(k)} v_padded={_seq_len_padded(v)} "
                    f"tile_target_pre_norm={tile_target_pre_norm}",
                    flush=True,
                )
            except Exception:
                pass

        # Head-dim RMSNorm on q and k.
        q = ttnn.rms_norm(
            q, weight=self.q_norm_w, epsilon=self.eps, memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        )
        k = ttnn.rms_norm(
            k, weight=self.k_norm_w, epsilon=self.eps, memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        )
        if debug is not None and debug.get("enabled", False):
            debug[f"{debug_prefix}q_norm"] = q
            debug[f"{debug_prefix}k_norm"] = k

        if _trace:
            _ace_step_log_ttnn_tensor(f"{debug_prefix}q_after_rmsnorm", q, ttnn=ttnn)
            _ace_step_log_ttnn_tensor(f"{debug_prefix}k_after_rmsnorm", k, ttnn=ttnn)

        # HF parity: apply RoPE in self-attention only.
        if encoder_hidden_states is None and self._rotary is not None:
            if _trace:
                print(
                    f"[ace_step_v1_5][attn_trace][ttnn] {debug_prefix}rope "
                    f"source={getattr(self._rotary, '_rope_source', '?')}",
                    flush=True,
                )
            q = self._rotary(q)
            k = self._rotary(k)
            # RoPE may return Q/K with tile-padded sequence length (e.g. 64) while V stayed at the
            # logical S (63). Slice Q/K back to `S` (query length from the linear) so SDPA sees
            # matching Q/K/V lengths and matches HF/torch_ref.
            if is_self_attn:
                if int(q.shape[2]) > S:
                    q = ttnn.slice(q, (0, 0, 0, 0), (B, H, S, Dh))
                if int(k.shape[2]) > S:
                    k = ttnn.slice(k, (0, 0, 0, 0), (B, H, S, Dh))
                if int(v.shape[2]) > S:
                    v = ttnn.slice(v, (0, 0, 0, 0), (B, H, S, Dh))
            if _trace:
                _ace_step_log_ttnn_tensor(f"{debug_prefix}q_after_rope", q, ttnn=ttnn)
                _ace_step_log_ttnn_tensor(f"{debug_prefix}k_after_rope", k, ttnn=ttnn)
            if debug is not None and debug.get("enabled", False):
                debug[f"{debug_prefix}q_rope"] = q
                debug[f"{debug_prefix}k_rope"] = k

        # Self-attn: optional additive mask from the pipeline (patchified latents).
        # Cross-attn: match torch_ref `TorchAceStepDiTCoreRef` — **do not**
        # apply HF `encoder_attention_mask` (text padding). The reference attends over full S_enc from the
        # conditioned encoder tensor; applying prepare_condition masks here caused large PCC gaps vs ref.
        # We only mask keys **past** the encoder logical length (tile pad / storage tail, e.g. 258→288).
        sdpa_attn_mask = attn_mask if is_self_attn else None
        if not is_self_attn:
            s_enc_log = _ace_step_logical_seq_len_dim2(encoder_hidden_states)
            S_q0 = int(q.shape[2])
            W = int(k.shape[2])
            tgt_k = _ceil_tile(W)
            if tgt_k > W:
                k = _pad_seq_dim2_bh_sd(k, tgt_k)
                v = _pad_seq_dim2_bh_sd(v, tgt_k)
                W = tgt_k
            if W > s_enc_log:
                pad_np = np.zeros((B, 1, S_q0, W), dtype=np.float32)
                pad_np[:, :, :, s_enc_log:W] = -1e9
                mem_m = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
                mapper_m = (
                    ttnn.ReplicateTensorToMesh(self.mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
                )
                pad_m = ttnn.as_tensor(
                    pad_np,
                    device=self.mesh_device,
                    dtype=self.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mem_m,
                    mesh_mapper=mapper_m,
                )
                sdpa_attn_mask = pad_m if sdpa_attn_mask is None else ttnn.add(sdpa_attn_mask, pad_m)

        # Self-attn: pad Q/K/V together to a tile multiple for SDPA, and mask padded key columns so
        # softmax matches the unpadded reference (PyTorch/HF uses exact S).
        if is_self_attn:
            S_rope = int(q.shape[2])
            if int(k.shape[2]) != S_rope or int(v.shape[2]) != S_rope:
                raise ValueError(
                    f"Self-attn Q/K/V seq mismatch before SDPA pad: q={int(q.shape[2])} k={int(k.shape[2])} v={int(v.shape[2])}"
                )
            target_sdpa = _ceil_tile(S_rope)
            if target_sdpa > S_rope:
                q = _pad_seq_dim2_bh_sd(q, target_sdpa)
                k = _pad_seq_dim2_bh_sd(k, target_sdpa)
                v = _pad_seq_dim2_bh_sd(v, target_sdpa)
                # Additive mask (0 keep, -1e9 on invalid keys); SDPA adds this to QK (see ttnn sdpa.cpp).
                pad_np = np.zeros((B, 1, target_sdpa, target_sdpa), dtype=np.float32)
                pad_np[:, :, :, S_rope:] = -1e9
                mem_m = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
                mapper_m = (
                    ttnn.ReplicateTensorToMesh(self.mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
                )
                # Match Q/K/V dtype so SDPA uniform-dataformat paths accept the mask tensor.
                pad_m = ttnn.as_tensor(
                    pad_np,
                    device=self.mesh_device,
                    dtype=self.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mem_m,
                    mesh_mapper=mapper_m,
                )
                if sdpa_attn_mask is None:
                    sdpa_attn_mask = pad_m
                else:
                    sdpa_attn_mask = ttnn.add(sdpa_attn_mask, pad_m)

        # Cross-attn: decomposed matmul + softmax + matmul (no fused SDPA), with FP32 scores/softmax/@V
        # when `ttnn.float32` exists — mirrors torch_ref `_sdpa`. Tail-pad mask keeps tile-multiple K/V
        # aligned with logical encoder length (e.g. 258→288).
        if not is_self_attn:
            ctx = _ace_step_cross_attention_decomposed(
                ttnn,
                q=q,
                k=k,
                v=v,
                b=B,
                h=H,
                s_q=int(q.shape[2]),
                scale=self.scale,
                additive_mask_b1qk=sdpa_attn_mask,
                activations_dtype=self.dtype,
            )
        else:
            sdpa_kwargs = dict(attn_mask=sdpa_attn_mask, is_causal=is_causal, scale=self.scale)
            if sliding_window_size is not None:
                sdpa_kwargs["sliding_window_size"] = int(sliding_window_size)
            ctx = self._sdpa(q, k, v, **sdpa_kwargs)
        if _trace:
            _ace_step_log_ttnn_tensor(f"{debug_prefix}ctx_after_sdpa_BHSD", ctx, ttnn=ttnn)
        # [B,H,S_ctx,Dh] -> [B,1,S_ctx,H*Dh]. SDPA can return tile-aligned logical
        # sequence lengths, so reshape using the returned length and slice back to query S.
        S_ctx = int(ctx.shape[2])
        if _trace:
            print(
                f"[ace_step_v1_5][attn_trace][ttnn] {debug_prefix}sdpa_out "
                f"S_ctx={S_ctx} S_q_expected={S} slice_back={S_ctx != S}",
                flush=True,
            )
        ctx = ttnn.permute(ctx, (0, 2, 1, 3))
        ctx = ttnn.reshape(ctx, (B, 1, S_ctx, H * Dh))
        if S_ctx != S:
            ctx = ttnn.slice(ctx, (0, 0, 0, 0), (B, 1, S, H * Dh))
        out = ttnn.linear(ctx, self.wo, bias=self.bo, transpose_b=True)
        if _trace:
            _ace_step_log_ttnn_tensor(f"{debug_prefix}attn_out_B1SD", out, ttnn=ttnn)
        if debug is not None and debug.get("enabled", False):
            debug[f"{debug_prefix}attn_out"] = out
        return out


class TtQwen3MLP:
    """
    Qwen3-style gated MLP: down_proj(silu(gate_proj(x)) * up_proj(x))
    """

    def __init__(
        self, *, state_dict: dict, base_address: str, mesh_device, hidden_size: int, intermediate_size: int, dtype=None
    ):
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.mesh_device = mesh_device
        self.dtype = dtype or getattr(ttnn, "bfloat16", None) or getattr(ttnn, "float16", None)
        if self.dtype is None:
            raise RuntimeError("TTNN build missing a usable dtype (bfloat16/float16)")

        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None

        def as_w(name: str):
            return ttnn.as_tensor(
                _maybe_get(state_dict, f"{base_address}.{name}.weight"),
                device=mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        self.w_gate = as_w("gate_proj")
        self.w_up = as_w("up_proj")
        self.w_down = as_w("down_proj")

        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)

    def __call__(self, x, *, debug: Optional[dict] = None, debug_prefix: str = ""):
        ttnn = self.ttnn
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        gate = ttnn.linear(x, self.w_gate, bias=None, transpose_b=True)
        up = ttnn.linear(x, self.w_up, bias=None, transpose_b=True)
        if debug is not None and debug.get("enabled", False):
            debug[f"{debug_prefix}gate_lin"] = gate
            debug[f"{debug_prefix}up_lin"] = up
        gate = ttnn.silu(gate) if hasattr(ttnn, "silu") else ttnn.gelu(gate)
        if debug is not None and debug.get("enabled", False):
            debug[f"{debug_prefix}gate_act"] = gate
        h = ttnn.multiply(gate, up)
        out = ttnn.linear(h, self.w_down, bias=None, transpose_b=True)
        if debug is not None and debug.get("enabled", False):
            debug[f"{debug_prefix}mlp_raw_out"] = out
        return out


class TtAceStepDiTLayer:
    """
    TTNN port of `AceStepDiTLayer` (self-attn + cross-attn + gated MLP) with modulation.
    """

    def __init__(
        self,
        *,
        cfg: AceStepDecoderConfigTTNN,
        state_dict: dict,
        layer_idx: int,
        mesh_device,
        dtype=None,
        rotary_embedding: Optional[TtHfRotaryEmbedding] = None,
    ) -> None:
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.mesh_device = mesh_device
        self.layer_idx = int(layer_idx)
        self.dtype = dtype or getattr(ttnn, "bfloat16", None) or getattr(ttnn, "float16", None)
        if self.dtype is None:
            raise RuntimeError("TTNN build missing a usable dtype (bfloat16/float16)")

        d = int(cfg.hidden_size)
        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None

        # Norm weights
        self.self_norm_w = ttnn.as_tensor(
            _maybe_get(state_dict, f"layers.{layer_idx}.self_attn_norm.weight"),
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        self.cross_norm_w = ttnn.as_tensor(
            _maybe_get(state_dict, f"layers.{layer_idx}.cross_attn_norm.weight"),
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        self.mlp_norm_w = ttnn.as_tensor(
            _maybe_get(state_dict, f"layers.{layer_idx}.mlp_norm.weight"),
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        self.eps = float(cfg.rms_norm_eps)
        self.attention_type = "full_attention"
        self.sliding_window = int(cfg.sliding_window) if cfg.sliding_window is not None else None
        if cfg.layer_types is not None:
            try:
                self.attention_type = str(list(cfg.layer_types)[int(layer_idx)])
            except Exception:
                self.attention_type = "full_attention"

        # Attention modules
        self.self_attn = TtAceStepAttentionSDPA(
            cfg=cfg,
            state_dict=state_dict,
            base_address=f"layers.{layer_idx}.self_attn",
            mesh_device=mesh_device,
            dtype=self.dtype,
            rotary_embedding=rotary_embedding,
        )
        self.cross_attn = TtAceStepAttentionSDPA(
            cfg=cfg,
            state_dict=state_dict,
            base_address=f"layers.{layer_idx}.cross_attn",
            mesh_device=mesh_device,
            dtype=self.dtype,
            rotary_embedding=None,
        )

        # MLP sizes (from config.json; store in state dict as well, but we pass explicit)
        # Keys exist under layers.{i}.mlp.(gate_proj/up_proj/down_proj).weight
        gate_w = _maybe_get(state_dict, f"layers.{layer_idx}.mlp.gate_proj.weight")
        intermediate = int(gate_w.shape[0])
        self.mlp = TtQwen3MLP(
            state_dict=state_dict,
            base_address=f"layers.{layer_idx}.mlp",
            mesh_device=mesh_device,
            hidden_size=d,
            intermediate_size=intermediate,
            dtype=self.dtype,
        )

        # Scale-shift table: [1,6,D]
        sst = _maybe_get(state_dict, f"layers.{layer_idx}.scale_shift_table")
        self.scale_shift_table = ttnn.as_tensor(
            sst,
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )

    def __call__(
        self,
        hidden_states,
        timestep_proj_b6d,
        encoder_hidden_states,
        *,
        encoder_attention_mask_b1qk=None,
        self_attention_mask_b1qq=None,
        debug: Optional[dict] = None,
    ):
        """
        Args:
            hidden_states: [B, 1, S, D] TILE/ROW_MAJOR ok
            timestep_proj_b6d: [B, 6, D] row-major
            encoder_hidden_states: [B, 1, S_enc, D]
        """
        ttnn = self.ttnn
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        b = int(hidden_states.shape[0])

        temb = timestep_proj_b6d
        if tuple(temb.shape) != (b, 6, int(hidden_states.shape[-1])):
            raise ValueError(f"Expected timestep_proj [B,6,D], got {tuple(temb.shape)}")

        # (scale_shift_table + temb) -> chunk 6 along dim=1 => each [B,1,D]
        sst = ttnn.add(self.scale_shift_table, temb)
        d = int(hidden_states.shape[-1])

        def chunk(i: int):
            # Slice [B, 1, D] then view as [B,1,1,D] for broadcast.
            c = ttnn.slice(sst, (0, i, 0), (b, i + 1, d))
            return ttnn.reshape(c, (b, 1, 1, d))

        shift_msa = chunk(0)
        scale_msa = chunk(1)
        gate_msa = chunk(2)
        c_shift = chunk(3)
        c_scale = chunk(4)
        c_gate = chunk(5)
        li = int(getattr(self, "layer_idx", 0))
        core_pfx = f"core.layer{li}."
        if debug is not None and debug.get("enabled", False):
            debug[f"{core_pfx}c_gate"] = c_gate

        # Self-attn AdaLN: norm(x) * (1+scale) + shift
        x_norm = ttnn.rms_norm(
            hidden_states,
            weight=self.self_norm_w,
            epsilon=self.eps,
            memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None),
        )
        ones = ttnn.ones_like(scale_msa)
        one_plus = ttnn.add(scale_msa, ones)
        h = ttnn.add(ttnn.multiply(x_norm, one_plus), shift_msa)
        if debug is not None and debug.get("enabled", False):
            debug[f"{core_pfx}adaln_self_in"] = h

        self_dbg_pfx = f"{core_pfx}self." if (debug is not None and debug.get("enabled", False)) else ""
        attn_out = self.self_attn(
            h,
            encoder_hidden_states=None,
            is_causal=False,
            sliding_window_size=self.sliding_window if self.attention_type == "sliding_attention" else None,
            attn_mask=self_attention_mask_b1qq,
            debug=debug,
            debug_prefix=self_dbg_pfx,
        )
        gated = ttnn.multiply(attn_out, gate_msa)
        hidden_states = ttnn.add(hidden_states, gated)
        if debug is not None and debug.get("enabled", False):
            debug[f"{core_pfx}after_self"] = hidden_states

        # Cross-attn
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        x2 = ttnn.rms_norm(
            hidden_states,
            weight=self.cross_norm_w,
            epsilon=self.eps,
            memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None),
        )
        cross_dbg_pfx = f"{core_pfx}cross." if (debug is not None and debug.get("enabled", False)) else ""
        ca = self.cross_attn(
            x2,
            encoder_hidden_states=encoder_hidden_states,
            is_causal=False,
            attn_mask=encoder_attention_mask_b1qk,
            debug=debug,
            debug_prefix=cross_dbg_pfx,
        )
        hidden_states = ttnn.add(hidden_states, ca)
        if debug is not None and debug.get("enabled", False):
            debug[f"{core_pfx}cross.attn_out"] = ca
            debug[f"{core_pfx}after_cross"] = hidden_states

        # MLP AdaLN + gated residual
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        x3 = ttnn.rms_norm(
            hidden_states,
            weight=self.mlp_norm_w,
            epsilon=self.eps,
            memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None),
        )
        if debug is not None and debug.get("enabled", False):
            debug[f"{core_pfx}mlp_norm_out"] = x3
        ones2 = ttnn.ones_like(c_scale)
        one_plus2 = ttnn.add(c_scale, ones2)
        h3 = ttnn.add(ttnn.multiply(x3, one_plus2), c_shift)
        if debug is not None and debug.get("enabled", False):
            debug[f"{core_pfx}mlp_in"] = h3
        mlp_pfx = f"{core_pfx}mlp." if (debug is not None and debug.get("enabled", False)) else ""
        ff = self.mlp(
            h3,
            debug=debug if (debug is not None and debug.get("enabled", False)) else None,
            debug_prefix=mlp_pfx,
        )
        if debug is not None and debug.get("enabled", False):
            debug[f"{core_pfx}mlp_out"] = ff
        ff = ttnn.multiply(ff, c_gate)
        if debug is not None and debug.get("enabled", False):
            debug[f"{core_pfx}mlp_out_gated"] = ff
        hidden_states = ttnn.add(hidden_states, ff)
        if debug is not None and debug.get("enabled", False):
            debug[f"{core_pfx}block_out"] = hidden_states
        return hidden_states


class TtAceStepDiTCore:
    """
    Decoder core for `AceStepDiTModel` (proj_in already handled elsewhere).
    """

    def __init__(
        self,
        *,
        cfg: AceStepDecoderConfigTTNN,
        state_dict: dict,
        mesh_device,
        dtype=None,
    ) -> None:
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.mesh_device = mesh_device
        self.dtype = dtype or getattr(ttnn, "bfloat16", None) or getattr(ttnn, "float16", None)
        if self.dtype is None:
            raise RuntimeError("TTNN build missing a usable dtype (bfloat16/float16)")

        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None

        # condition_embedder: Linear(condition_dim -> hidden_size)
        self.cond_w = ttnn.as_tensor(
            _maybe_get(state_dict, "condition_embedder.weight"),
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        self.cond_b = ttnn.as_tensor(
            _maybe_get(state_dict, "condition_embedder.bias").reshape(1, 1, 1, -1),
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )

        self._rotary = TtHfRotaryEmbedding(
            mesh_device=mesh_device,
            head_dim=int(cfg.head_dim),
            max_seq_len=int(cfg.max_position_embeddings),
            rope_theta=float(cfg.rope_theta),
            hidden_size=int(cfg.hidden_size),
            num_attention_heads=int(cfg.num_attention_heads),
            num_key_value_heads=int(cfg.num_key_value_heads),
            dtype=self.dtype,
        )

        self.layers: List[TtAceStepDiTLayer] = [
            TtAceStepDiTLayer(
                cfg=cfg,
                state_dict=state_dict,
                layer_idx=i,
                mesh_device=mesh_device,
                dtype=self.dtype,
                rotary_embedding=self._rotary,
            )
            for i in range(int(cfg.num_hidden_layers))
        ]

    def __call__(
        self,
        hidden_states_patches,
        timestep_proj_b6d,
        encoder_hidden_states,
        encoder_attention_mask_b1qk=None,
        self_attention_mask_b1qq=None,
        debug: Optional[dict] = None,
    ):
        ttnn = self.ttnn
        # encoder_hidden_states: [B, S_enc, cond_dim] row-major -> [B,1,S_enc,D]
        enc = ttnn.to_layout(encoder_hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        enc = ttnn.unsqueeze(enc, 1)
        enc = ttnn.to_layout(enc, ttnn.TILE_LAYOUT)
        enc = ttnn.linear(enc, self.cond_w, bias=self.cond_b, transpose_b=True)
        if debug is not None and debug.get("enabled", False):
            debug["core.enc_conditioned"] = enc

        x = ttnn.to_layout(hidden_states_patches, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.unsqueeze(x, 1)  # [B,1,S,D]
        if debug is not None and debug.get("enabled", False):
            debug["core.x_patches_in"] = x
        for layer in self.layers:
            x = layer(
                x,
                timestep_proj_b6d,
                enc,
                encoder_attention_mask_b1qk=encoder_attention_mask_b1qk,
                self_attention_mask_b1qq=self_attention_mask_b1qq,
                debug=debug,
            )
        x = ttnn.squeeze(x, 1)  # [B,S,D]
        if debug is not None and debug.get("enabled", False):
            debug["core.out"] = x
        return x
