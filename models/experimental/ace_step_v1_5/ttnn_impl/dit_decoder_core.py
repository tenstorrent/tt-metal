# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import os
import traceback
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np


def _ace_step_env_truthy(name: str) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on", "y")


def _sliding_window_attn_bias_np(*, seq_len: int, window: int, batch: int = 1) -> np.ndarray:
    """Additive SDPA mask [B,1,S,S]: 0 inside |i-j|<=window, -1e9 outside (bidirectional)."""
    s = int(seq_len)
    i = np.arange(s, dtype=np.int32)[:, None]
    j = np.arange(s, dtype=np.int32)[None, :]
    keep = np.abs(i - j) <= int(window)
    sw = np.where(keep, np.float32(0.0), np.float32(-1e9)).astype(np.float32)
    return np.broadcast_to(sw.reshape(1, 1, s, s), (int(batch), 1, s, s)).copy()


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


def _ace_step_flush_device_profiler(ttnn, device) -> None:
    """Drain the per-RISC device-profiler ring buffer when Tracy / device profiling is enabled.

    The AceStep DiT core has ~24 layers; a single denoise step can emit > 12000 markers per RISC
    (the on-device profiler ring-buffer cap), causing kernels to silently drop markers and
    stock ``tools/tracy/process_ops_logs.py`` to abort with ``Device data missing`` (use
    ``perf/process_ace_tracy_ops_logs.py`` for lenient Tracy merge). Flushing after
    every Nth layer keeps the buffer within capacity. No-op when neither
    ``TT_METAL_DEVICE_PROFILER`` nor ``TTNN_OP_PROFILER`` is set, so production runs incur zero cost.

    Also no-op during an active DiT trace session: ``ttnn.synchronize_device`` is illegal inside
    ``begin_trace_capture``. Tracy + e2e trace are mutually exclusive anyway.
    """
    if os.environ.get("TTNN_OP_PROFILER") != "1" and os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        return
    try:
        from models.experimental.ace_step_v1_5.ttnn_impl.e2e_model_tt import ace_step_trace_session_active

        if ace_step_trace_session_active():
            return
    except Exception:
        pass
    try:
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
    except Exception:
        pass


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


from models.experimental.ace_step_v1_5.utils.tt_device import (
    ace_step_device_num_chips,
    ace_step_dit_weight_mesh_mapper,
    ace_step_synchronize_device,
)

import ttnn
from .math_perf_env import (
    _mcast_1d_linear_program_config,
    ace_step_add_one,
    ace_step_attn_qo_weight_dtype,
    ace_step_binary_kwargs,
    ace_step_dense_linear_program_config,
    ace_step_dit_attn_linear_program_config,
    ace_step_dit_fused_qwkv_linear_program_config,
    ace_step_dit_fused_wkv_linear_program_config,
    ace_step_dit_linear_l1_memory_config,
    ace_step_dit_mlp_down_proj_linear_program_config,
    ace_step_dit_mlp_fused_gate_up_linear_program_config,
    ace_step_dit_prefers_dram_activations,
    ace_step_dit_rms_norm_kwargs,
    ace_step_dit_weight_dtype,
    ace_step_eltwise_l1_memory_config,
    ace_step_ensure_dit_activation,
    ace_step_ensure_dram_activation,
    ace_step_ensure_l1_activation,
    ace_step_ensure_tile_layout,
    ace_step_from_torch_activation,
    ace_step_init_dit_linear_compute_kernel_config,
    ace_step_linear_kwargs_memory_config,
    ace_step_matmul_activation,
    ace_step_nlp_concat_heads,
    ace_step_permute_kwargs,
    ace_step_reshape_kwargs,
    ace_step_safe_deallocate,
    ace_step_sdpa_activation_kwargs,
    ace_step_sdpa_mask_memory_config,
    ace_step_split_qkv_heads_bhsd,
)


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
        linear_output_l1_memory_config=None,
    ) -> None:
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
        mapper = ace_step_dit_weight_mesh_mapper(mesh_device)

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

        self._linear_out_l1 = linear_output_l1_memory_config
        _t_freq_mc = linear_output_l1_memory_config or mem
        self.t_freq_table = ttnn.as_tensor(
            emb.reshape(self.num_steps, 1, 1, self.in_channels),
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=_t_freq_mc,
            mesh_mapper=mapper,
        )

        # `from_timestep_value` rebuilds sin/cos on host and uploads a [1,1,1,256] tensor every call.
        # In the Euler denoise loop `time_embed_r` is invoked with `delta_tr=0.0` every step (no
        # `timestep_r_index` passed by `e2e_model_tt.run_ttnn_denoise_loop`). Cache the (temb,tp) result
        # per scalar value so subsequent calls are pure device lookups and the trace stays stable.
        self._value_cache: dict[float, tuple["ttnn.Tensor", "ttnn.Tensor"]] = {}

    def __call__(self, timestep_index: int):
        ttnn = self.ttnn
        if not (0 <= int(timestep_index) < self.num_steps):
            raise ValueError(f"timestep_index out of range: {timestep_index} not in [0,{self.num_steps})")

        _l1 = self._linear_out_l1 or getattr(ttnn, "DRAM_MEMORY_CONFIG", None)

        t_freq = ttnn.slice(self.t_freq_table, (timestep_index, 0, 0, 0), (timestep_index + 1, 1, 1, self.in_channels))
        temb = ttnn.linear(t_freq, self.w1, bias=self.b1, transpose_b=True, memory_config=_l1)
        temb = ttnn.silu(temb, memory_config=_l1) if hasattr(ttnn, "silu") else ttnn.gelu(temb, memory_config=_l1)
        temb = ttnn.linear(temb, self.w2, bias=self.b2, transpose_b=True, memory_config=_l1)

        # time_proj(act2(temb)) -> [1,1,1,6*D] -> [1,6,D] (single reshape, not 6,1,D then 6,D)
        h = ttnn.silu(temb, memory_config=_l1) if hasattr(ttnn, "silu") else ttnn.gelu(temb, memory_config=_l1)
        tp = ttnn.linear(h, self.wt, bias=self.bt, transpose_b=True, memory_config=_l1)
        d = self.time_embed_dim
        _sr = ace_step_reshape_kwargs(ttnn)
        tp = ttnn.reshape(tp, (1, 6, d), **_sr)
        temb = ttnn.reshape(temb, (1, d), **_sr)
        if _l1 is not None:
            temb = ace_step_ensure_l1_activation(ttnn, temb, _l1)
            tp = ace_step_ensure_l1_activation(ttnn, tp, _l1)
        return temb, tp

    def from_timestep_value(self, timestep: float):
        """
        HF-parity path: compute sinusoidal embedding for an arbitrary timestep value.

        This matches HF's `TimestepEmbedding` contract of generating sin/cos at runtime, but
        it uploads a tiny `[1,1,1,256]` tensor each call (host -> device).

        Results are cached per scalar so repeated calls (e.g. `delta_tr=0.0` every denoise step)
        return the same device tensors and do not re-upload sin/cos.
        """
        ttnn = self.ttnn
        _l1 = self._linear_out_l1 or getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        key = float(timestep)
        cached = self._value_cache.get(key)
        if cached is not None:
            if _l1 is not None:
                temb_c, tp_c = cached
                return (
                    ace_step_ensure_l1_activation(ttnn, temb_c, _l1),
                    ace_step_ensure_l1_activation(ttnn, tp_c, _l1),
                )
            return cached

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
        t_freq = ace_step_from_torch_activation(
            ttnn,
            emb,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            l1_mc=_l1,
        )
        temb = ttnn.linear(t_freq, self.w1, bias=self.b1, transpose_b=True, memory_config=_l1)
        temb = ttnn.silu(temb, memory_config=_l1) if hasattr(ttnn, "silu") else ttnn.gelu(temb, memory_config=_l1)
        temb = ttnn.linear(temb, self.w2, bias=self.b2, transpose_b=True, memory_config=_l1)
        h = ttnn.silu(temb, memory_config=_l1) if hasattr(ttnn, "silu") else ttnn.gelu(temb, memory_config=_l1)
        tp = ttnn.linear(h, self.wt, bias=self.bt, transpose_b=True, memory_config=_l1)
        d = self.time_embed_dim
        _sr = ace_step_reshape_kwargs(ttnn)
        tp = ttnn.reshape(tp, (1, 6, d), **_sr)
        temb = ttnn.reshape(temb, (1, d), **_sr)
        # Keep t_freq allocated only for the duration of this call.
        if hasattr(ttnn, "deallocate"):
            try:
                ttnn.deallocate(t_freq)
            except Exception:
                pass
        if _l1 is not None:
            temb = ace_step_ensure_l1_activation(ttnn, temb, _l1)
            tp = ace_step_ensure_l1_activation(ttnn, tp, _l1)
        self._value_cache[key] = (temb, tp)
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

        mapper = ace_step_dit_weight_mesh_mapper(mesh_device)
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
        _rope_out_mc = ace_step_eltwise_l1_memory_config(ttnn) or getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        return ttnn.experimental.rotary_embedding(
            x,
            self.cos_cached,
            self.sin_cached,
            token_idx,
            memory_config=_rope_out_mc,
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
    use_fp32: Optional[bool] = None,
    softmax_fp32: bool = False,
    compute_kernel_config=None,
    eltwise_memory_config=None,
):
    """
    Cross-attention without fused SDPA: matches torch_ref `dit_decoder_core._sdpa`
    (QK^T * scale, softmax on keys, @ V) while allowing additive padding / encoder masks.

    torch_ref (DiT) runs attention matmuls and softmax in FP32, then casts context back to BF16.
    When `ttnn.float32` is available and ``use_fp32`` is not False, this path does the same;
    otherwise it stays in ``activations_dtype`` for the full decomposed op.

    Pass ``use_fp32=False`` when the reference model runs attention in BF16 (e.g. causal LM with
    HF ``torch_dtype=torch.bfloat16``) to avoid a precision mismatch that compounds over layers.

    Pass ``softmax_fp32=True`` along with ``use_fp32=False`` to upcast only the softmax to float32
    and cast back (matches HF Qwen3 ``nn.functional.softmax(..., dtype=torch.float32).to(bf16)``).

    Shapes:
      q,k,v: [B, H, S_q or S_k, Dh] TILE
      additive_mask_b1qk: optional [B, 1, S_q, S_k] added to pre-softmax logits (broadcast over heads).
    """
    s_k = int(k.shape[2])
    dh = int(q.shape[3])
    bh = int(b * h)
    el_mc = (
        eltwise_memory_config or ace_step_eltwise_l1_memory_config(ttnn) or getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    )
    mem_kw = ace_step_binary_kwargs(ttnn, el_mc)
    ck_kw = dict(compute_kernel_config=compute_kernel_config) if compute_kernel_config is not None else {}
    fp32 = getattr(ttnn, "float32", None)
    if use_fp32 is None:
        use_fp32 = fp32 is not None
    else:
        use_fp32 = bool(use_fp32) and (fp32 is not None)
    # softmax_fp32 only applies when not already running the full path in fp32
    do_softmax_fp32 = bool(softmax_fp32) and (not use_fp32) and (fp32 is not None)
    sr = ace_step_reshape_kwargs(ttnn)
    pk = ace_step_permute_kwargs(ttnn)

    def _l1(t):
        return ace_step_ensure_l1_activation(ttnn, t, el_mc)

    qf = _l1(ttnn.reshape(q, (bh, s_q, dh), **sr))
    kf = _l1(ttnn.reshape(k, (bh, s_k, dh), **sr))
    vf = _l1(ttnn.reshape(v, (bh, s_k, dh), **sr))
    if use_fp32:
        qf = _l1(ttnn.typecast(qf, dtype=fp32))
        kf = _l1(ttnn.typecast(kf, dtype=fp32))
        vf = _l1(ttnn.typecast(vf, dtype=fp32))

    kt = ttnn.permute(kf, (0, 2, 1), **pk)
    scores = _l1(ttnn.matmul(qf, kt, **mem_kw, **ck_kw))
    scores = ttnn.multiply(scores, float(scale), **mem_kw)
    if additive_mask_b1qk is not None:
        m = ttnn.repeat(additive_mask_b1qk, (1, int(h), 1, 1))
        m = ttnn.reshape(m, (bh, s_q, s_k), **sr)
        if use_fp32:
            m = ttnn.typecast(m, dtype=fp32)
        scores = ttnn.add(scores, m, **mem_kw)
    if do_softmax_fp32:
        scores = ttnn.typecast(scores, dtype=fp32)
    attn = ttnn.softmax(scores, dim=-1, **mem_kw)
    if do_softmax_fp32 and activations_dtype is not None:
        attn = ttnn.typecast(attn, dtype=activations_dtype)
    ctx = _l1(ttnn.matmul(attn, vf, **mem_kw, **ck_kw))
    if use_fp32 and activations_dtype is not None:
        ctx = _l1(ttnn.typecast(ctx, dtype=activations_dtype))
    return _l1(ttnn.reshape(ctx, (b, h, s_q, dh), **sr))


class TtAceStepAttentionSDPA:
    """
    AceStep DiT attention on TTNN.

    Self-attention uses ``ttnn.transformer.scaled_dot_product_attention``. Cross-attention uses
    fused SDPA by default (set ``ACE_STEP_CROSS_ATTN_DECOMPOSED=1`` to restore FP32 decomposed
    path for PCC debugging).

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
        linear_compute_kernel_config=None,
        activation_l1_memory_config=None,
        linear_output_l1_memory_config=None,
    ):
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
        mapper = ace_step_dit_weight_mesh_mapper(mesh_device)  # replicate (None on BH 2×2)

        # TP: attention is head-parallel — Q/K/V column-parallel by head, output proj row-parallel.
        # OFF path is byte-identical (deg==1 makes _interleave the identity of the old concat order,
        # and col/row mappers fall back to the legacy replicate mapper).
        from models.experimental.ace_step_v1_5.ttnn_impl.tp_config import resolve_tp_config, tp_weight_mesh_mapper

        self._tp = resolve_tp_config(mesh_device)
        deg = self._tp.degree
        tp_on = self._tp.enabled and deg > 1
        if tp_on and (self.n_heads % deg != 0 or self.n_kv % deg != 0):
            raise ValueError(f"TP degree {deg} must divide n_heads {self.n_heads} and n_kv {self.n_kv}")
        self._n_heads_local = self.n_heads // deg if tp_on else self.n_heads
        self._n_kv_local = self.n_kv // deg if tp_on else self.n_kv
        self._d_model_local = self._n_heads_local * self.d_head
        self._fused_kv_dim_local = int(self._n_kv_local * self.d_head) * 2

        w_dtype = ace_step_dit_weight_dtype(ttnn, self.dtype)
        qo_dtype = ace_step_attn_qo_weight_dtype(ttnn, self.dtype)
        col_mapper = tp_weight_mesh_mapper(mesh_device, shard_dim=0, cfg=self._tp) if tp_on else mapper
        row_mapper = tp_weight_mesh_mapper(mesh_device, shard_dim=1, cfg=self._tp) if tp_on else mapper

        def _interleave(mats: list):
            """Per-chip head-contiguous interleave along dim 0: [m0_d0,m1_d0,..,m0_d1,m1_d1,..].
            With deg==1 this is exactly ``np.concatenate(mats, axis=0)`` (legacy layout)."""
            splits = [np.split(m, deg, axis=0) for m in mats]
            return np.concatenate([splits[j][d] for d in range(deg) for j in range(len(mats))], axis=0)

        def _as(host, *, dtype, mapper_):
            return ttnn.as_tensor(
                host, device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=mem, mesh_mapper=mapper_
            )

        def _b_raw(suffix: str):
            b = state_dict.get(f"{base_address}.{suffix}.bias", None)
            return None if b is None else _to_numpy_host_array(b).reshape(-1)

        wq_host = _to_numpy_host_array(_maybe_get(state_dict, f"{base_address}.q_proj.weight"))
        wk_host = _to_numpy_host_array(_maybe_get(state_dict, f"{base_address}.k_proj.weight"))
        wv_host = _to_numpy_host_array(_maybe_get(state_dict, f"{base_address}.v_proj.weight"))

        # Q: heads already contiguous → plain dim-0 shard (column-parallel).
        self.wq = _as(_interleave([wq_host]), dtype=qo_dtype, mapper_=col_mapper)
        # Fused KV: interleave [k_d, v_d] per chip so a dim-0 shard gives each chip its k+v heads.
        self.wkv = _as(_interleave([wk_host, wv_host]), dtype=w_dtype, mapper_=col_mapper)
        # Self-attn fused QKV: interleave [q_d, k_d, v_d] per chip.
        self.w_qwkv = _as(_interleave([wq_host, wk_host, wv_host]), dtype=w_dtype, mapper_=col_mapper)

        bq_np = _b_raw("q_proj")
        bk_np = _b_raw("k_proj")
        bv_np = _b_raw("v_proj")
        self.bq = (
            _as(_interleave([bq_np]).reshape(1, 1, 1, -1), dtype=w_dtype, mapper_=col_mapper)
            if bq_np is not None
            else None
        )
        if bk_np is not None and bv_np is not None:
            self.bkv = _as(_interleave([bk_np, bv_np]).reshape(1, 1, 1, -1), dtype=w_dtype, mapper_=col_mapper)
        else:
            self.bkv = None
        if bq_np is not None and bk_np is not None and bv_np is not None:
            self.b_qwkv = _as(
                _interleave([bq_np, bk_np, bv_np]).reshape(1, 1, 1, -1), dtype=w_dtype, mapper_=col_mapper
            )
        else:
            self.b_qwkv = None

        # Output proj: row-parallel — shard input (q_dim, head-contiguous) across TP; all-reduce output.
        wo_host = _to_numpy_host_array(_maybe_get(state_dict, f"{base_address}.o_proj.weight"))
        self.wo = _as(wo_host, dtype=qo_dtype, mapper_=row_mapper)
        bo_np = _b_raw("o_proj")
        # bo is added to the FULL output; under TP add it once AFTER all-reduce (not per shard).
        self.bo = _as(bo_np.reshape(1, 1, 1, -1), dtype=w_dtype, mapper_=mapper) if bo_np is not None else None

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

        self._linear_ck = linear_compute_kernel_config
        self._act_l1 = activation_l1_memory_config
        self._linear_out_l1 = linear_output_l1_memory_config
        self._fused_kv_dim = int(self.n_kv * self.d_head) * 2
        self._wkv_pc_cache: dict = {}
        self._qwkv_pc_cache: dict = {}
        # Per-call mask uploads (additive tail/pad masks) used to rebuild the same NumPy zeros tensor
        # and call ttnn.as_tensor on every forward, every layer, every step. Cache them by shape so
        # they stay device-resident across denoise steps and become trace+2CQ friendly.
        # Keys:
        #   - self-attn pad mask:  (B, s_rope, target_sdpa)        value broadcasts over S_q==S_k==target_sdpa
        #   - cross-attn tail mask:(B, S_q0, W, s_enc_log)         pads keys past s_enc_log up to W
        self._self_pad_mask_cache: dict[tuple[int, int, int], "ttnn.Tensor"] = {}
        self._cross_tail_mask_cache: dict[tuple[int, int, int, int], "ttnn.Tensor"] = {}
        self._sliding_window_mask_cache: dict[tuple[int, int, int], "ttnn.Tensor"] = {}

    def _get_self_pad_mask(self, *, batch: int, target_sdpa: int, s_rope: int) -> "ttnn.Tensor":
        """Return a cached additive self-attn pad mask ``[B,1,target_sdpa,target_sdpa]``.

        Columns ``[s_rope, target_sdpa)`` get ``-1e9`` so SDPA's softmax ignores tile-pad keys
        while matching the unpadded PyTorch/HF result over the logical sequence.
        """
        ttnn = self.ttnn
        key = (int(batch), int(s_rope), int(target_sdpa))
        cached = self._self_pad_mask_cache.get(key)
        if cached is not None:
            return cached
        pad_np = np.zeros((int(batch), 1, int(target_sdpa), int(target_sdpa)), dtype=np.float32)
        pad_np[:, :, :, int(s_rope) :] = np.float32(-1e9)
        mem_m = ace_step_sdpa_mask_memory_config(ttnn) or getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper_m = ace_step_dit_weight_mesh_mapper(self.mesh_device)
        pad_m = ttnn.as_tensor(
            pad_np,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_m,
            mesh_mapper=mapper_m,
        )
        self._self_pad_mask_cache[key] = pad_m
        return pad_m

    def _get_cross_tail_mask(self, *, batch: int, s_q0: int, w: int, s_enc_log: int) -> "ttnn.Tensor":
        """Return a cached additive cross-attn tail mask ``[B,1,S_q0,W]`` masking ``[s_enc_log, W)``."""
        ttnn = self.ttnn
        key = (int(batch), int(s_q0), int(w), int(s_enc_log))
        cached = self._cross_tail_mask_cache.get(key)
        if cached is not None:
            return cached
        pad_np = np.zeros((int(batch), 1, int(s_q0), int(w)), dtype=np.float32)
        pad_np[:, :, :, int(s_enc_log) : int(w)] = np.float32(-1e9)
        mem_m = ace_step_sdpa_mask_memory_config(ttnn) or getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper_m = ace_step_dit_weight_mesh_mapper(self.mesh_device)
        pad_m = ttnn.as_tensor(
            pad_np,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_m,
            mesh_mapper=mapper_m,
        )
        self._cross_tail_mask_cache[key] = pad_m
        return pad_m

    def _get_sliding_window_mask(self, *, batch: int, seq_len: int, window: int) -> "ttnn.Tensor":
        """Return a cached additive bidirectional sliding-window mask ``[B,1,S,S]``."""
        ttnn = self.ttnn
        key = (int(batch), int(seq_len), int(window))
        cached = self._sliding_window_mask_cache.get(key)
        if cached is not None:
            return cached
        sw_np = _sliding_window_attn_bias_np(seq_len=int(seq_len), window=int(window), batch=int(batch))
        mem_m = ace_step_sdpa_mask_memory_config(ttnn) or getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper_m = ace_step_dit_weight_mesh_mapper(self.mesh_device)
        sw_m = ttnn.as_tensor(
            sw_np,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_m,
            mesh_mapper=mapper_m,
        )
        self._sliding_window_mask_cache[key] = sw_m
        return sw_m

    def _l1_activation(self, t):
        if self._act_l1 is None:
            return t
        return self.ttnn.to_memory_config(t, self._act_l1)

    def _linear_kwargs(self, *, batch_size: int, seq_len: int, in_dim: int, out_dim: int) -> dict:
        kw: dict = {}
        if self._linear_ck is not None:
            kw["compute_kernel_config"] = self._linear_ck
        pc = ace_step_dit_attn_linear_program_config(
            self.mesh_device,
            seq_len=int(seq_len),
            in_dim=int(in_dim),
            out_dim=int(out_dim),
            batch_size=int(batch_size),
        )
        if pc is None:
            pc = _mcast_1d_linear_program_config(
                self.mesh_device,
                seq_len=int(seq_len),
                in_dim=int(in_dim),
                out_dim=int(out_dim),
                batch_size=int(batch_size),
                in0_block_w_cap=2,
                out_subblock_h_cap=4,
                out_subblock_w=1,
            )
        if pc is not None:
            kw["program_config"] = pc
        _dram = getattr(self.ttnn, "DRAM_MEMORY_CONFIG", None)
        kw["memory_config"] = ace_step_linear_kwargs_memory_config(pc, linear_out_l1=self._linear_out_l1, dram=_dram)
        return kw

    def _qwkv_linear_kwargs(self, *, batch_size: int, seq_len: int, in_dim: int) -> dict:
        """LoFi + L1 program config for fused self-attn ``q`` + ``wkv``."""
        kw: dict = {}
        if self._linear_ck is not None:
            kw["compute_kernel_config"] = self._linear_ck
        key = (int(batch_size), int(seq_len), int(in_dim))
        pc = self._qwkv_pc_cache.get(key)
        if pc is None:
            pc = ace_step_dit_fused_qwkv_linear_program_config(
                self.mesh_device,
                seq_len=int(seq_len),
                in_dim=int(in_dim),
                hidden_size=self.d_model,
                fused_kv_dim=self._fused_kv_dim,
                batch_size=int(batch_size),
            )
            if pc is not None:
                self._qwkv_pc_cache[key] = pc
        if pc is not None:
            kw["program_config"] = pc
        _dram = getattr(self.ttnn, "DRAM_MEMORY_CONFIG", None)
        kw["memory_config"] = ace_step_linear_kwargs_memory_config(pc, linear_out_l1=self._linear_out_l1, dram=_dram)
        return kw

    def _wkv_linear_kwargs(self, *, batch_size: int, seq_len: int, in_dim: int) -> dict:
        """LoFi + L1 + wide-N program config for fused ``wkv`` (256×2048×2048 family)."""
        kw: dict = {}
        if self._linear_ck is not None:
            kw["compute_kernel_config"] = self._linear_ck
        key = (int(batch_size), int(seq_len), int(in_dim))
        pc = self._wkv_pc_cache.get(key)
        if pc is None:
            pc = ace_step_dit_fused_wkv_linear_program_config(
                self.mesh_device,
                seq_len=int(seq_len),
                hidden_size=int(in_dim),
                fused_kv_dim=self._fused_kv_dim,
                batch_size=int(batch_size),
            )
            if pc is not None:
                self._wkv_pc_cache[key] = pc
        if pc is not None:
            kw["program_config"] = pc
        _dram = getattr(self.ttnn, "DRAM_MEMORY_CONFIG", None)
        kw["memory_config"] = ace_step_linear_kwargs_memory_config(pc, linear_out_l1=self._linear_out_l1, dram=_dram)
        return kw

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
        use_dram_activations: bool = False,
    ):
        ttnn = self.ttnn
        _sr = ace_step_reshape_kwargs(ttnn)
        _pk = ace_step_permute_kwargs(ttnn)
        _trace = _ace_step_attn_trace_print(debug_prefix)
        _dram_mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        _qk_mc = _dram_mc if use_dram_activations else (self._linear_out_l1 or _dram_mc)
        _concat_mc = _dram_mc if use_dram_activations else ace_step_eltwise_l1_memory_config(ttnn)
        if use_dram_activations:

            def _act_fn(t):
                return ace_step_ensure_dram_activation(ttnn, t, _dram_mc)

            _op_mc = _dram_mc
            _head_mc = _dram_mc
        else:

            def _act_fn(t):
                return self._l1_activation(t)

            _op_mc = self._act_l1
            _head_mc = self._act_l1
        x = ace_step_ensure_tile_layout(ttnn, hidden_states)  # [B,1,S,D]
        b_x = int(x.shape[0])
        s_q = int(x.shape[2])
        d_in = int(x.shape[-1])
        if encoder_hidden_states is None:
            lin_qwkv = self._qwkv_linear_kwargs(batch_size=b_x, seq_len=s_q, in_dim=d_in)
            x_qwkv = ace_step_matmul_activation(ttnn, x, lin_qwkv, l1_fn=self._l1_activation, dram_mc=_dram_mc)
            qwkv = ttnn.linear(x_qwkv, self.w_qwkv, bias=self.b_qwkv, transpose_b=True, **lin_qwkv)
            d_q = int(self._d_model_local)
            kv_end = d_q + int(self._fused_kv_dim_local)
            s_lin = int(qwkv.shape[2])
            q = ttnn.slice(qwkv, (0, 0, 0, 0), (b_x, 1, s_lin, d_q))
            kv = ttnn.slice(qwkv, (0, 0, 0, d_q), (b_x, 1, s_lin, kv_end))
            ace_step_safe_deallocate(ttnn, qwkv)
            q = _act_fn(q)
            kv = _act_fn(kv)
        else:
            lin_q = self._linear_kwargs(
                batch_size=b_x,
                seq_len=s_q,
                in_dim=d_in,
                out_dim=self.d_model,
            )
            x_q = ace_step_matmul_activation(ttnn, x, lin_q, l1_fn=self._l1_activation, dram_mc=_dram_mc)
            q = ttnn.linear(x_q, self.wq, bias=self.bq, transpose_b=True, **lin_q)
            enc = ace_step_ensure_tile_layout(ttnn, encoder_hidden_states)
            b_enc = int(enc.shape[0])
            s_enc = int(enc.shape[2])
            d_enc = int(enc.shape[-1])
            lin_kv = self._wkv_linear_kwargs(batch_size=b_enc, seq_len=s_enc, in_dim=d_enc)
            enc_kv = ace_step_matmul_activation(ttnn, enc, lin_kv, l1_fn=self._l1_activation, dram_mc=_dram_mc)
            kv = ttnn.linear(enc_kv, self.wkv, bias=self.bkv, transpose_b=True, **lin_kv)

        B = int(q.shape[0])
        S = int(q.shape[2])
        H = self._n_heads_local  # local heads under TP (== n_heads when off)
        Dh = self.d_head
        if _trace:
            print(
                f"[ace_step_v1_5][attn_trace][ttnn] {debug_prefix}enter "
                f"B={B} S_q_linear={S} H={H} Dh={Dh} cross={encoder_hidden_states is not None} "
                f"scale={self.scale:.6g} attn_mask={'set' if attn_mask is not None else 'None'}",
                flush=True,
            )
            _ace_step_log_ttnn_tensor(f"{debug_prefix}q_after_linear_B1SD", q, ttnn=ttnn)

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
            _pad_kw = {"memory_config": _op_mc} if _op_mc is not None else {}
            return ttnn.pad(
                x,
                padding=((0, 0), (0, 0), (0, pad), (0, 0)),
                value=0.0,
                **_pad_kw,
            )

        def _pad_seq_dim2_bh_sd(x, target_s: int):
            """Pad/slice sequence dim (index 2) for tensors shaped [B, H, S, Dh]."""
            b0, h0, s0, d0 = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
            if s0 == target_s:
                return x
            if s0 > target_s:
                return ttnn.slice(x, (0, 0, 0, 0), (b0, h0, target_s, d0))
            pad = target_s - s0
            _pad_kw = {"memory_config": _op_mc} if _op_mc is not None else {}
            return ttnn.pad(
                x,
                padding=((0, 0), (0, 0), (0, pad), (0, 0)),
                value=0.0,
                **_pad_kw,
            )

        is_self_attn = encoder_hidden_states is None

        q, k, v = ace_step_split_qkv_heads_bhsd(
            ttnn,
            q,
            kv,
            num_heads=H,
            num_kv_heads=self._n_kv_local,
            l1_mc=_head_mc,
        )

        # Safety: TTNN SDPA requires K/V to have the same logical and padded sequence length.
        S_k_raw = max(_seq_len_logical(k), _seq_len_padded(k))
        S_v_raw = max(_seq_len_logical(v), _seq_len_padded(v))
        if S_k_raw != S_v_raw:
            target = _ceil_tile(max(S_k_raw, S_v_raw))
            k = _pad_seq_dim2_bh_sd(k, target)
            v = _pad_seq_dim2_bh_sd(v, target)

        S_k = int(k.shape[2])
        kv_h = self._n_kv_local
        if _trace:
            _ace_step_log_ttnn_tensor(f"{debug_prefix}q_after_reshape_BHSD", q, ttnn=ttnn)
        if debug is not None and debug.get("enabled", False):
            debug[f"{debug_prefix}q_lin"] = q
            debug[f"{debug_prefix}k_lin"] = k
            debug[f"{debug_prefix}v_lin"] = v

        if kv_h != H:
            # Grouped-query attention: repeat kv heads to match q heads.
            if H % kv_h != 0:
                raise ValueError(f"num_attention_heads {H} not divisible by num_key_value_heads {kv_h}")
            rep = H // kv_h
            # HF GQA semantics: kv0 -> q0,q1 ; kv1 -> q2,q3 ; ...
            #
            # ttnn.repeat_interleave and the old concat-based fallback both route through DRAM:
            # repeat_interleave calls ttnn::concat / ttnn::to_layout without memory_config, so
            # they default to DRAM.  The final to_layout(TILE) emits
            # ``TilizeDeviceOperation (in0:dram_interleaved)`` (~10 μs × 96 = ~960 μs = 2.08 %
            # of device time), followed by a ``CopyDeviceOperation (in0:dram_interleaved)``
            # from the _l1_activation() call.
            #
            # Fix: replicate the same unsqueeze→concat→reshape→tilize sequence but pass
            # memory_config=L1 at every step.  ROW_MAJOR L1 inputs bypass the
            # build_untilize_rm_retilize_concat path inside ttnn.concat and go straight to
            # concat_impl with output_memory_config=L1 → ROW_MAJOR L1 output.  The final
            # to_layout(TILE, memory_config=L1) then reads from L1, emitting
            # ``TilizeDeviceOperation (in0:l1_interleaved)`` and eliminating the subsequent
            # CopyDeviceOperation entirely (result is already L1 TILE).
            _gqa_mc = _op_mc or getattr(ttnn, "L1_MEMORY_CONFIG", None)
            _gqa_kw = {"memory_config": _gqa_mc} if _gqa_mc is not None else {}
            _rm_layout = getattr(ttnn, "ROW_MAJOR_LAYOUT", None)
            _tile_layout = getattr(ttnn, "TILE_LAYOUT", None)

            def _gqa_expand(x: Any) -> Any:
                """Repeat-interleave x along head dim (dim=1) × rep with pinned activation memory."""
                b_, kv_, s_, dh_ = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3]))
                xrm = (
                    ttnn.to_layout(x, _rm_layout, **_gqa_kw)
                    if _rm_layout is not None
                    else ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
                )
                xus = ttnn.unsqueeze(xrm, 2)
                _concat_fn = getattr(ttnn, "concat", None) or ttnn.concatenate
                xrep = _concat_fn([xus] * rep, dim=2, **_gqa_kw)
                xflat = ttnn.reshape(xrep, (b_, kv_ * rep, s_, dh_), **_gqa_kw)
                return (
                    ttnn.to_layout(xflat, _tile_layout, **_gqa_kw)
                    if _tile_layout is not None
                    else ttnn.to_layout(xflat, ttnn.TILE_LAYOUT)
                )

            k = _gqa_expand(k)
            v = _gqa_expand(v)

        q = _act_fn(q)
        k = _act_fn(k)
        v = _act_fn(v)

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

        # Head-dim RMSNorm on q and k — L1 on short clips; DRAM on long clips (SDPA-safe).
        _rms_kw = ace_step_dit_rms_norm_kwargs(ttnn, _qk_mc, device=self.mesh_device)
        q = ttnn.rms_norm(q, weight=self.q_norm_w, epsilon=self.eps, **_rms_kw)
        k = ttnn.rms_norm(k, weight=self.k_norm_w, epsilon=self.eps, **_rms_kw)
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
            q = _act_fn(q)
            k = _act_fn(k)
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
                q = _act_fn(q)
                k = _act_fn(k)
                v = _act_fn(v)
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
                k = _act_fn(k)
                v = _act_fn(v)
                W = tgt_k
            if W > s_enc_log:
                pad_m = self._get_cross_tail_mask(batch=B, s_q0=S_q0, w=W, s_enc_log=s_enc_log)
                if sdpa_attn_mask is None:
                    sdpa_attn_mask = pad_m  # cached tensor, not owned
                else:
                    _mask_mc = ace_step_sdpa_mask_memory_config(ttnn)
                    sdpa_attn_mask = ttnn.add(sdpa_attn_mask, pad_m, memory_config=_mask_mc)

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
                q = _act_fn(q)
                k = _act_fn(k)
                v = _act_fn(v)
                # Additive mask (0 keep, -1e9 on invalid keys); SDPA adds this to QK (see ttnn sdpa.cpp).
                # Match Q/K/V dtype so SDPA uniform-dataformat paths accept the mask tensor.
                pad_m = self._get_self_pad_mask(batch=B, target_sdpa=target_sdpa, s_rope=S_rope)
                if sdpa_attn_mask is None:
                    sdpa_attn_mask = pad_m  # cached tensor, not owned
                else:
                    _mask_mc = ace_step_sdpa_mask_memory_config(ttnn)
                    sdpa_attn_mask = ttnn.add(sdpa_attn_mask, pad_m, memory_config=_mask_mc)

        # Cross-attn: fused SDPA by default (L1 Q/K/V, DRAM mask only). The old decomposed FP32 path
        # dominated E2E traces as ``BinaryNgDeviceOperation (in0:dram_interleaved)`` (~77 ms) while
        # BF16 AdaLN ``add``/``multiply`` L1 tuning moved <~10 ms. Set ``ACE_STEP_CROSS_ATTN_DECOMPOSED=1``
        # to restore FP32 decomposed scores/softmax for debugging PCC.
        if not is_self_attn and _ace_step_env_truthy("ACE_STEP_CROSS_ATTN_DECOMPOSED"):
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
                eltwise_memory_config=_dram_mc if use_dram_activations else self._act_l1,
            )
        else:
            # TTNN SDPA rejects sliding_window_size together with a dense attn_mask; bake the
            # bidirectional sliding window into the additive mask instead (matches condition_encoder).
            effective_sliding = sliding_window_size
            if is_self_attn and sliding_window_size is not None and sdpa_attn_mask is not None:
                sw_m = self._get_sliding_window_mask(
                    batch=B,
                    seq_len=int(q.shape[2]),
                    window=int(sliding_window_size),
                )
                _mask_mc = ace_step_sdpa_mask_memory_config(ttnn)
                sdpa_attn_mask = ttnn.add(sdpa_attn_mask, sw_m, memory_config=_mask_mc)
                effective_sliding = None
            sdpa_kwargs = dict(attn_mask=sdpa_attn_mask, is_causal=is_causal, scale=self.scale)
            if effective_sliding is not None:
                sdpa_kwargs["sliding_window_size"] = int(effective_sliding)
            sdpa_kwargs.update(ace_step_sdpa_activation_kwargs(ttnn, _op_mc))
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
        # [B,H,S_ctx,Dh] -> [B,1,S_ctx,H*Dh] (nlp_concat_heads by default)
        ctx = ace_step_nlp_concat_heads(ttnn, ctx, l1_mc=_concat_mc)
        if S_ctx != S:
            ctx = ttnn.slice(ctx, (0, 0, 0, 0), (B, 1, S, H * Dh))
        ctx = _act_fn(ctx)
        lin_o = self._linear_kwargs(
            batch_size=B,
            seq_len=S,
            in_dim=H * Dh,
            out_dim=self.d_model,
        )
        ctx_o = ace_step_matmul_activation(ttnn, ctx, lin_o, l1_fn=self._l1_activation, dram_mc=_dram_mc)
        _tp_on = self._tp.enabled and self._tp.degree > 1
        # Row-parallel o_proj: under TP each chip holds a q_dim shard → partial output. Defer the
        # bias (it applies to the full sum) and all-reduce across the TP axis to rebuild hidden.
        out = ttnn.linear(ctx_o, self.wo, bias=(None if _tp_on else self.bo), transpose_b=True, **lin_o)
        if _tp_on:
            from models.experimental.ace_step_v1_5.ttnn_impl.tp_config import tp_all_reduce

            out = tp_all_reduce(out, self.mesh_device, cfg=self._tp)
            if self.bo is not None:
                out = ttnn.add(out, self.bo)
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
        self,
        *,
        state_dict: dict,
        base_address: str,
        mesh_device,
        hidden_size: int,
        intermediate_size: int,
        dtype=None,
        linear_compute_kernel_config=None,
        activation_l1_memory_config=None,
        linear_output_l1_memory_config=None,
    ):
        self.ttnn = ttnn
        self.mesh_device = mesh_device
        self.dtype = dtype or getattr(ttnn, "bfloat16", None) or getattr(ttnn, "float16", None)
        if self.dtype is None:
            raise RuntimeError("TTNN build missing a usable dtype (bfloat16/float16)")

        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        # Replicate mapper (None on BH 2×2 — avoids large-upload stalls); unchanged when TP off.
        mapper = ace_step_dit_weight_mesh_mapper(mesh_device)
        w_dtype = ace_step_dit_weight_dtype(ttnn, self.dtype)

        # TP: shard the gated MLP across the mesh (column-parallel gate/up, row-parallel down).
        # OFF path is byte-identical to before (mapper stays the legacy replicate mapper).
        from models.experimental.ace_step_v1_5.ttnn_impl.tp_config import resolve_tp_config, tp_weight_mesh_mapper

        self._tp = resolve_tp_config(mesh_device)
        tp_deg = self._tp.degree
        self._local_intermediate = int(intermediate_size) // tp_deg if self._tp.enabled else int(intermediate_size)

        w_gate_host = _to_numpy_host_array(_maybe_get(state_dict, f"{base_address}.gate_proj.weight"))
        w_up_host = _to_numpy_host_array(_maybe_get(state_dict, f"{base_address}.up_proj.weight"))
        if self._tp.enabled and tp_deg > 1:
            # Fused gate/up column-parallel: a contiguous dim-0 shard must give each chip its slice
            # of BOTH gate and up, so interleave per-chip chunks [g0,u0,g1,u1,...] before sharding.
            g_chunks = np.split(w_gate_host, tp_deg, axis=0)
            u_chunks = np.split(w_up_host, tp_deg, axis=0)
            w_gate_up_host = np.concatenate([c for pair in zip(g_chunks, u_chunks) for c in pair], axis=0)
            gate_up_mapper = tp_weight_mesh_mapper(mesh_device, shard_dim=0, cfg=self._tp)  # shard output
            down_mapper = tp_weight_mesh_mapper(mesh_device, shard_dim=1, cfg=self._tp)  # shard input (intermediate)
        else:
            w_gate_up_host = np.concatenate([w_gate_host, w_up_host], axis=0)
            gate_up_mapper = mapper
            down_mapper = mapper

        self.w_gate_up = ttnn.as_tensor(
            w_gate_up_host,
            device=mesh_device,
            dtype=w_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=gate_up_mapper,
        )
        self.w_down = ttnn.as_tensor(
            _maybe_get(state_dict, f"{base_address}.down_proj.weight"),
            device=mesh_device,
            dtype=w_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=down_mapper,
        )

        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)

        self._linear_ck = linear_compute_kernel_config
        self._act_l1 = activation_l1_memory_config
        self._linear_out_l1 = linear_output_l1_memory_config
        self._fused_gate_up_pc_cache: dict = {}
        self._down_pc_cache: dict = {}

    def _l1_activation(self, t):
        if self._act_l1 is None:
            return t
        return self.ttnn.to_memory_config(t, self._act_l1)

    def _fused_gate_up_linear_kwargs(self, *, batch_size: int, seq_len: int) -> dict:
        """LoFi + L1 program config for fused ``gate_proj`` + ``up_proj`` (one matmul)."""
        kw: dict = {}
        if self._linear_ck is not None:
            kw["compute_kernel_config"] = self._linear_ck
        key = (int(batch_size), int(seq_len))
        pc = self._fused_gate_up_pc_cache.get(key)
        if pc is None:
            pc = ace_step_dit_mlp_fused_gate_up_linear_program_config(
                self.mesh_device,
                seq_len=int(seq_len),
                hidden_size=self.hidden_size,
                intermediate_size=self._local_intermediate,
                batch_size=int(batch_size),
            )
            if pc is not None:
                self._fused_gate_up_pc_cache[key] = pc
        if pc is not None:
            kw["program_config"] = pc
        _dram = getattr(self.ttnn, "DRAM_MEMORY_CONFIG", None)
        kw["memory_config"] = ace_step_linear_kwargs_memory_config(pc, linear_out_l1=self._linear_out_l1, dram=_dram)
        return kw

    def _down_linear_kwargs(self, *, batch_size: int, seq_len: int) -> dict:
        """LoFi + L1 + 1D-mcast program config for ``down_proj`` (intermediate→hidden)."""
        kw: dict = {}
        if self._linear_ck is not None:
            kw["compute_kernel_config"] = self._linear_ck
        key = (int(batch_size), int(seq_len))
        pc = self._down_pc_cache.get(key)
        if pc is None:
            pc = ace_step_dit_mlp_down_proj_linear_program_config(
                self.mesh_device,
                seq_len=int(seq_len),
                intermediate_size=self._local_intermediate,
                hidden_size=self.hidden_size,
                batch_size=int(batch_size),
            )
            if pc is not None:
                self._down_pc_cache[key] = pc
        if pc is not None:
            kw["program_config"] = pc
        _dram = getattr(self.ttnn, "DRAM_MEMORY_CONFIG", None)
        kw["memory_config"] = ace_step_linear_kwargs_memory_config(pc, linear_out_l1=self._linear_out_l1, dram=_dram)
        return kw

    def __call__(self, x, *, debug: Optional[dict] = None, debug_prefix: str = "", use_dram_activations: bool = False):
        ttnn = self.ttnn
        x = ace_step_ensure_tile_layout(ttnn, x)
        b_x = int(x.shape[0])
        s = int(x.shape[2])
        _dram_mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        lin_gu = self._fused_gate_up_linear_kwargs(batch_size=b_x, seq_len=s)
        lin_down = self._down_linear_kwargs(batch_size=b_x, seq_len=s)
        _use_l1 = not use_dram_activations and "program_config" in lin_gu
        _use_l1_down = not use_dram_activations and "program_config" in lin_down
        if use_dram_activations:
            for lin in (lin_gu, lin_down):
                lin.pop("program_config", None)
                if _dram_mc is not None:
                    lin["memory_config"] = _dram_mc

            def _act_fn(t):
                return ace_step_ensure_dram_activation(ttnn, t, _dram_mc)

            _elt_mc = _dram_mc
        else:

            def _act_fn(t):
                return self._l1_activation(t)

            _elt_mc = self._act_l1 if _use_l1 else _dram_mc
        x_lin = ace_step_matmul_activation(ttnn, x, lin_gu, l1_fn=self._l1_activation, dram_mc=_dram_mc)
        gate_up = ttnn.linear(x_lin, self.w_gate_up, bias=None, transpose_b=True, **lin_gu)
        # Local intermediate under TP (== full when off): gate/up are the per-chip halves.
        i_sz = int(self._local_intermediate)
        gate = ttnn.slice(gate_up, (0, 0, 0, 0), (b_x, 1, s, i_sz))
        up = ttnn.slice(gate_up, (0, 0, 0, i_sz), (b_x, 1, s, 2 * i_sz))
        ace_step_safe_deallocate(ttnn, gate_up)
        if debug is not None and debug.get("enabled", False):
            debug[f"{debug_prefix}gate_lin"] = gate
            debug[f"{debug_prefix}up_lin"] = up
        gate = (
            ttnn.silu(gate, memory_config=_elt_mc) if hasattr(ttnn, "silu") else ttnn.gelu(gate, memory_config=_elt_mc)
        )
        if debug is not None and debug.get("enabled", False):
            debug[f"{debug_prefix}gate_act"] = gate
        if _use_l1:
            gate = _act_fn(gate)
            up = _act_fn(up)
        h = ttnn.multiply(gate, up, memory_config=_elt_mc)
        _bf16 = getattr(ttnn, "bfloat16", None)
        if _bf16 is not None:
            h = ttnn.typecast(h, _bf16, memory_config=_elt_mc)
        if _use_l1 or _use_l1_down:
            h = _act_fn(h)
        h_lin = ace_step_matmul_activation(ttnn, h, lin_down, l1_fn=self._l1_activation, dram_mc=_dram_mc)
        out = ttnn.linear(h_lin, self.w_down, bias=None, transpose_b=True, **lin_down)
        # Row-parallel down_proj: each chip produced a partial sum over its intermediate shard;
        # all-reduce across the TP axis to reconstruct the full hidden output on every chip.
        if self._tp.enabled and self._tp.degree > 1:
            from models.experimental.ace_step_v1_5.ttnn_impl.tp_config import tp_all_reduce

            out = tp_all_reduce(out, self.mesh_device, cfg=self._tp)
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
        linear_compute_kernel_config=None,
        activation_l1_memory_config=None,
        linear_output_l1_memory_config=None,
    ) -> None:
        self.ttnn = ttnn
        self.mesh_device = mesh_device
        self.layer_idx = int(layer_idx)
        self.dtype = dtype or getattr(ttnn, "bfloat16", None) or getattr(ttnn, "float16", None)
        if self.dtype is None:
            raise RuntimeError("TTNN build missing a usable dtype (bfloat16/float16)")

        d = int(cfg.hidden_size)
        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ace_step_dit_weight_mesh_mapper(mesh_device)

        # Norm weights stay DRAM (small); rms_norm outputs use ``memory_config=_el_mc`` for activations.
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

        self._act_l1 = activation_l1_memory_config
        self._linear_out_l1 = linear_output_l1_memory_config

        _attn_kw = dict(
            linear_compute_kernel_config=linear_compute_kernel_config,
            activation_l1_memory_config=activation_l1_memory_config,
            linear_output_l1_memory_config=linear_output_l1_memory_config,
        )
        # Attention modules
        self.self_attn = TtAceStepAttentionSDPA(
            cfg=cfg,
            state_dict=state_dict,
            base_address=f"layers.{layer_idx}.self_attn",
            mesh_device=mesh_device,
            dtype=self.dtype,
            rotary_embedding=rotary_embedding,
            **_attn_kw,
        )
        self.cross_attn = TtAceStepAttentionSDPA(
            cfg=cfg,
            state_dict=state_dict,
            base_address=f"layers.{layer_idx}.cross_attn",
            mesh_device=mesh_device,
            dtype=self.dtype,
            rotary_embedding=None,
            **_attn_kw,
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
            **_attn_kw,
        )

        # Scale-shift table: tiny [1,6,D] parameter — host in L1 so AdaLN ``add`` does not read
        # ``in0`` from DRAM interleaved (Tracy bucket ``BinaryNgDeviceOperation (in0:dram_interleaved)``).
        sst = _maybe_get(state_dict, f"layers.{layer_idx}.scale_shift_table")
        _sst_mc = ace_step_dit_linear_l1_memory_config(ttnn)
        if _sst_mc is None:
            _sst_mc = mem
        self.scale_shift_table = ttnn.as_tensor(
            sst,
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=_sst_mc,
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
        use_dram_activations: bool = False,
    ):
        """
        Args:
            hidden_states: [B, 1, S, D] TILE/ROW_MAJOR ok
            timestep_proj_b6d: [B, 6, D] TILE (from ``TtTimestepEmbedding`` + pipeline). ROW_MAJOR is not used:
            it forces DRAM placement and Tracy labels AdaLN BinaryNg inputs as DRAM-backed.
            encoder_hidden_states: [B, 1, S_enc, D]
        """
        ttnn = self.ttnn
        hidden_states = ace_step_ensure_tile_layout(ttnn, hidden_states)
        encoder_hidden_states = ace_step_ensure_tile_layout(ttnn, encoder_hidden_states)
        b = int(hidden_states.shape[0])
        _sr = ace_step_reshape_kwargs(ttnn)

        # Short clips: L1 activations + 1D-mcast matmul. Long clips: DRAM-only (no L1/DRAM mix).
        _dram_mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        if use_dram_activations:
            _el_mc = _dram_mc
            _bin_kw = ace_step_binary_kwargs(ttnn, _dram_mc)

            def _l1_act(t):
                return ace_step_ensure_dram_activation(ttnn, t, _dram_mc)

        else:
            _el_mc = self._act_l1 or self._linear_out_l1 or _dram_mc
            _bin_kw = ace_step_binary_kwargs(ttnn, _el_mc)

            def _l1_act(t):
                return ace_step_ensure_l1_activation(ttnn, t, _el_mc)

        hidden_states = _l1_act(hidden_states)
        if len(encoder_hidden_states.shape) == 3:
            encoder_hidden_states = ttnn.unsqueeze(encoder_hidden_states, 1)
        encoder_hidden_states = _l1_act(encoder_hidden_states)

        temb = ace_step_ensure_tile_layout(ttnn, timestep_proj_b6d)
        temb = _l1_act(temb)
        if tuple(temb.shape) != (b, 6, int(hidden_states.shape[-1])):
            raise ValueError(f"Expected timestep_proj [B,6,D], got {tuple(temb.shape)}")

        # (scale_shift_table + temb) -> chunk 6 along dim=1 => each [B,1,D]
        # Commutative add; put timestep activation first so Tracy tags ``in0`` as the activation buffer
        # (and after L1 placement above both operands can be L1-interleaved).
        sst = _l1_act(ttnn.add(temb, self.scale_shift_table, **_bin_kw))
        d = int(hidden_states.shape[-1])

        def chunk(i: int):
            c = ttnn.slice(sst, (0, i, 0), (b, i + 1, d))
            c = ttnn.reshape(c, (b, 1, 1, d), **_sr)
            return _l1_act(c)

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

        _rms_kw = ace_step_dit_rms_norm_kwargs(ttnn, _el_mc, device=self.mesh_device)

        # Self-attn AdaLN: norm(x) * (1+scale) + shift
        x_norm = ttnn.rms_norm(
            hidden_states,
            weight=self.self_norm_w,
            epsilon=self.eps,
            **_rms_kw,
        )
        x_norm = _l1_act(x_norm)
        one_plus = _l1_act(ace_step_add_one(ttnn, scale_msa, **_bin_kw))
        x_scaled = _l1_act(ttnn.multiply(x_norm, one_plus, **_bin_kw))
        h = _l1_act(ttnn.add(x_scaled, shift_msa, **_bin_kw))
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
            use_dram_activations=use_dram_activations,
        )
        attn_out = _l1_act(attn_out)
        gated = _l1_act(ttnn.multiply(attn_out, gate_msa, **_bin_kw))
        hidden_states = _l1_act(ttnn.add(_l1_act(hidden_states), gated, **_bin_kw))
        if debug is not None and debug.get("enabled", False):
            debug[f"{core_pfx}after_self"] = hidden_states

        x2 = ttnn.rms_norm(
            _l1_act(hidden_states),
            weight=self.cross_norm_w,
            epsilon=self.eps,
            **_rms_kw,
        )
        x2 = _l1_act(x2)
        cross_dbg_pfx = f"{core_pfx}cross." if (debug is not None and debug.get("enabled", False)) else ""
        ca = self.cross_attn(
            x2,
            encoder_hidden_states=encoder_hidden_states,
            is_causal=False,
            attn_mask=encoder_attention_mask_b1qk,
            debug=debug,
            debug_prefix=cross_dbg_pfx,
            use_dram_activations=use_dram_activations,
        )
        ca = _l1_act(ca)
        hidden_states = _l1_act(ttnn.add(_l1_act(hidden_states), ca, **_bin_kw))
        if debug is not None and debug.get("enabled", False):
            debug[f"{core_pfx}cross.attn_out"] = ca
            debug[f"{core_pfx}after_cross"] = hidden_states

        x3 = ttnn.rms_norm(
            hidden_states,
            weight=self.mlp_norm_w,
            epsilon=self.eps,
            **_rms_kw,
        )
        if debug is not None and debug.get("enabled", False):
            debug[f"{core_pfx}mlp_norm_out"] = x3
        x3 = _l1_act(x3)
        one_plus2 = _l1_act(ace_step_add_one(ttnn, c_scale, **_bin_kw))
        x3_scaled = _l1_act(ttnn.multiply(x3, one_plus2, **_bin_kw))
        h3 = _l1_act(ttnn.add(x3_scaled, c_shift, **_bin_kw))
        if debug is not None and debug.get("enabled", False):
            debug[f"{core_pfx}mlp_in"] = h3
        mlp_pfx = f"{core_pfx}mlp." if (debug is not None and debug.get("enabled", False)) else ""
        ff = _l1_act(
            self.mlp(
                h3,
                debug=debug if (debug is not None and debug.get("enabled", False)) else None,
                debug_prefix=mlp_pfx,
                use_dram_activations=use_dram_activations,
            )
        )
        if debug is not None and debug.get("enabled", False):
            debug[f"{core_pfx}mlp_out"] = ff
        ff = _l1_act(ttnn.multiply(ff, c_gate, **_bin_kw))
        if debug is not None and debug.get("enabled", False):
            debug[f"{core_pfx}mlp_out_gated"] = ff
        hidden_states = _l1_act(ttnn.add(_l1_act(hidden_states), ff, **_bin_kw))
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
        self.ttnn = ttnn
        self.mesh_device = mesh_device
        self.dtype = dtype or getattr(ttnn, "bfloat16", None) or getattr(ttnn, "float16", None)
        if self.dtype is None:
            raise RuntimeError("TTNN build missing a usable dtype (bfloat16/float16)")

        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ace_step_dit_weight_mesh_mapper(mesh_device)

        print("[ace_step_v1_5] DiT core: condition embedder …", flush=True)
        _w_dtype = ace_step_dit_weight_dtype(ttnn, self.dtype)
        self.cond_w = ttnn.as_tensor(
            _maybe_get(state_dict, "condition_embedder.weight"),
            device=mesh_device,
            dtype=_w_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        self.cond_b = ttnn.as_tensor(
            _maybe_get(state_dict, "condition_embedder.bias").reshape(1, 1, 1, -1),
            device=mesh_device,
            dtype=_w_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        if ace_step_device_num_chips(mesh_device) > 1:
            ace_step_synchronize_device(ttnn, mesh_device)
        print("[ace_step_v1_5] DiT core: condition embedder ready", flush=True)

        print("[ace_step_v1_5] DiT core: RoPE cache …", flush=True)
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
        print("[ace_step_v1_5] DiT core: RoPE ready", flush=True)
        if ace_step_device_num_chips(mesh_device) > 1:
            ace_step_synchronize_device(ttnn, mesh_device)

        linear_ck = ace_step_init_dit_linear_compute_kernel_config(mesh_device)
        l1_mc = ace_step_dit_linear_l1_memory_config(ttnn)
        self._linear_ck = linear_ck
        self._l1_mc = l1_mc
        self._cond_embed_pc_cache: dict = {}

        num_layers = int(cfg.num_hidden_layers)
        _init_sync = _ace_step_env_truthy("ACE_STEP_DIT_INIT_SYNC")
        self.layers: List[TtAceStepDiTLayer] = []
        for i in range(num_layers):
            print(f"[ace_step_v1_5] DiT core: layer {i + 1}/{num_layers} …", flush=True)
            self.layers.append(
                TtAceStepDiTLayer(
                    cfg=cfg,
                    state_dict=state_dict,
                    layer_idx=i,
                    mesh_device=mesh_device,
                    dtype=self.dtype,
                    rotary_embedding=self._rotary,
                    linear_compute_kernel_config=linear_ck,
                    activation_l1_memory_config=l1_mc,
                    linear_output_l1_memory_config=l1_mc,
                )
            )
            if _init_sync or (i == 0 and ace_step_device_num_chips(mesh_device) > 1):
                ace_step_synchronize_device(ttnn, mesh_device)
        print(f"[ace_step_v1_5] DiT core: {num_layers} layers uploaded", flush=True)
        if ace_step_device_num_chips(mesh_device) > 1:
            ace_step_synchronize_device(ttnn, mesh_device)

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
        # encoder_hidden_states: [B, S_enc, D] or [B, 1, S_enc, D] -> [B, 1, S_enc, D] TILE (no ROW_MAJOR hop).
        l1_mc = ace_step_dit_linear_l1_memory_config(ttnn)
        enc = ace_step_ensure_dit_activation(ttnn, encoder_hidden_states, l1_mc)
        if len(enc.shape) == 3:
            enc = ttnn.unsqueeze(enc, 1)
        # Build program_config for condition_embedder (e.g. 256×2048→1024).
        # Do not pass L1 ``memory_config`` into ``linear`` here: bias broadcast validation fails
        # on small test shapes (e.g. ``[1,1,4,128]`` @ ``[256,128]`` + ``[1,1,1,256]`` bias).
        _b_enc = int(enc.shape[0])
        _s_enc = int(enc.shape[2])
        _d_enc = int(enc.shape[-1])
        _d_out = int(self.cond_w.shape[0])
        _ce_key = (_b_enc, _s_enc, _d_enc, _d_out)
        _ce_pc = self._cond_embed_pc_cache.get(_ce_key)
        if _ce_pc is None:
            _ce_pc = ace_step_dense_linear_program_config(
                self.mesh_device,
                seq_len=_s_enc,
                in_dim=_d_enc,
                out_dim=_d_out,
                batch_size=_b_enc,
            )
            if _ce_pc is not None:
                self._cond_embed_pc_cache[_ce_key] = _ce_pc
        _ce_kw: dict = {}
        if self._linear_ck is not None:
            _ce_kw["compute_kernel_config"] = self._linear_ck
        if _ce_pc is not None:
            _ce_kw["program_config"] = _ce_pc
        if l1_mc is not None:
            _ce_kw["memory_config"] = l1_mc
        enc = ttnn.linear(enc, self.cond_w, bias=self.cond_b, transpose_b=True, **_ce_kw)
        if l1_mc is not None:
            enc = ace_step_ensure_l1_activation(ttnn, enc, l1_mc)
        if debug is not None and debug.get("enabled", False):
            debug["core.enc_conditioned"] = enc

        x = ace_step_ensure_dit_activation(ttnn, hidden_states_patches, l1_mc)
        if len(x.shape) == 3:
            x = ttnn.unsqueeze(x, 1)  # [B,1,S,D]
        _dram_mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        _use_dram_act = ace_step_dit_prefers_dram_activations(
            batch_size=int(x.shape[0]),
            seq_len=int(x.shape[2]),
        )
        if _use_dram_act and _dram_mc is not None:
            x = ace_step_ensure_dram_activation(ttnn, x, _dram_mc)
            enc = ace_step_ensure_dram_activation(ttnn, enc, _dram_mc)
        elif l1_mc is not None:
            enc = ace_step_ensure_l1_activation(ttnn, enc, l1_mc)
        if debug is not None and debug.get("enabled", False):
            debug["core.x_patches_in"] = x
        # Periodic device-profiler drain (no-op without TT_METAL_DEVICE_PROFILER / TTNN_OP_PROFILER).
        # Default flushes every layer to keep the 12000-marker per-RISC ring buffer from overflowing.
        try:
            _layer_flush_every = int(os.environ.get("ACE_STEP_PROFILER_FLUSH_EVERY_LAYER", "1"))
        except ValueError:
            _layer_flush_every = 1
        if _layer_flush_every < 0:
            _layer_flush_every = 0
        for _layer_idx, layer in enumerate(self.layers):
            x = layer(
                x,
                timestep_proj_b6d,
                enc,
                encoder_attention_mask_b1qk=encoder_attention_mask_b1qk,
                self_attention_mask_b1qq=self_attention_mask_b1qq,
                debug=debug,
                use_dram_activations=_use_dram_act,
            )
            if _layer_flush_every and ((_layer_idx + 1) % _layer_flush_every) == 0:
                _ace_step_flush_device_profiler(ttnn, self.mesh_device)
        x = ttnn.squeeze(x, 1)  # [B,S,D]
        if debug is not None and debug.get("enabled", False):
            debug["core.out"] = x
        return x
