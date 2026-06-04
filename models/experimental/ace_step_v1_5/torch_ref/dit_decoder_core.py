from __future__ import annotations

"""
Torch reference of the ACE-Step v1.5 DiT decoder core.

This module provides small, explicit reference implementations used by TTNN PCC
tests. The reference mirrors the HF behavior at a functional level:
- Qwen3-style RMSNorm
- Qwen3-style rotary embeddings (RoPE) via transformers' helpers
- Qwen3 MLP (SiLU-gated)
- ACE-Step DiT layer modulation via `scale_shift_table` and timestep projection

The weight key naming matches the TTNN implementation (`ttnn_impl/dit_decoder_core.py`)
and the unit tests in `tests/test_pcc_dit_decoder_core.py`.
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding, apply_rotary_pos_emb

from models.experimental.ace_step_v1_5.ttnn_impl.dit_decoder_core import AceStepDecoderConfigTTNN


def _torch_env_truthy(name: str) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on", "y")


def _torch_attn_trace_print(prefix: Optional[str]) -> bool:
    """Same layer filter as TTNN: ACE_STEP_DEBUG_ATTN_TRACE / ACE_STEP_DEBUG_ATTN_TRACE_LAYER."""
    if not _torch_env_truthy("ACE_STEP_DEBUG_ATTN_TRACE"):
        return False
    layer_filt = (os.environ.get("ACE_STEP_DEBUG_ATTN_TRACE_LAYER") or "").strip()
    if not layer_filt:
        return True
    needle = f"layer{layer_filt}."
    return prefix is not None and needle in prefix


def _torch_log_tensor(tag: str, t: torch.Tensor) -> None:
    xf = t.detach().float().reshape(-1)
    m = torch.isfinite(xf)
    if not bool(m.any()):
        print(f"[ace_step_v1_5][attn_trace][torch] {tag} shape={tuple(t.shape)} NO_FINITE", flush=True)
        return
    xf = xf[m]
    print(
        f"[ace_step_v1_5][attn_trace][torch] {tag} shape={tuple(t.shape)} "
        f"min={float(xf.min()):.6g} max={float(xf.max()):.6g} "
        f"mean={float(xf.mean()):.6g} std={float(xf.std(unbiased=False)):.6g} n={int(xf.numel())}",
        flush=True,
    )


def rmsnorm_qwen3(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + float(eps))
    return (x_f * weight.float()).to(x.dtype)


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * float(scale)
    attn = torch.softmax(attn, dim=-1)
    ctx = torch.matmul(attn, v.float())
    return ctx.to(q.dtype)


def _attention_sdpa(
    x: torch.Tensor,
    *,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    n_heads: int,
    head_dim: int,
    eps: float,
    encoder_hidden_states: torch.Tensor | None,
    n_kv_heads: int | None = None,
    rope_cos: torch.Tensor | None = None,
    rope_sin: torch.Tensor | None = None,
    trace_prefix: Optional[str] = None,
) -> torch.Tensor:
    # x: [B,1,S,D]
    _trace = _torch_attn_trace_print(trace_prefix)
    scale_attn = 1.0 / math.sqrt(float(head_dim))
    if _trace:
        print(
            f"[ace_step_v1_5][attn_trace][torch] {trace_prefix or ''}enter "
            f"cross={encoder_hidden_states is not None} scale={scale_attn:.6g}",
            flush=True,
        )

    if encoder_hidden_states is None:
        k_in = x
        v_in = x
    else:
        k_in = encoder_hidden_states
        v_in = encoder_hidden_states

    # Keep matmuls in FP32 for determinism across CPU builds, cast back to BF16.
    q = torch.matmul(x.float(), wq.float().t()).to(x.dtype)  # [B,1,S,H*Dh]
    k = torch.matmul(k_in.float(), wk.float().t()).to(x.dtype)
    v = torch.matmul(v_in.float(), wv.float().t()).to(x.dtype)
    if _trace:
        _torch_log_tensor(f"{trace_prefix or ''}q_after_linear_B1SD", q)
        _torch_log_tensor(f"{trace_prefix or ''}k_after_linear_B1SD", k)
        _torch_log_tensor(f"{trace_prefix or ''}v_after_linear_B1SD", v)

    b, _one, s, _d = q.shape
    h = int(n_heads)
    kv = int(n_kv_heads) if n_kv_heads is not None else h
    dh = int(head_dim)
    s_k = int(k.shape[2])

    q = q.view(b, 1, s, h, dh).permute(0, 3, 2, 4, 1).reshape(b, h, s, dh)
    k = k.view(b, 1, s_k, kv, dh).permute(0, 3, 2, 4, 1).reshape(b, kv, s_k, dh)
    v = v.view(b, 1, s_k, kv, dh).permute(0, 3, 2, 4, 1).reshape(b, kv, s_k, dh)
    if kv != h:
        if h % kv != 0:
            raise ValueError(f"n_heads {h} must be divisible by n_kv_heads {kv}")
        rep = h // kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    if _trace:
        _torch_log_tensor(f"{trace_prefix or ''}q_after_reshape_BHSD", q)
        print(
            f"[ace_step_v1_5][attn_trace][torch] {trace_prefix or ''}kv_pre_norm "
            f"S_q={s} S_k={s_k} k_shape={tuple(k.shape)} v_shape={tuple(v.shape)}",
            flush=True,
        )
        _torch_log_tensor(f"{trace_prefix or ''}k_pre_rmsnorm", k)
        _torch_log_tensor(f"{trace_prefix or ''}v_pre_rmsnorm", v)

    q = rmsnorm_qwen3(q, q_norm_w, eps)
    k = rmsnorm_qwen3(k, k_norm_w, eps)

    if _trace:
        _torch_log_tensor(f"{trace_prefix or ''}q_after_rmsnorm", q)
        _torch_log_tensor(f"{trace_prefix or ''}k_after_rmsnorm", k)

    if encoder_hidden_states is None and rope_cos is not None and rope_sin is not None:
        if _trace:
            print(f"[ace_step_v1_5][attn_trace][torch] {trace_prefix or ''}rope apply_rotary_pos_emb", flush=True)
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin, unsqueeze_dim=1)
        if _trace:
            _torch_log_tensor(f"{trace_prefix or ''}q_after_rope", q)
            _torch_log_tensor(f"{trace_prefix or ''}k_after_rope", k)

    scale = 1.0 / math.sqrt(float(dh))
    ctx = _sdpa(q, k, v, scale)  # [B,H,S,Dh]
    if _trace:
        _torch_log_tensor(f"{trace_prefix or ''}ctx_after_sdpa_BHSD", ctx)
        print(
            f"[ace_step_v1_5][attn_trace][torch] {trace_prefix or ''}sdpa_out S_ctx={s} (matches S_q)",
            flush=True,
        )
    ctx = ctx.permute(0, 2, 1, 3).reshape(b, 1, s, h * dh)
    out = torch.matmul(ctx.float(), wo.float().t()).to(x.dtype)  # [B,1,S,D]
    if _trace:
        _torch_log_tensor(f"{trace_prefix or ''}attn_out_B1SD", out)
    return out


def qwen3_mlp(x: torch.Tensor, *, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    gate = torch.matmul(x, w_gate.t())
    up = torch.matmul(x, w_up.t())
    gate = torch.nn.functional.silu(gate.float()).to(x.dtype)
    h = gate * up
    return torch.matmul(h, w_down.t())


def adaln_modulate_torch(
    x: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """ACE DiT AdaLN-style modulation used in ``TtAceStepDiTLayer``: norm(x) * (1 + scale) + shift."""
    x_norm = rmsnorm_qwen3(x, norm_w, eps)
    return x_norm * (1.0 + scale) + shift


def make_tiny_state_dict(
    *,
    d_model: int,
    n_heads: int,
    head_dim: int,
    cond_dim: int,
    intermediate: int,
    num_layers: int,
    n_kv_heads: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Generate a synthetic (tiny) state dict with the key structure expected by
    `TorchAceStepDiTCoreRef` and `TtAceStepDiTCore`.
    """
    torch.manual_seed(0)
    n_kv = int(n_kv_heads) if n_kv_heads is not None else int(n_heads)

    def r(shape):
        return torch.randn(*shape, dtype=torch.bfloat16)

    sd: dict[str, np.ndarray] = {}

    sd["condition_embedder.weight"] = r((d_model, cond_dim)).float().cpu().numpy()
    sd["condition_embedder.bias"] = r((d_model,)).float().cpu().numpy()

    for i in range(int(num_layers)):
        for attn_name in ("self_attn", "cross_attn"):
            base = f"layers.{i}.{attn_name}"
            sd[f"{base}.q_proj.weight"] = r((n_heads * head_dim, d_model)).float().cpu().numpy()
            sd[f"{base}.k_proj.weight"] = r((n_kv * head_dim, d_model)).float().cpu().numpy()
            sd[f"{base}.v_proj.weight"] = r((n_kv * head_dim, d_model)).float().cpu().numpy()
            sd[f"{base}.o_proj.weight"] = r((d_model, n_heads * head_dim)).float().cpu().numpy()
            sd[f"{base}.q_norm.weight"] = r((head_dim,)).float().cpu().numpy()
            sd[f"{base}.k_norm.weight"] = r((head_dim,)).float().cpu().numpy()

        sd[f"layers.{i}.self_attn_norm.weight"] = r((d_model,)).float().cpu().numpy()
        sd[f"layers.{i}.cross_attn_norm.weight"] = r((d_model,)).float().cpu().numpy()
        sd[f"layers.{i}.mlp_norm.weight"] = r((d_model,)).float().cpu().numpy()

        sd[f"layers.{i}.mlp.gate_proj.weight"] = r((intermediate, d_model)).float().cpu().numpy()
        sd[f"layers.{i}.mlp.up_proj.weight"] = r((intermediate, d_model)).float().cpu().numpy()
        sd[f"layers.{i}.mlp.down_proj.weight"] = r((d_model, intermediate)).float().cpu().numpy()

        sd[f"layers.{i}.scale_shift_table"] = r((1, 6, d_model)).float().cpu().numpy()

    return sd


def make_time_embed_state_dict(*, hidden_size: int) -> dict[str, np.ndarray]:
    torch.manual_seed(0)

    def r(shape):
        return torch.randn(*shape, dtype=torch.float32).cpu().numpy()

    in_ch = 256
    sd: dict[str, np.ndarray] = {}
    for base in ("time_embed", "time_embed_r"):
        sd[f"{base}.linear_1.weight"] = r((hidden_size, in_ch))
        sd[f"{base}.linear_1.bias"] = r((hidden_size,))
        sd[f"{base}.linear_2.weight"] = r((hidden_size, hidden_size))
        sd[f"{base}.linear_2.bias"] = r((hidden_size,))
        sd[f"{base}.time_proj.weight"] = r((6 * hidden_size, hidden_size))
        sd[f"{base}.time_proj.bias"] = r((6 * hidden_size,))
    return sd


@dataclass(frozen=True)
class TorchTimestepEmbeddingRef:
    hidden_size: int
    state_dict: dict[str, np.ndarray]
    base: str
    timesteps_host: np.ndarray
    scale: float = 1000.0

    def __call__(self, timestep_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        in_ch = 256
        t = torch.from_numpy(self.timesteps_host.astype(np.float32)) * float(self.scale)  # [N]
        half = in_ch // 2
        freqs = torch.exp((-math.log(10000.0)) * (torch.arange(0, half, dtype=torch.float32) / float(half)))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [N,256]

        t_freq = emb[int(timestep_index) : int(timestep_index) + 1, :].view(1, 1, 1, in_ch).to(torch.bfloat16)

        sd = self.state_dict
        base = self.base
        hidden_size = int(self.hidden_size)

        w1 = torch.from_numpy(sd[f"{base}.linear_1.weight"]).to(torch.bfloat16)
        b1 = torch.from_numpy(sd[f"{base}.linear_1.bias"]).to(torch.bfloat16).view(1, 1, 1, hidden_size)
        w2 = torch.from_numpy(sd[f"{base}.linear_2.weight"]).to(torch.bfloat16)
        b2 = torch.from_numpy(sd[f"{base}.linear_2.bias"]).to(torch.bfloat16).view(1, 1, 1, hidden_size)
        wt = torch.from_numpy(sd[f"{base}.time_proj.weight"]).to(torch.bfloat16)
        bt = torch.from_numpy(sd[f"{base}.time_proj.bias"]).to(torch.bfloat16).view(1, 1, 1, 6 * hidden_size)

        temb = torch.matmul(t_freq, w1.t()) + b1
        temb = torch.nn.functional.silu(temb.float()).to(torch.bfloat16)
        temb = torch.matmul(temb, w2.t()) + b2  # [1,1,1,D]

        h = torch.nn.functional.silu(temb.float()).to(torch.bfloat16)
        tp = torch.matmul(h, wt.t()) + bt  # [1,1,1,6D]
        tp = tp.view(1, 6, 1, hidden_size).view(1, 6, hidden_size)
        temb2 = temb.view(1, hidden_size)
        return temb2, tp

    def from_timestep_value(self, t_value: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Same MLP head as :meth:`__call__`, but driven by a continuous timestep delta
        (e.g. ``t - r`` in HF ``time_embed_r``), not an index into ``timesteps_host``.
        """
        in_ch = 256
        hidden_size = int(self.hidden_size)
        t_scalar = torch.tensor(float(t_value) * float(self.scale), dtype=torch.float32)
        half = in_ch // 2
        freqs = torch.exp((-math.log(10000.0)) * (torch.arange(0, half, dtype=torch.float32) / float(half)))
        args = t_scalar * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        t_freq = emb.view(1, 1, 1, in_ch).to(torch.bfloat16)

        sd = self.state_dict
        base = self.base

        w1 = torch.from_numpy(sd[f"{base}.linear_1.weight"]).to(torch.bfloat16)
        b1 = torch.from_numpy(sd[f"{base}.linear_1.bias"]).to(torch.bfloat16).view(1, 1, 1, hidden_size)
        w2 = torch.from_numpy(sd[f"{base}.linear_2.weight"]).to(torch.bfloat16)
        b2 = torch.from_numpy(sd[f"{base}.linear_2.bias"]).to(torch.bfloat16).view(1, 1, 1, hidden_size)
        wt = torch.from_numpy(sd[f"{base}.time_proj.weight"]).to(torch.bfloat16)
        bt = torch.from_numpy(sd[f"{base}.time_proj.bias"]).to(torch.bfloat16).view(1, 1, 1, 6 * hidden_size)

        temb = torch.matmul(t_freq, w1.t()) + b1
        temb = torch.nn.functional.silu(temb.float()).to(torch.bfloat16)
        temb = torch.matmul(temb, w2.t()) + b2
        h = torch.nn.functional.silu(temb.float()).to(torch.bfloat16)
        tp = torch.matmul(h, wt.t()) + bt
        tp = tp.view(1, 6, 1, hidden_size).view(1, 6, hidden_size)
        temb2 = temb.view(1, hidden_size)
        return temb2, tp


class TorchAceStepDiTCoreRef:
    def __init__(self, *, cfg: AceStepDecoderConfigTTNN, state_dict: dict[str, np.ndarray]) -> None:
        self.cfg = cfg
        self.sd = state_dict
        # Cache host->device converted weights (keyed by (param_key, device_str)).
        self._wcache: dict[tuple[str, str], torch.Tensor] = {}
        # Cache RoPE modules per device (device_str -> module).
        self._rope_cache: dict[str, Qwen3RotaryEmbedding] = {}

    def _devkey(self, device: torch.device) -> str:
        # torch.device is not hash-stable across processes; normalize to string.
        return str(device)

    def _w(self, key: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ck = (key, self._devkey(device))
        t = self._wcache.get(ck)
        if t is None:
            t = torch.from_numpy(self.sd[key]).to(device=device, dtype=dtype)
            self._wcache[ck] = t
        return t

    def _rope(
        self, *, d_model: int, n_heads: int, head_dim: int, max_seq_len: int, device: torch.device
    ) -> Qwen3RotaryEmbedding:
        dk = self._devkey(device)
        rope = self._rope_cache.get(dk)
        if rope is None:
            qc = Qwen3Config(
                vocab_size=8192,
                hidden_size=int(d_model),
                intermediate_size=max(int(d_model), 512),
                num_hidden_layers=1,
                num_attention_heads=int(n_heads),
                num_key_value_heads=int(self.cfg.num_key_value_heads),
                head_dim=int(head_dim),
                max_position_embeddings=int(max_seq_len),
                rope_theta=float(self.cfg.rope_theta),
            )
            rope = Qwen3RotaryEmbedding(qc).to(device=device)
            self._rope_cache[dk] = rope
        return rope

    @staticmethod
    def _debug_put(debug: Optional[dict[str, Any]], key: str, t: torch.Tensor) -> None:
        if debug is None or not debug.get("enabled", False):
            return
        debug[key] = t.detach().float().cpu().clone()

    def forward(
        self,
        *,
        x_patches: torch.Tensor,  # [B,S,D]
        timestep_proj_b6d: torch.Tensor,  # [B,6,D]
        encoder_hidden_states: torch.Tensor,  # [B,S_enc,cond_dim]
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        debug: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        b, s, d = x_patches.shape
        eps = float(self.cfg.rms_norm_eps)
        h = int(self.cfg.num_attention_heads)
        dh = int(self.cfg.head_dim)

        # Reference weights are loaded as BF16; host conditioning may be float32 (e.g. lightweight text path).
        compute_dtype = torch.bfloat16
        device = x_patches.device
        x_patches = x_patches.to(dtype=compute_dtype, device=device)
        timestep_proj_b6d = timestep_proj_b6d.to(dtype=compute_dtype, device=device)
        encoder_hidden_states = encoder_hidden_states.to(dtype=compute_dtype, device=device)

        cond_w = self._w("condition_embedder.weight", device=device, dtype=compute_dtype)
        cond_b = self._w("condition_embedder.bias", device=device, dtype=compute_dtype)

        enc = encoder_hidden_states.unsqueeze(1)  # [B,1,S_enc,cond_dim]
        enc = torch.matmul(enc, cond_w.t()) + cond_b.view(1, 1, 1, d)
        self._debug_put(debug, "core.enc_conditioned", enc.squeeze(1))

        x = x_patches.unsqueeze(1)  # [B,1,S,D]
        self._debug_put(debug, "core.x_patches_in", x.squeeze(1))

        if rope_cos is None or rope_sin is None:
            rope = self._rope(d_model=d, n_heads=h, head_dim=dh, max_seq_len=max(s, 512), device=device)
            dummy = torch.zeros(b, s, d, dtype=torch.float32, device=device)
            pos = torch.arange(s, dtype=torch.long, device=device).unsqueeze(0).expand(b, -1)
            with torch.no_grad():
                rope_cos, rope_sin = rope(dummy, pos)

        for layer_idx in range(int(self.cfg.num_hidden_layers)):
            x = self.forward_layer(
                layer_idx=layer_idx,
                hidden_states=x,
                timestep_proj_b6d=timestep_proj_b6d,
                encoder_hidden_states=enc,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                debug=debug,
            )

        self._debug_put(debug, "core.out", x.squeeze(1))
        return x.squeeze(1)  # [B,S,D]

    def condition_encoder_hidden_states(
        self, encoder_hidden_states: torch.Tensor, *, device: torch.device | None = None
    ) -> torch.Tensor:
        """Project raw encoder states to hidden size (``condition_embedder``), shape ``[B,1,S_enc,D]``."""
        compute_dtype = torch.bfloat16
        device = device or encoder_hidden_states.device
        d = int(self.cfg.hidden_size)
        enc = encoder_hidden_states.to(dtype=compute_dtype, device=device).unsqueeze(1)
        cond_w = self._w("condition_embedder.weight", device=device, dtype=compute_dtype)
        cond_b = self._w("condition_embedder.bias", device=device, dtype=compute_dtype)
        return torch.matmul(enc, cond_w.t()) + cond_b.view(1, 1, 1, d)

    def forward_layer(
        self,
        *,
        layer_idx: int,
        hidden_states: torch.Tensor,
        timestep_proj_b6d: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        debug: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        One ``layers.{i}`` block (self-attn + cross-attn + gated MLP).

        ``hidden_states``: ``[B,1,S,D]``; ``encoder_hidden_states``: ``[B,1,S_enc,D]`` after
        ``condition_embedder`` (same as ``TtAceStepDiTLayer`` input).
        """
        layer_idx = int(layer_idx)
        eps = float(self.cfg.rms_norm_eps)
        h = int(self.cfg.num_attention_heads)
        dh = int(self.cfg.head_dim)
        compute_dtype = torch.bfloat16
        device = hidden_states.device
        x = hidden_states.to(dtype=compute_dtype, device=device)
        timestep_proj_b6d = timestep_proj_b6d.to(dtype=compute_dtype, device=device)
        enc = encoder_hidden_states.to(dtype=compute_dtype, device=device)

        b, _one, s, d = x.shape
        if rope_cos is None or rope_sin is None:
            rope = self._rope(d_model=d, n_heads=h, head_dim=dh, max_seq_len=max(s, 512), device=device)
            dummy = torch.zeros(b, s, d, dtype=torch.float32, device=device)
            pos = torch.arange(s, dtype=torch.long, device=device).unsqueeze(0).expand(b, -1)
            with torch.no_grad():
                rope_cos, rope_sin = rope(dummy, pos)

        sst = self._w(f"layers.{layer_idx}.scale_shift_table", device=device, dtype=compute_dtype)
        sst = sst + timestep_proj_b6d
        shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = [sst[:, i : i + 1, :].unsqueeze(2) for i in range(6)]

        self_norm_w = self._w(f"layers.{layer_idx}.self_attn_norm.weight", device=device, dtype=compute_dtype)
        cross_norm_w = self._w(f"layers.{layer_idx}.cross_attn_norm.weight", device=device, dtype=compute_dtype)
        mlp_norm_w = self._w(f"layers.{layer_idx}.mlp_norm.weight", device=device, dtype=compute_dtype)

        x_norm = rmsnorm_qwen3(x, self_norm_w, eps)
        h_in = x_norm * (1 + scale_msa) + shift_msa
        base = f"layers.{layer_idx}.self_attn"
        attn_out = _attention_sdpa(
            h_in,
            wq=self._w(f"{base}.q_proj.weight", device=device, dtype=compute_dtype),
            wk=self._w(f"{base}.k_proj.weight", device=device, dtype=compute_dtype),
            wv=self._w(f"{base}.v_proj.weight", device=device, dtype=compute_dtype),
            wo=self._w(f"{base}.o_proj.weight", device=device, dtype=compute_dtype),
            q_norm_w=self._w(f"{base}.q_norm.weight", device=device, dtype=compute_dtype),
            k_norm_w=self._w(f"{base}.k_norm.weight", device=device, dtype=compute_dtype),
            n_heads=h,
            n_kv_heads=int(self.cfg.num_key_value_heads),
            head_dim=dh,
            eps=eps,
            encoder_hidden_states=None,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trace_prefix=f"torch_ref.layer{layer_idx}.self.",
        )
        self._debug_put(debug, f"layer{layer_idx}.after_self", x.squeeze(1))
        x = x + attn_out * gate_msa

        x2 = rmsnorm_qwen3(x, cross_norm_w, eps)
        base = f"layers.{layer_idx}.cross_attn"
        ca = _attention_sdpa(
            x2,
            wq=self._w(f"{base}.q_proj.weight", device=device, dtype=compute_dtype),
            wk=self._w(f"{base}.k_proj.weight", device=device, dtype=compute_dtype),
            wv=self._w(f"{base}.v_proj.weight", device=device, dtype=compute_dtype),
            wo=self._w(f"{base}.o_proj.weight", device=device, dtype=compute_dtype),
            q_norm_w=self._w(f"{base}.q_norm.weight", device=device, dtype=compute_dtype),
            k_norm_w=self._w(f"{base}.k_norm.weight", device=device, dtype=compute_dtype),
            n_heads=h,
            n_kv_heads=int(self.cfg.num_key_value_heads),
            head_dim=dh,
            eps=eps,
            encoder_hidden_states=enc,
            rope_cos=None,
            rope_sin=None,
            trace_prefix=f"torch_ref.layer{layer_idx}.cross.",
        )
        x = x + ca

        x3 = rmsnorm_qwen3(x, mlp_norm_w, eps)
        h3 = x3 * (1 + c_scale) + c_shift
        ff = qwen3_mlp(
            h3,
            w_gate=self._w(f"layers.{layer_idx}.mlp.gate_proj.weight", device=device, dtype=compute_dtype),
            w_up=self._w(f"layers.{layer_idx}.mlp.up_proj.weight", device=device, dtype=compute_dtype),
            w_down=self._w(f"layers.{layer_idx}.mlp.down_proj.weight", device=device, dtype=compute_dtype),
        )
        x = x + ff * c_gate
        self._debug_put(debug, f"layer{layer_idx}.block_out", x.squeeze(1))
        return x

    def __call__(
        self,
        x_patches: torch.Tensor,
        timestep_proj_b6d: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        debug: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        return self.forward(
            x_patches=x_patches,
            timestep_proj_b6d=timestep_proj_b6d,
            encoder_hidden_states=encoder_hidden_states,
            debug=debug,
        )


class TorchAceStepDiTLayerRef:
    """Thin wrapper around :meth:`TorchAceStepDiTCoreRef.forward_layer` for PCC tests."""

    def __init__(self, *, cfg: AceStepDecoderConfigTTNN, state_dict: dict[str, np.ndarray], layer_idx: int = 0) -> None:
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self._core = TorchAceStepDiTCoreRef(cfg=cfg, state_dict=state_dict)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        timestep_proj_b6d: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._core.forward_layer(
            layer_idx=self.layer_idx,
            hidden_states=hidden_states,
            timestep_proj_b6d=timestep_proj_b6d,
            encoder_hidden_states=encoder_hidden_states,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
