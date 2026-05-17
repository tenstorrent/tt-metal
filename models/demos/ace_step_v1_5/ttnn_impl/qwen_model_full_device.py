# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full-device causal LM stack for ACE-Step 5 Hz (``--experimental-5hz-ttnn-causal-lm``).

``QwenModelFullDevice`` mirrors :class:`~models.demos.ace_step_v1_5.ttnn_impl.ace_step_ds_r1_qwen.QwenModel`
but replaces hybrid ``Attention`` (torch softmax / CPU KV) with :class:`AttentionFullDevice`:

- TTNN embedding prefix (via :mod:`qwen_causal_prefix_ttnn`)
- TTNN prefill causal additive mask (no host mask tensor)
- ``TtHfRotaryEmbedding`` + ``ttnn.experimental.rotary_embedding`` (device cos/sin caches)
- TTNN ``rms_norm`` on Q/K heads
- TTNN KV history via ``ttnn.concat`` along sequence
- TTNN matmul + softmax attention via ``_ace_step_cross_attention_decomposed`` (same pattern as DiT)

**Limitations**

- ``sliding_attention`` layers are **not** supported here (raise at init). ACE-Step 5 Hz 1.7B configs in the
  wild are often all ``full_attention``; if your checkpoint uses sliding windows, keep the legacy
  :class:`~models.demos.ace_step_v1_5.ttnn_impl.ace_step_ds_r1_qwen.QwenModel` path.
- Batch size must remain **1** (same as the experimental wrapper contract).

The HF handler still receives **torch** logits: :class:`AceStepFiveHzExperimentalTtnnCausalLM` performs a
single ``to_torch_auto_compose`` at the model boundary.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM

import ttnn
from models.demos.ace_step_v1_5.ttnn_impl.ace_step_ds_r1_qwen import MLP
from models.demos.ace_step_v1_5.ttnn_impl.dit_decoder_core import (
    TtHfRotaryEmbedding,
    _ace_step_cross_attention_decomposed,
)
from models.demos.ace_step_v1_5.ttnn_impl.qwen_causal_prefix_ttnn import build_prefix_full_device


def _concat_heads(ttnn: Any, parts: list) -> Any:
    if len(parts) == 1:
        return parts[0]
    return ttnn.concat(parts, dim=1) if hasattr(ttnn, "concat") else ttnn.concatenate(parts, dim=1)


def _repeat_kv_interleave(
    ttnn: Any,
    k: Any,
    *,
    num_kv_heads: int,
    n_rep: int,
    sk: int,
    head_dim: int,
) -> Any:
    """GQA repeat-interleave along head dim (dim=1), matching ``torch.repeat_interleave`` order."""
    parts: list = []
    for kv in range(num_kv_heads):
        ki = ttnn.slice(k, (0, kv, 0, 0), (1, kv + 1, sk, head_dim))
        for _ in range(n_rep):
            if n_rep > 1:
                if not hasattr(ttnn, "clone"):
                    raise RuntimeError("ttnn.clone is required for GQA repeat when num_queries_per_kv > 1")
                parts.append(ttnn.clone(ki))
            else:
                parts.append(ki)
    return _concat_heads(ttnn, parts)


def _slice_seq_len2_if_needed(
    ttnn: Any,
    t: Any,
    *,
    bsz: int,
    n_heads: int,
    seq_cap: int,
    head_dim: int,
) -> Any:
    """Trim TILE-padded sequence (dim 2) so Q/K/V match residual hidden states ``x.shape[2]``."""
    sq = int(t.shape[2])
    if sq <= int(seq_cap):
        return t
    return ttnn.slice(t, (0, 0, 0, 0), (int(bsz), int(n_heads), int(seq_cap), int(head_dim)))


class AttentionFullDevice:
    """Self-attention with Q/K/V linear, TTNN RoPE, device KV cache, and TTNN decomposed SDPA."""

    def __init__(
        self,
        device: Any,
        layer_id: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        bq: Optional[torch.Tensor] = None,
        bk: Optional[torch.Tensor] = None,
        bv: Optional[torch.Tensor] = None,
        max_seq_len: int = 2048,
        sliding_window: Optional[int] = None,
        q_norm_weight: Optional[torch.Tensor] = None,
        k_norm_weight: Optional[torch.Tensor] = None,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        compute_kernel_config=None,
    ):
        if sliding_window is not None and int(sliding_window) > 0:
            raise RuntimeError(
                "AttentionFullDevice does not implement sliding-window masks on TTNN yet. "
                "Use ace_step_ds_r1_qwen.QwenModel for checkpoints with sliding_attention layers."
            )
        self.layer_id = int(layer_id)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.hidden_size = int(hidden_size)
        self.device = device
        self.rms_norm_eps = float(rms_norm_eps)
        self.compute_kernel_config = compute_kernel_config

        self.wq = ttnn.from_torch(
            wq.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.wk = ttnn.from_torch(
            wk.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.wv = ttnn.from_torch(
            wv.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.wo = ttnn.from_torch(
            wo.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.bq = (
            ttnn.from_torch(bq.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            if bq is not None
            else None
        )
        self.bk = (
            ttnn.from_torch(bk.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            if bk is not None
            else None
        )
        self.bv = (
            ttnn.from_torch(bv.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            if bv is not None
            else None
        )

        if q_norm_weight is None or k_norm_weight is None:
            raise RuntimeError("AttentionFullDevice requires Qwen3 q_norm/k_norm weights.")
        # Stage q_norm/k_norm weights as fp32 so the head RMSNorm runs in fp32 on the upgraded
        # residual stream (matches HF Qwen3RMSNorm semantics).
        self.q_norm_w_tt = ttnn.from_torch(
            q_norm_weight.detach().to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        )
        self.k_norm_w_tt = ttnn.from_torch(
            k_norm_weight.detach().to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        )

        self._k_hist: Any = None
        self._v_hist: Any = None

        self._rotary = TtHfRotaryEmbedding(
            mesh_device=device,
            head_dim=self.head_dim,
            max_seq_len=int(max_seq_len),
            rope_theta=float(rope_theta),
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
        )

    def reset_kv_device(self) -> None:
        if self._k_hist is not None and hasattr(ttnn, "deallocate"):
            try:
                ttnn.deallocate(self._k_hist)
            except Exception:
                pass
        if self._v_hist is not None and hasattr(ttnn, "deallocate"):
            try:
                ttnn.deallocate(self._v_hist)
            except Exception:
                pass
        self._k_hist = None
        self._v_hist = None

    def __call__(self, x: Any, start_pos: int, mask_tt: Any = None) -> Any:
        # Force fp32 output dtype so the residual stream the caller adds these into stays fp32 —
        # bf16 ttnn.add at deep-layer activations (mean_abs ~700, max ~3000 on Qwen3 1.7B)
        # quantizes residuals to multiples of ~2-16 and dominates per-layer PCC drift otherwise.
        ck_kw = dict(compute_kernel_config=self.compute_kernel_config) if self.compute_kernel_config is not None else {}
        ck_kw["dtype"] = ttnn.float32
        xq = ttnn.matmul(x, self.wq, **ck_kw)
        xk = ttnn.matmul(x, self.wk, **ck_kw)
        xv = ttnn.matmul(x, self.wv, **ck_kw)
        if self.bq is not None:
            xq = ttnn.add(xq, self.bq)
        if self.bk is not None:
            xk = ttnn.add(xk, self.bk)
        if self.bv is not None:
            xv = ttnn.add(xv, self.bv)

        B = int(x.shape[1])
        S = int(x.shape[2])
        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)

        q5 = ttnn.reshape(xq, (B, 1, S, self.num_heads, self.head_dim))
        q = ttnn.permute(q5, (0, 3, 2, 4, 1))
        q = ttnn.reshape(q, (B, self.num_heads, S, self.head_dim))

        k5 = ttnn.reshape(xk, (B, 1, S, self.num_kv_heads, self.head_dim))
        k = ttnn.permute(k5, (0, 3, 2, 4, 1))
        k = ttnn.reshape(k, (B, self.num_kv_heads, S, self.head_dim))

        v5 = ttnn.reshape(xv, (B, 1, S, self.num_kv_heads, self.head_dim))
        v = ttnn.permute(v5, (0, 3, 2, 4, 1))
        v = ttnn.reshape(v, (B, self.num_kv_heads, S, self.head_dim))

        q = ttnn.rms_norm(q, weight=self.q_norm_w_tt, epsilon=self.rms_norm_eps, memory_config=mem)
        k = ttnn.rms_norm(k, weight=self.k_norm_w_tt, epsilon=self.rms_norm_eps, memory_config=mem)

        tok = int(start_pos) if S == 1 else None
        q = self._rotary(q, token_idx=tok)
        k = self._rotary(k, token_idx=tok)

        S_log = int(x.shape[2])
        q = _slice_seq_len2_if_needed(ttnn, q, bsz=B, n_heads=self.num_heads, seq_cap=S_log, head_dim=self.head_dim)
        k = _slice_seq_len2_if_needed(ttnn, k, bsz=B, n_heads=self.num_kv_heads, seq_cap=S_log, head_dim=self.head_dim)
        v = _slice_seq_len2_if_needed(ttnn, v, bsz=B, n_heads=self.num_kv_heads, seq_cap=S_log, head_dim=self.head_dim)

        if self._k_hist is None:
            new_k, new_v = k, v
        else:
            new_k = (
                ttnn.concat([self._k_hist, k], dim=2)
                if hasattr(ttnn, "concat")
                else ttnn.concatenate([self._k_hist, k], dim=2)
            )
            new_v = (
                ttnn.concat([self._v_hist, v], dim=2)
                if hasattr(ttnn, "concat")
                else ttnn.concatenate([self._v_hist, v], dim=2)
            )
            if hasattr(ttnn, "deallocate"):
                try:
                    ttnn.deallocate(self._k_hist)
                    ttnn.deallocate(self._v_hist)
                except Exception:
                    pass
        self._k_hist, self._v_hist = new_k, new_v

        sk = int(new_k.shape[2])
        k_exp = _repeat_kv_interleave(
            ttnn,
            new_k,
            num_kv_heads=self.num_kv_heads,
            n_rep=self.num_queries_per_kv,
            sk=sk,
            head_dim=self.head_dim,
        )
        v_exp = _repeat_kv_interleave(
            ttnn,
            new_v,
            num_kv_heads=self.num_kv_heads,
            n_rep=self.num_queries_per_kv,
            sk=sk,
            head_dim=self.head_dim,
        )

        act = ttnn.bfloat16
        s_q = int(q.shape[2])
        ctx = _ace_step_cross_attention_decomposed(
            ttnn,
            q=q,
            k=k_exp,
            v=v_exp,
            b=B,
            h=self.num_heads,
            s_q=s_q,
            scale=1.0 / math.sqrt(float(self.head_dim)),
            additive_mask_b1qk=mask_tt,
            activations_dtype=act,
            use_fp32=False,
            softmax_fp32=True,  # HF Qwen3: softmax(..., dtype=torch.float32).to(bf16)
            compute_kernel_config=self.compute_kernel_config,
        )

        out = ttnn.permute(ctx, (0, 2, 1, 3))
        S_out = int(out.shape[1])
        out = ttnn.reshape(out, (1, B, S_out, self.hidden_size))
        # wo matmul: bf16 weight × fp32 act × fp32 dest accumulator, fp32 output to keep residual
        # stream in fp32 (avoids deep-layer bf16 quantization on Qwen3 1.7B).
        wo_kw = dict(ck_kw)
        wo_kw["dtype"] = ttnn.float32
        return ttnn.matmul(out, self.wo, **wo_kw)


class TransformerBlockFullDevice:
    """Decoder block using :class:`AttentionFullDevice` (no host RoPE tensors)."""

    def __init__(
        self,
        layer_id: int,
        config: Any,
        layer_weights: dict[str, torch.Tensor],
        device: Any,
        max_seq_len: int = 2048,
        compute_kernel_config=None,
    ):
        self.layer_id = int(layer_id)

        layer_types = getattr(config, "layer_types", None)
        sliding_window = None
        if (
            layer_types is not None
            and layer_id < len(layer_types)
            and layer_types[layer_id] == "sliding_attention"
            and getattr(config, "sliding_window", None) is not None
        ):
            sliding_window = int(config.sliding_window)

        head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        q_norm_w = layer_weights.get("self_attn.q_norm.weight")
        k_norm_w = layer_weights.get("self_attn.k_norm.weight")
        if getattr(config, "model_type", None) == "qwen3" and (q_norm_w is None or k_norm_w is None):
            raise RuntimeError(f"Qwen3 layer {layer_id} missing q_norm/k_norm weights.")

        rope_theta = float(getattr(config, "rope_theta", 1_000_000.0))
        self.attention = AttentionFullDevice(
            device,
            layer_id,
            config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            wq=layer_weights["self_attn.q_proj.weight"],
            wk=layer_weights["self_attn.k_proj.weight"],
            wv=layer_weights["self_attn.v_proj.weight"],
            wo=layer_weights["self_attn.o_proj.weight"],
            bq=layer_weights.get("self_attn.q_proj.bias"),
            bk=layer_weights.get("self_attn.k_proj.bias"),
            bv=layer_weights.get("self_attn.v_proj.bias"),
            max_seq_len=max_seq_len,
            sliding_window=sliding_window,
            q_norm_weight=q_norm_w,
            k_norm_weight=k_norm_w,
            rms_norm_eps=float(config.rms_norm_eps),
            rope_theta=rope_theta,
            compute_kernel_config=compute_kernel_config,
        )

        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            gate_proj=layer_weights["mlp.gate_proj.weight"],
            up_proj=layer_weights["mlp.up_proj.weight"],
            down_proj=layer_weights["mlp.down_proj.weight"],
            device=device,
            compute_kernel_config=compute_kernel_config,
        )
        self.rms_norm_eps = float(config.rms_norm_eps)
        # fp32 RMSNorm weights so the variance reduction stays in fp32 on the upgraded residual
        # stream (matches HF Qwen3RMSNorm precision).
        self.input_layernorm_w = ttnn.from_torch(
            layer_weights["input_layernorm.weight"].detach().to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        )
        self.post_attention_layernorm_w = ttnn.from_torch(
            layer_weights["post_attention_layernorm.weight"].detach().to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        )

    def __call__(self, x: Any, start_pos: int, mask_tt: Any = None, position_embeddings: Any = None) -> Any:
        del position_embeddings
        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        h = ttnn.rms_norm(x, weight=self.input_layernorm_w, epsilon=self.rms_norm_eps, memory_config=mem)
        h = self.attention(h, start_pos, mask_tt)
        x = ttnn.add(x, h)
        h = ttnn.rms_norm(x, weight=self.post_attention_layernorm_w, epsilon=self.rms_norm_eps, memory_config=mem)
        h = self.mlp(h)
        return ttnn.add(x, h)


class QwenModelFullDevice:
    """Qwen-style decoder with a fully TTNN attention path (see module docstring)."""

    def __init__(self, model_name: str, device: Any, max_seq_len: int = 2048, *, validate_against_hf: bool = False):
        if validate_against_hf:
            raise RuntimeError("QwenModelFullDevice does not support validate_against_hf=True.")

        print(f"Loading {model_name} (full-device attention)...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.config = hf_model.config
        self.device = device
        self.max_seq_len = int(max_seq_len)

        # HiFi4 matmul with fp32 accumulation matches PyTorch bfloat16 matmul precision.
        init_ck = getattr(ttnn, "init_device_compute_kernel_config", None)
        self._compute_kernel_config = None
        if callable(init_ck) and hasattr(device, "arch"):
            try:
                self._compute_kernel_config = init_ck(
                    device.arch(),
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=True,
                )
            except Exception:
                pass

        state_dict = hf_model.state_dict()

        emb_w = state_dict["model.embed_tokens.weight"].to(torch.bfloat16).contiguous()
        self.embed_tokens_tt = ttnn.from_torch(
            emb_w,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        del emb_w

        print("Building transformer layers (full-device attention)...")
        self.layers: list = []
        for layer_id in range(self.config.num_hidden_layers):
            layer_weights = {
                key.replace(f"model.layers.{layer_id}.", ""): value
                for key, value in state_dict.items()
                if f"model.layers.{layer_id}." in key
            }
            self.layers.append(
                TransformerBlockFullDevice(
                    layer_id=layer_id,
                    config=self.config,
                    layer_weights=layer_weights,
                    device=device,
                    max_seq_len=max_seq_len,
                    compute_kernel_config=self._compute_kernel_config,
                )
            )
            if (layer_id + 1) % 4 == 0:
                print(f"  Loaded {layer_id + 1}/{self.config.num_hidden_layers} layers")

        self.rms_norm_eps = float(self.config.rms_norm_eps)
        # Final norm weight in fp32 to preserve precision on the fp32 residual stream.
        self.norm_w = ttnn.from_torch(
            state_dict["model.norm.weight"].detach().to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        )
        self.lm_head = ttnn.from_torch(
            state_dict["lm_head.weight"].T.unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        print("Model loaded successfully (full-device)!")
        del hf_model
        del state_dict

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> Any:
        h_tt, mask_tt = build_prefix_full_device(
            tokens=tokens,
            embed_tokens_tt=self.embed_tokens_tt,
            device=self.device,
            hidden_size=int(self.config.hidden_size),
            start_pos=int(start_pos),
        )
        # Upgrade the residual stream to fp32. Each layer's matmuls now output fp32, the RMSNorm
        # weights are fp32, and ttnn.add stays in fp32 — no per-layer bf16 quantization on the
        # residual stream (the dominant remaining drift on Qwen3 1.7B before this fix).
        h_tt = ttnn.typecast(h_tt, dtype=ttnn.float32)
        for layer in self.layers:
            h_tt = layer(h_tt, int(start_pos), mask_tt, position_embeddings=None)
        if mask_tt is not None and hasattr(ttnn, "deallocate"):
            try:
                ttnn.deallocate(mask_tt)
            except Exception:
                pass
        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        h_tt = ttnn.rms_norm(h_tt, weight=self.norm_w, epsilon=self.rms_norm_eps, memory_config=mem)
        ck_kw = (
            dict(compute_kernel_config=self._compute_kernel_config) if self._compute_kernel_config is not None else {}
        )
        # lm_head matmul: bf16 weight × fp32 act × fp32 dest accumulator → fp32 logits, matches
        # the host-attention QwenModel path so the experimental wrapper's downstream consumers
        # see the same precision regardless of which body they pick.
        ck_kw["dtype"] = ttnn.float32
        return ttnn.matmul(h_tt, self.lm_head, **ck_kw)

    def reset_kv_cache(self) -> None:
        for layer in self.layers:
            layer.attention.reset_kv_device()


__all__ = ["AttentionFullDevice", "QwenModelFullDevice", "TransformerBlockFullDevice"]
