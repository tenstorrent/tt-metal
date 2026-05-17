# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step demo fork of the hybrid TTNN + torch Qwen causal stack.

Upstream reference: ``models/experimental/tt_transformers_v2/ds_r1_qwen.py`` (do **not** modify that
tree for ACE-Step work). This copy carries Qwen3-specific fixes (RoPE, q_norm/k_norm, sliding-window
attention path, ``validate_against_hf``, etc.) used by
:class:`~models.demos.ace_step_v1_5.ttnn_impl.five_hz_causal_lm_experimental.AceStepFiveHzExperimentalTtnnCausalLM`.

:class:`QwenModel` stages token embedding with ``ttnn.embedding``, prefill causal masks with
``ttnn.full``/``ttnn.triu``, and position ids with ``ttnn.arange`` (see ``qwen_causal_prefix_ttnn``);
HF rotary still computes ``cos``/``sin`` on CPU for parity with ``hf_apply_rotary_pos_emb`` inside
``Attention``.

The original file description (DeepSeek-R1-Distill-Qwen-1.5B TTNN prototype) still broadly applies.
"""

import math
import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding as HFQwen2RotaryEmbedding
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as hf_apply_rotary_pos_emb

import ttnn
from models.common.auto_compose import to_torch_auto_compose

# Validation framework
from models.common.validation_tools import Metric, compare_to_torch, get_validation_registry
from models.demos.ace_step_v1_5.ttnn_impl.qwen_causal_prefix_ttnn import build_prefix_ttnn


def _build_rotary_embedding(config):
    """Pick HF RoPE that matches the checkpoint (Qwen3 vs Qwen2).

    ACE-Step 5 Hz LMs are Qwen3; using ``Qwen2RotaryEmbedding`` on a ``Qwen3Config`` produces
    incompatible cos/sin and garbage token predictions even when logits tensor shapes are correct.
    """
    if getattr(config, "model_type", None) == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        return Qwen3RotaryEmbedding(config=config, device=None)
    return HFQwen2RotaryEmbedding(config)


def _head_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm on the last dimension (``head_dim``). Matches Qwen3 ``q_norm`` / ``k_norm``."""
    w = weight.to(device=x.device, dtype=torch.float32)
    x_f = x.float()
    variance = x_f.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_f * torch.rsqrt(variance + eps)
    return (x_normed * w).to(dtype=x.dtype)


def _hf_grade_compute_kernel_config(device):
    """HF-grade matmul precision: HiFi4 / approx=False / fp32 dest accumulator / packer L1 acc.

    Per-layer PCC drift in :func:`test_llm_per_layer_pcc_debug` was traced to bf16 matmul
    accumulator drift in deep layers where activations grow to ~3000+ (bf16 precision ~16).
    This config matches the recipe used by ``qwen3_embedding_encoder``, ``audio_code_detokenizer``,
    ``condition_encoder``, ``vae/conv1d`` for their high-precision matmul paths.
    """
    init_ck = getattr(ttnn, "init_device_compute_kernel_config", None)
    if not callable(init_ck) or not hasattr(device, "arch"):
        print(
            "[ace_step_v1_5][qwen] WARN: init_device_compute_kernel_config unavailable, "
            "matmuls will use TTNN defaults (bf16 dest accumulator).",
            flush=True,
        )
        return None
    try:
        ck = init_ck(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        print(f"[ace_step_v1_5][qwen] HF-grade compute_kernel_config built: {ck}", flush=True)
        return ck
    except Exception as e:
        print(f"[ace_step_v1_5][qwen] WARN: init_device_compute_kernel_config failed ({e!r})", flush=True)
        return None


# ============================================================================
# Model Implementation by Claude and Codex with manually edits
# ============================================================================


class RMSNorm:
    """RMS Normalization in TTNN.

    Weights staged in fp32 so the manual chain (mul/mean/sqrt/reciprocal/mul) stays uniformly in
    fp32 when called with an fp32 residual stream. With Qwen3 1.7B, after the matmuls were moved
    to fp32 dest accumulator (see :func:`_hf_grade_compute_kernel_config`) and the residual stream
    upgraded to fp32 in :meth:`QwenModel.forward`, RMSNorm precision becomes the next-largest
    drift source — fp32 weight + fp32 ops keeps the variance reduction faithful to HF.
    """

    def __init__(self, weight: torch.Tensor, eps: float, device):
        self.eps = eps
        self.device = device
        self.weight_torch = weight
        self.ref_module = None  # optional HF module for validation
        self.weight = ttnn.from_torch(
            weight.detach().to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        )

    @compare_to_torch(
        reference_fn=lambda self, x: self._ref_call(x),
        metric_tolerances={
            # Metric.MAX_ABS_ERROR: 1e-2,
            # Metric.MEAN_ABS_ERROR: 1e-3,
            Metric.PCC: 0.99,
        },
        enabled=False,
    )
    def __call__(self, x):
        # x shape: [1, seq_len, hidden_size]
        # Compute RMS: sqrt(mean(x^2) + eps)
        x_squared = ttnn.mul(x, x)
        mean_x_squared = ttnn.mean(x_squared, dim=-1, keepdim=True)
        rms = ttnn.sqrt(ttnn.add(mean_x_squared, self.eps))
        # Normalize and scale
        x_normed = ttnn.mul(x, ttnn.reciprocal(rms))
        return ttnn.mul(x_normed, self.weight)

    def _ref_call(self, x_torch):
        # Use the HF RMSNorm module only; no manual fallback
        if self.ref_module is None:
            raise RuntimeError("HF RMSNorm reference module not set for validation")
        mod_dtype = next(self.ref_module.parameters()).dtype
        return self.ref_module(x_torch.to(mod_dtype))


class RotaryEmbedding:
    """Deprecated local RoPE helper (kept for reference). Use HFQwen2RotaryEmbedding instead."""

    def __init__(self, *args, **kwargs):
        pass

    def apply_rotary_emb(self, x: torch.Tensor, position: int) -> torch.Tensor:
        raise NotImplementedError("Use HFQwen2RotaryEmbedding via position_embeddings from QwenModel.forward")


class Attention:
    """Multi-head attention with GQA support"""

    def __init__(
        self,
        device: ttnn.MeshDevice,
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
    ):
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.hidden_size = hidden_size
        self.device = device
        self.sliding_window = sliding_window
        self.q_norm_weight = q_norm_weight
        self.k_norm_weight = k_norm_weight
        self.rms_norm_eps = float(rms_norm_eps)

        # Convert weights to TTNN
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

        # Optional biases
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

        # KV cache storage (on CPU as torch tensors for now)
        self.cache_k = torch.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=torch.bfloat16)
        self.cache_v = torch.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=torch.bfloat16)

        # HF-grade compute kernel config for Q/K/V/O matmuls (fp32 destination accumulator).
        self._mm_ck = _hf_grade_compute_kernel_config(device)

        # Optional HF attention module and past KV for validation
        self.hf_attn = None
        self.hf_past_kv = None
        # HF rotary embedding (for reference path)
        self.hf_rotary_emb = None

    @compare_to_torch(
        reference_fn=lambda self, x, start_pos, mask: self._ref_call(x, start_pos, mask),
        input_to_torch=lambda self, x, start_pos, mask: (
            self,
            to_torch_auto_compose(x).squeeze(0),
            start_pos,
            mask,
        ),
        metric_tolerances={
            # Metric.MAX_ABS_ERROR: 2e-1,
            # Metric.MEAN_ABS_ERROR: 2e-2,
            Metric.PCC: 0.95,
        },
        enabled=False,
    )
    def __call__(
        self,
        x,
        start_pos: int,
        mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):  # TTNN tensor [1, seq_len, hidden_size]
        # Project to Q, K, V (+ bias if available). Force fp32 output so the residual stream the
        # caller adds these into stays fp32 (bf16 would quantize ~2/elem at deep-layer activations
        # which is the dominant remaining drift on Qwen3 1.7B PCC).
        ck_kw = dict(compute_kernel_config=self._mm_ck) if self._mm_ck is not None else dict()
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

        # Convert to torch for reshaping and RoPE
        xq_torch = to_torch_auto_compose(xq).squeeze(0)
        xk_torch = to_torch_auto_compose(xk).squeeze(0)
        xv_torch = to_torch_auto_compose(xv).squeeze(0)
        # Ensure batch dim exists
        if xq_torch.dim() == 3:
            pass  # [batch, seq, hidden]
        elif xq_torch.dim() == 2:
            xq_torch = xq_torch.unsqueeze(0)
            xk_torch = xk_torch.unsqueeze(0)
            xv_torch = xv_torch.unsqueeze(0)
        else:
            raise RuntimeError(f"Unexpected xq_torch dim: {xq_torch.shape}")

        batch_size, seq_len, _ = xq_torch.shape

        # Reshape to [batch, num_heads, seq_len, head_dim]; Qwen3 applies per-head RMSNorm before RoPE.
        xq_torch = xq_torch.view(batch_size, seq_len, self.num_heads, self.head_dim)
        if self.q_norm_weight is not None:
            xq_torch = _head_rmsnorm(xq_torch, self.q_norm_weight, self.rms_norm_eps)
        xq_torch = xq_torch.transpose(1, 2)

        xk_torch = xk_torch.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        if self.k_norm_weight is not None:
            xk_torch = _head_rmsnorm(xk_torch, self.k_norm_weight, self.rms_norm_eps)
        xk_torch = xk_torch.transpose(1, 2)

        xv_torch = xv_torch.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings from HF for exact parity
        if position_embeddings is None:
            raise RuntimeError("position_embeddings (cos,sin) must be provided from QwenModel.forward")
        cos, sin = position_embeddings
        xq_torch, xk_torch = hf_apply_rotary_pos_emb(xq_torch, xk_torch, cos, sin)

        # Update KV cache (keep on CPU)
        # Ensure dtype matches cache dtype
        xk_torch = xk_torch.to(self.cache_k.dtype)
        xv_torch = xv_torch.to(self.cache_v.dtype)
        self.cache_k[:, :, start_pos : start_pos + seq_len] = xk_torch
        self.cache_v[:, :, start_pos : start_pos + seq_len] = xv_torch

        # Get keys and values up to current position
        keys = self.cache_k[:, :, : start_pos + seq_len]
        values = self.cache_v[:, :, : start_pos + seq_len]

        # Repeat KV heads for GQA
        if self.num_queries_per_kv > 1:
            keys = keys.repeat_interleave(self.num_queries_per_kv, dim=1)
            values = values.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Compute attention scores (softmax in fp32)
        scores = torch.matmul(xq_torch.to(torch.float32), keys.transpose(-2, -1).to(torch.float32))
        scores = scores * (1.0 / math.sqrt(self.head_dim))

        # Qwen3 sliding_attention layers: causal + local window (full triu mask alone is wrong).
        if self.sliding_window is not None and self.sliding_window > 0:
            _, _, sq, sk = scores.shape
            device = scores.device
            q_global = torch.arange(start_pos, start_pos + sq, device=device, dtype=torch.int64).view(1, 1, sq, 1)
            k_global = torch.arange(0, sk, device=device, dtype=torch.int64).view(1, 1, 1, sk)
            sw = int(self.sliding_window)
            left_bound = q_global - sw + 1
            invalid = (k_global > q_global) | (k_global < left_bound)
            scores = scores.masked_fill(invalid, float("-inf"))

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask.to(scores.dtype)

        attn = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(xq_torch.dtype)

        # Apply attention to values. ``xq_torch`` may now be fp32 (when the upstream Q/K/V matmul
        # output dtype is fp32 to keep the residual stream precise), so ``attn`` is fp32 too.
        # The cache stays in bf16 to keep host RAM bounded; cast values to ``attn``'s dtype just
        # for this matmul so PyTorch doesn't reject the mixed-dtype matmul.
        if values.dtype != attn.dtype:
            output = torch.matmul(attn, values.to(attn.dtype))
        else:
            output = torch.matmul(attn, values)

        # Reshape back to [batch, seq_len, hidden_size]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Convert back to TTNN as fp32 so the wo matmul input is fp32 (matches the upgraded
        # residual stream's precision and avoids the deep-layer bf16 quantization observed in
        # the per-layer PCC diagnostic).
        output_tt = ttnn.from_torch(
            output.float().unsqueeze(0), device=self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
        )

        ck_kw = dict(compute_kernel_config=self._mm_ck) if self._mm_ck is not None else dict()
        ck_kw["dtype"] = ttnn.float32
        output_tt = ttnn.matmul(output_tt, self.wo, **ck_kw)

        return output_tt

    def _ref_call(self, x_torch: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Use the HF attention module only; no manual fallback
        if self.hf_attn is None:
            raise RuntimeError("HF Attention reference module not set for validation")
        x_t = x_torch
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(0)
        bsz, seq_len, _ = x_t.shape
        position_ids = torch.arange(start_pos, start_pos + seq_len, device=x_t.device).unsqueeze(0).expand(bsz, -1)
        mod_dtype = next(self.hf_attn.parameters()).dtype
        x_cast = x_t.to(mod_dtype)

        # Use HF rotary embedding module for position embeddings (cos, sin)
        if self.hf_rotary_emb is None:
            raise RuntimeError("HF RotaryEmbedding reference module not set for validation")
        cos, sin = self.hf_rotary_emb(x_cast, position_ids)

        out = self.hf_attn(
            hidden_states=x_cast,
            attention_mask=None,  # rely on HF's internal causal masking
            position_ids=position_ids,
            position_embeddings=(cos, sin),
            past_key_value=self.hf_past_kv,
            output_attentions=False,
            use_cache=True,
        )
        if isinstance(out, tuple):
            attn_out = out[0]
            self.hf_past_kv = out[-1]
        else:
            attn_out = getattr(out, "hidden_states", out)
            self.hf_past_kv = getattr(out, "past_key_value", self.hf_past_kv)
        return attn_out


class MLP:
    """Feed-forward network"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        device,
        compute_kernel_config=None,
    ):
        # Optional HF MLP module for validation
        self.ref_module = None
        self.compute_kernel_config = compute_kernel_config
        self.gate_proj = ttnn.from_torch(
            gate_proj.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.up_proj = ttnn.from_torch(
            up_proj.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.down_proj = ttnn.from_torch(
            down_proj.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    @compare_to_torch(
        reference_fn=lambda self, x: self._ref_call(x),
        metric_tolerances={
            # Metric.MAX_ABS_ERROR: 2e-1,
            # Metric.MEAN_ABS_ERROR: 2e-2,
            Metric.PCC: 0.95,
        },
        enabled=False,
    )
    def __call__(self, x):
        # SwiGLU activation: gate(x) * up(x) then down projection.
        # Output every matmul in fp32 so the caller's residual add stays in fp32 (avoids bf16
        # quantization that dominates remaining PCC drift in deep layers).
        ck_kw = dict(compute_kernel_config=self.compute_kernel_config) if self.compute_kernel_config is not None else {}
        ck_kw["dtype"] = ttnn.float32
        gate = ttnn.matmul(x, self.gate_proj, **ck_kw)
        gate = ttnn.silu(gate)

        up = ttnn.matmul(x, self.up_proj, **ck_kw)

        hidden = ttnn.mul(gate, up)
        output = ttnn.matmul(hidden, self.down_proj, **ck_kw)

        return output

    def _ref_call(self, x_torch):
        # Use the HF MLP module only; no manual fallback
        if self.ref_module is None:
            raise RuntimeError("HF MLP reference module not set for validation")
        mod_dtype = next(self.ref_module.parameters()).dtype
        return self.ref_module(x_torch.to(mod_dtype))


class TransformerBlock:
    """Single transformer layer"""

    def __init__(self, layer_id: int, config, layer_weights, device, max_seq_len: int = 2048, hf_layer=None):
        self.layer_id = layer_id

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
            raise RuntimeError(
                "Qwen3 checkpoint requires self_attn.q_norm.weight and self_attn.k_norm.weight in layer weights; "
                f"missing for layer {layer_id}."
            )

        # Attention
        self.attention = Attention(
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
        )

        # MLP — share the HF-grade compute kernel config built by the attention layer so every
        # large bf16 matmul (gate/up/down) runs with fp32 destination accumulator.
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            gate_proj=layer_weights["mlp.gate_proj.weight"],
            up_proj=layer_weights["mlp.up_proj.weight"],
            down_proj=layer_weights["mlp.down_proj.weight"],
            device=device,
            compute_kernel_config=self.attention._mm_ck,
        )

        # Norms
        self.input_layernorm = RMSNorm(layer_weights["input_layernorm.weight"], config.rms_norm_eps, device)
        self.post_attention_layernorm = RMSNorm(
            layer_weights["post_attention_layernorm.weight"], config.rms_norm_eps, device
        )

        # Hook up HF reference submodules for validation if provided
        if hf_layer is not None:
            self.hf_layer = hf_layer
            self._hf_past_kv = None  # HF DynamicCache for this layer
            self.attention.hf_attn = hf_layer.self_attn
            self.attention.hf_rotary_emb = _build_rotary_embedding(config)
            self.input_layernorm.ref_module = hf_layer.input_layernorm
            self.post_attention_layernorm.ref_module = hf_layer.post_attention_layernorm
            self.mlp.ref_module = hf_layer.mlp

    def _ref_call(
        self,
        x_torch: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Reference path using the HF decoder layer with proper args.

        Inputs are torch tensors; x_torch shape [batch, seq, hidden].
        """
        if not hasattr(self, "hf_layer") or self.hf_layer is None:
            raise RuntimeError("HF decoder layer not set for validation")

        x_t = x_torch
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(0)
        bsz, seq_len, _ = x_t.shape

        # Dtype match to HF module
        # Prefer attention weights dtype if present, else norm weight
        try:
            mod_dtype = self.hf_layer.self_attn.q_proj.weight.dtype
        except Exception:
            mod_dtype = next(self.hf_layer.parameters()).dtype
        x_cast = x_t.to(mod_dtype)

        # Position ids / cache_position
        device = x_cast.device
        cache_position = torch.arange(start_pos, start_pos + seq_len, device=device)
        position_ids = cache_position.unsqueeze(0).expand(bsz, -1)

        # Position embeddings (cos, sin)
        if position_embeddings is None:
            if not hasattr(self.attention, "hf_rotary_emb") or self.attention.hf_rotary_emb is None:
                raise RuntimeError("HF RotaryEmbedding module not set; cannot compute position embeddings")
            cos, sin = self.attention.hf_rotary_emb(x_cast, position_ids)
        else:
            cos, sin = position_embeddings
            cos = cos.to(x_cast.dtype)
            sin = sin.to(x_cast.dtype)

        # Maintain a dynamic cache across calls
        if self._hf_past_kv is None:
            self._hf_past_kv = DynamicCache()

        # For prefill we need a causal mask; for decode we can skip
        attn_mask = None
        if mask is not None:
            attn_mask = mask.to(x_cast.dtype)

        out = self.hf_layer(
            hidden_states=x_cast,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_value=self._hf_past_kv,
            output_attentions=False,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=(cos, sin),
        )
        # Qwen2DecoderLayer returns (hidden_states, [attn_weights?])
        hidden_states = out[0] if isinstance(out, tuple) else out
        return hidden_states

    @compare_to_torch(
        reference_fn=lambda self, x, start_pos, mask, position_embeddings: self._ref_call(
            x, start_pos, mask, position_embeddings
        ),
        input_to_torch=lambda self, x, start_pos, mask, position_embeddings: (
            self,
            to_torch_auto_compose(x).squeeze(0),
            start_pos,
            mask,
            position_embeddings,
        ),
        metric_tolerances={
            # Metric.MAX_ABS_ERROR: 2e-1,
            Metric.PCC: 0.99,
        },
        # Disabled: Qwen3 uses alternating full / sliding_attention layers. The prefill mask here is a
        # plain triu(S,S); HF expects per-layer masks from create_causal_mask / create_sliding_window_causal_mask.
        # Running the full DecoderLayer ref with the wrong mask breaks SDPA (e.g. seq 424 vs 213 on dim 3).
        enabled=False,
        return_reference_output=True,
        raise_exceptions=True,
    )
    def __call__(
        self,
        x,
        start_pos: int,
        mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        # Attention with residual
        h = self.input_layernorm(x)
        h = self.attention(h, start_pos, mask, position_embeddings=position_embeddings)
        x = ttnn.add(x, h)

        # MLP with residual
        h = self.post_attention_layernorm(x)
        h = self.mlp(h)
        x = ttnn.add(x, h)

        return x


class QwenModel:
    """Complete Qwen transformer model"""

    def __init__(self, model_name: str, device, max_seq_len: int = 2048, *, validate_against_hf: bool = True):
        print(f"Loading {model_name}...")

        # Load HuggingFace model
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        self.config = hf_model.config
        self.device = device
        self.max_seq_len = max_seq_len
        # Keep HF model as reference when validation is enabled (host RAM heavy; off for demo inference).
        self._validate = bool(validate_against_hf)
        self.hf_model = hf_model.eval() if self._validate else None
        # HF cache for decode validation
        self._hf_past_kv = None
        # HF rotary embedding for position embeddings parity
        self.rotary_emb = _build_rotary_embedding(self.config)

        print(f"Model config:")
        print(f"  Hidden size: {self.config.hidden_size}")
        print(f"  Num layers: {self.config.num_hidden_layers}")
        print(f"  Num attention heads: {self.config.num_attention_heads}")
        print(f"  Num KV heads: {self.config.num_key_value_heads}")
        print(f"  Intermediate size: {self.config.intermediate_size}")
        print(f"  Vocab size: {self.config.vocab_size}")

        # Extract weights
        state_dict = hf_model.state_dict()

        # Token embedding weights on device (``ttnn.embedding``); no host gather per forward.
        emb_w = state_dict["model.embed_tokens.weight"].to(torch.bfloat16).contiguous()
        self.embed_tokens_tt = ttnn.from_torch(
            emb_w,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        del emb_w

        # Build transformer layers
        print("Building transformer layers...")
        self.layers = []
        for layer_id in range(self.config.num_hidden_layers):
            layer_weights = {
                key.replace(f"model.layers.{layer_id}.", ""): value
                for key, value in state_dict.items()
                if f"model.layers.{layer_id}." in key
            }

            layer = TransformerBlock(
                layer_id=layer_id,
                config=self.config,
                layer_weights=layer_weights,
                device=device,
                max_seq_len=max_seq_len,
                hf_layer=(hf_model.model.layers[layer_id] if self._validate else None),
            )
            self.layers.append(layer)

            if (layer_id + 1) % 4 == 0:
                print(f"  Loaded {layer_id + 1}/{self.config.num_hidden_layers} layers")

        # Final norm
        self.norm = RMSNorm(state_dict["model.norm.weight"], self.config.rms_norm_eps, device)
        # Hook HF final norm for validation
        if self._validate:
            self.norm.ref_module = hf_model.model.norm

        # LM head (output projection) — large bf16 matmul (hidden=2048 -> vocab~152k);
        # use HF-grade compute kernel so the per-token logits inherit the same fp32 accumulator
        # precision as Q/K/V/O / MLP.
        self.lm_head = ttnn.from_torch(
            state_dict["lm_head.weight"].T.unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        self._lm_head_mm_ck = _hf_grade_compute_kernel_config(device)

        print("Model loaded successfully!")

        # Free HF model weights if not validating
        if not self._validate:
            del hf_model
        del state_dict

    # Reference forward using the original HF model (host/CPU)
    # Supports both prefill (start_pos == 0) and decode (start_pos > 0) by
    # maintaining HF past_key_values alongside the TTNN KV cache.
    def _ref_forward(self, tokens: torch.Tensor, start_pos: int = 0):
        with torch.inference_mode():
            if self.hf_model is None:
                raise RuntimeError("HF reference model not available for validation")

            # Prefill path: compute logits over full prompt and initialize cache
            if start_pos == 0:
                out = self.hf_model(input_ids=tokens, use_cache=True)
                # Store past for subsequent decode steps
                self._hf_past_kv = out.past_key_values if hasattr(out, "past_key_values") else None
                return out.logits  # [batch, seq, vocab]

            # Decode path: use past_kv and only pass new tokens
            out = self.hf_model(input_ids=tokens, use_cache=True, past_key_values=self._hf_past_kv)
            self._hf_past_kv = out.past_key_values if hasattr(out, "past_key_values") else self._hf_past_kv
            return out.logits  # [batch, seq(=len(tokens)), vocab]

    @compare_to_torch(
        reference_fn=lambda self, tokens, start_pos=0: self._ref_forward(tokens, start_pos),
        metric_tolerances={
            # Metric.MAX_ABS_ERROR: 1e-1,
            # Metric.MEAN_ABS_ERROR: 1e-2,
            Metric.PCC: 0.90,
        },
        enabled=False,
        return_reference_output=True,
    )
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):  # [batch, seq_len]
        h_tt, mask, position_embeddings = build_prefix_ttnn(
            tokens=tokens,
            embed_tokens_tt=self.embed_tokens_tt,
            device=self.device,
            hidden_size=int(self.config.hidden_size),
            start_pos=int(start_pos),
            rotary_emb=self.rotary_emb,
        )
        # Upgrade the residual stream to fp32. The bf16 ttnn.add at deep layers (mean_abs ~700,
        # max ~3000) was quantizing residuals to multiples of ~2-16, which dominated the
        # remaining PCC drift after Q/K/V/O/MLP/lm_head matmuls were moved to fp32 dest accum.
        # With fp32 residual + fp32 matmul outputs, the only bf16 step is the matmul **input**,
        # which HF Qwen3 also uses (weights are bf16, not fp32).
        h_tt = ttnn.typecast(h_tt, dtype=ttnn.float32)

        for layer in self.layers:
            h_tt = layer(h_tt, start_pos, mask, position_embeddings=position_embeddings)

        # Final norm
        h_tt = self.norm(h_tt)

        # LM head — input is fp32 (preserved residual precision), weight bf16, output fp32 logits.
        lm_head_kw = dict(dtype=ttnn.float32)
        if self._lm_head_mm_ck is not None:
            lm_head_kw["compute_kernel_config"] = self._lm_head_mm_ck
        logits = ttnn.matmul(h_tt, self.lm_head, **lm_head_kw)

        return logits

    def reset_kv_cache(self):
        """Clear KV cache for all layers"""
        for layer in self.layers:
            layer.attention.cache_k.zero_()
            layer.attention.cache_v.zero_()
            # Reset reference HF attention cache if present
            if hasattr(layer.attention, "hf_past_kv"):
                layer.attention.hf_past_kv = None
        # Reset HF cache used for validation during decode
        self._hf_past_kv = None


def generate_hf_ref(model: QwenModel, tokenizer, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0):
    """Generate text using HF reference path for speed/validation."""

    # Tokenize
    tokens = tokenizer.encode(prompt, return_tensors="pt")

    # Reset caches (also clears HF past_kv)
    model.reset_kv_cache()

    # Prefill using HF model
    logits = model._ref_forward(tokens, start_pos=0)  # [1, seq_len, vocab]
    next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()

    generated = [next_token_id]
    current_pos = tokens.shape[1]

    # Decode using cached HF past_kv
    for _ in range(max_new_tokens - 1):
        next_token = torch.tensor([[next_token_id]], dtype=torch.long)
        logits = model._ref_forward(next_token, start_pos=current_pos)  # [1, 1, vocab]

        if temperature > 0:
            probs = torch.nn.functional.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token_id = torch.multinomial(probs.squeeze(0), num_samples=1).item()
        else:
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()

        if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
            break

        generated.append(next_token_id)
        current_pos += 1

    return tokenizer.decode(generated, skip_special_tokens=True)


def generate(model: QwenModel, tokenizer, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0):
    """Generate text from prompt"""

    # Tokenize
    tokens = tokenizer.encode(prompt, return_tensors="pt")

    # Reset cache
    model.reset_kv_cache()

    # Prefill phase
    print(f"Prefill: {tokens.shape[1]} tokens")
    logits = model.forward(tokens, start_pos=0)

    # Get last token logits
    logits_torch = to_torch_auto_compose(logits)
    if len(logits.shape) > 3:
        logits_torch = logits_torch.squeeze(0)  # [batch, seq_len, vocab_size]
    next_token_id = torch.argmax(logits_torch[:, -1, :], dim=-1).item()  # Take last position

    generated = [next_token_id]
    current_pos = tokens.shape[1]

    # Decode phase
    print("Generating...")
    for i in range(max_new_tokens - 1):
        next_token = torch.tensor([[next_token_id]], dtype=torch.long)  # [1, 1]
        logits = model.forward(next_token, start_pos=current_pos)
        logits_torch = to_torch_auto_compose(logits)
        if len(logits.shape) > 3:
            logits_torch = logits_torch.squeeze(0)  # [batch, 1, vocab_size]

        # Sample next token
        if temperature > 0:
            probs = torch.nn.functional.softmax(logits_torch[:, -1, :] / temperature, dim=-1)
            next_token_id = torch.multinomial(probs.squeeze(0), num_samples=1).item()
        else:
            next_token_id = torch.argmax(logits_torch[:, -1, :], dim=-1).item()

        # Check for EOS
        if next_token_id == tokenizer.eos_token_id:
            break

        generated.append(next_token_id)
        current_pos += 1

        # Print token as it's generated
        if i % 5 == 0:
            print(".", end="", flush=True)

    print()
    return tokenizer.decode(generated, skip_special_tokens=True)


def test_ds_r1_qwen(is_ci_v2_env, model_location_generator):
    if is_ci_v2_env:
        hf_model = os.getenv("HF_MODEL", "")
        model_location = model_location_generator(hf_model, download_if_ci_v2=True, ci_v2_timeout_in_s=900)
        # update env var HF_MODEL to the model location
        os.environ["HF_MODEL"] = str(model_location)

    model_name = os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    print(f"Pure TTNN implementation of {model_name}")
    print("=" * 80)

    # Setup device
    if os.environ.get("MESH_DEVICE") == "N150":
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([1, 1]))
    elif os.environ.get("MESH_DEVICE") == "N300":
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([1, 2]))
    elif os.environ.get("MESH_DEVICE") == "T3K":
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([1, 8]))
    elif os.environ.get("MESH_DEVICE") == "T3K2x4":
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([2, 4]))
    else:
        device_ids = ttnn.get_device_ids()
        num_devices = len(device_ids)
        if num_devices >= 1:
            mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([1, 1]))
        else:
            raise RuntimeError("No devices found")

    print(f"Using {mesh_device.get_num_devices()} device(s)")
    print()

    # Get validation registry
    registry = get_validation_registry()
    registry.enabled = True  # we can use this to disable validation using the validations tools

    # Load model
    model = QwenModel(model_name, mesh_device, max_seq_len=2048)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare prompt
    messages = [{"role": "user", "content": "What is 2+2? Answer briefly."}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print()
    print("Prompt:", prompt)
    print()

    # Generate
    # response = generate_hf_ref(model, tokenizer, prompt, max_new_tokens=50)
    response = generate(model, tokenizer, prompt, max_new_tokens=50)

    print()
    print("Response:", response)
    print()

    # Print validation report if any validations were run
    if registry.results:
        registry.print_report(verbose=True)

    # Cleanup
    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    test_ds_r1_qwen(is_ci_v2_env=False, model_location_generator=None)
