# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN implementation of one dots.ocr Qwen2 language-model decoder layer.

Reference: models/demos/rednote_hilab_dots.ocr/reference/functional.py
           :func:`decoder_layer_forward`

Qwen2DecoderLayer (pre-norm, residual):

    residual = h
    h = input_layernorm(h)
    h = residual + self_attn(h)
    residual = h
    h = post_attention_layernorm(h)
    h = residual + mlp(h)

This is the LM composite block. It does NOT re-implement the leaf maths -- it
imports and composes the already-verified leaf modules:

    TtRMSNorm  (x2, eps 1e-6)        -- tt/rmsnorm.py
    TtAttention (GQA 12/2, QKV bias, 1D RoPE theta 1e6, causal) -- tt/attention.py
    TtMLP       (SwiGLU FFN, no bias) -- tt/mlp.py

The 1D-RoPE cos/sin tables and the additive causal mask are threaded into
TtAttention exactly as the standalone attention block did (precomputed on host
at construction time, uploaded to device like the norm gamma weight). The
forward() runs entirely with ttnn ops (no host-side matmul / softmax / activation).

hidden_size 1536, num_heads 12, num_kv_heads 2, head_dim 128, intermediate 8960,
rms_norm_eps 1e-6, attention_bias True.

The model dir name (rednote_hilab_dots.ocr) contains a dot, so the sibling leaf
modules cannot be imported via the normal dotted package path -- they are loaded
by file path with importlib (the same convention the tests use).

Reference TTNN impl this follows: models/demos/rednote_hilab_dots.ocr/tt/vision_block.py
"""
import importlib.util
import os

import ttnn
from models.common.lightweightmodule import LightweightModule

_TT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_leaf(module_name: str, file_name: str, symbol: str):
    """Import a sibling leaf module by file path (dir name has a dot)."""
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(_TT_DIR, file_name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, symbol)


TtRMSNorm = _load_leaf("dots_tt_rmsnorm", "rmsnorm.py", "TtRMSNorm")
TtAttention = _load_leaf("dots_tt_attention", "attention.py", "TtAttention")
TtMLP = _load_leaf("dots_tt_mlp", "mlp.py", "TtMLP")


class TtDecoderLayer(LightweightModule):
    """dots.ocr Qwen2 LM decoder layer (pre-norm residual).

    Composes the three verified leaf modules. Args mirror the layer golden's
    prefixed state_dict (input_layernorm / self_attn.{q,k,v,o}_proj /
    post_attention_layernorm / mlp.{gate,up,down}_proj).

    Args:
        device: ttnn Device or MeshDevice.
        input_layernorm_weight: torch.Tensor [hidden] -- pre-attention RMSNorm gamma.
        q_weight/k_weight/v_weight: torch projection weights.
        q_bias/k_bias/v_bias: torch projection biases.
        o_weight: torch [hidden, hidden] output projection weight (no bias).
        post_attention_layernorm_weight: torch.Tensor [hidden] -- pre-MLP RMSNorm gamma.
        gate_weight/up_weight/down_weight: torch SwiGLU MLP weights (no bias).
        cos, sin: torch [seq, head_dim] 1D-RoPE tables (theta 1e6).
        attention_mask: torch additive causal mask [seq, seq].
        seq_len: sequence length.
        num_heads: 12 query heads.
        num_kv_heads: 2 KV heads.
        head_dim: 128.
        eps: RMSNorm epsilon (1e-6).
        dtype: activation/weight dtype (bf16).
    """

    def __init__(
        self,
        device,
        input_layernorm_weight,
        q_weight,
        k_weight,
        v_weight,
        q_bias,
        k_bias,
        v_bias,
        o_weight,
        post_attention_layernorm_weight,
        gate_weight,
        up_weight,
        down_weight,
        cos,
        sin,
        attention_mask,
        seq_len,
        num_heads: int = 12,
        num_kv_heads: int = 2,
        head_dim: int = 128,
        eps: float = 1e-6,
        dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device
        hidden = num_heads * head_dim

        self.input_layernorm = TtRMSNorm(
            device=device,
            dim=hidden,
            weight=input_layernorm_weight,
            eps=eps,
            weight_dtype=dtype,
            weight_memory_config=weight_memory_config,
        )
        self.self_attn = TtAttention(
            device=device,
            q_weight=q_weight,
            k_weight=k_weight,
            v_weight=v_weight,
            q_bias=q_bias,
            k_bias=k_bias,
            v_bias=v_bias,
            o_weight=o_weight,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            seq_len=seq_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            weight_memory_config=weight_memory_config,
        )
        self.post_attention_layernorm = TtRMSNorm(
            device=device,
            dim=hidden,
            weight=post_attention_layernorm_weight,
            eps=eps,
            weight_dtype=dtype,
            weight_memory_config=weight_memory_config,
        )
        self.mlp = TtMLP(
            device=device,
            gate_weight=gate_weight,
            up_weight=up_weight,
            down_weight=down_weight,
            dtype=dtype,
            weight_memory_config=weight_memory_config,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [seq, hidden] (TILE layout) -> [seq, hidden].

        residual = h; h = residual + attn(input_layernorm(h));
        residual = h; h = residual + mlp(post_attention_layernorm(h)).
        """
        attn_out = self.self_attn(self.input_layernorm(x))
        x = ttnn.add(x, attn_out)

        mlp_out = self.mlp(self.post_attention_layernorm(x))
        x = ttnn.add(x, mlp_out)
        return x
