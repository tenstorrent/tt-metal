# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Quantization and compute precision configuration for the LTX pipeline.

Ported from ``pipelines/wan/quant_config.py``: the config dataclass field definitions are
identical. The presets encode LTX's block structure (attn1/attn2/ffn, plus an optional audio
path) and its carve-outs.

The config is consumed at module CONSTRUCTION. ``LTXAttention``, ``LTXTransformerBlock`` and
``ParallelFeedForward`` take a ``quant_config`` and bake the preset's weight dtypes, activation
casts, output pins and compute-kernel fidelities into the modules as they are built, so a cache
miss loads weights direct-to-quant and the cache write holds the quantized tensorbins. The LTX
linears do NOT read each ``Linear.compute_config`` — the attention matmuls take
``LTXAttention.mm_compute_kernel_config`` and the FFN takes
``LTXTransformerBlock.ff_compute_kernel_config``, so a fidelity change sets those attributes.

The pipeline resolves the ``LTX_QUANT`` preset before building the transformer and threads the
``QuantConfig`` through construction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import ttnn

# When set, activations are narrowed to bf8 wherever they cross the fabric — both the linear activation
# casts and the ring-SDPA Q/K/V inputs — so this is the one knob that changes what the collectives move
# (the linear payloads and the SP-gathered K/V; SDPA itself stays HiFi2, only its inputs narrow). More
# aggressive than the weight quant it rides on: weights are a fixed, well-conditioned distribution,
# activations are not. On by default because the shipped 1080p tier is measured and VBench-gated with it
# on; set to 0 to A/B against the weight-only preset. Keeping the SDPA input cast on this same knob means
# a run can't narrow the linear activations while silently leaving SDPA at bf16 (or vice versa).
LTX_QUANT_ACTIVATIONS = os.environ.get("LTX_QUANT_ACTIVATIONS", "1") in ("1", "true", "True")

# ---------------------------------------------------------------------------
# Config dataclasses (field definitions identical to the Wan template)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearQuantConfig:
    """Precision config for a single linear layer type."""

    weight_dtype: ttnn.DataType = ttnn.bfloat16  # Weight storage dtype
    activation_dtype: ttnn.DataType | None = None  # None = no cast (activation quantization deferred)
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    fp32_dest_acc: bool = True


@dataclass(frozen=True)
class SDPAQuantConfig:
    """Precision config for ring SDPA (self-attention)."""

    input_dtype: ttnn.DataType | None = None  # None = no cast
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    fp32_dest_acc: bool = False


@dataclass
class QuantConfig:
    """Per-component quantization and compute precision configuration.

    LTX self-attn (attn1): to_qkv, to_out. Cross-attn (attn2): to_q, to_kv, to_out.
    FFN: ff1, ff2. Ring SDPA is the self-attention; cross-attention SDPA stays default.
    """

    # Self-attention projections
    self_attn_qkv: LinearQuantConfig = field(default_factory=LinearQuantConfig)
    self_attn_out: LinearQuantConfig = field(default_factory=LinearQuantConfig)

    # Cross-attention projections
    cross_attn_q: LinearQuantConfig = field(default_factory=LinearQuantConfig)
    cross_attn_kv: LinearQuantConfig = field(default_factory=LinearQuantConfig)
    cross_attn_out: LinearQuantConfig = field(default_factory=LinearQuantConfig)

    # FFN
    ffn_ff1: LinearQuantConfig = field(default_factory=LinearQuantConfig)
    ffn_ff2: LinearQuantConfig = field(default_factory=LinearQuantConfig)

    # Ring SDPA (self-attention only; cross-attention stays default)
    ring_sdpa: SDPAQuantConfig = field(default_factory=SDPAQuantConfig)

    @staticmethod
    def all_bf8_lofi() -> QuantConfig:
        """Weights bfloat8_b, LoFi compute. SDPA stays bf16/HiFi2.

        ``activation_dtype`` is bf8 and active by default (``LTX_QUANT_ACTIVATIONS`` defaults on), so
        the collectives move bf8; set ``LTX_QUANT_ACTIVATIONS=0`` for a matmul-internal-only cast that
        keeps the collectives at bf16.

        Carve-out: both ``self_attn_out`` (attn1) and the video ``cross_attn_out`` (attn2)
        run the fused ``dit_minimal_matmul_addcmul_fused`` / ``all_gather_minimal_matmul_async``
        epilogue (see ``attention_ltx.py:_to_out_fused_addcmul``, called from the block's
        attn1 and the cross_attention_adaln attn2). That kernel's ternary addcmul inputs
        (residual, gate) are bf16 and must match the weight tile format, so those weights
        stay bf16. ``ffn_ff2`` uses the RowParallel RS-fused addcmul, which Wan runs at bf8
        with no issue, so it is quantized. SDPA stays fully unquantized here (FastVideo kept
        attention higher precision); ``LTX_QUANT_ACTIVATIONS`` also narrows the SDPA inputs to bf8.
        """
        lc = LinearQuantConfig(
            weight_dtype=ttnn.bfloat8_b,
            activation_dtype=ttnn.bfloat8_b,
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc=False,
        )
        # Fused matmul+addcmul (attn1 to_out and video attn2 to_out): bf16 weights to match
        # the bf16 ternary addcmul inputs. Compute stays LoFi.
        lc_out = LinearQuantConfig(
            weight_dtype=ttnn.bfloat16,
            activation_dtype=ttnn.bfloat8_b,
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc=False,
        )
        sc = SDPAQuantConfig()  # SDPA unchanged: bf16 / HiFi2
        return QuantConfig(
            self_attn_qkv=lc,
            self_attn_out=lc_out,
            cross_attn_q=lc,
            cross_attn_kv=lc,
            cross_attn_out=lc_out,
            ffn_ff1=lc,
            ffn_ff2=lc,
            ring_sdpa=sc,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_compute_config(arch, math_fidelity, fp32_dest_acc, math_approx_mode=False, packer_l1_acc=True):
    return ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=math_fidelity,
        math_approx_mode=math_approx_mode,
        fp32_dest_acc_en=fp32_dest_acc,
        packer_l1_acc=packer_l1_acc,
    )
