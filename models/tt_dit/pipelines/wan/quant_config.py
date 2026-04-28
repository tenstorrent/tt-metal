# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Quantization and compute precision configuration for the Wan pipeline.

Usage:
    from models.tt_dit.pipelines.wan.quant_config import QuantConfig, apply_quant_config

    config = QuantConfig.all_weights_bf8()
    apply_quant_config(model, config)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

import ttnn

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LinearQuantConfig:
    """Precision config for a single linear layer type."""

    weight_dtype: ttnn.DataType = ttnn.bfloat16  # Weight storage dtype
    activation_dtype: ttnn.DataType | None = None  # None = no cast (activation quantization deferred)
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    fp32_dest_acc: bool = True


@dataclass
class SDPAQuantConfig:
    """Precision config for ring SDPA (self-attention)."""

    input_dtype: ttnn.DataType | None = None  # None = no cast
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    fp32_dest_acc: bool = False


@dataclass
class QuantConfig:
    """Per-component quantization and compute precision configuration.

    Each transformer block has 7 linear layers and 1 ring SDPA (self-attention).
    Cross-attention SDPA stays at default settings.
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
    def default() -> QuantConfig:
        """All bfloat16, HiFi2 -- matches current baseline."""
        lc = LinearQuantConfig()
        return QuantConfig(
            self_attn_qkv=lc,
            self_attn_out=lc,
            cross_attn_q=lc,
            cross_attn_kv=lc,
            cross_attn_out=lc,
            ffn_ff1=lc,
            ffn_ff2=lc,
            ring_sdpa=SDPAQuantConfig(),
        )

    @staticmethod
    def all_weights_bf8() -> QuantConfig:
        """All linear weights bfloat8_b, rest default."""
        lc = LinearQuantConfig(weight_dtype=ttnn.bfloat8_b)
        return QuantConfig(
            self_attn_qkv=lc,
            self_attn_out=lc,
            cross_attn_q=lc,
            cross_attn_kv=lc,
            cross_attn_out=lc,
            ffn_ff1=lc,
            ffn_ff2=lc,
            ring_sdpa=SDPAQuantConfig(),
        )

    @staticmethod
    def all_lofi() -> QuantConfig:
        """All compute LoFi, rest default."""
        lc = LinearQuantConfig(math_fidelity=ttnn.MathFidelity.LoFi)
        sc = SDPAQuantConfig(math_fidelity=ttnn.MathFidelity.LoFi)
        return QuantConfig(
            self_attn_qkv=lc,
            self_attn_out=lc,
            cross_attn_q=lc,
            cross_attn_kv=lc,
            cross_attn_out=lc,
            ffn_ff1=lc,
            ffn_ff2=lc,
            ring_sdpa=sc,
        )

    @staticmethod
    def all_bf8_lofi() -> QuantConfig:
        """All weights + activations bfloat8_b, LoFi compute, SDPA bf8 HiFi2."""
        lc = LinearQuantConfig(
            weight_dtype=ttnn.bfloat8_b,
            activation_dtype=ttnn.bfloat8_b,
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc=False,
        )
        sc = SDPAQuantConfig(
            input_dtype=ttnn.bfloat8_b,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc=False,
        )
        return QuantConfig(
            self_attn_qkv=lc,
            self_attn_out=lc,
            cross_attn_q=lc,
            cross_attn_kv=lc,
            cross_attn_out=lc,
            ffn_ff1=lc,
            ffn_ff2=lc,
            ring_sdpa=sc,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _quantize_weight(param, new_dtype: ttnn.DataType) -> None:
    """Typecast a Parameter's weight tensor in-place, bypassing dtype check."""
    if param.dtype == new_dtype:
        return
    param._data = ttnn.typecast(param._data, new_dtype)
    param.dtype = new_dtype


def _make_linear_compute_config(arch, lc: LinearQuantConfig):
    return ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=lc.math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=lc.fp32_dest_acc,
        packer_l1_acc=True,
    )


def _make_sdpa_compute_config(arch, sc: SDPAQuantConfig):
    return ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=sc.math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=sc.fp32_dest_acc,
    )


def _apply_linear_config(linear, lc: LinearQuantConfig, arch, name: str) -> None:
    """Apply quantization + compute config to a single linear layer."""
    if lc.weight_dtype != ttnn.bfloat16:
        _quantize_weight(linear.weight, lc.weight_dtype)
        if hasattr(linear, "bias") and linear.bias is not None:
            _quantize_weight(linear.bias, lc.weight_dtype)
        logger.debug(f"  {name}: weights -> {lc.weight_dtype}")

    linear.compute_config = _make_linear_compute_config(arch, lc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_quant_config(model, config: QuantConfig) -> None:
    """Apply quantization config to a WanTransformer3DModel after weight loading.

    Modifies the model in-place:
    - Typecasts weight parameters to the configured dtype
    - Replaces compute_config on each linear layer
    - Replaces mm_compute_kernel_config on attention blocks
    - Replaces ff_compute_kernel_config on transformer blocks
    - Replaces sdpa_compute_kernel_config on self-attention

    Wan-specific attribute mapping:
    - Self-attn QKV matmuls: block.attn1.mm_compute_kernel_config
    - Self-attn output: block.attn1.to_out.compute_config
    - Cross-attn matmuls: block.attn2.mm_compute_kernel_config
    - Cross-attn output: block.attn2.to_out.compute_config
    - FFN: block.ff_compute_kernel_config
    - Ring SDPA: block.attn1.sdpa_compute_kernel_config
    """
    arch = model.mesh_device.arch()
    n_blocks = len(model.blocks)

    logger.info(f"Applying quant config to {n_blocks} transformer blocks")

    for idx, block in enumerate(model.blocks):
        # Self-attention linears
        _apply_linear_config(block.attn1.to_qkv, config.self_attn_qkv, arch, f"blocks[{idx}].attn1.to_qkv")
        _apply_linear_config(block.attn1.to_out, config.self_attn_out, arch, f"blocks[{idx}].attn1.to_out")

        # Self-attention mm_compute_kernel_config (used by matmuls in attention forward)
        block.attn1.mm_compute_kernel_config = _make_linear_compute_config(arch, config.self_attn_qkv)

        # Cross-attention linears
        _apply_linear_config(block.attn2.to_q, config.cross_attn_q, arch, f"blocks[{idx}].attn2.to_q")
        _apply_linear_config(block.attn2.to_kv, config.cross_attn_kv, arch, f"blocks[{idx}].attn2.to_kv")
        _apply_linear_config(block.attn2.to_out, config.cross_attn_out, arch, f"blocks[{idx}].attn2.to_out")

        # Cross-attention mm_compute_kernel_config
        block.attn2.mm_compute_kernel_config = _make_linear_compute_config(arch, config.cross_attn_q)

        # FFN linears
        _apply_linear_config(block.ffn.ff1, config.ffn_ff1, arch, f"blocks[{idx}].ffn.ff1")
        _apply_linear_config(block.ffn.ff2, config.ffn_ff2, arch, f"blocks[{idx}].ffn.ff2")

        # FFN block-level compute config
        block.ff_compute_kernel_config = _make_linear_compute_config(arch, config.ffn_ff1)

        # Ring SDPA config (self-attention only)
        block.attn1.sdpa_compute_kernel_config = _make_sdpa_compute_config(arch, config.ring_sdpa)

    # Log summary
    weight_dtypes = set()
    for lc in [
        config.self_attn_qkv,
        config.self_attn_out,
        config.cross_attn_q,
        config.cross_attn_kv,
        config.cross_attn_out,
        config.ffn_ff1,
        config.ffn_ff2,
    ]:
        weight_dtypes.add(lc.weight_dtype)
    fidelities = set()
    for lc in [
        config.self_attn_qkv,
        config.self_attn_out,
        config.cross_attn_q,
        config.cross_attn_kv,
        config.cross_attn_out,
        config.ffn_ff1,
        config.ffn_ff2,
    ]:
        fidelities.add(lc.math_fidelity)

    logger.info(f"  Weight dtypes: {weight_dtypes}")
    logger.info(f"  Math fidelities: {fidelities}")
    logger.info(
        f"  Ring SDPA: input_dtype={config.ring_sdpa.input_dtype}, "
        f"fidelity={config.ring_sdpa.math_fidelity}, fp32_acc={config.ring_sdpa.fp32_dest_acc}"
    )
