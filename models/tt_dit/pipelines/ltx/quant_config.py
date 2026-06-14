# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Quantization and compute precision configuration for the LTX pipeline.

Ported from ``pipelines/wan/quant_config.py``. The dataclasses are identical; only
``apply_quant_config``'s block-walk differs — LTX block structure (attn1/attn2/ffn,
plus an optional audio path) and its linear hook surface differ from Wan's.

LTX hook surface: the LTX linears do NOT read each ``Linear.compute_config`` — the
attention matmuls take ``LTXAttention.mm_compute_kernel_config`` explicitly and the
FFN takes ``LTXTransformerBlock.ff_compute_kernel_config``. So a fidelity change here
means rewriting those two attributes, not per-linear ``compute_config``.

Usage:
    from models.tt_dit.pipelines.ltx.quant_config import QuantConfig, set_quant_config
    set_quant_config(pipeline, QuantConfig.all_bf8_lofi())
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

import ttnn

# ---------------------------------------------------------------------------
# Config dataclasses (identical to the Wan template)
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
    def all_bf8_lofi() -> QuantConfig:
        """Weights + activations bfloat8_b, LoFi compute. SDPA stays bf16/HiFi2.

        Carve-out: both ``self_attn_out`` (attn1) and the video ``cross_attn_out`` (attn2)
        run the fused ``dit_minimal_matmul_addcmul_fused`` / ``all_gather_minimal_matmul_async``
        epilogue (see ``attention_ltx.py:_to_out_fused_addcmul``, called from the block's
        attn1 and the cross_attention_adaln attn2). That kernel's ternary addcmul inputs
        (residual, gate) are bf16 and must match the weight tile format, so those weights
        stay bf16. ``ffn_ff2`` uses the RowParallel RS-fused addcmul, which Wan runs at bf8
        with no issue, so it is quantized. SDPA stays fully unquantized for the first landing
        (FastVideo kept attention higher precision; casting SDPA inputs to bf8 is a separate
        bandwidth-only tweak).
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


def _quantize_weight(param, new_dtype: ttnn.DataType) -> None:
    """Typecast a Parameter's weight tensor in-place, bypassing dtype check."""
    if param is None or param._data is None:
        return
    if param.dtype == new_dtype:
        return
    param._data = ttnn.typecast(param._data, new_dtype)
    param.dtype = new_dtype


def _make_compute_config(arch, math_fidelity, fp32_dest_acc, math_approx_mode=False, packer_l1_acc=True):
    return ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=math_fidelity,
        math_approx_mode=math_approx_mode,
        fp32_dest_acc_en=fp32_dest_acc,
        packer_l1_acc=packer_l1_acc,
    )


def _quantize_linear_weights(linear, lc: LinearQuantConfig) -> None:
    """Typecast a linear's weight (and bias) to the configured dtype, if loaded."""
    if linear is None or lc.weight_dtype == ttnn.bfloat16:
        return
    _quantize_weight(linear.weight, lc.weight_dtype)
    if getattr(linear, "bias", None) is not None:
        _quantize_weight(linear.bias, lc.weight_dtype)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_quant_config(model, config: QuantConfig) -> None:
    """Apply quantization config to an LTXTransformerModel.

    Safe whether or not weights are loaded: weight typecasts no-op when ``_data is None``,
    and the compute-config attributes are always (re)set. With ``dynamic_load=True`` the
    pipeline evicts/reloads transformer weights, so this must run again after each reload
    (see ``set_quant_config``).

    LTX attribute mapping (per block):
    - Self-attn QKV / out:  block.attn1.to_qkv, block.attn1.to_out
    - Self-attn matmul fidelity:  block.attn1.mm_compute_kernel_config
    - Cross-attn Q / KV / out:  block.attn2.to_q, block.attn2.to_kv, block.attn2.to_out
    - Cross-attn matmul fidelity:  block.attn2.mm_compute_kernel_config
    - FFN:  block.ffn.ff1, block.ffn.ff2, block.ff_compute_kernel_config
    - Ring SDPA fidelity:  block.attn1.sdpa_compute_kernel_config
    Audio block linears (audio_attn1/2, audio_ff, a2v/v2a cross) mirror the video ones.
    """
    arch = model.mesh_device.arch()
    blocks = model.transformer_blocks
    n_blocks = len(blocks)
    logger.info(f"Applying LTX quant config to {n_blocks} transformer blocks (has_audio={model.has_audio})")

    qkv_compute = _make_compute_config(arch, config.self_attn_qkv.math_fidelity, config.self_attn_qkv.fp32_dest_acc)
    cross_compute = _make_compute_config(arch, config.cross_attn_q.math_fidelity, config.cross_attn_q.fp32_dest_acc)
    ffn_compute = _make_compute_config(arch, config.ffn_ff1.math_fidelity, config.ffn_ff1.fp32_dest_acc)
    sdpa_compute = ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=config.ring_sdpa.math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=config.ring_sdpa.fp32_dest_acc,
    )

    def _quant_self_attn(attn):
        if attn is None:
            return
        _quantize_linear_weights(attn.to_qkv, config.self_attn_qkv)
        _quantize_linear_weights(attn.to_out, config.self_attn_out)
        attn.mm_compute_kernel_config = qkv_compute
        attn.sdpa_compute_kernel_config = sdpa_compute
        if config.ring_sdpa.input_dtype is not None:
            attn._sdpa_input_dtype = config.ring_sdpa.input_dtype

    def _quant_cross_attn(attn):
        if attn is None:
            return
        _quantize_linear_weights(attn.to_q, config.cross_attn_q)
        _quantize_linear_weights(attn.to_kv, config.cross_attn_kv)
        _quantize_linear_weights(attn.to_out, config.cross_attn_out)
        attn.mm_compute_kernel_config = cross_compute

    for block in blocks:
        _quant_self_attn(block.attn1)
        _quant_cross_attn(block.attn2)
        _quantize_linear_weights(block.ffn.ff1, config.ffn_ff1)
        _quantize_linear_weights(block.ffn.ff2, config.ffn_ff2)
        block.ff_compute_kernel_config = ffn_compute

        if model.has_audio:
            _quant_self_attn(getattr(block, "audio_attn1", None))
            _quant_cross_attn(getattr(block, "audio_attn2", None))
            # A2V / V2A cross-attn use a separate (non-fused) addcmul on their outputs, so their
            # to_out could be bf8; keep them on cross_attn_out (bf16 carve-out) conservatively —
            # audio is already optimized and these are small relative to the video GEMMs.
            _quant_cross_attn(getattr(block, "audio_to_video_attn", None))
            _quant_cross_attn(getattr(block, "video_to_audio_attn", None))
            audio_ff = getattr(block, "audio_ff", None)
            if audio_ff is not None:
                _quantize_linear_weights(audio_ff.ff1, config.ffn_ff1)
                _quantize_linear_weights(audio_ff.ff2, config.ffn_ff2)

    weight_dtypes = {
        config.self_attn_qkv.weight_dtype,
        config.self_attn_out.weight_dtype,
        config.cross_attn_q.weight_dtype,
        config.cross_attn_kv.weight_dtype,
        config.cross_attn_out.weight_dtype,
        config.ffn_ff1.weight_dtype,
        config.ffn_ff2.weight_dtype,
    }
    fidelities = {
        config.self_attn_qkv.math_fidelity,
        config.cross_attn_q.math_fidelity,
        config.ffn_ff1.math_fidelity,
    }
    logger.info(f"  Weight dtypes: {weight_dtypes} | Linear fidelities: {fidelities}")
    logger.info(f"  Ring SDPA: input_dtype={config.ring_sdpa.input_dtype}, fidelity={config.ring_sdpa.math_fidelity}")


def set_quant_config(pipeline, config: QuantConfig) -> None:
    """Install a quantization config on an LTX pipeline.

    Applies compute configs now (they survive eviction — module attributes, not Parameter
    data) and patches ``_prepare_transformer`` so the weight typecast re-runs after every
    reload (required under ``dynamic_load=True``, which evicts/reloads transformer weights).
    """
    for state in pipeline.transformer_states:
        apply_quant_config(state.model, config)

    if not hasattr(pipeline, "_orig_prepare_transformer"):
        pipeline._orig_prepare_transformer = pipeline._prepare_transformer
    original_prepare = pipeline._orig_prepare_transformer

    def _prepare_with_quant(idx: int = 0):
        was_loaded = pipeline.transformer_states[idx].model.is_loaded()
        original_prepare(idx)
        if not was_loaded:
            apply_quant_config(pipeline.transformer_states[idx].model, config)

    pipeline._prepare_transformer = _prepare_with_quant
