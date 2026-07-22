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

import os
from dataclasses import dataclass, field

from loguru import logger

import ttnn

# ``activation_dtype`` is honoured only when this is set. It is a more aggressive change than the
# weight quant it rides on — weights are a fixed, well-conditioned distribution, activations are not
# — and it is the one knob here that changes what the *collectives* move. It is on by default because
# the shipped 1080p tier is measured and VBench-gated with it on; set it to 0 to A/B against the
# weight-only preset.
LTX_QUANT_ACTIVATIONS = os.environ.get("LTX_QUANT_ACTIVATIONS", "1") in ("1", "true", "True")

# Forces the ring-SDPA input cast on top of whatever preset is selected. Same knob as
# ``all_bf8_lofi_sdpa_bf8``, reachable without changing the preset name — which matters because the
# pipeline's tensorbin cache is keyed on that name, so the preset form cannot be measured on the
# pipeline without re-materialising the 22B checkpoint it shares weights with.
LTX_QUANT_SDPA_BF8 = os.environ.get("LTX_QUANT_SDPA_BF8", "0") in ("1", "true", "True")

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
        attention higher precision); ``all_bf8_lofi_sdpa_bf8`` is the separate SDPA-input arm.
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

    @staticmethod
    def all_bf8_lofi_sdpa_lofi() -> QuantConfig:
        """all_bf8_lofi plus the self-attention ring SDPA dropped HiFi2 -> LoFi.

        Only the ``sdpa_compute_kernel_config`` fidelity changes; SDPA inputs stay bf16. Ring SDPA
        is the O(seq^2) self-attention (video seq ~9.7k at stage-2), so if stage-2 is attention-
        compute-bound this roughly halves those matmul phases. Cross-attention SDPA (attn2,
        seq x prompt_len) is untouched. Quality-sensitive: gate on PCC / a frame check."""
        cfg = QuantConfig.all_bf8_lofi()
        cfg.ring_sdpa = SDPAQuantConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc=False)
        return cfg

    @staticmethod
    def all_bf8_lofi_sdpa_bf8() -> QuantConfig:
        """all_bf8_lofi with the SDPA inputs (Q/K/V) cast to bf8, fidelity left at HiFi2.

        Independent of ``LTX_QUANT_ACTIVATIONS``: that flag gates the *linear* activation cast, this
        preset carries the SDPA one. On the ring paths K/V are the collective's payload, so this is a
        bandwidth lever, not just a math one — but SDPA inputs carry the block's widest dynamic range
        and SDPA-LoFi alone has already been measured failing the PCC bar, so it gates on PCC first.

        Weights are byte-identical to ``all_bf8_lofi``, but the pipeline's tensorbin cache is keyed on
        the preset *name*, so a pipeline run under this name is a cache MISS and re-materialises the
        22B checkpoint. The PCC oracle builds from the torch reference and is unaffected; a traced run
        would need the cache key taught to hash the weight dtypes instead.
        """
        cfg = QuantConfig.all_bf8_lofi()
        cfg.ring_sdpa = SDPAQuantConfig(input_dtype=ttnn.bfloat8_b, math_fidelity=ttnn.MathFidelity.HiFi2)
        return cfg

    @staticmethod
    def all_bf8_lofi_sdpa_lofi_fp32acc() -> QuantConfig:
        """all_bf8_lofi with the ring self-attention SDPA at LoFi and fp32 dest accumulation.
        fp32 dest keeps SDPA's running softmax max/sum in full precision under LoFi, where the
        packed-dest path leaves the reduced-mantissa accumulators too coarse."""
        cfg = QuantConfig.all_bf8_lofi()
        cfg.ring_sdpa = SDPAQuantConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc=True)
        return cfg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _quantize_weight(param, new_dtype: ttnn.DataType) -> None:
    """Set a Parameter's weight dtype, typecasting resident data in place.

    ``param.dtype`` must be set even when ``_data is None`` (weights not yet loaded):
    under ``dynamic_load`` the cache load runs ``Parameter.load`` which checks the
    on-disk tile dtype against ``param.dtype``, and the torch fallback casts to it — both
    need the quantized dtype set BEFORE load, or a bf8 cache hit clashes with a bf16 param.
    """
    if param is None or param.dtype == new_dtype:
        return
    if param._data is not None:
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


def _apply_linear_quant(linear, lc: LinearQuantConfig) -> None:
    """Typecast a linear's weight (and bias) to the configured dtype, and install its activation cast.

    ``activation_dtype`` is only defined on ``ColParallelLinear`` — the one variant whose input is a
    collective's payload — so the cast lands exactly where it shrinks fabric bytes and nowhere else.
    """
    if linear is None:
        return
    if LTX_QUANT_ACTIVATIONS and hasattr(linear, "activation_dtype"):
        linear.activation_dtype = lc.activation_dtype
    if lc.weight_dtype == ttnn.bfloat16:
        return
    _quantize_weight(linear.weight, lc.weight_dtype)
    if getattr(linear, "bias", None) is not None:
        _quantize_weight(linear.bias, lc.weight_dtype)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_quant_config_to_block(block, config: QuantConfig, arch, has_audio: bool) -> None:
    """Quantize a single ``LTXTransformerBlock`` in place (weights + compute configs).

    Factored out so the transformer-block PCC test can apply the exact same quant path the
    pipeline uses against the diffusers torch oracle. ``arch`` is the device arch enum.

    Per-block attribute mapping:
    - Self-attn QKV / out:  block.attn1.to_qkv, block.attn1.to_out (mm/sdpa compute on attn1)
    - Cross-attn Q / KV / out:  block.attn2.to_q, block.attn2.to_kv, block.attn2.to_out
    - FFN:  block.ffn.ff1, block.ffn.ff2, block.ff_compute_kernel_config
    Audio block linears (audio_attn1/2, audio_ff, a2v/v2a cross) mirror the video ones.
    """
    qkv_compute = _make_compute_config(arch, config.self_attn_qkv.math_fidelity, config.self_attn_qkv.fp32_dest_acc)
    cross_compute = _make_compute_config(arch, config.cross_attn_q.math_fidelity, config.cross_attn_q.fp32_dest_acc)
    ffn_compute = _make_compute_config(arch, config.ffn_ff1.math_fidelity, config.ffn_ff1.fp32_dest_acc)
    sdpa_compute = ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=config.ring_sdpa.math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=config.ring_sdpa.fp32_dest_acc,
    )

    sdpa_input_dtype = ttnn.bfloat8_b if LTX_QUANT_SDPA_BF8 else config.ring_sdpa.input_dtype

    def _quant_self_attn(attn):
        if attn is None:
            return
        _apply_linear_quant(attn.to_qkv, config.self_attn_qkv)
        _apply_linear_quant(attn.to_out, config.self_attn_out)
        attn.mm_compute_kernel_config = qkv_compute
        attn.sdpa_compute_kernel_config = sdpa_compute
        # Assign unconditionally (incl. None) so re-applying a preset that requests no SDPA cast
        # clears a prior preset's cast instead of leaving it active.
        attn._sdpa_input_dtype = sdpa_input_dtype

    def _quant_cross_attn(attn):
        if attn is None:
            return
        _apply_linear_quant(attn.to_q, config.cross_attn_q)
        _apply_linear_quant(attn.to_kv, config.cross_attn_kv)
        _apply_linear_quant(attn.to_out, config.cross_attn_out)
        attn.mm_compute_kernel_config = cross_compute

    _quant_self_attn(block.attn1)
    _quant_cross_attn(block.attn2)
    _apply_linear_quant(block.ffn.ff1, config.ffn_ff1)
    _apply_linear_quant(block.ffn.ff2, config.ffn_ff2)
    block.ff_compute_kernel_config = ffn_compute

    if has_audio:
        _quant_self_attn(getattr(block, "audio_attn1", None))
        _quant_cross_attn(getattr(block, "audio_attn2", None))
        # A2V / V2A cross-attn use a separate (non-fused) addcmul on their outputs, so their
        # to_out could be bf8; keep them on cross_attn_out (bf16 carve-out) conservatively —
        # audio is already optimized and these are small relative to the video GEMMs.
        _quant_cross_attn(getattr(block, "audio_to_video_attn", None))
        _quant_cross_attn(getattr(block, "video_to_audio_attn", None))
        audio_ff = getattr(block, "audio_ff", None)
        if audio_ff is not None:
            _apply_linear_quant(audio_ff.ff1, config.ffn_ff1)
            _apply_linear_quant(audio_ff.ff2, config.ffn_ff2)


def apply_quant_config(model, config: QuantConfig) -> None:
    """Apply quantization config to an LTXTransformerModel.

    Safe whether or not weights are loaded: weight typecasts no-op when ``_data is None``,
    and the compute-config attributes are always (re)set. With ``dynamic_load=True`` the
    pipeline evicts/reloads transformer weights, so this must run again after each reload
    (see ``set_quant_config``).
    """
    arch = model.mesh_device.arch()
    blocks = model.transformer_blocks
    logger.info(f"Applying LTX quant config to {len(blocks)} transformer blocks (has_audio={model.has_audio})")

    for block in blocks:
        apply_quant_config_to_block(block, config, arch, model.has_audio)

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
    act = config.self_attn_qkv.activation_dtype if LTX_QUANT_ACTIVATIONS else None
    logger.info(
        f"  LTX_QUANT_ACTIVATIONS={LTX_QUANT_ACTIVATIONS} -> linear activation cast: {act} "
        f"(None = collectives move bf16)"
    )
    sdpa_in = ttnn.bfloat8_b if LTX_QUANT_SDPA_BF8 else config.ring_sdpa.input_dtype
    logger.info(f"  Ring SDPA: input_dtype={sdpa_in}, fidelity={config.ring_sdpa.math_fidelity}")


def set_quant_config(pipeline, config: QuantConfig) -> None:
    """Install a quantization config on an LTX pipeline.

    Applies compute configs now (they survive eviction — module attributes, not Parameter
    data) and installs a post-load hook so the weight typecast re-runs after every reload
    (required under ``dynamic_load=True``, which evicts/reloads transformer weights). The
    hook runs inside ``load_model`` BEFORE the cache write, so cached tensorbins hold the
    quantized dtype and a reload's cache hit matches the module's expected dtype.
    """
    for state in pipeline.transformer_states:
        apply_quant_config(state.model, config)

    pipeline._transformer_post_load_hook = lambda model: apply_quant_config(model, config)
