# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Quantization and compute precision configuration for the Wan pipeline.

Usage:
    from models.tt_dit.pipelines.wan.quant_config import QuantConfig, apply_quant_config

    config = QuantConfig.all_weights_bf8()
    apply_quant_config(model, config)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

from loguru import logger

import ttnn

# ---------------------------------------------------------------------------
# Config dataclasses
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

    # Sprint 7 (A1). `activation_dtype` on the LinearQuantConfigs was dead until 2026-10-07:
    # `_apply_linear_config` never assigned it, so every preset still gathered bf16 activations
    # over the TP ring. It is wired for real only when this flag is set, so the measured presets
    # (`all_bf8_lofi` ...) keep their exact behaviour; `all_bf8_lofi_act8*` opt in.
    apply_activation_dtype: bool = False
    # Output dtype the block's norm1 / norm2 / norm3 emit (None = the input dtype, bf16). With
    # bfloat8_b the AGMM gathers of qkv, cross to_q and ff1 move half the bytes with no extra
    # program (the fused norms take `dtype=`); their outputs are pinned back to bf16 because the
    # fused norms require a bf16 input.
    norm_output_dtype: ttnn.DataType | None = None
    # Sprint 7 (A3). Dtype of the AdaLN shift and 1+scale fed to norm1 / norm3 / norm_out (None =
    # the fp32 reference). bf16 halves the fused norms' affine operand and drops the two `1 +
    # scale` programs per block (the +1 is folded into fp32 table rows once per load). A rounding
    # change: presets only, PCC- and fidelity-gated.
    modulation_affine_dtype: ttnn.DataType | None = None
    # Sprint 7 (A4, plumbing fix). Three per-linear compute configs were silently ignored:
    # attn1.to_out ran the config derived from self_attn_qkv, attn2.to_kv / to_out the one from
    # cross_attn_q, ff2 the one from ffn_ff1 (only their weight dtypes took effect, so e.g. the
    # sprint-6 "cross_attn_out at HiFi2" pin only changed the weight dtype). True makes every
    # linear run its own LinearQuantConfig. Off for the measured presets so their numbers stay
    # reproducible; on in the act8 family.
    per_linear_compute_config: bool = False
    # Sprint 7 (A6). Output dtype of ff1 (ff2's input). bfloat8_b halves the 16.7 MB per-block
    # intermediate ff1 writes and ff2 reads (the largest DRAM stream in the block) and lets ff2's
    # MMRS read block-float pages; ff2's output is pinned back to bf16. None = default.
    ff1_output_dtype: ttnn.DataType | None = None
    # Sprint 7 (A7). Output dtype of norm_out (proj_out input + its fp32 TP all-gather); None = fp32.
    norm_out_output_dtype: ttnn.DataType | None = None

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
        """All linear weights bfloat8_b, rest default.

        self_attn_out keeps bf16 weights, as in `all_bf8_lofi`: the fused matmul+addcmul
        kernels (`all_gather_minimal_matmul_async` with the addcmul ternary, and
        `dit_minimal_matmul_addcmul_fused`) require the residual and gate tiles to match the
        weight tile format, and those are bf16 activations. With bf8 weights there the kernel
        asserts `ternary_a_tile_size == in1_tile_size` (seen on the TI2V-5B, 2026-09-22).
        """
        lc = LinearQuantConfig(weight_dtype=ttnn.bfloat8_b)
        return QuantConfig(
            self_attn_qkv=lc,
            self_attn_out=LinearQuantConfig(),
            cross_attn_q=lc,
            cross_attn_kv=lc,
            # bf16 weights for the same reason as self_attn_out: since sprint 6 the
            # cross-attention residual is fused into attn2.to_out through the ternary kernel.
            cross_attn_out=LinearQuantConfig(),
            ffn_ff1=lc,
            ffn_ff2=lc,
            ring_sdpa=SDPAQuantConfig(),
        )

    @staticmethod
    def bf8_weights_sdpa_bf8() -> QuantConfig:
        """`all_weights_bf8` plus bfloat8_b inputs to the ring SDPA (HiFi2, no fp32 accumulate).

        The middle step between `all_weights_bf8` and `all_bf8_lofi`: halves the K/V bytes the
        sequence-parallel ring moves per step (~120 GB/device/generation at 720p in bf16) while
        keeping every matmul at HiFi2. Added 2026-09-25 for the TI2V-5B.
        """
        base = QuantConfig.all_weights_bf8()
        base.ring_sdpa = SDPAQuantConfig(
            input_dtype=ttnn.bfloat8_b, math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc=False
        )
        return base

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
        """All weights + activations bfloat8_b, LoFi compute, SDPA bf8 HiFi2.

        self_attn_out keeps bf16 weights because the fused matmul+addcmul kernel
        (dit_minimal_matmul_addcmul_fused) requires ternary inputs (residual,
        gate) to match the weight tile format, and those are bf16 activations.
        """
        lc = LinearQuantConfig(
            weight_dtype=ttnn.bfloat8_b,
            activation_dtype=ttnn.bfloat8_b,
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc=False,
        )
        # self_attn_out: same compute config but bf16 weights (fused addcmul constraint)
        lc_out = LinearQuantConfig(
            weight_dtype=ttnn.bfloat16,
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
            self_attn_out=lc_out,
            cross_attn_q=lc,
            cross_attn_kv=lc,
            # Since sprint 6 the cross-attention residual is fused into attn2.to_out through the
            # same ternary kernel, so it carries the same bf16-weight constraint. It also runs
            # HiFi2 with fp32 accumulate: with the residual add inside a LoFi epilogue the
            # transformer PCC fell from 99.9651 % to 99.886 % (2026-09-28), and the N=768
            # projections are bound by the TP-ring bytes, not by math rate (notes 7.11), so the
            # higher fidelity is free here.
            cross_attn_out=LinearQuantConfig(weight_dtype=ttnn.bfloat16, activation_dtype=ttnn.bfloat8_b),
            ffn_ff1=lc,
            ffn_ff2=lc,
            ring_sdpa=sc,
        )


def _all_bf8_lofi_sdpa_lofi() -> QuantConfig:
    """`all_bf8_lofi` with the ring SDPA itself at LoFi (bf8 inputs, no fp32 accumulate).

    The last precision step: the self-attention SDPA is ~28 % of the TI2V-5B denoise and is the
    only compute `all_bf8_lofi` leaves at HiFi2. Measured option, not a default candidate
    (added 2026-09-25); gate it on PCC, CLIP and a visual check before use.
    """
    base = QuantConfig.all_bf8_lofi()
    base.ring_sdpa = SDPAQuantConfig(
        input_dtype=ttnn.bfloat8_b, math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc=False
    )
    return base


QuantConfig.all_bf8_lofi_sdpa_lofi = staticmethod(_all_bf8_lofi_sdpa_lofi)


def _all_bf8_lofi_act8() -> QuantConfig:
    """`all_bf8_lofi` with bfloat8_b activations actually fed to the TP all-gathers (sprint 7, A1).

    norm1 / norm2 / norm3 emit bfloat8_b, so the qkv, cross-attention to_q and ff1 AGMMs gather
    half the bytes of the bf16 preset with no extra program; every AGMM output is pinned back to
    bf16 (the fused norms and the residual stream require bf16). The two to_out projections keep
    their bf16 input (the SDPA output): casting it costs a program per call, so that variant is
    `all_bf8_lofi_act8_out8`, to be measured against this one. The three N=768 projections are
    fabric-bound (~10.8 MB per call at ~41 GB/s, notes 7.11), so the bytes are the lever:
    expected up to ~23 ms/step at 720p across them plus a share on qkv / ff1. PCC-gated; the
    bit-exact gates do not exercise it (opt-in preset).
    """
    base = QuantConfig.all_bf8_lofi()
    base.apply_activation_dtype = True
    base.norm_output_dtype = ttnn.bfloat8_b
    # A4 plumbing fix: with this the preset's cross_attn_out really runs HiFi2 + fp32 accumulate
    # (the sprint-6 pin only reached its weight dtype) and ff2 / to_kv run their own configs
    # (identical to the inherited ones in this preset).
    base.per_linear_compute_config = True
    # The SDPA outputs feeding the two to_out projections stay bf16 here (no cast program);
    # the hoisted prompt K/V projection keeps bf16 in and out (norm_k needs bf16).
    base.self_attn_out = replace(base.self_attn_out, activation_dtype=None)
    base.cross_attn_out = replace(base.cross_attn_out, activation_dtype=None)
    base.cross_attn_kv = replace(base.cross_attn_kv, activation_dtype=None)
    return base


def _all_bf8_lofi_act8_out8() -> QuantConfig:
    """`all_bf8_lofi_act8` plus a bfloat8_b cast of the SDPA outputs into the two to_out AGMMs.

    Adds one typecast program per to_out call (120 per step) to halve the bytes of the two
    remaining bf16 gathers (`attn1.to_out`, `attn2.to_out`, both fabric-bound). Measure against
    `all_bf8_lofi_act8`: the cast pays off only if the gather saving exceeds its launch cost.
    New LoFi combination (bf8 in0 with the bf16 to_out weight): run the 20 s PCC test under a
    watchdog first.
    """
    base = _all_bf8_lofi_act8()
    base.self_attn_out = replace(base.self_attn_out, activation_dtype=ttnn.bfloat8_b)
    base.cross_attn_out = replace(base.cross_attn_out, activation_dtype=ttnn.bfloat8_b)
    return base


def _all_bf8_lofi_act8_mod16() -> QuantConfig:
    """`all_bf8_lofi_act8` plus bf16 AdaLN shift / 1+scale into norm1, norm3 and norm_out (sprint 7, A3).

    The fp32 affine operand is the whole 84 vs 56 us gap between norm1/norm3 and the bf16-affine
    norm2 (same kernel, same phases); emitting the modulation as bf16 from its add also removes
    the two `1 + scale` programs per block (62 per step). Expected ~-4 ms/step T2V, ~-7 ms I2V
    (the per-token affine is fetched as full tiles per tile-row). Deviates from the diffusers
    fp32 modulation like Flux2 / MiniMax-H3 / SD3.5 do; PCC- and same-seed-fidelity gated.
    """
    base = _all_bf8_lofi_act8()
    base.modulation_affine_dtype = ttnn.bfloat16
    return base


def _all_bf8_lofi_act8_mod16_ff8() -> QuantConfig:
    """`all_bf8_lofi_act8_mod16` plus a bfloat8_b ff1 output into ff2 (sprint 7, A6).

    Halves the [M, inner/tp] intermediate between ff1 and ff2 (16.7 MB per block at 720p, the
    largest DRAM stream per block) and feeds the ff2 MMRS block-float pages; ff2's output is pinned
    bf16. New LoFi combination (bf8 in0 x bf8 ff2 weight): 20 s PCC test under a watchdog first.
    Expected 1-3 % of the step; PCC- and fidelity-gated.
    """
    base = _all_bf8_lofi_act8_mod16()
    base.ff1_output_dtype = ttnn.bfloat8_b
    # A7: norm_out emits bf16 (halves the fp32 all-gather into proj_out; the diffusers reference
    # casts to bf16 before proj_out as well).
    base.norm_out_output_dtype = ttnn.bfloat16
    return base


QuantConfig.all_bf8_lofi_act8 = staticmethod(_all_bf8_lofi_act8)
QuantConfig.all_bf8_lofi_act8_mod16_ff8 = staticmethod(_all_bf8_lofi_act8_mod16_ff8)
QuantConfig.all_bf8_lofi_act8_out8 = staticmethod(_all_bf8_lofi_act8_out8)
QuantConfig.all_bf8_lofi_act8_mod16 = staticmethod(_all_bf8_lofi_act8_mod16)


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


def _apply_linear_config(
    linear, lc: LinearQuantConfig, arch, name: str, *, apply_activation_dtype: bool = False
) -> None:
    """Apply quantization + compute config to a single linear layer."""
    if lc.weight_dtype != ttnn.bfloat16 and linear.weight._data is not None:
        _quantize_weight(linear.weight, lc.weight_dtype)
        if hasattr(linear, "bias") and linear.bias is not None:
            _quantize_weight(linear.bias, lc.weight_dtype)
        logger.debug(f"  {name}: weights -> {lc.weight_dtype}")

    linear.compute_config = _make_linear_compute_config(arch, lc)

    if apply_activation_dtype and hasattr(linear, "activation_dtype"):
        # ColParallelLinear: cast the activation before its TP all-gather (a no-op when the
        # producer already emitted that dtype) and pin the matmul output back to bf16 so the
        # residual stream and the fused norms (bf16 input only) never see a block-float tensor.
        linear.activation_dtype = lc.activation_dtype
        linear.pin_output_bf16 = lc.activation_dtype is not None
        logger.debug(f"  {name}: activation_dtype -> {lc.activation_dtype}, pin_output_bf16={linear.pin_output_bf16}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_quant_config(model, config: QuantConfig) -> None:
    """Apply quantization config to a WanTransformer3DModel.

    Safe to call whether or not weights are currently loaded. When weights are
    loaded (``_data is not None``), typecasts them to the configured dtype.
    Compute configs are always applied (they are module attributes, not weight
    data).

    With ``dynamic_load=True`` the pipeline evicts transformer weights between
    uses.  In that case, call this function again after each reload so the
    freshly-loaded weights get quantized.  See ``set_quant_config`` for a
    convenience wrapper that patches ``_prepare_transformer`` automatically.

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

    act = {"apply_activation_dtype": config.apply_activation_dtype}
    for idx, block in enumerate(model.blocks):
        # Norm output dtype (A1): what norm1 / norm2 / norm3 hand to the qkv, cross to_q and ff1 gathers.
        block.norm_output_dtype = config.norm_output_dtype
        # AdaLN affine dtype (A3): shift / 1+scale emitted as bf16 with the +1 pre-folded.
        block.modulation_affine_dtype = config.modulation_affine_dtype
        # Per-linear compute configs (A4): honour self_attn_out / cross_attn_kv / cross_attn_out / ffn_ff2.
        block.per_linear_compute_config = config.per_linear_compute_config
        block.attn1.per_linear_compute_config = config.per_linear_compute_config
        block.attn2.per_linear_compute_config = config.per_linear_compute_config
        # ff1 output dtype (A6).
        block.ffn.ff1_output_dtype = config.ff1_output_dtype

        # Self-attention linears
        _apply_linear_config(block.attn1.to_qkv, config.self_attn_qkv, arch, f"blocks[{idx}].attn1.to_qkv", **act)
        _apply_linear_config(block.attn1.to_out, config.self_attn_out, arch, f"blocks[{idx}].attn1.to_out", **act)

        # Self-attention mm_compute_kernel_config (used by matmuls in attention forward)
        block.attn1.mm_compute_kernel_config = _make_linear_compute_config(arch, config.self_attn_qkv)

        # Cross-attention linears
        _apply_linear_config(block.attn2.to_q, config.cross_attn_q, arch, f"blocks[{idx}].attn2.to_q", **act)
        _apply_linear_config(block.attn2.to_kv, config.cross_attn_kv, arch, f"blocks[{idx}].attn2.to_kv", **act)
        _apply_linear_config(block.attn2.to_out, config.cross_attn_out, arch, f"blocks[{idx}].attn2.to_out", **act)

        # Cross-attention mm_compute_kernel_config
        block.attn2.mm_compute_kernel_config = _make_linear_compute_config(arch, config.cross_attn_q)

        # FFN linears
        _apply_linear_config(block.ffn.ff1, config.ffn_ff1, arch, f"blocks[{idx}].ffn.ff1", **act)
        _apply_linear_config(block.ffn.ff2, config.ffn_ff2, arch, f"blocks[{idx}].ffn.ff2", **act)

        # FFN block-level compute config
        block.ff_compute_kernel_config = _make_linear_compute_config(arch, config.ffn_ff1)

        # Ring SDPA config (self-attention only)
        block.attn1.sdpa_compute_kernel_config = _make_sdpa_compute_config(arch, config.ring_sdpa)
        if config.ring_sdpa.input_dtype is not None:
            block.attn1._sdpa_input_dtype = config.ring_sdpa.input_dtype
            # Materialise the typed empty joint input now, outside any trace capture.
            block.attn1.dummy_joint_as(config.ring_sdpa.input_dtype)

    model.norm_out_affine_dtype = config.modulation_affine_dtype
    model.norm_out_output_dtype = config.norm_out_output_dtype or ttnn.float32

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
        f"  Activations: apply_activation_dtype={config.apply_activation_dtype}, "
        f"norm_output_dtype={config.norm_output_dtype}, modulation_affine_dtype={config.modulation_affine_dtype}, "
        f"per_linear_compute_config={config.per_linear_compute_config}, ff1_output_dtype={config.ff1_output_dtype}, "
        f"norm_out_output_dtype={config.norm_out_output_dtype}"
    )
    logger.info(
        f"  Ring SDPA: input_dtype={config.ring_sdpa.input_dtype}, "
        f"fidelity={config.ring_sdpa.math_fidelity}, fp32_acc={config.ring_sdpa.fp32_dest_acc}"
    )


def set_quant_config(pipeline, config: QuantConfig) -> None:
    """Install a quantization config on a WanPipeline.

    Patches ``pipeline._prepare_transformer`` so that ``apply_quant_config``
    runs automatically after every weight load (including dynamic reloads).
    This is necessary because ``dynamic_load=True`` evicts and reloads
    transformer weights from cache, which would overwrite any earlier
    typecast.

    Also applies compute configs immediately (safe even if weights are
    currently evicted).
    """
    # Apply compute configs now (these survive eviction since they are
    # module attributes, not Parameter data).
    for state in pipeline.transformer_states:
        apply_quant_config(state.model, config)

    # Patch _prepare_transformer to re-apply weight typecast after each reload.
    # Stash the original so repeated calls don't stack wrappers.
    if not hasattr(pipeline, "_orig_prepare_transformer"):
        pipeline._orig_prepare_transformer = pipeline._prepare_transformer
    original_prepare = pipeline._orig_prepare_transformer

    def _prepare_with_quant(idx: int):
        was_loaded = pipeline.transformer_states[idx].model.is_loaded()
        original_prepare(idx)
        if not was_loaded:
            # Weights were just loaded from cache — typecast them.
            apply_quant_config(pipeline.transformer_states[idx].model, config)

    pipeline._prepare_transformer = _prepare_with_quant


# ---------------------------------------------------------------------------
# Environment opt-in (test / experiment plumbing; never changes a default)
# ---------------------------------------------------------------------------

QUANT_CONFIG_ENV = "WAN5B_QUANT_CONFIG"
"""Name of a `QuantConfig` preset (`all_weights_bf8`, `all_lofi`, `all_bf8_lofi`) to opt into."""


def quant_config_from_env(var: str = QUANT_CONFIG_ENV) -> tuple[str, QuantConfig] | None:
    """Return `(name, QuantConfig)` for the preset named in `var`, or None when unset.

    `default` is accepted and means "apply the explicit bf16/HiFi2 config", which is a no-op on
    the weights but replaces the layers' compute configs with the preset's, so it is a useful
    control when comparing presets.
    """
    name = os.environ.get(var, "").strip()
    if not name:
        return None
    factory = getattr(QuantConfig, name, None) if not name.startswith("_") else None
    if factory is None or not callable(factory):
        presets = [
            "default",
            "all_weights_bf8",
            "bf8_weights_sdpa_bf8",
            "all_lofi",
            "all_bf8_lofi",
            "all_bf8_lofi_sdpa_lofi",
            "all_bf8_lofi_act8",
            "all_bf8_lofi_act8_out8",
            "all_bf8_lofi_act8_mod16",
            "all_bf8_lofi_act8_mod16_ff8",
        ]
        msg = f"{var}={name!r} is not a QuantConfig preset; expected one of {presets}"
        raise ValueError(msg)
    return name, factory()


def set_quant_config_from_env(pipeline, *, rewarm: bool = True, var: str = QUANT_CONFIG_ENV) -> str | None:
    """Apply the preset named in `var` to `pipeline` (see `set_quant_config`); return its name.

    Meant to be called right after the pipeline is constructed and before any trace is captured.
    With `rewarm=True` the pipeline's eager warmup runs again afterwards: the construction-time
    warmup compiled programs for bf16 weights and the baseline compute configs, and a trace
    capture cannot compile the new binaries a different dtype or fidelity needs.
    """
    found = quant_config_from_env(var)
    if found is None:
        return None
    name, config = found
    logger.info(f"{var}={name}: applying QuantConfig.{name}() to the pipeline")
    set_quant_config(pipeline, config)
    if rewarm:
        logger.info("re-running the eager warmup so the quantized programs are compiled before trace capture")
        pipeline._warmup()
    return name


# ---------------------------------------------------------------------------
# Trace-execution mode opt-in (same spirit: test plumbing, default unchanged)
# ---------------------------------------------------------------------------

TRACE_MODE_ENV = "WAN5B_TRACE_MODE"
TRACE_MODES = ("blocking", "nonblocking", "2cq")
DEFAULT_TRACE_MODE = "2cq"
"""Production default for the 5B tests and demo since 2026-09-22: non-blocking trace with the
per-step uploads on a second command queue. Measured on the 4x8 BH Galaxy, 720p T2V, mean of 3
vs mean of 3 the same hour: denoise 10.747 -> 10.434 s (-2.9 %), total 11.83 -> 11.50 s, with
bit-identical latents and an unchanged 12x10 compute grid. `blocking` is the previous path."""


def trace_mode_from_env(var: str = TRACE_MODE_ENV) -> str:
    """`2cq` (default: non-blocking + input uploads on queue 1), `nonblocking`, or `blocking`."""
    mode = os.environ.get(var, DEFAULT_TRACE_MODE).strip().lower() or DEFAULT_TRACE_MODE
    if mode not in TRACE_MODES:
        msg = f"{var}={mode!r}; expected one of {TRACE_MODES}"
        raise ValueError(msg)
    return mode


def device_params_for_trace_mode(base: dict, var: str = TRACE_MODE_ENV) -> dict:
    """Add `num_command_queues=2` to `base` when the env selects the two-queue mode."""
    if trace_mode_from_env(var) == "2cq":
        return {**base, "num_command_queues": 2}
    return dict(base)


def configure_trace_mode_from_env(pipeline, var: str = TRACE_MODE_ENV) -> str:
    """Apply the env-selected mode through `pipeline.configure_trace_execution`; return it."""
    mode = trace_mode_from_env(var)
    if mode == "nonblocking":
        pipeline.configure_trace_execution(blocking=False)
    elif mode == "2cq":
        pipeline.configure_trace_execution(blocking=False, input_cq_id=1)
    return mode
