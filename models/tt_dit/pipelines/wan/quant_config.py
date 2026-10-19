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

from ...utils.matmul import FusedMMRSConfig, register_fused_mmrs_configs, register_matmul_configs

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
    # (q_chunk, k_chunk), e.g. larger K chunks that fit L1 with bfloat8_b K / V; None = the model's chunk sizes
    chunk_size: tuple[int, int] | None = None


@dataclass
class QuantConfig:
    """Per-component quantization and compute precision configuration.

    Each transformer block has 7 linear layers and 1 ring SDPA (self-attention).
    Cross-attention SDPA stays at default settings, except for its input dtype.
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

    # Cross-attention SDPA Q / K / V dtype (the K / V are cast once, when computed); None = no cast
    cross_sdpa_input_dtype: ttnn.DataType | None = None

    # dtype the norms write the inputs of to_qkv, cross to_q and ff1 in, and the attention outputs are cast to before
    # concat_heads (the inputs of to_out): the all-gather-matmuls then gather it. Their outputs stay bf16. None = no
    # cast. With LoFi projections, bfloat8_b here also keeps CFG batching on (see apply_quant_config).
    projection_input_dtype: ttnn.DataType | None = None

    # dtype ff1 writes the FFN intermediate (ff2's input) in; ff2's output stays bf16. None = ff1's own.
    ffn_intermediate_dtype: ttnn.DataType | None = None

    # Blockings swept with this config for matmuls that accumulate in bf16 DEST, in the format of
    # register_matmul_configs and register_fused_mmrs_configs; apply_quant_config registers them with
    # fp32_dest_acc=False, so no fp32-DEST matmul uses them. Once registered, they stay for the process.
    bf16_dest_matmul_configs: dict = field(default_factory=dict)
    bf16_dest_fused_mmrs_configs: dict = field(default_factory=dict)

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
            cross_attn_out=lc,
            ffn_ff1=lc,
            ffn_ff2=lc,
            ring_sdpa=sc,
        )

    @staticmethod
    def wh_galaxy_bf8_lofi() -> QuantConfig:
        """bfloat8_b weights and activations with LoFi matmuls, as tuned on WH Galaxy 4x8 (CFG batching).

        - bf8 weights, except the two output projections: the fused addcmul of the self-attention one needs bf16
          weights, and the cross-attention one keeps bf16 weights as measured and quality-checked (bf8 untested).
        - The norms write the projection inputs in bf8, and ff1 writes the FFN intermediate in bf8; the projection
          outputs and the residual stream stay bf16.
        - LoFi matmuls accumulating in bf16 DEST, with blockings swept for it at 480p (8-tile subblocks) and reused at
          720p, where they beat the generic default.
        - Ring SDPA: bf8 Q / K / V and (128, 1024) chunks, which fit L1 with bf8 K / V; its matmuls stay HiFi2
          (LoFi there degrades I2V). Cross-attention SDPA: bf8 Q / K / V.
        """
        lc = LinearQuantConfig(
            weight_dtype=ttnn.bfloat8_b,
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc=False,
        )
        lc_out = LinearQuantConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc=False)
        return QuantConfig(
            self_attn_qkv=lc,
            self_attn_out=lc_out,
            cross_attn_q=lc,
            cross_attn_kv=lc,
            cross_attn_out=lc_out,
            ffn_ff1=lc,
            ffn_ff2=lc,
            ring_sdpa=SDPAQuantConfig(
                input_dtype=ttnn.bfloat8_b,
                math_fidelity=ttnn.MathFidelity.HiFi2,
                fp32_dest_acc=False,
                chunk_size=(128, 1024),
            ),
            cross_sdpa_input_dtype=ttnn.bfloat8_b,
            projection_input_dtype=ttnn.bfloat8_b,
            ffn_intermediate_dtype=ttnn.bfloat8_b,
            bf16_dest_matmul_configs={
                # CFG-batched rows per device (SP=8, TP=4): 8192 at 480p, 18944 at 720p; 8x8 all-gather-matmul grid
                "8x8": {
                    (8192, 5120, 3840): (16, 10, 10, (4, 2)),  # to_qkv
                    (8192, 5120, 3456): (16, 8, 14, (4, 2)),  # ff1 + GELU
                    (8192, 5120, 1280): (16, 8, 5, (8, 1)),  # to_out + addcmul; cross to_q, to_out
                    (18944, 5120, 3840): (16, 10, 10, (4, 2)),
                    (18944, 5120, 3456): (16, 8, 14, (4, 2)),
                    (18944, 5120, 1280): (16, 8, 5, (8, 1)),
                }
            },
            bf16_dest_fused_mmrs_configs={
                ttnn.CoreCoord(8, 9): {
                    # ff2 + addcmul, matmul on 8x7: 4129.9 us at 480p
                    (8192, 3456, 5120): FusedMMRSConfig(ttnn.CoreCoord(8, 7), 8, 3, 10, 4, 2, None, 1),
                    (18944, 3456, 5120): FusedMMRSConfig(ttnn.CoreCoord(8, 7), 8, 3, 10, 4, 2, None, 1),
                }
            },
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
    if lc.weight_dtype != ttnn.bfloat16 and linear.weight._data is not None:
        _quantize_weight(linear.weight, lc.weight_dtype)
        if hasattr(linear, "bias") and linear.bias is not None:
            _quantize_weight(linear.bias, lc.weight_dtype)
        logger.debug(f"  {name}: weights -> {lc.weight_dtype}")

    linear.compute_config = _make_linear_compute_config(arch, lc)


def _needs_separate_cfg_passes(config: QuantConfig) -> bool:
    linears = (
        config.self_attn_qkv,
        config.self_attn_out,
        config.cross_attn_q,
        config.cross_attn_kv,
        config.cross_attn_out,
        config.ffn_ff1,
        config.ffn_ff2,
    )
    # only bf8 projection inputs are known to run CFG-batched LoFi projections without a hang
    bf8_inputs = config.projection_input_dtype == ttnn.bfloat8_b
    return not bf8_inputs and any(lc.math_fidelity == ttnn.MathFidelity.LoFi for lc in linears)


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
    - Ring SDPA: block.attn1.sdpa_compute_kernel_config, block.attn1.set_ring_sdpa_config
    - Cross-attn SDPA input dtype: block.attn2
    - Projection input dtype: the block's norms, the attention outputs, pin_output_bf16 on those projections
    - FFN intermediate dtype: block.ffn.intermediate_dtype
    """
    arch = model.mesh_device.arch()
    n_blocks = len(model.blocks)

    logger.info(f"Applying quant config to {n_blocks} transformer blocks")

    # Workaround: with CFG batching (8192 rows per device at 480p on WH Galaxy 4x8), LoFi projections on bf16
    # activations hang in the bias add of a projection all-gather-matmul (one core's unpacker and math wait on each
    # other); bf8 projection inputs do not. Such configs run the CFG passes as two forwards.
    model.cfg_batch = model.cfg_batch_requested and not _needs_separate_cfg_passes(config)
    if model.cfg_batch_requested and not model.cfg_batch:
        logger.warning("LoFi projections on bf16 activations: running the CFG passes as two forwards")

    if config.bf16_dest_matmul_configs:
        register_matmul_configs(config.bf16_dest_matmul_configs, fp32_dest_acc=False)
    if config.bf16_dest_fused_mmrs_configs:
        register_fused_mmrs_configs(config.bf16_dest_fused_mmrs_configs, fp32_dest_acc=False)

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
        if config.ring_sdpa.input_dtype is not None:
            block.attn1._sdpa_input_dtype = config.ring_sdpa.input_dtype
        if config.ring_sdpa.chunk_size is not None:
            block.attn1.set_ring_sdpa_config(chunk_size=config.ring_sdpa.chunk_size)

        # Cross-attention SDPA input dtype
        if config.cross_sdpa_input_dtype is not None:
            block.attn2._sdpa_input_dtype = config.cross_sdpa_input_dtype

        # Inputs of the all-gathered projections, and the FFN intermediate
        if config.projection_input_dtype is not None:
            block.projection_input_dtype = config.projection_input_dtype
            block.attn1.projection_input_dtype = config.projection_input_dtype
            block.attn2.projection_input_dtype = config.projection_input_dtype
            for linear in (block.attn1.to_qkv, block.attn1.to_out, block.attn2.to_q, block.attn2.to_out, block.ffn.ff1):
                linear.pin_output_bf16 = True
        if config.ffn_intermediate_dtype is not None:
            block.ffn.intermediate_dtype = config.ffn_intermediate_dtype

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
    batched = [state.model.cfg_batch for state in pipeline.transformer_states]
    for state in pipeline.transformer_states:
        apply_quant_config(state.model, config)
    if any(was and not state.model.cfg_batch for was, state in zip(batched, pipeline.transformer_states)):
        # The pipeline's allocation run sized the CCL buffers for the batched rows; free them so that those of the
        # two forwards fit. Captured traces hold the batched graph and these buffers, so release them first.
        pipeline.release_traces()
        pipeline.dit_ccl_manager.free_ping_pong_buffers()

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
