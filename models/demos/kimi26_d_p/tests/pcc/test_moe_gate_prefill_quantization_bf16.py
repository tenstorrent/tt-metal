# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Weight-quantization experiment for the BFLOAT16 MoE gate prefill.

The BF16 analogue of ``test_moe_gate_prefill_quantization.py``: everything on
device runs in ``ttnn.bfloat16`` instead of ``ttnn.bfloat4_b``. BF16 is IEEE-
style (1 sign + 8 exponent + 7 mantissa bits), so the quantization step is
element-wise rather than block-shared-exponent; there is no "16 elements
share one exponent" structure to exploit.

Because the device already emits BF16 logits, the typecast that the BFP4
test inserts right before ``deepseek_grouped_gate`` is not needed here —
``tt_model`` is constructed with its native BF16 weight path and the
grouped-gate op consumes logits directly.

Applicable variants (the BFP4-only variants — ``sim_bfp4``, ``dither``,
``rank_exp``, ``dither_rank_exp``, ``soft_rank_*``, ``perm_sim_bfp4`` — are
all block-float preprocessing tricks and are dropped from this file):

    ``baseline``  — ttnn packs fp32 weight directly at upload (rounds to
                    bf16).
    ``perm``      — intra-group expert permutation by descending L2 norm.
                    Dtype-agnostic; we keep it as a control to confirm the
                    permutation itself has no effect on bf16 (where there
                    are no shared exponents to align).

Bias-correction suffixes are kept, now computed against the bf16 round-trip
of the weight (instead of the bfp4 round-trip):

    ``*_bc``   — Tier 0: ``Δbias[e] = − s'(0) · X_BAR_GLOBAL · Σ_i ΔW[e, i]``
                 with ``ΔW = bf16(W) − W``. BF16 rounding error is ~4 bits
                 smaller per element than bfp4, so this is an even more
                 extreme null test than in the bfp4 file.
    ``*_bcpc`` — Tier 1: ``Δbias[e] = − s'(0) · (ΔW[e, :] · x_bar_per_channel)``.
                 Requires the gate input cache (see
                 ``_try_load_x_bar_per_channel``); falls back to the Tier 0
                 formula with a constant ``x_bar`` vector when unavailable.

For each variant we log PCC / recall against the fp32 reference *and* the
relative reconstruction error ``||w_host − w_fp32|| / ||w_fp32||`` — here
that's exactly the bf16 rounding error of the weight, typically
``O(2^-8) ≈ 4e-3``.
"""

import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate as ReferenceMoEGate
from models.demos.kimi26_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    create_fabric_router_config,
    create_gate_weights,
    get_max_payload_size,
    get_sp_mesh_composer,
    load_gate_weights_from_hf,
)
from models.demos.kimi26_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, TtMoEGateConfig, TtMoEGatePrefill
from models.demos.kimi26_d_p.tt.moe.validation_helpers import (
    ValidationResult,
    calculate_average_recall,
    compare_pcc,
    compare_recall,
    validate_composed,
)
from models.demos.kimi26_d_p.tt.moe.visualization_helpers import log_validation_results
from models.demos.kimi26_d_p.utils.test_utils import adjust_shapes_for_testing, get_input_mem_config

_DEFAULT_HF_REPO = "deepseek-ai/DeepSeek-V3"
_LOCAL_FALLBACKS = (
    "models/demos/kimi26_d_p/reference",
    "/proj_sw/user_dev/moonshotai/Kimi-K2.5",
)

# Both activations and weights run in bf16 on the device; the on-host weight
# "preprocessing" for the bf16 path is just an optional expert permutation.
ACTIVATION_DTYPE = ttnn.bfloat16

# Global activation statistics at layer 3 (post-RMSNorm gate input), measured
# once offline on ``gate_input_layer3_seq100000.pt``:
#   mean = 0.0004,  std = 0.1147,  min ≈ -7.69,  max ≈ 8.19
# These four scalars are all the "input info" the "_bc" variants may use —
# no per-channel statistics, no forward pass on the cache. At runtime the
# bias correction is a fixed vector computed purely from the weight and its
# bf16 round-trip error, multiplied by ``X_BAR_GLOBAL``.
X_BAR_GLOBAL = 0.0004
X_STD_GLOBAL = 0.1147
# Sigmoid derivative at the working point ``logit = 0`` — linearization
# coefficient that maps a logit-space bias correction into score-space (the
# bias in ``deepseek_grouped_gate`` is added after sigmoid).
SIGMOID_PRIME_AT_ZERO = 0.25


# ---------------------------------------------------------------------------
# Cross-variant summary (prints once all parametrized tests have finished)
# ---------------------------------------------------------------------------
# Each test invocation appends its global PCC / recall here, and the
# session-scoped autouse fixture below prints a single comparison table at
# the very end of the pytest session. Running the whole file now gives you
# a side-by-side view of every variant at the bottom of the log — no need
# to scroll through per-variant summaries and cross-reference manually.

_RESULTS: list[dict] = []


@pytest.fixture(scope="session", autouse=True)
def _print_results_summary():
    yield
    if not _RESULTS:
        return
    logger.info("=" * 125)
    logger.info(f"Gate prefill quantization — cross-variant summary (dtype={ACTIVATION_DTYPE})")
    logger.info("-" * 125)
    header = (
        f"{'variant':>28} | {'mesh':>7} | {'logits_pcc':>12} | "
        f"{'scores_pcc':>12} | {'recall':>10} | {'w_rel_err':>12} | {'bf16_rt_err':>12}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for r in _RESULTS:
        logger.info(
            f"{r['variant']:>28} | {r['mesh']:>7} | {r['logits_pcc']:>12.6f} | "
            f"{r['scores_pcc']:>12.6f} | {r['recall']:>10.4f} | "
            f"{r['w_rel_err']:>12.6e} | {r['bf16_rt_err']:>12.6e}"
        )
    logger.info("=" * 125)


def _resolve_model_id() -> str:
    env_path = os.getenv("KIMI_K25_HF_MODEL")
    if env_path and (Path(env_path) / "model.safetensors.index.json").exists():
        return env_path
    for fallback in _LOCAL_FALLBACKS:
        if (Path(fallback) / "model.safetensors.index.json").exists():
            return fallback
    return _DEFAULT_HF_REPO


def _try_load_real_gate_weights(n_routed_experts: int, dim: int) -> dict | None:
    model_id = _resolve_model_id()
    try:
        gate_w = load_gate_weights_from_hf(model_id, layer_idx=3, dtype=torch.bfloat16)
        gate_w["weight"] = gate_w["weight"][:n_routed_experts, :dim]
        gate_w["e_score_correction_bias"] = gate_w["e_score_correction_bias"][:n_routed_experts]
        return gate_w
    except (FileNotFoundError, KeyError) as e:
        logger.warning(f"Could not load real gate weights ({model_id}): {e}. Using random weights.")
        return None


def _try_load_real_gate_input_fp32(max_seq_len: int, dim: int) -> torch.Tensor | None:
    """Load a saved gate input tensor and return it in fp32; None on failure."""
    gate_input_cache = os.environ.get("DEEPSEEK_V3_GATE_INPUT_CACHE")
    moe_dir = Path(gate_input_cache) if gate_input_cache else Path(__file__).parent.parent.parent / "tt" / "moe"

    for name in ("gate_input_layer3_seq100000.pt", "gate_input_layer3.pt"):
        path = moe_dir / name
        if path.exists():
            saved = torch.load(path, weights_only=True)
            real_input = saved["gate_input"].squeeze(0).to(torch.float32)
            if real_input.shape[0] >= max_seq_len:
                result = real_input[:max_seq_len, :dim]
            else:
                repeats = (max_seq_len + real_input.shape[0] - 1) // real_input.shape[0]
                result = real_input.repeat(repeats, 1)[:max_seq_len, :dim]
            logger.info(f"Loaded real gate input from {path} (raw {real_input.shape}, sliced to {result.shape})")
            return result

    return None


def _try_load_x_bar_per_channel(dim: int) -> torch.Tensor | None:
    """Load the gate input cache and return its per-channel mean (shape ``(dim,)``).

    This is the single pre-computed vector that Tier 1 bias correction (the
    ``_bcpc`` variants) relies on — a ``.mean(0)`` over the same file that
    ``_try_load_real_gate_input_fp32`` reads. No forward pass, no model
    calibration, just one summary statistic.
    """
    gate_input_cache = os.environ.get("DEEPSEEK_V3_GATE_INPUT_CACHE")
    moe_dir = Path(gate_input_cache) if gate_input_cache else Path(__file__).parent.parent.parent / "tt" / "moe"

    for name in ("gate_input_layer3_seq100000.pt", "gate_input_layer3.pt"):
        path = moe_dir / name
        if path.exists():
            saved = torch.load(path, weights_only=True)
            real_input = saved["gate_input"].squeeze(0).to(torch.float32)
            x_bar = real_input[:, :dim].mean(dim=0)
            logger.info(
                f"Computed per-channel x_bar from {path} (over {real_input.shape[0]} tokens): "
                f"mean={x_bar.mean().item():.4e}, std={x_bar.std().item():.4e}, "
                f"min={x_bar.min().item():.4e}, max={x_bar.max().item():.4e}, "
                f"|x_bar|_2={x_bar.norm().item():.4e}"
            )
            return x_bar

    return None


# ---------------------------------------------------------------------------
# Weight preprocessing variants (bf16 path)
# ---------------------------------------------------------------------------
#
# The bf16 path has no shared-exponent structure to align with, so the only
# non-trivial host-side weight transform we keep from the bfp4 file is the
# expert permutation. It's dtype-agnostic and serves as a control here: if
# ``perm`` and ``baseline`` agree on bf16, that confirms the permutation's
# observed effect in the bfp4 file is block-float-specific (shared-exponent
# re-grouping), not something introduced by the permutation mechanics.
#
# All variants operate on ``(n_routed_experts, dim)`` (HF layout) and return
# the same layout; the test transposes to device layout ``(dim, n_experts)``
# at upload.
#
# Bias-correction variants (``_bc`` / ``_bcpc`` suffix, orthogonal to weight
# variant): see the module docstring and the two ``_compute_bias_correction_*``
# functions for the math. The only change from the bfp4 file is that ``ΔW``
# here is the bf16 rounding error (``bf16(W) − W``), which is ~4 bits
# smaller per element than the bfp4 error.


def _compute_expert_permutation(weight_fp32: torch.Tensor, n_expert_groups: int) -> torch.Tensor:
    """Return ``perm`` of shape ``(n_routed_experts,)`` s.t. ``W[perm, :]``
    has experts sorted by descending L2 norm within each expert group.

    With ``W_perm = W[perm, :]`` handed to the device, device-expert index
    ``i`` corresponds to original-expert index ``perm[i]``; apply ``perm[]``
    to the topk indices returned by the device to recover original indices.
    """
    n_experts = weight_fp32.shape[0]
    assert (
        n_experts % n_expert_groups == 0
    ), f"n_routed_experts={n_experts} is not divisible by n_expert_groups={n_expert_groups}"
    experts_per_group = n_experts // n_expert_groups

    scores = weight_fp32.pow(2).sum(dim=-1).sqrt()  # (n_experts,) L2 norm per expert

    perm = torch.empty(n_experts, dtype=torch.long)
    for g in range(n_expert_groups):
        start = g * experts_per_group
        end = start + experts_per_group
        local = torch.argsort(scores[start:end], descending=True)
        perm[start:end] = start + local
    return perm


def _preprocess_weight(
    variant: str,
    weight_fp32: torch.Tensor,
    n_expert_groups: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Produce the fp32 weight to hand to ttnn, plus an optional expert
    permutation that was applied to it.

    Input / output layout is HF convention: ``(n_routed_experts, dim)``.

    Returns:
        ``(weight_for_upload, perm_or_None)``. When ``perm`` is not None,
        the caller must also permute the bias by ``bias[perm]`` and apply
        ``perm[]`` to the device's topk indices before comparison.
    """
    if variant == "baseline":
        return weight_fp32, None
    if variant == "perm":
        perm = _compute_expert_permutation(weight_fp32, n_expert_groups)
        return weight_fp32[perm, :].contiguous(), perm
    raise ValueError(f"unknown weight variant: {variant!r}")


def _relative_frobenius_error(approx: torch.Tensor, ref: torch.Tensor) -> float:
    return torch.norm(approx - ref).item() / (torch.norm(ref).item() + 1e-30)


def _bf16_round_trip(w_fp32: torch.Tensor) -> torch.Tensor:
    """Approximate the weight ttnn materializes on device for the bf16 path.

    ``ttnn.from_torch(..., dtype=ttnn.bfloat16)`` rounds fp32 → bf16
    element-wise (round-to-nearest-even on a 7-bit mantissa). Casting fp32
    → bfloat16 → fp32 reproduces that rounding exactly, so we use it as the
    host-side ground truth for ``W_device`` in the bias-correction math.
    """
    return w_fp32.to(torch.bfloat16).to(torch.float32)


# ---------------------------------------------------------------------------
# Weight-only bias correction (Option A: global scalar x_bar)
# ---------------------------------------------------------------------------
#
# Derivation (identical to the bfp4 file; reproduced here for readability).
# Logit error per token / expert:
#
#     err[t, e] = x[t] · ΔW[e, :]    with ΔW = W_device − W_fp32
#
# For the bf16 path ``W_device`` is the bf16 round-trip of ``weight_for_upload``
# (the bfp4 file used ``quantize_bfp`` here). The rest of the derivation is
# unchanged: assuming ``E[x_i] = x_bar_global`` uniformly across channels,
#
#     E_t[err[t, e]] = x_bar_global · Σ_i ΔW[e, i]
#
# The bias in DeepSeek's grouped-gate is added AFTER sigmoid, so the logit
# correction has to be linearized through sigmoid. At the working point
# ``logit = 0``, ``s'(0) = 0.25``, giving the score-space correction:
#
#     Δbias_score[e] = − s'(0) · x_bar_global · Σ_i ΔW[e, i]
#                    = − 0.25 · x_bar_global · Σ_i ΔW[e, i]
#
# Expected magnitude in bf16. The bf16 rounding error per element is at most
# ``|w| · 2^-8 ≈ 4e-3 · |w|`` (with weight magnitude ~0.1 that's O(4e-4) per
# element). The 7168-wide row sum is a near-zero-mean random walk, ~√dim
# larger than per-element, so ``Σ_i ΔW[e, i] ~ 3e-2``; times
# ``X_BAR_GLOBAL ≈ 4e-4`` and ``s'(0) = 0.25`` gives ``Δbias_score ~ 3e-6``
# per expert. That's well below bf16 resolution at bias magnitudes of
# 0.01–0.1 (≈4e-4), so the correction will round to exactly zero on cast.
# This is an even more extreme null test than the bfp4 version.


def _compute_bias_correction_global(
    ref_weight: torch.Tensor,
    weight_for_upload_fp32: torch.Tensor,
    x_bar_global: float = X_BAR_GLOBAL,
) -> torch.Tensor:
    """Per-expert score-space bias correction from ΔW + scalar activation mean.

    Args:
        ref_weight: fp32 weight the device is expected to represent, in the
            same expert ordering as ``weight_for_upload_fp32`` (i.e. already
            permuted for ``perm*`` variants). Shape ``(n_experts, dim)``.
        weight_for_upload_fp32: fp32 weight handed to ttnn, shape
            ``(n_experts, dim)``. Round-tripped through bf16 internally to
            approximate the weight the device will materialize.
        x_bar_global: scalar assumed ``E[x_i]`` for every channel.
    Returns:
        ``Δbias_score`` of shape ``(n_experts,)``. Add this to the shipped
        bias.
    """
    w_dev_bf16 = _bf16_round_trip(weight_for_upload_fp32)
    delta_w = w_dev_bf16 - ref_weight  # (n_experts, dim)
    row_sum = delta_w.sum(dim=-1)  # (n_experts,)
    return -SIGMOID_PRIME_AT_ZERO * x_bar_global * row_sum


# ---------------------------------------------------------------------------
# Weight-based bias correction with per-channel ``x_bar`` (Tier 1 / "_bcpc")
# ---------------------------------------------------------------------------
#
# Same derivation as Tier 0, but replaces the uniform-channel assumption
# ``E[x_i] = x_bar_global`` with an actual per-channel mean
# ``x_bar_per_channel[i] = mean_t x[t, i]`` computed once offline over the
# gate input cache (see ``_try_load_x_bar_per_channel``).
#
#     E_t[err[t, e]] = Σ_i x_bar_per_channel[i] · ΔW[e, i]
#                    = ΔW[e, :] · x_bar_per_channel        (dot product)
#
# Score-space correction, linearized through sigmoid at logit = 0:
#
#     Δbias_score[e] = − s'(0) · (ΔW[e, :] · x_bar_per_channel)
#
# Why this might matter for bf16 specifically. BF16 rounding is tiny, but
# structured: elements with larger magnitudes pick up the bigger absolute
# rounding errors, and if the per-channel ``x_bar`` has any systematic
# alignment with those rows, the dot product can dodge the cancellation
# that kills the global-scalar version.


def _compute_bias_correction_per_channel(
    ref_weight: torch.Tensor,
    weight_for_upload_fp32: torch.Tensor,
    x_bar_per_channel: torch.Tensor,
) -> torch.Tensor:
    """Per-expert score-space bias correction from ΔW + per-channel activation mean.

    Args:
        ref_weight: fp32 weight the device is expected to represent, in the
            same expert ordering as ``weight_for_upload_fp32``. Shape
            ``(n_experts, dim)``.
        weight_for_upload_fp32: fp32 weight handed to ttnn, shape
            ``(n_experts, dim)``. Round-tripped through bf16 internally to
            approximate the weight the device will materialize.
        x_bar_per_channel: per-channel activation mean, shape ``(dim,)``.
    Returns:
        ``Δbias_score`` of shape ``(n_experts,)``. Add this to the shipped
        bias.
    """
    w_dev_bf16 = _bf16_round_trip(weight_for_upload_fp32)
    delta_w = w_dev_bf16 - ref_weight  # (n_experts, dim)
    delta_logit_per_expert = delta_w @ x_bar_per_channel.to(delta_w.dtype)
    return -SIGMOID_PRIME_AT_ZERO * delta_logit_per_expert


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weight_variant",
    [
        pytest.param("baseline", id="baseline"),
        pytest.param("perm", id="perm"),
        # "_bc" suffix: weight-only bias correction using the scalar global
        # ``X_BAR_GLOBAL``. See ``_compute_bias_correction_global`` — predicted
        # deeply null for bf16 (ΔW is ~16x smaller than in the bfp4 file).
        pytest.param("baseline_bc", id="baseline_bc"),
        pytest.param("perm_bc", id="perm_bc"),
        # "_bcpc" suffix: weight-only bias correction using a per-channel
        # ``x_bar`` precomputed from the gate input cache (one ``.mean(0)``,
        # no forward pass). See ``_compute_bias_correction_per_channel``.
        pytest.param("baseline_bcpc", id="baseline_bcpc"),
        pytest.param("perm_bcpc", id="perm_bcpc"),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (2, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
            id="mesh-2x2",
        ),
        pytest.param(
            (4, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
        pytest.param(
            (2, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="mesh-2x4",
        ),
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_forward_pass(
    mesh_device,
    num_links,
    topology,
    weight_variant,
):
    random.seed(42)
    torch.manual_seed(42)

    gate_fallback_mode = GateComputeMode.DEVICE  # on-device matmul + grouped-gate

    config = TtMoEGateConfig()
    config.ccl_config["NUM_LINKS"] = num_links
    adjust_shapes_for_testing(config, mesh_device)

    ref_config = SimpleNamespace(
        num_experts_per_tok=config.n_activated_experts,
        n_routed_experts=config.n_routed_experts,
        routed_scaling_factor=config.route_scale,
        scoring_func=config.score_func,
        topk_method="noaux_tc",
        n_group=config.n_expert_groups,
        topk_group=config.n_limited_groups,
        norm_topk_prob=True,
        hidden_size=config.dim,
    )
    reference_model = ReferenceMoEGate(ref_config, use_bitonic_sort=True)

    # Load / generate weights in fp32. ``_try_load_real_gate_weights`` returns
    # bf16; upcasting is lossless, so ``weight_fp32`` is bit-exact with the
    # stored HF bf16 weight when that path is taken.
    gate_w = _try_load_real_gate_weights(config.n_routed_experts, config.dim)
    if gate_w is None:
        gate_w = create_gate_weights(config.n_routed_experts, config.dim)
    weight_fp32 = gate_w["weight"].to(torch.float32)
    bias_fp32 = gate_w["e_score_correction_bias"].to(torch.float32)

    # --- Reference (fp32) ---------------------------------------------------
    reference_model.weight.data = weight_fp32
    reference_model.e_score_correction_bias.data = bias_fp32
    reference_model.eval()
    reference_model.to(torch.float32)

    n_sp_devices = mesh_device.shape[0]
    n_tp_devices = mesh_device.shape[1]
    total_seq_len = config.sp_dim * n_sp_devices

    torch_input_fp32 = _try_load_real_gate_input_fp32(total_seq_len, config.dim)
    if torch_input_fp32 is None:
        # 0.1147 is the std of the real gate input; scale for smaller dims.
        torch_input_fp32 = torch.randn(total_seq_len, config.dim, dtype=torch.float32) * 0.1147 * (7168 / config.dim)

    reference_topk_indices, reference_topk_scores = reference_model.grouped_forward(torch_input_fp32.unsqueeze(0))
    reference_logits = torch_input_fp32 @ weight_fp32.T

    # --- Weight preprocessing (the one thing that differs across variants) --
    # Bias-correction suffixes are orthogonal to the weight variant: peel
    # them off, run the base variant's weight preprocessing, and optionally
    # tweak the bias at upload time. ``_bcpc`` is checked before ``_bc``
    # because it contains ``_bc`` as a prefix.
    if weight_variant.endswith("_bcpc"):
        bc_mode = "per_channel"
        base_variant = weight_variant[: -len("_bcpc")]
    elif weight_variant.endswith("_bc"):
        bc_mode = "global_scalar"
        base_variant = weight_variant[: -len("_bc")]
    else:
        bc_mode = "none"
        base_variant = weight_variant
    logger.info(f"Weight variant: {weight_variant} (base={base_variant}, bias_correction={bc_mode})")
    weight_for_upload_fp32, expert_perm = _preprocess_weight(base_variant, weight_fp32, config.n_expert_groups)
    # Reconstruction error is measured against the **permuted** fp32 weight
    # for ``perm*`` variants, because un-permuting is free and doesn't count
    # as quantization noise. For non-permuted variants ``expert_perm`` is
    # None and this collapses to the original weight. For the bf16 path
    # ``baseline`` has zero host-side error (we ship fp32 unchanged and let
    # ttnn round); ``perm`` likewise — only the expert ordering changes.
    ref_weight_for_err = weight_fp32 if expert_perm is None else weight_fp32[expert_perm, :]
    weight_rel_err = _relative_frobenius_error(weight_for_upload_fp32, ref_weight_for_err)
    logger.info(
        f"[{weight_variant}] weight pre-upload reconstruction error "
        f"||w_host - w_fp32|| / ||w_fp32|| = {weight_rel_err:.6e}"
    )
    # Also log the bf16 round-trip error, which is the *actual* quantization
    # the device will apply. For bf16 this is always non-trivial (unlike
    # bfp4, where host-side preprocessing can reduce it), and serves as the
    # denominator for whether any bias correction has room to matter.
    bf16_rt_err = _relative_frobenius_error(_bf16_round_trip(weight_for_upload_fp32), ref_weight_for_err)
    logger.info(
        f"[{weight_variant}] bf16 round-trip error (host sim of device weight) "
        f"||bf16(w_host) - w_fp32|| / ||w_fp32|| = {bf16_rt_err:.6e}"
    )
    if expert_perm is not None:
        logger.info(
            f"[{weight_variant}] applied intra-group expert permutation "
            f"(n_expert_groups={config.n_expert_groups}, "
            f"experts_per_group={config.n_routed_experts // config.n_expert_groups})"
        )

    # --- Ship input to the mesh as bf16 (identical in all variants) ---------
    sharded_mem_config = get_input_mem_config(config, mesh_device.shape)
    tt_input = ttnn.from_torch(
        torch_input_fp32,
        device=mesh_device,
        dtype=ACTIVATION_DTYPE,
        memory_config=sharded_mem_config,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(0, -1),  # tensor parallel
            mesh_shape=mesh_device.shape,
        ),
    )

    # --- Build the TT gate with a bf16 weight -------------------------------
    # ``TtMoEGatePrefill.__init__`` already uploads the weight as bf16 in HF
    # layout (it transposes internally), so we hand it ``weight_for_upload_fp32``
    # directly and let the constructor do the round-trip. The bias stays in
    # bf16 — it is tiny and lives inside ``deepseek_grouped_gate``.
    n_routed_experts = config.n_routed_experts
    dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=n_routed_experts,
        dispatch_group_size=n_sp_devices,
        num_dispatch_groups=n_tp_devices,
    )
    # Permute the bias the same way as the weight so device-expert ``i``'s
    # weight row and bias entry both correspond to original-expert ``perm[i]``.
    bias_for_upload = gate_w["e_score_correction_bias"]
    if expert_perm is not None:
        bias_for_upload = bias_for_upload[expert_perm]

    # Optional weight-only bias correction. Two tiers:
    #   * ``_bc``   — scalar global ``X_BAR_GLOBAL`` (Tier 0).
    #   * ``_bcpc`` — per-channel ``x_bar`` from cache (Tier 1).
    # Both produce a per-expert ``Δbias_score`` added to the bias in fp32
    # before casting back to the bf16 dtype the device expects.
    if bc_mode != "none":
        ref_weight = weight_fp32 if expert_perm is None else weight_fp32[expert_perm]
        if bc_mode == "global_scalar":
            delta_bias = _compute_bias_correction_global(
                ref_weight=ref_weight,
                weight_for_upload_fp32=weight_for_upload_fp32,
            )
            bc_description = f"x_bar_global={X_BAR_GLOBAL} (Tier 0, ΔW row-sum method)"
        elif bc_mode == "per_channel":
            x_bar_pc = _try_load_x_bar_per_channel(config.dim)
            if x_bar_pc is None:
                logger.warning(
                    f"[{weight_variant}] per-channel x_bar not available — "
                    f"falling back to constant vector of X_BAR_GLOBAL "
                    f"(this collapses Tier 1 back onto Tier 0)."
                )
                x_bar_pc = torch.full((config.dim,), X_BAR_GLOBAL, dtype=torch.float32)
            delta_bias = _compute_bias_correction_per_channel(
                ref_weight=ref_weight,
                weight_for_upload_fp32=weight_for_upload_fp32,
                x_bar_per_channel=x_bar_pc,
            )
            bc_description = "per-channel x_bar from cache (Tier 1, ΔW · x_bar method)"
        else:
            raise ValueError(f"unknown bc_mode: {bc_mode!r}")

        bias_before_fp32 = bias_for_upload.to(torch.float32)
        bias_after_fp32 = bias_before_fp32 + delta_bias
        bias_for_upload = bias_after_fp32.to(torch.bfloat16)
        effective_delta = bias_for_upload.to(torch.float32) - bias_before_fp32
        logger.info(f"[{weight_variant}] weight-only bias correction — {bc_description}:")
        logger.info(
            f"  requested Δbias: min={delta_bias.min().item():.3e}, "
            f"max={delta_bias.max().item():.3e}, "
            f"mean={delta_bias.mean().item():.3e}, "
            f"|Δbias|_2={delta_bias.norm().item():.3e}"
        )
        logger.info(f"  effective (post-bf16-cast) Δbias: " f"|effective|_2={effective_delta.norm().item():.3e}")

    tt_model = TtMoEGatePrefill(
        config,
        mesh_device,
        dispatch_table=dispatch_table,
        weight=weight_for_upload_fp32,
        bias=bias_for_upload,
        fallback_mode=gate_fallback_mode,
    )

    # Note: unlike the bfp4 file, no ``_device_grouped_gate_bf16`` monkey-patch
    # is needed here. The matmul already emits bf16 logits, which is what
    # ``deepseek_grouped_gate`` wants.

    tt_topk_scores, tt_topk_indices, tt_logits, _, _ = tt_model(tt_input)

    # --- Validation ---------------------------------------------------------
    # Loose thresholds — the whole point of this test is to measure
    # quantization sensitivity, so we do not want pytest to fail on a small PCC
    # drop. We keep these in line with the device-mode thresholds from
    # ``test_moe_gate_prefill2d.py`` and let the logged numbers tell the story.
    recall_threshold = 0.70
    logits_pcc_threshold = 0.70
    scores_pcc_threshold = 0.70

    seq_len_per_device = reference_logits.shape[0] // mesh_device.shape[0]
    sp_composer = get_sp_mesh_composer(mesh_device)

    # For ``perm*`` variants the device works in the permuted expert space,
    # so we need to un-permute every tensor whose expert axis we compare to
    # the fp32 reference. For the full per-expert logit tensor the mapping
    # is ``logits_orig[:, j] = logits_device[:, inv_perm[j]]``, since device
    # column ``i`` corresponds to original expert ``perm[i]``. For topk
    # indices the device emits permuted-space IDs, which we map back with
    # ``perm[]`` directly.
    if expert_perm is not None:
        inv_perm = torch.empty_like(expert_perm)
        inv_perm[expert_perm] = torch.arange(expert_perm.numel(), dtype=expert_perm.dtype)
    else:
        inv_perm = None

    host_tt_topk_indices = ttnn.to_torch(tt_topk_indices, mesh_composer=sp_composer)
    if expert_perm is not None:
        host_tt_topk_indices = expert_perm[host_tt_topk_indices.long()]
    host_tt_topk_indices = host_tt_topk_indices.view(1, n_sp_devices, seq_len_per_device, -1).sort(dim=-1).values
    reference_topk_indices = reference_topk_indices.view(1, n_sp_devices, seq_len_per_device, -1).sort(dim=-1).values

    recall_topk_indices = validate_composed(
        host_tt_topk_indices,
        reference_topk_indices,
        1,
        n_sp_devices,
        compare_recall(recall_threshold),
        name="recall_topk_indices",
        broadcast_groups=n_tp_devices,
    )

    host_tt_logits = ttnn.to_torch(tt_logits, mesh_composer=sp_composer)
    if inv_perm is not None:
        # Un-permute the expert axis: ``logits_orig[..., j] = logits_dev[..., inv_perm[j]]``.
        host_tt_logits = host_tt_logits[..., inv_perm]
    host_tt_logits = host_tt_logits.view(1, n_sp_devices, seq_len_per_device, -1)
    reference_logits = reference_logits.view(1, n_sp_devices, seq_len_per_device, -1)

    pcc_logits = validate_composed(
        host_tt_logits,
        reference_logits,
        1,
        n_sp_devices,
        compare_pcc(logits_pcc_threshold),
        name="pcc_logits",
        broadcast_groups=n_tp_devices,
    )

    host_tt_topk_scores = ttnn.to_torch(tt_topk_scores, mesh_composer=sp_composer)
    host_tt_topk_scores = host_tt_topk_scores.view(1, n_sp_devices, seq_len_per_device, -1)
    reference_topk_scores = reference_topk_scores.view(1, n_sp_devices, seq_len_per_device, -1)

    pcc_scores = validate_composed(
        host_tt_topk_scores,
        reference_topk_scores,
        1,
        n_sp_devices,
        compare_pcc(scores_pcc_threshold),
        name="pcc_scores",
        broadcast_groups=n_tp_devices,
    )

    all_results = [recall_topk_indices, pcc_logits, pcc_scores]

    for res in all_results:
        res.log_mismatches()

    log_validation_results(
        results=all_results,
        num_dispatch_groups=n_tp_devices,
        dispatch_group_size=n_sp_devices,
        title=f"Gate Prefill Quantization Validation ({weight_variant})",
    )

    # ---------------------------------------------------------------------
    # Explicit PCC / recall summary
    # ---------------------------------------------------------------------
    # ``validate_composed`` only records pass/fail plus mismatch details, so a
    # passing run prints nothing numeric. For a quantization-sensitivity
    # experiment that is exactly the information we want, so re-compute the
    # PCC and recall directly here and log the scalar values. Per-chip
    # numbers come from iterating the SP-composed tensors the same way
    # ``validate_composed`` does.
    per_chip_logits_pcc = []
    per_chip_scores_pcc = []
    per_chip_recall = []
    for c in range(n_sp_devices):
        _, chip_logits_pcc = comp_pcc(reference_logits[0, c].float(), host_tt_logits[0, c].float(), pcc=0.0)
        _, chip_scores_pcc = comp_pcc(reference_topk_scores[0, c].float(), host_tt_topk_scores[0, c].float(), pcc=0.0)
        chip_recall = calculate_average_recall(host_tt_topk_indices[0, c], reference_topk_indices[0, c])
        per_chip_logits_pcc.append(chip_logits_pcc)
        per_chip_scores_pcc.append(chip_scores_pcc)
        per_chip_recall.append(chip_recall)

    _, global_logits_pcc = comp_pcc(reference_logits.reshape(-1).float(), host_tt_logits.reshape(-1).float(), pcc=0.0)
    _, global_scores_pcc = comp_pcc(
        reference_topk_scores.reshape(-1).float(), host_tt_topk_scores.reshape(-1).float(), pcc=0.0
    )
    global_recall = sum(per_chip_recall) / len(per_chip_recall)

    logger.info("=" * 80)
    logger.info(
        f"Gate prefill quantization summary [{weight_variant}] — "
        f"dtype={ACTIVATION_DTYPE}, mesh={tuple(mesh_device.shape)}"
    )
    logger.info(f"  weight reconstruction error (pre-upload, vs fp32): {weight_rel_err:.6e}")
    logger.info(f"  bf16 round-trip error (host sim of device weight): {bf16_rt_err:.6e}")
    logger.info("-" * 80)
    header = f"{'chip':>6} | {'logits_pcc':>12} | {'scores_pcc':>12} | {'recall':>10}"
    logger.info(header)
    logger.info("-" * len(header))
    for c in range(n_sp_devices):
        logger.info(
            f"{c:>6d} | {per_chip_logits_pcc[c]:>12.6f} | {per_chip_scores_pcc[c]:>12.6f} | "
            f"{per_chip_recall[c]:>10.4f}"
        )
    logger.info("-" * len(header))
    logger.info(f"{'GLOBAL':>6} | {global_logits_pcc:>12.6f} | {global_scores_pcc:>12.6f} | {global_recall:>10.4f}")
    logger.info("=" * 80)

    _RESULTS.append(
        {
            "variant": weight_variant,
            "mesh": f"{mesh_device.shape[0]}x{mesh_device.shape[1]}",
            "logits_pcc": global_logits_pcc,
            "scores_pcc": global_scores_pcc,
            "recall": global_recall,
            "w_rel_err": weight_rel_err,
            "bf16_rt_err": bf16_rt_err,
        }
    )

    merged = ValidationResult.merge(all_results, name=f"gate_prefill_quantization_{weight_variant}")
    merged.assert_passed(f"Gate prefill quantization validation failed ({weight_variant})")
