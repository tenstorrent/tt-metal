# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Weight-quantization experiment for the BFLOAT8_B MoE gate prefill.

The on-device gate matmul runs in BFLOAT8_B for both activations and weights.
BFP8 is a block-float format (one shared 8-bit exponent per 16-element group,
sign-magnitude mantissa with 1 sign + 7 mantissa bits → 255 distinct levels).
That's ~18x finer per-element resolution than BFP4 (1s + 3m → 15 levels),
so BFP8 rounding noise is usually well under the tolerance of the grouped-
gate topk step — but on the few expert scores that happen to be within a
rounding step of each other BFP8 can still flip selections, which is what
this test measures.

``deepseek_grouped_gate`` does not accept BFP8 (same restriction as BFP4),
so the matmul logits are typecast back to BFLOAT16 right before grouped-
gate. That cast is the only BF16 step in the pipeline.

The test compares several host-side weight pre-processing strategies, all of
them input-agnostic. The activation path is identical across variants (plain
``ttnn.from_torch(dtype=bfloat8_b)``); only the weight tensor differs.

Variants:

    ``baseline``        — ttnn packs fp32 weight directly at upload time.
    ``sim_bfp8``        — host bfp8 with offline-optimal shared exponent
                          chosen per 16-element group (``simulate_bfp``'s
                          default: tries ``E`` and ``E+1``, picks lower MSE).
                          ttnn re-packs at upload but sees values already on
                          the bfp8 grid, so the re-pack is near-idempotent.
    ``dither``          — add uniform noise on ``[-step/2, +step/2]`` per
                          element before rounding (non-subtractive /
                          Schuchman dither). Makes the quantization error
                          statistically independent of the signal so it
                          averages out in the dim-axis reduction. Much
                          smaller-footprint than in bfp4 since ``step`` is
                          ~18x smaller.
    ``rank_exp``        — pick the per-group shared exponent that minimizes
                          the number of within-group ties (with MSE as a
                          secondary key), over an extended search set
                          ``{E-1, E, E+1}``. Directly targets collisions
                          on the 255-level bfp8 grid — but ties are already
                          rare at this resolution, so expected upside is
                          smaller than in bfp4.
    ``dither_rank_exp`` — A + B stacked: dither first, then tie-minimizing
                          exponent search.

In addition, any variant can be suffixed with ``_bc`` or ``_bcpc`` to enable
a weight-only bias correction:

  * ``_bc``   — scalar-global x_bar (Tier 0 in our hierarchy). The correction
                is ``Δbias[e] = − s'(0) · X_BAR_GLOBAL · Σ_i ΔW[e, i]`` (see
                ``_compute_bias_correction_global``). With the observed
                ``X_BAR_GLOBAL ≈ 4e-4`` this is a deliberate null test, and
                even more null-ish for bfp8 than bfp4 because ``ΔW`` is
                ~18x smaller per element here.

  * ``_bcpc`` — per-channel x_bar precomputed from the gate input cache with
                a single ``.mean(0)`` (see ``_try_load_x_bar_per_channel``).
                The correction is ``Δbias[e] = − s'(0) · ΔW[e, :] · x_bar``
                (see ``_compute_bias_correction_per_channel``). Still purely
                weight-derived at runtime — the per-channel ``x_bar`` is a
                single 7168-dim vector loaded once.

For each variant we log PCC / recall against the fp32 reference *and* the
relative reconstruction error ``||w_host - w_fp32|| / ||w_fp32||`` of the
weight tensor shipped to the device (pre the final ttnn re-quantization),
so we can separate "the host step built a better tensor" from "ttnn's
packer ended up with a better shared exponent".
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
from models.demos.kimi26_d_p.utils.simulate_bfp import GROUP_SIZE, MAX_EXPONENT, MIN_EXPONENT, quantize_bfp
from models.demos.kimi26_d_p.utils.test_utils import adjust_shapes_for_testing, get_input_mem_config

_DEFAULT_HF_REPO = "deepseek-ai/DeepSeek-V3"
_LOCAL_FALLBACKS = (
    "models/demos/kimi26_d_p/reference",
    "/proj_sw/user_dev/moonshotai/Kimi-K2.5",
)

# All variants quantize both activations and weights to this dtype on device;
# only the on-host weight preprocessing differs between variants.
ACTIVATION_DTYPE = ttnn.bfloat8_b
BFP_MANTISSA_BITS = 8  # matches ACTIVATION_DTYPE (bfp8 = 1 sign + 7 mantissa bits)

# Dither seed — fixed so every run produces the same dithered weight.
DITHER_SEED = 4321

# Global activation statistics at layer 3 (post-RMSNorm gate input), measured
# once offline on ``gate_input_layer3_seq100000.pt``:
#   mean = 0.0004,  std = 0.1147,  min ≈ -7.69,  max ≈ 8.19
# These four scalars are all the "input info" the "_bc" variants may use —
# no per-channel statistics, no forward pass on the cache. At runtime the
# bias correction is a fixed vector computed purely from the weight and its
# bfp8 round-trip error, multiplied by ``X_BAR_GLOBAL``.
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
    logger.info("=" * 110)
    logger.info(f"Gate prefill quantization — cross-variant summary (dtype={ACTIVATION_DTYPE})")
    logger.info("-" * 110)
    header = (
        f"{'variant':>28} | {'mesh':>7} | {'logits_pcc':>12} | "
        f"{'scores_pcc':>12} | {'recall':>10} | {'w_rel_err':>12}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for r in _RESULTS:
        logger.info(
            f"{r['variant']:>28} | {r['mesh']:>7} | {r['logits_pcc']:>12.6f} | "
            f"{r['scores_pcc']:>12.6f} | {r['recall']:>10.4f} | {r['w_rel_err']:>12.6e}"
        )
    logger.info("=" * 110)


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
# Weight preprocessing variants
# ---------------------------------------------------------------------------
#
# IMPORTANT — layout / blocking alignment:
#
# The matmul wants the weight in device layout ``(dim, n_routed_experts)``,
# which we produce with ``.T.contiguous()`` right before uploading. ttnn packs
# that tile-by-tile with **groups of 16 along the last (contiguous) axis**,
# i.e. groups of 16 experts share a shared exponent for every fixed
# ``dim`` position.
#
# ``simulate_bfp.quantize_bfp`` likewise groups 16 elements along the last
# axis of whatever tensor it receives. To make the host simulation and the
# device packer group the **same** 16 elements together, the host step has
# to operate on the device-layout tensor ``w_dev = w_hf.T``. Otherwise we
# pre-quantize along one axis, ttnn re-quantizes along an orthogonal axis,
# and the two rounding steps compound instead of cancel.
#
# All variants operate on ``(dim, n_experts)`` and return the same layout,
# and the test transposes once more at the very end before uploading.
#
# Variants (all input-agnostic):
#
#   * ``baseline``        — no host preprocessing, ttnn packs fp32 at upload.
#   * ``sim_bfp8``        — offline-optimal MSE exponent (``simulate_bfp``).
#   * ``dither``          — uniform non-subtractive dither on
#       ``[-step/2, step/2]`` per element before rounding, step set by each
#       group's base exponent. Quantization error becomes signal-independent,
#       so the dim-axis reduction in ``X @ W.T`` averages it out. Bfp8's
#       step is ~18x smaller than bfp4, so dither helps less here.
#   * ``rank_exp``        — per-group shared exponent chosen to *minimize the
#       number of within-group ties* (MSE as a strict secondary key,
#       lexicographic order via ``score = 1e3*ties + mse``). Extends the
#       search to ``{E-1, E, E+1}``. Aggressive: will pick a tight exponent
#       and clip an outlier badly if it produces one extra distinct level.
#       At 255 levels ties are already uncommon, so this is mostly a
#       control for "is tie-count the right objective?".
#   * ``dither_rank_exp`` — dither + hard rank_exp, stacked.
#   * ``soft_rank_01``    — soft rank-aware: ``score = mse + 0.1 * step^2 *
#       ties``. Ties are penalized in MSE-comparable units (``step^2`` ≈ the
#       MSE cost of one lost level of resolution), so the search only picks
#       a non-MSE-optimal exponent when the tie reduction outweighs the
#       MSE sacrifice.
#   * ``soft_rank_1``     — same soft objective with ``ties_weight = 1.0``.
#       More willing to trade MSE for fewer ties.
#   * ``perm``            — global expert permutation within each expert
#       group: sort by descending per-expert L2 norm so similar-magnitude
#       experts share a bfp8 group of 16. Keeps grouped-gate topk semantics
#       intact (group selection is permutation-invariant within groups).
#       ttnn packs as usual at upload; only the bfp8 group membership
#       changes.
#   * ``perm_sim_bfp8``   — ``perm`` + ``sim_bfp8`` stacked.
#
# Bias-correction variants (``_bc`` / ``_bcpc`` suffix, orthogonal to
# weight variant):
#
#   * ``baseline_bc``, ``sim_bfp8_bc``, ``perm_sim_bfp8_bc`` — Tier 0.
#       Adjust ``e_score_correction_bias`` by
#       ``− s'(0) · X_BAR_GLOBAL · Σ_i ΔW[e, i]``. Predicted null under
#       ``X_BAR_GLOBAL ≈ 4e-4``, and even more so for bfp8 where ``ΔW``
#       is ~18x smaller per element than bfp4.
#
#   * ``baseline_bcpc``, ``sim_bfp8_bcpc``, ``perm_sim_bfp8_bcpc`` —
#       Tier 1. Adjust bias by ``− s'(0) · (ΔW[e, :] · x_bar_per_channel)``,
#       with ``x_bar_per_channel`` computed once from
#       ``gate_input_layer3_seq100000.pt`` via a single ``.mean(0)``.
#       Channels with non-zero mean stop cancelling into the global scalar,
#       so the correction has room to be non-trivial.


def _bfp_quantize_with_search(
    flat: torch.Tensor,
    mantissa_bits: int,
    offsets: tuple[int, ...],
    objective: str,
    ties_weight: float = 1.0,
) -> torch.Tensor:
    """Block-float quantize ``flat`` of shape ``(num_groups, group_size)``.

    Per group, tries each exponent offset (relative to ``floor(log2(max|x|))``)
    and keeps the one minimizing ``objective``:

        ``"mse"``       — sum of squared errors within the group.
        ``"ties_hard"`` — lexicographic: fewest ties, MSE breaks remaining
                          ties (``score = 1e3 * ties + mse``). Aggressive.
        ``"ties_soft"`` — soft: ``score = mse + ties_weight * step^2 * ties``.
                          Ties penalized in MSE-comparable units. ``ties_weight``
                          is a tunable tradeoff: 0 → pure MSE, ∞ → pure ties.
    """
    max_abs = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    base_exp = torch.floor(torch.log2(max_abs)).clamp(MIN_EXPONENT, MAX_EXPONENT)
    n_levels = 2 ** (mantissa_bits - 1)
    max_mantissa = n_levels - 1

    best_dequant: torch.Tensor | None = None
    best_score: torch.Tensor | None = None

    for offset in offsets:
        exp = (base_exp + offset).clamp(min=MIN_EXPONENT, max=MAX_EXPONENT)
        scale = (2.0**exp) / n_levels
        mantissa = torch.round(flat / scale).clamp(-max_mantissa, max_mantissa)
        dequant = mantissa * scale

        mse = (flat - dequant).pow(2).sum(dim=-1, keepdim=True)
        if objective == "mse":
            score = mse
        elif objective in ("ties_hard", "ties_soft"):
            sorted_vals, _ = dequant.sort(dim=-1)
            diffs = sorted_vals[..., 1:] - sorted_vals[..., :-1]
            ties = (diffs == 0).sum(dim=-1, keepdim=True).to(flat.dtype)
            if objective == "ties_hard":
                score = ties * 1e3 + mse
            else:  # ties_soft
                # ``step^2 = scale^2`` — the MSE cost of "one lost level of
                # resolution" at the current exponent. This puts ties and
                # MSE in comparable units so ``ties_weight`` is an
                # interpretable tradeoff knob (0 = pure MSE, 1 ≈ each
                # tie costs one step^2, etc.).
                score = mse + ties_weight * scale.pow(2) * ties
        else:
            raise ValueError(f"unknown objective: {objective!r}")

        if best_dequant is None:
            best_dequant = dequant
            best_score = score
        else:
            improved = score < best_score
            best_dequant = torch.where(improved, dequant, best_dequant)
            best_score = torch.where(improved, score, best_score)

    assert best_dequant is not None
    return best_dequant


def _quantize_bfp_custom(
    tensor: torch.Tensor,
    *,
    dither: bool = False,
    rank_mode: str = "mse",
    ties_weight: float = 1.0,
    mantissa_bits: int = BFP_MANTISSA_BITS,
    group_size: int = GROUP_SIZE,
    dither_seed: int = DITHER_SEED,
) -> torch.Tensor:
    """Extended BFP quantizer (bfp8 here): optional dither + rank-aware exponent search.

    ``dither``       — inject ``U(-step/2, step/2)`` per element before
                       rounding, with ``step`` derived from each group's
                       base exponent (``offset = 0``). This is non-subtractive
                       dither: quantization error becomes uniform and
                       statistically independent of the signal, so the
                       7168-wide reduction in the gate matmul averages it out.
    ``rank_mode``    — ``"mse"`` (``simulate_bfp`` default, offsets ``{0, 1}``),
                       ``"ties_hard"`` (lexicographic ties-then-MSE, offsets
                       ``{-1, 0, 1}``), or ``"ties_soft"`` (soft ties penalty
                       with ``ties_weight``, offsets ``{-1, 0, 1}``).
    ``ties_weight``  — tradeoff scalar for ``"ties_soft"``. Each within-group
                       tie adds ``ties_weight * step^2`` to the score. Units
                       match MSE, so 0 means "pure MSE", 1 means "each lost
                       distinct level costs about one step^2 of MSE".
    """
    orig_shape = tensor.shape
    flat = tensor.reshape(-1, group_size).contiguous()

    if dither:
        max_abs = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
        base_exp = torch.floor(torch.log2(max_abs)).clamp(MIN_EXPONENT, MAX_EXPONENT)
        n_levels = 2 ** (mantissa_bits - 1)
        step = (2.0**base_exp) / n_levels

        gen = torch.Generator().manual_seed(dither_seed)
        noise = (torch.rand(flat.shape, generator=gen, dtype=flat.dtype) - 0.5) * step
        flat = flat + noise

    if rank_mode == "mse":
        offsets: tuple[int, ...] = (0, 1)
        objective = "mse"
    elif rank_mode in ("ties_hard", "ties_soft"):
        offsets = (-1, 0, 1)
        objective = rank_mode
    else:
        raise ValueError(f"unknown rank_mode: {rank_mode!r}")

    dequant = _bfp_quantize_with_search(flat, mantissa_bits, offsets, objective, ties_weight=ties_weight)
    return dequant.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Expert permutation (intra-group)
# ---------------------------------------------------------------------------
#
# grouped-gate picks top-k **expert groups** first (sum of top-2 scores per
# group), then top-k experts within the selected groups. Group selection is
# permutation-invariant *within* a group but not *across* groups, so we can
# only safely permute experts inside their existing expert group.
#
# Within a group of ``experts_per_group = n_routed_experts / n_expert_groups``
# experts, we sort by descending per-expert L2 norm and take the experts in
# that order. Contiguous binning of 16 then produces bfp8 groups of similar-
# magnitude experts so the shared exponent fits everyone.


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
    Internally we transpose to the device layout ``(dim, n_experts)`` so
    host-side bfp8 blocking aligns with the ttnn packer's blocking.

    Returns:
        ``(weight_for_upload, perm_or_None)``. When ``perm`` is not None,
        the caller must also permute the bias by ``bias[perm]`` and apply
        ``perm[]`` to the device's topk indices before comparison.
    """
    perm: torch.Tensor | None = None
    w_hf = weight_fp32
    if variant in ("perm", "perm_sim_bfp8"):
        perm = _compute_expert_permutation(weight_fp32, n_expert_groups)
        w_hf = weight_fp32[perm, :]

    # (n_experts, dim) -> (dim, n_experts). bfp8 groups of 16 run along
    # ``n_experts`` here — same as ttnn's packer at upload time.
    w_dev = w_hf.T.contiguous()

    if variant in ("baseline", "perm"):
        w_dev_out = w_dev
    elif variant in ("sim_bfp8", "perm_sim_bfp8"):
        w_dev_out = quantize_bfp(w_dev, mantissa_bits=BFP_MANTISSA_BITS)
    elif variant == "dither":
        w_dev_out = _quantize_bfp_custom(w_dev, dither=True, rank_mode="mse")
    elif variant == "rank_exp":
        w_dev_out = _quantize_bfp_custom(w_dev, dither=False, rank_mode="ties_hard")
    elif variant == "dither_rank_exp":
        w_dev_out = _quantize_bfp_custom(w_dev, dither=True, rank_mode="ties_hard")
    elif variant == "soft_rank_01":
        w_dev_out = _quantize_bfp_custom(w_dev, rank_mode="ties_soft", ties_weight=0.1)
    elif variant == "soft_rank_1":
        w_dev_out = _quantize_bfp_custom(w_dev, rank_mode="ties_soft", ties_weight=1.0)
    else:
        raise ValueError(f"unknown weight variant: {variant!r}")

    # Back to HF ``(n_experts, dim)``; the test does another ``.T.contiguous()``
    # later for the upload.
    return w_dev_out.T.contiguous(), perm


def _relative_frobenius_error(approx: torch.Tensor, ref: torch.Tensor) -> float:
    return torch.norm(approx - ref).item() / (torch.norm(ref).item() + 1e-30)


# ---------------------------------------------------------------------------
# Weight-only bias correction (Option A: global scalar x_bar)
# ---------------------------------------------------------------------------
#
# Derivation. Logit error per token / expert:
#
#     err[t, e] = x[t] · ΔW[e, :]    with ΔW = W_q − W
#
# Expected error, assuming ``E[x_i] = x_bar_global`` uniformly across channels
# (the weakest "input info" assumption compatible with just a scalar mean):
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
# Expected magnitude. ``x_bar_global ≈ 4e-4`` and ``Σ_i ΔW[e, i]`` is at most
# a fraction in magnitude (much smaller by CLT across 7168 near-zero-mean
# rounding errors, and ~18x smaller per element for bfp8 than bfp4), so
# ``Δbias_score`` comes out at well under O(1e-5) per expert. Bias values in
# the HF checkpoint are on the order of 0.01–0.1, so this is a < 0.01%
# perturbation — and bf16 resolution at those magnitudes is 4e-4, so the
# correction is guaranteed to round away on cast. This experiment is
# deliberately a null test to confirm the theoretical prediction (Option A in
# the "no calibration inputs" hierarchy): with only a near-zero global mean,
# purely weight-derived bias correction has no room to work.
#
# Device-facing ΔW. The device sees a bfp8 version of whatever we upload. For
# ``sim_bfp8_*`` variants ``weight_for_upload`` is already on the bfp8 grid
# and ttnn's re-pack is near-idempotent; for ``baseline`` it's the raw fp32
# weight and ttnn does the full quantization. To stay consistent across
# variants we always run ``weight_for_upload`` through ``simulate_bfp``
# (bit-exact with the ttnn packer, parameterized by ``BFP_MANTISSA_BITS``)
# and diff against the permutation-matched reference.


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
            ``(n_experts, dim)``. Passed through ``simulate_bfp`` internally
            to approximate the weight the device will materialize.
        x_bar_global: scalar assumed ``E[x_i]`` for every channel.
    Returns:
        ``Δbias_score`` of shape ``(n_experts,)``. Add this to the shipped
        bias.
    """
    # Simulate the bfp8 round-trip ttnn performs at upload. Layout note:
    # ttnn packs with groups of 16 along the last (contiguous) axis of the
    # device-layout tensor ``(dim, n_experts)``, which ``quantize_bfp`` also
    # does — so we transpose here, quantize, and transpose back.
    w_dev = weight_for_upload_fp32.T.contiguous()
    w_dev_q = quantize_bfp(w_dev, mantissa_bits=BFP_MANTISSA_BITS)
    delta_w = w_dev_q.T.contiguous() - ref_weight  # (n_experts, dim)
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
# Why this might work where Tier 0 failed. The global mean of the cached
# activations is 0.0004 — the per-expert row sum ``Σ_i ΔW[e, i]`` multiplies
# that to produce a near-zero correction. But different channels can have
# per-channel means of much larger magnitude (e.g. 0.01 − 0.1) that cancel
# when averaged to a global scalar. The dot product ``ΔW[e, :] · x_bar``
# recovers exactly the per-expert logit bias that the global scalar washes
# out.


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
            ``(n_experts, dim)``. Passed through ``simulate_bfp`` internally
            to approximate the weight the device will materialize.
        x_bar_per_channel: per-channel activation mean, shape ``(dim,)``.
    Returns:
        ``Δbias_score`` of shape ``(n_experts,)``. Add this to the shipped
        bias.
    """
    w_dev = weight_for_upload_fp32.T.contiguous()
    w_dev_q = quantize_bfp(w_dev, mantissa_bits=BFP_MANTISSA_BITS)
    delta_w = w_dev_q.T.contiguous() - ref_weight  # (n_experts, dim)
    # Per-expert expected logit error: ΔW[e, :] · x_bar.
    delta_logit_per_expert = delta_w @ x_bar_per_channel.to(delta_w.dtype)
    return -SIGMOID_PRIME_AT_ZERO * delta_logit_per_expert


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weight_variant",
    [
        pytest.param("baseline", id="baseline"),
        pytest.param("sim_bfp8", id="sim_bfp8"),
        pytest.param("dither", id="dither"),
        pytest.param("rank_exp", id="rank_exp"),
        pytest.param("dither_rank_exp", id="dither_rank_exp"),
        pytest.param("soft_rank_01", id="soft_rank_01"),
        pytest.param("soft_rank_1", id="soft_rank_1"),
        pytest.param("perm", id="perm"),
        pytest.param("perm_sim_bfp8", id="perm_sim_bfp8"),
        # "_bc" suffix: weight-only bias correction using the scalar global
        # ``X_BAR_GLOBAL``. See ``_compute_bias_correction_global`` for the
        # math — expected to be near-zero by construction. Confirmed null in
        # practice; kept in the matrix for reference.
        pytest.param("baseline_bc", id="baseline_bc"),
        pytest.param("sim_bfp8_bc", id="sim_bfp8_bc"),
        pytest.param("perm_sim_bfp8_bc", id="perm_sim_bfp8_bc"),
        # "_bcpc" suffix: weight-only bias correction using a per-channel
        # ``x_bar`` precomputed from the gate input cache (one ``.mean(0)``,
        # no forward pass). See ``_compute_bias_correction_per_channel``.
        pytest.param("baseline_bcpc", id="baseline_bcpc"),
        pytest.param("sim_bfp8_bcpc", id="sim_bfp8_bcpc"),
        pytest.param("perm_sim_bfp8_bcpc", id="perm_sim_bfp8_bcpc"),
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
    # tweak the bias at upload time (below, right before
    # ``TtMoEGatePrefill`` is built). ``_bcpc`` is checked before ``_bc``
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
    # None and this collapses to the original weight.
    ref_weight_for_err = weight_fp32 if expert_perm is None else weight_fp32[expert_perm, :]
    weight_rel_err = _relative_frobenius_error(weight_for_upload_fp32, ref_weight_for_err)
    logger.info(
        f"[{weight_variant}] weight pre-upload reconstruction error "
        f"||w_host - w_fp32|| / ||w_fp32|| = {weight_rel_err:.6e}"
    )
    if expert_perm is not None:
        logger.info(
            f"[{weight_variant}] applied intra-group expert permutation "
            f"(n_expert_groups={config.n_expert_groups}, "
            f"experts_per_group={config.n_routed_experts // config.n_expert_groups})"
        )

    # --- Ship input to the mesh as bfp8 (identical in all variants) ---------
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

    # --- Build the TT gate with a bfp8 weight -------------------------------
    # ``TtMoEGatePrefill.__init__`` hard-codes ttnn.bfloat16 for its weight
    # tensor. To use a different dtype here we construct the model with
    # ``weight=None`` (allocates a zero placeholder) and then swap
    # ``self.weight`` for a ttnn tensor in ACTIVATION_DTYPE built from our
    # pre-processed fp32 weight. The bias stays in bf16 — it is tiny and
    # lives inside ``deepseek_grouped_gate``, which wants bf16 anyway.
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
    #   * ``_bc``   — scalar global ``X_BAR_GLOBAL`` (Tier 0, confirmed null).
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
        weight=None,
        bias=bias_for_upload,
        fallback_mode=gate_fallback_mode,
    )

    ttnn.deallocate(tt_model.weight)
    # TTNN matmul expects the weight in (dim, n_routed_experts) layout.
    tt_model.weight = ttnn.from_torch(
        weight_for_upload_fp32.T.contiguous(),
        device=mesh_device,
        dtype=ACTIVATION_DTYPE,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0),
            mesh_shape=mesh_device.shape,
        ),
    )

    # ``deepseek_grouped_gate`` does not accept BFP8 (same restriction as
    # BFP4), so cast the logits to bf16 right before the grouped-gate op. We
    # intercept the method call rather than patching the model class so this
    # stays local to the test.
    _orig_grouped_gate_bf16 = tt_model._device_grouped_gate_bf16

    def _grouped_gate_with_bf16_cast(logits):
        logits_bf16 = ttnn.typecast(logits, ttnn.bfloat16)
        out = _orig_grouped_gate_bf16(logits_bf16)
        ttnn.deallocate(logits_bf16)
        return out

    tt_model._device_grouped_gate_bf16 = _grouped_gate_with_bf16_cast

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
        }
    )

    merged = ValidationResult.merge(all_results, name=f"gate_prefill_quantization_{weight_variant}")
    merged.assert_passed(f"Gate prefill quantization validation failed ({weight_variant})")
