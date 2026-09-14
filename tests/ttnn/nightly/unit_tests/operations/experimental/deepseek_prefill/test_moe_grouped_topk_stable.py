# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Strict-bar tests for moe_grouped_topk's stable top-k path, as the Kimi / DeepSeek prefill MoE gate
(``tt_moe_gate_prefill.py``) requests it on every gated layer.

- ``test_stable_matches_unstable``: stable and unstable must agree exactly wherever the sorted keys
  are distinct; device against device, no golden needed.
- ``test_stable_exact_vs_golden``: exact expert-set match against the torch reference on inputs with
  a designed top-k boundary margin (``separated_gate_inputs``), swept over the blocks::topk chain depth.
- ``test_index_domain``: every returned id valid and distinct, every padded row the sentinel.

Exact tie ORDER is covered by test_moe_grouped_topk_ties.py.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    assert_gate_output,
    assert_index_domain,
    build_padding_config,
    distinct_logits,
    grouped_gate_golden_act,
    score_activation,
)
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP

EPSILON = 1e-20

# (n_groups, total_experts, summed_experts_per_group, topk_groups, n_activated_experts, route_scale)
# Kimi is the configuration that regressed; the other two broaden which stable blocks run --
# n_groups == 1 reaches only blocks::topk, n_groups == 8 also reaches process_and_sort_tiles and
# topk_group_scores.
KIMI = (1, 384, 1, 1, 8, 2.827)
DSV4_FLASH = (1, 256, 1, 1, 6, 1.5)
DEEPSEEK_GROUPED = (8, 256, 2, 4, 8, 0.5)

ROUTING_CONFIGS = [KIMI, DSV4_FLASH, DEEPSEEK_GROUPED]
ROUTING_CONFIG_IDS = ["kimi-1g384e", "dsv4flash-1g256e", "deepseek-8g256e"]

# The differential covers the single-group configs only. On the grouped path the final expert
# boundary is not the only decision point: sum_top_experts_per_group + topk_group_scores first
# choose which topk_groups of n_groups survive, so a near-tie between two GROUP SUMS swaps a whole
# group of experts while the expert-level boundary still looks well separated. Masking correctly
# there means reproducing the group-sum ranking in the test, which is a separate piece of work;
# without it the predicate admits legitimately ambiguous tokens and reports a defect on main just
# as readily (measured: main disagrees on 1-4 of ~590 tokens per case, this branch on 3-7, and the
# disagreement does not vanish even at a 6.4e-2 relative boundary tolerance -- i.e. it is not a
# boundary-proximity effect at all). Index-domain invariants below still cover all three configs.
DIFFERENTIAL_CONFIGS = [KIMI, DSV4_FLASH]
DIFFERENTIAL_CONFIG_IDS = ["kimi-1g384e", "dsv4flash-1g256e"]

# total_experts -> width tiles -> merge rounds in the blocks::topk insertion chain (tiles - 2).
# n_groups == 1 only requires experts % 32 == 0, so the expert axis is free to sweep here.
WIDTH_SWEEP = [64, 128, 256, 384, 512]
WIDTH_SWEEP_IDS = [f"e{e}-{e // 32}tiles-{max(e // 32 - 2, 0)}rounds" for e in WIDTH_SWEEP]

INPUT_DTYPES = [ttnn.float32, ttnn.bfloat16]
INPUT_DTYPE_IDS = ["in_fp32", "in_bf16"]

SCORE_FUNCS = ["sigmoid", "sqrtsoftplus"]

# One prefill chunk's worth of tokens on one chip (5120 // PREFILL_SP_FACTOR), i.e. exactly the
# per-call token count the failing model runs.
SEQ_LEN = PREFILL_CHUNK_TOKENS_PER_CHIP

# Boundary margin for the designed-separation inputs, in biased-score units. Large enough that
# neither bf16 input quantization (~2e-2 at these magnitudes) nor the device/torch activation
# difference can reorder the selection, small enough to keep every value well inside bf16 range.
MARGIN = 1.0

# Effective comparison resolution of the top-k datapath, as a RELATIVE gap between two keys.
# The network does not compare at fp32: measured on Blackhole over 5077 tokens (both input dtypes,
# 4 seeds, Kimi geometry), selection is exactly torch's from a relative gap of 8e-4 upward and
# starts deviating below it -- consistent with roughly a 10-bit effective mantissa in the sort
# datapath. Keys closer than this are ties to the hardware however far apart fp32 says they are,
# so every tie-freedom predicate below is a tolerance rather than an exact comparison. 2e-3 keeps
# ~93% of tokens assertable with headroom over the measured 8e-4 floor.
#
# This is a property of the datapath, not of this PR: main deviates at the same ~1% rate on the
# same near-boundary tokens. An exact predicate here reports a defect on every branch.
SORT_RESOLUTION = 2e-3


def separated_gate_inputs(shape, k, score_func, margin=MARGIN, to_bf16=False):
    """``(logits, bias)`` whose biased scores have a designed top-k boundary margin.

    The op sorts ``score_activation(logits) + bias`` and the activation is bounded, so the bias is
    solved for any target score: ``bias = target - activation(logits)``. Per token: ``k`` winners on
    a descending ladder from ``2 * margin``, the rest scattered below zero, the row shuffled.
    """
    *lead, total_experts = shape
    rows = int(torch.tensor(lead).prod().item())

    selected = margin * torch.arange(k, 0, -1, dtype=torch.float32) + margin  # (k+1)*m .. 2*m
    rejected = -margin * (0.5 + 3.5 * torch.rand(rows, total_experts - k))  # -0.5*m .. -4*m
    targets = torch.cat([selected.expand(rows, k), rejected], dim=1)

    shuffle = torch.argsort(torch.rand(rows, total_experts), dim=1)
    targets = torch.gather(targets, 1, shuffle).reshape(shape)

    # The logits need only be non-degenerate, not distinct: the biased score is pinned by
    # ``targets`` and the activation is cancelled out exactly. A plain uniform draw also keeps the
    # fixture usable at widths where the bf16 grid cannot supply ``total_experts`` distinct values
    # -- ``distinct_logits`` raises above ~384 experts in bf16.
    logits = torch.empty(shape, dtype=torch.float32).uniform_(-6.0, 6.0)
    if to_bf16:
        logits = logits.to(torch.bfloat16).float()
    # Solve the bias against the already-quantized logits, so only the bias's own rounding can
    # move a biased score away from its target.
    bias = targets - score_activation(logits, score_func)
    if to_bf16:
        bias = bias.to(torch.bfloat16).float()
    return logits, bias


def realistic_gate_inputs(shape, score_func, to_bf16=False):
    """The same input distribution ``test_moe_grouped_topk.py`` uses: distinct logits plus a
    normal bias. Crowded near the selection boundary, which is where a comparator defect is most
    likely to show -- safe to use for the device-vs-device differential, which needs no reference.
    """
    del score_func  # kept for symmetry with separated_gate_inputs
    gen_dtype = torch.bfloat16 if to_bf16 else torch.float32
    logits = distinct_logits(shape, dtype=gen_dtype).float()
    bias = torch.randn(*shape, dtype=torch.float32)
    if to_bf16:
        bias = bias.to(torch.bfloat16).float()
    return logits, bias


def boundary_strict(biased, k, resolution=SORT_RESOLUTION):
    """Tokens whose k-th and (k+1)-th largest biased scores are more than ``resolution`` apart, so
    the top-k set cannot depend on the tie-break policy. Keys closer than ``resolution`` are ties as
    far as the network is concerned, even when fp32 can tell them apart."""
    top = torch.topk(biased, k + 1, dim=-1).values
    scale = top[..., k - 1].abs().clamp(min=1e-6)
    return (top[..., k - 1] - top[..., k]) > resolution * scale


def topk_strict(biased, k, resolution=SORT_RESOLUTION):
    """Tokens whose top-k biased scores are pairwise separated beyond the network's resolution, so
    their *order* is determined too -- the precondition for comparing the returned score vectors
    position by position.
    """
    top = torch.topk(biased, k, dim=-1).values
    scale = top[..., 1:].abs().clamp(min=1e-6)
    return ((top[..., :-1] - top[..., 1:]) > resolution * scale).all(dim=-1)


def run_gate(device, logits, bias, routing, score_func, stable_sort, input_dtype, num_real=None):
    """One moe_grouped_topk call; returns ``(weights, indices, biased_scores)`` trimmed to the logical
    shape. ``biased_scores`` is the op's own debug output (the tensor the top-k sorted), so the
    tie-freedom preconditions are evaluated against what the hardware saw."""
    n_groups, total_experts, summed, topk_groups, k, route_scale = routing
    num_batches, batch_size, seq_len, _ = logits.shape

    dev_logits = ttnn.from_torch(logits, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    dev_bias = ttnn.from_torch(bias, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    dev_biased = ttnn.from_torch(torch.zeros_like(logits), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    padding_config = build_padding_config(device, num_real) if num_real is not None else None

    weights, indices = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
        dev_logits,
        dev_bias,
        n_groups=n_groups,
        summed_experts_per_group=summed,
        topk_groups=topk_groups,
        n_activated_experts=k,
        route_scale=route_scale,
        stable_sort=stable_sort,
        epsilon=EPSILON,
        score_func=score_func,
        padding_config=padding_config,
        biased_scores=dev_biased,
    )

    return (
        ttnn.to_torch(weights)[:num_batches, :batch_size, :seq_len, :k],
        ttnn.to_torch(indices)[:num_batches, :batch_size, :seq_len, :k],
        ttnn.to_torch(dev_biased)[:num_batches, :batch_size, :seq_len, :total_experts].float(),
    )


def assert_same_selection(lhs_indices, rhs_indices, k, mask, lhs_name, rhs_name):
    """Assert two index tensors pick the same expert SET per token, over ``mask`` tokens only.

    Set comparison rather than element-wise: MoE sums the selected experts, so the order of the k
    slots carries no meaning downstream, and reporting an ordering difference as a selection
    failure would misdirect the reader.
    """
    lhs = lhs_indices.reshape(-1, k).long().sort(dim=-1).values
    rhs = rhs_indices.reshape(-1, k).long().sort(dim=-1).values
    flat_mask = mask.reshape(-1)

    differing = (lhs != rhs).any(dim=-1) & flat_mask
    if not differing.any():
        return

    first = int(differing.nonzero()[0])
    lhs_row, rhs_row = set(lhs[first].tolist()), set(rhs[first].tolist())
    raise AssertionError(
        f"{int(differing.sum())}/{int(flat_mask.sum())} tie-free tokens select a different expert "
        f"set under {lhs_name} vs {rhs_name}. First at token {first}: "
        f"{lhs_name}-only={sorted(lhs_row - rhs_row)}, {rhs_name}-only={sorted(rhs_row - lhs_row)}"
    )


@pytest.mark.parametrize("input_dtype", INPUT_DTYPES, ids=INPUT_DTYPE_IDS)
@pytest.mark.parametrize("score_func", SCORE_FUNCS)
@pytest.mark.parametrize("routing", DIFFERENTIAL_CONFIGS, ids=DIFFERENTIAL_CONFIG_IDS)
def test_stable_matches_unstable(device, routing, score_func, input_dtype):
    """stable_sort=True and stable_sort=False must agree exactly wherever keys are distinct: the two
    networks are the same function there. Realistic input distribution; the tie-freedom precondition
    is evaluated per token against the op's own biased_scores."""
    torch.manual_seed(42)
    _, total_experts, _, _, k, _ = routing
    shape = (1, 1, SEQ_LEN, total_experts)
    to_bf16 = input_dtype == ttnn.bfloat16

    logits, bias = realistic_gate_inputs(shape, score_func, to_bf16=to_bf16)

    w_stable, idx_stable, biased_stable = run_gate(device, logits, bias, routing, score_func, True, input_dtype)
    w_unstable, idx_unstable, biased_unstable = run_gate(device, logits, bias, routing, score_func, False, input_dtype)

    # The activation + bias stage is upstream of the sort and cannot depend on stable_sort, so any
    # difference here would mean the two runs never saw the same keys -- which would invalidate
    # everything below rather than reveal a top-k bug.
    assert torch.equal(biased_stable, biased_unstable), (
        "biased_scores differ between the stable and unstable runs; the two networks were not "
        "given identical keys, so this comparison says nothing about top-k"
    )

    set_mask = boundary_strict(biased_stable, k)
    order_mask = topk_strict(biased_stable, k)
    tokens = set_mask.numel()
    logger.info(
        f"tie-free tokens: boundary {int(set_mask.sum())}/{tokens}, " f"full top-{k} {int(order_mask.sum())}/{tokens}"
    )
    # Guard against the input generator silently degenerating (e.g. bf16 quantization collapsing
    # the candidate grid) and leaving nothing meaningful to assert on.
    assert set_mask.float().mean() > 0.5, (
        f"only {int(set_mask.sum())}/{tokens} tokens have a strict top-{k} boundary; the input "
        "distribution is too tied for this comparison to mean anything"
    )

    assert_index_domain(idx_stable, k, total_experts)
    assert_index_domain(idx_unstable, k, total_experts)

    assert_same_selection(idx_stable, idx_unstable, k, set_mask, "stable", "unstable")

    # Where the set AND the order are both determined, the returned score vectors must match
    # position for position -- so this also catches an ordering-only regression. Both conditions
    # are required: topk_strict alone separates ranks 1..k from each other but says nothing about
    # rank k vs k+1, so on its own it would admit tokens whose *set* is legitimately ambiguous and
    # blame the resulting weight difference on ordering.
    ordered = (order_mask & set_mask).reshape(-1)
    w_s = w_stable.reshape(-1, k)[ordered].float()
    w_u = w_unstable.reshape(-1, k)[ordered].float()
    if not torch.equal(w_s, w_u):
        # Same set, same order, same keys: the gathered scores should be bit-identical. Report how
        # far off they are rather than just that they differ.
        worst = (w_s - w_u).abs().max().item()
        passed, pcc = comp_pcc(w_s, w_u, pcc=0.9999)
        assert passed, (
            f"stable vs unstable weights differ on fully-determined tokens: PCC {pcc:.6f} < 0.9999, "
            f"largest absolute difference {worst:.3e}"
        )


@pytest.mark.parametrize("input_dtype", INPUT_DTYPES, ids=INPUT_DTYPE_IDS)
@pytest.mark.parametrize("score_func", SCORE_FUNCS)
@pytest.mark.parametrize("total_experts", WIDTH_SWEEP, ids=WIDTH_SWEEP_IDS)
def test_stable_exact_vs_golden(device, total_experts, score_func, input_dtype):
    """Exact expert-set match against torch, swept over the blocks::topk chain depth.

    Single-group routing only (``n_groups == 1``): that is the path Kimi takes, and it is the one
    whose expert width is free to sweep -- the grouped path is pinned to 256 experts. The designed
    boundary margin means recall is required to be exactly 1.0 rather than merely above a bar.
    """
    torch.manual_seed(42)
    k = 8
    routing = (1, total_experts, 1, 1, k, 2.827)
    shape = (1, 1, SEQ_LEN, total_experts)
    to_bf16 = input_dtype == ttnn.bfloat16

    logits, bias = separated_gate_inputs(shape, k, score_func, to_bf16=to_bf16)
    ref_indices, ref_weights = grouped_gate_golden_act(logits, bias, routing[5], EPSILON, 1, 1, 1, k, score_func)

    weights, indices, biased = run_gate(device, logits, bias, routing, score_func, True, input_dtype)

    # Self-check the fixture: the margin must have survived the device's own activation and any
    # bf16 quantization, or the "no legitimate ambiguity" premise does not hold.
    top = torch.topk(biased, k + 1, dim=-1).values
    gap = (top[..., k - 1] - top[..., k]).min().item()
    logger.info(f"e{total_experts} ({total_experts // 32} tiles): narrowest device boundary gap {gap:.4f}")
    assert gap > MARGIN / 2, (
        f"designed boundary margin did not survive to the device: narrowest gap {gap:.4f} "
        f"<= {MARGIN / 2}; an exact comparison is not justified"
    )
    assert topk_strict(biased, k).all(), "designed top-k is not strictly ordered on device"

    assert_index_domain(indices, k, total_experts)
    all_tokens = torch.ones(biased.shape[:-1], dtype=torch.bool)
    assert_same_selection(indices, ref_indices, k, all_tokens, "device", "torch")
    assert_gate_output(
        indices,
        weights,
        ref_indices,
        ref_weights,
        k,
        total_experts,
        num_real=0,
        apply_padding=False,
        exact_recall=True,
        pcc_threshold=0.999,
    )


@pytest.mark.parametrize("input_dtype", INPUT_DTYPES, ids=INPUT_DTYPE_IDS)
@pytest.mark.parametrize("stable_sort", [True, False], ids=["stable", "unstable"])
@pytest.mark.parametrize("padded_percent", [0, 50], ids=["pad0", "pad50"])
@pytest.mark.parametrize("routing", ROUTING_CONFIGS, ids=ROUTING_CONFIG_IDS)
def test_index_domain(device, routing, padded_percent, stable_sort, input_dtype):
    """Index-domain invariants, with and without padding, on both networks. Reference-free, so it
    runs on the realistic input distribution without any tie caveat."""
    torch.manual_seed(7)
    _, total_experts, _, _, k, _ = routing
    shape = (1, 1, SEQ_LEN, total_experts)

    logits, bias = realistic_gate_inputs(shape, score_func="sigmoid", to_bf16=input_dtype == ttnn.bfloat16)

    total_tokens = SEQ_LEN
    num_real = total_tokens - int(total_tokens * padded_percent / 100)
    apply_padding = 0 < num_real < total_tokens

    _, indices, _ = run_gate(
        device,
        logits,
        bias,
        routing,
        "sigmoid",
        stable_sort,
        input_dtype,
        num_real=num_real if apply_padding else None,
    )

    assert_index_domain(indices, k, total_experts, num_real=num_real, apply_padding=apply_padding)
