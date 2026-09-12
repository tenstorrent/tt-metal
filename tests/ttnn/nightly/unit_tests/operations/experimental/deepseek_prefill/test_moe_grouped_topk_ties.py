# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tie-ORDER coverage for moe_grouped_topk's single-group path.

Every input here plants EXACT ties: experts sharing a tie class are given the same logit and the
same bias, so their biased scores are bitwise equal. Each test asserts the returned indices
element-wise against a stable-argsort golden, with no tolerance.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    assert_index_domain,
    assert_indices_exact,
    build_padding_config,
    grouped_gate_golden_act,
    score_activation,
)

EPSILON = 1e-20

# (n_groups, total_experts, summed_experts_per_group, topk_groups, n_activated_experts, route_scale)
KIMI = (1, 384, 1, 1, 8, 2.827)
SCORE_FUNC = "sigmoid"  # Kimi's router affinity

# bf16 only: production feeds bf16 and the op upcasts internally, and the fp32 rows of this
# directory are pruned on Blackhole in CI, so an fp32-only case would silently not run there.
INPUT_DTYPE = ttnn.bfloat16

TOKENS = 32  # one tile of rows; every row can carry a different tie layout

# Per-class logit and bias both DECREASE with the class id, so the biased score is monotonic in it
# and no bias progression can invert the intended order. Every value below has at most 3 mantissa
# bits, so it survives the bf16 round-trip exactly and a shared class keeps a bitwise-equal score.
LEVEL_LOGIT_BASE, LEVEL_LOGIT_STEP = 2.0, 0.75
LEVEL_BIAS_STEP = 0.25

# Smallest gap the construction must leave between adjacent classes, in biased-score units. The
# realised gap is ~0.28; the datapath's effective comparison resolution is ~8e-4 relative, so this
# leaves two orders of magnitude of headroom and only exact ties are ties.
MIN_CLASS_GAP = 0.05


def levels_to_inputs(levels, num_levels):
    """Map a per-expert tie-class id to (logits, bias). Same class -> identical logit AND identical
    bias -> bitwise-equal biased score. Class 0 is the highest-scoring."""
    logit_of = torch.tensor([LEVEL_LOGIT_BASE - j * LEVEL_LOGIT_STEP for j in range(num_levels)], dtype=torch.float32)
    bias_of = torch.tensor([-j * LEVEL_BIAS_STEP for j in range(num_levels)], dtype=torch.float32)

    lv = torch.as_tensor(levels, dtype=torch.long)
    logits = logit_of[lv].reshape(1, 1, *lv.shape)
    bias = bias_of[lv].reshape(1, 1, *lv.shape)
    return logits, bias


def assert_ties_landed(biased, levels, context=""):
    """Precondition on the device's OWN biased scores: bitwise equal within a class, and separated
    by MIN_CLASS_GAP between classes. Without it a tie test can pass or fail for the wrong reason."""
    lv = torch.as_tensor(levels, dtype=torch.long)
    for row in range(biased.shape[-2]):
        row_biased = biased[0, 0, row]
        row_levels = lv[row] if lv.dim() == 2 else lv
        class_value = {}
        for j in row_levels.unique().tolist():
            members = row_biased[row_levels == j]
            assert torch.equal(
                members, members[:1].expand_as(members)
            ), f"{context}row {row} class {j} not bitwise tied on device: {members.unique().tolist()[:4]}"
            class_value[j] = members[0].item()

        ordered = [class_value[j] for j in sorted(class_value)]
        gaps = [a - b for a, b in zip(ordered, ordered[1:])]
        assert all(
            g >= MIN_CLASS_GAP for g in gaps
        ), f"{context}row {row} classes not separated / not descending: values {ordered} gaps {gaps}"


def run_gate(device, logits, bias, stable_sort, num_real=None):
    """One moe_grouped_topk call. Returns (weights, indices, biased_scores) trimmed to shape."""
    n_groups, total_experts, summed, topk_groups, k, route_scale = KIMI
    seq_len = logits.shape[-2]

    dev_logits = ttnn.from_torch(logits, dtype=INPUT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    dev_bias = ttnn.from_torch(bias, dtype=INPUT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
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
        score_func=SCORE_FUNC,
        padding_config=padding_config,
        biased_scores=dev_biased,
    )
    return (
        ttnn.to_torch(weights)[:1, :1, :seq_len, :k],
        ttnn.to_torch(indices)[:1, :1, :seq_len, :k],
        ttnn.to_torch(dev_biased)[:1, :1, :seq_len, :total_experts].float(),
    )


def stable_golden(logits, bias):
    """Reference indices under the stable contract, from the same bf16-quantised values the device
    sees (ttnn.from_torch(bf16) rounds, so the golden must round first or the two can disagree)."""
    q_logits = logits.to(torch.bfloat16).float()
    q_bias = bias.to(torch.bfloat16).float()
    n_groups, _, summed, topk_groups, k, route_scale = KIMI
    indices, _ = grouped_gate_golden_act(
        q_logits, q_bias, route_scale, EPSILON, n_groups, summed, topk_groups, k, SCORE_FUNC, stable=True
    )
    return indices


# --------------------------------------------------------------------------------------------
# Tie patterns. Each returns a per-expert tie-class id array of shape [TOKENS, total_experts].
# --------------------------------------------------------------------------------------------

TOTAL_EXPERTS = KIMI[1]
K = KIMI[4]


def pattern_all_equal():
    """Every expert in one tie class. Stable must return 0..k-1 in order."""
    return torch.zeros(TOKENS, TOTAL_EXPERTS, dtype=torch.long), 1


def pattern_boundary_tie():
    """Six strictly ordered winners, then a tie class straddling the k cut, then the rest.

    Slots 0-5 are forced; slots 6 and 7 must be the two LOWEST-INDEXED members of the tied class,
    which is the decision that actually changes routing in production.
    """
    levels = torch.full((TOKENS, TOTAL_EXPERTS), 7, dtype=torch.long)  # class 7 = the floor
    for row in range(TOKENS):
        g = torch.Generator().manual_seed(1000 + row)
        perm = torch.randperm(TOTAL_EXPERTS, generator=g)
        levels[row, perm[:6]] = torch.arange(6)  # six singleton winners, classes 0..5
        levels[row, perm[6:16]] = 6  # ten-member tie class at the boundary
    return levels, 8


def pattern_chain_spanning():
    """One tie class of 12, deliberately spread across the first and last of the 12 width tiles.

    The insertion chain merges 11 times; a bitonic network is not stable even when every comparator
    preserves order on equal keys, and that only shows when tied elements meet across non-adjacent
    positions after several merges. k=8 of the 12 must come back in ascending index order.
    """
    members = [0, 3, 31, 32, 160, 200, 351, 352, 370, 380, 382, 383]
    levels = torch.ones(TOKENS, TOTAL_EXPERTS, dtype=torch.long)  # class 1 = the floor
    levels[:, members] = 0
    return levels, 2


PATTERNS = {
    "all_equal": pattern_all_equal,
    "boundary_tie": pattern_boundary_tie,
    "chain_spanning": pattern_chain_spanning,
}


def build(pattern_name):
    levels, num_levels = PATTERNS[pattern_name]()
    logits, bias = levels_to_inputs(levels, num_levels)
    return levels, logits, bias


# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "pattern",
    [
        "all_equal",
        pytest.param("boundary_tie", marks=BROKEN_ON_MAIN),
        pytest.param("chain_spanning", marks=BROKEN_ON_MAIN),
    ],
)
def test_tie_order_matches_stable_golden(device, pattern):
    """Exact index parity against a stable-argsort golden, on exact ties."""
    levels, logits, bias = build(pattern)
    _, indices, biased = run_gate(device, logits, bias, stable_sort=True)

    assert_ties_landed(biased, levels, context=f"[{pattern}] ")
    assert_index_domain(indices, K, TOTAL_EXPERTS)
    assert_indices_exact(indices, stable_golden(logits, bias), K, context=f"[{pattern}] ")


def test_all_equal_row_is_identity(device):
    """The strongest single assertion: an all-equal row must return exactly 0..k-1."""
    levels, logits, bias = build("all_equal")
    _, indices, biased = run_gate(device, logits, bias, stable_sort=True)

    assert_ties_landed(biased, levels, context="[all_equal] ")
    expected = torch.arange(K).expand(1, 1, TOKENS, K)
    assert_indices_exact(indices, expected, K, context="[all_equal] ")


def test_stable_and_unstable_differ_on_ties(device):
    """The flag must change something on an exact tie. Without this every other assertion here
    would still pass if stable_sort were wired to nothing."""
    levels, logits, bias = build("chain_spanning")
    _, stable_idx, biased = run_gate(device, logits, bias, stable_sort=True)
    _, unstable_idx, _ = run_gate(device, logits, bias, stable_sort=False)

    assert_ties_landed(biased, levels, context="[chain_spanning] ")
    differing = int((stable_idx != unstable_idx).any(dim=-1).sum())
    logger.info(f"[chain_spanning] stable vs unstable differ on {differing}/{TOKENS} tokens")
    assert differing > 0, "stable_sort=True and False returned identical indices on an exact tie"


def test_tie_order_is_deterministic(device):
    """Repeat the boundary tie; the tie-break must be a function of the input, not of the run."""
    _, logits, bias = build("boundary_tie")
    _, first, _ = run_gate(device, logits, bias, stable_sort=True)
    for i in range(1, 4):
        _, again, _ = run_gate(device, logits, bias, stable_sort=True)
        assert torch.equal(first, again), f"run {i} differs from run 0"


@pytest.mark.xfail(reason="stable tie order across the merge chain, #33492", strict=True)
def test_tie_order_under_padding(device):
    """Ties plus right-padding: padded rows go to the sentinel, real rows keep stable order."""
    num_real = TOKENS // 2
    levels, logits, bias = build("boundary_tie")
    _, indices, biased = run_gate(device, logits, bias, stable_sort=True, num_real=num_real)

    assert_ties_landed(biased, levels, context="[padded] ")
    assert_index_domain(indices, K, TOTAL_EXPERTS, num_real=num_real, apply_padding=True)
    golden = stable_golden(logits, bias)[:, :, :num_real]
    assert_indices_exact(indices[:, :, :num_real], golden, K, context="[padded] ")


def negative_zeros(t):
    return ((t == 0) & torch.signbit(t)).sum().item()


def test_negative_zero_is_unreachable():
    """The -0.0 canonicalisation sweep compiles in only under stable + fp32 dest, i.e. exactly this
    op's configuration. Both score functions are non-negative and IEEE gives x + (-x) = +0.0, so no
    input can put -0.0 into the biased scores. If this ever fails the sweep is live and needs a
    device test."""
    logits = torch.tensor([-1e30, -80.0, -1.0, 0.0, 1.0, 80.0, 1e30], dtype=torch.float32)
    for func in ("sigmoid", "sqrtsoftplus"):
        scores = score_activation(logits, func)
        assert negative_zeros(scores) == 0, f"{func} emitted -0.0"

        for name, bias in (
            ("exact_cancel", -scores),
            ("neg_zero_bias", torch.full_like(scores, -0.0)),
            ("tiny_neg", torch.full_like(scores, -1e-45) - scores),
            ("large_neg", torch.full_like(scores, -1.0)),
        ):
            biased = scores + bias
            assert negative_zeros(biased) == 0, f"{func}/{name} produced -0.0 in the biased scores"
