# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tie-ORDER coverage for moe_grouped_topk: the single-group path (Kimi) and the grouped path (DeepSeek).

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
DEEPSEEK = (8, 256, 2, 4, 8, 2.5)  # the only grouped configuration the op accepts
GROUP_SIZE = 32  # one tile per group on the grouped path
SCORE_FUNC = "sigmoid"  # Kimi's router affinity

# Widths for the all-equal identity check. The comparator handles indices as 32-bit words, and a
# fault that reads only part of the word stays invisible while every index fits in that part: 256
# is the control that passes even then, 288 and 384 need bit 8, 544 needs bit 9. Only 384 is a
# production width.
WIDTH_SWEEP = [256, 288, 384, 544]

# The FPU keeps only 10 mantissa bits when it moves fp32 tiles, so the sort compares scores at that
# precision. Every sigmoid value for logits from 8 up truncates to the same number, so experts with
# logits in that range tie on the score alone. Values below are bf16-exact.
PLATEAU_LOGITS = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 15.5, 9.5]
COMPARED_MANTISSA_BITS = 10

# Random-input test: enough rows that many of them have a tie at the k-th slot by chance.
NATURAL_TOKENS = 640
MIN_TIES_AT_CUT = 32

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


def run_gate(device, logits, bias, stable_sort, num_real=None, routing=KIMI):
    """One moe_grouped_topk call. Returns (weights, indices, biased_scores) trimmed to shape."""
    n_groups, total_experts, summed, topk_groups, k, route_scale = routing
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


def stable_golden(logits, bias, routing=KIMI):
    """Reference indices under the stable contract, from the same bf16-quantised values the device
    sees (ttnn.from_torch(bf16) rounds, so the golden must round first or the two can disagree)."""
    q_logits = logits.to(torch.bfloat16).float()
    q_bias = bias.to(torch.bfloat16).float()
    n_groups, _, summed, topk_groups, k, route_scale = routing
    indices, _ = grouped_gate_golden_act(
        q_logits, q_bias, route_scale, EPSILON, n_groups, summed, topk_groups, k, SCORE_FUNC, stable=True
    )
    return indices


def as_compared(biased):
    """The scores as the sort sees them: fp32 with the low 13 mantissa bits dropped. This reproduces
    the device's order exactly; using the full fp32 values does not."""
    drop = 23 - COMPARED_MANTISSA_BITS
    keep = torch.tensor(-(1 << drop), dtype=torch.int32)
    return (biased.contiguous().view(torch.int32) & keep).view(torch.float32)


def device_stable_golden(biased, k):
    """Stable argsort of the scores the device computed, at the precision the sort compares them.
    Use this when a tie exists only on device; torch's fp32 scores would not tie."""
    return torch.argsort(as_compared(biased), dim=-1, descending=True, stable=True)[..., :k]


def ties_at_cut(biased, k):
    """Number of rows where the k-th and (k+1)-th largest scores are equal as the sort sees them."""
    top = as_compared(biased).reshape(-1, biased.shape[-1]).sort(dim=-1, descending=True).values
    return int((top[:, k - 1] == top[:, k]).sum())


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


def pattern_grouped_ties():
    """DeepSeek routing with exact ties at both places the grouped path decides by order.

    Five groups share the top group sum (one class-0 anchor plus three class-1 experts each) and only
    four may win, so stable keeps the four lowest group ids and drops group 6. Among the winners four
    anchors and twelve class-1 experts tie across four tiles for eight slots, so the last four slots
    must be the lowest-indexed class-1 members, which sit in three different groups.
    """
    levels = torch.full((TOKENS, DEEPSEEK[1]), 2, dtype=torch.long)  # class 2 = the floor
    for g in (0, 2, 3, 5, 6):
        base = g * GROUP_SIZE
        levels[:, base + 9] = 0
        levels[:, [base + 1, base + 17, base + 31]] = 1
    return levels, 3


# Anchor class per group: the four winners in sum order are groups 6, 1, 5, 3, which is not their id
# order, so the final top-k sees its tiles out of index order. The other four groups lose clearly.
INVERTED_GROUP_ANCHORS = {6: 0, 1: 1, 5: 2, 3: 3, 0: 4, 2: 5, 4: 6, 7: 7}
GROUP_FLOOR_CLASS = 8


def pattern_grouped_inverted_order():
    """DeepSeek routing where the winning groups arrive at the final top-k in an order that is not
    their id order, and the last slots are decided by a tie that spans those groups.

    Every group has one anchor expert (position 9) of a distinct class, and all other experts share
    the floor class, so each group sum is anchor + floor and the sums are at least one class gap
    apart: the cut is unambiguous, whatever the datapath rounds. The four anchors take the first four
    slots; the remaining four slots are contested by 124 tied floor experts across the winning groups
    and must go to the lowest indices, which sit in group 1 even though its tile is not the first."""
    levels = torch.full((TOKENS, DEEPSEEK[1]), GROUP_FLOOR_CLASS, dtype=torch.long)
    for g, cls in INVERTED_GROUP_ANCHORS.items():
        levels[:, g * GROUP_SIZE + 9] = cls
    return levels, GROUP_FLOOR_CLASS + 1


# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pattern", list(PATTERNS))
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
    """The flag must change something on exact ties. Without this every other assertion here would
    still pass if stable_sort were wired to nothing. Checked across all patterns: on some the unstable
    network happens to return index order, so no single pattern is a safe witness."""
    differing = {}
    for pattern in PATTERNS:
        levels, logits, bias = build(pattern)
        _, stable_idx, biased = run_gate(device, logits, bias, stable_sort=True)
        _, unstable_idx, _ = run_gate(device, logits, bias, stable_sort=False)
        assert_ties_landed(biased, levels, context=f"[{pattern}] ")
        differing[pattern] = int((stable_idx != unstable_idx).any(dim=-1).sum())
    logger.info(f"stable vs unstable differing tokens per pattern: {differing}")
    assert any(differing.values()), f"stable_sort=True and False identical on every tie pattern: {differing}"


def test_tie_order_is_deterministic(device):
    """Repeat the boundary tie; the tie-break must be a function of the input, not of the run."""
    _, logits, bias = build("boundary_tie")
    _, first, _ = run_gate(device, logits, bias, stable_sort=True)
    for i in range(1, 4):
        _, again, _ = run_gate(device, logits, bias, stable_sort=True)
        assert torch.equal(first, again), f"run {i} differs from run 0"


def test_tie_order_under_padding(device):
    """Ties plus right-padding: padded rows go to the sentinel, real rows keep stable order."""
    num_real = TOKENS // 2
    levels, logits, bias = build("boundary_tie")
    _, indices, biased = run_gate(device, logits, bias, stable_sort=True, num_real=num_real)

    assert_ties_landed(biased, levels, context="[padded] ")
    assert_index_domain(indices, K, TOTAL_EXPERTS, num_real=num_real, apply_padding=True)
    golden = stable_golden(logits, bias)[:, :, :num_real]
    assert_indices_exact(indices[:, :, :num_real], golden, K, context="[padded] ")


@pytest.mark.parametrize("total_experts", WIDTH_SWEEP)
def test_all_equal_identity_across_widths(device, total_experts):
    """0..k-1 from an all-equal row at every width. A tie-break that sees only the low byte of an
    index still passes at 256 and fails from 288 up; see WIDTH_SWEEP."""
    routing = (1, total_experts, 1, 1, K, KIMI[5])
    levels = torch.zeros(TOKENS, total_experts, dtype=torch.long)
    logits, bias = levels_to_inputs(levels, 1)
    _, indices, biased = run_gate(device, logits, bias, stable_sort=True, routing=routing)

    assert_ties_landed(biased, levels, context=f"[all_equal/{total_experts}] ")
    expected = torch.arange(K).expand(1, 1, TOKENS, K)
    assert_indices_exact(indices, expected, K, context=f"[all_equal/{total_experts}] ")


def test_grouped_tie_order_matches_stable_golden(device):
    """The grouped path's own comparator blocks, group selection and the cross-group expert top-k, on
    exact ties. The single-group tests above never reach them. Before stable_sort existed (#33492) the
    cut between the five tied groups dropped group 5 instead of group 6, and the tied experts came back
    out of index order."""
    levels, num_levels = pattern_grouped_ties()
    logits, bias = levels_to_inputs(levels, num_levels)
    golden = stable_golden(logits, bias, routing=DEEPSEEK)
    # Construction check: the four anchors, then the four lowest class-1 members, nothing from group 6.
    assert golden[0, 0, 0].tolist() == [9, 73, 105, 169, 1, 17, 31, 65]

    _, indices, biased = run_gate(device, logits, bias, stable_sort=True, routing=DEEPSEEK)
    assert_ties_landed(biased, levels, context="[grouped] ")
    assert_index_domain(indices, K, DEEPSEEK[1])
    assert_indices_exact(indices, golden, K, context="[grouped] ")


def test_grouped_inverted_group_order_ties(device):
    """The final top-k of the grouped path receives the winning groups ordered by group sum, not by
    group id. A tie-break that ranks candidates by their position in that chain instead of by their
    index passes every test above (there the sum order happens to be the id order) and fails here."""
    levels, num_levels = pattern_grouped_inverted_order()
    logits, bias = levels_to_inputs(levels, num_levels)
    golden = stable_golden(logits, bias, routing=DEEPSEEK)
    # Construction check: the anchors of groups 6, 1, 5, 3 in that order, then the four lowest
    # floor experts of the lowest-numbered winning group.
    assert golden[0, 0, 0].tolist() == [201, 41, 169, 105, 32, 33, 34, 35]

    _, indices, biased = run_gate(device, logits, bias, stable_sort=True, routing=DEEPSEEK)
    assert_ties_landed(biased, levels, context="[grouped inverted] ")
    assert_index_domain(indices, K, DEEPSEEK[1])
    assert_indices_exact(indices, golden, K, context="[grouped inverted] ")


# --------------------------------------------------------------------------------------------
# Ties as they happen in production: made by the device's sigmoid, not by identical inputs
# --------------------------------------------------------------------------------------------


def natural_inputs(seed):
    """Random bf16 logits and a per-expert bias with many repeated bf16 values, like Kimi's bias after
    the bf16 cast. No ties are planted."""
    g = torch.Generator().manual_seed(seed)
    logits = (torch.randn(1, 1, NATURAL_TOKENS, TOTAL_EXPERTS, generator=g) * 3.0).to(torch.bfloat16).float()
    bias = (torch.randn(TOTAL_EXPERTS, generator=g) * 0.03).to(torch.bfloat16).float()
    return logits, bias.expand(1, 1, NATURAL_TOKENS, TOTAL_EXPERTS).clone()


def test_natural_inputs_follow_stable_order(device):
    """Random inputs, every row checked. The golden is a stable argsort of the scores the device
    computed, at the precision the sort compares them, so only the sort is under test. With 10
    mantissa bits, ties at the k-th slot happen by themselves in about a fifth of the rows."""
    logits, bias = natural_inputs(seed=7)
    _, indices, biased = run_gate(device, logits, bias, stable_sort=True)

    n_ties = ties_at_cut(biased, K)
    logger.info(f"[natural] rows with a tie at slot {K}: {n_ties}/{NATURAL_TOKENS}")
    assert n_ties >= MIN_TIES_AT_CUT, f"only {n_ties} rows have a tie at the cut; the test would prove little"
    assert_index_domain(indices, K, TOTAL_EXPERTS)
    assert_indices_exact(indices, device_stable_golden(biased, K), K, context="[natural] ")


def pattern_plateau_top():
    """Nine experts on the sigmoid plateau, all with bias 0, spread over the tiles and across index
    256. They tie on the score; stable keeps the eight lowest indices. Everyone else is far below."""
    members = [5, 40, 100, 200, 255, 256, 300, 351, 383]
    logits = torch.full((1, 1, TOKENS, TOTAL_EXPERTS), -2.0)
    bias = torch.zeros(1, 1, TOKENS, TOTAL_EXPERTS)
    logits[..., members] = torch.tensor(PLATEAU_LOGITS)
    levels = torch.ones(TOKENS, TOTAL_EXPERTS, dtype=torch.long)
    levels[:, members] = 0
    return levels, logits, bias, sorted(members)[:K]


def pattern_plateau_at_cut():
    """Six clear winners (same logit, different bias), then four plateau experts tied for the last
    two slots, then the floor. Stable takes the two lowest-indexed plateau experts, 200 and 255."""
    winners = [10, 70, 130, 190, 250, 310]
    plateau = [200, 255, 256, 300]
    logits = torch.full((1, 1, TOKENS, TOTAL_EXPERTS), -2.0)
    bias = torch.zeros(1, 1, TOKENS, TOTAL_EXPERTS)
    logits[..., winners] = 2.0
    bias[..., winners] = torch.tensor([0.875, 0.75, 0.625, 0.5, 0.375, 0.25])
    logits[..., plateau] = torch.tensor(PLATEAU_LOGITS[:4])
    levels = torch.full((TOKENS, TOTAL_EXPERTS), 7, dtype=torch.long)
    levels[:, winners] = torch.arange(6)
    levels[:, plateau] = 6
    return levels, logits, bias, winners + [200, 255]


ACTIVATION_PATTERNS = {"plateau_top": pattern_plateau_top, "plateau_at_cut": pattern_plateau_at_cut}


@pytest.mark.parametrize("pattern", list(ACTIVATION_PATTERNS))
def test_ties_made_by_the_activation(device, pattern):
    """Ties between experts with different logits, created by the 10-bit precision of the compared
    scores. This is how most ties arise in Kimi. The golden comes from the device's own scores; torch
    would not tie them."""
    levels, logits, bias, expected_row = ACTIVATION_PATTERNS[pattern]()
    _, indices, biased = run_gate(device, logits, bias, stable_sort=True)

    assert_ties_landed(biased, levels, context=f"[{pattern}] ")
    golden = device_stable_golden(biased, K)
    assert (
        golden[0, 0, 0].tolist() == expected_row
    ), f"[{pattern}] construction check failed: {golden[0, 0, 0].tolist()}"
    assert_index_domain(indices, K, TOTAL_EXPERTS)
    assert_indices_exact(indices, golden, K, context=f"[{pattern}] ")


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
