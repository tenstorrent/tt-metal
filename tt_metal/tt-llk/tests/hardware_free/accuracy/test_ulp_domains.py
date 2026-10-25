# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for ``helpers/ulp_domains.py``: what a ``domains`` list means, and
the rule that draws one from a cell's binade profile.

No device: the rule is pure, and a profile is a dict. The tensor side
(``ulp_sweep.binade_maxima``, ``domain_figures``) is held in test_ulp_sweep.py.
"""

import math

import pytest
from helpers.format_config import DataFormat
from helpers.ulp import ULP_FORMATS
from helpers.ulp_domains import (
    BINADE_OFFSET,
    DOMAIN_MIN_BIN_LANES,
    DOMAIN_MIN_LANES,
    Domain,
    Problem,
    binade,
    binade_edges,
    check_domains,
    contract_of,
    entries_of,
    largest_figure,
    partition,
    usable_budget_ceiling,
)
from helpers.ulp_testing import TANH_DOMAINS, problem_id, refused

INF = math.inf
BF16 = usable_budget_ceiling(DataFormat.Float16_b)


def _profile(worst_of, lanes=128):
    """A bfloat16-like profile: 254 binades a side, *lanes* present in each, the worst
    ranked lane from ``worst_of(k, sign)`` (``None`` for an unranked binade), and the
    single zero lane."""
    bins = {}
    for k in range(-126, 128):
        for sign in (1, -1):
            worst = worst_of(k, sign)
            index = sign * (k + BINADE_OFFSET)
            bins[index] = (
                lanes,
                0 if worst is None else lanes,
                -1 if worst is None else worst,
            )
    bins[0] = (1, 1, 0)
    return bins


def test_binades_order_as_the_values_do():
    assert binade(-0.3) < binade(-1e-30) < binade(0.0) < binade(1e-30) < binade(0.3)
    assert binade_edges(binade(0.3)) == (0.25, 0.5)
    assert binade_edges(binade(-0.3)) == (-0.5, -0.25)
    assert binade_edges(binade(1.0)) == (1.0, 2.0)
    assert binade(2.0**-126) == -126 + BINADE_OFFSET == 25  # bf16's smallest normal
    assert binade_edges(0) == (0.0, 0.0)


@pytest.mark.parametrize(
    "x", [-2.0, -1.5, -1.0, -0.75, -0.5, -(2.0**-149), 2.0**-149, 0.5, 1.0, 1.5, 2.0]
)
def test_a_binade_holds_its_lanes_as_an_entry_does(x):
    """``low <= x < high`` on both sides: -1 opens [-1, -0.5), as it opens an entry that
    starts at -1, so no binade's lanes straddle an entry's edge. The smallest subnormal
    still gets a binade of its own, apart from zero's."""
    low, high = binade_edges(binade(x))
    assert low <= x < high
    assert binade(x) != 0


def test_a_flat_profile_is_one_entry_over_the_whole_range():
    assert partition(_profile(lambda k, s: 1), BF16) == ((-INF, INF, True),)


def test_a_profile_splits_where_the_worst_lane_crosses_the_ceiling():
    """Approximate tanh: 10 steps below |x| = 1, 3 steps above; bf16's ceiling is 6."""
    spans = partition(_profile(lambda k, s: 10 if k < 0 else 3), BF16)
    assert spans == ((-INF, -1.0, True), (-1.0, 1.0, True), (1.0, INF, True))


def test_a_step_class_flickering_between_binades_is_one_entry():
    """The pack path rounds some binades and not others, so the worst lane alternates
    between 0 and 1 from one binade to the next on hundreds of cells. Both are within
    the ceiling, and the binary rule does not tell them apart."""
    assert partition(_profile(lambda k, s: k % 2), BF16) == ((-INF, INF, True),)


def test_an_unranked_region_is_its_own_entry_with_zero_for_an_edge():
    """`log` below zero: no lane is ranked. The edge between the two is where the
    swept values stop, which is written as 0 rather than 2**-126."""
    spans = partition(_profile(lambda k, s: None if s < 0 else 1), BF16)
    assert spans == ((-INF, 0.0, False), (0.0, INF, True))


def test_a_stray_ranked_lane_in_an_unranked_binade_is_noise():
    """erfinv's [1, 2) binade has one ranked lane, at x = 1: fewer than
    DOMAIN_MIN_BIN_LANES, so the binade is unranked and the list ends at 1."""

    def profile():
        bins = _profile(lambda k, s: 4 if k < 0 else None)
        for sign in (1, -1):
            bins[sign * BINADE_OFFSET] = (128, 1, 0)  # [1, 2) and [-2, -1): one lane
        return bins

    assert partition(profile(), BF16) == (
        (-INF, -1.0, False),
        (-1.0, 1.0, True),
        (1.0, INF, False),
    )


def test_a_narrow_gateable_island_folds_into_its_looser_neighbour_but_a_bad_one_stays():
    """One good binade (128 lanes, under DOMAIN_MIN_LANES) inside a bad range is not
    advertised on its own; one bad binade inside a good range always is."""
    assert DOMAIN_MIN_LANES > 128 >= DOMAIN_MIN_BIN_LANES
    good_island = partition(_profile(lambda k, s: 2 if (k, s) == (3, 1) else 10), BF16)
    assert good_island == ((-INF, INF, True),)
    bad_island = partition(_profile(lambda k, s: 10 if (k, s) == (3, 1) else 1), BF16)
    assert bad_island == ((-INF, 8.0, True), (8.0, 16.0, True), (16.0, INF, True))


def test_the_zero_lane_never_forms_an_entry():
    for spans in (
        partition(_profile(lambda k, s: 10 if k < 0 else 3), BF16),
        partition(_profile(lambda k, s: None if s < 0 else 1), BF16),
        partition({0: (1, 1, 0)}, BF16),
    ):
        assert all(low < high for low, high, _ in spans)
        assert any(low <= 0 < high for low, high, _ in spans)


def test_the_profile_of_a_strided_float32_input_partitions_the_same_way():
    """~129 samples per binade rather than 128 values: the rule reads lanes, not
    values, so nothing about it is bf16-specific."""
    spans = partition(
        _profile(lambda k, s: 500000 if k > 20 else 2, lanes=129),
        usable_budget_ceiling(DataFormat.Float32),
    )
    assert spans == (
        (-INF, -(2.0**21), True),
        (-(2.0**21), 2.0**21, True),
        (2.0**21, INF, True),
    )


def test_a_finer_step_class_splits_the_gateable_range(monkeypatch):
    """``DOMAIN_CLASS_BITS``: with 1, classes a factor of 2 apart (0 and 1 together)
    split the gateable range as well; a narrow class still folds into a looser
    neighbour."""
    import helpers.ulp_domains as ulp_domains

    monkeypatch.setattr(ulp_domains, "DOMAIN_CLASS_BITS", 1)
    lanes = 1024  # one binade per side is a domain of its own at DOMAIN_MIN_LANES
    spans = partition(_profile(lambda k, s: 4 if s > 0 else 1, lanes=lanes), BF16)
    assert spans == ((-INF, 0.0, True), (0.0, INF, True))
    narrow = partition(
        _profile(lambda k, s: 1 if (k, s) == (3, 1) else 4, lanes=128), BF16
    )
    assert narrow == ((-INF, INF, True),)


def _worst_of(figures):
    """*worst(low, high)* over a dict of ``(low, high) -> worst``."""
    return lambda low, high: figures.get((low, high))


def test_a_null_span_keeps_null_while_its_strays_fit_the_largest_figure():
    spans = ((-INF, 0.0, False), (0.0, 1.0, True), (1.0, INF, True))
    figures = {(-INF, 0.0): 2, (0.0, 1.0): 3, (1.0, INF): 1}
    assert entries_of(spans, _worst_of(figures)) == (
        Domain(-INF, 0, None),
        Domain(0, 1, 3),
        Domain(1, INF, 1),
    )


def test_the_null_span_whose_strays_are_worst_takes_a_figure():
    """The sweep holds a null entry's ranked lanes to the row's largest figure, so a
    list whose strays exceed it would fail the next run on identical output. Only the
    worst such span needs a figure: the others then fit the new largest."""
    spans = ((-INF, -1.0, False), (-1.0, 0.0, False), (0.0, INF, True))
    figures = {(-INF, -1.0): 40, (-1.0, 0.0): 1000, (0.0, INF): 3}
    assert entries_of(spans, _worst_of(figures)) == (
        Domain(-INF, -1, None),
        Domain(-1, 0, 1000),
        Domain(0, INF, 3),
    )


def test_spans_with_no_figure_draw_no_list():
    """Every binade ranks a few lanes: no span is ranked, and a list with no figure is
    one ``check_domains`` refuses when the registry loads. The cell gets a plain row."""
    assert entries_of(((-INF, INF, False),), lambda low, high: 1000) == ()
    assert entries_of(((-INF, 0.0, True), (0.0, INF, False)), lambda lo, hi: None) == ()


def test_a_well_formed_list_is_accepted():
    check_domains(TANH_DOMAINS)
    check_domains((Domain(-INF, 0), Domain(0, INF, 1)))


@pytest.mark.parametrize(
    "domains, problem",
    [
        ((), Problem.EMPTY_DOMAINS),
        ((Domain(-1, INF, 0),), Problem.NOT_COVERING),
        ((Domain(-INF, 1, 0),), Problem.NOT_COVERING),
        ((Domain(-INF, -1, 3), Domain(0, INF, 1)), Problem.GAP),
        ((Domain(-INF, 3, 1), Domain(3, INF, 1)), Problem.EDGE_NOT_POWER_OF_TWO),
        ((Domain(-INF, INF),), Problem.NO_FIGURE),
    ],
    ids=problem_id,
)
def test_a_malformed_list_is_refused(domains, problem):
    with refused(problem):
        check_domains(domains)


@pytest.mark.parametrize(
    "low, high, ulp, problem",
    [(1, 1, 0, Problem.EMPTY_RANGE), (-INF, INF, -1, Problem.BAD_NUMBER)],
    ids=problem_id,
)
def test_a_malformed_entry_is_refused(low, high, ulp, problem):
    with refused(problem):
        Domain(low, high, ulp)


def test_a_list_implies_its_whole_cell_contract():
    """Gated at its largest figure when every figure is within the ceiling."""
    assert contract_of(TANH_DOMAINS, BF16) == (False, 10)
    assert contract_of(TANH_DOMAINS[:1] + (Domain(-1, INF, 5),), BF16) == (True, 5)


def test_a_null_entry_is_held_to_the_largest_figure_of_its_list():
    """The one rule the sweep, ``diff`` and ``headroom`` read a null entry by."""
    assert largest_figure((Domain(-INF, 0, 2), Domain(0, 1), Domain(1, INF, 7))) == 7


def test_the_ceilings_are_the_rtol_they_replace():
    """``usable_budget_ceiling`` is the format's ``rtol`` in steps, rounded down."""
    assert [usable_budget_ceiling(fmt) for fmt in ULP_FORMATS] == [6.0, 51.0, 419430.0]
