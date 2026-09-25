# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the SFPU accuracy budget registry.

No kernel, no device: a table and a resolution rule. Both need guarding, because the rule
reduces a four-dimensional lookup to "most specific key wins" and a budget resolved from
the wrong key is a silently wrong gate, not an error -- and because a *widened* budget is
invisible to every device test, which it makes pass.

The resolution tests build their own small tables through :func:`resolve_contract` rather
than querying the live registry, so enrolling an op does not break them; only the tests
about enrolment touch the real one.
"""

import textwrap
from dataclasses import fields

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, MathOperation
from helpers.sfpu_accuracy_budget import (
    _BUDGET_KEY_TYPES,
    _SFPU_ACCURACY_BUDGET,
    DEFAULT,
    MEASURED_ARCH,
    TOLERANCE_CONTRACT,
    AccuracyContract,
    BudgetKey,
    Metric,
    _load_table,
    accuracy_contract,
    resolve_contract,
    validate_registry,
)
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
from helpers.ulp import ULP_FORMATS
from helpers.utils import passed_test

TILE_SIZE = DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM

F32 = BudgetKey(output_format=DataFormat.Float32)
BF16 = BudgetKey(output_format=DataFormat.Float16_b)


def _refuses(match, kind=ValueError):
    """The suite's ``expect_error`` fixture needs a device; these are host-only tests."""
    return pytest.raises(kind, match=match)  # allow-pytest.raises: host-only test


def _table(tmp_path, text):
    """*text* as a budget table on disk, loaded the way the real one is."""
    path = tmp_path / "budget.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return _load_table(path)


# ── The contract's own coherence ──────────────────────────────────────────────


def test_a_ulp_contract_needs_a_budget():
    with _refuses("needs max_ulp"):
        AccuracyContract(metric=Metric.ULP)


def test_a_ulp_contract_rejects_a_tolerance():
    """Both cannot apply, and an entry carrying both is a half-finished conversion that
    would read as deliberate."""
    with _refuses("silently ignored"):
        AccuracyContract(max_ulp=1, atol=0.13)


def test_a_tolerance_contract_rejects_a_budget():
    with _refuses("belong to the ulp metric"):
        AccuracyContract(metric=Metric.TOLERANCE, max_ulp=1)


def test_a_negative_budget_is_rejected():
    with _refuses("must not be negative"):
        AccuracyContract(max_ulp=-1)


@pytest.mark.parametrize("bogus", [True, 1.0, "3"], ids=repr)
def test_a_budget_that_is_not_an_int_is_rejected(bogus):
    """YAML 1.1 reads ``true`` as a bool -- and bool is an int, so it would pass the sign
    check and enforce a 1-step budget; ``1.0e+1`` lands as a float."""
    with _refuses("must be an int step count"):
        AccuracyContract(max_ulp=bogus)


@pytest.mark.parametrize("field", ["atol", "rtol", "near_zero_atol"], ids=str)
def test_a_negative_tolerance_field_is_refused(field):
    """``passed_test`` applies an override only when it is ``>= 0``, so a negative
    ``atol``/``rtol`` would read in the table as a declared tolerance and then silently
    run against the per-format default. A negative ``near_zero_atol`` is an inert floor.
    """
    metric = Metric.ULP if field == "near_zero_atol" else Metric.TOLERANCE
    kwargs = {"metric": metric, field: -0.001}
    if metric is Metric.ULP:
        kwargs["max_ulp"] = 1
    with _refuses("must not be negative"):
        AccuracyContract(**kwargs)


@pytest.mark.parametrize("bogus", [True, float("nan"), float("inf")], ids=repr)
def test_a_tolerance_field_must_be_a_finite_number(bogus):
    """``atol: yes`` would apply as 1.0; ``.nan`` is silently ignored by ``passed_test``
    and ``.inf`` makes its gate unconditional."""
    with _refuses("must be (a number|finite)"):
        AccuracyContract(metric=Metric.TOLERANCE, atol=bogus)


def test_the_metric_is_a_closed_set():
    assert set(Metric) == {Metric.ULP, Metric.TOLERANCE}
    assert AccuracyContract(max_ulp=1).metric is Metric.ULP
    assert TOLERANCE_CONTRACT.metric is Metric.TOLERANCE


@pytest.mark.parametrize("bogus", ["ulp", "tolerance", "pcc", 0, None], ids=repr)
def test_a_metric_that_is_not_a_metric_member_is_refused(bogus):
    """A type annotation is not a check. ``metric="ulp"`` is not ``Metric.ULP`` -- the
    enum is bare, so the string compares unequal -- and it would otherwise fall through
    to the *tolerance* arm, silently switching off the gate the entry meant to declare.
    """
    with _refuses("must be a Metric member"):
        AccuracyContract(metric=bogus, max_ulp=1)


#: Both translations: only `tolerance_kwargs` has a production caller, and `passed_test`
#: takes `max_ulp` as an optional kwarg, so a `tolerance_kwargs` that regressed to the
#: `passed_test_kwargs` shape would not raise -- it would swap every enrolled op from
#: tolerance+PCC to a whole-format ULP budget, silently.
_TRANSLATIONS = [
    (AccuracyContract(max_ulp=3), {"max_ulp": 3, "near_zero_atol": None}, {}),
    (
        AccuracyContract(max_ulp=3, near_zero_atol=1e-7),
        {"max_ulp": 3, "near_zero_atol": 1e-7},
        {},
    ),
    (
        AccuracyContract(metric=Metric.TOLERANCE, atol=0.13, rtol=0.05),
        {"custom_atol": 0.13, "custom_rtol": 0.05},
        {"custom_atol": 0.13, "custom_rtol": 0.05},
    ),
    (
        TOLERANCE_CONTRACT,
        {"custom_atol": None, "custom_rtol": None},
        {"custom_atol": None, "custom_rtol": None},
    ),
]


@pytest.mark.parametrize("contract, by_ulp, by_tolerance", _TRANSLATIONS)
def test_a_contract_translates_to_passed_test_arguments(contract, by_ulp, by_tolerance):
    assert contract.passed_test_kwargs() == by_ulp
    assert contract.tolerance_kwargs() == by_tolerance


@pytest.mark.parametrize("method", ["passed_test_kwargs", "tolerance_kwargs"])
def test_every_contract_is_accepted_by_passed_test(method):
    """The translation has to be callable, not merely shaped right -- a renamed keyword
    would otherwise only surface on hardware."""
    golden = torch.ones(TILE_SIZE, dtype=torch.bfloat16)
    for contract, _, _ in _TRANSLATIONS:
        assert passed_test(
            golden, golden.clone(), DataFormat.Float16_b, **getattr(contract, method)()
        )


# ── Key resolution ────────────────────────────────────────────────────────────


def test_the_default_key_matches_every_variant():
    assert DEFAULT.specificity == 0
    assert DEFAULT.matches(
        BudgetKey(
            approx_mode=ApproximationMode.Yes,
            output_format=DataFormat.Float32,
            dest_acc=DestAccumulation.No,
            arch=ChipArchitecture.WORMHOLE,
        )
    )
    assert DEFAULT.matches(DEFAULT)


def test_a_more_specific_key_wins_over_the_default():
    table = {DEFAULT: AccuracyContract(max_ulp=64), F32: AccuracyContract(max_ulp=4)}
    assert resolve_contract(table, F32).max_ulp == 4
    assert resolve_contract(table, BF16).max_ulp == 64


def test_specificity_counts_every_set_dimension():
    table = {
        F32: AccuracyContract(max_ulp=4),
        BudgetKey(
            output_format=DataFormat.Float32, dest_acc=DestAccumulation.No
        ): AccuracyContract(max_ulp=8),
    }
    no = BudgetKey(output_format=DataFormat.Float32, dest_acc=DestAccumulation.No)
    yes = BudgetKey(output_format=DataFormat.Float32, dest_acc=DestAccumulation.Yes)
    assert resolve_contract(table, no).max_ulp == 8
    assert resolve_contract(table, yes).max_ulp == 4


def test_an_unset_query_dimension_only_matches_a_wildcard():
    """A caller that does not know ``dest_acc`` must not be handed a budget measured for
    one setting of it."""
    table = {BudgetKey(dest_acc=DestAccumulation.Yes): AccuracyContract(max_ulp=1)}
    assert resolve_contract(table, F32) is TOLERANCE_CONTRACT


def test_equally_specific_keys_are_an_error_not_a_tie_break():
    """Two keys, one dimension each, both matching: the table's iteration order must not
    decide a budget."""
    table = {
        BudgetKey(approx_mode=ApproximationMode.No): AccuracyContract(max_ulp=4),
        F32: AccuracyContract(max_ulp=64),
    }
    query = BudgetKey(
        output_format=DataFormat.Float32, approx_mode=ApproximationMode.No
    )
    with _refuses("Ambiguous has 2 equally specific"):
        resolve_contract(table, query, label="Ambiguous")


def test_an_empty_table_falls_back_to_the_tolerance_metric():
    assert resolve_contract({}, F32) is TOLERANCE_CONTRACT


def test_a_key_describes_itself_for_an_error_message():
    assert DEFAULT.describe() == "DEFAULT"
    described = BudgetKey(
        output_format=DataFormat.Float32, dest_acc=DestAccumulation.No
    ).describe()
    assert "output_format" in described and "dest_acc" in described


@pytest.mark.parametrize(
    "field, bogus",
    [
        ("approx_mode", True),
        ("output_format", "Float32"),
        ("dest_acc", True),
        ("dest_acc", False),
        ("arch", "wormhole"),
    ],
    ids=lambda v: str(v),
)
def test_a_budget_key_dimension_that_is_not_an_enum_member_is_refused(field, bogus):
    """All four are bare ``Enum``s, so ``DestAccumulation.No.value is False`` never equals
    its member. As a table key such a value is *inert*: counted as set by ``specificity``
    and matched by nothing. As a query it would silently fall through to a broader row --
    and ``sfpu_domains`` types ``dest_acc`` as ``Union[bool, Enum]``, so a driver can hand
    the same bool to both modules. The query is a ``BudgetKey`` too, so one check covers
    both sides."""
    with _refuses(f"BudgetKey.{field} must be a"):
        BudgetKey(**{field: bogus})


def test_every_budget_key_field_is_guarded():
    """``_BUDGET_KEY_TYPES`` is hand-maintained beside the dataclass, so a dimension
    added to one and not the other would be unguarded and silently inert."""
    assert {f.name for f in fields(BudgetKey)} == set(_BUDGET_KEY_TYPES)
    # ...and the declared member of each really is accepted.
    assert BudgetKey(
        approx_mode=ApproximationMode.No,
        output_format=DataFormat.Float32,
        dest_acc=DestAccumulation.Yes,
        arch=ChipArchitecture.WORMHOLE,
    ).specificity == len(_BUDGET_KEY_TYPES)


# ── The live registry ─────────────────────────────────────────────────────────


def test_the_registry_resolves_unambiguously_for_every_variant():
    """Exhaustive over the variant space, which is small. An ambiguity that only appears
    for one format is exactly what a reader of the table misses."""
    validate_registry()


def test_an_unenrolled_op_keeps_todays_gate():
    """Enrolment is incremental: nothing changes for an op until it is in the table."""
    assert MathOperation.Exp not in _SFPU_ACCURACY_BUDGET
    for fmt in ULP_FORMATS:
        assert (
            accuracy_contract(MathOperation.Exp, output_format=fmt, arch=MEASURED_ARCH)
            is TOLERANCE_CONTRACT
        )


@pytest.mark.parametrize(
    "arch", [a for a in ChipArchitecture if a != MEASURED_ARCH], ids=lambda a: a.name
)
@pytest.mark.parametrize(
    "op", [MathOperation.SigmoidAppx, MathOperation.GeluAppx], ids=lambda o: o.name
)
def test_a_declared_tolerance_survives_an_unswept_architecture(op, arch):
    """A declared *tolerance* is not a Wormhole measurement, so the arch gate must not
    take it: downgrading before the lookup would drop SigmoidAppx's and GeluAppx's
    atol=0.13 everywhere but Wormhole, back to the default those numbers exist to widen.
    """
    contract = accuracy_contract(
        op,
        output_format=DataFormat.Float16_b,
        approx_mode=ApproximationMode.Yes,
        dest_acc=DestAccumulation.No,
        arch=arch,
    )
    assert contract.metric is Metric.TOLERANCE
    assert contract.atol == 0.13 and contract.rtol == 0.05


@pytest.mark.parametrize("arch", list(ChipArchitecture), ids=lambda a: a.name)
@pytest.mark.parametrize(
    "output_format", [DataFormat.Float16_b, DataFormat.Bfp4_b], ids=lambda f: f.name
)
def test_a_downgrade_lands_on_the_ops_own_tolerance_row(
    arch, output_format, monkeypatch
):
    """Both downgrades re-resolve against the tolerance rows rather than returning the
    global default: a ULP row winning on specificity must not shadow a *broader*
    tolerance row the same op declares, or an op carrying both shapes falls back past
    its own atol to the per-format default. Built here rather than taken from the
    registry, since no live op carries a ULP row yet."""
    op = MathOperation.Abs
    monkeypatch.setitem(
        _SFPU_ACCURACY_BUDGET,
        op,
        {
            DEFAULT: AccuracyContract(metric=Metric.TOLERANCE, atol=0.13, rtol=0.05),
            BF16: AccuracyContract(max_ulp=3),
        },
    )
    contract = accuracy_contract(op, output_format=output_format, arch=arch)
    if output_format is DataFormat.Float16_b and arch is MEASURED_ARCH:
        assert contract.metric is Metric.ULP and contract.max_ulp == 3
    else:
        # Bfp4_b has no per-element ULP, and off Wormhole the unkeyed budget does not
        # bind -- either way the op keeps the 0.13 it declared, not the global default.
        assert contract.metric is Metric.TOLERANCE
        assert contract.atol == 0.13 and contract.rtol == 0.05


@pytest.mark.parametrize(
    "arch", [a for a in ChipArchitecture if a != MEASURED_ARCH], ids=lambda a: a.name
)
def test_a_ulp_row_that_names_its_arch_binds_there(arch, monkeypatch):
    """The arch gate exists because unkeyed numbers are Wormhole measurements. A row
    whose key names another arch *is* a measurement taken there, and must bind."""
    op = MathOperation.Abs
    monkeypatch.setitem(
        _SFPU_ACCURACY_BUDGET,
        op,
        {
            BF16: AccuracyContract(max_ulp=3),
            BudgetKey(output_format=DataFormat.Float16_b, arch=arch): AccuracyContract(
                max_ulp=5
            ),
        },
    )
    contract = accuracy_contract(op, output_format=DataFormat.Float16_b, arch=arch)
    assert contract.metric is Metric.ULP and contract.max_ulp == 5
    # ...while the unkeyed row is still downgraded on any other unswept arch.
    for other in ChipArchitecture:
        if other not in (arch, MEASURED_ARCH):
            assert (
                accuracy_contract(op, output_format=DataFormat.Float16_b, arch=other)
                is TOLERANCE_CONTRACT
            )


@pytest.mark.parametrize("enrolled", [False, True], ids=["unenrolled", "enrolled"])
def test_a_bad_query_is_refused_whether_or_not_the_op_is_enrolled(enrolled):
    """Validation must not depend on the table's contents, or a miswired driver passes
    until the day its op is enrolled. The op is picked from the table, not named, so
    enrolling more ops later does not change what this checks."""
    op = next(
        op
        for op in sorted(MathOperation, key=lambda op: op.name)
        if (op in _SFPU_ACCURACY_BUDGET) == enrolled
    )
    with _refuses("BudgetKey.arch must be a"):
        accuracy_contract(op, output_format=DataFormat.Float32, arch="wormhole")


def test_arch_must_be_passed_explicitly():
    """``arch`` is the one dimension where the numbers do not transfer, so unlike the
    others it cannot be left unset and quietly resolved against the Wormhole table."""
    with _refuses("arch", TypeError):
        accuracy_contract(MathOperation.Abs, output_format=DataFormat.Float32)


def test_a_variant_specific_tolerance_needs_no_driver_override():
    """Why removing ``custom_atol``/``custom_rtol`` from the driver loses nothing: a
    tolerance narrower than the op's default is a more specific :class:`BudgetKey`, so it
    need not move the op's other variants."""
    table = {
        DEFAULT: AccuracyContract(metric=Metric.TOLERANCE, atol=0.13, rtol=0.05),
        F32: AccuracyContract(metric=Metric.TOLERANCE, atol=0.001, rtol=0.001),
    }
    assert resolve_contract(table, F32).atol == 0.001
    assert resolve_contract(table, BF16).atol == 0.13


# ── The loader ────────────────────────────────────────────────────────────────


def test_a_repeated_op_in_the_table_is_refused(tmp_path):
    """YAML keeps only the last of two identical mapping keys, so the earlier op's whole
    budget would vanish with nothing downstream able to see it."""
    with _refuses("duplicate entry for 'Abs'"):
        _table(
            tmp_path,
            """\
            Abs:
              - {max_ulp: 0}
            Neg:
              - {max_ulp: 1}
            Abs:
              - {max_ulp: 99}
            """,
        )


def test_a_duplicate_row_is_refused_rather_than_deduplicated(tmp_path):
    """Two rows with the same key are two list items, not one -- so nothing collapses
    them, and a copy-pasted row replacing a measured budget would otherwise take effect
    silently as the later of the two."""
    with _refuses("repeats BudgetKey"):
        _table(
            tmp_path,
            """\
            Abs:
              - {out: Float16_b, max_ulp: 1}
              - {out: Float16_b, max_ulp: 4}
            """,
        )
    both = _table(
        tmp_path,
        """\
        Abs:
          - {max_ulp: 4}
          - {out: Float16_b, max_ulp: 1}
        """,
    )
    assert len(both[MathOperation.Abs]) == 2


def test_the_loader_refuses_what_it_cannot_turn_into_a_contract(tmp_path):
    """Every failure here is the author's, so each one names the op it came from."""
    with _refuses("'Nope' is not a MathOperation"):
        _table(tmp_path, "Nope:\n  - {max_ulp: 1}\n")
    with _refuses("Abs: unknown field"):
        _table(tmp_path, "Abs:\n  - {max_ulp: 1, budget: 2}\n")
    with _refuses("Abs: 'Float17' is not a DataFormat"):
        _table(tmp_path, "Abs:\n  - {out: Float17, max_ulp: 1}\n")
    with _refuses("Abs: 'pcc' is not a Metric"):
        _table(tmp_path, "Abs:\n  - {metric: pcc, max_ulp: 1}\n")
    with _refuses("Abs has no rows"):
        _table(tmp_path, "Abs:\n")
    # ...and the contract invariants still come from AccuracyContract itself.
    with _refuses("a ulp contract replaces the tolerance gate"):
        _table(tmp_path, "Abs:\n  - {max_ulp: 1, atol: 0.5}\n")


@pytest.mark.parametrize("alias", ["1", "0", "1.0"])
def test_a_numeric_alias_for_a_boolean_enum_is_refused(tmp_path, alias):
    """``True == 1`` in Python, so a by-value lookup alone would load ``approx: 1`` as
    ``Yes``: a typo in a key dimension would select a contract instead of failing."""
    with _refuses("is not a ApproximationMode"):
        _table(tmp_path, f"Abs:\n  - {{approx: {alias}, max_ulp: 1}}\n")


def test_a_quoted_and_an_unquoted_no_mean_the_same_thing(tmp_path):
    """YAML 1.1 reads a bare ``No`` as ``False``, and ``ApproximationMode.No`` is spelled
    ``False`` too, so the two spellings must not disagree. The table quotes them; the
    loader takes either."""
    quoted = _table(tmp_path, 'Abs:\n  - {approx: "No", dest: "Yes", max_ulp: 1}\n')
    bare = _table(tmp_path, "Abs:\n  - {approx: No, dest: Yes, max_ulp: 1}\n")
    assert quoted == bare
    key = next(iter(quoted[MathOperation.Abs]))
    assert key.approx_mode is ApproximationMode.No
    assert key.dest_acc is DestAccumulation.Yes
