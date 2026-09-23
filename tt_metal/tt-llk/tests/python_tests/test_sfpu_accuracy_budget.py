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

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
)
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
    enrolled_ops,
    resolve_contract,
    validate_registry,
)
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
from helpers.ulp import (
    ULP_FORMATS,
)
from helpers.utils import passed_test

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


def test_the_metric_is_a_closed_set():
    """Two members and no third gate to fall through to.

    Not a substitute for ``test_a_metric_that_is_not_a_metric_member_is_refused`` further
    down: a bare ``Enum`` does not stop ``metric="ulp"`` from being *constructed*, only
    from being one of these. The closed set is what makes the ``__post_init__`` check a
    complete one."""
    assert set(Metric) == {Metric.ULP, Metric.TOLERANCE}
    assert AccuracyContract(max_ulp=1).metric is Metric.ULP
    assert TOLERANCE_CONTRACT.metric is Metric.TOLERANCE


#: Both translations, because only `tolerance_kwargs` has a production caller and only
#: `passed_test_kwargs` is otherwise pinned. `passed_test` takes `max_ulp` and
#: `near_zero_atol` as optional kwargs, so a `tolerance_kwargs` that regressed from
#: `return {}` to the `passed_test_kwargs` shape would not raise -- it would swap every
#: enrolled op from tolerance+PCC to a whole-format ULP budget, silently.
_KWARGS_METHODS = ("passed_test_kwargs", "tolerance_kwargs")

#: The ULP arm differs between them: `tolerance_kwargs` deliberately declines to gate.
_TRANSLATIONS = [
    (
        AccuracyContract(max_ulp=3),
        {"max_ulp": 3, "near_zero_atol": None},
        {},
    ),
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


@pytest.mark.parametrize("method", _KWARGS_METHODS)
def test_every_contract_is_accepted_by_passed_test(method):
    """The translation has to be callable, not merely shaped right — a renamed keyword
    would otherwise only surface on hardware."""
    golden = torch_ones()
    for contract, _, _ in _TRANSLATIONS:
        assert passed_test(
            golden,
            golden.clone(),
            DataFormat.Float16_b,
            **getattr(contract, method)(),
        )


TILE_SIZE = DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM


def torch_ones():
    return torch.ones(TILE_SIZE, dtype=torch.bfloat16)


# ── Key resolution ────────────────────────────────────────────────────────────


def test_the_default_key_matches_every_variant():
    key = DEFAULT
    assert key.specificity == 0
    assert key.matches(
        approx_mode=ApproximationMode.Yes,
        output_format=DataFormat.Float32,
        dest_acc=DestAccumulation.No,
        arch=ChipArchitecture.WORMHOLE,
    )
    assert key.matches(approx_mode=None, output_format=None, dest_acc=None, arch=None)


def test_a_more_specific_key_wins_over_the_default():
    table = {
        DEFAULT: AccuracyContract(max_ulp=64),
        BudgetKey(output_format=DataFormat.Float32): AccuracyContract(max_ulp=4),
    }
    assert (
        resolve_contract(table, label="op", output_format=DataFormat.Float32).max_ulp
        == 4
    )
    assert (
        resolve_contract(table, label="op", output_format=DataFormat.Float16_b).max_ulp
        == 64
    )


def test_specificity_counts_every_set_dimension():
    table = {
        BudgetKey(output_format=DataFormat.Float32): AccuracyContract(max_ulp=4),
        BudgetKey(
            output_format=DataFormat.Float32, dest_acc=DestAccumulation.No
        ): AccuracyContract(max_ulp=8),
    }
    resolved = resolve_contract(
        table,
        label="op",
        output_format=DataFormat.Float32,
        dest_acc=DestAccumulation.No,
    )
    assert resolved.max_ulp == 8
    resolved = resolve_contract(
        table,
        label="op",
        output_format=DataFormat.Float32,
        dest_acc=DestAccumulation.Yes,
    )
    assert resolved.max_ulp == 4


def test_an_unset_query_dimension_only_matches_a_wildcard():
    """A caller that does not know ``dest_acc`` must not be handed a budget measured for
    one setting of it."""
    table = {BudgetKey(dest_acc=DestAccumulation.Yes): AccuracyContract(max_ulp=1)}
    assert (
        resolve_contract(table, label="op", output_format=DataFormat.Float32)
        is TOLERANCE_CONTRACT
    )


def test_equally_specific_keys_are_an_error_not_a_tie_break():
    """Two keys, one dimension each, both matching: the table's iteration order must not
    decide a budget."""
    table = {
        BudgetKey(approx_mode=ApproximationMode.No): AccuracyContract(max_ulp=4),
        BudgetKey(output_format=DataFormat.Float32): AccuracyContract(max_ulp=64),
    }
    with _refuses("equally specific"):
        resolve_contract(
            table,
            label="Ambiguous",
            output_format=DataFormat.Float32,
            approx_mode=ApproximationMode.No,
        )


def test_an_empty_table_falls_back_to_the_tolerance_metric():
    assert (
        resolve_contract({}, label="op", output_format=DataFormat.Float32)
        is TOLERANCE_CONTRACT
    )


def test_a_key_describes_itself_for_an_error_message():
    assert DEFAULT.describe() == "DEFAULT"
    described = BudgetKey(
        output_format=DataFormat.Float32, dest_acc=DestAccumulation.No
    ).describe()
    assert "output_format" in described and "dest_acc" in described


# ── The live registry ─────────────────────────────────────────────────────────


def test_the_registry_resolves_unambiguously_for_every_variant():
    """Exhaustive over the variant space, which is small. An ambiguity that only appears
    for one format is exactly what a reader of the table misses."""
    validate_registry()


def test_an_unenrolled_op_keeps_todays_gate():
    """Enrolment is incremental: nothing changes for an op until it is in the table."""
    assert MathOperation.Exp not in enrolled_ops()
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
    take it: downgrading before the lookup dropped SigmoidAppx's and GeluAppx's atol=0.13
    everywhere but Wormhole, back to the 0.05 default those numbers exist to widen."""
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
    "output_format",
    [DataFormat.Float16_b, DataFormat.Bfp4_b],
    ids=lambda f: f.name,
)
def test_a_downgrade_lands_on_the_ops_own_tolerance_row(
    arch, output_format, monkeypatch
):
    """Both downgrades re-resolve against the tolerance rows rather than returning the
    global default: a ULP row winning on specificity must not shadow a *broader*
    tolerance row the same op declares, or an op carrying both shapes falls back past
    its own atol to the per-format default.

    Built here rather than taken from the registry -- no live op carries both a ULP row
    and a wider declared atol yet, and the shadow is only observable through
    ``passed_test_kwargs()``.
    """
    op = MathOperation.Abs
    monkeypatch.setitem(
        _SFPU_ACCURACY_BUDGET,
        op,
        {
            DEFAULT: AccuracyContract(metric=Metric.TOLERANCE, atol=0.13, rtol=0.05),
            BudgetKey(output_format=DataFormat.Float16_b): AccuracyContract(max_ulp=3),
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


def test_arch_must_be_passed_explicitly():
    """``arch`` is the one dimension where the numbers do not transfer, so unlike the
    other three it cannot be left unset and quietly resolved against the Wormhole table.
    A second enroller that forgets the keyword fails at the call."""
    with _refuses("arch", TypeError):
        accuracy_contract(MathOperation.Abs, output_format=DataFormat.Float32)


def test_a_variant_specific_tolerance_needs_no_driver_override():
    """Why removing ``custom_atol``/``custom_rtol`` from the driver loses nothing: a
    tolerance narrower than the op's default is a more specific :class:`BudgetKey`, so it
    need not move the op's other variants."""
    table = {
        DEFAULT: AccuracyContract(metric=Metric.TOLERANCE, atol=0.13, rtol=0.05),
        BudgetKey(output_format=DataFormat.Float32): AccuracyContract(
            metric=Metric.TOLERANCE, atol=0.001, rtol=0.001
        ),
    }
    narrow = resolve_contract(
        table, label="probe", output_format=DataFormat.Float32, arch=MEASURED_ARCH
    )
    assert narrow.atol == 0.001
    broad = resolve_contract(
        table, label="probe", output_format=DataFormat.Float16_b, arch=MEASURED_ARCH
    )
    assert broad.atol == 0.13


def test_enrolled_ops_is_sorted_and_stable():
    ops = enrolled_ops()
    assert list(ops) == sorted(ops, key=lambda op: op.name)
    assert len(set(ops)) == len(ops)


def _refuses(match, kind=ValueError):
    """The suite's ``expect_error`` fixture needs a device; these are host-only tests."""
    return pytest.raises(kind, match=match)  # allow-pytest.raises: host-only test


# ── Integers never reach the ULP metric through the registry ──────────────────


# ── The contract refuses what its annotations cannot ──────────────────────────


@pytest.mark.parametrize("bogus", ["ulp", "tolerance", "pcc", 0, None], ids=repr)
def test_a_metric_that_is_not_a_metric_member_is_refused(bogus):
    """A type annotation is not a check. ``metric="ulp"`` is not ``Metric.ULP`` -- the
    enum is bare, so the string compares unequal -- and it used to fall through to the
    *tolerance* arm of ``__post_init__``, silently switching off the gate the entry meant
    to declare. Exactly the typo a registry edit makes."""
    with _refuses("must be a Metric member"):
        AccuracyContract(metric=bogus, max_ulp=1)


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
    """The same rule as ``AccuracyContract.metric``, on the four dimensions that had no
    check. All are bare ``Enum``s, so ``DestAccumulation.No.value is False`` never equals
    its member -- and such a key is *inert*: counted as set by ``specificity``, matched by
    nothing, and rendered identically by ``describe()``."""
    with _refuses(f"BudgetKey.{field} must be a"):
        BudgetKey(**{field: bogus})


def test_every_budget_key_field_is_guarded():
    """``_BUDGET_KEY_TYPES`` is hand-maintained beside the dataclass, so a dimension
    added to one and not the other would be unguarded and silently inert."""
    from dataclasses import fields

    assert {f.name for f in fields(BudgetKey)} == set(_BUDGET_KEY_TYPES)
    # ...and the declared member of each really is accepted.
    assert BudgetKey(
        approx_mode=ApproximationMode.No,
        output_format=DataFormat.Float32,
        dest_acc=DestAccumulation.Yes,
        arch=ChipArchitecture.WORMHOLE,
    ).specificity == len(_BUDGET_KEY_TYPES)


def _table(tmp_path, text):
    """*text* as a budget table on disk, loaded the way the real one is."""
    path = tmp_path / "budget.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return _load_table(path)


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
    with _refuses("unknown field"):
        _table(tmp_path, "Abs:\n  - {max_ulp: 1, budget: 2}\n")
    with _refuses("not a DataFormat"):
        _table(tmp_path, "Abs:\n  - {out: Float17, max_ulp: 1}\n")
    with _refuses("has no rows"):
        _table(tmp_path, "Abs:\n")
    # ...and the contract invariants still come from AccuracyContract itself.
    with _refuses("a ulp contract replaces the tolerance gate"):
        _table(tmp_path, "Abs:\n  - {max_ulp: 1, atol: 0.5}\n")


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


@pytest.mark.parametrize("field", ["atol", "rtol", "near_zero_atol"], ids=str)
def test_a_negative_tolerance_field_is_refused(field):
    """``passed_test`` applies an override only ``if custom_atol is not None and
    custom_atol >= 0``, so a negative ``atol``/``rtol`` read in the table as a declared
    tolerance and then silently ran against the per-format default. A negative
    ``near_zero_atol`` is a silently inert floor. The ULP arm already rejected a negative
    ``max_ulp`` -- the one case that would have failed loudly anyway."""
    metric = Metric.ULP if field == "near_zero_atol" else Metric.TOLERANCE
    kwargs = {"metric": metric, field: -0.001}
    if metric is Metric.ULP:
        kwargs["max_ulp"] = 1
    with _refuses("must not be negative"):
        AccuracyContract(**kwargs)
