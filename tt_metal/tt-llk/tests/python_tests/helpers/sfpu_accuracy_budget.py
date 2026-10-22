# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The accuracy contract an SFPU op declares, and the registry that holds them.

An op's budget is a *property of the op*, not a magic number in a test body: that makes
it reviewable, and a kernel improvement visible as the number dropping in the same PR.
The table is ``sfpu_accuracy_budget.yaml``, read by :mod:`helpers.ulp_table`; this
module resolves one variant of one op to its contract.

**Keyed on more than the op.** Approximation mode moves the error by orders of magnitude,
and the output format decides what is even visible, so a row is keyed on the formats,
approximation mode, Dest accumulation and architecture. Unset dimensions match any
value, the most specific key wins, and two equally specific matches are an authoring
error rather than a tie-break.

**Enrolment is per op and incremental.** An op with no entry gets
:data:`TOLERANCE_CONTRACT`. A step budget binds only between the formats the SFPU
computes in (``ulp.ULP_FORMATS``) and on the architecture it was measured on
(:func:`steps_bind`); a variant elsewhere keeps the op's declared tolerance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import product
from typing import Dict, Optional, Tuple, TypeVar

from .chip_architecture import ChipArchitecture
from .format_config import DataFormat, InputOutputFormat
from .llk_params import ApproximationMode, DestAccumulation, MathOperation
from .ulp import (  # FLUSH_SUBNORMAL_OUTPUTS re-exported: the step gates read it here
    FLUSH_SUBNORMAL_OUTPUTS,
    ULP_FORMATS,
)
from .ulp_domains import Domain, Problem, TableError, usable_budget_ceiling
from .ulp_table import (
    DEFAULT,
    TOLERANCE_CONTRACT,
    AccuracyContract,
    BudgetKey,
    Metric,
    load_table,
)

__all__ = [
    "DEFAULT",
    "FLUSH_SUBNORMAL_OUTPUTS",
    "TOLERANCE_CONTRACT",
    "AccuracyContract",
    "BudgetKey",
    "Metric",
    "usable_budget_ceiling",
]

#: The architecture every unkeyed budget was measured on.
MEASURED_ARCH = ChipArchitecture.WORMHOLE


@dataclass(frozen=True)
class Query:
    """The variant :func:`accuracy_contract` was last asked about, in *test*: what the
    ``--ulp-measure`` recorder files the next comparison under."""

    test: str
    op: MathOperation
    key: BudgetKey


#: The query nothing has consumed yet. Two lookups in one test with no comparison
#: between them make the association ambiguous (:data:`PENDING_AMBIGUOUS`), and the
#: recorder drops that reading rather than file it under the wrong variant.
LAST_QUERY: Optional[Query] = None
PENDING_AMBIGUOUS: bool = False


def _current_test() -> str:
    """The test a query was made in, so a stale one cannot cross a test boundary."""
    return os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]


#: One op's keyed budgets.
_Contracts = Dict[BudgetKey, AccuracyContract]

_SFPU_ACCURACY_BUDGET: Dict[MathOperation, _Contracts] = {
    op: {row.key: row.contract for row in rows} for op, rows in load_table().items()
}

V = TypeVar("V")


def _winner(
    table: Dict[BudgetKey, V], query: BudgetKey
) -> Optional[Tuple[BudgetKey, V]]:
    """The single most specific entry of *table* covering *query*, or ``None``."""
    matched = [(key, value) for key, value in table.items() if key.matches(query)]
    if not matched:
        return None
    best = max(key.specificity for key, _ in matched)
    winners = [(key, value) for key, value in matched if key.specificity == best]
    if len(winners) > 1:
        raise TableError(Problem.AMBIGUOUS, [key.describe() for key, _ in winners])
    return winners[0]


def resolve_contract(table: _Contracts, query: BudgetKey) -> AccuracyContract:
    """The most specific contract in *table* covering *query*, or
    :data:`TOLERANCE_CONTRACT`."""
    found = _winner(table, query)
    return TOLERANCE_CONTRACT if found is None else found[1]


def steps_bind(key: BudgetKey, query: BudgetKey) -> bool:
    """Whether a step budget on the row keyed *key* binds the variant *query*: every
    format named is one the SFPU computes in, and the variant is on the architecture
    the row was measured on -- :data:`MEASURED_ARCH` for a row that names none."""
    formats_ok = all(
        fmt is None or fmt in ULP_FORMATS
        for fmt in (query.input_format, query.output_format)
    )
    return formats_ok and (query.arch == MEASURED_ARCH or key.arch is not None)


def accuracy_contract(
    op: MathOperation,
    *,
    output_format: DataFormat,
    arch: ChipArchitecture,
    input_format: Optional[DataFormat] = None,
    approx_mode: Optional[ApproximationMode] = None,
    dest_acc: Optional[DestAccumulation] = None,
) -> AccuracyContract:
    """The contract for one op variant, or :data:`TOLERANCE_CONTRACT` if it has none.

    Not a pure lookup: every call also publishes the variant as :data:`LAST_QUERY` for
    the ``--ulp-measure`` recorder, and a second call in the same test marks it
    ambiguous. Resolve once, right before the comparison.
    """
    query = BudgetKey(
        approx_mode=approx_mode,
        input_format=input_format,
        output_format=output_format,
        dest_acc=dest_acc,
        arch=arch,
    )
    global LAST_QUERY, PENDING_AMBIGUOUS
    here = _current_test()
    PENDING_AMBIGUOUS = LAST_QUERY is not None and LAST_QUERY.test == here
    LAST_QUERY = Query(here, op, query)

    table = _SFPU_ACCURACY_BUDGET.get(op)
    if table is None:
        return TOLERANCE_CONTRACT
    found = _winner(table, query)
    if found is None:
        return TOLERANCE_CONTRACT
    key, contract = found
    if contract.metric is Metric.ULP:
        if steps_bind(key, query):
            return contract
        return _declared_tolerance(table, query, TOLERANCE_CONTRACT)
    if not contract.declares_tolerance:
        # A numberless tolerance row opts its cell out of the ULP metric without
        # retracting an atol/rtol the op declares on a broader row.
        return _declared_tolerance(table, query, contract)
    return contract


def _declared_tolerance(
    table: _Contracts, query: BudgetKey, fallback: AccuracyContract
) -> AccuracyContract:
    """The most specific tolerance row *with numbers* covering *query*, or
    *fallback*."""
    numbered = {
        key: contract for key, contract in table.items() if contract.declares_tolerance
    }
    found = _winner(numbered, query)
    return fallback if found is None else found[1]


def assert_against_contract(
    op: MathOperation,
    formats: InputOutputFormat,
    dest_acc: DestAccumulation,
    golden_tensor,
    res_tensor,
    *,
    approx_mode: Optional[ApproximationMode] = None,
) -> None:
    """Resolve *op*'s declared contract for the variant that ran, and gate on it: the
    binary, ternary and scalar drivers' shared last line. Their rows were measured over
    their own sweeps, so the whole contract applies, step budget included.

    *approx_mode* is left unset for a kernel that compiles no ``APPROX_MODE``; where the
    kernel does compile it, passing it is required, or a row keyed on it never matches.
    """
    from .chip_architecture import get_chip_architecture
    from .utils import passed_test

    contract = accuracy_contract(
        op,
        output_format=formats.output_format,
        input_format=formats.input_format,
        approx_mode=approx_mode,
        dest_acc=dest_acc,
        arch=get_chip_architecture(),
    )
    if not passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        **contract.passed_test_kwargs(flush_subnormals=FLUSH_SUBNORMAL_OUTPUTS),
    ):
        raise AssertionError("Assert against golden failed")


def domains_of(
    op: MathOperation,
    *,
    output_format: DataFormat,
    arch: ChipArchitecture,
    input_format: Optional[DataFormat] = None,
    approx_mode: Optional[ApproximationMode] = None,
    dest_acc: Optional[DestAccumulation] = None,
) -> Tuple[Tuple[Domain, ...], Optional[float]]:
    """The ``domains`` list one variant is held to, with the row's near-zero floor, or
    ``((), None)``. Pure, unlike :func:`accuracy_contract`: it publishes no query, so the
    sweep driver calls it beside its one lookup."""
    table = _SFPU_ACCURACY_BUDGET.get(op)
    if table is None:
        return (), None
    query = BudgetKey(
        approx_mode=approx_mode,
        input_format=input_format,
        output_format=output_format,
        dest_acc=dest_acc,
        arch=arch,
    )
    found = _winner(table, query)
    if found is None:
        return (), None
    key, contract = found
    if not contract.domains or not steps_bind(key, query):
        return (), None
    return contract.domains, contract.near_zero_atol


def enrolled_ops() -> Tuple[MathOperation, ...]:
    """Every op with a declared contract, in name order."""
    return tuple(sorted(_SFPU_ACCURACY_BUDGET, key=lambda op: op.name))


def validate_registry() -> None:
    """Raise if any op can resolve ambiguously, for any variant a driver may ask about.
    Per op, an axis is the values its own rows pin, plus ``None``; the output format and
    the architecture stay whole, because :func:`steps_bind` reads them."""
    for op, table in _SFPU_ACCURACY_BUDGET.items():

        def pinned(field: str) -> list:
            return sorted({getattr(k, field) for k in table} - {None}, key=str) + [None]

        for approx_mode, input_format, output_format, dest_acc, arch in product(
            pinned("approx_mode"),
            pinned("input_format"),
            DataFormat,
            pinned("dest_acc"),
            ChipArchitecture,
        ):
            accuracy_contract(
                op,
                output_format=output_format,
                input_format=input_format,
                approx_mode=approx_mode,
                dest_acc=dest_acc,
                arch=arch,
            )
