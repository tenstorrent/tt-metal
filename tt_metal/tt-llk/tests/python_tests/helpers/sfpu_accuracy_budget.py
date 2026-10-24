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
from typing import Dict, Optional, Tuple

from .chip_architecture import ChipArchitecture
from .format_config import DataFormat, InputOutputFormat
from .llk_params import ApproximationMode, DestAccumulation, MathOperation
from .ulp import (  # re-exported: the step gates read it here
    FLUSH_SUBNORMAL_OUTPUTS,
)
from .ulp_domains import Domain, usable_budget_ceiling
from .ulp_table import (
    DEFAULT,
    TOLERANCE_CONTRACT,
    AccuracyContract,
    BudgetKey,
    Metric,
    _winner,
    contracts_of,
    load_table,
    resolve_variant,
    steps_bind,
    validate_contracts,
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

_SFPU_ACCURACY_BUDGET: Dict[MathOperation, _Contracts] = contracts_of(load_table())


def resolve_contract(table: _Contracts, query: BudgetKey) -> AccuracyContract:
    """The most specific contract in *table* covering *query*, or
    :data:`TOLERANCE_CONTRACT`."""
    found = _winner(table, query)
    return TOLERANCE_CONTRACT if found is None else found[1]


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
    return resolve_variant(table, query)


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
    """Raise if any op of the loaded registry can resolve ambiguously; the writer runs
    the same check on a proposed table (:func:`~helpers.ulp_table.validate_contracts`).
    """
    validate_contracts(_SFPU_ACCURACY_BUDGET)
