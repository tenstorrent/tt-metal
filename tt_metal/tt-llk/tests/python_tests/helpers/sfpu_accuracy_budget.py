# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Which contract one variant of an SFPU op is held to.

The rows of the table (``sfpu_accuracy_budget_wh.yaml`` or ``_bh.yaml``, the file of the
chip this session targets: :data:`~.ulp_table.TABLE_PATHS`) are keyed on the formats,
approximation mode, Dest accumulation and architecture; an unset dimension matches any
value and the most specific key wins. An op with no row gets :data:`TOLERANCE_CONTRACT`.
A step budget binds only between the formats the SFPU computes in and on the
architecture it was measured on (:func:`steps_bind`); elsewhere the op's declared
tolerance applies.
"""

import os
from dataclasses import dataclass

from .chip_architecture import ChipArchitecture
from .format_config import DataFormat, InputOutputFormat
from .llk_params import ApproximationMode, DestAccumulation, MathOperation
from .ulp import FLUSH_SUBNORMAL_OUTPUTS
from .ulp_domains import Domain
from .ulp_table import (
    TOLERANCE_CONTRACT,
    AccuracyContract,
    BudgetKey,
    contracts_of,
    load_table,
    most_specific,
    resolve_variant,
    steps_bind,
    validate_contracts,
)


@dataclass(frozen=True)
class Query:
    """The variant :func:`accuracy_contract` was last asked about, in *test*: what the
    ``--ulp-measure`` recorder files the next comparison under."""

    test: str
    op: MathOperation
    key: BudgetKey


# The query nothing has consumed yet. Two lookups in one test with no comparison between
# them are ambiguous, and the recorder drops that reading rather than misfile it.
LAST_QUERY: Query | None = None
PENDING_AMBIGUOUS: bool = False


def _current_test() -> str:
    return os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]


# One op's keyed budgets.
_Contracts = dict[BudgetKey, AccuracyContract]

_SFPU_ACCURACY_BUDGET: dict[MathOperation, _Contracts] = contracts_of(load_table())


def resolve_contract(table: _Contracts, query: BudgetKey) -> AccuracyContract:
    """The most specific contract in *table* covering *query*, or
    :data:`TOLERANCE_CONTRACT`."""
    found = most_specific(table, query)
    return TOLERANCE_CONTRACT if found is None else found[1]


def accuracy_contract(
    op: MathOperation,
    *,
    output_format: DataFormat,
    arch: ChipArchitecture,
    input_format: DataFormat | None = None,
    approx_mode: ApproximationMode | None = None,
    dest_acc: DestAccumulation | None = None,
) -> AccuracyContract:
    """The contract for one op variant, or :data:`TOLERANCE_CONTRACT`. Not a pure lookup:
    it also publishes the variant as :data:`LAST_QUERY` for the ``--ulp-measure``
    recorder, so resolve once, right before the comparison."""
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
    approx_mode: ApproximationMode | None = None,
) -> None:
    """The binary, ternary and scalar drivers' last line: resolve the contract for the
    variant that ran and gate on the whole of it, step budget included. Pass
    *approx_mode* whenever the kernel compiles ``APPROX_MODE``, or a row keyed on it
    never matches."""
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
    input_format: DataFormat | None = None,
    approx_mode: ApproximationMode | None = None,
    dest_acc: DestAccumulation | None = None,
) -> tuple[tuple[Domain, ...], float | None]:
    """The ``domains`` list one variant is held to and the row's floor, or ``((),
    None)``. Pure: publishes no query."""
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
    found = most_specific(table, query)
    if found is None:
        return (), None
    key, contract = found
    if not contract.domains or not steps_bind(key, query):
        return (), None
    return contract.domains, contract.near_zero_atol


def enrolled_ops() -> tuple[MathOperation, ...]:
    """Every op with a row, in name order."""
    return tuple(sorted(_SFPU_ACCURACY_BUDGET, key=lambda op: op.name))


def validate_registry() -> None:
    """Raise if any op of the loaded registry can resolve ambiguously; the writer runs
    the same check on a proposed table (:func:`~helpers.ulp_table.validate_contracts`).
    """
    validate_contracts(_SFPU_ACCURACY_BUDGET)
