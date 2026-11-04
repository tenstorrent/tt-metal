# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The SFPU accuracy table as typed rows, and the ``--ulp-measure`` records.

The table is data, one file per architecture it was measured on
(:data:`TABLE_PATHS`): each op maps to a list of :class:`Row`, written by
:func:`dump_table` as one flow mapping per row with enum members by name.
:func:`from_plain` reads any dataclass here from that plain form and refuses anything
else with a typed :class:`TableError`.
"""

import functools
import json
import math
import types
from dataclasses import MISSING, dataclass, fields, is_dataclass
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

from .chip_architecture import ChipArchitecture, get_chip_architecture
from .format_config import DataFormat
from .llk_params import ApproximationMode, DestAccumulation, MathOperation
from .ulp import ULP_FORMATS
from .ulp_domains import (
    Domain,
    Problem,
    TableError,
    check_domains,
    largest_figure,
    usable_budget_ceiling,
)

# One table per architecture the sweep has measured on. Every unkeyed step budget of a
# file binds on that file's architecture only (:func:`steps_bind`), and a session reads
# the file of the chip it targets; Quasar, with no table of its own, reads Wormhole's,
# where none binds. A figure carried over from another architecture's table is written
# as a tolerance row with its ``measured``, which binds nowhere.
TABLE_PATHS = {
    ChipArchitecture.WORMHOLE: Path(__file__).with_name("sfpu_accuracy_budget_wh.yaml"),
    ChipArchitecture.BLACKHOLE: Path(__file__).with_name(
        "sfpu_accuracy_budget_bh.yaml"
    ),
}


def table_arch(arch: ChipArchitecture) -> ChipArchitecture:
    """The architecture whose table a session on *arch* reads."""
    return arch if arch in TABLE_PATHS else ChipArchitecture.WORMHOLE


def arch_of_table(path: Path) -> ChipArchitecture | None:
    """The architecture of the table file *path* by its name (a copy such as
    ``base_sfpu_accuracy_budget_bh.yaml`` included), or ``None``."""
    name = Path(path).name
    found = [arch for arch, table in TABLE_PATHS.items() if name.endswith(table.name)]
    return found[0] if len(found) == 1 else None


@functools.cache
def measured_arch() -> ChipArchitecture:
    """The architecture every unkeyed row of this session's table was measured on.
    Asked of the chip on first use, not at import, so a host-only tool that never
    resolves a variant needs neither ``CHIP_ARCH`` nor a card; fixed from then on."""
    return table_arch(get_chip_architecture())


def __getattr__(name: str) -> Any:
    """``MEASURED_ARCH``, ``OTHER_ARCH`` (an architecture this session's table is not
    for) and ``TABLE_PATH`` (its file), resolved on first access (PEP 562). Code in this
    module calls :func:`measured_arch`: a global lookup never reaches here."""
    if name == "MEASURED_ARCH":
        return measured_arch()
    if name == "OTHER_ARCH":
        return next(a for a in ChipArchitecture if a is not measured_arch())
    if name == "TABLE_PATH":
        return TABLE_PATHS[measured_arch()]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class Metric(Enum):
    ULP = "ulp"
    TOLERANCE = "tolerance"


def _check_number(value: Any, *, integer: bool = False) -> None:
    """A finite non-negative number of the right kind, and not a bool (YAML reads ``yes``
    as one, and bool is an int)."""
    kinds = int if integer else (int, float)
    if (
        isinstance(value, bool)
        or not isinstance(value, kinds)
        or not math.isfinite(value)
        or value < 0
    ):
        raise TableError(Problem.BAD_NUMBER, value)


@dataclass(frozen=True)
class AccuracyContract:
    """How closely an op's output must match its golden: every element within *max_ulp*
    steps (``Metric.ULP``, the tolerance and PCC checks skipped), or the per-format
    tolerance. A *near_zero_atol* rides with a ULP contract or with a list."""

    metric: Metric = Metric.ULP
    max_ulp: int | None = None
    atol: float | None = None
    rtol: float | None = None
    near_zero_atol: float | None = None
    domains: tuple[Domain, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.metric, Metric):
            raise TableError(Problem.NOT_A_MEMBER, self.metric)
        for value in (self.atol, self.rtol, self.near_zero_atol):
            if value is not None:
                _check_number(value)
        if self.metric is Metric.ULP:
            if self.max_ulp is None:
                raise TableError(Problem.MISSING_FIGURE, self)
            _check_number(self.max_ulp, integer=True)
            if self.atol is not None or self.rtol is not None:
                raise TableError(Problem.MIXED_METRIC, self)
        elif self.max_ulp is not None or (
            self.near_zero_atol is not None and not self.domains
        ):
            raise TableError(Problem.MIXED_METRIC, self)

    @property
    def declares_tolerance(self) -> bool:
        """A tolerance contract with an ``atol`` or ``rtol``, as opposed to a numberless
        one that only opts its cell out of the ULP metric."""
        return self.metric is Metric.TOLERANCE and (
            self.atol is not None or self.rtol is not None
        )

    def tolerance_kwargs(self) -> dict[str, Any]:
        """``passed_test`` arguments for a caller that keeps today's tolerance on a ULP
        cell (the unary functional driver samples too few values for a sweep budget)."""
        if self.metric is Metric.ULP:
            return {}
        return {"custom_atol": self.atol, "custom_rtol": self.rtol}

    def passed_test_kwargs(
        self, flush_subnormals: bool | None = None
    ) -> dict[str, Any]:
        """The contract as ``passed_test`` keyword arguments; *flush_subnormals* overrides
        the ULP arm's per-dtype default."""
        if self.metric is Metric.ULP:
            kwargs = {"max_ulp": self.max_ulp, "near_zero_atol": self.near_zero_atol}
            if flush_subnormals is not None:
                kwargs["flush_subnormals"] = flush_subnormals
            return kwargs
        return {"custom_atol": self.atol, "custom_rtol": self.rtol}


# What an unenrolled op, or an enrolled one off the formats the SFPU computes in, is
# judged by: the per-format defaults plus PCC > 0.99.
TOLERANCE_CONTRACT = AccuracyContract(metric=Metric.TOLERANCE)


@dataclass(frozen=True)
class BudgetKey:
    """Which variants a row covers, or which variant is asked about: an unset field
    matches any value, more fields set is more specific, the most specific match wins,
    and two equally specific matches are an authoring error."""

    approx_mode: ApproximationMode | None = None
    input_format: DataFormat | None = None
    output_format: DataFormat | None = None
    dest_acc: DestAccumulation | None = None
    arch: ChipArchitecture | None = None

    def __post_init__(self) -> None:
        values = tuple(getattr(self, f.name) for f in fields(self))
        for value, expected in zip(values, _KEY_TYPES):
            # A bool never equals `DestAccumulation.No`: such a key would match nothing.
            if value is not None and not isinstance(value, expected):
                raise TableError(Problem.NOT_A_MEMBER, value)
        object.__setattr__(self, "_values", values)

    @property
    def specificity(self) -> int:
        return sum(value is not None for value in self._values)

    def matches(self, query: "BudgetKey") -> bool:
        """Whether this key covers *query*; an unset query dimension matches only a
        wildcard. Spelled out, with `_values` cached: this is `validate_contracts`' hot
        loop (via `most_specific`), and a generator over `zip` here costs that host
        test 1.3 s."""
        a, i, o, d, r = self._values
        qa, qi, qo, qd, qr = query._values
        return (
            (a is None or a == qa)
            and (i is None or i == qi)
            and (o is None or o == qo)
            and (d is None or d == qd)
            and (r is None or r == qr)
        )

    @property
    def sort_key(self) -> tuple:
        """The table's row order: unset fields first, then by member name, formats
        before flags."""
        return tuple(
            (value is not None, value.name if value is not None else "")
            for value in (
                self.input_format,
                self.output_format,
                self.approx_mode,
                self.dest_acc,
                self.arch,
            )
        )

    def describe(self) -> str:
        set_fields = [
            f"{f.name}={getattr(self, f.name).name}"
            for f in fields(self)
            if getattr(self, f.name) is not None
        ]
        return f"BudgetKey({', '.join(set_fields)})" if set_fields else "DEFAULT"


# The enum behind each field's `X | None` annotation, in field order.
_KEY_TYPES = tuple(
    next(a for a in get_args(hint) if a is not type(None))
    for hint in get_type_hints(BudgetKey).values()
)

V = TypeVar("V")


def most_specific(
    table: dict[BudgetKey, V],
    query: BudgetKey,
    only: "Any" = lambda value: True,
) -> tuple[BudgetKey, V] | None:
    """The single most specific entry of *table* covering *query* whose value passes
    *only*, or ``None``; two equally specific entries are :attr:`Problem.AMBIGUOUS`."""
    matched = [
        (key, value)
        for key, value in table.items()
        if key.matches(query) and only(value)
    ]
    if not matched:
        return None
    best = max(key.specificity for key, _ in matched)
    winners = [(key, value) for key, value in matched if key.specificity == best]
    if len(winners) > 1:
        raise TableError(Problem.AMBIGUOUS, [key.describe() for key, _ in winners])
    return winners[0]


def steps_bind(
    key: BudgetKey, query: BudgetKey, arch: ChipArchitecture | None = None
) -> bool:
    """Whether a step budget on the row keyed *key* binds the variant *query*: every
    format named is one the SFPU computes in, and the variant runs on the architecture
    the row was measured on: for a row that names none, the table's *arch* (default:
    this session's, :func:`measured_arch`)."""
    formats_ok = all(
        fmt is None or fmt in ULP_FORMATS
        for fmt in (query.input_format, query.output_format)
    )
    if not formats_ok:
        return False
    return key.arch is not None or query.arch == (arch or measured_arch())


@dataclass(frozen=True)
class Row:
    """One row of the table: the key fields say which variants it covers, the rest is
    its contract and the measurement behind it. A ULP row's gate is ``measured``; a
    sweep-written row carries ``domains`` and nothing beside it but a floor."""

    input: DataFormat | None = None
    output: DataFormat | None = None
    approx: ApproximationMode | None = None
    dest: DestAccumulation | None = None
    arch: ChipArchitecture | None = None
    metric: Metric | None = None
    measured: int | None = None
    nonfinite: int | None = None
    near_zero_atol: float | None = None
    atol: float | None = None
    rtol: float | None = None
    domains: tuple[Domain, ...] = ()
    note: str | None = None

    def __post_init__(self) -> None:
        for fmt in (self.input, self.output):
            if fmt is not None and fmt not in ULP_FORMATS:
                # A block float's error is the unpacker's and packer's, not the op's.
                raise TableError(Problem.NOT_A_COMPUTE_FORMAT, fmt)
        if self.measured is not None:
            _check_number(self.measured, integer=True)
        if self.nonfinite is not None:
            _check_number(self.nonfinite, integer=True)
            if self.nonfinite == 0:
                raise TableError(Problem.BAD_NUMBER, self.nonfinite)
        if self.domains:
            if (
                self.metric is not None
                or self.measured is not None
                or self.nonfinite is not None
                or self.atol is not None
                or self.rtol is not None
            ):
                raise TableError(Problem.BESIDE_DOMAINS, self)
            if self.input is None or self.output is None:
                raise TableError(Problem.UNPINNED_DOMAINS, self)
            check_domains(self.domains)
            if self.near_zero_atol is not None:
                _check_number(self.near_zero_atol)
            return
        if self.metric is not Metric.TOLERANCE:
            if self.measured is None:
                raise TableError(Problem.MISSING_FIGURE, self)
            if self.nonfinite is not None:
                raise TableError(Problem.NONFINITE_ON_BUDGET, self)
        # Built once, so a half-converted row is refused here, not on first use.
        object.__setattr__(self, "_contract", self._derive_contract())

    @property
    def key(self) -> BudgetKey:
        return BudgetKey(
            approx_mode=self.approx,
            input_format=self.input,
            output_format=self.output,
            dest_acc=self.dest,
            arch=self.arch,
        )

    @property
    def contract(self) -> AccuracyContract:
        """A list row is gated at its largest figure when every figure is within the
        output's ceiling, and on tolerance otherwise."""
        if self.domains:
            figure = largest_figure(self.domains)
            gated = figure <= usable_budget_ceiling(self.output)
            return AccuracyContract(
                metric=Metric.ULP if gated else Metric.TOLERANCE,
                max_ulp=figure if gated else None,
                near_zero_atol=self.near_zero_atol,
                domains=self.domains,
            )
        return self._contract

    def _derive_contract(self) -> AccuracyContract:
        if self.metric is Metric.TOLERANCE:
            return AccuracyContract(
                metric=Metric.TOLERANCE,
                atol=self.atol,
                rtol=self.rtol,
                near_zero_atol=self.near_zero_atol,
            )
        return AccuracyContract(
            max_ulp=self.measured,
            atol=self.atol,
            rtol=self.rtol,
            near_zero_atol=self.near_zero_atol,
        )

    @property
    def unmeasurable(self) -> bool:
        """Lanes disagreeing with the golden about being finite held the cell on
        tolerance."""
        return self.nonfinite is not None


# The table: each op's rows, in the order dump_table writes them.
Table = dict[MathOperation, tuple[Row, ...]]


def declared_tolerance(
    contracts: dict[BudgetKey, AccuracyContract],
    query: BudgetKey,
    fallback: AccuracyContract,
) -> AccuracyContract:
    """The most specific tolerance row *with numbers* covering *query*, or
    *fallback*."""
    found = most_specific(contracts, query, only=lambda c: c.declares_tolerance)
    return fallback if found is None else found[1]


def resolve_variant(
    contracts: dict[BudgetKey, AccuracyContract], query: BudgetKey
) -> AccuracyContract:
    """The contract one op variant is held to, given that op's keyed *contracts*: the
    most specific row, a step budget only where :func:`steps_bind` says it binds, and
    the op's declared tolerance where it does not. Pure; the driver-facing lookup is
    ``sfpu_accuracy_budget.accuracy_contract``."""
    found = most_specific(contracts, query)
    if found is None:
        return TOLERANCE_CONTRACT
    key, contract = found
    if contract.metric is Metric.ULP:
        if steps_bind(key, query):
            return contract
        return declared_tolerance(contracts, query, TOLERANCE_CONTRACT)
    if not contract.declares_tolerance:
        # A numberless tolerance row opts its cell out of the ULP metric without
        # retracting an atol/rtol the op declares on a broader row.
        return declared_tolerance(contracts, query, contract)
    return contract


def contracts_of(
    table: Table,
) -> dict[MathOperation, dict[BudgetKey, AccuracyContract]]:
    """Each op of *table* as its keyed contracts, the shape the resolver reads."""
    return {op: {row.key: row.contract for row in rows} for op, rows in table.items()}


def validate_contracts(
    by_op: dict[MathOperation, dict[BudgetKey, AccuracyContract]],
) -> None:
    """Raise if any op of *by_op* can resolve ambiguously, for any variant a driver may
    ask about, on any architecture: both lookups :func:`resolve_variant` can make are
    checked, whether or not a step budget binds. Per op, an axis is the values its own
    rows pin, plus ``None``: any other value matches the same rows ``None`` does."""
    for op, contracts in by_op.items():

        def pinned(field: str) -> list:
            return sorted(
                {getattr(key, field) for key in contracts} - {None}, key=str
            ) + [None]

        for approx_mode, input_format, output_format, dest_acc, arch in product(
            *map(
                pinned,
                ("approx_mode", "input_format", "output_format", "dest_acc", "arch"),
            )
        ):
            query = BudgetKey(
                approx_mode=approx_mode,
                input_format=input_format,
                output_format=output_format,
                dest_acc=dest_acc,
                arch=arch,
            )
            try:
                most_specific(contracts, query)
                most_specific(contracts, query, only=lambda c: c.declares_tolerance)
            except TableError as error:
                raise TableError(error.problem, f"{op.name}: {error}") from error


# ── Plain values: the one conversion between the dataclasses and YAML/JSON ─────


def to_plain(value: Any) -> Any:
    """*value* as YAML/JSON holds it: a dataclass as a mapping of the fields that differ
    from their defaults, an enum member by name, a tuple as a list."""
    if is_dataclass(value):
        return {
            f.name: to_plain(getattr(value, f.name))
            for f in fields(value)
            if getattr(value, f.name) != f.default
        }
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (tuple, list)):
        return [to_plain(item) for item in value]
    return value


def from_plain(cls: type, data: Any) -> Any:
    """The dataclass *cls* from :func:`to_plain`'s output, each field converted by its
    annotation; anything that does not fit is a :class:`TableError`."""
    if not isinstance(data, dict):
        raise TableError(Problem.WRONG_SHAPE, data)
    hints = get_type_hints(cls)
    unknown = set(data) - set(hints)
    if unknown:
        raise TableError(Problem.UNKNOWN_FIELD, sorted(map(str, unknown)))
    required = {
        f.name
        for f in fields(cls)
        if f.default is MISSING and f.default_factory is MISSING
    }
    missing = required - set(data)
    if missing:
        raise TableError(Problem.MISSING_FIELD, sorted(missing))
    return cls(**{name: _convert(hints[name], value) for name, value in data.items()})


def _convert(hint: Any, value: Any) -> Any:
    origin = get_origin(hint)
    if origin is Union or origin is types.UnionType:
        (inner,) = [a for a in get_args(hint) if a is not type(None)]
        return None if value is None else _convert(inner, value)
    if origin is tuple:
        args = get_args(hint)
        if not isinstance(value, (list, tuple)):
            raise TableError(Problem.WRONG_SHAPE, value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], item) for item in value)
        if len(args) != len(value):  # a fixed shape, such as a `bins` entry
            raise TableError(Problem.WRONG_SHAPE, value)
        return tuple(_convert(a, item) for a, item in zip(args, value))
    if is_dataclass(hint):
        return from_plain(hint, value)
    if isinstance(hint, type) and issubclass(hint, Enum):
        if not isinstance(value, str) or value not in hint.__members__:
            raise TableError(Problem.NOT_A_MEMBER, value)
        return hint[value]
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TableError(Problem.BAD_NUMBER, value)
        return value
    if hint is float:
        if isinstance(value, str) and value in _INFINITIES:
            return float(value)  # JSON has no infinity; `to_json` spells it this way
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TableError(Problem.BAD_NUMBER, value)
        return float(value)
    if not isinstance(value, hint):
        raise TableError(Problem.WRONG_SHAPE, value)
    return value


# ── The table file ───────────────────────────────────────────────────────────


class _Loader(getattr(yaml, "CSafeLoader", yaml.SafeLoader)):
    """PyYAML's safe loader, refusing a repeated mapping key (it would keep the last)."""


def _construct_mapping(loader: _Loader, node: yaml.MappingNode, deep: bool = False):
    seen = set()
    for key_node, _ in node.value:
        if not isinstance(key_node, yaml.ScalarNode):
            # Doubled braces pasted from an f-string: the row's one key is a mapping.
            raise TableError(
                Problem.WRONG_SHAPE,
                f"line {key_node.start_mark.line + 1}: a key that is not a name",
            )
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise TableError(
                Problem.DUPLICATE_YAML_KEY, f"line {key_node.start_mark.line + 1}"
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_Loader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def read_table(text: str) -> Table:
    """*text* as a :data:`Table`; a :class:`TableError` names the op and row it
    refuses."""
    try:
        loaded = yaml.load(text, Loader=_Loader)
    except yaml.YAMLError as error:
        raise TableError(Problem.BAD_YAML, error)
    if loaded is None:  # an empty document; `false`, `0` and `[]` are not tables
        loaded = {}
    if not isinstance(loaded, dict):
        raise TableError(Problem.WRONG_SHAPE, "the top level")
    table: Table = {}
    for name, rows in loaded.items():
        if not isinstance(name, str) or name not in MathOperation.__members__:
            raise TableError(Problem.UNKNOWN_OP, name)
        if not isinstance(rows, list) or not rows:
            raise TableError(Problem.NO_ROWS, name)
        typed = []
        for number, row in enumerate(rows, 1):
            try:
                typed.append(from_plain(Row, row))
            except TableError as error:
                raise TableError(error.problem, f"{name} row {number}", *error.where)
        keys = [row.key for row in typed]
        if len(set(keys)) != len(keys):
            raise TableError(Problem.REPEATED_KEY, name)
        table[MathOperation[name]] = tuple(typed)
    return table


def load_table(path: Path | None = None) -> Table:
    """The table at *path*, by default this session's."""
    path = TABLE_PATHS[measured_arch()] if path is None else Path(path)
    return read_table(path.read_text(encoding="utf-8"))


class _Dumper(yaml.SafeDumper):
    """One flow mapping per row, enum members by name, no line wrapping."""


_Dumper.add_representer(
    Row,
    lambda dumper, row: dumper.represent_mapping(
        "tag:yaml.org,2002:map", to_plain(row), flow_style=True
    ),
)


def dump_table(table: Table) -> str:
    """*table* as the file holds it: ops by name, each op's rows by key."""
    data = {
        op.name: sorted(table[op], key=lambda row: row.key.sort_key)
        for op in sorted(table, key=lambda op: op.name)
    }
    header = (
        "# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        "#\n"
        "# The accuracy contract each SFPU op declares (helpers/ulp_table.py, tests/SFPU_ULP.md).\n"
        "# Written by `test_unary_sfpu_ulp.py --ulp-emit` and helpers/ulp_table.dump_table.\n"
        "\n"
    )
    return header + yaml.dump(
        data,
        Dumper=_Dumper,
        sort_keys=False,
        width=math.inf,
        allow_unicode=True,
    )


def write_table_file(table: Table, path: Path | None = None) -> None:
    """Dump *table* to *path* (by default this session's table) after reading the dump
    back: a table that would not load is never written."""
    text = dump_table(table)
    validate_contracts(contracts_of(read_table(text)))
    path = TABLE_PATHS[measured_arch()] if path is None else Path(path)
    path.write_text(text, encoding="utf-8")


# ── Measurement records (--ulp-measure) ──────────────────────────────────────


@dataclass(frozen=True)
class Reading:
    """One entry of a cell's list as measured: the worst lane outside the floor
    (``None`` where none is ranked) and the ranked lane count."""

    low: float
    high: float
    worst: int | None
    lanes: int


@dataclass(frozen=True)
class Measurement:
    """One comparison, filed under the variant it ran: the worst ranked lane, the lane
    counts, the lanes disagreeing with the golden about being finite, and for a sweep
    cell its per-binade profile (``bins``: ``(binade, present, ranked, worst)``) and
    its readings over the table's list."""

    op: MathOperation
    key: BudgetKey
    worst: int
    lanes: int
    unmeasurable: int = 0
    nonfinite: int = 0
    test: str = ""
    bins: tuple[tuple[int, int, int, int], ...] = ()
    readings: tuple[Reading, ...] = ()

    def to_json(self) -> str:
        """One JSON object. An infinite entry edge is written as ``"inf"`` or
        ``"-inf"``: JSON has no infinity, and the ``-Infinity`` ``json.dumps`` would
        write only Python reads back."""
        return json.dumps(_json_safe(to_plain(self)))

    @classmethod
    def from_json(cls, line: str) -> "Measurement":
        return from_plain(cls, json.loads(line))


#: How :meth:`Measurement.to_json` spells an infinite float, and what
#: :func:`from_plain` reads back as one.
_INFINITIES = ("inf", "-inf")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and math.isinf(value):
        return str(value)
    return value


def read_measurements(path: Path) -> list[Measurement]:
    """The records in *path*; none for a missing file (a sweep that died before its
    first cell)."""
    path = Path(path)
    if not path.exists():
        return []
    return [
        Measurement.from_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
