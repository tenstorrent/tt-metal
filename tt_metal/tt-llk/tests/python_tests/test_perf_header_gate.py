# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Merge gate: per-perf-test CSV schema versioning + metric-vocabulary drift.

The per-test gate re-derives, for every WH/BH perf test, the columns its CSV
would carry (statically, from that test's ``PerfConfig`` parameters) and compares
them to the recorded schema in ``helpers/perf_test_schemas.py``. A change to one
test's columns fails with a per-test diff, so you can see exactly which test
changed and how — and, because each test carries a ``version``, two reports of
the same test are comparable by that number.

Quasar perf tests derive the same way from their sibling test_*_quasar.py source
(see perf_schema_derive), so ``PERF_TEST_SCHEMAS_QSR`` is gated exactly like WH/BH.

Two smaller global gates keep the metric-column vocabulary (run-type names and
efficiency-metric names) from drifting. Derivation helpers live in the device-free
``perf_schema_derive`` module, so the gate needs no hardware.
"""

import ast

from perf_schema_derive import (
    ROOT,
    derive_perf_test_schemas,
    enum_member_names,
    load_pure_module,
)

# ── Per-test schema catalog (WH/BH) ───────────────────────────────────


def _assert_schemas_match(catalog, live, arch):
    problems = []
    for name, cols in sorted(live.items()):
        entry = catalog.get(name)
        if entry is None:
            problems.append(
                f"  '{name}': perf test found in source but missing from the "
                f"catalog. Add an entry with its columns and version=1."
            )
            continue
        added = sorted(set(cols) - set(entry["columns"]))
        removed = sorted(set(entry["columns"]) - set(cols))
        if added or removed:
            problems.append(
                f"  '{name}' (schema v{entry['version']}): +{added} -{removed}"
            )
    for name in sorted(set(catalog) - set(live)):
        problems.append(
            f"  '{name}': in catalog but no longer a {arch} perf test "
            f"(renamed or deleted?). Remove its catalog entry."
        )

    msg = [
        f"{arch} per-perf-test CSV schema(s) drifted from "
        f"helpers/perf_test_schemas.py:",
        "",
        *problems,
        "",
        "If intentional: update that test's 'columns', bump its 'version', and "
        "for a renamed column add an 'aliases' entry (old -> new).",
    ]
    assert not problems, "\n".join(msg)


def test_perf_test_schemas_match():
    ps = load_pure_module("perf_test_schemas.py")
    _assert_schemas_match(
        ps.PERF_TEST_SCHEMAS, derive_perf_test_schemas(quasar=False), "WH/BH"
    )


def test_perf_test_schemas_match_qsr():
    ps = load_pure_module("perf_test_schemas.py")
    _assert_schemas_match(
        ps.PERF_TEST_SCHEMAS_QSR, derive_perf_test_schemas(quasar=True), "Quasar"
    )


# ── Metric-vocabulary drift (global) ──────────────────────────────────


def test_run_type_names_match_source():
    ps = load_pure_module("perf_schema.py")
    live = enum_member_names("llk_params.py", "PerfRunType")
    assert live == set(ps.RUN_TYPE_NAMES), (
        f"PerfRunType members {sorted(live)} drifted from "
        f"perf_schema.RUN_TYPE_NAMES {sorted(ps.RUN_TYPE_NAMES)}. Update the "
        f"catalog: a run-type name prefixes every metric/counter header."
    )


def test_metric_bases_match_source():
    """The catalog's metric bases must equal the *_pct dict keys metrics.py exports."""
    tree = ast.parse((ROOT / "helpers" / "metrics.py").read_text())
    # export_metrics keeps exactly the metric-dict keys ending in "_pct" (see
    # _exportable()). Read the dict keys via ast, not a text scan, so an
    # unrelated "_pct" string literal (log line, docstring, m.get() arg) can
    # neither trip nor evade the gate.
    live = {
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant)
        and isinstance(key.value, str)
        and key.value.endswith("_pct")
    }
    ps = load_pure_module("perf_schema.py")
    assert live == set(ps.METRIC_BASES), (
        f"Efficiency metric names drifted. In source but not catalog: "
        f"{sorted(live - set(ps.METRIC_BASES))}; in catalog but not source: "
        f"{sorted(set(ps.METRIC_BASES) - live)}. Update perf_schema.METRIC_BASES."
    )
