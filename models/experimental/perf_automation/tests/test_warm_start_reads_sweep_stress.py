# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HARD STRESS: the warm-start lookup must read the file the sweep actually writes.

`_warm_start_for` read `data.get("entries")` and bailed unless it was a list. But matmul_sweep.py
writes summarize()'s output -- `seeds` (pre-picked winners) and the full `table` -- and NEVER an
`entries` key. So the guard tripped on every call, every lookup returned None, and the 14 measured,
PCC-gated configs were silently discarded (next_target["warm_start"] was never set).

The two row layouts are DIFFERENT and both must be handled:
  seeds row : {"shape": {"m","k","n"}, "fidelity", "dtype", "pcc", ...}   # shape NESTED, config flat
  table row : {"m","k","n", "best": {"fidelity","dtype","ms","pcc"}, ...} # shape FLAT, config in `best`

We drive matmul_sweep.summarize() itself to BUILD the file (so the test tracks the real schema, not a
hand-rolled guess), then assert _warm_start_for resolves the shape to the recommended config.

perf_mcp imports `mcp` (an external server pkg absent in CI here), so we stub it before load -- every
other import (agent.*) resolves in this env, exactly like the sibling warm-start test.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"
sys.path.insert(0, str(_PA))


def _stub_mcp():
    """Minimal stand-in for mcp.server.fastmcp.FastMCP: every method returns a pass-through decorator,
    so perf_mcp's module-level `mcp = FastMCP(...)` and its `@mcp.tool()` decorators import cleanly."""
    if "mcp" in sys.modules:
        return

    class _FastMCP:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, _name):
            def _method(*a, **k):
                if len(a) == 1 and callable(a[0]) and not k:
                    return a[0]  # bare  @mcp.tool

                def _decorator(fn=None):
                    return fn

                return _decorator  # parametrised  @mcp.tool(...)

            return _method

    mcp = types.ModuleType("mcp")
    server = types.ModuleType("mcp.server")
    fastmcp = types.ModuleType("mcp.server.fastmcp")
    fastmcp.FastMCP = _FastMCP
    server.fastmcp = fastmcp
    mcp.server = server
    sys.modules["mcp"] = mcp
    sys.modules["mcp.server"] = server
    sys.modules["mcp.server.fastmcp"] = fastmcp


def _load_perf_mcp():
    _stub_mcp()
    spec = importlib.util.spec_from_file_location("perf_mcp_ws_ut", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["perf_mcp_ws_ut"] = m
    spec.loader.exec_module(m)
    return m


def _load_sweep():
    spec = importlib.util.spec_from_file_location("matmul_sweep_ws_ut", str(_CC / "matmul_sweep.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sweep_row(m, k, n, fidelity, dtype, ms, base_ms, pcc=0.999):
    """A table row shaped like matmul_sweep's real per-shape entry: candidates + a `best`."""
    return {
        "m": m,
        "k": k,
        "n": n,
        "candidates": [
            {"fidelity": "hifi4", "dtype": "bfloat16", "ms": base_ms, "pcc": 1.0},
            {"fidelity": fidelity, "dtype": dtype, "ms": ms, "pcc": pcc},
        ],
        "best": {"fidelity": fidelity, "dtype": dtype, "ms": ms, "pcc": pcc},
    }


def _write_real_sweep_json(tmp_path):
    """Build matmul_sweep.json via the REAL summarize(), so seeds/table match production exactly."""
    sweep = _load_sweep()
    table = [
        _sweep_row(64, 4096, 1024, "lofi", "bfloat8_b", ms=0.5, base_ms=1.0),
        _sweep_row(512, 512, 512, "hifi2", "bfloat16", ms=0.8, base_ms=1.0),
        _sweep_row(32, 1024, 1024, "hifi4", "bfloat16", ms=1.0, base_ms=1.0),  # no win -> still seeded
    ]
    summary = sweep.summarize(table)
    summary["ok"] = True
    summary["table"] = table
    assert "entries" not in summary, "precondition: the sweep never writes an `entries` key"
    assert summary["seeds"], "summarize must produce seeds"
    (tmp_path / "matmul_sweep.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- the bug
def test_lookup_resolves_the_winner_from_the_real_sweep_file(tmp_path):
    pm = _load_perf_mcp()
    _write_real_sweep_json(tmp_path)
    # the op_code carries the shape fingerprint the loop passes in
    got = pm._warm_start_for(str(tmp_path), "MatmulDeviceOperation 64x4096x1024")
    assert got == {"fidelity": "lofi", "dtype": "bfloat8_b"}, f"warm-start not delivered: {got}"


def test_regression_entries_only_file_would_have_returned_none(tmp_path):
    """Pin the exact bug: a file with ONLY the old `entries` key (never written now) yields nothing --
    proving the fix reads seeds/table, and that the guard used to trip here."""
    pm = _load_perf_mcp()
    (tmp_path / "matmul_sweep.json").write_text(
        json.dumps({"entries": [{"m": 64, "k": 4096, "n": 1024, "fidelity": "lofi", "dtype": "bfloat8_b"}]})
    )
    assert pm._warm_start_for(str(tmp_path), "matmul 64x4096x1024") is None


def test_falls_back_to_table_when_seeds_absent(tmp_path):
    """If only `table` is present (no seeds rolled up), the flat-shape + `best`-config layout still
    resolves -- the second source in the fix."""
    pm = _load_perf_mcp()
    table = [_sweep_row(128, 256, 256, "lofi", "bfloat8_b", ms=0.4, base_ms=1.0)]
    (tmp_path / "matmul_sweep.json").write_text(json.dumps({"ok": True, "table": table}))
    got = pm._warm_start_for(str(tmp_path), "matmul 128x256x256")
    assert got == {"fidelity": "lofi", "dtype": "bfloat8_b"}, f"table fallback failed: {got}"


def test_no_matching_shape_and_hostile_inputs_return_none(tmp_path):
    pm = _load_perf_mcp()
    _write_real_sweep_json(tmp_path)
    assert pm._warm_start_for(str(tmp_path), "matmul 7x7x7") is None  # shape not swept
    assert pm._warm_start_for(str(tmp_path), "LayerNorm 64x4096") is None  # not a matmul
    assert pm._warm_start_for(str(tmp_path), "") is None
    # missing file
    assert pm._warm_start_for(str(tmp_path / "nope"), "matmul 64x4096x1024") is None
    # garbage json
    (tmp_path / "matmul_sweep.json").write_text("{not json")
    assert pm._warm_start_for(str(tmp_path), "matmul 64x4096x1024") is None


def test_seeds_layout_is_nested_and_the_fix_reads_it(tmp_path):
    """Directly pin the layout hazard: the seeds row nests the shape under ['shape'], so a naive
    row['m'] would KeyError. Assert the delivered config comes from the nested-shape seed."""
    pm = _load_perf_mcp()
    summary = _write_real_sweep_json(tmp_path)
    seed = next(s for s in summary["seeds"] if s["shape"] == {"m": 512, "k": 512, "n": 512})
    assert "m" not in seed and seed["shape"]["m"] == 512  # precondition: shape IS nested
    got = pm._warm_start_for(str(tmp_path), "matmul 512x512x512")
    assert got == {"fidelity": seed["fidelity"], "dtype": seed["dtype"]}


# --------------------------------------------------------------------------- source guard
def test_source_no_longer_reads_entries():
    src = (_CC / "perf_mcp.py").read_text()
    i = src.index("def _warm_start_for(")
    j = src.index("\ndef ", i + 1)
    body = src[i:j]
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    assert '"entries"' not in code, "the fix must not read the non-existent `entries` key"
    # The KEYS must be read -- not one particular spelling of the read. Pinning the literal
    # `data.get("seeds")` broke the moment those lookups moved behind a _rows() helper guarding
    # against a non-list value. A source guard should survive a refactor that preserves its intent,
    # otherwise it just taxes every future edit.
    assert '"seeds"' in code and '"table"' in code, "must read seeds + table"
