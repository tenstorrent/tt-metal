#!/usr/bin/env python3
"""Tests for the compute-API ordering rule table.

The table is data, so the tests are what keep it honest: every symbol has to exist in the
compute API, every argument index has to fit the real signature, every citation has to point at
a real line, and every deprecation entry has to match .github/deprecations.json.
"""
import json
import os
import re

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TABLE = os.path.join(HERE, "..", "compute_api_rules.yaml")
API_DIR = os.path.join(REPO, "tt_metal", "hw", "inc", "api", "compute")
DEPRECATIONS = os.path.join(REPO, ".github", "deprecations.json")

THREADS = {"UNPACK", "MATH", "PACK"}
ARCHES = {"wormhole_b0", "blackhole"}
REQUIRED = {"id", "op", "init", "thread", "arg_match", "source", "note"}


def load():
    with open(TABLE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def api_sources():
    """Every header under the compute API, keyed by absolute path."""
    out = {}
    for root, _, files in os.walk(API_DIR):
        for name in files:
            if name.endswith((".h", ".hpp")):
                path = os.path.join(root, name)
                with open(path, encoding="utf-8", errors="replace") as f:
                    out[path] = f.read()
    return out


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def signature(text, name):
    """Return the parameter list of `void name(...)` in text, or None."""
    m = re.search(r"ALWI\s+void\s+" + re.escape(name) + r"\s*\(", text)
    if not m:
        return None
    i, depth = m.end(), 1
    while i < len(text) and depth:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        i += 1
    return text[m.end() : i - 1]


def params(text, name):
    """Top-level parameters, without the trailing call_line that only carries diagnostics."""
    raw = signature(text, name)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return []
    out, depth, cur = [], 0, ""
    for ch in raw:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return [p for p in out if not re.search(r"\bcall_line\b", p)]


def symbols():
    """Map every compute-API function name to (path, source)."""
    index = {}
    for path, text in api_sources().items():
        clean = strip_comments(text)
        for name in re.findall(r"ALWI\s+void\s+([A-Za-z_]\w*)\s*\(", clean):
            index.setdefault(name, (path, clean))
    return index


def test_schema_and_unique_ids():
    data = load()
    assert data["version"] == 1
    assert set(data["architectures"]) == ARCHES
    ids = [r["id"] for r in data["operations"]]
    assert len(ids) == len(set(ids)), f"duplicate rule ids: {ids}"
    for rule in data["operations"]:
        missing = REQUIRED - set(rule)
        assert not missing, f"{rule['id']} missing {missing}"
        assert rule["thread"] in THREADS, rule["id"]
        assert rule["source"].startswith("tt_metal/") or rule["source"].startswith(".github/")
        assert len(rule["note"]) > 20, f"{rule['id']} needs a real note"
        for pair in rule["arg_match"]:
            assert len(pair) == 2 and all(isinstance(i, int) and i >= 0 for i in pair), rule["id"]
        if rule.get("requires_uninit"):
            assert rule.get("uninit"), f"{rule['id']} requires_uninit without uninit"


def test_every_symbol_exists():
    index = symbols()
    for rule in load()["operations"]:
        for key in ("op", "init"):
            assert rule[key] in index, f"{rule['id']}: {rule[key]} not in the compute API"
        if rule.get("uninit"):
            assert rule["uninit"] in index, f"{rule['id']}: {rule['uninit']} not found"
        for name in rule.get("invalidated_by", []):
            assert name in index, f"{rule['id']}: invalidator {name} not found"


def test_argument_indices_fit_the_signatures():
    index = symbols()
    for rule in load()["operations"]:
        init_params = params(index[rule["init"]][1], rule["init"])
        op_params = params(index[rule["op"]][1], rule["op"])
        assert init_params is not None and op_params is not None, rule["id"]
        for init_idx, op_idx in rule["arg_match"]:
            assert init_idx < len(init_params), (
                f"{rule['id']}: init arg {init_idx} out of range for "
                f"{rule['init']}({len(init_params)} params)"
            )
            assert op_idx < len(op_params), (
                f"{rule['id']}: op arg {op_idx} out of range for "
                f"{rule['op']}({len(op_params)} params)"
            )


def test_sources_point_at_real_lines():
    for rule in load()["operations"]:
        path, _, line = rule["source"].partition(":")
        full = os.path.join(REPO, path)
        assert os.path.isfile(full), f"{rule['id']}: missing {path}"
        assert line.isdigit(), f"{rule['id']}: source needs file:line"
        with open(full, encoding="utf-8", errors="replace") as f:
            n = sum(1 for _ in f)
        assert 1 <= int(line) <= n, f"{rule['id']}: {path}:{line} past end ({n} lines)"


def test_deprecated_inits_match_the_deprecation_tracker():
    if not os.path.isfile(DEPRECATIONS):
        pytest.skip(".github/deprecations.json is not checked out")
    with open(DEPRECATIONS, encoding="utf-8") as f:
        tracked = {d["id"] for d in json.load(f)["deprecations"]}
    index = symbols()
    for entry in load()["deprecated_inits"]:
        assert entry["id"] in tracked, f"{entry['id']} is not a tracked deprecation"
        for name in entry["inits"]:
            assert name in index, f"deprecated init {name} is no longer in the compute API"


def test_matmul_argument_order_is_not_assumed():
    """matmul_init calls state_configure(in1, in0); the rule must keep per-argument mapping."""
    rule = next(r for r in load()["operations"] if r["id"] == "matmul-tiles")
    assert rule["arg_match"] == [[0, 0], [1, 1]]
    assert "state_configure" in rule["note"]
