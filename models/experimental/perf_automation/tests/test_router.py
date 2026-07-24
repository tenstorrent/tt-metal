"""M2 tests: routing core + cache_playbook (PLAN section 7.2)."""

import json

import pytest

from agent.router import (
    WILDCARD,
    build_index,
    cache_playbook,
    coverage_lint,
    read_section,
    route,
)


def _by_id(index):
    return {e["id"]: e for e in index}


# ---- build_index (against the real GUIDELINES/*.md) ----


def test_build_index_harvests_route_blocks():
    index = build_index()
    by_id = _by_id(index)

    # A known lever section with a route block.
    assert "mlp-fidelity-walk" in by_id
    entry = by_id["mlp-fidelity-walk"]
    assert entry["file"] == "05_MLP.md"
    assert entry["lever_type"] == "walk"
    assert "matmul" in entry["op_class"]
    assert "time" in entry["rank"]
    assert "flop" in entry["bound"]
    # Omitted dimensions default to wildcard.
    assert entry["regime"] == [WILDCARD]

    # Process-doc anchors without a route block are NOT indexed.
    assert "anchor" not in by_id  # the AGENT_INDEX.md example anchor


# ---- route ----


def test_route_matches_by_tag_equality():
    index = build_index()
    results = route(index, {"op_class": "matmul", "rank": "time", "bound": "flop"})
    ids = [e["id"] for e in results]
    assert "mlp-fidelity-walk" in ids
    # Every result is compatible with op_class=matmul.
    for e in results:
        assert WILDCARD in e["op_class"] or "matmul" in e["op_class"]
    # Results preserve document (index) order.
    assert results == [e for e in index if e in results]


def test_route_fidelity_exhaustion():
    index = build_index()
    base = {"op_class": "matmul", "rank": "time", "bound": "flop"}
    assert "mlp-fidelity-walk" in [e["id"] for e in route(index, base)]
    # A LoFi bucket: the hifi-only walk section is excluded (exhausted).
    lofi = dict(base, fidelity="lofi")
    assert "mlp-fidelity-walk" not in [e["id"] for e in route(index, lofi)]


def test_route_wildcard():
    index = [
        {
            "id": "a",
            "title": "",
            "file": "x.md",
            "lever_type": "walk",
            "op_class": ["matmul"],
            "bound": [WILDCARD],
            "rank": [WILDCARD],
            "fidelity": [WILDCARD],
            "grid": [WILDCARD],
            "dispatch": [WILDCARD],
            "memory": [WILDCARD],
            "regime": [WILDCARD],
        },
    ]
    # Section wildcard on grid matches a query that sets grid.
    assert route(index, {"op_class": "matmul", "grid": "tiny"})
    # Query omits every dim (all wildcard) -> matches.
    assert route(index, {})
    # Non-wildcard mismatch on op_class -> no match.
    assert route(index, {"op_class": "eltwise"}) == []


def test_route_rejects_unknown_dim():
    index = build_index()
    # `rank_axis` is the old POC dim name -> a real, likely-silent mistake.
    with pytest.raises(ValueError):
        route(index, {"rank_axis": "time"})


def test_route_rejects_invalid_value():
    index = build_index()
    # `gemm`/`saturated` are not in the closed vocabulary (PLAN section 4.1).
    with pytest.raises(ValueError):
        route(index, {"op_class": "gemm"})
    with pytest.raises(ValueError):
        route(index, {"fidelity": "saturated"})


# ---- read_section ----


def test_read_section_by_anchor():
    text = read_section("mlp-fidelity-walk")
    first_line = text.splitlines()[0]
    assert first_line.startswith("## ")
    assert "{#mlp-fidelity-walk}" in first_line
    assert "<!-- route" in text
    # Stops before the next section.
    assert "{#mlp-l1-handoff}" not in text


def test_read_section_unknown_raises():
    with pytest.raises(KeyError):
        read_section("does-not-exist")


# ---- coverage_lint ----


def test_coverage_lint_flags_uncovered_key():
    index = [
        {
            "id": "a",
            "title": "",
            "file": "x.md",
            "lever_type": "walk",
            "op_class": ["matmul"],
            "bound": [WILDCARD],
            "rank": [WILDCARD],
            "fidelity": [WILDCARD],
            "grid": [WILDCARD],
            "dispatch": [WILDCARD],
            "memory": [WILDCARD],
            "regime": [WILDCARD],
        },
    ]
    possible = [{"op_class": "matmul"}, {"op_class": "ccl"}]
    uncovered = coverage_lint(index, possible)
    assert {"op_class": "ccl"} in uncovered
    assert {"op_class": "matmul"} not in uncovered


# ---- cache_playbook ----


def _section(anchor, op_class="matmul"):
    return (
        f"## Lever {anchor} {{#{anchor}}}\n"
        "<!-- route\n"
        f"op_class: {op_class}\n"
        "lever_type: walk\n"
        "-->\n\nbody text\n\n"
    )


def test_index_cache_invalidates_on_content_change(tmp_path):
    pb = tmp_path / "playbook"
    pb.mkdir()
    md = pb / "01.md"
    md.write_text(_section("sec-a"))
    cache = tmp_path / ".cache" / "playbook_index.json"

    idx1 = cache_playbook(pb, cache)
    assert "sec-a" in [e["id"] for e in idx1]
    hash1 = json.loads(cache.read_text())["hash"]
    mtime1 = cache.stat().st_mtime_ns

    # No change -> cache hit, no rewrite.
    cache_playbook(pb, cache)
    assert cache.stat().st_mtime_ns == mtime1

    # Content change -> hash changes -> rebuild.
    md.write_text(_section("sec-a") + _section("sec-b"))
    idx2 = cache_playbook(pb, cache)
    ids2 = [e["id"] for e in idx2]
    assert "sec-b" in ids2
    assert json.loads(cache.read_text())["hash"] != hash1
