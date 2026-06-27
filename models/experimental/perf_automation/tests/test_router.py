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


# ---- ROUTE handler: off-menu (from-principles) is ALWAYS offered, not just on empty bucket ----
def test_route_handler_always_offers_from_principles(tmp_path):
    """Even when the bucket HAS matching playbook levers, FROM_PRINCIPLES must be appended
    to the candidate list so the brain can choose to reason freely (levers exist but none
    fits — the nemotron/Tilize case)."""
    from agent import states
    from agent.handlers.route import route as route_handler
    from agent.loop_context import LoopContext
    from agent.run import Run

    run = Run.create(tmp_path / "runs", config={"config": {}, "pathmap": {}}, run_id="R")
    prof = {
        "device_ms": 10.0,
        "wall_ms": 10.0,
        "buckets": [
            {
                "id": "matmul",
                "device_ms": 8.0,
                "count": 5,
                "tags": {"op_class": "matmul", "rank": "time", "bound": "flop"},
                "top_ops": [],
            }
        ],
    }
    (run.profiles_dir / "baseline_profile.json").write_text(json.dumps(prof))
    run.state_path.write_text(json.dumps({"state": "ROUTE", "exec_scope_done": True, "iteration": 0, "cost_usd": 0.0}))
    ctx = LoopContext.from_run(run, index=build_index())

    nxt = route_handler(ctx)
    cands = ctx.state["candidates"]
    assert nxt == states.SELECT
    assert states.FROM_PRINCIPLES in cands, "from-principles must always be a candidate"
    # matmul has real levers -> candidates must be levers PLUS from-principles (not fallback-only)
    assert len(cands) > 1, f"expected real levers + from-principles, got {cands}"


def test_route_handler_emits_bucket_landscape(tmp_path):
    """ROUTE-as-evidence: the brief must carry the FULL bucket landscape (all bottlenecks),
    not just the one bucket the deterministic ranker picked."""
    import json as _json

    from agent.handlers.route import route as route_handler
    from agent.loop_context import LoopContext
    from agent.run import Run

    run = Run.create(tmp_path / "runs", config={"config": {}, "pathmap": {}}, run_id="RL")
    prof = {
        "device_ms": 10.0,
        "wall_ms": 10.0,
        "buckets": [
            {"id": "datamove", "device_ms": 6.0, "count": 50, "tags": {"op_class": "datamove"}, "top_ops": []},
            {
                "id": "matmul",
                "device_ms": 4.0,
                "count": 5,
                "tags": {"op_class": "matmul", "rank": "time", "bound": "flop"},
                "top_ops": [],
            },
        ],
    }
    (run.profiles_dir / "baseline_profile.json").write_text(_json.dumps(prof))
    run.state_path.write_text(_json.dumps({"state": "ROUTE", "exec_scope_done": True, "iteration": 0, "cost_usd": 0.0}))
    ctx = LoopContext.from_run(run, index=build_index())

    route_handler(ctx)
    brief = _json.loads([l for l in (run.dir / "route_briefs.jsonl").read_text().splitlines()][-1])
    land = brief.get("bucket_landscape")
    assert land and {b["id"] for b in land} == {"datamove", "matmul"}  # ALL buckets present, not just the picked one


def test_route_handler_regime_verdict_and_kernel_reorder(tmp_path):
    """knob-vs-kernel diagnosis: the brief carries a regime_verdict, and once the bucket's
    TTNN knobs are all tried the verdict flips to 'kernel' and the kernel lever is moved to the
    front of the candidate list (routed directly, no exhaustion grind)."""
    import json as _json

    from agent import states
    from agent.handlers.route import _tt_lang_available
    from agent.handlers.route import route as route_handler
    from agent.loop_context import LoopContext
    from agent.run import Run

    if not _tt_lang_available():
        import pytest

        pytest.skip("tt-lang (ttl) not installed in this env")

    def _run_route(run_id, tried, iteration):
        run = Run.create(tmp_path / run_id, config={"config": {}, "pathmap": {}}, run_id=run_id)
        prof = {
            "device_ms": 10.0,
            "wall_ms": 10.0,
            "buckets": [
                {
                    "id": "matmul",
                    "device_ms": 8.0,
                    "count": 5,
                    "tags": {"op_class": "matmul", "rank": "time"},
                    "top_ops": [],
                }
            ],
        }
        (run.profiles_dir / "baseline_profile.json").write_text(_json.dumps(prof))
        run.state_path.write_text(
            _json.dumps(
                {"state": "ROUTE", "exec_scope_done": True, "iteration": iteration, "cost_usd": 0.0, "tried": tried}
            )
        )
        ctx = LoopContext.from_run(run, index=build_index())
        route_handler(ctx)
        brief = _json.loads((run.dir / "route_briefs.jsonl").read_text().splitlines()[-1])
        return ctx, brief

    # 1) fresh: brief carries a regime_verdict, and the kernel lever is offered for matmul
    ctx, brief = _run_route("RK0", tried=[], iteration=0)
    assert "regime_verdict" in brief and brief["regime_verdict"]["verdict"] in ("knob", "kernel")
    cands = ctx.state["candidates"]
    assert states.KERNEL_LEVER in cands, "kernel lever must be offered for a matmul bucket when ttl is available"

    # 2) all TTNN knobs tried -> verdict flips to kernel -> kernel lever reordered to front
    knobs = [c for c in cands if c not in (states.KERNEL_LEVER, states.FROM_PRINCIPLES)]
    ctx2, _ = _run_route("RK1", tried=knobs, iteration=1)
    assert ctx2.state["regime"]["verdict"] == "kernel"
    assert ctx2.state["candidates"][0] == states.KERNEL_LEVER
