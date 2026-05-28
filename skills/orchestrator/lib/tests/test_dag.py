# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for skills.orchestrator.lib.dag — candidate selection + deadlock detection.

Each test exercises one branch of the SPEC §3 tick decision tree. The helpers
``_component`` and ``_state`` build minimal valid dicts; tests then inject the
specific phase statuses / dependencies under test and assert on the shape of
``eligible_blocks``'s return.
"""

from skills.orchestrator.lib.state import bootstrap
from skills.orchestrator.lib.dag import eligible_blocks


def _component(
    name,
    *,
    deps=None,
    reference="pending",
    ttnn="pending",
    debug="n/a",
    optimization="pending",
    real_weights=None,
    host_resident=False,
):
    """Build a single component dict in the canonical shape.

    Defaults mirror a freshly-emitted architecture block: reference pending,
    ttnn pending, no debug needed yet, optimization pending. Tests override
    only the fields they care about.

    ``real_weights`` defaults to None (= phase dict absent) so legacy tests
    that don't care about the post-bringup phase continue to work; set
    explicitly when a test exercises the real_weights branch.
    """
    c = {
        "name": name,
        "depends_on": deps or [],
        "reference": {"status": reference},
        "ttnn": {"status": ttnn},
        "debug": {"status": debug},
        "optimization": {"status": optimization},
    }
    if real_weights is not None:
        c["real_weights"] = {"status": real_weights}
    if host_resident:
        c["host_resident"] = {
            "allowed": True,
            "justification": "test",
            "reference_link": "tests/dummy.py",
        }
    return c


def _state(components=None, use_cases=None):
    """Bootstrap a valid state and inject ``components`` / ``use_cases`` (defaults: empty)."""
    s = bootstrap("Acme/Foo-1B", "n150", "wormhole_b0")
    s["components"] = components or []
    s["use_cases"] = use_cases or []
    return s


def _use_case(
    name,
    *,
    components_used=None,
    needs_ar=True,
    needs_audio_out=False,
    generation="pending",
    perf="pending",
):
    """Build a single use_case dict in the canonical shape.

    Defaults mirror a freshly-emitted architecture use_case entry:
    generation pending, perf pending, no components_used. Tests override
    only the fields they care about.
    """
    return {
        "name": name,
        "description": "test",
        "input_modality": "text",
        "output_modality": "text",
        "components_used": components_used or [],
        "needs_ar": needs_ar,
        "needs_audio_out": needs_audio_out,
        "hf_class": "XxxForX",
        "validation_metric": "bleu",
        "validation_threshold": "HF - 1.0",
        "hybrid_notes": None,
        "generation": {"status": generation},
        "perf": {"status": perf},
    }


# ---------------------------------------------------------------------------
# Rule 1: architecture pending
# ---------------------------------------------------------------------------


def test_architecture_pending_first():
    """Empty component list → architecture worker should run."""
    assert eligible_blocks(_state([])) == {"phase": "architecture"}


def test_architecture_returned_only_when_components_empty():
    """Any non-empty component list moves past Rule 1."""
    result = eligible_blocks(_state([_component("A")]))
    assert result["phase"] != "architecture"


def test_reference_done_means_arch_phase_already_returned():
    """Rule 1 keys off the literal value of state['components'].

    Defensive guard: a state with components=[] must return architecture, not
    fall through to e.g. ``done`` (which would happen if the check were
    ``state.get("components")`` and the loop ran zero times).
    """
    assert eligible_blocks(_state([])) == {"phase": "architecture"}


# ---------------------------------------------------------------------------
# Rule 2: reference fan-out
# ---------------------------------------------------------------------------


def test_reference_pending_fanout():
    """All three components reference=pending → all three dispatched in order."""
    comps = [_component("A"), _component("B"), _component("C")]
    assert eligible_blocks(_state(comps)) == {
        "phase": "reference",
        "blocks": ["A", "B", "C"],
    }


def test_reference_capped_at_max_parallel():
    """Six pending components but default cap is 4 → exactly 4 in order."""
    comps = [_component(n) for n in ["A", "B", "C", "D", "E", "F"]]
    result = eligible_blocks(_state(comps))
    assert result["phase"] == "reference"
    assert result["blocks"] == ["A", "B", "C", "D"]


def test_reference_failing_is_eligible():
    """reference.status=failing is also a candidate for the reference phase."""
    comps = [_component("A", reference="failing")]
    result = eligible_blocks(_state(comps))
    assert result == {"phase": "reference", "blocks": ["A"]}


def test_reference_blocked_is_excluded():
    """A reference=blocked block is the only candidate → falls through to deadlock."""
    comps = [_component("A", reference="blocked")]
    result = eligible_blocks(_state(comps))
    assert result["phase"] == "deadlock"
    assert {"name": "A", "blocks_downstream": []} in result["blocking"]


# ---------------------------------------------------------------------------
# Rule 3: device queue
# ---------------------------------------------------------------------------


def test_device_prefers_first_component_with_remaining_work():
    """Under global priority (SPEC §3 step 3), pending-ttnn beats optimization
    across the queue regardless of component order. A has only optimization
    left (ttnn=done, opt=pending); B has ttnn=pending. B wins via the
    pending→ttnn priority scan."""
    comps = [
        _component("A", reference="done", ttnn="done"),
        _component("B", deps=["A"], reference="done", ttnn="pending"),
    ]
    result = eligible_blocks(_state(comps))
    assert result == {"phase": "device", "block": "B", "worker": "ttnn"}


def test_device_skips_block_with_unfinished_deps():
    """A and B both ttnn=pending, B depends on A → A picked first."""
    comps = [
        _component("A", reference="done", ttnn="pending"),
        _component("B", deps=["A"], reference="done", ttnn="pending"),
    ]
    result = eligible_blocks(_state(comps))
    assert result == {"phase": "device", "block": "A", "worker": "ttnn"}


def test_device_priority_failing_routes_to_debug():
    """ttnn=failing takes priority over ttnn=pending → debug worker."""
    comps = [
        _component("A", reference="done", ttnn="failing", debug="in_progress"),
        _component("B", reference="done", ttnn="pending"),
    ]
    result = eligible_blocks(_state(comps))
    assert result == {"phase": "device", "block": "A", "worker": "debug"}


def test_device_priority_debug_blocked_falls_through():
    """A.debug=blocked → A is not eligible; B's pending ttnn wins."""
    comps = [
        _component("A", reference="done", ttnn="failing", debug="blocked"),
        _component("B", reference="done", ttnn="pending"),
    ]
    result = eligible_blocks(_state(comps))
    assert result == {"phase": "device", "block": "B", "worker": "ttnn"}


def test_device_priority_optimization_after_ttnn_done():
    """ttnn=done, optimization=pending → optimization worker."""
    comps = [_component("A", reference="done", ttnn="done", optimization="pending")]
    result = eligible_blocks(_state(comps))
    assert result == {"phase": "device", "block": "A", "worker": "optimization"}


def test_device_priority_failing_beats_optimization_across_blocks():
    """Cross-block: A has only optimization to do; B is failing on ttnn.
    Per SPEC §3 step 3, failing→debug has global priority over
    optimization across the candidate queue, regardless of component order."""
    state = _state(
        [
            _component("A", reference="done", ttnn="done", optimization="pending"),
            _component("B", reference="done", ttnn="failing", debug="in_progress", optimization="pending"),
        ]
    )
    result = eligible_blocks(state)
    assert result == {"phase": "device", "block": "B", "worker": "debug"}


def test_device_priority_pending_ttnn_beats_optimization_across_blocks():
    """Cross-block: A has only optimization to do; B has ttnn pending.
    Pending ttnn beats optimization across the queue."""
    state = _state(
        [
            _component("A", reference="done", ttnn="done", optimization="pending"),
            _component("B", reference="done", ttnn="pending"),
        ]
    )
    result = eligible_blocks(state)
    assert result == {"phase": "device", "block": "B", "worker": "ttnn"}


def test_device_skips_fully_finished_components():
    """A is fully done (ttnn=done, opt=done) so it shouldn't be a candidate;
    B has remaining work."""
    state = _state(
        [
            _component("A", reference="done", ttnn="done", optimization="done"),
            _component("B", reference="done", ttnn="pending"),
        ]
    )
    result = eligible_blocks(state)
    assert result == {"phase": "device", "block": "B", "worker": "ttnn"}


def test_reference_zero_max_parallel_returns_empty_blocks():
    """A misconfigured max_parallel_reference=0 should return no blocks.
    The orchestrator can then deadlock-detect or treat as 'no work this tick'."""
    state = _state([_component("A", reference="pending"), _component("B", reference="pending")])
    state["config"]["max_parallel_reference"] = 0
    result = eligible_blocks(state)
    # Either the function returns reference with empty blocks list,
    # OR it skips to the next rule. Pin whichever behavior you implement.
    if result["phase"] == "reference":
        assert result["blocks"] == []
    else:
        # If you decided to skip Rule 2 when cap is 0, then components
        # with pending reference but no other progress means deadlock
        # (no reference work AND ttnn requires reference done).
        assert result["phase"] in {"deadlock", "device"}  # whichever your design lands on


# ---------------------------------------------------------------------------
# Rule 4: done
# ---------------------------------------------------------------------------


def test_done_when_all_phases_finished():
    """Every component ttnn=done + optimization=done + real_weights=done → done.

    The done check requires every per-component phase finished (including the
    post-bringup ``real_weights`` axis); no use_cases means the use_case half
    of the check is vacuously satisfied.
    """
    comps = [
        _component("A", reference="done", ttnn="done", optimization="done", real_weights="done"),
        _component("B", reference="done", ttnn="done", optimization="done", real_weights="done"),
    ]
    assert eligible_blocks(_state(comps)) == {"phase": "done"}


def test_done_with_skipped_phases():
    """ttnn=skipped + optimization=skipped + real_weights=skipped → done.

    ``skipped`` counts as finished, same as ``done`` — host_resident-allowed
    blocks may skip all their on-device phases including real_weights.
    """
    comps = [
        _component(
            "A",
            reference="done",
            ttnn="skipped",
            optimization="skipped",
            real_weights="skipped",
            host_resident=True,
        ),
    ]
    assert eligible_blocks(_state(comps)) == {"phase": "done"}


def test_done_with_host_resident_satisfies_ttnn():
    """host_resident.allowed=True satisfies the ttnn requirement for Rule 4.

    Even though ttnn is still pending, the escape hatch is what lets a
    bringup finish without TTNN, so the pipeline is complete. The
    ``real_weights`` axis is also satisfied via skipped here.
    """
    comps = [
        _component(
            "A",
            reference="done",
            ttnn="pending",
            optimization="done",
            real_weights="skipped",
            host_resident=True,
        ),
    ]
    assert eligible_blocks(_state(comps)) == {"phase": "done"}


# ---------------------------------------------------------------------------
# Rule 5: deadlock
# ---------------------------------------------------------------------------


def test_deadlock_when_dep_is_blocked():
    """A.ttnn=blocked and B depends on A → deadlock with B downstream of A."""
    comps = [
        _component("A", reference="done", ttnn="blocked"),
        _component("B", deps=["A"], reference="done", ttnn="pending"),
    ]
    result = eligible_blocks(_state(comps))
    assert result == {
        "phase": "deadlock",
        "blocking": [{"name": "A", "blocks_downstream": ["B"]}],
    }


def test_deadlock_no_downstream_for_isolated_blocked():
    """A single ttnn=blocked block with nothing depending on it → still deadlock."""
    comps = [_component("X", reference="done", ttnn="blocked")]
    result = eligible_blocks(_state(comps))
    assert result == {
        "phase": "deadlock",
        "blocking": [{"name": "X", "blocks_downstream": []}],
    }


def test_deadlock_lists_multiple_blocked_in_order():
    """Three blocked blocks reported in component order."""
    comps = [
        _component("A", reference="done", ttnn="blocked"),
        _component("B", reference="done", ttnn="blocked"),
        _component("C", reference="done", ttnn="blocked"),
    ]
    result = eligible_blocks(_state(comps))
    assert result["phase"] == "deadlock"
    assert [b["name"] for b in result["blocking"]] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Post-bringup Rule 4 (NEW): real_weights candidates
# ---------------------------------------------------------------------------


def test_real_weights_dispatched_after_optimization():
    """ttnn=done + optimization=done, real_weights absent (pending) → real_weights worker."""
    state = _state([_component("A", reference="done", ttnn="done", optimization="done")])
    result = eligible_blocks(state)
    assert result == {"phase": "device", "block": "A", "worker": "real_weights"}


def test_real_weights_not_dispatched_if_optimization_pending():
    """optimization still pending → optimization wins; real_weights gated on opt=done."""
    state = _state([_component("A", reference="done", ttnn="done", optimization="pending")])
    result = eligible_blocks(state)
    assert result == {"phase": "device", "block": "A", "worker": "optimization"}


def test_real_weights_not_dispatched_if_ttnn_pending():
    """ttnn still pending → ttnn wins; real_weights gated on ttnn=done."""
    state = _state([_component("A", reference="done", ttnn="pending", optimization="pending")])
    result = eligible_blocks(state)
    assert result["worker"] == "ttnn"


def test_real_weights_failing_routes_to_worker():
    """real_weights=failing is retried via the same real_weights worker (no debug fan-out)."""
    comp = _component("A", reference="done", ttnn="done", optimization="done")
    comp["real_weights"] = {"status": "failing"}
    result = eligible_blocks(_state([comp]))
    assert result == {"phase": "device", "block": "A", "worker": "real_weights"}


def test_real_weights_blocked_excluded():
    """real_weights=blocked → not dispatched; only block alone → falls through to deadlock."""
    comp = _component("A", reference="done", ttnn="done", optimization="done")
    comp["real_weights"] = {"status": "blocked"}
    result = eligible_blocks(_state([comp]))
    # Not real_weights — should be deadlock since A is the only block and is stuck.
    assert result["phase"] == "deadlock"
    assert {"name": "A", "blocks_downstream": []} in result["blocking"]


def test_real_weights_picks_first_eligible():
    """Two components ready for real_weights → component-order pick."""
    comps = [
        _component("A", reference="done", ttnn="done", optimization="done"),
        _component("B", reference="done", ttnn="done", optimization="done"),
    ]
    result = eligible_blocks(_state(comps))
    assert result == {"phase": "device", "block": "A", "worker": "real_weights"}


# ---------------------------------------------------------------------------
# Post-bringup Rule 5 (NEW): generation candidates (per-use-case)
# ---------------------------------------------------------------------------


def test_generation_dispatched_after_all_components_real_weights_done():
    """All components_used satisfied → generation worker for the use_case."""
    comp = _component("Linear", reference="done", ttnn="done", optimization="done", real_weights="done")
    state = _state([comp], use_cases=[_use_case("uc1", components_used=["Linear"])])
    result = eligible_blocks(state)
    assert result == {"phase": "device", "use_case": "uc1", "worker": "generation"}


def test_generation_blocked_if_dep_real_weights_pending():
    """A components_used entry with real_weights still pending → that wins, not generation."""
    comp = _component("Linear", reference="done", ttnn="done", optimization="done")
    # real_weights NOT set (= None / pending semantics)
    state = _state([comp], use_cases=[_use_case("uc1", components_used=["Linear"])])
    result = eligible_blocks(state)
    assert result["worker"] == "real_weights"
    assert result["block"] == "Linear"


def test_generation_failing_is_dispatched():
    """generation=failing should re-dispatch via the same generation worker."""
    comp = _component("Linear", reference="done", ttnn="done", optimization="done", real_weights="done")
    state = _state(
        [comp],
        use_cases=[_use_case("uc1", components_used=["Linear"], generation="failing")],
    )
    result = eligible_blocks(state)
    assert result == {"phase": "device", "use_case": "uc1", "worker": "generation"}


def test_generation_blocked_excluded():
    """generation=blocked → not dispatched; use_case effectively dead-ends here."""
    comp = _component("Linear", reference="done", ttnn="done", optimization="done", real_weights="done")
    uc = _use_case("uc1", components_used=["Linear"])
    uc["generation"] = {"status": "blocked"}
    state = _state([comp], use_cases=[uc])
    result = eligible_blocks(state)
    assert result["phase"] == "deadlock"


def test_generation_picks_first_eligible_use_case():
    """Two use_cases ready → use_case-order pick."""
    comp = _component("Linear", reference="done", ttnn="done", optimization="done", real_weights="done")
    state = _state(
        [comp],
        use_cases=[
            _use_case("uc1", components_used=["Linear"]),
            _use_case("uc2", components_used=["Linear"]),
        ],
    )
    result = eligible_blocks(state)
    assert result == {"phase": "device", "use_case": "uc1", "worker": "generation"}


def test_generation_missing_dep_component_blocks():
    """A use_case references an unknown component → not dispatched (deps unsatisfiable)."""
    comp = _component("Linear", reference="done", ttnn="done", optimization="done", real_weights="done")
    state = _state(
        [comp],
        use_cases=[_use_case("uc1", components_used=["NotPresent"])],
    )
    result = eligible_blocks(state)
    # uc1 cannot dispatch generation (dep missing); no real_weights pending
    # either (Linear done). Result should fall to deadlock since no progress
    # is possible; assert it is NOT a stale generation dispatch.
    assert result.get("worker") != "generation"


# ---------------------------------------------------------------------------
# Post-bringup Rule 6 (NEW): perf candidates (per-use-case)
# ---------------------------------------------------------------------------


def test_perf_dispatched_after_generation_done():
    """generation=done + perf pending → perf worker for the use_case."""
    comp = _component("Linear", reference="done", ttnn="done", optimization="done", real_weights="done")
    state = _state(
        [comp],
        use_cases=[_use_case("uc1", components_used=["Linear"], generation="done")],
    )
    result = eligible_blocks(state)
    assert result == {"phase": "device", "use_case": "uc1", "worker": "perf"}


def test_perf_blocked_excluded():
    """perf=blocked → not dispatched; falls through (eventually to deadlock)."""
    comp = _component("Linear", reference="done", ttnn="done", optimization="done", real_weights="done")
    uc = _use_case("uc1", components_used=["Linear"], generation="done")
    uc["perf"] = {"status": "blocked"}
    state = _state([comp], use_cases=[uc])
    result = eligible_blocks(state)
    assert result["phase"] == "deadlock"


def test_perf_failing_is_dispatched():
    """perf=failing → re-dispatched via perf worker."""
    comp = _component("Linear", reference="done", ttnn="done", optimization="done", real_weights="done")
    state = _state(
        [comp],
        use_cases=[_use_case("uc1", components_used=["Linear"], generation="done", perf="failing")],
    )
    result = eligible_blocks(state)
    assert result == {"phase": "device", "use_case": "uc1", "worker": "perf"}


def test_perf_skipped_if_generation_not_done():
    """generation still pending → generation wins, perf not dispatched."""
    comp = _component("Linear", reference="done", ttnn="done", optimization="done", real_weights="done")
    state = _state(
        [comp],
        use_cases=[_use_case("uc1", components_used=["Linear"], generation="pending")],
    )
    result = eligible_blocks(state)
    assert result["worker"] == "generation"


# ---------------------------------------------------------------------------
# Done check (extended): both axes must be complete
# ---------------------------------------------------------------------------


def test_done_requires_all_use_cases_complete():
    """All components + all use_cases finished → done."""
    comp = _component("Linear", reference="done", ttnn="done", optimization="done", real_weights="done")
    uc = _use_case("uc1", components_used=["Linear"], generation="done", perf="done")
    state = _state([comp], use_cases=[uc])
    assert eligible_blocks(state) == {"phase": "done"}


def test_done_not_yet_when_use_case_perf_pending():
    """Use_case still has perf pending → perf is dispatched, not done."""
    comp = _component("Linear", reference="done", ttnn="done", optimization="done", real_weights="done")
    uc = _use_case("uc1", components_used=["Linear"], generation="done", perf="pending")
    state = _state([comp], use_cases=[uc])
    result = eligible_blocks(state)
    assert result["phase"] == "device"
    assert result["worker"] == "perf"


def test_done_not_yet_when_component_real_weights_pending():
    """Component still has real_weights pending → real_weights wins, not done."""
    comp = _component("Linear", reference="done", ttnn="done", optimization="done")
    # No use_cases at all — even so, real_weights must complete first.
    state = _state([comp])
    result = eligible_blocks(state)
    assert result == {"phase": "device", "block": "Linear", "worker": "real_weights"}


# ---------------------------------------------------------------------------
# Deadlock (extended): scans use_case phases too
# ---------------------------------------------------------------------------


def test_deadlock_reports_use_case_generation_blocked():
    """A use_case with generation=blocked surfaces in the deadlock report."""
    comp = _component("Linear", reference="done", ttnn="done", optimization="done", real_weights="done")
    uc = _use_case("uc1", components_used=["Linear"])
    uc["generation"] = {"status": "blocked"}
    state = _state([comp], use_cases=[uc])
    result = eligible_blocks(state)
    assert result["phase"] == "deadlock"
    names = [entry["name"] for entry in result["blocking"]]
    assert "uc1" in names


def test_deadlock_reports_use_case_perf_blocked():
    """A use_case with perf=blocked surfaces in the deadlock report."""
    comp = _component("Linear", reference="done", ttnn="done", optimization="done", real_weights="done")
    uc = _use_case("uc1", components_used=["Linear"], generation="done")
    uc["perf"] = {"status": "blocked"}
    state = _state([comp], use_cases=[uc])
    result = eligible_blocks(state)
    assert result["phase"] == "deadlock"
    names = [entry["name"] for entry in result["blocking"]]
    assert "uc1" in names


def test_deadlock_reports_component_real_weights_blocked():
    """A component with real_weights=blocked surfaces in the deadlock report."""
    comp = _component("Linear", reference="done", ttnn="done", optimization="done")
    comp["real_weights"] = {"status": "blocked"}
    state = _state([comp])
    result = eligible_blocks(state)
    assert result["phase"] == "deadlock"
    assert {"name": "Linear", "blocks_downstream": []} in result["blocking"]
