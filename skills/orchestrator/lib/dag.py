# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""DAG candidate selection + deadlock detection for the bringup orchestrator.

Implements SPEC.md §3 "Tick decision tree" as a pure function over the state
dict. The orchestrator calls ``eligible_blocks(state)`` once per tick to learn
what to do next; no I/O, no mutation. The returned dict is one of five shapes:

- ``{"phase": "architecture"}``
- ``{"phase": "reference", "blocks": [name, ...]}``
- ``{"phase": "device", "block": name, "worker": "ttnn"|"debug"|"optimization"}``
- ``{"phase": "done"}``
- ``{"phase": "deadlock", "blocking": [{"name": ..., "blocks_downstream": [...]}, ...]}``

The rules are evaluated in order; the first matching rule's result is returned.
See ``SPEC.md`` §3 for the canonical specification.
"""

from __future__ import annotations

# Statuses that count as "finished" — both successful completion (``done``) and
# explicit operator-approved skipping (``skipped`` via the host-resident escape
# hatch) free downstream work to proceed. ``blocked`` is NOT finished; it's a
# stuck state that requires human intervention.
_FINISHED_STATUSES = frozenset({"done", "skipped"})

# Statuses that make a component eligible for the reference fan-out. ``failing``
# is included so the orchestrator automatically retries reference workers
# whose previous attempts produced PCC < 0.99. ``blocked`` is intentionally
# excluded — once a phase is blocked, the orchestrator stops dispatching it
# and waits for ``--redo`` / ``--skip`` from the operator.
_REFERENCE_PENDING_STATUSES = frozenset({"pending", "failing"})


def _phase_status(component: dict, phase: str) -> str | None:
    """Return the status string for ``component[phase]``, or None if absent.

    Phase dicts may be entirely missing on a freshly-emitted component that
    has not yet been touched by any worker; treating a missing phase as
    ``None`` (rather than raising) lets the decision tree handle that case
    via "missing == pending" semantics where appropriate.
    """
    phase_dict = component.get(phase)
    if not isinstance(phase_dict, dict):
        return None
    return phase_dict.get("status")


def _is_finished(component: dict, phase: str) -> bool:
    """True if the named phase is in a terminal-success state.

    Used to gate both reference→device handoff and DAG dependency satisfaction:
    a block's ttnn is "finished enough" to unblock its dependents whether it
    completed normally or was explicitly skipped via host_resident.
    """
    return _phase_status(component, phase) in _FINISHED_STATUSES


def _ttnn_satisfied_for_completion(component: dict) -> bool:
    """True if a component's ttnn requirement is met for Rule 4 (pipeline done).

    The host_resident escape hatch counts as satisfying ttnn — that's the
    whole point of the hatch. See SPEC §"No shortcuts" guard.
    """
    if _is_finished(component, "ttnn"):
        return True
    hr = component.get("host_resident") or {}
    return hr.get("allowed") is True


def _deps_finished_for_ttnn(state: dict, component: dict) -> bool:
    """True iff every ``depends_on`` entry has ttnn finished.

    A missing dependency (named in ``depends_on`` but absent from
    ``state["components"]``) is treated as NOT finished — we can't schedule
    a block whose stated prerequisites don't exist in the state.
    """
    by_name = {c["name"]: c for c in state["components"]}
    for dep_name in component.get("depends_on", []):
        dep = by_name.get(dep_name)
        if dep is None or not _is_finished(dep, "ttnn"):
            return False
    return True


def _downstream_of(state: dict, name: str) -> list[str]:
    """Return all components transitively depending on ``name``.

    Used by Rule 5 to render the deadlock report: when ``name`` is blocked,
    these are the components that can never progress as a result.

    Returns a list sorted alphabetically for deterministic output.
    """
    # Adjacency: dep_name -> list of dependents
    dependents: dict[str, list[str]] = {}
    for c in state["components"]:
        for dep in c.get("depends_on", []):
            dependents.setdefault(dep, []).append(c["name"])

    seen: set[str] = set()
    stack = list(dependents.get(name, []))
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(dependents.get(cur, []))
    return sorted(seen)


def _is_device_candidate(state: dict, component: dict) -> bool:
    """True iff ``component`` is in the device candidate set for Rule 3.

    A candidate must (a) not be host_resident-allowed (escape hatch means
    no device work), (b) have reference finished, (c) have all upstream
    deps ttnn-finished, and (d) have *some* remaining device work — either
    ttnn or optimization in a non-finished state. The per-priority
    "current phase not blocked" check is applied in the priority scans
    themselves, not here.
    """
    if (component.get("host_resident") or {}).get("allowed") is True:
        return False
    if not _is_finished(component, "reference"):
        return False
    if not _deps_finished_for_ttnn(state, component):
        return False
    ttnn_status = _phase_status(component, "ttnn")
    opt_status = _phase_status(component, "optimization")
    # Remaining work if ttnn isn't finished OR optimization isn't finished.
    ttnn_done = ttnn_status in _FINISHED_STATUSES
    opt_done = opt_status in _FINISHED_STATUSES
    return not (ttnn_done and opt_done)


def eligible_blocks(state: dict) -> dict:
    """Decide the next orchestrator action for ``state``.

    Pure function — does not read I/O or mutate the argument. Returns one of
    the five shapes documented in this module's docstring. See ``SPEC.md`` §3
    (Tick decision tree) for the rules; they are evaluated in order and the
    first match wins.
    """
    components = state["components"]

    # ---- Rule 1: architecture pending ------------------------------------
    # Literal empty-list check (not falsy) so we don't accidentally match
    # ``None`` or some other oddity that should have been caught by schema
    # validation upstream.
    if components == []:
        return {"phase": "architecture"}

    # ---- Rule 2: reference fan-out ---------------------------------------
    # Design choice: a non-positive ``max_parallel_reference`` is treated as
    # "no reference work this tick" — we return ``{phase: reference,
    # blocks: []}`` so the orchestrator sees that Rule 2 matched (preventing
    # accidental fall-through to a misleading "device" or "done" decision)
    # but dispatches zero workers. The operator can then detect the cap is
    # misconfigured. We return rather than skip so the contract "Rule 2
    # matches whenever there is pending reference work" stays intact.
    max_parallel = state["config"]["max_parallel_reference"]
    reference_candidates: list[str] = []
    if max_parallel > 0:
        for c in components:
            ref_status = _phase_status(c, "reference")
            # Missing reference dict is treated as pending — see SPEC §3 step 2.
            if ref_status is None or ref_status in _REFERENCE_PENDING_STATUSES:
                reference_candidates.append(c["name"])
                if len(reference_candidates) == max_parallel:
                    break
        if reference_candidates:
            return {"phase": "reference", "blocks": reference_candidates}
    else:
        # Cap is zero/negative — check whether any component would have been
        # eligible. If so, surface an empty reference dispatch; if nothing
        # was eligible anyway, fall through to later rules normally.
        for c in components:
            ref_status = _phase_status(c, "reference")
            if ref_status is None or ref_status in _REFERENCE_PENDING_STATUSES:
                return {"phase": "reference", "blocks": []}

    # ---- Rule 3: device queue --------------------------------------------
    # SPEC §3 step 3: "Priority within the queue" is GLOBAL across the
    # candidate set, not per-component. We compute the candidate set once,
    # then do three priority scans in component order:
    #   (a) ttnn=failing + debug not blocked → debug worker
    #   (b) ttnn=pending (or missing) + ttnn not blocked → ttnn worker
    #   (c) ttnn done/skipped + optimization=pending (or missing)
    #       + optimization not blocked → optimization worker
    # First match in each scan wins; if scan (a) has no hit we move on.
    candidates = [c for c in components if _is_device_candidate(state, c)]

    # Priority (a): failing → debug.
    for c in candidates:
        if _phase_status(c, "ttnn") == "failing" and _phase_status(c, "debug") != "blocked":
            return {"phase": "device", "block": c["name"], "worker": "debug"}

    # Priority (b): pending (or missing) ttnn → ttnn worker. ``blocked`` ttnn
    # is excluded so a single blocked block does not pin the queue.
    for c in candidates:
        ttnn_status = _phase_status(c, "ttnn")
        if ttnn_status in ("pending", None):
            return {"phase": "device", "block": c["name"], "worker": "ttnn"}

    # Priority (c): ttnn finished + optimization pending → optimization worker.
    for c in candidates:
        ttnn_status = _phase_status(c, "ttnn")
        opt_status = _phase_status(c, "optimization")
        if ttnn_status in _FINISHED_STATUSES and opt_status in ("pending", None):
            return {"phase": "device", "block": c["name"], "worker": "optimization"}

    # ---- Rule 4: done -----------------------------------------------------
    if all(_ttnn_satisfied_for_completion(c) and _is_finished(c, "optimization") for c in components):
        return {"phase": "done"}

    # ---- Rule 5: deadlock -------------------------------------------------
    # Anything blocked at any phase is a candidate to report. We list them
    # in component order so the operator reads them top-down through the
    # DAG, with each entry's downstream closure sorted for determinism.
    blocking = []
    for c in components:
        is_blocked = any(
            _phase_status(c, phase) == "blocked" for phase in ("reference", "ttnn", "debug", "optimization")
        )
        if is_blocked:
            blocking.append(
                {
                    "name": c["name"],
                    "blocks_downstream": _downstream_of(state, c["name"]),
                }
            )
    return {"phase": "deadlock", "blocking": blocking}
