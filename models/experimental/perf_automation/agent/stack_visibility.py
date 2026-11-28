# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A model hides a block stack; the reference it carries says so, and the agent makes it visible.

THE WALK'S RULE IS NARROWER THAN REALITY. find_all_stacks accepts a list of same-typed objects, or a
hybrid whose classes share a base. A pipeline that wraps each layer in a DIFFERENT class -- a counting
proxy here, a parts-assembled layer there -- holds a real stack that reads as three unrelated
objects, so the walk never sees it. Everything downstream inherits that: one coverage number for a
model with several sections, one knob, one stack capped, the rest profiled at full depth, and no
error anywhere because each step did its job with what it was given.

WIDENING THE RULE BY INFERENCE DOES NOT WORK, and this module exists because two attempts proved it.
Comparing attribute sets scored every pair of torch modules as identical (they all carry
_parameters, _buffers, _modules, training), so three unrelated top-level submodules registered as a
stack and shadowed the real ones. Comparing child-module names instead, with framework internals
excluded, still could not separate "three wrappers around one kind of layer" from "three submodules
of a model" -- and the mean similarity included the reference compared with itself, so any two-element
list passed automatically. Both attempts made the walk worse than leaving it alone.

WHAT IS MEASURABLE IS THE DISCREPANCY. A pipeline that holds its reference model -- and one that
builds from HF weights necessarily does -- carries stacks the walk CAN see, because torch holds them
as ModuleLists of one class. Two reference stacks against one device stack is a fact, not a guess: it
does not say what the missing stack looks like, only that the model has more sections than the device
side exposes. That is exactly enough to ask for a repair, and it needs no naming rule, no config and
no per-model code.

The repair is the same mechanism the depth knob already uses: hand the agent what was measured, then
re-measure. An agent reading the source can see that a list is built from `LM.layers[0..2]` and run
in sequence by the forward -- the distinction inference could not make from outside.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

CENSUS_TOKEN = "PERF_STACK_CENSUS="

_SYSTEM = (
    "You are editing a TTNN model pipeline so its repeated block stacks are discoverable by a "
    "structural walk. Make the smallest change that works: do not restructure the model, do not "
    "touch stub bodies, and do not change any numerics."
)


def split_stacks(stacks) -> tuple:
    """(device stacks, reference stacks) from one walk.

    Reference stacks are torch modules: weights the pipeline builds FROM, which never dispatch a ttnn
    op. They are useless for sizing a profiling window and are the only independent statement of how
    many block stacks the model actually has.
    """
    try:
        import torch as _t

        mod = _t.nn.Module
    except Exception:  # noqa: BLE001
        mod = ()
    dev, ref = [], []
    for st in stacks or []:
        blocks = getattr(st, "stack", None) or []
        if not blocks:
            continue
        head = blocks[0]
        if mod and isinstance(head, mod):
            ref.append(st)
        elif callable(head):
            dev.append(st)
    return dev, ref


def census(stacks) -> str:
    """The walk's result as one parseable line, emitted by the probe.

    THE WALK RUNS IN THE PROBE, NOT IN THE RUN. Only the probe process holds the built pipeline, and
    the built pipeline is the only place the stacks exist as objects -- by the time the run sees a
    result it has a signpost sequence and a list of op signatures, from which the REFERENCE stacks
    are already gone, filtered out before tagging because they never dispatch an op. Filtering them
    is right for sizing and destroys the only evidence of what is missing. So the probe states both
    kinds before it filters, and the run reads the census rather than re-deriving it.
    """
    dev, ref = split_stacks(stacks)
    rows = [_row("reference", st) for st in ref] + [_row("device", st) for st in dev]
    return CENSUS_TOKEN + json.dumps(rows)


def _row(kind, st) -> dict:
    blocks = getattr(st, "stack", None) or []
    head = blocks[0] if blocks else None
    return {
        "kind": kind,
        "path": str(getattr(st, "path", "?")),
        "blocks": len(blocks),
        "cls": type(head).__name__ if head is not None else "?",
    }


def parse_census(raw: str) -> list:
    """Rows from a probe's output; [] when the probe emitted none (an older probe, or a dead run)."""
    for line in reversed((raw or "").splitlines()):
        if line.startswith(CENSUS_TOKEN):
            try:
                rows = json.loads(line[len(CENSUS_TOKEN) :])
            except (ValueError, TypeError):
                return []
            return [r for r in rows if isinstance(r, dict)]
    return []


def hidden_stack_count(stacks, expected: int = 0) -> int:
    """How many block stacks exist that the device side does not expose.

    Accepts either the walk's stack objects (in the probe) or census rows (in the run).

    `expected` is any independent count of the model's sections -- the checkpoint's key structure, or
    the containers observed to actually run -- for models that carry no HF reference at all. The
    strongest witness wins rather than the first one available, so a model with a reference AND a
    checkpoint is held to whichever declares more.

    Never negative: a device side with MORE stacks than declared is not a defect -- a pipeline may
    split one reference stack across two resident towers, which is what Voxtral does.
    """
    dev, ref = stack_counts(stacks)
    return max(0, max(ref, int(expected or 0)) - dev)


def _rows(stacks) -> list:
    """Census rows from either representation, so one implementation serves both sides."""
    if stacks and isinstance(stacks[0], dict):
        return list(stacks)
    dev, ref = split_stacks(stacks)
    return [_row("reference", st) for st in ref] + [_row("device", st) for st in dev]


def stack_counts(stacks) -> tuple:
    rows = _rows(stacks or [])
    return (
        sum(1 for r in rows if r.get("kind") == "device"),
        sum(1 for r in rows if r.get("kind") == "reference"),
    )


def _describe(stacks) -> str:
    lines = [
        "  %-10s %-44s %d blocks of %s"
        % (r.get("kind", "?"), r.get("path", "?"), r.get("blocks", 0), r.get("cls", "?"))
        for r in sorted(_rows(stacks or []), key=lambda r: r.get("kind") != "reference")
    ]
    return "\n".join(lines) or "  (none)"


def repair_prompt(stacks, evidence=None) -> str:
    """The task, stated as what the walk measured.

    `evidence` is whatever else the run knows about the model's structure and does not need an HF
    reference to know: the section counts its checkpoint declares, and the containers observed to
    actually run. A model with no reference at all is repaired from those alone, and where they name
    a PATH the agent is pointed at a specific list instead of asked to search.
    """
    dev, ref = stack_counts(stacks)
    return (
        "A structural walk of the object build_pipeline returns found these repeated block "
        "stacks:\n\n%s\n\n%s"
        "The model has at least %d block stack(s); the device side exposes %d. The missing one(s) "
        "exist and run -- they are simply not held in a shape the walk recognises.\n\n"
        "The walk counts a list as ONE stack when its elements are the SAME CLASS (any length), or "
        "when their classes share a common base AND the list holds at least 4 elements and at most "
        "3 distinct classes. A list of differently-typed per-layer wrappers reads as unrelated "
        "objects and is skipped.\n\n"
        "Find the list(s) in tt/pipeline.py that hold the layers for the missing stack -- they are "
        "the ones built from the model's weights and run in sequence by the forward -- and "
        "make them read as one stack. PREFER ONE CLASS for all the wrappers: a shared base is not "
        "enough for a list that can hold 3 blocks or fewer, and these lists get short exactly when "
        "the profiler caps the depth, which is when the walk matters most. If one class is not "
        "workable, an empty common base is the fallback and costs nothing at runtime.\n\n"
        "Do not merge the wrappers, do not change what any of them does, and do not touch numerics. "
        "Edit tt/pipeline.py only." % (_describe(stacks), _evidence(evidence), max(ref, _expected(evidence)), dev)
    )


def _expected(evidence) -> int:
    """How many stacks the non-walk witnesses say exist. 0 when none of them spoke."""
    if not evidence:
        return 0
    return max(len(evidence.get("checkpoint") or {}), len(evidence.get("observed") or []))


def _evidence(evidence) -> str:
    """The witnesses that need no HF reference, written out for the agent.

    The observed containers are the useful half: they name the exact attribute path that RAN and how
    many blocks it holds, so the agent is told which list to fix rather than asked to find it. The
    checkpoint half says how many sections the weights describe, which is the count the walk is being
    checked against.
    """
    if not evidence:
        return ""
    out = []
    obs = evidence.get("observed") or []
    if obs:
        out.append(
            "Containers that RAN as repeated blocks (observed by bracketing each call, so this "
            "is execution, not inference) -- any of these missing from the walk above is the "
            "stack to fix:"
        )
        for o in obs:
            out.append("  %-44s %d blocks ran" % (o.get("path", "?"), o.get("depth", 0)))
    ck = evidence.get("checkpoint") or {}
    if ck:
        out.append("The checkpoint's own keys declare %d section(s):" % len(ck))
        for name, depth in sorted(ck.items()):
            out.append("  %-44s %d deep" % (name, depth))
    return "\n".join(out) + "\n\n" if out else ""


def retry_prompt(stacks) -> str:
    """Feedback after a re-walk: the stacks as they now read, and what is still absent.

    A parsed fact rather than a verdict -- the same reason the depth repair states the factory's
    parameter list instead of saying it failed. An agent told "it did not work" repeats itself; an
    agent told what the walk now returns fixes what the walk reads.
    """
    dev, ref = stack_counts(stacks)
    return (
        "After the edit the walk now returns:\n\n%s\n\n"
        "That is still %d reference stack(s) against %d device stack(s). Check EVERY element of the "
        "list you edited, including any built through a helper or a proxy wrapper: if a wrapper "
        "forwards to an inner object, the WRAPPER's class is what the walk sees, not the inner one. "
        "A shared base also does NOT work below 4 elements -- if the list can be that short, the "
        "wrappers have to be one single class.\n\n"
        "Keep the work already done; only make the remaining list read as one stack." % (_describe(stacks), ref, dev)
    )


def repair(model_root, stacks, timeout_s: int = 1200, feedback: bool = False, evidence=None) -> dict:
    """Ask the agent to make the hidden stack visible. One round; the caller re-walks and may retry.

    `feedback` sends the retry wording instead of the opening task: the same discipline the depth
    repair follows, where the second round states what the walk NOW returns rather than that the
    first round failed. An agent told it failed repeats itself.

    Returns {"attempted": bool, "hidden_before": int}. Whether it worked is not decided here -- the
    caller re-walks the rebuilt model, which is the only evidence that cannot be talked past.
    """
    from .agent_bin import resolve_claude_bin

    root = Path(model_root)
    hidden = hidden_stack_count(stacks, _expected(evidence))
    if hidden <= 0 or not (root / "tt" / "pipeline.py").is_file():
        return {"attempted": False, "hidden_before": hidden}
    env = dict(os.environ)
    for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(k, None)
    cmd = [
        resolve_claude_bin(),
        "-p",
        retry_prompt(stacks) if feedback else repair_prompt(stacks, evidence),
        "--system-prompt",
        _SYSTEM,
        "--allowedTools",
        "Read,Write,Edit,Glob,Grep",
        "--permission-mode",
        "bypassPermissions",
        "--max-turns",
        os.environ.get("PERF_MCP_STACK_REPAIR_TURNS", "40"),
        "--output-format",
        "text",
    ]
    try:
        subprocess.run(cmd, cwd=str(root), env=env, capture_output=True, text=True, timeout=timeout_s)
    except Exception:  # noqa: BLE001 -- a repair that cannot run leaves the model as it was
        return {"attempted": True, "hidden_before": hidden}
    return {"attempted": True, "hidden_before": hidden}
