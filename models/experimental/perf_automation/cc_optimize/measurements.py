# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The measurement ledger: the ONE place a reported number comes from.

Every headline defect this tool has produced came from the same shape -- the report needing a
"before" number and not having one recorded, so it searched a chain of files and found something
with different provenance:

    eager per-op device time (all layers):  0.06 ms -> 648.17 ms  (-1062476.1%)
        a sub-millisecond anchor from an unrelated model's run, against a real 648 ms reading

    baseline 832.93 ms -> final 1088.15 ms  (-30.6%)
        a 2-layer profile paired with a 16-layer one; a regression that never happened

    before 47.10 ms [eager] -> after 100.00 ms [trace+1cq]
        two different units subtracted from each other

Each was fixed by hardening one link of the chain. The chain is the defect: a number reached the
report carrying no statement of WHAT it measured, so nothing could tell that it did not belong.

Here a measurement is APPENDED when it is taken, with its provenance, and the report reads only
this. There is nowhere to fall back to, so a foreign or stale value cannot be promoted into an
anchor; and two rows are subtracted only when they describe the same work.

THE MODELLED FLOOR IS KEPT HERE TOO (KIND_FLOOR), for the same reason the measured numbers are: it
is a value the report ANCHORS on, so it needs the earliest-reading-wins durability and the fitness
checks in `record` rather than a second store beside them. Recomputing it each round let the
optimized build lower its own target (537 -> 332 ms while measuring FASTER), so at-floor% fell during
a run that improved.

DURABILITY IS THE POINT. The ledger is keyed by (model, task) and is never truncated -- not by a
rerun, not by a fresh ladder, not by clearing the kernel log. `first("eager_per_op", "before")` is
the earliest before-reading ever taken for this model, so re-running optimize on an
already-optimized model still reports against the TRUE original. Without that, the second run
measures the optimized model, calls that its baseline, and the 2464 -> 648 result becomes
unreportable the moment you restart.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

_SCHEMA = 1
_MAX_KEY_LEN = 180

KIND_EAGER = "eager_per_op"
KIND_TRACE_PASS = "trace_pass"
KIND_FULLPIPE = "fullpipe_e2e"

KIND_FLOOR = "modeled_floor"

# Bytes streamed per unit of work -- the ceiling's numerator. Anchored here for the same reason the
# floor is: it must be the BASELINE figure, and it must survive. Kept in perf_target_inputs.json it
# was rolled back twice by the optimize loop's directory restore, which put a 16-layer ceiling next to
# a full-model measurement in the report.
KIND_ACTIVE_BYTES = "active_bytes"

# The COMPUTE roof's denominator -- peak FLOP/s at the math fidelity the model runs at. Anchored for
# the same reason the floor and the bytes are, and it was the one ceiling input that never was.
#
# Blackhole's peak spans 4x across the modes (LoFi 702, HiFi2 351, HiFi3 234, HiFi4 175.5 TFLOPS), so
# the `fidelity` rung moves this ceiling every time it lands. _promote_baseline already protects the
# BAR on exactly this reasoning -- "a re-profile must not redefine what wins are graded against" --
# and then refreshes the PICTURE the peak is read from, in the same function.
#
# Nothing announced it, which is why it outlived the other two. The memory roof divides by a fixed
# 512 GB/s, so drift there prints an impossible bandwidth and gets caught. Here the peak IS what
# moves, and the measurement moves with it: a stage that got 2x faster reports the same % of a
# ceiling that also doubled. The error and the win cancel, and the reader sees neither.
KIND_PEAK_FLOPS = "peak_flops"

# The MEMORY roof's numerator, per stage: the bytes trace_replay observed that stage's ops reading.
# Anchored for the third time for the same reason, because the reason has not changed and the shape
# of the mistake is identical each time:
#
#   "The floor is a property of the IMPLEMENTATION, not a goal: halving a weight's dtype halves the
#    bytes it must move, so recomputing it each round makes the target retreat ahead of the
#    measurement and it is never reached."
#
# Measuring the read set instead of inferring it makes the number RIGHT; it does nothing about the
# number MOVING, and the dtype rung moves it by construction -- bf16 -> bf8_b halves a weight, the
# observed bytes halve, and the ceiling follows the build down. Keyed per stage, because each stage
# has its own read set and decode's is the one that binds.
KIND_STAGE_BYTES = "stage_bytes"

# The COMPUTE roof's numerator: the parameters a matmul actually multiplies, per checkpoint section.
#
# THE CEILING IS PINNED OR IT IS NOT, and which roof binds is irrelevant to that. The other three
# inputs were anchored one at a time, each after it moved something: the floor, the bytes, the peak.
# This was the last one left loose, and it is loose for the same reason the others were -- blocks[]
# lives in the arch mirror, written `{**prev, **keep}`, last-write-wins. The mirror calls itself safe
# to cache without expiry because "a dtype or grid knob ... cannot change how many towers the model
# has", which is true of the towers and NOT true of matmul_params: that figure subtracts the gathers
# the profile OBSERVED, so a run that observes a different gather set recomputes it, and the compute
# roof moves under a measurement that did not.
#
# Pinned per SECTION rather than per stage: prefill and decode share a subtree and must not be able
# to disagree about how many parameters it multiplies.
KIND_MATMUL_PARAMS = "matmul_params"

# ITEMS ONE CALL OF A STAGE RETIRES -- the `tokens` in the compute floor's 2 x params x tokens.
# Anchored for the same reason every other ceiling input is: the THEORETICAL column must describe
# the state the campaign started from, and this one is re-observed from the run on every step. A
# prefill chunk that changes size would move the ceiling under the measurement chasing it.
KIND_STAGE_TOKENS = "stage_tokens"

PHASE_BEFORE = "before"
PHASE_AFTER = "after"


def is_win(attempt) -> bool:
    """Did this attempt make the model measurably faster? THE ONE definition of a win.

    Lives here, beside `record`, because it is the same kind of rule: what a claimed number must
    carry before anything may act on it. A win requires a MEASUREMENT, not a commit -- and the
    difference is not cosmetic:

      * the report set a tick from `beat_baseline` alone, so housekeeping commits (once even a
        comment-only one) rendered as wins: 48 of 75 "wins" in one report had never been timed;
      * three renderers in summary.py and the KV-cache GATE in perf_mcp each re-derived this, so a
        fix to one left the others claiming the opposite about the same attempt -- the gate's own
        docstring promised "clears ONLY on a MEASURED reduction" while its code checked the flag.

    Callers must not re-derive it. tests/test_single_source_of_truth.py fails if they do, because
    every defect in this class came from a second site rather than a wrong rule.
    """
    if not isinstance(attempt, dict):
        return False
    if not attempt.get("beat_baseline"):
        return False
    ms = attempt.get("measured_ms")
    if isinstance(ms, bool) or not isinstance(ms, (int, float)):
        return False
    return math.isfinite(ms) and ms > 0


def staircase_value(attempt) -> tuple:
    """(value, ruler) an attempt's win may be judged on. The RULER matters as much as the value.

    `measured_ms` is whatever the caller measured, and callers measure different things: a per-token
    trace reading for a host/dispatch lever, a per-profile device_ms sum for an op lever. Comparing
    them in one staircase is how a per-token 21.11 became the running best and every later per-profile
    290-354 row rendered "no gain" -- four git-committed wins shown as failures in one report.

    `fullpipe_ms` is the end-to-end number the win DEFINITION is already stated in (gate_set_new_best:
    full_pipeline_ms below the previous best), so when it is present it is the correct and
    unit-consistent thing to rank by. Otherwise fall back to measured_ms, in its own separate ruler.
    """
    for key, ruler in (("fullpipe_ms", "fullpipe"), ("measured_ms", "measured")):
        try:
            v = float(attempt.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0:
            return v, ruler
    return None, ""


def winning_indices(attempts, baseline_ms=None) -> set:
    """Indices of the attempts that ACTUALLY made the model faster. THE ONE win rule for a sequence.

    `is_win` answers "was this timed and kept", which is necessary but not sufficient: an attempt can
    be committed, timed, and still not have reduced anything. Reporting those as wins is how a run
    with 3 real improvements showed 75 ticks -- every one of them inviting the reader to believe a
    lever worked.

    A win here must be a NEW BEST: strictly faster than the baseline and than every win before it. So
    the ✓ marks read as the staircase the run actually walked down, and their count is the number of
    times the end-to-end time moved.

    Order matters, so this takes the attempt list rather than one row; callers must not re-derive it.

    WHEN THE ATTEMPT CARRIES ITS OWN VERDICT, THAT IS THE ANSWER. perf_mcp now makes the end-to-end
    comparison ONCE per attempt and stamps `fullpipe_delta_ms` (this attempt's own trace+1cq minus
    the running best); the sign is the verdict. Re-deriving a staircase here is what made three
    components disagree about the same row -- 16 raw flags, 3 ticks, 2 real improvements. The
    staircase below remains for logs written before that stamp existed.
    """
    stamped = [
        i
        for i, a in enumerate(attempts or [])
        if isinstance(a, dict) and isinstance(a.get("fullpipe_delta_ms"), (int, float))
    ]
    if stamped:
        return {i for i in stamped if attempts[i]["fullpipe_delta_ms"] < 0}

    # ONE STAIRCASE PER RULER. A single `best` across mixed units let the smallest-scoped reading win
    # once and then disqualify everything else; see staircase_value.
    best: dict = {}
    try:
        b = float(baseline_ms)
        if math.isfinite(b) and b > 0:
            best["measured"] = b
    except (TypeError, ValueError):
        pass
    out = set()
    for i, a in enumerate(attempts or []):
        if not is_win(a):
            continue
        ms, ruler = staircase_value(a)
        if ms is None:
            continue
        prev = best.get(ruler)
        if prev is None and ruler == "fullpipe":
            # SEED FROM THE GATE'S OWN PREVIOUS BEST -- same ruler, same scope. Crediting the first
            # end-to-end attempt unconditionally would mark a lever that merely held steady.
            try:
                _pb = float(a.get("fullpipe_best_ms"))
                prev = _pb if math.isfinite(_pb) and _pb > 0 else None
            except (TypeError, ValueError):
                prev = None
        # prev may still be None (no baseline, no gate best): the first timed attempt then STARTS this
        # ruler's staircase, which is deliberate -- test_without_a_baseline_the_first_timed_commit_starts
        # _the_staircase pins it, so a run with no recorded baseline still shows relative progress.
        if prev is None or ms < prev:
            out.add(i)
            best[ruler] = ms
    return out


def trace_ms_from_profile(profile) -> float | None:
    """The trace-pass latency carried by a device profile, or None. THE ONE extractor.

    Reported separately from the eager per-op number because they measure different things over the
    same window; collapsing them is how a 'regression' appears out of nowhere. Lives here so the
    writer and the renderer read the same key from the same shape: the renderer had its own copy and
    the writer's call was guarded on a name that was never defined, so the durable row was never
    written and the report fell back to reading the per-profile file -- which every profile
    overwrites, making a CURRENT number carry the word BASELINE.
    """
    if not isinstance(profile, dict):
        return None
    for key in ("per_token_ms", "trace_per_token_ms", "trace_ms"):
        try:
            v = float(profile.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0:
            return v
    return None


def _ledger_dir() -> Path:
    """Where keyed ledgers live. PERF_MCP_LEDGER_DIR redirects the whole namespace.

    PERF_MCP_LEDGER redirects ONE file, which is not enough to isolate a caller: the moment a test
    deletes it to exercise unkeyed behaviour, any KEYED call in the same test resolves back to the
    shared temp dir and writes a real ledger there. That is how a stray
    perf_measurements_named_model_main.jsonl -- from test_floor_anchor_writeonce's deliberate delenv
    -- appeared beside a live run's ledger and was mistaken for the run having split its anchors
    across two files. Redirecting the DIRECTORY survives the delenv, so the isolation cannot be
    switched off by accident.
    """
    # FALL BACK TO THE STATE DIR, not to bare /tmp. Nothing in the run sets PERF_MCP_LEDGER_DIR --
    # the MCP config carries PERF_MCP_STATE_DIR only -- so every production run resolved its ledger
    # to tempfile.gettempdir() while every other artifact went to the state dir. The anchors were
    # therefore written and read in a directory no one else looked at, and the report's "THE LEDGER
    # WINS" block could never win: anchor_value returned None on every real run and the ceiling fell
    # through to the throughput snapshot. On gemma-3-12b-it that printed 45.8 tok/s/u (512/11.18 GB,
    # the reverted-directory vintage) instead of 42.7 (512/12 GB, the operator-confirmed anchor) --
    # a 7% optimistic ceiling in every report written so far.
    #
    # PERF_MCP_LEDGER_DIR still redirects the whole namespace for test isolation; it is only the
    # DEFAULT that changes, from "somewhere in /tmp" to "beside the rest of this run's state".
    _explicit = os.environ.get("PERF_MCP_LEDGER_DIR")
    if _explicit:
        return Path(_explicit)
    _state = os.environ.get("PERF_MCP_STATE_DIR")
    return Path(_state) if _state else Path(tempfile.gettempdir())


def ledger_path(model: str = "", task: str = "") -> Path:
    """Keyed by (model, task), like every other per-run artifact. An unkeyed file is how another
    run's number became this run's baseline."""
    override = os.environ.get("PERF_MCP_LEDGER")
    if override:
        return Path(override)
    _keyed = bool(model or os.environ.get("PERF_MCP_MODEL_NAME") or os.environ.get("PERF_MCP_MODEL_ROOT"))
    model = (
        model
        or os.environ.get("PERF_MCP_MODEL_NAME")
        or Path(os.environ.get("PERF_MCP_MODEL_ROOT", "") or "model").name
    )
    task = task or os.environ.get("PERF_MCP_TASK", "main")
    if not _keyed and os.environ.get("PERF_MCP_STRICT_LEDGER_KEY") == "1":
        # An UNKEYED call silently produces perf_measurements_model_main.jsonl -- a file that looks
        # like a real ledger and belongs to no model. That is how ONE gemma-3-12b-it run ended up
        # with two: the write-once BEFORE/AFTER rule was then applied per FILE, so a committed-best
        # reading found no BEFORE in the model's own ledger, claimed that slot, and the report
        # announced the OPTIMIZED number as the baseline. Opt-in (tests/CI) so a new call site that
        # forgets the key fails loudly here rather than being discovered in a report months later;
        # production still degrades rather than crashing a long run over a ledger name.
        raise ValueError(
            "unkeyed ledger access: pass model=/task= (or set PERF_MCP_MODEL_NAME / "
            "PERF_MCP_MODEL_ROOT). An unkeyed call writes the shared 'model_main' ledger, which "
            "splits one run's anchors across two files."
        )
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", "%s_%s" % (model, task)).strip("_") or "model_main"
    if len(safe) > _MAX_KEY_LEN:
        # A long HF id (org/very-long-name) produced a filename over the 255-byte limit, so EVERY
        # write failed -- silently, since record() swallows OSError. Truncate but keep a digest so
        # two long ids that share a prefix still get different ledgers.
        digest = hashlib.sha1(safe.encode()).hexdigest()[:12]
        safe = "%s_%s" % (safe[: _MAX_KEY_LEN - 13], digest)
    return _ledger_dir() / ("perf_measurements_%s.jsonl" % safe)


def record(
    kind: str,
    phase: str,
    value_ms,
    *,
    depth: str = "",
    mode: str = "",
    stage: str = "",
    source: str = "",
    model: str = "",
    task: str = "",
    derived: bool = False,
) -> bool:
    """Append one measurement. Returns False when it is not worth recording.

    A reading with no depth or no mode is REFUSED rather than stored blank: an unlabelled number is
    exactly what the report cannot safely use, and storing it would rebuild the guessing problem
    inside the ledger.
    """
    try:
        v = float(value_ms)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(v) or v <= 0:
        # NaN slips past a `v <= 0` test -- every comparison with NaN is False -- so an unusable
        # reading could become the PERMANENT anchor, and no later run could dislodge it.
        return False
    if not str(depth).strip() or not str(mode).strip():
        return False
    row = {
        "schema": _SCHEMA,
        "kind": str(kind),
        "phase": str(phase),
        "value_ms": round(v, 4),
        # DERIVED IS NOT MEASURED. A value computed from other readings can be the only figure
        # available -- the baseline per-token latency was never recorded, so it could only be scaled
        # out of the baseline device_ms -- but the report renders value_ms and not source, so a
        # hand-written row was indistinguishable from a profiler reading. Flagged here so the
        # renderer can say which it is; a report that goes out for confirmation must not present
        # arithmetic as a measurement.
        "derived": bool(derived),
        "depth": str(depth).strip(),
        "mode": str(mode).strip(),
        "stage": str(stage or "").strip(),
        "source": str(source or "").strip(),
    }
    try:
        p = ledger_path(model, task)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        return True
    except Exception:  # noqa: BLE001
        return False


def rows(kind: str = "", phase: str = "", model: str = "", task: str = "") -> list:
    """Every matching row, oldest first. A malformed line is skipped, never fatal -- a corrupt
    ledger must degrade to 'not measured', not crash the report."""
    out = []
    try:
        text = ledger_path(model, task).read_text()
    except Exception:  # noqa: BLE001
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(r, dict):
            continue
        if kind and r.get("kind") != kind:
            continue
        if phase and r.get("phase") != phase:
            continue
        out.append(r)
    return out


def is_identified(model: str = "", task: str = "") -> bool:
    """Is the ledger being addressed a KNOWN one, rather than the unkeyed default?

    An anchor is permanent, so reading one out of the shared unkeyed file is how another run's number
    becomes this run's target -- the same defect that made a foreign 0.06 ms an anchor, one level up.
    An explicit model, or an explicit ledger path, counts as identified; nothing else does.
    """
    if str(model or "").strip():
        return True
    return bool(
        os.environ.get("PERF_MCP_LEDGER")
        or os.environ.get("PERF_MCP_MODEL_NAME")
        or os.environ.get("PERF_MCP_MODEL_ROOT")
    )


def anchor_value(kind: str, *, depth: str = "", model: str = "", task: str = ""):
    """READ the pinned anchor for (kind, depth), or None. Never writes.

    Split from `anchor` so RENDERING cannot mutate state. The renderer originally did the pinning,
    which made producing the report a side effect: the first report written pinned a value that every
    later report then inherited, whatever its own input said.
    """
    if not is_identified(model, task):
        return None
    d = str(depth)
    for r in rows(kind, PHASE_BEFORE, model, task):
        if str(r.get("depth")) == d:
            try:
                return float(r.get("value_ms"))
            except (TypeError, ValueError):
                return None
    return None


@contextlib.contextmanager
def _anchor_lock(model: str = "", task: str = ""):
    """Serialise the anchor's check-then-write across PROCESSES.

    `anchor` reads "is anything pinned?" and then appends -- two writers can both see nothing and both
    append, after which which value is pinned depends on which append landed first. Stress-testing 64
    concurrent writers pinned 1000.0 on one run and 1001.0 on the next, i.e. a NON-DETERMINISTIC
    baseline, which is the one thing an anchor exists to prevent. Two writers is a real configuration:
    the producer anchors at setup and the MCP server anchors when it rebuilds facts a revert deleted.

    Best-effort: an unavailable lock yields anyway rather than blocking a measurement.
    """
    f = None
    try:
        f = open(str(ledger_path(model, task)) + ".lock", "a+")
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    except Exception:  # noqa: BLE001
        pass
    try:
        yield
    finally:
        if f is not None:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:  # noqa: BLE001
                pass
            f.close()


def anchor(
    kind: str,
    value_ms,
    *,
    depth: str = "",
    mode: str = "roofline",
    source: str = "",
    model: str = "",
    task: str = "",
):
    """Pin `value_ms` for (kind, depth) if nothing is pinned yet; return whatever is pinned.

    Called by whoever PRODUCES the value, at the moment it is produced -- which is the only place
    that knows it describes the state the run started from. `record` decides whether the value is fit
    to be permanent, so this adds only the write-once rule.
    """
    if not is_identified(model, task):
        return None
    with _anchor_lock(model, task):
        held = anchor_value(kind, depth=depth, model=model, task=task)
        if held is not None:
            return held
        if record(kind, PHASE_BEFORE, value_ms, depth=depth, mode=mode, source=source, model=model, task=task):
            return float(value_ms)
    return None


def first(kind: str, phase: str = PHASE_BEFORE, model: str = "", task: str = ""):
    """The EARLIEST matching reading -- the true original, surviving every rerun."""
    rs = rows(kind, phase, model, task)
    return rs[0] if rs else None


def last(kind: str, phase: str = PHASE_AFTER, model: str = "", task: str = ""):
    """The most recent matching reading -- the current state."""
    rs = rows(kind, phase, model, task)
    return rs[-1] if rs else None


def comparable(a, b) -> tuple:
    """(ok, why). Two readings may be subtracted only when they describe the SAME work: same depth,
    same mode, same stage. This is the structural version of the checks that were previously spread
    across the renderer as magnitude heuristics and mode string comparisons."""
    if not a or not b:
        return False, "not measured"
    for axis in ("depth", "mode", "stage"):
        av, bv = str(a.get(axis) or ""), str(b.get(axis) or "")
        if av != bv:
            return False, "%s differs: %s vs %s" % (axis, av or "unknown", bv or "unknown")
    return True, "comparable"


def delta_pct(a, b):
    """Percent improvement from a -> b, or None when the pair may not be compared."""
    ok, _ = comparable(a, b)
    if not ok:
        return None
    before, after = float(a.get("value_ms") or 0.0), float(b.get("value_ms") or 0.0)
    if before <= 0 or after <= 0:
        return None
    return (before - after) / before * 100.0
