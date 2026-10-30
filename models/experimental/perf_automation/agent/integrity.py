"""Shared integrity primitives: one source of truth for the four defect shapes found in the
2026-07-25 audits. ~50 distinct bugs reduced to four, and every one of them was a site-local
decision that should never have been site-local.

  1. A CLASSIFIER whose terminal branch returns a REAL category.
       normalize_memory("")     -> "dram_interleaved"   (a routing dimension)
       _bytes_per_elem("fp8")   -> 2.0                  (moved the tok/s ceiling -> premature DONE)
       ARCH_FACTS.get(a, blackhole)                     (wrong roofline floor for any other arch)
     RULE: the terminal branch is UNKNOWN. A caller that cannot handle UNKNOWN must say so.

  2. A GUARD initialised to the PASSING value and wrapped in `except`.
       measurement_ok = True; try: ... except: pass
     A guard that could not run has NOT cleared anything.
     RULE: PASS / FAIL / UNKNOWN, and UNKNOWN is never truthy.

  3. A NUMERIC DEFAULT flowing into comparison arithmetic.
       float(prof.get("device_ms") or 0.0)  -> 0.0 read as "infinitely fast" -> 100% win, cached
     RULE: an absent measurement is not a number.

  4. POSITIVE-MARKER detection whose non-match means "fine".
       bool(_parse_trace_path(out))  -> bool("eager") is True -> eager banked as trace
     RULE: absence of the marker is UNKNOWN, not OK.

  5. A NUMBER COMPARED WITHOUT ITS PROVENANCE (2026-07-27 audit).
       baseline 832.93 -> final 1088.15 (-30.6%)   2-layer profile vs a 16-layer one
       before 47.10 -> after 100.00 (-112.3%)      eager wall-clock vs trace+1cq per-token
       0.0612 ms pinned as the permanent baseline  an empty capture, correctly stamped
       714.94 -> 714.94 (+0.0%)                    anchor fell back to the CURRENT value
     Four separate headlines, all reporting a change that never happened, all fixed one at a time
     with a bespoke check -- which is how the fifth arrives. A ms figure is meaningless without the
     work it covers (DEPTH), the method that produced it (MODE) and where in the run it was taken
     (STAGE). Two numbers may only be differenced when all three agree.
     RULE: carry provenance with the value; refuse the delta, never guess.

And one cross-cutting lesson worth its own primitive: a single predicate with a default was used for
two decisions needing OPPOSITE conservatism ("reset the board?" vs "was this lever fairly tried?").
Fixing one broke the other. Three states, each caller picking its own safe side, is the fix.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# ONE state directory for every durable temp artifact -- see cc_optimize/tmpstate.py.
import importlib.util as _ilu_ts

_ts_spec = _ilu_ts.spec_from_file_location(
    "_tmpstate", str(Path(__file__).resolve().parent.parent / "cc_optimize" / "tmpstate.py")
)
_tmpstate = _ilu_ts.module_from_spec(_ts_spec)
_ts_spec.loader.exec_module(_tmpstate)
state_dir = _tmpstate.state_dir


class _Unknown:
    """Falsy, self-describing sentinel: `if classify(...)` cannot silently pass."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "UNKNOWN"

    def __str__(self) -> str:
        return "unknown"


UNKNOWN = _Unknown()


def ask_cli(prompt: str, timeout: int = 120) -> str:
    """Ask the local `claude` CLI one question and return its text. "" when it cannot be asked.

    THE SAME FOUR LINES, THREE TIMES -- twice in this file and once in summary.py: resolve the
    binary, run it with -p and --output-format text, guard the timeout, take stdout. What each
    caller does with the answer differs and is worth keeping separate; getting the answer is not.
    """
    import shutil as _shutil
    import subprocess as _sp

    claude = _shutil.which("claude")
    if not claude:
        return ""
    try:
        r = _sp.run([claude, "-p", prompt, "--output-format", "text"], capture_output=True, text=True, timeout=timeout)
    except Exception:  # noqa: BLE001 -- an unanswerable question is "", never a failed run
        return ""
    return (r.stdout or "") if r.returncode == 0 else ""


def normalise(token) -> str:
    """The ONE token normaliser for rungs, levers, dtypes and op classes.

    Shape 1's most expensive instance was a prefix mismatch: rungs are minted `"knob:grid"` and were
    compared `== "grid"`, so `*_tries` never incremented, `_MAX_KNOB_RETRIES` never saturated and the
    ladder re-issued the same rung forever. The same mismatch appeared as a report column, a dtype
    spelling (`DataType.BFLOAT8_B`), and an op_class. One normaliser, used everywhere, or it recurs.
    """
    s = str(token or "").strip().lower()
    for sep in (":", "."):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return " ".join(s.split())


def _cache_path():
    """Resolved per call: a module constant freezes the path at import, before any redirect."""
    return state_dir() / "perf_integrity_resolve_cache.json"


def _cache_get(key: str):
    try:
        return json.loads(_cache_path().read_text()).get(key)
    except Exception:  # noqa: BLE001
        return None


def _cache_put(key: str, val: str) -> None:
    try:
        cur = {}
        if _cache_path().exists():
            cur = json.loads(_cache_path().read_text())
        cur[key] = val
        tmp = _cache_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(cur))
        tmp.replace(_cache_path())
    except Exception:  # noqa: BLE001
        pass


def _ask_agent(question: str, options) -> str:
    """One Claude Code call, answer constrained to `options`. Empty string when unavailable."""
    if os.environ.get("PERF_MCP_NO_AGENT_CLASSIFY") == "1":
        return ""
    opts = [str(o) for o in options]
    prompt = (
        question
        + "\n\nAnswer with EXACTLY ONE of these, and nothing else: "
        + ", ".join(opts)
        + "\nIf none genuinely applies, answer: unknown"
    )
    try:
        _out = ask_cli(prompt)
        ans = normalise(_out.strip().splitlines()[-1] if _out.strip() else "")
        return ans if ans in {normalise(o) for o in opts} else ""
    except Exception:  # noqa: BLE001
        return ""


def classify(value, vocab, what: str = "value", evidence: str = ""):
    """Map a free-form value onto `vocab`, reasoning about MEANING rather than spelling.

    Exact and normalised matches resolve locally (free, stable). Anything unrecognised is resolved
    ONCE by a Claude Code agent and cached by content hash -- because the failures this replaces were
    all spelling drift: `knob:grid` vs `grid`, `DataType.BFLOAT8_B` vs `bfloat8_b`, `bfp8_b`, an
    op_class the router has never seen, a TT_FATAL phrased a new way. A hand-maintained alias table
    misses the next spelling by construction; that is why this is not a table.

    Returns a member of `vocab`, or UNKNOWN. UNKNOWN is falsy and is never a guessed real category.
    """
    key = normalise(value)
    if not key:
        return UNKNOWN
    canon = {normalise(v): v for v in vocab}
    if key in canon:
        return canon[key]

    if os.environ.get("PERF_MCP_NO_AGENT_CLASSIFY") == "1":
        return UNKNOWN
    ck = hashlib.sha256(("|".join((what, key, evidence[:400], *sorted(canon)))).encode()).hexdigest()[:32]
    hit = _cache_get(ck)
    if hit is not None:
        return canon.get(hit, UNKNOWN) if hit else UNKNOWN

    ans = _ask_agent(
        "In a Tenstorrent TTNN model-performance tool, map this %s onto one of the listed "
        "categories by what it MEANS, not by string similarity.\n\n%s: %r%s"
        % (what, what, str(value), ("\nevidence: " + evidence[:400]) if evidence else ""),
        list(canon),
    )
    _cache_put(ck, ans)
    return canon.get(ans, UNKNOWN) if ans else UNKNOWN


def ask_number(question: str, lo: float, hi: float, cache_key: str = "") -> float:
    """A numeric estimate from the agent, clamped to [lo, hi]. 0.0 when unavailable.

    Used for budgets that have no observation to scale from yet. The alternative -- a table of
    per-operation multipliers -- is a guess about every future model baked in at authoring time,
    which is what made a 240 s build cap collide with llama's real 872 s build.
    """
    if os.environ.get("PERF_MCP_NO_AGENT_CLASSIFY") == "1":
        return 0.0
    ck = hashlib.sha256(("num|" + (cache_key or question)).encode()).hexdigest()[:32]
    hit = _cache_get(ck)
    if hit is not None:
        try:
            return max(lo, min(hi, float(hit)))
        except (TypeError, ValueError):
            return 0.0
    # ask_cli resolves the binary and returns "" when there is none.
    prompt = (
        question
        + "\n\nAnswer with ONE integer number of seconds and nothing else. It must be between "
        + "%d and %d. Prefer a GENEROUS estimate: a budget that is too tight kills healthy work, "
        "while one that is too loose only delays detection (a separate watchdog judges liveness "
        "from evidence)." % (int(lo), int(hi))
    )
    try:
        digits = "".join(ch for ch in ask_cli(prompt) if ch.isdigit() or ch == " ").split()
        if not digits:
            return 0.0
        val = max(lo, min(hi, float(digits[-1])))
        _cache_put(ck, str(int(val)))
        return val
    except Exception:  # noqa: BLE001
        return 0.0


PASS, FAIL = "pass", "fail"


class Verdict:
    """A guard result that cannot be mistaken for a pass.

    `bool(v)` is True ONLY for an explicit PASS, so the shape-2 pattern
    `ok = True; try: ok = guard() except: pass` becomes impossible: an UNKNOWN verdict is falsy and
    carries the reason it could not decide.
    """

    __slots__ = ("state", "reason")

    def __init__(self, state, reason: str = ""):
        self.state = state
        self.reason = reason

    @classmethod
    def passed(cls, reason: str = "") -> "Verdict":
        return cls(PASS, reason)

    @classmethod
    def failed(cls, reason: str) -> "Verdict":
        return cls(FAIL, reason)

    @classmethod
    def unknown(cls, reason: str) -> "Verdict":
        return cls(UNKNOWN, reason)

    @property
    def is_pass(self) -> bool:
        return self.state == PASS

    @property
    def is_unknown(self) -> bool:
        return self.state is UNKNOWN

    def __bool__(self) -> bool:
        return self.is_pass

    def __repr__(self) -> str:
        return "Verdict(%s%s)" % (self.state, (": " + self.reason) if self.reason else "")


class Unmeasured(Exception):
    """Raised when an absent measurement is used as a number."""


class Measurement:
    """A number that was measured, or an explicit absence.

    Shape 3: `float(prof.get("device_ms") or 0.0)` turned a failed capture into 0.0 ms, which read as
    infinitely fast -- a 100% win, written back as the new baseline and cached, so every later
    comparison used base 0.0. An absence must not be arithmetic-compatible.
    """

    __slots__ = ("_value", "reason")

    def __init__(self, value, reason: str = ""):
        self._value = value
        self.reason = reason

    @classmethod
    def measured(cls, value) -> "Measurement":
        v = float(value)
        if v <= 0.0:
            return cls(None, "non-positive measurement (%r) is an absence, not a speed" % (value,))
        return cls(v)

    @classmethod
    def unmeasured(cls, reason: str) -> "Measurement":
        return cls(None, reason)

    @property
    def ok(self) -> bool:
        return self._value is not None

    @property
    def value(self) -> float:
        if self._value is None:
            raise Unmeasured(self.reason or "no measurement")
        return self._value

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        return "Measurement(%s)" % (self._value if self.ok else "unmeasured: " + self.reason)


class Reading:
    """A measured ms value together with everything needed to know what it may be compared to.

    Every reporting bug found in the 2026-07-27 audit was the same shape: a bare float, correct in
    isolation, differenced against another bare float that measured something else. Naming the axes
    once -- and refusing the subtraction when they disagree -- removes the whole class rather than the
    instance, so a number added to the report later inherits the guard instead of needing its own.

    depth  how much of the model the number covers ("16", "all"). The roofline floor SUMS per-op
           floors over the profiled window, and device_ms scales with it, so a 2-layer figure is
           simply a different quantity from a 16-layer one.
    mode   how it was produced ("eager", "trace+1cq"). The same field carries a per-token decode step
           in one mode and a whole-forward wall clock in another.
    stage  when in the run it was taken ("baseline", "current"). An anchor must predate every lever;
           falling back to the current value silently reports +0.0%.
    """

    __slots__ = ("value", "depth", "mode", "stage", "source")

    def __init__(self, value, depth: str = "", mode: str = "", stage: str = "", source: str = ""):
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = None
        self.value = v if (v is not None and v > 0) else None
        self.depth = _norm_axis(depth)
        self.mode = _norm_axis(mode)
        self.stage = _norm_axis(stage)
        self.source = source

    @property
    def ok(self) -> bool:
        return self.value is not None

    def __bool__(self) -> bool:
        return self.ok

    def comparable_to(self, other) -> "Verdict":
        """May these two be differenced? PASS / FAIL / UNKNOWN, and UNKNOWN is never truthy.

        An axis unknown on BOTH sides is tolerated -- legacy readings carry no provenance and refusing
        every one of them would be noise. An axis known on ONE side only is NOT assumed to match: that
        is precisely the case where a mode or depth changed underneath a value captured earlier.
        """
        if not isinstance(other, Reading):
            return Verdict.unknown("not a Reading")
        if not self.ok or not other.ok:
            return Verdict.unknown("one side has no usable measurement")
        for axis in ("depth", "mode", "stage"):
            a, b = getattr(self, axis), getattr(other, axis)
            if a and b and a != b:
                return Verdict.failed("%s differs: %s vs %s" % (axis, a, b))
            if bool(a) != bool(b):
                return Verdict.unknown("%s known on one side only (%r vs %r)" % (axis, a, b))
        return Verdict.passed("comparable")

    def delta_pct_vs(self, other):
        """Improvement of THIS reading against `other` as a percentage, or None when not comparable.

        None is the whole point: the callers that fabricated regressions all computed a percentage
        from two numbers they had no business subtracting.
        """
        if not self.comparable_to(other).is_pass:
            return None
        return (other.value - self.value) / other.value * 100.0

    def label(self) -> str:
        """How the value should be written in a report: never bare."""
        if not self.ok:
            return "n/a"
        bits = [
            b for b in (self.depth and "%s layers" % self.depth if self.depth.isdigit() else self.depth, self.mode) if b
        ]
        return "%.2f ms%s" % (self.value, (" [%s]" % ", ".join(bits)) if bits else "")

    def __repr__(self) -> str:
        return "Reading(%s, depth=%r, mode=%r, stage=%r)" % (self.value, self.depth, self.mode, self.stage)


def _norm_axis(v) -> str:
    return str(v or "").strip().lower()


_WIN_FRAC = float(os.environ.get("PERF_MCP_WIN_FRAC", "0.01") or "0.01")
_WIN_ABS_MS = float(os.environ.get("PERF_MCP_WIN_ABS_MS", "0.05") or "0.05")


def win_threshold(baseline_ms, spread_ms=None) -> float:
    """Smallest delta worth calling a win, in the metric's own units.

    A fixed absolute floor is unit-agnostic: 0.05 ms against llama's 2266 ms baseline accepts three
    thousandths of one percent, far inside this hardware's documented thermal drift, so noise was
    banked as a win. Scale with the baseline, never claim a win inside a measured spread, and keep
    the absolute value only as a floor for very fast modules.
    """
    floors = [_WIN_ABS_MS]
    for v in (baseline_ms and abs(float(baseline_ms)) * _WIN_FRAC, spread_ms and abs(float(spread_ms))):
        try:
            if v:
                floors.append(float(v))
        except (TypeError, ValueError):
            pass
    return max(floors)


def two_sided(evidence_a: bool, evidence_b: bool, a, b):
    """Pick between two states from POSITIVE evidence only, else UNKNOWN.

    The cross-cutting lesson: one predicate with a default served two decisions needing opposite
    conservatism, so fixing one broke the other. Force both sides to be evidenced and make the
    caller choose what UNKNOWN means for its own decision.
    """
    if evidence_a and not evidence_b:
        return a
    if evidence_b and not evidence_a:
        return b
    return UNKNOWN
