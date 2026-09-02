# SPDX-License-Identifier: Apache-2.0
"""End-of-run optimization summary for the cc engine.

Reads the per-op kernel-attempts log + the baseline profile and renders a table of what was attempted
at each ladder level (grid / dtype / tt-lang / cpp / host) per op, the best device_ms reached, and the
overall old->new runtime with the percentage speedup. Pure stdlib; additive (touches no opt logic).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

# ONE state directory for every durable temp artifact -- see cc_optimize/tmpstate.py. Loaded by path
# because cc_optimize is not a package: these modules run both as scripts and as plain imports.
import importlib.util as _ilu_ts

_ts_spec = _ilu_ts.spec_from_file_location("_tmpstate", str(Path(__file__).resolve().parent / "tmpstate.py"))
_tmpstate = _ilu_ts.module_from_spec(_ts_spec)
_ts_spec.loader.exec_module(_tmpstate)
state_dir = _tmpstate.state_dir


# THE LADDER HAS EIGHT RUNGS AND THIS SHOWED SIX OF THEM.
#
# perf_mcp._RUNG_PRIORITY is the climb order -- grid, fidelity, dtype, shard, host, structural,
# tt-lang, cpp -- and `structural` had no column here. _HOST_KINDS folded it into `host`, so an
# ALGORITHMIC RESTRUCTURE and a trace/dispatch fix rendered as the same lever: run 13 shows
# ReshapeView and TilizeWithValPadding with a host win, and the ledger records both as structural.
# `tp-fracture`, which the ladder also mints, fell through to the anonymous `other` bucket, and so
# did every one of the specific structural levers the gates hand out (structural-conv,
# structural-fold, structural-order, kv-cache/structural-decode). Four different levers, one
# indistinguishable column.
#
# So structural is its own column, and the specific gate kinds resolve to it rather than to `other`.
# `host` keeps trace/2-CQ/dispatch work, which is what it always meant.
_LEVEL_COLS = ("grid", "fidelity", "dtype", "shard", "host", "structural", "tp-fracture", "tt-lang", "cpp")
_ALL_COLS = _LEVEL_COLS + ("other",)  # "other" holds unclassifiable levers; rendered only when used
_HOST_KINDS = {"trace", "fusion", "fuse", "cache"}
# Every lever the gates mint is a structural one; they are named for WHICH restructure, so a report
# can tell a conv-weight prep from a KV-cache from a fold. Kept beside _HOST_KINDS because the two
# together decide the column, and splitting them is how `structural` ended up inside `host`.
_STRUCTURAL_KINDS = {
    "structural",
    "structural-conv",
    "structural-fold",
    "structural-order",
    "structural-decode",
    "conv-prep",
    "fold",
    "order",
    "kv-cache",
    "gather",
    "sparse",
}

_REPORT_NAME = "RUN_REPORT.md"


def _runs_root() -> Path:
    return Path(__file__).resolve().parent.parent / "runs"


def _latest_belongs_to(latest: Path, model_root) -> bool:
    """Is the run behind ``runs/latest`` the run for ``model_root``?

    report_path returned the run directory for ANY caller, ignoring the model_root it was handed.
    That is right for the run itself and wrong for everyone else: the tool's own suite passes a
    temporary model directory and expects the report there, so every test run overwrote whatever
    real report `latest` happened to name -- observed replacing a finished 6.2 KB report with a
    57-byte fixture. A run declares its model in its manifest, so the pointer can be checked rather
    than assumed.

    Unreadable or silent manifests keep the previous behaviour: a run that has not written one yet
    still gets its own directory, which is the case this function must not regress.
    """
    try:
        declared = (json.loads((latest / "manifest.json").read_text()).get("config") or {}).get("model_root")
    except (OSError, ValueError, AttributeError):
        return True
    if not str(declared or "").strip():
        return True
    try:
        return Path(declared).resolve() == Path(model_root).resolve()
    except OSError:
        return True


def report_path(model_root) -> Path:
    """Where RUN_REPORT.md is written: the RUN directory when there is one.

    The report used to live in the model directory, INSIDE the optimize worktree, where git can see
    it. The first commit of a run sweeps up whatever is untracked, so the report -- a generated
    artifact -- landed in commit e952cec6ce beside the lever's source change. Later commits stage
    only the file a lever touched, so it was never re-committed: permanently "modified, unstaged".
    Every git_revert after a no-gain attempt then discarded those modifications and restored the
    committed blob, rewinding a live report from 30 attempts back to the 7 it held when that first
    commit was made. Observed on gemma-3-12b-it at 2026-07-31 21:47:10, five seconds after a correct
    render, and it stayed wrong until the next attempt re-rendered it.

    The run directory cannot suffer this: models/experimental/perf_automation/.gitignore ignores
    `runs/`, nothing under it is tracked, and git does not restore what it cannot see. It is also
    per-run, so one run's report never overwrites another's, and that .gitignore is committed in the
    repo, so every worktree inherits the protection with no per-worktree setup to forget.

    Falls back to the model directory when no run directory exists, so callers outside a run (tests,
    a bare render) behave exactly as before.
    """
    latest = _runs_root() / "latest"
    try:
        if latest.is_dir() and _latest_belongs_to(latest, model_root):
            return latest / _REPORT_NAME
    except OSError:  # noqa: BLE001
        pass
    return Path(model_root) / _REPORT_NAME


def upsert_report_section(model_root, key: str, block_md: str):
    try:
        path = report_path(model_root)
        begin, end = f"<!-- BEGIN {key} -->", f"<!-- END {key} -->"
        block = f"{begin}\n{block_md.strip()}\n{end}"
        existing = path.read_text() if path.exists() else ""
        if begin in existing and end in existing:
            pre = existing.split(begin, 1)[0].rstrip()
            post = existing.split(end, 1)[1].lstrip()
            parts = [p for p in (pre, block, post) if p]
        else:
            parts = [existing.strip(), block] if existing.strip() else [block]
        path.parent.mkdir(parents=True, exist_ok=True)
        _text = "\n\n".join(parts) + "\n"
        path.write_text(_text)
        _mirror_report(_text)
        return path
    except Exception:  # noqa: BLE001
        return None


def _mirror_report(text: str) -> None:
    """Keep a copy of the live report somewhere a reboot cannot reach.

    THE REPORT HAD NO DURABLE HOME. It was moved out of the model directory for a real reason -- git
    revert after a no-gain attempt restored the committed blob and rewound a live report from 30
    attempts back to 7 -- into the run directory, which is gitignored and safe from that. But an
    optimize run works in a throwaway worktree under /tmp, so the run directory is under /tmp too,
    and /tmp is cleared on boot.

    Run 10, 2026-08-19: the box rebooted at 12:56 after fourteen hours. The report was being written
    live -- 18805 bytes at 11:50, 18935 at 12:13 -- and went with the worktree. Every measurement
    survived, because --persist keeps those in ~/.perf_mcp; only the rendered report was lost, and
    only because it was the one artifact still living in the disposable copy.

    So: the same text, written a second time into the durable state directory. Outside git, so no
    revert can rewind it; outside the worktree, so no reboot can take it. Best-effort and silent --
    a mirror that cannot be written must never cost the report that can.
    """
    try:
        from cc_optimize.tmpstate import state_dir
    except Exception:  # noqa: BLE001
        try:
            from .tmpstate import state_dir
        except Exception:  # noqa: BLE001
            return
    try:
        import os as _os

        # Only when a durable directory was actually configured (--persist). Without it state_dir()
        # is the system temp dir, which is the very place this exists to escape.
        if not (_os.environ.get("PERF_MCP_STATE_DIR") or "").strip():
            return
        for d in _durable_report_dirs(state_dir()):
            d.mkdir(parents=True, exist_ok=True)
            (d / _REPORT_NAME).write_text(text)
    except Exception:  # noqa: BLE001
        pass


def _durable_report_dirs(state: Path) -> list:
    """Every place outside the worktree that should hold the live report.

    The state directory is the one that always qualifies. The second is the run's counterpart
    directory in the main checkout, which before_loop creates and points `latest` at under
    --persist: without the report in it, following that pointer from the tree the operator has open
    reaches an empty directory, while the worktree's identical pointer reaches a live report. The
    two views must agree, or the pointer is a trap rather than a shortcut.

    Derived from the run's own `latest`, so no run id has to be threaded here, and skipped whenever
    the main checkout IS this checkout -- then the run directory already holds the report.
    """
    dirs = [state]
    try:
        from agent.run import main_runs_root
    except Exception:  # noqa: BLE001
        return dirs
    try:
        runs = _runs_root()
        main_runs = main_runs_root(runs)
        if main_runs is None:
            return dirs
        run_id = os.readlink(runs / "latest")
        mirror = main_runs / Path(run_id).name
        if mirror.resolve() != (runs / Path(run_id).name).resolve():
            dirs.append(mirror)
    except (OSError, ValueError):
        pass
    return dirs


def optimize_block(model_root, attempts_len: int, text: str, when_note: str) -> str:
    from pathlib import Path as _P

    return (
        f"# Optimize (perf) — `{_P(model_root).name}`\n\n" f"_{when_note}_\n\n" "```\n" + (text or "").strip() + "\n```"
    )


def module_optimize_block(
    model_root,
    attempts_len: int,
    text: str,
    when_note: str,
    *,
    module: str,
    index: str = "",
    pcc_gate: str = "",
    outcome: str = "optimizing…",
) -> str:
    """Per-module optimize block for module-level runs: the standard optimize
    block wrapped in the module's ``## Module:`` header, so it renders INSIDE
    that module's own (pre-seeded, correctly-positioned) section and is labelled
    with the module name — instead of a single floating global block that stays
    pinned under whichever module was optimized first."""
    idx = f" — {index}" if index else ""
    head = f"## Module: `{module}`{idx}\n\n- pcc gate: `{pcc_gate}`\n- outcome: **{outcome}**\n\n"
    body = optimize_block(model_root, attempts_len, text, when_note)
    if body.startswith("# Optimize (perf)"):
        body = body.split("\n\n", 1)[1]
    return head + body


def _level_alias_cache():
    """Resolved per call: a module constant freezes the path at import, before any redirect."""
    return state_dir() / "perf_mcp_lever_alias_cache.json"


_LEVEL_SEMANTICS = (
    "grid = core-grid occupancy / spreading work across cores; "
    "fidelity = math fidelity (HiFi/LoFi); "
    "dtype = weight or activation precision (bf16/bf8_b/bf4_b); "
    "shard = memory sharding, L1 pinning, memory-config changes; "
    "host = host-side or dispatch-side work, tracing, fusion, caching; "
    "structural = an algorithmic restructure (KV-cache, gather, conv weight prep, folding repeats, "
    "reordering a projection against a slice); "
    "tp-fracture = the weights fractured across chips; "
    "tt-lang = custom kernel authored in the tt-lang DSL; "
    "cpp = custom C++ Metalium kernel"
)


def _normalise_kind(kind: str) -> str:
    """Strip rung/knob prefixes and whitespace so `knob:grid` matches the `grid` column."""
    k = (kind or "").strip().lower()
    for pfx in ("knob:", "rung:", "lever:"):
        if k.startswith(pfx):
            k = k[len(pfx) :].strip()
    return k


def _level_of(kind: str) -> str:
    """Ladder column for an attempt, deterministic fast path.

    An unrecognised name returns "other" -- never "host", which previously swallowed
    every `knob:*` lever and any new naming silently. Semantic classification of
    unknown names lives in classify_level().
    """
    k = _normalise_kind(kind)
    if k in _LEVEL_COLS and k != "host":
        return k
    if k in _STRUCTURAL_KINDS:
        return "structural"
    if k in _HOST_KINDS or k == "host":
        return "host"
    return "other"


def _alias_cache() -> dict:
    try:
        return json.loads(_level_alias_cache().read_text())
    except Exception:  # noqa: BLE001
        return {}


def _alias_cache_put(key: str, col: str) -> None:
    try:
        c = _alias_cache()
        c[key] = col
        _level_alias_cache().write_text(json.dumps(c))
    except Exception:  # noqa: BLE001
        pass


def _classify_via_agent(kind: str, note: str, op_signature: str) -> str:
    if os.environ.get("PERF_MCP_NO_AGENT_CLASSIFY") == "1":
        return "other"
    # ask_cli resolves the binary and returns "" when there is none.
    prompt = (
        "Classify ONE optimization attempt into exactly one ladder column.\n\n"
        f"columns: {', '.join(_LEVEL_COLS)}, other\n"
        f"semantics: {_LEVEL_SEMANTICS}\n\n"
        f"attempt label (a HINT only, may be wrong): {kind!r}\n"
        f"op: {op_signature!r}\n"
        f"what the attempt actually did: {(note or '(no note)')[:600]!r}\n\n"
        "Judge from what it DID, not the label. If nothing fits, answer other.\n"
        'Reply with ONLY: {"column":"<one of the above>"}'
    )
    try:
        # integrity owns the CLI call; this was the third copy of it.
        from agent.integrity import ask_cli

        out = ask_cli(prompt).strip()
        i, j = out.find("{"), out.rfind("}")
        col = json.loads(out[i : j + 1]).get("column", "other")
        return col if (col in _LEVEL_COLS or col == "other") else "other"
    except Exception:  # noqa: BLE001
        return "other"


def classify_level(kind: str, note: str = "", op_signature: str = "") -> str:
    """Ladder column derived from what the attempt DID, not from its name.

    `kernel_kind` is a HINT only: reports have carried `tt-lang` on rows whose note
    described L1 weight-pinning (a shard change). Known names resolve deterministically;
    anything else is classified once by a Claude Code agent from the note, then cached by
    content hash so re-renders stay deterministic and cost nothing. Falls back to "other"
    (never "host") when no agent is available.
    """
    det = _level_of(kind)
    if det != "other":
        return det
    key = hashlib.sha1(f"{_normalise_kind(kind)}|{(note or '')[:400]}".encode()).hexdigest()[:16]
    cached = _alias_cache().get(key)
    if cached in _LEVEL_COLS or cached == "other":
        return cached
    col = _classify_via_agent(kind, note, op_signature)
    _alias_cache_put(key, col)
    return col


def _ttl_absent() -> bool:
    """True when the tt-lang (ttl) toolchain is not installed in this env. Rendering runs in the same
    env as the run, so this reflects the real availability the agent had."""
    return importlib.util.find_spec("ttl") is None


def _disp_level(label: str) -> str:
    """DISPLAY-only relabel: a tt-lang rung with no ttl toolchain is really a ttnn implementation the
    agent improvised, so show it as 'ttnn'. Internal column keys / kernel_kind stay 'tt-lang' — this
    changes nothing in the ladder or credit logic."""
    return "ttnn" if label == "tt-lang" and _ttl_absent() else label


# A lever whose scope is the whole model, not one op: prefetch, the host decode loop. Prefixed so the
# per-op matrix can exclude it and the report can list it where it belongs.
_MODEL_LEVER_PREFIX = "model:"


def _dominant_bound_by(profile: dict | None) -> str:
    """What the model's LARGEST reachable gap is waiting on, from the profile's own annotation.

    The same `bound_by` tag the per-op ladder gate reads, so the order this report prints is the
    order the optimizer actually walks -- not a second opinion about the same model. Weighted by
    gap_ms rather than by op count: the ladder is spent on the ops with headroom, and a hundred tiny
    eltwise ops should not outvote the matmul that owns the residual."""
    if not isinstance(profile, dict):
        return ""
    weight: dict = {}
    for b in profile.get("buckets") or []:
        for o in (b.get("top_ops") or []) if isinstance(b, dict) else []:
            if not isinstance(o, dict):
                continue
            bound = (o.get("bound_by") or "").strip().lower()
            gap = o.get("gap_ms")
            if not bound or not isinstance(gap, (int, float)) or gap <= 0:
                continue
            weight[bound] = weight.get(bound, 0.0) + float(gap)
    return max(weight, key=weight.get) if weight else ""


def _levels_display(bound_by: str = "") -> str:
    """Render the ladder from its single definition in perf_mcp, not from a second hardcoded copy.

    The order depends on what the model is waiting on -- fidelity speeds the math engine, so it
    leads on a compute-bound model and trails on a memory-bound one -- so the binding travels with
    the request. Without one, perf_mcp's own default row is used; there is no display-side default,
    because a display-side default is how the two copies drifted the first time."""
    try:
        from .perf_mcp import ladder_order
    except Exception:  # noqa: BLE001
        try:
            from perf_mcp import ladder_order  # type: ignore
        except Exception:  # noqa: BLE001
            return "grid -> dtype -> shard -> fidelity -> host -> structural -> tt-lang -> cpp"
    return " -> ".join(_disp_level(r) if r == "tt-lang" else r for r in ladder_order(bound_by))


def _op_label(sig: str, width: int = 34) -> str:
    """Display label for an op, KEEPING THE SHAPE that distinguishes it.

    This took `split(" ")[0]`, discarding the shape the signature carries -- so five different
    matmuls (32x4096x14336 ff1/ff3, 32x14336x4096 ff2, 32x4096x6144 QKV, 32x4096x4096 wo,
    128x4096x14336 prefill) rendered as seven identical `MatmulDeviceOperation` rows in the ladder
    matrix and the reader could not tell which one a ✓ belonged to. The redundant "DeviceOperation"
    suffix is dropped to make room, since every op in the table carries it.
    """
    s = (sig or "?").strip()
    if not s:
        return "?"
    head, _, rest = s.partition(" ")
    name = head.replace("DeviceOperation", "").replace("Operation", "") or head
    return ("%s %s" % (name, rest.replace(" ", ""))).strip()[:width]


def _read_json(path) -> object:
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return None


_PERF_MCP = None
_SIBLINGS: list = []


def _siblings():
    """Load the sibling resolver itself -- the one import that cannot use the resolver.

    Four lines by path, because this module may have no package and no sys.path entry; everything
    after this point goes through siblings.load(). See cc_optimize/siblings.py.
    """
    if _SIBLINGS:
        return _SIBLINGS[0]
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("cc_optimize_siblings", str(Path(__file__).resolve().parent / "siblings.py"))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _SIBLINGS.append(_mod)
    return _mod


def _perf_mcp():
    """perf_mcp, reachable under every load style. Delegates to cc_optimize/siblings.py.

    The resolution order used to live here, and was then rewritten a second time in run.py for the
    thermal gate -- two implementations of one idea, and the copy in run.py was the one that failed.
    One owner now: cc_optimize/siblings.py. Contract is unchanged: the module, or None.
    """
    return _siblings().load("perf_mcp")


def _measured_stage_paths(model: str = "", task: str = "") -> dict:
    """How each stage was measured: "trace+1cq", "eager", or absent/unknown.

    A stage that fell back to eager is not comparable to a traced one -- eager pays per-op host
    dispatch that trace removes, which on gemma-3's prefill is half the wall clock. Held against a
    band that assumes trace it scores an automatic miss, so the report needs to know."""
    m = _perf_mcp()
    try:
        return (m.read_stage_paths(model=model, task=task) or {}) if m else {}
    except Exception:  # noqa: BLE001
        return {}


def _measured_stage_ops(model: str = "", task: str = "") -> dict:
    """Op dispatches observed per stage -- what the "eager"/"trace+1cq" label was derived from."""
    m = _perf_mcp()
    try:
        return (m.read_stage_ops(model=model, task=task) or {}) if m else {}
    except Exception:  # noqa: BLE001
        return {}


def _measured_stage_ms(model: str = "", task: str = "") -> dict:
    """trace_replay's per-stage timings, or {}.

    These come from the PIPELINE_STAGES the MODEL declares -- prefill/decode measured by the
    harness -- so they are the only phase numbers in this report that are not prose.
    """
    m = _perf_mcp()
    try:
        return (m.read_stage_ms(model=model, task=task) or {}) if m else {}
    except Exception:  # noqa: BLE001
        return {}


def _pinned_stage_bytes(stage, model: str = "", task: str = ""):
    """The pinned BASELINE read set for one stage, or None. Read-only.

    Measuring the bytes made them right. It did not stop them moving: the dtype rung halves a
    weight and the observed read set halves with it, so a ceiling recomputed each round retreats
    ahead of the measurement -- "the floor is a property of the IMPLEMENTATION, not a goal". The
    baseline is the target; the current reading is kept beside it to tell a real win from a stale
    number, exactly as _floor_anchor's caller does.
    """
    try:
        led = _ledger()
        v = led.anchor_value(led.KIND_STAGE_BYTES, depth=str(stage or "").strip().lower(), model=model, task=task)
        return int(round(float(v) * 1e6)) if v and float(v) > 0 else None
    except Exception:  # noqa: BLE001
        return None


def _measured_stage_bytes(model: str = "", task: str = "") -> dict:
    """trace_replay's per-stage READ SET in bytes, or {}.

    The only measurement of what a stage streams. What stood here before -- reading a per-stage
    `bytes` figure off the profile's buckets -- could never work: buckets carry no `bytes` key and
    their regime tag is "na" for every one, so it returned 0 always and _bytes_for silently used the
    checkpoint estimate while appearing to prefer a measurement.
    """
    m = _perf_mcp()
    try:
        return (m.read_stage_bytes(model=model, task=task) or {}) if m else {}
    except Exception:  # noqa: BLE001
        return {}


def _stage_rows(rows: list, indent: str = "    ", share: bool = True) -> list:
    """name / ms / proportional bar, and a share of the block total when the rows share a currency.

    NO "hottest" MARKER, on either block. It marked the largest row, and the only block that got it
    was the agent's annotation -- the one the report labels `not measurement` -- so the single word a
    reader acts on sat beside the numbers it is told not to trust. The bars already show which row is
    largest, and the optimizer ranks targets from the profile's gap_ms, never from this table.

    `share` is False when the rows are NOT in one currency. A per-request prefill and a per-token
    decode do not sum: at OSL 128 the request spends 128 decode steps, so prefill is ~2% of it, not
    the 71% a naive sum reports. Without a meaningful total there is no meaningful percentage, so
    both are dropped rather than printed wrong.
    """
    if not rows:
        return []
    peak = max((ms for _n, ms in rows)) or 1.0
    total = sum(ms for _n, ms in rows) or 1.0
    out = []
    for name, ms in rows:
        n = max(0, min(_BAR_W, int(round((ms / peak) * _BAR_W))))
        bar = "\u2588" * n + "\u2591" * (_BAR_W - n)
        pct = ("  %5.1f%%" % (100.0 * ms / total)) if share else ""
        out.append("%s%-22s %8.2f ms  %s%s" % (indent, str(name)[:22], ms, bar, pct))
    return out


def _stage_table_lines(stages: list, model: str = "", task: str = "", stages_measured: bool = False) -> list:
    """The block-level timing view, in two parts that do NOT have the same standing.

    MEASURED (top): trace_replay's per-stage timings, derived from the PIPELINE_STAGES the model
    declares. Real numbers, phase-correct, and absent when the pipeline declares no stages.

    BREAKDOWN (bottom): where the ms inside a stage went. Its provenance is stated in its own
    header, because two different things can supply it. The profile's op-class buckets are device
    measurements and are used whenever they exist. The agent's stages_json is free text -- nothing
    validates the names, and on this model four of fourteen carry no phase at all -- so it appears
    only when the profile has no buckets to answer with, and says so.

    The two are kept apart rather than merged because decode ms and prefill ms are not the same
    currency: decode recurs on every token and sets tok/s/u, prefill happens once and sets TTFT. A
    phase split guessed from prose can put time in the wrong pool and be acted on, which is worse
    than having no split. Each block therefore carries its OWN total, and percentages are within a
    block -- the two totals describe different things and should not be added.
    """
    out = []
    meas = _measured_stage_ms(model, task)
    if meas:
        rows = sorted(meas.items(), key=lambda kv: -kv[1])
        # NO COMBINED TOTAL. These stages are measured in different units -- prefill per request,
        # decode per token -- so their sum describes nothing that happens, and the percentages taken
        # from it are wrong in the direction that matters: it read prefill as 71% of the work when at
        # OSL 128 a request spends 128 decode steps and prefill is ~2% of it. Each stage states its
        # own ms, which is the number that is true.
        out.append("  measured by trace_replay")
        out.extend(_stage_rows(rows, share=False))
    st = [x for x in (stages or []) if isinstance(x, dict) and (x.get("ms") or 0) > 0]
    if st:
        rows = [(x.get("name", "?"), float(x.get("ms") or 0)) for x in st]
        if out:
            out.append("")
        # These rows ARE one currency (one profile, one window), so a share is meaningful here --
        # unlike the measured block above, whose two stages are counted per request and per token.
        _hdr = (
            "op-class breakdown (measured, same profile as above)"
            if stages_measured
            else "agent breakdown (annotation, not measurement)"
        )
        out.append("  %-44s %8.2f ms" % (_hdr, sum(m for _n, m in rows)))
        out.extend(_stage_rows(rows))
    return out


def _floor_status(floor_ms: float, measured_ms: float) -> str:
    """BELOW_BAND | IN_BAND | ABOVE_BAND for a floor-derived (non-decode) target.

    Delegates to perf_target.score so the "measured beat the ceiling -> target suspect, never bank
    it" judgement lives in ONE place; this table used to re-implement that check and then guess
    which side was stale.
    """
    try:
        from agent.perf_target import score, target_from_floor_ms
    except Exception:  # noqa: BLE001
        return "UNKNOWN"
    return str(score(target_from_floor_ms(floor_ms), measured_ms).get("status") or "UNKNOWN")


def _throughput_from_profile(baseline_profile: dict | None) -> dict | None:
    """Compute the roofline-target snapshot from the always-written baseline device profile, so the
    Roofline section renders deterministically even when the per-profile persist did not fire. Uses the
    pure-python roofline + perf_target modules via the agent package. Never raises."""
    if not isinstance(baseline_profile, dict):
        return None
    try:
        _pa = str(Path(__file__).resolve().parent.parent)
        if _pa not in sys.path:
            sys.path.insert(0, _pa)
        from agent import perf_target as _pt
        from agent import roofline as _rl

        rep = _rl.residual_report(baseline_profile, {})
        floor = rep.get("modeled_floor_ms")
        tgt = _pt.target_from_floor_ms(floor)
        return {
            "scope": "model",
            "has_unit_ceiling": False,
            "theoretical_rate": tgt.theoretical_rate,
            "band": [tgt.band[0], tgt.band[1]],
            "active_bytes": tgt.active_bytes,
            "peak_bw_gbps": 0.0,
            "tp_degree": tgt.tp_degree,
            "modeled_floor_ms": floor,
        }
    except Exception:  # noqa: BLE001
        return None


def _stages_from_profile(baseline_profile: dict | None) -> list:
    """Derive block-level stage rows from the profile op-class buckets, so the Block-level timing table
    renders deterministically even when no attempt carried per-stage timings."""
    prof = baseline_profile if isinstance(baseline_profile, dict) else {}
    rows = []
    for b in prof.get("buckets") or []:
        if not isinstance(b, dict):
            continue
        rows.append({"name": str(b.get("id", "?")), "ms": float(b.get("device_ms") or 0.0)})
    if not rows:
        return []
    rows.sort(key=lambda r: -r["ms"])
    rows[0]["dominant"] = True
    return rows


def _ledger():
    """The measurement ledger (cc_optimize/measurements.py), loaded by path."""
    global _LEDGER_MOD
    try:
        return _LEDGER_MOD
    except NameError:
        pass
    import importlib.util as _ilu

    _p = Path(__file__).with_name("measurements.py")
    _spec = _ilu.spec_from_file_location("tt_measurements", str(_p))
    _m = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    globals()["_LEDGER_MOD"] = _m
    return _m


def _ledger_pair(kind: str, model: str = "", task: str = ""):
    """(before, after) for one measurement kind, straight from the ledger.

    THE POINT: these carry their own depth and mode, so the report never has to infer what a number
    measured, and there is no chain to fall through when one is missing. `first` is the earliest
    reading ever taken for this (model, task), so it is the TRUE original even on the fifth rerun.
    """
    try:
        led = _ledger()
        return (
            led.first(kind, led.PHASE_BEFORE, model=model, task=task),
            led.last(kind, led.PHASE_AFTER, model=model, task=task),
        )
    except Exception:  # noqa: BLE001
        return None, None


def _single_reading(title: str, row) -> str | None:
    """One measurement, stated plainly, with no delta implied.

    The depth travels WITH the number because that is the axis that made the pair incomparable in the
    first place: "547.80 ms (96 layers)" cannot be silently re-read as a full-model result the way a
    bare 547.80 can.
    """
    if not isinstance(row, dict):
        return None
    v = row.get("value_ms")
    if not isinstance(v, (int, float)) or v <= 0:
        return None
    d = str(row.get("depth") or "unknown")
    # JUST THE NUMBER. The depth in the header already says what was measured; a trailing explanation
    # of what the line is NOT is noise in a report that goes out for confirmation, and the em dash it
    # carried mojibaked in the reader's terminal. No arrow means no delta -- that is the whole signal.
    return "%s (%s):  %.2f ms" % (title, "all layers" if d == "all" else "%s layers" % d, v)


def _ledger_line(kind: str, title: str, model: str = "", task: str = ""):
    """Render one before/after line from the ledger.

    A DELTA needs two readings of the same work. Without that pair there is still a measurement worth
    showing, so the line degrades to the single latest reading rather than to a subtraction nobody can
    trust -- see _single_reading for why the arrow is dropped instead of the whole line.
    """
    try:
        led = _ledger()
        a, b = _ledger_pair(kind, model, task)
        if not a and not b:
            return None
        if not a or not b:
            # One reading only: the normal state for most of a run (before, no after yet), or an after
            # with no anchor. Show the number, not a half-drawn arrow.
            return _single_reading(title, b or a)
        av, bv = a.get("value_ms"), b.get("value_ms")
        depth = a.get("depth") or "unknown"
        dl = "all layers" if str(depth) == "all" else "%s layers" % depth
        ok, _why = led.comparable(a, b)
        if not ok:
            # DO NOT SUBTRACT, and do not disclaim either. Two reports shipped a headline whose numbers
            # described different work -- "before 547.90 -> after 547.80 (all vs 96)", the same
            # unoptimized build profiled twice, and "before 296.70 -> after 117.40 (2 vs all)", a 60%
            # "win" that is purely the depth stamp. The disclaimer was printed after the numbers and
            # the numbers are what got read. The latest reading is still a real measurement of a real
            # build, so it survives; only the arrow between two incommensurable ones goes.
            return _single_reading(title, b)
        pct = led.delta_pct(a, b)
        spd = (av / bv) if bv else 1.0
        _der = " ".join(t for t, r in (("before", a), ("after", b)) if isinstance(r, dict) and r.get("derived"))
        _mark = "   [%s DERIVED, not measured]" % _der if _der else ""
        return "%s (%s):  %.2f ms  ->  %.2f ms   (%+.1f%%, %.2fx)%s" % (title, dl, av, bv, pct, spd, _mark)
    except Exception:  # noqa: BLE001
        return None


def _depth_in_force() -> str:
    """The one answer to "what depth is this". See layer_depth.depth_in_force."""
    try:
        from agent.layer_depth import depth_in_force

        return depth_in_force()
    except Exception:  # noqa: BLE001 -- unknown depth reads as full, exactly as it did before
        return "all"


def _depth_label(profile: dict | None = None) -> str:
    """How much of the model the tracy numbers cover. A ms figure means nothing without it: the whole
    2-layer-vs-16-layer confusion came from a headline that printed neither side's depth.

    Read the depth the PROFILE was stamped with, not this process's env. The depth is exported into
    the profiling SUBPROCESS, so the renderer's own TT_PERF_LAYERS is usually empty -- which made the
    label fall through to "all layers" for a run profiled at 16, printing a depth that was simply
    wrong. Env stays as the fallback for profiles written before stamping.
    """
    raw = ""
    if isinstance(profile, dict):
        raw = str(profile.get("perf_layers") or "").strip()
    if not raw:
        raw = "" if _depth_in_force() == "all" else _depth_in_force()
    # A partial window means the tracy numbers are a COVERAGE SAMPLE -- a few layers holding every op
    # type -- not the whole model. But the COUNT is not reliable: it comes from TT_PERF_LAYERS, which
    # usually still holds the coverage default (2) and does NOT track the depth the search actually
    # profiled at, so "%s layers" prints a wrong number -- an 8- or 16-layer slice is reported as "2".
    # Disclose that it is a coverage sample WITHOUT the bogus count; only full-depth is stated exactly.
    if raw.isdigit() and int(raw) > 0:
        return "a coverage sample (not the full model)"
    return "all layers"


def _baseline_trace_ms(baseline_profile: dict | None):
    """Delegates to the ledger, which owns how a trace-pass reading is extracted."""
    try:
        return _ledger().trace_ms_from_profile(baseline_profile)
    except Exception:  # noqa: BLE001
        return None


def _floor_basis(profile: dict | None) -> str:
    """What the floor actually is, and how much of the profile it covers.

    "Σ per-op roofline floors" reads like an arbitrary sum. It is physics -- each op's floor is
    max(FLOPs/peak, bytes/bandwidth, count x dispatch) -- and for a weight-streaming model the memory
    term dominates, so the total is essentially bytes/bandwidth. But it sums each bucket's top_ops,
    not every op, and skips host_overhead as non-device work: on llama3_1_8b_p150 that is 86% of
    device time, so the number is a LOWER bound. A confirmation report should state the coverage
    rather than let the figure read as complete.
    """
    base = "Σ per-op max(FLOPs/peak, bytes/BW, dispatch)"
    try:
        buckets = (profile or {}).get("buckets") or []
        total = float((profile or {}).get("device_ms") or 0.0)
        covered = sum(
            float(o.get("device_ms") or 0.0)
            for b in buckets
            if b.get("id") != "host_overhead"
            for o in (b.get("top_ops") or [])
        )
        if total > 0 and covered > 0:
            return "%s; covers %.0f%% of device time" % (base, covered / total * 100.0)
    except Exception:  # noqa: BLE001
        pass
    return base


def _is_win(attempt) -> bool:
    """Delegates to the ledger, which owns what a win is. Not a second implementation."""
    try:
        return _ledger().is_win(attempt)
    except Exception:  # noqa: BLE001
        return False


def _win_set(attempts, baseline_ms=None) -> set:
    """Delegates to the ledger, which owns which attempts actually reduced the measured time."""
    try:
        return _ledger().winning_indices(attempts, baseline_ms)
    except Exception:  # noqa: BLE001
        return set()


def _peak_for_stage(stage, profile, model: str = "", task: str = ""):
    """(peak FLOP/s, dominant fidelity) for ONE stage, or (0.0, "") when the capture did not mark it.

    Pinned first, for the reason every roof input is pinned: the fidelity rung moves the mode a stage
    runs at, and a ceiling recomputed from it retreats ahead of the measurement. Keyed per STAGE --
    the shared per-unit key was only ever a consequence of there being one number to key.

    Zero means "no per-stage evidence", and the caller keeps the whole-profile figure it already had,
    so an unmarked capture behaves exactly as before stage marks existed.
    """
    try:
        led = _ledger()
        _p = led.anchor_value(led.KIND_PEAK_FLOPS, depth=str(stage or "").strip().lower(), model=model, task=task)
        if _p and float(_p) > 0:
            return float(_p), ""
    except Exception:  # noqa: BLE001
        pass
    try:
        _sb = ((profile or {}).get("stage_buckets") or {}).get(stage)
        if not _sb:
            return 0.0, ""
        _rows, _ = _fidelity_breakdown({"buckets": _sb})
        if not _rows:
            return 0.0, ""
        _top = max(_rows, key=lambda r: r[1])
        if not _top[1]:
            return 0.0, ""
        from agent.environment import ARCH_FACTS
        from agent.perf_target import chip_peak_flops as _cpf

        _arch = str(os.environ.get("PERF_MCP_ARCH") or "blackhole").strip().lower()
        return float(_cpf(ARCH_FACTS.get(_arch) or {}, str(_top[0])) or 0.0), str(_top[0])
    except Exception:  # noqa: BLE001
        return 0.0, ""


def _pinned_peak_flops(unit, model: str = "", task: str = ""):
    """The pinned peak FLOP/s for this (model, task, unit), or None if nothing is pinned.

    READ-ONLY, and keyed on the UNIT rather than the profiling window -- the same key the byte anchor
    uses, because both describe the whole model rather than the slice the profiler built. Pinning
    happens where the value is PRODUCED (perf_mcp._persist_throughput); rendering a report must not
    change what the next report says.
    """
    try:
        led = _ledger()
        d = str(unit or "token").strip().lower() or "token"
        return led.anchor_value(led.KIND_PEAK_FLOPS, depth=d, model=model, task=task)
    except Exception:  # noqa: BLE001
        return None


def _floor_anchor(current_ms, depth, model: str = "", task: str = ""):
    """The pinned modeled floor for this (model, task, depth), or None if nothing is pinned yet.

    READ-ONLY. The floor is a property of the IMPLEMENTATION, not a goal: halving a weight's dtype
    halves the bytes it must move, so recomputing it each round makes the target retreat ahead of the
    measurement and it is never reached. The pin therefore has to happen once, where the floor is
    PRODUCED (perf_mcp._persist_throughput) -- not here, because rendering a report must not change
    what the next report says.
    """
    try:
        led = _ledger()
        d = str(depth if depth not in (None, "") else "unknown")
        return led.anchor_value(led.KIND_FLOOR, depth=d, model=model, task=task)
    except Exception:  # noqa: BLE001
        return None


_BAR_W = 20

# Dispatch has no achievable band -- its target is zero, since a fully traced model replays from
# device-resident queues and the op gaps vanish. This is the share of a step above which the residual
# is worth a round of the ladder rather than being noise. Named and printed, because a verdict the
# reader cannot check against the stated target is not a verdict.
_DISPATCH_FLAG_PCT = 10


def _env_float(name: str):
    try:
        v = float(os.environ.get(name) or "")
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _wrap_note(text: str, width: int) -> list:
    """Wrap a withheld-reason to the table width. Kept local rather than using textwrap so the
    report has no import that could fail while it is being written."""
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w) if cur else w
    if cur:
        lines.append(cur)
    return lines or [""]


def _split(width, a=60, b=98):
    """Divider aligned to the column format: breaks at the two column boundaries.

    The positions are arguments rather than the literals 46/69 they used to be, because the table
    widened and a divider that keeps the old breaks stops being a divider -- it becomes two crosses
    landing in the middle of the number fields."""
    return "\u2500" * a + "\u253c" + "\u2500" * max(0, b - a - 1) + "\u253c" + "\u2500" * max(0, width - b - 1)


def _bar(frac, width=None):
    """Proportional fill. `frac` is 0..1, or None for "no data" (an all-empty bar).

    `width` is an argument because the utilisation panel and the block-timing panel are read
    differently: block timing compares rows against the block's own peak, where 20 cells is plenty,
    while utilisation compares FRACTIONS OF A PEAK across roofs that differ by three orders of
    magnitude -- and at 20 cells a 0.1% bar and an 11% bar both render as empty.
    """
    w = int(width or _BAR_W)
    if frac is None:
        return "\u2591" * w
    n = max(0, min(w, int(round(float(frac) * w))))
    return "\u2588" * n + "\u2591" * (w - n)


def _dispatch_ms_per_unit(profile, per_unit_ms):
    """Launch overhead in the same unit as the headline, or None.

    host_overhead is the profiler's op_gap bucket -- time the device spent waiting between kernels.
    Scaled into the headline unit by its share of total device_ms, because the two are measured over
    different windows (a profiling capture vs one traced token).
    """
    try:
        buckets = [b for b in (profile or {}).get("buckets") or [] if isinstance(b, dict)]
        host = next((b for b in buckets if b.get("id") == "host_overhead"), None)
        total = float((profile or {}).get("device_ms") or 0.0)
        if not host or total <= 0 or not per_unit_ms:
            return None
        share = float(host.get("device_ms") or 0.0) / total
        # A share at or above 1 is not a dispatch measurement. host_overhead sums per-op GAPS, and op
        # intervals OVERLAP, so on a profile with concurrency the sum runs past total device_ms
        # (634.55 vs 293.20 on gemma-3-12b-it). Scaling that onto a token would claim more launch
        # overhead than there is time in the step. Refusing is right; the caller states the refusal
        # rather than dropping the row, so an absent number is never mistaken for a zero one.
        return float(per_unit_ms) * share if 0 < share < 1 else None
    except Exception:  # noqa: BLE001
        return None


def _ops_per_unit(profile):
    try:
        return (
            sum(int(b.get("count") or 0) for b in (profile or {}).get("buckets") or [] if isinstance(b, dict)) or None
        )
    except Exception:  # noqa: BLE001
        return None


def _capacity_bytes():
    """Per-chip DRAM from the detected arch, or None. Never a hardcoded number."""
    try:
        from agent.environment import ARCH_FACTS

        import os as _os

        arch = (_os.environ.get("PERF_MCP_ARCH") or "blackhole").strip().lower()
        return int((ARCH_FACTS.get(arch) or {}).get("dram_capacity_bytes") or 0) or None
    except Exception:  # noqa: BLE001
        return None


_MODEL_ROOT_HINT = None


def _facts_params(facts) -> int:
    """The param count these facts can yield, literal or derived. 0 when they cannot yield one."""
    try:
        from agent.perf_target import ceiling_params as _cp

        return int(_cp(facts or {}) or 0)
    except Exception:  # noqa: BLE001
        return int((facts or {}).get("total_params") or 0)


def _model_facts():
    """perf_target_inputs.json — total_params / layers / weight bytes. None if unobtainable."""
    # the caller's model dir first: it is the only source that is right by construction
    if _MODEL_ROOT_HINT:
        try:
            _p = Path(_MODEL_ROOT_HINT) / "perf_target_inputs.json"
            if _p.is_file():
                f = json.loads(_p.read_text())
                # ACCEPTED IF IT CAN YIELD A PARAM COUNT, not if it spells one. ceiling_params
                # derives the count from device_weight_bytes / bytes_per_param when total_params is
                # absent, so gating on the literal field rejected the model's own facts before that
                # derivation could run -- and the caller then fell back to a search that does not
                # know the model root. Measured on voxtral: a census-written file carrying
                # 10604865536 bytes at 2.0 bytes/param was discarded for lack of their quotient, and
                # every per-stage compute ceiling printed "no param count" as a result.
                if isinstance(f, dict) and _facts_params(f):
                    return f
        except Exception:  # noqa: BLE001
            pass
    m = _perf_mcp()
    if m is None:
        return None
    try:
        f = m._load_perf_target_inputs()
        return f if isinstance(f, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _prompt_tokens() -> int:
    """The input length the PREFILL STAGE ACTUALLY RAN, so the FLOP term describes the request the
    harness issued rather than a number from a neighbouring knob.

    TT_PERF_ISL_TOKENS is the one the generated test reads (perf_test_gen.py) and prints back as
    PERF_ISL_TOKENS; it is what `_prompt_ids_for_isl` sizes the prompt to and therefore what the
    traced prefill stage consumes.

    TT_PERF_SEQ_LEN is a DIFFERENT knob -- before_loop walks it down a shape ladder when a baseline
    crashes on a pinned program config, and the full-pipeline gate quotes it on its scorecard. Reading
    it here priced prefill's arithmetic against a sequence the stage never saw, and on a run where it
    is unset (the common case) prefill got no compute roof at all. It stays as a fallback because a
    run that set it and nothing else still declares something; the variable the test reads wins.

    0 when neither is declared, which WITHHOLDS the prefill compute roof rather than inventing a
    sequence length for it."""
    # OBSERVED FIRST. The generated test prints PERF_ISL_TOKENS after tokenizing -- the length that
    # actually reached the model -- and perf_mcp now persists it beside the stage timings. Everything
    # below is a fallback for a report rendered before any run has measured one.
    # Both import spellings, like every other reader here: summary is loaded as `cc_optimize.summary`
    # by the tool and as a top-level `summary` by spec_from_file_location, and a bare relative import
    # fails silently in the second case -- which would skip the observed value and quietly fall back
    # to a default, the exact bug this is fixing.
    m = _perf_mcp()
    if m is not None:
        try:
            _obs = int(m.read_stage_isl() or 0)
            if _obs > 0:
                return _obs
        except Exception:  # noqa: BLE001
            pass
    for var in ("TT_PERF_ISL_TOKENS", "TT_PERF_SEQ_LEN"):
        raw = str(os.environ.get(var) or "").strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    # THE GENERATOR'S OWN DEFAULT, READ FROM THE SKELETON. 128 in / 128 out is the tool's defined
    # benchmark point, stated once as a literal in the emitted test -- which is where it has to be,
    # since generated code cannot import from its generator. This reads that literal rather than
    # keeping a second copy of it: two copies is how the roof came to price a length the run never
    # used. Reached only when nothing was observed and nothing declared, which is exactly the case
    # where the run is about to use this number.
    try:
        from agent.perf_test_gen import DEFAULT_ISL_TOKENS

        return int(DEFAULT_ISL_TOKENS)
    except Exception:  # noqa: BLE001
        return 0


def _retires_one_per_user(rf, batch: int) -> bool:
    """Does one call of this stage retire exactly one item per user?

    THE HEADING AND THE UNIT ASK THIS SAME QUESTION, and used to answer it with two different tests:
    the unit compared the stage's item count against the BATCH, the heading against the literal 1. At
    batch 1 the two agree, and until per-stage item counts existed every stage fell back to a single
    item -- so both held and the disagreement could not appear. Once real counts arrived, a batch-8
    recurring stage retiring 8 rows per call satisfied the unit (printing tok/s/u) and failed the
    heading (printing "per request"), and the report contradicted itself in adjacent lines.

    One predicate, so the two cannot drift apart again. Nothing is assumed about what the stage is
    called or what its item is: `batch` is what the run served and `tokens` is what the stage's own
    ops retired, both discovered.
    """
    return int((rf or {}).get("tokens") or 0) == int(batch)


def _request_batch() -> int:
    """How many requests the prefill stage processed at once. 1 when nothing says otherwise.

    THE UNIT OF WORK IS A BATCH, NOT A SEQUENCE. Both the byte and the FLOP term were computed from
    seq_len alone, so a run at batch 8 was priced as if it prefilled 128 tokens when it prefilled
    1024. FLOPs are linear in the token count and the activation bytes are too, so the roof came out
    8x low and the stage read memory-bound when it was compute-bound -- the binding roof is the one
    thing this table exists to state, and batch decided it.

    Declared rather than observed: trace_replay prints batch on its TRACE_REPLAY_PATH line but that
    is not persisted, so the environment the harness set is the best available source. 1 is the
    honest default -- every model has at least one request in flight."""
    # WHAT THE RUN SERVED, before what the environment asked for. TT_PERF_BATCH carries 0 -- the
    # "ask the pipeline" sentinel -- so reading the environment first resolved an eight-user run to 1
    # and priced every ceiling for a single user against an eight-user measurement.
    try:
        from cc_optimize.perf_mcp import read_stage_batch

        _b = int(read_stage_batch() or 0)
        if _b > 0:
            return _b
    except Exception:  # noqa: BLE001 -- not recorded: fall through to what the operator asked for
        pass
    for var in ("TT_PERF_BATCH", "PERF_MCP_BATCH", "TT_PERF_BATCH_SIZE"):
        raw = str(os.environ.get(var) or "").strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return 1


_SECTION_BYTES = None


def _section_bytes_cached() -> dict:
    """{tower: bytes} from the checkpoint header, read once per report.

    Header-only: every tensor's byte span is in `data_offsets`, so a tower's size is a sum over names
    with no tensor data read and no device involved.
    """
    global _SECTION_BYTES
    if _SECTION_BYTES is not None:
        return _SECTION_BYTES
    _SECTION_BYTES = {}
    try:
        # THE HUB ID FROM THE MODEL'S OWN SOURCE. This called _hf_repo_ids(Path(...)) -- and that
        # function takes a parsed Source, not a path: `for _path, tree in src.trees.items()` raises
        # AttributeError on a Path, the bare except below swallowed it, and _SECTION_BYTES was {} on
        # every model since the day it was written. _stage_share then found no section, returned 1.0,
        # and EVERY stage was priced at the whole model's bytes -- the apportionment this cache exists
        # to provide never ran once. model_id_from_source answers the same question and takes a path,
        # which is what the one caller has.
        from agent.checkpoint_sections import hf_cache_dir, section_bytes
        from agent.stack_survey import model_id_from_source

        _mids = [model_id_from_source(_MODEL_ROOT_HINT)] if _MODEL_ROOT_HINT else []
        for _mid in [m for m in _mids if m]:
            snap = hf_cache_dir(_mid)
            if snap:
                _SECTION_BYTES = section_bytes(snap) or {}
                if _SECTION_BYTES:
                    break
    except Exception:  # noqa: BLE001 -- no checkpoint reachable: every stage keeps the whole-model share
        _SECTION_BYTES = {}
    return _SECTION_BYTES


def _roofline_stage_share(mf, stage) -> float:
    """Fraction of resident weights this stage's subtree holds. See _share_and_basis."""
    return _share_and_basis(mf, stage)[0]


def _share_and_basis(mf, stage):
    """Fraction of the model's resident weights the subtree THIS stage runs holds; 1.0 when unknown.

    A stage streams its own tower and nothing else -- a decode token reads the language backbone and
    never the audio encoder -- so handing every stage the whole-model byte count overprices the
    backbone stages and leaves a third tower with nothing at all.

    MEASURED FIRST, APPORTIONED SECOND. The census walks the built model and credits every tensor to
    the attribute names it was reached through, so a stage's resident bytes are a lookup by the name
    the model declared. Before that existed the only split available was the CHECKPOINT's, which
    states disk precision: voxtral's language tower is 85.8% of a 9.36 GB bf16 file, and the loader
    put 1.72 GB on the chip at widths it chose per tensor. 85.8% of the file is 85.8% of the chip only
    if both towers were quantised alike -- nothing requires that, and the error lands whole on the
    stage's ceiling. It published a decode floor of 3.15 ms against a 2.89 ms measurement: 109% of
    peak, which is not a fast model but an unusable row.

    The checkpoint ratio remains the fallback rather than being removed, because it is right whenever
    the widths do match and it is the only answer for a model whose census never ran.
    """
    root = str(((mf or {}).get("stage_roots") or {}).get(str(stage)) or "").strip()
    if not root:
        # NO MAPPING. Decided by how many towers the checkpoint has, not by the stage's NAME.
        # A single-tower model has only one subtree, so every stage streams all of it and the
        # whole-model figure is exactly right -- the behaviour before any of this existed. With
        # two or more towers there is no such answer: pricing one at another's byte count is the
        # wrong divisor rather than an approximation, and a refused ceiling beside a real
        # measurement is more use to a reader than a confident wrong one.
        # HOW MANY TOWERS, FROM THE MODEL'S OWN FACTS. `blocks` is one entry per subtree with its own
        # geometry, published by the run; the checkpoint's section map answers the same question for a
        # model whose census never ran. Either is structural. The stage's NAME is not, and was what
        # decided this before: perf_target accepted `decode` and `prefill`, so those two were handed
        # the whole model and a third tower was refused, by spelling.
        #
        # ONE TOWER, OR NO EVIDENCE OF MORE THAN ONE: the whole model is every stage's subtree, which
        # is the behaviour before any of this existed. Refusing on an EMPTY map would strip the
        # ceiling off every model whose census never ran and whose checkpoint could not be read --
        # absence of evidence charged as evidence of two towers.
        _towers = len((mf or {}).get("blocks") or {}) or len(_section_bytes_cached())
        if _towers <= 1:
            return 1.0, "whole-model"
        # OTHERWISE REFUSE, FOR EVERY STAGE ALIKE. This asked perf_target to price the stage and
        # read the ANSWER as the verdict -- accepted means "its read set is the backbone", raised
        # means "refuse" -- which sounds structural and was a stage-name list wearing an exception:
        # perf_target accepted exactly `decode` and `prefill`, so those two got the whole-model
        # figure and everything else got nothing, by name, from a function whose comment said it was
        # not doing that. Deleting the name gate there (it prices from `items` now, not from what a
        # stage is called) left this handing 1.0 to every unmapped stage instead.
        #
        # The right answer is the one this file already argues for one test above: on a model with
        # more than one tower, an unmapped stage has NO known read set. The whole-model figure is
        # not a conservative approximation of it -- it charges a stage for a tower it never touches,
        # which is the precise defect `test_the_backbone_stops_paying_for_the_tower_it_never_reads`
        # exists to catch. A refused ceiling beside a real measurement beats a confident wrong one,
        # and that holds for the backbone stages too, not only for the third one.
        #
        # A single-tower model is unaffected: it returns 1.0 above, because there the whole model
        # genuinely IS every stage's subtree.
        return 0.0, "refused"
    _dev = (mf or {}).get("device_section_bytes") or {}
    _res = float((mf or {}).get("device_weight_bytes") or 0.0)
    _mine_dev = float(_dev.get(root) or 0.0)
    if _res > 0 and _mine_dev > 0:
        # A group is credited at every depth it was reached through, so a name high enough in the
        # tree covers the whole model and yields 1.0 -- the true share for that root, not an
        # overflow. ABOVE 1.0 is impossible and means the two figures came from different walks;
        # fall through rather than publish a share the census cannot support.
        _s = _mine_dev / _res
        if _s <= 1.0:
            return _s, "measured"
    secs = _section_bytes_cached()
    total = float(sum(secs.values()) or 0.0)
    mine = float(secs.get(root) or 0.0)
    # THE CHECKPOINT'S SPLIT, AND THE READER IS TOLD. This states DISK precision: the loader picks a
    # dtype per tensor, so a tower quantised more heavily than its neighbour is a smaller slice of
    # what is resident than of what is on the file. Voxtral's backbone is 85.8% of a 9.36 GB bf16
    # checkpoint, and its decode measurement implies 77.9% of the 1.72 GB actually on the chip --
    # which is why that row read 109% of peak, a number that looks like a fast model and is a guessed
    # divisor. The basis travels with the value so the report can say which one it used.
    return ((mine / total) if (total > 0 and mine > 0) else 1.0), (
        "checkpoint" if (total > 0 and mine > 0) else "whole-model"
    )


def _per_request_stages() -> set:
    """Stages the run recorded a PER-REQUEST item count for -- i.e. that consume the prompt.

    The legacy PERF_ISL_TOKENS marker names exactly the stage that takes the prompt, which is the
    stage that writes the KV the recurring stage later reads. Used to tell a stage inside the
    autoregressive loop from one outside it, without asking what any of them are called.
    """
    try:
        from cc_optimize.perf_mcp import read_stage_isl_per_request_map

        return {str(k) for k in (read_stage_isl_per_request_map() or {})}
    except Exception:  # noqa: BLE001 -- unknown means the recurring test alone decides
        return set()


def _stage_items_observed(stage, profile) -> int:
    """How many items ONE call of `stage` retired, from the matmuls that stage actually ran. 0 if it
    ran none with parseable dims.

    THE COUNT THE RUN CAN SEE FOR ITSELF. A stage that states nothing was priced at ONE item, which
    is right for a recurring step and 1500x wrong for a tower over 1500 frames, and the only escape
    was a workload marker filed under a typed stage name -- so exactly one stage per model could be
    sized, and only if it happened to be called that. A matmul's M is the number of rows the stage
    pushed through: batch included, per call, which is the same quantity <stage>_trace_items states.
    The dominant matmul by FLOPs is the stage's real shape; smaller projections beside it share it.

    Same joining principle as stage_roots, which pairs a 32-block stack with a 32-block checkpoint
    section: read what the model did and join on a number, never on a name.
    """
    try:
        from agent.roofline import parse_matmul_shape
    except Exception:  # noqa: BLE001
        try:
            from models.experimental.perf_automation.agent.roofline import parse_matmul_shape
        except Exception:  # noqa: BLE001
            return 0
    _rows: dict = {}
    for _b in ((profile or {}).get("stage_buckets") or {}).get(str(stage)) or []:
        for _o in (_b or {}).get("top_ops") or []:
            # LOGICAL ROWS FIRST. `shape` carries the PADDED dim -- what the kernel computed -- so a
            # decode step retiring one row per user reads 32 for a batch of 8, and the stage would
            # look like it retires 32 items and stop being a per-user rate. `rows` is the count the
            # model asked for. Falling back to the fingerprint keeps profiles written before it.
            # MATMULS ONLY, AND THE SHAPE PARSE IS THAT TEST -- it is the one thing that says this op
            # is a matmul rather than a LayerNorm or a datamove. `rows` is recorded for EVERY op, so
            # preferring it before this check counted every elementwise and movement op in the stage,
            # and those outnumber the matmuls: their row counts could carry the mode away from the
            # arithmetic the ceiling is about.
            _parsed = parse_matmul_shape(str((_o or {}).get("shape") or ""))
            if not _parsed:
                continue
            # The parse gives the PADDED M; `rows` is what the stage asked for. Prefer the latter,
            # fall back to the former for a profile written before the field existed.
            try:
                _m = int((_o or {}).get("rows") or 0) or _parsed[0]
            except (TypeError, ValueError):  # a field this did not write: the fingerprint still holds
                _m = _parsed[0]
            if _m > 0:
                _rows[_m] = _rows.get(_m, 0) + max(1, int((_o or {}).get("count") or 1))
    if not _rows:
        return 0
    # THE ROW COUNT MOST OF THE STAGE'S MATMULS RUN AT, not the biggest one. Ranking by FLOPs picks
    # whichever single matmul is widest, and on a decode step that is the vocab head -- which runs at
    # a TILE-PADDED 32 rows for a batch of 8. The stage would then have read as retiring 32 items and
    # been labelled a request-rate stage instead of one token per user. Every other matmul in the
    # step runs at the true row count, so the mode carries it and the padded outlier does not.
    # TIES GO TO THE LARGER COUNT. A tie is genuinely ambiguous -- equal numbers of matmuls at two
    # row counts -- and the two directions are not equally bad. Under-counting is the failure this
    # whole path exists to fix: it shrinks the compute roof and reports a compute-bound stage as
    # memory-bound, sending the reader after bandwidth. Over-counting overstates the roof, which
    # reads as headroom rather than as the wrong wall.
    return int(max(_rows.items(), key=lambda kv: (kv[1], kv[0]))[0])


def _stage_units(stage, prompt_tokens, profile=None) -> int:
    """How many items this stage retires in one unit of work.

    FROM WHAT THE RUN RECORDED, not from the stage's name and not inferred from the byte model. The
    first attempt asked perf_target whether a stage's read set grows with the prompt, and that says
    yes for a recurring stage too: decode READS the whole KV history to emit one token. It would have
    priced decode as though it processed 1024 items.

    The run knows: it records the prompt length against the stage that consumed it. A stage with a
    recorded item count retires that many per request; one without retires a single item, which is
    what a recurring stage does and the safe answer for a stage nobody has measured that way.
    """
    # TWO UNITS, AND ONLY ONE OF THEM IS MULTIPLIED BY THE BATCH.
    #
    # A count a stage STATES through <stage>_trace_items() is the total for one call, batch included
    # -- voxtral's prefill_trace_items returns PREFILL_C * B, and its encode traces at batch 1
    # whatever the pipeline serves. The legacy PERF_ISL_TOKENS marker is the prompt length PER
    # REQUEST and has to be multiplied. Reading both from one map and multiplying uniformly counted
    # prefill at 8x its real work and gave encode a batch it does not have.
    try:
        from cc_optimize.perf_mcp import read_stage_isl_map, read_stage_isl_per_request_map

        _total = int((read_stage_isl_map() or {}).get(str(stage)) or 0)
        if _total > 0:
            return _total
        _each = int((read_stage_isl_per_request_map() or {}).get(str(stage)) or 0)
    except Exception:  # noqa: BLE001
        _each = 0
    if _each > 0:
        return int(_each) * max(1, _request_batch())
    # OBSERVED, BEFORE DEFAULTED. Neither map named this stage, which used to mean ONE item -- a
    # number that is right for a recurring step and silently wrong for everything else. What the
    # stage actually ran is on record; ask that before falling back.
    _seen = _stage_items_observed(stage, profile)
    return int(_seen) if _seen > 0 else 1


# What the prompt-consuming row is called when the model declared no stages and therefore gave it no
# name. A display label only -- nothing is looked up by it.
_PROMPT_ROW_LABEL = "prefill"


def _unit_word(unit) -> str:
    """The vocabulary word behind a unit label, via the table in model_bytes that produced it.

    GUARDED LIKE EVERY OTHER `agent.` IMPORT IN THIS FILE. They resolve because the tool puts the
    perf_automation dir on sys.path (perf_test_mcp.py:21), which an importer that reaches this module
    another way has not done -- so every such site here degrades instead of raising, and these two
    did not. An unresolvable table means the unit cannot be read, which is not a token.
    """
    for _mod in ("agent.model_bytes", "models.experimental.perf_automation.agent.model_bytes"):
        try:
            import importlib

            return importlib.import_module(_mod).unit_word(unit)
        except Exception:  # noqa: BLE001
            continue
    return str(unit or "").strip().lower()


def _unit_is_token(unit) -> bool:
    """Whether the measured unit is one token, asked of the table rather than of the string's spelling."""
    return _unit_word(unit) == "token"


def _unit_work_name(unit) -> str:
    """What one unit of this model's work is called, from the unit the run measured.

    trace_replay's headline_unit already answers this structurally -- token / step / inference -- and
    prints the rate's unit alongside. Used only when the model declared no stages at all, so the one
    aggregate row is not labelled with a stage the model does not have.
    """
    # THE WORD BEHIND THE UNIT, from the table that produced it. This substring-matched its own
    # keyword list ("tok" / "step" / "denoise" / "diffus"), which was a guess about wording and a
    # second copy of a question model_bytes already answers by inverting _UNIT_LABEL.
    w = _unit_word(unit)
    if w == "token":
        return "decode"
    if w == "step":
        return "step"
    return "inference"


def stage_read_bytes(
    stage, *, model: str = "", task: str = "", measured=None, estimate=None, with_source: bool = False
):
    """THE bytes one unit of `stage` reads. Every consumer asks here; nobody derives it again.

    Three sources, in this order, and the order is the point:

      1. THE PINNED BASELINE (ledger KIND_STAGE_BYTES). The ladder shrinks the read set by design --
         bf16 -> bf8_b halves a weight -- so a ceiling recomputed each round retreats ahead of the
         measurement and is never reached. "The floor is a property of the IMPLEMENTATION, not a
         goal."
      2. THIS BUILD'S MEASUREMENT (trace_replay's dispatch hook: the distinct device tensors this
         stage's ops touched, observed while the stage ran alone). The first actual measurement of a
         stage's read set -- everything before it inferred the number from the checkpoint.
      3. THE ESTIMATE, last: the checkpoint apportioned by a tower NAME LIST, which is a guess about
         how a model spells its encoder.

    WHY THIS IS A FUNCTION AND NOT A LINE INSIDE _stage_roofs. Eleven commits have been about this
    one quantity, and each fixed whichever consumer looked wrong -- the census total, the anchor's
    key, the checkpoint inference, the pinning, the per-stage split. None began by asking who else
    read it, so each left the others on a different source. The twelfth symptom was a DECODE row
    whose floor said 2.350 GB and whose bandwidth said 4.784 GB, three lines apart, because the
    derivation lived inside one function's loop and the other renderer reached for the model-level
    figure instead. A quantity with one answer needs one place to ask.

    `estimate` is a callable so the fallback is only paid for when the two measurements are absent --
    it prices activations and KV from the model's geometry and is not free.
    """

    # WHICH OF THE THREE ANSWERED, reported by the one function that knows. A caller that needs to
    # label the number must not re-run the chain to find out -- that is a second opinion on the same
    # question, and two opinions on this quantity is the defect this function was extracted to end.
    def _ret(val, src):
        return (int(val), src) if with_source else int(val)

    try:
        _p = _pinned_stage_bytes(stage, model, task)
        if _p and int(_p) > 0:
            return _ret(_p, "pinned")
    except Exception:  # noqa: BLE001
        pass
    try:
        _m = int((measured or {}).get(stage) or 0)
        if _m > 0:
            return _ret(_m, "measured")
    except Exception:  # noqa: BLE001
        pass
    try:
        _e = estimate() if callable(estimate) else estimate
        return _ret(_e or 0, "estimate")
    except Exception:  # noqa: BLE001
        return 0


def _measured_bw_gbps(rf: dict, ms):
    """Achieved DRAM bandwidth for ONE stage: the bytes IT reads over the time IT took.

    THE ONLY PLACE THIS IS COMPUTED. It stood at two call sites as the same expression twice, each
    carrying a special case for the recurring stage that substituted the caller's MODEL-LEVEL
    bandwidth:

        _mb = bw_gbps if (rf["tokens"] == 1 and bw_gbps) else rf["bytes"] / ms

    and its comment explained why -- "the caller already computed this one from the same bytes;
    recomputing it here differs in the last digit". True while a stage's read set WAS the model-level
    one. It stopped being true the moment stages got their own measured bytes, and the special case
    became a second source rather than a rounding nicety: decode printed a 4.59 ms floor (2.350 GB)
    beside 360.5 GB/s (4.784 GB), two answers to one question, 2.04x apart, in the same three rows.
    Every other stage was self-consistent because only the recurring one took that branch.

    So the row divides by the bytes THAT ROW is built from, always. One owner, and the arithmetic is
    the definition of the quantity rather than a choice between two sources.
    """
    try:
        b = float((rf or {}).get("bytes") or 0)
        t = float(ms or 0)
        return (b / (t / 1000.0)) / 1e9 if (b > 0 and t > 0) else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


# The two ceiling inputs the renderer reads but must never write. Named here so the lookup is one
# function rather than a ledger import repeated at each use.
_LED_PARAMS = "matmul_params"
_LED_TOKENS = "stage_tokens"


def _pinned_ceiling_input(kind: str, stage, model: str = "", task: str = ""):
    """The pinned value for (kind, stage), or None. READ-ONLY -- see measurements.anchor_value."""
    try:
        led = _ledger()
        if led is None:
            return None
        return led.anchor_value(kind, depth=str(stage or "").strip().lower(), model=model, task=task)
    except Exception:  # noqa: BLE001
        return None


def _stage_roofs(active_bytes, peak_bw_gbps, tp_degree, unit, profile=None, stage_ms=None, model="", task=""):
    """Both ceilings for both stages, from the MODEL'S OWN facts rather than from summing annotated ops.

    THE ROOFS ARE ANALYTIC, WHICH IS WHY THIS NEEDS NO PER-OP STAGE LABEL. A stage's memory floor is
    the weights it streams over peak bandwidth, and its compute floor is 2 x params x tokens-in-the-unit
    over peak FLOPs -- perf_target already owns both formulas and already takes `tokens_per_unit`. The
    profile cannot answer this: `_top_ops` keys on (op_code, shape, memory) and records nothing about
    which phase an op ran in, so any attempt to split the op sums by stage is a guess. These two
    numbers are not.

    That distinction is the whole reason the roofline could previously state a compute band over a
    memory-bound stage: `annotate_op` kept only the WINNING floor, so compute was the only term that
    survived to the report, and the only renderable number is not the same as the right one.

    tokens_per_unit is what separates the stages: 1 for a decoded token, the declared sequence length
    for a prefill. The bytes are flat in it and the FLOPs are linear, which is exactly why the two
    stages sit under different roofs at all.

    Returns {stage: {memory_ms, compute_ms, flops, bytes, tokens, binds}} for the stages it can answer.
    """
    out = {}
    mf = _model_facts()
    if not (active_bytes and peak_bw_gbps):
        return out
    tp = max(1, int(tp_degree or 1))
    per_dev_bytes = float(active_bytes) / tp

    _share_bases: dict = {}
    _meas_bytes = _measured_stage_bytes(model, task)

    def _root_of(stage) -> str:
        """The subtree this stage runs, as stage_roots resolved it. "" when unmapped."""
        return str(((mf or {}).get("stage_roots") or {}).get(str(stage)) or "").strip()

    def _stage_share(stage) -> float:
        _v, _b = _share_and_basis(mf, stage)
        _share_bases[str(stage)] = _b
        return _v

    def _stage_block(stage):
        """The geometry of the block this stage runs, or None when it cannot be established.

        None means "use the model root", which is right for a single-block model and is the only
        shape that still publishes root geometry -- a multi-tower model emits no flat keys at all,
        precisely so a stage cannot silently inherit another tower's widths.
        """
        _root = str(((mf or {}).get("stage_roots") or {}).get(str(stage)) or "").strip()
        _blocks = (mf or {}).get("blocks") or {}
        _b = _blocks.get(_root) if _root else None
        return dict(_b) if isinstance(_b, dict) and _b else None

    def _bytes_for(stage, toks):
        """This stage's read set: the weights its subtree streams, plus what it alone carries.

        ONE RULE FOR EVERY STAGE. prefill and decode had a branch each and everything else fell
        through to zero, so a declared third stage got a row with no ceiling and a model declaring
        neither of those two got nothing at all. They are not special: they are stages whose subtree
        happens to be the language backbone.

        The per-item terms still come from perf_target, which owns the byte model and prices KV and
        activations from the model's own geometry. It knows two regimes; a stage outside them has no
        extra term, which is a missing addend rather than a missing ceiling -- the subtree's weights
        still give it a floor.

        The extra is a DIFFERENCE against the same regime with nothing in flight, so the weights
        cancel and this can never become a second opinion on them -- which is what kept decode
        excluded before, two ceilings 2.18x apart with the gate and the report disagreeing.
        """
        _share = _stage_share(stage)
        # A REFUSED SHARE IS NOT A SMALL ONE. 0.0 means "this stage's subtree is unknown on a model
        # with more than one", and the terms below ADD to the base -- so a refused stage came back
        # with a ceiling built from its activations alone, a confident number resting on no weights
        # at all. Nothing is the answer here, and the caller renders the row without a roof.
        if _share <= 0.0:
            return 0
        base = per_dev_bytes * _share
        # GATHERED WEIGHTS ARE RESIDENT, NOT STREAMED. The rule above -- one unit of work reads the
        # whole weight set once -- holds for every weight a matmul consumes and fails for a lookup
        # table. Voxtral's embed_tokens is [131072, 3072]: a decode step reads the ROW for its token,
        # about 6 KB, and is charged all 805 MB of it. That is 25% of decode's modelled read set,
        # which is why decode alone printed above 100% of peak; prefill carries the same error at 5%
        # of a much larger number and encode not at all, since audio_tower has no lookup.
        #
        # THE TOOL CANNOT TELL. embed_tokens and lm_head are the same shape, the same size and the
        # same checkpoint section -- one is gathered, one is streamed by the vocab matmul, and
        # nothing in the file separates them. The profiler could (EmbeddingsDeviceOperation against
        # MatmulDeviceOperation) but its buckets carry no byte counts. The pipeline holds both
        # tensors and knows, so it says, exactly as it states its per-stage item counts.
        #
        # Subtracted in full. What is actually read is items x row, which is 6 KB for a decode step
        # and 25 MB across a 4096-token prefill -- under half a percent of either read set, against
        # the 25% error it replaces.
        base = max(0.0, base - float(((mf or {}).get("gathered_weight_bytes") or {}).get(_root_of(stage), 0.0)))
        if not mf:
            return base
        # NO NAME TEST. perf_target owns which regimes it can price and raises for the rest, so ASK
        # it rather than keeping a second list here that has to be updated in step. A stage it cannot
        # price simply has no extra term -- a missing addend, not a missing ceiling, since the
        # subtree's weights already give it a floor.
        # CONTEXT IS THE WORKLOAD'S, NOT THE STAGE'S. Every stage in a run sees the same prompt; what
        # differs is how many units each retires, and that is `toks`, which prices FLOPs above. Passing
        # `toks` here conflated the two -- decode's unit count of 1 became a context of one token -- so
        # the run's ISL is passed uniformly and batch is passed beside it, exactly as the byte model
        # expects. For a prompt-consuming stage that reproduces seq_len x batch as before.
        _seq = int(_prompt_tokens() or 0)
        if not _seq:
            return base
        try:
            from agent.perf_target import active_bytes as _ab

            _b = max(1, _request_batch())
            # ITEMS PER REQUEST is what this stage does in one unit -- `toks` already includes the
            # batch, so dividing it out keeps the two factors from being applied twice.
            _items = max(1, int(round(float(toks or 1) / float(_b)))) if toks else 0
            # THIS STAGE'S OWN GEOMETRY. KV and activations are computed from layer count, hidden and
            # intermediate width -- properties of the BLOCK the stage runs, not of the model. Read
            # from the root they were a chimera on any multi-tower model: the deepest tower's depth
            # with another tower's widths, so the audio encoder carried 3072-wide activations through
            # 32 layers it does not have. None for a single-block model, where root geometry IS the
            # block's and nothing changes.
            _blk = _stage_block(stage)
            # A KV TERM BELONGS TO A STAGE IN THE KV LOOP, AND NOTHING ELSE PRICES IT OUT.
            #
            # `_seq` is the WORKLOAD's prompt, and active_bytes turns any non-zero seq_len into a KV
            # read of 2 x layers x kv_heads x head_dim x seq_len x batch. For voxtral's audio encoder
            # that is 32 layers x 20 kv heads x 64 head_dim x 128 text tokens x 8 -- about 168 MB of
            # cache traffic charged to a stage that never sees the text prompt and keeps no cache at
            # all. A Whisper-style encoder attends bidirectionally within one forward pass; its K and
            # V are intermediates, and the activation term below already prices those.
            #
            # WHICH STAGES ARE IN THE LOOP is recorded, not guessed. The recurring stage retires one
            # item per unit and reads the accumulated history; the prompt-consuming stage is the one
            # the per-request marker names, and it writes the cache the recurring stage then reads.
            # A stage that is neither -- an encoder, a vocoder, a diffusion step -- gets seq_len 0,
            # which zeroes the KV term and leaves its activations untouched.
            _in_kv_loop = int(_items or 0) <= 1 or str(stage) in _per_request_stages()
            _seq_for_stage = _seq if _in_kv_loop else 0
            _extra = float(
                _ab(mf, regime=stage, seq_len=_seq_for_stage, batch=_b, items=_items, block=_blk) or 0.0
            ) - float(_ab(mf, regime=stage, seq_len=0, batch=1, items=0, block=_blk) or 0.0)
        except Exception:  # noqa: BLE001 -- regime unknown to the byte model, or no byte model at all
            return base
        return base + max(0.0, _extra) / tp

    params = 0
    try:
        from agent.perf_target import ceiling_params as _cp

        params = int(_cp(mf or {}) or 0)
    except Exception:  # noqa: BLE001
        params = int((mf or {}).get("total_params") or 0)
    # THE PEAK IS THE ONE THE MODEL ACTUALLY RUNS AT. chip_peak_flops defaults to HiFi4 when handed no
    # fidelity, and HiFi4 is a QUARTER of LoFi on Blackhole -- so a LoFi model was being priced against
    # 175 TFLOPS instead of 702, making its compute roof 4x too slow and its utilisation 4x too
    # flattering. The profile reports the fidelity every matmul ran at; the dominant one by FLOP share
    # is a measurement, not a default.
    peak_flops, _dom = 0.0, ""
    try:
        _fid_rows, _ = _fidelity_breakdown(profile or {})
        if _fid_rows:
            _t = max(_fid_rows, key=lambda r: r[1])
            _dom = str(_t[0]) if _t[1] else ""
    except Exception:  # noqa: BLE001
        _dom = ""
    try:
        from agent.environment import ARCH_FACTS
        from agent.perf_target import chip_peak_flops as _cpf

        _arch = str(os.environ.get("PERF_MCP_ARCH") or "blackhole").strip().lower()
        peak_flops = float(_cpf(ARCH_FACTS.get(_arch) or {}, _dom) or 0.0)
    except Exception:  # noqa: BLE001
        peak_flops = 0.0
    # THE PINNED PEAK OUTRANKS THE ONE JUST DERIVED, for the same reason the byte anchor outranks the
    # snapshot: the value above describes the CURRENT picture, and _promote_baseline replaces that
    # picture on every profile_model call. A ceiling recomputed from it retreats ahead of the
    # measurement -- the defect KIND_FLOOR was introduced to fix, on a roof that never got the fix.
    #
    # The derived value is kept as _peak_flops_now: it is what separates a real fidelity win from a
    # stale reading when a stage measures faster than its pinned roof, exactly as _floor_anchor's
    # caller uses the current floor. Falling back to it when nothing is pinned keeps every existing
    # path alive -- a report rendered before any anchor exists still gets a ceiling.
    _peak_now = peak_flops
    _pinned = _pinned_peak_flops(unit, model, task)
    if _pinned and _pinned > 0:
        peak_flops = float(_pinned)
    # THE RECURRING STAGE EXISTS FOR EVERY UNIT, not only for tokens. Gating it on "tok" deleted the
    # whole ceiling for a diffusion model, whose unit of work is a denoise STEP and which is exactly
    # as memory-bound per step as an LLM is per token. PREFILL is the part that is token-specific:
    # a model that does not consume a prompt has no prefill to price, so it gets one stage, not two.
    # DECODE'S unit is one token per USER (tok/s/u is per user), so batch does not multiply it.
    # PREFILL'S unit is the whole in-flight request set: seq_len tokens for each of `batch`.
    # THE MODEL SAYS WHICH STAGES IT HAS, and nothing else does.
    #
    # This was the literal [("decode", 1)] plus an optional prefill, so the table could only ever
    # describe an autoregressive text model. Two ways that is wrong. A stage the model declares and
    # the harness MEASURES could not appear -- voxtral's audio encoder, timed at 12.79 ms, twice its
    # own decode, shown in the stage breakdown and then absent from the roofline with no indication
    # the reader was looking at 116 of 129 ms. And a model with no decode at all -- a classifier, a
    # vision tower, a vocoder, a diffusion denoiser -- was still given a DECODE row it does not have.
    #
    # stage_ms is written from the model's own PIPELINE_STAGES by the run that measured them, so it
    # is the authority. Declared order is kept: it is the order the pipeline runs in.
    _pt = _prompt_tokens() if _unit_is_token(unit) else 0
    _declared = [str(k) for k in (stage_ms or {}) if k]
    if _declared:
        stages = [(n, _stage_units(n, _pt, profile)) for n in _declared]
    else:
        # NOTHING DECLARED. A model that never reported its stages still gets the recurring unit --
        # every model has one -- and a prompt-consuming stage when it consumes a prompt. This is the
        # old behaviour, reached only when there is nothing better to go on.
        #
        # NAMED FOR THE UNIT THE RUN MEASURED, not for the two an LLM has. `unit` comes from
        # trace_replay, which derives it from the decode_step CONTRACT rather than from any stage
        # name: tok/s/u means the pipeline retires one token per call and the pair below describes
        # it. Anything else -- a diffusion step, a classifier's forward pass -- was handed a DECODE
        # row it does not have, on the exact reasoning this file's docstring calls the original bug
        # ("a model with NO decode was still handed a DECODE row"), left in place because it was the
        # fallback rather than the main path.
        # THE UNIT NAMES THE ROW, in both arms. The token arm hardcoded "decode" while the arm right
        # beside it already asked _unit_work_name -- which returns exactly "decode" for a token unit,
        # so this is the same output from the one function that owns the question.
        stages = [(_unit_work_name(unit), 1)]
        if _pt and _unit_is_token(unit):
            # The prompt-consuming pass. A model that declared no stages has no name for it, so this
            # label is the tool's own and is not looked up anywhere: _stage_block returns None for it
            # either way. Kept rather than invented from the unit ("decode-prompt") because the label
            # is what the report prints and the convention is what a reader expects.
            stages.insert(0, (_PROMPT_ROW_LABEL, _pt * max(1, _request_batch())))
    for name, toks in stages:
        # EVERY TERM FROM THIS STAGE'S OWN BLOCK. params, layers and hidden were read from the model
        # root for every stage alike, so on a multi-tower model the audio encoder was priced with the
        # language backbone's 4.014e9 parameters and 3072-wide attention -- 0.041 ms against a
        # measurement in the tens of milliseconds. A single-block model has root geometry equal to
        # its one block's, so nothing changes there.
        _blk = _stage_block(name) or {}
        # THE PARAMS THAT ARE MULTIPLIED, NOT THE PARAMS THAT EXIST.
        #
        # `2 x params x tokens` counts a multiply-accumulate for every parameter, once per token --
        # which is right for a WEIGHT in a matmul and wrong for a lookup table. An embedding is read
        # by INDEX: one row per token, no multiply. blocks[root]["params"] is the tower's SIZE, so it
        # includes the table, and feeding it to this formula charges work that never happens.
        #
        # Voxtral prefill: 4.014B (language_model) instead of 3.611B, so 2 x 0.403B x 4096 = 3.30
        # TFLOP of phantom matmul -- 18.8 ms of a 222.61 ms floor, 9.2% too slow, making the stage
        # look closer to its ceiling than it is. Encode is unaffected: an audio tower has no token
        # embedding, which is why only the text stages drifted.
        #
        # The rule already exists -- model_bytes._LOOKUP_ONLY, and total_params is built with it --
        # but blocks[] records a tower's size and never inherited it. Prefer the block's own
        # matmul_params when the producer supplies it; fall back to subtracting the declared
        # lookup-only bytes; and only then to the raw size, so a model whose producer predates this
        # behaves exactly as before.
        _blk = dict(_blk)
        _mm = int(_blk.get("matmul_params") or 0)
        if not _mm:
            _lo = int(_blk.get("lookup_params") or 0)
            _mm = max(0, int(_blk.get("params") or 0) - _lo) if _lo else 0
        _params = _mm or int(_blk.get("params") or 0) or int(params or 0)
        # PINNED FIRST, like every other ceiling input. Read-only here: anchor_value never writes, so
        # rendering a report cannot move the number it is reporting -- the producers (the arch mirror
        # for params, trace_replay's observed counts for items) do the pinning where the value is
        # made. Without this the compute roof was the last input still re-derived from whatever the
        # model currently looks like, so a lever that reshaped a block moved the ceiling under the
        # measurement chasing it.
        try:
            _lp = _pinned_ceiling_input(_LED_PARAMS, name, model, task)
            if _lp and int(_lp) > 0:
                _params = int(_lp)
        except Exception:  # noqa: BLE001
            pass
        _L = int(_blk.get("layers") or (mf or {}).get("layers") or 0)
        _H = int(_blk.get("hidden_size") or (mf or {}).get("hidden_size") or 0)
        # 2 x params x tokens counts every WEIGHT matmul -- each parameter is multiplied once per
        # token, so every projection in every layer is already in there. What it omits is the
        # attention score path, QK^T and A.V, which uses no parameters at all and scales with the
        # SQUARE of the sequence: 4 x layers x tokens^2 x hidden. Absent, prefill's compute floor is
        # understated by 0.4% at ISL 128 -- invisible, which is why it survived -- but 3.3% at 1024
        # and 21.3% at 8192, where it decides whether the stage reads compute- or memory-bound.
        try:
            _lt = _pinned_ceiling_input(_LED_TOKENS, name, model, task)
            if _lt and int(_lt) > 0:
                toks = int(_lt)
        except Exception:  # noqa: BLE001
            pass
        _attn = (4.0 * _L * float(toks) * float(toks) * _H) if (_L and _H) else 0.0
        flops = ((2.0 * float(_params) * float(toks) + _attn) / tp) if _params else 0.0
        # THIS STAGE'S OWN PEAK, when the capture marked its ops. The value resolved above is the
        # dominant fidelity across the WHOLE profile, applied to every stack -- one variable, used
        # three times. It is right only while every stack runs the same math mode: on voxtral encode,
        # the projector and prefill are pure HiFi4 (so 175.5 is their true constant) while decode
        # carries a HiFi2 lm_head worth 3.299e12 of its 5.907e12 FLOPs, whose true peak is 351.0.
        # Harmless there because decode binds on memory by ~230x, and wrong the moment the fidelity
        # rung lands on a stack that binds on compute.
        # THIS STAGE'S OWN PEAK when the capture marked its ops; the whole-profile figure otherwise.
        # One variable shared by three stacks was only ever a consequence of there being one number.
        _stage_peak, _stage_dom = _peak_for_stage(name, profile, model, task)
        _pk_use = _stage_peak or peak_flops
        comp_ms = ((flops / _pk_use) * 1000.0) if (flops and _pk_use > 0) else None
        # MEASURED FIRST, APPORTIONED SECOND -- which is what _roofline_stage_share's docstring has
        # always claimed and what the order here contradicted. The profile records what a stage
        # ACTUALLY read; a share is that same quantity estimated from the checkpoint's split. Asking
        # the estimate first meant a recorded read set was consulted only when apportionment produced
        # nothing at all, so the better evidence was reachable only through the worse one's failure.
        #
        # A THIRD TOWER IS NOT THE BACKBONE. An audio encoder streams its own weights and nothing of
        # the language model, so the whole-model divisor is wrong for it rather than approximate. When
        # neither a measurement nor a subtree share is available the roof is WITHHELD -- and the stage
        # still gets its row, because the measurement is real and the gap is worth seeing.
        # PINNED BASELINE, THEN THIS BUILD'S READING, THEN THE ESTIMATE. _measured_stage_bytes is
        # trace_replay's observation of the distinct device tensors this stage's ops touched, taken
        # while the stage ran in isolation. The pin keeps the target still while the ladder shrinks
        # the read set underneath it; _bytes_for is the last resort, apportioning the checkpoint with
        # a tower NAME LIST.
        #
        # There used to be a fourth entry here, _stage_measured_bytes(profile, name), which looked
        # like the measurement and was not one: it summed a `bytes` key buckets do not carry, keyed on
        # a `stage`/`regime` field it read at the wrong level, and the right level says "na" for every
        # bucket anyway. Three ways of returning 0, so the line always fell through to the estimate
        # while reading as though it preferred a measurement. Deleted rather than repaired -- the real
        # observation is the one above it now.
        # ONE PLACE TO ASK. See stage_read_bytes: the order pinned -> measured -> estimate is the
        # whole property, and it lived inline here while other renderers reached elsewhere.
        _b, _b_src = stage_read_bytes(
            name,
            model=model,
            task=task,
            measured=_meas_bytes,
            estimate=lambda: _bytes_for(name, toks),
            with_source=True,
        )
        _b_now = int((_meas_bytes or {}).get(name) or 0)
        # WHERE THE NUMBER CAME FROM, carried beside it. The three sources render identically today --
        # a pinned measurement, this build's measurement and a checkpoint estimate are all just a
        # figure in the GB/s column -- so a read set that was never measured looks exactly like one
        # that was. That is not a cosmetic gap: the byte hook watched a ttnn class this build does not
        # instantiate and returned 0 for every stage of every run for two days, and nothing on the
        # page disagreed, because the estimate fallback kept printing plausible numbers.
        mem_ms = (_b / (float(peak_bw_gbps) * 1e9)) * 1000.0 if _b else None
        out[name] = {
            "share_basis": _share_bases.get(name, ""),
            "memory_ms": mem_ms,
            "compute_ms": comp_ms,
            "flops": flops or None,
            "bytes": _b,
            "bytes_source": _b_src,
            "tokens": toks,
            "peak_flops": _pk_use or None,
            # THE RUNG THIS STAGE RESOLVED FOR ITSELF, empty when it fell back to the whole-profile
            # figure. `fidelity` below cannot answer that -- it is `_stage_dom or _dom`, so it is
            # populated either way. This key was READ by the shared-peak caveat and written nowhere,
            # so `any(...)` was always False and the report claimed "one peak shared by every stack"
            # even when every stage had resolved its own from its own ops. Measured on voxtral run
            # 38: encode, prefill and decode each resolved hifi4 independently and the caveat still
            # printed.
            "peak_stage": _stage_dom or "",
            # This build's observed read set beside the pinned one, so a dtype win shows as the
            # bytes genuinely falling rather than as the ceiling quietly following them down.
            "bytes_now": _b_now or None,
            # What the CURRENT build's mode implies, carried beside the pinned roof so a reader (and
            # the >100% classifier) can tell "the model got faster" from "these numbers are stale".
            "peak_flops_now": _peak_now or None,
            "fidelity": _stage_dom or _dom,
            # The binding roof is the SLOWEST one -- the stage cannot beat its tightest floor. Stated
            # per stage because it genuinely differs: prefill's FLOPs scale with the sequence and
            # decode's do not, so the same model can be compute-bound in one stage and not the other.
            "binds": (
                "compute"
                if (comp_ms is not None and mem_ms is not None and comp_ms > mem_ms)
                else ("memory" if mem_ms else ("compute" if comp_ms else None))
            ),
        }
    return out


def _fidelity_breakdown(profile):
    """[(fidelity, flops, peak_tflops, floor_ms), ...] and the summed compute ceiling, or (None, None).

    Each op against ITS OWN peak, not one blanket figure. Peak differs 4x across the fidelity modes
    (Blackhole: LoFi 5.4 / HiFi2 2.7 / HiFi3 1.8 / HiFi4 1.35 TFLOPS per core), so assuming LoFi gives
    a ceiling unreachable without dropping precision, and assuming HiFi4 punishes a model for the LoFi
    work it has already done. The mix is the only honest total.

    The SPLIT is the actionable part: a fidelity slice that is a small share of the FLOPs but a large
    share of the floor is exactly where the `fidelity` rung pays, and a single blended number hides it.

    MATMUL ONLY -- a compute floor needs parsed MxKxN dims, so LayerNorm, BinaryNg and the rest
    contribute nothing. This is a lower bound on prefill compute, not the whole of it.
    """
    try:
        from agent import roofline as _rf
        from agent.environment import ARCH_FACTS

        # PASS THE ARCH. _facts() keys off env["arch"]; an empty env leaves peak_tflops_per_core
        # unset, ideal_ms_compute returns None, and no op ever carries a compute floor -- the
        # breakdown then silently renders "not modelled" on a profile that has everything it needs.
        _arch = str((profile or {}).get("arch") or os.environ.get("PERF_MCP_ARCH") or "blackhole").lower()
        rep = _rf.residual_report(profile or {}, {"arch": _arch})
        ops = [o for o in (rep.get("open_ops") or []) if o.get("compute_ms") and o.get("flops")]
        if not ops:
            return None, None
        bh = ARCH_FACTS.get(_arch) or ARCH_FACTS.get("blackhole") or {}
        cores = int(bh.get("grid_x") or 0) * int(bh.get("grid_y") or 0)
        peaks = bh.get("peak_tflops_per_core") or {}
        agg = {}
        for o in ops:
            f = str(o.get("fidelity") or "hifi4").lower()
            fl, ms = agg.get(f, (0, 0.0))
            agg[f] = (fl + int(o["flops"]), ms + float(o["compute_ms"]))
        # ALWAYS all four modes, present or not. A mode with no ops still states its peak, so the
        # reader sees the whole ladder the model could be sitting on rather than only where it
        # happens to sit today -- and a zero row is the visible answer to "what would HiFi4 cost".
        order = ["lofi", "hifi2", "hifi3", "hifi4"]
        keys = order + [f for f in agg if f not in order]
        rows = [(f, agg.get(f, (0, 0.0))[0], (peaks.get(f) or 0.0) * cores, agg.get(f, (0, 0.0))[1]) for f in keys]
        return rows, sum(r[3] for r in rows)
    except Exception:  # noqa: BLE001
        return None, None


def _roofline_tables(
    *,
    unit,
    theo,
    band,
    measured,
    bw_gbps,
    peak_bw_gbps,
    active_bytes,
    per_unit_ms,
    profile,
    tag="",
    note="",
    stage_ms=None,
    stage_paths=None,
    stage_ops=None,
    tp_degree=1,
    model="",
    task="",
):
    """The Roofline / Overheads / Utilization blocks.

    Three blocks rather than one, because the middle column means different things. A roofline row
    has a spec ceiling and a sustained band (60-80% dense, 37.5-50% MoE -- what silicon actually
    delivers). Dispatch and capacity have neither: dispatch is overhead whose target is zero, and
    capacity is a hard wall with a safety margin. Printing "achievable" over all four invited reading
    26% dispatch as a grade rather than as a quarter of every token being wasted.

    Every row renders only from inputs that exist; a missing one prints `not measured` instead of
    being dropped, so the report states what it does not know.

    `tag` and `note` are the two things a rate cannot be published without. `tag` is the profiling
    depth: tok/s/u is an ABSOLUTE throughput, so a 16-layer window on a 32-layer model reads about
    twice the real figure, and a rate printed bare is a rate somebody will quote. `note` is the
    reason a measurement was withheld -- "not measured" alone reads as a gap in the harness when the
    truth is that the value was computable and refused (a truncated window against a full-depth
    ceiling makes the ratio meaningless, not merely optimistic).
    """
    W = 100
    rule = "\u2500" * W
    out = []
    # The unit word, derived ONCE from the declared unit. It used to be hardcoded "step" one column
    # from a MEASURED cell already reading "ms/token" -- two names for one unit, side by side.
    _step = unit.split("/")[0].replace("tok", "token") if "/" in unit else "step"
    # THE SUSTAINED FRACTION IS THE MODEL'S, NOT A CONSTANT. Dense silicon delivers 60-80% of spec;
    # an MoE reads a fraction of its weights per token and perf_target bands it at 37.5-50%. Hardcoding
    # 0.60/0.80 here would print a dense band over an MoE ceiling -- so it is read back off the band
    # perf_target already computed, and only falls to 60-80 when there is no band to read.
    _LOF, _HIF = 0.60, 0.80
    if band and band[0] and band[1] and theo:
        _LOF, _HIF = float(band[0]) / float(theo), float(band[1]) / float(theo)

    # COLUMN GEOMETRY IN ONE PLACE. Number and unit occupy fixed sub-fields, so the digits form a
    # straight line down the page and the unit words start at their own column. Left-aligned on the
    # FIRST DIGIT rather than the decimal point: the magnitudes here run from 0.03 ms to 29412
    # tok/s/u, and no single decimal column serves both.
    def _n(v):
        """A duration, to hundredths -- and to thousandths below 0.1 ms. One fixed %.1f printed
        decode's compute roof as 0.0: a roof three orders below the memory roof IS the finding, so
        rounding it away deletes the answer."""
        return ("%.3f" if abs(float(v)) < 0.1 else "%.2f") % float(v)

    def _pct_of_ceiling(got, ceiling):
        """Tenths, and hundredths below 1% -- a roof the run barely touches (decode reaches 0.1% of
        its compute ceiling) must not round to the same 0% as no data."""
        p = 100.0 * float(got) / float(ceiling)
        return "%s%% of ceiling" % (("%.2f" if p < 1 else "%.1f") % p)

    def _r(v):
        """A rate. Tenths, except where the number is large enough that a tenth is noise."""
        a = abs(float(v))
        return ("%.2f" if a < 1 else "%.0f" if a >= 10000 else "%.1f") % float(v)

    def _al(num):
        """ALIGNED ON THE FIRST DIGIT: every number starts in the same column.

        The alternative is aligning on the decimal point, which packs the digits more tidily but
        leaves each number starting somewhere different -- and the column then has no left edge to
        scan down. The field is wide enough that the widest value still clears its unit."""
        return "%-8s" % str(num)[:8]

    def _cell(num, u=""):
        return "%-10s%-12s" % (num, u)

    def _ncell(num, u=""):
        return "%s%-8s" % (_al(num), u)

    def _bandcell(lo, hi, u=""):
        # THE LOW VALUE IS RIGHT-ALIGNED so the dash sits BETWEEN the two numbers instead of
        # drifting. Left-aligned, a short low value ("27.5") left a gap and the dash ended up hard
        # against the high one -- reading as a sign on it rather than as a range between them. The
        # dash now lands on one fixed column, tight to both.
        return "%8s \u2013 %s%-8s" % (str(lo)[:8], _al(hi), u)

    def _row(label, theo_s="", band_s="", meas_s=""):
        # A divider closes the LABEL column too. Without it the stage name and the roof names ran
        # straight into the THEORETICAL numbers with no edge between them, so the first column was
        # the only one on the page without a boundary.
        return (" %-30s\u2502 %-16s\u2502 %-28s\u2502 %s" % (label, theo_s, band_s, meas_s)).rstrip()

    _row2 = _row

    def _rule4():
        """Derived from a real row, so the crosses cannot drift from the dividers when a field
        width changes -- which is exactly what happened when they were counted by hand."""
        return "".join("\u253c" if c == "\u2502" else "\u2500" for c in _row("", "", "", "x").ljust(W))

    def _hrule():
        return "\u2500" * W

    out.append("Roofline")
    # SAY WHAT THE NUMBERS BELOW ARE PER. Every figure in this table is per unit -- per request, per
    # token, per pass -- and the batch decides how many of those a single step retires, so the same
    # measurement reads eight ways on an eight-user run. Voxtral serves 8 and the table said nothing,
    # so a reader had no way to tell a per-user figure from an aggregate one.
    #
    # Stated as UNKNOWN rather than 1 when nothing resolved it: TT_PERF_BATCH carries 0 for "ask the
    # pipeline", so a 1 here would be a guess dressed as a fact, and it is exactly the guess that made
    # an 8-user step get priced as a 1-user step.
    _bs = _request_batch()
    # A RESOLVED batch counts as reported, however it was resolved: the run recording 8 is a stronger
    # statement than the operator exporting it, and reading only the environment is what let a real
    # eight-user run print "not reported".
    _bs_declared = _bs > 1 or any(
        str(os.environ.get(v) or "").strip().isdigit() and int(os.environ.get(v)) > 0
        for v in ("TT_PERF_BATCH", "PERF_MCP_BATCH", "TT_PERF_BATCH_SIZE")
    )
    out.append("  batch: %s" % (("%d" % _bs) if _bs_declared else "not reported; ceilings assume 1"))
    out.append(rule)
    out.append(_row("", "THEORETICAL", "ACHIEVABLE %.0f-%.0f%%" % (_LOF * 100, _HIF * 100), "MEASURED"))
    out.append(_rule4())

    # A MEASUREMENT ABOVE THE CEILING IS NOT A GOOD SCORE. The ceiling is peak bandwidth over the
    # model's bytes -- exceeding it means the pair is inconsistent (a stale target, or a reading from
    # a shallower window), not that the model beat physics.
    _exceeds = bool(measured and theo and measured > theo)
    # TTFT IS THE PROMPT-CONSUMING STAGE'S TIME, and which stage that is follows from what it does --
    # retiring many items per unit -- not from being called "prefill". Resolved after the roofs are
    # built, below, so the item counts are available; None until then.
    _pf_measured = None
    _roofs = _stage_roofs(active_bytes, peak_bw_gbps, tp_degree, unit, profile, stage_ms, model=model, task=task)
    # ONE CEILING, NOT TWO THAT NEARLY AGREE. `theo` is what perf_target published and what the stop
    # gate judges against -- and on the anchored path it is recomputed from the ledger, not from the
    # snapshot this function was handed. Re-deriving the decode memory roof from bytes here would put
    # a second, almost-identical ceiling in the same report, and "almost" is how a run gets banked
    # against one number and reported against another.
    # THE HEADLINE CEILING BELONGS TO THE RECURRING STAGE -- the one retiring a single item per unit,
    # which is what a per-token/per-step figure measures. Keyed on the name "decode", a model whose
    # recurring stage is called anything else kept a second, almost-identical ceiling in the same
    # report, and "almost" is how a run gets banked against one number and reported against another.
    # THE LAST recurring stage, in declared order: a pipeline's output comes from the stage it ends
    # on. Taking the first put the model's per-unit ceiling on an encoder that happens to retire one
    # item per pass, and printed it under that heading instead of the generating stage's.
    for _st_x, _rf_x in _roofs.items():
        if int((_rf_x or {}).get("tokens") or 0) > 1:
            _v = (stage_ms or {}).get(_st_x)
            if isinstance(_v, (int, float)) and _v > 0:
                _pf_measured = float(_v)
            break
    _recurring_st = next(
        (st for st, rf in reversed(list(_roofs.items())) if int((rf or {}).get("tokens") or 0) == 1), None
    )
    # ONE CEILING, AND IT IS THE BETTER-INFORMED ONE. This replaced the recurring stage's own roof
    # with the model-level ceiling, to stop two almost-identical numbers appearing in one report --
    # right about the risk, wrong about which to keep. The model-level figure divides by the WHOLE
    # resident model, including towers the recurring stage never reads; the stage roof divides by
    # that stage's own subtree and adds the KV it carries. On voxtral: 3.36 ms against 3.15 ms, the
    # difference being an audio encoder a decoded token does not touch.
    #
    # So the overwrite happens only when the stage has no roof of its own to keep -- a model whose
    # towers cannot be established still gets the whole-model ceiling, exactly as before.
    #
    # NOT YET RECONCILED WITH THE STOP GATE, which computes from the same whole-model anchor
    # (perf_mcp.py, compute_target(mf, ...)). The gate is therefore ~7% more generous than this row
    # for a two-tower model. That is a real disagreement and it is the next thing to fix; printing
    # the wrong number here to match it would only hide it.
    if _recurring_st and theo and not (_roofs.get(_recurring_st) or {}).get("memory_ms"):
        _roofs[_recurring_st]["memory_ms"] = 1000.0 / float(theo)
    _fid, _cc = _fidelity_breakdown(profile)
    _cc_floor = float(_cc) if (_cc and _cc > 0) else None
    # Which fidelity the model ACTUALLY runs at, by FLOP share. The per-fidelity rows below price the
    # stage's own FLOPs at every rung of the ladder; without this mark they state what each rung would
    # cost but not which one is being paid for.
    _in_use = ""
    if _fid:
        _top = max(_fid, key=lambda r: r[1])
        _in_use = str(_top[0]) if _top[1] else ""
    # AND SAY WHEN IT IS ONE VERDICT FOR EVERY STACK. Without stage signposts the capture cannot
    # attribute an op to a stage, so this whole-profile figure is stamped across encode, prefill and
    # decode alike -- stacks that do not share a rung. voxtral: 60.3% of matmul FLOPs are LoFi and
    # 39.7% HiFi4, so every stack was priced at LoFi's 702 TFLOPS while encode and prefill run HiFi4
    # at 175.5, a ceiling 4x too generous. The fallback is correct (inventing a per-stage peak would
    # be worse); printing it as though it were three measurements is not.
    _shared_peak = bool(_in_use) and not any((_roofs.get(_s) or {}).get("peak_stage") for _s in (_roofs or {}))

    # BOTH ROOFS, BOTH STAGES. Only the WINNING floor used to survive `annotate_op`, so compute was
    # the only renderable term and the report printed a compute band over a memory-bound stage. Being
    # the only number that can be drawn is not the same as being the right one.
    if tag:
        out.append(_row(" " + tag.strip()))
    if _exceeds:
        out.append(_row(" \u2717 measured EXCEEDS ceiling \u2014 target stale/suspect (re-profile)"))
    if note and not measured:
        # Wrapped to the LABEL field, not to the page. At W-40 the note ran past the first divider
        # and pushed the column edges out on its own rows.
        for _ln in _wrap_note(str(note), 56):
            out.append(_row(" " + _ln))

    def _fidelity_section():
        """The precision ladder, ONCE, with the stages as columns.

        It used to render inside each stage's compute roof, which put it in the three-column grid --
        where its two values (a peak, and what this stage's FLOPs cost at that peak) landed under
        THEORETICAL and ACHIEVABLE 60-80%, and the second read as a sustained band it is not. Adding
        sub-labels only put two headers over one column, and the block still duplicated per stage.

        It is not a measurement, it is a what-if: what the arithmetic WOULD cost at each precision.
        That is its own kind of statement and gets its own section, and since the peaks are the same
        for every stage the stages are columns rather than repeated blocks."""
        # EVERY STAGE WITH A COMPUTE TERM, in declared order -- not the two an LLM has.
        _cols = [(st, _roofs[st]) for st in _roofs if _roofs.get(st) and _roofs[st].get("flops")]
        if not (_fid and _cols):
            # NAME THE MISSING INPUT. This returned [] and the section simply was not there, which
            # reads as "the tool does not do that" rather than "the tool could not". On gemma-3 the
            # ladder vanished because the model facts had no param count -- no params, no FLOPs, no
            # compute roof, no ladder -- and the report gave the reader nothing to act on. One line
            # naming the cause turns a silent gap into a fixable one.
            if not _cols:
                return [
                    "",
                    "Fidelity ladder",
                    "\u2500" * W,
                    "  not shown: no compute roof for any stage " "(needs a param count in perf_target_inputs.json)",
                ]
            return [
                "",
                "Fidelity ladder",
                "\u2500" * W,
                "  not shown: the profile carries no matmul with "
                "parsed MxKxN dims, so no per-precision floor can be computed",
            ]
        _o = ["", "Fidelity ladder"]  # the rules are appended once the table's own width is known
        # Ruled and spaced like the tables above it, so the section does not read as loose text
        # dropped between two grids.
        # EACH STAGE'S OWN RUNG, MARKED IN ITS OWN COLUMN. One trailing arrow can only ever name a
        # single precision, and it was chosen by whole-profile FLOP share -- so the stack with the
        # most FLOPs spoke for all of them. On voxtral prefill carries ~190 of ~365 GFLOP, so the
        # first fidelity lever landing on prefill alone would have moved the arrow to prefill's rung
        # and the table would have read as though encode and decode moved with it. They had not, and
        # the remaining headroom on two of three stacks would have looked already spent.
        #
        # A stage that could not resolve its own rung shows the whole-profile one, which is what it
        # is actually priced at; the caveat below then says so once rather than per column.
        _own = {}
        for _st_o, _rf_o in _cols:
            _own[_st_o] = str((_rf_o or {}).get("peak_stage") or "").strip().lower() or _in_use

        def _cell(_st_c, _rf_c, _rung, _peak):
            """This stage's cost at this rung, marked when it is the rung the stage runs at."""
            _v = _n((_rf_c["flops"] / (_peak * 1e12)) * 1000.0)
            return ("%s \u2190 in use" % _v) if _own.get(_st_c) == str(_rung) else _v

        # Widened from 18 so a marked cell does not overflow its column; the rule under the header is
        # derived from the header itself, so it follows automatically.
        _fr = " %-14s\u2502 %-18s" + "\u2502 %-20s" * len(_cols) + "\u2502 %s"
        _hdr = _fr % (("precision", "peak") + tuple("%s ms" % st for st, _ in _cols) + ("",))
        # THE FRAME FOLLOWS THE TABLE. All three rules were a flat W wide while the header grew with
        # the stage count and the column width, so a wide table overflowed its own frame by however
        # much it exceeded W -- visible the moment the columns widened to fit a marked cell. W stays
        # the floor so a narrow table still lines up with the sections above it.
        _hdr = _hdr.rstrip()
        _rows = []
        for _f, _fl, _pk, _fms in _fid:
            if not _pk:
                continue
            _nm = str(_f).replace("lofi", "LoFi").replace("hifi", "HiFi")
            _rows.append(
                (
                    _fr
                    % (
                        (_nm, "%s TFLOPS" % _r(_pk))
                        + tuple(_cell(_st_c, rf, _f, _pk) for _st_c, rf in _cols)
                        + (
                            (
                                "(no per-stage attribution: one peak shared by every stack)"
                                if _shared_peak and str(_f) == _in_use
                                else ""
                            ),
                        )
                    )
                ).rstrip()
            )
        # THE FRAME FOLLOWS THE TABLE, and is sized once every row exists. All three rules were a
        # flat W while the header grew with the stage count and the column width, so a wide table
        # overflowed its own frame -- visible as soon as the columns widened to fit a marked cell.
        # Sizing off the header alone is not enough either: the shared-peak caveat rides in the
        # trailing column and can be the longest line in the block. W stays the floor so a narrow
        # table still lines up with the sections above it.
        _w = max([W, len(_hdr)] + [len(_r) for _r in _rows])
        _o.append("\u2500" * _w)
        _o.append(_hdr)
        _o.append("".join("\u253c" if c == "\u2502" else "\u2500" for c in _hdr.ljust(_w)))
        _o.extend(_rows)
        _o.append("\u2500" * _w)
        return _o

    # A stage retiring one item per unit reports the model's own unit; one retiring many reports
    # requests. Both follow from the item count the stage list already carries, not from a name.
    # ONE ITEM PER USER is what "recurring" means, and that is `batch`, not the literal 1. The two
    # agree at batch 1, which is why == 1 held while every stage fell back to a single item. Once a
    # stage's real count is observed, a batch-8 decode retires 8 rows per call and would have read as
    # a request-rate stage; it is still one token per user. Encode, retiring 1500 frames for a batch
    # it does not have, correctly stops claiming to emit tokens.
    _su_batch = max(1, _request_batch())
    _STAGE_UNIT = {st: (unit if _retires_one_per_user(rf, _su_batch) else "req/s") for st, rf in _roofs.items()}
    # A DECLARED STAGE GETS A UNIT TOO. Only prefill and decode were named here, so a third stage --
    # voxtral's audio encoder, measured at 12.79 ms -- had no unit, no title, and no row. Its unit is
    # one pass of that tower, which is what "per pass" says without pretending it emits tokens.
    for _extra in _roofs:
        _STAGE_UNIT.setdefault(_extra, "pass/s")
    # The recurring stage is named for the unit the model actually reports, so a diffusion run reads
    # "per step" rather than being told it decodes.
    _STAGE_TITLE = {
        st: (
            "%s \u2014 per %s" % (str(st).upper(), _step)
            if _retires_one_per_user(rf, _su_batch)
            else "%s \u2014 per request" % str(st).upper()
        )
        for st, rf in _roofs.items()
    }
    for _extra in _roofs:
        _STAGE_TITLE.setdefault(_extra, "%s \u2014 per pass" % str(_extra).upper())
    _rendered = []
    # EVERY STAGE THE MODEL DECLARED, in the order _stage_roofs built them -- not a hardcoded pair.
    for _st in _roofs:
        _rf = _roofs.get(_st)
        if not _rf:
            continue
        _u = _STAGE_UNIT.get(_st) or unit
        _path = str((stage_paths or {}).get(_st) or "").strip().lower()
        _traced = _path.startswith("trace") or not _path
        # STATE THE EVIDENCE BESIDE THE LABEL. "eager" is a CONCLUSION trace_replay drew from the op
        # dispatches it counted during the last warmup -- one for a traced stage, hundreds for an
        # eager one. Printing the count makes the label checkable; without it, gemma-3's prefill was
        # diagnosed as eager twice from the label alone, once wrongly.
        _ndisp = (stage_ops or {}).get(_st)
        _lab = "%s, %d op dispatches" % (_path, _ndisp) if isinstance(_ndisp, int) else _path
        # THE MEASURED COLUMN FOLLOWS THE SAME AUTHORITY AS THE STAGE LIST. prefill and decode keep
        # their existing sources; any other declared stage reads its own measurement out of stage_ms,
        # which is where it was recorded. Without this a declared stage rendered as an empty row --
        # present in the table and blank in the only column the reader looks at first.
        # THE RUN'S OWN PER-STAGE TIMING, for every stage alike. This read stage_ms for prefill, the
        # headline for decode and stage_ms again for anything else -- three sources keyed on two
        # names. stage_ms is written by the run that measured each stage, so it is the authority;
        # the headline stays only as the fallback for a stage the run did not time separately.
        _sv = (stage_ms or {}).get(_st)
        _ms = float(_sv) if isinstance(_sv, (int, float)) and _sv > 0 else None
        # THE HEADLINE BELONGS TO THE RECURRING STAGE -- the one retiring a single item per unit,
        # which is what a per-token/per-step figure measures. Handing it to a prompt-consuming
        # stage would print one stage's measurement under another's heading.
        # Read BEFORE the fallback below overwrites it: afterwards every stage looks self-timed.
        _own_ms = _ms is not None
        if _ms is None and per_unit_ms and int((_rf or {}).get("tokens") or 0) == 1:
            _ms = float(per_unit_ms)
        # THE HEADLINE RATE BELONGS TO THE STAGE THE HEADLINE TIMED, and to no other.
        #
        # This asked whether the stage retires ONE item per unit, which is true of a decoded token
        # and equally true of an encoder pass -- so encode, a stage with its own 12.80 ms in
        # stage_ms, printed decode's 345.7 tok/s/u. 1000/2.8926 on the encode row: a number from a
        # different stage, in a unit encode does not use, beside encode's own measured latency.
        #
        # The question is not what the stage retires, it is where its milliseconds came from. A
        # stage the run timed separately has a rate of its own -- 1000/12.80 = 78.1 pass/s -- and
        # only a stage with no timing of its own falls back to the headline.
        _mrate = (1000.0 / _ms) if (_own_ms and _ms) else (measured if measured else ((1000.0 / _ms) if _ms else None))
        out.append(
            _row(
                "%s%s"
                % (_STAGE_TITLE[_st], "" if _traced else "   [%s \u2014 not comparable to a traced band]" % _lab),
                "",
                "",
                "",
            )
        )
        _rendered.append(_st)

        for _roof in ("memory", "compute"):
            _c = _rf.get("%s_ms" % _roof)
            _binding = _rf.get("binds") == _roof
            _mark = "   \u2190 binds" if _binding else ""
            _lbl = "  %s%s" % (_roof, _mark)
            if not _c:
                # SAY WHICH INPUT IS MISSING, not merely that the cell is empty. A compute roof
                # needs a param count and a peak; "not measured" for both named neither, so a
                # report missing its model facts looked identical to one whose model has no
                # compute term at all.
                # THE MEASUREMENT IS NOT MISSING JUST BECAUSE THE ROOF IS. A declared stage with no
                # modelled read set -- a separate tower, whose bytes are not the backbone's -- still
                # has a real time, and blanking that column made the row look like a failed
                # measurement rather than an unpriced one. Show what was measured, and say the
                # ceiling is the part that is absent.
                out.append(
                    _row(
                        _lbl,
                        # A REFUSED roof and an UNCOMPUTABLE one read differently: the first means this
                        # stage's read set could not be attributed, the second that an input was missing.
                        # Told apart by whether a share was found, not by the stage's name.
                        "not modelled" if (_roof == "memory" and not _rf.get("bytes")) else "not measured",
                        "not measured",
                        ("%-8.2f ms" % _ms) if (_ms and _roof == "memory") else "",
                    )
                )
                # WHY, on its own line rather than in the cell -- the columns are 16 wide and a
                # reason does not fit one. "not measured" for both halves named nothing, so a report
                # missing its model facts looked identical to one whose model has no compute term.
                if _roof == "compute" and not _rf.get("flops"):
                    out.append("   (no compute roof: perf_target_inputs.json carries no param count)")
                continue
            # The time band is INVERTED: lower ms is better, so 80% efficiency costs MORE time.
            _lo_ms, _hi_ms = _c / _HIF, _c / _LOF
            # ONLY THE BINDING ROOF EARNS A VERDICT. Held against the roof that is NOT the limit, the
            # measurement scores an automatic miss -- prefill reads 22x off its compute ceiling -- and
            # that miss is meaningless: the stage cannot reach the lower ceiling because the higher one
            # forbids it. A cross there says "close this gap" about a gap physics will not let anyone
            # close, and it duplicates the binding row's measurement to say it. The non-binding roof
            # reports its SLACK instead, which is the actual finding: how much headroom that resource
            # has before it would start to matter.
            _bind_ms = _rf.get("%s_ms" % _rf.get("binds")) if _rf.get("binds") else None
            # THE MS ROW BELONGS TO THE BINDING ROOF, like the rate row and for the same reason.
            #
            # There is one elapsed time per stage. Printed on both roofs it was the same 91.33 ms
            # twice, and on the non-binding roof it also said something false: compute did not take
            # 91.33 ms, the STAGE did, while the arithmetic units sat idle waiting on memory. The
            # verdict beside it was worse -- an automatic cross against a ceiling the stage cannot
            # reach because the higher one forbids it.
            #
            # So the roof that is not the limit reduces to ONE row, in its own currency: 0.69 of
            # 702.0 TFLOPS. Nothing repeats, no cell is blank, and the row still says plainly that
            # the arithmetic is idle.
            if not _binding:
                # NO MS ROW ON THE ROOF THAT IS NOT THE LIMIT -- the same rule as the rate row.
                #
                # Elapsed time is stage-level: compute and memory run at once, so the stage's 91.33 ms
                # cannot be split into "X ms memory, Y ms compute". Printed here it was either the
                # stage wall-clock (false -- compute did not take 91.33 ms, the STAGE did, while the
                # arithmetic idled waiting on memory) or a dash. Neither earns a row.
                #
                # The roof reduces to its own currency, where it does have something to say: 0.69 of
                # 702.0 TFLOPS, which differs by stage even though the peak does not.
                if _roof == "memory" and peak_bw_gbps:
                    _mb0 = _measured_bw_gbps(_rf, _ms)
                    out.append(
                        _row(
                            _lbl,
                            _ncell(_r(peak_bw_gbps), "GB/s"),
                            _bandcell(_r(peak_bw_gbps * _LOF), _r(peak_bw_gbps * _HIF), "GB/s"),
                            _ncell(_r(_mb0), "GB/s") if _mb0 else "n/a \u2014 not measured",
                        )
                    )
                elif _roof == "compute" and _rf.get("peak_flops"):
                    _pk0 = _rf["peak_flops"] / 1e12
                    _mt0 = ((_rf["flops"] / (_ms / 1000.0)) / 1e12) if (_ms and _rf.get("flops")) else None
                    out.append(
                        _row(
                            _lbl,
                            _ncell(_r(_pk0), "TFLOPS"),
                            _bandcell(_r(_pk0 * _LOF), _r(_pk0 * _HIF), "TFLOPS"),
                            _ncell(_r(_mt0), "TFLOPS") if _mt0 else "n/a \u2014 not measured",
                        )
                    )
                continue
            if _ms:
                # AN EAGER STAGE EARNS NO VERDICT. The ACHIEVABLE band is 60-80% of a hardware
                # ceiling, which assumes the measurement is a traced replay; an eager stage also
                # pays per-op host dispatch, and on gemma-3's prefill that is half the wall clock.
                # Scoring it against the band marks the harness's measuring method as a model
                # defect. The number still prints -- it is what the pipeline does today -- it just
                # carries how it was taken instead of a cross.
                # NO VERDICT GLYPH. The three columns already state ceiling, band and measurement
                # side by side; a tick or cross adds no fact, and it kept asserting one on stages the
                # band does not describe -- an eager-measured prefill, a roof that is not the limit.
                # The reader compares the numbers, which is what they are there for.
                _meas = _ncell(_n(_ms), "ms")
            else:
                _meas = "n/a \u2014 not measured"
            out.append(_row(_lbl, _ncell(_n(_c), "ms"), _bandcell(_n(_lo_ms), _n(_hi_ms), "ms"), _meas))
            if _roof == "memory" and peak_bw_gbps:
                _mb = _measured_bw_gbps(_rf, _ms)
                # SAY WHICH BYTES THESE ARE. Both the memory ceiling and this GB/s figure divide by
                # the stage's read set, and the three possible sources rendered identically: a pinned
                # measurement, this build's measurement, and a checkpoint estimate all printed as a
                # bare number. So a read set that was never measured was indistinguishable from one
                # that was -- which is precisely how a dispatch hook that watched an op class the
                # build never instantiates returned 0 for every stage of every run for two days,
                # without the page disagreeing.
                _bsrc = str(_rf.get("bytes_source") or "")
                out.append(
                    _row(
                        "" if _bsrc in ("", "pinned") else ("bytes: %s" % _bsrc),
                        _ncell(_r(peak_bw_gbps), "GB/s"),
                        _bandcell(_r(peak_bw_gbps * _LOF), _r(peak_bw_gbps * _HIF), "GB/s"),
                        _ncell(_r(_mb), "GB/s") if _mb else "",
                    )
                )
            if _roof == "compute" and _rf.get("peak_flops"):
                _pk_t = _rf["peak_flops"] / 1e12
                _mt = ((_rf["flops"] / (_ms / 1000.0)) / 1e12) if (_ms and _rf.get("flops")) else None
                out.append(
                    _row(
                        "",
                        _ncell(_r(_pk_t), "TFLOPS"),
                        _bandcell(_r(_pk_t * _LOF), _r(_pk_t * _HIF), "TFLOPS"),
                        _ncell(_r(_mt), "TFLOPS") if _mt else "",
                    )
                )
            # THE RATE ROW BELONGS TO THE BINDING ROOF, because the rate is that roof's to explain.
            #
            # There is one achieved rate -- 30.8 tok/s/u -- and memory sets it. Printed under compute
            # it was that memory-set number held against a compute ceiling of 18850-25088 tok/s/u: not
            # a comparison but a non-sequitur, since nothing about the arithmetic can be read off a
            # number the arithmetic did not determine. What CAN be read off it is already stated one
            # row up, as 0.69 of 702.0 TFLOPS.
            #
            # It follows `binds`, so a compute-bound stage gets the mirror image: compute carries
            # ms + TFLOPS + the rate, and memory keeps ms + GB/s.
            if _binding:
                _tr = 1000.0 / _c
                out.append(
                    _row(
                        "",
                        _ncell(_r(_tr), _u),
                        _bandcell(_r(_tr * _LOF), _r(_tr * _HIF), _u),
                        _ncell(_r(_mrate), _u) if _mrate else "n/a \u2014 not measured",
                    )
                )
            # THE FIDELITY LADDER, PRICED IN THIS STAGE'S OWN FLOPs. Peak differs 4x across the modes,
            # so a blanket peak is either optimistic (LoFi, unreachable without dropping precision) or
            # pessimistic (HiFi4, punishing LoFi work already done). Every rung prints, present or
            # not, so the reader sees the whole ladder the stage could sit on rather than only where
            # it sits today -- and the rung actually in use is marked.

        if int((_rf or {}).get("tokens") or 0) != 1:
            out.append(_rule4())

    if not _rendered:
        _na = "n/a \u2014 not measured"
        out.append(_row("  memory", _na, _na, _na))
        out.append(_row("  compute", _na, _na, _na))
    # ONE resolution of the prefill figure, measurement first. The roofline cell read the measured
    # stage while this row read only the estimate, so the same report answered the same question two
    # ways: "15.90 ms (trace_replay)" above, "TTFT never measured" below. A guess lit the bar and a
    # measurement did not.
    _prefill_ms = _pf_measured
    # close the roofline block BEFORE the ladder starts its own, or the two rules stack
    out.append(rule)
    # SAY WHICH SPLIT PRICED THESE ROWS. A multi-tower stage's ceiling is the resident bytes times
    # its subtree's share, and that share has two sources: the census, which WALKED the built model,
    # and the checkpoint, which states the precision the file was written at. The loader picks a
    # dtype per tensor, so a tower quantised more heavily than its neighbour is a smaller slice of
    # the chip than of the file -- voxtral's backbone is 85.8% of its checkpoint and its decode
    # measurement implies 77.9% of what is resident. Priced on the checkpoint, that row read 109% of
    # peak: a number indistinguishable from a fast model, and actually a guessed divisor. Unmarked,
    # a reader had no way to tell the two apart, which is the whole reason the row was doubted.
    _guessed = sorted(st for st, rf in (_roofs or {}).items() if (rf or {}).get("share_basis") == "checkpoint")
    if _guessed:
        out.append("")
        out.append(
            "  note: %s priced from the CHECKPOINT's tower split (disk precision), not a device "
            "census -- a stage can read >100%% of peak when the served split differs. Re-run once "
            "the census records device_section_bytes." % ", ".join(_guessed)
        )
    out.extend(_fidelity_section())
    disp = _dispatch_ms_per_unit(profile, per_unit_ms)
    cap = _capacity_bytes()
    out.append("")

    if disp is not None or cap:
        out.append("Overheads & limits")
        out.append(rule)
        out.append(_row2("", "TARGET", "MEASURED"))
        out.append(_rule4())
        # No "ok" marker on a healthy row. It read as a verdict against the TARGET beside it, but was
        # judged on a DIFFERENT, invisible rule -- a hardcoded 10% tolerance -- so a row could print
        # a target of ~0 ms, measure 2.46 ms, and call itself ok. The tolerance is now stated in the
        # TARGET column, and only the breach is marked: on a threshold row, passing is the silent
        # case and the exception is the whole signal. `\u2717` matches the roofline block's out-of-band
        # glyph, so the report has one vocabulary rather than two.
        if disp is not None and per_unit_ms:
            _share = 100.0 * disp / float(per_unit_ms)
            _ops = _ops_per_unit(profile)
            out.append(
                (
                    " %-30s\u2502 %-16s\u2502 %-28s\u2502 %.0f%% of %-8s %s"
                    % (
                        "Dispatch",
                        _ncell("~0", "ms"),
                        _ncell("%.2f" % disp, "ms/%s" % _step),
                        _share,
                        _step,
                        "",
                    )
                ).rstrip()
            )
            # NO THRESHOLD LINE AND NO OP COUNT.
            #
            # The threshold was the rule behind a "OVER" verdict; with the verdict gone it grades
            # nothing. And the op count was a MEASUREMENT sitting in the TARGET column -- nobody
            # targets 10784 ops -- counted over the whole profiling window (one prefill plus six
            # decode steps) on a row that reads per token, so it was out by roughly 7x as well.
            # Op counts live in the Op breakdown table below, per class, correctly attributed.
        if cap and active_bytes:
            _used = 100.0 * active_bytes / cap
            out.append(
                (
                    " %-30s\u2502 %-16s\u2502 %-28s\u2502 %.0f%% used%s"
                    % (
                        "DRAM capacity",
                        _ncell("%.1f" % (cap * 0.9 / 1024**3), "GiB 90%"),
                        _ncell("%.2f" % (active_bytes / 1024**3), "GiB"),
                        _used,
                        "      \u2717 OVER" if _used >= 90 else "",
                    )
                ).rstrip()
            )
            out.append(rule)
        else:
            out.append(rule)
        out.append("")

    # One list, direction marked on the ROW it qualifies. Two direction-headed groups put the label
    # furthest from the bars it describes, and forced an ordering on rows that otherwise read
    # roofline-then-overhead top to bottom. A row with nothing measured gets no arrow -- there is no
    # direction to want when there is no number.
    out.append("Utilization")
    out.append(rule)
    # An over-ceiling ratio is withheld, not drawn. _bar clamps at full, so 129% rendered as a
    # SATURATED bar -- the inconsistent pair reading as a flawless score, which is the opposite of
    # what it means.
    _u1 = None if _exceeds else ((measured / theo) if (measured and theo) else None)
    _rows = []
    # ONE BAR PER STAGE AND ROOF, each a fraction of a PEAK so the bars are directly comparable. The
    # list used to hold memory for decode and compute for prefill and nothing else -- the same
    # one-sided view the roofline had, which left "is decode compute-bound?" unanswerable from the
    # report. An empty compute bar beside a two-thirds-full memory bar is the whole optimisation
    # story at a glance: there is no compute headroom to win because compute was never the constraint.
    for _st in _roofs:
        _rf = _roofs.get(_st)
        if not _rf:
            continue
        # Same authority as the table above: the run's own per-stage timing, with the pipeline
        # headline only as the fallback for a single-stage model.
        _sv = (stage_ms or {}).get(_st)
        _ms = float(_sv) if isinstance(_sv, (int, float)) and _sv > 0 else None
        if _ms is None and per_unit_ms and int((_rf or {}).get("tokens") or 0) == 1:
            _ms = float(per_unit_ms)
        if not _ms:
            continue
        # ONE BAR PER STAGE: the roof that BINDS it. Same rule the ms and rate rows follow above --
        # a resource that is not the limit has no utilisation worth acting on, and drawing it invited
        # the reading that an empty compute bar is something to go and fill.
        _bind = _rf.get("binds")
        if _bind == "memory" and peak_bw_gbps and _rf.get("bytes"):
            # Same figure as the roofline row above, not a second computation of it: recomputing from
            # bytes differed in the last digit and put 345.0 and 344.4 in one report.
            _g = (
                bw_gbps
                if (int((_rf or {}).get("tokens") or 0) == 1 and bw_gbps)
                else (_rf["bytes"] / (_ms / 1000.0)) / 1e9
            )
            _rows.append(
                (
                    "%-9s memory" % _st,
                    (None if (_exceeds and int((_rf or {}).get("tokens") or 0) == 1) else _g / peak_bw_gbps),
                    (
                        "inconsistent \u2014 see above"
                        if (_exceeds and int((_rf or {}).get("tokens") or 0) == 1)
                        else "%.1f / %.1f GB/s" % (_g, peak_bw_gbps)
                    ),
                    "\u2191 better",
                )
            )
        if _bind == "compute" and _rf.get("flops") and _rf.get("peak_flops"):
            _t = (_rf["flops"] / (_ms / 1000.0)) / 1e12
            _pk_t = _rf["peak_flops"] / 1e12
            _rows.append(("%-9s compute" % _st, _t / _pk_t, "%.1f / %.1f TFLOPS" % (_t, _pk_t), "\u2191 better"))
    # DISPATCH AND CAPACITY BELONG HERE TOO. Their percentages also appear in the Overheads block,
    # but this panel is the one place every resource is drawn on the same 0-100% scale -- which is
    # what makes 67% bandwidth, 17% dispatch and 33% capacity comparable at a glance. Overheads
    # answers "is this row in breach"; this one answers "which resource is full".
    if disp is not None and per_unit_ms:
        _d = disp / float(per_unit_ms)
        _rows.append(("dispatch  overhead", _d, "%.2f / %.2f ms" % (disp, per_unit_ms), "\u2193 better"))
    if cap and active_bytes:
        _c = active_bytes / cap
        _rows.append(
            ("DRAM      capacity", _c, "%.2f / %.0f GiB" % (active_bytes / 1024**3, cap / 1024**3), "\u2193 better")
        )
    for _name, _frac, _detail, _dir in _rows:
        # An estimated row draws a HATCHED bar, never the solid fill a measurement gets. A bar is
        # read as data at a glance, so the distinction has to survive a glance.
        _b = _bar(_frac, 30)
        # Two decimals below 1%: a compute bar reading 0% and a bar reading "no data" are opposite
        # findings, and "%.0f%%" printed both as 0.
        _pc = (
            (("%.1f%%" % (_frac * 100)) if (_frac and _frac < 0.01) else ("%.0f%%" % (_frac * 100)))
            if _frac
            else "\u2014"
        )
        out.append(("  %-20s %s  %6s   %-24s %s" % (_name, _b, _pc, _detail, _dir)).rstrip())
    out.append(rule)
    return out


def _roofline_lines(
    throughput: dict | None,
    forward_ms: float | None,
    profile: dict | None = None,
    model: str = "",
    task: str = "",
    per_token_ms: float | None = None,
    measured_depth: str = "",
) -> list:
    """The adaptive 'Roofline & utilization' table. MEASURED values (tok/s, mem BW, utilization,
    at-floor) are computed HERE from the ms actually being reported (`forward_ms`) against the STATIC
    target snapshot in `throughput` — so a stale measured can never leak in, and any missing/zero
    input renders 'n/a' rather than a fake 0.0 (the fix for the old '+0.0%' readout). LLM-decode
    pipelines get the tok/s/u form; everything else gets the roofline-floor (ms) form."""
    if not isinstance(throughput, dict):
        return []
    fm = forward_ms if (isinstance(forward_ms, (int, float)) and forward_ms > 0) else None
    # BACKWARD-COMPATIBLE READ. The snapshot on disk may predate the rename (it lives in /tmp and is
    # rewritten every profile, but a run in flight can be holding the old spelling).
    theo = throughput.get("theoretical_rate")
    if theo is None:
        theo = throughput.get("theoretical_tok_s")
    out = ["Roofline & utilization"]
    _has_ceiling = throughput.get("has_unit_ceiling")
    if _has_ceiling is None:
        _has_ceiling = throughput.get("is_llm_decode")
    if _has_ceiling and isinstance(theo, (int, float)) and theo > 0:
        band = throughput.get("band") or [None, None]
        # THE LEDGER WINS. The throughput snapshot is rewritten from perf_target_inputs.json, which
        # lives in the model directory the optimize loop restores -- so a 16-layer 3.33 GB vintage kept
        # coming back and printing a 153.8 tok/s/u ceiling next to a full-model measurement. The
        # ledger's anchor is keyed, append-only and outside that directory, so it is read first and the
        # ceiling and band are recomputed from it rather than trusting the snapshot's arithmetic.
        active_bytes = throughput.get("active_bytes") or 0
        try:
            _led = _ledger()
            # DEPTH IS THE UNIT, NOT THE STRING "token". The producer anchors the bytes under the
            # model's own unit of work (run.py: depth=facts["unit"]), so hardcoding "token" here read
            # the wrong key for every step-unit (diffusion) and inference-unit (classifier) model:
            # the lookup missed, the code fell back to the snapshot, and the whole point of the anchor
            # -- surviving the model directory being reverted mid-run -- was lost for everything but an
            # LLM. It worked only because the one model that has ever had an anchor is per-token.
            _anchor_depth = str(throughput.get("unit") or "token").strip().lower() or "token"
            _mb = _led.anchor_value(_led.KIND_ACTIVE_BYTES, depth=_anchor_depth, model=model, task=task)
            if _mb and float(_mb) > 0:
                active_bytes = int(round(float(_mb) * 1e6))
                _pk = float((throughput or {}).get("peak_bw_gbps") or 0.0) * 1e9
                if _pk > 0:
                    # perf_target OWNS this arithmetic. This used to be its own `_pk / bytes` with a
                    # hardcoded (0.60, 0.80) band, which kept the pre-sustained-fraction physics after
                    # the ceiling moved: an anchored run printed 84.0 while the stop gate reading the
                    # same snapshot judged against 51.2.
                    from agent.perf_target import rate_and_band as _rab

                    theo, _b = _rab(
                        active_bytes,
                        _pk,
                        frac=float((throughput or {}).get("bw_fraction") or 0.0) or 1.0,
                        tp_degree=int(throughput.get("tp_degree") or 1),
                    )
                    band = [_b[0], _b[1]]
        except Exception:  # noqa: BLE001
            pass
        tp = max(1, int(throughput.get("tp_degree") or 1))
        per_dev_bytes = (active_bytes / tp) if active_bytes else 0
        # THE UNIT MUST MATCH THE CEILING. This ceiling is per TOKEN (peak_BW / bytes-per-token), and
        # `fm` here is the headline number -- a per-profile device_ms sum over the profiling window.
        # Dividing 1000 by that reported 1.9 tok/s/u against a 64 tok/s/u ceiling: a 3% utilisation
        # readout for a model running at 84%. The per-token reading is the profile's own, and when
        # there is none the line says so rather than printing a number of the wrong kind.
        _pt_ms = per_token_ms
        if not (isinstance(_pt_ms, (int, float)) and _pt_ms > 0):
            try:
                _pt_ms = _ledger().trace_ms_from_profile(profile)
            except Exception:  # noqa: BLE001
                _pt_ms = None
        # THE MEASUREMENT AND THE CEILING MUST DESCRIBE THE SAME MODEL. The ceiling is peak_BW over
        # the bytes of the WHOLE model; a per-token reading taken on a truncated profiling window
        # streams a fraction of those bytes, so pairing them reports roughly depth_total/depth_window
        # times the real throughput -- 107.1 tok/s/u from a 16-layer window against a 32-layer
        # ceiling, when the model does 43.9. Depths disagreeing is not a detail to annotate, it makes
        # the ratio meaningless, so the value is withheld and the reason given.
        _ceil_depth = str((throughput or {}).get("perf_layers") or "").strip().lower()
        _meas_depth = str(measured_depth or "").strip().lower()
        _mismatch = bool(_ceil_depth and _meas_depth and _ceil_depth != _meas_depth)
        if isinstance(_pt_ms, (int, float)) and _pt_ms > 0 and not _mismatch:
            fm = float(_pt_ms)
        measured = (1000.0 / fm) if fm else None
        util = (measured / theo) if measured else None
        bw_gbps = ((per_dev_bytes / (fm / 1000.0)) / 1e9) if (per_dev_bytes and fm) else None
        # SAY WHAT DEPTH THIS IS. tok/s/u is an absolute, user-facing throughput, and a truncated
        # profiling window makes it wrong as a statement about the model: a 16-layer window on a
        # 32-layer model reads ~2x the real throughput. The ratios below (GB/s, utilisation) are
        # depth-invariant and stay valid; the rates are not, so they carry the depth.
        # THE UNIT IS PART OF THE NUMBER. The same formula gives tok/s/u for a decoded token, steps/s
        # for a denoise step and inferences/s for one forward pass -- printing "tok/s/u" for a
        # diffusion model would be a category error, not a label slip.
        _unit = str((throughput or {}).get("unit") or "token").strip().lower()
        try:
            from agent.model_bytes import unit_label as _ul

            _u = _ul(_unit) or "tok/s/u"
        except Exception:  # noqa: BLE001
            _u = {"step": "steps/s", "inference": "inferences/s"}.get(_unit, "tok/s/u")
        _depth = str((throughput or {}).get("perf_layers") or "").strip()
        _partial = _depth and _depth.lower() not in ("all", "0", "none")
        _tag = "   [%s-layer window, NOT the full model]" % _depth if _partial else ""
        if str(os.environ.get("PERF_MCP_ROOFLINE_TABLE", "1")).lower() not in ("0", "false", "no"):
            # NEW LAYOUT: three blocks. The old five lines conflated a roofline (spec ceiling +
            # sustained band) with overheads that have neither, so "achievable" spanned rows where it
            # is meaningless. See _roofline_tables.
            try:
                return _roofline_tables(
                    unit=_u,
                    theo=theo,
                    band=band,
                    measured=measured,
                    bw_gbps=bw_gbps,
                    peak_bw_gbps=float((throughput or {}).get("peak_bw_gbps") or 0.0) or None,
                    active_bytes=active_bytes,
                    per_unit_ms=fm,
                    profile=profile,
                    tag=_tag,
                    # The MEASURED phase split, so the compute row can state a real prefill time
                    # instead of a hardcoded "not measured" while the block below prints one.
                    stage_ms=_measured_stage_ms(model, task),
                    # HOW each stage was measured. A stage that fell back to eager cannot be graded
                    # against a band that assumes trace.
                    stage_paths=_measured_stage_paths(model, task),
                    # The dispatch count each path label was derived from, so the label is evidence
                    # rather than an assertion.
                    stage_ops=_measured_stage_ops(model, task),
                    # The pinned peak is keyed per (model, task) like every other anchor, so the
                    # ceiling code needs both to look it up.
                    model=model,
                    task=task,
                    # The stage roofs shard the bytes and the FLOPs the same way the ceiling does.
                    tp_degree=tp,
                    # WHY the measurement is absent, not merely that it is. A depth mismatch means the
                    # value was computable and REFUSED: a truncated window streams a fraction of the
                    # bytes the ceiling assumes, so the ratio is meaningless rather than optimistic.
                    # Rendered as "n/a" alone it reads as a harness gap, and the next run repeats it.
                    note=(
                        "the per-token reading is from a %s-layer window, the ceiling is for %s layers "
                        "(re-profile at full depth)" % (_meas_depth, _ceil_depth)
                        if _mismatch
                        else ""
                    ),
                )
            except Exception as _rt_exc:  # noqa: BLE001
                # SAY THAT IT FELL BACK, AND WHY. This was `pass`, so a table that raised was
                # indistinguishable from a tool that never had the section: the report quietly
                # printed the legacy five lines with correct-looking numbers, and the per-stage
                # ceilings -- the whole reason the stage marks exist -- were simply absent with
                # nothing to act on. Measured: run 37 marked all three stages, sliced their buckets
                # (137.6 / 190.1 / 37.8 GFLOP, correctly different) and still reported one aggregate
                # ceiling, because this line discarded the reason. Two hundred lines above, the same
                # module already names its missing input rather than rendering nothing; this now
                # does the same. The fallback still happens -- losing the roofline entirely would be
                # worse -- but it no longer happens in silence.
                import traceback as _tb

                print(
                    "  [summary] per-stage roofline table unavailable (%s: %s); reporting the "
                    "aggregate ceiling instead" % (type(_rt_exc).__name__, str(_rt_exc)[:160]),
                    file=sys.stderr,
                    flush=True,
                )
                _tb.print_exc(file=sys.stderr)
                out.append("  per-stage ceilings unavailable: %s: %s" % (type(_rt_exc).__name__, str(_rt_exc)[:120]))
        out.append(f"  theoretical ceiling : {theo:.1f} {_u}{_tag}")
        if band[0] is not None:
            # The percentages are DERIVED from the band, not hardcoded: dense sustains 60-80% of peak
            # and MoE 37.5-50%, so a fixed "(60-80%)" string would describe the wrong physics for an
            # MoE while printing its correct numbers.
            _lo_pct = (100.0 * band[0] / theo) if theo else 0.0
            _hi_pct = (100.0 * band[1] / theo) if theo else 0.0
            out.append(f"  achievable ({_lo_pct:.0f}-{_hi_pct:.0f}%) : {band[0]:.1f} - {band[1]:.1f} {_u}")
        out.append(
            f"  measured            : {measured:.1f} {_u}   (1000 / {fm:.2f} ms){_tag}"
            if measured
            else (
                "  measured            : n/a — the per-token reading is from a %s-layer window, the "
                "ceiling is for %s layers (re-profile at full depth)" % (_meas_depth, _ceil_depth)
                if _mismatch
                else "  measured            : n/a (no valid forward ms)"
            )
        )
        if bw_gbps is not None:
            out.append(f"  measured mem BW     : {bw_gbps:.0f} GB/s   ({per_dev_bytes / 1e9:.2f} GB / {fm:.2f} ms)")
        if util is None:
            out.append("  utilization         : n/a")
        elif util > 1.0:
            # measured beat the theoretical ceiling -> active_bytes / the target is stale/suspect
            # (the target and the measured ms are from different states); do NOT print a >100% util.
            out.append(f"  utilization         : measured EXCEEDS ceiling — target stale/suspect (re-profile)")
        else:
            # "BASELINE ceiling" was needed when the divisor was the streamed BYTES: optimization
            # shrinks bytes (bf8_b -> bf4_b), so the build's own bound moved during the run -- 84.0 at
            # the baseline vs 121.3 once every weight group reached bf4 -- and the figure had to say
            # which one it was against. The divisor is now the PARAM count, which no lever changes, so
            # the baseline ceiling and the current ceiling are the same number and the qualifier only
            # added noise.
            out.append(f"  utilization         : {util * 100:.0f}%   (measured / ceiling)")
    else:
        _current = throughput.get("modeled_floor_ms")
        floor = _floor_anchor(_current, throughput.get("perf_layers"), model, task) or _current
        have_floor = isinstance(floor, (int, float)) and floor > 0
        out.append(
            f"  modeled floor       : {floor:.2f} ms   ({_floor_basis(profile)})"
            if have_floor
            else "  modeled floor       : n/a"
        )
        # NO ACHIEVABLE BAND IN THE FLOOR FORM, and no line pretending otherwise. This printed a
        # hardcoded "achievable (60-80%) : … ms" from _achievable_band_ms(floor) -- which asks
        # target_from_floor_ms for a band and is answered (0.0, 0.0) on purpose, because 60-80% of
        # 1000/floor is not a bandwidth statement and has no hardware peak behind it. So the line could
        # never render, and it survived only as a stale label describing physics the tool rejected. The
        # rate ceiling is where a band belongs, and every model reaches that form now.
        out.append(
            f"  measured            : {fm:.2f} ms" if fm else "  measured            : n/a (no valid forward ms)"
        )
        if have_floor and fm:
            status = _floor_status(floor, fm)
            if status == "ABOVE_BAND":
                # Measured beat the floor. Two very different things produce that, and calling both
                # "impossible" puts a false claim in a confirmation document:
                #
                #   the work SHRANK -- the reported floor is pinned to the BASELINE, and optimization
                #       removes bytes (bf8_b -> bf4_b halves a weight read), so the current build's own
                #       bound is lower and beating the baseline's bound is exactly the goal. On
                #       llama3_1_8b_p150 the run reached 534.44 ms against a 537.23 ms baseline floor
                #       while its own floor stood at 331.86 ms: a real win the report called impossible
                #       and told the reader never to bank.
                #
                #   the numbers DISAGREE -- measured beats even the current build's floor, which no
                #       kernel can do, so one side is stale (a 2-layer measurement against a 16-layer
                #       floor produced this).
                #
                # The current floor is the only thing that separates them. It is read here and NOT
                # printed: the report states the baseline, one floor line.
                _beats_own = isinstance(_current, (int, float)) and _current > 0 and fm < _current
                if _beats_own:
                    out.append(
                        f"  status              : ABOVE_BAND — stale/suspect: beats this build's own floor "
                        f"{_current:.2f} ms (re-profile; never bank this as a win)"
                    )
                else:
                    out.append("  status              : PAST BASELINE FLOOR — keep optimizing")
            else:
                _hint = {
                    "IN_BAND": "reached the achievable band — done",
                    "NO_BAND": "no bandwidth band for this pipeline — keep optimizing",
                }.get(status, "keep optimizing")
                out.append(
                    f"  at-floor            : {floor / fm * 100:.0f}%   ({fm - floor:.2f} ms reachable headroom)"
                )
                out.append(f"  status              : {status} — {_hint}")
        # Do not assert WHY unless it is known. This line claimed "not an LLM decode pipeline" for
        # Llama-3.1-8B, which is false -- the pipeline runs a real traced KV-cache decode. The ms
        # branch is taken whenever the tok/s target is unavailable, and for this model that is because
        # active_bytes is 0: perf_target_inputs.json was never produced, so the weight-bytes-per-token
        # physics has no numerator. A report that goes out for confirmation must not explain a missing
        # input by inventing a property of the model.
        _ab = throughput.get("active_bytes") if isinstance(throughput, dict) else None
        if not _ab:
            # NAME THE MISSING STEP, NOT JUST THE MISSING VALUE. Without the weight bytes this table
            # silently becomes its floor-only form -- no band, no per-stage rows, no fidelity ladder --
            # which reads as a different KIND of report rather than a report missing an input. The
            # snapshot that carries those bytes is written by profile_model; when that step could not
            # run it now says so on stderr, and this line points at it so the two can be connected.
            out.append(
                "  (rate ceiling — n/a: no weight-bytes input for this pipeline — the roofline inputs "
                "were never persisted for this run, so there is no bandwidth band, no per-stage row and "
                "no fidelity ladder. See '[perf-mcp] roofline inputs NOT persisted' in the run log.)"
            )
        else:
            out.append("  (rate ceiling — n/a: no single unit of work for this pipeline)")
    out.append("")
    return out


def _baseline_bucket_lines(baseline_profile: dict | None, report_csv: str = "") -> list:
    """Render the baseline op-class breakdown (device time per op class, ranked) so an operator can
    read WHAT to target directly from RUN_REPORT.md instead of the terminal/CSV. Sourced from the
    baseline profile's `buckets`; falls back to the baseline_profile.json beside report_csv. Empty
    list when no bucket data is available, so the section silently skips rather than blocking."""
    prof = baseline_profile
    if not (isinstance(prof, dict) and prof.get("buckets")) and report_csv:
        prof = _read_json(Path(report_csv).parent / "baseline_profile.json") or {}
    buckets = prof.get("buckets") if isinstance(prof, dict) else None
    if not buckets:
        return []
    # STATE WHICH MEASUREMENT THIS IS, do not name it. `baseline_profile` names TWO DIFFERENT FILES
    # depending on who called render_summary, and only one of them is a baseline:
    #
    #   run.py:_read_baseline_profile_for_report -> runs/*/profiles/baseline_profile.json
    #       written ONCE by before_loop, predates every lever. Genuinely the baseline.
    #   perf_mcp:_read_baseline_profile -> .state/perf_mcp_baseline_<model>_<task>.json
    #       rewritten by _promote_baseline on EVERY profile_model call: `out = dict(prof)` replaces
    #       the whole picture and only device_ms is ratcheted ("picture refreshed, BAR unchanged").
    #       This is the LIVE report's source, so here the word BASELINE is a claim about provenance
    #       the number does not have.
    #
    # On llama3_1_8b_p150 these rows summed to 615.69 ms of device time -- the PREVIOUS reading --
    # while the roofline above reported 534.44 ms and the true baseline was 2464.18 ms. Three points
    # in the run, one of them labelled the other. Printing the total the rows sum to lets the reader
    # tie the table to a measurement instead of trusting a label.
    #
    # Spelled out because the single-sentence version of this ("perf_mcp REWRITES that file on every
    # profile") is true of one caller and false of the other, and a later audit read it, believed the
    # ceiling floated on both paths, and had to be walked back.
    _tot = sum((b.get("device_ms") or 0.0) for b in buckets if isinstance(b, dict))
    _depth = _depth_label(prof) if isinstance(prof, dict) else ""
    _hdr_note = "totalling %.2f ms" % _tot if _tot > 0 else "total unknown"
    if _depth:
        _hdr_note += " over %s" % _depth
    # Heading only. The subtitle restated the column names ("device time by op class"), the ranking
    # is visible from the order, and the profile total is the first row's denominator -- so it was
    # three facts the table already carries, in a line as wide as the table itself.
    out = ["Op breakdown"]
    # A TABLE BUILT FROM A TRUNCATED CAPTURE MUST SAY SO. Tracy stops instrumenting after 32K source
    # locations and saves what it has, losing roughly a third of the rows on a full-model forward --
    # and the run still exits 0 with a CSV, so nothing downstream could tell. These counts and totals
    # were rendered as if they described the whole run. The profile now carries the reason; printing
    # it here means the reader learns it beside the numbers it qualifies, not in a log.
    _trunc = (prof or {}).get("capture_truncated") if isinstance(prof, dict) else None
    if _trunc:
        out.append(
            "  INCOMPLETE: the profiler stopped recording partway through this capture (%s), so these "
            "counts and totals describe only the part it kept." % _trunc
        )
    hdr = f"{'op class':<15} {'device_ms':>10} {'%':>6} {'count':>7} {'bound':>6}  dominant op (shape)"
    out.append(hdr)
    # Ruled to the same width as every other section in this report, rather than a length derived
    # from the header string -- which left this one table 99 wide against their 100.
    out.append("\u2500" * 100)
    for b in sorted(buckets, key=lambda x: -(x.get("device_ms") or 0.0)):
        if not isinstance(b, dict):
            continue
        top = (b.get("top_ops") or [{}])[0] if b.get("top_ops") else {}
        dom = str(top.get("op_code", "") or top.get("shape", "")).strip()
        ms = b.get("device_ms") or 0.0
        # Recomputed against the total of the rows SHOWN, so the column sums to 100%. The stored pct
        # used a denominator that excluded host_overhead, so host_overhead's own share was overstated
        # (3.5% of 615.69 rather than 3.4% of 637.24) and the column did not add up.
        pct = (ms / _tot * 100.0) if _tot > 0 else (b.get("pct") or 0.0)
        cnt = b.get("count") or 0
        bound = (b.get("tags") or {}).get("bound") or "—"
        # rstrip: a bucket with no top_ops (host_overhead has none -- it is the op-gap bucket, not an
        # op) left an empty dominant-op cell and a trailing space on the row.
        out.append((f"{str(b.get('id', '?')):<15} {ms:>10.2f} {pct:>5.1f}% {cnt:>7} {bound:>6}  {dom[:52]}").rstrip())
    out.append("\u2500" * 100)
    out.append("")
    return out


def render_summary(
    kernel_log_path: str | Path,
    baseline_ms: float | None = None,
    *,
    model: str = "",
    task: str = "main",
    metric: str = "device_ms",
    committed_wins: int | None = None,
    opt_branch: str = "",
    perf_test: str = "",
    report_csv: str = "",
    residual: dict | None = None,
    baseline_profile: dict | None = None,
    finalized: bool = True,
    final_override_ms: float | None = None,
    throughput: dict | None = None,
    model_root: str | Path = "",
) -> str:
    """Return a markdown summary. Degrades gracefully when data is partial."""
    # THE MODEL DIRECTORY IS PASSED, NOT DISCOVERED. perf_mcp resolves it from PERF_MCP_MODEL_ROOT or
    # a manifest, and under the by-path load this module gets it falls back to "." -- so
    # _load_perf_target_inputs read a directory with no perf_target_inputs.json, returned None, and
    # both compute roofs plus the whole fidelity ladder rendered "not measured" while the file sat in
    # the model dir the caller already knew. Everything the caller can hand over should be handed
    # over; this module has been wrong every time it went looking for something itself.
    global _MODEL_ROOT_HINT
    if model_root:
        _MODEL_ROOT_HINT = Path(model_root)
    attempts = _read_json(kernel_log_path) or []
    if not isinstance(attempts, list):
        attempts = []

    # THE baseline and THE win-set, computed once, before anything renders. A win must have reduced
    # the model's measured time, which is a property of the SEQUENCE, not of one row -- so it is
    # decided here and every section below reads this set instead of re-judging rows.
    _base_row = _ledger_pair(_ledger().KIND_EAGER, model, task)[0]
    hdr_base = float(_base_row["value_ms"]) if _base_row else None
    # MODEL-LEVEL LEVERS ARE NOT A ROW IN A PER-OP LADDER. Every row below is one op walking its own
    # rungs; a lever that rebuilds how the whole model streams has no owning op. The DRAM prefetcher is
    # built once, every layer's weights are registered with it and global_cb goes to all the decode
    # matmuls together -- run 20 recorded that attempt as "Matmul 32x3840x15360 / shard", true of where
    # the agent was standing and useless to anyone asking whether the prefetcher had been tried.
    # Attaching it to all six matmuls instead would count one change six times. They get their own
    # block, keyed by a `model:` prefix, so the matrix stays one-ladder-per-op.
    _model_levers = [
        a for a in attempts if isinstance(a, dict) and str(a.get("op_signature", "")).startswith(_MODEL_LEVER_PREFIX)
    ]
    attempts = [
        a
        for a in attempts
        if not (isinstance(a, dict) and str(a.get("op_signature", "")).startswith(_MODEL_LEVER_PREFIX))
    ]

    _wins = _win_set(attempts, hdr_base)

    by_op: dict[str, dict] = {}
    for _i, a in enumerate(attempts):
        if not isinstance(a, dict):
            continue
        sig = a.get("op_signature", "?")
        lvl = classify_level(a.get("kernel_kind", ""), a.get("note", ""), sig)
        ms = a.get("measured_ms")
        won = _i in _wins
        op = by_op.setdefault(sig, {c: None for c in _ALL_COLS})
        cur = op.get(lvl)
        # 'win' beats 'try'; track best (lowest) measured ms per cell
        status = "win" if won else ("wedge" if a.get("wedged") else "try")
        if cur is None:
            op[lvl] = (status, ms)
        elif status == "win" and cur[0] != "win":
            op[lvl] = (status, ms)
        elif cur[0] == "wedge" and status == "try":
            op[lvl] = (status, ms)
        elif cur and ms is not None and cur[1] is not None and ms < cur[1] and cur[0] != "win":
            op[lvl] = (cur[0], ms)

    win_ms = [attempts[i].get("measured_ms") for i in _wins]
    final_ms = final_override_ms if final_override_ms is not None else (min(win_ms) if win_ms else baseline_ms)
    # ANCHOR for the headline, most trustworthy first. Falling straight back to `baseline_ms` is
    # wrong: run.py passes the CURRENT committed ms in that slot, so refusing a stale original made a
    # 3.45x run print "714.94 -> 714.94 (+0.0%)". This run's own baseline_profile.json is written once
    # at the start and is the only value guaranteed to predate every lever.
    # The baseline is a measured fact, never a derived one. This used to substitute the SLOWEST
    # measurement ever recorded whenever the real baseline was not better than the final -- i.e.
    # precisely when the run achieved NOTHING -- so 100 -> 105 with a 180 ms failed experiment
    # printed `baseline 180.00 -> final 105.00 (+41.7%)`, and every `gain vs base` cell was then
    # computed against that invented number. A run that did not improve must say so.

    lines = []
    title = f"Optimization summary — {model or 'model'} · {task} ({metric})"
    lines.append(title)
    lines.append("=" * len(title))
    if not finalized:
        lines.append(
            "optimizing… — baseline->final speedup is finalized when the module converges (per-attempt detail below is live)"
        )
    else:
        _eager = _ledger_line(_ledger().KIND_EAGER, "eager per-op device time", model, task)
        if _eager:
            lines.append(_eager)
        elif not _ledger().rows(_ledger().KIND_EAGER, model=model, task=task):
            # Only claim "not measured" when the ledger really holds nothing. When it holds a pair the
            # renderer declined to subtract, this sentence is false.
            lines.append("eager per-op device time: not measured (no ledger reading for this run)")
    _tp = _ledger_line(_ledger().KIND_TRACE_PASS, "tracy trace pass", model, task)
    if _tp:
        lines.append(_tp)
    _trace_scope = f"module ({task})" if os.environ.get("TT_PERF_MODULE_LEVEL") == "1" else "full-pipeline e2e"
    _fp = _ledger_line(_ledger().KIND_FULLPIPE, "trace+1CQ %s" % _trace_scope, model, task)
    if _fp:
        lines.append(_fp)
    lines.append("")

    if not isinstance(throughput, dict):
        throughput = _throughput_from_profile(baseline_profile)
    # The decode ceiling is per TOKEN, so hand the renderer the per-token reading EXPLICITLY rather
    # than letting it divide the headline per-profile device_ms: 1000/534 ms reads 1.9 tok/s/u against
    # a 64 tok/s/u ceiling, i.e. 3% utilisation for a model running at 84%.
    # THE WIN GATE'S OWN NUMBER IS THE MEASURED ONE. check_full_pipeline_latency confirms every win by
    # comparing trace_replay end-to-end against trace_replay end-to-end, ratcheting best-so-far into
    # this file -- "validate/bank EVERY win here". So the report must divide the ceiling by THAT, not by
    # a per-profile per_token_ms (a 16-layer window) or the e2e bookend taken at the run's start. Using
    # the stale bookend reported 43.9 tok/s/u for a model the gate had already measured at 58.6.
    # Layer cap is OFF for this measurement, so it is full-model and comparable to a full-model ceiling.
    _tok_ms, _tok_depth = None, ""
    try:
        import json as _j

        _gp = state_dir() / ("perf_mcp_full_pipeline_baseline_1cq_%s_%s.json" % (model or "model", task or "main"))
        _g = _j.loads(_gp.read_text())
        if str(_g.get("method")) == "trace" and float(_g.get("full_pipeline_ms") or 0) > 0:
            _tok_ms, _tok_depth = float(_g["full_pipeline_ms"]), "all"
    except Exception:  # noqa: BLE001
        _tok_ms, _tok_depth = None, ""
    try:
        if _tok_ms is None:
            # Only the reading's OWN producer may label its depth. This stamped the profile's depth
            # onto whatever _tok_ms held, so the gate's full-model 17.05 ms was relabelled as a
            # 16-layer window and the scope guard then refused it -- reporting n/a for a number that
            # was correct and comparable.
            _tok_ms = _ledger().trace_ms_from_profile(baseline_profile)
            if _tok_ms is not None and isinstance(baseline_profile, dict):
                _tok_depth = str(baseline_profile.get("perf_layers") or "")
        if _tok_ms is None:
            _row = _ledger_pair(_ledger().KIND_TRACE_PASS, model, task)[1]
            if _row:
                _tok_ms = float(_row["value_ms"])
                _tok_depth = str(_row.get("depth") or "")
    except Exception:  # noqa: BLE001
        _tok_ms, _tok_depth = None, ""
    # For a per-token ceiling the per-profile sum is not a fallback, it is a WRONG ANSWER, so it is
    # never offered: with no per-token reading the line reads n/a instead of "3% utilisation".
    _is_decode = bool(throughput.get("has_unit_ceiling")) if isinstance(throughput, dict) else False
    # An EXPLICIT final_override_ms is the caller stating the measured value, and in a decode run that
    # is already the per-token figure (_reliable_forward_ms), so it is honoured. What is withheld is
    # the DERIVED final_ms -- a per-profile sum -- which is not a fallback for a per-token ceiling.
    _fm_for_roofline = final_override_ms if final_override_ms is not None else (None if _is_decode else final_ms)
    lines.extend(
        _roofline_lines(
            throughput,
            _fm_for_roofline,
            baseline_profile,
            model,
            task,
            per_token_ms=_tok_ms,
            measured_depth=_tok_depth,
        )
    )
    lines.extend(_baseline_bucket_lines(baseline_profile, report_csv))

    # MEASURED FIRST, PROSE ONLY WHERE THERE IS NO MEASUREMENT. This preferred the agent's
    # stages_json -- free text, unvalidated, frozen at capture time -- and fell back to the
    # profile's own op-class buckets only when no attempt happened to carry any. So the one source
    # with a device measurement behind it was the LAST resort: the table has carried "QKV, still
    # HiFi2 -- same lever untried" long after that lever was tried and won, and summed to 529.43 ms
    # while the op breakdown directly above it summed to 556.80.
    _pstages = _stages_from_profile(baseline_profile)
    _st = None if _pstages else next((a for a in reversed(attempts) if isinstance(a, dict) and a.get("stages")), None)
    _stages = _pstages or (_st["stages"] if _st else [])
    if _stages:
        _lbl = (
            f"latest lever on {_op_label(_st.get('op_signature', '?'))}"
            if _st
            else "op-class breakdown (same profile as the table above)"
        )
        # STATE THE VINTAGE AND WHOSE WORDS THESE ARE. The per-stage names are free text the agent
        # wrote when it recorded the stage snapshot, so they freeze at capture time: this table has
        # carried "QKV, still HiFi2 -- same lever untried" and "wo, still HiFi2 x bf8_b" long after
        # both levers were tried, committed and won. The numbers are equally frozen -- this table
        # summed to 529.43 ms while the op breakdown above it summed to 556.80 and the headline said
        # 534.44, three profiles in adjacent sections. Neither can be refreshed here (the snapshot is
        # all there is), so say so rather than let it read as current.
        # The vintage caveat MOVED INTO THE TABLE. It used to ride on this header -- "totals 201.92
        # ms; per-stage notes are the agent's words AT CAPTURE TIME" -- because the table itself gave
        # the reader no way to tell annotation from measurement. It does now: the rows sit under
        # "agent breakdown (annotation, not measurement)" with their own total, beneath the block
        # trace_replay actually measured. Repeating it here restated the total a line above where the
        # table prints it, and put a 90-character disclaimer between the reader and the numbers.
        lines.append(f"Block-level timing (per-stage trace) — {_lbl}:")
        lines.extend(_stage_table_lines(_stages, model, task, stages_measured=bool(_pstages)))
        lines.append("")

    if _model_levers:
        lines.append("Model-level levers")
        lines.append("=" * len("Model-level levers"))
        _h = f"{'lever':<16} {'result':<9} {'e2e Δ':>10}  note"
        lines.append(_h)
        lines.append("-" * len(_h))
        # THE OWNING HELPER, not beat_baseline. "is a win" has one derivation (_win_set); a second
        # reader is how the flag and the marks drifted apart in the first place.
        _lever_wins = _win_set(_model_levers, hdr_base)
        for _j, a in enumerate(_model_levers):
            _name = str(a.get("op_signature", ""))[len(_MODEL_LEVER_PREFIX) :] or "?"
            _d = a.get("fullpipe_delta_ms")
            _dl = f"{_d:+.2f} ms" if isinstance(_d, (int, float)) else "n/m"
            _res = "✓ win" if _j in _lever_wins else ("· wedged" if a.get("wedged") else "· try")
            _n = " ".join((a.get("note") or "").split())[:110] or "(no reason recorded)"
            lines.append(f"{_name:<16} {_res:<9} {_dl:>10}  {_n}")
        lines.append("")

    if by_op:
        _cols = list(_LEVEL_COLS) + (["other"] if any(o.get("other") for o in by_op.values()) else [])
        # ONE WIDTH FOR THE HEADER AND THE CELLS. `:<8` pads to a minimum but never truncates, so a
        # column whose name is longer than 8 -- `structural` at 10, `tp-fracture` at 11 -- pushed the
        # header right while the cells stayed on the 8-wide grid. Everything after the first long name
        # drifted, by 2 characters and then 5, and a reader following a column upward from a ✓win
        # arrived at the wrong lever's name. Widened to fit the longest heading actually rendered, so
        # adding a column can never silently shear the table again.
        _w = max(8, *(len(_disp_level(c)) for c in _cols))
        hdr = f"{'op':<34} " + "  ".join(f"{_disp_level(c):<{_w}}" for c in _cols) + f"  {'best ms':>9}"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for sig in sorted(by_op):
            op = by_op[sig]
            cells = []
            best = None
            for c in _cols:
                cell = op.get(c)
                if cell is None:
                    cells.append(f"{'—':<{_w}}")
                else:
                    st, ms = cell
                    mark = "✓win" if st == "win" else ("·wedge" if st == "wedge" else "·try")
                    cells.append(f"{mark:<{_w}}")
                    if ms is not None and (best is None or ms < best):
                        best = ms
            best_s = f"{best:.2f}" if best is not None else "—"
            lines.append(f"{_op_label(sig):<34} " + "  ".join(cells) + f"  {best_s:>9}")
    else:
        lines.append("(no kernel attempts recorded — nothing was tried, or the run stopped before any lever)")

    lines.append("")
    if committed_wins is not None:
        suffix = f" (branch {opt_branch})" if opt_branch else ""
        lines.append(f"committed wins: {committed_wins}{suffix}")

    # --- Per-attempt detail: gain of EVERY optimization tried (#5a) ---
    if attempts:
        lines.append("")
        lines.append("Per-attempt detail:")
        # NO REASON COLUMN. It carried the agent's own prose, truncated at 200 characters, so every
        # row ended mid-sentence and the table's four measured columns were pushed off the width by
        # text that was never a measurement. The reasoning lives in the kernel log, in full.
        # SAME FURNITURE AS THE TABLES ABOVE: ruled columns, one field per fact. Packed right-aligned
        # against each other the numbers ran together and the op name had no column edge to end at.
        # Sized to the coverage matrix above it, so the two tables in this section share a width.
        _ar = " %-44s\u2502 %-18s\u2502 %-20s\u2502 %-22s\u2502 %s"
        ah = _ar % ("op", "lever", "eager device_ms", "1CQ \u0394 vs current", "result")
        lines.append(ah)
        # THE RULE IS DERIVED FROM THE HEADER, not counted by hand. Hand-counted it drifted the
        # moment a field width changed -- crosses at 33/48/64/82 under dividers at 33/49/66/85.
        lines.append("".join("\u253c" if c == "\u2502" else "\u2500" for c in ah.ljust(128)))
        _unmeasured = 0
        for _i, a in enumerate(attempts):
            if not isinstance(a, dict):
                continue
            sig = _op_label(a.get("op_signature", "?"))
            lever = _disp_level(a.get("kernel_kind") or "?")
            ms = a.get("measured_ms")
            ms_s = f"{ms:.2f}" if isinstance(ms, (int, float)) else "—"
            # READ THE ONE COMPARISON, DO NOT RE-DERIVE IT. This subtracted fullpipe_ms -
            # fullpipe_best_ms itself, and since attempts that never ran an end-to-end inherited the
            # last verdict's numbers, the SAME delta printed on row after row -- so "-41.77 ms",
            # earned by one win, appeared beside a dozen rows marked "no gain". perf_mcp now stamps
            # the attempt's own delta once (_attempt_fullpipe_verdict); absent means this attempt
            # measured no end-to-end of its own, which is a real distinction and reads as "n/m".
            _d = a.get("fullpipe_delta_ms")
            if isinstance(_d, (int, float)):
                gain_s = f"{_d:+.2f} ms"
            elif "fullpipe_measured_here" not in a:
                # LEGACY ROW, written before the stamp existed: subtract as the renderer used to, so
                # old kernel logs still render. Same fallback winning_indices keeps for the ✓ marks.
                _fp, _fb = a.get("fullpipe_ms"), a.get("fullpipe_best_ms")
                gain_s = (
                    f"{_fp - _fb:+.2f} ms" if isinstance(_fp, (int, float)) and isinstance(_fb, (int, float)) else "—"
                )
            elif a.get("fullpipe_measured_here"):
                gain_s = "—"  # measured its own, but nothing to compare against
            else:
                gain_s = "n/m"  # this attempt ran no end-to-end of its own
            # AN UNMEASURED ATTEMPT IS NOT A ROW IN A MEASUREMENT TABLE. The delta column answers one
            # question -- what did this attempt do to the end-to-end number -- and an attempt that ran
            # none has no answer. "n/m" was honest, but on gemma-3-12b-it it was 97 of 105 rows, and
            # ninety-seven of them in front of eight real deltas is not a table anyone reads. They
            # cannot be given a number either: they predate the attempt gate, so subtracting a baseline
            # they never measured against is how a fake win gets manufactured. Dropped here and counted
            # below. The op x rung matrix above still marks every one of them tried, and the kernel log
            # is untouched -- resume still reads all of them. Self-limiting: record_kernel_attempt now
            # refuses an attempt owning no end-to-end, so this count goes to zero on its own.
            if gain_s == "n/m":
                _unmeasured += 1
                continue
            res = "✓ win" if _i in _wins else ("· wedged" if a.get("wedged") else "· no gain")
            lines.append((_ar % (sig, lever, ms_s, gain_s, res)).rstrip())
        if _unmeasured:
            lines.append("")
            lines.append(
                "(%d earlier attempt(s) omitted: no end-to-end measurement of their own. They are "
                "marked tried in the matrix above.)" % _unmeasured
            )

    # NO CODE-CHANGES SECTION. It printed the full source diff of every attempt, win or fail, which
    # on a long run is thousands of lines of patch in a document read for its numbers. The diffs are
    # in the kernel log and in git; a report is not a second copy of the tree.

    # --- Limitations / suggested manual next steps (#5c) ---
    _won_ops = {attempts[i].get("op_signature") for i in _wins}
    _no_gain = sorted({o for o in by_op} - {o for o in _won_ops if o})
    lines.append("")
    lines.append("Limitations / suggested manual next steps:")
    if _no_gain:
        shown = ", ".join(_op_label(o, 26) for o in _no_gain[:8]) + (" …" if len(_no_gain) > 8 else "")
        lines.append(f"- {len(_no_gain)} op(s) tried but no lever beat baseline: {shown}")
        lines.append("  -> inspect the per-op device report and consider a hand-written kernel or a structural change.")
    _measured = [a for a in attempts if isinstance(a, dict) and a.get("measured_ms") is not None]
    _any_win = bool(_wins)
    if not _measured and attempts:
        _unmeasured = len(attempts)
        lines.append(
            f"- INCONCLUSIVE — {_unmeasured} attempt(s) were made but NONE produced a valid measurement, "
            "so no statement about speedup or the ttnn floor is possible. See the per-attempt reasons above."
        )
    elif baseline_ms and final_ms and final_ms >= baseline_ms and _measured and not _any_win:
        lines.append(
            "- No net speedup recorded — the model may already be at its ttnn floor, or the dominant op needs a custom kernel."
        )
    if residual:
        _rv = residual.get("verdict") or residual.get("summary") or residual.get("reason")
        if _rv:
            lines.append(f"- Roofline residual: {str(_rv)[:200]}")
    if not _no_gain and not residual and not (baseline_ms and final_ms and final_ms >= baseline_ms):
        lines.append("- (none flagged automatically — see the per-op device report for remaining headroom.)")

    # --- Reproduce these numbers (#6) ---
    lines.append("")
    lines.append("Reproduce:")
    # CHECK THE PATH, DO NOT JUST PRINT IT. This node-id is carried from the manifest, and the perf test
    # is often GENERATED -- regenerating or renaming it leaves a command that cannot run. The demo and
    # PCC lines below already list their directory at render time, so they self-validate; this one did
    # not. A command handed over for confirmation must not be one that fails on paste.
    if perf_test:
        _pt_file = str(perf_test).split("::")[0]
        try:
            # is_file() RAISES on a pathological name (OSError: File name too long, embedded NUL), and a
            # renderer must never fail on the value it is describing.
            _pt_missing = bool(_pt_file) and not Path(_pt_file).is_file()
        except (OSError, ValueError):
            _pt_missing = False
        lines.append(
            f"  trace+1CQ perf:  python -m pytest {perf_test} -svv"
            + ("   [path no longer exists — perf test regenerated?]" if _pt_missing else "")
        )
    else:
        lines.append("  trace+1CQ perf:  (node-id not provided)")
    # Derive the demo (real input/output) + full-model e2e PCC test from the perf-test path
    # (perf tests live under models/demos/<model>/tests/...); best-effort, pointer only.
    _demo_root = ""
    if perf_test:
        _pt = perf_test.split("::")[0]
        _mi = _pt.find("/tests/")
        if _mi > 0:
            _demo_root = _pt[:_mi]
    if _demo_root:
        import os as _os

        _demo_dir = _os.path.join(_demo_root, "demo")
        _e2e_dir = _os.path.join(_demo_root, "tests", "e2e")
        try:
            _demos = sorted(f for f in _os.listdir(_demo_dir) if f.startswith("demo_") and f.endswith(".py"))
        except Exception:
            _demos = []
        try:
            _pccs = sorted(
                f for f in _os.listdir(_e2e_dir) if f.startswith("test_") and f.endswith(".py") and "perf" not in f
            )
        except Exception:
            _pccs = []
        if _demos:
            lines.append(f"  demo (real input→output):  python {_demo_dir}/{_demos[0]}")
        if _pccs:
            lines.append(f"  full-model e2e PCC:  python -m pytest {_e2e_dir}/{_pccs[0]} -svv")
    if report_csv:
        lines.append(f"  per-op device report (tt-metal format): {report_csv}")

    lines.append("")
    lines.append(
        f"levels: {_levels_display(_dominant_bound_by(baseline_profile))}   |   ✓win = new best so far, ·try = measured no-gain, ·wedge = wedged/crashed when tried, — = not attempted"
    )
    return "\n".join(lines)
