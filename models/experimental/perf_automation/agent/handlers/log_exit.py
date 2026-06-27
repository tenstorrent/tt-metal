"""LOG + CHECK_EXIT handlers (Member 2) — REAL, deterministic.

LOG:        append one ledger row from ctx.state["last_decision"], update
            counters + metric.current (on keep), mark the lever tried. -> CHECK_EXIT
CHECK_EXIT: delegate to exit_policy.check_exit(state). -> ROUTE | DONE | STOPPED

Template for the other M2 handlers. Note the idempotent experiment_id so a
resumed LOG never double-writes.
"""

from __future__ import annotations

from .. import exit_policy, states


def log(ctx) -> str:
    d = ctx.state.get("last_decision") or {}
    it = ctx.state.get("iteration", 0)
    lever = ctx.state.get("selected_lever")
    before, after = d.get("before"), d.get("after")
    row = {
        "experiment_id": f"{ctx.run.run_id}#{it}",  # idempotent replay key
        "iteration": it,
        "bucket": ctx.state.get("current_bucket"),
        "lever": lever,
        "result": d.get("result"),
        "reason": d.get("reason"),
        "before": before,
        "after": after,
        "delta": (None if before is None or after is None else round(after - before, 4)),
        "pcc": d.get("pcc"),
        "hypothesis": d.get("hypothesis"),
    }
    ctx.ledger.append(row)

    edited_on_live_line = _edited_files_are_live(ctx)
    if d.get("reason") == "edit_inert" and not ctx.state.get("exec_scoped_files") and not edited_on_live_line:
        # REACTIVE mode only (no exec-scope) AND the edit was NOT on a known-live line:
        # the file may genuinely be off-path. Record it and -- up to a cap -- DON'T spend
        # the lever: retry it on an on-path file. In SCOPED mode every file already runs,
        # so an inert edit falls through to the else branch (spend the lever, move on).
        #
        # CRITICAL: if attribution proved the edited line IS executed, inert means the
        # EDIT was a no-op, NOT that the file is dead -- so we do NOT come here (the
        # DECIDE fixer already fed the agent the measured op-delta + its own diff). Blaming
        # file location there was the misdiagnosis that churned the lever 6x for nothing.
        for f in (ctx.state.get("last_edit") or {}).get("files") or []:
            if f not in ctx.state.setdefault("inert_files", []):
                ctx.state["inert_files"].append(f)
        retries = ctx.state.get("inert_retries", 0)
        if retries < states.MAX_INERT_RETRY:
            ctx.state["inert_retries"] = retries + 1
        else:  # cap hit -> stop steering this lever, let SELECT move on
            ctx.state["inert_retries"] = 0
            if lever and lever not in ctx.state.setdefault("tried", []):
                ctx.state["tried"].append(lever)
    else:
        ctx.state["inert_retries"] = 0
        if lever and lever not in ctx.state.setdefault("tried", []):
            ctx.state["tried"].append(lever)
        if d.get("result") == "keep" and after is not None:
            ctx.state["metric"]["current"] = after

    # Per-bucket measured history + no-gain streak — the EVIDENCE the agentic waste
    # judge (CHECK_EXIT) reasons over. History is measured results only.
    cur_bucket = ctx.state.get("current_bucket")
    if cur_bucket is not None:
        hist = ctx.state.setdefault("bucket_history", {}).setdefault(cur_bucket, [])
        hist.append(
            {
                "lever": lever,
                "result": d.get("result"),
                "reason": d.get("reason"),
                "before": before,
                "after": after,
                "delta": row["delta"],
            }
        )
    # Streak counts ONLY MEASURED no-gains (an edit that ran on-device and didn't help).
    # already_applied / edit_inert / measure_failed / edit_failed never ran a real
    # measurement, so they are NOT evidence the bucket is tapped -- counting them let the
    # waste-judge bail after a few no-op levers with most levers/buckets still untried.
    if d.get("result") == "keep":
        ctx.state["nogain_streak"] = 0
    elif d.get("reason") == "no_gain":
        same = ctx.state.get("streak_bucket") == cur_bucket
        ctx.state["nogain_streak"] = (ctx.state.get("nogain_streak", 0) + 1) if same else 1
    # else (no-op / non-measured result): leave the streak unchanged
    ctx.state["streak_bucket"] = cur_bucket

    # Manual-perf TARGET visibility (the "Make Fast Models Fast" Slide-4 gap): when a
    # --target is set, report current-vs-target each iter so the loop is chasing a number,
    # not just sweeping knobs. The actual stop-on-target lives in exit_policy (rule 1).
    m = ctx.state.get("metric") or {}
    tgt = m.get("target")
    cur = m.get("current")
    if tgt is not None and cur is not None:
        unit = m.get("unit", "")
        gap = (cur - tgt) if m.get("direction", "min") == "min" else (tgt - cur)
        pct = (gap / tgt * 100.0) if tgt else 0.0
        status = "TARGET MET" if gap <= 0 else f"gap {gap:+.4f}{unit} ({pct:+.1f}% from target)"
        ctx.log_event(states.LOG, "info", f"vs manual target {tgt}{unit}: current {cur:.4f}{unit} — {status}")

    ctx.state["iteration"] = it + 1
    return states.CHECK_EXIT


def _emit_residual(ctx) -> None:
    """END-OF-RUN certification (once): how far the final profile is from its ttnn-reachable
    roofline floor + where the gap lives. Lets the run SAY 'nothing ttnn-reachable left' vs
    just 'ran out of levers'. Best-effort; written to runs/<id>/residual_report.json + logged."""
    if ctx.state.get("_residual_emitted"):
        return
    ctx.state["_residual_emitted"] = True
    try:
        import json as _json

        from .. import roofline

        env = (ctx.manifest or {}).get("env", {}) if isinstance(ctx.manifest, dict) else {}
        rep = roofline.residual_report(ctx.current_profile(), env or {})
        (ctx.run.dir / "residual_report.json").write_text(_json.dumps(rep, indent=2, sort_keys=True))
        top = rep.get("open_ops") or []
        verdict = (
            "AT ttnn-floor (no reachable gain left in modeled hot ops)"
            if rep.get("at_floor")
            else f"{rep.get('residual_gap_ms')}ms above modeled floor across {rep.get('n_open')} open op(s)"
        )
        head = (
            f"; biggest open: {top[0].get('op_code')} [{top[0].get('shape')}] "
            f"gap {top[0].get('gap_ms')}ms bound_by={top[0].get('bound_by')}"
            if top
            else ""
        )
        ctx.log_event(
            states.CHECK_EXIT,
            "info",
            f"RESIDUAL vs roofline: measured {rep.get('total_device_ms')}ms, modeled floor "
            f"{rep.get('modeled_floor_ms')}ms -> {verdict}{head}",
        )
    except Exception as exc:
        ctx.log_event(states.CHECK_EXIT, "warn", f"residual report skipped: {exc}")


def check_exit(ctx) -> str:
    decision = exit_policy.check_exit(ctx.state)  # "continue" | "DONE" | "STOPPED"
    if decision in ("DONE", "STOPPED"):  # target met / budget / max-iter
        _emit_residual(ctx)
        return decision

    # Per-bucket lever exhaustion: if every candidate for the CURRENT bucket has
    # been tried, mark that bucket exhausted and ROUTE to the next-slowest one
    # (ROUTE already skips exhausted_buckets). Only STOP once ALL buckets are out.
    state = ctx.state
    candidates = state.get("candidates") or []
    tried = set(state.get("tried") or [])
    untried = [c for c in candidates if c not in tried]

    # AGENTIC waste decision (not a static rule): after a streak of MEASURED no-gains
    # in this bucket, an agent reasons over the measured history and decides whether to
    # keep spending device measurements here. Fires only while knobs remain untried
    # (full exhaustion is handled deterministically below). Fails open to 'continue'.
    if untried and state.get("nogain_streak", 0) >= states.JUDGE_STREAK_THRESHOLD:
        cur = state.get("current_bucket")
        judge = ctx.deps.get("progress_judge") or _default_judge()
        verdict = judge(bucket=cur, untried=untried, history=(state.get("bucket_history") or {}).get(cur) or [])
        ctx.record_agent_call(states.CHECK_EXIT, "progress", verdict.get("model", "?"), verdict.get("usage"))
        dec = verdict.get("decision", "continue")
        ctx.log_event(states.CHECK_EXIT, "info", f"waste-judge: {dec} — {verdict.get('reasoning', '')}")
        state["nogain_streak"] = 0  # reset so the judge isn't re-invoked every iter
        # The waste-judge may SKIP a tapped bucket (advance to the next-slowest) but must
        # NEVER end the whole run while other buckets still have headroom -- that early-quit
        # left datamove/reduction (40% of device time) completely unexplored. Both 'stop' and
        # 'exhaust' therefore exhaust THIS bucket and advance; the run stops ONLY when every
        # bucket is exhausted (full sweep matmul->datamove->reduction->eltwise).
        if dec in ("stop", "exhaust"):
            exhausted = set(state.get("exhausted_buckets") or [])
            if cur:
                exhausted.add(cur)
            state["exhausted_buckets"] = sorted(exhausted)
            try:
                allb = {b["id"] for b in (ctx.current_profile().get("buckets") or [])}
                if (state.get("metric") or {}).get("name", "device_ms") == "device_ms":
                    allb.discard("host_overhead")
            except Exception:
                allb = set()
            if allb and exhausted >= allb:
                ctx.log_event(
                    states.CHECK_EXIT, "info", f"waste-judge: all buckets exhausted {sorted(exhausted)}; stopping"
                )
                _emit_residual(ctx)
                return states.STOPPED
            ctx.log_event(
                states.CHECK_EXIT, "info", f"waste-judge -> advance past '{cur}' (exhausted={sorted(exhausted)})"
            )
            return states.ROUTE
        # 'continue' falls through to the deterministic logic below

    # Exhaust the bucket when all its candidates are tried OR it has no candidates
    # at all (a no-hit bucket the playbook can't optimize — never leave it routable).
    if not candidates or all(c in tried for c in candidates):
        cur = state.get("current_bucket")
        exhausted = set(state.get("exhausted_buckets") or [])
        if cur:
            exhausted.add(cur)
        state["exhausted_buckets"] = sorted(exhausted)
        try:
            all_buckets = {b["id"] for b in (ctx.current_profile().get("buckets") or [])}
            # host_overhead is non-routable under the device-floor metric, so it must
            # not count toward "all buckets exhausted" or the run never stops.
            if (state.get("metric") or {}).get("name", "device_ms") == "device_ms":
                all_buckets.discard("host_overhead")
        except Exception:
            all_buckets = set()
        if all_buckets and exhausted >= all_buckets:
            ctx.log_event(states.CHECK_EXIT, "info", f"all buckets exhausted {sorted(exhausted)}; stopping")
            _emit_residual(ctx)
            return states.STOPPED
        ctx.log_event(
            states.CHECK_EXIT, "info", f"bucket '{cur}' exhausted; advancing to next (exhausted={sorted(exhausted)})"
        )
        return states.ROUTE

    return states.ROUTE


def _edited_files_are_live(ctx) -> bool:
    """True iff a file the edit touched also appears in attribution's hot_sources
    (the executed-and-hot lines). If so, an inert result is a no-op EDIT, not a dead
    FILE -- so the reactive 'wrong file, retry elsewhere' path must NOT fire."""
    edited = (ctx.state.get("last_edit") or {}).get("files") or []
    if not edited:
        return False
    live_files = set()
    for h in ctx.state.get("hot_sources") or []:
        src = (h.get("src") or "").split(":")[0]  # strip :lineno
        if src:
            live_files.add(src)
    for f in edited:
        base = f.replace("\\", "/").rsplit("/", 1)[-1]
        if any(
            lf.replace("\\", "/").endswith("/" + base) or lf.replace("\\", "/").rsplit("/", 1)[-1] == base
            for lf in live_files
        ):
            return True
    return False


def _default_judge():
    from ..progress_agent import make_progress_judge_runner

    return make_progress_judge_runner()
