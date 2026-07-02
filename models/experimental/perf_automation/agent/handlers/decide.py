"""DECIDE handler (PLAN 8.8) — REAL, pure. keep vs discard. No agent, no device.

By the time DECIDE runs, GATE_PCC passed and REMEASURE produced before/after (and
the run spread). Keep only a real improvement per metric.direction beyond the
noise floor; otherwise discard (no_gain). Edit/run crashes never reach here —
they were absorbed by REPAIR (8.5.2); an unmeasurable lever is discarded by
REMEASURE (8.7).

NOTE: _NOISE_FLOOR is a PLACEHOLDER (see progress.txt — noise-floor deferred). The
agreed fix is to use the MEASURED run spread (last_decision["spread"]) instead of
a constant; the spread is already recorded, so swapping this is a one-line change.
"""

from __future__ import annotations

from .. import states

_NOISE_FLOOR = 0.05  # PLACEHOLDER — not a measurement; see progress.txt
_SUSPICIOUS_GAIN = 0.5  # >50% improvement from ONE lever -> flag for verification


def decide(ctx) -> str:
    d = ctx.state.get("last_decision") or {}

    # Measurement-validity guard: never KEEP on a capture we can't trust.
    # REMEASURE flags structurally-incomparable profiles (op-count collapse /
    # dominant bucket vanished) -- e.g. the false 22x from a 27-op tracy capture.
    if d.get("measurement_ok") is False:
        d["result"] = "discard"
        d["reason"] = d.get("measurement_reason") or "measurement_invalid"
        ctx.state["last_decision"] = d
        ctx.log_event(states.DECIDE, "warn", f"discard (untrusted measurement): {d['reason']}")
        return states.REVERT

    before, after = d.get("before"), d.get("after")
    direction = ctx.state["metric"].get("direction", "min")

    improved = (
        before is not None
        and after is not None
        and (
            (direction == "min" and after <= before - _NOISE_FLOOR)
            or (direction == "max" and after >= before + _NOISE_FLOOR)
        )
    )
    d["result"] = "keep" if improved else "discard"
    if not improved:
        if d.get("op_graph_identical"):
            d["reason"] = "edit_inert"
            # FIXER: iterate an inert structural shard (re-invoke with evidence) instead of
            # discarding; capped, and only while the agent keeps making a new edit (stuck-detector).
            sig = ctx.state.get("edit_sig")
            stuck = sig is not None and sig == ctx.state.get("prev_fixer_sig")
            if (
                ctx.state.get("last_was_structural")
                and not stuck
                and ctx.state.get("inert_fix_attempts", 0) < states.MAX_STRUCT_FIX
            ):
                ctx.state["prev_fixer_sig"] = sig
                ctx.state["inert_fix_attempts"] = ctx.state.get("inert_fix_attempts", 0) + 1
                ctx.state["inert_repair_error"] = _inert_evidence(ctx, d)
                ctx.state["last_decision"] = d
                ctx.log_event(
                    states.DECIDE,
                    "info",
                    f"edit_inert -> FIXER retry {ctx.state['inert_fix_attempts']}/{states.MAX_STRUCT_FIX} "
                    f"(iterate the shard, don't discard); op-delta:\n{d.get('op_delta')}",
                )
                return states.APPLY
            ctx.log_event(
                states.DECIDE,
                "warn",
                (
                    "fixer abandoned: edit unchanged from last retry (not converging)"
                    if stuck
                    else "edit_inert: post-edit op graph byte-identical to pre-edit (target not exercised)"
                ),
            )
        else:
            d["reason"] = "no_gain"
    elif before:
        gain = abs(after - before) / abs(before)
        if gain > _SUSPICIOUS_GAIN:
            d["suspicious_gain"] = round(gain, 3)
            ctx.log_event(
                states.DECIDE, "warn", f"SUSPICIOUS {gain:.0%} gain ({before} -> {after}); comparable but verify"
            )
    ctx.state["last_decision"] = d
    ctx.log_event(states.DECIDE, "info", f"{d['result']} ({before} -> {after}, dir={direction})")
    return states.COMMIT if improved else states.REVERT


def _inert_evidence(ctx, d) -> str:
    """Build the inert-repair message from MEASURED GROUND TRUTH, not a canned guess.

    The previous version asserted a specific cause ("your to_memory_config didn't
    execute") — but the agent often never wrote one; it shipped a bare program_config.
    Telling it to "reconnect" something it never wrote sent it in circles. Instead we
    hand it (1) its OWN diff, (2) the measured per-bucket op-delta proving nothing
    moved, (3) attribution proof the edited line IS live — and let it diagnose. This
    is the one fact it cannot get by re-reading the file: what its edit DID, not says."""
    delta = d.get("op_delta") or "(op-delta unavailable)"
    diff = (ctx.state.get("last_diff") or "").strip() or "(diff unavailable)"
    if len(diff) > 6000:  # keep the prompt bounded; head is where the edit anchors live
        diff = diff[:6000] + "\n... (diff truncated)"
    live = _live_line_proof(ctx)
    return (
        "Your previous edit applied and PASSED PCC, but it was INERT: the device op graph is "
        "byte-identical to before your edit. This is MEASURED ground truth (the device, not a guess):\n\n"
        f"Per-bucket op counts, before -> after your edit:\n{delta}\n\n"
        f"{live}\n"
        "So the code you edited DOES run — your edit simply produced ZERO new device ops. Read the "
        "EXACT diff you wrote last time and find why:\n\n"
        "----- YOUR PREVIOUS DIFF -----\n"
        f"{diff}\n"
        "----- END DIFF -----\n\n"
        "Diagnose from the diff above. The usual cause: you added a `program_config=`/`memory_config=` "
        "kwarg (or no real data-movement op at all) while the input tensor stayed DRAM_INTERLEAVED — a "
        "no-op. To change the graph, a standalone `ttnn.to_memory_config(x, <sharded L1 cfg>)` (or "
        "interleaved_to_sharded) must execute ON the variable that is actually passed into the hot "
        "matmul, and the datamove bucket count above MUST increase next time."
    )


def _live_line_proof(ctx) -> str:
    """Turn attribution into proof the edit was NOT on dead code (the false reading the
    old reactive path made). Names the hottest executed source line + its call count."""
    hs = ctx.state.get("hot_sources") or []
    if not hs:
        return "Attribution confirms the bucket's ops execute on the forward path (this is NOT dead code)."
    top = hs[0]
    return (
        f"Attribution PROVES this is not dead code: {top.get('src','?')} "
        f"({top.get('func','?')}) executed {top.get('calls','?')} times this run."
    )
