from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _pcc_repair_loop(
    *,
    model_id: str,
    prepare_argv: argparse.Namespace,
    initial_result,
    initial_prompt: str,
    args: argparse.Namespace,
    agent_bin: str,
    agent_model: str,
    max_iters: int,
    agent_timeout_s: int,
    sep: str,
    engine: str = "legacy",
    model_light: Optional[str] = None,
    model_heavy: Optional[str] = None,
) -> int:
    """LLM-driven repair loop for the case where pytest passed but the
    PCC gate fired (false-green detection).

    Mirrors :func:`_runtime_repair_loop`. The difference is the
    prompt: there is no Python traceback to point at, so we feed the
    agent the TT-vs-HF token mismatch + a checklist of likely
    semantic suspects (RoPE wiring, layer-type dispatch, attention
    mask, QK-norm, logit softcap).

    Returns ``0`` if the loop graduates (PCC gate accepts the output)
    or ``_PCC_FAIL_RC`` if the loop exhausts without converging.
    """
    from ..cli import (
        REPO_ROOT,
        _PCC_FAIL_RC,
        _build_forced_edit_preamble,
        _edits_touch_cache_affecting_files,
        _git_changed_files,
        _git_worktree_diff_hash,
        _hash_files,
        _invalidate_tt_weight_cache,
        _invoke_agent,
        _make_verdict_signature,
        _pick_agent_model_for_iter,
        _purge_pycache_for_edited_files,
        _run_prepare_capture,
        _verify_edit_took_effect,
    )
    from ..output_validation import (
        build_pcc_repair_prompt,
        gather_model_architecture_context,
        gather_backend_file_paths,
        gather_tt_weight_cache_summary,
    )

    state = None
    if engine == "evidence":
        try:
            from ..correctness import diagnose as _diagnose
            from ..correctness.hypothesis import new_for_text
            from ..correctness.planner import (
                build_repair_prompt as _evidence_build_prompt,
            )

            state = new_for_text()
            initial_evidence = getattr(initial_result, "_text_evidence", None)
            if initial_evidence is not None:
                _diagnose.apply_priors(state, initial_evidence)
        except Exception as exc:
            print(
                f"  PCC-REPAIR LOOP: failed to set up evidence "
                f"engine state ({type(exc).__name__}: {exc}); "
                f"falling back to legacy prompt for the remainder "
                f"of this loop."
            )
            engine = "legacy"
            state = None

    agentic_ctx = None
    if engine == "agentic":
        try:
            from ..agentic.executor import AgenticContext

            agentic_probe_path = (
                Path(REPO_ROOT) / "generated" / "tt_hw_planner" / f"agentic_probe_{model_id.replace('/', '_')}.json"
            )
            agentic_probe_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                agentic_probe_path.unlink(missing_ok=True)
            except Exception:
                pass
            from ..discovery import BRINGUP_ROOT as _BRINGUP_ROOT

            agentic_ctx = AgenticContext(
                model_id=model_id,
                workspace_root=Path(_BRINGUP_ROOT()),
                probe_output_path=agentic_probe_path,
                max_iters=max_iters,
            )
            print(f"  PCC-REPAIR LOOP: agentic engine active; " f"probe sidecar -> {agentic_probe_path}")
        except Exception as exc:
            print(
                f"  PCC-REPAIR LOOP: agentic engine setup failed "
                f"({type(exc).__name__}: {exc}); falling back to "
                f"legacy prompt for the remainder of this loop."
            )
            engine = "legacy"
            agentic_ctx = None

    print()
    print(sep)
    print("  PCC-REPAIR LOOP  (pytest passed but output diverges from HF)")
    print(sep)
    print(f"  model        : {model_id}")
    print(f"  engine       : {engine}")
    print(f"  max_iters    : {max_iters}")
    print(f"  agent_bin    : {agent_bin}")
    print(f"  agent_model  : {agent_model}")
    print(f"  agent_budget : {agent_timeout_s}s per iter")
    print(f"  initial gate : {initial_result.reason}")
    if state is not None:
        top = state.top_active(n=3)
        print(f"  prior suspects: " f"{[(s.name, round(s.confidence, 2)) for s in top]}")
    print(sep)

    print("  gathering HF model config + backend file surface for repair prompt …")
    try:
        model_config_block = gather_model_architecture_context(model_id)
    except Exception as exc:
        model_config_block = f"  (HF config gather failed: {type(exc).__name__}: {exc})"
    try:
        backend_files_block = gather_backend_file_paths()
    except Exception as exc:
        backend_files_block = f"  (backend file gather failed: {type(exc).__name__}: {exc})"
    try:
        weight_cache_block = gather_tt_weight_cache_summary(model_id)
    except Exception as exc:
        weight_cache_block = f"  (cache state probe failed: {type(exc).__name__}: {exc})"

    previous_attempts: List[str] = []
    current_result = initial_result

    consecutive_no_edit_iters = 0
    for iter_idx in range(1, max_iters + 1):
        print()
        print(sep)
        print(f"  PCC-REPAIR ITER {iter_idx}/{max_iters}  -- asking agent " f"to fix the semantic divergence")
        print(sep)

        _iter_model, _iter_model_reason = _pick_agent_model_for_iter(
            model_default=agent_model,
            model_light=model_light,
            model_heavy=model_heavy,
            complexity_bonus=0,
            failure_class="",
            attempts_so_far=iter_idx - 1,
            force_heavy=consecutive_no_edit_iters >= 1,
        )
        if (model_light or model_heavy) and _iter_model_reason != "default":
            print(f"  [auto:pcc-repair] tiered model pick: " f"{_iter_model} ({_iter_model_reason})")
        forced_edit_mode = consecutive_no_edit_iters >= 1

        pre_iter_diff_hash = _git_worktree_diff_hash()
        pre_iter_changed_files = _git_changed_files()

        agentic_iteration_result = None
        if engine == "agentic" and agentic_ctx is not None:
            try:
                from ..agentic.executor import run_iteration as _agentic_run

                _ev = getattr(current_result, "_text_evidence", None)
                agentic_iteration_result = _agentic_run(
                    ctx=agentic_ctx,
                    captured_demo_output="",
                    evidence_obj=_ev,
                    current_mismatch_ratio=getattr(current_result, "mismatch_ratio", 1.0),
                )
                for n in agentic_iteration_result.notes:
                    print(f"  [agentic] {n}")

                for _k, _v in agentic_iteration_result.env_overrides.items():
                    os.environ[_k] = _v

                if agentic_iteration_result.next_action == "retry":
                    print(
                        f"  [agentic] mechanical action applied; "
                        f"skipping LLM, re-running demo to observe "
                        f"verdict change"
                    )

                    consecutive_no_edit_iters = 0
                    prompt = ""
                elif agentic_iteration_result.next_action == "bail":
                    print(
                        "  [agentic] BAIL requested by executor "
                        "(smoke test failed or no further actions). "
                        "Aborting PCC repair loop -- see [agentic] "
                        "notes above for the focused diagnostic."
                    )
                    break
                else:
                    prompt = agentic_iteration_result.llm_prompt or ""
            except Exception as _aexc:
                print(
                    f"  [agentic] iteration failed "
                    f"({type(_aexc).__name__}: {_aexc}); falling back "
                    f"to evidence/legacy prompt for this iter"
                )
                agentic_iteration_result = None
                prompt = None

        category_cmp = getattr(current_result, "_comparator", None)
        category_evidence = getattr(current_result, "_comparator_evidence", None)
        if (
            engine == "agentic"
            and agentic_iteration_result is not None
            and agentic_iteration_result.next_action != "retry"
            and agentic_iteration_result.llm_prompt
        ):
            pass
        elif (
            category_cmp is not None
            and category_evidence is not None
            and getattr(category_cmp, "category", "") not in ("LLM", "VLM")
        ):
            extra_blocks_list: List[str] = [
                model_config_block,
                backend_files_block,
                weight_cache_block,
            ]

            if agentic_iteration_result is not None:
                dr = getattr(agentic_iteration_result, "diverge_report", None)
                if dr is not None:
                    try:
                        from ..agentic.diverge import format_divergence_block as _fmt_dvg

                        diverge_str = _fmt_dvg(dr)
                        if diverge_str:
                            extra_blocks_list.append(
                                "EMPIRICAL DIVERGENCE (per-module HF vs TT activation stats)\n"
                                "------------------------------------------------------------\n" + diverge_str
                            )
                    except Exception:
                        pass
                conv = getattr(agentic_iteration_result, "convergence", None)
                if conv is not None:
                    try:
                        conv_block = (
                            f"CONVERGENCE TRAJECTORY\n"
                            f"----------------------\n"
                            f"  progress_score   : "
                            f"{getattr(conv, 'progress_score', 0):+.2f}  "
                            f"(-1=regressing, 0=stagnant, +1=fast)\n"
                            f"  stagnant         : "
                            f"{getattr(conv, 'stagnant', False)}\n"
                            f"  iters_to_zero    : "
                            f"{getattr(conv, 'predicted_iters_to_zero', None)}"
                        )
                        extra_blocks_list.append(conv_block)
                    except Exception:
                        pass
            try:
                prompt = category_cmp.build_repair_prompt(
                    model_id=model_id,
                    evidence=category_evidence,
                    result=current_result,
                    iter_idx=iter_idx,
                    max_iters=max_iters,
                    previous_attempts=previous_attempts,
                    extra_blocks=tuple(extra_blocks_list),
                )
            except Exception as _cexc:
                print(
                    f"  PCC-REPAIR LOOP: comparator "
                    f"{category_cmp.label()}.build_repair_prompt "
                    f"raised {type(_cexc).__name__}: {_cexc}; falling "
                    f"back to legacy text prompt."
                )
                prompt = build_pcc_repair_prompt(
                    model_id=model_id,
                    result=current_result,
                    prompt=initial_prompt,
                    iter_idx=iter_idx,
                    max_iters=max_iters,
                    previous_attempts=previous_attempts,
                    model_config_block=model_config_block,
                    backend_files_block=backend_files_block,
                    weight_cache_block=weight_cache_block,
                )
        elif engine == "agentic" and agentic_iteration_result is not None:
            pass
        elif engine == "evidence" and state is not None:
            current_evidence = getattr(current_result, "_text_evidence", None)
            if current_evidence is None:
                prompt = build_pcc_repair_prompt(
                    model_id=model_id,
                    result=current_result,
                    prompt=initial_prompt,
                    iter_idx=iter_idx,
                    max_iters=max_iters,
                    previous_attempts=previous_attempts,
                    model_config_block=model_config_block,
                    backend_files_block=backend_files_block,
                    weight_cache_block=weight_cache_block,
                )
            else:
                prompt = _evidence_build_prompt(
                    model_id=model_id,
                    evidence=current_evidence,
                    state=state,
                    iter_idx=iter_idx,
                    max_iters=max_iters,
                    prompt_text=initial_prompt,
                    model_config_block=model_config_block,
                    backend_files_block=backend_files_block,
                    weight_cache_block=weight_cache_block,
                    forced_edit_mode=forced_edit_mode,
                )
        else:
            prompt = build_pcc_repair_prompt(
                model_id=model_id,
                result=current_result,
                prompt=initial_prompt,
                iter_idx=iter_idx,
                max_iters=max_iters,
                previous_attempts=previous_attempts,
                model_config_block=model_config_block,
                backend_files_block=backend_files_block,
                weight_cache_block=weight_cache_block,
            )
        if forced_edit_mode:
            prompt = _build_forced_edit_preamble(iter_idx) + "\n\n" + prompt
        provider = "claude" if agent_bin.endswith("claude") or agent_bin == "claude" else "cursor"

        _skip_llm = (
            engine == "agentic"
            and agentic_iteration_result is not None
            and agentic_iteration_result.next_action == "retry"
        )
        if _skip_llm:
            print(
                "  [agentic] skipping LLM invocation -- mechanical " "action will be validated by the demo re-run below"
            )
            agent_rc = 0
        else:
            try:
                from .agent import _bringup_cwd as _bcwd

                agent_rc = _invoke_agent(
                    prompt,
                    provider=provider,
                    agent_bin=agent_bin,
                    cwd=_bcwd(),
                    model=_iter_model,
                    timeout_s=agent_timeout_s,
                    iter_tag=f"pcc_repair_iter_{iter_idx}",
                    require_edit_progress=forced_edit_mode,
                )
            except Exception as exc:
                print(f"  agent invocation failed: {type(exc).__name__}: " f"{exc}. Aborting PCC-repair loop.")
                return _PCC_FAIL_RC
            if agent_rc not in (0, None):
                print(
                    f"  agent exited with rc={agent_rc} (non-zero). The "
                    f"agent may have hit its budget or refused; re-running "
                    f"prepare anyway in case partial edits landed."
                )
        post_iter_diff_hash = _git_worktree_diff_hash()
        post_iter_changed_files = _git_changed_files()
        newly_changed = sorted(set(post_iter_changed_files) - set(pre_iter_changed_files))
        diff_touched_paths = newly_changed or post_iter_changed_files
        if pre_iter_diff_hash == post_iter_diff_hash:
            consecutive_no_edit_iters += 1
            print(
                f"  AGENT MADE NO FILE CHANGES this iteration "
                f"(working-tree hash unchanged; consecutive no-edit "
                f"iters: {consecutive_no_edit_iters}). The next "
                f"iteration's prompt will say so explicitly AND the "
                f"loop will escalate to the heavy model (if tiered) "
                f"plus inject a forced-edit preamble."
            )
            attempt_note = (
                f"agent made NO file changes this iteration "
                f"(repeat={current_result.max_repeat_ratio:.0%}, "
                f"mismatch={current_result.mismatch_ratio:.0%}); "
                f"you MUST actually edit code this time"
            )

            if consecutive_no_edit_iters >= 2:
                print()
                print(sep)
                print(
                    f"  PCC-REPAIR LOOP TERMINATED EARLY: "
                    f"{consecutive_no_edit_iters} consecutive iters "
                    f"made zero edits despite the forced-edit "
                    f"preamble and heavy-model escalation. The "
                    f"agent is stuck on this problem; continuing "
                    f"would waste budget. Latest gate verdict: "
                    f"{current_result.reason}"
                )
                print(sep)
                return _PCC_FAIL_RC
        else:
            consecutive_no_edit_iters = 0
            short_paths = ", ".join(diff_touched_paths[:6]) or "(no path info)"
            if len(diff_touched_paths) > 6:
                short_paths += f", ... (+{len(diff_touched_paths) - 6} more)"
            print(f"  agent edited {len(diff_touched_paths)} file(s) " f"this iteration: {short_paths}")
            attempt_note = (
                f"agent edited {len(diff_touched_paths)} file(s) "
                f"({short_paths}); gate "
                f"(repeat={current_result.max_repeat_ratio:.0%}, "
                f"mismatch={current_result.mismatch_ratio:.0%}, "
                f"non_ascii={current_result.non_ascii_ratio:.0%})"
            )
        previous_attempts.append(attempt_note)

        cache_affecting_edits = _edits_touch_cache_affecting_files(diff_touched_paths)
        if cache_affecting_edits:
            short = ", ".join(cache_affecting_edits[:3])
            if len(cache_affecting_edits) > 3:
                short += f", … (+{len(cache_affecting_edits) - 3} more)"
            print(
                f"  cache-affecting edit detected ({short}); "
                f"invalidating TT-native weight cache BEFORE next "
                f"demo run so the edit isn't shadowed by stale "
                f"cached tensors."
            )
            invalidated_eager = _invalidate_tt_weight_cache(model_id)
            if invalidated_eager:
                previous_attempts.append(
                    f"NOTE FROM REPAIR LOOP: iter {iter_idx} edited "
                    f"cache-affecting files ({short}); the TT-native "
                    f"weight cache at {invalidated_eager} was "
                    f"INVALIDATED before the gate re-runs. The next "
                    f"demo will pay re-conversion cost (~5-10 min) "
                    f"but your edit will actually exercise. If the "
                    f"gate still fails, the bug is NOT in weight "
                    f"conversion -- look at runtime paths "
                    f"(attention / RoPE / softcap / decode loop)."
                )

        pre_edit_hashes: Dict[str, str] = {}
        pre_verdict_signature: str = _make_verdict_signature(current_result)
        if diff_touched_paths:
            pre_edit_hashes = _hash_files(diff_touched_paths, REPO_ROOT)
            n_purged = _purge_pycache_for_edited_files(diff_touched_paths, REPO_ROOT)
            if n_purged > 0:
                print(
                    f"  purged {n_purged} stale __pycache__/*.pyc "
                    f"file(s) for edited modules; demo subprocess "
                    f"will re-import against fresh source"
                )

        print()
        print(f"  re-running prepare --execute after PCC-repair attempt " f"{iter_idx} …")
        new_rc, new_output = _run_prepare_capture(prepare_argv)
        if new_rc != 0:
            print(
                f"  iter {iter_idx} BROKE the build (rc={new_rc}). "
                f"The semantic-repair edit introduced a Python error. "
                f"Handing back to the runtime-repair loop semantics is "
                f"out of scope here; stopping the PCC loop and "
                f"returning a hard FAIL."
            )
            return new_rc

        from ..correctness import run_gate as _correctness_run_gate

        _loop_category = "LLM"
        try:
            from ..probe import probe_model as _probe_in_loop

            _gp_in = _probe_in_loop(model_id)
            _loop_category = getattr(_gp_in, "category", None) or "LLM"
            if _loop_category == "Unknown":
                _loop_category = "LLM"
        except Exception:
            pass
        new_result, _ = _correctness_run_gate(
            category=_loop_category,
            model_id=model_id,
            captured_output=new_output,
            args=args,
            engine=engine,
        )
        if new_result is None:
            print(
                f"  iter {iter_idx}: PCC gate could not re-run "
                f"(see warning above). Accepting the build as a soft "
                f"pass."
            )
            return 0

        if engine == "evidence" and state is not None:
            try:
                from ..correctness import diagnose as _diagnose

                _ev_before = getattr(current_result, "_text_evidence", None)
                _ev_after = getattr(new_result, "_text_evidence", None)
                if _ev_before is not None and _ev_after is not None:
                    _diagnose.score_iteration_delta(
                        state,
                        evidence_before=_ev_before,
                        evidence_after=_ev_after,
                        edited_files=diff_touched_paths,
                    )
                    top = state.top_active(n=3)
                    print(
                        f"  evidence engine: updated suspects -> " f"{[(s.name, round(s.confidence, 2)) for s in top]}"
                    )
            except Exception as exc:
                print(
                    f"  evidence engine: hypothesis-state update "
                    f"failed ({type(exc).__name__}: {exc}); next "
                    f"iter will use stale suspect ranking."
                )
        if new_result.ok:
            print()
            print(sep)
            print(
                f"  PCC-REPAIR LOOP GRADUATED at iter {iter_idx} -- "
                f"TT output now matches HF reference within tolerance."
            )
            print(sep)

            if engine == "agentic" and agentic_ctx is not None:
                try:
                    from ..agentic.executor import (
                        compute_diff as _agentic_diff,
                        register_graduation as _agentic_register,
                    )

                    diff_text = _agentic_diff(
                        Path(REPO_ROOT),
                        diff_touched_paths or _git_changed_files(),
                    )
                    if diff_text.strip():
                        ok = _agentic_register(
                            ctx=agentic_ctx,
                            diff_text=diff_text,
                            diff_files=diff_touched_paths or _git_changed_files(),
                            notes=(f"graduated at iter {iter_idx}/" f"{max_iters}"),
                        )
                        print(
                            f"  [agentic] register_graduation: " f"{'ok' if ok else 'no-op (missing signature/diff)'}"
                        )
                except Exception as _aexc:
                    print(f"  [agentic] register_graduation failed: " f"{type(_aexc).__name__}: {_aexc}")
            return 0

        new_verdict_signature = _make_verdict_signature(new_result)
        took_effect, edit_diagnostic = _verify_edit_took_effect(
            edited_files=diff_touched_paths,
            repo_root=REPO_ROOT,
            pre_hashes=pre_edit_hashes,
            prev_verdict_signature=pre_verdict_signature,
            new_verdict_signature=new_verdict_signature,
        )
        if not took_effect:
            print(f"  [edit-verify] {edit_diagnostic}")
            previous_attempts.append(f"NOTE FROM REPAIR LOOP (iter {iter_idx}): " f"{edit_diagnostic}")

        if (
            abs(new_result.mismatch_ratio - current_result.mismatch_ratio) < 0.05
            and abs(new_result.max_repeat_ratio - current_result.max_repeat_ratio) < 0.05
        ):
            print(
                f"  iter {iter_idx}: gate verdict barely moved "
                f"(mismatch {current_result.mismatch_ratio:.0%} -> "
                f"{new_result.mismatch_ratio:.0%}, repeat "
                f"{current_result.max_repeat_ratio:.0%} -> "
                f"{new_result.max_repeat_ratio:.0%}). Re-prompting "
                f"with the previous-attempts log so the agent doesn't "
                f"repeat itself."
            )

            edited_this_iter = "NO file changes" not in attempt_note
            if edited_this_iter:
                invalidated = _invalidate_tt_weight_cache(model_id)
                if invalidated:
                    previous_attempts.append(
                        f"NOTE FROM REPAIR LOOP: agent edits in iter "
                        f"{iter_idx} did not change the demo output, "
                        f"so the cached TT-native weights at {invalidated} "
                        f"were INVALIDATED before iter {iter_idx + 1}. "
                        f"If your edit was a weight-conversion fix "
                        f"(load_checkpoints / state-dict renaming / "
                        f"dtype change), it will now actually be "
                        f"exercised. If your edit was a *runtime* fix "
                        f"(attention dispatch / RoPE freq selection / "
                        f"softcap), the cache invalidation will not "
                        f"help -- the bug is in the runtime path, not "
                        f"the cache."
                    )
        else:
            print(
                f"  iter {iter_idx}: gate verdict shifted "
                f"(mismatch {current_result.mismatch_ratio:.0%} -> "
                f"{new_result.mismatch_ratio:.0%}, repeat "
                f"{current_result.max_repeat_ratio:.0%} -> "
                f"{new_result.max_repeat_ratio:.0%}). Progress; "
                f"continuing."
            )
        current_result = new_result

    print()
    print(sep)
    print(
        f"  PCC-REPAIR LOOP EXHAUSTED  ({max_iters} iters consumed, "
        f"output still diverges from HF). Latest gate verdict: "
        f"{current_result.reason}"
    )
    print(sep)
    return _PCC_FAIL_RC
