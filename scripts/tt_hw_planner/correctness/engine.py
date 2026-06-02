"""Engine-level dispatcher for the correctness gate.

The dispatcher is the single seam between :mod:`cli` and the
per-category :class:`Comparator` plugins. Its responsibilities are:

1. Decide WHICH engine to use (``"legacy"`` vs ``"evidence"``) based
   on the ``--pcc-engine`` CLI flag.
2. Decide WHICH comparator to use (the registry lookup).
3. Run the four-step gate (extract → load_reference → compare →
   build_repair_prompt) defensively, treating any internal
   exception as a soft skip so the dispatcher cannot turn a
   successful bring-up into a failed return code.

Why a separate module
---------------------
Putting the dispatcher into :mod:`correctness.__init__` would have
worked but would tangle import order: the registry needs the
package's submodules to import (so they self-register) before the
dispatcher can lookup a comparator. By making this a leaf module
imported lazily by ``__init__.run_gate``, we avoid the chicken-and-
egg cycle.

Phase 0 vs Phase 1
------------------
* Phase 0: both engines call the legacy implementation. The
  ``--pcc-engine`` flag is accepted but inert.
* Phase 1 (this commit): ``engine="evidence"`` widens the
  comparison window from 32 → 256 tokens and adds a collapse
  detector (see :mod:`.evidence`). The legacy 32-token gate
  reported PASS for medgemma-4b-it whose output collapsed at
  token ~36; the evidence gate catches it.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple


_COMPLETED_DEMO_MARKERS = (
    "==USER 0 - OUTPUT",
    "==EMBED 0 - OUTPUT",
    "==CLASS 0 - OUTPUT",
    "==SEG 0 - OUTPUT",
    "==DET 0 - OUTPUT",
    "==ASR 0 - OUTPUT",
    "Finished decoding",
    "=== Performance metrics ===",
)


def _looks_like_completed_demo(captured: str) -> bool:
    """True iff the pytest stdout shows ANY known demo's terminal
    markers. Used to distinguish 'demo never ran' (genuine soft-skip)
    from 'demo ran but produced markers from the wrong category'
    (false-green mismatch -- must fail loud)."""
    if not captured:
        return False
    return any(m in captured for m in _COMPLETED_DEMO_MARKERS)


def _synthesize_mismatch_failure(comparator_label: str, reason: str) -> Any:
    """Build a :class:`ValidationResult` that demotes a category/demo
    mismatch to a hard failure. Lazily imports ValidationResult to
    avoid a hard dependency at module-load time."""
    from scripts.tt_hw_planner.output_validation import ValidationResult

    return ValidationResult(
        ok=False,
        reason=(
            f"CATEGORY/DEMO MISMATCH ({comparator_label}): {reason}. "
            f"The demo ran but produced markers from a different "
            f"category; this is the classic false-green pattern "
            f"(the model was routed to the wrong backend). Refusing "
            f"to soft-pass."
        ),
        tt_text="",
        hf_text="",
        compared_tokens=0,
    )


def run_gate(
    *,
    category: str,
    model_id: str,
    captured_output: str,
    args: Any,
    engine: str = "legacy",
    compare_tokens: Optional[int] = None,
    instruct: bool = True,
) -> Tuple[Optional[Any], Optional[str]]:
    """Dispatch the correctness gate for ``model_id`` of
    ``category``.

    The signature deliberately mirrors :func:`cli._run_pcc_gate`
    so the cli.py call sites can be migrated by a name change
    only. Returns ``(result, prompt_used)``:

    * ``result`` is a :class:`ValidationResult` (or ``None`` if the
      gate could not run; the caller treats ``None`` as a soft pass
      with a printed warning).
    * ``prompt_used`` is the demo's first prompt text (or ``None``
      if not available); the caller passes this to the repair loop
      so the agent sees the same prompt the demo saw.

    ``engine`` selects between:

    * ``"legacy"`` — call into :func:`cli._run_pcc_gate` directly.
      The pre-2026-05-24 behaviour. 32-token compare window, no
      collapse detection, ignores anything past the gate's window.
    * ``"evidence"`` — call into :func:`run_evidence_gate` below.
      256-token compare window, collapse detection, returns a
      richer :class:`evidence.TextEvidence` attached to the
      :class:`ValidationResult` for the repair loop to consume.
      Currently only the text comparator (LLM/VLM) supports this
      engine; other categories fall back to legacy until their
      comparators are wired in later phases.
    * ``"agentic"`` — same gate engine as ``"evidence"`` (so the
      repair loop sees the rich text-evidence) and the difference
      is in the *repair* path, not the gate path. Historically the
      now-deleted ``_pcc_repair_loop`` switched to the agentic
      planner when ``engine == "agentic"``; today Path 2 escalates
      via ``_maybe_escalate_pcc_fail`` into Path 1's per-component
      iterate, which always uses the agentic planner.
    """

    if category and category not in ("LLM", "VLM", "Unknown"):
        cmp_obj = _resolve_category_comparator(category, model_id)
        if cmp_obj is not None:
            return _run_via_comparator(
                comparator=cmp_obj,
                model_id=model_id,
                captured_output=captured_output,
            )

    if engine in ("evidence", "agentic"):
        return run_evidence_gate(
            category=category,
            model_id=model_id,
            captured_output=captured_output,
            args=args,
            compare_tokens=compare_tokens,
            instruct=instruct,
        )

    from scripts.tt_hw_planner.cli import _run_pcc_gate

    return _run_pcc_gate(
        model_id=model_id,
        captured_output=captured_output,
        args=args,
        compare_tokens=compare_tokens,
        instruct=instruct,
    )


def _resolve_category_comparator(category: str, model_id: str):
    """Look up the registered Comparator for ``category`` (and the
    optional fine-grained model_id). Returns ``None`` if no
    comparator claims this category."""
    try:
        from .registry import get_comparator

        cmp_obj = get_comparator(category, model_id)
        if cmp_obj is None:
            if "/" in category:
                cmp_obj = get_comparator(category.split("/", 1)[0], model_id)
        return cmp_obj
    except Exception:
        return None


def _run_via_comparator(
    *,
    comparator,
    model_id: str,
    captured_output: str,
) -> Tuple[Optional[Any], Optional[str]]:
    """Drive a registered Comparator through its full pipeline:
    extract → load_reference → compare. Returns the same
    ``(result, prompt)`` shape :func:`run_gate` produces for the
    LLM/VLM text path.

    Any exception from the comparator is soft-skipped — a buggy
    plugin must NEVER block bring-up of a model that actually
    works."""
    try:
        evidence = comparator.extract(captured_output, model_id)
    except Exception as exc:
        print(f"  PCC gate ({comparator.label()}): extract() raised " f"{type(exc).__name__}: {exc}. Soft-skipping.")
        return None, None
    if not getattr(evidence, "ok", True):
        reason = getattr(evidence, "reason", "unknown reason")
        if _looks_like_completed_demo(captured_output):
            print(
                f"  PCC gate ({comparator.label()}): MISMATCH FAIL "
                f"({reason}). A different category's demo markers "
                f"are present in the output; refusing to soft-pass."
            )
            return _synthesize_mismatch_failure(comparator.label(), reason), None
        print(
            f"  PCC gate ({comparator.label()}): no evidence "
            f"({reason}). Soft-skipping (no completed-demo markers "
            f"found, so this looks like 'demo never ran' rather than "
            f"a routing mismatch)."
        )
        return None, None
    try:
        reference = comparator.load_reference(evidence, model_id)
    except Exception as exc:
        print(
            f"  PCC gate ({comparator.label()}): load_reference() "
            f"raised {type(exc).__name__}: {exc}. Soft-skipping."
        )
        return None, None
    try:
        result = comparator.compare(evidence, reference)
    except Exception as exc:
        print(f"  PCC gate ({comparator.label()}): compare() raised " f"{type(exc).__name__}: {exc}. Soft-skipping.")
        return None, None

    try:
        setattr(result, "_comparator_evidence", evidence)
        setattr(result, "_comparator", comparator)
    except Exception:
        pass

    prompt_hint = ""
    try:
        ih = getattr(evidence, "input_hint", None)
        if isinstance(ih, str):
            prompt_hint = ih
        elif ih is not None:
            prompt_hint = repr(ih)[:512]
    except Exception:
        pass
    return result, prompt_hint or None


def run_evidence_gate(
    *,
    category: str,
    model_id: str,
    captured_output: str,
    args: Any,
    compare_tokens: Optional[int] = None,
    instruct: bool = True,
) -> Tuple[Optional[Any], Optional[str]]:
    """The evidence-engine PCC gate.

    Mirrors :func:`cli._run_pcc_gate` in shape (extract → load HF
    → compare) but widens the comparison window and produces a
    :class:`evidence.TextEvidence` attached as
    ``result._text_evidence`` so the repair loop can use it.

    Soft-skips on any non-fatal failure: a missing
    ``==USER 0 - OUTPUT`` block, a missing HF mirror, an HF
    generation timeout, a tokenizer reload failure. The caller
    treats ``None`` results as "gate did not engage" (soft pass)
    so this function CANNOT downgrade a real bring-up into a
    false-fail.
    """
    from scripts.tt_hw_planner.output_validation import (
        DEFAULT_COMPARE_TOKENS,
        ValidationResult,
        compare_token_sequences,
        extract_demo_user_output,
        generate_hf_reference,
        load_demo_first_prompt,
        tokenize_text_for_compare,
    )
    from .evidence import (
        DEFAULT_SCAN_LIMIT,
        build_text_evidence,
    )

    n_compare = compare_tokens or DEFAULT_COMPARE_TOKENS
    scan_limit = max(DEFAULT_SCAN_LIMIT, n_compare)

    tt_text = extract_demo_user_output(captured_output, user_idx=0)
    if tt_text is None:
        print(
            "  PCC gate (evidence): could not find a "
            "'==USER 0 - OUTPUT' block in the pytest output. "
            "Skipping the gate (the demo may not be "
            "simple_text_demo, or the test path may not emit the "
            "canonical output marker)."
        )
        return None, None
    if not tt_text.strip():
        result = ValidationResult(
            ok=False,
            reason=(
                "TT demo produced an empty decoded-text block; "
                "this is almost certainly a silent generation "
                "failure that pytest didn't catch."
            ),
            tt_text="",
            hf_text="",
            compared_tokens=0,
        )
        return result, None

    prompt = load_demo_first_prompt()
    if not prompt:
        print("  PCC gate (evidence): could not load the demo's " "first prompt. Skipping the gate.")
        return None, None

    print(
        "  PCC gate (evidence): running HF CPU reference for the "
        f"same prompt (model={model_id}, scan window={scan_limit} "
        f"tokens, greedy). The evidence engine compares the WHOLE "
        f"output (not just the first 32 tokens) and looks for a "
        f"mid-sequence collapse position; this catches the "
        f"'first N tokens fine, then garbage' pattern that the "
        f"legacy gate misses."
    )
    # Always request return_logits=True so the strict logit-PCC gate
    # (see below) has the HF step-0 logits to compare against. Without
    # this, the strict gate flips to fail-closed UNVERIFIED — which is
    # safer than a false-green but defeats the comparison the user
    # wanted.
    try:
        hf = generate_hf_reference(
            model_id,
            prompt,
            max_new_tokens=scan_limit,
            instruct=instruct,
            return_logits=True,
        )
    except Exception as exc:
        print(
            f"  PCC gate (evidence): HF reference generation "
            f"FAILED ({type(exc).__name__}: {exc}). Skipping the "
            f"gate (soft pass)."
        )
        return None, prompt

    try:
        tt_token_ids = tokenize_text_for_compare(model_id, tt_text)
    except Exception as exc:
        print(f"  PCC gate (evidence): tokenizer re-load FAILED " f"({type(exc).__name__}: {exc}). Skipping the gate.")
        return None, prompt

    evidence_record = build_text_evidence(
        tt_text=tt_text,
        tt_tokens=tt_token_ids,
        hf_text=hf.text,
        hf_tokens=hf.token_ids,
        scan_limit=scan_limit,
        input_hint=prompt,
    )

    legacy_result = compare_token_sequences(
        tt_token_ids,
        hf.token_ids,
        tt_text=tt_text,
        hf_text=hf.text,
        compare_tokens=n_compare,
    )

    evidence_ok = legacy_result.ok and evidence_record.collapse_position is None
    if evidence_record.collapse_position is not None and legacy_result.ok:
        legacy_result.reason = (
            f"WIDE-SCAN collapse at token "
            f"{evidence_record.collapse_position} after "
            f"{evidence_record.prefix_match_count} coherent tokens "
            f"(legacy 32-token gate would have falsely passed; the "
            f"evidence engine caught the 'good prefix then garbage' "
            f"pattern)."
        )
    legacy_result.ok = evidence_ok

    setattr(legacy_result, "_text_evidence", evidence_record)

    # Strict logit-PCC ≥ 0.99 gate — MUST fire on every LLM/VLM
    # bring-up. Without this, the evidence engine stamps SUCCESS on
    # token-overlap alone (which tolerates up to 70% mismatch by
    # design). The check is fail-closed: missing TT logits, missing
    # HF logits, or PCC < threshold all flip ok=False.
    from .text import _TT_LOGITS_PATH_RE, apply_strict_logit_pcc_gate

    _tt_logits_match = _TT_LOGITS_PATH_RE.search(captured_output)
    _tt_logits_path = _tt_logits_match.group("path") if _tt_logits_match else None
    _hf_step0_logits = getattr(hf, "step0_logits", None)
    _token_reason_pre_strict = legacy_result.reason
    apply_strict_logit_pcc_gate(
        legacy_result,
        _tt_logits_path,
        _hf_step0_logits,
        informational_token_reason=_token_reason_pre_strict,
    )

    print(f"  {legacy_result.summary()}")
    if evidence_record.collapse_position is not None:
        print(
            f"  evidence: collapse@tok{evidence_record.collapse_position}, "
            f"prefix_match={evidence_record.prefix_match_count}, "
            f"regime_shifts={[r.kind for r in evidence_record.regime_shifts]}"
        )
    if not legacy_result.ok:
        print()
        print("  ----- TT-demo output (first 200 chars) -----")
        print(f"  {(legacy_result.tt_text or '')[:200]}")
        print("  ----- HF-reference output (first 200 chars) -----")
        print(f"  {(legacy_result.hf_text or '')[:200]}")
        print("  --------------------------------------------------")
    return legacy_result, prompt


__all__ = ["run_gate", "run_evidence_gate"]
