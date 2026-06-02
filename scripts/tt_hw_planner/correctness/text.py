"""Text-category correctness (LLM + VLM).

Phase 0: this module is a thin :class:`Comparator` that wraps the
existing :mod:`tt_hw_planner.output_validation` helpers. The point of
existing right now is to (a) register a comparator instance under
the ``"LLM"`` and ``"VLM"`` categories so the dispatcher has someone
to call, and (b) provide a stable surface that Phase 1 will swap
out for the Evidence/Hypothesis/Diagnose/Planner machinery without
touching the call sites in :mod:`cli`.

In Phase 1 the same class gains real intelligence:

* :meth:`extract` will start returning a richer :class:`Evidence`
  including the full decoded text, the position of the first
  collapse (defined as the first ``k`` such that the next 32 tokens
  all match the same loguru-printable pattern, or the first
  non-ASCII run, whichever comes first), and the regime-shift map.
* :meth:`compare` will widen the comparison window from 32 → 128
  tokens and add a mid-sequence collapse penalty so the medgemma
  pattern (first ~36 tokens fine, then garbage) can no longer slip
  through as a green build.
* :meth:`build_repair_prompt` will consume the running
  :class:`hypothesis.HypothesisState` and emit a focused, ranked
  suspect list instead of the current static checklist.

For Phase 0 every method is a pass-through. The Phase-0 success
criterion is "behavior is byte-identical to today, but a
``correctness.get_comparator('LLM')`` call returns a usable object."
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence

from .base import Comparator, Evidence, ValidationResult
from .registry import register_comparator


_TEXT_CATEGORIES = ("LLM", "VLM")


_TT_LOGITS_PATH_RE = re.compile(r"^\s*==LOGITS\s+PATH:\s*(?P<path>\S+)\s*$", re.M)


_LOGIT_PCC_MIN = 0.99


def _compute_step0_logit_pcc(tt_logits_path: str, hf_logits: Any) -> Optional[float]:
    """Load TT step-0 logits from ``tt_logits_path`` and compute the
    Pearson PCC against the HF step-0 logits.

    Reuses :func:`scripts.tt_hw_planner.activation_diff._pcc` -- the
    same mean-centred cosine the scaffolded per-component tests use,
    so the LLM/VLM gate and the cold-start gate report comparable
    numbers. Returns ``None`` on any failure (missing file, shape
    mismatch beyond the tail-pad rescue, NaNs, no torch); the caller
    treats ``None`` as "post-screen could not run" rather than
    "post-screen failed".
    """
    try:
        import numpy as np
        import torch
    except Exception:
        return None
    from pathlib import Path

    p = Path(tt_logits_path)
    if not p.is_file():
        return None
    try:
        tt_arr = np.load(str(p))
    except Exception:
        return None
    try:
        tt_t = torch.from_numpy(np.asarray(tt_arr))
        hf_t = torch.from_numpy(np.asarray(hf_logits))
    except Exception:
        return None
    from scripts.tt_hw_planner.activation_diff import _pcc as _ad_pcc

    return _ad_pcc(tt_t, hf_t)


class TextComparator(Comparator):
    """Comparator for tt_transformers-style text-generation demos.

    Reads ``==USER N - OUTPUT`` blocks out of pytest stdout, runs
    HF on CPU for the same prompt, and compares token sequences
    with the heuristics in :mod:`output_validation`.

    Phase 0: every method delegates to the module-level functions
    in :mod:`output_validation`. Phase 1: this class grows internal
    state (the :class:`hypothesis.HypothesisState`) and starts
    short-circuiting prompts based on diagnostic rules.
    """

    category: str = "LLM"
    discriminator: str = ""

    def supports(self, category: str, model_id: str) -> bool:
        return category in _TEXT_CATEGORIES

    def extract(
        self,
        captured_output: str,
        model_id: str,
    ) -> Evidence:
        """Pull the first ``==USER 0 - OUTPUT`` block out of the
        captured pytest stdout and wrap it in an :class:`Evidence`.

        Phase 0: delegates straight to
        :func:`output_validation.extract_demo_user_output`. Phase 1
        will also populate ``Evidence.payload`` with the full
        decoded string and ``Evidence.input_hint`` with the prompt
        text (today the prompt is loaded separately in
        :meth:`load_reference`)."""
        from scripts.tt_hw_planner.output_validation import (
            extract_demo_user_output,
            load_demo_first_prompt,
        )

        tt_text = extract_demo_user_output(captured_output, user_idx=0)
        if tt_text is None:
            return Evidence(
                payload=None,
                ok=False,
                reason=(
                    "could not find a '==USER 0 - OUTPUT' block in "
                    "the pytest output (the demo may not be "
                    "simple_text_demo, or the test path may not "
                    "emit the canonical output marker)"
                ),
            )
        if not tt_text.strip():
            return Evidence(
                payload="",
                ok=True,
                reason=("TT demo printed an empty output block " "(zero decoded tokens; this is itself a fail)"),
            )
        ev = Evidence(
            payload=tt_text,
            input_hint=load_demo_first_prompt(),
            ok=True,
            reason="",
        )

        m = _TT_LOGITS_PATH_RE.search(captured_output)
        if m:
            setattr(ev, "_tt_logits_path", m.group("path"))
        return ev

    def load_reference(
        self,
        evidence: Evidence,
        model_id: str,
    ) -> Any:
        """Generate the HF CPU reference for the prompt the demo
        saw. Returns a :class:`output_validation._HFRefOutput` or
        raises. Phase 0: delegates straight to
        :func:`output_validation.generate_hf_reference`. The caller
        (the dispatcher) is responsible for catching exceptions and
        treating them as a soft skip."""
        from scripts.tt_hw_planner.output_validation import (
            DEFAULT_COMPARE_TOKENS,
            generate_hf_reference,
        )

        prompt = evidence.input_hint
        if not prompt:
            raise RuntimeError(
                "TextComparator.load_reference called without an "
                "input_hint (the demo's first prompt). The Evidence "
                "must carry a prompt; check extract()."
            )

        # Always request step-0 logits from HF. The strict logit-PCC
        # gate in compare() requires them, and a missing-logits result
        # now fail-closes (UNVERIFIED) so SUCCESS can't be stamped
        # without numerical comparison. Previously this was gated on
        # whether TT-side had captured (which cascaded the env-var
        # opt-in problem in instrumentation.py:_install_logit_dump);
        # decoupling them means each side captures independently and
        # the gate is the single source of truth for whether logits
        # are usable.
        return generate_hf_reference(
            model_id,
            prompt,
            max_new_tokens=DEFAULT_COMPARE_TOKENS,
            instruct=True,
            return_logits=True,
        )

    def compare(
        self,
        evidence: Evidence,
        reference: Any,
    ) -> ValidationResult:
        """Run the existing token-overlap PCC against the HF
        reference. Phase 0: delegates to
        :func:`output_validation.compare_token_sequences`. Phase 1
        will wrap this with a wider window and the collapse
        detector."""
        from scripts.tt_hw_planner.output_validation import (
            DEFAULT_COMPARE_TOKENS,
            compare_token_sequences,
            tokenize_text_for_compare,
        )

        model_id = getattr(reference, "_model_id", "") or ""
        if not model_id:
            raise RuntimeError(
                "TextComparator.compare: reference object lacks "
                "_model_id; the dispatcher must annotate it before "
                "calling compare()"
            )
        tt_token_ids = tokenize_text_for_compare(model_id, evidence.payload or "")
        result = compare_token_sequences(
            tt_token_ids,
            reference.token_ids,
            tt_text=evidence.payload or "",
            hf_text=reference.text,
            compare_tokens=DEFAULT_COMPARE_TOKENS,
        )

        # Strict logit-PCC gate — MUST always fire. A SUCCESS verdict
        # requires both gates (token-overlap heuristic AND numerical
        # PCC >= 0.99) to fire AND pass. If logits are missing on
        # either side, or the comparison cannot be computed, that is
        # a verification gap, NOT a free pass — flip result.ok to
        # False so _maybe_escalate_pcc_fail routes the run to Path A's
        # per-component iterate loop. Fail-closed, never fail-open.
        tt_logits_path = getattr(evidence, "_tt_logits_path", None)
        hf_logits = getattr(reference, "step0_logits", None)
        if not tt_logits_path:
            result.ok = False
            result.reason = (
                f"LOGIT-PCC UNVERIFIED: TT-side step-0 logits not captured "
                f"(no `==LOGITS PATH:` marker emitted by the demo). The "
                f"strict gate requires both heuristic AND numerical "
                f"PCC>={_LOGIT_PCC_MIN:.2f}; token-overlap alone is not a "
                f"SUCCESS criterion. Token-overlap verdict (informational): "
                f"{result.reason}"
            )
        elif hf_logits is None:
            result.ok = False
            result.reason = (
                f"LOGIT-PCC UNVERIFIED: HF reference did not return step-0 "
                f"logits (return_logits path missing or HF generation failed). "
                f"The strict gate requires both heuristic AND numerical "
                f"PCC>={_LOGIT_PCC_MIN:.2f}. Token-overlap verdict "
                f"(informational): {result.reason}"
            )
        else:
            pcc = _compute_step0_logit_pcc(tt_logits_path, hf_logits)
            if pcc is None:
                result.ok = False
                result.reason = (
                    f"LOGIT-PCC UNVERIFIED: could not compute PCC from "
                    f"{tt_logits_path} (file missing, shape mismatch, or "
                    f"numpy/torch unavailable). The strict gate requires a "
                    f"numerical PCC>={_LOGIT_PCC_MIN:.2f}; treating as "
                    f"unverified rather than pass. Token-overlap verdict "
                    f"(informational): {result.reason}"
                )
            elif pcc < _LOGIT_PCC_MIN:
                result.ok = False
                result.reason = (
                    f"LOGIT-PCC FAIL: step-0 PCC={pcc:.4f} < "
                    f"{_LOGIT_PCC_MIN:.2f} (token-overlap verdict, "
                    f"informational: {result.reason})"
                )
            else:
                result.reason = f"{result.reason} [logit-PCC: {pcc:.4f}]"
        return result

    def build_repair_prompt(
        self,
        model_id: str,
        evidence: Evidence,
        result: ValidationResult,
        *,
        iter_idx: int,
        max_iters: int,
        previous_attempts: Optional[List[str]] = None,
        extra_blocks: Optional[Sequence[str]] = None,
    ) -> str:
        """Render the LLM-repair prompt for this iteration. Phase 0:
        delegates to :func:`output_validation.build_pcc_repair_prompt`
        and pulls in the context blocks (config / backend files /
        weight cache state) the legacy path already builds. Phase 1
        will replace this with the :mod:`planner` module's
        hypothesis-aware prompt."""
        from scripts.tt_hw_planner.output_validation import (
            build_pcc_repair_prompt,
            gather_backend_file_paths,
            gather_model_architecture_context,
            gather_tt_weight_cache_summary,
        )
        from .base import render_extra_blocks

        try:
            cfg = gather_model_architecture_context(model_id)
        except Exception as exc:
            cfg = f"  (HF config gather failed: {type(exc).__name__}: {exc})"
        try:
            files = gather_backend_file_paths()
        except Exception as exc:
            files = f"  (backend file gather failed: {type(exc).__name__}: {exc})"
        try:
            cache = gather_tt_weight_cache_summary(model_id)
        except Exception as exc:
            cache = f"  (cache state probe failed: {type(exc).__name__}: {exc})"

        base = build_pcc_repair_prompt(
            model_id=model_id,
            result=result,
            prompt=evidence.input_hint or "",
            iter_idx=iter_idx,
            max_iters=max_iters,
            previous_attempts=previous_attempts or [],
            model_config_block=cfg,
            backend_files_block=files,
            weight_cache_block=cache,
        )

        return base + render_extra_blocks(extra_blocks)


_singleton = TextComparator()
register_comparator(_singleton)


class _VLMTextComparator(TextComparator):
    category: str = "VLM"


register_comparator(_VLMTextComparator())


__all__ = ["TextComparator"]
