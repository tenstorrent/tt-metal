"""Stateful hypothesis tracker for the PCC-repair loop.

The legacy repair prompt was *stateless*: every iteration showed
the LLM the same static checklist of suspects ("RoPE wiring,
softcap, layer-type dispatch, …") and asked it to pick one. The
LLM had no memory of what it tried in iteration 1 versus iteration
3, so on a 4-iter run it would routinely test the same suspect
twice — and on suspects that were partially right ("ah, I touched
the RoPE freq but the demo still produces garbage; let me touch a
different RoPE thing") the lack of state caused thrashing.

This module replaces that with a running :class:`HypothesisState`:

* It holds a list of :class:`Suspect` objects, each tagged with a
  ``status`` ∈ ``{active, tested-no-improvement, ruled-out,
  fixed}``.
* :func:`HypothesisState.update_from_iteration` consumes the
  *delta* between iteration N's evidence and iteration N+1's
  evidence to update suspect statuses (e.g. "agent edited file X,
  collapse-position didn't move → demote suspects keyed on X").
* The current top-N active suspects, with confidence, get fed
  into the planner's prompt instead of the static checklist.

Why a tracker instead of just appending to ``previous_attempts``
----------------------------------------------------------------
The existing code already appends a one-line "agent edited file
X" note per iteration into ``previous_attempts``. That gives the
LLM a reading list but not a *ranking*. With the tracker, by
iteration 3 the prompt can say "Suspect #1 (confidence 0.7):
sliding-window KV cache. Already tested in iter 2 — file X
touched, but collapse moved from token 36 → 38. Now consider
suspect #2 (confidence 0.55): RoPE position wrap at 4096 …" That
ranking, drawn from actual deltas, is what closes the loop.

The tracker is *not* a probabilistic Bayesian network. It's a
small finite-state machine over a small set of suspects (~ a
dozen for LLM/VLM, slightly different sets for other categories).
Phase 1 ships with hard-coded LLM/VLM suspects; later phases will
let each :class:`Comparator` contribute category-specific ones
via :meth:`Comparator.diagnostic_rules`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


INITIAL_CONFIDENCE = 0.5


DEMOTE_ON_NO_PROGRESS = 0.3


DEMOTE_ON_REGRESSION = 0.45


PROMOTE_ON_PROGRESS = 0.25


RULED_OUT_FLOOR = 0.05


@dataclass
class Suspect:
    """A single architectural feature suspected of causing the
    PCC divergence.

    Attributes:
      name        — short, stable id (used in prompts and tests).
      description — human-readable explanation of WHY this is a
                    suspect ("Gemma-3 alternates sliding-window
                    and full attention per layer; if the dispatch
                    table is off by one, the model sees the wrong
                    K/V context after the first sliding window
                    fills up").
      files       — list of source-file globs the LLM should
                    inspect to test this hypothesis.
      kinds       — set of regime-shift kinds this suspect
                    explains (e.g. {"repetition",
                    "non_ascii_run"} for KV-cache bugs;
                    {"non_ascii_run"} for tokenizer issues).
      collapse_window — tuple ``(low, high)`` of token indices
                    where this suspect's symptoms would appear.
                    For sliding-window KV: ``(32, 96)`` (first
                    sliding boundary). For RoPE wrap: depends on
                    model max_position_embeddings. None means
                    "any position".
      confidence  — running confidence in [0,1]. Updated by
                    :meth:`HypothesisState.update_from_iteration`.
      status      — see status constants below.
      history     — per-iteration log of confidence updates and
                    notes (used in the prompt's "what's been
                    tried" section).
    """

    name: str
    description: str
    files: Tuple[str, ...] = ()
    kinds: Tuple[str, ...] = ()
    collapse_window: Optional[Tuple[int, int]] = None
    confidence: float = INITIAL_CONFIDENCE
    status: str = "active"
    history: List[str] = field(default_factory=list)

    def is_active(self) -> bool:
        return self.status in ("active", "tested-no-improvement")

    def matches(
        self,
        regime_kinds: Sequence[str],
        collapse_position: Optional[int],
    ) -> bool:
        """Does this suspect plausibly explain the current
        evidence shape?"""

        if self.kinds:
            if not any(k in regime_kinds for k in self.kinds):
                return False

        if self.collapse_window is not None and collapse_position is not None:
            lo, hi = self.collapse_window
            if not (lo <= collapse_position <= hi):
                return False
        return True


STATUS_ACTIVE = "active"
STATUS_TESTED_NO_IMPROVEMENT = "tested-no-improvement"
STATUS_RULED_OUT = "ruled-out"
STATUS_FIXED = "fixed"


def default_llm_suspects() -> List[Suspect]:
    """Return the default suspect list for LLM/VLM correctness
    failures.

    These are *priors*, not posterior beliefs. The tracker starts
    each repair loop with this set and updates the confidences as
    evidence accumulates.

    The list is drawn from the failure modes we've seen in the
    audit log (medgemma, gemma-2-27b, qwen2-vl) plus the
    architectural features that are most commonly mis-ported when
    a new model lands on tt_transformers. Order does NOT imply
    confidence — every entry starts at INITIAL_CONFIDENCE.
    """
    return [
        Suspect(
            name="sliding_window_kv_cache",
            description=(
                "Per-layer sliding-window vs full-attention dispatch. "
                "Many Gemma-3 / Qwen-2.5 / Llama-3.1 layers alternate "
                "between sliding-window (typ. 4096 tokens) and full "
                "attention. If the dispatch table is wrong, K/V "
                "lookups index out-of-window data starting at the "
                "first sliding-window boundary."
            ),
            files=(
                "models/tt_transformers/tt/attention.py",
                "models/tt_transformers/tt/decoder.py",
                "models/tt_transformers/tt/model_config.py",
                "models/tt_transformers/tt/load_checkpoints.py",
            ),
            kinds=("repetition", "non_ascii_run"),
            collapse_window=(20, 200),
        ),
        Suspect(
            name="rope_freq_dispatch",
            description=(
                "RoPE frequency selection (global vs local theta). "
                "Gemma-3 in particular uses TWO distinct rope_theta "
                "values, one per attention type. Wiring both layers "
                "to the same theta causes positional confusion "
                "after a handful of decode steps. The `rope_scaling_"
                "model_factory` in `tt/common.py` is where Gemma-3's "
                "nested {'full_attention': ..., 'sliding_attention': "
                "...} format needs special handling -- mis-handled, "
                "it collapses both attention types to one theta and "
                "produces the 'good prefix, then stuck token' "
                "signature."
            ),
            files=(
                "models/tt_transformers/tt/rope.py",
                "models/tt_transformers/tt/rope_utils.py",
                "models/tt_transformers/tt/common.py",
                "models/tt_transformers/tt/model_config.py",
            ),
            kinds=("repetition", "non_ascii_run"),
            collapse_window=(10, 120),
        ),
        Suspect(
            name="rope_position_wrap",
            description=(
                "RoPE position index wraps at max_position_embeddings "
                "instead of being clamped or extended. Manifests as "
                "the model losing coherence right around position "
                "max_position_embeddings."
            ),
            files=(
                "models/tt_transformers/tt/rope.py",
                "models/tt_transformers/tt/rope_utils.py",
            ),
            kinds=("repetition", "non_ascii_run"),
            collapse_window=None,
        ),
        Suspect(
            name="qk_norm_per_head",
            description=(
                "Per-head Q/K normalization (e.g. Gemma-2, Gemma-3, "
                "Qwen-2.5). If applied to the joint Q/K tensor "
                "instead of per-head, normalisation is wrong and the "
                "attention logits drift over decode steps. A common "
                "variant: q_norm/k_norm weights get accidentally "
                "RoPE-permuted (real/imag interleave) during weight "
                "conversion, which scrambles them into garbage because "
                "they are plain RMSNorm scale vectors, not Q/K "
                "matrices. Inspect both `tt/attention.py` AND the "
                "weight-conversion path in `tt/load_checkpoints.py` "
                "(or the per-arch multimodal variant)."
            ),
            files=(
                "models/tt_transformers/tt/attention.py",
                "models/tt_transformers/tt/load_checkpoints.py",
                "models/demos/multimodal/gemma3/tt/load_checkpoints.py",
            ),
            kinds=("repetition", "non_ascii_run"),
            collapse_window=None,
        ),
        Suspect(
            name="final_logit_softcap",
            description=(
                "Final-logit softcap (Gemma-2 / Gemma-3). If "
                "missing or applied at the wrong scale, the "
                "argmax over vocabulary becomes unstable, "
                "favouring rare tokens in the long tail (often "
                "non-ASCII)."
            ),
            files=(
                "models/tt_transformers/tt/lm_head.py",
                "models/tt_transformers/tt/model_config.py",
            ),
            kinds=("non_ascii_run",),
            collapse_window=None,
        ),
        Suspect(
            name="attention_logit_softcap",
            description=(
                "Per-layer attention-logit softcap (Gemma-2). "
                "Distinct from final_logit_softcap; applies inside "
                "each attention block before softmax."
            ),
            files=(
                "models/tt_transformers/tt/attention.py",
                "models/tt_transformers/tt/model_config.py",
            ),
            kinds=("repetition",),
            collapse_window=None,
        ),
        Suspect(
            name="layer_type_dispatch_table",
            description=(
                "config.layer_types (per-layer attention type) is not "
                "consumed by the backend, so all layers run with the "
                "default (full or sliding) regardless of what the "
                "HF config says."
            ),
            files=(
                "models/tt_transformers/tt/model_config.py",
                "models/tt_transformers/tt/decoder.py",
            ),
            kinds=("repetition", "non_ascii_run"),
            collapse_window=(20, 200),
        ),
        Suspect(
            name="tokenizer_special_tokens",
            description=(
                "BOS/EOS/PAD wiring differs between HF and TT, so the "
                "model sees a different decode-time prefix. Less "
                "likely on simple_text_demo (it shares the HF "
                "tokenizer) but possible for newer chat templates."
            ),
            files=(
                "models/tt_transformers/tt/load_checkpoints.py",
                "models/tt_transformers/demo/simple_text_demo.py",
            ),
            kinds=(),
            collapse_window=(0, 8),
        ),
        Suspect(
            name="weight_conversion_state_dict",
            description=(
                "load_checkpoints.py does the wrong tensor rename or "
                "reshape (e.g. q_proj/k_proj are still in "
                "(num_heads*head_dim,) layout but the TT code "
                "expects (num_heads, head_dim)). For multimodal "
                "models (Gemma-3 vision, Mistral 24B vision, etc.) "
                "the per-model variant lives at "
                "`models/demos/multimodal/<arch>/tt/load_checkpoints.py` "
                "AND `tt_transformers/tt/load_checkpoints.py` -- "
                "both need to be considered. Likely if the agent "
                "ALSO has to invalidate the cached weights to see "
                "their edits take effect."
            ),
            files=(
                "models/tt_transformers/tt/load_checkpoints.py",
                "models/tt_transformers/tt/model_config.py",
                "models/demos/multimodal/gemma3/tt/load_checkpoints.py",
                "models/demos/multimodal/mistral_24b/tt/load_checkpoints.py",
            ),
            kinds=("repetition", "non_ascii_run"),
            collapse_window=None,
        ),
    ]


@dataclass
class HypothesisState:
    """Running state across the PCC-repair loop's iterations.

    Construct once at loop start (via :func:`new_for_text`),
    update at each iteration with the new evidence (via
    :meth:`update_from_iteration`), and call :meth:`top_active`
    to get the suspects the planner should surface to the LLM.
    """

    suspects: List[Suspect] = field(default_factory=list)

    iteration_log: List[Dict[str, Any]] = field(default_factory=list)

    def by_name(self, name: str) -> Optional[Suspect]:
        for s in self.suspects:
            if s.name == name:
                return s
        return None

    def active(self) -> List[Suspect]:
        return [s for s in self.suspects if s.is_active()]

    def top_active(self, n: int = 3) -> List[Suspect]:
        """Return the top-N active suspects by descending
        confidence. Ties are broken in insertion order."""
        return sorted(
            self.active(),
            key=lambda s: -s.confidence,
        )[:n]

    def update_from_iteration(
        self,
        *,
        edited_files: Sequence[str],
        evidence_before: Any,
        evidence_after: Any,
        verdict_improved: bool,
        verdict_worsened: bool,
    ) -> None:
        """Update suspect confidences based on what changed
        between two iterations.

        Rules:

        * If the agent edited a file that's in suspect S's
          ``files`` AND the verdict didn't improve, demote S
          (the test was performed and didn't pan out).
        * If the agent edited a suspect S's files AND the verdict
          got worse, demote S harder (likely a wrong-direction
          fix).
        * If the agent edited a suspect S's files AND the verdict
          improved, promote S (we're getting warmer; the LLM may
          want to keep iterating in that area).
        * Move any suspect whose confidence falls below
          :data:`RULED_OUT_FLOOR` to status ``"ruled-out"``.
        """
        ef_norm = [f.replace("\\", "/").lower() for f in edited_files]

        def _touched(suspect_files: Sequence[str]) -> bool:
            for sf in suspect_files:
                sf_l = sf.replace("\\", "/").lower()

                for ef in ef_norm:
                    if sf_l in ef or ef in sf_l:
                        return True
            return False

        for suspect in self.suspects:
            if not _touched(suspect.files):
                continue
            if verdict_improved:
                delta = +PROMOTE_ON_PROGRESS
                note = "files touched + verdict IMPROVED"
            elif verdict_worsened:
                delta = -DEMOTE_ON_REGRESSION
                note = "files touched + verdict WORSENED"
            else:
                delta = -DEMOTE_ON_NO_PROGRESS
                note = "files touched + verdict unchanged"
            suspect.confidence = max(0.0, min(1.0, suspect.confidence + delta))
            suspect.history.append(f"iter+1: {note}; confidence -> {suspect.confidence:.2f}")
            if suspect.confidence <= RULED_OUT_FLOOR:
                suspect.status = STATUS_RULED_OUT
            elif delta < 0:
                suspect.status = STATUS_TESTED_NO_IMPROVEMENT

        self.iteration_log.append(
            {
                "edited_files": list(edited_files),
                "verdict_improved": verdict_improved,
                "verdict_worsened": verdict_worsened,
                "before_summary": evidence_before.summary()
                if hasattr(evidence_before, "summary")
                else str(evidence_before),
                "after_summary": evidence_after.summary()
                if hasattr(evidence_after, "summary")
                else str(evidence_after),
            }
        )

    def update_from_evidence_shape(self, evidence: Any) -> None:
        """At iteration 0 (before any agent action), narrow the
        active suspect list down to those whose ``.matches()``
        function approves of the observed evidence shape.

        This prevents the planner from showing suspects whose
        ``collapse_window`` rules them out — e.g. "RoPE position
        wrap" with window (4096, ∞) is silly to flag when the
        model collapses at token 36."""
        regime_kinds = [r.kind for r in getattr(evidence, "regime_shifts", [])]
        cpos = getattr(evidence, "collapse_position", None)
        for s in self.suspects:
            if not s.matches(regime_kinds, cpos):
                s.confidence = max(0.0, s.confidence - DEMOTE_ON_NO_PROGRESS)
                s.history.append(
                    f"iter0: doesn't match observed evidence shape "
                    f"(kinds={regime_kinds}, collapse@{cpos}); "
                    f"confidence -> {s.confidence:.2f}"
                )


def new_for_text() -> HypothesisState:
    """Construct a fresh hypothesis state seeded with the
    LLM/VLM suspect library. Used by :mod:`.text` when a repair
    loop starts."""
    return HypothesisState(suspects=default_llm_suspects())


__all__ = [
    "DEMOTE_ON_NO_PROGRESS",
    "DEMOTE_ON_REGRESSION",
    "HypothesisState",
    "INITIAL_CONFIDENCE",
    "PROMOTE_ON_PROGRESS",
    "RULED_OUT_FLOOR",
    "STATUS_ACTIVE",
    "STATUS_FIXED",
    "STATUS_RULED_OUT",
    "STATUS_TESTED_NO_IMPROVEMENT",
    "Suspect",
    "default_llm_suspects",
    "new_for_text",
]
