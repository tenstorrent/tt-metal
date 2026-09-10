"""Category-agnostic correctness contract.

Every supported model category implements one :class:`Comparator`.
The :func:`tt_hw_planner.correctness.run_gate` dispatcher picks a
comparator by category and runs the four-step gate pipeline:

::

    captured_pytest_output  ──►  extract(...)  ─►  Evidence
                                                     │
                              load_reference(...)  ◄─┘
                                       │
                                       ▼
                                  reference object
                                       │
                                       ▼
                              compare(evidence, ref)  ─►  ValidationResult
                                       │
                                       ▼
                            build_repair_prompt(...)  ─►  str

The shape of ``Evidence`` and the comparator's reference object are
intentionally category-defined (a token list for text, a mask tensor
for segmentation, a logit vector for classification, an audio
transcript for ASR, etc.). The planner only sees the
:class:`ValidationResult` and the repair prompt — it never inspects
the category-specific intermediates. That keeps the dispatcher
category-agnostic.

Why an ABC and not just functions
---------------------------------
A registered :class:`Comparator` is just a callable bundle of pure
strategies plus a label. Using an ABC instead of a tuple of
functions has three concrete benefits in this codebase:

1. ``isinstance(c, Comparator)`` lets the registry reject bad
   plugins at register-time, not at first use.
2. The ABC documents the contract once, in one place, instead of
   spreading "expected callable signature" comments across 8
   category modules.
3. Subclasses can share helpers via inheritance (``TextComparator``
   and ``VLMComparator`` will share most of ``text.py``; the
   per-category overrides are small).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence


from scripts.tt_hw_planner.output_validation import (
    ValidationResult,
)


@dataclass
class Evidence:
    """What the comparator extracted from a single pytest run.

    The dataclass is deliberately loose: ``payload`` is ``Any`` so
    each category can put what it needs (a token list, a numpy mask
    array, a list of bounding-box dicts, etc.) without the base
    class caring. The strict-typed members are the ones the planner
    inspects directly to decide whether to escalate / invalidate /
    re-prompt.

    Fields:
      payload      — category-defined raw output (tokens, mask, …).
      input_hint   — what input was given to the demo (prompt text,
                     image path, audio file). Used by
                     :meth:`Comparator.load_reference`.
      seen_at      — wall-clock timestamp the evidence was captured
                     (so the planner can compute iteration deltas).
      ok           — whether the evidence even exists. ``False``
                     means the comparator couldn't find anything to
                     compare; the dispatcher treats this as a soft
                     skip rather than a fail (the existing legacy
                     behaviour we want to preserve).
      reason       — a short human-readable explanation. Empty when
                     ``ok=True``.
    """

    payload: Any
    input_hint: Any = None
    seen_at: float = 0.0
    ok: bool = True
    reason: str = ""


class Comparator(abc.ABC):
    """Category-keyed correctness implementation.

    A concrete subclass declares the category string it serves
    (e.g. ``"LLM"``, ``"VLM"``, ``"CNN/segmentation"``) and
    implements four methods. The dispatcher calls them in this
    order:

    1. :meth:`extract` parses the demo's captured pytest stdout
       and returns an :class:`Evidence` instance. Pure — no
       network, no GPU.
    2. :meth:`load_reference` produces the HF reference for the
       same input. This is the impure step; it may load a HF model
       on CPU, run a few forward passes, etc. It returns whatever
       :meth:`compare` expects on the reference side.
    3. :meth:`compare` runs the actual TT-vs-HF comparison and
       returns a :class:`ValidationResult`. Pure — no I/O.
    4. :meth:`build_repair_prompt` generates the LLM-repair prompt
       used when the gate fails and ``--auto`` is on. Pure.

    Subclasses MAY also override:

    * :meth:`supports` — by default returns ``True`` for any
      category string the subclass declares; override to narrow
      (e.g. only enable for a given model_type).
    * :meth:`diagnostic_rules` — return a list of pattern rules
      the Phase-1 ``diagnose`` module can use to update hypotheses.
      Default is empty list.
    """

    category: str = ""

    discriminator: str = ""

    def label(self) -> str:
        """Human-readable label used in log lines."""
        if self.discriminator:
            return f"{self.category}/{self.discriminator}"
        return self.category

    def supports(self, category: str, model_id: str) -> bool:
        """Return ``True`` iff this comparator handles ``(category,
        model_id)``. Default: exact match on ``self.category``.
        Override to narrow by model_id or model_type."""
        return category == self.category

    @abc.abstractmethod
    def extract(
        self,
        captured_output: str,
        model_id: str,
    ) -> Evidence:
        """Parse the demo's captured pytest stdout and return an
        :class:`Evidence`.

        Implementations must NEVER raise on malformed input; return
        ``Evidence(payload=None, ok=False, reason="...")`` instead.
        The dispatcher treats ``ok=False`` as a soft skip so a
        broken comparator can't tank a successful bring-up.
        """

    @abc.abstractmethod
    def load_reference(
        self,
        evidence: Evidence,
        model_id: str,
    ) -> Any:
        """Generate the HF reference for the same input the demo
        saw. Impure; may load a HF model.

        Implementations should respect ``HF_TRUST_REMOTE_CODE`` from
        the environment (passed through by :func:`run_gate`)."""

    @abc.abstractmethod
    def compare(
        self,
        evidence: Evidence,
        reference: Any,
    ) -> ValidationResult:
        """Pure TT-vs-HF comparison. Returns a
        :class:`ValidationResult` describing pass/fail and the
        diagnostic signal."""

    @abc.abstractmethod
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
        """Render the LLM-repair prompt used when the gate fails."""

    def diagnostic_rules(self) -> Sequence[Any]:
        """Phase-1 hook. Return a sequence of pattern rules that
        :mod:`.diagnose` will use to update the hypothesis state
        across iterations. Default: empty.

        The shape of the rule objects is defined by :mod:`.diagnose`;
        Phase-0 comparators leave this empty and gain no behavior."""
        return ()


def render_extra_blocks(
    extra_blocks: Optional[Sequence[str]],
    *,
    heading: str = "ADDITIONAL CONTEXT (model config / backend / divergence / convergence)",
) -> str:
    """Universal renderer for ``extra_blocks`` to append to a repair
    prompt.

    The per-component iterate loop
    (``auto_iterate._run_auto_iterate_loop``) passes
    ``extra_blocks`` containing (historically also called by
    ``_pcc_repair_loop`` in the deleted ``_cli_helpers/pcc_repair.py``):

      * the HF ``config.json`` model-architecture summary,
      * the list of TT backend files for this category,
      * the TT-native weight-cache summary,
      * (when the agentic executor ran) an empirical per-module
        divergence table, and
      * (when the agentic executor ran) a convergence trajectory.

    Every comparator's :meth:`Comparator.build_repair_prompt`
    should append the output of this function to its returned prompt
    so the LLM gets the same caliber of context regardless of
    category. Filters out empty/None blocks; returns the empty string
    if nothing useful was passed.
    """
    if not extra_blocks:
        return ""
    parts = [b.strip() for b in extra_blocks if b and b.strip()]
    if not parts:
        return ""
    body = "\n\n".join(parts)
    return f"\n\n  {heading}\n  {'-' * len(heading)}\n{body}\n"


__all__ = ["Comparator", "Evidence", "ValidationResult", "render_extra_blocks"]
