"""Generic agentic primitives for the bring-up loop.

This package implements the generic discovery primitives (G1-G9) that the
PCC-repair / runtime-repair loops use to identify and fix bring-up bugs
WITHOUT category-specific hardcoded knowledge. The primitives are:

  G1  Universal module-tree dual-probe (HF vs TT, per-submodule activation
      stats) -- :mod:`.probe` (HF side) + :mod:`.tt_probe` (TT side).
  G2  Source-file resolver (`inspect.getsourcefile` on whatever class
      diverges) -- :mod:`.resolve`.
  G3  Empirical suspect synthesis (the first-diverging pair IS the
      suspect; no hardcoded prior lists) -- :mod:`.diverge`.
  G4  Mechanical action library (cache invalidate, env toggle, edit
      revert, dtype switch -- all category-agnostic toggles tried
      BEFORE invoking the LLM) -- :mod:`.actions`.
  G5  Plan executor (deterministic order: probe -> learnings -> mech ->
      LLM -> bisect -> persist) -- :mod:`.executor`.
  G6  LLM context maximizer (probe table + diverging source files;
      same prompt shape for every category) -- :mod:`.executor` helper.
  G7  Persistent learnings keyed by (arch_signature,
      first_diverging_qualified_name) -- :mod:`.learnings`.
  G8  Convergence + budget bail (linear fit on mismatch_ratio history)
      -- :mod:`.convergence`.
  G9  Provider / device recovery -- reused from the existing
      :mod:`scripts.tt_hw_planner.cli` infrastructure
      (``_invoke_agent`` failover, ``_run_tt_smi_reset``).

Why a sub-package
-----------------
The legacy ``_pcc_repair_loop`` and the legacy ``correctness/`` package
both have category-specific knowledge baked in (hardcoded suspects,
hardcoded decoder paths, hardcoded prompt templates). Rewriting those
in place would risk regressions; ``agentic/`` is a parallel,
generic-only implementation that the legacy loop opts into via
``--auto-engine=agentic``. When the agentic path proves out across
several models the legacy code becomes dead and gets removed; until
then both coexist.

Categories supported
--------------------
Every primitive is category-agnostic. The same code path runs for
LLM, VLM, STT, NLP, CNN-classify, CNN-segment, CNN-detect, embed,
diffusion, TTS, and video. The only category-keyed step left in the
loop is the FINAL-OUTPUT comparator (text vs mask vs class vs ...) --
that lives in :mod:`.correctness` and is untouched here.
"""

from __future__ import annotations


from .executor import AgenticIterationResult, run_iteration

__all__ = ["AgenticIterationResult", "run_iteration"]
