"""LLM verify pass for the end-to-end chained forward (Item 2).

When the strict end-to-end PCC gate fails after per-component PCC has
already passed, the failure is somewhere in the chaining — either the
top-level forward's data flow (template / synthesized) doesn't match
HF's forward, or a glue component is missing/wrong. This module asks
the LLM to perform a static code review comparing the two and emit a
structured JSON verdict naming what's missing.

Distinct from the chain-divergence diagnostic (Item 1):

  * Item 1 (chain divergence) — runtime: compares HF vs TT activation
    statistics per module to localize WHERE numerical drift starts.
  * Item 2 (LLM verify) — static: reads HF's forward() and the TT
    chained forward source, asks the LLM what's structurally missing
    (args, branches, intermediate ops, modules).

Both feed into the LLM repair prompt — Item 1 says "module X drifted",
Item 2 says "the chain is missing op Y between module X and module Z."

Design contract
---------------
Pure-ish: the prompt builder and verdict parser are pure functions;
the orchestrator (``run_llm_verify_pass``) is the one impure step
(invokes ``_invoke_agent`` subprocess and reads back the JSON
deliverable). Both sides are tested independently:

  * Prompt builder + verdict parser → unit-tested with no LLM.
  * Orchestrator → unit-tested with ``_invoke_agent`` mocked and a
    pre-populated verdict file.

Best-effort: every failure mode (missing demo source, malformed
verdict JSON, agent timeout, schema mismatch) degrades to ``None``
rather than raising. Caller treats ``None`` as "no verify signal
this iter" and continues with whatever fallback signal it has.

The LLM is NEVER asked to modify source files here — verify is a
read-only static analysis. The repair step happens elsewhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


# ─── Verdict schema ──────────────────────────────────────────────────


@dataclass
class LLMVerifyVerdict:
    """Result of one LLM verify pass.

    The verdict is intentionally coarse (PASS / FAIL) because the
    downstream router only needs to decide "escalate to synthesis or
    not." The diagnostic dict carries the structured details an LLM
    repair prompt would re-render.
    """

    verdict: str  # "PASS" or "FAIL"
    diagnostic: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""  # source bytes of the verdict file (for debugging)

    @property
    def ok(self) -> bool:
        """True iff verdict == 'PASS' — the chained forward looks
        correct to the LLM and no synthesis pass is needed."""
        return self.verdict.upper() == "PASS"


# Strict whitelist for the verdict value. The parser rejects anything
# outside this set so an LLM emitting "MAYBE" / "NEEDS_REVIEW" / etc.
# can't smuggle a value the downstream router has no rule for.
_VALID_VERDICTS = frozenset({"PASS", "FAIL"})


# ─── Prompt builder ──────────────────────────────────────────────────


def build_verify_prompt(
    *,
    model_id: str,
    hf_forward_src: str,
    tt_chained_src: Optional[str],
    drift_summary: str,
    verdict_path: Path,
) -> str:
    """Render the LLM-verify prompt.

    Pure function — no I/O. Tested independently of the orchestrator.

    Parameters
    ----------
    model_id
        HuggingFace model id (printed in the prompt for context).
    hf_forward_src
        Source of HF's top-level ``forward()`` method (extracted via
        ``inspect.getsource``). May be empty if extraction fails;
        caller is responsible for resolving.
    tt_chained_src
        Source of the TT chained forward — typically the demo .py
        that wires the graduated TTNN components together. ``None``
        means "no TT chained forward exists yet" — the LLM should
        respond FAIL with ``"missing_modules": ["entire forward"]``.
    drift_summary
        One-paragraph summary from Item 1's chain-divergence
        diagnostic, e.g. "first divergence at vision_encoder.layer_0
        with mean drift 0.5". Empty string if no diagnostic ran.
    verdict_path
        Absolute path where the LLM must write its JSON verdict.

    Returns
    -------
    A prompt string ready to feed to ``_invoke_agent``.
    """
    schema_block = (
        "{\n"
        '  "verdict": "PASS" | "FAIL",\n'
        '  "diagnostic": {\n'
        '    "missing_args":             ["arg_name", ...],\n'
        '    "missing_branches":         ["if X then Y", ...],\n'
        '    "missing_intermediate_ops": ["op description", ...],\n'
        '    "missing_modules":          ["module path", ...],\n'
        '    "summary":                  "one paragraph human-readable"\n'
        "  }\n"
        "}"
    )
    tt_block = tt_chained_src if tt_chained_src else "(no TT chained forward exists yet — needs synthesis)"
    drift_block = drift_summary or "(no chain-divergence diagnostic available)"
    return f"""You are a code-review-only verifier for the tt_hw_planner bring-up tool.

CONTEXT
-------
Model: {model_id}

Per-component PCC tests have already passed (every TTNN submodule is
individually numerically faithful to its torch reference at PCC >= 0.99).
But the end-to-end CHAINED forward — where the output of one TTNN
module feeds the next — has a numerical or structural problem.

YOUR TASK
---------
Compare the HF reference forward() source to the TT chained forward
source, and write a structured JSON verdict naming what's missing or
wrong. DO NOT modify any source files. ONLY write the verdict JSON
at the specified path.

The downstream router uses your verdict to decide:
  • verdict == "PASS"  → trust the existing chain, skip synthesis
  • verdict == "FAIL"  → escalate to LLM synthesis with your
                         diagnostic as input

Be specific and conservative — answer FAIL if you're not sure the
chain is correct. A false PASS wastes user trust; a false FAIL only
wastes one synthesis iter.

REFERENCE: HF forward() source
------------------------------
{hf_forward_src}

CANDIDATE: TT chained forward source
------------------------------------
{tt_block}

CHAIN-DIVERGENCE DIAGNOSTIC (from runtime activation comparison)
---------------------------------------------------------------
{drift_block}

OUTPUT
------
Write this exact JSON shape to {verdict_path}:

{schema_block}

Fields:
  • missing_args            — kwargs HF.forward accepts that the TT chain ignores
  • missing_branches        — conditional paths in HF that the TT chain skips
  • missing_intermediate_ops — reshapes / permutes / projections HF does that
                                the TT chain doesn't (or does wrong)
  • missing_modules         — nn.Module instances HF calls that the TT chain
                                doesn't replace and doesn't fall back on
  • summary                 — one paragraph naming the most likely cause

Lists may be empty when nothing in that category is missing. The
"summary" field is required; the others may be empty arrays.

DO NOT make any edits to source files. Write ONLY the verdict JSON.
"""


# ─── Verdict parser ──────────────────────────────────────────────────


def parse_verify_verdict(verdict_path: Path) -> Optional[LLMVerifyVerdict]:
    """Read the LLM's verdict file and validate the schema.

    Pure with respect to non-file inputs. Returns ``None`` on every
    failure mode (file missing, malformed JSON, missing required
    fields, unrecognized verdict value). Never raises — the caller
    treats None as "no verify signal".

    The strict validation here is intentional: a bad LLM response
    shouldn't propagate garbage into the downstream router. Better
    to silently degrade and let the run continue without the verify
    signal than to act on noise.
    """
    if not verdict_path.is_file():
        return None
    try:
        raw_text = verdict_path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        blob = json.loads(raw_text)
    except Exception:
        return None
    if not isinstance(blob, dict):
        return None
    verdict = blob.get("verdict")
    if not isinstance(verdict, str):
        return None
    verdict_upper = verdict.strip().upper()
    if verdict_upper not in _VALID_VERDICTS:
        return None
    diagnostic = blob.get("diagnostic", {})
    if not isinstance(diagnostic, dict):
        diagnostic = {}
    return LLMVerifyVerdict(
        verdict=verdict_upper,
        diagnostic=diagnostic,
        raw_text=raw_text,
    )


# ─── HF forward source resolver ──────────────────────────────────────


def resolve_hf_forward_source(model_id: str) -> Optional[str]:
    """Best-effort extract HF's top-level ``forward()`` source for
    inclusion in the verify prompt.

    Loads the model class via ``transformers.AutoModel*`` cascade,
    grabs ``inspect.getsource(model_cls.forward)``, returns the
    source string. Returns ``None`` on any failure — caller falls
    through to passing empty HF source (the LLM verify will still
    attempt the analysis, just with less context).

    Pure-ish: triggers HF library imports + model class loading
    (NOT weights — just the class object via ``AutoConfig`` +
    class registry). No network if HF caches the config.
    """
    try:
        import inspect

        from transformers import AutoConfig
    except Exception:
        return None
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        return None
    # Walk the transformers model-class registry for this model_type.
    # We don't instantiate — just resolve the class object so we can
    # inspect its forward.
    try:
        import transformers

        cls = None
        for loader_name in (
            "AutoModelForCausalLM",
            "AutoModelForSpeechSeq2Seq",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForImageClassification",
            "AutoModel",
        ):
            loader = getattr(transformers, loader_name, None)
            if loader is None:
                continue
            # The _model_mapping (or similar) carries config-class -> model-class.
            mapping = getattr(loader, "_model_mapping", None)
            if mapping is None:
                continue
            try:
                cls = mapping.get(type(cfg))
            except Exception:
                cls = None
            if cls is not None:
                break
        if cls is None:
            return None
        return inspect.getsource(cls.forward)
    except Exception:
        return None


# ─── TT chained forward source resolver ──────────────────────────────


def resolve_tt_chained_source(demo_dir: Path) -> Optional[str]:
    """Best-effort extract the TT chained forward source.

    For LLMs this is typically ``tt_transformers/demo/simple_text_demo.py``,
    but the per-model bring-up emits a demo.py under ``demo_dir/demo.py``
    (via :mod:`bringup_loop._MIXED_EXEC_DEMO_TEMPLATE`). We look at the
    per-model demo.py FIRST — that's the source the model actually runs
    — and fall back to None if it doesn't exist.

    Returns the source string, or ``None`` if no demo.py exists yet
    (e.g. brand-new model with no template, or scaffold step hasn't
    fired). ``None`` is a signal to the verify prompt that "synthesis
    is needed from scratch."
    """
    if demo_dir is None or not demo_dir.is_dir():
        return None
    demo_py = demo_dir / "demo" / "demo.py"
    if not demo_py.is_file():
        demo_py = demo_dir / "demo.py"
    if not demo_py.is_file():
        return None
    try:
        return demo_py.read_text(encoding="utf-8")
    except Exception:
        return None


# ─── Orchestrator ───────────────────────────────────────────────────


def run_llm_verify_pass(
    *,
    model_id: str,
    demo_dir: Path,
    agent_bin: str = "claude",
    agent_model: str = "haiku",
    timeout_s: int = 300,
    drift_summary: str = "",
    hf_forward_src: Optional[str] = None,
    tt_chained_src: Optional[str] = None,
) -> Optional[LLMVerifyVerdict]:
    """One LLM verify pass: ask the LLM to compare HF.forward() to the
    TT chained forward and return its structured verdict.

    Orchestrates: resolve HF source → resolve TT source → build prompt
    → invoke agent → parse verdict file → return.

    Parameters
    ----------
    model_id
        HuggingFace model id (used in prompt + HF source resolution).
    demo_dir
        Model's demo directory. Verdict file lands at
        ``demo_dir/_verify/verdict.json`` — under a subdir so it
        doesn't collide with synthesis artifacts.
    agent_bin
        Path/name of the LLM CLI to invoke (default ``claude``).
    agent_model
        Model tier for the call. Defaults to ``haiku`` because verify
        is cheap and bounded — no need for sonnet/opus.
    timeout_s
        Hard wall-clock budget for the LLM call.
    drift_summary
        Optional one-paragraph summary from Item 1's diagnostic.
    hf_forward_src / tt_chained_src
        Pre-resolved source if the caller already has it. When
        ``None``, the orchestrator resolves them via the helpers.

    Returns
    -------
    Parsed :class:`LLMVerifyVerdict` on success. ``None`` on every
    failure mode — caller treats as "no verify signal this iter."
    """
    from .agent import _invoke_agent

    if hf_forward_src is None:
        hf_forward_src = resolve_hf_forward_source(model_id) or ""
    if tt_chained_src is None:
        tt_chained_src = resolve_tt_chained_source(demo_dir)

    verify_dir = demo_dir / "_verify"
    try:
        verify_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    verdict_path = verify_dir / "verdict.json"
    try:
        if verdict_path.exists():
            verdict_path.unlink()
    except Exception:
        pass

    prompt = build_verify_prompt(
        model_id=model_id,
        hf_forward_src=hf_forward_src,
        tt_chained_src=tt_chained_src,
        drift_summary=drift_summary,
        verdict_path=verdict_path,
    )

    try:
        rc = _invoke_agent(
            prompt,
            provider="claude",
            agent_bin=agent_bin,
            cwd=demo_dir,
            model=agent_model,
            timeout_s=timeout_s,
            iter_tag="verify",
            expected_deliverable_files=[verdict_path],
        )
    except Exception:
        return None
    if rc != 0:
        return None

    return parse_verify_verdict(verdict_path)


__all__ = [
    "LLMVerifyVerdict",
    "build_verify_prompt",
    "parse_verify_verdict",
    "resolve_hf_forward_source",
    "resolve_tt_chained_source",
    "run_llm_verify_pass",
]
