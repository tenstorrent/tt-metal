"""Critic sub-agent: targeted precision-debugging for PCC plateau failures.

When the auto-iterate loop hits a PCC plateau — the LLM produces
byte-identical (or functionally identical) code across iterations and
the PCC value doesn't move — generic retry doesn't help. The LLM needs
a *directed* signal: "change line N from X to Y because Z".

The critic is a thin one-shot LLM call that takes the failed code +
PCC value + recipes-applied and returns one structured recommendation.
The loop injects that recommendation into the next iteration's prompt,
breaking the plateau.

Design:
  - Cheap: small input, small output, no filesystem tools.
  - Cached: keyed by (component_name, code_sha1, pcc_value). The same
    failing code → same diagnosis without re-spending tokens.
  - Robust: malformed critic output never crashes the loop; absent
    diagnosis just means the next iter doesn't get the extra hint.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


_CRITIC_SYSTEM_PROMPT = """\
You are a TTNN precision critic. Your job is to diagnose WHY a generated \
ttnn implementation is producing a specific PCC value below 0.99, and \
recommend the SPECIFIC change most likely to close the gap.

You will receive:
  - The current ttnn implementation (Python source).
  - The PCC value achieved (e.g. 0.9877).
  - The captured input shape + dtype.
  - The catalog recipes that were already applied.
  - (Optionally) the previous critic diagnosis that didn't help.

Output ONE structured diagnosis in the exact YAML-like format below. Do \
NOT add extra commentary, code blocks, or explanations outside the \
fields. Keep each field to one or two short sentences.

  ROOT_CAUSE: <one sentence — which math step loses precision>
  SPECIFIC_CHANGE: <imperative — what code to change and how, with file/line if you can pinpoint it>
  WHY: <one sentence — why this change moves PCC>
  ALTERNATIVE: <a SECOND change to try if SPECIFIC_CHANGE doesn't help; one line>
  CONFIDENCE: high | medium | low

Common precision pitfalls to consider (not exhaustive):
  - bf16 accumulation in reductions (should be fp32 — set \
fp32_dest_acc_en=True in the kernel config).
  - Missing or wrong math fidelity (HiFi4 vs default LoFi).
  - eps applied in the wrong position relative to rsqrt.
  - Tile padding leaking into reductions (the "padding-poisoned" pattern: \
ttnn.mean / ttnn.var over sub-tile dims include padded zeros).
  - Quantization losses across unnecessary layout conversions.
  - Wrong broadcast direction for affine multiply/add.
  - Order of operations differing from the torch reference (e.g. compute \
variance before vs after centering).

Focus on the math, not style. Aim for ONE small change that, if applied, \
would move PCC. Do NOT propose architectural rewrites or recipe \
replacements — the catalog recipe is fixed; you are tuning within it.
"""


@dataclass
class CriticDiagnosis:
    """One critic output, parsed into structured fields."""

    component: str
    pcc: float
    root_cause: str = ""
    specific_change: str = ""
    why: str = ""
    alternative: str = ""
    confidence: str = "low"
    raw_response: str = ""
    error: str = ""

    @property
    def is_actionable(self) -> bool:
        return bool(self.specific_change) and not self.error

    def to_prompt_block(self) -> str:
        """Render as a markdown block suitable for splicing into the
        next iteration's prompt. Empty string if nothing actionable."""
        if not self.is_actionable:
            return ""
        return (
            "\n"
            "PRECISION CRITIC DIAGNOSIS — apply BEFORE iterating\n"
            "----------------------------------------------------\n"
            f"Previous iter scored PCC={self.pcc:.4f} (target 0.99). The "
            f"critic analysis flags ONE specific change as the most likely "
            f"culprit for the residual precision gap. Apply it (or the "
            f"ALTERNATIVE if you have a strong reason not to) BEFORE making "
            f"other edits.\n\n"
            f"ROOT_CAUSE: {self.root_cause}\n"
            f"SPECIFIC_CHANGE: {self.specific_change}\n"
            f"WHY: {self.why}\n"
            f"ALTERNATIVE: {self.alternative}\n"
            f"CONFIDENCE: {self.confidence}\n"
            "\n"
        )


def _hash_code(code: str) -> str:
    return hashlib.sha1(code.encode("utf-8", errors="replace")).hexdigest()[:12]


_FIELD_RE = re.compile(r"^\s*([A-Z_]+):\s*(.+?)\s*$", re.MULTILINE)


def _parse_critic_response(text: str) -> Dict[str, str]:
    """Pull ``KEY: value`` lines out of the critic's response. Multi-line
    values that wrap to the next non-key line are joined."""
    text = text.strip()
    if text.startswith("```"):
        # Strip code fences if the LLM ignored the no-fence instruction.
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]

    fields: Dict[str, str] = {}
    matches = list(_FIELD_RE.finditer(text))
    for i, m in enumerate(matches):
        key = m.group(1).upper()
        # Value runs from this match's end up to the next match's start.
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = text[m.end() : end].strip()
        # Re-prepend the inline value captured in group 2.
        inline = m.group(2).strip()
        if value:
            value = f"{inline} {value}".strip()
        else:
            value = inline
        fields[key] = value
    return fields


def _build_critic_user_prompt(
    *,
    component: str,
    code: str,
    pcc: float,
    input_shape: Optional[List[int]],
    input_dtype: Optional[str],
    recipes_applied: List[str],
    previous_diagnosis: Optional[str],
) -> str:
    shape_s = "unknown" if input_shape is None else str(input_shape)
    dtype_s = input_dtype or "unknown"
    recipes_s = ", ".join(recipes_applied) if recipes_applied else "(none reported)"

    prev_block = ""
    if previous_diagnosis:
        prev_block = (
            "\nPREVIOUS CRITIC DIAGNOSIS (did NOT close the gap — propose something different):\n"
            "------------------------------------------------------------\n"
            f"{previous_diagnosis}\n"
        )

    return (
        f"COMPONENT: {component}\n"
        f"CURRENT_PCC: {pcc}\n"
        f"INPUT_SHAPE: {shape_s}\n"
        f"INPUT_DTYPE: {dtype_s}\n"
        f"CATALOG_RECIPES_APPLIED: {recipes_s}\n"
        f"\n"
        f"CURRENT TTNN IMPLEMENTATION (the file that scored PCC={pcc:.4f}):\n"
        f"------------------------------------------------------------\n"
        f"{code}\n"
        f"------------------------------------------------------------\n"
        f"{prev_block}\n"
        f"Diagnose the precision loss. Output ONLY the structured "
        f"ROOT_CAUSE/SPECIFIC_CHANGE/WHY/ALTERNATIVE/CONFIDENCE fields. "
        f"Do not write code blocks or commentary outside those fields."
    )


# Module-level cache: (component, code_hash, pcc_rounded) -> CriticDiagnosis
_CACHE: Dict[tuple, CriticDiagnosis] = {}


def _cache_key(component: str, code: str, pcc: float) -> tuple:
    return (component, _hash_code(code), round(pcc, 4))


def invoke_critic(
    *,
    component: str,
    code: str,
    pcc: float,
    input_shape: Optional[List[int]] = None,
    input_dtype: Optional[str] = None,
    recipes_applied: Optional[List[str]] = None,
    previous_diagnosis: Optional[str] = None,
    model: str = "sonnet",
    agent_bin: str = "claude",
    timeout_s: int = 180,
    _call_llm=None,
) -> CriticDiagnosis:
    """Run the critic. Returns a (possibly empty/error-tagged)
    CriticDiagnosis. NEVER raises — the loop must continue even if the
    critic call fails entirely.

    ``_call_llm`` is an injection seam for tests: pass a function with the
    same signature as ``invoke_llm_cli_one_shot`` to mock the LLM call.
    """
    recipes_applied = recipes_applied or []

    key = _cache_key(component, code, pcc)
    if key in _CACHE:
        return _CACHE[key]

    user_prompt = _build_critic_user_prompt(
        component=component,
        code=code[:8000],  # cap code length to keep critic prompt small
        pcc=pcc,
        input_shape=input_shape,
        input_dtype=input_dtype,
        recipes_applied=recipes_applied,
        previous_diagnosis=previous_diagnosis,
    )
    full_prompt = _CRITIC_SYSTEM_PROMPT + "\n\n---\n\n" + user_prompt

    if _call_llm is None:
        from ..llm_synth import invoke_llm_cli_one_shot

        _call_llm = invoke_llm_cli_one_shot

    diagnosis = CriticDiagnosis(component=component, pcc=pcc)
    try:
        response = _call_llm(
            prompt=full_prompt,
            agent_bin=agent_bin,
            model=model,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        diagnosis.error = f"{type(exc).__name__}: {exc}"
        _CACHE[key] = diagnosis
        return diagnosis

    diagnosis.raw_response = response
    fields = _parse_critic_response(response)
    diagnosis.root_cause = fields.get("ROOT_CAUSE", "")
    diagnosis.specific_change = fields.get("SPECIFIC_CHANGE", "")
    diagnosis.why = fields.get("WHY", "")
    diagnosis.alternative = fields.get("ALTERNATIVE", "")
    confidence = fields.get("CONFIDENCE", "low").lower()
    diagnosis.confidence = confidence if confidence in ("high", "medium", "low") else "low"

    _CACHE[key] = diagnosis
    return diagnosis


def persist_diagnosis(diagnosis: CriticDiagnosis, demo_dir: Path) -> Optional[Path]:
    """Write the diagnosis under ``demo_dir/_critic/<component>.json`` so
    future runs can read it. Best-effort; returns the path or None."""
    out_dir = demo_dir / "_critic"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{diagnosis.component}.json"
        out_path.write_text(json.dumps(asdict(diagnosis), indent=2))
        return out_path
    except Exception:
        return None


def load_diagnosis(component: str, demo_dir: Path) -> Optional[CriticDiagnosis]:
    """Load a previously-persisted diagnosis. Returns None if absent or
    malformed."""
    p = demo_dir / "_critic" / f"{component}.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
        return CriticDiagnosis(
            component=data.get("component", component),
            pcc=float(data.get("pcc", 0.0) or 0.0),
            root_cause=data.get("root_cause", "") or "",
            specific_change=data.get("specific_change", "") or "",
            why=data.get("why", "") or "",
            alternative=data.get("alternative", "") or "",
            confidence=data.get("confidence", "low") or "low",
            raw_response=data.get("raw_response", "") or "",
            error=data.get("error", "") or "",
        )
    except Exception:
        return None


def clear_cache() -> None:
    """Drop the in-process cache. Tests use this between cases; the loop
    doesn't need it (caching across iters is the point)."""
    _CACHE.clear()
