"""Late-discovery classifier (Item 5).

When :func:`run_e2e_synthesis_loop` discovers a piece the chained
forward needs that wasn't in the decomposition output, this module
routes the piece to one of three handlers:

  * **Case A** — pure TTNN-expressible op (reshape, permute, slice,
    view, transpose). Action: LLM writes it inline in the synthesized
    forward as a single ttnn call. No per-component round-trip.
  * **Case B** — CPU-acceptable glue (small op, dict access, indexing,
    control flow). Action: drop on CPU inside the forward with a
    `.cpu() → op → .to(device)` bridge. Tagged ``GLUE_CPU`` in the
    manifest.
  * **Case C** — real graduate-able submodule the decomposer missed.
    Action: kick back to :mod:`late_graduate.run_late_graduate` for
    per-component iterate + LATE_GRADUATE status.

The classifier itself is an LLM call: given the missing-piece
description plus the HF source and component list, pick A / B / C
and return a structured verdict. The router (a separate function
caller) acts on that verdict.

Distinct from the LLM verify pass (Item 2):

  * Item 2 — "is the existing chain correct?" — outputs PASS / FAIL.
  * Item 5 — "given this missing piece, how should we handle it?" —
    outputs A / B / C + the data each handler needs.

Best-effort: malformed verdicts degrade to None so the caller can
fall back to a conservative default (typically Case B, the
CPU-glue route).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


_VALID_CASES = frozenset({"A", "B", "C"})


# ─── Classification schema ──────────────────────────────────────────


@dataclass
class MissingPieceClassification:
    """One verdict from the late-discovery classifier.

    Carries the routing decision plus the per-case payload the router
    needs. Fields:

      * ``case``         — "A" / "B" / "C"
      * ``piece_kind``   — informational, one of "tensor_op",
                            "control_flow", "submodule", or "other"
      * ``description``  — short human-readable name for the piece
      * ``ttnn_call``    — for Case A: the inline ttnn op + args
      * ``cpu_module``   — for Case B: the dotted HF module path that
                            should run on CPU
      * ``submodule_spec`` — for Case C: the
                            :class:`LateGraduateComponentSpec`-compatible
                            dict the router passes to
                            :func:`late_graduate.run_late_graduate`
      * ``notes``        — free-form LLM justification
    """

    case: str
    piece_kind: str = "other"
    description: str = ""
    ttnn_call: Optional[Dict[str, Any]] = None
    cpu_module: Optional[str] = None
    submodule_spec: Optional[Dict[str, Any]] = None
    notes: str = ""

    @property
    def is_case_a(self) -> bool:
        return self.case == "A"

    @property
    def is_case_b(self) -> bool:
        return self.case == "B"

    @property
    def is_case_c(self) -> bool:
        return self.case == "C"


# ─── Prompt builder ──────────────────────────────────────────────────


def build_classify_prompt(
    *,
    model_id: str,
    missing_piece_description: str,
    hf_forward_src: str,
    graduated_components: List[Dict[str, Any]],
    verdict_path: Path,
) -> str:
    """Render the classification prompt.

    Pure function — no I/O. The LLM gets the missing-piece description
    (from synthesis's diagnostic), the HF forward source (so it knows
    the surrounding context), and the list of already-graduated
    components (so it doesn't suggest Case C for something we already
    have).
    """
    comps_block = (
        "\n".join(f"  • {c.get('name', '?')} ({c.get('class_name', '?')})" for c in graduated_components) or "  (none)"
    )
    return f"""You are a routing classifier for the tt_hw_planner bring-up tool.

CONTEXT
-------
Model: {model_id}

The e2e synthesis loop is trying to build a chained forward for this
model. It identified a piece it needs that wasn't in the original
decomposition. Your job: decide how to handle it.

THE MISSING PIECE
-----------------
{missing_piece_description}

HF REFERENCE forward() (for surrounding context)
-------------------------------------------------
{hf_forward_src}

ALREADY-GRADUATED TTNN COMPONENTS (don't suggest Case C for these)
------------------------------------------------------------------
{comps_block}

CLASSIFY INTO ONE OF THREE CASES
--------------------------------
  Case A — Pure TTNN-expressible op
    The piece is a single tensor op (reshape, permute, slice, view,
    transpose, contiguous, etc.) expressible as one ttnn call.
    Action: LLM writes inline in the synthesized forward. No
    per-component pass.

  Case B — CPU-acceptable glue
    Small / non-critical-path piece (dict access, indexing,
    conditional branching, list construction). CPU latency hop is
    fine. Action: drop on CPU inside the forward.

  Case C — Real graduate-able submodule
    A genuine nn.Module the decomposer missed (e.g. a small conv,
    norm, projection). It's worth a full per-component PCC pass.
    Action: kick back to the per-component iterate loop.

DECISION RULES
--------------
  • Default to Case A if it's a single ttnn op.
  • Default to Case B if Case A doesn't apply AND it's small /
    non-critical AND a CPU hop wouldn't measurably hurt perf.
  • Case C only when the piece IS a real nn.Module that:
       - Has tunable weights / state, OR
       - Is on a hot/critical path (called once per token / layer), OR
       - Would meaningfully benefit from a TTNN port.

A false Case C wastes a synthesis iter; a false Case B leaves perf
on the table. Lean toward A → B → C in cost order.

OUTPUT
------
Write this exact JSON shape to {verdict_path}:

{{
  "case": "A" | "B" | "C",
  "piece_kind": "tensor_op" | "control_flow" | "submodule" | "other",
  "description": "short name for the piece",
  "ttnn_call":      {{ "op": "ttnn.reshape", "args": [...] }}    // Case A only
  "cpu_module":     "vision_encoder.fpn_extract"                 // Case B only
  "submodule_spec": {{ "name": "fpn", "hf_reference": "...",
                       "class_name": "FPN" }}                     // Case C only
  "notes":          "one sentence justifying the case"
}}

Include ONLY the per-case field for the case you picked. Leave the
others absent or null. The "notes" field is always required.

DO NOT make any edits to source files. Write ONLY the verdict JSON.
"""


# ─── Verdict parser ──────────────────────────────────────────────────


def parse_classify_verdict(verdict_path: Path) -> Optional[MissingPieceClassification]:
    """Read the LLM's classification verdict and validate the schema.

    Returns None on every malformed-input path (file missing, bad JSON,
    missing case, invalid case value). Strict whitelist on the ``case``
    field — only A/B/C accepted.
    """
    if not verdict_path.is_file():
        return None
    try:
        blob = json.loads(verdict_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(blob, dict):
        return None
    case = blob.get("case")
    if not isinstance(case, str):
        return None
    case_upper = case.strip().upper()
    if case_upper not in _VALID_CASES:
        return None
    piece_kind = blob.get("piece_kind") or "other"
    if not isinstance(piece_kind, str):
        piece_kind = "other"
    return MissingPieceClassification(
        case=case_upper,
        piece_kind=piece_kind,
        description=str(blob.get("description") or ""),
        ttnn_call=blob.get("ttnn_call") if isinstance(blob.get("ttnn_call"), dict) else None,
        cpu_module=blob.get("cpu_module") if isinstance(blob.get("cpu_module"), str) else None,
        submodule_spec=blob.get("submodule_spec") if isinstance(blob.get("submodule_spec"), dict) else None,
        notes=str(blob.get("notes") or ""),
    )


# ─── Heuristic fallback ──────────────────────────────────────────────


_TENSOR_OP_KEYWORDS = (
    "reshape",
    "permute",
    "transpose",
    "view",
    "contiguous",
    "slice",
    "squeeze",
    "unsqueeze",
    "expand",
    "repeat",
    "flatten",
)
_CONTROL_FLOW_KEYWORDS = (
    "dict access",
    "if branch",
    "conditional",
    "list construction",
    "tuple unpack",
    "indexing",
)


def heuristic_classify(missing_piece_description: str) -> Optional[MissingPieceClassification]:
    """Cheap regex-style classifier used as a fallback (or pre-check)
    when the LLM isn't available.

    Returns ``None`` if no keyword matches — caller falls through to
    the LLM classifier. Pure function, no I/O.

    Coverage is deliberately narrow: catches the obvious Cases A and B
    without trying to be exhaustive. Case C decisions need the LLM
    because they require semantic understanding of the HF forward.
    """
    if not missing_piece_description:
        return None
    text = missing_piece_description.lower()
    for kw in _TENSOR_OP_KEYWORDS:
        if kw in text:
            return MissingPieceClassification(
                case="A",
                piece_kind="tensor_op",
                description=missing_piece_description.strip(),
                notes=f"heuristic: matched tensor-op keyword '{kw}'",
            )
    for kw in _CONTROL_FLOW_KEYWORDS:
        if kw in text:
            return MissingPieceClassification(
                case="B",
                piece_kind="control_flow",
                description=missing_piece_description.strip(),
                notes=f"heuristic: matched control-flow keyword '{kw}'",
            )
    return None


# ─── Orchestrator ───────────────────────────────────────────────────


def run_classify_pass(
    *,
    model_id: str,
    demo_dir: Path,
    missing_piece_description: str,
    hf_forward_src: str = "",
    graduated_components: Optional[List[Dict[str, Any]]] = None,
    agent_invoker: Optional[Callable[..., int]] = None,
    use_heuristic_first: bool = True,
    agent_bin: str = "claude",
    agent_model: str = "haiku",
    timeout_s: int = 180,
) -> Optional[MissingPieceClassification]:
    """Classify one missing piece into Case A / B / C.

    Tries the heuristic first (cheap, no LLM cost) if
    ``use_heuristic_first`` is set. On no-heuristic-match (or when
    ``use_heuristic_first=False``), invokes the LLM via
    ``_invoke_agent`` to get a structured verdict.

    The agent invocation pattern mirrors Items 2 and 3: writes the
    verdict to ``<demo_dir>/_classify/verdict.json`` and parses it
    back. Same best-effort contract — None on any failure.
    """
    if graduated_components is None:
        graduated_components = []

    if use_heuristic_first:
        h = heuristic_classify(missing_piece_description)
        if h is not None:
            return h

    classify_dir = demo_dir / "_classify"
    try:
        classify_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    verdict_path = classify_dir / "verdict.json"
    try:
        if verdict_path.exists():
            verdict_path.unlink()
    except Exception:
        pass

    prompt = build_classify_prompt(
        model_id=model_id,
        missing_piece_description=missing_piece_description,
        hf_forward_src=hf_forward_src,
        graduated_components=graduated_components,
        verdict_path=verdict_path,
    )

    if agent_invoker is None:
        # Default invoker: wraps _invoke_agent with the verify-style
        # deliverable contract.
        def _default(prompt_text, *, expected_deliverable_files, timeout_s, **_):
            from .agent import _invoke_agent

            return _invoke_agent(
                prompt_text,
                provider="claude",
                agent_bin=agent_bin,
                cwd=demo_dir,
                model=agent_model,
                timeout_s=timeout_s,
                iter_tag="classify",
                expected_deliverable_files=list(expected_deliverable_files),
            )

        agent_invoker = _default

    try:
        rc = agent_invoker(prompt, expected_deliverable_files=[verdict_path], timeout_s=timeout_s)
    except Exception:
        return None
    if rc != 0:
        return None

    return parse_classify_verdict(verdict_path)


__all__ = [
    "MissingPieceClassification",
    "build_classify_prompt",
    "heuristic_classify",
    "parse_classify_verdict",
    "run_classify_pass",
]
