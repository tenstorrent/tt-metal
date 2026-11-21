"""Failure-class-keyed edit scope: which files the LLM may also edit when a failure cannot be fixed in its stub."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..discovery import BRINGUP_ROOT, safe_relative_to_root, ttt_path


_EDIT_SCOPE_FOR_FAILURE_CLASS: Dict[str, List[str]] = {
    "L1_SMALL_ZERO": ["conftest.py", "tests/pcc/conftest.py"],
    "EMBEDDING_DTYPE": ["tests/pcc/conftest.py", "_captured/**/*.py"],
    "HANG": ["tests/pcc/conftest.py"],
}


# Repo-relative escalation paths (resolved against BRINGUP_ROOT, not demo_dir).
# Used when the failure originates outside the demo dir — e.g. a
# NotImplementedError from `models/tt_transformers/tt/rope.py` whose real
# fix lives in `models/tt_transformers/tt/model_config.py`. Without this,
# the LLM is given no path to fix bugs in canonical code that it didn't
# author. The Phi-3.5 attention failure (use_qk_fused for head_dim=96)
# is the motivating case. Stack paths are resolved per checkout by
# ttt_path, so a checkout on either layout gets its own files.
_REPO_RELATIVE_EDIT_SCOPE_FOR_FAILURE_CLASS: Dict[str, List[str]] = {
    "CONFIG_PARAM": [
        "models/ttt_compat/tt/model_config.py",
        "models/ttt_compat/tt/rope.py",
        "models/common/rmsnorm.py",
    ],
}


def _resolve_extra_edit_paths(
    demo_dir: Path,
    failure_class: str,
) -> List[Path]:
    """Expand the failure-class-keyed permission table into concrete
    file paths. Returns an empty list when the class is unknown or has
    no escalation.

    Used by the prompt-assembly path to emit an ESCALATED EDIT SCOPE
    section instructing the LLM that, for this iteration only, it may
    additionally edit the listed files. The default `_stubs/<comp>.py`
    write-scope is always implicitly allowed.

    Two tables are consulted:
      * ``_EDIT_SCOPE_FOR_FAILURE_CLASS`` — paths globbed under demo_dir
        (e.g. conftest.py inside the per-model test tree).
      * ``_REPO_RELATIVE_EDIT_SCOPE_FOR_FAILURE_CLASS`` — paths resolved
        against BRINGUP_ROOT (e.g. canonical tt_transformers source).
        Used when the bug lives in shared code, not the demo-local tree.
    """
    out: List[Path] = []

    # demo_dir-relative patterns (existing behavior).
    patterns = _EDIT_SCOPE_FOR_FAILURE_CLASS.get(failure_class, [])
    for pattern in patterns:
        try:
            for p in demo_dir.glob(pattern):
                if p.is_file():
                    out.append(p)
        except Exception:
            continue

    # repo-relative patterns (canonical source fixes).
    repo_patterns = _REPO_RELATIVE_EDIT_SCOPE_FOR_FAILURE_CLASS.get(failure_class, [])
    if repo_patterns:
        try:
            repo_root = BRINGUP_ROOT()
        except Exception:
            repo_root = None
        if repo_root is not None:
            for rel in repo_patterns:
                p = repo_root / ttt_path(rel, repo_root)
                if p.is_file():
                    out.append(p)

    return sorted(set(out))


def _format_escalated_edit_scope_block(
    demo_dir: Path,
    failure_class: str,
) -> str:
    """Render the ESCALATED EDIT SCOPE section for the LLM prompt.

    Returns an empty string when the failure class does not unlock
    any extra files — keeps the prompt small and avoids tempting the
    LLM to touch files it has no business in."""
    extra = _resolve_extra_edit_paths(demo_dir, failure_class)
    if not extra:
        return ""
    lines = [
        "ESCALATED EDIT SCOPE (failure-class-specific):",
        f"  Failure class `{failure_class}` cannot be fixed inside the per-",
        "  component stub alone. For THIS iteration ONLY, you are also",
        "  permitted to edit the following file(s):",
    ]
    for p in extra:
        try:
            rel = safe_relative_to_root(p)
        except Exception:
            rel = p
        lines.append(f"    - {rel}")
    if failure_class == "L1_SMALL_ZERO":
        lines.extend(
            [
                "",
                "  For L1_SMALL_ZERO specifically: edit the device fixture in",
                "  the listed conftest.py to pass `l1_small_size=16384` (or",
                "  a larger power-of-two if the diagnostic reports a bigger",
                "  buffer requirement) to whatever ttnn device-creation helper",
                "  the fixture uses. Look for `ttnn.open_device(...)`,",
                "  `ttnn.open_mesh_device(...)`, or",
                "  `MeshDevice.create(...)` invocations and add the kwarg.",
                "  This single change unblocks every conv2d/halo/sliding-window",
                "  op the model uses; no stub rewrite is needed once it lands.",
            ]
        )
    elif failure_class == "EMBEDDING_DTYPE":
        lines.extend(
            [
                "",
                "  For EMBEDDING_DTYPE: the test harness may be passing index",
                "  tensors as float32/bfloat16 when ttnn.embedding requires",
                "  uint32. Cast indices to uint32 either inside `__call__`",
                "  (preferred — robust to harness changes) or in the conftest",
                "  fixture's input prep step.",
            ]
        )
    elif failure_class == "HANG":
        lines.extend(
            [
                "",
                "  For HANG: if multiple components hang at the SAME stage",
                "  (e.g. `ttnn_to_torch`), this is likely a harness bug",
                "  rather than a stub bug. Inspect `_ttnn_to_torch_mesh_safe`",
                "  in the test conftest and consider replacing it with a",
                "  per-output type-dispatch (handle Tensor, tuple, list, dict)",
                "  that calls `ttnn.synchronize_device(device)` before drain.",
            ]
        )
    elif failure_class == "CONFIG_PARAM":
        lines.extend(
            [
                "",
                "  For CONFIG_PARAM: a NotImplementedError was raised from",
                "  canonical tt_transformers code because a model_config flag",
                "  (e.g. `use_qk_fused`, `use_hf_rope`) is set inconsistently",
                "  with the model's architecture (e.g. head_dim not divisible",
                "  by 64). The fix is usually a one-line addition to the",
                "  derivation in `model_config.py` to also gate on the",
                "  relevant architecture constraint. Trace the exception back",
                "  from the `raise NotImplementedError` site to wherever the",
                "  offending flag is set; add the missing condition there.",
                "  DO NOT just override the flag in the wrapper — that won't",
                "  stick if downstream code re-reads from configuration.",
            ]
        )
    lines.append("")
    return "\n".join(lines)
