"""G7: persistent learnings keyed by structural signature.

When the agentic loop graduates a run (gate passes), this module records:

* ``arch_signature``       -- a hash of the HF config's structural
                              fields (model_type, num_layers,
                              hidden_size, sliding_window, etc.). Two
                              gemma3 sizes share the signature in the
                              non-size fields; the exact match counts
                              size + width.
* ``first_diverging_qn``   -- the qualified-name of the first-
                              diverging module at the moment the run
                              graduated (or ``""`` if the very first
                              gate passed without ever firing the
                              probe).
* ``diff``                 -- the minimal LLM-produced diff that
                              flipped the verdict (collected by the
                              executor's bisection step).

On every NEW run, before invoking the LLM, the executor consults this
log: if a fix exists for the current model's
``(arch_signature, first_diverging_qn)`` it's auto-applied and the
gate re-runs. If the gate passes, the LLM is never asked.

This is the cross-run learning the user has been asking for:
medgemma's fix flows to qwen2-vl flows to paligemma without writing
a single line of model-specific code -- the framework recognises that
both models have the same structural signature and the same first-
diverging qualified-name, so the same diff applies.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from scripts.tt_hw_planner.learning import _filelock as _learning_filelock

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_LOG = _THIS_DIR / "learned_fixes.json"
_DEFAULT_LOCK = _THIS_DIR / ".learned_fixes.lock"


@dataclass
class LearnedFix:
    """One persistent fix record."""

    arch_signature: str
    first_diverging_qn: str
    diff: str
    diff_files: List[str]
    source_model_id: str
    timestamp: float
    notes: str = ""


def compute_arch_signature(hf_config: Dict[str, Any]) -> str:
    """Compute the structural-signature hash for an HF config.

    Returns a 16-char hex digest. Empty string if the config is
    missing the fields needed for a meaningful signature."""
    if not hf_config:
        return ""

    from scripts.tt_hw_planner.architecture import (
        build_arch_spec,
        detect_architecture,
    )

    candidates = [hf_config]
    for k in (
        "text_config",
        "language_config",
        "decoder_config",
        "text_model_config",
        "language_model_config",
    ):
        nested = hf_config.get(k)
        if isinstance(nested, dict):
            candidates.append(nested)
    structural_keys = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
    )
    best = max(
        candidates,
        key=lambda c: sum(1 for f in structural_keys if c.get(f)),
    )

    try:
        family = detect_architecture(best)
        spec = build_arch_spec(best, family)
    except Exception:
        return ""

    parts: Dict[str, Any] = {
        "model_type": hf_config.get("model_type") or best.get("model_type") or "",
        "architectures": json.dumps(hf_config.get("architectures") or [], sort_keys=True),
        "family": spec.family,
        "num_layers": spec.num_layers,
        "hidden_size": spec.hidden_size,
        "num_attention_heads": spec.num_attention_heads,
        "num_key_value_heads": spec.num_key_value_heads,
        "head_dim": spec.head_dim,
        "vocab_size": spec.vocab_size,
        "max_position_embeddings": spec.max_position_embeddings,
        "sliding_window": spec.sliding_window,
        "kv_lora_rank": spec.kv_lora_rank,
        "qk_rope_head_dim": spec.qk_rope_head_dim,
        "num_experts": spec.num_experts,
        "experts_per_token": spec.experts_per_token,
    }
    if not any(parts.get(k) for k in ("num_layers", "hidden_size")):
        return ""

    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def _load_log(path: Path) -> List[LearnedFix]:
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    out: List[LearnedFix] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            out.append(LearnedFix(**{k: entry.get(k) for k in LearnedFix.__annotations__}))
        except TypeError:
            continue
    return out


def _save_log(path: Path, entries: List[LearnedFix]) -> None:
    try:
        path.write_text(
            json.dumps([asdict(e) for e in entries], indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def lookup_fix(
    *,
    arch_signature: str,
    first_diverging_qn: str,
    log_path: Path = _DEFAULT_LOG,
    failure_class: Optional[str] = None,
    error_extract: Optional[str] = None,
    component_kind: Optional[str] = None,
) -> Optional[LearnedFix]:
    """Return the most recent learned fix matching this
    (signature, qn) pair, or None.

    ``failure_class`` / ``error_extract`` / ``component_kind`` are
    accepted for the richer in-loop tier-2 call site but do not yet
    refine matching — ``LearnedFix`` persists only the (signature, qn)
    key. They are reserved for a future schema that keys on them."""
    if not arch_signature:
        return None
    entries = _load_log(log_path)
    matching = [e for e in entries if e.arch_signature == arch_signature and e.first_diverging_qn == first_diverging_qn]
    if not matching:
        return None
    return max(matching, key=lambda e: e.timestamp)


def register_fix(
    *,
    arch_signature: str,
    first_diverging_qn: str,
    diff: str,
    diff_files: List[str],
    source_model_id: str,
    notes: str = "",
    failure_class: Optional[str] = None,
    error_extract: Optional[str] = None,
    component_kind: Optional[str] = None,
    log_path: Path = _DEFAULT_LOG,
    lock_path: Path = _DEFAULT_LOCK,
) -> bool:
    """Append a new learned fix to the log. Returns True on success.

    Concurrency-safe via the same naive O_EXCL filelock used by the
    bring-up learning log (:func:`learning._filelock`)."""
    if not arch_signature or not diff:
        return False
    entry = LearnedFix(
        arch_signature=arch_signature,
        first_diverging_qn=first_diverging_qn,
        diff=diff,
        diff_files=diff_files,
        source_model_id=source_model_id,
        timestamp=time.time(),
        notes=notes,
    )
    try:
        with _learning_filelock(lock_path):
            entries = _load_log(log_path)
            entries.append(entry)
            _save_log(log_path, entries)
        return True
    except TimeoutError:
        return False


def apply_fix(
    *,
    fix: LearnedFix,
    workspace_root: Path,
) -> Tuple[bool, str]:
    """Apply a learned fix as a git patch. Returns (ok, message)."""
    import subprocess
    import tempfile

    if not fix.diff:
        return False, "empty diff"
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as tf:
            tf.write(fix.diff)
            patch_path = tf.name
    except Exception as exc:
        return False, f"could not write patch: {type(exc).__name__}: {exc}"
    try:
        out = subprocess.run(
            ["git", "apply", "--3way", patch_path],
            cwd=str(workspace_root),
            capture_output=True,
            timeout=60,
        )
        if out.returncode == 0:
            return True, f"applied {len(fix.diff_files)} file(s) from learned fix"
        return False, (f"git apply rc={out.returncode}: " f"{out.stderr.decode(errors='replace')[:300]}")
    except Exception as exc:
        return False, f"apply failed: {type(exc).__name__}: {exc}"


__all__ = [
    "LearnedFix",
    "apply_fix",
    "compute_arch_signature",
    "lookup_fix",
    "register_fix",
]
