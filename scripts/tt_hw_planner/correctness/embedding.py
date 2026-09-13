"""Embedding-category correctness (BGE / sentence-BERT / E5 / ...).

Compares the demo's per-sentence embedding vector against the HF
CPU reference for the same sentence using cosine similarity. The
gate is:

  cosine(tt_vec, hf_vec) >= 0.95  (per sentence)
  AND
  mean cosine across all probe sentences >= 0.97

Why two thresholds
------------------
A single per-sentence cosine misses the "model is rotated"
failure where every single output vector is close to its
reference but the inter-vector relationships are scrambled. The
mean-across-probes check catches that (it forces consistency
across multiple sentences). It's the same idea as the per-class
IoU floor for segmentation.

Demo-output protocol
--------------------
Looks for:

1. ``==EMBED N - OUTPUT`` marker followed by a base64 npy blob
   (one per probe sentence; ``N`` is 0..probe_count-1).
2. ``embeddings: <path>.npy`` line referencing a file with
   shape (num_probes, embed_dim).
3. Falls back to soft skip.
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from .base import Comparator, Evidence, ValidationResult
from .registry import register_comparator


DEFAULT_COSINE_PER_SENTENCE_MIN = 0.95
DEFAULT_COSINE_MEAN_MIN = 0.97
DEFAULT_PROBE_SENTENCES = (
    "The quick brown fox jumps over the lazy dog.",
    "Tenstorrent makes AI hardware in Toronto, Canada.",
    "Embeddings represent text as vectors in a high-dimensional space.",
)


_EMBED_MARKER_RE = re.compile(r"^==EMBED\s+(?P<idx>\d+)\s+-\s+OUTPUT\s*$", re.M)
_EMBED_END_RE = re.compile(r"^==EMBED\s+(?P<idx>\d+)\s+-\s+END\s*$", re.M)
_BASE64_LINE_RE = re.compile(r"^[A-Za-z0-9+/=]+$")
_EMBED_PATH_RE = re.compile(r"^\s*embeddings:\s*(?P<path>\S+\.npy)\s*$", re.M)


def cosine_similarity(a: Any, b: Any, *, eps: float = 1e-12) -> float:
    import numpy as np

    va = np.asarray(a, dtype=float).reshape(-1)
    vb = np.asarray(b, dtype=float).reshape(-1)
    if va.shape != vb.shape:
        return 0.0
    n = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if n < eps:
        return 0.0
    return float(np.dot(va, vb) / n)


def extract_embeddings_from_pytest_output(
    captured_output: str,
) -> Optional[Any]:
    return _extract_embeddings_and_indices(captured_output)[0]


def _extract_embeddings_and_indices(
    captured_output: str,
) -> Tuple[Optional[Any], List[int]]:
    """Return (matrix, indices). Matrix shape (num_emitted, embed_dim).
    Indices are the per-probe ids that were actually emitted (so the
    HF-reference compute can match them 1:1)."""
    import numpy as np

    rows: List[Any] = []
    indices: List[int] = []
    for m in _EMBED_MARKER_RE.finditer(captured_output):
        after = captured_output[m.end() :]
        chunks: List[str] = []
        for line in after.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if _EMBED_END_RE.match(line) or _EMBED_MARKER_RE.match(line):
                break
            if not _BASE64_LINE_RE.match(stripped):
                continue
            chunks.append(stripped)
        payload = "".join(chunks)
        try:
            raw = base64.b64decode(payload, validate=False)
            arr = np.load(io.BytesIO(raw), allow_pickle=False)
            rows.append(arr.reshape(-1))
            indices.append(int(m.group("idx")))
        except Exception:
            continue
    if rows:
        return np.stack(rows, axis=0), indices

    pm = _EMBED_PATH_RE.search(captured_output)
    if pm:
        path = Path(pm.group("path"))
        if path.exists():
            try:
                arr = np.load(path)
                if arr.ndim == 1:
                    arr = arr[None, :]
                return arr, list(range(arr.shape[0]))
            except Exception:
                pass
    return None, []


@dataclass
class _EmbedRef:
    matrix: Any
    probe_sentences: Sequence[str]
    source_model_id: str = ""


class EmbeddingComparator(Comparator):
    """Comparator for embedding backbones (BGE / sentence-BERT /
    E5)."""

    category: str = "Embed"

    def supports(self, category: str, model_id: str) -> bool:
        return category == self.category

    def extract(
        self,
        captured_output: str,
        model_id: str,
    ) -> Evidence:
        emb, indices = _extract_embeddings_and_indices(captured_output)
        if emb is None:
            return Evidence(
                payload=None,
                ok=False,
                reason=(
                    "could not find embeddings in the pytest output. "
                    "Expected '==EMBED N - OUTPUT' markers (one per "
                    "probe sentence) or an 'embeddings: <path>.npy' "
                    "line."
                ),
            )
        probes_all = list(DEFAULT_PROBE_SENTENCES)
        if indices:
            probes = [probes_all[i] for i in indices if 0 <= i < len(probes_all)]
            if len(probes) != len(indices):
                probes = probes_all[: emb.shape[0]]
        else:
            probes = probes_all[: emb.shape[0]]
        return Evidence(
            payload=emb,
            input_hint=probes,
            ok=True,
            reason=f"embeddings extracted ({len(probes)} probe(s), indices={indices})",
        )

    def load_reference(
        self,
        evidence: Evidence,
        model_id: str,
    ) -> _EmbedRef:
        import numpy as np
        from transformers import AutoTokenizer, AutoModel
        import torch

        probes = list(evidence.input_hint or DEFAULT_PROBE_SENTENCES)
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModel.from_pretrained(model_id)
        mdl.eval()
        rows: List[Any] = []
        is_decoder_only = bool(getattr(getattr(mdl, "config", None), "is_decoder", False)) or (
            getattr(getattr(mdl, "config", None), "model_type", "") or ""
        ).lower() in (
            "qwen3",
            "qwen2",
            "llama",
            "phi3",
            "phi4",
            "mistral",
            "gemma",
            "gemma2",
            "gemma3",
            "olmo2",
            "olmoe",
        )
        with torch.no_grad():
            for s in probes:
                t = tok(s, return_tensors="pt", truncation=True, max_length=128)
                out = mdl(**t)
                last = out.last_hidden_state.squeeze(0).to(torch.float32)
                if is_decoder_only:
                    mask = t["attention_mask"].squeeze(0)
                    last_idx = int(mask.sum().item()) - 1
                    pooled = last[last_idx]
                else:
                    mask = t["attention_mask"].squeeze(0).float().unsqueeze(-1)
                    pooled = (last * mask).sum(dim=0) / mask.sum().clamp(min=1)
                rows.append(pooled.detach().cpu().numpy())
        return _EmbedRef(
            matrix=np.stack(rows, axis=0),
            probe_sentences=probes,
            source_model_id=model_id,
        )

    def compare(
        self,
        evidence: Evidence,
        reference: Any,
    ) -> ValidationResult:
        import numpy as np

        if not isinstance(reference, _EmbedRef):
            return ValidationResult(
                ok=False,
                reason="embedding comparator: reference is not an _EmbedRef",
            )
        tt_mat = np.asarray(evidence.payload)
        hf_mat = np.asarray(reference.matrix)
        if tt_mat.shape[0] != hf_mat.shape[0]:
            return ValidationResult(
                ok=False,
                reason=(f"row count mismatch tt={tt_mat.shape[0]} vs " f"hf={hf_mat.shape[0]}"),
            )
        per_row = [cosine_similarity(tt_mat[i], hf_mat[i]) for i in range(tt_mat.shape[0])]
        worst = min(per_row)
        mean = sum(per_row) / len(per_row)
        ok = worst >= DEFAULT_COSINE_PER_SENTENCE_MIN and mean >= DEFAULT_COSINE_MEAN_MIN
        return ValidationResult(
            ok=ok,
            reason=(
                f"{'PASS' if ok else 'FAIL'}: "
                f"min cosine={worst:.3f} (>= {DEFAULT_COSINE_PER_SENTENCE_MIN}); "
                f"mean cosine={mean:.3f} (>= {DEFAULT_COSINE_MEAN_MIN})"
            ),
            tt_text=f"matrix shape={tt_mat.shape}",
            hf_text=f"matrix shape={hf_mat.shape}",
        )

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
        from .base import render_extra_blocks

        prev = "\n    ".join(previous_attempts or []) or "(none)"
        return (
            f"You are debugging a TT-hardware bring-up of {model_id!r} "
            f"(embedding backbone). The output vectors disagree with "
            f"HF beyond the cosine threshold.\n\n"
            f"  GATE VERDICT (iter {iter_idx}/{max_iters}):\n"
            f"    {result.reason}\n\n"
            f"  LIKELY SUSPECTS:\n"
            f"    1. Pooling layer mismatch (mean / CLS / max).\n"
            f"    2. Attention-mask propagation to pooling.\n"
            f"    3. LayerNorm epsilon mismatch.\n"
            f"    4. Output normalisation (L2 vs no-norm).\n"
            f"    5. Tokenizer pad-truncation differs from HF.\n\n"
            f"  WHAT WAS ALREADY TRIED:\n"
            f"    {prev}\n\n"
            f"  BUDGET: ~25 min/iter. Make at least one Edit.\n" + render_extra_blocks(extra_blocks)
        )


_singleton = EmbeddingComparator()
register_comparator(_singleton)


__all__ = [
    "DEFAULT_COSINE_MEAN_MIN",
    "DEFAULT_COSINE_PER_SENTENCE_MIN",
    "DEFAULT_PROBE_SENTENCES",
    "EmbeddingComparator",
    "cosine_similarity",
    "extract_embeddings_from_pytest_output",
]
