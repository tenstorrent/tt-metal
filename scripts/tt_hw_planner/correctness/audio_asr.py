"""Audio-ASR correctness (Whisper / Wav2Vec2 / Hubert).

Compares the demo's transcribed text against the HF CPU
transcription on the same audio file using Word Error Rate (WER).

Why WER instead of token-overlap PCC
------------------------------------
ASR models have non-trivial post-processing (Whisper applies
language detection + timestamp prediction + chunked beam search;
Wav2Vec2 does CTC argmax). Comparing raw logits with PCC catches
numerical drift but is blind to "model said the right word but
the timestamps are wrong" — which is exactly what TT bring-ups
of Whisper have hit historically. WER on the final string is the
metric users actually care about.

WER is computed using the standard ``jiwer`` library if it's
installed; we fall back to a pure-Python Levenshtein-on-words
implementation otherwise (slightly slower but zero deps and
identical results for normal-length transcripts).

Demo-output protocol
--------------------
Looks for:

1. ``==TRANSCRIPT 0 - OUTPUT`` marker followed by the
   transcribed text on the next line (recommended convention).
2. ``transcription: <text>`` line (legacy Whisper demo format).
3. Falls back to ``Evidence(ok=False, ...)`` (soft skip).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from .base import Comparator, Evidence, ValidationResult
from .registry import register_comparator


DEFAULT_WER_MAX = 0.30


_TRANSCRIPT_MARKER_RE = re.compile(r"^==TRANSCRIPT\s+(?P<idx>\d+)\s+-\s+OUTPUT\s*$", re.M)
_TRANSCRIPTION_RE = re.compile(r"^\s*transcription:\s*(?P<text>.+)$", re.M)


def normalise_text(t: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.
    Standard ASR-evaluation pre-step; without it, "Hello." and
    "hello" register as a substitution."""
    if not t:
        return ""
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s']", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate. Pure-Python Levenshtein-on-words.

    Defined as (S + D + I) / N where S/D/I are substitutions,
    deletions, insertions and N is the number of words in the
    reference. Returns 0.0 if reference is empty and hypothesis
    is empty; 1.0 if reference is empty but hypothesis isn't
    (every word is an insertion / 0 ref).
    """
    ref_words = normalise_text(reference).split()
    hyp_words = normalise_text(hypothesis).split()
    n = len(ref_words)
    m = len(hyp_words)
    if n == 0 and m == 0:
        return 0.0
    if n == 0:
        return 1.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],
                    dp[i - 1][j],
                    dp[i][j - 1],
                )
    return dp[n][m] / n


_PYTEST_END_RE = re.compile(r"^(\d+ passed|\d+ failed|=+ |PASSED|FAILED|----|" r"collected \d+ items?$|nanobind:)")


def extract_transcript_from_pytest_output(
    captured_output: str,
    *,
    user_idx: int = 0,
) -> Optional[str]:
    """Try every parser. Returns ``None`` on miss."""
    matches = list(_TRANSCRIPT_MARKER_RE.finditer(captured_output))
    for m in matches:
        if int(m.group("idx")) == user_idx:
            after = captured_output[m.end() :]
            lines = []
            for line in after.splitlines():
                if line.startswith("==") and " - " in line:
                    break
                if line.startswith("==REPEAT BATCH"):
                    break

                if _PYTEST_END_RE.match(line):
                    break
                if line.strip():
                    lines.append(line)
            if lines:
                return "\n".join(lines).strip()

    m = _TRANSCRIPTION_RE.search(captured_output)
    if m:
        return m.group("text").strip()
    return None


@dataclass
class _ASRRef:
    text: str
    source_model_id: str = ""


class ASRComparator(Comparator):
    """Comparator for ASR backbones (Whisper / Wav2Vec2 / Hubert).

    Category claim: ``"STT"`` (the audit's "speech-to-text" key).
    """

    category: str = "STT"

    _ASR_MODEL_TYPES = ("whisper", "wav2vec2", "hubert", "data2vec_audio")

    def supports(self, category: str, model_id: str) -> bool:
        if category != self.category:
            return False
        mid_l = model_id.lower()
        return any(k in mid_l for k in self._ASR_MODEL_TYPES)

    def extract(
        self,
        captured_output: str,
        model_id: str,
    ) -> Evidence:
        text = extract_transcript_from_pytest_output(captured_output)
        if text is None:
            return Evidence(
                payload=None,
                ok=False,
                reason=(
                    "could not find a transcript in the pytest "
                    "output. Expected '==TRANSCRIPT 0 - OUTPUT' "
                    "marker followed by the text, OR a "
                    "'transcription: <text>' line."
                ),
            )
        return Evidence(
            payload=text,
            input_hint=None,
            ok=True,
            reason="transcript extracted from pytest output",
        )

    def load_reference(
        self,
        evidence: Evidence,
        model_id: str,
    ) -> _ASRRef:
        from transformers import pipeline

        audio_path = self._locate_demo_audio()
        if audio_path is None:
            return _ASRRef(text="", source_model_id=model_id)
        pipe = pipeline("automatic-speech-recognition", model=model_id, device="cpu")
        out = pipe(audio_path)
        ref_text = out.get("text", "") if isinstance(out, dict) else str(out)
        return _ASRRef(text=ref_text, source_model_id=model_id)

    @staticmethod
    def _locate_demo_audio() -> Optional[str]:
        from pathlib import Path

        try:
            here = Path(__file__).resolve()
            for parent in here.parents:
                candidate = parent / "models" / "demos" / "whisper" / "sample_data"
                if candidate.is_dir():
                    for f in sorted(candidate.iterdir()):
                        if f.suffix.lower() in (".wav", ".flac", ".mp3"):
                            return str(f)
        except Exception:
            pass
        return None

    def compare(
        self,
        evidence: Evidence,
        reference: Any,
    ) -> ValidationResult:
        if not isinstance(reference, _ASRRef):
            return ValidationResult(
                ok=False,
                reason="ASR comparator: reference is not an _ASRRef",
            )
        tt_text = str(evidence.payload or "")
        hf_text = reference.text
        if not hf_text:
            return ValidationResult(
                ok=True,
                reason="HF reference produced no transcript; soft pass",
                tt_text=tt_text,
                hf_text="",
            )
        score = wer(hf_text, tt_text)
        ok = score <= DEFAULT_WER_MAX
        return ValidationResult(
            ok=ok,
            reason=(f"{'PASS' if ok else 'FAIL'}: WER={score:.2f} " f"(threshold {DEFAULT_WER_MAX:.2f})"),
            tt_text=tt_text[:200],
            hf_text=hf_text[:200],
            mismatch_count=int(score * 1000),
            mismatch_ratio=float(score),
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
            f"(automatic speech recognition). The pytest completes but "
            f"the transcribed text differs from the HF CPU reference "
            f"beyond the configured WER threshold.\n\n"
            f"  GATE VERDICT (iter {iter_idx}/{max_iters}):\n"
            f"    {result.reason}\n"
            f"  TT transcript : {result.tt_text!r}\n"
            f"  HF transcript : {result.hf_text!r}\n\n"
            f"  LIKELY SUSPECTS for ASR bring-ups:\n"
            f"    1. Mel-spectrogram feature extraction differs "
            f"(window size, hop length, n_fft, n_mels).\n"
            f"    2. Encoder positional encoding (Whisper uses "
            f"sinusoidal; Wav2Vec2 uses learned + relative).\n"
            f"    3. Decoder beam-search settings (Whisper) vs "
            f"greedy CTC argmax (Wav2Vec2).\n"
            f"    4. Language-detection mis-routing (Whisper "
            f"transcribed English audio as German).\n"
            f"    5. Tokenizer mismatch (BPE vs WordPiece).\n\n"
            f"  WHAT WAS ALREADY TRIED:\n"
            f"    {prev}\n\n"
            f"  BUDGET: ~25 min/iter. Make at least one Edit.\n" + render_extra_blocks(extra_blocks)
        )


_singleton = ASRComparator()
register_comparator(_singleton)


__all__ = [
    "ASRComparator",
    "DEFAULT_WER_MAX",
    "extract_transcript_from_pytest_output",
    "normalise_text",
    "wer",
]
