"""
Parse Molmo2 ``verification/test.jsonl`` rows and resolve repo-local cached videos.

Used by ``demo/eval_video.py`` and related scripts.

Prompt + video handling matches ``demo.py`` video mode: ``<|video|>`` + space + user text
(see ``demo.py`` ``run_video_demo`` / ``--video`` + ``--prompt``), then ``preprocess_video``
with ``apply_chat_template=True``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

# ``models/demos/molmo2/tests/verification_jsonl.py`` -> molmo2 package root
_MOLMO2_ROOT = Path(__file__).resolve().parent.parent

# Must match ``models.demos.molmo2.tt.utils.VIDEO_PROMPT`` and demo CLI behavior.
DEFAULT_VIDEO_TOKEN = "<|video|>"


def _url_basename(url: str) -> str:
    url_path = url.split("?")[0]
    name = url_path.rstrip("/").split("/")[-1]
    return name or "video.mp4"


def extract_video_filename_from_url(url: str) -> str:
    """
    Return only the video file name from a full URL or path, e.g.
    ``d02d399003cca14e2a5c822d389821e553e6b7942c5640d48c075703e533d1dc.mp4``.

    Works for ``https://.../molmo2-eval-media/<hash>.mp4`` and for local paths.
    """
    if not url:
        return "video.mp4"
    s = url.strip()
    if s.startswith("http://") or s.startswith("https://"):
        return _url_basename(s)
    # Local path: use Path name
    return Path(s).name or "video.mp4"


def find_local_video_for_url(url: str) -> Optional[Path]:
    """
    Return a filesystem path if this URL's file already exists under the standard
    verification layout (same basename as the URL path).

    Checks, in order:

    - ``<molmo2>/verification/video_cache/<filename>``
    - ``<molmo2>/verification/<filename>``
    """
    filename = extract_video_filename_from_url(url)
    for subdir in ("video_cache", ""):
        base = _MOLMO2_ROOT / "verification"
        if subdir:
            base = base / subdir
        p = base / filename
        if p.is_file():
            return p
    return None


def normalize_verification_prompt(text: str) -> str:
    """
    Normalize user question text to stable newlines (LF only), matching JSONL / demo strings.

    Does not alter internal wording; only line-ending and outer trim.
    """
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    return t.strip()


def build_hf_video_prompt_like_demo(
    question_text: str,
    *,
    video_token: str = DEFAULT_VIDEO_TOKEN,
) -> str:
    """
    Build the same ``prompt`` string as ``demo.py`` for ``--video`` + ``--prompt``:

    - If ``question_text`` already contains ``video_token``, return it as-is (after normalize).
    - Else ``f"{video_token} {normalized_question}"`` — **space** after ``<|video|>``, like demo.

    Pass this to ``preprocess_video(..., prompt=..., apply_template=True)`` as in
    ``run_video_demo``.
    """
    q = normalize_verification_prompt(question_text)
    if video_token in q:
        return q
    return f"{video_token} {q}"


def extract_prompt_video_max(entry: dict[str, Any]) -> tuple[str, str, int]:
    """
    From one JSONL object (OpenAI-style chat + ``max_tokens``), return:

    - User prompt text (video question), **normalized** (LF newlines, stripped)
    - Full video URL string (for cache / download resolution)
    - ``max_tokens`` for generation (defaults to 16)

    **Content parts:** Walks ``messages[0].content`` in order. If there are multiple
    ``text`` or ``video_url`` parts, the **last** occurrence wins (typical files have
    one of each). ``verification/test.jsonl`` often repeats the same video URL on
    several consecutive lines — that is benchmark data, not a parsing bug.

    Use :func:`extract_video_filename_from_url` for logs or cache keys matching
    ``video_cache/<hash>.mp4``.
    """
    messages = entry.get("messages")
    if not messages:
        raise ValueError("entry has no messages")

    msg0 = messages[0]
    content = msg0.get("content")
    if not isinstance(content, list):
        raise ValueError("messages[0].content must be a list of parts")

    prompt_text: Optional[str] = None
    video_url: Optional[str] = None
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            prompt_text = part.get("text")
        elif ptype == "video_url":
            vu = part.get("video_url") or {}
            if isinstance(vu, dict):
                video_url = vu.get("url")

    if prompt_text is None:
        raise ValueError("no text part in messages[0].content")
    if not video_url:
        raise ValueError("no video_url in messages[0].content")

    prompt_text = normalize_verification_prompt(prompt_text)
    max_tokens = int(entry.get("max_tokens", 16))
    return prompt_text, video_url, max_tokens
