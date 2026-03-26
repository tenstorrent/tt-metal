#!/usr/bin/env python3
"""Extract final markdown report section from Cursor stream-json raw log."""

from __future__ import annotations

from difflib import SequenceMatcher
import json
import sys
from pathlib import Path


MARKER = "===FINAL_REPORT_MD===\n"


def normalize(text: str) -> str:
    return " ".join(text.replace("\r", "").replace("\n", " ").split())


def overlap_suffix_prefix(old: str, new: str) -> int:
    max_k = min(len(old), len(new))
    for k in range(max_k, 0, -1):
        if old[-k:] == new[:k]:
            return k
    return 0


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: extract_final_report.py <raw_output_path> <report_path>", file=sys.stderr)
        return 2

    raw_path = Path(sys.argv[1])
    report_path = Path(sys.argv[2])
    full_text = ""
    last_snapshot = ""

    for line in raw_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue

        if obj.get("type") != "assistant":
            continue

        message = obj.get("message", {})
        content = message.get("content", [])
        if not isinstance(content, list):
            continue

        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                if text.startswith(last_snapshot):
                    delta = text[len(last_snapshot) :]
                elif last_snapshot.startswith(text):
                    # Older replayed snapshot; ignore.
                    delta = ""
                else:
                    # Suppress near-identical corrected snapshots.
                    if last_snapshot:
                        similarity = SequenceMatcher(None, normalize(last_snapshot), normalize(text)).ratio()
                        if similarity >= 0.985:
                            last_snapshot = text
                            continue
                    overlap = overlap_suffix_prefix(last_snapshot, text)
                    delta = text[overlap:]
                last_snapshot = text
                if delta:
                    full_text += delta

    # Use the last marker in case the model emits retries/corrections.
    final_md = full_text.rsplit(MARKER, 1)[1].strip() if MARKER in full_text else ""
    report_path.write_text(final_md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
