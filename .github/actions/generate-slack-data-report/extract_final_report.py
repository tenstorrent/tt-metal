#!/usr/bin/env python3
"""Extract final markdown report section from Cursor stream-json raw log."""

from __future__ import annotations

import json
import sys
from pathlib import Path


MARKER = "===FINAL_REPORT_MD===\n"


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: extract_final_report.py <raw_output_path> <report_path>", file=sys.stderr)
        return 2

    raw_path = Path(sys.argv[1])
    report_path = Path(sys.argv[2])
    chunks: list[str] = []

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
                chunks.append(text)

    full_text = "".join(chunks)
    final_md = full_text.split(MARKER, 1)[1].strip() if MARKER in full_text else ""
    report_path.write_text(final_md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
