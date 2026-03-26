#!/usr/bin/env python3
"""Pretty-print Cursor stream-json output while preserving raw logs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def safe_get_text(obj: dict[str, Any]) -> str:
    message = obj.get("message", {})
    content = message.get("content", [])
    parts: list[str] = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: stream_json_pretty.py <raw_output_path>", file=sys.stderr)
        return 2

    raw_path = Path(sys.argv[1])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    assistant_chars = 0
    with raw_path.open("w", encoding="utf-8") as raw_out:
        for line in sys.stdin:
            raw_out.write(line)
            raw_out.flush()

            stripped = line.strip()
            if not stripped:
                continue

            try:
                obj = json.loads(stripped)
            except Exception:
                continue

            event_type = obj.get("type")
            subtype = obj.get("subtype")

            if event_type == "system":
                if subtype == "init":
                    model = obj.get("model", "unknown")
                    print(f"[system] initialized model={model}", flush=True)
                continue

            if event_type == "tool_call":
                if subtype == "started":
                    print("[tool] started", flush=True)
                elif subtype == "completed":
                    print("[tool] completed", flush=True)
                continue

            if event_type == "assistant":
                text = safe_get_text(obj)
                if text:
                    assistant_chars += len(text)
                    preview = text.replace("\n", " ").strip()
                    if len(preview) > 140:
                        preview = f"{preview[:137]}..."
                    if preview:
                        print(f"[assistant] {preview}", flush=True)
                continue

            if event_type == "result":
                duration = obj.get("duration_ms")
                print(f"[result] completed duration_ms={duration}", flush=True)
                continue

    print(f"[summary] assistant_chars_streamed={assistant_chars}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
