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


def normalize_for_log(text: str) -> str:
    return " ".join(text.replace("\r", "").replace("\n", " ").split())


def tool_label(obj: dict[str, Any]) -> str:
    tool_call = obj.get("tool_call", {})
    if isinstance(tool_call, dict):
        for key in tool_call:
            if key.endswith("ToolCall"):
                return key[: -len("ToolCall")]
    return "unknown"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: stream_json_pretty.py <raw_output_path>", file=sys.stderr)
        return 2

    raw_path = Path(sys.argv[1])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    assistant_chars = 0
    assistant_buffer = ""
    last_assistant_snapshot = ""

    def flush_assistant(force: bool = False) -> None:
        nonlocal assistant_buffer
        while assistant_buffer:
            split_idx = -1
            for punct in ("\n", ".", "!", "?"):
                idx = assistant_buffer.rfind(punct)
                if idx > split_idx:
                    split_idx = idx

            if force:
                chunk = assistant_buffer
                assistant_buffer = ""
            elif split_idx >= 80:
                chunk = assistant_buffer[: split_idx + 1]
                assistant_buffer = assistant_buffer[split_idx + 1 :]
            elif len(assistant_buffer) >= 420:
                # Force periodic emission on long unpunctuated spans.
                chunk = assistant_buffer[:420]
                assistant_buffer = assistant_buffer[420:]
            else:
                break

            preview = normalize_for_log(chunk)
            if preview:
                print(f"[assistant] {preview}", flush=True)

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
                    print(f"[tool] started {tool_label(obj)}", flush=True)
                elif subtype == "completed":
                    print(f"[tool] completed {tool_label(obj)}", flush=True)
                continue

            if event_type == "assistant":
                text = safe_get_text(obj)
                if text:
                    # stream-json may emit cumulative assistant snapshots.
                    # Keep only novel suffix to avoid duplicate/repeated logs.
                    if text.startswith(last_assistant_snapshot):
                        delta = text[len(last_assistant_snapshot) :]
                        last_assistant_snapshot = text
                    elif last_assistant_snapshot.startswith(text):
                        # Older snapshot replayed; ignore.
                        delta = ""
                    else:
                        # Snapshot discontinuity (new turn / reset): emit separator.
                        if assistant_buffer:
                            flush_assistant(force=True)
                        print("[assistant] ---", flush=True)
                        delta = text
                        last_assistant_snapshot = text

                    if delta:
                        assistant_chars += len(delta)
                        assistant_buffer += delta
                        flush_assistant(force=False)
                continue

            if event_type == "result":
                flush_assistant(force=True)
                duration = obj.get("duration_ms")
                print(f"[result] completed duration_ms={duration}", flush=True)
                last_assistant_snapshot = ""
                continue

    flush_assistant(force=True)
    print(f"[summary] assistant_chars_streamed={assistant_chars}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
