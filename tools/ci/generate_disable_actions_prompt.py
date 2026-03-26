#!/usr/bin/env python3
"""Generate Cursor prompt for structured stale-disable action planning."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prompt from template for disable action planning.")
    parser.add_argument("--template", required=True, help="Prompt template markdown path")
    parser.add_argument("--candidates-json", required=True, help="Stale candidate JSON path")
    parser.add_argument("--output", required=True, help="Rendered prompt output path")
    parser.add_argument("--max-actions", type=int, default=3, help="Hard cap for disable actions")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    template = Path(args.template).read_text(encoding="utf-8")
    text = template.replace("__CANDIDATES_JSON_PATH__", args.candidates_json)
    text = text.replace("__MAX_ACTIONS__", str(args.max_actions))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
