# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end agentic coding acceptance harness for a running vLLM server.

The harness creates a disposable Python project, exposes only bounded file and
test tools, and drives the OpenAI chat-completions tool loop until the model has
inspected, repaired, and tested the fixture.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in the coding workspace.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the coding workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Replace a UTF-8 text file in the coding workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the workspace's fixed unittest suite.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _workspace_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {relative!r}") from exc
    if candidate.suffix != ".py":
        raise ValueError("only Python source files are accessible")
    return candidate


def execute_tool(root: Path, name: str, arguments: dict[str, Any]) -> str:
    if name == "list_files":
        return json.dumps([str(path.relative_to(root)) for path in sorted(root.glob("*.py"))])
    if name == "read_file":
        return _workspace_file(root, arguments["path"]).read_text()
    if name == "write_file":
        path = _workspace_file(root, arguments["path"])
        if not path.exists():
            raise ValueError(f"refusing to create an unexpected file: {arguments['path']}")
        path.write_text(arguments["content"])
        return json.dumps({"written": arguments["path"]})
    if name == "run_tests":
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "-v"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return json.dumps(
            {
                "exit_code": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        )
    raise ValueError(f"unknown tool: {name}")


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"server returned HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc


def _make_fixture(root: Path) -> None:
    (root / "math_utils.py").write_text(
        '"""Small arithmetic helpers."""\n\n' "def add(left: int, right: int) -> int:\n" "    return left - right\n"
    )
    (root / "test_math_utils.py").write_text(
        "import unittest\n\n"
        "from math_utils import add\n\n\n"
        "class AddTests(unittest.TestCase):\n"
        "    def test_positive_numbers(self):\n"
        "        self.assertEqual(add(7, 5), 12)\n\n"
        "    def test_negative_number(self):\n"
        "        self.assertEqual(add(-2, 5), 3)\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )


def run_harness(base_url: str, model: str, max_turns: int) -> None:
    with tempfile.TemporaryDirectory(prefix="muse-glimmer-agent-") as tmp:
        root = Path(tmp)
        _make_fixture(root)
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    "Act as a coding agent. Inspect the workspace, repair the failing "
                    "implementation without changing the tests, run the tests, and only "
                    "then give a concise final answer. Use the provided tools for every "
                    "workspace operation."
                ),
            }
        ]
        used: set[str] = set()
        tests_passed = False

        for turn in range(1, max_turns + 1):
            response = _post_json(
                f"{base_url.rstrip('/')}/v1/chat/completions",
                {
                    "model": model,
                    "messages": messages,
                    "tools": TOOLS,
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                    "temperature": 0,
                    "max_tokens": 1024,
                },
            )
            choice = response["choices"][0]
            assistant = choice["message"]
            messages.append(assistant)
            calls = assistant.get("tool_calls") or []
            print(f"turn {turn}: finish_reason={choice.get('finish_reason')} calls={len(calls)}")

            if not calls:
                required = {"read_file", "write_file", "run_tests"}
                missing = sorted(required - used)
                if missing:
                    raise AssertionError(f"model ended before required tool calls: {missing}")
                if not tests_passed:
                    raise AssertionError("model ended without a passing test run")
                final = assistant.get("content") or ""
                if not final.strip():
                    raise AssertionError("model returned no final answer")
                print("PASS: multi-turn coding task completed with passing tests")
                return

            for call in calls:
                name = call["function"]["name"]
                arguments = json.loads(call["function"].get("arguments") or "{}")
                try:
                    result = execute_tool(root, name, arguments)
                except Exception as exc:  # return tool failures to the agent to recover
                    result = json.dumps({"error": str(exc)})
                used.add(name)
                if name == "run_tests":
                    tests_passed = json.loads(result).get("exit_code") == 0
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": result,
                    }
                )

        raise AssertionError(f"model did not finish within {max_turns} turns")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="meta-models/Muse-Glimmer-30B")
    parser.add_argument("--max-turns", type=int, default=12)
    args = parser.parse_args()
    run_harness(args.base_url, args.model, args.max_turns)


if __name__ == "__main__":
    main()
