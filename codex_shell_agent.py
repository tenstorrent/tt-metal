#!/usr/bin/env python3
"""
codex_shell_agent.py
--------------------
Run a safeguarded loop where an OpenAI model proposes JSON-encoded shell
commands, the script validates them against a whitelist, executes the allowed
ones, feeds the outputs back to the model, and stops when the model signals
completion.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

# Default prefixes that commands must start with (after leading whitespace).
DEFAULT_ALLOWED_PREFIXES = [
    "gh",
    "python",
    "python3",
    "cat",
    "ls",
    "pwd",
]

JSON_SCHEMA_DOC = textwrap.dedent(
    """
    Respond ONLY with valid JSON matching this schema:
    {
      "status": "running" | "complete",
      "commands": [
        {
          "cmd": "<literal shell command>",
          "reason": "<why this command is needed>"
        }
      ],
      "final_artifact": {
        "summary": "<short description>",
        "content": "<full text (markdown/json/plain)>",
        "write_path": "<relative or absolute path for content>"
      } | null
    }

    Rules:
    - While status == "running", include one or more commands and set
      final_artifact to null.
    - When the task is finished set status == "complete", leave commands empty,
      and populate final_artifact.
    - DO NOT emit any prose outside the JSON.
    - Use only commands that begin with the allowed prefixes given later.
    - If you require a command outside the whitelist, explain why and wait for
      new tool outputs rather than emitting unapproved commands.
    """
).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded Codex/GPT command runner with JSON protocol.")
    parser.add_argument(
        "--api-key",
        required=True,
        help="OpenAI API key (required each invocation; not stored).",
    )
    parser.add_argument(
        "--input-json",
        required=True,
        type=Path,
        help="Path to the JSON file the model should analyze.",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="High-level task instructions passed to the model.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model ID with JSON-friendly output.",
    )
    parser.add_argument(
        "--allowed-prefix",
        dest="allowed_prefixes",
        action="append",
        default=None,
        help="Additional allowed command prefixes (can repeat).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="Safety cap on number of model-command cycles.",
    )
    parser.add_argument(
        "--plan-file",
        type=Path,
        default=Path("agent_plan.json"),
        help="File where each model JSON plan is written.",
    )
    parser.add_argument(
        "--final-output",
        type=Path,
        default=Path("agent_final_output.md"),
        help="Destination file for the model's final artifact.",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("."),
        help="Working directory for executed shell commands.",
    )
    return parser.parse_args()


def build_allowed_prefixes(args: argparse.Namespace) -> List[str]:
    prefixes = DEFAULT_ALLOWED_PREFIXES.copy()
    if args.allowed_prefixes:
        prefixes.extend(args.allowed_prefixes)
    # Normalize whitespace and duplicates.
    cleaned = []
    seen = set()
    for prefix in prefixes:
        prefix = prefix.strip()
        if prefix and prefix not in seen:
            seen.add(prefix)
            cleaned.append(prefix)
    return cleaned


def command_is_allowed(command: str, allowed_prefixes: List[str]) -> bool:
    stripped = command.lstrip()
    if not stripped:
        return False
    first_token = stripped.split(maxsplit=1)[0]
    return any(first_token == prefix or stripped.startswith(f"{prefix} ") for prefix in allowed_prefixes)


def run_shell(command: str, cwd: Path) -> Dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "cmd": command,
        "exit_code": completed.returncode,
        "output": completed.stdout,
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_json_response(text_payload: str) -> Dict[str, Any]:
    try:
        return json.loads(text_payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response was not valid JSON: {exc}\n{text_payload}") from exc


def main() -> None:
    args = parse_args()
    allowed_prefixes = build_allowed_prefixes(args)

    client = OpenAI(api_key=args.api_key)
    input_payload = args.input_json.read_text(encoding="utf-8")

    system_message = textwrap.dedent(
        f"""
        You orchestrate shell work on behalf of the user.
        {JSON_SCHEMA_DOC}
        Allowed prefixes: {', '.join(allowed_prefixes)}.
        Always reason internally and only emit the JSON object.
        """
    ).strip()

    base_user_message = textwrap.dedent(
        f"""
        Task request:
        {args.task}

        Input file: {args.input_json}
        JSON contents:
        ```json
        {input_payload}
        ```

        Respond with JSON per the schema. Begin when ready.
        """
    ).strip()

    conversation: List[Dict[str, str]] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": base_user_message},
    ]

    working_dir = args.working_dir.resolve()

    for iteration in range(1, args.max_iterations + 1):
        print(f"\n=== Iteration {iteration} ===")
        response = client.responses.create(model=args.model, input=conversation)
        response_text = response.output_text.strip()

        try:
            plan = ensure_json_response(response_text)
        except ValueError as exc:
            warning = (
                "Your previous response was not valid JSON per the agreed schema. "
                "Please reply with a valid JSON object only."
            )
            print(f"Model JSON error: {exc}")
            conversation.append({"role": "assistant", "content": response_text})
            conversation.append({"role": "user", "content": warning})
            continue

        conversation.append({"role": "assistant", "content": response_text})
        save_json(args.plan_file, plan)
        status = plan.get("status", "")
        commands = plan.get("commands", []) or []

        if status not in {"running", "complete"}:
            conversation.append(
                {
                    "role": "user",
                    "content": ("Status must be 'running' or 'complete'. " f"Got '{status}'. Please correct the JSON."),
                }
            )
            continue

        if status == "complete":
            final_artifact = plan.get("final_artifact") or {}
            content = final_artifact.get("content", "")
            summary = final_artifact.get("summary", "")
            write_path = final_artifact.get("write_path") or str(args.final_output)

            final_path = Path(write_path)
            if not final_path.is_absolute():
                final_path = working_dir / final_path
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_text(content, encoding="utf-8")
            print(f"Final summary: {summary}")
            print(f"Wrote final artifact to {final_path}")
            break

        disallowed = [
            cmd_info["cmd"]
            for cmd_info in commands
            if not command_is_allowed(cmd_info.get("cmd", ""), allowed_prefixes)
        ]

        if disallowed:
            feedback = textwrap.dedent(
                f"""
                The following commands are not permitted because they violate the whitelist
                {allowed_prefixes}:
                {json.dumps(disallowed, indent=2)}
                Please respond again with only allowed commands.
                """
            ).strip()
            conversation.append({"role": "user", "content": feedback})
            continue

        execution_results = []
        for idx, cmd_info in enumerate(commands, start=1):
            cmd = cmd_info["cmd"]
            reason = cmd_info.get("reason", "")
            print(f"[{idx}/{len(commands)}] Running: {cmd}")
            if reason:
                print(f"Reason: {reason}")
            result = run_shell(cmd, cwd=working_dir)
            execution_results.append(result)
            print(f"Exit code: {result['exit_code']}")

        feedback_message = {
            "role": "user",
            "content": textwrap.dedent(
                f"""
                Command execution complete. Here are the results:
                ```json
                {json.dumps(execution_results, indent=2)}
                ```
                Continue the task and respond with the required JSON schema.
                """
            ).strip(),
        }
        conversation.append(feedback_message)
    else:
        raise RuntimeError(f"Reached maximum iterations ({args.max_iterations}) without completion.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user.")
