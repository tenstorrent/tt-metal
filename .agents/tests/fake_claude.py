#!/usr/bin/env python3
"""Fake `claude` CLI for multigoal runner tests.

Driven by $FAKE_CLAUDE_DIR, a directory holding:
  - responses/001.json, 002.json, ...  one per expected invocation, in order:
      {
        "session_id": "sess-1",           # default: the --session-id/--resume value
        "subtype": "success",             # result event subtype
        "is_error": false,
        "terminal_reason": "completed",
        "verdict": {"status": "complete", "summary": "..."},   # -> structured_output
        "result_text": "...",             # raw final message (default: verdict JSON)
        "errors": ["..."],                # error-variant detail array
        "exit_code": 0,
        "stderr": "",
        "sleep": 0,                       # seconds before the result event
        "no_result_event": false,         # emit init but no result event
        "hang": false                     # emit init then sleep forever
      }
  - calls.jsonl (appended by this shim): argv, stdin, flag values, and whether
    ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN leaked into the environment.
  - counter (managed by this shim)
"""
import json
import os
import pathlib
import sys
import time


def flag_value(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def main() -> int:
    root = pathlib.Path(os.environ["FAKE_CLAUDE_DIR"])
    counter_file = root / "counter"
    n = int(counter_file.read_text()) + 1 if counter_file.exists() else 1
    counter_file.write_text(str(n))

    stdin_text = sys.stdin.read()
    argv = sys.argv[1:]
    resume = flag_value(argv, "--resume")
    requested_session = flag_value(argv, "--session-id")

    with (root / "calls.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "n": n,
                    "argv": argv,
                    "stdin": stdin_text,
                    "stdin_len": len(stdin_text),
                    "resume": resume,
                    "session_id_flag": requested_session,
                    "json_schema": flag_value(argv, "--json-schema"),
                    "settings": flag_value(argv, "--settings"),
                    "api_key_leaked": "ANTHROPIC_API_KEY" in os.environ,
                    "auth_token_leaked": "ANTHROPIC_AUTH_TOKEN" in os.environ,
                }
            )
            + "\n"
        )

    response_file = root / "responses" / f"{n:03d}.json"
    spec = json.loads(response_file.read_text()) if response_file.exists() else {}
    session_id = spec.get("session_id") or requested_session or resume or f"sess-{n}"

    print(json.dumps({"type": "system", "subtype": "init", "session_id": session_id, "model": "fake"}), flush=True)

    if spec.get("hang"):
        while True:
            time.sleep(3600)
    if spec.get("sleep"):
        time.sleep(spec["sleep"])

    if not spec.get("no_result_event"):
        subtype = spec.get("subtype", "success")
        verdict = spec.get("verdict", {"status": "complete", "summary": "done"})
        event: dict = {
            "type": "result",
            "subtype": subtype,
            "is_error": spec.get("is_error", False),
            "session_id": session_id,
            "num_turns": 1,
            "total_cost_usd": 0.0,
            "terminal_reason": spec.get("terminal_reason", "completed"),
        }
        if subtype == "success" and not spec.get("is_error"):
            # Success variant: carries result + structured_output.
            event["api_error_status"] = None
            if verdict is not None:
                event["structured_output"] = verdict
            event["result"] = spec.get("result_text", json.dumps(verdict) if verdict else "")
        else:
            # Error variant: no result / structured_output, detail in `errors`.
            if "errors" in spec:
                event["errors"] = spec["errors"]
            if "result_text" in spec:
                event["result"] = spec["result_text"]
        print(json.dumps(event), flush=True)

    if spec.get("stderr"):
        print(spec["stderr"], file=sys.stderr, flush=True)
    return int(spec.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
