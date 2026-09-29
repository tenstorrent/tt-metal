# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Terminal client for the Qwen3.8 chat server (standard library only).

Keeps the whole conversation and sends it with every turn, so the server
reuses the device state when the rendered prefix matches; the status line
after each reply shows ``prefix_reused`` and the decode rate.  Reasoning
deltas print between ``<think>`` and ``</think>`` markers.  With ``--tools``
the ``add_integers`` tool is offered and executed locally (the OpenAI shape
Hermes uses: string ``arguments``, a ``tool`` message per call), then the
turn continues until the model answers.  Commands: ``/reset`` clears the
history, ``/quit`` exits.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
from urllib.parse import urlsplit

try:
    import readline  # noqa: F401  line editing and history for input()
except ImportError:  # pragma: no cover  platforms without readline
    pass

MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
ADD_INTEGERS_TOOL = {
    "type": "function",
    "function": {
        "name": "add_integers",
        "description": "Add two bounded integers locally without network or filesystem access.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}
MAX_TOOL_ROUNDS = 4


def stream_completion(url: str, body: dict, *, timeout: float) -> tuple[dict, dict]:
    """POST one streaming request; print deltas as they arrive; return the assembled message and the final chunk."""

    parts = urlsplit(url)
    connection_class = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    connection = connection_class(
        parts.hostname, parts.port or (443 if parts.scheme == "https" else 80), timeout=timeout
    )
    payload = json.dumps({**body, "stream": True}).encode("utf-8")
    connection.request(
        "POST",
        parts.path.rstrip("/") + "/chat/completions",
        body=payload,
        headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
    )
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read().decode('utf-8', 'replace')}")
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: dict[int, dict] = {}
    thinking = False
    final: dict = {}
    buffer = b""
    while True:
        block = response.read1(4096) if hasattr(response, "read1") else response.read(4096)
        if not block:
            break
        buffer += block
        while b"\n\n" in buffer:
            event, buffer = buffer.split(b"\n\n", 1)
            for line in event.split(b"\n"):
                if not line.startswith(b"data: "):
                    continue
                data = line[len(b"data: ") :]
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise RuntimeError(f"server error: {chunk['error']}")
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})
                if delta.get("reasoning_content"):
                    if not thinking:
                        print("<think> ", end="", flush=True)
                        thinking = True
                    reasoning.append(delta["reasoning_content"])
                    print(delta["reasoning_content"], end="", flush=True)
                if delta.get("content"):
                    if thinking:
                        print(" </think>\n", end="", flush=True)
                        thinking = False
                    content.append(delta["content"])
                    print(delta["content"], end="", flush=True)
                for call in delta.get("tool_calls") or ():
                    slot = tool_calls.setdefault(
                        call["index"], {"id": call.get("id"), "type": "function", "function": {}}
                    )
                    if call.get("id"):
                        slot["id"] = call["id"]
                    function = call.get("function", {})
                    if function.get("name"):
                        slot["function"]["name"] = function["name"]
                    slot["function"]["arguments"] = slot["function"].get("arguments", "") + function.get(
                        "arguments", ""
                    )
                    print(
                        f"\n<tool_call {slot['function'].get('name')}({slot['function']['arguments']})>",
                        end="",
                        flush=True,
                    )
                if choice.get("finish_reason") is not None:
                    final = chunk
    connection.close()
    if thinking:
        print(" </think>", end="", flush=True)
    message: dict = {"role": "assistant", "content": "".join(content)}
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    return message, final


def execute_add_integers(call: dict) -> str:
    """The local tool: bounded integer addition; anything else is reported back to the model as an error."""

    function = call["function"]
    try:
        arguments = json.loads(function.get("arguments") or "{}")
        if function.get("name") != "add_integers" or set(arguments) != {"a", "b"}:
            raise ValueError(f"unknown tool call {function.get('name')}({sorted(arguments)})")
        values = [arguments[name] for name in ("a", "b")]
        if any(isinstance(value, bool) or not isinstance(value, int) or abs(value) > 10**9 for value in values):
            raise ValueError(f"add_integers needs bounded integers, got {values}")
        return json.dumps({"a": values[0], "b": values[1], "sum": sum(values)}, separators=(",", ":"))
    except (ValueError, TypeError) as error:
        return json.dumps({"error": str(error)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1", help="base URL (through the ssh port forward)")
    parser.add_argument("--system", default=None, help="system prompt (default: none; the server adds none either)")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="completion budget; omitted by default so the server grants the remaining context",
    )
    parser.add_argument(
        "--thinking", action="store_true", help="enable_thinking=true (reasoning shown in <think> markers)"
    )
    parser.add_argument("--reasoning-effort", default="low", choices=("low", "medium", "high", "xhigh"))
    parser.add_argument("--thinking-budget", type=int, default=None, help="reasoning tokens before </think> is forced")
    parser.add_argument("--tools", action="store_true", help="offer add_integers and execute its calls locally")
    parser.add_argument("--stop", action="append", default=None, help="stop string (repeatable, up to 4)")
    parser.add_argument(
        "--timeout", type=float, default=3600.0, help="socket timeout in seconds (prefill is 50 ms/token)"
    )
    args = parser.parse_args()

    history: list[dict] = [] if args.system is None else [{"role": "system", "content": args.system}]
    print(f"qwen38 chat at {args.url} ({MODEL_ID}); /reset clears the history, /quit exits", flush=True)
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line.strip():
            continue
        if line.strip() == "/quit":
            return 0
        if line.strip() == "/reset":
            history = history[:1] if history and history[0]["role"] == "system" else []
            print("history cleared (the server resets on the next request)", flush=True)
            continue
        history.append({"role": "user", "content": line})
        for _round in range(MAX_TOOL_ROUNDS + 1):
            print("assistant> ", end="", flush=True)
            body = {
                "model": MODEL_ID,
                "messages": history,
                **({"max_tokens": args.max_tokens} if args.max_tokens is not None else {}),
                "enable_thinking": args.thinking,
                "reasoning_effort": args.reasoning_effort,
            }
            if args.thinking_budget is not None:
                body["thinking_budget"] = args.thinking_budget
            if args.tools:
                body["tools"] = [ADD_INTEGERS_TOOL]
            if args.stop:
                body["stop"] = args.stop
            try:
                message, final = stream_completion(args.url, body, timeout=args.timeout)
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                print(f"\n[request failed: {error}]", file=sys.stderr, flush=True)
                while history and history[-1]["role"] != "user":
                    history.pop()
                history.pop()
                break
            print(flush=True)
            usage, extension = final.get("usage", {}), final.get("qwen38", {})
            finish = final.get("choices", [{}])[0].get("finish_reason")
            print(
                f"[{finish}: prompt {usage.get('prompt_tokens')} tok, completion {usage.get('completion_tokens')} tok, "
                f"prefix_reused {extension.get('prefix_reused')}{' (reset)' if extension.get('reset') else ''}, "
                f"queue {extension.get('queue_wait_seconds')} s, ttft {extension.get('ttft_seconds')} s, "
                f"{extension.get('tokens_per_second')} tok/s]",
                flush=True,
            )
            # Echo the reply as Hermes does: string arguments, no reasoning_content.
            echo = {key: value for key, value in message.items() if key != "reasoning_content"}
            history.append(echo)
            if not message.get("tool_calls") or not args.tools:
                break
            for call in message["tool_calls"]:
                result = execute_add_integers(call)
                print(f"tool> {call['function'].get('name')} -> {result}", flush=True)
                history.append({"role": "tool", "tool_call_id": call.get("id"), "content": result})


if __name__ == "__main__":
    sys.exit(main())
