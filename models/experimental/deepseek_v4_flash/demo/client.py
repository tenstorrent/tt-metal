# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""HTTP client for the DeepSeek-V4-Flash inference server (``demo/server.py``).

Pure ``requests`` -- it never touches tt-metal, so it runs in any Python
environment with a network path to the server. The API mirrors the OpenAI API
(``/v1/models``, ``/v1/chat/completions``, ``/v1/completions``) plus the server's
session extensions (``/v1/sessions``, ``DELETE /v1/sessions/<user>``).

Two CLI modes:

* **One shot** -- ask one question and print the answer::

      python models/experimental/deepseek_v4_flash/demo/client.py \\
          --base-url http://10.0.0.5:8000 --message "What is the capital of France?" \\
          --stream --think

* **Interactive REPL** -- multi-turn chat; each turn sends the full conversation and the
  server feeds only the new tokens, because it keeps the KV cache of the conversation
  this history continues. Several clients (or several people) hold independent
  conversations against the one model, and their turns generate concurrently -- the
  server hands out a cache slot per request, so this does not depend on ``--user``, which
  only labels the requests (and picks what ``/reset`` rewinds)::

      python models/experimental/deepseek_v4_flash/demo/client.py --user alice --system-prompt "You are terse."

  Commands at the prompt: ``/user``, ``/system``, ``/think``, ``/reset``,
  ``/sessions``, ``/models``, ``/help``, ``/exit``.

Programmatic use::

    from models.experimental.deepseek_v4_flash.demo.client import Client

    client = Client("http://127.0.0.1:8000", model="deepseek-v4-flash")
    reply = client.chat(
        [{"role": "user", "content": "Hello"}],
        user="alice", max_tokens=256, temperature=0.7,
    )
    print(reply["choices"][0]["message"]["content"])

    for chunk in client.chat([{"role": "user", "content": "Hello"}], stream=True):
        print(chunk["choices"][0]["delta"].get("content", ""), end="")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterator, Optional
from urllib.parse import quote

import requests

_DEFAULT_BASE_URL = "http://127.0.0.1:8000"
_DEFAULT_MODEL = "deepseek-v4-flash"

_ANSI = {"gray": "\033[90m", "blue": "\033[94m", "reset": "\033[0m"}


class OpenAIError(RuntimeError):
    """An error reply from the server (carries the OpenAI ``status``/message)."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` from a stored reply (display helper only)."""
    return text.replace("<think>", "").replace("</think>", "")


class Client:
    """A thin OpenAI-compatible client for ``demo/server.py``."""

    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or os.environ.get("DEEPSEEK_V4_BASE_URL", _DEFAULT_BASE_URL)).rstrip("/")
        self.model = model or os.environ.get("DEEPSEEK_V4_MODEL_ID", _DEFAULT_MODEL)
        self.timeout = timeout
        self._http = requests.Session()

    # -- transport -------------------------------------------------------------- #
    def _request(self, method: str, path: str, *, json_body: Optional[dict] = None, stream: bool = False):
        url = self.base_url + path
        try:
            return self._http.request(method, url, json=json_body, stream=stream, timeout=self.timeout)
        except requests.RequestException as e:
            raise OpenAIError(0, f"cannot reach {url}: {e}") from e

    @staticmethod
    def _check(r) -> requests.Response:
        if r.status_code >= 400:
            try:
                body = r.json()
            except ValueError:
                body = {}
            err = body.get("error", {}) if isinstance(body, dict) else {}
            raise OpenAIError(r.status_code, err.get("message") or r.text[:300])
        return r

    @staticmethod
    def _sse(r: requests.Response) -> Iterator[dict]:
        """Parse an SSE body into ``data:`` JSON events, stopping at ``[DONE]``."""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            obj = json.loads(data)
            if isinstance(obj, dict) and "error" in obj:
                err = obj["error"]
                raise OpenAIError(err.get("code") or 0, err.get("message") or "stream error")
            yield obj

    # -- OpenAI endpoints ------------------------------------------------------- #
    def models(self) -> dict:
        return self._check(self._request("GET", "/v1/models")).json()

    def chat(self, messages: list[dict], *, stream: bool = False, **params):
        """One ``/v1/chat/completions`` call.

        Non-stream returns the full response dict; with ``stream=True`` it returns
        an iterator of chunk dicts whose ``choices[0]["delta"]`` carries
        ``content`` / ``reasoning_content`` deltas.
        """
        body: dict = {"model": self.model, "messages": messages, "stream": bool(stream)}
        body.update({k: v for k, v in params.items() if v is not None})
        r = self._check(self._request("POST", "/v1/chat/completions", json_body=body, stream=stream))
        if stream:
            return self._sse(r)
        return r.json()

    def completions(self, prompt: str, *, stream: bool = False, **params):
        """One ``/v1/completions`` call (legacy text completion)."""
        body: dict = {"model": self.model, "prompt": prompt, "stream": bool(stream)}
        body.update({k: v for k, v in params.items() if v is not None})
        r = self._check(self._request("POST", "/v1/completions", json_body=body, stream=stream))
        if stream:
            return self._sse(r)
        return r.json()

    # -- server extensions ------------------------------------------------------ #
    def sessions(self) -> dict:
        return self._check(self._request("GET", "/v1/sessions")).json()

    def reset_session(self, user: str = "default") -> dict:
        return self._check(self._request("DELETE", f"/v1/sessions/{quote(user, safe='')}")).json()

    def health(self) -> dict:
        return self._check(self._request("GET", "/health")).json()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_chat(reply: dict) -> None:
    message = reply["choices"][0]["message"]
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    color = sys.stdout.isatty()
    if reasoning:
        if color:
            print(f"{_ANSI['gray']}{reasoning}{_ANSI['reset']}")
        else:
            print(reasoning)
    print(content)


def _stream_chat(client: Client, messages: list[dict], params: dict) -> tuple[str, str]:
    """Stream one completion, printing deltas as they arrive; returns the
    assembled ``(reasoning, content)`` so the caller can store the message."""
    reasoning, content = "", ""
    color = sys.stdout.isatty()
    for chunk in client.chat(messages, stream=True, **params):
        delta = chunk["choices"][0]["delta"]
        r = delta.get("reasoning_content") or ""
        c = delta.get("content") or ""
        if r:
            reasoning += r
            sys.stdout.write(f"{_ANSI['gray']}{r}{_ANSI['reset']}" if color else r)
            sys.stdout.flush()
        if c:
            content += c
            sys.stdout.write(c)
            sys.stdout.flush()
    print()
    return reasoning, content


def _user_params(
    user: str, thinking: bool | None, max_tokens: int | None, temperature: float | None, top_p: float | None
) -> dict:
    params: dict = {"user": user}
    if thinking is not None:
        params["thinking"] = thinking
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    return params


def _cmd_chat(args) -> int:
    client = Client(args.base_url, args.model, args.timeout)
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.message})
    params = _user_params(args.user, args.think or None, args.max_tokens, args.temperature, args.top_p)
    try:
        if args.stream:
            _stream_chat(client, messages, params)
        else:
            reply = client.chat(messages, **params)
            _print_chat(reply)
    except OpenAIError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _cmd_repl(args) -> int:
    client = Client(args.base_url, args.model, args.timeout)
    user = args.user
    thinking = args.think or None
    system = args.system_prompt or None
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})

    print(
        f"\nDeepSeek-V4-Flash chat via {client.base_url} (model {client.model!r}). "
        "Talk to session user={user!r}; /help for commands.\n"
    )
    if thinking:
        print("[thinking mode: the reasoning block is shown in gray before the answer]\n")

    while True:
        try:
            line = input(f"you({user})> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.startswith("/"):
            head, _, rest = line.partition(" ")
            cmd, rest = head.lower(), rest.strip()
            if cmd in ("/exit", "/quit"):
                return 0
            elif cmd == "/help":
                print(_HELP)
            elif cmd == "/user":
                if not rest:
                    print(f"[talking to session user={user!r}]")
                    continue
                user = rest
                messages = [m for m in messages if m["role"] == "system"]
                print(f"[switched to user {user!r}; new conversation]")
            elif cmd == "/system":
                system = rest or None
                messages = [m for m in messages if m["role"] != "system"]
                if system:
                    messages.insert(0, {"role": "system", "content": system})
                print(f"[system prompt: {system!r}]")
            elif cmd == "/think":
                if rest not in ("on", "off", ""):
                    print("usage: /think [on|off]")
                    continue
                on = rest == "on" if rest else thinking is not True
                thinking = on
                print(f"[thinking {'on' if thinking else 'off'}]")
            elif cmd == "/reset":
                try:
                    client.reset_session(user)
                except OpenAIError as e:
                    print(f"[{e}]")
                messages = [m for m in messages if m["role"] == "system"]
                print(f"[session {user!r} reset; conversation cleared]")
            elif cmd == "/sessions":
                try:
                    rows = client.sessions().get("data", [])
                except OpenAIError as e:
                    print(f"[{e}]")
                    continue
                if not rows:
                    print("[no active sessions]")
                for row in rows:
                    think = " think" if row.get("thinking") else ""
                    print(
                        f"  {row['id']}: {row['tokens']}/{row['max_context']} tokens, "
                        f"{row['messages']} messages{think}"
                    )
            elif cmd == "/models":
                try:
                    for m in client.models().get("data", []):
                        print(f"  {m['id']}")
                except OpenAIError as e:
                    print(f"[{e}]")
            else:
                print(f"unknown command {cmd}; /help for the list")
            continue

        messages.append({"role": "user", "content": line})
        params = _user_params(user, thinking, args.max_tokens, args.temperature, args.top_p)
        try:
            if args.stream:
                reasoning, content = _stream_chat(client, messages, params)
            else:
                reply = client.chat(messages, **params)
                message = reply["choices"][0]["message"]
                reasoning, content = message.get("reasoning_content") or "", message.get("content") or ""
                _print_chat(reply)
        except OpenAIError as e:
            print(f"\n[error: {e}]")
            messages.pop()  # the turn was not fed; drop it and carry on
            continue
        assistant: dict = {"role": "assistant", "content": content}
        if reasoning:
            assistant["reasoning_content"] = reasoning
        messages.append(assistant)


_HELP = """commands:
  /user KEY       switch to another server-side KV-cache session (each is an
                  independent conversation; bare /user shows who you are)
  /system TEXT    replace the system prompt (bare /system clears it)
  /think [on|off] thinking mode for the following turns (no argument flips it)
  /reset          rewind this session's KV cache on the server and clear the chat
  /sessions       list the server's active sessions and their token usage
  /models         list the served model id
  /help           this message
  /exit           quit (also ctrl-D)
anything else is sent to the model as a user turn."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("DEEPSEEK_V4_BASE_URL", _DEFAULT_BASE_URL), help="server URL")
    p.add_argument("--model", default=os.environ.get("DEEPSEEK_V4_MODEL_ID", _DEFAULT_MODEL), help="model id")
    p.add_argument("--user", default="default", help="server-side session key (OpenAI 'user' field)")
    p.add_argument("--message", help="ask one question and exit instead of starting the REPL")
    p.add_argument("--system-prompt", default=os.environ.get("DEEPSEEK_V4_SYSTEM_PROMPT", ""), help="system prompt")
    p.add_argument("--think", action="store_true", help="thinking mode (streams a reasoning block in gray)")
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("DEEPSEEK_V4_MAX_NEW_TOKENS", "0")) or None)
    p.add_argument("--temperature", type=float, default=None, help="sampling temperature (default: greedy)")
    p.add_argument("--top-p", type=float, default=None, help="nucleus sampling probability")
    p.add_argument("--stream", action="store_true", help="stream the reply as it is decoded")
    p.add_argument("--timeout", type=float, default=None, help="per-request timeout in seconds")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.message:
        return _cmd_chat(args)
    return _cmd_repl(args)


if __name__ == "__main__":
    sys.exit(main())
