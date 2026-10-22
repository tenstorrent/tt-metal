# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Two users of a running ``demo/server.py`` talking to each other, both sides printed as they stream.

Each speaker is its own ``user`` with its own system prompt (persona) and its own copy of the conversation: its
lines are ``assistant`` turns and the other speaker's are ``user`` turns. Every request therefore continues the
conversation the speaker's slot already holds, so after the first turn only the other speaker's latest line is
prefilled -- the parked-slot path of the server. Both sessions are deleted at the end (``--keep`` leaves them).

Needs only ``requests``; it never opens the device. Start the server first (see ``demo/server.py``), then::

    python models/experimental/deepseek_v41_flash/demo/test_two_users_chat.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid

from models.experimental.deepseek_v4_flash.demo.client import Client, OpenAIError

_PERSONAS = (
    (
        "Alice",
        "You are Alice, a curious astronomer. You are chatting with Bob. Reply in two to four sentences, "
        "speak only as Alice, and end with a question for Bob.",
    ),
    (
        "Bob",
        "You are Bob, a practical marine engineer. You are chatting with Alice. Reply in two to four sentences, "
        "speak only as Bob, and end with a question for Alice.",
    ),
)
_OPENING = "Hi Bob! I spent last night looking at Jupiter's moons. What have you been working on lately?"


class Speaker:
    """One side of the conversation: a ``user`` key on the server and the history as this side sees it."""

    def __init__(self, name: str, persona: str, run_id: str):
        self.name = name
        self.user = f"{name.lower()}-{run_id}"
        self.messages: list[dict] = [{"role": "system", "content": persona}]

    def reply(self, client: Client, heard: str, args) -> str:
        """Send ``heard`` (the other side's line) and stream this side's answer to stdout; returns the answer."""
        self.messages.append({"role": "user", "content": heard})
        params = {"user": self.user, "max_tokens": args.max_tokens, "temperature": args.temperature}
        print(f"\n[{self.name}] ", end="", flush=True)
        text = ""
        t0 = time.perf_counter()
        for chunk in client.chat(self.messages, stream=True, **params):
            piece = chunk["choices"][0]["delta"].get("content") or ""
            if piece:
                print(piece, end="", flush=True)
                text += piece
        print(f"\n    ({len(text)} chars in {time.perf_counter() - t0:.1f}s)", flush=True)
        text = text.strip()
        self.messages.append({"role": "assistant", "content": text})
        return text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="deepseek-v4.1-flash")
    p.add_argument("--turns", type=int, default=6, help="replies in total, alternating Bob, Alice, Bob, ...")
    p.add_argument("--max-tokens", type=int, default=256, help="cap on each reply")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--opening", default=_OPENING, help="Alice's first line, which Bob answers first")
    p.add_argument("--timeout", type=float, default=1800, help="per-request timeout in seconds")
    p.add_argument("--keep", action="store_true", help="leave both sessions on the server afterwards")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    client = Client(args.base_url, model=args.model, timeout=args.timeout)
    print(f"server {client.base_url}: {client.health()}")

    run_id = uuid.uuid4().hex[:8]
    alice, bob = (Speaker(name, persona, run_id) for name, persona in _PERSONAS)
    # Alice's opening is her answer to a kickoff turn, so her first request already continues a conversation.
    alice.messages.append({"role": "user", "content": "Bob has just joined. Say hello and start the conversation."})
    alice.messages.append({"role": "assistant", "content": args.opening})
    print(f"\n[{alice.name}] {args.opening}")

    line, speakers = args.opening, (bob, alice)
    try:
        for turn in range(args.turns):
            line = speakers[turn % 2].reply(client, line, args)
            if not line:
                print(f"{speakers[turn % 2].name} gave an empty reply; stopping", file=sys.stderr)
                return 1
        print("\nsessions:")
        for row in client.sessions()["data"]:
            if row["id"] in (alice.user, bob.user):
                print(
                    f"  {row['id']}: slot {row['index']}, {row.get('tier', '?')}, {row['tokens']} tokens, "
                    f"{row['messages']} messages"
                )
    except OpenAIError as e:
        print(f"\nrequest failed: {e}", file=sys.stderr)
        return 1
    finally:
        if not args.keep:
            for speaker in (alice, bob):
                try:
                    client.reset_session(speaker.user)
                except OpenAIError as e:
                    print(f"could not delete session {speaker.user}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
