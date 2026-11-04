# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Multi-user checks against a running ``demo/server.py``: decode rate while users switch, slot eviction past
``--num-users``, and whether interleaving users changes anyone's output.

Three subcommands, each a separate run (``all`` runs them in turn):

* ``perf`` -- decode rate when the scheduler switches between users. First a sweep: ``c`` users start together
  for each ``c`` in ``--concurrency`` (default 1 2 4 8) and decode ``--max-tokens`` each; reported are the
  per-user rate (mean and slowest), the total, and the longest gap between streamed chunks (a stall while users
  join or leave shows up there). Then churn: ``--churn-users`` users arrive at random times over
  ``--stagger`` seconds with random reply lengths, so turns join and leave while the others decode, each then
  sending ``--turns - 1`` follow-ups that continue their parked conversation.

* ``evict`` -- more users than slots (``--users``, default 80 against the server's 64): every user first stores a
  secret code ("remember AMBER-4821"), then, once all have, asks for it back. With the slots full a new user
  takes the slot idle longest and its conversation moves to host RAM (``--host-users``), and past that it is
  dropped, so the follow-ups are served from a parked slot, from host RAM or re-prefilled. Each must still answer
  with its own code: another user's code is a failure (a cache that followed the wrong user), as is an error. The
  report splits the follow-ups by where the conversation was (``/v1/sessions`` just before) and whether the
  server reports a cache hit. Needs the server started with a short ``--slot-idle-timeout`` (e.g. 1), else a
  user past the slots gets 429s until a parked slot has been idle 300 s (the client retries for
  ``--retry-seconds``).

* ``compare`` -- each of ``--users`` two-turn conversations (default 12, more than the 8 that run at once) first
  alone, one after the other, then all at once with random starts over ``--stagger`` seconds, so turns queue,
  join and leave mid-reply. Under greedy decoding a user's reply should not depend on who else ran: each pair is
  compared exactly. Every prompt carries a tag the reply must start with, so a reply carrying another user's tag
  (state from the wrong session) is a hard failure; so is a tag the lone run produced and the interleaved one
  lost, or an error. Any other difference is reported with where it starts; it fails the run with ``--strict``.
  A prompt may be fed by the prefill model in one run and replayed through decode in the other (the server picks
  by its measured replay rate), which can shift a long greedy reply late without being a switching bug.

Start the server greedy (no ``--temperature``) for ``compare``; ``--temperature`` here sends one with every
request. Sessions are deleted afterwards unless ``--keep``. Needs only ``requests``; it never opens the device::

    python models/experimental/deepseek_v41_flash/demo/test_server_users.py perf --base-url http://127.0.0.1:8000
    python models/experimental/deepseek_v41_flash/demo/test_server_users.py evict --users 80
    python models/experimental/deepseek_v41_flash/demo/test_server_users.py compare --users 12
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from models.experimental.deepseek_v4_flash.demo.client import Client, OpenAIError

_WORDS = (
    "AMBER BASALT CEDAR DELTA EMBER FALCON GRANITE HARBOR INDIGO JASPER KESTREL LUMEN MARBLE NECTAR ONYX "
    "PEPPER QUARTZ RAVEN SABLE TUNDRA UMBER VELVET WILLOW XENON YARROW ZEPHYR COBALT FJORD GLACIER HELIX"
).split()
_CODE = re.compile(r"\b([A-Z]+-\d{4})\b")
_TOPICS = (
    "explain how a lighthouse lens focuses light",
    "describe the life cycle of a monarch butterfly",
    "explain why the sky is blue at noon and red at sunset",
    "describe how a sourdough starter works",
    "explain how a bicycle stays upright",
    "describe the water cycle over a mountain range",
    "explain how a suspension bridge carries its load",
    "describe how bees communicate the location of flowers",
    "explain how a refrigerator moves heat",
    "describe how a glacier carves a valley",
    "explain how vaccines train the immune system",
    "describe how a coral reef forms",
    "explain how a jet engine produces thrust",
    "describe the phases of the moon",
    "explain how a compass finds north",
    "describe how a volcano island chain forms",
)
_STORY = "Write a long, detailed story about {topic}. Use at least 1500 words and do not stop early."


def _codes(n: int, rng: random.Random) -> list[str]:
    codes: set[str] = set()
    while len(codes) < n:
        codes.add(f"{rng.choice(_WORDS)}-{rng.randrange(1000, 10000)}")
    return sorted(codes, key=lambda _: rng.random())


@dataclass
class Reply:
    user: str
    text: str = ""
    completion_tokens: int = 0
    cache_hit_tokens: int | None = None
    finish: str | None = None
    error: str | None = None
    retries: int = 0  # 429s / 503s before the request was taken
    t_sent: float = 0.0
    t_first: float | None = None
    t_last: float | None = None
    gaps: list[float] = field(default_factory=list)  # seconds between streamed chunks
    tier: str | None = None  # where the conversation was just before the request (evict)

    @property
    def ttft(self) -> float | None:
        return None if self.t_first is None else self.t_first - self.t_sent

    @property
    def rate(self) -> float | None:
        """Decode tok/s of this reply: ``(tokens - 1) / (last chunk - first chunk)``."""
        if self.t_first is None or self.completion_tokens < 2 or self.t_last <= self.t_first:
            return None
        return (self.completion_tokens - 1) / (self.t_last - self.t_first)


_local = threading.local()


def _client(args) -> Client:
    """One client (HTTP session) per thread."""
    if getattr(_local, "client", None) is None:
        _local.client = Client(args.base_url, model=args.model, timeout=args.timeout)
    return _local.client


def _stream(args, user: str, messages: list[dict], max_tokens: int) -> Reply:
    """One streamed chat turn, retrying while the server is full (429/503) for up to ``--retry-seconds``."""
    reply = Reply(user)
    params = {"user": user, "max_tokens": max_tokens, "temperature": args.temperature}
    deadline = time.perf_counter() + args.retry_seconds
    backoff = 0.5
    while True:
        reply.t_sent = time.perf_counter()
        try:
            stream = _client(args).chat(messages, stream=True, **params)
            break
        except OpenAIError as e:
            if e.status not in (429, 503) or time.perf_counter() + backoff > deadline:
                reply.error = f"{e.status}: {e.message}"
                return reply
            reply.retries += 1
            time.sleep(backoff)
            backoff = min(backoff * 2, 5.0)
    try:
        for chunk in stream:
            if chunk.get("usage"):
                usage = chunk["usage"]
                reply.completion_tokens = usage.get("completion_tokens", 0)
                reply.cache_hit_tokens = usage.get("prompt_cache_hit_tokens")
            choices = chunk.get("choices") or []
            if not choices:
                continue
            reply.finish = choices[0].get("finish_reason") or reply.finish
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or ""
            if not (piece or delta.get("reasoning_content")):
                continue
            now = time.perf_counter()
            if reply.t_first is None:
                reply.t_first = now
            else:
                reply.gaps.append(now - reply.t_last)
            reply.t_last = now
            reply.text += piece
    except OpenAIError as e:
        reply.error = f"stream: {e.message}"
    return reply


class Conversation:
    """One user's side of a chat: its ``user`` key and the history it sends (which the server continues)."""

    def __init__(self, user: str, system: str | None = None):
        self.user = user
        self.messages: list[dict] = [{"role": "system", "content": system}] if system else []
        self.replies: list[Reply] = []

    def turn(self, args, text: str, max_tokens: int) -> Reply:
        self.messages.append({"role": "user", "content": text})
        reply = _stream(args, self.user, self.messages, max_tokens)
        if reply.error:
            self.messages.pop()
        else:
            self.messages.append({"role": "assistant", "content": reply.text.strip()})
        self.replies.append(reply)
        return reply


def _run_all(jobs, workers: int, stagger: float = 0.0, rng: random.Random | None = None) -> list:
    """Run the callables ``jobs`` on ``workers`` threads, each first sleeping a random start delay in
    ``[0, stagger)``; their results in order."""
    delays = [rng.uniform(0, stagger) if stagger and rng else 0.0 for _ in jobs]
    t0 = time.perf_counter()

    def run(job, delay):
        time.sleep(max(0.0, t0 + delay - time.perf_counter()))
        return job()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(run, jobs, delays))


def _cleanup(args, users) -> None:
    if args.keep:
        return
    client = Client(args.base_url, model=args.model, timeout=args.timeout)
    for user in users:
        try:
            client.reset_session(user)
        except OpenAIError as e:
            print(f"could not delete session {user}: {e}", file=sys.stderr)


def _pct(values, q: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def _rates_line(replies: list[Reply]) -> str:
    rates = [r.rate for r in replies if r.rate]
    gaps = [g for r in replies for g in r.gaps]
    if not rates:
        return "no decode rate (every reply had fewer than 2 tokens or failed)"
    return (
        f"{statistics.mean(rates):6.2f} tok/s/u mean, {min(rates):6.2f} slowest, "
        f"chunk gap p50 {1e3 * _pct(gaps, 0.5):5.1f} ms / p99 {1e3 * _pct(gaps, 0.99):6.1f} ms / "
        f"max {1e3 * max(gaps, default=0.0):7.1f} ms"
    )


def _errors(replies: list[Reply]) -> list[str]:
    return [f"{r.user}: {r.error}" for r in replies if r.error]


# --------------------------------------------------------------------------- #
# perf
# --------------------------------------------------------------------------- #
def cmd_perf(args, run_id: str) -> tuple[dict, list[str]]:
    rng = random.Random(args.seed)
    failures: list[str] = []
    out: dict = {"sweep": [], "churn": {}}

    print(f"\n== perf: concurrency sweep, {args.max_tokens} tokens each ==")
    for c in args.concurrency:
        convs = [Conversation(f"perf-{run_id}-c{c}-{i}") for i in range(c)]
        jobs = [
            (lambda conv=conv, i=i: conv.turn(args, _STORY.format(topic=_TOPICS[i % len(_TOPICS)]), args.max_tokens))
            for i, conv in enumerate(convs)
        ]
        t0 = time.perf_counter()
        replies = _run_all(jobs, c)
        wall = time.perf_counter() - t0
        failures += _errors(replies)
        tokens = sum(r.completion_tokens for r in replies)
        rates = [r.rate for r in replies if r.rate]
        print(
            f"  {c:2d} users: {_rates_line(replies)}; {tokens / wall:7.2f} tok/s total, "
            f"TTFT max {max((r.ttft or 0) for r in replies):5.2f} s"
        )
        out["sweep"].append(
            {
                "users": c,
                "tok_s_u_mean": statistics.mean(rates) if rates else None,
                "tok_s_u_min": min(rates) if rates else None,
                "tok_s_total": tokens / wall,
                "max_gap_s": max((g for r in replies for g in r.gaps), default=0.0),
            }
        )
        _cleanup(args, [conv.user for conv in convs])

    print(
        f"\n== perf: churn, {args.churn_users} users arriving over {args.stagger:.0f} s, "
        f"{args.turns} turn(s) each, {args.churn_workers} at a time =="
    )
    convs = [Conversation(f"perf-{run_id}-churn-{i}") for i in range(args.churn_users)]

    def churn(conv: Conversation, i: int) -> list[Reply]:
        n = rng.randint(max(2, args.max_tokens // 4), args.max_tokens)
        conv.turn(args, _STORY.format(topic=_TOPICS[i % len(_TOPICS)]), n)
        for _ in range(args.turns - 1):
            time.sleep(rng.uniform(0, 2))
            conv.turn(args, "Continue the story from where you stopped.", rng.randint(max(2, n // 4), n))
        return conv.replies

    t0 = time.perf_counter()
    per_user = _run_all(
        [lambda conv=conv, i=i: churn(conv, i) for i, conv in enumerate(convs)], args.churn_workers, args.stagger, rng
    )
    wall = time.perf_counter() - t0
    firsts = [rs[0] for rs in per_user]
    follow = [r for rs in per_user for r in rs[1:]]
    replies = firsts + follow
    failures += _errors(replies)
    print(f"  first turns:  {_rates_line(firsts)}")
    if follow:
        ttfts = [r.ttft for r in follow if r.ttft is not None]
        hits = sum(1 for r in follow if r.cache_hit_tokens)
        print(f"  follow-ups:   {_rates_line(follow)}")
        print(
            f"  follow-up TTFT p50 {_pct(ttfts, 0.5):.2f} s / max {max(ttfts, default=0):.2f} s; "
            f"{hits}/{len(follow)} continued from the cache"
        )
    tokens = sum(r.completion_tokens for r in replies)
    print(
        f"  {tokens} tokens in {wall:.1f} s = {tokens / wall:.2f} tok/s total; {sum(r.retries for r in replies)} retries"
    )
    out["churn"] = {
        "tok_s_total": tokens / wall,
        "first_tok_s_u": [r.rate for r in firsts],
        "follow_tok_s_u": [r.rate for r in follow],
        "follow_ttft_s": [r.ttft for r in follow],
        "max_gap_s": max((g for r in replies for g in r.gaps), default=0.0),
    }
    _cleanup(args, [conv.user for conv in convs])
    return out, failures


# --------------------------------------------------------------------------- #
# evict
# --------------------------------------------------------------------------- #
def _tiers(client: Client) -> dict[str, str]:
    """``user -> tier`` from ``/v1/sessions``: a device slot's tier, or ``host``."""
    tiers: dict[str, str] = {}
    for row in client.sessions()["data"]:
        if row.get("id") and (row["id"] not in tiers or row.get("index") is not None):
            tiers[row["id"]] = row.get("tier", "?")
    return tiers


def cmd_evict(args, run_id: str) -> tuple[dict, list[str]]:
    rng = random.Random(args.seed + 1)
    client = Client(args.base_url, model=args.model, timeout=args.timeout)
    health = client.health()
    slots = health.get("sessions")
    print(f"\n== evict: {args.users} users against {slots} slots ({health.get('slot_states')}) ==")
    if slots is not None and args.users <= slots:
        print(f"  warning: {args.users} users fit the {slots} slots, so nothing is evicted; raise --users")
    codes = _codes(args.users, rng)
    known = set(codes)
    convs = [Conversation(f"evict-{run_id}-{i:03d}") for i in range(args.users)]
    failures: list[str] = []

    def store(i: int) -> Reply:
        return convs[i].turn(
            args, f"Remember this secret code: {codes[i]}. Reply with just the word OK.", args.store_tokens
        )

    t0 = time.perf_counter()
    stored = _run_all([lambda i=i: store(i) for i in range(args.users)], args.workers)
    failures += _errors(stored)
    print(
        f"  stored {args.users - len(_errors(stored))}/{args.users} codes in {time.perf_counter() - t0:.1f} s, "
        f"{sum(r.retries for r in stored)} retries while the slots were full"
    )
    after = client.health()
    print(f"  slots now {after.get('slot_states')}, {after.get('host_sessions')} conversations in host RAM")

    order = list(range(args.users))
    rng.shuffle(order)

    def recall(i: int) -> Reply:
        tier = _tiers(_client(args)).get(convs[i].user, "gone")
        reply = convs[i].turn(
            args, "What was the secret code I asked you to remember? Reply with just the code.", args.recall_tokens
        )
        reply.tier = tier
        return reply

    if stored and all(r.error for r in stored):
        return {}, failures
    t0 = time.perf_counter()
    recalled = _run_all([lambda i=i: recall(i) for i in order], args.workers)
    by_user = dict(zip(order, recalled))
    rows = []
    for i in range(args.users):
        r = by_user[i]
        if stored[i].error:
            continue
        found = set(_CODE.findall(r.text.upper())) & known
        if r.error:
            verdict = "error"
            failures.append(f"{r.user}: {r.error}")
        elif found - {codes[i]}:
            verdict = "WRONG USER"
            failures.append(f"{r.user} (code {codes[i]}) answered another user's {sorted(found - {codes[i]})}")
        elif codes[i] in found:
            verdict = "ok"
        else:
            verdict = "no code"
        rows.append(
            {
                "user": r.user,
                "code": codes[i],
                "tier": r.tier,
                "hit": r.cache_hit_tokens,
                "ttft": r.ttft,
                "verdict": verdict,
                "text": r.text.strip()[:80],
            }
        )
    print(f"  recalled in {time.perf_counter() - t0:.1f} s")
    print(f"  {'was':>8} {'cache':>6} {'n':>4} {'ok':>4} {'no code':>8} {'wrong':>6} {'error':>6} {'TTFT p50':>9}")
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["tier"], "hit" if row["hit"] else "miss"), []).append(row)
    for (tier, hit), group in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        count = {v: sum(1 for row in group if row["verdict"] == v) for v in ("ok", "no code", "WRONG USER", "error")}
        ttft = _pct([row["ttft"] for row in group if row["ttft"] is not None], 0.5)
        print(
            f"  {tier:>8} {hit:>6} {len(group):>4} {count['ok']:>4} {count['no code']:>8} "
            f"{count['WRONG USER']:>6} {count['error']:>6} {ttft:>8.2f}s"
        )
    for row in rows:
        if row["verdict"] == "no code":
            print(f"    no code: {row['user']} ({row['tier']}, code {row['code']}): {row['text']!r}")
    _cleanup(args, [conv.user for conv in convs])
    return {"rows": rows}, failures


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def cmd_compare(args, run_id: str) -> tuple[dict, list[str]]:
    rng = random.Random(args.seed + 2)
    codes = _codes(args.users, rng)
    known = set(codes)
    turns = [
        lambda i: (
            f"Tag: {codes[i]}. Start your reply with the tag in square brackets, then "
            f"{_TOPICS[i % len(_TOPICS)]} in about 150 words."
        ),
        lambda i: "Now summarize what you wrote in two sentences. Start with the tag in square brackets again.",
    ]

    def conversation(prefix: str, i: int) -> list[Reply]:
        conv = Conversation(f"cmp-{run_id}-{prefix}-{i:02d}")
        for make in turns:
            if conv.turn(args, make(i), args.max_tokens).error:
                break
        return conv.replies

    print(f"\n== compare: {args.users} two-turn conversations, alone and then interleaved ==")
    t0 = time.perf_counter()
    alone = [conversation("alone", i) for i in range(args.users)]
    print(f"  alone, one at a time: {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    mixed = _run_all([lambda i=i: conversation("mixed", i) for i in range(args.users)], args.workers, args.stagger, rng)
    print(f"  interleaved ({args.workers} clients, starts over {args.stagger:.0f} s): {time.perf_counter() - t0:.1f} s")

    failures: list[str] = []
    rows = []
    same = differ = 0
    for i in range(args.users):
        for t in range(len(turns)):
            a = alone[i][t] if t < len(alone[i]) else None
            m = mixed[i][t] if t < len(mixed[i]) else None
            where = f"conversation {i} turn {t + 1}"
            if a is None or m is None or a.error or m.error:
                errors = [x.error for x in (a, m) if x is not None and x.error]
                failures.append(f"{where}: {'; '.join(errors) or 'missing reply'}")
                continue
            foreign = (set(_CODE.findall(m.text)) & known) - {codes[i]}
            if foreign:
                failures.append(f"{where}: the interleaved reply carries another user's tag {sorted(foreign)}")
            if codes[i] in a.text and codes[i] not in m.text:
                failures.append(f"{where}: the interleaved reply lost its tag {codes[i]}")
            if a.text == m.text:
                same += 1
                continue
            differ += 1
            k = _common_prefix(a.text, m.text)
            row = {
                "conversation": i,
                "turn": t + 1,
                "prefix_chars": k,
                "alone_chars": len(a.text),
                "mixed_chars": len(m.text),
                "alone": a.text[k : k + 60],
                "mixed": m.text[k : k + 60],
            }
            rows.append(row)
            if args.strict:
                failures.append(f"{where}: replies differ from char {k}")
    print(f"  {same} replies identical, {differ} differ (of {same + differ} compared)")
    for row in rows:
        print(
            f"    conversation {row['conversation']} turn {row['turn']}: same for {row['prefix_chars']} of "
            f"{row['alone_chars']}/{row['mixed_chars']} chars, then alone {row['alone']!r} vs "
            f"interleaved {row['mixed']!r}"
        )
    _cleanup(args, [r.user for rs in alone + mixed for r in rs[:1]])
    return {"identical": same, "differ": rows}, failures


# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("perf", "evict", "compare", "all"))
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="deepseek-v4.1-flash")
    p.add_argument("--temperature", type=float, default=None, help="sent with every request (default: none)")
    p.add_argument("--timeout", type=float, default=3600, help="per-request timeout in seconds")
    p.add_argument("--retry-seconds", type=float, default=600, help="how long a request retries 429/503s")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep", action="store_true", help="leave the sessions on the server")
    p.add_argument("--json", help="also write the results to this file")
    g = p.add_argument_group("perf")
    g.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8], help="users decoding together")
    g.add_argument("--max-tokens", type=int, default=256, help="tokens a reply decodes (perf, compare)")
    g.add_argument("--churn-users", type=int, default=16)
    g.add_argument("--churn-workers", type=int, default=8, help="churn users in flight at once")
    g.add_argument("--turns", type=int, default=2, help="turns per churn user (follow-ups continue the chat)")
    g.add_argument("--stagger", type=float, default=20.0, help="seconds over which users start (perf, compare)")
    g = p.add_argument_group("evict / compare")
    g.add_argument("--users", type=int, default=None, help="evict: 80, compare: 12")
    g.add_argument("--workers", type=int, default=8, help="requests in flight at once")
    g.add_argument("--store-tokens", type=int, default=8)
    g.add_argument("--recall-tokens", type=int, default=24)
    g.add_argument("--strict", action="store_true", help="compare: any difference fails the run")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    client = Client(args.base_url, model=args.model, timeout=args.timeout)
    health = client.health()
    print(f"server {client.base_url}: {health.get('sessions')} slots, {health.get('slot_states')}")
    run_id = uuid.uuid4().hex[:6]
    commands = ("perf", "evict", "compare") if args.command == "all" else (args.command,)
    results: dict = {}
    failures: list[str] = []
    users = args.users
    for name in commands:
        args.users = users if users is not None else {"evict": 80, "compare": 12}.get(name)
        result, failed = {"perf": cmd_perf, "evict": cmd_evict, "compare": cmd_compare}[name](args, run_id)
        results[name] = result
        failures += [f"{name}: {f}" for f in failed]
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=lambda o: asdict(o) if isinstance(o, Reply) else str(o))
        print(f"\nwrote {args.json}")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
