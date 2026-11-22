# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Test client for ``demo/prefill_server.py``: plays a decode server, sends prompts over TCP and checks the KV cache
entries that come back. Needs only torch (no ttnn, no device), so it runs on the decode host.

Examples::

    # 16384 random tokens, three times
    python models/experimental/deepseek_v41_flash/demo/prefill_client.py --server 10.82.97.40:9041 --random 16384 --repeat 3
    # a saved prompt (``test_d_p_export``'s file), entries compared with the saved ones
    python .../prefill_client.py --server 10.82.97.40:9041 --ids-file d_p_states_16k.pt --compare d_p_states_16k.pt
    # four decode servers at once (their prefills queue on the server), plus the error paths
    python .../prefill_client.py --server 10.82.97.40:9041 --random 8192 --clients 4 --check-errors
    # the prefix cache: a prompt's first half, then the whole prompt continued from it (only the new entries sent),
    # then the same prompt again; the continued entries compared with a cold prefill's
    python .../prefill_client.py --server 10.82.97.40:9041 --random 16384 --prefix-check
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import torch

from models.experimental.deepseek_v41_flash.tt.prefill.d_p_wire import PrefillClient, PrefillServerError


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))


def _prompt(args, info: dict) -> list[int]:
    if args.ids_file:
        saved = torch.load(args.ids_file, map_location="cpu", weights_only=False)
        ids = saved["ids"] if isinstance(saved, dict) else saved
        ids = [int(t) for t in torch.as_tensor(ids).flatten()]
    elif args.text_file:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        ids = tok.encode(Path(args.text_file).read_text())
    else:
        gen = torch.Generator().manual_seed(args.seed)
        ids = torch.randint(0, info["vocab"], (args.random,), generator=gen).tolist()
    n = min(len(ids), info["max_seq"]) // info["align"] * info["align"]
    if n <= 0:
        raise SystemExit(f"prompt of {len(ids)} tokens: the server needs at least {info['align']}")
    return ids[:n]


def _rate(tokens: int, seconds: float) -> str:
    return f"{tokens / seconds:.0f} tok/s" if seconds > 0 else "nothing computed"


def _check(states: dict, info: dict, n: int, have: int = 0) -> list[str]:
    """Shapes and finiteness of one reply against the server's contract (``d_p_wire`` docstring)."""
    bad = []
    if states["seq_len"] != n:
        bad.append(f"seq_len {states['seq_len']} for a {n}-token prompt")
    if states["base"] != have:
        bad.append(f"entries from {states['base']}, asked from {have}")
    if not 0 <= states["cached"] <= n:
        bad.append(f"{states['cached']} cached tokens of {n}")
    if sorted(states["layers"]) != sorted(info["layers"]):
        bad.append(f"layers {sorted(states['layers'])}, the server announced {info['layers']}")
    for i, layer in states["layers"].items():
        want = {"kv_tail": info["window"]}
        ratio = info["kv_sources"].get(str(i))
        if ratio:
            want.update(compressed_kv=(n - have) // ratio, idx_keys=(n - have) // ratio)
        if set(layer) != set(want):
            bad.append(f"layer {i}: fields {sorted(layer)}, want {sorted(want)}")
            continue
        for name, rows in want.items():
            t = layer[name]
            if t.shape[0] != rows:
                bad.append(f"layer {i} {name}: {t.shape[0]} rows, want {rows}")
            if not torch.isfinite(t.float()).all():
                bad.append(f"layer {i} {name}: non-finite values")
    return bad


def _compare(states: dict, path: str, min_pcc: float) -> list[str]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    bad = []
    for i, layer in saved["layers"].items():
        for name, want in layer.items():
            have = states["layers"].get(i, {}).get(name)
            if have is None or have.shape != want.shape:
                bad.append(f"layer {i} {name}: missing or shaped {None if have is None else tuple(have.shape)}")
                continue
            p = _pcc(have, want)
            if p < min_pcc:
                bad.append(f"layer {i} {name}: PCC {p:.6f} < {min_pcc}")
    return bad


def _run_client(k: int, args, ids: list[int], results: list) -> None:
    with PrefillClient(args.server) as client:
        for r in range(args.repeat):
            chunks = []
            t0 = time.perf_counter()
            states = client.prefill(ids, on_progress=chunks.append)
            wall = time.perf_counter() - t0
            results.append((k, r, wall, states, len(chunks)))
            n, mb, cached = states["seq_len"], states["bytes"] / 1e6, states["cached"]
            print(
                f"client {k} request {r}: {n} tokens ({cached} from the prefix cache), server prefill "
                f"{states['seconds']:.2f} s ({_rate(n - cached, states['seconds'])}), queued {states['queued']:.2f} s, "
                f"round trip {wall:.2f} s ({n / wall:.0f} tok/s), {mb:.1f} MB back, {len(chunks)} progress messages",
                flush=True,
            )


def _check_errors(server: str, info: dict) -> list[str]:
    """Bad requests get an error reply, and the connection keeps working after it."""
    bad = []
    with PrefillClient(server) as client:
        for what, ids in (
            ("unaligned length", [1] * (info["align"] + 1)),
            ("too long", [1] * (info["max_seq"] + info["align"])),
            ("id past the vocabulary", [1 << 30] * info["align"]),
        ):
            try:
                client.prefill(ids)
                bad.append(f"{what}: accepted")
            except PrefillServerError as e:
                print(f"error path ({what}): {e}")
        for what, have in (("unaligned have", info["align"] // 2), ("have past the prompt", 2 * info["align"])):
            try:
                client.prefill([1] * info["align"], have=have)
                bad.append(f"{what}: accepted")
            except PrefillServerError as e:
                print(f"error path ({what}): {e}")
        if not client.ping():
            bad.append("no pong after the errors")
    return bad


def _prefix_check(server: str, info: dict, ids: list[int], prefix: int, min_pcc: float) -> list[str]:
    """The prefix cache, as a decode server continuing a conversation uses it: the prompt's first ``prefix`` tokens
    computed cold; the whole prompt continued from them, sent with have=prefix (a hit: ``cached`` = prefix, only the
    entries of [prefix, n) back); the whole prompt cold (no lookup), whose entries the continued ones must match; the
    same prompt again with have=n (a hit with nothing to compute: the windows only)."""
    bad = []
    n, align = len(ids), info["align"]
    if not 0 < prefix < n or prefix % align:
        return [f"--prefix-len {prefix}: a multiple of {align} inside the {n}-token prompt"]
    with PrefillClient(server) as client:

        def ask(what, part, have=0, cache=True):
            t0 = time.perf_counter()
            got = client.prefill(part, have=have, cache=cache)
            print(
                f"{what}: {len(part)} tokens, {got['cached']} cached, prefill {got['seconds']:.2f} s "
                f"({_rate(len(part) - got['cached'], got['seconds'])}), round trip {time.perf_counter() - t0:.2f} s, "
                f"{got['bytes'] / 1e6:.1f} MB back",
                flush=True,
            )
            bad.extend(f"{what}: {b}" for b in _check(got, info, len(part), have))
            return got

        ask("prefix, cold", ids[:prefix], cache=False)
        hit = ask("continued", ids, have=prefix)
        cold = ask("whole prompt, cold", ids, cache=False)
        again = ask("same prompt again", ids, have=n)
    # (a slot an earlier run left holding more of the same prompt continues further: only a miss fails)
    if hit["cached"] < prefix:
        bad.append(f"continued: {hit['cached']} tokens from the cache, want at least {prefix}")
    if again["cached"] != n:
        bad.append(f"same prompt again: {again['cached']} tokens from the cache, want {n}")
    pairs = []  # (what, reply's tensor, the cold prefill's rows it must match)
    for i, layer in cold["layers"].items():
        ratio = info["kv_sources"].get(str(i))
        pairs.append((f"same prompt again layer {i} kv_tail", again["layers"][i]["kv_tail"], layer["kv_tail"]))
        for name, want in layer.items():
            rows = want[prefix // ratio :] if name != "kv_tail" else want
            pairs.append((f"continued layer {i} {name}", hit["layers"][i][name], rows))
    for what, got, want in pairs:
        p = _pcc(got, want) if got.shape == want.shape else float("nan")
        if not p >= min_pcc:
            bad.append(
                f"{what}: PCC {p:.6f} against the cold prefill (< {min_pcc}), shapes {tuple(got.shape)} / "
                f"{tuple(want.shape)}"
            )
    return bad


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="10.82.97.40:9041", help="host:port of prefill_server.py")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--random", type=int, default=16384, help="random prompt of this many tokens (default)")
    src.add_argument("--ids-file", help="torch file with the prompt's ids (a tensor, or a dict with 'ids')")
    src.add_argument("--text-file", help="text to tokenize with --tokenizer")
    p.add_argument("--tokenizer", default="/mnt/tt-data/sdawle/models/DeepSeek-V4.1-Flash-1017")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=1, help="requests per client")
    p.add_argument("--clients", type=int, default=1, help="connections sending at once")
    p.add_argument("--compare", help="host-states file (test_d_p_export) to compare the entries with")
    p.add_argument("--min-pcc", type=float, default=0.99)
    p.add_argument("--save", help="save the first reply's entries here (d_p_bridge host format)")
    p.add_argument("--check-errors", action="store_true", help="also check the server's error replies")
    p.add_argument("--prefix-check", action="store_true", help="check the prefix cache (and only it) with the prompt")
    p.add_argument(
        "--prefix-len", type=int, default=None, help="--prefix-check's cached prefix (default: half the prompt)"
    )
    args = p.parse_args(argv)

    with PrefillClient(args.server) as client:
        info = client.info
        t0 = time.perf_counter()
        client.ping()
        print(f"server {args.server}: {info}, ping {1e3 * (time.perf_counter() - t0):.1f} ms")
    ids = _prompt(args, info)
    print(f"prompt: {len(ids)} tokens")
    if args.prefix_check:
        prefix = args.prefix_len if args.prefix_len is not None else len(ids) // 2 // info["align"] * info["align"]
        bad = _prefix_check(args.server, info, ids, prefix, args.min_pcc)
        print(("FAIL\n  " + "\n  ".join(bad)) if bad else "PASS")
        return 1 if bad else 0

    results: list = []
    errors: list = []

    def run(k: int) -> None:
        try:
            _run_client(k, args, ids, results)
        except Exception as e:  # noqa: BLE001 - reported below
            errors.append(f"client {k}: {type(e).__name__}: {e}")

    t0 = time.perf_counter()
    threads = [threading.Thread(target=run, args=(k,)) for k in range(args.clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.perf_counter() - t0

    bad = list(errors)
    first = results[0][3] if results else None
    for k, r, _, states, _ in results:
        bad += [f"client {k} request {r}: {b}" for b in _check(states, info, len(ids))]
        if first is not None and states is not first:
            same = all(
                torch.equal(states["layers"][i][name], first["layers"][i][name])
                for i in first["layers"]
                for name in first["layers"][i]
            )
            if not same:
                bad.append(f"client {k} request {r}: entries differ from the first reply's for the same prompt")
    if first is not None:
        fields = sum(len(layer) for layer in first["layers"].values())
        print(f"reply: {len(first['layers'])} layers, {fields} tensors, {first['bytes'] / 1e6:.1f} MB")
        for i, layer in sorted(first["layers"].items()):
            print(f"  layer {i:>2}: " + ", ".join(f"{name} {tuple(t.shape)}" for name, t in layer.items()))
        if args.compare:
            bad += _compare(first, args.compare, args.min_pcc)
        if args.save:
            torch.save({**{k: v for k, v in first.items() if k != "bytes"}, "ids": torch.tensor(ids)}, args.save)
            print(f"saved to {args.save}")
    if results:
        n = len(ids) * len(results)
        print(
            f"{len(results)} request(s) from {args.clients} client(s) in {total:.2f} s: {n / total:.0f} tok/s overall"
        )
    if args.check_errors:
        bad += _check_errors(args.server, info)

    if bad:
        print("FAIL\n  " + "\n  ".join(bad))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
