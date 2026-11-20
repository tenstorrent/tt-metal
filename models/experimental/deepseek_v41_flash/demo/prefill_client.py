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


def _check(states: dict, info: dict, n: int) -> list[str]:
    """Shapes and finiteness of one reply against the server's contract (``d_p_wire`` docstring)."""
    bad = []
    if states["seq_len"] != n:
        bad.append(f"seq_len {states['seq_len']} for a {n}-token prompt")
    if sorted(states["layers"]) != sorted(info["layers"]):
        bad.append(f"layers {sorted(states['layers'])}, the server announced {info['layers']}")
    for i, layer in states["layers"].items():
        want = {"kv_tail": info["window"]}
        ratio = info["kv_sources"].get(str(i))
        if ratio:
            want.update(compressed_kv=n // ratio, idx_keys=n // ratio)
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
            n, mb = states["seq_len"], states["bytes"] / 1e6
            print(
                f"client {k} request {r}: {n} tokens, server prefill {states['seconds']:.2f} s "
                f"({n / states['seconds']:.0f} tok/s), queued {states['queued']:.2f} s, round trip {wall:.2f} s "
                f"({n / wall:.0f} tok/s), {mb:.1f} MB back, {len(chunks)} progress messages",
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
        if not client.ping():
            bad.append("no pong after the errors")
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
    args = p.parse_args(argv)

    with PrefillClient(args.server) as client:
        info = client.info
        t0 = time.perf_counter()
        client.ping()
        print(f"server {args.server}: {info}, ping {1e3 * (time.perf_counter() - t0):.1f} ms")
    ids = _prompt(args, info)
    print(f"prompt: {len(ids)} tokens")

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
