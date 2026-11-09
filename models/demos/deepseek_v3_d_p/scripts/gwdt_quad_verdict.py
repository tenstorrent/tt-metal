# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-stage verdicts for the quad prefill-runner screen, from the files each runner rank writes.

Each rank r writes rank{r}.json (progress, first mismatch, error) and rank{r}.hashes.jsonl (per-device hashes of
the input it received and the output it sent, per iteration and chunk). Stage k's input is stage k-1's output
moved over the inter-galaxy link, so a mismatch is placed on the earliest stage that shows it: on its input it is
the link into the stage, on its output the stage's own galaxy.
"""

import json
import time
from pathlib import Path

from gwdt_checks import Result, Status

MAX_CHIPS_SHOWN = 6


def load_ranks(results_dir: Path, n_ranks: int, retries: int = 3, delay: float = 0.5) -> dict[int, dict | None]:
    """rank -> parsed rank file, None when absent or still unreadable after retries (a writer mid-rename)."""
    out: dict[int, dict | None] = {}
    for r in range(n_ranks):
        out[r] = None
        path = Path(results_dir) / f"rank{r}.json"
        for _ in range(retries):
            try:
                out[r] = json.loads(path.read_text())
                break
            except FileNotFoundError:
                break
            except (OSError, ValueError):
                time.sleep(delay)
    return out


def load_hashes(results_dir: Path, n_ranks: int) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r in range(n_ranks):
        path = Path(results_dir) / f"rank{r}.hashes.jsonl"
        if not path.is_file():
            continue
        lines = []
        for line in path.read_text().splitlines():
            try:
                lines.append(json.loads(line))
            except ValueError:
                # The writer appends line by line; only a torn last line can fail to parse.
                continue
        out[r] = lines
    return out


def chips(devices: list) -> str:
    shown = ",".join(f"({r},{c})" for r, c in devices[:MAX_CHIPS_SHOWN])
    more = f" +{len(devices) - MAX_CHIPS_SHOWN} more" if len(devices) > MAX_CHIPS_SHOWN else ""
    return f"chips {shown}{more}"


def _where(bad: dict) -> str:
    return f"iter {bad['iter']} chunk {bad['chunk']}"


def _link(k: int) -> str:
    return "host→galaxy 0 (H2D)" if k == 0 else f"link {k - 1}→{k}"


def _hash_link_mismatch(prev: list[dict], cur: list[dict], mesh_cols: int) -> tuple[dict, list] | None:
    """First (iter, chunk) where stage k's input hashes differ from stage k-1's output hashes."""
    sent = {(h["iter"], h["chunk"]): h["out"] for h in prev}
    for h in cur:
        out = sent.get((h["iter"], h["chunk"]))
        if out is None:
            continue
        bad = [i for i, (a, b) in enumerate(zip(out, h["in"])) if a != b]
        if bad:
            return h, [[i // mesh_cols, i % mesh_cols] for i in bad]
    return None


def _pos(p) -> tuple:
    return tuple(p) if p is not None else (-1, -1)


def _stage(k: int, doc: dict | None, prev: dict | None, hashes: dict, hung: bool, mesh_cols: int, host: str) -> Result:
    if doc is None:
        return Result(Status.FAIL, f"rank {k} never started")
    if doc.get("error"):
        return Result(Status.FAIL, f"galaxy {k} ({host}): {doc['error']}")
    bad = doc.get("first_bad")
    if bad is not None:
        where = f"{_where(bad)}, {chips(bad['devices'])}"
        if bad["what"] == "output":
            return Result(Status.FAIL, f"galaxy {k} ({host}) compute: output differs at {where}")
        upstream = (prev or {}).get("first_bad")
        if k > 0 and upstream is not None and upstream["what"] == "output":
            return Result(Status.FAIL, f"input differs at {where}; follows stage {k - 1} output")
        return Result(Status.FAIL, f"{_link(k)}: input differs at {where}")
    if k > 0 and k - 1 in hashes and k in hashes:
        hit = _hash_link_mismatch(hashes[k - 1], hashes[k], mesh_cols)
        if hit is not None:
            h, devices = hit
            return Result(Status.FAIL, f"{_link(k)}: received != sent at {_where(h)}, {chips(devices)}")
    if hung and not doc.get("done"):
        last_in, last_out = _pos(doc.get("last_in")), _pos(doc.get("last_out"))
        if k == 0 and doc.get("last_in") is None:
            return Result(Status.HANG, f"{_link(0)}: no input arrived")
        if last_in > last_out:
            return Result(Status.HANG, f"galaxy {k} ({host}) compute: stuck after input {list(last_in)}")
        if k > 0 and prev is not None and _pos(prev.get("last_out")) > last_in:
            return Result(Status.HANG, f"{_link(k)}: stage {k - 1} sent {prev['last_out']}, got {doc.get('last_in')}")
        return Result(Status.PASS, f"idle at {doc.get('last_out')} when the run hung")
    if not doc.get("done"):
        return Result(Status.FAIL, f"galaxy {k} ({host}): rank exited before the run finished")
    return Result(Status.PASS, f"{doc['iterations']} iters, {doc['chunks_done']} chunks, bit-identical")


def stage_verdicts(
    ranks: dict[int, dict | None], hashes: dict[int, list[dict]], hosts: list[str], hung: bool, mesh_cols: int = 4
) -> list[tuple[str, Result]]:
    rows = []
    for k in range(len(hosts)):
        doc = ranks.get(k)
        host = (doc or {}).get("host") or hosts[k]
        rows.append((f"stage {k} {host}", _stage(k, doc, ranks.get(k - 1), hashes, hung, mesh_cols, host)))
    return rows


def overall(rows: list[tuple[str, Result]], hung: bool = False) -> Result:
    for label, r in rows:
        if r.status != Status.PASS:
            return Result(r.status, f"{label.split()[0]} {label.split()[1]}: {r.reason}")
    if hung:
        return Result(Status.HANG, "run hung with every stage idle between chunks")
    return Result(Status.PASS, f"{len(rows)}/{len(rows)} stages bit-identical")
