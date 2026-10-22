# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Out-of-process watcher for one prefill stage: python hang_watcher.py <probe.json>

Polls the stage's chunk counters on chip (0,0). When none has moved for the timeout and the counts say
this stage is at fault, or chip (0,0) stops answering reads, it prints one report, writes the verdict
file, asks the rank for its Python stacks (SIGUSR1), runs the triage command and SIGKILLs the rank.

Five counts, in chunks, follow a chunk through the stage:

    A       arrived    landed in our input, from the upstream stage or (rank 0) the producer
    C       consumed   our inbound op copied it out
    P       produced   our outbound op handed it to our sender; last rank: all its layer acks fired
    pushed             our sender handed it to fabric
    F       forwarded  the next stage took it; last rank: F = P

Once nothing has moved for the timeout, the first match decides:

    A == F                      -> not us   nothing is inside the stage
    P > pushed and pushed == F  -> culprit  our sender never pushed a ready chunk
    C > P, on every chip        -> culprit  our compute holds a chunk
    A > C and P == F            -> culprit  a chunk landed and we never took it
    otherwise                   -> not us   waiting on a neighbour

Each chip queues its next inbound op behind a WAIT for our previous push (_lease_reclaim), so a stage
held back by its downstream shows C == P. Each chip's D2D pipe runs on its own, so a stalled pipe on
one chip leaves a healthy neighbour's chip (0,0) holding a chunk too; hence "on every chip".
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time

from ttexalens import tt_exalens_lib as lib

MASK32 = 0xFFFFFFFF
TRIAGE_TIMEOUT_S = 600
# On the first read the stage has barely started; a word past this is not the counter we meant.
PLAUSIBLE_CHUNKS = 1024


def read(word: dict) -> int:
    return lib.read_word_from_device(f"{word['x']},{word['y']}", word["addr"], device_id=word["chip"]) & MASK32


def counts(probe: dict, raw: dict[str, int]) -> dict[str, int]:
    """Unwrapped words -> chunk counts, warm-up record left out. acks stays a raw count."""
    n = {k: raw[k] // w["unit"] for k, w in probe["words"].items()}
    kind, w = probe["kind"], probe["warmup_transfers"]
    if kind in ("last", "single"):
        # The last rank's output is its layer acks; the capture's warm pass stands for the warm-up record.
        warm = probe["warmup_acks"]
        n["P"] = n["F"] = w + (n["acks"] - warm) // probe["acks_per_chunk"] if n["acks"] >= warm else 0
    # Rank 0 never receives a warm-up, so only its outbound words carry it, once produced. Never clamp.
    if kind in ("middle", "last") or (kind == "first" and n["P"] >= 1):
        for k in ("P", "pushed", "F") if kind == "first" else ("A", "C", "P", "pushed", "F"):
            if k in n:
                n[k] -= w
    return n


def _chunk(i: int) -> str:
    """The i-th chunk, numbered like the rank log's c=."""
    return f"chunk c={i - 1}" if i >= 1 else "the warm-up record"


def chips_behind(probe: dict, consumed: int) -> tuple[list, list]:
    """Mesh coords whose inbound op has not taken `consumed` chunks, and those that did not answer."""
    w = probe["warmup_transfers"] if probe["kind"] in ("middle", "last") else 0
    behind, unreadable = [], []
    for word in probe["consumed_on_every_chip"]:
        try:
            if read(word) // word["unit"] - w < consumed:
                behind.append(word["coord"])
        except Exception:
            unreadable.append(word["coord"])
    return behind, unreadable


def _where(probe: dict, n: dict[str, int]) -> str:
    """Place a held chunk by its layer acks; each fires after a layer's KV write, before its FFN/MoE."""
    acks_per_chunk, warm, layers = probe["acks_per_chunk"], probe["warmup_acks"], probe["ack_layers"]
    j = n["acks"] - warm - max(n["P"], 0) * acks_per_chunk
    if j < 0:
        return f"warming up for trace capture, {n['acks']} of its {warm} warm-pass acks in"
    if j >= acks_per_chunk:
        return f"all {acks_per_chunk} layer acks in: in its last layer's FFN/MoE, or stalled before the D2D send"
    done = f"{j} of its {acks_per_chunk} layer acks in"
    if j == 0 and layers:
        return f"{done}: before layer {layers[0]}'s KV write"
    if 0 < j < len(layers):
        return f"{done}: between layer {layers[j - 1]}'s KV write and layer {layers[j]}'s"
    if 0 < j == len(layers):
        return f"{done}: after layer {layers[-1]}'s KV write"
    return done


def classify(probe: dict, n: dict[str, int]) -> tuple[bool, str]:
    """(our stage is the culprit, why). Only the culprit's watcher acts; waiting on a neighbour is not us."""
    A, C, P, F, pushed = n["A"], n["C"], n["P"], n["F"], n.get("pushed")
    if A == F:
        return False, "nothing inside the stage"
    # pushed == F: our sender may only push once the next stage has taken the previous chunk.
    if pushed is not None and P > pushed == F:
        return True, f"our sender never pushed {_chunk(pushed + 1)}, ready since our outbound op"
    if C > P:
        behind, unreadable = chips_behind(probe, C)
        if behind:
            return False, f"chips {behind} never took {_chunk(C)}: one pipe stalled, left to triage"
        took = "every chip took it" + (f", {unreadable} did not answer" if unreadable else "")
        return True, f"holding {_chunk(C)}, {_where(probe, n)}; {took}"
    # P == F: while we wait on our output, upstream or the producer may already have filled our input.
    if A > C and P == F:
        return True, f"{_chunk(C + 1)} landed and was never taken"
    return False, "waiting on a neighbour"


def report(probe: dict, n: dict[str, int], detail: str, frozen_s: float) -> str:
    """The one thing this process prints: why it is about to kill the rank."""
    shown = " ".join(f"{k}={n[k]}" for k in ("A", "C", "P", "pushed", "F") if k in n)
    return (
        f"PREFILL HANG: rank {probe['rank']} (layers {probe['layers']}) is the culprit -- {detail}\n"
        f"  {shown} chunks, acks={n['acks']} on chip (0,0); frozen for {frozen_s:.0f} s"
    )


def _signal_rank(pid: int, sig: int) -> None:
    # Only while the rank is still our parent: once it is gone its pid may be reused.
    if os.getppid() == pid:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, sig)


def run_command(probe: dict, detail: str) -> None:
    """Triage this host while the rank is alive, so it sees the device exactly as the hang left it."""
    env = {
        **os.environ,
        "PREFILL_HANG_RANK": str(probe["rank"]),
        "PREFILL_HANG_PID": str(probe["pid"]),
        "PREFILL_HANG_DETAIL": detail,
    }
    proc = subprocess.Popen(probe["command"], shell=True, env=env, start_new_session=True)
    try:
        proc.wait(timeout=TRIAGE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def convict(probe: dict, text: str) -> None:
    print(text, file=sys.stderr, flush=True)
    if probe["verdict_path"]:
        with contextlib.suppress(OSError), open(probe["verdict_path"], "w") as f:
            f.write(text + "\n")
    # Python stacks while the rank still runs: they tell a host stall from a device hang.
    _signal_rank(probe["pid"], signal.SIGUSR1)
    time.sleep(2)  # let faulthandler finish before anything can kill the rank
    if probe["command"]:
        run_command(probe, text)
    # A rank dying takes the MPI job down with it.
    _signal_rank(probe["pid"], signal.SIGKILL)


def _say(probe: dict, msg: str) -> None:
    print(f"PREFILL HANG WATCHER: rank {probe['rank']}: {msg}", file=sys.stderr, flush=True)


def main(probe_path: str) -> int:
    with open(probe_path) as f:
        probe = json.load(f)
    pid, names = probe["pid"], list(probe["words"])
    per_chunk = {k: probe["acks_per_chunk"] if k == "acks" else w["unit"] for k, w in probe["words"].items()}
    prev: dict[str, int] | None = None
    wraps = dict.fromkeys(names, 0)
    moved_at = time.monotonic()
    failing_since: float | None = None
    try:
        # Stop judging the moment the rank is gone: a rank killed mid-chunk must not be convicted.
        while os.getppid() == pid:
            try:
                snap = {k: read(probe["words"][k]) for k in names}
                failing_since = None
            except Exception as e:
                if prev is None:
                    _say(probe, f"cannot read chip (0,0) through exalens ({type(e).__name__}: {e}); not watching")
                    return 1
                # Keep the last values, which counts as frozen; reads that stay down are a hang too.
                snap = prev
                failing_since = time.monotonic() if failing_since is None else failing_since
            # After the read: the first one also brings exalens up, which takes a while.
            now = time.monotonic()

            if prev is None:
                odd = {k: snap[k] for k in names if snap[k] // per_chunk[k] > PLAUSIBLE_CHUNKS}
                if odd:
                    _say(probe, f"first read {odd} is not a fresh chunk counter; not watching")
                    return 1
                moved_at = now
            else:
                # The words are uint32: count each one's wraps and add them back.
                for k in names:
                    wraps[k] += prev[k] - snap[k] > 1 << 31
                if snap != prev:
                    moved_at = now
            prev = snap

            n = counts(probe, {k: snap[k] + (wraps[k] << 32) for k in names})
            limit = probe["timeout_s"] if n["P"] >= 1 else probe["first_chunk_timeout_s"]
            if failing_since is not None and now - failing_since >= limit:
                convict(probe, report(probe, n, "chip (0,0) stopped answering exalens reads", now - failing_since))
                return 1
            if now - moved_at >= limit:
                culprit, detail = classify(probe, n)
                if culprit:
                    convict(probe, report(probe, n, detail, now - moved_at))
                    return 1
                # Not us: judge again after another full window.
                moved_at = now
            time.sleep(probe["poll_s"])
        return 0
    finally:
        with contextlib.suppress(OSError):
            os.unlink(probe_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
