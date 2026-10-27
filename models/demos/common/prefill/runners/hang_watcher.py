# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Out-of-process watcher for one prefill stage: python hang_watcher.py <probe.json>

Poll the stage's counters on chip (0,0); if none has moved for timeout_s, read every chip and classify.
On a verdict of culprit it prints the verdict and chip (0,0)'s counts to stderr, runs handoff.command,
and SIGKILLs the rank. With no command it exits instead and leaves the rank hung for inspection.

A chunk goes into a chip through its inputs and out through its outputs. Five counts in chunks follow it:

    A       arrived    landed in an input: from the upstream stage, or from the producer on rank 0
    C       consumed   our inbound op copied it out
    P       produced   our outbound op handed it to an output's sender; last and single rank: all its acks fired
    pushed             the sender handed it to fabric
    F       forwarded  the next stage took it

Inputs are listed in the order the run loop receives them and outputs in the order it sends them: the
activation first, then side streams such as GLM-5.3's indices. A chip has a chunk in hand once every input
took it, C_in = the lowest C, and it is out of compute once the first output has it, P_out = that output's
P. A chip of a still stage waits on the first of:

    SENDER       an output's P > pushed == F                    a ready chunk its sender never pushed
    OUTBOUND     an output's P < the one before's P == pushed   the one before went out, its send never ran
    COMPUTE      C_in > P_out                                   took a chunk in, never produced it
    DOWNSTREAM   an output's P > F                              the next stage has not taken our last chunk
    INBOUND      the next input's A > C                         a chunk landed and we never took it
    UPSTREAM     anything else                                  idle, or the stage before us is stuck

The next input is the one the run loop waits on: the first still at C_in. COMPUTE also covers the first
output's outbound op: both sit between C and P.

The stage is the culprit if a chip waits on SENDER, OUTBOUND or INBOUND, or on COMPUTE with its P_out below
every chip's C_in. Waiting on a neighbour is never us: each chip queues its next inbound op behind a WAIT for
its previous push (_lease_reclaim), so a stage held back by its downstream never has a chunk in hand on every
chip.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto

from ttexalens import tt_exalens_lib as lib

MASK32 = 0xFFFFFFFF


class WaitingOn(Enum):
    """What a chip of a still stage waits on; Chip.state takes the first that matches, in this order."""

    SENDER = auto()
    OUTBOUND = auto()
    COMPUTE = auto()
    DOWNSTREAM = auto()
    INBOUND = auto()
    UPSTREAM = auto()


@dataclass
class State:
    waiting_on: WaitingOn
    why: str  # what the verdict says when it is ours, e.g. "our indices sender never pushed a ready chunk"


@dataclass
class Word:
    """A 32-bit counter in a core's L1, and how far it moves per chunk."""

    x: int
    y: int
    addr: int
    unit: int
    value: int = 0  # watcher only: the last read, unwrapped

    @property
    def chunks(self) -> int:
        return self.value // self.unit


@dataclass
class Input:
    """A stream into the chip: H2D on rank 0, D2D from the upstream stage elsewhere."""

    name: str
    A: Word
    C: Word


@dataclass
class Output:
    """A stream out of the chip. On the last and single rank, its layer acks: P, pushed and F are one counter."""

    name: str
    P: Word
    pushed: Word
    F: Word


@dataclass
class Chip:
    chip_uid: int
    r: int
    c: int
    inputs: list[Input]  # in the order the run loop receives them
    outputs: list[Output]  # in the order it sends them
    acks: Word | None  # D2H layer acks: a heartbeat
    dummy: int = 0  # watcher only: Handoff.dummy

    def words(self) -> list[Word]:
        """Every word, in the same order on every chip of the stage."""
        words = [w for i in self.inputs for w in (i.A, i.C)] + [w for o in self.outputs for w in (o.P, o.pushed, o.F)]
        return words if self.acks is None else words + [self.acks]

    @property
    def p_out(self) -> int:
        return self.outputs[0].P.chunks

    @property
    def c_in(self) -> int:
        """Rank 0 counts the dummy it sent as taken in too, once it is out."""
        return min(i.C.chunks for i in self.inputs) + (self.dummy if self.p_out >= 1 else 0)

    def read(self, wraps_from: Chip | None = None) -> bool:
        """Read every word; True if any moved since this chip's last read.

        The counters are uint32 and overflow within a run. Chip (0,0), read every poll, counts its own wraps: each
        word adds what it moved since. The other chips, read once the stage is still, take theirs from chip (0,0):
        a stage's chips exchange data throughout every chunk, so none gets more than a chunk or so away from it,
        and each word is the value nearest chip (0,0)'s.
        """
        before = [w.value for w in self.words()]
        for w, w00 in zip(self.words(), (wraps_from or self).words()):
            raw = lib.read_word_from_device(f"{w.x},{w.y}", w.addr, device_id=self.chip_uid) & MASK32
            if wraps_from is None:
                w.value += (raw - w.value) & MASK32
            else:
                w.value = w00.value + ((raw - w00.value + (1 << 31)) & MASK32) - (1 << 31)
        return [w.value for w in self.words()] != before

    @property
    def state(self) -> State:
        for o in self.outputs:
            if o.P.chunks > o.pushed.chunks == o.F.chunks:
                return State(WaitingOn.SENDER, f"our {o.name} sender never pushed a ready chunk")
        for before, o in zip(self.outputs, self.outputs[1:]):
            if o.P.chunks < before.P.chunks == before.pushed.chunks:
                return State(WaitingOn.OUTBOUND, f"our {before.name} went out, our {o.name} send never ran")
        if self.c_in > self.p_out:
            return State(WaitingOn.COMPUTE, "stuck computing a chunk every chip took")
        # Our next inbound op queues behind a WAIT for our last push (_lease_reclaim).
        if any(o.P.chunks != o.F.chunks for o in self.outputs):
            return State(WaitingOn.DOWNSTREAM, "the next stage has not taken our last chunk")
        nxt = min(self.inputs, key=lambda i: i.C.chunks)
        if nxt.A.chunks > nxt.C.chunks:
            return State(WaitingOn.INBOUND, f"a chunk reached our {nxt.name} input and we never took it")
        return State(WaitingOn.UPSTREAM, "nothing has reached us")

    def counts(self) -> str:
        """This chip's counts, in chunks."""
        parts = [f"{i.name} in A={i.A.chunks} C={i.C.chunks}" for i in self.inputs]
        parts += [f"{o.name} out P={o.P.chunks} pushed={o.pushed.chunks} F={o.F.chunks}" for o in self.outputs]
        if self.acks is not None:
            parts.append(f"acks={self.acks.chunks}")
        if self.dummy:
            parts.append(f"dummy={self.dummy}")
        return f"  chip ({self.r},{self.c}): " + ", ".join(parts)


@dataclass
class Handoff:
    """What HangDetector.arm writes. chips[0] is chip (0,0), the one polled."""

    stage_idx: int
    pid: int
    layers: str
    dummy: int
    timeout_s: float
    first_chunk_timeout_s: float
    poll_s: float
    command: str
    chips: list[Chip]


def load(probe_path: str) -> Handoff:
    with open(probe_path) as f:
        d = json.load(f)
    chips = [
        Chip(
            chip_uid=chip["chip_uid"],
            r=chip["r"],
            c=chip["c"],
            inputs=[Input(i["name"], Word(**i["A"]), Word(**i["C"])) for i in chip["inputs"]],
            outputs=[Output(o["name"], Word(**o["P"]), Word(**o["pushed"]), Word(**o["F"])) for o in chip["outputs"]],
            acks=None if chip["acks"] is None else Word(**chip["acks"]),
            dummy=d["dummy"],
        )
        for chip in d["chips"]
    ]
    return Handoff(**{**d, "chips": chips})


def verdict(chips: list[Chip]) -> str | None:
    """Why our stage is the culprit, or None when it is idle or waiting on a neighbour."""
    took = min(chip.c_in for chip in chips)
    for on in (WaitingOn.SENDER, WaitingOn.OUTBOUND, WaitingOn.COMPUTE, WaitingOn.INBOUND):
        # Compute is judged across chips: a chip that never got its share keeps the minimum down.
        stuck = [c for c in chips if c.state.waiting_on is on and (on is not WaitingOn.COMPUTE or c.p_out < took)]
        if stuck:
            why = stuck[0].state.why
            hit = [c for c in stuck if c.state.why == why]
            where = "every chip" if len(hit) == len(chips) else "chips " + " ".join(f"({c.r},{c.c})" for c in hit)
            return f"{why}, on {where}"
    return None


def run_command(handoff: Handoff, detail: str, lines: str) -> None:
    """Hand the verdict to the configured command. Its environment is the only output channel."""
    subprocess.run(
        handoff.command,
        shell=True,
        env={
            **os.environ,
            "PREFILL_HANG_STAGE": str(handoff.stage_idx),
            "PREFILL_HANG_PID": str(handoff.pid),
            "PREFILL_HANG_DETAIL": detail,
            "PREFILL_HANG_COUNTERS": lines,
        },
    )


def main(probe_path: str) -> int:
    handoff = load(probe_path)
    os.unlink(probe_path)
    chip00 = handoff.chips[0]
    frozen_since: float | None = None
    left_out: list[str] = []
    while True:
        try:
            # kill 0 is just a liveness probe badly named
            os.kill(handoff.pid, 0)
        except ProcessLookupError:
            return 0

        try:
            moved = chip00.read()
        except Exception:
            return 1
        # Until the first real chunk is out, trace capture and cold compiles can take minutes.
        timeout_s = handoff.timeout_s if chip00.p_out >= 1 + handoff.dummy else handoff.first_chunk_timeout_s

        now = time.monotonic()
        if moved:
            frozen_since = None
        elif frozen_since is None:
            frozen_since = now
        elif now - frozen_since >= timeout_s:
            frozen_since = now
            # The rest of the stage stopped with chip (0,0), so one read of each chip shows where it stands.
            for chip in handoff.chips[1:]:
                chip.read(wraps_from=chip00)
            # A word read from the wrong place must not convict: leave out chips more than a chunk off chip (0,0).
            off = [c for c in handoff.chips if abs(c.c_in - chip00.c_in) > 1 or abs(c.p_out - chip00.p_out) > 1]
            coords = [f"({c.r},{c.c})" for c in off]
            if coords != left_out:
                print(f"PREFILL HANG: stage {handoff.stage_idx}: left out chips {' '.join(coords)}", file=sys.stderr)
                left_out = coords
            detail = verdict([c for c in handoff.chips if c not in off])
            if detail:
                lines = chip00.counts()
                print(
                    f"PREFILL HANG: stage {handoff.stage_idx} (layers {handoff.layers}) is the culprit -- "
                    f"{detail}\n{lines}",
                    file=sys.stderr,
                    flush=True,
                )
                if not handoff.command:
                    return 0
                run_command(handoff, detail, lines)
                with contextlib.suppress(ProcessLookupError):
                    os.kill(handoff.pid, signal.SIGKILL)
                return 1

        time.sleep(handoff.poll_s)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
