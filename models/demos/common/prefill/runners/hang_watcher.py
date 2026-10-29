# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Out-of-process watcher for one prefill stage: python hang_watcher.py <probe.json>

Poll the stage's counters on device (0,0); if none has moved for the timeout, read every device and classify.
On a verdict of culprit it prints the verdict and a stuck device's counts to stderr, runs handoff.command,
and SIGKILLs the rank. With no command it exits instead and leaves the rank hung for inspection.

A chunk goes into a device through its inputs and out through its outputs. Five counts in chunks follow it:

    arrived     landed in an input: from the upstream stage, or from the producer on rank 0
    consumed    our inbound op copied it out
    produced    our outbound op handed it to an output's sender; last and single rank: all its acks fired
    pushed      the sender handed it to fabric
    forwarded   the next stage took it

Inputs are listed in the order the run loop receives them and outputs in the order it sends them: the
activation first, then side streams such as GLM-5.3's indices. A device has consumed a chunk once every
input has, and produced it once the first output has. A device of a still stage waits on the first of:

    SENDER       an output's produced > pushed == forwarded: a ready chunk its sender never pushed
    OUTBOUND     an output's produced < the one before's produced == pushed: its send never ran
    COMPUTE      consumed > produced: took a chunk in and never produced it
    DOWNSTREAM   an output's produced > forwarded: the next stage has not taken our last chunk
    INBOUND      the next input's arrived > consumed: a chunk arrived and was never consumed
    UPSTREAM     anything else: idle, or the stage before us is stuck

The next input is the one the run loop waits on: the first with the lowest consumed. COMPUTE also covers
the first output's outbound op: both sit between consumed and produced.

The stage is the culprit if a device waits on SENDER, OUTBOUND or INBOUND, or on COMPUTE having produced
fewer chunks than every device consumed. Waiting on a neighbour is never us: each device queues its next
inbound op behind a WAIT for its previous push (_lease_reclaim), so a stage held back by its downstream
never has a chunk in hand on every device.
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
    """What a device of a still stage waits on; Device.state takes the first that matches, in this order."""

    SENDER = auto()
    OUTBOUND = auto()
    COMPUTE = auto()
    DOWNSTREAM = auto()
    INBOUND = auto()
    UPSTREAM = auto()


# What the verdict says for each state that blames our stage, in the order we take the blame.
BLAME = {
    WaitingOn.SENDER: "a sender never pushed a ready chunk",
    WaitingOn.OUTBOUND: "an outbound op never ran after the output before it went out",
    WaitingOn.COMPUTE: "never produced a chunk every device consumed",
    WaitingOn.INBOUND: "a chunk arrived and was never consumed",
}


@dataclass
class Word:
    """A 32-bit counter in a core's L1, and how far it moves per chunk."""

    x: int
    y: int
    addr: int
    unit: int
    raw: int = 0
    wraps: int = 0

    @property
    def chunks(self) -> int:
        return (self.raw + (self.wraps << 32)) // self.unit


@dataclass
class Input:
    """A stream into the device: H2D on rank 0, D2D from the upstream stage elsewhere."""

    name: str
    arrived: Word
    consumed: Word


@dataclass
class Output:
    """A stream out of the device. On the last and single rank, its layer acks: all three are one counter."""

    name: str
    produced: Word
    pushed: Word
    forwarded: Word


@dataclass
class Device:
    uid: int
    r: int
    c: int
    inputs: list[Input]  # in the order the run loop receives them
    outputs: list[Output]  # in the order it sends them
    acks: Word | None  # D2H layer acks
    dummy: int = 0  # Handoff.dummy

    def words(self) -> list[Word]:
        """Every word, in the same order on every device of the stage."""
        words = [w for i in self.inputs for w in (i.arrived, i.consumed)]
        words += [w for o in self.outputs for w in (o.produced, o.pushed, o.forwarded)]
        return words if self.acks is None else words + [self.acks]

    @property
    def produced(self) -> int:
        return self.outputs[0].produced.chunks

    @property
    def consumed(self) -> int:
        """Rank 0 counts the dummy it sent as consumed too, once it is out."""
        return min(i.consumed.chunks for i in self.inputs) + (self.dummy if self.produced >= 1 else 0)

    def read(self) -> bool:
        """Read every word; True if any moved. The counters are uint32: a word below its last read has wrapped."""
        moved = False
        for w in self.words():
            raw = lib.read_word_from_device(f"{w.x},{w.y}", w.addr, device_id=self.uid) & MASK32
            w.wraps += raw < w.raw
            moved |= raw != w.raw
            w.raw = raw
        return moved

    def borrow_wraps(self, dev00: Device) -> None:
        """Take device (0,0)'s wrap counts, one off where the two straddle a wrap.

        For a device read only at verdict time: a stage's devices exchange data throughout every chunk, so none
        gets more than a chunk or so away from device (0,0).
        """
        for w, w00 in zip(self.words(), dev00.words()):
            w.wraps = w00.wraps + round((w00.raw - w.raw) / (1 << 32))

    @property
    def state(self) -> WaitingOn:
        for o in self.outputs:
            if o.produced.chunks > o.pushed.chunks == o.forwarded.chunks:
                return WaitingOn.SENDER
        for before, o in zip(self.outputs, self.outputs[1:]):
            if o.produced.chunks < before.produced.chunks == before.pushed.chunks:
                return WaitingOn.OUTBOUND
        if self.consumed > self.produced:
            return WaitingOn.COMPUTE
        # Our next inbound op queues behind a WAIT for our last push (_lease_reclaim).
        if any(o.produced.chunks != o.forwarded.chunks for o in self.outputs):
            return WaitingOn.DOWNSTREAM
        nxt = min(self.inputs, key=lambda i: i.consumed.chunks)
        return WaitingOn.INBOUND if nxt.arrived.chunks > nxt.consumed.chunks else WaitingOn.UPSTREAM

    def counts(self) -> str:
        """This device's counts, in chunks."""
        parts = [f"{i.name} in arrived={i.arrived.chunks} consumed={i.consumed.chunks}" for i in self.inputs]
        parts += [
            f"{o.name} out produced={o.produced.chunks} pushed={o.pushed.chunks} forwarded={o.forwarded.chunks}"
            for o in self.outputs
        ]
        if self.acks is not None:
            parts.append(f"acks={self.acks.chunks}")
        if self.dummy:
            parts.append(f"dummy={self.dummy}")
        return f"  device ({self.r},{self.c}): " + ", ".join(parts)


@dataclass
class Handoff:
    """What HangDetector.arm writes. devices[0] is device (0,0), the one polled."""

    stage_idx: int
    pid: int
    layers: str
    dummy: int
    timeout_s: float
    first_chunk_timeout_s: float
    poll_s: float
    command: str
    devices: list[Device]


def load(probe_path: str) -> Handoff:
    with open(probe_path) as f:
        d = json.load(f)
    devices = [
        Device(
            uid=dev["uid"],
            r=dev["r"],
            c=dev["c"],
            inputs=[Input(i["name"], Word(**i["arrived"]), Word(**i["consumed"])) for i in dev["inputs"]],
            outputs=[
                Output(o["name"], Word(**o["produced"]), Word(**o["pushed"]), Word(**o["forwarded"]))
                for o in dev["outputs"]
            ],
            acks=None if dev["acks"] is None else Word(**dev["acks"]),
            dummy=d["dummy"],
        )
        for dev in d["devices"]
    ]
    return Handoff(**{**d, "devices": devices})


def culprits(devices: list[Device]) -> list[Device]:
    """The devices our stage is to blame for, all in one state; none when it is idle or waiting on a neighbour."""
    took = min(d.consumed for d in devices)
    for on in BLAME:
        # Compute is judged across devices: a device that never got its share keeps the minimum down.
        hit = [d for d in devices if d.state is on and (on is not WaitingOn.COMPUTE or d.produced < took)]
        if hit:
            return hit
    return []


def main(probe_path: str) -> int:
    handoff = load(probe_path)
    os.unlink(probe_path)
    dev00 = handoff.devices[0]
    frozen_since: float | None = None
    left_out: list[str] = []
    while True:
        try:
            # kill 0 is just a liveness probe badly named
            os.kill(handoff.pid, 0)
        except ProcessLookupError:
            return 0

        try:
            moved = dev00.read()
        except Exception:
            return 1
        # Until the first real chunk is out, trace capture and cold compiles can take minutes.
        limit_s = handoff.timeout_s if dev00.produced >= 1 + handoff.dummy else handoff.first_chunk_timeout_s

        now = time.monotonic()
        if moved:
            frozen_since = None
        elif frozen_since is None:
            frozen_since = now
        elif now - frozen_since >= limit_s:
            frozen_since = now
            # The rest of the stage stopped with device (0,0), so one read of each device shows where it stands.
            for device in handoff.devices[1:]:
                device.read()
                device.borrow_wraps(dev00)
            # A word read from the wrong place must not convict: leave out devices more than a chunk off device (0,0).
            off = [
                d
                for d in handoff.devices
                if abs(d.consumed - dev00.consumed) > 1 or abs(d.produced - dev00.produced) > 1
            ]
            coords = [f"({d.r},{d.c})" for d in off]
            if coords != left_out:
                print(f"PREFILL HANG: stage {handoff.stage_idx}: left out devices {' '.join(coords)}", file=sys.stderr)
                left_out = coords
            judged = [d for d in handoff.devices if d not in off]
            hit = culprits(judged)
            if hit:
                where = (
                    "every device" if len(hit) == len(judged) else "devices " + " ".join(f"({d.r},{d.c})" for d in hit)
                )
                detail = f"{BLAME[hit[0].state]}, on {where}"
                lines = hit[0].counts()
                print(
                    f"PREFILL HANG: stage {handoff.stage_idx} (layers {handoff.layers}) is the culprit -- "
                    f"{detail}\n{lines}",
                    file=sys.stderr,
                    flush=True,
                )
                if not handoff.command:
                    return 0
                subprocess.run(handoff.command, shell=True)
                with contextlib.suppress(ProcessLookupError):
                    os.kill(handoff.pid, signal.SIGKILL)
                return 1

        time.sleep(handoff.poll_s)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
