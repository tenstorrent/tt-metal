# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side serial work of a speculative round, made cheaper without changing a single result bit.

Every switch is an environment variable, default OFF (the shipped path is untouched when unset):

    DSV41_HOSTOPT_FORCE=1     SpecDecoder.set_force: skip the re-upload when the value is unchanged
    DSV41_HOSTOPT_ENGRAM=1    Engram n-gram hash (numpy, gathers only what is read) + table gather / fp8 dequant (LUT, preallocated buffers, written straight into the packed upload)
    DSV41_HOSTOPT_PACK=1      (with ENGRAM) the two host tensors of the packed upload are built once and refilled in place; sampler params: only the uniforms are rebuilt per round
    DSV41_HOSTOPT_READBACK=1  the read-back is enqueued right behind the trace (hidden under the device time): one synchronize instead of two
    DSV41_HOSTOPT_POST=1      vectorised post-process loop of AdaptiveSpec.run / SpecRunner.run
    DSV41_HOSTOPT_WARM=1      keep the round thread's core out of the deep idle state (idle-priority busy loop pinned to the same core)

Test / benchmark hooks (never needed in production):

    DSV41_HOSTOPT_CYCLE=none,all,allnw   the switch set changes every DSV41_HOSTOPT_WIN rounds (default 4) of a spec pass, cycling through the list ('none', 'all', 'allnw' = all but WARM, or items joined
                                         by '+', e.g. 'engram+pack'): modes are measured interleaved on the SAME token stream (a fresh prefill after a spec pass hangs the compile pass of the next
                                         one, so passes of one process cannot be compared)
    DSV41_HOSTOPT_SAMPLE_WIN=16          rounds alternate greedy / sampled (DSV41_HOSTOPT_SAMPLE=T:top_k:top_p, default 1.0:0:0.95; generator seeded 1234 per pass), like the round timeline's windows
    DSV41_HOSTOPT_DUMP=<path>            at the end of every spec pass write <path>.<n>.json: tokens, accepted counts, per-round (end time, k, mode, sampled) and, with DSV41_HOSTOPT_FULL=1,
                                         every round's block / accepted ids / drafts / confidences (bit-exactness proofs: dsv4-logs/host_opt/compare.py)
"""

import os
import time

import numpy as np
import torch

ITEMS = ("FORCE", "ENGRAM", "PACK", "READBACK", "POST", "WARM")
ACTIVE = {n: os.environ.get(f"DSV41_HOSTOPT_{n}", "0") == "1" for n in ITEMS}
CYCLE = [m for m in os.environ.get("DSV41_HOSTOPT_CYCLE", "").split(",") if m]
WIN = int(os.environ.get("DSV41_HOSTOPT_WIN", "4"))
SAMPLE_WIN = int(os.environ.get("DSV41_HOSTOPT_SAMPLE_WIN", "0"))
CUR_MODE = "env"
_ST = {"armed": False, "n": 0, "gen": None, "t0": 0.0}
_REC, _TIMES = [], []


def on(name):
    return ACTIVE[name]


_WARM = {"proc": None, "mask": None, "cpu": None, "stopped": False}


def warm_start():
    """DSV41_HOSTOPT_WARM=1: keep the core of the round loop out of the deep idle state. The host thread sleeps ~50 ms of every ~55 ms round inside the device wait; on these hosts (AMD EPYC, acpi_idle C2:
    0.8 ms exit latency, the core's caches are lost) every host phase then runs cold (the same Engram hash takes 0.55 ms after a sleep, 0.29 ms after a busy wait, 0.13 ms back to back). A SCHED_IDLE busy
    loop pinned to the same core (a separate process: no GIL, it never runs while the round thread is runnable) keeps the core out of C2. Numerically a no-op."""
    import signal
    import subprocess

    p = _WARM["proc"]
    if p is not None:
        if _WARM["stopped"]:
            os.sched_setaffinity(0, {_WARM["cpu"]})
            os.kill(p.pid, signal.SIGCONT)
            _WARM["stopped"] = False
        return
    with open("/proc/thread-self/stat") as f:
        cpu = int(f.read().rsplit(")", 1)[1].split()[36])  # field 39: the core this thread runs on now
    _WARM["mask"], _WARM["cpu"] = os.sched_getaffinity(0), cpu
    os.sched_setaffinity(0, {cpu})  # the calling thread only
    try:
        _WARM["proc"] = subprocess.Popen(
            [
                "chrt",
                "-i",
                "0",
                "taskset",
                "-c",
                str(cpu),
                "sh",
                "-c",
                f"while kill -0 {os.getpid()} 2>/dev/null; do :; done",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        os.sched_setaffinity(0, _WARM["mask"])


def warm_stop():
    """(pause, not kill: a mode cycle switches it on and off every few rounds; the spinner is gone when the process ends)"""
    import signal

    p = _WARM["proc"]
    if p is not None and not _WARM["stopped"]:
        os.kill(p.pid, signal.SIGSTOP)
        os.sched_setaffinity(0, _WARM["mask"])
        _WARM["stopped"] = True


def _set_mode(mode):
    sel = set(mode.upper().split("+"))
    for n in ITEMS:
        ACTIVE[n] = mode == "all" or (mode == "allnw" and n != "WARM") or n in sel
    (warm_start if ACTIVE["WARM"] else warm_stop)()
    global CUR_MODE
    CUR_MODE = mode


def begin_run(obj):
    """Start of a spec pass (SpecRunner.run / AdaptiveSpec.run)."""
    _ST.update(armed=True, n=0, t0=time.perf_counter())
    _REC.clear()
    _TIMES.clear()
    if SAMPLE_WIN:
        _ST["gen"] = torch.Generator().manual_seed(1234)
    if CYCLE:
        _set_mode(CYCLE[0])
    elif ACTIVE["WARM"]:
        warm_start()


def round_hook(r):
    """Top of SpecRunner._round: the cycle of switch sets / greedy-sampled windows of the benchmark hooks (rounds of a spec pass only: the seeding rounds run with the environment's switches)."""
    if not _ST["armed"] or not (CYCLE or SAMPLE_WIN):
        return
    j = _ST["n"]
    _ST["n"] = j + 1
    if CYCLE:
        m = CYCLE[(j // WIN) % len(CYCLE)]
        if m != CUR_MODE:
            _set_mode(m)
    if SAMPLE_WIN and r.samp is not None:
        if (j // SAMPLE_WIN) % 2:
            t_, k_, p_ = os.environ.get("DSV41_HOSTOPT_SAMPLE", "1.0:0:0.95").split(":")
            r.sample_cfg = dict(temperature=float(t_), top_k=int(k_), top_p=float(p_), gens=_ST["gen"])
        else:
            r.sample_cfg = None


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# bit-exactness / timing test hook
# ---------------------------------------------------------------------------------------------------------------------------------------------------
DUMP = os.environ.get("DSV41_HOSTOPT_DUMP")
FULL = os.environ.get("DSV41_HOSTOPT_FULL") == "1"
_NFLUSH = [0]


def rec(r, k, X, a, mm, d, conf=None):
    if DUMP:
        _TIMES.append((time.perf_counter(), k, CUR_MODE, int(r.sample_cfg is not None)))
        if FULL:
            _REC.append([k, X.tolist(), a.tolist(), mm.tolist(), d.tolist(), None if conf is None else conf.tolist()])


def flush(gen, m_hist, walls=None):
    _ST["armed"] = False
    if DUMP:
        import json

        with open(f"{DUMP}.{_NFLUSH[0]}.json", "w") as f:
            json.dump({"rounds": _REC, "times": _TIMES, "gen": gen, "m_hist": m_hist, "walls": walls, "t0": _ST["t0"]}, f)
        _NFLUSH[0] += 1
        _REC.clear()
        _TIMES.clear()
