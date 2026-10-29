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
import threading
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


_WARM = {"proc": None, "mask": None, "cpu": None, "stopped": False, "tids": set()}


def warm_start():
    """DSV41_HOSTOPT_WARM=1: keep the core of the round loop out of the deep idle state. The host thread sleeps ~50 ms of every ~55 ms round inside the device wait; on these hosts (AMD EPYC, acpi_idle C2:
    0.8 ms exit latency, the core's caches are lost) every host phase then runs cold (the same Engram hash takes 0.55 ms after a sleep, 0.29 ms after a busy wait, 0.13 ms back to back). A SCHED_IDLE busy
    loop pinned to the same core (a separate process: no GIL, it never runs while the round thread is runnable) keeps the core out of C2. Numerically a no-op.
    """
    import signal
    import subprocess

    p = _WARM["proc"]
    if p is not None:
        if _WARM["stopped"]:
            os.sched_setaffinity(0, {_WARM["cpu"]})
            os.kill(p.pid, signal.SIGCONT)
            _WARM["stopped"] = False
            _WARM["tids"] = {threading.get_ident()}
        elif threading.get_ident() not in _WARM["tids"]:
            # (server) a round thread other than the one that started the spinner: pin it to the same core too
            os.sched_setaffinity(0, {_WARM["cpu"]})
            _WARM["tids"].add(threading.get_ident())
            print(
                f"HOSTOPT_WARM: round thread {threading.current_thread().name} (ident {threading.get_ident()}) pinned to cpu {_WARM['cpu']}",
                flush=True,
            )
        return
    with open("/proc/thread-self/stat") as f:
        cpu = int(f.read().rsplit(")", 1)[1].split()[36])  # field 39: the core this thread runs on now
    _WARM["mask"], _WARM["cpu"] = os.sched_getaffinity(0), cpu
    _WARM["tids"] = {threading.get_ident()}
    print(
        f"HOSTOPT_WARM: round thread {threading.current_thread().name} (ident {threading.get_ident()}, pid {os.getpid()}) pinned to cpu {cpu}",
        flush=True,
    )
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
# Engram: hash
# ---------------------------------------------------------------------------------------------------------------------------------------------------
class FastNgramHash:
    """Numpy twin of ``RaggedNgramHash.__call__`` (same state, same cache writes, identical ids). Returns int64 [B, L, n_engram_layers, n_hash_cols] (numpy)."""

    def __init__(self, state):
        self.st = state
        self.tmap = state.token_map.numpy().astype(np.int64)
        self.mult = state.multipliers.numpy().astype(np.int64)  # [layers, ngram]
        self.primes = state.primes.numpy().astype(np.int64)  # [layers, ngram - 1, heads]
        self.offsets = state.offsets.numpy().astype(np.int64)  # [layers, (ngram - 1) * heads]
        self.pad = int(state.pad_id)
        self.dead = int(state.DEAD)
        self.ng = int(state.layout.max_ngram_size)
        self.cache = state.cache.numpy()  # shares the memory of the torch cache: both paths see the same token history
        self._shifts = np.arange(self.ng).reshape(1, 1, self.ng)

    def __call__(self, input_ids, start, rows=None, token_mask=None):
        if token_mask is not None:
            raise NotImplementedError("decode hash has no token mask")
        ids = input_ids.numpy() if isinstance(input_ids, torch.Tensor) else np.asarray(input_ids)
        B, L = ids.shape
        start = (start.numpy() if isinstance(start, torch.Tensor) else np.asarray(start)).reshape(B, 1).astype(np.int64)
        pos = start + np.arange(L, dtype=np.int64).reshape(1, L)  # [B, L]
        if rows is None:
            crow = np.arange(B, dtype=np.int64)
        else:
            crow = (rows.numpy() if isinstance(rows, torch.Tensor) else np.asarray(rows)).astype(np.int64).reshape(B)
        crow = crow.reshape(B, 1, 1)
        cache = self.cache
        cache[crow[:, :, 0], pos] = self.tmap[ids]
        # all look-backs at once: [B, L, ngram]
        idx = np.maximum(pos[:, :, None] - self._shifts, 0)
        src = cache[crow, idx]
        blocked = np.logical_or.accumulate((pos[:, :, None] < self._shifts) | (src == self.dead), axis=2)
        toks = np.where(blocked, self.pad, src)  # [B, L, ngram]
        prod = toks[:, :, None, :] * self.mult  # [B, L, layers, ngram] (int64 wrap-around exactly like torch)
        roll = np.bitwise_xor.accumulate(prod, axis=3)
        out = [
            roll[..., i : i + 1] % self.primes[:, i - 1] for i in range(1, self.ng)
        ]  # python-style modulo, like torch's
        return np.concatenate(out, axis=-1) + self.offsets


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Engram: table gather + dequant
# ---------------------------------------------------------------------------------------------------------------------------------------------------
_LUT16 = None
LUT16_MAX_T = int(
    os.environ.get("DSV41_HOSTOPT_LUT16_MAX_T", "20")
)  # block tokens (B * n) up to which the 16-bit LUT path is used on a cold core (above: torch's multi-threaded conversion wins, see bench_rows.py)
LUT16_MAX_T_WARM = int(
    os.environ.get("DSV41_HOSTOPT_LUT16_MAX_T_WARM", "48")
)  # ... and with DSV41_HOSTOPT_WARM on (a warm core favours the LUT: 0.49 vs 0.81 ms at T=24, 0.71 vs 0.88 ms at T=32)


def _lut16():
    """65536-entry table (e8m0 scale byte << 8 | fp8 e4m3fn byte) -> bf16 bits, built with torch's own conversions and the same float32 multiply + round-to-nearest-even as ``_MappedTable.rows``, so a lookup is bit-identical to
    ``(w.float() * s.float()).to(bfloat16)`` for every byte pair (also NaN / subnormal / overflow)."""
    global _LUT16
    if _LUT16 is None:
        b = torch.arange(256, dtype=torch.uint8)
        prod = b.view(torch.float8_e8m0fnu).float().unsqueeze(1) * b.view(torch.float8_e4m3fn).float().unsqueeze(
            0
        )  # [scale, byte]
        _LUT16 = prod.to(torch.bfloat16).contiguous().view(torch.int16).numpy().reshape(-1).view(np.uint16).copy()
    return _LUT16


class FastRows:
    """Rows of every Engram layer straight into one preallocated bf16 buffer [T, sum(Kin)] (layer-id order = the order ``prepare_packed_inputs`` concatenates).
    Needs the tables in RAM (``in_ram``): the NFS / memmap paths keep the shipped code. Two dequant paths (both bit-identical to the shipped one, checked by bench_rows.py / the round-by-round dumps):
    small T (see LUT16_MAX_T): gather the fp8 rows + scales (numpy take into reused buffers), one 16-bit LUT lookup per element; larger T: torch index_select + float conversion into reused buffers (multi-threaded).
    """

    def __init__(self, host_rows, layer_ids):
        self.hr = host_rows
        self.order = sorted(layer_ids)
        self._bufs = {}

    def usable(self):
        return all(t.in_ram for t in self.hr.tables.values())

    def _buf(self, T, nh):
        lut = T <= (LUT16_MAX_T_WARM if ACTIVE["WARM"] else LUT16_MAX_T)
        b = self._bufs.get((T, lut))
        if b is None:
            R = T * nh
            out = torch.empty(T, len(self.order) * nh * 256, dtype=torch.bfloat16)
            b = dict(out=out, lut=lut)
            if b["lut"]:
                b.update(
                    lut16=_lut16(),
                    w=np.empty((R, 256), np.uint8),
                    s=np.empty((R, 8), np.uint8),
                    s16=np.empty((R, 8), np.uint16),
                    i16=np.empty((R, 8, 32), np.uint16),
                    o16=np.empty((R, 8, 32), np.uint16),
                    dst=out.view(torch.int16).numpy().reshape(T, len(self.order), nh, 8, 32),
                )
            else:
                b.update(
                    w=torch.empty(R, 256, dtype=torch.uint8),
                    s=torch.empty(R, 8, dtype=torch.uint8),
                    wf=torch.empty(R, 8, 32),
                    sf=torch.empty(R, 8, 1),
                    pf=torch.empty(R, 8, 32),
                    dst=out.view(T, len(self.order), nh, 8, 32),
                )
            self._bufs[(T, lut)] = b
        return b

    def __call__(self, hashes, layer_ids=None):
        """hashes: int64 [B, L, n_engram_layers, n_hash_cols] (numpy). -> bf16 torch [B*L, 1, 1, len(order) * Kin] (a view of a reused buffer: valid until the next call)."""
        B, L = hashes.shape[:2]
        T = B * L
        nh = hashes.shape[-1]
        bf = self._buf(T, nh)
        for j, lid in enumerate(self.order):
            e = self.hr.engram.mods[lid]
            flat = np.ascontiguousarray(hashes[:, :, e.layer_hash_index, :]).reshape(-1)
            tab = self.hr.tables[lid]
            if bf["lut"]:
                if not hasattr(tab, "w_u8"):
                    tab.w_u8, tab.s_u8 = tab.w.view(torch.uint8).numpy(), tab.s.view(torch.uint8).numpy()
                np.take(tab.w_u8, flat, axis=0, out=bf["w"], mode="clip")
                np.take(tab.s_u8, flat, axis=0, out=bf["s"], mode="clip")
                np.copyto(bf["s16"], bf["s"])
                np.left_shift(bf["s16"], 8, out=bf["s16"])
                np.bitwise_or(bf["s16"][:, :, None], bf["w"].reshape(-1, 8, 32), out=bf["i16"])
                np.take(bf["lut16"], bf["i16"], out=bf["o16"], mode="clip")
                bf["dst"][:, j] = bf["o16"].view(np.int16).reshape(T, nh, 8, 32)
            else:
                if not hasattr(tab, "w_t"):
                    tab.w_t, tab.s_t = tab.w.view(torch.uint8), tab.s.view(torch.uint8)
                fl = torch.from_numpy(flat)
                torch.index_select(tab.w_t, 0, fl, out=bf["w"])
                torch.index_select(tab.s_t, 0, fl, out=bf["s"])
                bf["wf"].copy_(bf["w"].view(torch.float8_e4m3fn).view(-1, 8, 32))
                bf["sf"].copy_(bf["s"].view(torch.float8_e8m0fnu).view(-1, 8, 1))
                torch.mul(bf["wf"], bf["sf"], out=bf["pf"])
                bf["dst"][:, j].copy_(bf["pf"].view(T, nh, 8, 32))  # float32 -> bf16 round-to-nearest-even
        return bf["out"].view(T, 1, 1, -1)


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# post-process of a round (python lists instead of per-element tensor reads)
# ---------------------------------------------------------------------------------------------------------------------------------------------------
_NAN = float("nan")


def post_round(
    run,
    a,
    mm,
    d,
    X,
    X_cols,
    base,
    done,
    gen,
    emitted,
    ms,
    eos,
    max_new,
    top,
    conf=None,
    newconf=None,
    k=None,
    cal=None,
    draft=True,
):
    """Same state updates as the per-user loop of ``SpecRunner.run`` / ``AdaptiveSpec.run`` (near-tie gaps off: nan), on python lists; the tensors are written once at the end.
    ``X`` [B, W] block tokens for the next round, columns 0 (accepted token) and 1..X_cols (drafts); ``conf``/``newconf``: AdaptiveSpec only (``cal`` collects the calibration records of a drafting round).
    """
    B = base.shape[0]
    a_l, mm_l, d_l = a.tolist(), mm.tolist(), d.tolist()
    done_l, base_l = done.tolist(), base.tolist()
    conf_l = conf.tolist() if cal is not None else None
    base_n, X_n, done_n = (
        base.numpy(),
        X.numpy(),
        done.numpy(),
    )  # views of the torch tensors: element writes without torch dispatch
    conf_n, nc_n = (conf.numpy(), newconf.numpy()) if newconf is not None else (None, None)
    for b in range(B):
        if done_l[b]:
            continue
        mb = mm_l[b]
        if cal is not None:
            cal.append((b, k, conf_l[b], mb))
        toks = a_l[b][: mb + 1]
        if eos is not None and eos in toks:
            toks = toks[: toks.index(eos) + 1]
            done_l[b] = True
        gen[b] += toks
        run.gaps[b] += [_NAN] * len(toks)
        emitted.append(len(toks))
        ms.append(mb)
        run.m_hist[b].append(mb)
        base_l[b] += mb + 1
        base_n[b] = base_l[b]
        X_n[b, 0] = a_l[b][mb]
        if draft:
            X_n[b, 1 : 1 + X_cols] = d_l[b][:X_cols]
        if nc_n is not None:
            conf_n[b] = nc_n[b]
        if len(gen[b]) >= max_new or base_l[b] > top:
            done_l[b] = True
    done_n[:] = done_l


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
            json.dump(
                {"rounds": _REC, "times": _TIMES, "gen": gen, "m_hist": m_hist, "walls": walls, "t0": _ST["t0"]}, f
            )
        _NFLUSH[0] += 1
        _REC.clear()
        _TIMES.clear()
