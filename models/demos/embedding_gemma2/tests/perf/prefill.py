# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill measurement harness for the EmbeddingGemma 2 text encoder (prefill-optimization stage 1).

Contract: doc/prefill/target_contract.json. Batch 1, buckets 128 ... 65536, masked path: a real
input of logical length 3/4 of the bucket (`prefill_refs.logical_len`) padded into the bucket, so a
quarter of the keys are padding. One trace per bucket (`TextEncoder.capture_all`), timed as
`execute_trace` replays on the resident inputs, with the selected precision config.

A plain run is quick: buckets 128, 1024 and 8192, burst only (53 s with a warm cache, measured).
Everything else is opt-in (env, "1" = on); options combine, `-k` picks cases, `--collect-only`
shows the selection. Rough costs with a warm weight / kernel cache (doc/prefill/01-baseline/README.md;
ALL measured as 18.7 min + 11.9 min for its 65536 sustained part, the rest estimated):

    option (env)                               adds                                              ~min
    (none)                                     buckets 128 / 1024 / 8192, burst                   1
    EMBEDDING_GEMMA2_PREFILL_SHAPES=1          all 10 contract buckets (16384 / 32768 / 65536     +4
                                               each in its own device session)
    EMBEDDING_GEMMA2_PREFILL_HF_CHECK=1        gate per bucket on the traced replay output vs     +1 (cached refs)
                                               HF fp32 (last-hidden PCC, embedding cosine), and
                                               replay vs eager on the same input; builds missing
                                               references first (prefill_refs.py; 65536: 4.3 min,
                                               44.5 GB host RAM)
    EMBEDDING_GEMMA2_PREFILL_HOST=1            host prep, H2D, D2H (embedding, last hidden),      +1
                                               host round trip per bucket
    EMBEDDING_GEMMA2_PREFILL_SUSTAINED=1       sustained: >= 30 s and >= 300 replays back to      +3 (quick buckets)
                                               back per bucket; 65536 alone takes 10.4 min        +25 (all buckets)
    EMBEDDING_GEMMA2_PREFILL_TELEMETRY=1       AICLK / power / current / vcore / temperature
                                               sampled every 100 ms in a second process during
                                               sustained runs (sysfs, prefill_telemetry.py)       +0
    EMBEDDING_GEMMA2_PREFILL_ALL=1             all of the above: the stage-1 baseline sweep       ~30

Phases run per device session in this order: gate (all buckets of the session), burst, host,
sustained, so every burst runs before any sustained load of its session.

Definitions (records in $EMBEDDING_GEMMA2_PREFILL_OUT/prefill.jsonl, default
generated/embedding_gemma2_prefill/):

    burst      BURST_REPS (5) windows, each after BURST_IDLE_S (3 s) idle, of n back-to-back
               replays and one sync (n = 200 ms / forward, clamped to 3 ... 20); ms per forward per
               window; reported avg (mean of windows), best (min), median
    sustained  back-to-back replays in windows of ~500 ms (>= 1 replay), one sync per window, for
               max(30 s, 300 replays); reported median of the windows in the last third of the
               run, plus first window and mean of all; telemetry summarized over the same last third
    gate       replay output after loading the real input: last-hidden PCC over the valid rows and
               embedding cosine vs the cached HF fp32 reference (bars 0.99 / 0.99); replay vs eager
               (same input, eager run before capture) PCC / cosine
    host       host_prep (pad + host tensors), h2d (copy ids + keep into the trace inputs + sync),
               d2h_embedding ([1, 768] to torch), d2h_last_hidden ([1, S, 768] to torch, sliced),
               round_trip (h2d + execute_trace + embedding readback; the request a caller sees
               after host prep)

Grid: 12 x 10 compute grid by default (contract compute_grid, Galaxy target; on a P150a the device
is opened with TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=11,9, see tt/config.py), recorded as
`compute_grid` in every record. EMBEDDING_GEMMA2_GRID=13x10 runs on the P150 grid. The override is
read once per process: run this file in its own pytest process.

Other env: EMBEDDING_GEMMA2_PREFILL_BUCKETS (comma list, overrides the bucket selection),
EMBEDDING_GEMMA2_PREFILL_REFS (reference cache, default generated/embedding_gemma2_prefill/refs),
EMBEDDING_GEMMA2_PREFILL_TRACE_MB (trace region, default 128), EMBEDDING_GEMMA2_SUSTAINED_S (30),
EMBEDDING_GEMMA2_PREFILL_SUSTAINED_REPLAYS (300), EMBEDDING_GEMMA2_PREFILL_BURST_IDLE_S (3),
EMBEDDING_GEMMA2_PREFILL_DUMP (dir: the gate phase saves the replay outputs per bucket, +0 min),
EMBEDDING_GEMMA2_PREFILL_FULL_INPUT=1 (all-valid input: logical length = the bucket, no key padding; for
A/Bs of padding-dependent paths such as the E9.4 padding-aware block map; burst / host / sustained only,
the HF gate needs the contract input). Every record carries `logical_len` and `valid_fraction`.
Model switches (EG2_*, e.g. the E9.2 topology A/B switches in tt/masks.py, tt/per_layer_input.py,
tt/pooler.py) are recorded in run_info `model_env`.
Profiling (one warmed eager forward per bucket between signposts) is perf.py::test_tracy_forward.

    pytest models/demos/embedding_gemma2/tests/perf/prefill.py -s                    # quick
    EMBEDDING_GEMMA2_PREFILL_ALL=1 pytest models/demos/embedding_gemma2/tests/perf/prefill.py -s
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.embedding_gemma2.conftest import DEVICE_PARAMS, MODEL_ID, open_eg2_device
from models.demos.embedding_gemma2.tests.common import COSINE_EMBEDDING, make_batch, pcc
from models.demos.embedding_gemma2.tests.perf import prefill_telemetry as telemetry
from models.demos.embedding_gemma2.tests.perf.perf import _git, _stats
from models.demos.embedding_gemma2.tests.perf.prefill_refs import load_ref, logical_len
from models.demos.embedding_gemma2.tt.config import GALAXY_GRID, GRID_ENV, requested_grid
from models.demos.embedding_gemma2.tt.weights import resolve_snapshot

MODEL_DIR = Path(__file__).resolve().parents[2]
TT_METAL_ROOT = MODEL_DIR.parents[2]
OUT_DIR = Path(
    os.environ.get("EMBEDDING_GEMMA2_PREFILL_OUT", str(TT_METAL_ROOT / "generated" / "embedding_gemma2_prefill"))
)
REFS_DIR = Path(os.environ.get("EMBEDDING_GEMMA2_PREFILL_REFS", str(OUT_DIR / "refs")))

CONTRACT_BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
QUICK_BUCKETS = (128, 1024, 8192)
PCC_LAST_HIDDEN = 0.99  # contract gate (doc/prefill/target_contract.json)
N_WARMUP = 3
BURST_REPS = 5
BURST_IDLE_S = float(os.environ.get("EMBEDDING_GEMMA2_PREFILL_BURST_IDLE_S", "3"))
BURST_TARGET_MS, BURST_MIN_N, BURST_MAX_N = 200.0, 3, 20
SUSTAINED_S = float(os.environ.get("EMBEDDING_GEMMA2_SUSTAINED_S", "30"))
SUSTAINED_REPLAYS = int(os.environ.get("EMBEDDING_GEMMA2_PREFILL_SUSTAINED_REPLAYS", "300"))
SUSTAINED_WINDOW_MS = 500.0
HOST_N, HOST_BUDGET_MS = 20, 3000.0
TRACE_MB = int(os.environ.get("EMBEDDING_GEMMA2_PREFILL_TRACE_MB", "128"))
# Gate phase only: save the traced replay outputs (embedding, valid last-hidden rows) per bucket to
# <dir>/b1_s<bucket>.pt, for bitwise comparisons between code paths (E9.2 A/B).
_dump = os.environ.get("EMBEDDING_GEMMA2_PREFILL_DUMP", "").strip()
DUMP_DIR = Path(_dump) if _dump else None
RUN_ID = time.strftime("%Y%m%d-%H%M%S")
# Compute grid (contract compute_grid [12, 10], the Galaxy target): EMBEDDING_GEMMA2_GRID overrides
# (e.g. 13x10 for the P150 grid). The device is opened on it and checked (conftest.open_eg2_device).
GRID = requested_grid(GALAXY_GRID)
_DEVICE_GRID: list[int] | None = None  # what the open device reports; in every record


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on", "all")


PREFILL_ALL = _flag("EMBEDDING_GEMMA2_PREFILL_ALL")
SEL = {
    k: PREFILL_ALL or _flag(f"EMBEDDING_GEMMA2_PREFILL_{k.upper()}")
    for k in ("shapes", "hf_check", "host", "sustained", "telemetry")
}
_env_buckets = os.environ.get("EMBEDDING_GEMMA2_PREFILL_BUCKETS", "").strip()
BUCKETS = (
    tuple(int(b) for b in _env_buckets.split(","))
    if _env_buckets
    else (CONTRACT_BUCKETS if SEL["shapes"] else QUICK_BUCKETS)
)
assert all(b in CONTRACT_BUCKETS for b in BUCKETS), f"buckets must be contract buckets: {BUCKETS}"
# all-valid input (module doc); the contract input is 3/4 valid (prefill_refs.logical_len)
FULL_INPUT = _flag("EMBEDDING_GEMMA2_PREFILL_FULL_INPUT")
assert not (FULL_INPUT and SEL["hf_check"]), "the HF gate runs on the contract input: drop FULL_INPUT or HF_CHECK"


def input_len(bucket: int) -> int:
    """Logical length of the harness input in `bucket`: the contract's 3/4, or the bucket (FULL_INPUT)."""
    return bucket if FULL_INPUT else logical_len(bucket)


def session_key(bucket: int) -> int:
    """Device session (= the TextEncoder max_seq_len) a bucket runs in: <= 8192 share one session."""
    return max(bucket, 8192)


PHASES = ("gate",) * SEL["hf_check"] + ("burst",) + ("host",) * SEL["host"] + ("sustained",) * SEL["sustained"]
CASES = [
    (b, ph)
    for sk in sorted({session_key(b) for b in BUCKETS})
    for ph in PHASES
    for b in BUCKETS
    if session_key(b) == sk
]


# ---------------------------------------------------------------------- records
def _emit(rec: dict) -> dict:
    rec = {
        "run_id": RUN_ID,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "compute_grid": _DEVICE_GRID or list(GRID),
        **rec,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "prefill.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    brief = {k: v for k, v in rec.items() if k not in ("series", "samples_ms", "telemetry_rows", "windows_ms")}
    logger.info(f"PREFILL {json.dumps(brief)}")
    return rec


@pytest.fixture(scope="session", autouse=True)
def prefill_run_info():
    from models.demos.embedding_gemma2.tt.precision import default_precision_path

    try:
        board = telemetry.identity()
        board["idle"] = telemetry.read()
    except OSError as err:
        board = {"error": str(err)}
    info = {
        "run_id": RUN_ID,
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": [ln for ln in _git("status", "--porcelain").splitlines() if ln],
        "model_id": MODEL_ID,
        "precision_file": str(default_precision_path()),
        "selection": {"all": PREFILL_ALL, **SEL, "buckets": list(BUCKETS), "phases": list(PHASES)},
        "input": "all-valid (FULL_INPUT)" if FULL_INPUT else "contract: 3/4 valid, padded into the bucket",
        "device_params": {**DEVICE_PARAMS, "trace_region_size": TRACE_MB * 1024 * 1024},
        "compute_grid_request": list(GRID),
        "grid_env": os.environ.get(GRID_ENV, ""),
        "burst": {"reps": BURST_REPS, "idle_s": BURST_IDLE_S, "target_ms": BURST_TARGET_MS},
        "sustained": {"min_s": SUSTAINED_S, "min_replays": SUSTAINED_REPLAYS, "window_ms": SUSTAINED_WINDOW_MS},
        "board": board,
        "trace_alloc_tracking": os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING", ""),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hostname": platform.node(),
        "env": {k: os.environ.get(k, "") for k in ("TT_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "OMP_NUM_THREADS")},
        # model switches (EG2_*, e.g. the E9.2 topology A/B switches); empty = code defaults
        "model_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("EG2_")},
        "dump_dir": str(DUMP_DIR) if DUMP_DIR else "",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"run_info_{RUN_ID}.json").write_text(json.dumps(info, indent=2) + "\n")
    logger.info(f"prefill run {RUN_ID}: commit {info['commit']} buckets {BUCKETS} phases {PHASES}")
    return info


# ---------------------------------------------------------------------- device session
class Session:
    """One device + TextEncoder (max_seq_len = session key) with one trace per bucket of the session."""

    def __init__(self, key: int, buckets: list[int]):
        from models.demos.embedding_gemma2.tt.text_encoder import TextEncoder

        self.key, self.buckets = key, buckets
        self.snapshot = resolve_snapshot(MODEL_ID)
        self.refs = {}
        if SEL["hf_check"]:
            for b in buckets:  # before the device opens: references are host-RAM heavy
                self.refs[b] = _get_ref(b)
        global _DEVICE_GRID
        self.device = open_eg2_device(0, grid=GRID, trace_region_size=TRACE_MB * 1024 * 1024)
        try:
            g = self.device.compute_with_storage_grid_size()
            _DEVICE_GRID = [g.x, g.y]
            t0 = time.perf_counter()
            self.enc = TextEncoder.from_pretrained(self.device, str(self.snapshot), max_seq_len=key)
            self.raw = {b: make_batch(self.snapshot, [input_len(b)]) for b in buckets}
            self.hosts = {b: self.enc.prepare_inputs(*self.raw[b]) for b in buckets}
            for b, h in self.hosts.items():
                assert (h.batch, h.seq_len, h.logical_seq_len) == (1, b, input_len(b))
            self.eager = {}
            if SEL["hf_check"]:  # eager on the same real input, before any capture
                for b in buckets:
                    emb, last = self.enc.encode(*self.raw[b], return_last_hidden=True)
                    self.eager[b] = (emb, last)
            self.enc.capture_all([(1, b) for b in buckets])
            self.setup_s = time.perf_counter() - t0
        except Exception:
            ttnn.close_device(self.device)
            raise

    def load_inputs(self, b):
        """Copy the real input into the trace inputs and replay once; returns (embedding, last hidden)."""
        last, emb = self.enc.replay(self.hosts[b])
        e = self.enc.read_embedding(emb, 1)
        h = self.enc.read_last_hidden(last, 1, input_len(b))
        return e, h

    def execute(self, b):
        self.enc.execute_captured(1, b)

    def sync(self):
        ttnn.synchronize_device(self.device)

    def close(self):
        try:
            self.enc.release_traces()
        finally:
            ttnn.close_device(self.device)


def _get_ref(b: int):
    L = logical_len(b)
    ref = load_ref(REFS_DIR, b, L)
    if ref is None:
        logger.info(f"building HF fp32 reference for bucket {b} (L={L}) into {REFS_DIR}")
        cmd = [sys.executable, "-m", "models.demos.embedding_gemma2.tests.perf.prefill_refs"]
        subprocess.run(cmd + ["--bucket", str(b), "--out", str(REFS_DIR)], check=True)
        ref = load_ref(REFS_DIR, b, L)
    return ref


@pytest.fixture(scope="module")
def sessions():
    holder = {"cur": None}

    def get(bucket):
        key = session_key(bucket)
        cur = holder["cur"]
        if cur is None or cur.key != key:
            if cur is not None:
                cur.close()
                holder["cur"] = None
            holder["cur"] = Session(key, [b for b in BUCKETS if session_key(b) == key])
            _emit(
                {
                    "phase": "session",
                    "session_key": key,
                    "buckets": holder["cur"].buckets,
                    "setup_s": holder["cur"].setup_s,
                    "status": "ok",
                }
            )
        return holder["cur"]

    yield get
    if holder["cur"] is not None:
        holder["cur"].close()


# ---------------------------------------------------------------------- phases
def _window(s: Session, b: int, n: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        s.execute(b)
    s.sync()
    return (time.perf_counter() - t0) * 1e3 / n


def _base(b):
    L = input_len(b)
    return {"shape": f"b1_s{b}", "batch": 1, "seq": b, "logical_len": L, "valid_fraction": L / b, "path": "masked"}


def phase_gate(s: Session, b: int):
    emb, last = s.load_inputs(b)
    ref = s.refs[b]
    L = logical_len(b)
    assert last.shape[-2] == L and ref["last_hidden"].shape[0] == L
    assert torch.equal(
        s.raw[b][0][0].to(torch.int32), ref["input_ids"]
    ), "reference input differs from the harness input"
    p_last = pcc(ref["last_hidden"], last[0])
    cos = float(torch.nn.functional.cosine_similarity(emb[0], ref["embedding"], dim=0))
    e_emb, e_last = s.eager[b]
    p_eager = pcc(e_last[0], last[0])
    cos_eager = float(torch.nn.functional.cosine_similarity(emb[0], e_emb[0].float(), dim=0))
    finite = bool(torch.isfinite(last).all() and torch.isfinite(emb).all())
    good = finite and p_last >= PCC_LAST_HIDDEN and cos >= COSINE_EMBEDDING
    if DUMP_DIR is not None:
        DUMP_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"embedding": emb, "last_hidden": last, "run_id": RUN_ID}, DUMP_DIR / f"b1_s{b}.pt")
    _emit(
        {
            **_base(b),
            "phase": "gate",
            "status": "pass" if good else "fail",
            "last_hidden_pcc": p_last,
            "embedding_cosine": cos,
            "bars": [PCC_LAST_HIDDEN, COSINE_EMBEDDING],
            "replay_vs_eager_last_hidden_pcc": p_eager,
            "replay_vs_eager_embedding_cosine": cos_eager,
            "replay_equals_eager_bitwise": bool(torch.equal(e_last, last) and torch.equal(e_emb, emb)),
            "reference": {"length": ref["length"], **ref["meta"]},
        }
    )
    assert good, f"b1_s{b}: last-hidden PCC {p_last:.6f}, embedding cosine {cos:.6f} (bars 0.99)"


def phase_burst(s: Session, b: int):
    emb, _ = s.load_inputs(b)
    assert torch.isfinite(emb).all()
    est = _window(s, b, N_WARMUP)
    n = min(BURST_MAX_N, max(BURST_MIN_N, math.ceil(BURST_TARGET_MS / est)))
    tel_start = telemetry.read()
    windows = []
    for _ in range(BURST_REPS):
        time.sleep(BURST_IDLE_S)
        windows.append(_window(s, b, n))
    tel_end = telemetry.read()
    tok = b / (statistics.mean(windows) / 1e3)
    _emit(
        {
            **_base(b),
            "phase": "burst",
            "status": "ok",
            "n_per_window": n,
            "windows": BURST_REPS,
            "avg_ms": statistics.mean(windows),
            "best_ms": min(windows),
            "median_ms": statistics.median(windows),
            "worst_ms": max(windows),
            "windows_ms": windows,
            "tokens_per_s_per_chip": tok,
            "telemetry_start": tel_start,
            "telemetry_end": tel_end,
        }
    )


def phase_host(s: Session, b: int):
    enc, host = s.enc, s.hosts[b]
    tr = enc._traces[(1, b)]
    s.load_inputs(b)
    est = _window(s, b, N_WARMUP)
    n = min(HOST_N, max(5, math.ceil(HOST_BUDGET_MS / est)))

    def timed(fn, reps):
        out = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            out.append((time.perf_counter() - t0) * 1e3)
        return _stats(out)

    def h2d():
        ttnn.copy_host_to_device_tensor(host.input_ids, tr.input_ids)
        ttnn.copy_host_to_device_tensor(host.keep, tr.keep)
        s.sync()

    def round_trip():
        _, emb = enc.replay(host)
        enc.read_embedding(emb, 1)

    s.execute(b)
    s.sync()
    rec = {
        "host_prep": timed(lambda: enc.prepare_inputs(*s.raw[b]), HOST_N),
        "h2d": timed(h2d, HOST_N),
        "d2h_embedding": timed(lambda: enc.read_embedding(tr.embedding, 1), HOST_N),
        "d2h_last_hidden": timed(lambda: enc.read_last_hidden(tr.last_hidden, 1, input_len(b)), min(HOST_N, n)),
        "round_trip": timed(round_trip, n),
        "replay_sync": timed(lambda: (s.execute(b), s.sync()), n),
    }
    nbytes = {
        "h2d_bytes": host.input_ids.volume() * host.input_ids.element_size()
        + host.keep.volume() * host.keep.element_size(),
        "d2h_embedding_bytes": tr.embedding.volume() * tr.embedding.element_size(),
        "d2h_last_hidden_bytes": tr.last_hidden.volume() * tr.last_hidden.element_size(),
    }
    _emit({**_base(b), "phase": "host", "status": "ok", "n_round_trip": n, **rec, **nbytes})


def phase_sustained(s: Session, b: int):
    emb, _ = s.load_inputs(b)
    assert torch.isfinite(emb).all()
    est = _window(s, b, N_WARMUP)
    k = max(1, math.ceil(SUSTAINED_WINDOW_MS / est))
    duration = max(SUSTAINED_S, SUSTAINED_REPLAYS * est / 1e3)
    time.sleep(BURST_IDLE_S)
    tel_path = OUT_DIR / f"telemetry_{RUN_ID}_b1_s{b}.csv"
    sampler = telemetry.Sampler(tel_path) if SEL["telemetry"] else None
    if sampler:
        sampler.__enter__()
    try:
        series, replays = [], 0
        wall0 = time.time()
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < duration or replays < SUSTAINED_REPLAYS:
            t = time.perf_counter() - t_start
            series.append((t, _window(s, b, k)))
            replays += k
        total_s = time.perf_counter() - t_start
    finally:
        if sampler:
            sampler.__exit__(None, None, None)
    last_third = [ms for t, ms in series if t >= total_s * 2 / 3] or [series[-1][1]]
    med = statistics.median(last_third)
    tel = {}
    if sampler:
        rows = [r for r in sampler.rows() if wall0 <= r["t"] <= wall0 + total_s]
        tail = [r for r in rows if r["t"] >= wall0 + total_s * 2 / 3]
        if tail:
            tel = {
                "samples": len(rows),
                "period_ms": sampler.period_ms,
                "csv": str(tel_path),
                "last_third": {
                    f: {
                        "median": statistics.median(r[f] for r in tail),
                        "min": min(r[f] for r in tail),
                        "max": max(r[f] for r in tail),
                    }
                    for f in ("aiclk_mhz", "power_w", "current_a", "vcore_v", "asic_temp_c")
                },
                "first_second": {
                    f: statistics.median(r[f] for r in rows if r["t"] < wall0 + 1.0) for f in ("aiclk_mhz", "power_w")
                }
                if any(r["t"] < wall0 + 1.0 for r in rows)
                else {},
            }
    _emit(
        {
            **_base(b),
            "phase": "sustained",
            "status": "ok",
            "window_replays": k,
            "replays": replays,
            "duration_s": total_s,
            "median_last_third_ms": med,
            "first_window_ms": series[0][1],
            "mean_all_ms": total_s * 1e3 / replays,
            "windows_last_third": len(last_third),
            "tokens_per_s_per_chip": b / (med / 1e3),
            "telemetry": tel,
            "series": series,
        }
    )


PHASE_FNS = {"gate": phase_gate, "burst": phase_burst, "host": phase_host, "sustained": phase_sustained}


# the repo default (pytest.ini timeout = 300 s) is shorter than a 65536 sustained run (300 x ~2.1 s) plus
# a long-bucket session setup
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("bucket,phase", CASES, ids=[f"b1_s{b}-{ph}" for b, ph in CASES])
def test_prefill(sessions, bucket, phase):
    PHASE_FNS[phase](sessions(bucket), bucket)
