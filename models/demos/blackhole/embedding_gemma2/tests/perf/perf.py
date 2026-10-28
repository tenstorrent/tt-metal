# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""EmbeddingGemma 2 benchmark on one Blackhole P150a: warmed, traced latency and throughput of every
tower (text, vision, audio) and of the full multimodal model, with the selected precision config
(`doc/datatype_sweep/selected_precision_config.json`, or the file named by EMBEDDING_GEMMA2_PRECISION).
Report: doc/benchmark/REPORT.md. Layout follows models/demos/wormhole/bge_m3/tests/perf/perf.py.

A plain run is quick: headline shapes, `forward` mode only (text 1 x 512, vision 1 x 2560, audio 1 x 768,
full text 472 tokens), finite-output asserts, no HF check. Everything else is opt-in (env, "1" = on);
costs are from the full-sweep run 20261028-063930 (doc/benchmark/REPORT.md, model loads included):

    option (env)                        adds                                                    ~min
    (none)                              headline shapes x forward                                1.5
    EMBEDDING_GEMMA2_PERF_SHAPES=1      every shape of TEXT / VISION / AUDIO / FULL_CASES        +0.5
    EMBEDDING_GEMMA2_PERF_MODES=1       h2d_d2h, api (full), 2cq (records "not supported")      +0.3
    EMBEDDING_GEMMA2_PERF_LONG=1        text 1 x 32768, own device session                      +0.8
    EMBEDDING_GEMMA2_PERF_PROBE=1       text batch-capacity probe at 128 tokens                  +0.8
    EMBEDDING_GEMMA2_PERF_SUSTAINED=1   30 s sustained vs burst per tower (text, vision, audio)  +2.5
    EMBEDDING_GEMMA2_PERF_HF_CHECK=1    HF fp32 comparison on the timed output (CPU reference)  +0.5
    EMBEDDING_GEMMA2_PERF_ALL=1         all of the above: the full sweep of REPORT.md            5.5

Options combine (e.g. SHAPES=1 MODES=1 gives every shape in every mode); `-k` picks single cases
among the collected ones. The selection decides collection, so `--collect-only` shows what will run.

Full sweep (batch x physical bucket; one trace per shape, all shapes of a group captured by one
capture_all):

    text    b1 x 128, 512, 2048, 8192; b8 x 512; b32 x 128 (TextEncoder, max_seq_len 8192)
            b1 x 32768 in its own device session (max_seq_len 32768)
            batch probe: eager b {32 .. 8192} x 128, ascending, until the first device allocation failure
    vision  1 x 2560 (landscape), 4 x 2560 (landscape, portrait, square, small), 8 x 1280 (8 video
            frames); patch buckets as in bench_vision.py (2560 = one image, 1280 = one video frame)
    audio   1 x 256 (10 s speech), 1 x 768 (30 s speech), 4 x 256 (4 x 10 s); token buckets as in
            bench_audio.py
    full    EmbeddingGemma2 public API: text only (472 tokens, bucket 512), image + text
            (`image_dog_caption`), audio + text (`audio_speech10s_text`); inputs from
            reference/full_model_inputs.py, the catalogue of the demo and test_full_model.py

Modes (timing boundaries; `mode` in the records):

    forward    trace replay only: execute_trace + synchronize_device per iteration, persistent inputs
               already resident (device time + one host sync per replay). Also logged per case:
               `forward_pipelined`, N back-to-back replays and one sync (mean only; the E5 / E6
               bench protocol)
    h2d_d2h    one request: copy the prepared host input tensors into the trace inputs, replay, read
               the output to torch (text / full: the [B, 768] embedding; vision / audio: the soft
               tokens, padding rows stripped). Host input preparation (padding, RoPE tables, masks,
               merge index) is timed separately as `host_prep` and is not inside h2d_d2h
    api        full model only: `EmbeddingGemma2.encode([item], traced=True)`, the public call,
               including the HF processor (tokenizer, image patchify, log-mel), batch preparation,
               upload, replay and readback
    2cq        not supported (NO_2CQ); the case records the reason and skips
    sustained  per tower at its b1 headline shape (text 1 x 512, vision 1 x 2560, audio 1 x 768):
               burst = median of 5 windows of 20 back-to-back replays, each after 3 s idle;
               sustained = 30 s of back-to-back replays in ~0.5 s windows (P150a current-limit
               throttling, see doc/optimized_text/README.md)

Every timed case: N_WARMUP >= 3 replays first, then N timed iterations (default 20); mean, p50, p90,
p99, min, max (numpy linear percentiles; with N = 20, p99 is close to max). Throughput: text tokens/s
and embeddings/s; vision / audio items/s (images, video frames, clips); full embeddings/s and
tokens/s (logical tokens). Checks: every case asserts a finite output (full: unit norm). With
EMBEDDING_GEMMA2_PERF_HF_CHECK=1, for one shape per tower (text 1 x 512, vision 1 x 2560, audio
1 x 768) and every selected full case, the output of the timed replay path is compared with HF fp32
(text / full: embedding cosine >= 0.99; vision / audio: soft-token PCC >= 0.99, the tower tests' bar,
and flattened soft-token cosine >= 0.99).

Raw records: doc/benchmark/run/perf.jsonl (with per-iteration samples) and perf.csv (summary), one
`run_id` per pytest session; doc/benchmark/run/run_info.json (commit, precision file hash, versions)
and device_info.json.

Other env: EMBEDDING_GEMMA2_PERF_ITERS (20), EMBEDDING_GEMMA2_PERF_WARMUP (3, minimum 3),
EMBEDDING_GEMMA2_SUSTAINED_S (30), EMBEDDING_GEMMA2_BENCH_DIR (doc/benchmark/run).

Run (from the tt-metal root, HF_MODEL = the pinned snapshot, transformers overlay on PYTHONPATH):

    pytest models/demos/blackhole/embedding_gemma2/tests/perf/perf.py -s                    # quick
    EMBEDDING_GEMMA2_PERF_ALL=1 pytest models/demos/blackhole/embedding_gemma2/tests/perf/perf.py -s
    EMBEDDING_GEMMA2_PERF_SHAPES=1 pytest models/demos/blackhole/embedding_gemma2/tests/perf/perf.py -k TestVision -s

test_tracy_forward (text, one warmed eager forward between Tracy signposts PERF_TEXT / PERF_TEXT_END)
runs only under the device profiler (`python -m tracy -r -p -v -m pytest ...`, see
doc/optimized_text/README.md).
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.embedding_gemma2.conftest import DEVICE_PARAMS, MODEL_ID
from models.demos.blackhole.embedding_gemma2.tests.common import COSINE_EMBEDDING, make_batch, pcc
from models.demos.blackhole.embedding_gemma2.tt.weights import resolve_snapshot

try:
    from tracy import signpost
except ImportError:  # tracy is optional outside profiling runs

    def signpost(*_args, **_kwargs):
        pass


MODEL_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = Path(os.environ.get("EMBEDDING_GEMMA2_BENCH_DIR", str(MODEL_DIR / "doc" / "benchmark" / "run")))
N_WARMUP = max(3, int(os.environ.get("EMBEDDING_GEMMA2_PERF_WARMUP", "3")))
N_ITERS = int(os.environ.get("EMBEDDING_GEMMA2_PERF_ITERS", "20"))
SUSTAINED_S = float(os.environ.get("EMBEDDING_GEMMA2_SUSTAINED_S", "30"))
BURST_IDLE_S, BURST_REPS, BURST_N = 3.0, 5, 20
SUSTAINED_WINDOW_MS = 500.0
SOFT_TOKEN_PCC = 0.99  # tower tests' soft-token bar

TEXT_CASES = ((1, 128), (1, 512), (1, 2048), (1, 8192), (8, 512), (32, 128))
TEXT_LONG_CASES = ((1, 32768),)
TEXT_PROBE_SEQ = 128
TEXT_PROBE_BATCHES = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
VISION_CASES = ((1, 2560), (4, 2560), (8, 1280))
AUDIO_CASES = ((1, 256), (1, 768), (4, 256))
FULL_TEXT_CHARS = 1200  # DOC prefix + the first 1200 chars of llms.md -> 472 tokens (bucket 512)
FULL_CASES = ("text_b1x512", "image_text", "audio_text")
FULL_ITEMS = {"image_text": "image_dog_caption", "audio_text": "audio_speech10s_text"}
HEADLINE = {"text": (1, 512), "vision": (1, 2560), "audio": (1, 768)}  # sustained run + HF check
MODES = ("forward", "h2d_d2h", "2cq")
FULL_MODES = ("forward", "h2d_d2h", "api", "2cq")
NO_2CQ = (
    "not supported: TextEncoder / VisionEncoder / AudioEncoder / EmbeddingGemma2 copy inputs and replay "
    "on command queue 0 only (cq_id=0 in capture_all / replay) and the device opens with "
    "num_command_queues=1; a second-queue input path is a code change outside the benchmark stage"
)
RUN_ID = time.strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------- selection (see the docstring)
def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on", "all")


PERF_ALL = _flag("EMBEDDING_GEMMA2_PERF_ALL")
SEL = {
    "shapes": PERF_ALL or _flag("EMBEDDING_GEMMA2_PERF_SHAPES"),
    "modes": PERF_ALL or _flag("EMBEDDING_GEMMA2_PERF_MODES"),
    "long": PERF_ALL or _flag("EMBEDDING_GEMMA2_PERF_LONG"),
    "probe": PERF_ALL or _flag("EMBEDDING_GEMMA2_PERF_PROBE"),
    "sustained": PERF_ALL or _flag("EMBEDDING_GEMMA2_PERF_SUSTAINED"),
    "hf_check": PERF_ALL or _flag("EMBEDDING_GEMMA2_PERF_HF_CHECK"),
}
PROFILER = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
RUN_TEXT = TEXT_CASES if SEL["shapes"] else (HEADLINE["text"],)
RUN_VISION = VISION_CASES if SEL["shapes"] else (HEADLINE["vision"],)
RUN_AUDIO = AUDIO_CASES if SEL["shapes"] else (HEADLINE["audio"],)
RUN_FULL = FULL_CASES if SEL["shapes"] else ("text_b1x512",)
RUN_MODES = MODES if SEL["modes"] else ("forward",)
RUN_FULL_MODES = FULL_MODES if SEL["modes"] else ("forward",)

CSV_FIELDS = (
    "run_id",
    "tower",
    "case",
    "batch",
    "seq_len",
    "mode",
    "n_warmup",
    "n_iters",
    "mean_ms",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "min_ms",
    "max_ms",
    "tokens_per_s",
    "embeddings_per_s",
    "items_per_s",
    "status",
    "note",
)


# ---------------------------------------------------------------------- records
def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(MODEL_DIR), *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _stats(samples: list[float]) -> dict:
    a = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "std_ms": float(a.std()),
    }


def _emit(rec: dict) -> dict:
    """Append one record to perf.jsonl (all fields) and perf.csv (CSV_FIELDS), and log it."""
    rec = {"run_id": RUN_ID, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "perf.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    path = OUT_DIR / "perf.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: rec.get(k, "") for k in CSV_FIELDS})
    msg = f"BENCH {rec.get('tower')} {rec.get('case')} {rec.get('mode')}"
    if "mean_ms" in rec:
        msg += f": mean {rec['mean_ms']:.3f} ms"
    if "p50_ms" in rec:
        msg += f" p50 {rec['p50_ms']:.3f} p90 {rec['p90_ms']:.3f} p99 {rec['p99_ms']:.3f} min {rec['min_ms']:.3f}"
    for k in ("tokens_per_s", "embeddings_per_s", "items_per_s"):
        if rec.get(k):
            msg += f" {k} {rec[k]:.1f}"
    if rec.get("status") and rec["status"] != "ok":
        msg += f" [{rec['status']}]"
    if rec.get("note"):
        msg += f" ({rec['note']})"
    logger.info(msg)
    return rec


def _throughput(tower: str, batch: int, tokens: int, mean_ms: float) -> dict:
    per_s = 1e3 / mean_ms
    if tower in ("text", "text_long", "full"):
        return {"tokens_per_s": tokens * per_s, "embeddings_per_s": batch * per_s}
    return {"items_per_s": batch * per_s}


@pytest.fixture(scope="session", autouse=True)
def run_info():
    """Provenance of this session (written once): commit, precision file, versions."""
    import transformers

    from models.demos.blackhole.embedding_gemma2.tt.precision import PRECISION_ENV, default_precision_path

    p = Path(default_precision_path())
    cfg = json.loads(p.read_text())
    info = {
        "run_id": RUN_ID,
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": [line for line in _git("status", "--porcelain").splitlines() if line],
        "model_id": MODEL_ID,
        "snapshot": str(resolve_snapshot(MODEL_ID)),
        "precision_file": str(p),
        "precision_env": os.environ.get(PRECISION_ENV, ""),
        "precision_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "precision_config_id": cfg.get("config_id"),
        "precision_sections": {
            k: (cfg[k].get("note") or cfg.get("config_id", "")).split(":")[0] for k in ("text", "vision", "audio")
        },
        "selection": {"all": PERF_ALL, **SEL},
        "device_params": dict(DEVICE_PARAMS),
        "n_warmup": N_WARMUP,
        "n_iters": N_ITERS,
        "sustained_s": SUSTAINED_S,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ttnn": getattr(ttnn, "__version__", ""),
        "transformers": transformers.__version__,
        "hostname": platform.node(),
        "env": {k: os.environ.get(k, "") for k in ("TT_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "OMP_NUM_THREADS")},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    logger.info(f"benchmark run {RUN_ID}: commit {info['commit']} precision {p} ({info['precision_sha256'][:12]})")
    return info


def _open_device():
    device = ttnn.open_device(device_id=0, **DEVICE_PARAMS)
    path = OUT_DIR / "device_info.json"
    if not path.exists():
        grid = device.compute_with_storage_grid_size()
        dram = device.dram_grid_size()
        info = {"arch": str(device.arch()), "compute_grid": [grid.x, grid.y], "dram_grid": [dram.x, dram.y]}
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, indent=2) + "\n")
    return device


# ---------------------------------------------------------------------- generic timing
class Bench:
    """One tower (or the full model) on one device. `execute(shape)` replays the trace on the resident
    inputs; `request(shape)` uploads the prepared host inputs, replays and reads the output back."""

    tower = ""

    def __init__(self, device):
        self.device = device
        self.reference = {}
        self._checked = set()

    def sync(self):
        ttnn.synchronize_device(self.device)

    def key_shape(self, shape):
        return shape

    def tokens(self, shape) -> int:
        return 0

    def check(self, shape, out) -> None:
        outs = out if isinstance(out, list) else [out]
        assert all(torch.isfinite(o).all() for o in outs), f"{self.tower} {shape}: non-finite output"

    def record_check(self, shape, good: bool, note: str, **fields) -> None:
        """Log an HF comparison once per (shape, item); the assert runs on every call."""
        tag = (shape, fields.get("item", 0))
        if tag not in self._checked:
            self._checked.add(tag)
            batch, seq = self.key_shape(shape)
            _emit(
                {
                    "tower": self.tower,
                    "case": _case_id(shape),
                    "batch": batch,
                    "seq_len": seq,
                    "mode": "check_hf_fp32",
                    "status": "ok" if good else "fail",
                    "note": note,
                    **fields,
                }
            )


def _case_id(shape) -> str:
    return f"b{shape[0]}x{shape[1]}" if isinstance(shape, tuple) else str(shape)


def run_mode(bench: Bench, shape, mode: str) -> dict:
    tower = bench.tower
    batch, seq = bench.key_shape(shape)
    base = {
        "tower": tower,
        "case": _case_id(shape),
        "batch": batch,
        "seq_len": seq,
        "n_warmup": N_WARMUP,
        "n_iters": N_ITERS,
    }
    if mode == "2cq":
        _emit({**base, "mode": "2cq", "status": "not_supported", "note": NO_2CQ})
        pytest.skip(NO_2CQ)

    out = bench.request(shape)  # load this case's real inputs into the persistent trace inputs
    bench.check(shape, out)
    for _ in range(N_WARMUP):
        bench.execute(shape)
    bench.sync()

    if mode == "forward":
        samples = []
        for _ in range(N_ITERS):
            t0 = time.perf_counter()
            bench.execute(shape)
            bench.sync()
            samples.append((time.perf_counter() - t0) * 1e3)
        st = _stats(samples)
        tput = _throughput(tower, batch, bench.tokens(shape), st["mean_ms"])
        rec = _emit({**base, "mode": "forward", **st, **tput, "status": "ok", "samples_ms": samples})
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            bench.execute(shape)
        bench.sync()
        mean = (time.perf_counter() - t0) * 1e3 / N_ITERS
        tput = _throughput(tower, batch, bench.tokens(shape), mean)
        note = "N back-to-back replays, one sync; mean only"
        _emit({**base, "mode": "forward_pipelined", "mean_ms": mean, **tput, "status": "ok", "note": note})
        out = bench.request(shape)
    elif mode in ("h2d_d2h", "api"):
        fn = bench.request if mode == "h2d_d2h" else bench.api
        samples = []
        for _ in range(N_ITERS):
            t0 = time.perf_counter()
            out = fn(shape)
            samples.append((time.perf_counter() - t0) * 1e3)
        st = _stats(samples)
        extra = {}
        if mode == "h2d_d2h":
            prep = []
            for _ in range(N_ITERS):
                t0 = time.perf_counter()
                bench.host_prep(shape)
                prep.append((time.perf_counter() - t0) * 1e3)
            extra = {"host_prep": _stats(prep)}
        tput = _throughput(tower, batch, bench.tokens(shape), st["mean_ms"])
        rec = _emit({**base, "mode": mode, **st, **tput, "status": "ok", "samples_ms": samples, **extra})
    else:
        raise ValueError(mode)
    bench.check(shape, out)
    return rec


def run_sustained(bench: Bench, shape) -> dict:
    """Burst (idle, then a short window) vs sustained (30 s back-to-back) device time per forward."""

    def window(n):
        t0 = time.perf_counter()
        for _ in range(n):
            bench.execute(shape)
        bench.sync()
        return (time.perf_counter() - t0) * 1e3 / n

    bench.check(shape, bench.request(shape))
    window(N_WARMUP)
    burst = []
    for _ in range(BURST_REPS):
        time.sleep(BURST_IDLE_S)
        burst.append(window(BURST_N))
    burst_med = statistics.median(burst)
    k = max(5, math.ceil(SUSTAINED_WINDOW_MS / burst_med))
    time.sleep(BURST_IDLE_S)
    series, replays = [], 0
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < SUSTAINED_S:
        t = time.perf_counter() - t_start
        series.append((t, window(k)))
        replays += k
    total_s = time.perf_counter() - t_start
    sus_last = statistics.median([ms for t, ms in series if t >= total_s - 10.0])
    batch, seq = shape
    rec = _emit(
        {
            "tower": bench.tower,
            "case": _case_id(shape),
            "batch": batch,
            "seq_len": seq,
            "mode": "sustained",
            "n_warmup": N_WARMUP,
            "n_iters": replays,
            "mean_ms": total_s * 1e3 / replays,
            **_throughput(bench.tower, batch, bench.tokens(shape), sus_last),
            "status": "ok",
            "note": f"{total_s:.1f} s back-to-back in windows of {k}; throughput at the last-10 s median",
            "burst_ms_median": burst_med,
            "burst_windows_ms": burst,
            "sustained_s": total_s,
            "sustained_window_replays": k,
            "sustained_first_window_ms": series[0][1],
            "sustained_ms_median_last10s": sus_last,
            "sustained_ms_mean_all": total_s * 1e3 / replays,
            "sustained_over_burst": sus_last / burst_med,
            "series": series,
        }
    )
    bench.check(shape, bench.request(shape))
    return rec


# ---------------------------------------------------------------------- text
class TextBench(Bench):
    tower = "text"

    def __init__(self, device, cases, max_seq_len, state_dict=None):
        from models.demos.blackhole.embedding_gemma2.tt.text_encoder import TextEncoder

        super().__init__(device)
        self.snapshot = resolve_snapshot(MODEL_ID)
        self.enc = TextEncoder.from_pretrained(
            device, str(self.snapshot), max_seq_len=max_seq_len, state_dict=state_dict
        )
        self.raw = {s: make_batch(self.snapshot, [s[1]] * s[0]) for s in cases}
        self.hosts = {s: self.enc.prepare_inputs(*self.raw[s]) for s in cases}
        for s, h in self.hosts.items():
            assert (h.batch, h.seq_len) == s
        self.enc.capture_all(cases)

    def tokens(self, shape):
        return shape[0] * shape[1]

    def execute(self, shape):
        self.enc.execute_captured(*shape)

    def request(self, shape):
        _, emb = self.enc.replay(self.hosts[shape])
        return self.enc.read_embedding(emb, shape[0])

    def host_prep(self, shape):
        return self.enc.prepare_inputs(*self.raw[shape])

    def check(self, shape, out):
        super().check(shape, out)
        if shape in self.reference:
            c = float(torch.nn.functional.cosine_similarity(out, self.reference[shape].float(), dim=-1).min())
            good = c >= COSINE_EMBEDDING
            self.record_check(shape, good, f"embedding cosine {c:.6f} (bar {COSINE_EMBEDDING})", cosine=c)
            assert good, f"text {shape}: embedding cosine {c:.6f} < {COSINE_EMBEDDING}"


def _text_reference(snapshot, state_dict, shape) -> torch.Tensor:
    from models.demos.blackhole.embedding_gemma2.reference.hf_reference import load_reference_text_model, run_reference

    model, _ = load_reference_text_model(snapshot, state_dict)
    ids, mask = make_batch(snapshot, [shape[1]] * shape[0])
    emb = run_reference(model, ids, mask, capture_layers=()).embedding
    del model
    gc.collect()
    return emb


class TestText:
    @pytest.fixture(scope="class")
    def bench(self):
        from models.demos.blackhole.embedding_gemma2.tt.weights import load_text_state_dict

        snapshot = resolve_snapshot(MODEL_ID)
        sd = load_text_state_dict(snapshot)
        ref = _text_reference(snapshot, sd, HEADLINE["text"]) if SEL["hf_check"] else None
        device = _open_device()
        try:
            b = TextBench(device, RUN_TEXT, 8192, state_dict=sd)
            if ref is not None:
                b.reference[HEADLINE["text"]] = ref
            yield b
            b.enc.release_traces()
        finally:
            ttnn.close_device(device)

    @pytest.mark.parametrize("mode", RUN_MODES)
    @pytest.mark.parametrize("shape", RUN_TEXT, ids=[_case_id(s) for s in RUN_TEXT])
    def test_perf(self, bench, shape, mode):
        run_mode(bench, shape, mode)

    def test_sustained(self, bench):
        run_sustained(bench, HEADLINE["text"])

    test_sustained.__test__ = SEL["sustained"]


class TestTextLong:
    __test__ = SEL["long"]

    @pytest.fixture(scope="class")
    def bench(self):
        device = _open_device()
        try:
            b = TextBench(device, TEXT_LONG_CASES, 32768)
            b.tower = "text_long"
            yield b
            b.enc.release_traces()
        finally:
            ttnn.close_device(device)

    @pytest.mark.parametrize("mode", RUN_MODES)
    @pytest.mark.parametrize("shape", TEXT_LONG_CASES, ids=[_case_id(s) for s in TEXT_LONG_CASES])
    def test_perf(self, bench, shape, mode):
        run_mode(bench, shape, mode)


def _is_device_oom(err: Exception) -> bool:
    s = str(err)
    return "Out of Memory" in s or "Not enough space to allocate" in s


class TestTextBatchProbe:
    """Largest text batch at seq 128: one eager forward per batch, ascending, until the first device
    allocation failure (recorded, not timed). Batches above the first failure are skipped."""

    __test__ = SEL["probe"]

    @pytest.fixture(scope="class")
    def probe(self):
        from models.demos.blackhole.embedding_gemma2.tt.text_encoder import TextEncoder

        device = _open_device()
        try:
            snapshot = resolve_snapshot(MODEL_ID)
            enc = TextEncoder.from_pretrained(device, str(snapshot), max_seq_len=8192)
            yield {"enc": enc, "snapshot": snapshot, "oom_at": None, "fits": []}
        finally:
            ttnn.close_device(device)

    @pytest.mark.parametrize("batch", TEXT_PROBE_BATCHES, ids=[f"b{b}" for b in TEXT_PROBE_BATCHES])
    def test_batch_probe(self, probe, batch):
        if probe["oom_at"] is not None:
            pytest.skip(f"above the first OOM batch {probe['oom_at']}")
        base = {
            "tower": "text",
            "case": f"b{batch}x{TEXT_PROBE_SEQ}",
            "batch": batch,
            "seq_len": TEXT_PROBE_SEQ,
            "mode": "batch_probe",
        }
        ids, mask = make_batch(probe["snapshot"], [TEXT_PROBE_SEQ] * batch)
        t0 = time.perf_counter()
        try:
            out = probe["enc"].encode(ids, mask)
        except RuntimeError as err:  # a device allocation failure is the measurement; anything else fails
            if not _is_device_oom(err):
                raise
            probe["oom_at"] = batch
            lines = [line for line in str(err).splitlines() if "Memory" in line or "allocate" in line]
            first = (lines[0] if lines else str(err)[:300]).strip()[:400]
            largest = max(probe["fits"], default=None)
            _emit({**base, "status": "oom", "note": first, "largest_fit": largest})
            pytest.skip(f"device OOM at batch {batch} (largest fit {largest}): {first[:200]}")
        ms = (time.perf_counter() - t0) * 1e3
        assert torch.isfinite(out).all()
        probe["fits"].append(batch)
        _emit({**base, "status": "fits", "note": f"eager first run incl. compile {ms:.0f} ms"})


# ---------------------------------------------------------------------- vision / audio
class TowerBench(Bench):
    """VisionEncoder / AudioEncoder: the output is the soft tokens per item (padding rows stripped)."""

    def __init__(self, tower, device, enc, cases, inputs_fn, counts_attr):
        super().__init__(device)
        self.tower = tower
        self.enc = enc
        self.counts_attr = counts_attr
        snapshot = resolve_snapshot(MODEL_ID)
        self.raw = {s: inputs_fn(snapshot, *s) for s in cases}
        self.hosts = {s: enc.prepare_inputs(*self.raw[s]) for s in cases}
        for s, h in self.hosts.items():
            assert (h.batch, h.seq_len) == s
        enc.capture_all(cases)

    def execute(self, shape):
        self.enc.execute_captured(*shape)

    def request(self, shape):
        h = self.hosts[shape]
        soft, _ = self.enc.replay(h)
        return self.enc.read_rows(soft, getattr(h, self.counts_attr))

    def host_prep(self, shape):
        return self.enc.prepare_inputs(*self.raw[shape])

    def check(self, shape, out):
        super().check(shape, out)
        if shape not in self.reference:
            return
        ok = True
        for i, (t, r) in enumerate(zip(out, self.reference[shape])):
            p = pcc(r, t)
            c = float(torch.nn.functional.cosine_similarity(t.flatten(), r.flatten().float(), dim=0))
            good = p >= SOFT_TOKEN_PCC and c >= COSINE_EMBEDDING
            ok &= good
            note = f"item {i}: soft-token PCC {p:.6f}, flattened cosine {c:.6f} (bars 0.99)"
            self.record_check(shape, good, note, soft_pcc=p, cosine=c, item=i)
        assert ok, f"{self.tower} {shape}: soft tokens below the 0.99 bars vs HF fp32"


class TestVision:
    @pytest.fixture(scope="class")
    def bench(self):
        from models.demos.blackhole.embedding_gemma2.reference.hf_vision_reference import (
            load_reference_vision,
            run_vision_reference,
        )
        from models.demos.blackhole.embedding_gemma2.tests.perf.bench_vision import _inputs
        from models.demos.blackhole.embedding_gemma2.tt.vision.vision_encoder import VisionEncoder

        snapshot = resolve_snapshot(MODEL_ID)
        ref = None
        if SEL["hf_check"]:
            tower, embed, _ = load_reference_vision(snapshot)
            pv, pid = _inputs(snapshot, *HEADLINE["vision"])
            ref = run_vision_reference(tower, embed, pv, pid, capture_layers=()).soft_tokens
            del tower, embed
            gc.collect()
        device = _open_device()
        try:
            enc = VisionEncoder.from_pretrained(device, str(snapshot))
            b = TowerBench("vision", device, enc, RUN_VISION, _inputs, "num_soft_tokens")
            if ref is not None:
                b.reference[HEADLINE["vision"]] = ref
            yield b
            b.enc.release_traces()
        finally:
            ttnn.close_device(device)

    @pytest.mark.parametrize("mode", RUN_MODES)
    @pytest.mark.parametrize("shape", RUN_VISION, ids=[_case_id(s) for s in RUN_VISION])
    def test_perf(self, bench, shape, mode):
        run_mode(bench, shape, mode)

    def test_sustained(self, bench):
        run_sustained(bench, HEADLINE["vision"])

    test_sustained.__test__ = SEL["sustained"]


class TestAudio:
    @pytest.fixture(scope="class")
    def bench(self):
        from models.demos.blackhole.embedding_gemma2.reference.hf_audio_reference import (
            load_reference_audio,
            run_audio_reference,
        )
        from models.demos.blackhole.embedding_gemma2.tests.perf.bench_audio import _inputs
        from models.demos.blackhole.embedding_gemma2.tt.audio.audio_encoder import AudioEncoder

        snapshot = resolve_snapshot(MODEL_ID)
        ref = None
        if SEL["hf_check"]:
            torch.set_num_threads(max(torch.get_num_threads(), 16))
            tower, embed, _ = load_reference_audio(snapshot)
            feats, mask = _inputs(snapshot, *HEADLINE["audio"])
            ref = run_audio_reference(tower, embed, feats, mask, capture_layers=()).soft_tokens
            del tower, embed
            gc.collect()
        device = _open_device()
        try:
            enc = AudioEncoder.from_pretrained(device, str(snapshot))
            b = TowerBench("audio", device, enc, RUN_AUDIO, _inputs, "num_tokens")
            if ref is not None:
                b.reference[HEADLINE["audio"]] = ref
            yield b
            b.enc.release_traces()
        finally:
            ttnn.close_device(device)

    @pytest.mark.parametrize("mode", RUN_MODES)
    @pytest.mark.parametrize("shape", RUN_AUDIO, ids=[_case_id(s) for s in RUN_AUDIO])
    def test_perf(self, bench, shape, mode):
        run_mode(bench, shape, mode)

    def test_sustained(self, bench):
        run_sustained(bench, HEADLINE["audio"])

    test_sustained.__test__ = SEL["sustained"]


# ---------------------------------------------------------------------- full model
def _full_items(cases) -> dict:
    from models.demos.blackhole.embedding_gemma2.reference import full_model_inputs as F

    out = {"text_b1x512": F.DOC + F.long_text(FULL_TEXT_CHARS)}
    if any(c in FULL_ITEMS for c in cases):
        catalogue = F.items()  # decodes every catalogue image / clip / video
        out.update({case: catalogue[name] for case, name in FULL_ITEMS.items()})
    return {c: out[c] for c in cases}


class FullBench(Bench):
    """EmbeddingGemma2 (public API): one item per case, one trace per ShapeKey."""

    tower = "full"

    def __init__(self, device, items):
        from models.demos.blackhole.embedding_gemma2.tt.embedding_model import EmbeddingGemma2

        super().__init__(device)
        self.model = EmbeddingGemma2.from_pretrained(device, str(resolve_snapshot(MODEL_ID)), max_seq_len=8192)
        self.items = items
        self.processed = {c: self.model.process(items[c]) for c in items}
        self.hosts = {c: self.model.prepare_batch([self.processed[c]]) for c in items}
        if "text_b1x512" in self.hosts:
            assert self.hosts["text_b1x512"].key.seq_len == 512
        self.model.capture_all([h.key for h in self.hosts.values()])

    def key_shape(self, case):
        k = self.hosts[case].key
        return k.batch, k.seq_len

    def tokens(self, case):
        return self.processed[case].length

    def execute(self, case):
        self.model.execute_captured(self.hosts[case].key)

    def request(self, case):
        return self.model.read_embedding(self.model.replay(self.hosts[case]), 1)

    def api(self, case):
        return self.model.encode([self.items[case]], traced=True)

    def host_prep(self, case):
        return self.model.prepare_batch([self.processed[case]])

    def check(self, case, out):
        super().check(case, out)
        assert torch.allclose(out.norm(dim=-1), torch.ones(out.shape[0]), atol=1e-3), f"full {case}: not unit norm"
        if case not in self.reference:
            return
        c = float(torch.nn.functional.cosine_similarity(out[0], self.reference[case].float(), dim=-1))
        good = c >= COSINE_EMBEDDING
        key = self.hosts[case].key
        note = f"embedding cosine {c:.6f} (bar {COSINE_EMBEDDING}); {self.tokens(case)} tokens, key {key}"
        self.record_check(case, good, note, cosine=c, key=str(key), tokens=self.tokens(case))
        assert good, f"full {case}: embedding cosine {c:.6f} < {COSINE_EMBEDDING}"


class TestFull:
    @pytest.fixture(scope="class")
    def bench(self):
        from transformers import AutoProcessor

        from models.demos.blackhole.embedding_gemma2.reference.hf_full_reference import embed, load_reference_full_model
        from models.demos.blackhole.embedding_gemma2.tt.embedding_model import process_item

        snapshot = resolve_snapshot(MODEL_ID)
        items = _full_items(RUN_FULL)
        ref = {}
        if SEL["hf_check"]:
            proc = AutoProcessor.from_pretrained(str(snapshot))
            hf = load_reference_full_model(snapshot)
            ref = {c: embed(hf, process_item(proc, items[c]))[0] for c in RUN_FULL}
            del hf
            gc.collect()
        device = _open_device()
        try:
            b = FullBench(device, items)
            b.reference = ref
            yield b
            b.model.release_traces()
        finally:
            ttnn.close_device(device)

    @pytest.mark.parametrize("mode", RUN_FULL_MODES)
    @pytest.mark.parametrize("case", RUN_FULL)
    def test_perf(self, bench, case, mode):
        run_mode(bench, case, mode)


# ---------------------------------------------------------------------- Tracy (profiling only)
@pytest.mark.skipif(
    os.environ.get("TT_METAL_DEVICE_PROFILER") != "1", reason="profiling only: run under python -m tracy -p"
)
@pytest.mark.parametrize("batch,seq_len", [(1, 512), (8, 512)], ids=["b1_s512", "b8_s512"])
def test_tracy_forward(batch, seq_len, eg2_device, tt_text_model, eg2_snapshot):
    from models.demos.blackhole.embedding_gemma2.tt.text_encoder import TextEncoder

    enc = TextEncoder(tt_text_model, eg2_device)
    ids, mask = make_batch(eg2_snapshot, [seq_len] * batch)
    host = enc.prepare_inputs(ids, mask)
    enc.model.prepare(host.seq_len)
    ids_d, keep_d = enc._to_device(host)
    enc.forward(ids_d, keep_d)  # compile
    ttnn.synchronize_device(eg2_device)
    ttnn.ReadDeviceProfiler(eg2_device)  # flush compile-run records
    signpost("PERF_TEXT")
    enc.forward(ids_d, keep_d)
    ttnn.synchronize_device(eg2_device)
    signpost("PERF_TEXT_END")
    ttnn.ReadDeviceProfiler(eg2_device)


# collected under the profiler, or with the all-on switch (where it skips without the profiler, as
# in the full-sweep run)
test_tracy_forward.__test__ = PROFILER or PERF_ALL
