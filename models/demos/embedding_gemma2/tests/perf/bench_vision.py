# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Burst and sustained traced latency of the vision forward (patch embed -> 16 layers -> pool -> embed).

Also holds the tower bench machinery shared with `bench_audio.py` (`TowerSpec`, `run_tower_bench`).

Device-only timing: inputs already resident, `execute_trace` + one synchronize per window. Inputs are the
real processor outputs of the vision test cases (`vision_common.load_case`).

A plain run is quick: the headline shape (1 x 2560, one image), burst only, on the 12 x 10 grid. Everything
else is opt-in (env, "1" = on; options combine). Rough costs per shape with a warm weight / kernel cache
(E10.1, doc/prefill_towers/work_log.md):

    option (env)                               adds                                              ~min
    (none)                                     headline shape, burst (5 windows after 3 s idle)  1
    EMBEDDING_GEMMA2_BENCH_CONTRACT=1          every batch-1 contract shape of the tower           +0.3 per shape
                                               (vision: 1 x 2560 only; audio: 1 x 256, 1 x 768)
    EMBEDDING_GEMMA2_BENCH_SUSTAINED=1         sustained: >= 30 s and >= 300 replays back to      +0.6 per shape
                                               back, windows of ~500 ms, median of the last third
    EMBEDDING_GEMMA2_BENCH_TELEMETRY=1         AICLK / power / current / vcore / temperature      +0
                                               sampled every 100 ms in a second process during
                                               sustained runs (sysfs, prefill_telemetry.py)
    EMBEDDING_GEMMA2_BENCH_CHECK=1             gate on the traced replay output vs HF fp32:       +0.5 (first run
                                               soft-token PCC (>= 0.99) and embedding cosine        builds the refs)
                                               (cosine of the mean soft token, >= 0.99), per-token
                                               cosine min / mean; HF refs cached in
                                               EMBEDDING_GEMMA2_BENCH_REFS
    EMBEDDING_GEMMA2_BENCH_ALL=1               all of the above

Phases: check + burst for every shape first, then sustained for every shape (no burst after sustained load).

Definitions
    burst      `--idle` s idle, then `--burst` back-to-back replays timed with one sync; repeated `--reps`
               times; median / min / max of the per-window means
    sustained  the prefill harness definition (tests/perf/prefill.py): `--idle` s idle, then windows of
               ~500 ms of back-to-back replays (one sync per window) for max(30 s, 300 replays); median of
               the windows in the last third; telemetry summarized over the same last third

Grid: 12 x 10 by default (the Galaxy target, contract compute_grid; the device is opened on it and checked,
conftest.open_eg2_device); EMBEDDING_GEMMA2_GRID=13x10 runs on the P150 grid. The grid override is read
once per process, so run one grid per process. Every record holds `compute_grid` (what the device reports),
the commit, dirty files and the EG2_* model switches.

Run:
    python models/demos/embedding_gemma2/tests/perf/bench_vision.py [--cases 1x2560] [--json out.jsonl] [--label name]
    EMBEDDING_GEMMA2_BENCH_ALL=1 python models/demos/embedding_gemma2/tests/perf/bench_vision.py --json out.jsonl

Other env: EMBEDDING_GEMMA2_BENCH_OUT (telemetry CSV dir, default generated/embedding_gemma2_towers),
EMBEDDING_GEMMA2_BENCH_REFS (HF reference cache, default <OUT>/refs), EMBEDDING_GEMMA2_SUSTAINED_S (30).
`--cases` overrides the shape selection (batched shapes run, but are outside the batch-1 contract).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

import ttnn

GALAXY = (12, 10)
_ROOT = Path(__file__).resolve().parents[5]  # tt-metal checkout


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on", "all")


def selection() -> dict:
    every = _flag("EMBEDDING_GEMMA2_BENCH_ALL")
    sel = {
        k: every or _flag(f"EMBEDDING_GEMMA2_BENCH_{k.upper()}")
        for k in ("contract", "sustained", "telemetry", "check")
    }
    sel["all"] = every
    return sel


OUT_DIR = Path(os.environ.get("EMBEDDING_GEMMA2_BENCH_OUT", str(_ROOT / "generated" / "embedding_gemma2_towers")))
REFS_DIR = Path(os.environ.get("EMBEDDING_GEMMA2_BENCH_REFS", str(OUT_DIR / "refs")))
SUSTAINED_S = float(os.environ.get("EMBEDDING_GEMMA2_SUSTAINED_S", "30"))
SUSTAINED_REPLAYS = 300
SUSTAINED_WINDOW_MS = 500.0
GATE = 0.99  # soft-token PCC and embedding cosine vs HF fp32 (contract)


@dataclass
class TowerSpec:
    """What `run_tower_bench` needs from a tower."""

    name: str  # "vision" / "audio"
    unit: str  # "img" / "clip" (ms per item in the log line)
    headline: tuple[str, ...]  # shapes of a plain run, "BxS"
    contract: tuple[str, ...]  # batch-1 contract shapes (EMBEDDING_GEMMA2_BENCH_CONTRACT)
    burst_default: int
    build: Callable  # (device, snapshot) -> encoder
    inputs: Callable  # (snapshot, b, s) -> (case names, prepare_inputs args)
    counts: Callable  # host inputs -> valid rows per item
    reference: Callable  # (snapshot, args) -> list of HF fp32 soft tokens per item


def _git(*a) -> str:
    return subprocess.run(["git", "-C", str(_ROOT), *a], capture_output=True, text=True).stdout.rstrip()


def run_info(spec: TowerSpec, sel: dict, grid) -> dict:
    from models.demos.embedding_gemma2.tests.perf import prefill_telemetry as telemetry
    from models.demos.embedding_gemma2.tt.precision import default_precision_path

    try:
        board = telemetry.identity()
        board["idle"] = telemetry.read()
    except OSError as err:
        board = {"error": str(err)}
    return {
        "tower": spec.name,
        "commit": _git("rev-parse", "--short=10", "HEAD"),
        "dirty_files": [ln for ln in _git("status", "--porcelain").splitlines() if ln and "embedding_gemma2" in ln],
        "compute_grid": list(grid),
        "grid_env": os.environ.get("EMBEDDING_GEMMA2_GRID", ""),
        "precision_file": str(default_precision_path()),
        "selection": sel,
        "model_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("EG2_")},
        "board": board,
    }


def _window_ms(enc, b: int, s: int, n: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        enc.execute_captured(b, s)
    ttnn.synchronize_device(enc.device)
    return (time.perf_counter() - t0) * 1e3 / n


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


def check(spec: TowerSpec, snapshot, names, args, out: list[torch.Tensor]) -> dict:
    """Traced replay output vs the cached HF fp32 reference."""
    from models.demos.embedding_gemma2.tests.common import pcc

    REFS_DIR.mkdir(parents=True, exist_ok=True)
    path = REFS_DIR / f"{spec.name}_{'+'.join(names)}.pt"
    if path.exists():
        ref = torch.load(path)
    else:
        t0 = time.perf_counter()
        ref = [r.float() for r in spec.reference(snapshot, args)]
        torch.save(ref, path)
        print(f"BENCH built HF fp32 reference {path} in {time.perf_counter() - t0:.1f} s", flush=True)
    items = []
    for hf, tt in zip(ref, out):
        assert tuple(hf.shape) == tuple(tt.shape), f"{tuple(tt.shape)} vs HF {tuple(hf.shape)}"
        rows = torch.nn.functional.cosine_similarity(hf, tt, dim=-1)
        items.append(
            {
                "soft_tokens": int(hf.shape[0]),
                "soft_token_pcc": pcc(hf, tt),
                "embedding_cos": _cos(hf.mean(0), tt.mean(0)),
                "token_cos_min": float(rows.min()),
                "token_cos_mean": float(rows.mean()),
            }
        )
    ok = all(i["soft_token_pcc"] >= GATE and i["embedding_cos"] >= GATE for i in items)
    return {"ref": str(path), "items": items, "pass": ok}


def sustained(enc, b: int, s: int, idle: float, tel_path: Path | None) -> dict:
    from models.demos.embedding_gemma2.tests.perf import prefill_telemetry as telemetry

    est = _window_ms(enc, b, s, 3)
    k = max(1, math.ceil(SUSTAINED_WINDOW_MS / est))
    duration = max(SUSTAINED_S, SUSTAINED_REPLAYS * est / 1e3)
    time.sleep(idle)
    sampler = telemetry.Sampler(tel_path) if tel_path else None
    if sampler:
        sampler.__enter__()
    try:
        series, replays = [], 0
        wall0 = time.time()
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < duration or replays < SUSTAINED_REPLAYS:
            t = time.perf_counter() - t_start
            series.append((t, _window_ms(enc, b, s, k)))
            replays += k
        total_s = time.perf_counter() - t_start
    finally:
        if sampler:
            sampler.__exit__(None, None, None)
    last = [ms for t, ms in series if t >= total_s * 2 / 3] or [series[-1][1]]
    rec = {
        "sustained_ms_median": statistics.median(last),
        "sustained_ms_min": min(last),
        "sustained_ms_max": max(last),
        "sustained_first_window_ms": series[0][1],
        "sustained_mean_all_ms": total_s * 1e3 / replays,
        "sustained_replays": replays,
        "sustained_window_replays": k,
        "sustained_duration_s": total_s,
        "sustained_windows_last_third": len(last),
    }
    if sampler:
        rows = [r for r in sampler.rows() if wall0 <= r["t"] <= wall0 + total_s]
        tail = [r for r in rows if r["t"] >= wall0 + total_s * 2 / 3]
        if tail:
            rec["telemetry"] = {
                "csv": str(tel_path),
                "samples": len(rows),
                "last_third": {
                    f: {
                        "median": statistics.median(r[f] for r in tail),
                        "min": min(r[f] for r in tail),
                        "max": max(r[f] for r in tail),
                    }
                    for f in ("aiclk_mhz", "power_w", "current_a", "vcore_v", "asic_temp_c")
                },
            }
    return rec


def run_tower_bench(spec: TowerSpec) -> None:
    from models.demos.embedding_gemma2.conftest import MODEL_ID, open_eg2_device
    from models.demos.embedding_gemma2.tt.config import requested_grid
    from models.demos.embedding_gemma2.tt.weights import resolve_snapshot

    sel = selection()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cases",
        default="",
        help="comma list of BxS; default: headline shape (contract with EMBEDDING_GEMMA2_BENCH_CONTRACT=1)",
    )
    ap.add_argument("--burst", type=int, default=spec.burst_default)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--idle", type=float, default=3.0)
    ap.add_argument("--no-sustained", action="store_true", help="force sustained off (overrides the env switches)")
    ap.add_argument("--json", default="")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    if args.no_sustained:
        sel["sustained"] = sel["telemetry"] = False

    names = args.cases or ",".join(spec.contract if sel["contract"] else spec.headline)
    cases = [tuple(int(v) for v in c.split("x")) for c in names.split(",")]
    snapshot = resolve_snapshot(MODEL_ID)
    device = open_eg2_device(0, grid=requested_grid(GALAXY))
    g = device.compute_with_storage_grid_size()
    grid = (g.x, g.y)
    info = run_info(spec, sel, grid)
    print(f"BENCH {spec.name} run_info {json.dumps(info)}", flush=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")

    def emit(rec):
        rec = {"label": args.label, "tower": spec.name, "run_id": run_id, "compute_grid": list(grid), **rec}
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    try:
        enc = spec.build(device, str(snapshot))
        enc.capture_all(cases)
        recs = {}
        for b, s in cases:
            case_names, inp = spec.inputs(snapshot, b, s)
            host = enc.prepare_inputs(*inp)
            assert (host.batch, host.seq_len) == (b, s)
            soft, _ = enc.replay(host)
            out = type(enc).read_rows(soft, spec.counts(host))
            assert all(torch.isfinite(o).all() for o in out)
            rec = {"batch": b, "seq_len": s, "inputs": list(case_names)}
            if sel["check"]:
                rec["check"] = check(spec, snapshot, case_names, inp, out)
                worst = min(rec["check"]["items"], key=lambda i: i["soft_token_pcc"])
                print(
                    f"BENCH {args.label} {spec.name} b{b} s{s} check {'PASS' if rec['check']['pass'] else 'FAIL'}: "
                    f"soft-token PCC {worst['soft_token_pcc']:.5f}, embedding cos {worst['embedding_cos']:.5f}, "
                    f"token cos min {worst['token_cos_min']:.4f}",
                    flush=True,
                )
            _window_ms(enc, b, s, 3)  # warm
            burst = []
            for _ in range(args.reps):
                time.sleep(args.idle)
                burst.append(_window_ms(enc, b, s, args.burst))
            rec.update(
                burst_ms_median=statistics.median(burst),
                burst_ms_min=min(burst),
                burst_ms_max=max(burst),
                burst_windows=burst,
                burst_replays_per_window=args.burst,
            )
            print(
                f"BENCH {args.label} {spec.name} b{b} s{s} grid {grid[0]}x{grid[1]}: burst median {rec['burst_ms_median']:.3f} ms "
                f"[{rec['burst_ms_min']:.3f}, {rec['burst_ms_max']:.3f}] ({rec['burst_ms_median'] / b:.3f} ms/{spec.unit})",
                flush=True,
            )
            recs[(b, s)] = rec
        if sel["sustained"]:
            for b, s in cases:
                tel = OUT_DIR / f"telemetry_{spec.name}_{run_id}_b{b}_s{s}.csv" if sel["telemetry"] else None
                rec = recs[(b, s)]
                rec.update(sustained(enc, b, s, args.idle, tel))
                msg = (
                    f"BENCH {args.label} {spec.name} b{b} s{s} grid {grid[0]}x{grid[1]}: sustained median "
                    f"{rec['sustained_ms_median']:.3f} ms [{rec['sustained_ms_min']:.3f}, {rec['sustained_ms_max']:.3f}] "
                    f"({rec['sustained_replays']} replays, {rec['sustained_duration_s']:.1f} s)"
                )
                if "telemetry" in rec:
                    lt = rec["telemetry"]["last_third"]
                    msg += f"; AICLK {lt['aiclk_mhz']['median']:.0f} MHz, power {lt['power_w']['median']:.1f} W"
                print(msg, flush=True)
        for b, s in cases:
            emit({**recs[(b, s)], "run_info": info})
        enc.release_traces()
    finally:
        ttnn.close_device(device)


# ------------------------------------------------------------------------------------------- vision
# (batch, bucket) -> processor cases concatenated along the batch (as test_vision_perf.PERF_CASES)
CASE_INPUTS = {
    (1, 2560): ("landscape",),
    (2, 2560): ("square", "small"),
    (4, 2560): ("landscape", "portrait", "square", "small"),
    (8, 1280): ("video8",),
}


def _case_inputs(snapshot, b: int, s: int):
    """(case names, (pixel_values, position_ids)) for shape b x s."""
    from models.demos.embedding_gemma2.tests.pcc.vision.vision_common import load_case

    names = CASE_INPUTS.get((b, s))
    if names is None:  # repeat the bucket's batch-1 case
        names = ("landscape",) * b if s == 2560 else None
    if names is None:
        raise ValueError(f"no input case for {b}x{s}")
    pvs, pids = zip(*(load_case(str(snapshot), c) for c in names))
    pv, pid = torch.cat(pvs), torch.cat(pids)
    if pv.shape[0] != b:
        raise ValueError(f"case {names} gives batch {pv.shape[0]}, wanted {b}")
    return names, (pv, pid)


def _inputs(snapshot, b: int, s: int):
    """prepare_inputs args for shape b x s (also used by dtype_sweep_towers.py)."""
    return _case_inputs(snapshot, b, s)[1]


def _vision_reference(snapshot, args):
    from models.demos.embedding_gemma2.reference.hf_vision_reference import load_reference_vision, run_vision_reference

    tower, embed, _ = load_reference_vision(snapshot)
    pv, pid = args
    return run_vision_reference(tower, embed, pv, pid, capture_layers=()).soft_tokens


def _vision_build(device, snapshot):
    from models.demos.embedding_gemma2.tt.vision.vision_encoder import VisionEncoder

    return VisionEncoder.from_pretrained(device, snapshot)


VISION = TowerSpec(
    name="vision",
    unit="img",
    headline=("1x2560",),
    contract=("1x2560",),
    burst_default=10,
    build=_vision_build,
    inputs=_case_inputs,
    counts=lambda host: host.num_soft_tokens,
    reference=_vision_reference,
)


if __name__ == "__main__":
    run_tower_bench(VISION)
