# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU goldens for the TTNN Toto 2.0 port, produced by the pinned upstream reference.

Runs only in the reference venv (Python 3.12, torch 2.14, ``toto-2`` @ 9213d2c), never in python_env,
with the worktree kept off the import path:

    ttq run --model toto2 --kind cpu --sh 'env -u PYTHONPATH -u VIRTUAL_ENV HF_HUB_OFFLINE=1 \
        /home/andy/bringup/venvs/toto2/bin/python models/experimental/toto2_forecast/tests/golden/make_goldens.py \
        --ckpt 22m'

Every golden is one ``.pt`` file under ``$TOTO2_GOLDEN_DIR`` (default ``/home/andy/bringup/goldens/toto2``),
``<ckpt>/<case>.pt``. It holds the call's inputs (``target`` with NaN already replaced, ``target_mask``,
``series_ids``, ``horizon``, the forecast kwargs) and the captures:

    loc, scale        (B, V, P_total)    hook on m.scaler, one value per patch (pre-backfill)
    loc_bf, scale_bf  (B, V, P_total)    after backfill_short_patches (what forecast() uses)
    tokens_context    (B, V, P, 64)      patch_proj input, context; tokens_horizon (B, V, h, 64)
    embed_context     (B, V, P, d)       patch_proj output, context; embed_horizon (B, V, h, d)
    layer_<i>         time (B*V, S, d), variate (B*S, V, d)    selected layers
    out_norm          (B, V, S, d)
    head_in           (B, V, n, d)       output_head input (the read-out tokens)
    head_out          (9, B, V, n, 32)   output_head output, asinh-normalized space
    q                 (9, B, V, H)       final quantiles

Large cases run in chunks of series (each series is independent) and keep partial chunk files, so
a timed-out job resumes where it stopped. ``MANIFEST.json`` next to this script records the code pin,
the venv's ``pip freeze``, the HF revisions, the safetensors sha256, the thread count, the input
generator and the sha256 of every golden.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

REFERENCE_REPO = "https://github.com/DataDog/toto"
REFERENCE_COMMIT = "9213d2c86f716c43010a0a3b2b6a0e3f440e6743"
THREADS = 8
GOLDEN_DIR = Path(os.environ.get("TOTO2_GOLDEN_DIR", "/home/andy/bringup/goldens/toto2"))
MANIFEST = Path(__file__).resolve().parent / "MANIFEST.json"

CHECKPOINTS = {
    "4m": ("Datadog/Toto-2.0-4m", "8306a9801cf98c0f5ffe4b2dcc8f496e616d84d9"),
    "22m": ("Datadog/Toto-2.0-22m", "685e4ae3e2be8d8998025e53dd98e7fdcb296a89"),
    "313m": ("Datadog/Toto-2.0-313m", "a7bab288f5e95f8606f8306f86659357e1c001ef"),
    "2.5B-FT": ("Datadog/Toto-2.0-2.5B-FT", "b4f030fb09ba0fe3f61dd11cf730539bda568eb1"),
}

FORECAST_KWARGS = dict(
    decode_block_size=None, has_missing_values=True, scaler_fallback_min_obs=8, quantile_real_cap_k=1e4
)


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    variates: int
    context: int
    horizon: int
    pattern: str = "clean"  # clean | missing
    series_ids: str = (
        "shared"  # shared (all variates of an item share id 0) | mixed (item b: ids [0,..,0,1,..,1] for odd b)
    )
    capture: str = "all"  # all | some (layers 0, 1, L/2, L-2, L-1) | head (scaler, head, q only)
    seed: int = 0
    chunk: int = 0  # items per chunk; 0 = one call
    mix: str = "smooth"  # smooth (trend + sinusoid + noise, the unit-test goldens) | realistic (``REALISTIC``)
    kind: str = ""  # realistic only: force one series kind for every row ("" = draw from the mix)


SMALL = (
    Case("uni_4x2048", 4, 1, 2048, 64),
    Case("missing_5x2048", 5, 1, 2048, 64, pattern="missing", seed=1),
    Case("mv_2x4x2048", 2, 4, 2048, 64, pattern="missing", series_ids="mixed", seed=2),
    Case("long_2x8192", 2, 1, 8192, 1024, seed=3),
)
# Single-series cases in the realistic mix (PLAN section 9: a per-row offset cannot carry their PCC).
SINGLE = (
    Case("single_rw_1x2048", 1, 1, 2048, 64, capture="some", seed=300, mix="realistic", kind="rw_seasonal"),
    Case("single_sparse_1x2048", 1, 1, 2048, 64, capture="head", seed=301, mix="realistic", kind="intermittent"),
    Case("single_shift_1x2048", 1, 1, 2048, 64, capture="head", seed=302, mix="realistic", kind="level_shift"),
    Case("single_long_rw_1x8192", 1, 1, 8192, 1024, capture="head", seed=303, mix="realistic", kind="rw_seasonal"),
)
CASES = {
    "4m": SMALL,
    "22m": SMALL + SINGLE,
    "313m": (
        Case("uni_16x2048", 16, 1, 2048, 64, capture="some", seed=10),
        Case("missing_16x2048", 16, 1, 2048, 64, pattern="missing", capture="some", seed=11),
        Case("mv_4x8x2048", 4, 8, 2048, 64, capture="some", seed=12),
        # Smooth-mix bench input (round 0 before PLAN v1.1); kept for the collapsed-output comparison.
        Case("paper_1024x2048", 1024, 1, 2048, 64, capture="head", seed=100, chunk=64),
        *SINGLE,
        Case("paper_mix_1024x2048", 1024, 1, 2048, 64, capture="head", seed=110, chunk=64, mix="realistic"),
    ),
    "2.5B-FT": (
        Case("uni_16x2048", 16, 1, 2048, 64, capture="some", seed=10),
        Case("mv_4x8x2048", 4, 8, 2048, 64, capture="some", seed=12),
        Case("missing_16x2048", 16, 1, 2048, 64, pattern="missing", capture="head", seed=11),
        # Smooth-mix bench inputs (round 0 before PLAN v1.1); kept for the collapsed-output comparison.
        Case("paper_1024x2048", 1024, 1, 2048, 64, capture="head", seed=100, chunk=64),
        Case("long_context_32x8192", 32, 1, 8192, 1024, capture="head", seed=101, chunk=8),
        *SINGLE,
        Case("long_context_mix_32x8192", 32, 1, 8192, 1024, capture="head", seed=111, chunk=8, mix="realistic"),
        Case("paper_mix_1024x2048", 1024, 1, 2048, 64, capture="head", seed=110, chunk=64, mix="realistic"),
        Case("multivariate_mix_128x8x2048", 128, 8, 2048, 64, capture="head", seed=112, chunk=8, mix="realistic"),
    ),
}
# The GluonTS end-to-end case (stage E): Toto2GluonTSModel, context_length 4096, 64 series, ffill, left zero padding.
GLUONTS = {"2.5B-FT": dict(name="gluonts_64x4096", series=64, context_length=4096, horizon=64, seed=200)}

GENERATOR = (
    "per row: level U(10,500) + trend U(-0.05,0.2)*t + amplitude U(1,30)*sin(2*pi*t/P + phase U(0,2pi)), "
    "P in {24,48,96,168}, + N(0, U(0.5,3)) noise; numpy default_rng(seed), rows drawn in (item, variate) order. "
    "pattern 'missing' (row r, r % 5): 0 clean; 1 ten scattered NaN; 2 NaN over t in [640,704) (two fully missing "
    "patches) plus scattered; 3 the first 90 steps NaN (backfill: patch 2 has 6 observations); 4 NaN in the last "
    "context patch [T-20,T-10) plus a fully missing patch at [1024,1056). NaN -> mask False, value 0. "
    "mix 'realistic' (the bench, single-series and GluonTS goldens, PLAN section 9): per row a kind drawn from "
    "REALISTIC (rw_seasonal 0.35, intermittent 0.15, level_shift 0.15, multi_seasonal 0.15, smooth_trend 0.10, "
    "spiky 0.10) unless the case forces one; level 10^U(-2,4), negative for 20 % of rows, unit |level| * U(0.05,0.5); "
    "random-walk, seasonal and level-shift kinds add Gaussian noise at SNR ~ 1 (std x U(0.7,1.4)); intermittent = "
    "Bernoulli(U(0.02,0.3)) x Poisson(U(1,20)); multivariate items mix a shared latent series (weight U(0.3,0.9)) "
    "into each variate. See _realistic() for the exact draws; seeds per case below."
)


# The realistic mix (PLAN section 9, v1.1): kind -> weight.
REALISTIC = {
    "rw_seasonal": 0.35,  # random walk + one seasonality + noise at SNR ~ 1
    "intermittent": 0.15,  # sparse counts: mostly zeros, Poisson bursts
    "level_shift": 0.15,  # piecewise-constant level with 1-4 shifts, noise, weak seasonality
    "multi_seasonal": 0.15,  # daily + weekly seasonality, noise at SNR ~ 1
    "smooth_trend": 0.10,  # the smooth trend + sinusoid of the unit-test goldens
    "spiky": 0.10,  # heavy-tailed (Student-t) noise and rare spikes on a seasonal base
}


def _smooth(rng, t: int) -> np.ndarray:
    steps = np.arange(t)
    level, trend = rng.uniform(10, 500), rng.uniform(-0.05, 0.2)
    amp, period, phase = rng.uniform(1, 30), rng.choice([24, 48, 96, 168]), rng.uniform(0, 2 * np.pi)
    return level + trend * steps + amp * np.sin(2 * np.pi * steps / period + phase) + rng.normal(0, rng.uniform(0.5, 3.0), t)


def _seasonal(rng, t: int, periods) -> np.ndarray:
    steps = np.arange(t)
    return sum(
        rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * steps / p + rng.uniform(0, 2 * np.pi)) for p in periods
    )


def _realistic(rng, t: int, kind: str) -> np.ndarray:
    """One series of ``kind`` with a log-uniform level (1e-2 .. 1e4) and a random sign for 20 % of the rows."""
    level = 10 ** rng.uniform(-2, 4) * (-1 if rng.random() < 0.2 else 1)
    unit = abs(level) * rng.uniform(0.05, 0.5)
    if kind == "smooth_trend":
        return _smooth(rng, t)
    if kind == "intermittent":
        active = rng.random(t) < rng.uniform(0.02, 0.3)
        return active * rng.poisson(rng.uniform(1, 20), t).astype(np.float64)
    if kind == "rw_seasonal":
        signal = np.cumsum(rng.normal(0, rng.uniform(0.02, 0.1), t)) + _seasonal(rng, t, [rng.choice([12, 24, 48, 96, 168])])
    elif kind == "multi_seasonal":
        signal = _seasonal(rng, t, [24, 168]) + np.cumsum(rng.normal(0, 0.01, t))
    elif kind == "level_shift":
        signal = np.zeros(t)
        for at in np.sort(rng.choice(np.arange(t // 8, t), rng.integers(1, 5), replace=False)):
            signal[at:] += rng.normal(0, 2.0)
        signal += 0.2 * _seasonal(rng, t, [24])
    elif kind == "spiky":
        signal = _seasonal(rng, t, [24]) + 0.3 * rng.standard_t(rng.uniform(2, 3), t)
        spikes = rng.random(t) < 0.005
        signal[spikes] += rng.choice([-1, 1], spikes.sum()) * rng.uniform(5, 15, spikes.sum())
        return level + unit * signal
    else:
        raise ValueError(kind)
    noise = rng.normal(0, signal.std() * rng.uniform(0.7, 1.4) + 1e-3, t)  # SNR ~ 1
    return level + unit * (signal + noise)


def _realistic_rows(rng, case: Case) -> np.ndarray:
    """(B * V, T): each item's variates share a latent series (weight 0.3-0.9) plus their own series."""
    kinds, weights = list(REALISTIC), np.array(list(REALISTIC.values()))
    t = case.context
    out = np.empty((case.batch * case.variates, t))
    for b in range(case.batch):
        pick = lambda: case.kind or kinds[rng.choice(len(kinds), p=weights / weights.sum())]
        shared = _realistic(rng, t, pick()) if case.variates > 1 else None
        for v in range(case.variates):
            own = _realistic(rng, t, pick())
            if shared is None:
                out[b * case.variates + v] = own
            else:
                w = rng.uniform(0.3, 0.9)
                out[b * case.variates + v] = w * shared + (1 - w) * own
    return out


def make_series(case: Case) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(target, target_mask, series_ids)``: (B, V, T) float32 with NaN replaced by 0, bool, (B, V) long."""
    rng = np.random.default_rng(case.seed)
    rows, t = case.batch * case.variates, case.context
    if case.mix == "realistic":
        data = _realistic_rows(rng, case)
    else:
        data = np.empty((rows, t), dtype=np.float64)
        for r in range(rows):
            data[r] = _smooth(rng, t)
    if case.pattern == "missing":
        for r in range(rows):
            kind = r % 5
            if kind in (1, 2):
                data[r, rng.choice(t, 10, replace=False)] = np.nan
            if kind == 2:
                data[r, 640:704] = np.nan
            if kind == 3:
                data[r, :90] = np.nan
            if kind == 4:
                data[r, t - 20 : t - 10] = np.nan
                data[r, 1024:1056] = np.nan
    x = torch.from_numpy(data.astype(np.float32)).reshape(case.batch, case.variates, t)
    mask = torch.isfinite(x)
    x = torch.where(mask, x, torch.zeros_like(x))
    ids = torch.zeros(case.batch, case.variates, dtype=torch.long)
    if case.series_ids == "mixed":
        ids[1::2, case.variates // 2 :] = 1
    return x, mask, ids


class Capture:
    """Forward hooks on the reference modules that ``forecast()`` calls directly."""

    def __init__(self, model, layers: set[int], capture: str):
        import toto2.model as tm

        self.store: dict[str, list[torch.Tensor]] = {}
        self.handles = []
        self._tm = tm
        self._orig_backfill = tm.backfill_short_patches
        p = model.config.patch_size

        def keep(name, t):
            self.store.setdefault(name, []).append(t.detach().float().clone())

        def scaler_hook(_m, _args, out):
            _, loc, scale = out
            keep("loc", loc[..., p - 1 :: p])
            keep("scale", scale[..., p - 1 :: p])

        def backfill(*args, **kwargs):
            loc, scale = self._orig_backfill(*args, **kwargs)
            keep("loc_bf", loc[..., p - 1 :: p])
            keep("scale_bf", scale[..., p - 1 :: p])
            return loc, scale

        tm.backfill_short_patches = backfill
        self.handles.append(model.scaler.register_forward_hook(scaler_hook))

        def head_hook(_m, args, out):
            keep("head_in", args[0])
            keep("head_out", out)

        self.handles.append(model.output_head.register_forward_hook(head_hook))
        if capture == "head":
            return
        calls = {"n": 0}

        def patch_hook(_m, args, out):
            part = "context" if calls["n"] % 2 == 0 else "horizon"
            calls["n"] += 1
            keep(f"tokens_{part}", args[0])
            keep(f"embed_{part}", out)

        self.handles.append(model.patch_proj.register_forward_hook(patch_hook))
        for i in sorted(layers):
            self.handles.append(
                model.transformer.layers[i].register_forward_hook(lambda _m, _a, out, i=i: keep(f"layer_{i}", out))
            )
        self.handles.append(model.transformer.out_norm.register_forward_hook(lambda _m, _a, out: keep("out_norm", out)))

    def close(self):
        self._tm.backfill_short_patches = self._orig_backfill
        for h in self.handles:
            h.remove()


def captured_layers(capture: str, num_layers: int) -> set[int]:
    if capture == "all":
        return set(range(num_layers))
    if capture == "some":
        return {0, 1, num_layers // 2, num_layers - 2, num_layers - 1}
    return set()


def run_case(model, case: Case, out_path: Path) -> dict:
    target, mask, ids = make_series(case)
    layers = captured_layers(case.capture, model.config.num_layers)
    chunk = case.chunk or case.batch
    parts_dir = out_path.with_suffix(".parts")
    pieces, cpu_s = [], 0.0
    for start in range(0, case.batch, chunk):
        part = parts_dir / f"{start:05d}.pt"
        if part.is_file():
            saved = torch.load(part)
            pieces.append(saved["captures"])
            cpu_s += saved["cpu_s"]
            continue
        sl = slice(start, start + chunk)
        cap = Capture(model, layers, case.capture)
        try:
            t0 = time.perf_counter()
            q = model.forecast(
                {"target": target[sl], "target_mask": mask[sl], "series_ids": ids[sl]},
                horizon=case.horizon,
                **FORECAST_KWARGS,
            )
            dt = time.perf_counter() - t0
        finally:
            cap.close()
        captures = {k: (v[0] if len(v) == 1 else v) for k, v in cap.store.items()}
        captures["q"] = q.detach().float().clone()
        bad = [k for k, v in captures.items() if isinstance(v, list)]
        assert not bad, f"captures called more than once: {bad}"
        cpu_s += dt
        pieces.append(captures)
        if case.chunk:
            parts_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"captures": captures, "cpu_s": dt}, part)
        print(f"  {case.name}: items {start}..{min(start + chunk, case.batch) - 1} in {dt:.1f} s", flush=True)
    if len(pieces) == 1:
        captures = pieces[0]
    else:
        captures = {k: torch.cat([p[k] for p in pieces], dim=1 if k in ("head_out", "q") else 0) for k in pieces[0]}
    golden = {
        "case": asdict(case),
        "inputs": {"target": target, "target_mask": mask, "series_ids": ids, "horizon": case.horizon},
        "forecast_kwargs": FORECAST_KWARGS,
        "captures": captures,
        "cpu_s": cpu_s,
        "threads": torch.get_num_threads(),
    }
    _save(golden, out_path)
    if parts_dir.is_dir():
        for p in parts_dir.iterdir():
            p.unlink()
        parts_dir.rmdir()
    return {"cpu_s": cpu_s, "shapes": {k: list(v.shape) for k, v in captures.items()}}


def run_gluonts(model, spec: dict, out_path: Path) -> dict:
    """The leaderboard wrapper on 64 series of mixed length (some shorter than the context, some with NaN)."""
    from gluonts.dataset.common import ListDataset
    from toto2.model import Toto2GluonTSModel
    from toto2.configuration import Toto2GluonTSModelConfig

    rng = np.random.default_rng(spec["seed"])
    n, ctx, h = spec["series"], spec["context_length"], spec["horizon"]
    series = []
    kinds, weights = list(REALISTIC), np.array(list(REALISTIC.values()))
    for i in range(n):
        length = 5000 if i % 4 else int(rng.integers(300, ctx))
        kind = kinds[rng.choice(len(kinds), p=weights / weights.sum())]
        x = _realistic(rng, length, kind).astype(np.float32)
        if i % 3 == 1:
            x[rng.choice(length, 12, replace=False)] = np.nan
        if i % 7 == 2:
            x[length // 2 : length // 2 + 64] = np.nan
        series.append(x)
    dataset = ListDataset([{"start": "2020-01-01 00:00", "target": s} for s in series], freq="h")
    config = Toto2GluonTSModelConfig(prediction_length=h, context_length=ctx, target_dim=1)
    wrapper = Toto2GluonTSModel(model, config)

    seen = {}
    orig = model.forecast

    def forecast(inputs, horizon, **kwargs):
        seen["inputs"] = {k: v.detach().clone() for k, v in inputs.items()}
        seen["kwargs"] = dict(kwargs)
        out = orig(inputs, horizon, **kwargs)
        seen["q"] = out.detach().float().clone()
        return out

    model.forecast = forecast
    cap = Capture(model, set(), "head")
    try:
        t0 = time.perf_counter()
        forecasts = list(wrapper.create_predictor(batch_size=n, device="cpu").predict(dataset))
        cpu_s = time.perf_counter() - t0
    finally:
        cap.close()
        del model.forecast
    arrays = torch.from_numpy(np.stack([f.forecast_array for f in forecasts]).astype(np.float32))  # (n, 9, h)
    captures = {k: v[0] for k, v in cap.store.items()}
    captures["q"] = seen["q"]
    golden = {
        "case": dict(spec, kind="gluonts", wrapper_config=config.__dict__),
        "series": [torch.from_numpy(s) for s in series],
        "inputs": {**seen["inputs"], "horizon": h},
        "forecast_kwargs": {k: v for k, v in seen["kwargs"].items()},
        "captures": captures,
        "forecast_array": arrays,
        "cpu_s": cpu_s,
        "threads": torch.get_num_threads(),
    }
    _save(golden, out_path)
    return {
        "cpu_s": cpu_s,
        "shapes": {"forecast_array": list(arrays.shape), **{k: list(v.shape) for k, v in captures.items()}},
    }


def _save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.rename(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def pip_freeze() -> list[str]:
    out = []
    for dist in importlib.metadata.distributions():
        name, version = dist.metadata["Name"], dist.version
        direct = dist.read_text("direct_url.json")
        if direct:
            info = json.loads(direct)
            vcs = info.get("vcs_info")
            if vcs:
                sub = info.get("subdirectory")
                out.append(
                    f"{name} @ {vcs['vcs']}+{info['url']}@{vcs['commit_id']}" + (f"#subdirectory={sub}" if sub else "")
                )
                continue
        out.append(f"{name}=={version}")
    return sorted(out, key=str.lower)


def update_manifest(ckpt: str, snapshot: Path, entries: dict):
    """Read-modify-write under an exclusive lock: several golden jobs may run at once."""
    import fcntl

    with open(MANIFEST.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _update_manifest(ckpt, snapshot, entries)


def _update_manifest(ckpt: str, snapshot: Path, entries: dict):
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
    manifest.update(
        {
            "reference": {
                "repo": REFERENCE_REPO,
                "commit": REFERENCE_COMMIT,
                "package": "toto-2 2.0.0 (toto2/), dd-unit-scaling 0.1.0, unit-scaling 0.3.5",
                "install": f"toto-2 @ git+{REFERENCE_REPO}.git@{REFERENCE_COMMIT}#subdirectory=toto2",
            },
            "environment": {
                "venv": sys.prefix,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "threads": torch.get_num_threads(),
                "pip_freeze": pip_freeze(),
            },
            "calls": {
                "load": "Toto2Model.from_pretrained(repo, revision=sha).eval(), HF_HUB_OFFLINE=1",
                "forecast": FORECAST_KWARGS,
            },
            "input_generator": GENERATOR,
            "golden_dir": str(GOLDEN_DIR),
        }
    )
    ck = manifest.setdefault("checkpoints", {}).setdefault(ckpt, {})
    repo, rev = CHECKPOINTS[ckpt]
    weights = snapshot / "model.safetensors"
    ck.update({"repo": repo, "revision": rev, "snapshot": str(snapshot)})
    if ck.get("safetensors_size") != weights.stat().st_size or "safetensors_sha256" not in ck:
        ck["safetensors_sha256"] = sha256(weights)
        ck["safetensors_size"] = weights.stat().st_size
    goldens = manifest.setdefault("goldens", {})
    for key, entry in entries.items():
        goldens[key] = {**goldens.get(key, {}), **entry}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, choices=list(CHECKPOINTS))
    parser.add_argument("--cases", default="", help="comma-separated case names (default: all for the checkpoint)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(THREADS)
    from huggingface_hub import hf_hub_download
    from toto2 import Toto2Model

    repo, rev = CHECKPOINTS[args.ckpt]
    # The shared cache holds only config.json and model.safetensors of some repos (snapshot_download refuses those).
    snapshot = Path(hf_hub_download(repo, "model.safetensors", revision=rev, local_files_only=True)).parent
    model = Toto2Model.from_pretrained(repo, revision=rev).eval()
    wanted = set(filter(None, args.cases.split(",")))
    entries = {}
    todo = [c for c in CASES[args.ckpt] if not wanted or c.name in wanted]
    for case in todo:
        out = GOLDEN_DIR / args.ckpt / f"{case.name}.pt"
        if out.is_file() and not args.force:
            print(f"skip {out} (exists)")
            info = None
        else:
            print(f"run {args.ckpt}/{case.name}", flush=True)
            info = run_case(model, case, out)
        entries[f"{args.ckpt}/{case.name}.pt"] = {
            "sha256": sha256(out),
            "bytes": out.stat().st_size,
            "case": asdict(case),
            **(info or {}),
        }
        update_manifest(args.ckpt, snapshot, {k: entries[k] for k in [f"{args.ckpt}/{case.name}.pt"]})
    spec = GLUONTS.get(args.ckpt)
    if spec and (not wanted or spec["name"] in wanted):
        out = GOLDEN_DIR / args.ckpt / f"{spec['name']}.pt"
        info = None
        if not out.is_file() or args.force:
            print(f"run {args.ckpt}/{spec['name']}", flush=True)
            info = run_gluonts(model, spec, out)
        update_manifest(
            args.ckpt,
            snapshot,
            {
                f"{args.ckpt}/{spec['name']}.pt": {
                    "sha256": sha256(out),
                    "bytes": out.stat().st_size,
                    "case": spec,
                    **(info or {}),
                }
            },
        )
    print("done")


if __name__ == "__main__":
    main()
