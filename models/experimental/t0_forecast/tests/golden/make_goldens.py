# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU fp32 goldens for t0-gamma from the pinned reference (tfc-t0 v0.6.0), written outside git.

Runs only inside the reference venv, isolated from the worktree's import path (it imports nothing from it):

    ttq run --model t0-gamma --kind cpu --sh 'env -u PYTHONPATH -u VIRTUAL_ENV HF_HUB_OFFLINE=1 \
        /home/andy/bringup/venvs/t0-gamma/bin/python -I <this file> --out /home/andy/bringup/goldens/t0-gamma \
        --manifest <worktree>/models/experimental/t0_forecast/tests/golden/MANIFEST.json --only single,modules,e2e'

Every golden saves its inputs; device tests load them and never regenerate them.

    single        1 x 2048 -> 64 (the model notes' recipe): scaled window, loc / scale, patch encoder, the 24
                  layer outputs, each attention module's input and output, out_norm, decoder, head,
                  ``_single_pass``, raw21, the 9-level and GIFT-Eval-settings outputs     (S = 66)
    single_long   1 x 8192 -> 1024, the same capture points                              (S = 288)
    modules       small random inputs through the real submodules: patch encoder (PAD / MISSING cells,
                  historical rows), a time layer (causal and PAD-masked), a group layer (G = 1 and G = 4),
                  an MLP, decoder + head
    paths         branch coverage: context 2000 (partial PAD patch), horizon 40, and 4 x 3 variates with
                  leading NaN trimmed to PAD (time and group masks at the full-model level)
    paper, long_context, multivariate
                  the bench shapes: inputs, head output at the read-out patches, raw21, cpu_s
    e2e           64 series x 2048 -> 64 with leading NaN, gaps and count rows; GIFT-Eval settings
                  (trim_leading_missing, symmetric_averaging, clip_nonnegative, snap_integers) at the 9 and
                  the 21 trained levels, plus the plain raw21
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ID = "theforecastingcompany/t0-gamma"
REVISION = "e08aee083778f1f32da4974c0a34276776a35c52"
SAFETENSORS_SHA256 = "1603e255629be22ea45518b502f16c55d06821c7090a1a851673e779e9c245a6"
REFERENCE = {
    "repo": "https://github.com/theforecastingcompany/tfc-t0",
    "tag": "v0.6.0",
    "commit": "4871c643c37d9bd8e7ade2500b9c3a5ba8519903",
    "package": "tfc-t0[torch]==0.6.0",
    "installed_from": "git+https://github.com/theforecastingcompany/tfc-t0@4871c643c37d9bd8e7ade2500b9c3a5ba8519903",
    "pins": {"rotary-embedding-torch": "0.9.1"},
}
CONVENTIONS = {
    "scaler_eps": 0.01,
    "scaler_eps_mode": "std_clamp",
    "xpos_scale_layout": "halves",
    "scaler_zero_no_value": False,
}
THREADS = 8
BENCH_SEEDS = {"paper": 41, "long_context": 42, "multivariate": 43}
GIFT_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SHAPES = {
    # name: (batch, variates, context, horizon, series per CPU chunk)
    "paper": (1024, 1, 2048, 64, 64),
    "long_context": (32, 1, 8192, 1024, 8),
    "multivariate": (128, 8, 2048, 64, 8),
}


def synthetic(n: int, width: int, seed: int) -> np.ndarray:
    """``[n, width]`` float32 series: level + trend + two seasonal harmonics + noise, per-row random parameters."""
    rng = np.random.default_rng(seed)
    t = np.arange(width, dtype=np.float64)
    col = lambda a: a[:, None]
    level = rng.uniform(-20.0, 200.0, n)
    amp = rng.uniform(0.2, 1.0, n) * np.maximum(np.abs(level), 1.0) * 0.1
    period = rng.choice([7.0, 12.0, 24.0, 48.0, 96.0, 168.0], n)
    phase = rng.uniform(0.0, 2 * np.pi, n)
    trend = rng.normal(0.0, 0.002, n) * np.maximum(np.abs(level), 1.0)
    noise = rng.uniform(0.02, 0.3, n) * amp
    y = (
        col(level)
        + col(trend) * t
        + col(amp) * np.sin(2 * np.pi * t / col(period) + col(phase))
        + 0.3 * col(amp) * np.sin(4 * np.pi * t / col(period) + 2 * col(phase))
        + col(noise) * rng.standard_normal((n, width))
    )
    return y.astype(np.float32)


KINDS = ("random_walk_seasonal", "noisy_seasonal", "intermittent", "level_shift", "trend")
KIND_PROBS = (0.30, 0.20, 0.15, 0.15, 0.20)


def _season(t: np.ndarray, period: float, phase: float) -> np.ndarray:
    return np.sin(2 * np.pi * t / period + phase) + 0.3 * np.sin(4 * np.pi * t / period + 2 * phase)


def _realistic_row(rng: np.random.Generator, kind: str, width: int) -> np.ndarray:
    """One series of ``kind``; magnitudes span 0.1..1000, seasonal SNR near 1 for the noisy kinds."""
    t = np.arange(width, dtype=np.float64)
    period = float(rng.choice([7.0, 12.0, 24.0, 48.0, 96.0, 168.0]))
    phase = rng.uniform(0.0, 2 * np.pi)
    scale = 10.0 ** rng.uniform(-1.0, 3.0)
    level = rng.normal(0.0, 3.0) * scale
    season = _season(t, period, phase)
    noise = rng.standard_normal(width)
    if kind == "random_walk_seasonal":
        walk = np.cumsum(rng.normal(0.0, rng.uniform(0.02, 0.1), width))
        return level + scale * (walk + season + rng.uniform(0.7, 1.3) * noise)
    if kind == "noisy_seasonal":
        envelope = 1.0 + 0.5 * np.sin(2 * np.pi * t / (7 * period) + phase)
        return level + scale * (envelope * season + rng.uniform(0.7, 1.3) * noise)
    if kind == "intermittent":
        rate = np.exp(rng.uniform(-0.5, 2.0) + 0.8 * season)
        return ((rng.random(width) < rng.uniform(0.05, 0.5)) * rng.poisson(rate)).astype(np.float64)
    if kind == "level_shift":
        steps = np.zeros(width)
        for start in rng.integers(0, width, int(rng.integers(1, 6))):
            steps[start:] += rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 5.0)
        return level + scale * (steps + 0.5 * season + rng.uniform(0.5, 1.0) * noise)
    trend = rng.normal(0.0, 0.002) * t
    return level + scale * (trend + season + 0.1 * noise)


def realistic(n: int, width: int, seed: int) -> tuple[np.ndarray, list[str]]:
    """``[n, width]`` float32 series drawn from ``KINDS`` with ``KIND_PROBS`` (seeded), and each row's kind."""
    rng = np.random.default_rng(seed)
    kinds = [KINDS[i] for i in rng.choice(len(KINDS), size=n, p=KIND_PROBS)]
    return np.stack([_realistic_row(rng, k, width) for k in kinds]).astype(np.float32), kinds


def realistic_groups(series: int, variates: int, width: int, seed: int) -> tuple[np.ndarray, list[str]]:
    """``[series, variates, width]``: each series' variates share a common component of its kind (loading 0.5..1.5)
    plus their own seasonal phase and noise, so group attention has cross-row signal."""
    rng = np.random.default_rng(seed)
    kinds = [KINDS[i] for i in rng.choice(len(KINDS), size=series, p=KIND_PROBS)]
    t = np.arange(width, dtype=np.float64)
    out = np.empty((series, variates, width))
    for b, kind in enumerate(kinds):
        common = _realistic_row(rng, kind, width)
        spread = max(float(np.std(common)), 1e-3)
        for v in range(variates):
            own = _season(t, float(rng.choice([12.0, 24.0, 168.0])), rng.uniform(0, 2 * np.pi))
            row = rng.uniform(0.5, 1.5) * common + spread * (0.3 * own + 0.5 * rng.standard_normal(width))
            out[b, v] = np.maximum(row, 0.0) if kind == "intermittent" else row
    return out.astype(np.float32), kinds


def e2e_inputs(width: int, seed: int) -> np.ndarray:
    """64 rows: 16 smooth, 16 nonnegative counts (some with exact zeros), 16 with leading NaN, 16 with gaps."""
    rng = np.random.default_rng(seed)
    x = synthetic(64, width, seed)
    t = np.arange(width)
    for r in range(16, 32):
        rate = np.clip(3.0 + 2.5 * np.sin(2 * np.pi * t / 24 + r), 0.05, None) * (0.2 if r % 4 == 0 else 1.0)
        x[r] = rng.poisson(rate).astype(np.float32)
    for r in range(32, 64):
        if r < 48 or r % 3 == 0:
            x[r, : int(rng.integers(40, 1500))] = np.nan
        if r >= 48:
            for _ in range(int(rng.integers(1, 6))):
                start = int(rng.integers(1600, width - 40))
                x[r, start : start + int(rng.integers(1, 40))] = np.nan
    return x


def save(data: dict, path: Path) -> None:
    """Write atomically: readers (device tests running meanwhile) see the old file or the new one."""
    tmp = path.with_suffix(".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model():
    from t0 import T0Forecaster

    m = T0Forecaster.from_pretrained(REPO_ID, revision=REVISION).eval()
    for key, value in CONVENTIONS.items():
        got = getattr(m.config, key)
        assert got == value, f"config {key} = {got!r}, expected {value!r}"
    return m


class Capture:
    """Forward hooks on the capture points of the contract; ``enabled`` gates recording."""

    def __init__(self, m, *, attention: bool = True):
        self.m = m
        self.data: dict[str, torch.Tensor] = {}
        self.enabled = False
        self.handles = []

        def out_hook(name):
            def hook(_module, _inputs, output):
                if self.enabled:
                    self.data[name] = output.detach().clone()

            return hook

        def io_hook(name):
            def hook(_module, inputs, output):
                if self.enabled:
                    self.data[f"{name}.in"] = inputs[0].detach().clone()
                    self.data[f"{name}.out"] = output.detach().clone()

            return hook

        self.handles.append(m.patch_encoder.register_forward_hook(out_hook("patch_encoder")))
        for i, layer in enumerate(m.transformer.layers):
            self.handles.append(layer.register_forward_hook(out_hook(f"layer_{i}")))
            if attention:
                self.handles.append(layer.attention_block.attention.register_forward_hook(io_hook(f"attention_{i}")))
        self.handles.append(m.transformer.out_norm.register_forward_hook(out_hook("out_norm")))
        self.handles.append(m.decoder.register_forward_hook(out_hook("decoder")))
        self.handles.append(m.head.register_forward_hook(out_hook("head")))

        scale_input, single_pass = m.scaler.scale_input, m._single_pass

        def wrapped_scale(series):
            scaled, loc_scale = scale_input(series)
            if self.enabled:
                self.data["scaled"] = scaled.variates.detach().clone()
                self.data["mask"] = series.mask.detach().clone()
                self.data["loc"] = loc_scale.loc.detach().clone()
                self.data["scale"] = loc_scale.scale.detach().clone()
            return scaled, loc_scale

        def wrapped_single(window):
            out = single_pass(window)
            if self.enabled:
                self.data["single_pass"] = out.detach().clone()
            return out

        m.scaler.scale_input = wrapped_scale
        m._single_pass = wrapped_single


class ReadoutHead:
    """Hook on ``m.head`` keeping only the read-out patches ``[P - n - 1, P - 1)`` of every forward."""

    def __init__(self, m, readout_count: int):
        self.n = readout_count
        self.chunks: list[torch.Tensor] = []
        self.handle = m.head.register_forward_hook(self.hook)

    def hook(self, _module, _inputs, output):
        p = output.shape[1]
        self.chunks.append(output[:, p - self.n - 1 : p - 1].detach().clone())


def golden_single(m, out: Path, context: int, horizon: int, name: str) -> dict:
    t = np.arange(context + horizon)
    y = (10 + np.sin(2 * np.pi * t / 24) + 0.01 * t + 0.1 * np.random.default_rng(0).standard_normal(t.size)).astype(
        np.float32
    )
    x = torch.from_numpy(y[None, :context])
    cap = Capture(m)
    cap.enabled = True
    raw21 = m.predict(x, horizon=horizon, quantile_levels=list(m.config.quantile_levels)).quantiles
    cap.enabled = False
    levels9 = m.predict(x, horizon=horizon, quantile_levels=GIFT_LEVELS).quantiles
    gift9 = m.predict(
        x,
        horizon=horizon,
        quantile_levels=GIFT_LEVELS,
        trim_leading_missing=True,
        symmetric_averaging=True,
        clip_nonnegative=True,
        snap_integers=True,
    ).quantiles
    data = {"context": x, "horizon": horizon, "raw21": raw21, "levels9": levels9, "gift9": gift9, **cap.data}
    path = out / f"{name}.pt"
    save(data, path)
    return {"path": path, "contents": sorted(data)}


def golden_modules(m, out: Path) -> dict:
    """Real submodules on random inputs (seed 1234)."""
    from t0.mask import build_attention_masks

    g = torch.Generator().manual_seed(1234)
    d, ps = m.config.embed_dim, m.config.patch_size
    data = {}
    # Patch encoder: 4 rows x 66 patches; PAD cells (left), MISSING cells, a historical row.
    values = torch.randn(4, 66, ps, generator=g)
    mask = torch.zeros(4, 66, ps, dtype=torch.int64)
    mask[1, :3] = 1
    mask[1, 3, :7] = 1
    mask[2, 10:12, ::3] = 2
    mask[3, 64:] = 4
    vtype = torch.zeros(4, 66, ps, dtype=torch.int64)
    vtype[3] = 1
    vtype[1, :3] = -1
    vtype[1, 3, :7] = -1
    data["patch_encoder"] = {
        "values": values,
        "mask": mask,
        "variate_type": vtype,
        "out": m.patch_encoder(values, mask, vtype).detach(),
    }
    # Layers on random activations, x ~ N(0, 4^2) (the real residual stream is O(1-10)).
    groups = torch.arange(4)[:, None, None].expand(4, 66, ps).clone()
    clean = torch.zeros(4, 66, ps, dtype=torch.int64)
    padded = clean.clone()
    padded[1, :3] = 1  # row 1: three all-PAD patches on the left
    padded_groups = groups.clone()
    padded_groups[1, :3] = -1
    vt = torch.zeros(4, 66, ps, dtype=torch.int64)
    vt_pad = vt.clone()
    vt_pad[1, :3] = -1
    x = 4.0 * torch.randn(4, 66, d, generator=g)
    for tag, (gids, cell_mask, types) in {
        "clean": (groups, clean, vt),
        "padded": (padded_groups, padded, vt_pad),
    }.items():
        tmask, gmask = build_attention_masks(gids, types, cell_mask)
        data[f"time_layer_{tag}"] = {
            "x": x,
            "group_ids": gids,
            "mask": cell_mask,
            "variate_type": types,
            "attention": m.transformer.layers[0].attention_block(x, tmask).detach(),
            "out": m.transformer.layers[0](x, tmask, gmask).detach(),
        }
    # Group layer: G = 1 (4 series) and G = 4 (one series of 4 variates), plus one PAD-masked G = 4 case.
    for tag, (row_groups, cell_mask, types) in {
        "g1": (torch.arange(4), clean, vt),
        "g4": (torch.zeros(4, dtype=torch.int64), clean, vt),
        "g4_padded": (torch.zeros(4, dtype=torch.int64), padded, vt_pad),
    }.items():
        gids = row_groups[:, None, None].expand(4, 66, ps).clone()
        gids[cell_mask == 1] = -1
        tmask, gmask = build_attention_masks(gids, types, cell_mask)
        data[f"group_layer_{tag}"] = {
            "x": x,
            "group_ids": gids,
            "mask": cell_mask,
            "variate_type": types,
            "attention": m.transformer.layers[2].attention_block(x, gmask).detach(),
            "out": m.transformer.layers[2](x, tmask, gmask).detach(),
        }
    data["mlp"] = {"x": x, "out": m.transformer.layers[0].mlp(m.transformer.layers[0].norm(x)).detach()}
    xd = 2.0 * torch.randn(4, 66, d, generator=g)
    decoded = m.decoder(m.transformer.out_norm(xd))
    data["head"] = {
        "x": xd,
        "decoder": decoded.detach(),
        "out": m.head(decoded.reshape(4, 66, ps, m.head.n_quantiles)).detach(),
    }
    path = out / "modules.pt"
    save(data, path)
    return {"path": path, "contents": sorted(data)}


def golden_bench(m, out: Path, name: str) -> dict:
    batch, variates, context, horizon, chunk = SHAPES[name]
    seed = BENCH_SEEDS[name]
    if variates > 1:
        values, kinds = realistic_groups(batch, variates, context, seed)
    else:
        values, kinds = realistic(batch, context, seed)
    x = torch.from_numpy(values)
    hook = ReadoutHead(m, horizon // m.config.patch_size)
    levels = list(m.config.quantile_levels)
    raw = []
    start = time.perf_counter()
    for i in range(0, batch, chunk):
        raw.append(m.predict(x[i : i + chunk], horizon=horizon, quantile_levels=levels).quantiles)
    cpu_s = time.perf_counter() - start
    hook.handle.remove()
    head = torch.cat(hook.chunks)
    data = {
        "context": x,
        "horizon": horizon,
        "raw21": torch.cat(raw),
        "head_readout": head,
        "cpu_s": cpu_s,
        "cpu_chunk": chunk,
        "kinds": kinds,
    }
    path = out / f"bench_{name}.pt"
    save(data, path)
    print(f"[golden] {name}: {tuple(x.shape)} -> {horizon}, cpu {cpu_s:.1f} s, head {tuple(head.shape)}", flush=True)
    return {"path": path, "contents": sorted(data), "cpu_s": cpu_s, "shape": list(x.shape)}


def golden_e2e(m, out: Path) -> dict:
    context, horizon = 2048, 64
    x = torch.from_numpy(e2e_inputs(context, seed=21))
    settings = dict(trim_leading_missing=True, symmetric_averaging=True, clip_nonnegative=True, snap_integers=True)
    levels21 = list(m.config.quantile_levels)
    hook = ReadoutHead(m, horizon // m.config.patch_size)
    start = time.perf_counter()
    gift9 = m.predict(x, horizon=horizon, quantile_levels=GIFT_LEVELS, **settings).quantiles
    cpu_s = time.perf_counter() - start
    gift_heads = list(hook.chunks)
    gift21 = m.predict(x, horizon=horizon, quantile_levels=levels21, **settings).quantiles
    hook.chunks.clear()
    raw21 = m.predict(x, horizon=horizon, quantile_levels=levels21).quantiles
    plain_head = hook.chunks[0]
    hook.handle.remove()
    data = {
        "context": x,
        "horizon": horizon,
        "settings": settings,
        "gift9": gift9,
        "gift21": gift21,
        "raw21": raw21,
        "head_readout": plain_head,
        "gift_head_readouts": gift_heads,
        "cpu_s": cpu_s,
    }
    path = out / "e2e.pt"
    save(data, path)
    return {"path": path, "contents": sorted(data), "cpu_s": cpu_s}


def golden_paths(m, out: Path) -> dict:
    """Branches the bench shapes never take: context 2000 (a partial left-PAD patch), horizon 40 (rounded up to
    two patches, 40 steps kept), and 4 series x 3 variates whose rows start with NaN, trimmed to PAD under the
    GIFT-Eval settings (all-PAD patches inside a series: the time mask and the group mask paths)."""
    levels21 = list(m.config.quantile_levels)
    uni = torch.from_numpy(realistic(8, 2000, seed=31)[0])
    multi = torch.from_numpy(realistic_groups(4, 3, 2048, seed=32)[0])
    for b, lead in enumerate((0, 100, 700, 1300)):
        multi[b, 1, :lead] = float("nan")
        multi[b, 2, : lead // 2] = float("nan")
    multi[1, :, :64] = float("nan")  # every row of series 1 starts with two all-missing patches
    # An explicit PAD mask on one row only: series 3 row 2 is padding for its first 20 patches, so its group
    # mixes real and all-PAD rows at those patches.
    multi_mask = torch.where(torch.isnan(multi), 2, 0).to(torch.int64)
    multi_mask[3, 2, :640] = 1
    settings = dict(trim_leading_missing=True, symmetric_averaging=True, clip_nonnegative=True, snap_integers=True)
    hook = ReadoutHead(m, 2)
    data = {
        "uni_context": uni,
        "uni_horizon": 40,
        "uni_raw21": m.predict(uni, horizon=40, quantile_levels=levels21).quantiles,
        "multi_context": multi,
        "multi_horizon": 64,
        "multi_mask": multi_mask,
        "multi_raw21": m.predict(multi, horizon=64, quantile_levels=levels21, mask=multi_mask).quantiles,
    }
    data["uni_head_readout"], data["multi_head_readout"] = hook.chunks[0], hook.chunks[1]
    hook.chunks.clear()
    data["multi_gift21"] = m.predict(multi, horizon=64, quantile_levels=levels21, **settings).quantiles
    data["multi_gift_head_readouts"] = list(hook.chunks)
    hook.handle.remove()
    path = out / "paths.pt"
    save(data, path)
    return {"path": path, "contents": sorted(data)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--only", default="single,single_long,modules,e2e,paper,long_context,multivariate")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(THREADS)
    args.out.mkdir(parents=True, exist_ok=True)
    m = load_model()

    import t0

    from huggingface_hub import hf_hub_download

    weights = Path(hf_hub_download(REPO_ID, "model.safetensors", revision=REVISION))
    st_sha = sha256(weights)
    assert st_sha == SAFETENSORS_SHA256, st_sha

    manifest = json.loads(args.manifest.read_text()) if args.manifest.is_file() else {}
    manifest.update(
        {
            "reference": {**REFERENCE, "t0_module": t0.__file__},
            "checkpoint": {
                "repo_id": REPO_ID,
                "revision": REVISION,
                "safetensors_sha256": st_sha,
                "config": json.loads((weights.parent / "config.json").read_text()),
            },
            "conventions": CONVENTIONS,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_num_threads": torch.get_num_threads(),
                "affinity_cpus": len(os.sched_getaffinity(0)),
                "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
                "pip_freeze": subprocess.run(
                    ["uv", "pip", "freeze", "--python", sys.executable], capture_output=True, text=True, check=True
                ).stdout.split(),
            },
            "generator": {
                "script": "models/experimental/t0_forecast/tests/golden/make_goldens.py",
                "script_sha256": sha256(Path(__file__)),
                "inputs": {
                    "single": "10 + sin(2 pi t / 24) + 0.01 t + 0.1 N(0,1) (numpy default_rng(0)), context 2048 / 8192",
                    "modules": "torch.Generator seed 1234; x = 4 N(0,1) for layers, 2 N(0,1) for the head",
                    "bench": "realistic / realistic_groups: per series a kind from KINDS with KIND_PROBS (random walk + seasonal at SNR ~1 0.30, noisy heteroscedastic seasonal 0.20, intermittent counts 0.15, level shifts 0.15, smooth trend 0.20), magnitudes 0.1..1000; multivariate variates share a series component; seeds paper 41, long 42, multi 43",
                    "e2e": "e2e_inputs(2048, seed=21): 16 smooth, 16 counts, 16 leading NaN, 16 gaps",
                    "paths": "realistic(8, 2000, 31) at horizon 40; realistic_groups(4, 3, 2048, 32) with leading NaN and a PAD row",
                },
                "seeds": {"torch.manual_seed": 0},
            },
        }
    )
    manifest.setdefault("goldens", {})
    jobs = {
        "single": lambda: golden_single(m, args.out, 2048, 64, "single"),
        "single_long": lambda: golden_single(m, args.out, 8192, 1024, "single_long"),
        "modules": lambda: golden_modules(m, args.out),
        "e2e": lambda: golden_e2e(m, args.out),
        "paths": lambda: golden_paths(m, args.out),
        **{name: (lambda n=name: golden_bench(m, args.out, n)) for name in SHAPES},
    }
    for name in args.only.split(","):
        start = time.perf_counter()
        info = jobs[name]()
        path = info.pop("path")
        manifest["goldens"][path.name] = {
            "sha256": sha256(path),
            "script_sha256": manifest["generator"]["script_sha256"],
            "bytes": path.stat().st_size,
            "created": datetime.now().isoformat(timespec="seconds"),
            "seconds": round(time.perf_counter() - start, 1),
            **info,
        }
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[golden] wrote {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
