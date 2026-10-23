# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Generate the TimesFM-3 CPU goldens with the pinned reference (run in the reference venv, offline).

    env -u PYTHONPATH -u VIRTUAL_ENV HF_HUB_OFFLINE=1 /home/andy/bringup/venvs/timesfm3/bin/python \
        models/experimental/timesfm3_forecast/tests/golden/make_goldens.py [--only a,b] [--out DIR]

Every golden saves its inputs; the device tests load them and never regenerate them. Files go to
``/home/andy/bringup/goldens/timesfm3`` (outside git); ``MANIFEST.json`` next to this script records the
provenance and the sha256 of each file.
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
from pathlib import Path

import numpy as np
import torch

REPO = "https://github.com/google-research/timesfm"
COMMIT = "e51928e27119cb17bebc005be2696b75e0a9e688"
HF_REPO = "google/timesfm-3.0-pytorch"
HF_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
THREADS = 8
DEFAULT_OUT = Path("/home/andy/bringup/goldens/timesfm3")
MANIFEST = Path(__file__).resolve().parent / "MANIFEST.json"

PATCH = 32


# ----------------------------------------------------------------------------------------------- inputs
def synthetic(rng: np.random.Generator, count: int, length: int) -> np.ndarray:
    """(count, length) float32: scaled level + trend + 1-3 seasonalities + AR(1) noise, varied per series."""
    t = np.arange(length, dtype=np.float64)
    out = np.empty((count, length))
    for i in range(count):
        level = rng.uniform(-20, 60)
        slope = rng.normal(0, 0.02) * (rng.random() < 0.6)
        y = level + slope * t
        for _ in range(rng.integers(1, 4)):
            period = rng.choice([7, 12, 24, 48, 96, 168, 365])
            y += rng.uniform(0.2, 4.0) * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
        phi, sigma = rng.uniform(0, 0.9), rng.uniform(0.1, 1.5)
        e = rng.normal(0, sigma, length)
        for k in range(1, length):
            e[k] += phi * e[k - 1]
        out[i] = (y + e) * 10.0 ** rng.uniform(-1.5, 2.5)
    return out.astype(np.float32)


def synthetic_multivariate(rng: np.random.Generator, items: int, variates: int, length: int) -> np.ndarray:
    """(items, variates, length) float32: a shared latent per item, mixed per variate with its own noise."""
    latent = synthetic(rng, items * 2, length).reshape(items, 2, length).astype(np.float64)
    latent = (latent - latent.mean(-1, keepdims=True)) / (latent.std(-1, keepdims=True) + 1e-6)
    mix = rng.normal(0, 1, (items, variates, 2))
    own = synthetic(rng, items * variates, length).reshape(items, variates, length).astype(np.float64)
    own = (own - own.mean(-1, keepdims=True)) / (own.std(-1, keepdims=True) + 1e-6)
    scale = 10.0 ** rng.uniform(-1, 2, (items, variates, 1))
    offset = rng.uniform(-10, 50, (items, variates, 1))
    lag = np.roll(np.einsum("ivk,ikt->ivt", mix, latent), 3, axis=-1)
    return ((lag + 0.5 * own) * scale + offset).astype(np.float32)


REALISTIC_MIX = {"walk_seasonal": 0.4, "intermittent": 0.2, "level_shift": 0.25, "smooth_trend": 0.15}


def realistic(rng: np.random.Generator, count: int, length: int) -> np.ndarray:
    """(count, length) float32 bench inputs where the forecast spread matters (PLAN section 9, v1.1):

    - ``walk_seasonal`` (40 %): a random walk plus 1-2 seasonalities, with white noise near SNR 1;
    - ``intermittent`` (20 %): mostly zeros, with lognormal demand at 2-30 % of the steps;
    - ``level_shift`` (25 %): piecewise-constant levels with 1-6 random jumps, mild seasonality and noise;
    - ``smooth_trend`` (15 %): the old smooth trend + seasonality + AR(1) generator.

    Each series is then scaled by 10^U(-1.5, 2.5) and offset by U(-20, 60) times its scale (intermittent
    series keep their zeros and get no offset).
    """
    t = np.arange(length, dtype=np.float64)
    kinds = rng.choice(list(REALISTIC_MIX), size=count, p=list(REALISTIC_MIX.values()))
    out = np.empty((count, length))
    for i, kind in enumerate(kinds):
        if kind == "walk_seasonal":
            walk = np.cumsum(rng.normal(0, rng.uniform(0.05, 0.5), length))
            amp = rng.uniform(0.5, 3.0)
            season = sum(
                amp * rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * t / p + rng.uniform(0, 2 * np.pi))
                for p in rng.choice([7, 12, 24, 48, 168], rng.integers(1, 3), replace=False)
            )
            y = walk + season + rng.normal(0, amp * rng.uniform(0.7, 1.4), length)
        elif kind == "intermittent":
            active = rng.random(length) < rng.uniform(0.02, 0.3)
            y = np.where(active, rng.lognormal(rng.uniform(0, 2), rng.uniform(0.3, 1.0), length), 0.0)
        elif kind == "level_shift":
            cuts = np.sort(rng.choice(np.arange(1, length), rng.integers(1, 7), replace=False))
            levels = np.cumsum(rng.normal(0, rng.uniform(1, 5), len(cuts) + 1))
            y = levels[np.searchsorted(cuts, t, side="right")]
            y = y + 0.3 * np.sin(2 * np.pi * t / rng.choice([24, 168]) + rng.uniform(0, 6.3))
            y = y + rng.normal(0, rng.uniform(0.2, 1.0), length)
        else:
            y = synthetic(rng, 1, length)[0].astype(np.float64)
            y = (y - y.mean()) / (y.std() + 1e-6)
        scale = 10.0 ** rng.uniform(-1.5, 2.5)
        offset = 0.0 if kind == "intermittent" else rng.uniform(-20, 60)
        out[i] = (y + offset) * scale
    return out.astype(np.float32)


def realistic_multivariate(rng: np.random.Generator, items: int, variates: int, length: int) -> np.ndarray:
    """(items, variates, length) float32: per item, a ``realistic`` latent mixed into every variate (lagged)
    plus each variate's own ``realistic`` series, so variate attention has real cross-series signal."""
    latent = realistic(rng, items, length).astype(np.float64)
    latent = (latent - latent.mean(-1, keepdims=True)) / (latent.std(-1, keepdims=True) + 1e-6)
    own = realistic(rng, items * variates, length).reshape(items, variates, length).astype(np.float64)
    own = (own - own.mean(-1, keepdims=True)) / (own.std(-1, keepdims=True) + 1e-6)
    mix = rng.normal(0, 1, (items, variates, 1))
    lag = rng.integers(0, 6, (items, variates))
    shared = np.stack([[np.roll(latent[i], lag[i, u]) for u in range(variates)] for i in range(items)])
    scale = 10.0 ** rng.uniform(-1, 2, (items, variates, 1))
    offset = rng.uniform(-10, 50, (items, variates, 1))
    return ((mix * shared + 0.7 * own + offset) * scale).astype(np.float32)


def nonnegative_series(rng: np.random.Generator, count: int, length: int) -> np.ndarray:
    """Count-like non-negative series (for ``make_positive``)."""
    t = np.arange(length)
    rate = rng.uniform(2, 40, (count, 1)) * (1 + 0.6 * np.sin(2 * np.pi * t / rng.choice([24, 7, 168], (count, 1))))
    return rng.poisson(np.clip(rate, 0, None)).astype(np.float32)


# ------------------------------------------------------------------------------------------- reference
def load_forecaster(batch_size: int = 64):
    from timesfm3 import TimesFM3Forecaster

    return TimesFM3Forecaster.from_pretrained(
        HF_REPO, device="cpu", revision=HF_REVISION, per_core_batch_size=batch_size, local_files_only=True
    )


class HeadCapture:
    """Forward hook on ``output_head``: the raw head output (normalized space) of every decode batch."""

    def __init__(self, model):
        self.outputs = []
        self.handle = model.output_head.register_forward_hook(lambda m, i, o: self.outputs.append(o.detach().clone()))

    def take(self, start: int, count: int) -> torch.Tensor:
        """(rows, count, 576) at tokens [start, start + count), rows = items * variates."""
        out = torch.cat(self.outputs)[:, :, start : start + count]
        self.outputs = []
        return out.reshape(-1, count, out.shape[-1]).clone()  # not a view: torch.save would keep every token

    def close(self):
        self.handle.remove()


def readout(context_len: int, horizon: int) -> tuple[int, int]:
    import math

    h = math.ceil(horizon / 64) * 64
    return context_len // PATCH - 1, max(math.ceil((h - 32) / 32), 1)


def predict(fc, contexts, horizon, **kw):
    out = list(fc.predict_batch(contexts=list(contexts), horizon=horizon, return_quantiles=True, **kw))
    return np.stack([o.quantiles for o in out]), np.stack([o.forecast for o in out])


def decode_capture(model, target: torch.Tensor, mask: torch.Tensor, horizon: int, layers=None, sublayers=False):
    """``model.decode`` with aux outputs and per-layer (optionally per-sublayer) hooks."""
    layers = range(len(model.transformer_stack.layers)) if layers is None else layers
    caps, handles = {}, []
    for i in layers:
        layer = model.transformer_stack.layers[i]

        def hook(name):
            def fn(m, inp, out):
                caps[name] = (out[0] if isinstance(out, tuple) else out).detach().clone()

            return fn

        handles.append(layer.register_forward_hook(hook(f"layer{i}")))
        if sublayers:
            for sub in ("seq_attn", "var_attn", "ff1", "post_seq_attn_ln", "post_var_attn_ln", "post_ff_ln"):
                handles.append(getattr(layer, sub).register_forward_hook(hook(f"layer{i}.{sub}")))
    head = HeadCapture(model)
    with torch.inference_mode():
        horizon_logits, aux = model.decode(target=target, horizon=horizon, mask=mask, return_aux_outputs=True)
    for h in handles:
        h.remove()
    raw = torch.cat(head.outputs)
    head.close()
    return horizon_logits, aux, caps, raw


def predict_recording(fc, contexts, horizon, **kw):
    """``predict_batch`` with ``model.decode`` wrapped to record the (target, mask, horizon) of each batch."""
    model, calls = fc.model, []
    orig = model.decode

    def wrapped(*args, **dkw):
        calls.append((dkw["target"].clone(), dkw["mask"].clone(), dkw["horizon"]))
        return orig(*args, **dkw)

    model.decode = wrapped
    try:
        t0 = time.perf_counter()
        q, f = predict(fc, contexts, horizon, **kw)
        cpu_s = time.perf_counter() - t0
    finally:
        model.decode = orig
    return q, f, cpu_s, calls


def with_batch(fc, batch: int):
    import dataclasses

    fc.config = dataclasses.replace(fc.config, per_core_batch_size=batch)
    return fc


# --------------------------------------------------------------------------------------------- goldens
def golden_single(fc):
    """One series, 2048 -> 64, every intermediate (the doc's golden workload)."""
    rng = np.random.default_rng(0)
    t = np.arange(2048)
    ctx = (10 + 0.01 * t + 3 * np.sin(2 * np.pi * t / 24) + rng.standard_normal(2048)).astype(np.float32)
    target, mask = torch.from_numpy(ctx)[None, None], torch.zeros(1, 2048, dtype=torch.bool)
    hl, aux, caps, raw = decode_capture(fc.model, target, mask, 64, sublayers=True)
    q, f = predict(with_batch(fc, 1), [ctx], 64)
    return {
        "context": torch.from_numpy(ctx),
        "horizon": 64,
        "resblock_input": aux["__call__:resblock_input"],
        "transformer_input": aux["__call__:transformer_input"],
        "seq_attn_mask": aux["__call__:seq_attn_mask"][0],
        "layers": {k: v for k, v in caps.items()},
        "transformer_output": aux["__call__:transformer_output"],
        "head_raw": raw,
        "logits": aux["logits"],
        "revin_mu": aux["revin_stats"][0],
        "revin_sigma": aux["revin_stats"][1],
        "horizon_logits": hl,
        "quantiles": torch.from_numpy(q[0]),
        "forecast": torch.from_numpy(f[0]),
    }


def random_module_inputs(rng: np.random.Generator, b: int, v: int, n: int, front: list[list[int]]):
    """Random residual-stream states and leading-pad patch masks (b, v, n)."""
    x = torch.from_numpy(rng.normal(0, 1, (b, v, n, 1280)).astype(np.float32))
    x = x * torch.from_numpy(rng.uniform(0.5, 4.0, (b, v, 1, 1)).astype(np.float32))
    mask = torch.zeros(b, v, n, dtype=torch.bool)
    for i in range(b):
        for u in range(v):
            mask[i, u, : front[i][u]] = True
    return x, mask


def golden_modules(fc):
    """Small random-input module goldens on real weights: tokenizer, every sublayer, a full layer, the head."""
    model = fc.model
    rng = np.random.default_rng(1)
    out = {}
    # Tokenizer: realistic features (normalized values, zero future window, mask bits).
    tok = torch.zeros(8, 66, 192)
    tok[..., :32] = torch.from_numpy(rng.normal(0, 1.5, (8, 66, 32)).astype(np.float32))
    tok[..., 96:128] = torch.from_numpy((rng.random((8, 66, 32)) < 0.05).astype(np.float32))
    tok[..., :32] *= 1 - tok[..., 96:128]
    tok[:, 64:, :32], tok[:, 64:, 96:128] = 0.0, 1.0
    tok[..., 128:] = 1.0
    with torch.inference_mode():
        out["tokenizer"] = {"x": tok, "out": model.pre_transformer_resblock(tok)}
        cases = {
            "uni": (6, 1, 66, [[0], [0], [3], [10], [33], [65]]),
            "mv": (2, 8, 66, [[0, 0, 0, 0, 0, 0, 0, 0], [5, 5, 0, 12, 5, 66, 0, 2]]),
            "mv32": (1, 32, 34, [[int(k) for k in rng.integers(0, 8, 32)]]),
        }
        for li, name in ((0, "uni"), (0, "mv"), (0, "mv32"), (19, "uni"), (19, "mv")):
            layer = model.transformer_stack.layers[li]
            b, v, n, front = cases[name]
            x, pm = random_module_inputs(rng, b, v, n, front)
            eff = torch.cumprod(pm.int(), dim=2).bool()
            y, _, _ = layer(x, eff)
            # Sublayers, run the way MixingTransformer.forward runs them.
            s_in = layer.pre_seq_attn_ln(x).reshape(b * v, n, 1280)
            s_out, _, _ = layer.seq_attn(s_in, patch_mask=eff.reshape(b * v, n))
            h1 = layer.post_seq_attn_ln(s_out.view(b, v, n, 1280)) + x
            v_in = layer.pre_var_attn_ln(h1).permute(0, 2, 1, 3).reshape(b * n, v, 1280)
            v_out, _, _ = layer.var_attn(v_in, patch_mask=eff.permute(0, 2, 1).reshape(b * n, v))
            h2 = layer.post_var_attn_ln(v_out.view(b, n, v, 1280).permute(0, 2, 1, 3)) + h1
            out[f"layer{li}_{name}"] = {"x": x, "patch_mask": pm, "h1": h1, "h2": h2, "out": y}
        hx = torch.from_numpy(rng.normal(0, 3, (4, 66, 1280)).astype(np.float32))
        out["head"] = {"x": hx, "out": model.output_head(hx)}
    return out


def golden_full(fc, contexts, horizon, *, layers=(0, 9, 19), light=False, **kw):
    """Full-model golden for one forecaster batch: decode intermediates, raw head, final quantiles.

    The decode inputs are the ones ``predict_batch`` itself built (recorded), so the goldens do not depend
    on the port's host code."""
    queries = len(contexts) * (2 if kw.get("use_symmetric_averaging") else 1)
    q, f, cpu_s, calls = predict_recording(with_batch(fc, queries), contexts, horizon, **kw)
    assert len(calls) == 1, f"expected one decode batch, got {len(calls)}"
    target, mask, h = calls[0]
    hl, aux, caps, raw = decode_capture(fc.model, target, mask, h, layers=() if light else layers)
    start, count = readout(target.shape[-1], horizon)
    heavy = ("resblock_input", "transformer_input", "transformer_output", "head_raw", "logits", "layers")
    out = {
        "contexts": [torch.from_numpy(np.asarray(c, dtype=np.float32)) for c in contexts],
        "horizon": horizon,
        "settings": kw,
        "target": target,
        "mask": mask,
        "resblock_input": aux["__call__:resblock_input"],
        "transformer_input": aux["__call__:transformer_input"],
        "seq_attn_mask": aux["__call__:seq_attn_mask"][0],
        "layers": caps,
        "transformer_output": aux["__call__:transformer_output"],
        "head_raw": raw,
        "head_readout": raw[:, :, start : start + count].reshape(-1, count, raw.shape[-1]).clone(),
        "readout_start": start,
        "logits": aux["logits"],
        "revin_mu": aux["revin_stats"][0],
        "revin_sigma": aux["revin_stats"][1],
        "horizon_logits": hl,
        "quantiles": torch.from_numpy(q),
        "forecast": torch.from_numpy(f),
        "cpu_s": cpu_s,
    }
    return {k: v for k, v in out.items() if not (light and k in heavy)}


def full16_contexts():
    rng = np.random.default_rng(2)
    series = list(synthetic(rng, 16, 2048))
    series[3] = series[3][-1500:]
    series[7] = series[7][-700:]
    s = series[11].copy()
    s[:40] = np.nan
    s[900:910] = np.nan
    series[11] = s
    series[13] = series[13][-64:]
    return series


def mv4_contexts():
    rng = np.random.default_rng(3)
    items = list(synthetic_multivariate(rng, 4, 8, 2048))
    items[2] = items[2][:, -1200:]
    items[3] = items[3].copy()
    items[3][5, 1000:1030] = np.nan
    return items


def golden_bench(fc, name: str, contexts, horizon: int, batch: int):
    """A bench shape: inputs, raw head output at the read-out tokens, final quantiles, cpu_s."""
    with_batch(fc, batch)
    cap = HeadCapture(fc.model)
    t0 = time.perf_counter()
    q, f = predict(fc, contexts, horizon)
    cpu_s = time.perf_counter() - t0
    length = np.asarray(contexts[0]).shape[-1]
    start, count = readout(-(-length // PATCH) * PATCH, horizon)
    raw = cap.take(start, count)
    cap.close()
    return {
        "name": name,
        "contexts": torch.from_numpy(np.stack(contexts)),
        "horizon": horizon,
        "head_readout": raw,
        "readout_start": start,
        "quantiles": torch.from_numpy(q),
        "forecast": torch.from_numpy(f),
        "cpu_s": cpu_s,
        "threads": torch.get_num_threads(),
    }


def gift_contexts():
    rng = np.random.default_rng(5)
    lengths = rng.integers(256, 2049, 64)
    pos = nonnegative_series(rng, 32, 2048)
    signed = synthetic(rng, 32, 2048)
    series = [pos[i // 2] if i % 2 == 0 else signed[i // 2] for i in range(64)]
    return [s[-int(n) :] for s, n in zip(series, lengths)]


def golden_host_fp64(fc):
    """The fp64 reference (``model.double()``) token features and running statistics for the full-model
    goldens' decode inputs: the host test's reference, since the fp32 reference's (x - mu) / sigma carries
    up to ~4e-5 of its own rounding on series with a large level / std ratio."""
    import copy

    model64 = copy.deepcopy(fc.model).double()
    model64.use_iterative_cpm_revin = False  # after the head, fp32-only; the captured tensors precede it
    t = np.arange(2048)
    rng = np.random.default_rng(0)
    single = (10 + 0.01 * t + 3 * np.sin(2 * np.pi * t / 24) + rng.standard_normal(2048)).astype(np.float32)
    cases = {
        "single_2048_h64": ([single], 64),
        "full16_2048_h64": (full16_contexts(), 64),
        "full16_2048_h128": (full16_contexts(), 128),
        "mv4x8_2048_h64": (mv4_contexts(), 64),
    }
    out = {}
    for name, (contexts, horizon) in cases.items():
        _, _, _, calls = predict_recording(with_batch(fc, len(contexts)), contexts, horizon)
        target, mask, h = calls[0]
        _, aux, _, _ = decode_capture(model64, target.double(), mask, h, layers=())
        out[name] = {
            "resblock_input": aux["__call__:resblock_input"],
            "revin_mu": aux["revin_stats"][0],
            "revin_sigma": aux["revin_stats"][1],
        }
    return out


GOLDENS = {
    "single_2048_h64": lambda fc: golden_single(fc),
    "modules": lambda fc: golden_modules(fc),
    "full16_2048_h64": lambda fc: golden_full(fc, full16_contexts(), 64),
    "full16_2048_h128": lambda fc: golden_full(fc, full16_contexts(), 128),
    "mv4x8_2048_h64": lambda fc: golden_full(fc, mv4_contexts(), 64),
    "gift64_h48": lambda fc: golden_full(
        fc, gift_contexts(), 48, light=True, use_symmetric_averaging=True, make_positive=True
    ),
    "host_fp64": golden_host_fp64,
    "bench_paper": lambda fc: golden_bench(
        fc, "paper", list(realistic(np.random.default_rng(20), 1024, 2048)), 64, 128
    ),
    "bench_long_context": lambda fc: golden_bench(
        fc, "long_context", list(realistic(np.random.default_rng(21), 32, 15360)), 1024, 32
    ),
    "bench_multivariate": lambda fc: golden_bench(
        fc, "multivariate", list(realistic_multivariate(np.random.default_rng(22), 128, 8, 2048)), 64, 16
    ),
}

GENERATORS = {
    "single_2048_h64": "doc recipe: 10 + 0.01 t + 3 sin(2 pi t / 24) + N(0, 1), default_rng(0)",
    "modules": "random states N(0,1) x U(0.5,4) per row, leading-pad masks per case, default_rng(1)",
    "full16_2048_h64": "synthetic(default_rng(2), 16, 2048); #3 len 1500, #7 len 700, #11 40 leading + 10 interior NaN, #13 len 64",
    "full16_2048_h128": "as full16_2048_h64, horizon 128",
    "mv4x8_2048_h64": "synthetic_multivariate(default_rng(3), 4, 8, 2048); item 2 len 1200, item 3 variate 5 30 NaN",
    "gift64_h48": "gift_contexts(): default_rng(5), 32 nonnegative + 32 signed, lengths U[256, 2048]",
    "host_fp64": "the full-model goldens' decode inputs through the fp64 reference (model.double())",
    "bench_paper": "realistic(default_rng(20), 1024, 2048): REALISTIC_MIX of walk+seasonal at SNR ~1, intermittent, level shifts, smooth trends",
    "bench_long_context": "realistic(default_rng(21), 32, 15360)",
    "bench_multivariate": "realistic_multivariate(default_rng(22), 128, 8, 2048)",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def pip_freeze() -> list[str]:
    try:
        res = subprocess.run(
            ["uv", "pip", "freeze", "--python", sys.executable], capture_output=True, text=True, check=True
        )
        return [l for l in res.stdout.splitlines() if l.strip()]
    except Exception:
        from importlib import metadata

        return sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())


def weights_file() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_REPO, "model.safetensors", revision=HF_REVISION, local_files_only=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=",".join(GOLDENS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    torch.set_num_threads(THREADS)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text()) if args.manifest.is_file() else {"files": {}}
    fc = load_forecaster()
    for name in args.only.split(","):
        t0 = time.perf_counter()
        data = GOLDENS[name](fc)
        path = args.out / f"{name}.pt"
        torch.save(data, path)
        manifest["files"][name] = {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "generator": GENERATORS[name],
            "cpu_s": data.get("cpu_s") if isinstance(data, dict) else None,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        print(f"[golden] {name}: {path.stat().st_size / 1e6:.1f} MB in {time.perf_counter() - t0:.1f} s", flush=True)
    wf = weights_file()
    manifest.update(
        {
            "repo": REPO,
            "commit": COMMIT,
            "package": "timesfm 3.0.2 (timesfm3.torch)",
            "hf_repo": HF_REPO,
            "hf_revision": HF_REVISION,
            "safetensors": {"path": str(wf), "sha256": sha256(wf), "bytes": wf.stat().st_size},
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threads": THREADS,
            "cpu_affinity": len(os.sched_getaffinity(0)),
            "pip_freeze": pip_freeze(),
            "venv": sys.prefix,
            "input_generator": {
                "script": "models/experimental/timesfm3_forecast/tests/golden/make_goldens.py",
                "sha256": sha256(Path(__file__)),
                "functions": "synthetic, synthetic_multivariate, realistic (REALISTIC_MIX), realistic_multivariate, "
                "nonnegative_series, full16_contexts, mv4_contexts, gift_contexts",
            },
            "settings": "predict_batch defaults (no symmetric averaging, no z-norm, detrend on, CPM refine on, "
            "stitching on, sorted) except gift64_h48 (use_symmetric_averaging, make_positive); one forecaster "
            "batch per full-model golden (per_core_batch_size = number of queries)",
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
