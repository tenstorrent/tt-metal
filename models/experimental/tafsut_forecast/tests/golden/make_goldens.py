# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU fp32 goldens for Tafsut (CONTRACT §5, §6A). Run in the reference venv, under ``ttq run --kind cpu``:

    HF_HUB_OFFLINE=1 /home/andy/bringup/venvs/tafsut/bin/python -m \
        models.experimental.tafsut_forecast.tests.golden.make_goldens --out <dir> --manifest <dir>/MANIFEST.json \
        --only test        # or: --only bench

Every file holds its inputs. ``head`` is the reference's ``output_patch_embedding`` output ``[B, 32, 288]`` per
forward (asinh space, before the fallback), the ``pcc_normalized`` tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from models.experimental.tafsut_forecast.reference import dataflow as df

THREADS = 32
WHEEL_SHA256 = "8c892f4f521ae60014073e43faabae2c8e3427db554f035729c29590b4d5de1a"
GIFT_EVAL_COMMIT = "cfaefe0"
MIX_KINDS = ("seasonal_walk", "intermittent", "level_shift", "smooth_trend", "multi_seasonal", "spiky")
MIX_WEIGHTS = (0.35, 0.15, 0.15, 0.1, 0.15, 0.1)


def make_series_mix(n: int, length: int, seed: int, nan_frac: float = 0.005):
    """``([n, length] fp32, [n] kind)``: the realistic mix (skill §1 A). Random walk + daily/weekly seasonality at
    SNR ~1; intermittent counts (some end in a long zero run, which triggers the fallback); level shifts; smooth
    trends; multi-seasonal with drifting amplitude and a missing segment; spikes. Own scale 1e-1..1e3 and offset per
    series; 0.5 % interior NaNs (never the last step)."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(length, dtype=torch.float64)
    u = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64)
    rn = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    kinds = torch.multinomial(torch.tensor(MIX_WEIGHTS), n, replacement=True, generator=g)
    out = torch.empty(n, length, dtype=torch.float64)
    for r in range(n):
        k = MIX_KINDS[int(kinds[r])]
        p1, p2 = 2 * math.pi * float(u(1)), 2 * math.pi * float(u(1))
        season = float(u(1) + 0.5) * torch.sin(2 * math.pi * t / 24 + p1) + float(u(1)) * torch.sin(
            2 * math.pi * t / 168 + p2
        )
        if k == "seasonal_walk":
            x = rn(length).cumsum(0) * float(0.05 + 0.2 * u(1)) + season + rn(length) * float(season.std())
        elif k == "intermittent":
            x = (u(length) < float(0.03 + 0.1 * u(1))).double() * torch.round(torch.exp(rn(length) * 0.7 + 1.0))
            if float(u(1)) < 0.4:  # a long zero tail: some trigger the fallback, some don't
                x[-int(length * float(0.3 + 0.4 * u(1))) :] = 0.0
        elif k == "level_shift":
            cuts = torch.sort((u(int(1 + 4 * u(1))) * length).long()).values
            x = (rn(len(cuts) + 1) * 3)[torch.searchsorted(cuts, torch.arange(length))] + rn(length) * float(
                0.2 + 0.5 * u(1)
            )
        elif k == "smooth_trend":
            tt = t / length
            x = float(rn(1)) * 3 * tt + float(rn(1)) * 2 * tt**2 + 0.3 * season + rn(length) * 0.05
        elif k == "multi_seasonal":
            amp = 1 + 0.5 * torch.sin(2 * math.pi * t / length * float(1 + 2 * u(1)))
            x = amp * season + 0.7 * torch.sin(2 * math.pi * t / 8766 + p1) + rn(length) * float(0.5 + u(1))
            s0 = int(u(1) * (length - 200))
            x[s0 : s0 + int(20 + 100 * u(1))] = float("nan")
        else:  # spiky
            x = season * 0.5 + rn(length) * 0.3
            x += (u(length) < 0.01).double() * rn(length).abs() * 20
        out[r] = x
    x = 10 ** (4 * u(n, 1) - 1) * (out + (u(n, 1) - 0.3) * 10)
    nan = u(n, length) < nan_frac
    nan[:, -1] = False
    x[nan] = float("nan")
    return x.to(torch.float32), kinds


def ragged(lengths: list[int], seed: int, all_nan_row: int | None, gap_rows: int = 4):
    """GIFT batch construction: each series at its own length, NaN-left-padded to the batch maximum. Interior gaps
    covering whole patches on the first ``gap_rows`` rows; one all-NaN row."""
    T = max(lengths)
    x = torch.full((len(lengths), T), float("nan"))
    mix, _ = make_series_mix(len(lengths), T, seed)
    g = torch.Generator().manual_seed(seed + 1)
    for r, L in enumerate(lengths):
        x[r, T - L :] = mix[r, T - L :]
        if r < gap_rows and L > 400:
            s0 = T - L + int(torch.randint(0, L - 300, (1,), generator=g))
            x[r, s0 : s0 + 96 + 32 * r] = float("nan")  # >= 2 whole patches
    if all_nan_row is not None:
        x[all_nan_row] = float("nan")
    return x


def fallback_set():
    """16 x 2048 around the fallback boundaries (CONTRACT §2 `fallback`, D0 amendment 3)."""
    T, g = 2048, torch.Generator().manual_seed(7)
    rn = lambda *s: torch.randn(*s, generator=g)
    x = torch.empty(16, T)
    x[0] = 3.25  # exactly constant
    x[1] = 100.0 * (1 + 0.9e-4 * torch.sin(torch.arange(T) / 7.0))  # constant within rtol
    x[2] = 100.0 * (1 + 1.2e-4 * torch.sin(torch.arange(T) / 7.0))  # just outside rtol
    x[3] = rn(T)
    x[3, -1025:] = 2.0  # trailing run longer than half
    x[4] = rn(T)
    x[4, -1024:] = 2.0  # trailing run exactly half: no fallback
    x[5] = (torch.rand(T, generator=g) < 0.1).float() * 3
    x[5, -1200:] = 0.0  # intermittent, long zero tail
    x[6] = (torch.rand(T, generator=g) < 0.1).float() * 3
    x[6, -900:] = 0.0  # intermittent, short zero tail: no fallback
    x[7] = rn(T)
    x[7, :600] = float("nan")
    x[7, -800:] = 0.0
    x[7, -700:-500] = float("nan")  # left padded, tail spans a gap: 600 zeros of 1248 observed, no fallback
    x[8] = x[7]
    x[8, -1000:-800] = 0.0  # now 800 zeros of 1248: fallback
    x[9] = float("nan")  # all NaN
    x[10] = rn(T)
    x[10, -1100:] = torch.tensor([1.0, 1.0001]).repeat(550)  # even-count tail median, values inside rtol
    x[11] = rn(T)
    x[11, -1:] = float("nan")
    x[11, -1101:-1] = 5.0 + torch.arange(1100) % 2 * 5e-4  # on the rtol edge (fp32 isclose): fallback
    x[12] = 1e-7 * rn(T)  # atol regime: constant
    x[13] = rn(T).cumsum(0)  # controls
    x[14] = torch.sin(torch.arange(T) / 3.8) * 10 + 50
    x[15] = rn(T) * 1e3
    return x


class Capture:
    """Hooks on the reference model, per forward."""

    def __init__(self, m, blocks: bool):
        self.m, self.rec, self.h = m, [], []
        self._orig = {}

        def wrap(name):
            fn = getattr(m, name)
            self._orig[name] = fn

            def w(*a, **k):
                out = fn(*a, **k)
                if name == "_constant_fallback_candidates":
                    self.rec.append({"fb_mask": out[0].clone(), "fb_value": out[1].clone()})
                else:
                    tok, am, (loc, scale) = out
                    self.rec[-1].update(tokens=tok.clone() if blocks else None, key_valid=am.clone(), loc=loc, scale=scale)
                return out

            setattr(m, name, w)

        wrap("_constant_fallback_candidates")
        wrap("_prepare_patched_context")
        self.h.append(m.output_patch_embedding.register_forward_hook(lambda mod, i, o: self.rec[-1].update(head=o.clone())))
        if blocks:
            emb = []
            self.h.append(m.input_patch_embedding.register_forward_hook(lambda mod, i, o: emb.append(o.clone())))
            self.emb = emb
            for i, b in enumerate(m.encoder.block):
                self.h.append(b.register_forward_hook(lambda mod, a, o, i=i: self.rec[-1].setdefault("blocks", []).append(o.clone())))
                if i == 0:
                    self.h.append(b.layer[0].register_forward_hook(lambda mod, a, o: self.rec[-1].update(attn0=o.clone())))
            self.h.append(m.encoder.final_layer_norm.register_forward_hook(lambda mod, i, o: self.rec[-1].update(final_norm=o.clone())))

    def close(self):
        for h in self.h:
            h.remove()
        for k, f in self._orig.items():
            setattr(self.m, k, f)
        if hasattr(self, "emb"):
            for r, (c, f) in zip(self.rec, zip(self.emb[0::2], self.emb[1::2])):
                r.update(ctx_emb=c, fut_emb=f)


def run(m, x, H, blocks=False):
    from tafsut import forecast

    cap = Capture(m, blocks)
    try:
        q = forecast(m, x, horizon=H)
    finally:
        cap.close()
    fwd = [{k: v for k, v in r.items() if v is not None} for r in cap.rec]
    for r in fwd:
        if "blocks" in r:
            r["blocks"] = torch.stack(r["blocks"], 0)
    return q, fwd


def save(out: Path, name: str, x, H, q, fwd, **extra):
    assert torch.isfinite(q).all() or name.startswith("gift") or name in ("ragged4", "fallback"), name
    for f in fwd:
        assert torch.isfinite(f["head"]).all(), f"{name}: non-finite head"
    torch.save({"x": x, "horizon": H, "quantiles": q, "forwards": fwd, **extra}, out / f"{name}.pt")
    return {"fallback_rows": [int(f["fb_mask"].sum()) for f in fwd], "rows": int(x.shape[0])}


# ------------------------------------------------------------------------------------------------- test goldens
def gen_test(m, out: Path, info: dict):
    t = torch.arange(2048 + 64, dtype=torch.float64)
    import numpy as np

    noise = torch.from_numpy(np.random.default_rng(0).standard_normal(t.numel()))
    y = (10 + torch.sin(2 * math.pi * t / 24) + 0.01 * t + 0.1 * noise).float()
    x = y[None, :2048]
    q, fwd = run(m, x, 64, blocks=True)
    assert not fwd[0]["fb_mask"].any()
    # sdpa vs eager once (CONTRACT §5 nondeterminism)
    for b in m.encoder.block:
        b.layer[0].self_attention.attn_implementation = "eager"
    qe, _ = run(m, x, 64)
    for b in m.encoder.block:
        b.layer[0].self_attention.attn_implementation = "sdpa"
    info["single_sdpa_vs_eager_maxrel"] = float((q - qe).abs().max() / q.abs().max())
    info["single"] = save(out, "single", x, 64, q, fwd)

    xl, _ = make_series_mix(1, 32768, seed=2)
    q, fwd = run(m, xl, 64, blocks=True)
    info["single_long"] = save(out, "single_long", xl, 64, q, fwd)

    xr = ragged([2048, 1500, 700, 2048], seed=3, all_nan_row=3, gap_rows=2)
    q, fwd = run(m, xr, 64, blocks=True)
    info["ragged4"] = save(out, "ragged4", xr, 64, q, fwd)

    x16, kinds = make_series_mix(16, 2048, seed=4)
    q, fwd = run(m, x16, 64)
    info["model16"] = save(out, "model16", x16, 64, q, fwd, kinds=kinds)

    xf = fallback_set()
    q, fwd = run(m, xf, 64)
    info["fallback"] = save(out, "fallback", xf, 64, q, fwd)

    xh, _ = make_series_mix(4, 2048, seed=5)
    q, fwd = run(m, xh, 1100)
    info["horizon4"] = save(out, "horizon4", xh, 1100, q, fwd)


def ragged_vs_alone(m, x, H):
    """Max |batch - alone| / max|alone| over rows: a row inside the padded batch vs. the row at its own length."""
    from tafsut import forecast

    qb = forecast(m, x, horizon=H)
    worst = 0.0
    for r in range(x.shape[0]):
        row = x[r]
        fin = torch.isfinite(row).nonzero()
        if fin.numel() == 0:
            continue
        qa = forecast(m, row[int(fin[0]) :][None], horizon=H)
        worst = max(worst, float((qb[r] - qa[0]).abs().max() / qa.abs().max().clamp(min=1e-12)))
    return worst


# ------------------------------------------------------------------------------------------------ bench goldens
BENCH = {
    "paper": dict(n=1024, context=2048, horizon=64, seed=30),
    "long_context": dict(n=32, context=32768, horizon=64, seed=31),
    "long_horizon": dict(n=512, context=2048, horizon=2048, seed=32),
    "gift_ragged": dict(n=32, horizon=48, seed=33),
    "gift_ragged_odd": dict(n=16, horizon=48, seed=34),
}


def gift_lengths(n, lo, hi, seed, exact_max):
    g = torch.Generator().manual_seed(seed + 100)
    L = torch.exp(torch.empty(n).uniform_(math.log(lo), math.log(hi), generator=g)).long().tolist()
    L[0] = exact_max
    return [min(v, exact_max) for v in L]


def gen_bench(m, name: str, out: Path, info: dict):
    s = BENCH[name]
    if name.startswith("gift"):
        L = gift_lengths(s["n"], 300, 32768, s["seed"], 32768) if name == "gift_ragged" else gift_lengths(
            s["n"], 300, 4931, s["seed"], 4931
        )
        x = ragged(L, s["seed"], all_nan_row=s["n"] - 1)
        q, fwd = run(m, x, s["horizon"])
        info[name] = save(out, name, x, s["horizon"], q, fwd, lengths=L)
        info[name]["ragged_vs_alone_maxrel"] = ragged_vs_alone(m, x, s["horizon"])
        return
    x, kinds = make_series_mix(s["n"], s["context"], s["seed"])
    chunk = 256 if s["context"] <= 2048 else s["n"]
    qs, fwds = [], []
    for c in range(0, s["n"], chunk):  # series are independent at uniform length (CONTRACT §5 batch rules)
        q, fwd = run(m, x[c : c + chunk], s["horizon"])
        qs.append(q)
        fwds.append(fwd)
    fwd = [
        {k: torch.cat([f[i][k] for f in fwds], 0) for k in fwds[0][i]} for i in range(len(fwds[0]))
    ]
    info[name] = save(out, name, x, s["horizon"], torch.cat(qs, 0), fwd, kinds=kinds)


# --------------------------------------------------------------------------------------------------- manifest
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_manifest(out: Path, manifest: Path, info: dict, timings: dict):
    import tafsut

    old = json.loads(manifest.read_text()) if manifest.is_file() else {}
    freeze = subprocess.run(["uv", "pip", "freeze", "--python", sys.executable], capture_output=True, text=True).stdout
    src = Path(tafsut.__file__).parent
    ck = Path(df.checkpoint_path()).parent
    data = {
        **old,
        "model": df.REPO,
        "hf_revision": df.REVISION,
        "code": {"package": "tafsut==0.1.0", "wheel_sha256": WHEEL_SHA256,
                 "repo": "https://github.com/Tafsut-FM/tafsut", "commit": "25a4ed55b2c8c67bbd62dcaf92a46680c0c9ac7e"},
        "gift_eval": {"repo": "https://github.com/SalesforceAIResearch/gift-eval", "commit": GIFT_EVAL_COMMIT,
                      "notebook": "notebooks/tafsut.ipynb",
                      "settings": "batch 32, truncate to 32768, NaN-left-pad to the batch maximum, 9 config quantiles"},
        "checkpoint_files": {n: sha256(ck / n) for n in ("model.safetensors", "config.json")},
        "source_sha256": {f"tafsut/{p.name}": sha256(p) for p in sorted(src.glob("*.py"))},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "pip_freeze": freeze.strip().splitlines(),
        "threads": THREADS,
        "env": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE",)},
        "calls": {"load": f'TafsutModel.from_pretrained("{df.REPO}", device="cpu", revision="{df.REVISION}", '
                  "local_files_only=True).eval()", "forecast": "tafsut.forecast(m, x, horizon=H)"},
        "input_generator": {"bench": f"make_series_mix kinds {list(MIX_KINDS)} weights {list(MIX_WEIGHTS)}",
                            "seeds": {"single": "numpy default_rng(0) (doc recipe)", "single_long": 2, "ragged4": 3,
                                      "model16": 4, "fallback": 7, "horizon4": 5,
                                      **{k: v["seed"] for k, v in BENCH.items()}}},
        "bench_shapes": BENCH,
        "cases": {**old.get("cases", {}), **info},
        "timings_s": {**old.get("timings_s", {}), **timings},
        "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(out.glob("*.pt"))},
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest.write_text(json.dumps(data, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=["test"], help="test, bench, or bench names")
    args = ap.parse_args()
    assert "python_env" not in sys.executable, "run in the reference venv"
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(THREADS)
    from tafsut import TafsutModel

    m = TafsutModel.from_pretrained(df.REPO, device="cpu", revision=df.REVISION, local_files_only=True).eval()
    info, timings = {}, {}
    todo = []
    for s in args.only:
        todo += list(BENCH) if s == "bench" else [s]
    with torch.inference_mode():
        for step in todo:
            t0 = time.perf_counter()
            gen_test(m, args.out, info) if step == "test" else gen_bench(m, step, args.out, info)
            timings[step] = round(time.perf_counter() - t0, 1)
            print(f"[golden] {step}: {timings[step]} s", flush=True)
            write_manifest(args.out, args.manifest, info, timings)


if __name__ == "__main__":
    main()
