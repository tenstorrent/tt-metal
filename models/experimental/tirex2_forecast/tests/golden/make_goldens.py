# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Generate the fp32 CPU goldens for TiRex-2 from the pinned upstream reference.

Runs only inside the reference venv (Python 3.12, torch 2.14.1+cpu, tirex-2 0.4.0), never from
tt-metal's python_env, and with the worktree kept off the import path:

    ttq run --model tirex2 --kind cpu --sh 'env -u PYTHONPATH -u VIRTUAL_ENV HF_HUB_OFFLINE=1 \
        TORCHDYNAMO_DISABLE=1 /home/andy/bringup/venvs/tirex2/bin/python \
        models/experimental/tirex2_forecast/tests/golden/make_goldens.py --out /home/andy/bringup/goldens/tirex2 \
        --manifest models/experimental/tirex2_forecast/tests/golden/MANIFEST.json'

Every golden saves its inputs next to the outputs; the device tests load those inputs and never
regenerate them. Files (all ``torch.save`` dicts):

    single.pt           1 series x 2048 -> 64, every intermediate of the single pass
    model16.pt          16 series x 2048 -> 64: every block output, the mLSTM internals of blocks 0 and 10,
                        head output, forward output, q_one and the sign-flip q_full
    slstm_short.pt      every sLSTM block over 32 steps (tokens 192..223 of model16) from a zero state,
                        with the per-step states (y, c, n, m)
    mlstm_chunks.pt     every mLSTM block of the single series: the per-chunk states (C, n, m) of the
                        chunkwise form
    multivar4.pt        4 groups x 8 rows (1 target + 7 past covariates) x 2048 -> 64
    bench_<shape>.pt    the three bench shapes: inputs, head output at the read-out tokens, q_one, cpu_s
    e2e64.pt            64 series sign-flip forecast (q_full) for stage E
    pad_check.json      the constant-left-pad hypothesis check (per-layer states after the 192 pad tokens)
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

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

THREADS = 8
torch.set_num_threads(THREADS)

REPO = "NX-AI/TiRex-2-gifteval-pretrain"
REVISION = "54f88ac06ccfba9fe3e727aa8856b39775481f0f"
CODE = {
    "tirex-2": {
        "repo": "https://github.com/NX-AI/tirex-2",
        "tag": "v0.4.0",
        "commit": "614f46fd7c60e332d8a32e525403b341c2c1ce1e",
    },
    "flashrnn": {
        "repo": "https://github.com/NX-AI/flashrnn",
        "version": "1.0.8",
        "commit": "0b84fc30120ecccbde97976bc5b086f3215572d2",
    },
    "mlstm_kernels": {
        "repo": "https://github.com/NX-AI/mlstm_kernels",
        "version": "2.0.6",
        "commit": "99b3677991c02595f08919dd18b28ec7b191608a",
    },
}
NUM_BLOCKS = 12
PAD_TOKENS = 192  # all-NaN left-pad tokens at context 2048: (8192 - 2048) / 32
READOUT_FIRST = 255  # token 255 predicts the first future patch


def make_series(n: int, length: int, seed: int, nan_frac: float = 0.005) -> torch.Tensor:
    """``[n, length]`` synthetic series: level + trend + daily and weekly seasonality + AR noise + a few NaNs.

    Every series gets its own scale (1e-1 .. 1e2), level, slope, phases and noise, so the scaler and the
    recurrent gates see a spread of inputs. A fraction ``nan_frac`` of interior steps is NaN (never the
    last step).
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(length, dtype=torch.float64)
    u = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64)
    scale = 10 ** (3 * u(n, 1) - 1)
    level = 20 * (u(n, 1) - 0.5)
    slope = (u(n, 1) - 0.5) * 6 / length
    p1, p2 = 2 * math.pi * u(n, 1), 2 * math.pi * u(n, 1)
    a1, a2 = 2 * u(n, 1), 1.5 * u(n, 1)
    season = a1 * torch.sin(2 * math.pi * t / 24 + p1) + a2 * torch.sin(2 * math.pi * t / 168 + p2)
    eps = torch.randn(n, length, generator=g, dtype=torch.float64) * (0.2 + 0.5 * u(n, 1))
    noise = torch.zeros_like(eps)
    for i in range(1, length):  # AR(1), phi 0.7
        noise[:, i] = 0.7 * noise[:, i - 1] + eps[:, i]
    x = scale * (level + slope * t * 10 + season + noise)
    nan = u(n, length) < nan_frac
    nan[:, -1] = False
    x[nan] = float("nan")
    return x.to(torch.float32)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Capture:
    """Forward hooks on the reference modules; ``keep`` selects tokens/rows to store (default: all)."""

    def __init__(self, m, blocks=None, internals=(), cell_io=True, head_tokens=None):
        self.m = m
        self.data: dict[str, torch.Tensor] = {}
        self.handles = []
        self.patched = []
        blocks = range(NUM_BLOCKS) if blocks is None else blocks
        self.hook(m, "forward_out", lambda i, o: o, inputs=lambda i: {"x_padded": i[0]["x"]})
        self.hook(m.input_patch_embedding, "embed", lambda i, o: o, inputs=lambda i: {"tokens": i[0]})
        self.hook(m.stack_out_norm, "stack_out_norm", lambda i, o: o)
        sel = (lambda o: o) if head_tokens is None else (lambda o: o[:, head_tokens])
        self.hook(m.output_patch_embedding, "head_out", lambda i, o: sel(o))
        for b in blocks:
            blk = m.stack[b]
            self.hook(blk, f"block{b}.out", lambda i, o: o[0])
            if cell_io:
                tm = blk.time_mixer
                self.hook(tm.norm_lstm, f"block{b}.norm_lstm", lambda i, o: o)
                self.hook(tm.fwd_cell, f"block{b}.cell_out", lambda i, o: o)
            if b in internals:
                self._internals(b)

    def hook(self, module, name, fn, inputs=None):
        def h(mod, i, o):
            self.data[name] = fn(i, o).detach().clone()
            if inputs is not None:
                for k, v in inputs(i).items():
                    self.data[k] = v.detach().clone()

        self.handles.append(module.register_forward_hook(h))

    def _internals(self, b):
        cell = self.m.stack[b].time_mixer.fwd_cell
        if b % 2 == 0:  # mLSTM: q, k, v, i, f into the chunkwise backend and its h

            def h(mod, args, kwargs, out):
                for k in ("q", "k", "v", "i", "f"):
                    self.data[f"block{b}.mlstm.{k}"] = kwargs[k].detach().clone()
                self.data[f"block{b}.mlstm.h"] = out.detach().clone()

            self.handles.append(cell.mlstm_backend.register_forward_hook(h, with_kwargs=True))
        else:  # sLSTM: Wx and all per-step states (y, c, n, m) from the recurrence
            orig = cell.recurrence

            def rec(Wx, states=None, _orig=orig):
                h_all, last = _orig(Wx, states)
                self.data[f"block{b}.slstm.Wx"] = Wx.detach().clone()  # [B, T, 4 gates, 4 heads, 128]
                self.data[f"block{b}.slstm.states"] = h_all.detach().clone()  # [4 (y,c,n,m), B, T, 4, 128]
                return h_all, last

            cell.recurrence = rec
            self.patched.append((cell, orig))

    def close(self):
        for h in self.handles:
            h.remove()
        for cell, orig in self.patched:
            del cell.recurrence  # drop the instance attribute; the bound method comes back
        self.handles, self.patched = [], []


def univariate(x: torch.Tensor):
    from tirex2 import TimeseriesType

    return [TimeseriesType(target=row[None], past_covariates=None, future_covariates=None) for row in x]


def forecast(fm, ts, horizon, sign_flip):
    out = fm.forecast(ts, prediction_length=horizon, output_type="torch", batch_size=len(ts), tta_sign_flip=sign_flip)
    return torch.stack([o if o.ndim == 3 else o[None] for o in out]).reshape(len(ts), *out[0].shape[-2:])


def scaler_stats(x_padded: torch.Tensor):
    loc = torch.nan_to_num(torch.nanmean(x_padded, dim=-1, keepdim=True), nan=0.0)
    scale = torch.nan_to_num(torch.nanmean((x_padded - loc) ** 2, dim=-1, keepdim=True).sqrt(), nan=1.0)
    return loc, torch.clamp_min(scale, 1e-8)


def gen_single(fm, out: Path):
    m = fm.model
    x = make_series(1, 2048, seed=1)
    cap = Capture(m, internals=range(NUM_BLOCKS))
    q_one = forecast(fm, univariate(x), 64, False)
    data = dict(cap.data)
    cap.close()
    q_full = forecast(fm, univariate(x), 64, True)
    loc, scale = scaler_stats(data["x_padded"])
    torch.save(
        {"context": x, "horizon": 64, "loc": loc, "scale": scale, "q_one": q_one, "q_full": q_full, **data},
        out / "single.pt",
    )
    return data


def gen_mlstm_chunks(single: dict, out: Path):
    """Per-chunk states of the chunkwise form for every mLSTM block (single series), from the reference kernel."""
    import torch.nn.functional as F
    from mlstm_kernels.torch.chunkwise.native.fw import mlstm_chunkwise__recurrent_fw_C

    res = {}
    for b in range(0, NUM_BLOCKS, 2):
        q, k, v, i, f = (single[f"block{b}.mlstm.{n}"] for n in "qkvif")
        S = q.shape[2]
        pad = (-S) % 64
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        i = F.pad(i, (0, pad), value=-25.0)
        f = F.pad(f, (0, pad), value=25.0)
        nc = (S + pad) // 64
        vb = F.logsigmoid(f.reshape(*f.shape[:2], nc, 64)).cumsum(-1)
        C, n, m_ = mlstm_chunkwise__recurrent_fw_C(
            matK=k,
            matV=v,
            vecB=vb,
            vecI=i.reshape(*i.shape[:2], nc, 64),
            qk_scale=64**-0.5,
            chunk_size=64,
            num_chunks=nc,
        )
        res[f"block{b}.C"] = C.reshape(*C.shape[:2], nc + 1, 64, 128)  # state before chunk k (k = 0..nc)
        res[f"block{b}.n"] = n.reshape(*n.shape[:2], nc + 1, 64)
        res[f"block{b}.m"] = m_
    torch.save(res, out / "mlstm_chunks.pt")


def gen_model16(fm, out: Path):
    m = fm.model
    x = make_series(16, 2048, seed=2)
    cap = Capture(m, internals=(0, 10), cell_io=False)
    # norm_lstm inputs to every sLSTM block feed the short-sequence goldens
    for b in range(1, NUM_BLOCKS, 2):
        cap.hook(m.stack[b].time_mixer.norm_lstm, f"block{b}.norm_lstm", lambda i, o: o)
    q_one = forecast(fm, univariate(x), 64, False)
    data = dict(cap.data)
    cap.close()
    q_full = forecast(fm, univariate(x), 64, True)
    loc, scale = scaler_stats(data["x_padded"])
    norm_inputs = {k: data.pop(k) for k in list(data) if k.endswith("norm_lstm")}
    torch.save(
        {"context": x, "horizon": 64, "loc": loc, "scale": scale, "q_one": q_one, "q_full": q_full, **data},
        out / "model16.pt",
    )
    return norm_inputs


def gen_slstm_short(fm, norm_inputs: dict, out: Path, t0: int = 192, steps: int = 32):
    """Each sLSTM block over ``steps`` tokens from a zero state, with every per-step state."""
    m = fm.model
    res = {"t0": t0, "steps": steps}
    with torch.no_grad():
        for b in range(1, NUM_BLOCKS, 2):
            cell = m.stack[b].time_mixer.fwd_cell
            xn = norm_inputs[f"block{b}.norm_lstm"][:, t0 : t0 + steps].contiguous()
            Wx = torch.stack((cell.fgate(xn), cell.igate(xn), cell.zgate(xn), cell.ogate(xn)), dim=2)
            Wx = Wx.reshape(*Wx.shape[:-1], 4, -1)
            h_all, _ = cell.recurrence(Wx)
            res[f"block{b}.x"] = xn
            res[f"block{b}.Wx"] = Wx
            res[f"block{b}.states"] = h_all  # [4 (y, c, n, m), B, steps, 4, 128]
            res[f"block{b}.out"] = cell(xn)
    torch.save(res, out / "slstm_short.pt")


def multivariate_inputs(groups: int, covariates: int, length: int, seed: int):
    from tirex2 import TimeseriesType

    x = make_series(groups * (1 + covariates), length, seed=seed).reshape(groups, 1 + covariates, length)
    # Couple the target to its covariates so variate attention has something to read
    x[:, 0] = x[:, 0] + 0.5 * torch.nan_to_num(x[:, 1:].mean(1))
    ts = [TimeseriesType(target=g[:1], past_covariates=g[1:], future_covariates=None) for g in x]
    return x, ts


def gen_multivar4(fm, out: Path):
    m = fm.model
    x, ts = multivariate_inputs(4, 7, 2048, seed=3)
    cap = Capture(m, blocks=(0, 1, 11), cell_io=False)
    tok = slice(160, 285)
    for b in (0, 11):
        cap.hook(
            m.stack[b].variate_mixer,
            f"block{b}.variate_out",
            lambda i, o: o[tok],
            inputs=lambda i, b=b: {f"block{b}.variate_in": i[0][tok]},
        )
    meta = {}

    def pre(mod, args, kwargs):
        meta["group_vector"] = kwargs["group_vector"].clone()
        meta["target_mask"] = kwargs["target_mask"].clone()

    hp = m.stack[0].variate_mixer.register_forward_pre_hook(pre, with_kwargs=True)
    q_one = forecast(fm, ts, 64, False)
    hp.remove()
    data = dict(cap.data)
    cap.close()
    loc, scale = scaler_stats(data["x_padded"])
    torch.save(
        {
            "context": x,
            "horizon": 64,
            "variate_tokens": [tok.start, tok.stop],
            "loc": loc,
            "scale": scale,
            "q_one": q_one,
            **meta,
            **data,
        },
        out / "multivar4.pt",
    )


BENCH = {
    "paper": dict(kind="uni", n=1024, context=2048, horizon=64, seed=10),
    "long_context": dict(kind="uni", n=32, context=8192, horizon=928, seed=11),
    "multivariate": dict(kind="multi", n=128, covariates=7, context=2048, horizon=64, seed=12),
}


def gen_bench(fm, name: str, out: Path):
    spec = BENCH[name]
    m = fm.model
    if spec["kind"] == "uni":
        x = make_series(spec["n"], spec["context"], seed=spec["seed"])
        ts = univariate(x)
    else:
        x, ts = multivariate_inputs(spec["n"], spec["covariates"], spec["context"], seed=spec["seed"])
    horizon = spec["horizon"]
    readout = list(range(READOUT_FIRST, READOUT_FIRST + math.ceil(horizon / 32)))
    cap = Capture(m, blocks=(), cell_io=False, head_tokens=readout)
    forecast(fm, ts[:2], horizon, False)  # warm-up (allocator, thread pool)
    start = time.perf_counter()
    q_one = forecast(fm, ts, horizon, False)
    cpu_s = time.perf_counter() - start
    data = {k: cap.data[k] for k in ("x_padded", "head_out")}
    cap.close()
    loc, scale = scaler_stats(data["x_padded"])
    torch.save(
        {
            "spec": spec,
            "context": x,
            "horizon": horizon,
            "readout_tokens": readout,
            "loc": loc,
            "scale": scale,
            "q_one": q_one,
            "cpu_s": cpu_s,
            "threads": THREADS,
            **data,
        },
        out / f"bench_{name}.pt",
    )
    print(f"[golden] bench {name}: cpu_s {cpu_s:.2f}", flush=True)


def gen_e2e64(fm, out: Path):
    x = make_series(64, 2048, seed=20)
    start = time.perf_counter()
    q_full = forecast(fm, univariate(x), 64, True)
    cpu_s = time.perf_counter() - start
    q_one = forecast(fm, univariate(x), 64, False)
    torch.save({"context": x, "horizon": 64, "q_full": q_full, "q_one": q_one, "cpu_s": cpu_s}, out / "e2e64.pt")


def pad_check(fm, out: Path, seeds=(100, 101, 102, 103)):
    """Are the per-layer states after the 192 all-NaN left-pad tokens input-independent (context 2048)?"""
    import torch.nn.functional as F
    from mlstm_kernels.torch.chunkwise.native.fw import mlstm_chunkwise__recurrent_fw_C

    m = fm.model
    runs = []
    for seed in seeds:
        x = make_series(1, 2048, seed=seed)
        cap = Capture(m, internals=range(NUM_BLOCKS), cell_io=False)
        forecast(fm, univariate(x), 64, False)
        d = dict(cap.data)
        cap.close()
        r = {"embed": d["embed"][:, :PAD_TOKENS], "tokens": d["tokens"][:, :PAD_TOKENS]}
        for b in range(NUM_BLOCKS):
            r[f"block{b}.out"] = d[f"block{b}.out"][:, :PAD_TOKENS]
            if b % 2 == 0:
                k, v, i, f = (d[f"block{b}.mlstm.{n}"][:, :, :PAD_TOKENS] for n in "kvif")
                vb = F.logsigmoid(f.reshape(1, 4, 3, 64)).cumsum(-1)
                C, n, mm = mlstm_chunkwise__recurrent_fw_C(
                    matK=k, matV=v, vecB=vb, vecI=i.reshape(1, 4, 3, 64), qk_scale=0.125, chunk_size=64, num_chunks=3
                )
                r[f"block{b}.state.C"] = C[:, :, -64:]
                r[f"block{b}.state.n"] = n[:, :, -64:]
                r[f"block{b}.state.m"] = mm[:, :, -1:]
            else:
                s = d[f"block{b}.slstm.states"][:, :, PAD_TOKENS - 1]  # after token 191: [4 (y,c,n,m), 1, 4, 128]
                for j, nm in enumerate("ycnm"):
                    r[f"block{b}.state.{nm}"] = s[j]
        runs.append(r)
    report = {"pad_tokens": PAD_TOKENS, "context": 2048, "seeds": list(seeds), "max_abs_diff": {}, "max_abs": {}}
    for key in runs[0]:
        ref = runs[0][key]
        report["max_abs_diff"][key] = max(float((r[key] - ref).abs().max()) for r in runs[1:])
        report["max_abs"][key] = float(ref.abs().max())
    worst = max(report["max_abs_diff"].values())
    report["worst_max_abs_diff"] = worst
    report["confirmed_1e-6"] = worst <= 1e-6
    # The cached per-layer states themselves, so candidate (b) can be checked against them
    torch.save({k: v.clone() for k, v in runs[0].items() if ".state." in k}, out / "pad_states.pt")
    (out / "pad_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[golden] pad check: worst max abs diff {worst:.3e} over {len(report['max_abs_diff'])} tensors", flush=True)


def write_manifest(out: Path, manifest: Path, timings: dict):
    import importlib.metadata as md

    freeze = subprocess.run(["uv", "pip", "freeze", "--python", sys.executable], capture_output=True, text=True).stdout
    if not freeze:
        freeze = "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions()))
    ckpt = (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub"
        / ("models--" + REPO.replace("/", "--"))
        / "snapshots"
        / REVISION
    )
    import tirex2

    src = Path(tirex2.__file__).parent
    key_files = [
        "model/tirex2.py",
        *(
            f"model/component/{n}.py"
            for n in (
                "variate_mixing_block",
                "bi_xlstm",
                "mlstm_block",
                "flashrnn_slstm",
                "attention_block",
                "feedforward",
                "norm",
                "residual_block",
                "scaler",
                "patch_tokenizer",
                "postprocessor",
            )
        ),
    ]
    import flashrnn
    import mlstm_kernels

    fr, mk = Path(flashrnn.__file__).parent, Path(mlstm_kernels.__file__).parent
    old = json.loads(manifest.read_text()) if manifest.is_file() else {}
    data = {
        "model": REPO,
        "hf_revision": REVISION,
        "checkpoint_files": {n: sha256(ckpt / n) for n in ("model.safetensors", "config.json")},
        "code": CODE,
        "source_sha256": {
            **{f"tirex2/{k}": sha256(src / k) for k in key_files},
            "flashrnn/flashrnn/vanilla/slstm.py": sha256(fr / "flashrnn/vanilla/slstm.py"),
            "flashrnn/flashrnn/vanilla/__init__.py": sha256(fr / "flashrnn/vanilla/__init__.py"),
            "mlstm_kernels/torch/chunkwise/native/fw.py": sha256(mk / "torch/chunkwise/native/fw.py"),
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "pip_freeze": freeze.strip().splitlines(),
        "threads": THREADS,
        "env": {k: os.environ.get(k) for k in ("TORCHDYNAMO_DISABLE", "HF_HUB_OFFLINE")},
        "calls": {
            "load": f'load_model("{REPO}", device="cpu", hf_kwargs={{"revision": "{REVISION}"}})',
            "q_one": 'fm.forecast(ts, prediction_length=H, output_type="torch", batch_size=len(ts), tta_sign_flip=False)',
            "q_full": 'fm.forecast(ts, prediction_length=H, output_type="torch", batch_size=len(ts), tta_sign_flip=True)',
            "pad_context": "default (True)",
        },
        "input_generator": {
            "function": "make_series(n, length, seed, nan_frac=0.005) in tests/golden/make_goldens.py",
            "description": "scale 10**U(-1,2) x (level + trend + daily/weekly sines + AR(1) noise), 0.5 % interior NaNs",
            "seeds": {
                "single": 1,
                "model16": 2,
                "multivar4": 3,
                **{f"bench_{k}": v["seed"] for k, v in BENCH.items()},
                "e2e64": 20,
                "pad_check": [100, 101, 102, 103],
            },
            "multivariate": "groups of 1 target + 7 past covariates; target += 0.5 * nanmean-free covariate mean",
        },
        "bench_shapes": BENCH,
        "timings_s": {**old.get("timings_s", {}), **timings},
        "files": {
            p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()
        },
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest.write_text(json.dumps(data, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="subset of: single model16 multivar4 bench e2e64 pad")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert "tt-metal" not in torch.__file__ and "python_env" not in sys.executable, "run in the reference venv"

    from tirex2 import load_model

    fm = load_model(REPO, device="cpu", hf_kwargs={"revision": REVISION})
    fm.model.eval()
    todo = args.only or ["single", "model16", "multivar4", "bench", "e2e64", "pad"]
    timings = {}
    with torch.no_grad():
        for step in todo:
            start = time.perf_counter()
            if step == "single":
                gen_mlstm_chunks(gen_single(fm, args.out), args.out)
            elif step == "model16":
                gen_slstm_short(fm, gen_model16(fm, args.out), args.out)
            elif step == "multivar4":
                gen_multivar4(fm, args.out)
            elif step == "bench":
                for name in BENCH:
                    gen_bench(fm, name, args.out)
            elif step == "e2e64":
                gen_e2e64(fm, args.out)
            elif step == "pad":
                pad_check(fm, args.out)
            timings[step] = time.perf_counter() - start
            print(f"[golden] {step}: {timings[step]:.1f} s", flush=True)
    write_manifest(args.out, args.manifest, timings)


if __name__ == "__main__":
    main()
