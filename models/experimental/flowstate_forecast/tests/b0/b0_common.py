# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Shared host pieces of the B0 micro-benchmark (CONTRACT v1.1 §6 B0): weights without tsfm, B0 goldens, metrics."""

import glob
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from models.experimental.flowstate_forecast.reference import dataflow as dfl

GOLDENS = Path("/home/andy/bringup/goldens/flowstate")
B0_CASES = ("b0_s6_L682", "b0_s1_L4096", "b0_s0.25_L16384", "b0_s1_12_L16384")
WEIGHTS = glob.glob(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--ibm-granite--granite-timeseries-flowstate-r1/snapshots/"
        "b80e0f1b1c2d90edebe3d4f243ad046a3f935daa/model.safetensors"
    )
)


def ssm_params(layer):
    sd = load_file(WEIGHTS[0])
    p = f"encoder.layers.{layer}.ssm."
    return SimpleNamespace(**{k[len(p) :]: v for k, v in sd.items() if k.startswith(p) and "." not in k[len(p) :]})


def case_inputs(case, layer):
    """Bu (L, B, P) complex64 from the bf16-rounded u and the golden Bbar, the reference kernel K, Lbar, the
    reference x (complex64 FFT at the reference's own n = 2L) and the float64 truth."""
    g = torch.load(GOLDENS / f"{case}.pt", weights_only=True)
    ssm = ssm_params(layer)
    _, Bbar, K = dfl.ssm_constants(ssm, g["s"], g["L"])
    assert torch.equal(Bbar, g[f"Bbar{layer}"]) and torch.equal(K[1], g[f"Lbar{layer}"])
    u = g[f"u{layer}"].float()
    bu = u @ torch.cat((Bbar.real, Bbar.imag), 0).T
    Bu = torch.complex(bu[..., :512], bu[..., 512:])
    ref = dfl.fft_conv(Bu, K, N=2 * g["L"])
    h = g[f"x_head{layer}"]
    assert torch.allclose(ref[: h.shape[0]], h, rtol=1e-5, atol=1e-6), "reference x does not match the golden head"
    return SimpleNamespace(g=g, s=g["s"], L=g["L"], Bu=Bu, K=K, lbar=K[1], ref=ref, truth=dfl.recurrence(Bu, K[1]))


def metrics(x, ref, truth):
    """CONTRACT §6 B0 items 1, 2, 4, 5: whole rel-L2 vs the reference, per-channel p99 / max, the reference's own
    per-channel error against the truth (item 4 floor), non-finite count."""

    def ch(a, b):
        return ((a - b).abs().pow(2).sum((0, 1)).sqrt() / b.abs().pow(2).sum((0, 1)).sqrt().clamp(min=1e-30)).double()

    x = x.to(torch.complex128)
    ref64 = ref.to(torch.complex128)
    e = ch(x, ref64)
    own = ch(ref64, truth)
    return {
        "rel_l2": ((x - ref64).norm() / ref64.norm()).item(),
        "rel_l2_vs_truth": ((x - truth).norm() / truth.norm()).item(),
        "ch_p99": torch.quantile(e, 0.99).item(),
        "ch_max": e.max().item(),
        "ref_own_ch_max": own.max().item(),
        "nonfinite": int((~torch.isfinite(torch.view_as_real(x))).sum()),
    }


def median_ms(fn, sync, n=5, warm=2):
    for _ in range(warm):
        fn()
    sync()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        sync()
        t.append((time.perf_counter() - t0) * 1e3)
    return sorted(t)[n // 2]


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))
