# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Operand probes of the fused row matmul on one core (probes.py has the inputs): every experiment is run at every
pass schedule and its FP32 sums are stored with its inputs in $HY4_PROBE_DIR (default generated/hy4_preview/logs/mk9_probe)
as <experiment>__<fidelity>.pt; analysis.py reads the files without a device. One summary line per file is logged
(HY4_MK9): how many sums equal exact arithmetic on the stored operands.

    wread_*   x one-hot with value 1.0: the weights as the multiplier sees them (random weights; a mantissa sweep
              under a known block exponent; the same with no other non-zero weight in the face column)
    xread_*   a 0 / 1 selection as the weight: x as the multiplier sees it (dense and sparse x)
    pairs_*   a diagonal weight: products of one x value and one weight value, all other products zero

On the mock Galaxy the programs are compiled only."""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.fused.fidelity import lab as LAB
from models.experimental.hy4_preview.fused.fidelity import probes as PR

OUT_DIR = Path(os.environ.get("HY4_PROBE_DIR", "generated/hy4_preview/logs/mk9_probe"))
W_BITS = {"bfp8": 7, "bfp4": 3, "bf16": 8, "fp32": 11}
X_BITS = {"bf16": 8, "fp32": 11}
FIDELITIES = ["ref", "lofi", "hifi2", "hifi3", "hifi4", "p02"]
pytestmark = pytest.mark.use_module_device


def experiment(name):
    """(kt, ct, weight format, x format, [(w, x), ...])."""
    kind, *rest = name.split("_")
    if kind == "wread":
        what, fmt = rest
        kt, ct = 8, 2
        if what == "gauss":
            w = PR.gauss_weights(kt, ct)
        elif what == "sweep":
            w = PR.sweep_weights(kt, ct, W_BITS[fmt])
        elif what == "nopin":
            w = PR.sweep_weights(kt, ct, W_BITS[fmt], pin=False)
        else:
            assert what == "isolated", name
            w = PR.isolate_rows(PR.sweep_weights(kt, ct, W_BITS[fmt]))
        order = torch.arange(5, PR.TILE * kt, 16) if what == "isolated" else None
        xs, _ = PR.one_hot_runs(PR.TILE * kt, order=order)
        return kt, ct, fmt, "bf16", [(w, x) for x in xs]
    if kind == "xread":
        what, x_fmt = rest
        kt, ct = 2, 2
        w = PR.selection(64)
        density = 1.0 if what == "dense" else 1.0 / 16
        runs = [(w, PR.float_grid(X_BITS[x_fmt], list(range(-3, 4)), seed=20 + i, density=density)) for i in range(8)]
        return kt, ct, "bfp8", x_fmt, runs
    assert kind == "pairs", name
    fmt, x_fmt = rest
    kt, ct = 2, 2
    runs = []
    for i in range(24):
        diag = PR.float_grid(W_BITS[fmt], [-1], seed=100 + i, shape=(64,))
        x = PR.float_grid(X_BITS[x_fmt], [0], seed=200 + i)
        runs.append((torch.diag(diag), x))
    return kt, ct, fmt, x_fmt, runs


EXPERIMENTS = (
    [f"wread_{w}_{f}" for f in ("bfp8", "bfp4") for w in ("gauss", "sweep", "nopin", "isolated")]
    + ["wread_gauss_bf16", "wread_gauss_fp32"]
    + [f"xread_{w}_{f}" for f in ("bf16", "fp32") for w in ("dense", "sparse")]
    + [f"pairs_{f}_{xf}" for f, xf in (("bfp8", "bf16"), ("bfp4", "bf16"), ("bf16", "bf16"), ("fp32", "fp32"))]
)


@pytest.mark.parametrize("fidelity", FIDELITIES)
@pytest.mark.parametrize("name", EXPERIMENTS)
def test_probe(device, name, fidelity):
    kt, ct, fmt, x_fmt, runs = experiment(name)
    passes, step = LAB.FIDELITIES["lofi" if fidelity == "ref" else fidelity]
    op = LAB.Lab(device, kt, ct, fmt=fmt, x_fmt=x_fmt, passes=passes, step=step, ref=fidelity == "ref")
    outs, last_w = [], None
    try:
        for w, x in runs:
            if w is not last_w:
                op.set_w(w)
                last_w = w
            op.set_x(x)
            op()
            if LAB.MOCK:
                return
            outs.append(op.read())
    finally:
        op.close()
    ws = [w for w, _ in runs]
    same_w = all(w is ws[0] for w in ws)
    w_all = ws[0][None] if same_w else torch.stack(ws)
    stored = torch.stack([LAB.stored(w, fmt) for w in w_all]) if fmt != "fp32" else w_all
    xs = torch.stack([x for _, x in runs])
    xs_stored = xs.to(torch.bfloat16).float() if x_fmt == "bf16" else xs
    out = torch.stack(outs)
    exact = (xs_stored.double() @ stored.double()).float()
    equal = (out == exact).float().mean().item()
    rel = ((out.double() - exact.double()).pow(2).sum().sqrt() / exact.double().pow(2).sum().sqrt()).item()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(name=name, fidelity=fidelity, fmt=fmt, x_fmt=x_fmt, w=w_all, stored=stored, x=xs, out=out),
        OUT_DIR / f"{name}__{fidelity}.pt",
    )
    logger.info(
        f"HY4_MK9 probe {name} {fidelity}: {equal:.4f} of the sums equal exact arithmetic on the stored operands, "
        f"relative rms difference {rel:.3e} ({out.numel()} sums)"
    )
