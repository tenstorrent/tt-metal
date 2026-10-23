# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU only: how far bf16 rounding alone moves the forecast median (scaled space), on the model16 golden.

    python models/experimental/tirex2_forecast/tests/golden/sim_precision.py

Runs the fp32 dataflow (checked against upstream to 1e-7) with (a) the matmul weights rounded to bf16, (b) also
the residual stream, (c) also every linear input, and prints the worst and mean per-row median error against
the upstream head output at the read-out tokens. Used as evidence for the median check in
``doc/optimize/iter_00/precision_median.md``.
"""

from __future__ import annotations

import copy
import dataclasses

import torch

from models.experimental.tirex2_forecast.reference import dataflow as df
from models.experimental.tirex2_forecast.tests.golden.goldens import golden


def bf(t):
    return t.bfloat16().float()


def round_weights(w):
    w = copy.deepcopy(w)

    def rec(o):
        if dataclasses.is_dataclass(o):
            for f in dataclasses.fields(o):
                v = getattr(o, f.name)
                if isinstance(v, torch.Tensor) and v.ndim == 2:
                    setattr(o, f.name, bf(v))
                else:
                    rec(v)
        elif isinstance(o, list):
            for v in o:
                rec(v)

    rec(w)
    return w


def main():
    torch.set_num_threads(16)
    g = golden("model16")
    W = df.build_weights(df.load_state_dict(df.default_checkpoint()))
    Wb = round_weights(W)
    prep = df.prepare(g["context"], g["horizon"])
    tok = df.readout_tokens(g["horizon"])
    rows = g["context"].shape[0]
    med = lambda h: h.reshape(rows, len(tok), df.QUANTILES, df.PATCH)[:, :, df.QUANTILES // 2]
    ref = med(g["head_out"][:, tok])

    def run(w, resid_bf16=False, lin_bf16=False):
        orig = df.linear
        if lin_bf16:
            df.linear = lambda x, ww, b=None: orig(bf(x), ww, b)
        try:
            x = df.residual_block(prep.tokens, w.embed)
            for blk in w.blocks:
                x = df.block(bf(x) if resid_bf16 else x, blk)
            x = df.t5_norm(bf(x) if resid_bf16 else x, w.w_out_norm)
            return df.residual_block(x, w.head)[:, tok]
        finally:
            df.linear = orig

    for name, w, kw in (
        ("fp32", W, {}),
        ("bf16 weights", Wb, {}),
        ("bf16 weights + residual", Wb, dict(resid_bf16=True)),
        ("bf16 weights + residual + linear inputs", Wb, dict(resid_bf16=True, lin_bf16=True)),
    ):
        e = (med(run(w, **kw)) - ref).abs().amax((1, 2))
        print(f"{name:42s} worst median |err| / scale {float(e.max()):.4f}  mean over rows {float(e.mean()):.4f}")


if __name__ == "__main__":
    main()
