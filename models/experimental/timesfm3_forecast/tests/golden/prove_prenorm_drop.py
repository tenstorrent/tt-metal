# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""R1-2 fp64 proof (D2 item 2): drop the gain-free pre-norms before W_ov (V = 1) and ff0.

Real eps, inputs with leading-pad and masked tokens. Reports the minimum per-token rms^2 of each removed norm's
input over valid tokens against eps, and the read-out difference. Rejects if any valid token is within 1e3x eps.
Runs in the reference venv (like ``check_dataflow.py``); writes ``prenorm_drop_report.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
sys.path.insert(0, str(HERE))

import dataflow as df  # noqa: E402
import host  # noqa: E402
import make_goldens as mg  # noqa: E402

EPS = torch.finfo(torch.float32).eps  # the device eps (nn.RMSNorm(eps=None) on fp32 weights)
stats: dict[str, float] = {}


def _ms(x, valid):
    return float((x * x).mean(-1)[valid].min())


def var_no_pre(h1, w, masks, token_valid):
    if masks.variates != 1:
        return orig_var(h1, w, masks, token_valid)
    stats["var"] = min(stats.get("var", float("inf")), _ms(h1, token_valid))
    o = (h1 @ w.w_ov.T) * token_valid[..., None].to(h1.dtype)
    return h1 + df.rms_norm(o, w.post_gain, EPS)


def ffn_no_pre(h2, lw):
    stats["ffn"] = min(stats.get("ffn", float("inf")), _ms(h2, cur_valid[0]))
    return h2 + df.rms_norm(torch.relu(h2 @ lw.ff0.T) @ lw.ff1.T, lw.post_ff_gain, EPS)


orig_var, orig_ffn, orig_rms = df.var_attention, df.ffn, df.rms_norm
cur_valid = [None]


def rms_real_eps(x, gain=None, eps=None):
    return orig_rms(x, gain, EPS if eps is None else eps)


def main():
    from safetensors.torch import load_file

    torch.set_num_threads(mg.THREADS)
    w64 = df.load_weights(load_file(str(mg.weights_file())), torch.float64)
    df.rms_norm = rms_real_eps
    rng = np.random.default_rng(5)
    cases = {
        "16 x 2048->64 (left-padded, NaN)": (mg.full16_contexts(), 64),
        "4 items x 8 variates": (mg.mv4_contexts(), 64),
        "realistic 16 x 2048->64": (list(mg.realistic(rng, 16, 2048)), 64),
    }
    report = []
    for name, (ctx, horizon) in cases.items():
        hi = host.prepare(ctx, horizon)
        ro = list(hi.readout_tokens)
        df.var_attention, df.ffn = orig_var, orig_ffn
        base = df.forward(hi.tokens, hi.num_front_masked, hi.variates, w64, ro)
        cur_valid[0] = hi.token_valid
        stats.clear()
        df.var_attention, df.ffn = var_no_pre, ffn_no_pre
        new = df.forward(hi.tokens, hi.num_front_masked, hi.variates, w64, ro)
        rel = float((new - base).abs().max() / base.abs().max())
        mins = {k: v / EPS for k, v in stats.items()}
        ok = rel < 1e-9 and all(v > 1e3 for v in mins.values())
        report.append({"case": name, "rel": rel, "min_rms2_over_eps": mins, "pass": ok})
        print(f"{name}: rel {rel:.2e}, min rms^2 / eps {mins}, {'PASS' if ok else 'REJECT'}")
    (HERE / "prenorm_drop_report.json").write_text(json.dumps(report, indent=1))
    sys.exit(0 if all(r["pass"] for r in report) else 1)


if __name__ == "__main__":
    main()
