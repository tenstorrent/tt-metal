# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU attribution of an exp grid exactly representable by LoFi's left operand.

Models the native INT16 ties-away grid, BF16 SFPU-store truncation, and LoFi
P truncation. FP64 QK, subtraction, correction, PV and recurrence deliberately
omit hardware arithmetic. This is not a device oracle or performance model.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

import direct_exp_grid_models as D
import bfp4_residual_preprocess as B

HERE = Path(__file__).resolve().parent


def grid7(delta):
    # Exactly preserve the native FP32 coefficient rounding before scaling.
    a = torch.tensor(256.0 * float(torch.tensor(1.4426950408889634)), dtype=torch.float32)
    a = (a * float(D.EXP.SCALE)) * 0.25
    b = torch.tensor(32500.818359375, dtype=torch.float32) * 0.25
    transformed = (delta.float().double() * a.double() + b.double()).float()
    magnitude = torch.floor(transformed.double().abs() + 0.5).clamp_max(32767).int()
    values = (magnitude << 17).contiguous().view(torch.float32)
    values = torch.where(transformed >= 0, values, 0)
    assert torch.isfinite(values).all() and (values >= 0).all()
    assert torch.equal(values, D.MODEL.round_significand(values, 7, "trunc"))
    return values.double()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 32768])
    args = parser.parse_args()
    assert Path(args.label).name == args.label
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    files = [Path(__file__).resolve(), HERE / "exp_grid7.hpp", Path(D.__file__),
             Path(D.EXP.__file__), Path(B.__file__), Path(D.REPRO.__file__),
             Path(D.MODEL.__file__)]
    pins = {str(p.relative_to(HERE.parents[2])): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with (HERE / (args.label + ".jsonl")).open("x") as output:
        def emit(value):
            line = json.dumps(value, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)
        emit(dict(kind="provenance", args=vars(args), source_sha256=pins,
                  contract=__doc__, sampled_q_rows=128, useful_timing=False))
        delta = torch.linspace(-80 * math.sqrt(128), 0, 262144, dtype=torch.float64)
        truth = (delta / math.sqrt(128)).exp()
        for name, p in (("native", D.native_grid(delta)), ("grid7", grid7(delta))):
            p = D.MODEL.round_significand(p, 8, "trunc").double()
            consumed = D.MODEL.round_significand(p, 7, "trunc").double()
            ratio = p / truth
            gain = ratio.mean()
            emit(dict(kind="scalar", exp=name, exp_common_gain=float(gain),
                      gain_normalized_rms_pct=float(100 * (ratio / gain - 1).square().mean().sqrt()),
                      pv_vs_denominator_mismatches=int((p != consumed).sum())))
        for length in args.lengths:
            assert length % 512 == 0
            for seed in (1240, 1241):
                for distribution in ("normal", "outliers", "scaled_qk", "constant_v"):
                    q, k, v = [x.squeeze().float() for x in D.REPRO.make_inputs(1, 128, length, 128, seed, distribution)]
                    ref = D.REPRO.reference(q, k, v)
                    qe = D.MODEL.round_significand(q, 7)
                    for kv in ("b8_b8", "b4_b4"):
                        if kv == "b8_b8":
                            ke, ve = [B.native_bfp8_rne5(x.bfloat16()) for x in (k, v)]
                            ke, ve = [D.MODEL.round_significand(x, 5, "trunc") for x in (ke, ve)]
                        else:
                            ke, ve = [B.host_rne_bfp4(x.bfloat16()) for x in (k, v)]
                        scores = (qe.double() @ ke.double().T).reshape(128, -1, 512)
                        maximum = scores.amax(-1, keepdim=True).cummax(1).values
                        correction = ((maximum - maximum[:, -1:]) / math.sqrt(128)).exp()
                        for name, p in (("native", D.native_grid(scores - maximum)),
                                        ("grid7", grid7(scores - maximum))):
                            packed = D.MODEL.round_significand(p, 8, "trunc").double()
                            consumed = D.MODEL.round_significand(packed, 7, "trunc").double()
                            numerator_weights = (consumed * correction).reshape(128, -1)
                            denominator = (packed * correction).sum((1, 2)).unsqueeze(-1)
                            actual = (numerator_weights @ ve.double() / denominator).bfloat16()
                            emit(dict(kind="attention_model", length=length, seed=seed,
                                      distribution=distribution, kv=kv, exp=name,
                                      p_mismatches=int((packed != consumed).sum()),
                                      metrics=D.REPRO.metrics(actual, ref)))
        assert pins == {str(p.relative_to(HERE.parents[2])): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        emit(dict(kind="complete", sources_unchanged=True))


if __name__ == "__main__":
    main()
