# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU width/sign comparison; reuse completed cells and compute only gaps."""

import argparse
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("width_diagonal", HERE / "diagonal_k_smoothing_models.py")
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)
HAD, RISK, MODEL, REPRO = DIAG.HAD, DIAG.RISK, DIAG.MODEL, DIAG.REPRO
OLD_HAD = HERE / "qk-hadamard-native-models-v1.jsonl"
OLD_DIAG = HERE / "diagonal-k-smoothing-v1.jsonl"
METRICS = ("l2_pct", "pcc", "gain", "gain_corrected_l2_pct", "row_l2_pct_median",
           "row_l2_pct_p95", "p_entropy_mean", "p_max_mean")


def reuse(distribution, seed, method, vfmt, old_had, old_diag):
    if method == "h128_plain" or (distribution == "channel_k" and method == "h128_signed"):
        return None
    if distribution == "channel_k":
        row = next(r for r in old_diag if r.get("kind") == "attention" and r["distribution"] == "channel_outlier_k"
                   and r["seed"] == seed and r["k_format"] == "b4" and r["v_format"] == vfmt
                   and r["method"] == ("none" if method == "none" else "h16_bf16"))
        return row, OLD_DIAG.name
    width = {"none": 1, "h16_signed": 16, "h128_signed": 128}[method]
    row = next(r for r in old_had if r.get("kind") == "attention" and r["distribution"] == distribution
               and r["seed"] == seed and r["v_format"] == vfmt and r["hadamard_width"] == width
               and not r["center_k"] and r["spill"] == ("none" if width == 1 else "bf16"))
    return row, OLD_HAD.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    signs = torch.randint(0, 2, (128,), generator=torch.Generator().manual_seed(HAD.SIGN_SEED)).float() * 2 - 1
    old_had = [json.loads(l) for l in OLD_HAD.read_text().splitlines()]
    old_diag = [json.loads(l) for l in OLD_DIAG.read_text().splitlines()]
    paths = [Path(__file__), Path(DIAG.__file__), Path(HAD.__file__), Path(RISK.__file__),
             HERE / "numerics.py", MODEL.V1 / "probe.py", Path(REPRO.__file__), OLD_HAD, OLD_DIAG]
    pinned = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    computed, reused, started = 0, 0, time.monotonic()
    with (HERE / (args.label + ".jsonl")).open("x") as output:
        def emit(record):
            line = json.dumps(record, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        emit(dict(kind="provenance", hostname=platform.node(), threads=4, length=32768, query_length=128,
                  dim=128, heads=1, seeds=[1240, 1241], sign_seed=HAD.SIGN_SEED, source_sha256=pinned,
                  contract="Original BF16 QKV reference; same signed or plain unnormalized Hadamard on Q/K, BF16 transform spills; scale divided by width; Q RNE7/K RNE BFP4; V RNE BFP4 or RNE5/nativeRNA BFP8/LoFi5; native exp, P trunc7 matched; FP64 QK/subtraction/online correction/recurrence; BF16 output",
                  scope="CPU only; no centering, no device FPU alignment, no performance/model-quality claims; H16 means previously tested SIGNED H16; only missing plainH128 and channelK signedH128 cells computed"))
        for seed in (1240, 1241):
            for distribution in ("normal", "outliers", "channel_k", "common_q"):
                if distribution == "channel_k":
                    q, k, v = DIAG.inputs(seed, "channel_outlier_k")
                else:
                    q, k, v = [x.squeeze().float() for x in REPRO.make_inputs(1, 128, 32768, 128, seed, distribution)]
                reference = REPRO.reference(q, k, v)
                values = {fmt: RISK.encode(v, fmt) for fmt in ("b8", "b4")}
                for method in ("none", "h16_signed", "h128_plain", "h128_signed"):
                    needed = reuse(distribution, seed, method, "b8", old_had, old_diag) is None
                    if needed:
                        diagonal = signs if method == "h128_signed" else torch.ones_like(signs)
                        qr, kr, details = HAD.prepare(q, k, 128, "bf16", False, diagonal)
                        qe, ke = MODEL.round_significand(qr, 7), RISK.encode(kr, "b4")
                        weights = HAD.weights(qe, ke, 128)
                    for vfmt, ve in values.items():
                        previous = reuse(distribution, seed, method, vfmt, old_had, old_diag)
                        if previous is not None:
                            row, source = previous
                            metrics = {key: row[key] for key in METRICS}
                            provenance = dict(computed_this_run=False, reused_record=source)
                            reused += 1
                        else:
                            metrics = {key: value for key, value in REPRO.metrics((weights @ ve).bfloat16(), reference).items()
                                       if key in METRICS}
                            metrics.update(p_entropy_mean=float(-(weights * weights.clamp_min(1e-300).log()).sum(-1).mean()),
                                           p_max_mean=float(weights.max(-1).values.mean()), **details)
                            provenance = dict(computed_this_run=True, reused_record=None)
                            computed += 1
                        emit(dict(kind="attention", seed=seed, distribution=distribution, method=method,
                                  k_format="b4", v_format=vfmt, **metrics, **provenance))
                emit(dict(kind="input_completed", seed=seed, distribution=distribution,
                          computed=computed, reused=reused, seconds=time.monotonic() - started))
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == pinned[str(p)] for p in paths)
        assert computed == 20 and reused == 44
        emit(dict(kind="completed", computed=computed, reused=reused, seconds=time.monotonic() - started,
                  sources_unchanged=True))


if __name__ == "__main__":
    main()
