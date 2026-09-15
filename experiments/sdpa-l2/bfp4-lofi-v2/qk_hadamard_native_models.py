# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU signed, unnormalized Hadamard Q/K smoothing with real BF16 spills."""

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("hadamard_asymmetric", HERE / "asymmetric_risk_models.py")
RISK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RISK)
MODEL, REPRO = RISK.MODEL, RISK.REPRO
SIGN_SEED = 20260915


def hadamard(x, width):
    shape = x.shape
    y = x.reshape(-1, width)
    stride = 1
    while stride < width:
        z = y.reshape(y.shape[0], -1, 2, stride)
        a, b = z[..., 0, :], z[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(-1, width)
        stride *= 2
    return y.reshape(shape)


def prepare(q, k, width, spill, center_k, signs):
    if width == 1:
        return q, k, dict(q_spill_roundtrip_l2_pct=0.0, k_spill_roundtrip_l2_pct=0.0)
    transformed = [hadamard(x * signs, width) for x in (q, k)]
    if spill == "bf16":
        transformed = [x.bfloat16().float() for x in transformed]
    details = {}
    for name, original, rotated in zip(("q", "k"), (q, k), transformed):
        reconstructed = hadamard(rotated.double(), width) * signs.double() / width
        details[name + "_spill_roundtrip_l2_pct"] = float(100 * (reconstructed - original).norm() / original.norm())
    if center_k:
        # Model existing fused FP32 center/quantize: no second BF16 spill.
        transformed[1] = (transformed[1].double() - transformed[1].double().mean(0, keepdim=True)).float()
    return *transformed, details


def weights(q, k, width):
    scale = 1 / (math.sqrt(128) * width)
    score = (q.double() @ k.double().T).reshape(128, -1, 512)
    maximum = score.amax(-1, keepdim=True).cummax(1).values
    delta = score - maximum
    a = np.float32(np.float32(256) * np.float32(1.4426950408889634) * np.float32(scale))
    transformed = (delta.float().double() * float(a) + 32500.818359375).float()
    integer = torch.floor(transformed.double().abs() + 0.5).clamp_max(32767).int()
    p = (integer << 15).contiguous().view(torch.float32)
    p = torch.where(transformed >= 0, p, 0)
    pe = MODEL.round_significand(p, 7, "trunc").double()
    weighted = (pe * ((maximum - maximum[:, -1:]) * scale).exp()).reshape(128, -1)
    return weighted / weighted.sum(-1, keepdim=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    signs = torch.randint(0, 2, (128,), generator=torch.Generator().manual_seed(SIGN_SEED)).float() * 2 - 1
    probe = torch.randn((32, 128), dtype=torch.float64, generator=torch.Generator().manual_seed(93))
    for width in (16, 128):
        torch.testing.assert_close(
            hadamard(hadamard(probe * signs, width), width) * signs / width, probe, atol=2e-14, rtol=2e-14
        )
    previous = [json.loads(x) for x in (HERE / "asymmetric-risk-models-v1.jsonl").read_text().splitlines()]
    started, count = time.monotonic(), 0
    sources = [Path(__file__), Path(RISK.__file__), HERE / "numerics.py", MODEL.V1 / "probe.py", Path(REPRO.__file__)]
    with (HERE / (args.label + ".jsonl")).open("x") as output:

        def emit(record):
            line = json.dumps(record, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        emit(
            dict(
                kind="provenance",
                hostname=platform.node(),
                threads=4,
                length=32768,
                seeds=[1240, 1241],
                sign_seed=SIGN_SEED,
                contract="Original BF16 QKV reference; same signed unnormalized Hadamard on Q and K; FP32 butterflies then BF16 spill (explicit FP32 controls); score scale1/(sqrtD*H); Q RNE7 and K RNE BFP4; V RNE BFP4 or RNE5/nativeBFP8/LoFi5; native FP32 exp grid; P trunc7 matched; FP64 QK/subtraction/online corrections/PV/state; BF16 output",
                k_center="After transformed spill, subtract token-column FP64 mean into FP32 before quantization; no extra BF16 spill; mean generation and transform performance unmeasured",
                source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            )
        )
        for seed in (1240, 1241):
            for distribution in ("normal", "outliers", "scaled_qk", "common_k", "common_q"):
                q, k, v = [x.squeeze().float() for x in REPRO.make_inputs(1, 128, 32768, 128, seed, distribution)]
                reference = REPRO.reference(q, k, v)
                values = {fmt: RISK.encode(v, fmt) for fmt in ("b8", "b4")}
                configs = [(1, "none", False)] + [(w, "bf16", center) for w in (16, 128) for center in (False, True)]
                if distribution in ("normal", "outliers"):
                    configs += [(w, "fp32", False) for w in (16, 128)]
                for width, spill, center in configs:
                    qr, kr, details = prepare(q, k, width, spill, center, signs)
                    qe = MODEL.round_significand(qr, 7)
                    ke = RISK.encode(kr, "b4")
                    normalized = weights(qe, ke, width)
                    for vfmt, ve in values.items():
                        actual = (normalized @ ve).bfloat16()
                        metrics = REPRO.metrics(actual, reference)
                        if seed == 1240 and width == 1:
                            old = next(
                                r
                                for r in previous
                                if r.get("kind") == "attention"
                                and r["length"] == 32768
                                and r["distribution"] == distribution
                                and r["k_format"] == "b4"
                                and r["v_format"] == vfmt
                                and r["exp"] == "native"
                            )
                            assert old["l2_pct"] == metrics["l2_pct"], "Native unrotated control must reproduce exactly"
                        emit(
                            dict(
                                kind="attention",
                                length=32768,
                                seed=seed,
                                distribution=distribution,
                                hadamard_width=width,
                                spill=spill,
                                center_k=center,
                                v_format=vfmt,
                                **metrics,
                                **details,
                                p_entropy_mean=float(-(normalized * normalized.clamp_min(1e-300).log()).sum(-1).mean()),
                                p_max_mean=float(normalized.max(-1).values.mean()),
                            )
                        )
                        count += 1
                emit(
                    dict(
                        kind="input_completed",
                        seed=seed,
                        distribution=distribution,
                        cases=count,
                        seconds=time.monotonic() - started,
                    )
                )
        emit(dict(kind="completed", cases=count, seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
