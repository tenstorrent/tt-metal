# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only reversible power-of-two channel balancing for LoFi Q/K."""

import argparse
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("diagonal_hadamard", HERE / "qk_hadamard_native_models.py")
HAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HAD)
RISK, MODEL, REPRO = HAD.RISK, HAD.MODEL, HAD.REPRO


def inputs(seed, distribution):
    q, k, v = [x.squeeze().float() for x in REPRO.make_inputs(1, 128, 32768, 128, seed, "normal")]
    if distribution in ("channel_outlier_k", "balanced_channel_qk"):
        k[:, ::16] *= 32
        if distribution == "balanced_channel_qk":
            q[:, ::16] /= 32
    elif distribution == "joint_channel_qk":
        q[:, ::16] *= 4
        k[:, ::16] *= 4
    elif distribution == "structured_balanced_qk":
        # Four shared smooth channel modes, independent token/query amplitudes.
        channels = torch.arange(128, dtype=torch.float32) + 0.5
        basis = torch.stack([torch.cos(channels * (j + 1) * torch.pi / 128) for j in range(4)])
        generator = torch.Generator().manual_seed(seed + 3000)
        q = (q + 0.5 * torch.randn((128, 4), generator=generator) @ basis).bfloat16().float()
        k = (k + 0.5 * torch.randn((32768, 4), generator=generator) @ basis).bfloat16().float()
        scale = torch.exp2((torch.arange(128) % 7 - 3).float())
        q /= scale
        k *= scale
    assert torch.equal(q, q.bfloat16().float()) and torch.equal(k, k.bfloat16().float())
    return q, k, v


def diagonal(q, k, limit):
    rms = k.double().square().mean(0).sqrt()
    # A common scale cancels analytically; median anchors clipping to typical channels.
    raw = torch.log2(rms / rms.median()).round()
    exponent = raw.clamp(-limit, limit)
    scale = torch.exp2(exponent).float()
    qr, kr = (q * scale).bfloat16().float(), (k / scale).bfloat16().float()
    assert torch.equal(qr / scale, q) and torch.equal(kr * scale, k)
    assert torch.equal(MODEL.round_significand(qr, 7) / scale, MODEL.round_significand(q, 7))
    return qr, kr, scale, dict(
        scale_exponents=exponent.int().tolist(),
        clipped_channels=int((raw != exponent).sum()),
        bf16_roundtrip_exact=True,
        q7_unscaled_exact=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    signs = torch.randint(0, 2, (128,), generator=torch.Generator().manual_seed(HAD.SIGN_SEED)).float() * 2 - 1
    started, count = time.monotonic(), 0
    sources = [Path(__file__), Path(HAD.__file__), Path(RISK.__file__), HERE / "numerics.py",
               MODEL.V1 / "probe.py", Path(REPRO.__file__)]
    with (HERE / (args.label + ".jsonl")).open("x") as output:
        def emit(record):
            line = json.dumps(record, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        emit(dict(kind="provenance", hostname=platform.node(), threads=4, length=32768,
                  heads=1, query_length=128, dim=128, seeds=[1240, 1241],
                  contract="Original BF16 reference; Q RNE7; K/V host RNE BFP4 or RNE5/native RNA BFP8/LoFi trunc5; native FP32 exp grid and trunc7 P with matched denominator; FP64 QK, subtraction, online corrections and PV state; BF16 output",
                  smoothing="K'=K/s, Q'=Q*s, s=2^clip(round(log2(RMS(K)/median_channel_RMS)),-limit,limit); real BF16 transformed spills; no mean subtraction or logit correction; no runtime statistics/transform cost measurement",
                  hadamard="Shared fixed signs, unnormalized H16, BF16 transformed spills, score scale divided by16; no centering",
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
        for seed in (1240, 1241):
            for distribution in ("normal", "channel_outlier_k", "balanced_channel_qk", "joint_channel_qk", "structured_balanced_qk"):
                q, k, v = inputs(seed, distribution)
                reference = REPRO.reference(q, k, v)
                exact_score = q.double() @ k.double().T
                exact_score -= exact_score.mean(-1, keepdim=True)
                values = {fmt: RISK.encode(v, fmt) for fmt in ("b8", "b4")}
                for method in ("none", "rms_pow2_clip3", "rms_pow2_clip6", "h16_bf16"):
                    width, scale, details = 1, torch.ones(128), {}
                    if method.startswith("rms"):
                        qr, kr, scale, details = diagonal(q, k, int(method[-1]))
                    elif method == "h16_bf16":
                        width = 16
                        qr, kr, details = HAD.prepare(q, k, width, "bf16", False, signs)
                    else:
                        qr, kr = q, k
                    qe = MODEL.round_significand(qr, 7)
                    for kfmt in ("b8", "b4"):
                        ke = RISK.encode(kr, kfmt)
                        score = qe.double() @ ke.T / width
                        score -= score.mean(-1, keepdim=True)
                        score_l2 = float(100 * (score - exact_score).norm() / exact_score.norm())
                        reconstructed = HAD.hadamard(ke, width) * signs.double() / width if width > 1 else ke * scale
                        k_l2 = float(100 * (reconstructed - k).norm() / k.norm())
                        normalized = HAD.weights(qe, ke, width)
                        for vfmt, ve in values.items():
                            emit(dict(kind="attention", seed=seed, distribution=distribution, method=method,
                                      k_format=kfmt, v_format=vfmt,
                                      **REPRO.metrics((normalized @ ve).bfloat16(), reference),
                                      centered_score_l2_pct=score_l2, original_coordinate_k_l2_pct=k_l2,
                                      p_entropy_mean=float(-(normalized * normalized.clamp_min(1e-300).log()).sum(-1).mean()),
                                      p_max_mean=float(normalized.max(-1).values.mean()), **details))
                            count += 1
                emit(dict(kind="input_completed", seed=seed, distribution=distribution, cases=count,
                          seconds=time.monotonic() - started))
        emit(dict(kind="completed", cases=count, seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
