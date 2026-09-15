# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only exponent search on TT's actual signed-integer BFP4 grid.

No device imports. Attention uses FP64 dot products/state, native exp's FP32
grid, phase-0 P truncation, matched denominator, and final BF16 rounding.
This is a representation model, NOT a bit-exact FPU or performance model.
"""

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch

POLICIES = ("baseline", "mse_minus", "mse_pm", "uniform_proxy")


def significant(x, bits, mode="rne"):
    x = x.float()
    exponent = torch.frexp(x.abs())[1].float() - 1
    step = torch.exp2(exponent - (bits - 1))
    scaled = x.abs() / step
    integer = scaled.round() if mode == "rne" else (scaled + 0.5).floor() if mode == "rna" else scaled.floor()
    return integer * step * x.sign()


def shared_rna(x, bits):
    groups = x.reshape(-1, 16)
    exponent = torch.frexp(groups.abs().amax(-1, keepdim=True))[1].float() - 1
    step = torch.exp2(exponent - (bits - 1))
    return (((groups.abs() / step + 0.5).floor().clamp_max(2**bits - 1)) * step * groups.sign()).reshape_as(x)


def metric(actual, reference):
    a, r = actual.double().flatten(), reference.double().flatten()
    error = a - r
    ac, rc = a - a.mean(), r - r.mean()
    return dict(
        l2_pct=float(100 * error.norm() / r.norm()),
        mse=float(error.square().mean()),
        gain=float((a * r).sum() / r.square().sum()),
        pcc=float((ac * rc).sum() / (ac.norm() * rc.norm())),
    )


def quantizers(x):
    groups = x.reshape(-1, 16)
    absolute = groups.abs()
    exponent = torch.frexp(absolute.amax(-1, keepdim=True))[1].float() - 1
    candidates, errors = {}, {}
    for offset in (0, -1, 1):
        step = torch.exp2(exponent + offset - 2)
        candidates[offset] = (absolute / step).round().clamp_max(7) * step * groups.sign()
        errors[offset] = (candidates[offset] - groups).square().sum(-1, keepdim=True)
    result = {}
    for policy in POLICIES:
        chosen = torch.zeros_like(exponent)
        if policy in ("mse_minus", "mse_pm"):
            best = errors[0]
            for offset in ((-1,) if policy == "mse_minus" else (-1, 1)):
                improve = errors[offset] < best
                chosen = torch.where(improve, offset, chosen)
                best = torch.minimum(best, errors[offset])
        elif policy == "uniform_proxy":
            # UOS-style integrated uniform-projection proxy for codes 0..7.
            # Its stationary boundary is 7.5 + sqrt(2), NOT MXFP4's 7.25.
            step_exp = torch.ceil(torch.log2(absolute.amax(-1, keepdim=True) / (7.5 + math.sqrt(2))))
            chosen = step_exp + 2 - exponent
            assert bool(((chosen == 0) | (chosen == -1)).all())
        value = candidates[0].clone()
        for offset in (-1, 1):
            value = torch.where(chosen == offset, candidates[offset], value)
        value = value.reshape_as(x)
        # Every final code has <=3 significant bits. Verify two native pack
        # pipeline possibilities independently rather than assume a raw route.
        roundtrips = [
            shared_rna(value, 3),
            shared_rna(significant(value, 7, "rna"), 3),
            shared_rna(significant(value, 3, "rna"), 3),
        ]
        assert all(torch.equal(value, y) for y in roundtrips)
        actual_exp = torch.frexp(value.reshape(-1, 16).abs().amax(-1, keepdim=True))[1].float() - 1
        mismatch = int((actual_exp != exponent + chosen).sum())
        assert mismatch == 0, "Selected exponent must be induced by actual output, not supplied as metadata"
        error = (value.reshape(-1, 16) - groups).square().sum(-1, keepdim=True)
        if policy.startswith("mse"):
            assert bool((error <= errors[0]).all())
        step = torch.exp2(exponent + chosen - 2)
        result[policy] = (
            value,
            dict(
                **metric(value, x),
                mse_improvement_pct=float(100 * (1 - error.sum() / errors[0].sum())),
                offset_fraction={str(o): float((chosen == o).float().mean()) for o in (-1, 0, 1)},
                clipped_fraction=float((absolute > 7 * step).float().mean()),
                canonical_exponent_mismatches=mismatch,
                pack_roundtrip_mismatches=0,
            ),
        )
    return result


def inputs(seed, distribution, length):
    tensors = []
    for index, count in enumerate((128, length, length)):
        gen = torch.Generator().manual_seed(seed + index * 1000)
        x = torch.randn((count, 128), generator=gen)
        if distribution == "outliers":
            x += 10 * torch.randn(x.shape, generator=gen) * (torch.rand(x.shape, generator=gen) < 0.001)
        if distribution == "scaled_qk" and index < 2:
            x *= 2
        if distribution == ("channel_outlier_q", "channel_outlier_k", "channel_outlier_v")[index]:
            x[:, ::16] *= 32
        tensors.append(x.bfloat16().float())
    return tensors


def weights(q, k):
    scale = 1 / math.sqrt(128)
    score = (q.double() @ k.double().T).reshape(128, -1, 512)
    maximum = score.amax(-1, keepdim=True).cummax(1).values
    a = np.float32(np.float32(256) * np.float32(1.4426950408889634) * np.float32(scale))
    transformed = ((score - maximum).float().double() * float(a) + 32500.818359375).float()
    integer = (transformed.double().abs() + 0.5).floor().clamp_max(32767).int()
    p = (integer << 15).contiguous().view(torch.float32)
    p = torch.where(transformed >= 0, p, 0)
    pe = significant(p, 7, "trunc").double()
    weighted = (pe * ((maximum - maximum[:, -1:]) * scale).exp()).reshape(128, -1)
    return weighted / weighted.sum(-1, keepdim=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--budget-seconds", type=float, default=300)
    parser.add_argument("--length", type=int, default=32768)
    args = parser.parse_args()
    assert args.length % 512 == 0
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    path = Path(__file__).with_name(args.label + ".jsonl")
    completed = 0
    with path.open("x") as output:

        def emit(record):
            line = json.dumps(record, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        emit(
            dict(
                kind="provenance",
                args=vars(args),
                host=platform.node(),
                threads=4,
                torch_version=torch.__version__,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                contract="Original BF16 inputs/reference; Q RNE7; K/V BFP4 RNE with empirical exponent selection; K8 RNE5 then native shared RNA7 and LoFi trunc5; native FP32 exp grid/P trunc7/matched denominator; FP64 QK/subtraction/online alpha/PV/state; BF16 output; no device or throughput claims",
                caveats="128 sampled Q rows, H1, N32768, D128, K512; no FPU product alignment, BF16 running max, FP32 state roundoff, or reciprocal approximation; not exact kernel simulation",
            )
        )
        for seed in (1240, 1241):
            for distribution in ("normal", "outliers", "scaled_qk", "channel_outlier_k", "channel_outlier_v"):
                if time.monotonic() - started > args.budget_seconds:
                    emit(dict(kind="budget_stop", inputs_completed=completed, seconds=time.monotonic() - started))
                    return
                q, k, v = inputs(seed, distribution, args.length)
                ref = torch.softmax(q.double() @ k.double().T / math.sqrt(128), -1) @ v.double()
                qe = significant(q, 7)
                kchoices, vchoices = quantizers(k), quantizers(v)
                for name, choices in (("k", kchoices), ("v", vchoices)):
                    for policy, (_, stats) in choices.items():
                        emit(
                            dict(
                                kind="representation",
                                seed=seed,
                                distribution=distribution,
                                tensor=name,
                                policy=policy,
                                **stats,
                            )
                        )
                k8 = significant(shared_rna(significant(k, 5), 7), 5, "trunc")
                emit(
                    dict(
                        kind="representation",
                        seed=seed,
                        distribution=distribution,
                        tensor="k",
                        policy="b8_rne5",
                        **metric(k8, k),
                    )
                )
                for kpolicy in (*POLICIES, "b8_rne5"):
                    ke = k8 if kpolicy == "b8_rne5" else kchoices[kpolicy][0]
                    normalized = weights(qe, ke)
                    vpolicies = POLICIES if kpolicy == "b8_rne5" else (kpolicy,)
                    if kpolicy == "mse_pm":
                        vpolicies = ("mse_pm", "baseline")
                    if kpolicy == "baseline":
                        vpolicies = ("baseline", "mse_pm")
                    for vpolicy in vpolicies:
                        actual = (normalized @ vchoices[vpolicy][0].double()).bfloat16()
                        emit(
                            dict(
                                kind="attention",
                                seed=seed,
                                distribution=distribution,
                                k_policy=kpolicy,
                                v_policy=vpolicy,
                                **metric(actual, ref),
                            )
                        )
                completed += 1
                emit(
                    dict(
                        kind="input_completed",
                        seed=seed,
                        distribution=distribution,
                        inputs_completed=completed,
                        seconds=time.monotonic() - started,
                    )
                )
        emit(dict(kind="completed", inputs_completed=completed, seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
