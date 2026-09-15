# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device LoFi matmul verification of residual quantization; CPU softmax/state.

This deliberately is NOT a fused-attention performance measurement.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("residual_model", HERE / "numerics.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
from probe import unpack, upload


def run(device, inputs, variant, route, preprocessing, seed):
    q, k, v, correction, v0 = M.V1_NUMERICS.preprocess(inputs, preprocessing, seed)
    qfmt, kfmts, pfmt, vfmts = M.VARIANTS[variant]
    config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    quantizer_checks = 0

    def pack(x, fmt, selected_route):
        nonlocal quantizer_checks
        if fmt in ("e7", "e5", "e5_b8"):
            x = M.round_significand(x, 7 if fmt == "e7" else 5)
        dtype = {
            "b4": ttnn.bfloat4_b,
            "b8": ttnn.bfloat8_b,
            "e5_b8": ttnn.bfloat8_b,
            "e7": ttnn.bfloat16,
            "e5": ttnn.bfloat16,
            "fp32": ttnn.float32,
        }[fmt]
        if selected_route == "device" and fmt in ("b4", "b8", "e5_b8"):
            src = upload(x.float(), ttnn.float32, device)
            tensor = ttnn.typecast(src, dtype)
            ttnn.deallocate(src)
        else:
            tensor = upload(x.float(), dtype, device)
        decoded = unpack(tensor)
        expected = M.encode(x, fmt, selected_route).float()
        assert torch.equal(decoded, expected), M.metrics(decoded, expected)
        quantizer_checks += 1
        return tensor, decoded

    def residuals(x, formats):
        residual, tensors, decoded = x.double(), [], []
        for fmt in formats:
            tensor, value = pack(residual, fmt, route)
            effective = M.effective(value, fmt, "right")
            tensors.append(tensor)
            decoded.append(effective)
            residual = residual - effective
        return tensors, decoded

    tq, dq = pack(q, qfmt, route)
    dq = M.effective(dq, qfmt, "left")
    numerator = torch.zeros((128, 128), dtype=torch.float32)
    matched = torch.zeros((128, 1), dtype=torch.float32)
    original = torch.zeros_like(matched)
    maximum = torch.full_like(matched, -torch.inf)
    qk_error_sq = qk_ref_sq = pv_error_sq = pv_ref_sq = 0.0
    for offset in range(0, k.shape[0], 512):
        tks, dks = residuals(k[offset : offset + 512], kfmts)
        tvs, dvs = residuals(v[offset : offset + 512], vfmts)
        scores = torch.zeros((128, 512), dtype=torch.float32)
        for tk, dk in zip(tks, dks):
            ts = ttnn.matmul(
                tq,
                tk,
                transpose_b=True,
                dtype=ttnn.float32,
                compute_kernel_config=config,
                core_grid=ttnn.CoreGrid(y=1, x=1),
            )
            actual = unpack(ts)
            reference = dq @ dk.T
            qk_error_sq += float((actual.double() - reference).square().sum())
            qk_ref_sq += float(reference.square().sum())
            scores += actual
            ttnn.deallocate(ts)
        scores = scores / math.sqrt(128) + correction[:, offset : offset + 512].float()
        new_max = torch.maximum(maximum, scores.amax(-1, keepdim=True))
        alpha = (maximum - new_max).exp()
        p = (scores - new_max).exp()
        tp, dp = pack(p, pfmt, "device")
        dp = M.effective(dp, pfmt, "left")
        partial = torch.zeros_like(numerator)
        for tv, dv in zip(tvs, dvs):
            to = ttnn.matmul(
                tp, tv, dtype=ttnn.float32, compute_kernel_config=config, core_grid=ttnn.CoreGrid(y=1, x=1)
            )
            actual = unpack(to)
            reference = dp @ dv
            pv_error_sq += float((actual.double() - reference).square().sum())
            pv_ref_sq += float(reference.square().sum())
            partial += actual
            ttnn.deallocate(to)
        numerator = numerator * alpha + partial
        matched = matched * alpha + dp.float().sum(-1, keepdim=True)
        original = original * alpha + p.sum(-1, keepdim=True)
        maximum = new_max
        for tensor in [*tks, *tvs, tp]:
            ttnn.deallocate(tensor)
    ttnn.deallocate(tq)
    oq, ok, ov = [x.squeeze().double() for x in inputs]
    reference = torch.softmax(oq @ ok.T / math.sqrt(128), -1) @ ov
    simulated = M.evaluate(inputs, variant, preprocessing, route, seed, reference)
    records = []
    for name, denominator, model in zip(("original", "matched"), (original, matched), simulated):
        actual = (numerator / denominator + v0.float()).bfloat16()
        records.append(
            dict(
                denominator=name,
                **M.metrics(actual, reference),
                model_l2_pct=model["l2_pct"],
                qk_arithmetic_l2_pct=100 * math.sqrt(qk_error_sq / qk_ref_sq),
                pv_arithmetic_l2_pct=100 * math.sqrt(pv_error_sq / pv_ref_sq),
                quantizer_exact_checks=quantizer_checks,
                output_sha256=hashlib.sha256(actual.view(torch.uint16).numpy().tobytes()).hexdigest(),
            )
        )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1240, 1241])
    parser.add_argument("--variants", nargs="+", default=["pervalue_rne", "residual48_pervalue", "residual44_pervalue"])
    parser.add_argument("--distributions", nargs="+", default=["normal", "outliers", "scaled_qk"])
    parser.add_argument("--route", choices=("host", "device"), default="device")
    parser.add_argument("--preprocessing", default="none")
    args = parser.parse_args()
    path = HERE / (args.label + ".jsonl")
    assert not path.exists()
    torch.set_num_threads(8)
    device = ttnn.open_device(device_id=0)
    try:
        with path.open("x") as output:
            output.write(
                json.dumps(
                    dict(
                        kind="provenance",
                        args=vars(args),
                        contract="Device LoFi QK/PV, CPU FP32 softmax/state, CPU residual preparation; NOT kernel timing",
                        source_sha256={
                            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__).resolve(), HERE / "numerics.py")
                        },
                    )
                )
                + "\n"
            )
            for seed in args.seeds:
                for distribution in args.distributions:
                    inputs = M.V1_NUMERICS.FRONTIER.inputs_for(args.length, seed, distribution)
                    for variant in args.variants:
                        for record in run(device, inputs, variant, args.route, args.preprocessing, seed):
                            record.update(
                                kind="device_residual",
                                length=args.length,
                                seed=seed,
                                distribution=distribution,
                                variant=variant,
                                route=args.route,
                                preprocessing=args.preprocessing,
                            )
                            output.write(json.dumps(record, allow_nan=False) + "\n")
                            output.flush()
                            print(json.dumps(record, allow_nan=False), flush=True)
    finally:
        ttnn.close_device(device)
