# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unfused device-QK/PV diagnostic; CPU softmax and FP32 recurrence.

This is an accuracy experiment, not an SDPA kernel or performance benchmark.
No production dispatch is used. Q128/K512/D128, explicit BF16 final output.
"""

import argparse
import hashlib
import json
import math
import platform
import time

import torch
import ttnn

from numerics import FRONTIER, VARIANTS, evaluate, preprocess
from probe import HERE, metrics, quantize, unpack, upload


def run(device, inputs, variant, preprocessing, qkv_route, seed):
    prepared = preprocess(inputs, preprocessing, seed)
    q, k, v, correction, v0 = prepared
    qb, kb, pb, vb = VARIANTS[variant]
    formats = {3: ttnn.bfloat4_b, 7: ttnn.bfloat8_b}
    config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    qkv_model = "host" if qkv_route == "host" else "device"
    quantizer_checks = []

    def packed_input(x, bits):
        if qkv_route == "host":
            tensor = upload(x.float(), formats[bits], device)
        else:
            original = upload(x.float(), ttnn.float32, device)
            tensor = ttnn.typecast(original, formats[bits])
            ttnn.deallocate(original)
        decoded = unpack(tensor)
        check = metrics(decoded, quantize(x, bits, qkv_model))
        assert check["unequal"] == 0, check
        quantizer_checks.append(check["unequal"])
        return tensor, decoded

    tq, dq = packed_input(q, qb)
    numerator = torch.zeros((128, 128), dtype=torch.float32)
    matched = torch.zeros((128, 1), dtype=torch.float32)
    original = torch.zeros_like(matched)
    maximum = torch.full_like(matched, -torch.inf)
    qk_delta_sq = qk_ref_sq = pv_delta_sq = pv_ref_sq = 0.0
    for offset in range(0, k.shape[0], 512):
        tk, dk = packed_input(k[offset : offset + 512], kb)
        tv, dv = packed_input(v[offset : offset + 512], vb)
        ts = ttnn.matmul(
            tq,
            tk,
            transpose_b=True,
            dtype=ttnn.float32,
            compute_kernel_config=config,
            core_grid=ttnn.CoreGrid(y=1, x=1),
        )
        raw_scores = unpack(ts)
        qk_ref = dq.double() @ dk.double().T
        qk_delta_sq += float((raw_scores.double() - qk_ref).square().sum())
        qk_ref_sq += float(qk_ref.square().sum())
        scores = raw_scores / math.sqrt(128) + correction[:, offset : offset + 512].float()
        new_max = torch.maximum(maximum, scores.amax(-1, keepdim=True))
        alpha = (maximum - new_max).exp()
        p = (scores - new_max).exp()
        tp32 = upload(p, ttnn.float32, device)
        tp = ttnn.typecast(tp32, formats[pb])
        dp = unpack(tp)
        check = metrics(dp, quantize(p, pb, "device"))
        assert check["unequal"] == 0, check
        quantizer_checks.append(check["unequal"])
        to = ttnn.matmul(tp, tv, dtype=ttnn.float32, compute_kernel_config=config, core_grid=ttnn.CoreGrid(y=1, x=1))
        partial = unpack(to)
        pv_ref = dp.double() @ dv.double()
        pv_delta_sq += float((partial.double() - pv_ref).square().sum())
        pv_ref_sq += float(pv_ref.square().sum())
        numerator = numerator * alpha + partial
        matched = matched * alpha + dp.sum(-1, keepdim=True)
        original = original * alpha + p.sum(-1, keepdim=True)
        maximum = new_max
        for tensor in (tk, tv, ts, tp32, tp, to):
            ttnn.deallocate(tensor)
    ttnn.deallocate(tq)
    oq, ok, ov = [x.squeeze().double() for x in inputs]
    reference = torch.softmax(oq @ ok.T / math.sqrt(128), dim=-1) @ ov
    simulation = evaluate(prepared, VARIANTS[variant], qkv_model, "device", reference)
    result = []
    for denom_name, denominator, model in zip(("original", "matched"), (original, matched), simulation):
        actual = (numerator / denominator + v0.float()).bfloat16()
        result.append(
            dict(
                denominator=denom_name,
                **metrics(actual, reference),
                model_l2_pct=model["l2_pct"],
                model_pcc=model["pcc"],
                qk_arithmetic_l2_pct=100 * math.sqrt(qk_delta_sq / qk_ref_sq),
                pv_arithmetic_l2_pct=100 * math.sqrt(pv_delta_sq / pv_ref_sq),
                quantizer_exact_checks=len(quantizer_checks),
                output_sha256=hashlib.sha256(actual.view(torch.uint16).numpy().tobytes()).hexdigest(),
            )
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--qkv-route", choices=("host", "device"), default="host")
    parser.add_argument("--distributions", nargs="+", default=["normal", "outliers", "common_k"])
    parser.add_argument("--preprocessing", nargs="+", default=["none", "center_qk", "hadamard"])
    args = parser.parse_args()
    torch.set_num_threads(8)
    assert args.length % 512 == 0
    path = HERE / (args.label + ".jsonl")
    assert not path.exists(), "Use a fresh label"
    with path.open("x") as output:

        def emit(record):
            line = json.dumps(record, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        emit(
            dict(
                kind="provenance",
                hostname=platform.node(),
                args=vars(args),
                contract="Device LoFi/FP32DST QK and PV; CPU FP32 softmax/recurrence; NOT fused-kernel performance",
                source_sha256={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (HERE / "device_attention.py", HERE / "numerics.py", HERE / "probe.py")
                },
            )
        )
        device = ttnn.open_device(device_id=0)
        started = time.monotonic()
        try:
            for distribution in args.distributions:
                inputs = FRONTIER.inputs_for(args.length, args.seed, distribution)
                for preprocessing in args.preprocessing:
                    for variant in ("all4", "q8k4p8v4"):
                        results = run(device, inputs, variant, preprocessing, args.qkv_route, args.seed)
                        for result in results:
                            emit(
                                dict(
                                    kind="device_attention",
                                    length=args.length,
                                    distribution=distribution,
                                    preprocessing=preprocessing,
                                    variant=variant,
                                    qkv_route=args.qkv_route,
                                    **result,
                                )
                            )
        finally:
            ttnn.close_device(device)
        emit(dict(kind="completed", seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
