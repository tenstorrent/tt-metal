# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Simplified QK-only INT4/BFP4 quantization comparison; CPU, not Sage execution.

FP64 softmax and PV isolate QK representation error. The explicit Q32/K64
strided-group variant follows published/default per-thread grouping but omits
GPU accumulation, PV quantization, and kernel scheduling. Other INT4 variants
are idealized scale-granularity ablations, not actual SageAttention kernels.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("sage_qk_v2_numerics", HERE / "numerics.py")
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)
VARIANTS = [
    "exact",
    "int4_perrow_ideal",
    "int4_block128_64_ideal",
    "int4_block128_512_ideal",
    "int4_thread32_64",
    "int8_thread32_64",
    "int4_shared16_ideal",
    "bfp4_host_rne",
    "bfp4_device_biased",
]


def scaled_integer(groups, bits=4, code_round=False):
    bound = 2 ** (bits - 1) - 1
    if code_round:
        groups = groups.float()
        scale = groups.abs().amax(-1, keepdim=True) / bound + 1e-7
        normalized = groups / scale
        integer = (normalized.abs() + 0.5).floor() * normalized.sign()
    else:
        groups = groups.double()
        scale = groups.abs().amax(-1, keepdim=True) / bound
        scale = torch.where(scale == 0, 1.0, scale)
        integer = (groups / scale).round()
    return integer.clamp(-bound, bound).double() * scale.double()


def encode_pair(q, k, variant):
    if variant == "exact":
        return q, k
    if variant.startswith("bfp4_"):
        route = "host" if variant == "bfp4_host_rne" else "device"
        return tuple(MODEL.quantize(x, 3, route).double() for x in (q, k))
    if variant == "int4_perrow_ideal":
        return scaled_integer(q), scaled_integer(k)
    if variant == "int4_shared16_ideal":
        return tuple(scaled_integer(x.reshape(-1, 16)).reshape(x.shape) for x in (q, k))
    if variant.startswith("int4_block"):
        kb = 64 if variant == "int4_block128_64_ideal" else 512
        return tuple(scaled_integer(x.reshape(-1, n * x.shape[-1])).reshape(x.shape) for x, n in ((q, 128), (k, kb)))
    if variant in ("int4_thread32_64", "int8_thread32_64"):
        bits = 4 if variant.startswith("int4") else 8
        # Q group: rows i, i+8, i+16, i+24, across all head channels.
        qg = q.reshape(-1, 4, 8, 128).permute(0, 2, 1, 3)
        qe = scaled_integer(qg.reshape(-1, 4 * 128), bits, True)
        qe = qe.reshape(qg.shape).permute(0, 2, 1, 3).reshape(q.shape)
        # K group: rows 2i,2i+1,2i+8,2i+9,...,2i+56,2i+57.
        kg = k.reshape(-1, 8, 4, 2, 128).permute(0, 2, 1, 3, 4)
        ke = scaled_integer(kg.reshape(-1, 16 * 128), bits, True)
        ke = ke.reshape(kg.shape).permute(0, 2, 1, 3, 4).reshape(k.shape)
        return qe, ke
    raise ValueError(variant)


def self_test():
    x = torch.arange(512 * 128).reshape(512, 128).double() % 17 - 8
    for variant in VARIANTS:
        q, k = encode_pair(x, x, variant)
        assert q.shape == x.shape and k.shape == x.shape
        assert torch.isfinite(q).all() and torch.isfinite(k).all()
    # Every member of the explicit thread group uses the same max/scale.
    q, k = encode_pair(x, x, "int4_thread32_64")
    qi = torch.arange(0, 32, 8)
    ki = torch.stack((torch.arange(0, 64, 8), torch.arange(1, 64, 8)), -1).flatten()
    assert torch.equal(q[qi], scaled_integer(x[qi].reshape(1, -1), code_round=True).reshape(4, 128))
    assert torch.equal(k[ki], scaled_integer(x[ki].reshape(1, -1), code_round=True).reshape(16, 128))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 32768])
    parser.add_argument("--seed", type=int, default=1243)
    parser.add_argument("--distributions", nargs="+", default=["normal", "scaled_down"])
    parser.add_argument(
        "--preprocessing", nargs="+", choices=["center_k", "center_qk"], default=["center_k", "center_qk"]
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    self_test()
    started = time.monotonic()
    with (HERE / (args.label + ".jsonl")).open("x") as output:

        def emit(record):
            line = json.dumps(record, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        sources = [Path(__file__), HERE / "numerics.py", MODEL.V1 / "probe.py", MODEL.V1 / "numerics.py"]
        emit(
            dict(
                kind="provenance",
                args=vars(args),
                hostname=platform.node(),
                contract="Simplified CPU QK-quantization simulation; FP64 softmax/PV; original BF16 input/output; NOT measured SageAttention or GPU/device performance",
                source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                primary_sources=[
                    "https://arxiv.org/html/2411.10958v7#A1.SS6",
                    "https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention/triton/quant_per_thread.py",
                ],
            )
        )
        for length in args.lengths:
            for distribution in args.distributions:
                inputs = MODEL.V1_NUMERICS.FRONTIER.inputs_for(length, args.seed, distribution)
                oq, ok, ov = [x.squeeze().double() for x in inputs]
                reference = torch.softmax(oq @ ok.T / math.sqrt(128), -1) @ ov
                for preprocessing in args.preprocessing:
                    q, k, v, correction, _ = MODEL.V1_NUMERICS.preprocess(inputs, preprocessing, args.seed)
                    exact_scores = q @ k.T / math.sqrt(128) + correction
                    for variant in args.variants:
                        qe, ke = encode_pair(q, k, variant)
                        scores = qe @ ke.T / math.sqrt(128) + correction
                        raw = torch.softmax(scores, -1) @ v
                        result = raw.bfloat16().double()
                        row_error = 100 * (result - reference).norm(dim=-1) / reference.norm(dim=-1)
                        score_error = scores - exact_scores
                        score_error -= score_error.mean(-1, keepdim=True)
                        emit(
                            dict(
                                kind="attention_model",
                                length=length,
                                seed=args.seed,
                                distribution=distribution,
                                preprocessing=preprocessing,
                                variant=variant,
                                **MODEL.metrics(result, reference),
                                before_output_rounding_l2_pct=MODEL.metrics(raw, reference)["l2_pct"],
                                row_p95_l2_pct=float(torch.quantile(row_error, 0.95)),
                                q_representation_l2_pct=MODEL.metrics(qe, q)["l2_pct"],
                                k_representation_l2_pct=MODEL.metrics(ke, k)["l2_pct"],
                                row_centered_score_error_rms=float(score_error.square().mean().sqrt()),
                            )
                        )
        emit(dict(kind="completed", seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
