"""Quantization-only model; not an exact Tensix arithmetic simulator."""
import json
import math
import runpy

import torch


def quantize(x, fraction_bits, rounding=True):
    shift = 23 - fraction_bits
    bits = x.float().view(torch.int32)
    if rounding:
        bits = bits + (1 << (shift - 1)) - 1 + ((bits >> shift) & 1)
    return (bits & ~((1 << shift) - 1)).view(torch.float32).double()


torch.set_num_threads(8)
repro = runpy.run_path("tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py")
q, k, v = repro["make_inputs"](1, 128, 4096, 128, 1234, "normal")
q, k, v = q.double(), k.double(), v.double()
gold = (q @ k.transpose(-1, -2) / math.sqrt(128)).softmax(-1) @ v
for q_scale in [1.0, 1.001, 1.002, 1.0025, 1.0027, 1.003, 1.004, 1.005, 1.125, math.sqrt(2)]:
    rq = quantize(q * q_scale, 6) / q_scale
    logits = rq @ k.transpose(-1, -2) / math.sqrt(128)
    weights = (logits - logits.amax(-1, keepdim=True)).exp()
    for p_round in [False, True]:
        p = quantize(weights, 6) if p_round else weights
        out = ((p @ v) / p.sum(-1, keepdim=True)).bfloat16()
        print(
            json.dumps(
                dict(
                    q_scale=q_scale,
                    p_round=p_round,
                    q_error_pct=100 * ((rq - q).norm() / q.norm()).item(),
                    **repro["metrics"](out, gold),
                )
            )
        )
