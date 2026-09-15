# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU ablations of rounding sites, not a bit-exact TT hardware simulator."""

import json
import math

import torch

from tests.ttnn.unit_tests.operations.sdpa.repro_sdpa_l2 import make_inputs, metrics, reference

torch.set_num_threads(8)
q, k, v = make_inputs(1, 128, 4096, 128, 1234, "normal")
raw = q.double() @ k.double().transpose(-1, -2)
gold = reference(q, k, v)
for name in ("exact", "qk_bf16", "qk_and_sub_bf16", "qk_sub_and_exp_bf16"):
    scores = raw if name == "exact" else raw.bfloat16().double()
    shifted = scores - scores.amax(-1, keepdim=True)
    if name in ("qk_and_sub_bf16", "qk_sub_and_exp_bf16"):
        shifted = shifted.bfloat16().double()
    weights = (shifted / math.sqrt(128)).exp()
    if name == "qk_sub_and_exp_bf16":
        weights = weights.bfloat16().double()
    result = (weights @ v.double()) / weights.sum(-1, keepdim=True)
    print(json.dumps(dict(experiment=name, **metrics(result, gold))), flush=True)

# Q=0 => exp(logits)=1. Each of the 32 partial-sum lanes receives Kchunk/32
# per update. At 512 chunks the BF16 running sum stops growing after 256
# equal additions. Real TT exp(0), pack rounding, and numerator rounding
# make the observed gain differ slightly from this idealized factor of two.
for length in (4096, 32768, 131072, 262144):
    partial = torch.tensor(0, dtype=torch.bfloat16)
    for _ in range(length // 512):
        partial = (partial.float() + 512 / 32).bfloat16()
    denominator = partial.float().item() * 32
    print(
        json.dumps(
            dict(
                experiment="uniform_bf16_denominator",
                kv_len=length,
                expected_denominator=length,
                bf16_denominator=denominator,
                implied_gain=length / denominator,
            )
        ),
        flush=True,
    )
