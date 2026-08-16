#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A falsifiable prediction that separates "wrong formula" from "bf16 rounding".

`presence_penalty_arithmetic_probe.py` reproduced the device's greedy token sequence
160/160 by taking vLLM's own presence rule and doing the subtraction in bfloat16, and it
showed that the one step where the device and vLLM's host sampler disagree is a step where
bf16 rounding turns a 0.05 fp32 margin into an exact tie.  That is an emulation, so it
should be made to stick its neck out.

The prediction.  The device stores the penalty in bf16 and subtracts it from bf16 logits.
Whether that subtraction rounds at all is decided purely by the penalty's own binary
expansion:

  * a penalty that is an exact multiple of the logit ULP -- 0.5, 1.25, 2.0, all multiples
    of 2^-3 -- leaves ``logit - penalty`` exactly representable on the logit's bf16 grid,
    on any binade the logits occupy here.  The device's arithmetic is then EXACT, and it
    must reproduce vLLM's fp32 host sampler token for token.
  * a penalty that is not -- 1.2 (bf16: 1.203125 = 77/64), 0.7 (bf16: 0.69921875), 1.1
    (bf16: 1.1015625) -- forces a rounding of up to half an ULP on every penalised token,
    and the two paths CAN split.  Only "can": rounding flips a decision only when a
    near-tie happens to fall inside the rounding window, so an unaligned penalty that
    never meets one stays identical.  The falsifiable half of the prediction is the
    aligned half, and it is one-sided by construction.

A "wrong formula" -- wrong token set, count instead of presence, penalty not landing --
predicts nothing of the kind: it would be just as wrong at 1.25 as at 1.2, and (per
`arithmetic_probe.json`) wrong from step 9, 30 or 88 rather than step 159.

So: greedy (temperature 0) device vs host, same prompt, six penalties.  `logprobs: true`
forces vLLM's host sampler on this 4-die mesh; the plain request runs on device.

Usage::

    python presence_penalty_grid_prediction.py --out <path.json>
"""

from __future__ import annotations

import argparse
import json
import struct
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "meta-models/Muse-Glimmer-30B"
PROMPT = [{"role": "user", "content": "Write a very repetitive story."}]

# (penalty, grid-aligned?) -- aligned means an exact multiple of 2^-3, so exact on every
# bf16 binade the logits of this model occupy (|logit| < 256 => ULP <= 1.0... in practice
# the top logits sit in [8,32), ULP 0.0625-0.125).
PENALTIES = [
    (0.5, True),
    (1.25, True),
    (2.0, True),
    (0.7, False),
    (1.1, False),
    (1.2, False),
]


def post(body: dict) -> dict:
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def to_bf16(x: float) -> float:
    """Round-to-nearest-even a float32 down to bfloat16, returned as a float."""
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    lower = b & 0xFFFF
    b &= 0xFFFF0000
    if lower > 0x8000 or (lower == 0x8000 and (b >> 16) & 1):
        b += 0x10000
    return struct.unpack("<f", struct.pack("<I", b))[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    cases = []
    for p, aligned in PENALTIES:
        base = {
            "model": MODEL,
            "messages": PROMPT,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "presence_penalty": p,
            "return_token_ids": True,
        }
        host = post({**base, "logprobs": True, "top_logprobs": 1})["choices"][0]
        dev = post(dict(base))["choices"][0]
        h, d = list(host["token_ids"]), list(dev["token_ids"])
        n = min(len(h), len(d))
        first = next((i for i in range(n) if h[i] != d[i]), -1)
        bf = to_bf16(p)
        cases.append(
            {
                "presence_penalty": p,
                "penalty_as_bf16": bf,
                "penalty_is_exact_in_bf16": bf == p,
                "penalty_is_multiple_of_0.125": abs(p / 0.125 - round(p / 0.125)) < 1e-9,
                "grid_aligned": aligned,
                "prediction": "device == host, exactly (subtraction is exact in bf16)"
                if aligned
                else "device MAY diverge (subtraction rounds; flips only at a near-tie)",
                "host_tokens": len(h),
                "device_tokens": len(d),
                "first_divergent_step": first,
                "identical": first == -1 and len(h) == len(d),
                # one-sided: only the aligned arm makes a falsifiable claim
                "falsifies_prediction": aligned and first != -1,
            }
        )
        print(json.dumps(cases[-1]))

    out = {
        "_what": __doc__.strip().splitlines()[0],
        "prompt": PROMPT[0]["content"],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "cases": cases,
        "grid_aligned_penalties_identical": sum(c["identical"] for c in cases if c["grid_aligned"]),
        "grid_aligned_penalties": sum(1 for c in cases if c["grid_aligned"]),
        "unaligned_penalties_diverged": sum(1 for c in cases if not c["grid_aligned"] and not c["identical"]),
        "unaligned_penalties": sum(1 for c in cases if not c["grid_aligned"]),
        "falsifications": sum(c["falsifies_prediction"] for c in cases),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "cases"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
