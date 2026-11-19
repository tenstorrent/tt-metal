# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only TF32 operand-loss hypothesis on saved actual-device GDN operands.

This simulates the three FPU consumers (state read, rank-one update, output
read), preserving FP32 carried state/decay/delta. It is not a hardware model:
FPU accumulation ordering and SFPU exp are not reproduced exactly.
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def quantize(value, mode):
    bits = value.contiguous().view(torch.int32)
    if mode == "rne":
        bits = bits + 4095 + ((bits >> 13) & 1)
    return (bits & -8192).view(torch.float32)


def simulate(fixture, mode):
    q = fixture["query"] * (128**-0.5)
    k, v, decay, beta = (fixture[n] for n in ("key", "value", "decay", "beta"))
    state = fixture["initial_state"].clone()
    result = []
    for token in range(q.shape[1]):
        state = state * decay[:, token].exp()[..., None, None]
        key = quantize(k[:, token], mode)
        read = (key[..., None] * quantize(state, mode)).sum(-2)
        delta = (v[:, token] - read) * beta[:, token, :, None]
        state = state + key[..., None] * quantize(delta, mode)[..., None, :]
        output = (quantize(q[:, token], mode)[..., None] * quantize(state, mode)).sum(-2)
        result.append(output)
    return torch.stack(result, dim=1), state


def error(expected, actual):
    difference = expected.double() - actual.double()
    return {
        "max_absolute": difference.abs().max().item(),
        "rms": difference.square().mean().sqrt().item(),
        "bf16_equal_fraction": (expected.bfloat16() == actual.bfloat16()).double().mean().item(),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    report = {"scope": __doc__, "fixtures": {}}
    for path in sorted(args.fixtures.glob("*.pt")):
        fixture = torch.load(path, weights_only=True)
        record = {"observed_error": {n: error(fixture["hf_" + n], fixture["tt_" + n]) for n in ("output", "state")}}
        for mode in ("trunc", "rne"):
            output, state = simulate(fixture, mode)
            record[mode] = {
                "simulation_vs_device": {
                    n: error(fixture["tt_" + n], v) for n, v in (("output", output), ("state", state))
                },
                "simulation_vs_hf": {
                    n: error(fixture["hf_" + n], v) for n, v in (("output", output), ("state", state))
                },
            }
        report["fixtures"][path.name] = record
    assert report["fixtures"] and "ttnn" not in sys.modules
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
