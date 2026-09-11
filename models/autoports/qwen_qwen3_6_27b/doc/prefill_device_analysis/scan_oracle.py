# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU FP64 full-sequence oracle on captured real device-preprocessed inputs."""

import argparse
import json

import torch


def compare(ref, actual):
    ref, actual = ref.double().flatten(), actual.double().flatten()
    return {
        "relative_l2": float((actual - ref).norm() / ref.norm()),
        "pcc": float(torch.corrcoef(torch.stack((ref, actual)))[0, 1]),
        "max_abs": float((actual - ref).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    chunks = torch.load(args.inputs, weights_only=False)
    state = torch.zeros(chunks[0][0].shape[0], 1, 128, 128, dtype=torch.float64)
    tokens = 0
    for chunk in chunks:
        q, k, v, beta, decay = [x.double() for x in chunk]
        for i in range(q.shape[1]):
            ki = k[:, i : i + 1]
            state = state * decay[:, i : i + 1]
            delta = (v[:, i : i + 1] - ki @ state) * beta[:, i : i + 1]
            state = state + ki.transpose(-1, -2) @ delta
            tokens += 1
    baseline = torch.load(args.baseline, weights_only=False)["cache_0_recurrent"]
    native = torch.load(args.native, weights_only=False)["cache_0_recurrent"]
    result = {
        "tokens": tokens,
        "reference": "FP64 sequential recurrence, captured real TT inputs, no intermediate cache quantization",
        "baseline": compare(state, baseline),
        "native": compare(state, native),
    }
    open(args.result, "w").write(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
