# SPDX-License-Identifier: Apache-2.0
"""Preserve failing outputs and finite/range diagnostics outside device execution."""

import argparse
import json

import torch

from . import multichip_checks

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--tokens", type=int, default=8193)
    p.add_argument("--repetitions", type=int, default=10)
    a = p.parse_args()
    a.baseline = False
    original = multichip_checks.pcc
    counter = [0]

    def diagnose(expected, actual):
        index = counter[0]
        counter[0] += 1
        if index == 0:
            rows = []
            for name, value in [("expected", expected), ("actual", actual)]:
                x = value.float()
                rows.append(
                    dict(
                        name=name,
                        shape=list(x.shape),
                        finite=bool(torch.isfinite(x).all()),
                        maximum=float(x.max()),
                        minimum=float(x.min()),
                        maximum_abs=float(x.abs().max()),
                    )
                )
            chunks = []
            for start in range(0, a.tokens, 1024):
                x = expected[..., start : start + 1024, :].double().flatten()
                y = actual[..., start : start + 1024, :].double().flatten()
                chunks.append(dict(start=start, pcc64=float(torch.corrcoef(torch.stack([x, y]))[0, 1])))
            result = dict(stats=rows, chunks=chunks)
            (multichip_checks.OUT / f"{a.tag}_diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
            torch.save(actual, multichip_checks.OUT / f"{a.tag}_actual.pt")
            print(json.dumps(result), flush=True)
        return original(expected, actual)

    multichip_checks.pcc = diagnose
    multichip_checks.run(a)
