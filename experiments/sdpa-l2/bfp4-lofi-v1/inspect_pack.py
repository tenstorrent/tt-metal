# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Print residual packer-model mismatches from probe artifacts."""

import json

import torch

from probe import HERE, quantize

for source in ("bf16", "fp32"):
    for case in ("thresholds", "random"):
        data = torch.load(HERE / f"pack-{case}-{source}-3.pt", weights_only=True)
        x, actual = data["input"], data["device"]
        expected = quantize(x, 3, "rna8_trunc")
        mismatch = (actual != expected).flatten().nonzero().flatten()
        examples = set()
        for i in mismatch:
            index = int(i)
            group = x.flatten()[index // 16 * 16 : index // 16 * 16 + 16]
            examples.add(
                (
                    float(x.flatten()[i]),
                    float(group.abs().max()),
                    float(actual.flatten()[i]),
                    float(expected.flatten()[i]),
                )
            )
        print(
            json.dumps(
                dict(
                    source=source,
                    case=case,
                    count=int(mismatch.numel()),
                    columns=["input", "group_absmax", "device", "rna8_trunc"],
                    examples=sorted(examples)[:80],
                )
            ),
            flush=True,
        )
