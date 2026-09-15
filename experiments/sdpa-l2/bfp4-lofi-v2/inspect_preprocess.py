# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Summarize saved rounding mismatches without treating failed probes as passing."""

from pathlib import Path
import torch

for path in sorted(Path(__file__).parent.glob("preprocess-finite-*.failure.pt")):
    data = torch.load(path, weights_only=True)
    x, actual, expected = [data[k].float().flatten() for k in ("input", "actual", "expected")]
    mask = actual != expected
    unique = torch.unique(x[mask])
    print(path.name, "mismatch", int(mask.sum()), "unique", unique.numel(), flush=True)
    exponents = torch.frexp(x[mask].abs())[1]
    print("exponent counts", torch.unique(exponents, return_counts=True), flush=True)
    indices = mask.nonzero().flatten()
    chosen = indices[torch.linspace(0, indices.numel() - 1, 24).long()]
    print([(float(x[i]), float(actual[i]), float(expected[i])) for i in chosen], flush=True)
