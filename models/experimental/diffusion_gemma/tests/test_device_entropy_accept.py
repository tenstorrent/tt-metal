# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device entropy-budget acceptance spike (#47463, plan risk R1).

The plan's #1 unknown: can the entropy-budget acceptance — sort-by-confidence +
cumulative-entropy cutoff + **scatter-back to original canvas positions** — run
on device and reproduce the pure-torch reference (`reference/sampling.py`)?
This validates the op chain (`ttnn.sort` -> `ttnn.cumsum` -> `ttnn.le` ->
`ttnn.scatter`) against the oracle on real hardware.

STATUS: **unvalidated draft** — device op composition (sort layout, le/scatter
dtypes) needs live iteration on hardware; not yet run (board reset pending). Run
with DG_RUN_DEVICE=1 on a healthy QB2. fp32 throughout to isolate the chain
logic from bf16 near-boundary drift. ``min_accept`` is omitted here (a trivial
host/slice op); the spike targets the sort/scatter mapping, the R1 risk.
"""

import os

import pytest
import torch

import ttnn
from models.experimental.diffusion_gemma.reference import sampling as S

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("DG_RUN_DEVICE") != "1",
        reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (needs sfpi >= 7.60.0)",
    ),
    pytest.mark.use_module_device,  # single open/teardown — avoid QB2 erisc cycling
]


def _device_entropy_accept(device, entropy: torch.Tensor, budget: float) -> torch.Tensor:
    """Run the acceptance chain on device; return the bool accept mask [B, L]."""
    ent = ttnn.from_torch(entropy.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    sorted_vals, sorted_idx = ttnn.sort(ent, dim=-1)  # ascending: most-confident first; idx uint16
    cum = ttnn.cumsum(sorted_vals, dim=-1)
    accept_sorted = ttnn.le(cum, float(budget))  # cum <= budget -> 1.0 / 0.0
    accept_sorted = ttnn.typecast(accept_sorted, ttnn.float32)  # match scatter input/src dtype

    # Positional form per the proven unit test (data_movement/test_scatter.py): fp32
    # input/src + uint16 index + TILE. uint16 index is accepted
    # (scatter_device_operation.cpp:40). L=128 (<256) dodges the tiled-integer issue #23407.
    zeros = ttnn.zeros_like(ent)
    accept = ttnn.scatter(zeros, -1, sorted_idx, accept_sorted)  # scatter-back to original positions

    return ttnn.to_torch(accept) > 0.5


@pytest.mark.parametrize("frac", [0.0, 0.3, 0.7, 1.0], ids=["accept~0", "accept~30", "accept~70", "accept~all"])
def test_entropy_budget_accept_matches_reference(device, frac):
    torch.manual_seed(7)
    batch, length = 1, 128
    # distinct entropy values (avoid sort-tie ambiguity between torch and ttnn)
    entropy = torch.rand(batch, length) + torch.arange(length).float() * 1e-4

    # pick a budget that accepts ~frac of the positions, off any exact boundary
    sorted_cum = torch.cumsum(torch.sort(entropy, dim=-1).values, dim=-1)
    k = int(frac * length)
    if k == 0:
        budget = float(sorted_cum[0, 0]) * 0.5
    elif k >= length:
        budget = float(sorted_cum[0, -1]) * 2.0
    else:
        budget = float((sorted_cum[0, k - 1] + sorted_cum[0, k]) / 2)

    ref = S.entropy_budget_accept(entropy, budget, min_accept=0)
    dev = _device_entropy_accept(device, entropy, budget)

    assert dev.shape == ref.shape
    assert torch.equal(dev, ref), f"accept mask mismatch (frac={frac}): {int((dev != ref).sum())} of {length} differ"
