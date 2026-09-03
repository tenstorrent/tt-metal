# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device entropy-budget acceptance spike (#47463, plan risk R1).

The plan's #1 unknown: can the entropy-budget acceptance — sort-by-confidence +
cumulative-entropy cutoff + **scatter-back to original canvas positions** — run
on device and reproduce the pure-torch reference (`reference/sampling.py`)?
Validates the op chain (`ttnn.sort` -> `ttnn.cumsum` -> `ttnn.le` ->
`ttnn.scatter`) against the oracle on real hardware.

STATUS: iterating on hardware (QB2). fp32 entropy/cumsum/threshold to isolate
chain logic from bf16 drift; the scatter mask is bf16 (ttnn.scatter rejects
fp32+TILE, scatter.cpp:109). ``min_accept`` omitted (host/slice op); the spike
targets the sort/scatter mapping, the R1 risk. Run with DG_RUN_DEVICE=1.
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


def _device_chain(device, entropy: torch.Tensor, budget: float) -> dict:
    """Run the acceptance chain on device; return every intermediate as torch."""
    ent = ttnn.from_torch(entropy.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    sorted_vals, sorted_idx = ttnn.sort(ent, dim=-1)  # ascending: most-confident first; idx uint16
    cum = ttnn.cumsum(sorted_vals, dim=-1)

    # EXCLUSIVE prefix (HF accept_canvas): position i accepts iff the sum over
    # *strictly more confident* positions stays <= budget, i.e. (cum - sorted_vals).
    # The most-confident position has an exclusive prefix of 0 -> always accepted.
    # (Inclusive cum <= budget wrongly drops the element that crosses the budget.)
    excl = ttnn.subtract(cum, sorted_vals)

    # tensor budget (unambiguous tensor-tensor compare; scalar overload misbehaved)
    budget_t = ttnn.full(list(entropy.shape), float(budget), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    accept_sorted = ttnn.le(excl, budget_t)  # exclusive prefix <= budget -> 1 / 0

    # ttnn.scatter rejects fp32+TILE (scatter.cpp:109); bf16+uint16+TILE is supported
    # (test_scatter.py:92). Mask is 0/1 -> exact in bf16. L<256 dodges issue #23407.
    accept_sorted_bf = ttnn.typecast(accept_sorted, ttnn.bfloat16)
    zeros = ttnn.typecast(ttnn.zeros_like(ent), ttnn.bfloat16)
    accept = ttnn.scatter(zeros, -1, sorted_idx, accept_sorted_bf)  # scatter-back to original positions

    return {
        "sorted_vals": ttnn.to_torch(sorted_vals).float(),
        "cum": ttnn.to_torch(cum).float(),
        "accept_sorted": ttnn.to_torch(accept_sorted_bf) > 0.5,
        "accept": ttnn.to_torch(accept) > 0.5,
    }


def _budget_for_fraction(entropy: torch.Tensor, frac: float) -> float:
    sorted_cum = torch.cumsum(torch.sort(entropy, dim=-1).values, dim=-1)
    k = int(frac * entropy.shape[-1])
    if k == 0:
        return float(sorted_cum[0, 0]) * 0.5
    if k >= entropy.shape[-1]:
        return float(sorted_cum[0, -1]) * 2.0
    return float((sorted_cum[0, k - 1] + sorted_cum[0, k]) / 2)


def test_acceptance_chain_stagewise_diagnostic(device):
    """One run that reports where (if anywhere) the device chain diverges."""
    torch.manual_seed(7)
    batch, length = 1, 128
    entropy = torch.rand(batch, length) + torch.arange(length).float() * 1e-4
    budget = _budget_for_fraction(entropy, 0.7)

    d = _device_chain(device, entropy, budget)

    t_vals, _ = torch.sort(entropy, dim=-1)
    t_cum = torch.cumsum(t_vals, dim=-1)
    ref_accept_sorted = (t_cum - t_vals) <= budget  # EXCLUSIVE prefix (matches device/HF), not inclusive
    ref = S.entropy_budget_accept(entropy, budget, min_accept=0)

    diag = (
        f"budget={budget:.3f} | "
        f"sort_close={torch.allclose(d['sorted_vals'], t_vals, atol=1e-2)} | "
        f"cum_close={torch.allclose(d['cum'], t_cum, atol=1e-1)} "
        f"(dev_total={float(d['cum'][0, -1]):.2f} torch_total={float(t_cum[0, -1]):.2f}) | "
        f"accept_sorted dev_sum={int(d['accept_sorted'].sum())} ref_sum={int(ref_accept_sorted.sum())} | "
        f"final dev_sum={int(d['accept'].sum())} ref_sum={int(ref.sum())}"
    )
    assert torch.equal(d["accept"], ref), diag


@pytest.mark.parametrize("frac", [0.0, 0.3, 0.7, 1.0], ids=["accept~0", "accept~30", "accept~70", "accept~all"])
def test_entropy_budget_accept_matches_reference(device, frac):
    torch.manual_seed(7)
    batch, length = 1, 128
    entropy = torch.rand(batch, length) + torch.arange(length).float() * 1e-4
    budget = _budget_for_fraction(entropy, frac)

    ref = S.entropy_budget_accept(entropy, budget, min_accept=0)
    dev = _device_chain(device, entropy, budget)["accept"]

    assert dev.shape == ref.shape
    assert torch.equal(dev, ref), f"accept mask mismatch (frac={frac}): {int((dev != ref).sum())} of {length} differ"
