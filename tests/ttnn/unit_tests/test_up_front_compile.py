# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke test + usage example for Tier-1 up-front parallel precompile.

The flow (op-agnostic — works for any model whose ops dispatch through the
device-op adapter, i.e. generic_op and every C++ ProgramDescriptor-migrated op):

    ttnn.graph.up_front_begin_collect()        # NO_DISPATCH: nothing runs on HW;
    model(dummy_input, device)                 #   each op stashes its program
    ttnn.graph.up_front_end_collect()
    ttnn.graph.up_front_compile(device, 4)     # parallel JIT -> warm on-disk cache
    out = model(real_input, device)            # now runs warm

For a real model (e.g. ResNet-50, models/demos/.../resnet50) the body in the
collect block is the model's forward; everything else is identical. Run on a
COLD program cache so every op is a cache miss (reaches the collector).
"""

import time

import pytest
import torch

import ttnn


def _chain(x):
    """A tiny multi-op graph -> a few distinct programs.

    c = (exp(x) + x) * exp(x)
    """
    a = ttnn.exp(x)
    b = ttnn.add(a, x)
    c = ttnn.multiply(b, a)
    return c


def test_up_front_compile(device):
    torch.manual_seed(0)
    t = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    # Collect on a cold cache: run the graph in NO_DISPATCH; each op stashes its
    # built-but-uncompiled program. Nothing executes on hardware here.
    ttnn.graph.up_front_begin_collect()
    try:
        _chain(x)
    finally:
        ttnn.graph.up_front_end_collect()

    n_collected = ttnn.graph.up_front_num_collected()
    n_unique = ttnn.graph.up_front_num_unique()
    print(f"\nup_front: collected {n_collected} ops, {n_unique} unique programs")
    # The 3-op chain should have routed at least one op through the collector.
    assert n_collected >= 1, "collect captured no programs — did the ops go through the device-op funnel?"
    assert n_unique >= 1

    # Parallel compile -> warms the on-disk kernel cache. Must report zero errors.
    num_programs, num_errors, workers, wall = ttnn.graph.up_front_compile(device, 4)
    print(f"up_front: compiled {num_programs} programs in {wall:.2f}s (workers={workers}, errors={num_errors})")
    assert num_errors == 0, "parallel compile reported errors"

    # The real run is now warm and must still be numerically correct. Use PCC
    # (the idiomatic ttnn metric) — bf16 has ~3 sig figs, so allclose on the
    # large magnitudes this chain produces is the wrong check.
    out = _chain(x)
    got = ttnn.to_torch(out).float().flatten()

    a = torch.exp(t.float())
    expected = ((a + t.float()) * a).flatten()
    pcc = torch.corrcoef(torch.stack([got, expected]))[0, 1].item()
    print(f"up_front: warm-run PCC vs torch reference = {pcc:.6f}")
    assert pcc > 0.999, f"warm run incorrect: PCC {pcc}"
