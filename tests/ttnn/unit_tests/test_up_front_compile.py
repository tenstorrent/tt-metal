# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke test + usage example for up-front parallel precompile.

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

    # The collector is a process-wide singleton — clear it so this test is independent
    # of whatever ran before it.
    ttnn.graph.up_front_clear()

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


def test_up_front_collect_mechanics(device):
    """Lock down the collector invariants the feature depends on, independent of
    timing or kernel-cache state: clear, stash/dedup accounting, compile↔unique
    parity, post-compile reset, and empty-compile safety.

    These are relational (no magic op counts) so they don't go stale if op→program
    decomposition changes; the point is the plumbing, not the exact numbers.
    """
    torch.manual_seed(0)
    t = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    # clear() empties the collector.
    ttnn.graph.up_front_clear()
    assert ttnn.graph.up_front_num_collected() == 0
    assert ttnn.graph.up_front_num_unique() == 0

    # A collect pass stashes a program per dispatched op; the deduped unique set is
    # non-empty and never larger than the total stashed.
    ttnn.graph.up_front_begin_collect()
    try:
        _chain(x)
    finally:
        ttnn.graph.up_front_end_collect()
    collected = ttnn.graph.up_front_num_collected()
    unique = ttnn.graph.up_front_num_unique()
    assert collected >= 1, "collect captured nothing — ops didn't reach the device-op funnel"
    assert 1 <= unique <= collected

    # compile() JIT-builds exactly the deduped unique set, error-free, then resets the
    # collector. If compile silently skipped programs this parity check would catch it.
    num_programs, num_errors, _, _ = ttnn.graph.up_front_compile(device, 4)
    assert num_errors == 0, "parallel compile reported errors"
    assert num_programs == unique, f"compiled {num_programs} programs, expected the {unique} unique collected"
    assert ttnn.graph.up_front_num_collected() == 0, "compile() should clear the collector"

    # Compiling an empty collector is a no-op, not an error.
    n, e, _, _ = ttnn.graph.up_front_compile(device, 4)
    assert (n, e) == (0, 0), f"empty compile should be a no-op, got ({n}, {e})"


def test_up_front_collect_accumulates(device):
    """clear=False accumulates across collect passes — the pytest-plugin usage where
    many test bodies feed one deduped set compiled once at the end."""
    torch.manual_seed(0)
    t = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    ttnn.graph.up_front_clear()

    ttnn.graph.up_front_begin_collect(clear=False)
    try:
        _chain(x)
    finally:
        ttnn.graph.up_front_end_collect()
    after_first = ttnn.graph.up_front_num_collected()

    ttnn.graph.up_front_begin_collect(clear=False)
    try:
        _chain(x)
    finally:
        ttnn.graph.up_front_end_collect()
    after_second = ttnn.graph.up_front_num_collected()

    assert after_first >= 1
    assert after_second > after_first, "clear=False should accumulate, not reset, across passes"
    assert ttnn.graph.up_front_num_unique() <= after_second
    ttnn.graph.up_front_clear()
