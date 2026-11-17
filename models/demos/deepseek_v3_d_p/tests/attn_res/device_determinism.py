# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Determinism checks that stay on device, for the AttnRes ops and the read they make up.

The pattern is `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py`'s. A host readback per
repeat serializes the chips, so it removes the timing skew a race needs, and it costs enough that
only a couple of repeats are affordable. Instead the first run's outputs stay on device as the
reference, every later run folds `max(ne(reference, output))` into one running marker, and the
marker is read back once at the end. Nothing syncs inside the loop, so a fast chip is free to get
ahead of a slow one, which is the condition a missing receiver-ready handshake needs.

Only a failure pays for a host diff, to say where the runs came apart.
"""

from __future__ import annotations

import torch

import ttnn


def deterministic_input(*shape, seed=0, scale=1.0, offset=0.0, dtype=torch.bfloat16):
    """Seeded normal draws, for runs whose accuracy is gated elsewhere.

    Drawn rather than patterned: a pattern smooth enough to be cheap rounds to the same bf16 values
    across neighbouring columns, which would hand every tensor-parallel rank the same shard, and a
    peer's stale or misplaced write could then land on numbers it happens to equal.
    """
    rng = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=rng, dtype=dtype) * scale + offset


def device_mismatch_marker(reference, actual):
    """A reduced device tensor that is non-zero on any chip where `actual` differs from `reference` anywhere."""
    mismatch = ttnn.ne(reference, actual, dtype=ttnn.bfloat16)
    marker = ttnn.max(mismatch)
    ttnn.deallocate(mismatch)
    return marker


def merge_device_mismatch_markers(current, new):
    """Fold `new` into the running marker `current`, consuming both."""
    if current is None:
        return new
    merged = ttnn.maximum(current, new)
    ttnn.deallocate(current)
    ttnn.deallocate(new)
    return merged


def fold_mismatches(marker, references, outputs):
    """Fold every output's mismatch against its reference into `marker`, without a sync."""
    for reference, output in zip(references, outputs, strict=True):
        marker = merge_device_mismatch_markers(marker, device_mismatch_marker(reference, output))
    return marker


def device_mismatch_marker_is_set(marker):
    """The one readback of the run: whether any chip saw a mismatch."""
    host = ttnn.from_device(marker)
    return any(float(ttnn.to_torch(shard).item()) != 0.0 for shard in ttnn.get_device_tensors(host))


def mismatch_report(names, references, outputs):
    """Host-side diff of the outputs that came apart, per chip, for the failure message only."""
    lines = []
    for name, reference, output in zip(names, references, outputs, strict=True):
        pairs = zip(
            ttnn.get_device_tensors(ttnn.from_device(reference)), ttnn.get_device_tensors(ttnn.from_device(output))
        )
        for chip, (want, got) in enumerate(pairs):
            want, got = ttnn.to_torch(want), ttnn.to_torch(got)
            if torch.equal(want, got):
                continue
            differs = want != got
            lines.append(
                f"{name} chip {chip}: {int(differs.sum())}/{differs.numel()} elements differ, "
                f"max |diff| {(want.float() - got.float()).abs().max().item():.6e}"
            )
    return "\n".join(lines) or "the marker was set but the last outputs match their references"
