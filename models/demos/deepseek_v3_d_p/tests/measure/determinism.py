# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Determinism checks that stay on device, as in tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py: the first
run's outputs stay on device as the reference, later runs fold max(ne(reference, output)) into a marker, and each
marker is read back once at the end. Nothing syncs inside the loops, so a fast chip can get ahead of a slow one, which
is what exposes a race. ttnn.ne compares values: a NaN fails, -0 equals +0."""

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.measure.timing import free
from models.demos.deepseek_v3_d_p.tests.measure.trace import drain

RUNS = 10


def tensors(output):
    """The device tensors of a step's output (a tensor or a sequence of them), in order."""
    return [output] if isinstance(output, ttnn.Tensor) else [t for item in output for t in tensors(item)]


def deterministic_input(*shape, seed=0, scale=1.0, offset=0.0, dtype=torch.bfloat16):
    """Seeded normal draws (not a smooth pattern, which would give every TP rank the same values)."""
    rng = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=rng, dtype=dtype) * scale + offset


def device_mismatch_marker(reference, actual):
    """Non-zero on any chip where `actual` differs from `reference`."""
    mismatch = ttnn.ne(reference, actual, dtype=ttnn.bfloat16)
    marker = ttnn.max(mismatch)
    ttnn.deallocate(mismatch)
    return marker


def merge_device_mismatch_markers(current, new):
    """Fold `new` into `current`, consuming both."""
    if current is None:
        return new
    merged = ttnn.maximum(current, new)
    ttnn.deallocate(current)
    ttnn.deallocate(new)
    return merged


def fold_mismatches(marker, references, outputs):
    for reference, output in zip(references, outputs, strict=True):
        marker = merge_device_mismatch_markers(marker, device_mismatch_marker(reference, output))
    return marker


def marker_chips(marker):
    """The chips whose marker is set (the marker's one readback)."""
    shards = ttnn.get_device_tensors(ttnn.from_device(marker))
    return [chip for chip, shard in enumerate(shards) if float(ttnn.to_torch(shard).item())]


def device_mismatch_marker_is_set(marker):
    return bool(marker_chips(marker))


def mismatch_report(names, references, outputs):
    """Per-chip host diff of the outputs that came apart, for the failure message only."""
    lines = []
    for name, reference, output in zip(names, references, outputs, strict=True):
        pairs = zip(
            ttnn.get_device_tensors(ttnn.from_device(reference)), ttnn.get_device_tensors(ttnn.from_device(output))
        )
        for chip, (want, got) in enumerate(pairs):
            want, got = ttnn.to_torch(want), ttnn.to_torch(got)
            if not torch.equal(want, got):
                differs = want != got
                lines.append(
                    f"{name} chip {chip}: {int(differs.sum())}/{differs.numel()} elements differ, "
                    f"max |diff| {(want.float() - got.float()).abs().max().item():.6e}"
                )
    return "\n".join(lines) or "the marker was set but the last outputs match their references"


def check_determinism(mesh_device, step, traced):
    """RUNS eager calls back to back, RUNS replays of `traced()` (a fresh TracedStep over the same step) each
    copied into a buffer allocated before the capture, and the first replay against the first eager run, all compared
    on device. Returns failure text, or None."""
    reference, last, copies, markers = tensors(step()), None, [], {}
    try:
        for _ in range(1, RUNS):
            out = tensors(step())
            markers["eager runs"] = fold_mismatches(markers.get("eager runs"), reference, out)
            free(last)  # safe: later work reusing it runs after the queued compare (one CQ)
            last = out
        with traced() as t:
            t.capture(
                before_capture=lambda out: copies.extend([ttnn.clone(x) for x in tensors(out)] for _ in range(RUNS))
            )
            for copy in copies:  # one CQ: each copy runs after its replay, before the next
                t.replay(sync=False)
                for src, dst in zip(tensors(t.output), copy, strict=True):
                    ttnn.copy(src, dst)
        for copy in copies[1:]:  # after the last replay
            markers["replays"] = fold_mismatches(markers.get("replays"), copies[0], copy)
        markers["traced vs eager"] = fold_mismatches(None, reference, copies[0])
        failed = {name: chips for name, marker in markers.items() if (chips := marker_chips(marker))}
        if not failed:
            return None
        try:  # a diagnostics error must not replace the failure itself
            detail = mismatch_report(
                ["eager last run", "last replay", "replay 0 vs eager"],
                [reference[0], copies[0][0], reference[0]],
                [(last or reference)[0], copies[-1][0], copies[0][0]],
            )
        except Exception as e:
            detail = f"(diagnostics failed: {e})"
        return "; ".join(f"{name} differ on chips {chips}" for name, chips in failed.items()) + f"\n{detail}"
    finally:
        drain(mesh_device)
        free([reference, last, copies, list(markers.values())])
