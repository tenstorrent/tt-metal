# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Determinism checks that stay on device.

The pattern is tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py's. A host readback per run serializes the chips,
removing the timing skew a race needs, and costs enough that only a couple of runs are affordable. Instead the first
run's outputs stay on device as the reference, every later run folds max(ne(reference, output)) into one running
marker, and each marker is read back once at the end. Nothing syncs inside the loops, so a fast chip is free to get
ahead of a slow one. Only a failure pays for a host diff, to say where the runs came apart.

ttnn.ne compares values, not bits: a NaN never equals itself (so a NaN fails the check) and -0 equals +0.

check_determinism runs, for one step:
  * eager: `runs` calls back to back; run 0's and the latest output are kept, each one in between is freed once the
    next run's compare is enqueued.
  * replays: `runs` replays of one captured call, each copied on device into a buffer allocated before the capture,
    folded after the last replay.
  * traced vs eager: the first replay must equal the eager reference.
A step returns a tensor or a sequence of tensors; every tensor is compared.
"""

from __future__ import annotations

from typing import Callable

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.measure.timing import free
from models.demos.deepseek_v3_d_p.tests.measure.trace import TracedStep, drain

RUNS = 10


def tensors(output) -> list:
    """The device tensors of a step's output, in order (a tensor, or a sequence of them)."""
    if isinstance(output, ttnn.Tensor):
        return [output]
    if isinstance(output, (list, tuple)):
        return [t for item in output for t in tensors(item)]
    raise TypeError(f"a step must return a tensor or a sequence of tensors, got {type(output).__name__}")


def deterministic_input(*shape, seed=0, scale=1.0, offset=0.0, dtype=torch.bfloat16):
    """Seeded normal draws, for runs whose accuracy is gated elsewhere.

    Drawn rather than patterned: a pattern smooth enough to be cheap rounds to the same bf16 values across
    neighbouring columns, which would hand every tensor-parallel rank the same shard, and a peer's stale or misplaced
    write could then land on numbers it happens to equal.
    """
    rng = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=rng, dtype=dtype) * scale + offset


def device_mismatch_marker(reference, actual, mask=None, memory_config=None):
    """A reduced device tensor that is non-zero on any chip where `actual` differs from `reference` anywhere. `mask`
    (0/1, broadcastable) limits the compare to defined elements, e.g. to skip NaN padding. `memory_config` places the
    elementwise mismatch (default: like the inputs; DRAM interleaved for ND-sharded inputs, which max cannot reduce)."""
    mismatch = ttnn.ne(reference, actual, dtype=ttnn.bfloat16, memory_config=memory_config)
    if mask is not None:
        # The 0/1 mismatch is finite, so masking it by multiplication is exact.
        masked = ttnn.multiply(mismatch, mask)
        ttnn.deallocate(mismatch)
        mismatch = masked
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
    """The one readback of a marker: whether any chip saw a mismatch."""
    return bool(marker_chips(marker))


def marker_chips(marker) -> list[int]:
    """The chips whose marker is set."""
    host = ttnn.from_device(marker)
    return [chip for chip, shard in enumerate(ttnn.get_device_tensors(host)) if float(ttnn.to_torch(shard).item())]


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


def eager_runs(mesh_device, step: Callable[[], object], runs: int = RUNS):
    """`runs` calls back to back with no sync. Returns (reference, last, marker): run 0's output tensors, the last
    run's, and max(ne(reference, output)) over runs 1.. folded on device. Outputs in between are freed once folded."""
    reference, last, marker = tensors(step()), None, None
    try:
        for _ in range(1, runs):
            out = tensors(step())
            marker = fold_mismatches(marker, reference, out)
            if last is not None:
                free(last)  # safe: later work reusing it runs after the queued compare (one CQ)
            last = out
        return reference, last, marker
    except BaseException:
        drain(mesh_device)
        free([reference, last])
        if marker is not None:
            ttnn.deallocate(marker)
        raise


def replay_runs(traced: TracedStep, copies: list, runs: int = RUNS):
    """`runs` replays of `traced` (not yet captured) back to back, each copied on device into buffers allocated before
    the capture and appended to `copies` (one list of tensors per replay). Returns max(ne(copies[0], copy)) over the
    rest, folded after the last replay. The caller frees `copies` once `traced` is closed (its `with` drains first)."""

    def allocate(out):
        for _ in range(runs):
            copies.append([ttnn.clone(t) for t in tensors(out)])

    traced.capture(before_capture=allocate)
    captured = tensors(traced.output)
    for copy in copies:  # same CQ: each copy runs after its replay, before the next replay
        traced.replay(sync=False)
        for src, dst in zip(captured, copy, strict=True):
            ttnn.copy(src, dst)
    marker = None
    for copy in copies[1:]:  # after the last replay
        marker = fold_mismatches(marker, copies[0], copy)
    return marker


def check_determinism(
    mesh_device,
    step: Callable[[], object],
    traced: Callable[[], TracedStep],
    runs: int = RUNS,
) -> str | None:
    """`runs` eager runs, `runs` replays and traced vs eager, compared on device. Returns failure text, or None.

    `traced()` returns a fresh, uncaptured TracedStep over the same step (so it can attach the model's controller)."""
    reference, last, eager_marker = eager_runs(mesh_device, step, runs)
    copies, replay_marker, traced_marker = [], None, None
    try:
        with traced() as t:
            replay_marker = replay_runs(t, copies, runs)
        traced_marker = fold_mismatches(None, reference, copies[0])
        failed = [
            name
            for name, marker in (
                ("eager runs", eager_marker),
                ("replays", replay_marker),
                ("traced vs eager", traced_marker),
            )
            if marker is not None and device_mismatch_marker_is_set(marker)  # the only readbacks on a pass
        ]
        if not failed:
            return None
        # Host diffs only on failure; a diagnostics error must not replace the failure itself.
        try:
            detail = ""
            if "eager runs" in failed:  # eager keeps only its last run: name every chip that differed in any run
                detail = f"eager runs differ on chips {marker_chips(eager_marker)}\n"
            n = len(reference)
            names, refs, outs = [], [], []
            for k in range(n):
                suffix = f" output {k}" if n > 1 else ""
                names += [f"eager last run{suffix}"] + [f"replay {i}{suffix}" for i in range(1, runs)]
                names += [f"replay 0 vs eager{suffix}"]
                refs += [reference[k]] + [copies[0][k]] * (runs - 1) + [reference[k]]
                outs += [(last or reference)[k]] + [c[k] for c in copies[1:]] + [copies[0][k]]
            detail += mismatch_report(names, refs, outs)
        except Exception as e:
            detail = f"(diagnostics failed: {e})"
        return f"{', '.join(failed)}:\n{detail}"
    finally:
        drain(mesh_device)
        free([reference, last, copies])
        for marker in (eager_marker, replay_marker, traced_marker):
            if marker is not None:
                ttnn.deallocate(marker)
