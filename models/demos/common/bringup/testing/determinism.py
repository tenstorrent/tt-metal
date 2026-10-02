# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bitwise determinism: the same input gives the same bytes, also right after a different input (A, B, A).

A race (a semaphore, a circular buffer, a multicast counter) or state left over from the previous call shows up as
run-to-run differences long before a PCC gate sees it. The check runs A, then B (same op and shapes, other data),
then A again until A has run ``repeats`` times, and compares every A output byte for byte with the first one.
Cost: ``repeats`` calls of A plus one of B; the module or op is already built and its programs compiled.

Component tests (testing/component.py) and fork tests (ttnn/ttnn/bringup/<fork>/tests) call it:

    determinism.assert_deterministic(run_a, run_b, first=out_a, label=case["id"])   # fork tests
    ok = determinism.check(run_a, run_b, first=out_a, label=..., metric=...)          # component tests

``run_a`` / ``run_b`` return the outputs: torch tensors, ttnn tensors (every device's shard is compared) or nested
tuples, lists and dicts of them. Switches (defaults.yaml): ``tests.determinism_repeats`` (A runs in total, 0 turns
the check off) and ``tests.nondeterministic`` (component steps exempt, each with its reason).
"""

from __future__ import annotations

from typing import Callable

import torch

from models.demos.common.bringup.core import defaults


def repeats(spec=None) -> int:
    key = "tests.determinism_repeats"
    return int(spec.get(key) if spec is not None else defaults.get(key))


def exempt(spec, step: str) -> str | None:
    """The recorded reason a component step is exempt from the check, else None."""
    ex = spec.get("tests.nondeterministic") or {}
    return ex.get(step) if isinstance(ex, dict) else ("listed in tests.nondeterministic" if step in ex else None)


def _flat(x, path: str = "out") -> list[tuple[str, torch.Tensor]]:
    if isinstance(x, torch.Tensor):
        return [(path, x)]
    if isinstance(x, dict):
        return [p for k in sorted(x, key=str) for p in _flat(x[k], f"{path}.{k}")]
    if isinstance(x, (list, tuple)):
        return [p for i, v in enumerate(x) for p in _flat(v, f"{path}[{i}]")]
    if x is None or isinstance(x, (int, float, bool, str)):
        return [(path, torch.tensor([hash(x)]))]
    import ttnn

    if isinstance(x, ttnn.Tensor):
        return [(f"{path}@dev{i}", ttnn.to_torch(t)) for i, t in enumerate(ttnn.get_device_tensors(x))]
    raise TypeError(f"{path}: cannot compare a {type(x).__name__}")


def snapshot(out) -> list[tuple[str, torch.Tensor]]:
    """Host copies of every output, taken at once (a later call may reuse the device buffers)."""
    return [(p, t.detach().cpu().contiguous().clone()) for p, t in _flat(out)]


def diff(a: list, b: list) -> str | None:
    """None when every tensor is byte-identical, else what differs first."""
    if [p for p, _ in a] != [p for p, _ in b]:
        return f"output structure differs: {[p for p, _ in a]} vs {[p for p, _ in b]}"
    for (p, x), (_, y) in zip(a, b):
        if x.shape != y.shape or x.dtype != y.dtype:
            return f"{p}: {tuple(x.shape)} {x.dtype} vs {tuple(y.shape)} {y.dtype}"
        bx, by = _bits(x), _bits(y)
        if not torch.equal(bx, by):
            ne = bx != by
            n, where = int(ne.sum()), int(ne.nonzero()[0])
            mx = (x.float() - y.float()).abs().nan_to_num(float("inf")).max().item() if x.is_floating_point() else ""
            return f"{p}: {n}/{x.numel()} elements differ in their bits (first at flat index {where}" + (
                f", max abs diff {mx:.3g})" if mx != "" else ")"
            )
    return None


_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _bits(t: torch.Tensor) -> torch.Tensor:
    """One integer per element holding its bits (signed zeros and NaN payloads count as different)."""
    t = t.reshape(-1)
    if t.dtype == torch.bool:
        return t.to(torch.uint8)
    return t.view(_INT[t.element_size()]) if t.element_size() in _INT else t.view(torch.uint8)


def run(run_a: Callable, run_b: Callable, first=None, n: int = 5) -> str | None:
    """A (``first``, if the caller already ran it), B, then A until A ran ``n`` times. None if every A matched."""
    if n <= 1:
        return None
    a0 = snapshot(run_a() if first is None else first)
    run_b()
    for i in range(1, n):
        d = diff(a0, snapshot(run_a()))
        if d:
            return f"A run {i + 1} of {n}{' (right after B)' if i == 1 else ''} differs from A run 1: {d}"
    return None


def check(run_a, run_b, first=None, *, label: str, n: int | None = None, metric: str | None = None) -> bool:
    """Print and record the result (metric 1.0 / 0.0); True when deterministic."""
    import time

    from models.demos.common.bringup.core import metrics

    n = repeats() if n is None else n
    t0 = time.time()
    err = run(run_a, run_b, first, n)
    dt = time.time() - t0
    if metric:
        metrics.record(metric, 0.0 if err else 1.0)
    print(
        f"{'FAIL' if err else 'ok  '} {label}: determinism A,B,A x{n} in {dt:.1f} s"
        + (f": {err}" if err else ", bit-identical")
    )
    return err is None


def assert_deterministic(run_a, run_b, first=None, *, label: str = "", n: int | None = None) -> None:
    """For pytest: fails with what differs."""
    n = repeats() if n is None else n
    err = run(run_a, run_b, first, n)
    assert err is None, f"{label}: not deterministic (A,B,A x{n}): {err}"
