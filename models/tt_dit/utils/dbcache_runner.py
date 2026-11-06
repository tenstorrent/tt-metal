# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The device side of DBCache: traced transformer pieces, persistent buffers and the diff readback.

The decisions are :class:`models.tt_dit.utils.dbcache.DBCacheContext`'s. :class:`DBCache` runs one
transformer per device, all of which cache or compute each step together, as CFG passes on
separate submeshes must: a step takes as long as its slowest device, so skipping on only some of
them saves nothing.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence
from typing import Any, NamedTuple, Protocol, TypeVar

import torch

import ttnn
from models.tt_dit.utils.dbcache import DBCacheConfig, DBCacheContext
from models.tt_dit.utils.tracing import Tracer

Carry = TypeVar("Carry")


class CachedTransformer(Protocol[Carry]):
    """A transformer whose forward is ``epilogue(*run_blocks(*prologue(**inputs), start=0, end=num_blocks))``.

    ``Carry`` is what the blocks pass along besides the hidden state. It must be traceable, i.e.
    tensors and scalars nested in tuples, lists and dicts.

    The hidden state ``run_blocks`` returns must not alias a buffer that later blocks write, such as
    a persistent collective buffer. ``epilogue`` receives the carry that ``run_blocks`` was called
    with, not the one it returned, as skipped blocks return none.
    """

    @property
    def num_blocks(self) -> int:
        ...

    # Takes the inputs as keyword arguments.
    prologue: Callable[..., tuple[ttnn.Tensor, Carry]]

    def run_blocks(self, hidden: ttnn.Tensor, carry: Carry, /, *, start: int, end: int) -> tuple[ttnn.Tensor, Carry]:
        ...

    def epilogue(self, hidden: ttnn.Tensor, carry: Carry, /) -> ttnn.Tensor:
        ...


class ResidualForecaster:
    """Stores the Bn residual of computed steps and predicts it for cached steps.

    ``order == 0`` reuses the last computed residual unchanged (plain DBCache). ``order == k``
    is TaylorSeer with ``k`` finite-difference derivatives, mirroring cache-dit's
    ``TaylorSeerState``::

        update(Y, t):   dY[0] = Y;  dY[i+1] = (dY[i] - dY_prev[i]) / (t - t_prev)
        predict(t):     sum_i dY[i] * (t - t_last)^i / i!

    All buffers are allocated up front (``alloc``) so they never overlap trace scratch memory.
    """

    def __init__(self, order: int, alloc: Callable[[], ttnn.Tensor]) -> None:
        self.order = order
        self.cur = [alloc() for _ in range(order + 1)]
        self.prev = [alloc() for _ in range(order + 1)]
        self.reset()

    def reset(self) -> None:
        self.valid_cur = 0  # number of valid entries in `cur` (0 = nothing cached yet)
        self.valid_prev = 0
        self.last_step: int | None = None

    @property
    def has_data(self) -> bool:
        return self.valid_cur > 0

    @staticmethod
    def steps_to_fill(order: int) -> int:
        """How many computed steps from step 0 form every derivative of ``order``.

        ``update`` forms the first derivative on step 2 and each further one a step later.
        """
        return 1 if order == 0 else order + 2

    def update(self, residual: ttnn.Tensor, step: int) -> None:
        """Record the residual of a computed step and refresh the derivative ladder."""
        self.cur, self.prev = self.prev, self.cur
        self.valid_prev = self.valid_cur
        ttnn.copy(residual, self.cur[0])
        valid = 1
        window = step - self.last_step if self.last_step is not None else 0
        for i in range(self.order):
            # cache-dit only forms the i-th derivative once the previous ladder has it.
            if self.valid_prev <= i or window <= 0 or step <= 1:
                break
            d = ttnn.multiply(ttnn.subtract(self.cur[i], self.prev[i]), 1.0 / window)
            ttnn.copy(d, self.cur[i + 1])
            ttnn.deallocate(d)
            valid = i + 2
        self.valid_cur = valid
        self.last_step = step

    def apply(self, spatial: ttnn.Tensor, step: int) -> ttnn.Tensor:
        """Return ``spatial + predicted_residual(step)``."""
        assert self.has_data and self.last_step is not None
        elapsed = step - self.last_step
        out = ttnn.add(spatial, self.cur[0])
        for i in range(1, self.valid_cur):
            term = ttnn.multiply(self.cur[i], float(elapsed**i) / math.factorial(i))
            out = ttnn.add(out, term)
            ttnn.deallocate(term)
        return out


class DBCache:
    """Runs one transformer per device with DBCache, one denoising step per ``start``/``finish``.

    Each step, ``start`` dispatches the head (``prologue`` and the leading blocks) on each device
    without waiting for it. ``finish`` then reads back the relative difference of every device,
    decides on the largest, and on each device skips or runs the body (the remaining blocks) and
    runs the tail (``epilogue``). The three pieces are traced separately, per input geometry.

    A geometry must first be run untraced, which compiles the programs and allocates the buffers
    kept across steps before any trace is captured.
    """

    def __init__(
        self,
        models: Sequence[CachedTransformer[Any]],
        config: DBCacheConfig,
        *,
        devices: Sequence[ttnn.MeshDevice],
        name: str = "dbcache",
    ) -> None:
        if len(models) != len(devices):
            msg = f"got {len(models)} models for {len(devices)} devices"
            raise ValueError(msg)
        if config.Bn_compute_blocks != 0:
            # The skipped blocks would leave the trailing ones a stale carry.
            msg = f"Bn_compute_blocks must be 0, got {config.Bn_compute_blocks}"
            raise ValueError(msg)
        for model in models:
            if config.Fn_compute_blocks >= model.num_blocks:
                msg = (
                    f"Fn_compute_blocks must be less than the {model.num_blocks} blocks, got {config.Fn_compute_blocks}"
                )
                raise ValueError(msg)

        self._default_config = config
        self._runners = [
            _Runner(m, d, fn_blocks=config.Fn_compute_blocks, taylorseer_order=config.taylorseer_order)
            for m, d in zip(models, devices, strict=True)
        ]
        self._context = DBCacheContext(config, name=name, num_branches=1)
        self.reset()

    def validate(self, config: DBCacheConfig) -> None:
        """Raise if ``config`` cannot replace the default for a run.

        The leading blocks and the TaylorSeer order are fixed, as the traces and the buffers
        depend on them.
        """
        fixed = self._default_config
        fields = ("Fn_compute_blocks", "Bn_compute_blocks", "taylorseer_order")
        changed = [
            f"{f} is {getattr(fixed, f)}, got {getattr(config, f)}"
            for f in fields
            if getattr(config, f) != getattr(fixed, f)
        ]
        if changed:
            msg = f"fixed DBCache fields differ: {'; '.join(changed)}"
            raise ValueError(msg)

    def reset(self, config: DBCacheConfig | None = None) -> None:
        """Start a denoising run, with ``config`` instead of the default for this run only."""
        config = config if config is not None else self._default_config
        self.validate(config)

        self._context.config = config
        self._context.reset()
        self._gate: str | None = None
        for runner in self._runners:
            runner.reset()

    @property
    def config(self) -> DBCacheConfig:
        return self._context.config

    @property
    def cached_steps(self) -> list[int]:
        """The steps of this run that skipped the body, counted from its first step."""
        return list(self._context.branch(0).cached_steps)

    @property
    def diffs(self) -> dict[int, float]:
        """The largest relative difference of each step a difference was needed for."""
        return dict(self._context.branch(0).residual_diffs)

    def inputs(self, index: int) -> dict[int | str, Any]:
        """The inputs of device ``index``'s last ``start``, as ``Tracer.inputs`` gives them."""
        return self._runners[index].inputs

    def start(self, index: int, *, traced: bool, diff_mask: ttnn.Tensor | None = None, **inputs: Any) -> None:
        """Dispatch the head of the step on device ``index``; ``inputs`` are the arguments of ``prologue``.

        ``diff_mask``, which broadcasts to the hidden state, is 1 on the positions the difference
        counts, such as the tokens that are not padding. It must be allocated before any trace is
        captured.
        """
        if self._gate is None:
            self._context.mark_step_begin()
            self._gate = self._context.gate(0)
        self._runners[index].start(traced=traced, measure=self._gate == "dynamic", diff_mask=diff_mask, inputs=inputs)

    def finish(self) -> list[ttnn.Tensor]:
        """Complete the step on every device, returning each device's output."""
        if self._gate is None or not all(r.started for r in self._runners):
            msg = "the step was not started on every device"
            raise RuntimeError(msg)
        context = self._context

        if self._gate == "dynamic":
            cache = context.decide(0, max(r.diff() for r in self._runners))
        else:
            cache = self._gate == "cache"
        self._gate = None

        step = context.current_step
        outputs = [r.finish(cache=cache, step=step) for r in self._runners]
        if cache:
            context.add_cached_step(0)
        else:
            context.add_computed_step(0)
        return outputs


class _Traces(NamedTuple):
    head: Tracer
    body: Tracer
    tail: Tracer


class _Buffers(NamedTuple):
    # The change the leading blocks made on the last computed step.
    prev_fn_residual: ttnn.Tensor
    # The change the remaining blocks made on the computed steps.
    forecaster: ResidualForecaster


class _Pending(NamedTuple):
    """A step between ``_Runner.start`` and ``finish``."""

    traces: _Traces
    traced: bool
    hidden: ttnn.Tensor
    carry: Any
    fn_residual: ttnn.Tensor
    buffers: _Buffers
    # The difference's per-device sums; None when the step needs no difference.
    sums: ttnn.Tensor | None


class _Runner:
    """One transformer on one device."""

    def __init__(
        self, model: CachedTransformer[Any], device: ttnn.MeshDevice, *, fn_blocks: int, taylorseer_order: int
    ) -> None:
        self._model = model
        self._device = device
        self._fn_blocks = fn_blocks
        self._taylorseer_order = taylorseer_order
        self._traces: dict[Hashable, _Traces] = {}
        self._current: _Traces | None = None
        # By the shape of the hidden state. The traces' outputs cannot hold these: replaying one
        # trace may overwrite the outputs of another captured after it.
        self._buffers: dict[Hashable, _Buffers] = {}
        self._pending: _Pending | None = None

    def reset(self) -> None:
        self._pending = None
        for buffers in self._buffers.values():
            buffers.forecaster.reset()

    @property
    def started(self) -> bool:
        return self._pending is not None

    @property
    def inputs(self) -> dict[int | str, Any]:
        assert self._current is not None
        return self._current.head.inputs

    def start(self, *, traced: bool, measure: bool, diff_mask: ttnn.Tensor | None, inputs: dict[str, Any]) -> None:
        if self._pending is not None:
            msg = "the previous step was not finished"
            raise RuntimeError(msg)

        traces = self._traces_for(inputs)
        self._current = traces

        hidden, carry, fn_residual = traces.head(**inputs, traced=traced, tracer_blocking_execution=False)
        buffers = self._buffers_for(fn_residual, traced=traced)
        sums = self._diff_sums(fn_residual, buffers.prev_fn_residual, diff_mask) if measure else None
        self._pending = _Pending(traces, traced, hidden, carry, fn_residual, buffers, sums)

    def diff(self) -> float:
        """The relative difference of the started step, the largest over the batch.

        It is the ratio of the two sums over all devices of the mesh. Devices holding replicas
        scale both sums alike, which leaves the ratio unchanged. A zero reference gives infinity,
        which never caches.
        """
        assert self._pending is not None and self._pending.sums is not None
        host = ttnn.to_torch(self._pending.sums, mesh_composer=ttnn.ConcatMeshToTensor(self._device, dim=0))
        batch_size = self._pending.fn_residual.shape[0]
        change, size = host.to(torch.float64).reshape(-1, batch_size, 2).sum(dim=0).unbind(dim=-1)
        ratios = torch.where(size > 0, change / size, torch.inf)
        return float(ratios.max())

    def finish(self, *, cache: bool, step: int) -> ttnn.Tensor:
        assert self._pending is not None
        traces, traced, hidden, carry, fn_residual, buffers, _ = self._pending
        self._pending = None
        run = {"traced": traced, "tracer_blocking_execution": False}

        if cache:
            hidden = buffers.forecaster.apply(hidden, step)
        else:
            hidden, residual = traces.body(hidden, carry, **run)
            buffers.forecaster.update(residual, step)
            ttnn.copy(fn_residual, buffers.prev_fn_residual)

        return traces.tail(hidden, carry, **run)

    def _traces_for(self, inputs: dict[str, Any]) -> _Traces:
        key = _signature(inputs)
        if key not in self._traces:
            # Untraced calls compile the programs, so no preparation run is needed.
            self._traces[key] = _Traces(
                head=Tracer(self._head, device=self._device, prep_run=False),
                body=Tracer(self._body, device=self._device, prep_run=False),
                tail=Tracer(self._model.epilogue, device=self._device, prep_run=False),
            )
        return self._traces[key]

    def _buffers_for(self, fn_residual: ttnn.Tensor, *, traced: bool) -> _Buffers:
        key = _signature(fn_residual)
        if key not in self._buffers:
            if traced:
                msg = "each input geometry must be run untraced before it is traced"
                raise RuntimeError(msg)
            self._buffers[key] = _Buffers(
                prev_fn_residual=ttnn.clone(fn_residual),
                forecaster=ResidualForecaster(self._taylorseer_order, lambda: ttnn.clone(fn_residual)),
            )
        return self._buffers[key]

    def _head(self, **inputs: Any) -> tuple[ttnn.Tensor, Any, ttnn.Tensor]:
        embedded, carry = self._model.prologue(**inputs)
        hidden, carry = self._model.run_blocks(embedded, carry, start=0, end=self._fn_blocks)
        return hidden, carry, hidden - embedded

    def _body(self, hidden: ttnn.Tensor, carry: Any) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        output, _ = self._model.run_blocks(hidden, carry, start=self._fn_blocks, end=self._model.num_blocks)
        return output, output - hidden

    @staticmethod
    def _diff_sums(fn_residual: ttnn.Tensor, prev_fn_residual: ttnn.Tensor, mask: ttnn.Tensor | None) -> ttnn.Tensor:
        """sum(|fn_residual - prev_fn_residual|) and sum(|prev_fn_residual|) of this device's shard.

        Only the positions where ``mask`` is 1 count. The result has shape [batch_size, 1, ..., 2].
        """
        dims = list(range(1, len(fn_residual.shape)))
        change = ttnn.abs(fn_residual - prev_fn_residual)
        size = ttnn.abs(prev_fn_residual)
        if mask is not None:
            change = change * mask
            size = size * mask
        change = ttnn.sum(change, dim=dims, keepdim=True)
        size = ttnn.sum(size, dim=dims, keepdim=True)
        return ttnn.concat([change, size], dim=-1)


def _signature(value: Any) -> Hashable:
    """The shapes and dtypes of the tensors in ``value``, and its other leaves."""
    if isinstance(value, ttnn.Tensor):
        return (tuple(value.shape), value.dtype)
    if isinstance(value, (tuple, list)):
        return tuple(_signature(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, _signature(v)) for k, v in sorted(value.items()))
    return value
