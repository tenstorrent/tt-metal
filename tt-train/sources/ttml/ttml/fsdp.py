# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""torch-style FSDP for TTML.

``fully_shard(module, ...)`` wraps a module in-place so its parameters are
sharded across a mesh axis (by default ``"fsdp"``, kept distinct from the
DDP axis ``"dp"`` so both can coexist on a 2D mesh for hybrid sharded data
parallel). On every forward the module's own parameters are all-gathered,
the original forward runs against the full weights, and on backward the
gradients are reduce-scattered back to shards. Each wrapper has an
``autograd_callback`` node on its forward output (fires as backward-pre and
re-gathers weights) and on its forward input (fires as backward-post,
reduce-scatters grads and reshards).

Intended usage (PyTorch FSDP2-style root), in forward order:

    for block in model.blocks:
        ttml.fsdp.fully_shard(block)
    ttml.fsdp.fully_shard(model)  # root: wraps only params NOT owned by a block

Hybrid FSDP + DDP (HSDP) on a 2D mesh ``[D, F]`` with axes
``("dp", "fsdp")``:

    # fully_shard uses axis "fsdp" (default). sync_gradients all-reduces
    # DDP-replicated shards across the "dp" axis; FSDP-sharded params are
    # skipped on the "fsdp" axis (already reduce-scattered in backward-post)
    # but still reduced on the "dp" axis to average across DP replicas.
    # The per-param axis filter in ttml.sync_gradients is what makes the
    # same single call cover pure DDP, pure FSDP, and HSDP.

Gradient accumulation keeps the accumulated gradient *sharded*: at backward-pre
the shard grad left by the previous micro-batch is set aside, the closures build
a fresh full-shape grad, and backward-post adds the new reduce-scattered shard to
the saved one. No collective is spent on the accumulated gradient.

Overlapping collectives with compute: ``enable_overlap()`` (before any
``fully_shard``) moves every all-gather and reduce-scatter onto a CCL sub-device
fed from a second hardware command queue, prefetching each unit's weights while
its neighbour computes. See :class:`_Overlap` and ``docs/FSDP.md``.

Contract:
    * Call ``fully_shard`` BEFORE ``create_optimizer`` so optimizer state
      (``zeros_like(param)``) is sized for the sharded weights.
    * Wrap units in forward order (the prefetch follows the wrap order).
    * The mesh axis must have the same name on every rank; auto-mode defaults
      to ``"fsdp"``, set up via ``ttml.open_device_mesh`` with that axis name.
    * Non-eltwise optimizers like Muon cannot be used with FSDP-managed
      parameters yet (guarded inside the Muon constructor).
"""

from __future__ import annotations

import contextlib
import math
import os
import warnings
from enum import Enum
from typing import Any, Callable, Iterator, List, Literal, Optional, Sequence, Tuple, Union

import ttnn

import ttml
from ttml.modules import AbstractModuleBase
from ttml.modules.parameter import Parameter, TensorMetadata, replace_lazy_mapper

TILE = 32

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class _ShardedState(Enum):
    """Current placement of the managed parameters held by an FSDPState."""

    SHARDED = 0
    UNSHARDED = 1


def _is_fsdp_wrapped_module(module: Any) -> bool:
    return getattr(module, "_fsdp_state", None) is not None


def _get_placements(autograd_tensor: Any) -> Optional[List[Any]]:
    """Best-effort read of ``tensor_topology().placements()`` from the ttnn tensor.

    Returns ``None`` if the topology is unavailable (e.g. unit mesh, older ttnn
    builds, or an edge case where ``placements()`` throws). Callers must treat
    a ``None`` return as "no useful placement info; assume replicated".
    """
    try:
        tensor = autograd_tensor.get_value()
        topology = tensor.tensor_topology()
        return list(topology.placements())
    except Exception:
        return None


def _is_shard(placement: Any) -> bool:
    return isinstance(placement, ttnn.PlacementShard)


def _sharded_tensor_dims(placements: Optional[List[Any]]) -> set:
    """Return the set of tensor dims already sharded on SOME mesh axis."""
    if placements is None:
        return set()
    return {p.dim for p in placements if _is_shard(p)}


def _already_fsdp_sharded(placements: Optional[List[Any]], axis_index: int) -> bool:
    """True if the param's placement on the FSDP axis is already a Shard."""
    if placements is None or axis_index >= len(placements):
        return False
    return _is_shard(placements[axis_index])


# ---------------------------------------------------------------------------
# Collective / compute overlap
# ---------------------------------------------------------------------------


class SlotSchedule:
    """Which compute release a gather into a reused buffer slot has to wait for.

    Pure bookkeeping, no device calls. Compute *releases* are numbered in the order they are
    recorded; ``release(slot)`` says "every compute program enqueued so far is done with
    ``slot``". A gather that overwrites ``slot`` may start once the release ``lookahead``
    positions *after* the slot's last release has completed. Waiting for the slot's own
    release would order the gather after this device's readers only: a collective writes
    into every device's copy of the slot, and a peer running a little behind may still be
    reading its copy. Every gather is a rendezvous, so devices cannot drift apart by more
    than ``lookahead`` units of compute; with ``2 * lookahead`` slots in rotation the slot
    being overwritten was last read at least ``lookahead`` units ago on every device.

    ``wait_target`` returns ``None`` when the slot has never been released (its buffer is
    new) or the target release is not recorded yet; the caller then waits for all compute
    enqueued so far.
    """

    def __init__(self, lookahead: int) -> None:
        if lookahead < 1:
            raise ValueError("lookahead must be at least 1")
        self.lookahead = lookahead
        self.num_slots = 2 * lookahead
        self._next_seq = 0
        self._last_release: dict = {}

    def release(self, slot: Any) -> int:
        seq = self._next_seq
        self._next_seq += 1
        self._last_release[slot] = seq
        return seq

    def wait_target(self, slot: Any) -> Optional[int]:
        last = self._last_release.get(slot)
        if last is None:
            return None
        target = last + self.lookahead
        return target if target < self._next_seq else None

    def oldest_needed(self) -> int:
        """Releases before this sequence number can no longer be a wait target (every slot's next
        target is after its own last release), so their events may be dropped."""
        return min(self._last_release.values(), default=0)

    def slot_of(self, unit_index: int) -> int:
        return unit_index % self.num_slots


class _Overlap:
    """Process-wide runtime for FSDP collectives on the CCL sub-device.

    Two hardware queues: compute runs on queue 0 inside the compute sub-device, every FSDP
    collective is issued on queue 1 and runs on the CCL sub-device. The dispatcher launches
    programs in order and a launch waits for the previous program on the same sub-device, so
    collectives issued on the compute queue would stall the compute launches behind them; on
    their own queue they never do. Rules:

    * **Gathers write persistent buffers.** All-gather outputs come from a pool keyed by
      ``(slot, parameter position, shape, dtype)``. Units that reshard rotate through
      ``2 * lookahead`` slots (see :class:`SlotSchedule`); the root and units kept gathered
      own a slot each. The host frees and reallocates addresses far ahead of the device, so a
      collective must never write into a freshly allocated buffer without an ordering point.
    * **Prefetch.** ``pre_forward`` of a unit waits for its own gather and issues its
      successor's; ``backward_pre`` does the same for its predecessor. The root is first in
      forward and last in backward, so it prefetches the first and last block respectively.
    * **Reduce-scatters go behind the next prefetch, after a compute drain.** ``backward_post``
      hands its gathered grads to the runtime; the next unit's ``backward_pre`` issues its own
      prefetch first (a gather has a deadline, a reduce-scatter has none) and then the deferred
      reduce-scatters, preceded by an event recorded on the compute queue that the CCL queue
      waits for (the grads are complete, and any buffer the host freed so far is really free).
      They write fresh shard-shaped outputs; the full-shape grads they read are freed two CCL
      barriers later, and the ``1/N`` scaling is applied once at the end of backward.
    * **Events, both ways.** A *CCL barrier* (record on queue 1, wait on queue 0) before
      compute consumes a gather; a *compute drain* (record on queue 0, wait on queue 1)
      before a collective reads compute output or overwrites a buffer compute may still use.
      The root's gather at the start of a step drains as well: the optimizer rewrote the
      shards in place on the compute queue.

    ``TTML_FSDP_SERIALIZE_COLLECTIVES=1`` follows every collective with a CCL barrier -- the
    bisect switch for "is it the overlap?".
    """

    def __init__(self) -> None:
        self.enabled = False
        self.schedule: Optional[SlotSchedule] = None
        self.units: List["FSDPState"] = []  # non-root units, in fully_shard (= forward) order
        self.root: Optional["FSDPState"] = None
        self._device = None
        self._ccl_ids: List[Any] = []
        self._compute_ids: List[Any] = []
        self._compute_cq = ttnn.QueueId(0)
        self._ccl_cq = ttnn.QueueId(1)
        self._pool: dict = {}
        self._release_events: dict = {}  # release seq -> event recorded on the compute queue
        # Buffers a collective read, freed two CCL barriers later: the barrier proves the collective done
        # on *this* device, a peer may still be finishing it, and the host would hand the address to the
        # next fresh allocation -- e.g. a tensor-parallel all-gather on queue 0 writing into every peer.
        self._free_queue: List[List[Any]] = [[], []]
        self._pending_reduce_scatters: List[Tuple[Any, Any, int, int]] = []  # (tensor, full grad, dim, axis)
        self._serialize = os.environ.get("TTML_FSDP_SERIALIZE_COLLECTIVES", "0") == "1"

    def enable(self, *, columns: int, rows: int, lookahead: int) -> None:
        if self.enabled:
            raise RuntimeError("ttml.fsdp overlap is already enabled")
        ctx = ttml.autograd.AutoContext.get_instance()
        if not ctx.has_ccl_sub_device():
            ctx.enable_ccl_sub_device(columns, rows)
        self._device = ctx.get_device()
        self._ccl_ids = [ttnn.SubDeviceId(ctx.ccl_sub_device_index())]
        self._compute_ids = [ttnn.SubDeviceId(ctx.compute_sub_device_index())]
        self.schedule = SlotSchedule(lookahead)
        self.enabled = True

    # -- registration ----------------------------------------------------------

    def register(self, state: "FSDPState", is_root: bool) -> None:
        if is_root:
            self.root = state
            state.slot = "root"
            return
        state.unit_index = len(self.units)
        self.units.append(state)
        # A unit that stays gathered keeps its slot for the whole step; a resharding one rotates.
        if state.reshard_after_forward:
            state.slot = self.schedule.slot_of(state.unit_index)
        else:
            state.slot = f"unit{state.unit_index}"

    def neighbour(self, state: "FSDPState", direction: int) -> Optional["FSDPState"]:
        """The unit that runs next in the current pass (+1 forward, -1 backward)."""
        if not self.units:
            return None
        if state is self.root:
            return self.units[0] if direction > 0 else self.units[-1]
        i = state.unit_index + direction
        return self.units[i] if 0 <= i < len(self.units) else None

    def finisher(self) -> Optional["FSDPState"]:
        """The unit whose backward_post is the last FSDP hook of a backward pass."""
        if self.root is not None:
            return self.root
        return self.units[0] if self.units else None

    # -- issuing -----------------------------------------------------------------

    def issue(self, op: Callable[[], Any]) -> Any:
        """Run ``op`` (which enqueues one collective) on the CCL queue."""
        with ttnn.command_queue(self._ccl_cq):
            result = op()
        if self._serialize:
            self.ccl_barrier()
        return result

    def ccl_barrier(self) -> None:
        """Compute must not pass this point until every collective issued so far has finished.
        Frees the buffers whose last reader that proves complete."""
        event = ttnn.record_event(self._device, cq_id=self._ccl_cq, sub_device_ids=self._ccl_ids)
        ttnn.wait_for_event(self._compute_cq, event)
        for tensor in self._free_queue[0]:
            ttnn.deallocate(tensor)
        self._free_queue = [self._free_queue[1], []]

    def compute_drain(self) -> None:
        """Collectives issued after this point start only once every compute program enqueued so
        far has finished."""
        event = ttnn.record_event(self._device, cq_id=self._compute_cq, sub_device_ids=self._compute_ids)
        ttnn.wait_for_event(self._ccl_cq, event)

    def release(self, slot: Any) -> None:
        """The compute enqueued so far was the last reader of ``slot`` on this device."""
        event = ttnn.record_event(self._device, cq_id=self._compute_cq, sub_device_ids=self._compute_ids)
        self._release_events[self.schedule.release(slot)] = event
        oldest = self.schedule.oldest_needed()
        for stale in [s for s in self._release_events if s < oldest]:
            del self._release_events[stale]

    def wait_slot_free(self, slot: Any) -> None:
        """Before a gather overwrites ``slot`` on every device (see :class:`SlotSchedule`)."""
        target = self.schedule.wait_target(slot)
        if target is None:
            self.compute_drain()
        else:
            ttnn.wait_for_event(self._ccl_cq, self._release_events[target])

    def free_after_barrier(self, tensor: Any) -> None:
        """Deallocate ``tensor`` (read by a collective issued so far) once two barriers have passed."""
        self._free_queue[1].append(tensor)

    # -- buffers -------------------------------------------------------------------

    def gather_buffer(self, slot: Any, param_index: int, shard: Any, shard_dim: int, axis_size: int) -> Any:
        """Persistent full-shape output for the ``param_index``-th parameter of the unit in ``slot``.
        Keyed by position, not only shape: same-shaped weights of one unit (q/out, w1/w3) must not share."""
        shape = list(shard.shape)
        shape[shard_dim] *= axis_size
        key = (slot, param_index, tuple(shape), shard.dtype)
        buf = self._pool.get(key)
        if buf is None:
            buf = ttnn.empty(shape, shard.dtype, shard.layout, self._device, shard.memory_config())
            # The new buffer may sit at an address the host freed a moment ago while a program on some
            # device still uses it (the no-grad forward of gradient checkpointing frees activations
            # eagerly). The gather into it is a cross-device write, so a local drain is not enough: wait
            # for every device to go idle. Happens once per pool buffer, in the first step.
            ttnn.synchronize_device(self._device)
            self._pool[key] = buf
        return buf

    # -- reduce-scatters -------------------------------------------------------------

    def defer_reduce_scatter(self, tensor: Any, full_grad: Any, shard_dim: int, axis_index: int) -> None:
        """Queue a reduce-scatter of ``full_grad`` (the gathered grad of ``tensor``) for the next flush."""
        self._pending_reduce_scatters.append((tensor, full_grad, shard_dim, axis_index))

    def flush_reduce_scatters(self) -> None:
        """Issue the deferred reduce-scatters, after a compute drain (their inputs are the grads the
        compute queue has just produced). Called right after the next unit's prefetch has been issued:
        the CCL queue is in order and a gather has a deadline (the compute waiting for it) while a
        reduce-scatter has none until the end of backward, so the gather must go first. Issuing the
        reduce-scatters in ``backward_post`` instead measured 6 % slower on TinyLlama at 1 sample per
        device -- every block's prefetch arrived late behind the previous block's reduce-scatters."""
        if not self._pending_reduce_scatters:
            return
        self.compute_drain()
        for tensor, full_grad, shard_dim, axis_index in self._pending_reduce_scatters:
            tensor.set_grad(self.issue(lambda: ttml.core.distributed.reduce_scatter(full_grad, shard_dim, axis_index)))
            self.free_after_barrier(full_grad)  # freed once a barrier has proved the reduce-scatter done
        self._pending_reduce_scatters.clear()

    # -- end of backward -------------------------------------------------------------

    def finish_backward(self) -> None:
        """Flush, wait for every collective, free what they read, then scale and merge the shard grads."""
        self.flush_reduce_scatters()
        self.ccl_barrier()
        self.ccl_barrier()  # second pass of the free queue: nothing of this backward outlives the step
        for unit in ([self.root] if self.root is not None else []) + self.units:
            unit._finalize_grads()


_overlap = _Overlap()


def enable_overlap(*, columns: int = 1, rows: int = 0, lookahead: int = 2) -> None:
    """Run FSDP collectives on a CCL sub-device from a second command queue, overlapped with compute.

    Call after ``ttml.open_device_mesh(..., num_command_queues=2)`` and before any ``fully_shard``.
    The CCL sub-device is the rightmost ``columns`` columns or the bottom ``rows`` rows of every
    chip's Tensix grid (exactly one of them non-zero). The CCL kernels use 1, 2 or 4 workers per
    link depending on the cores available -- on a 12x10 Blackhole grid one column (10 cores) gives
    1 worker, one row (12 cores) 2 (a 2.3x faster all-gather), two columns (20 cores) 4. The
    shape also changes the compute grid every op sizes itself to, and that matters more: one row
    leaves 12x9, measured 11 % *slower* than no overlap for TinyLlama at 6 samples per device while
    one column (11x10) gained 3 %, and 6 % slower than the column on 32 chips; the row only edges
    the column at 1 sample per device on 8 chips. Reserving cores costs that fraction of compute, so
    overlap pays when the exposed collectives are a larger share of the step than the reserved cores
    are of the grid (few tokens per device): 3-10 % on TinyLlama, 3 % on Llama-8B TP4 x FSDP8.

    ``lookahead`` is how many units the collective queue may run ahead of compute; the gather
    pool holds ``2 * lookahead`` full-shape copies of a unit's weights (see :class:`SlotSchedule`).
    """
    _overlap.enable(columns=columns, rows=rows, lookahead=lookahead)


def overlap_enabled() -> bool:
    """True if FSDP collectives run on the CCL sub-device and are prefetched (see :func:`enable_overlap`)."""
    return _overlap.enabled


# ---------------------------------------------------------------------------
# FSDPState: per-module state + hooks
# ---------------------------------------------------------------------------


class FSDPState:
    """State holder for a single FSDP shard group.

    One instance lives on the wrapped module as ``module._fsdp_state``. It owns
    the list of parameters that belong to this shard group (i.e. parameters
    registered under this module but NOT already managed by a nested
    FSDPState), and runs pre/post forward/backward hooks against them.
    """

    def __init__(
        self,
        module: AbstractModuleBase,
        mesh_axis: str,
        reshard_after_forward: bool,
    ) -> None:
        self.module = module
        self.mesh_axis_name = mesh_axis

        mesh = ttml.mesh()
        self.axis_index = mesh.axis_index(mesh_axis)
        self.axis_size = mesh.axis_size(mesh_axis)

        self.reshard_after_forward = reshard_after_forward
        self.sharded_state = _ShardedState.SHARDED

        # Each entry: ``(parameter, shard_dim)`` where ``parameter`` is the
        # Python :class:`~ttml.modules.parameter.Parameter` wrapper. Hooks
        # always go through ``parameter.tensor`` -- for eager-wrapped models
        # that's the autograd tensor at fully_shard time; for lazy-wrapped
        # ones it resolves after :func:`ttml.materialize_module` runs. Order
        # is stable across unshard/reshard so reducing/gathering pairs line up.
        self.managed: List[Tuple[Parameter, int]] = []

        # id(autograd_tensor) -> the sharded ttnn::Tensor, recorded at gather time and swapped
        # back at reshard so no host roundtrip is needed.
        self._shards: dict[int, Any] = {}
        # param index -> shard grad set aside at backward_pre (gradient accumulation).
        self._accumulated: dict[int, Any] = {}

        # Overlap bookkeeping (see _Overlap).
        self.overlap = _overlap.enabled
        self.slot: Any = None
        self.unit_index: Optional[int] = None
        self.gather_issued = False  # a gather into this unit's buffers is enqueued, maybe in flight

    # -- managed-param management --------------------------------------------

    def add_managed_param(self, parameter: Parameter, shard_dim: int) -> None:
        self.managed.append((parameter, shard_dim))

    # -- forward hooks -------------------------------------------------------

    def pre_forward(self, prefetch_next: bool = True) -> None:
        """Gather sharded parameters into their full form (idempotent) and, in overlap mode,
        prefetch the next unit's weights. ``prefetch_next`` is False for a gradient-checkpoint
        recompute: backward is walking the units in reverse and prefetches for itself."""
        self._unshard()
        if prefetch_next:
            self._prefetch(+1)

    def post_forward(self, keep_unsharded: bool = False) -> None:
        """Reshard back to the local slice unless the weights are wanted for the backward that
        follows: ``reshard_after_forward=False``, or ``keep_unsharded`` for a gradient-checkpoint
        recompute whose backward closures run right after this forward (resharding here would be
        undone by an immediate re-gather in ``backward_pre``)."""
        if self.reshard_after_forward and not keep_unsharded:
            self._reshard()

    # -- backward hooks (fired via autograd_callback) ------------------------

    def backward_pre(self) -> None:
        """Right before the module's internal backward closures run.

        The closures captured a ``weight`` TensorPtr that held the gathered value at forward
        time and will ``add_grad`` gathered-shape grads into it, so ``m_value`` must be gathered
        again for the shape check to pass. A shard grad carried over from a previous micro-batch
        is set aside so the closures start a fresh full-shape grad; ``backward_post`` adds the two
        shards back together.
        """
        self._unshard()
        for index, (parameter, _dim) in enumerate(self.managed):
            tensor = parameter.tensor
            if tensor.is_grad_initialized():
                self._accumulated[index] = tensor.get_grad()
                tensor.reset_grad()
        self._prefetch(-1)
        if self.overlap:
            _overlap.flush_reduce_scatters()  # the previous unit's, behind this unit's prefetch

    def backward_post(self) -> None:
        """After all of the module's internal backward closures have run.

        Every managed parameter with a grad has it at gathered shape: reduce-scatter it into a
        shard-shaped grad, swap the value back to the shard (``set_grad`` shape-checks against
        ``m_value``, so the value goes first), and free the full-size buffers. In overlap mode the
        reduce-scatters are handed to the runtime and issued at the next unit's ``backward_pre``,
        behind that unit's prefetch (see :meth:`_Overlap.flush_reduce_scatters`).
        """
        for index, (parameter, shard_dim) in enumerate(self.managed):
            tensor = parameter.tensor
            gathered_value = tensor.get_value()
            full_grad = tensor.get_grad() if tensor.is_grad_initialized() else None
            tensor.set_value(self._shards[id(tensor)])
            if self.overlap:
                if full_grad is not None:
                    _overlap.defer_reduce_scatter(tensor, full_grad, shard_dim, self.axis_index)
                continue
            ttnn.deallocate(gathered_value)
            if full_grad is None:
                continue
            reduced = ttml.core.distributed.reduce_scatter(full_grad, shard_dim, self.axis_index)
            tensor.set_grad(self._merge_accumulated(index, ttnn.multiply(reduced, 1.0 / float(self.axis_size))))
            ttnn.deallocate(reduced)
            ttnn.deallocate(full_grad)
        self.sharded_state = _ShardedState.SHARDED
        if self.overlap:
            _overlap.release(self.slot)  # this unit's backward closures were the slot's last readers
            if self is _overlap.finisher():
                _overlap.finish_backward()

    # -- low-level helpers ---------------------------------------------------

    def _issue(self, op: Callable[[], Any]) -> Any:
        return _overlap.issue(op) if self.overlap else op()

    def _unshard(self) -> None:
        if self.sharded_state == _ShardedState.UNSHARDED:
            return
        if not self.gather_issued:
            self._issue_gather()
        if self.overlap:
            _overlap.ccl_barrier()
        self.gather_issued = False
        self.sharded_state = _ShardedState.UNSHARDED

    def _prefetch(self, direction: int) -> None:
        if not self.overlap:
            return
        nxt = _overlap.neighbour(self, direction)
        if nxt is not None and nxt.sharded_state == _ShardedState.SHARDED and not nxt.gather_issued:
            nxt._issue_gather()

    def _issue_gather(self) -> None:
        """All-gather each managed param's value and swap it into the TensorPtr. In overlap mode the
        value points at the persistent output as soon as the gather is *issued*; nothing reads it
        before ``_unshard``'s CCL barrier."""
        if self.overlap:
            # The root gathers first in a step and has to wait for the optimizer, which rewrote the
            # shards in place on the compute queue; the CCL queue is in order, so that drain also
            # covers every gather issued after it.
            if self is _overlap.root:
                _overlap.compute_drain()
            else:
                _overlap.wait_slot_free(self.slot)
        for index, (parameter, shard_dim) in enumerate(self.managed):
            tensor = parameter.tensor
            shard = tensor.get_value()
            self._shards[id(tensor)] = shard
            out = _overlap.gather_buffer(self.slot, index, shard, shard_dim, self.axis_size) if self.overlap else None
            tensor.set_value(
                self._issue(lambda: ttml.core.distributed.all_gather(shard, shard_dim, self.axis_index, out))
            )
        self.gather_issued = True

    def _reshard(self) -> None:
        """Swap managed params back to their shards, freeing the gathered copies (or, in overlap mode,
        releasing the persistent slot for the next gather)."""
        if self.sharded_state == _ShardedState.SHARDED:
            return
        for parameter, _shard_dim in self.managed:
            tensor = parameter.tensor
            gathered = tensor.get_value()
            tensor.set_value(self._shards[id(tensor)])
            if not self.overlap:
                ttnn.deallocate(gathered)
        self.sharded_state = _ShardedState.SHARDED
        if self.overlap:
            _overlap.release(self.slot)

    def _merge_accumulated(self, index: int, shard_grad: Any) -> Any:
        """Add the shard grad set aside at ``backward_pre`` (gradient accumulation), if any."""
        previous = self._accumulated.pop(index, None)
        if previous is None:
            return shard_grad
        total = ttnn.add(previous, shard_grad)
        ttnn.deallocate(previous)
        ttnn.deallocate(shard_grad)
        return total

    def _finalize_grads(self) -> None:
        """Overlap mode, end of backward: the reduce-scatter outputs are complete; apply the mean
        scaling and merge any accumulated shard."""
        for index, (parameter, _dim) in enumerate(self.managed):
            tensor = parameter.tensor
            if not tensor.is_grad_initialized():
                continue
            reduced = tensor.get_grad()
            tensor.set_grad(self._merge_accumulated(index, ttnn.multiply(reduced, 1.0 / float(self.axis_size))))
            ttnn.deallocate(reduced)


# ---------------------------------------------------------------------------
# Auto shard-dim selection
# ---------------------------------------------------------------------------


def _shard_is_tile_aligned(dim_size: int, axis_size: int, tile_size: int = TILE) -> bool:
    """True if slicing ``dim_size`` into ``axis_size`` equal shards leaves each shard tile-aligned."""
    return dim_size % axis_size == 0 and (dim_size // axis_size) % tile_size == 0


def _pick_shard_dim_from_shape(
    shape: List[int],
    already_sharded: set,
    axis_index: int,
    axis_size: Optional[int] = None,
    tile_size: int = TILE,
) -> Optional[int]:
    """Shape-only ``_auto_shard_dim_for_param`` core, shared by eager and lazy paths.

    Candidates are ``rank-2`` (the first matmul weight dim on TTML's ``[1,1,O,I]``
    convention) then ``rank-1``. A candidate is dropped if it is already sharded by
    another mesh axis (e.g. TP), has size 1, or (when ``axis_size`` is given) is not
    divisible by ``axis_size``.

    Among the surviving candidates, prefer one whose per-rank shard is **tile-aligned**
    (``shape[dim] // axis_size`` is a multiple of ``tile_size``). The CCL ops FSDP relies on
    (``all_gather_async`` / ``reduce_scatter_minimal_async``) take a composite
    split/pad/concat fallback when the gathered-or-scattered dim of a TILE tensor is
    padded, which on a 32-chip Blackhole galaxy measured 4-20x slower per call and
    10-50x more host time than the direct kernels (see tools/profiling/fsdp_bench).
    Only if no candidate is tile-aligned fall back to the first survivor and let the
    caller warn.
    """
    rank = len(shape)
    if rank < 1:
        return None

    candidates = []
    if rank >= 2:
        candidates.append(rank - 2)
    candidates.append(rank - 1)

    survivors = []
    for cand in candidates:
        if cand in already_sharded:
            continue
        if shape[cand] == 1:
            continue
        if axis_size is not None and shape[cand] % axis_size != 0:
            continue
        survivors.append(cand)

    if not survivors:
        return None
    if axis_size is not None:
        for cand in survivors:
            if _shard_is_tile_aligned(shape[cand], axis_size, tile_size):
                return cand
    return survivors[0]


def _placements_from_mapper(mapper: Any) -> Optional[List[Any]]:
    """Best-effort read of the placements list backing a ``CppTensorToMesh``.

    The mapper is the ``ttnn.CppTensorToMesh`` returned by
    ``ttml.mesh().axis_mapper(...)`` / ``ttnn.create_mesh_mapper(...)``. C++
    binds ``TensorToMesh::config()`` so we can introspect the original
    ``MeshMapperConfig.placements`` without a side channel on
    :class:`TensorMetadata`. Returns ``None`` if ``mapper`` is ``None`` or the
    accessor isn't available (older ttnn build before ``config()`` was bound).
    """
    if mapper is None:
        return None
    try:
        return list(mapper.config().placements)
    except Exception:
        return None


def _param_shape(parameter: Parameter) -> List[int]:
    """Shape of a Parameter -- works for both lazy (``TensorMetadata.shape``)
    and materialized (``autograd_tensor.shape()``) wrappers."""
    inner = parameter.peek_tensor()
    if isinstance(inner, TensorMetadata):
        return list(inner.shape)
    return list(inner.shape())


def _auto_shard_dim_for_param(parameter: Parameter, axis_index: int, axis_size: int) -> Optional[int]:
    """Pick a shard dim for ``parameter``, or return ``None`` to skip it.

    Rules (shared between lazy and eager paths):
      1. Candidates are ``rank - 2`` (the first matmul weight dim on TTML's
         ``[1,1,O,I]`` convention, where ``O`` is typically large) then ``rank - 1``.
         A candidate is dropped if it is already sharded on another mesh axis
         (e.g. TP), has size 1 (e.g. LayerNorm gamma sized ``[1,1,1,F]``), or is
         not divisible by ``axis_size``.
      2. Prefer a candidate whose per-rank shard is tile-aligned (multiple of 32);
         a misaligned shard forces the CCL ops onto a much slower composite path.
      3. If no candidate survives, return ``None`` -- caller skips this parameter
         with a warning.

    "Already sharded on another mesh axis" is read from
    ``parameter.tensor.tensor_topology().placements()`` in the eager case and
    from ``parameter.peek_tensor().mapper.config().placements`` in the lazy
    case -- same logical question, different source depending on whether the
    tensor exists yet.
    """
    inner = parameter.peek_tensor()
    if isinstance(inner, TensorMetadata):
        placements = _placements_from_mapper(inner.mapper)
    else:
        placements = _get_placements(inner)
    already_sharded = _sharded_tensor_dims(placements)
    return _pick_shard_dim_from_shape(_param_shape(parameter), already_sharded, axis_index, axis_size)


# ---------------------------------------------------------------------------
# Parameter collection: FSDP2 root semantics
# ---------------------------------------------------------------------------


def _unique_parameters(module: AbstractModuleBase) -> Iterator[Tuple[str, Parameter]]:
    """``(dotted_name, Parameter)`` for every Python Parameter wrapper under ``module``, tied
    references counted once. Walking wrappers (not C++-registered autograd tensors) covers lazy
    models, whose ``named_parameters()`` is empty until :func:`ttml.materialize_module` runs."""
    seen: set = set()
    for prefix, mod in module.named_modules():
        for attr_name, val in list(mod.__dict__.items()):
            if isinstance(val, Parameter) and id(val) not in seen:
                seen.add(id(val))
                yield (f"{prefix}.{attr_name}" if prefix else attr_name), val


def _collect_root_param_wrappers(module: AbstractModuleBase) -> List[Tuple[str, Parameter]]:
    """Return ``[(dotted_name, Parameter), ...]`` for every Python
    :class:`~ttml.modules.parameter.Parameter` owned by ``module`` but NOT by
    any nested ``fully_shard``-wrapped submodule.

    Implicitly assumes every parameter the model owns is exposed via a Python
    ``Parameter`` wrapper -- true for every model in ``ttml.models``.
    """
    fsdp_prefixes = [
        name + "." for name, child in module.named_modules() if child is not module and _is_fsdp_wrapped_module(child)
    ]

    out: List[Tuple[str, Parameter]] = []
    for name, parameter in _unique_parameters(module):
        # Skip parameters of the wrapped sub-modules AND of their descendants. ``fsdp_prefixes``
        # items end in ".", so "blocks.0.attention.w" startswith "blocks.0." (excluded) but
        # "blocks.10.w" does not startswith "blocks.1.".
        if any(name.startswith(p) for p in fsdp_prefixes):
            continue
        out.append((name, parameter))
    return out


# ---------------------------------------------------------------------------
# Sharding: replicated -> local shard
# ---------------------------------------------------------------------------


def _shard_replicated_param(
    autograd_tensor: Any,
    shard_dim: int,
    axis_index: int,
    axis_size: int,
) -> Any:
    """Reshape a tensor's distribution to add Shard{shard_dim} on the FSDP axis,
    preserving any existing sharding on other mesh axes (e.g. TP).

    Approach: aggregate the tensor's data to host via a multi-axis composer,
    then redistribute with a new mapper whose placements are the original
    placements with the FSDP axis swapped from Replicate to Shard{shard_dim}.
    # TODO: This is very slow. Need lazy init to shard once on creation,
    # or at least a device-side slice-per-device op.

    Returns the new ttnn tensor (the caller is responsible for swapping it
    into the autograd::Tensor via ``set_value``).
    """
    device = ttml.autograd.AutoContext.get_instance().get_device()
    mesh_shape = ttml.mesh().shape
    n_axes = len(mesh_shape)

    placements = _get_placements(autograd_tensor)
    if placements is None or len(placements) != n_axes:
        # Tensor has no usable topology metadata (or it's stale). Default-
        # initialised tensors from ttml.init.* are fully replicated, so this
        # is the right fallback for the common path.
        placements = [ttnn.PlacementReplicate()] * n_axes

    # Build a multi-axis composer that gathers the tensor onto host:
    #   - For each Shard{tdim} axis, concat along tdim (gather the shards).
    #   - For each Replicate axis, all shards along it are identical; the
    #     composer doesn't have a "Replicate" placement, just a concat dim
    #     per axis. We pick any tensor dim NOT already taken by another
    #     axis and slice the resulting duplicates off afterwards.
    # MeshComposerConfig requires every concat dim to be UNIQUE, so we
    # build the assignment carefully: sharded axes claim their tdim first,
    # then replicate axes get successive unused dims.
    rank = autograd_tensor.get_rank()
    original_shape = list(autograd_tensor.shape())
    sharded_tensor_dims = {p.dim for p in placements if isinstance(p, ttnn.PlacementShard)}
    available_for_replicate = [d for d in range(rank) if d not in sharded_tensor_dims]

    composer_dims: List[int] = []
    replicate_dims_used: List[Tuple[int, int]] = []  # (tensor_dim, original_size)
    next_replicate_idx = 0
    for placement in placements:
        if isinstance(placement, ttnn.PlacementShard):
            composer_dims.append(placement.dim)
        else:
            if next_replicate_idx >= len(available_for_replicate):
                raise RuntimeError(
                    f"FSDP: tensor of rank {rank} has too many Replicate mesh "
                    f"axes to assign unique compose dims (placements={placements}, "
                    f"mesh_shape={list(mesh_shape)}). The fix is to add a "
                    f"slice-per-device op so we don't need to roundtrip via host."
                )
            d = available_for_replicate[next_replicate_idx]
            composer_dims.append(d)
            replicate_dims_used.append((d, original_shape[d]))
            next_replicate_idx += 1

    composer = ttnn.create_mesh_composer(device, ttnn.MeshComposerConfig(composer_dims))
    full_np = autograd_tensor.to_numpy(composer=composer)

    if replicate_dims_used:
        # Each replicate axis stacked the tensor mesh_shape[axis] times along
        # its assigned dim. Slice each stacking dim back to the original size
        # to recover one canonical copy. (Sharded axes don't need slicing --
        # concat across N shards along a dim of (size/N) gives back size.)
        slicer: List[Any] = [slice(None)] * full_np.ndim
        for tensor_dim, original_size in replicate_dims_used:
            slicer[tensor_dim] = slice(0, original_size)
        full_np = full_np[tuple(slicer)]

    # New placements: keep existing sharding on every other axis (e.g. TP),
    # install Shard{shard_dim} on the FSDP axis. fully_shard's pre-call
    # check already verified that ``placements[axis_index]`` was Replicate.
    new_placements = list(placements)
    if new_placements[axis_index] != ttnn.PlacementReplicate():
        raise RuntimeError(
            f"FSDP: tensor of rank {rank} is already sharded on axis {axis_index} (placements={placements})."
        )
    new_placements[axis_index] = ttnn.PlacementShard(shard_dim)
    new_mapper = ttnn.create_mesh_mapper(device, ttnn.MeshMapperConfig(new_placements))

    # Redistribute. Target dtype = the parameter's half-precision dtype
    # (typically bfloat16); to_numpy returned float32 because the autograd
    # FULL view is fp32, but from_numpy will convert during host->device.
    target_dtype = autograd_tensor.get_value().dtype
    new_at = ttml.autograd.Tensor.from_numpy(full_np, ttnn.Layout.TILE, target_dtype, new_mapper)
    return new_at.get_value()


def _shard_lazy_param(
    parameter: Parameter,
    shard_dim: int,
    axis_index: int,
    n_axes: int,
) -> None:
    """Install ``Shard{shard_dim}`` on the FSDP axis of a *lazy* parameter's mapper.

    Replaces the parameter's :class:`TensorMetadata` mapper with a fresh one
    whose placements equal the existing placements (e.g. TP's
    ``Shard{2}/Shard{3}`` on the ``"tp"`` axis, read back via
    ``mapper.config().placements``) plus ``Shard{shard_dim}`` on the FSDP
    axis. When :func:`ttml.materialize_module` later runs, the parameter is
    allocated already FSDP-sharded -- never going through the full-tensor host
    roundtrip the eager :func:`_shard_replicated_param` does, and never
    holding the gathered tensor in DRAM.
    """
    meta = parameter.peek_tensor()
    assert isinstance(meta, TensorMetadata)

    # Existing placements (from TP-aware modules' mappers) or all-Replicate
    # fallback when the param has no explicit mapper.
    existing = _placements_from_mapper(meta.mapper)
    if existing is None or len(existing) != n_axes:
        new_placements: List[Any] = [ttnn.PlacementReplicate()] * n_axes
    else:
        new_placements = list(existing)

    if isinstance(new_placements[axis_index], ttnn.PlacementShard):
        raise RuntimeError(
            f"FSDP: lazy parameter is already sharded on mesh axis {axis_index} "
            f"(placements={new_placements}). FSDP cannot layer a second shard "
            f"on the same axis."
        )
    new_placements[axis_index] = ttnn.PlacementShard(shard_dim)

    device = ttml.autograd.AutoContext.get_instance().get_device()
    new_mapper = ttnn.create_mesh_mapper(device, ttnn.MeshMapperConfig(new_placements))
    replace_lazy_mapper(parameter, new_mapper)


def _shard_eager_param(
    parameter: Parameter,
    shard_dim: int,
    axis_index: int,
    axis_size: int,
    mesh_axis_name: str,
) -> None:
    """Eager-path counterpart to :func:`_shard_lazy_param`.

    Host-roundtrips ``parameter.tensor`` via :func:`_shard_replicated_param`
    and swaps the result into the autograd tensor in place. Errors if the
    tensor is already sharded on the FSDP axis (e.g. if some upstream code
    layered TP onto this same axis).
    """
    param_tensor = parameter.tensor
    placements = _get_placements(param_tensor)
    if _already_fsdp_sharded(placements, axis_index):
        raise RuntimeError(
            f"Parameter is already sharded on mesh axis {mesh_axis_name!r} "
            f"(placements={placements}). FSDP cannot layer a second shard on "
            f"the same axis."
        )
    sharded = _shard_replicated_param(param_tensor, shard_dim, axis_index, axis_size)
    param_tensor.set_value(sharded)


# ---------------------------------------------------------------------------
# Forward monkey-patching
# ---------------------------------------------------------------------------


def _wrap_forward(module: AbstractModuleBase) -> None:
    """Replace ``module.forward`` with an FSDP-aware version (idempotent guard).

    The wrapped forward:
      1. Calls ``pre_forward`` (unshard managed params).
      2. Wraps the first tensor argument in ``autograd_callback`` whose
         callback is ``backward_post`` -- fires late in backward topo order.
      3. Calls the original forward.
      4. Wraps the autograd::Tensor return in another ``autograd_callback``
         whose callback is ``backward_pre`` -- fires early in backward.
      5. Calls ``post_forward`` (optional reshard).
    """
    state: FSDPState = module._fsdp_state  # set by fully_shard before us
    original_forward = module.forward

    def new_forward(*args, **kwargs):
        ctx = ttml.autograd.AutoContext.get_instance()
        # A forward that runs *inside* a backward pass with gradients enabled is a
        # gradient-checkpointing recompute (see memory_efficient_runner): the module's
        # backward closures run immediately after it, so keep the gathered weights
        # instead of resharding and re-gathering them.
        is_recompute = ctx.get_gradient_mode() == ttml.autograd.GradMode.ENABLED and ctx.is_backward_in_progress()

        state.pre_forward(prefetch_next=not is_recompute)  # Unshard managed params.

        # TODO: This is quite hacky. We should probably redesign the autograd graph
        # to support this properly.
        # Wrap the first autograd.Tensor we find; the backward_post callback
        # will fire AFTER all the module's internal backward closures have run.
        new_args = list(args)
        wrapped_input = False
        for idx, arg in enumerate(new_args):
            if isinstance(arg, ttml.autograd.Tensor):
                new_args[idx] = ttml.autograd.callback(arg, state.backward_post)
                wrapped_input = True
                break
        if not wrapped_input:
            for key, value in kwargs.items():
                if isinstance(value, ttml.autograd.Tensor):
                    kwargs[key] = ttml.autograd.callback(value, state.backward_post)
                    wrapped_input = True
                    break

        if not wrapped_input:
            raise RuntimeError("No input tensor found to wrap with backward_post callback.")

        out = original_forward(*new_args, **kwargs)

        # Wrap the output so backward_pre fires BEFORE internal bwd closures.
        if isinstance(out, ttml.autograd.Tensor):
            out = ttml.autograd.callback(out, state.backward_pre)
        elif isinstance(out, tuple):
            out = tuple(
                ttml.autograd.callback(o, state.backward_pre) if isinstance(o, ttml.autograd.Tensor) else o for o in out
            )

        state.post_forward(keep_unsharded=is_recompute)
        return out

    module.forward = new_forward


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fully_shard(
    module: AbstractModuleBase,
    shard_dim: Union[int, Literal["auto"]] = "auto",
    mesh_axis: str = "fsdp",
    reshard_after_forward: bool = True,
) -> AbstractModuleBase:
    """Wrap ``module`` with torch-style FSDP in place.

    Parameters directly owned by ``module`` (and by its non-FSDP descendants)
    are sharded along ``mesh_axis``; nested ``fully_shard``-ed submodules are
    left alone (they handle their own shard group).

    Args:
        module: An ``AbstractModuleBase`` instance to wrap.
        shard_dim: Tensor dim to shard along, or ``"auto"`` (default).
            Auto considers ``rank-2`` then ``rank-1``, dropping dims that are already
            sharded by another mesh axis (e.g. TP), have size 1, or are not divisible by
            the axis size, and prefers the one whose per-rank shard is tile-aligned (a
            multiple of 32) since misaligned shards fall onto a slow composite CCL path.
            Parameters with no usable dim are skipped (left replicated) with a warning.
        mesh_axis: Name of the mesh axis to shard across. Defaults to ``"fsdp"``
        reshard_after_forward: If ``True`` (default), the module's weights
            are resharded between forward and backward (after forward) to keep peak memory
            low; the backward-pre callback re-gathers just in time. If ``False``, weights stay
            gathered, which saves that all-gather (see :func:`blocks_to_keep_gathered`).
    Returns:
        ``module`` (modified in place).
    """
    if _is_fsdp_wrapped_module(module):
        raise RuntimeError(f"Module {module.get_name()!r} already wrapped with fully_shard.")

    mesh = ttml.mesh()
    if not mesh.has_axis(mesh_axis):
        raise RuntimeError(f"Mesh has no axis named {mesh_axis!r}; available: {mesh.axis_names}. ")

    axis_size = mesh.axis_size(mesh_axis)
    axis_index = mesh.axis_index(mesh_axis)
    if axis_size <= 1:
        # No sharding possible/needed. Don't wrap; behaves like an identity.
        warnings.warn(
            f"fully_shard called with axis {mesh_axis!r} of size {axis_size}; " "leaving module unchanged.",
            stacklevel=2,
        )
        return module

    state = FSDPState(module, mesh_axis=mesh_axis, reshard_after_forward=reshard_after_forward)
    module._fsdp_state = state

    n_axes = len(mesh.shape)

    # Walk Python ``Parameter`` wrappers (one collector for both eager and
    # lazy models -- see ``_collect_root_param_wrappers``). For each, decide
    # the shard dim from shape + existing placements, then dispatch to either
    # ``_shard_lazy_param`` (rewrites the mapper so materialize allocates
    # already-sharded -- required for ~70B models) or ``_shard_eager_param``
    # (host-roundtrips the materialized tensor).
    for rel_name, parameter in _collect_root_param_wrappers(module):
        # Already FSDP-managed somewhere else (e.g. tied weight claimed by
        # an inner block's FSDPState).
        if getattr(parameter, "_fsdp_managed", False):
            continue

        shape = _param_shape(parameter)
        if shard_dim == "auto":
            chosen = _auto_shard_dim_for_param(parameter, axis_index, axis_size)
        else:
            rank = len(shape)
            chosen = int(shard_dim)
            if chosen < 0:
                chosen = rank + chosen
            if not 0 <= chosen < rank:
                raise RuntimeError(
                    f"Invalid shard_dim {shard_dim!r} for parameter {rel_name!r} "
                    f"(rank {rank}): normalized dim {chosen} is out of range "
                    f"[0, {rank})."
                )

        if chosen is None:
            warnings.warn(
                f"Skipping FSDP sharding for parameter {rel_name!r} "
                f"(shape {shape}): no suitable shard dim could be auto-selected.",
                stacklevel=2,
            )
            continue
        if shape[chosen] % axis_size != 0:
            warnings.warn(
                f"Skipping FSDP sharding for parameter {rel_name!r} "
                f"(shape {shape}): chosen dim {chosen} has size {shape[chosen]} "
                f"which is not divisible by FSDP axis size {axis_size}.",
                stacklevel=2,
            )
            continue
        if not _shard_is_tile_aligned(shape[chosen], axis_size):
            message = (
                f"FSDP parameter {rel_name!r} (shape {shape}) sharded on dim {chosen}: the per-rank "
                f"shard of {shape[chosen] // axis_size} is not a multiple of {TILE}, so its all-gather / "
                f"reduce-scatter take the slow composite CCL path. Pad this dim to a multiple of "
                f"{TILE * axis_size} or pass an explicit shard_dim."
            )
            if state.overlap:
                # The composite path allocates its own output, which the overlap's persistent
                # buffers cannot tolerate (all_gather would fail later, less clearly).
                raise RuntimeError(message + " Overlap mode requires tile-aligned shards.")
            warnings.warn(message, stacklevel=2)

        if isinstance(parameter.peek_tensor(), TensorMetadata):
            _shard_lazy_param(parameter, chosen, axis_index, n_axes)
        else:
            _shard_eager_param(parameter, chosen, axis_index, axis_size, mesh_axis)

        state.add_managed_param(parameter, chosen)
        _mark_fsdp_managed(parameter, chosen, axis_index)

    if state.overlap:
        # Root = the wrapper that owns other wrappers (FSDP2 root semantics): gathered around the
        # whole step, it prefetches the first block and drives the end-of-backward finish.
        is_root = any(_is_fsdp_wrapped_module(child) for _n, child in module.named_modules() if child is not module)
        _overlap.register(state, is_root)

    _wrap_forward(module)

    # Expose convenience handles (manual use outside the training loop, e.g. inference).
    module.unshard = lambda: state.pre_forward(prefetch_next=False)  # type: ignore[attr-defined]
    module.reshard = state._reshard  # type: ignore[attr-defined]

    return module


def _mark_fsdp_managed(parameter: Parameter, shard_dim: int, axis_index: int) -> None:
    """Tag a Parameter wrapper as FSDP-managed and mirror the marker to its
    underlying autograd tensor.

    The wrapper-side marker is used by ``fully_shard``'s own dedup (so a
    second pass can skip already-managed wrappers, e.g. tied weights). The
    tensor-side marker is what downstream infra -- ``ttml.sync_gradients``,
    the Muon guard, ``ttml.fsdp.is_fsdp_managed`` -- looks at, since it
    iterates ``model.parameters()`` (autograd tensors).

    For eager wraps the autograd tensor exists already and the mirror runs
    immediately. For lazy wraps the tensor doesn't exist yet, so the mirror
    is deferred via :meth:`Parameter.add_post_materialize_callback`; it fires
    inside :func:`ttml.materialize_module` right after the tensor is bound.
    Either way, ``materialize_module`` itself stays FSDP-agnostic -- the
    FSDP-specific logic lives entirely in this file.
    """
    parameter._fsdp_managed = True
    parameter._fsdp_shard_dim = int(shard_dim)
    parameter._fsdp_axis = int(axis_index)

    def _mirror_to_tensor(p: Parameter) -> None:
        t = p.tensor
        t._fsdp_managed = True
        t._fsdp_shard_dim = int(shard_dim)
        t._fsdp_axis = int(axis_index)

    parameter.add_post_materialize_callback(_mirror_to_tensor)


def is_fsdp_managed(param_tensor: Any) -> bool:
    """Return True if ``param_tensor`` was sharded by an ``fully_shard`` call."""
    return bool(getattr(param_tensor, "_fsdp_managed", False))


def fsdp_axis_of(param_tensor: Any) -> Optional[int]:
    """Mesh axis index on which this parameter is FSDP-sharded, or None."""
    if hasattr(param_tensor, "_fsdp_axis"):
        return int(param_tensor._fsdp_axis)
    return None


def fsdp_units(module: AbstractModuleBase) -> List[AbstractModuleBase]:
    """Every ``fully_shard``-wrapped module under (and including) ``module``, root first."""
    units: List[AbstractModuleBase] = []
    if _is_fsdp_wrapped_module(module):
        units.append(module)
    for _name, child in module.named_modules():
        if child is not module and _is_fsdp_wrapped_module(child):
            units.append(child)
    return units


# ---------------------------------------------------------------------------
# Memory accounting and wrapping policy helpers
# ---------------------------------------------------------------------------

_ELEMENT_BYTES = {
    ttnn.DataType.BFLOAT16: 2,
    ttnn.DataType.FLOAT32: 4,
    ttnn.DataType.UINT32: 4,
    ttnn.DataType.INT32: 4,
    ttnn.DataType.UINT16: 2,
    ttnn.DataType.UINT8: 1,
    ttnn.DataType.BFLOAT8_B: 1,
    ttnn.DataType.BFLOAT4_B: 1,
}


def parameter_bytes(module: AbstractModuleBase, bytes_per_element: int = 2) -> int:
    """Bytes of the *unsharded* parameters under ``module`` (bf16 unless told otherwise), from the
    Parameter shapes -- works before materialization and before ``fully_shard``."""
    return sum(math.prod(_param_shape(p)) * bytes_per_element for _name, p in _unique_parameters(module))


def unsharded_bytes(module: AbstractModuleBase) -> int:
    """Bytes the gathered (full) weights of all FSDP units under ``module`` occupy per device."""
    total = 0
    for unit in fsdp_units(module):
        state: FSDPState = unit._fsdp_state
        for parameter, _dim in state.managed:
            value = parameter.tensor.get_value()
            numel = math.prod(int(d) for d in value.padded_shape)
            total += numel * _ELEMENT_BYTES.get(value.dtype, 2) * state.axis_size
    return total


def blocks_to_keep_gathered(blocks: Sequence[AbstractModuleBase], budget_gib: float = 0.0) -> set:
    """Indices of the trailing ``blocks`` to wrap with ``reshard_after_forward=False``.

    A block kept gathered between its forward and its backward skips one of FSDP's three
    collectives per parameter per step, at the cost of holding its full weights for that
    span. The last block is always kept: its backward starts right after the forward, so
    resharding it would be undone by an immediate re-gather. ``budget_gib`` extends that to
    as many preceding blocks as fit in that many GiB of unsharded bf16 weights per device --
    counted from the end because the last blocks' gathered weights live the shortest. Pass
    ``math.inf`` to keep every block. Sizes come from the Parameter shapes, so this works
    before materialization; call it before ``fully_shard``.
    """
    if not blocks:
        return set()
    keep = {len(blocks) - 1}
    budget = budget_gib * 2**30
    used = 0
    for i in range(len(blocks) - 2, -1, -1):
        used += parameter_bytes(blocks[i])
        if used > budget:
            break
        keep.add(i)
    return keep


def _free_dram_bytes() -> Optional[int]:
    """Free DRAM per device (sum over banks), or ``None`` if the query is unavailable."""
    try:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        view = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM)
        return int(view.total_bytes_free_per_bank) * int(view.num_banks)
    except Exception:  # noqa: BLE001
        return None


@contextlib.contextmanager
def unshard_for_inference(module: AbstractModuleBase, max_fraction_of_free_dram: float = 0.5) -> Iterator[bool]:
    """Keep every FSDP unit under ``module`` gathered for the duration of the block.

    Inside a ``fully_shard``-ed model each forward all-gathers all weights and frees them
    again. That is right for training (one forward per step) but not for autoregressive
    generation, where the whole model is re-gathered for every generated token (measured
    on TinyLlama 1.1B: 30 ms per forward, 60 % of a decode step; ~1 s per token for a 32B
    model).

    On entry, if the unsharded weights fit in ``max_fraction_of_free_dram`` of the free
    DRAM, every unit is gathered once and ``reshard_after_forward`` is suspended so the
    forwards inside the block reuse the gathered weights. On exit the units are resharded
    and their settings restored. If the weights do not fit (or the memory query fails)
    nothing changes and per-forward gathering continues. Yields whether the weights were
    kept gathered. Only for gradient-free use: a backward inside the block would see the
    suspended reshard as ``reshard_after_forward=False`` (correct, but memory-hungry).
    """
    units = fsdp_units(module)
    if not units:
        yield False
        return

    needed = unsharded_bytes(module)
    free = _free_dram_bytes()
    if free is None or needed > max_fraction_of_free_dram * free:
        warnings.warn(
            f"unshard_for_inference: gathered weights need {needed / 2**30:.2f} GiB per device but only "
            f"{(free or 0) / 2**30:.2f} GiB DRAM is free (limit {max_fraction_of_free_dram:.0%}); "
            f"falling back to per-forward all-gather.",
            stacklevel=3,
        )
        yield False
        return

    saved = [unit._fsdp_state.reshard_after_forward for unit in units]
    try:
        for unit in units:
            unit._fsdp_state.reshard_after_forward = False
            unit.unshard()
        yield True
    finally:
        for unit, previous in zip(units, saved):
            unit._fsdp_state.reshard_after_forward = previous
            if previous:
                unit.reshard()


__all__ = [
    "fully_shard",
    "FSDPState",
    "SlotSchedule",
    "enable_overlap",
    "overlap_enabled",
    "fsdp_units",
    "parameter_bytes",
    "unsharded_bytes",
    "blocks_to_keep_gathered",
    "unshard_for_inference",
    "is_fsdp_managed",
    "fsdp_axis_of",
]
