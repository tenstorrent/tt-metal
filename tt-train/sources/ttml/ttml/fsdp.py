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
``autograd_callback`` node on its forward output (fires as backward-pre,
re-gathers weights and any accumulated shard grads from a previous
micro-batch) and on its forward input (fires as backward-post,
reduce-scatters grads and reshards).

Intended usage (PyTorch FSDP2-style root):

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

Contract:
    * Call ``fully_shard`` BEFORE ``create_optimizer`` so optimizer state
      (``zeros_like(param)``) is sized for the sharded weights.
    * The mesh axis must have the same name on every rank; auto-mode defaults
      to ``"fsdp"``, set up via ``ttml.open_device_mesh`` with that axis name.
    * Non-eltwise optimizers like Muon cannot be used with FSDP-managed
      parameters yet (guarded inside the Muon constructor).
"""

from __future__ import annotations

import os
import warnings
from enum import Enum
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import ttnn

import ttml
from ttml.modules import AbstractModuleBase

# Set TTML_FSDP_TRACE=1 to print every gather/reshard/reduce-scatter hook with
# the owning module name and managed-param shapes. Useful for diagnosing
# backward re-gather ordering issues with reshard_after_forward=True.
_FSDP_TRACE = os.environ.get("TTML_FSDP_TRACE", "0") == "1"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class _ShardedState(Enum):
    """Current placement of the managed parameters held by an FSDPState."""

    SHARDED = 0
    UNSHARDED = 1


def _is_fsdp_wrapped_module(module: Any) -> bool:
    return getattr(module, "_fsdp_state", False) and module._fsdp_state is not None


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

        try:
            self._name = module.get_name()
        except Exception:  # noqa: BLE001
            self._name = type(module).__name__

        # Each entry: (param_autograd_tensor, shard_dim). We keep order stable
        # across unshard/reshard so reducing/gathering pairs line up.
        self.managed: List[Tuple[Any, int]] = []

        # id(param) -> cached sharded ttnn::Tensor, populated lazily during
        # the first pre-forward and reused across forwards to avoid a host
        # roundtrip on every reshard.
        self._cached_shards: dict[int, Any] = {}

    # -- managed-param management --------------------------------------------

    def add_managed_param(self, autograd_tensor: Any, shard_dim: int) -> None:
        self.managed.append((autograd_tensor, shard_dim))

    def _trace(self, action: str) -> None:
        if not _FSDP_TRACE:
            return
        try:
            shapes = [list(p.shape()) for p, _ in self.managed[:2]]
        except Exception:  # noqa: BLE001
            shapes = []
        print(
            f"[fsdp] {action:<14} module={self._name} state={self.sharded_state.name} "
            f"nparams={len(self.managed)} shapes(head)={shapes}",
            flush=True,
        )

    # -- forward hooks -------------------------------------------------------

    def pre_forward(self) -> None:
        """Gather sharded parameters into their full form (idempotent)."""
        if self.sharded_state == _ShardedState.UNSHARDED:
            return
        self._trace("pre_forward")
        self._gather_all_params()
        self.sharded_state = _ShardedState.UNSHARDED

    def post_forward(self) -> None:
        """Optionally reshard back to local slice, dropping the full copy.

        If reshard_after_forward is False we simply stay UNSHARDED, and
        backward_pre will see that and not re-gather.
        """
        if not self.reshard_after_forward:
            return
        if self.sharded_state == _ShardedState.SHARDED:
            return
        self._trace("post_forward")
        self._reshard_all_params()
        self.sharded_state = _ShardedState.SHARDED

    # -- backward hooks (fired via autograd_callback) ------------------------

    def backward_pre(self) -> None:
        """Called right before the module's internal backward closures run.

        Two responsibilities:

        1. If ``reshard_after_forward`` swapped back to sharded shape between
           forward-post and the backward entering this module, re-gather the
           managed weights. The autograd closures captured a ``weight``
           TensorPtr that at forward time held the gathered value, and their
           ``add_grad`` calls will pass grads of gathered shape into it, so
           ``m_value`` must be gathered for ``add_grad``'s shape check to
           pass. If ``reshard_after_forward`` was False, the module is
           already UNSHARDED and the param gather is skipped.

        2. Gradient accumulation: if any managed param's ``m_grad`` is
           already initialised (carried over from a previous micro-batch's
           ``backward_post`` reduce-scatter), it's currently in shard shape,
           so we all-gather the shard grad here.

        """
        self._trace("backward_pre")
        if self.sharded_state == _ShardedState.SHARDED:
            self._gather_all_params()
            self.sharded_state = _ShardedState.UNSHARDED

        self._gather_accumulated_grads()
        if _FSDP_TRACE:
            print(f"[fsdp] backward_pre   done module={self._name} (closures run next)", flush=True)

    def backward_post(self) -> None:
        """Called after all of the module's internal backward closures have run.

        At this point every managed parameter has its ``.grad`` populated at
        gathered shape. We reduce-scatter each grad into a shard-shaped grad,
        explicitly deallocate the gathered value and gathered grad, and
        restore the sharded m_value (in that order, so ``set_grad``'s shape
        check sees the shard-shaped m_value).
        """
        self._trace("backward_post")
        for param_tensor, shard_dim in self.managed:
            if not param_tensor.is_grad_initialized():
                continue
            # Pull a Python-side handle to the gathered grad before we reduce
            # it; we'll explicitly deallocate it after set_grad to release the
            # full-size buffer on device immediately.
            gathered_grad = param_tensor.get_grad()
            reduced = ttml.core.distributed.reduce_scatter(
                gathered_grad,
                shard_dim,
                self.axis_index,
            )
            # Mean-reduction: reduce_scatter sums one shard's contribution
            # per DP rank; divide by axis_size to get the per-shard average,
            # matching the implicit averaging a DDP all-reduce would do.
            reduced = ttnn.multiply(reduced, 1.0 / float(self.axis_size))

            # Swap m_value to the cached shard and free the gathered weight,
            # then set_grad to the reduced shard. Order matters: set_grad
            # shape-checks against m_value, so m_value must be shard-shape
            # before we install a shard-shape grad.
            gathered_value = param_tensor.get_value()
            shard = self._cached_shards.get(id(param_tensor))
            if shard is not None:
                param_tensor.set_value(shard)
            param_tensor.set_grad(reduced)

            # Explicit deallocate of the now-unreferenced full-size buffers.
            # Without this, Python GC may keep them around for too long
            ttnn.deallocate(gathered_value)
            ttnn.deallocate(gathered_grad)

        self.sharded_state = _ShardedState.SHARDED

    # -- low-level helpers ---------------------------------------------------

    def _gather_all_params(self) -> None:
        """All-gather each managed param's value and swap it into the TensorPtr.

        Does NOT touch ``m_grad`` — grad reshaping lives in
        ``_gather_accumulated_grads`` and ``backward_post``.
        """
        for idx, (param_tensor, shard_dim) in enumerate(self.managed):
            current = param_tensor.get_value()
            self._cached_shards[id(param_tensor)] = current
            if _FSDP_TRACE:
                print(
                    f"[fsdp]   gather[{idx}] module={self._name} shard_dim={shard_dim} "
                    f"shard_shape={list(param_tensor.shape())}",
                    flush=True,
                )
            gathered = ttml.core.distributed.all_gather(current, shard_dim, self.axis_index)
            param_tensor.set_value(gathered)
            if _FSDP_TRACE:
                print(f"[fsdp]   gather[{idx}] done full_shape={list(param_tensor.shape())}", flush=True)

    def _gather_accumulated_grads(self) -> None:
        """All-gather any managed param ``m_grad`` that survived from a prior
        micro-batch's ``backward_post``.

        Call AFTER ``_gather_all_params`` so ``set_grad`` can shape-check the
        gathered grad against the (now gathered) ``m_value``.

        """
        for param_tensor, shard_dim in self.managed:
            if not param_tensor.is_grad_initialized():
                continue
            shard_grad = param_tensor.get_grad()
            gathered_grad = ttml.core.distributed.all_gather(
                shard_grad,
                shard_dim,
                self.axis_index,
            )
            param_tensor.set_grad(gathered_grad)
            ttnn.deallocate(shard_grad)

    def _reshard_all_params(self) -> None:
        """Swap managed params back to their cached shard tensors.

        We grab the current gathered value before swapping so we can
        ``ttnn.deallocate`` it explicitly rather than relying on Python GC —
        this makes the memory-saving property of ``reshard_after_forward``
        actually take effect on the device side at the moment we expect.
        """
        for param_tensor, _shard_dim in self.managed:
            shard = self._cached_shards.get(id(param_tensor))
            if shard is None:
                raise RuntimeError("FSDP: managed parameter has no cached shard. This should never happen.")
            gathered = param_tensor.get_value()
            param_tensor.set_value(shard)
            ttnn.deallocate(gathered)


# ---------------------------------------------------------------------------
# Auto shard-dim selection
# ---------------------------------------------------------------------------


def _auto_shard_dim(autograd_tensor: Any, axis_index: int) -> Optional[int]:
    """Pick a shard dim for this tensor, or return ``None`` to skip it.

    Rules:
      1. Prefer ``rank - 2`` (the first matmul weight dim on TTML's
         ``[1,1,O,I]`` convention, where ``O`` is typically large). Fall back
         to ``rank - 1`` if either:
             (a) ``rank - 2`` is already sharded on another mesh axis (e.g. TP), or
             (b) shape on ``rank - 2`` is 1 (e.g. LayerNorm gamma sized ``[1,1,1,F]``).
      2. If no candidate survives those checks, return ``None`` — caller
         skips this parameter with a warning.
      3. Divisibility is enforced by the caller.
    """
    rank = autograd_tensor.get_rank()
    if rank < 1:
        return None

    placements = _get_placements(autograd_tensor)
    already_sharded = _sharded_tensor_dims(placements)
    shape = list(autograd_tensor.shape())

    candidates = []
    if rank >= 2:
        candidates.append(rank - 2)
    candidates.append(rank - 1)

    for cand in candidates:
        if cand in already_sharded:
            continue
        if shape[cand] == 1:
            continue
        return cand

    return None


# ---------------------------------------------------------------------------
# Parameter collection: FSDP2 root semantics
# ---------------------------------------------------------------------------


def _collect_root_params(module: AbstractModuleBase) -> List[Tuple[str, Any]]:
    """Return ``[(name, autograd_tensor), ...]`` for params owned by ``module``
    but NOT owned by any nested FSDP-wrapped submodules.

    Walks the param tree once via ``named_parameters`` (dot-separated paths
    relative to ``module``) and excludes any parameter whose path descends
    into a sub-module already wrapped with ``fully_shard``. Duplicate
    TensorPtrs (weight tying) are deduped by tensor identity, first owner
    wins.
    """
    # Build dot-separated prefixes of nested FSDP-wrapped modules, e.g.
    # "blocks.0." for an inner block. The trailing "." is required so
    # "blocks.10" doesn't accidentally match the "blocks.1" prefix.
    fsdp_prefixes: List[str] = []
    for name, child in module.named_modules():
        if child is module:
            continue
        if _is_fsdp_wrapped_module(child):
            fsdp_prefixes.append(name + ".")

    seen_ids: set = set()
    out: List[Tuple[str, Any]] = []
    for rel_name, tensor in module.named_parameters():
        if any(rel_name.startswith(pref) for pref in fsdp_prefixes):
            continue
        key = id(tensor)
        if key in seen_ids:
            continue
        seen_ids.add(key)
        out.append((rel_name, tensor))
    return out


# ---------------------------------------------------------------------------
# Sharding: replicated → local shard (stays on device via reduce_scatter trick)
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
        # to recover one canonical copy. (Sharded axes don't need slicing —
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
    # FULL view is fp32, but from_numpy will convert during host→device.
    target_dtype = autograd_tensor.get_value().dtype
    new_at = ttml.autograd.Tensor.from_numpy(full_np, ttnn.Layout.TILE, target_dtype, new_mapper)
    return new_at.get_value()


# ---------------------------------------------------------------------------
# Forward monkey-patching
# ---------------------------------------------------------------------------


def _wrap_forward(module: AbstractModuleBase) -> None:
    """Replace ``module.forward`` with an FSDP-aware version (idempotent guard).

    The wrapped forward:
      1. Calls ``pre_forward`` (unshard managed params).
      2. Wraps the first tensor argument in ``autograd_callback`` whose
         callback is ``backward_post`` — fires late in backward topo order.
      3. Calls the original forward.
      4. Wraps the autograd::Tensor return in another ``autograd_callback``
         whose callback is ``backward_pre`` — fires early in backward.
      5. Calls ``post_forward`` (optional reshard).
    """
    state: FSDPState = module._fsdp_state  # set by fully_shard before us
    original_forward = module.forward

    def new_forward(*args, **kwargs):
        state.pre_forward()  # Unshard managed params.

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

        state.post_forward()
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
            Auto prefers ``rank-2`` and falls back to ``rank-1`` if that's
            already sharded by another mesh axis (e.g. TP). Parameters whose
            chosen shard dim has size 1 are skipped (not sharded) with a warning. Any param
            whose chosen dim is not divisible by the mesh axis size raises an error.
        mesh_axis: Name of the mesh axis to shard across. Defaults to ``"fsdp"``
        reshard_after_forward: If ``True`` (default), the module's weights
            are resharded between forward and backward (after forward) to keep peak memory
            low; the backward-pre callback re-gathers just in time. If ``False``, weights stay gathered.
            # TODO: Try this on the last block
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

    params = _collect_root_params(module)

    for rel_name, param_tensor in params:
        # Already FSDP-managed somewhere else (e.g. tied weight — very rare).
        if getattr(param_tensor, "_fsdp_managed", False):
            # Either attached to *this* state (shouldn't happen due to seen_ids
            # dedup above) or the owning FSDPState is elsewhere — skip.
            continue

        # Resolve shard dim (explicit int, or auto).
        if shard_dim == "auto":
            chosen = _auto_shard_dim(param_tensor, axis_index)
        else:
            rank = param_tensor.get_rank()
            chosen = int(shard_dim)
            if chosen < 0:
                chosen = rank + chosen
            if not 0 <= chosen < rank:
                raise RuntimeError(
                    f"Invalid shard_dim {shard_dim!r} for parameter {rel_name!r} "
                    f"(rank {rank}): normalized dim {chosen} is out of range "
                    f"[0, {rank})."
                )

        shape = list(param_tensor.shape())
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

        # The tensor must be Replicate on the FSDP axis for us to install a
        # Shard there. If it's already a Shard, something else (e.g. TP)
        # sharded it on this axis and we have don't support "double sharding"
        # one the same axis.
        placements = _get_placements(param_tensor)
        if _already_fsdp_sharded(placements, axis_index):
            raise RuntimeError(
                f"Parameter {rel_name!r} is already sharded on mesh axis "
                f"{mesh_axis!r} (placements={placements}). FSDP cannot "
                f"layer a second shard on the same axis."
            )

        sharded = _shard_replicated_param(param_tensor, chosen, axis_index, axis_size)
        param_tensor.set_value(sharded)

        state.add_managed_param(param_tensor, chosen)
        _mark_fsdp_managed(param_tensor, chosen, axis_index)

    _wrap_forward(module)

    # Expose convenience handles.
    def _unshard() -> None:
        state.pre_forward()

    def _reshard() -> None:
        # Manual reshard outside the training loop; safe only when no grads in flight.
        if state.sharded_state == _ShardedState.SHARDED:
            return
        state._reshard_all_params()
        state.sharded_state = _ShardedState.SHARDED

    module.unshard = _unshard  # type: ignore[attr-defined]
    module.reshard = _reshard  # type: ignore[attr-defined]

    return module


def _mark_fsdp_managed(param_tensor: Any, shard_dim: int, axis_index: int) -> None:
    """Tag an autograd::Tensor so other infra (synchronize_gradients, Muon guard,
    serializers) can detect FSDP-managed params."""
    param_tensor._fsdp_managed = True
    param_tensor._fsdp_shard_dim = int(shard_dim)
    param_tensor._fsdp_axis = int(axis_index)


def is_fsdp_managed(param_tensor: Any) -> bool:
    """Return True if ``param_tensor`` was sharded by an ``fully_shard`` call."""
    if getattr(param_tensor, "_fsdp_managed", False):
        return True
    return False


def fsdp_axis_of(param_tensor: Any) -> Optional[int]:
    """Mesh axis index on which this parameter is FSDP-sharded, or None."""
    if hasattr(param_tensor, "_fsdp_axis"):
        return int(param_tensor._fsdp_axis)
    return None


__all__ = [
    "fully_shard",
    "FSDPState",
    "is_fsdp_managed",
    "fsdp_axis_of",
]
