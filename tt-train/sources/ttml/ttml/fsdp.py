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

    # fully_shard uses axis "fsdp" (default). ttml.sync_gradients averages
    # each gradient over the data axes its parameter is replicated on: an
    # FSDP-sharded param is Shard on "fsdp" (reduce-scattered in
    # backward-post), so it is averaged over "dp" only. The same single call
    # covers pure DDP, pure FSDP, and HSDP.

Contract:
    * Call ``fully_shard`` BEFORE ``create_optimizer`` so optimizer state
      (``zeros_like(param)``) is sized for the sharded weights.
    * The mesh axis must have the same name on every rank; auto-mode defaults
      to ``"fsdp"``, set up via ``ttml.open_device_mesh`` with that axis name.
    * Non-eltwise optimizers like Muon cannot be used with FSDP-managed
      parameters yet (guarded inside the Muon constructor).
"""

from __future__ import annotations

import re
import warnings
from enum import Enum
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Union

import ttnn

import ttml
from ttml.modules import AbstractModuleBase
from ttml.modules.parameter import Parameter, TensorMetadata, replace_lazy_mapper

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class _ShardedState(Enum):
    """Current placement of the managed parameters held by an FSDPState."""

    SHARDED = 0
    UNSHARDED = 1


def _is_fsdp_wrapped_module(module: Any) -> bool:
    return getattr(module, "_fsdp_state", False) and module._fsdp_state is not None


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
        # always go through ``parameter.tensor`` — for eager-wrapped models
        # that's the autograd tensor at fully_shard time; for lazy-wrapped
        # ones it resolves after :func:`ttml.materialize_module` runs. Order
        # is stable across unshard/reshard so reducing/gathering pairs line up.
        self.managed: List[Tuple[Parameter, int]] = []

        # Params kept replicated by ``replicate=``: name -> the patterns that matched it.
        # ``ttml.sync_gradients`` averages their grads.
        self.replicated: dict[str, List[str]] = {}

        # id(autograd_tensor) -> cached sharded ttnn::Tensor, populated lazily
        # during the first pre-forward and reused across forwards to avoid a
        # host roundtrip on every reshard.
        self._cached_shards: dict[int, Any] = {}

        # id(autograd_tensor) -> the shard-shaped gradient accumulated by earlier
        # micro-batches, set aside for the duration of one backward.
        self._accumulated_grads: dict[int, Any] = {}

    # -- managed-param management --------------------------------------------

    def add_managed_param(self, parameter: Parameter, shard_dim: int) -> None:
        self.managed.append((parameter, shard_dim))

    # -- forward hooks -------------------------------------------------------

    def pre_forward(self) -> None:
        """Gather sharded parameters into their full form (idempotent)."""
        if self.sharded_state == _ShardedState.UNSHARDED:
            return
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

        2. Gradient accumulation: a managed param's ``m_grad`` carried over from
           earlier micro-batches (or zeroed by the optimizer) is in shard shape.
           It is set aside, so this backward computes a fresh gathered-shape
           gradient, and ``backward_post`` adds it to the reduce-scattered one.

        """
        if self.sharded_state == _ShardedState.SHARDED:
            self._gather_all_params()
            self.sharded_state = _ShardedState.UNSHARDED

        self._set_aside_accumulated_grads()

    def backward_post(self) -> None:
        """Called after all of the module's internal backward closures have run.

        At this point every managed parameter has its ``.grad`` populated at
        gathered shape. We reduce-scatter each grad into a shard-shaped grad,
        explicitly deallocate the gathered value and gathered grad, and
        restore the sharded m_value (in that order, so ``set_grad``'s shape
        check sees the shard-shaped m_value).
        """
        for parameter, shard_dim in self.managed:
            param_tensor = parameter.tensor
            if not param_tensor.is_grad_initialized():
                # Unused by this backward: back to its shard, with whatever gradient it had accumulated.
                gathered_value = param_tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE)
                param_tensor.set_value(self._cached_shards[id(param_tensor)])
                ttnn.deallocate(gathered_value)
                accumulated = self._accumulated_grads.pop(id(param_tensor), None)
                if accumulated is not None:
                    param_tensor.set_grad(accumulated)
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
            accumulated = self._accumulated_grads.pop(id(param_tensor), None)
            if accumulated is not None:
                reduced = ttnn.add(accumulated, reduced)

            # Swap m_value to the cached shard and free the gathered weight,
            # then set_grad to the reduced shard. Order matters: set_grad
            # shape-checks against m_value, so m_value must be shard-shape
            # before we install a shard-shape grad.
            gathered_value = param_tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE)
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
        ``_set_aside_accumulated_grads`` and ``backward_post``.
        """
        for parameter, shard_dim in self.managed:
            param_tensor = parameter.tensor
            # Cache the shard as stored (NATIVE), so restoring it keeps the parameter's dtype. Gather the bf16 view:
            # forward and backward read it anyway, and it is half the bytes of an fp32 shard.
            self._cached_shards[id(param_tensor)] = param_tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE)
            gathered = ttml.core.distributed.all_gather(param_tensor.get_value(), shard_dim, self.axis_index)
            param_tensor.set_value(gathered)

    def _set_aside_accumulated_grads(self) -> None:
        """Take each managed param's shard-shaped ``m_grad`` out of the tensor until ``backward_post``."""
        for parameter, _shard_dim in self.managed:
            param_tensor = parameter.tensor
            if param_tensor.is_grad_initialized():
                self._accumulated_grads[id(param_tensor)] = param_tensor.get_grad()
                param_tensor.reset_grad()

    def _reshard_all_params(self) -> None:
        """Swap managed params back to their cached shard tensors.

        We grab the current gathered value before swapping so we can
        ``ttnn.deallocate`` it explicitly rather than relying on Python GC —
        this makes the memory-saving property of ``reshard_after_forward``
        actually take effect on the device side at the moment we expect.
        """
        for parameter, _shard_dim in self.managed:
            param_tensor = parameter.tensor
            shard = self._cached_shards.get(id(param_tensor))
            if shard is None:
                raise RuntimeError("FSDP: managed parameter has no cached shard. This should never happen.")
            gathered = param_tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE)
            param_tensor.set_value(shard)
            ttnn.deallocate(gathered)


# ---------------------------------------------------------------------------
# Auto shard-dim selection
# ---------------------------------------------------------------------------


def _is_tile_aligned_shard(shape: List[int], dim: int, axis_size: int) -> bool:
    """Whether ``shape[dim]`` splits over ``axis_size`` devices into whole tiles (else CCLs go composite)."""
    if shape[dim] % axis_size != 0:
        return False
    if dim < len(shape) - 2:
        return True
    return (shape[dim] // axis_size) % ttnn.TILE_SIZE == 0


def _pick_shard_dim_from_shape(
    shape: List[int],
    already_sharded: set,
    axis_index: int,
    axis_size: int,
) -> Optional[int]:
    """Shape-only core of :func:`_auto_shard_dim_for_param`, shared by eager and lazy paths."""
    rank = len(shape)
    if rank < 1:
        return None

    # Placements may count dims from the end (``Shard(-1)``); compare as non-negative dims.
    taken = {d % rank for d in already_sharded}
    candidates = []
    if rank >= 2:
        candidates.append(rank - 2)
    candidates.append(rank - 1)
    candidates = [c for c in candidates if c not in taken and shape[c] != 1]
    if not candidates:
        return None

    for cand in candidates:
        if _is_tile_aligned_shard(shape, cand, axis_size):
            return cand
    for cand in candidates:
        if shape[cand] % axis_size == 0:
            return cand
    return candidates[0]


def _param_shape(parameter: Parameter) -> List[int]:
    """Shape of a Parameter — works for both lazy (``TensorMetadata.shape``)
    and materialized (``autograd_tensor.shape()``) wrappers."""
    inner = parameter.peek_tensor()
    if isinstance(inner, TensorMetadata):
        return list(inner.shape)
    return list(inner.shape())


def _param_placements(parameter: Parameter) -> List[Any]:
    """The parameter's placement on each mesh axis: from its mapper while lazy, its topology once materialized."""
    inner = parameter.peek_tensor()
    if not isinstance(inner, TensorMetadata):
        return ttml.Sharding.from_tensor(inner).placements
    n_axes = len(ttml.mesh().shape)
    placements = list(inner.mapper.config().placements) if inner.mapper is not None else []
    if len(placements) == n_axes:
        return placements
    if not any(isinstance(p, ttnn.PlacementShard) for p in placements):
        return [ttnn.PlacementReplicate()] * n_axes
    raise RuntimeError(
        f"FSDP: a lazy parameter's mapper shards it over the whole mesh at once (placements={placements}); "
        f"create it with a mapper that has one placement per mesh axis (ttnn.create_mesh_mapper / mesh.axis_mapper)."
    )


def _auto_shard_dim_for_param(parameter: Parameter, axis_index: int, axis_size: int) -> Optional[int]:
    """Pick a shard dim for ``parameter``, or return ``None`` to skip it.

    Rules (shared between lazy and eager paths):
      1. Candidates are ``rank - 2`` (second-to-last dim, ``O`` in TTML's ``[1,1,O,I]``),
         then ``rank - 1`` (last dim), minus dims of size 1 or already sharded on
         another mesh axis (e.g. TP).
      2. Prefer whole-tile shards, then an even split, then the first candidate.
      3. No candidate: return ``None`` and the caller skips the parameter.
         Divisibility is enforced by the caller.

    "Already sharded on another mesh axis" is read by :func:`_param_placements`.
    """
    already_sharded = {p.dim for p in _param_placements(parameter) if isinstance(p, ttnn.PlacementShard)}
    return _pick_shard_dim_from_shape(_param_shape(parameter), already_sharded, axis_index, axis_size)


# Warn once per (shape, dim, axis_size): e.g. Qwen3-32B has 128 identical q/k-norm weights.
_warned_misaligned: set = set()


def _warn_misaligned_shard(rel_name: str, shape: List[int], dim: int, axis_size: int) -> None:
    key = (tuple(shape), dim, axis_size)
    if key in _warned_misaligned:
        return
    _warned_misaligned.add(key)
    warnings.warn(
        f"FSDP: {rel_name!r} {shape} has sub-tile shards on dim {dim}; "
        f"its CCLs will use the slower composite path. Reported once per shape, shard dim, and axis size.",
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# Parameter collection: FSDP2 root semantics
# ---------------------------------------------------------------------------


def _collect_root_param_wrappers(module: AbstractModuleBase) -> List[Tuple[List[str], Parameter]]:
    """Return ``[(dotted_names, Parameter), ...]`` for every Python
    :class:`~ttml.modules.parameter.Parameter` owned by ``module`` but NOT by
    any nested ``fully_shard``-wrapped submodule.

    Walking Python ``Parameter`` wrappers (rather than C++-registered autograd
    tensors via ``module.named_parameters()``) lets the same code handle both
    eager and lazy models: lazy params aren't C++-registered until
    :func:`ttml.materialize_module` runs, so ``named_parameters()`` returns
    nothing for them. ``Parameter.tensor`` then resolves to the autograd
    tensor in either case (raising for still-lazy ones, which we never hold
    past materialize time).

    Dedup is by ``Parameter`` identity, so weight-tied references (one
    ``Parameter`` referenced under multiple attribute names) count once;
    ``dotted_names`` lists every such alias visible to ``module``.
    Implicitly assumes every parameter the model owns is exposed via a Python
    ``Parameter`` wrapper — true for every model in ``ttml.models``.
    """
    fsdp_modules: set = set()
    fsdp_prefixes: List[str] = []
    for name, child in module.named_modules():
        if child is module:
            continue
        if _is_fsdp_wrapped_module(child):
            fsdp_modules.add(id(child))
            fsdp_prefixes.append(name + ".")

    names_by_id: dict[int, List[str]] = {}
    out: List[Tuple[List[str], Parameter]] = []
    for prefix, mod in module.named_modules():
        # Skip the wrapped sub-module roots themselves AND any of their
        # descendants. ``fsdp_prefixes`` items end in ".", so descendant paths
        # like "blocks.0.attention" startswith "blocks.0." (excluded) but
        # sibling paths like "blocks.10..." don't startswith "blocks.1.".
        if id(mod) in fsdp_modules:
            continue
        if any(prefix.startswith(p) for p in fsdp_prefixes):
            continue
        for attr_name, val in list(mod.__dict__.items()):
            if not isinstance(val, Parameter):
                continue
            full_name = f"{prefix}.{attr_name}" if prefix else attr_name
            if id(val) in names_by_id:
                names_by_id[id(val)].append(full_name)
                continue
            names_by_id[id(val)] = [full_name]
            out.append((names_by_id[id(val)], val))
    return out


# ---------------------------------------------------------------------------
# Sharding: replicated → local shard
# ---------------------------------------------------------------------------


def _shard_param(parameter: Parameter, shard_dim: int, axis_index: int, mesh_axis: str) -> None:
    """Add ``Shard{shard_dim}`` on the FSDP axis, keeping the parameter's placement on every other axis (e.g. TP).

    A lazy parameter gets a new mapper, so :func:`ttml.materialize_module` allocates it already sharded and never
    holds the full tensor (required for ~70B models). A materialized one keeps each device's slice on device.
    """
    placements = _param_placements(parameter)
    if isinstance(placements[axis_index], ttnn.PlacementShard):
        raise RuntimeError(
            f"Parameter is already sharded on mesh axis {mesh_axis!r} (placements={placements}). FSDP cannot layer "
            f"a second shard on the same axis."
        )
    if isinstance(parameter.peek_tensor(), TensorMetadata):
        placements[axis_index] = ttnn.PlacementShard(shard_dim)
        device = ttml.autograd.AutoContext.get_instance().get_device()
        replace_lazy_mapper(parameter, ttnn.create_mesh_mapper(device, ttnn.MeshMapperConfig(placements)))
        return
    param_tensor = parameter.tensor
    value = param_tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE)
    param_tensor.set_value(ttml.core.distributed.mesh_partition(value, shard_dim, axis_index))


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
    replicate: Sequence[str] = (),
) -> AbstractModuleBase:
    """Wrap ``module`` with torch-style FSDP in place.

    Parameters directly owned by ``module`` (and by its non-FSDP descendants)
    are sharded along ``mesh_axis``; nested ``fully_shard``-ed submodules are
    left alone (they handle their own shard group).

    Args:
        module: An ``AbstractModuleBase`` instance to wrap.
        shard_dim: Tensor dim to shard along, or ``"auto"`` (default).
            Auto picks ``rank-2`` or ``rank-1`` (second-to-last or last dim),
            preferring whole-tile shards and skipping dims of size 1 or already
            sharded by another mesh axis (e.g. TP).
            Parameters with no usable dim, or whose chosen dim isn't divisible by the
            axis size, stay replicated with a warning. Parameters whose shards aren't
            whole tiles are still sharded, with one warning per shape, shard dim
            and axis size.
        mesh_axis: Name of the mesh axis to shard across. Defaults to ``"fsdp"``
        reshard_after_forward: If ``True`` (default), the module's weights
            are resharded between forward and backward (after forward) to keep peak memory
            low; the backward-pre callback re-gathers just in time. If ``False``, weights stay gathered.
            # TODO: Try this on the last block
        replicate: Regex patterns of parameters to keep replicated instead of sharding,
            for params too small to shard into whole tiles. Each is searched for
            (``re.search``) in the parameter's dotted names relative to ``module``; a
            tied parameter matches if any of its names does.
            Raises if a matched parameter is already sharded on
            ``mesh_axis``. The training loop must average their gradients over
            ``mesh_axis`` with ``ttml.sync_gradients``; ``SFTTrainer`` and
            ``GRPOTrainer`` already do.
    Returns:
        ``module`` (modified in place).
    """
    if _is_fsdp_wrapped_module(module):
        raise RuntimeError(f"Module {module.get_name()!r} already wrapped with fully_shard.")
    replicate_res = _compile_replicate_patterns(replicate)

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

    # Walk Python ``Parameter`` wrappers (one collector for both eager and
    # lazy models — see ``_collect_root_param_wrappers``). For each, decide
    # the shard dim from shape + existing placements, then ``_shard_param``.
    for names, parameter in _collect_root_param_wrappers(module):
        rel_name = names[0]
        # Already claimed by another wrapper (e.g. tied weight sharded or
        # kept replicated by an inner block's FSDPState).
        if getattr(parameter, "_fsdp_managed", False) or getattr(parameter, "_fsdp_replicated", False):
            continue
        # Match every alias of a tied parameter, not just the first-seen one.
        matched = _matching_replicate_patterns(names, replicate_res)
        if matched:
            if isinstance(_param_placements(parameter)[axis_index], ttnn.PlacementShard):
                raise RuntimeError(
                    f"replicate: parameter {rel_name!r} (matched by {matched}) is already sharded on mesh "
                    f"axis {mesh_axis!r}, so it cannot be kept replicated on it."
                )
            state.replicated[rel_name] = matched
            # Claims the parameter so outer wrappers don't shard it through another alias.
            parameter._fsdp_replicated = True
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
        if not _is_tile_aligned_shard(shape, chosen, axis_size):
            _warn_misaligned_shard(rel_name, shape, chosen, axis_size)

        _shard_param(parameter, chosen, axis_index, mesh_axis)

        state.add_managed_param(parameter, chosen)
        _mark_fsdp_managed(parameter, chosen, axis_index)

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


def _mark_fsdp_managed(parameter: Parameter, shard_dim: int, axis_index: int) -> None:
    """Tag a Parameter wrapper as FSDP-managed and mirror the marker to its
    underlying autograd tensor.

    The wrapper-side marker is used by ``fully_shard``'s own dedup (so a
    second pass can skip already-managed wrappers, e.g. tied weights). The
    tensor-side marker is what downstream infra — ``ttml.sync_gradients``,
    the Muon guard, ``ttml.fsdp.is_fsdp_managed`` — looks at, since it
    iterates ``model.parameters()`` (autograd tensors).

    For eager wraps the autograd tensor exists already and the mirror runs
    immediately. For lazy wraps the tensor doesn't exist yet, so the mirror
    is deferred via :meth:`Parameter.add_post_materialize_callback`; it fires
    inside :func:`ttml.materialize_module` right after the tensor is bound.
    Either way, ``materialize_module`` itself stays FSDP-agnostic — the
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


def _compile_replicate_patterns(patterns: Sequence[str], what: str = "replicate") -> List[re.Pattern]:
    """Compile ``replicate`` patterns, raising if they aren't a list of usable regex strings.

    A pattern that matches the empty string (e.g. ``""`` or ``"q_norm|"``) matches every name, which
    would keep the whole model replicated, so it is rejected. ``what`` names the setting in errors.
    """
    if isinstance(patterns, (str, bytes)) or not isinstance(patterns, Sequence):
        raise TypeError(f"{what} must be a list of regex patterns, got {patterns!r}")
    compiled = []
    for p in patterns:
        if not isinstance(p, str):
            raise TypeError(f"{what} patterns must be strings, got {p!r}")
        try:
            pattern = re.compile(p)
        except re.error as e:
            raise ValueError(f"{what}: invalid pattern {p!r}: {e}") from e
        if pattern.search(""):
            raise ValueError(f"{what}: pattern {p!r} matches the empty string, so it would match every parameter")
        compiled.append(pattern)
    return compiled


def _matching_replicate_patterns(names: Union[str, Sequence[str]], patterns: Sequence[re.Pattern]) -> List[str]:
    """Return the patterns found by ``re.search`` in ``names`` (one name, or all aliases of a tied parameter)."""
    if isinstance(names, str):
        names = [names]
    return [p.pattern for p in patterns if any(p.search(n) for n in names)]


def replicated_parameters(module: AbstractModuleBase) -> dict[str, List[str]]:
    """Return ``{name: matching patterns}`` for every parameter ``replicate=`` kept replicated under ``module``."""
    out: dict[str, List[str]] = {}
    for prefix, mod in module.named_modules():
        if _is_fsdp_wrapped_module(mod):
            for name, patterns in mod._fsdp_state.replicated.items():
                out[f"{prefix}.{name}" if prefix else name] = list(patterns)
    return out


def unmatched_replicate_patterns(module: AbstractModuleBase, patterns: Sequence[str]) -> List[str]:
    """Return the ``patterns`` that kept no parameter under ``module`` replicated."""
    if not any(_is_fsdp_wrapped_module(mod) for _, mod in module.named_modules()):
        return []
    matched = {p for ps in replicated_parameters(module).values() for p in ps}
    return [p for p in patterns if p not in matched]


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
    "replicated_parameters",
    "unmatched_replicate_patterns",
    "fsdp_axis_of",
]
