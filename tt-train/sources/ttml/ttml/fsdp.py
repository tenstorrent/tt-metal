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

import warnings
from enum import Enum
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

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

        # Each entry: ``(parameter, shard_dim)`` where ``parameter`` is the
        # Python :class:`~ttml.modules.parameter.Parameter` wrapper. Hooks
        # always go through ``parameter.tensor`` — for eager-wrapped models
        # that's the autograd tensor at fully_shard time; for lazy-wrapped
        # ones it resolves after :func:`ttml.materialize_module` runs. Order
        # is stable across unshard/reshard so reducing/gathering pairs line up.
        self.managed: List[Tuple[Parameter, int]] = []

        # id(autograd_tensor) -> cached sharded ttnn::Tensor, populated lazily
        # during the first pre-forward and reused across forwards to avoid a
        # host roundtrip on every reshard.
        self._cached_shards: dict[int, Any] = {}

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

        2. Gradient accumulation: if any managed param's ``m_grad`` is
           already initialised (carried over from a previous micro-batch's
           ``backward_post`` reduce-scatter), it's currently in shard shape,
           so we all-gather the shard grad here.

        """
        if self.sharded_state == _ShardedState.SHARDED:
            self._gather_all_params()
            self.sharded_state = _ShardedState.UNSHARDED

        self._gather_accumulated_grads()

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
        for parameter, shard_dim in self.managed:
            param_tensor = parameter.tensor
            current = param_tensor.get_value()
            self._cached_shards[id(param_tensor)] = current
            gathered = ttml.core.distributed.all_gather(current, shard_dim, self.axis_index)
            param_tensor.set_value(gathered)

    def _gather_accumulated_grads(self) -> None:
        """All-gather any managed param ``m_grad`` that survived from a prior
        micro-batch's ``backward_post``.

        Call AFTER ``_gather_all_params`` so ``set_grad`` can shape-check the
        gathered grad against the (now gathered) ``m_value``.

        """
        for parameter, shard_dim in self.managed:
            param_tensor = parameter.tensor
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
        for parameter, _shard_dim in self.managed:
            param_tensor = parameter.tensor
            shard = self._cached_shards.get(id(param_tensor))
            if shard is None:
                raise RuntimeError("FSDP: managed parameter has no cached shard. This should never happen.")
            gathered = param_tensor.get_value()
            param_tensor.set_value(shard)
            ttnn.deallocate(gathered)


# ---------------------------------------------------------------------------
# Auto shard-dim selection
# ---------------------------------------------------------------------------


def _pick_shard_dim_from_shape(
    shape: List[int],
    already_sharded: set,
    axis_index: int,
) -> Optional[int]:
    """Shape-only ``_auto_shard_dim_for_param`` core, shared by eager and lazy paths.

    Same precedence rule as :func:`_auto_shard_dim_for_param`: prefer ``rank-2``
    (the first matmul weight dim on TTML's ``[1,1,O,I]`` convention), fall back
    to ``rank-1`` if ``rank-2`` is already sharded by another mesh axis (e.g. TP)
    or has size 1.
    """
    rank = len(shape)
    if rank < 1:
        return None

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
    """Shape of a Parameter — works for both lazy (``TensorMetadata.shape``)
    and materialized (``autograd_tensor.shape()``) wrappers."""
    inner = parameter.peek_tensor()
    if isinstance(inner, TensorMetadata):
        return list(inner.shape)
    return list(inner.shape())


def _auto_shard_dim_for_param(parameter: Parameter, axis_index: int) -> Optional[int]:
    """Pick a shard dim for ``parameter``, or return ``None`` to skip it.

    Rules (shared between lazy and eager paths):
      1. Prefer ``rank - 2`` (the first matmul weight dim on TTML's
         ``[1,1,O,I]`` convention, where ``O`` is typically large). Fall back
         to ``rank - 1`` if either:
             (a) ``rank - 2`` is already sharded on another mesh axis (e.g. TP), or
             (b) shape on ``rank - 2`` is 1 (e.g. LayerNorm gamma sized ``[1,1,1,F]``).
      2. If no candidate survives those checks, return ``None`` — caller
         skips this parameter with a warning.
      3. Divisibility is enforced by the caller.

    "Already sharded on another mesh axis" is read from
    ``parameter.tensor.tensor_topology().placements()`` in the eager case and
    from ``parameter.peek_tensor().mapper.config().placements`` in the lazy
    case — same logical question, different source depending on whether the
    tensor exists yet.
    """
    inner = parameter.peek_tensor()
    if isinstance(inner, TensorMetadata):
        placements = _placements_from_mapper(inner.mapper)
    else:
        placements = _get_placements(inner)
    already_sharded = _sharded_tensor_dims(placements)
    return _pick_shard_dim_from_shape(_param_shape(parameter), already_sharded, axis_index)


# ---------------------------------------------------------------------------
# Parameter collection: FSDP2 root semantics
# ---------------------------------------------------------------------------


def _collect_root_param_wrappers(module: AbstractModuleBase) -> List[Tuple[str, Parameter]]:
    """Return ``[(dotted_name, Parameter), ...]`` for every Python
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
    ``Parameter`` referenced under multiple attribute names) count once.
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

    seen_ids: set = set()
    out: List[Tuple[str, Parameter]] = []
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
            if id(val) in seen_ids:
                continue
            seen_ids.add(id(val))
            full_name = f"{prefix}.{attr_name}" if prefix else attr_name
            out.append((full_name, val))
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
    allocated already FSDP-sharded — never going through the full-tensor host
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

    n_axes = len(mesh.shape)

    # Walk Python ``Parameter`` wrappers (one collector for both eager and
    # lazy models — see ``_collect_root_param_wrappers``). For each, decide
    # the shard dim from shape + existing placements, then dispatch to either
    # ``_shard_lazy_param`` (rewrites the mapper so materialize allocates
    # already-sharded — required for ~70B models) or ``_shard_eager_param``
    # (host-roundtrips the materialized tensor).
    for rel_name, parameter in _collect_root_param_wrappers(module):
        # Already FSDP-managed somewhere else (e.g. tied weight claimed by
        # an inner block's FSDPState).
        if getattr(parameter, "_fsdp_managed", False):
            continue

        shape = _param_shape(parameter)
        if shard_dim == "auto":
            chosen = _auto_shard_dim_for_param(parameter, axis_index)
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

        if isinstance(parameter.peek_tensor(), TensorMetadata):
            _shard_lazy_param(parameter, chosen, axis_index, n_axes)
        else:
            _shard_eager_param(parameter, chosen, axis_index, axis_size, mesh_axis)

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
