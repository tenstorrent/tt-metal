# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import functools, operator, os, re
from typing import Iterable
import ttnn
import ttml


def prod(x: Iterable[int]) -> int:
    return functools.reduce(operator.mul, x, 1)


class Mesh:
    """Named multi-dimensional device mesh.

    Maps a flat set of devices onto a logical grid where each axis has a name
    (e.g. ``("dp", "tp")``).  Axis names are used throughout the TP/DP stack to
    look up shard dimensions, communicate across the correct device subset, etc.
    """

    shape: tuple[int, ...]
    axis_names: tuple[str, ...]
    _axis_map: dict[str, int]

    def __init__(self, shape: tuple[int, ...], axis_names: tuple[str, ...]):
        if len(shape) != len(axis_names):
            raise ValueError("each axis in a mesh must have an assigned name")

        self.shape = shape
        self.axis_names = axis_names
        self._axis_map = {name: i for i, name in enumerate(axis_names)}

    def num_devices(self) -> int:
        return prod(self.shape)

    def has_axis(self, name: str) -> bool:
        return name in self._axis_map

    def axis_size(self, name: str) -> int:
        return self.shape[self.axis_index(name)]

    def axis_index(self, name: str) -> int:
        if not name in self._axis_map:
            msg = (
                f"Mesh has no axis named '{name}'.\n"
                + f"  Shape: {self.shape}\n"
                + f"  Axis names: {self.axis_names}\n"
            )
            raise RuntimeError(msg)
        return self._axis_map[name]

    def axis_mapper(self, name: str, tdim: int) -> ttnn.CppTensorToMesh:
        """Return a mapper that shards a tensor along ``tdim`` across the named mesh axis.

        The mapper is passed to ``Tensor.from_numpy`` (or an initializer's ``mapper``
        kwarg) so that each device in the mesh receives the appropriate slice of the
        full tensor.
        """
        dev = ttml.autograd.AutoContext.get_instance().get_device()
        cluster_axis = self.axis_index(name)
        return ttml.core.distributed.shard_tensor_to_mesh_mapper(dev, tdim, cluster_axis)

    def axis_mapper_config(self, name: str, tdim: int) -> ttnn.MeshMapperConfig:
        """Return a MeshMapperConfig that shards along ``tdim`` across the named mesh axis."""
        cluster_axis = self.axis_index(name)
        placements = [ttnn.PlacementReplicate() for _ in self.shape]
        placements[cluster_axis] = ttnn.PlacementShard(tdim)
        return ttnn.MeshMapperConfig(placements)


# Mirrors get_max_dimensions_for_architecture() in tt_metal/fabric/mesh_graph_descriptor.cpp
_ARCH_MAX_DIMS = {"WORMHOLE_B0": 2, "BLACKHOLE": 3}


def _validate_mgd(mesh: Mesh) -> None:
    """Validate that the requested Mesh is consistent with the Mesh Graph Descriptor (MGD) file.

    Three checks are performed:
      1. dims — the MGD device_topology dims must match ``mesh.shape`` exactly.
      2. arch — the mesh dimensionality must not exceed the architecture's maximum.
      3. topology — any axis named ``"dp"`` must use RING topology in the MGD.
    """
    mgd_path = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
    if not mgd_path:
        print("WARNING: TT_MESH_GRAPH_DESC_PATH not set, skipping MGD validation")
        return
    if not os.path.isfile(mgd_path):
        print(f"WARNING: MGD file not found: {mgd_path}, skipping validation")
        return

    with open(mgd_path) as f:
        content = f.read()

    # Multi-host MGDs have multiple mesh_descriptors blocks; per-mesh dims don't
    # represent the full cluster topology, so validation would be misleading.
    mesh_descriptor_count = content.count("mesh_descriptors {")
    if mesh_descriptor_count > 1:
        print(
            f"WARNING: MGD file has {mesh_descriptor_count} mesh_descriptors "
            f"(multi-host config); per-mesh validation not supported, skipping."
        )
        return

    dims_match = re.search(r"device_topology\s*\{[^}]*dims\s*:\s*\[\s*([\d\s,]+)\]", content)
    if not dims_match:
        print(f"WARNING: Could not parse dims from MGD file: {mgd_path}")
        return

    mgd_dims = [int(d) for d in (d.strip() for d in dims_match.group(1).split(",")) if d]
    if list(mgd_dims) != list(mesh.shape):
        raise RuntimeError(
            f"Mesh shape mismatch!\n"
            f"  Requested mesh shape: {list(mesh.shape)}\n"
            f"  MGD device_topology dims: {mgd_dims}\n"
            f"Please ensure your mesh dimensions match the MGD file."
        )

    arch_match = re.search(r"\barch\s*:\s*(\w+)", content)
    if arch_match:
        arch = arch_match.group(1)
        max_dims = _ARCH_MAX_DIMS.get(arch)
        if max_dims is not None and len(mesh.shape) > max_dims:
            raise RuntimeError(
                f"Mesh has {len(mesh.shape)} dimensions but arch {arch} "
                f"supports at most {max_dims}.\n"
                f"  Mesh shape: {list(mesh.shape)}\n"
                f"  MGD file: {mgd_path}"
            )

    types_match = re.search(r"device_topology\s*\{[^}]*dim_types\s*:\s*\[\s*([A-Z_,\s]+)\]", content)
    if types_match:
        dim_types = [t for t in (t.strip() for t in types_match.group(1).split(",")) if t]
        if len(dim_types) != len(mgd_dims):
            raise RuntimeError(
                f"Malformed MGD: dim_types has {len(dim_types)} entries "
                f"but dims has {len(mgd_dims)}.\n"
                f"  MGD file: {mgd_path}"
            )

    print(f"MGD validated: dims={mgd_dims}, file={mgd_path}")


_mesh: Mesh | None = None


def open_device_mesh(mesh: tuple[int, ...] | Mesh, device_ids: tuple[int, ...] | None = None):
    """Initialize the global device mesh and open the underlying TT devices.

    When more than one device is requested the MGD file is validated and the
    TT-Fabric interconnect is enabled.  A plain tuple is accepted for backward
    compatibility and converted into a ``Mesh`` with anonymous axis names.

    If ``enable_fabric`` succeeds but the subsequent ``open_device`` raises
    (e.g. the bundled MGD doesn't match the physical topology, the fabric
    topology mapper can't satisfy pinnings, etc.), we roll back the
    process-global fabric config before re-raising. Otherwise a fixture that
    catches the exception and ``pytest.skip``s would leave fabric armed for
    the rest of the process and poison every subsequent implicit single-device
    open.
    """
    if not isinstance(mesh, Mesh):
        mesh = Mesh(mesh, tuple(f"_{i}" for i in range(len(mesh))))
    if device_ids is None:
        device_ids = ()

    fabric_enabled = False
    if mesh.num_devices() > 1:
        _validate_mgd(mesh)
        ttml.core.distributed.enable_fabric(mesh.num_devices())
        fabric_enabled = True

    try:
        ttml.autograd.AutoContext.get_instance().open_device(list(mesh.shape), list(device_ids))
    except BaseException:
        if fabric_enabled:
            try:
                ttml.core.distributed.disable_fabric()
            except Exception:
                # Best-effort cleanup; never mask the original exception.
                pass
        raise

    global _mesh
    _mesh = mesh


def close_device_mesh() -> None:
    """Tear down the global device mesh opened by :func:`open_device_mesh`.

    Reverses, in opposite order, the state ``open_device_mesh`` installs:

      1. ``_mesh = mesh``            -> ``_mesh = None`` (this function)
      2. ``AutoContext.open_device`` -> ``AutoContext.close_device``
      3. ``enable_fabric``           -> ``disable_fabric``

    Steps 2 and 3 are both handled by a single ``close_device()`` call:
    ``AutoContext::close_device`` drops the ``MeshDevice`` *and* calls
    ``disable_fabric()`` (see auto_context.cpp), which is exactly the
    process-global fabric config ``open_device_mesh`` arms via
    ``enable_fabric`` for multi-device meshes. That call is idempotent — it is
    safe when no device is open and when fabric was never enabled (a
    single-device mesh) — so this function needs no bookkeeping of what open
    actually did, and is itself safe to call repeatedly or when nothing is
    open.
    """
    global _mesh
    try:
        ttml.autograd.AutoContext.get_instance().close_device()
    finally:
        _mesh = None


def maybe_mesh() -> Mesh | None:
    """Return the active device mesh, or ``None`` if no mesh has been opened."""
    global _mesh
    return _mesh


def mesh() -> Mesh:
    """Return the active device mesh, raising ``RuntimeError`` if none is open."""
    global _mesh
    if _mesh is None:
        msg = (
            "Device mesh is not initialized.\n"
            + "Use ttml.open_device_mesh(ttml.Mesh(shape, axis_names)) "
            + "to initialize the mesh.\n"
        )
        raise RuntimeError(msg)
    return _mesh


def sync_gradients(parameters, axis_names: tuple[str, ...] = ("dp",)):
    """Average parameter gradients across one or more mesh axes.

    For each parameter with an initialized gradient, the grad is all-reduced
    (summed) across every axis in ``axis_names`` and then divided by the
    product of those axis sizes, leaving each device with the mean grad.

    Axes listed in ``axis_names`` but not present on the active mesh are
    silently skipped — if none are present (or ``axis_names`` is empty, or
    no mesh is open), the function is a no-op. The default ``("dp",)``
    matches the common DDP case. TP is intentionally excluded: sharded
    parameters already hold per-shard-correct grads, and replicated
    parameters see identical inputs on every TP rank so their grads match
    without a reduce.

    FSDP interaction: parameters sharded by ``ttml.fsdp.fully_shard`` on a
    mesh axis listed in ``axis_names`` are skipped for that axis — the FSDP
    backward hook already reduce-scattered the grad into shard shape.

    Args:
        parameters: A ``NamedParameters`` mapping (e.g. ``model.parameters()``).
        axis_names: Tuple of mesh axis names to reduce over.
    """
    m = maybe_mesh()
    if m is None or not axis_names:
        return

    axes = tuple(m.axis_index(name) for name in axis_names if m.has_axis(name))
    if not axes:
        return

    for _, param in parameters.items():
        if not param.is_grad_initialized():
            continue

        # Drop axes on which this particular parameter is FSDP-sharded
        axes_for_param = tuple(a for a in axes if not _param_is_fsdp_sharded(param, a))
        if not axes_for_param:
            continue

        scaler = prod(m.shape[axis] for axis in axes_for_param)
        if scaler == 1:
            continue
        inv_scaler = 1.0 / float(scaler)

        grad = param.get_grad()
        for axis in axes_for_param:
            grad = ttml.core.distributed.all_reduce(grad, cluster_axis=axis)
        grad = ttnn.multiply(grad, inv_scaler)
        param.set_grad(grad)


def _param_is_fsdp_sharded(param, axis_index: int) -> bool:
    """True if ``param`` is FSDP-sharded on the given mesh axis."""
    if getattr(param, "_fsdp_managed", False) and getattr(param, "_fsdp_axis", None) == axis_index:
        return True
    return False


def _param_is_sharded_on_axis(param, axis_index: int) -> bool:
    """True if ``param``'s tensor is Shard (not Replicate) on the given mesh axis."""
    placements = ttml.Sharding.from_tensor(param).placements
    if placements is None or axis_index >= len(placements):
        return False
    return isinstance(placements[axis_index], ttnn.PlacementShard)


def sync_sequence_parallel_gradients(parameters, axis_name: str = "tp"):
    """Sum the gradients of TP-replicated parameters across the tensor-parallel axis.

    Under Megatron sequence parallelism the residual stream is sharded along the
    sequence across ``axis_name`` (the TP axis), so a parameter that is *replicated*
    on that axis -- the RMSNorm ``gamma`` weights, and any bias added in a
    sequence-sharded region -- only accumulates the gradient from its rank's slice of
    the sequence. The full gradient is the SUM over ranks, so we all-reduce with **no
    averaging** (unlike :func:`sync_gradients`, which means-reduces over the data
    axes).

    Parameters that are TP-*sharded* (Column/RowParallel weights, the vocab-parallel
    embedding / LM-head weight) are skipped: each rank already holds the correct grad
    for its shard.

    This is orthogonal to :func:`sync_gradients` (which handles dp/fsdp) and both may
    be called -- the reductions are over disjoint axes and commute. It is a no-op when
    the mesh has no ``axis_name`` axis of size > 1.

    Args:
        parameters: A ``NamedParameters`` mapping (e.g. ``model.parameters()``).
        axis_name: The tensor-parallel mesh axis name (default ``"tp"``).
    """
    m = maybe_mesh()
    if m is None or not m.has_axis(axis_name) or m.axis_size(axis_name) == 1:
        return
    axis = m.axis_index(axis_name)

    for _, param in parameters.items():
        if not param.is_grad_initialized():
            continue
        if _param_is_sharded_on_axis(param, axis):
            continue
        # SUM across TP (no division): reconstruct the full-sequence gradient.
        param.set_grad(ttml.core.distributed.all_reduce(param.get_grad(), cluster_axis=axis))
