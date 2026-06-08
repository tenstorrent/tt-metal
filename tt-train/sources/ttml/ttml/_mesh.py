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
        if mesh.has_axis("dp"):
            dp_axis = mesh.axis_index("dp")
            if dp_axis < len(dim_types) and dim_types[dp_axis] != "RING":
                raise RuntimeError(
                    f"DDP axis (axis {dp_axis}) expected RING topology, "
                    f"but MGD has '{dim_types[dp_axis]}'.\n"
                    f"  MGD dim_types: {dim_types}\n"
                    f"  MGD file: {mgd_path}"
                )

    print(f"MGD validated: dims={mgd_dims}, file={mgd_path}")


_mesh: Mesh | None = None


def open_device_mesh(mesh: tuple[int, ...] | Mesh, device_ids: tuple[int, ...] | None = None):
    """Initialize the global device mesh and open the underlying TT devices.

    When more than one device is requested the MGD file is validated and the
    TT-Fabric interconnect is enabled.  A plain tuple is accepted for backward
    compatibility and converted into a ``Mesh`` with anonymous axis names.
    """
    if not isinstance(mesh, Mesh):
        mesh = Mesh(mesh, tuple(f"_{i}" for i in range(len(mesh))))
    if device_ids is None:
        device_ids = ()

    if mesh.num_devices() > 1:
        _validate_mgd(mesh)
        ttml.core.distributed.enable_fabric(mesh.num_devices())

    ttml.autograd.AutoContext.get_instance().open_device(list(mesh.shape), list(device_ids))

    global _mesh
    _mesh = mesh


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

    scaler = prod(m.shape[axis] for axis in axes)
    if scaler == 1:
        return
    inv_scaler = 1.0 / float(scaler)

    for _, param in parameters.items():
        if not param.is_grad_initialized():
            continue
        grad = param.get_grad()
        for axis in axes:
            grad = ttnn.all_reduce(grad, cluster_axis=axis)
        grad = ttnn.multiply(grad, inv_scaler)
        param.set_grad(grad)
