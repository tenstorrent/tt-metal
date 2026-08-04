# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Everything in ``ttnn`` that is not a tensor and not an op.

Enums, memory/program configs, core grids, the mesh device, and the mesh
mapper/composer protocol. None of it affects the graph except the mesh mappers,
which decide a tensor's distribution at creation -- that is what dissolves
roadmap blockers 7 and 8 (entry placements no longer have to be declared).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .context import CTX


class Enum:
    """A named ttnn enum value; compares by name so identity never matters."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return isinstance(other, Enum) and other.name == self.name

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(("dryrun-enum", self.name))


class Stub:
    """Permissive placeholder for a config object the model builds and passes on.

    Program configs, compute-kernel configs, semaphores and subdevices all reach
    ops as opaque arguments and are never read back, so a stub that accepts any
    attribute, call or index is exactly as informative as the real thing.
    """

    def __init__(self, what: str = "stub", **kw):
        self._what = what
        self.__dict__.update(kw)

    def __call__(self, *a, **k) -> "Stub":
        return self

    def __getattr__(self, name: str) -> "Stub":
        if name.startswith("__"):
            raise AttributeError(name)
        return Stub(self._what + "." + name)

    def __getitem__(self, key) -> "Stub":
        return self

    def __iter__(self):
        return iter(())

    def __eq__(self, other) -> bool:
        return isinstance(other, Stub) and other._what == self._what

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(("dryrun-stub", self._what))

    def __repr__(self) -> str:
        return "<%s>" % self._what


class Namespace:
    """A dotted namespace of fixed values (``ttnn.Layout.TILE``)."""

    def __init__(self, name: str, **members):
        self._name = name
        self.__dict__.update(members)

    def __repr__(self) -> str:
        return "<%s>" % self._name


# -- dtypes, layouts, memory --------------------------------------------------
bfloat16 = Enum("bfloat16")
bfloat8_b = Enum("bfloat8_b")
bfloat4_b = Enum("bfloat4_b")
float32 = Enum("float32")
uint8 = Enum("uint8")
uint16 = Enum("uint16")
uint32 = Enum("uint32")
int32 = Enum("int32")

TILE_LAYOUT = Enum("TILE_LAYOUT")
ROW_MAJOR_LAYOUT = Enum("ROW_MAJOR_LAYOUT")

# `ttnn.Layout.TILE` and `ttnn.TILE_LAYOUT` are the same value in real ttnn, and
# `Parameter._check_data` compares a tensor's layout against the module default,
# so these must be the identical object or every weight load fails.
Layout = Namespace("ttnn.Layout", TILE=TILE_LAYOUT, ROW_MAJOR=ROW_MAJOR_LAYOUT)
DataType = Namespace(
    "ttnn.DataType",
    BFLOAT16=bfloat16,
    BFLOAT8_B=bfloat8_b,
    BFLOAT4_B=bfloat4_b,
    FLOAT32=float32,
    UINT8=uint8,
    UINT16=uint16,
    UINT32=uint32,
    INT32=int32,
)

DRAM_MEMORY_CONFIG = Stub("DRAM_MEMORY_CONFIG")
L1_MEMORY_CONFIG = Stub("L1_MEMORY_CONFIG")
L1_BLOCK_SHARDED_MEMORY_CONFIG = Stub("L1_BLOCK_SHARDED_MEMORY_CONFIG")
L1_WIDTH_SHARDED_MEMORY_CONFIG = Stub("L1_WIDTH_SHARDED_MEMORY_CONFIG")
L1_HEIGHT_SHARDED_MEMORY_CONFIG = Stub("L1_HEIGHT_SHARDED_MEMORY_CONFIG")

Topology = Namespace("ttnn.Topology", Linear=Enum("Linear"), Ring=Enum("Ring"), Mesh=Enum("Mesh"))
MathFidelity = Namespace(
    "ttnn.MathFidelity", LoFi=Enum("LoFi"), HiFi2=Enum("HiFi2"), HiFi3=Enum("HiFi3"), HiFi4=Enum("HiFi4")
)
BufferType = Namespace("ttnn.BufferType", DRAM=Enum("DRAM"), L1=Enum("L1"))
TensorMemoryLayout = Namespace(
    "ttnn.TensorMemoryLayout",
    INTERLEAVED=Enum("INTERLEAVED"),
    HEIGHT_SHARDED=Enum("HEIGHT_SHARDED"),
    WIDTH_SHARDED=Enum("WIDTH_SHARDED"),
    BLOCK_SHARDED=Enum("BLOCK_SHARDED"),
)
ShardOrientation = Namespace("ttnn.ShardOrientation", ROW_MAJOR=Enum("ROW_MAJOR"), COL_MAJOR=Enum("COL_MAJOR"))
DumpTensorMode = Namespace("ttnn.DumpTensorMode", LOCAL=Enum("LOCAL"), GLOBAL=Enum("GLOBAL"))
FabricConfig = Namespace(
    "ttnn.FabricConfig",
    DISABLED=Enum("DISABLED"),
    FABRIC_1D=Enum("FABRIC_1D"),
    FABRIC_1D_RING=Enum("FABRIC_1D_RING"),
    FABRIC_2D=Enum("FABRIC_2D"),
    FABRIC_2D_DYNAMIC=Enum("FABRIC_2D_DYNAMIC"),
)
TILE_SIZE = 32


# -- core grids ---------------------------------------------------------------
class CoreCoord:
    def __init__(self, x: int = 0, y: int = 0):
        self.x, self.y = int(x), int(y)

    def __repr__(self) -> str:
        return "CoreCoord(%d, %d)" % (self.x, self.y)


class CoreGrid:
    def __init__(self, x: int = 1, y: int = 1, **k):
        self.x, self.y = int(x), int(y)

    @property
    def num_cores(self) -> int:
        return self.x * self.y

    def __repr__(self) -> str:
        return "CoreGrid(x=%d, y=%d)" % (self.x, self.y)


class MeshShape(tuple):
    def __new__(cls, *dims):
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])
        return super().__new__(cls, tuple(int(d) for d in dims))


class MeshCoordinate(tuple):
    def __new__(cls, *dims):
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])
        return super().__new__(cls, tuple(int(d) for d in dims))


# -- mesh device --------------------------------------------------------------
class MeshDevice:
    """Metadata-only mesh device: a shape, an arch and a core grid."""

    def __init__(self, shape: Sequence[int], arch_name: str = "blackhole", grid=(13, 10)):
        self.shape = MeshShape(*shape)
        self._arch = Enum(arch_name.upper())
        self._grid = grid

    def arch(self):
        return self._arch

    def compute_with_storage_grid_size(self) -> CoreCoord:
        return CoreCoord(*self._grid)

    def dram_grid_size(self) -> CoreCoord:
        return CoreCoord(8, 1)

    @property
    def core_grid(self) -> CoreGrid:
        return CoreGrid(*self._grid)

    def get_num_devices(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def id(self) -> int:
        return 0

    def create_submeshes(self, shape, *a, **k) -> List["MeshDevice"]:
        sub = tuple(shape)
        count = 1
        for whole, part in zip(self.shape, sub):
            count *= whole // max(1, part)
        # Every submesh is a separate mesh in the IR (roadmap blocker 22, phase 10);
        # until then a dry run that fans out over submeshes analyses only one.
        return [MeshDevice(sub, self._arch.name, self._grid) for _ in range(count)]

    def create_submesh(self, shape, *a, **k) -> "MeshDevice":
        return MeshDevice(tuple(shape), self._arch.name, self._grid)

    def get_view(self, *a, **k) -> "MeshDevice":
        return self

    def __getattr__(self, name: str):
        # Device bookkeeping (sub-device stall groups, program cache, profiler
        # hooks) never reaches a tensor, so an accepting stub cannot corrupt the
        # graph -- unlike an op, which would silently lose a collective.
        if name.startswith("__"):
            raise AttributeError(name)
        return Stub("MeshDevice." + name)

    def __eq__(self, other) -> bool:
        return other is self

    def __ne__(self, other) -> bool:
        return other is not self

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return "DryRunMeshDevice(%s, %s)" % (tuple(self.shape), self._arch.name)


# -- mesh mappers and composers ----------------------------------------------
class MeshMapper:
    """How a host tensor fractures over the mesh: one tensor dim per mesh axis."""

    def __init__(self, dims: Optional[Sequence[Optional[int]]] = None):
        self.dims = list(dims) if dims is not None else None

    def shard(self) -> dict:
        return {m: a for m, a in enumerate(self.dims or []) if a is not None}

    def __repr__(self) -> str:
        return "MeshMapper(%s)" % (self.dims,)


class MeshComposer:
    """The inverse: which mesh axis concatenates onto which tensor dim."""

    def __init__(self, dims: Optional[Sequence[Optional[int]]] = None):
        self.dims = list(dims) if dims is not None else None


class PlacementShard:
    def __init__(self, dim: int):
        self.dim = int(dim)


class PlacementReplicate:
    def __init__(self, *a, **k):
        self.dim = None


class MeshMapperConfig:
    def __init__(self, placements: Sequence[object] = ()):
        self.placements = list(placements)


class MeshComposerConfig:
    def __init__(self, placements: Sequence[object] = ()):
        self.placements = list(placements)


def _dims_of(config) -> List[Optional[int]]:
    return [getattr(p, "dim", None) for p in getattr(config, "placements", [])]


def create_mesh_mapper(device=None, config=None, *a, **k) -> MeshMapper:
    """`utils/tensor.from_torch` builds its mapper this way (placements per mesh axis)."""
    return MeshMapper(_dims_of(config))


def create_mesh_composer(device=None, config=None, *a, **k) -> MeshComposer:
    return MeshComposer(_dims_of(config))


def ShardTensor2dMesh(device=None, mesh_shape=None, dims=None, **k) -> MeshMapper:
    return MeshMapper(dims)


def ShardTensorToMesh(device=None, dim=None, **k) -> MeshMapper:
    dims: List[Optional[int]] = [None] * len(CTX.mesh.shape if CTX.mesh else (1, 1))
    dims[0] = dim
    return MeshMapper(dims)


def ReplicateTensorToMesh(device=None, **k) -> MeshMapper:
    return MeshMapper(None)


def ConcatMesh2dToTensor(device=None, mesh_shape=None, dims=None, **k) -> MeshComposer:
    return MeshComposer(dims)


def ConcatMeshToTensor(device=None, dim=None, **k) -> MeshComposer:
    return MeshComposer([dim, None])


def create_global_semaphore(*a, **k) -> Stub:
    # Real semaphore identity is only needed for the phase 11 buffer-liveness gate.
    return Stub("global_semaphore")


def init_device_compute_kernel_config(*a, **k) -> Stub:
    return Stub("compute_kernel_config")
