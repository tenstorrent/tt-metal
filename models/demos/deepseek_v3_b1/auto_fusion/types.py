# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Core type definitions for the auto-fusion infrastructure.

Defines MicroOpSpec (the hardware contract for each atomic operation),
CBPortSpec (circular buffer port descriptors), and RISCContract
(per-RISC behavior specification).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SDFRate:
    """Synchronous dataflow rate for a CB port.

    In SDF, each actor (micro-op) fires once per invocation and produces/consumes
    a fixed number of tokens (tiles) on each port. The rate can be:
    - A literal integer (e.g., 1 tile)
    - A parametric expression resolved from compile-time args (e.g., "num_tiles")
    """

    tokens: int = 1  # Default: 1 tile per firing
    is_parametric: bool = False  # True if resolved at compile time
    param_expr: str = ""  # CT arg name to resolve (e.g., "num_tiles")


class TransferType(Enum):
    """How data flows between producer and consumer cores."""

    SAME_CORE = "same_core"  # Same core set, output CB = input CB (zero-copy)
    MCAST = "mcast"  # One-to-many multicast
    GATHER = "gather"  # Many-to-one gather
    UNICAST = "unicast"  # Point-to-point
    SCATTER = "scatter"  # One-to-many scatter


class CBDirection(Enum):
    INPUT = "input"
    OUTPUT = "output"
    WEIGHT = "weight"
    SCRATCH = "scratch"


@dataclass(frozen=True)
class CBPortSpec:
    """Specification of a circular buffer port on a micro-op."""

    direction: CBDirection
    is_sharded: bool = False  # Pre-loaded weight/persistent buffer
    sdf_rate: Optional[SDFRate] = None  # SDF token rate (None = inferred as 1)


@dataclass(frozen=True)
class RISCContract:
    """What a micro-op does on a specific RISC processor."""

    # C++ type expression for CTArgs. Use {param} for template params.
    # e.g. "deepseek_b1_ops::Matmul::ComputeCTArgs<{out_w}, {transpose}, {fused_activation}>"
    ct_args_type: str

    # C++ struct type for runtime args
    # e.g. "deepseek_b1_ops::Matmul::ComputeArgs"
    rt_args_type: str

    # Named CT args this RISC reads (list of arg name strings)
    # These get emitted as get_named_compile_time_arg_val("name") in the kernel
    named_ct_args: List[str] = field(default_factory=list)

    # RT arg fields and their source expressions
    # Each entry is (field_name, source_expression) where source_expression
    # references named CT args or common RT args.
    # Prefixes: "ct:" for named CT args, "rt_uint32:" / "rt_float:" for common RT args,
    #           "semaphore:", "read_ptr:", "write_ptr:" for special lookups
    rt_args_fields: List[Tuple[str, str]] = field(default_factory=list)

    # Common runtime args for this RISC: list of (name, c_type) pairs
    # These are passed as common_runtime_args and accessed via get_common_arg_val<T>(idx)
    common_runtime_args: List[Tuple[str, str]] = field(default_factory=list)

    # CB ports this RISC reads/writes (port names from MicroOpSpec.cb_ports)
    cb_reads: List[str] = field(default_factory=list)
    cb_writes: List[str] = field(default_factory=list)

    # Sharded buffers to setup (port names) - NCRISC/BRISC only
    setup_sharded: List[str] = field(default_factory=list)

    is_noop: bool = False


@dataclass(frozen=True)
class MicroOpSpec:
    """
    Complete specification of an atomic micro-op.

    This captures the full hardware contract: what headers to include,
    what each RISC does, what CBs are needed, and how to instantiate the Op.
    """

    name: str  # e.g. "Matmul", "RMSNorm"
    header: str  # e.g. "unified_kernels/matmul.hpp"
    namespace: str  # e.g. "deepseek_b1_ops"
    struct_name: str  # e.g. "Matmul", "RMSNorm"

    # Per-RISC contracts
    ncrisc: RISCContract
    brisc: RISCContract
    trisc: RISCContract

    # CB port specifications: port_name -> CBPortSpec
    cb_ports: Dict[str, CBPortSpec] = field(default_factory=dict)

    # Op class instantiation template (C++ expression)
    # Use {CTArgs}, {role_flags}, etc. as placeholders
    # e.g. "Op<{CTArgs}, {is_active}, true, true>"
    op_template: str = "Op<{CTArgs}, {is_active}, true>"

    # Whether the op has init/teardown (like Mcast persistent pattern)
    has_init: bool = False
    has_teardown: bool = False

    # Additional includes needed
    extra_includes: List[str] = field(default_factory=list)

    # Estimated cycle counts per RISC (for ILP scheduling and MII computation)
    # Keys: "ncrisc", "brisc", "trisc"
    risc_latency: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CBConfig:
    """Configuration for an intermediate CB (not backed by a user tensor).

    Used to specify data format, page size, and page count for CBs that
    need L1 pool allocation (e.g., intermediate results between fused ops).
    """

    page_size: int  # Bytes per page/tile
    num_pages: int  # Number of pages in the CB
    data_format: str = "bfloat16"  # Data format name (resolved to ttnn type at build time)
    tile_height: int = 32  # Tile height (for TileDescriptor)
    tile_width: int = 32  # Tile width

    @property
    def total_size(self) -> int:
        return self.page_size * self.num_pages


@dataclass
class CorePlacement:
    """Which cores an op runs on."""

    core_range_set: object  # ttnn.CoreRangeSet
    role_flag: str  # e.g. "is_rmsnorm_core"


@dataclass
class OpNode:
    """A micro-op instance in the fusion graph."""

    id: str  # Unique ID, e.g. "rmsnorm1"
    spec: MicroOpSpec  # From registry
    placement: CorePlacement  # Where it runs
    ct_args: Dict[str, object]  # Compile-time arg values: name -> value
    rt_args: Dict[str, object]  # Runtime arg values: name -> value
    cb_bindings: Dict[str, int] = field(default_factory=dict)  # port_name -> cb_index


@dataclass
class DataEdge:
    """A data dependency between two op nodes."""

    src_node: str  # Producer node ID
    src_port: str  # Producer CB port name
    dst_node: str  # Consumer node ID
    dst_port: str  # Consumer CB port name
    transfer: TransferType  # How data moves between cores
