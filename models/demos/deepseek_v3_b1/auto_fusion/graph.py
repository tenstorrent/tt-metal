# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
FusionGraph: the spatio-temporal dataflow graph for auto-fusion.

Users build a graph of micro-op nodes with spatial placements and data
dependencies, then call build() to generate a fused unified kernel and
host-side descriptors.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from models.demos.deepseek_v3_b1.auto_fusion.cb_allocator import CBAllocator
from models.demos.deepseek_v3_b1.auto_fusion.codegen import UnifiedKernelCodegen
from models.demos.deepseek_v3_b1.auto_fusion.graph_coloring import ChaitinBriggsAllocator
from models.demos.deepseek_v3_b1.auto_fusion.ilp_scheduler import ILPScheduler
from models.demos.deepseek_v3_b1.auto_fusion.polyhedral import PolyhedralAnalyzer
from models.demos.deepseek_v3_b1.auto_fusion.sdf import SDFAnalyzer
from models.demos.deepseek_v3_b1.auto_fusion.software_pipeline import SoftwarePipeliner
from models.demos.deepseek_v3_b1.auto_fusion.types import (
    CBConfig,
    CBDirection,
    CorePlacement,
    DataEdge,
    MicroOpSpec,
    OpNode,
    TransferType,
)


def _cores_to_set(cores) -> set:
    """Convert a CoreRangeSet to a set of (x, y) tuples for comparison."""
    result = set()
    for cr in cores.ranges():
        # Support both real ttnn (start/end) and mock (start_coord/end_coord)
        start = getattr(cr, "start", None) or getattr(cr, "start_coord")
        end = getattr(cr, "end", None) or getattr(cr, "end_coord")
        for x in range(start.x, end.x + 1):
            for y in range(start.y, end.y + 1):
                result.add((x, y))
    return result


def _infer_transfer(src_cores, dst_cores) -> TransferType:
    """Infer TransferType from source and destination core sets."""
    src = _cores_to_set(src_cores)
    dst = _cores_to_set(dst_cores)

    if src == dst or src.issubset(dst) or dst.issubset(src):
        return TransferType.SAME_CORE
    elif len(src) == 1 and len(dst) > 1:
        return TransferType.MCAST
    elif len(src) > 1 and len(dst) == 1:
        return TransferType.GATHER
    elif len(src) == 1 and len(dst) == 1:
        return TransferType.UNICAST
    else:
        return TransferType.SCATTER


class FusionGraph:
    """
    A directed acyclic graph of micro-op nodes.

    Temporal ordering: topological sort of nodes defines execution sequence.
    Spatial mapping: each node's CorePlacement defines which cores run it.

    Usage:
        g = FusionGraph()
        g.add("rmsnorm", RMSNORM, cores=input_core, ct_args={...})
        g.add("mcast",   MCAST,   cores=grid, inputs={"src": ("rmsnorm", "output")})
        g.add("matmul",  MATMUL,  cores=grid, inputs={"in0": ("mcast", "dst")})
        source, schedule, cb_allocs = g.compile()
    """

    def __init__(self):
        self._nodes: Dict[str, OpNode] = {}
        self._edges: List[DataEdge] = []
        self._order: List[str] = []  # Insertion order = topological order
        self._cb_configs: Dict[Tuple[str, str], CBConfig] = {}  # (op_id, port) -> config

    def add(
        self,
        op_id: str,
        spec: MicroOpSpec,
        cores,  # ttnn.CoreRangeSet
        ct_args: Optional[Dict[str, object]] = None,
        rt_args: Optional[Dict[str, object]] = None,
        inputs: Optional[Dict[str, Tuple[str, str]]] = None,
        cb_config: Optional[Dict[str, CBConfig]] = None,
    ) -> OpNode:
        """
        Add a micro-op node to the graph.

        Args:
            op_id: Unique string ID for this op instance (e.g. "rmsnorm1")
            spec: MicroOpSpec from the registry
            cores: ttnn.CoreRangeSet where this op executes
            ct_args: Compile-time arg values (name -> value)
            rt_args: Runtime arg values (name -> value)
            inputs: Data dependencies: {dst_port: (src_op_id, src_port)}
            cb_config: Per-port CB configuration for intermediate CBs
                       (ports not backed by user tensors). Maps port_name -> CBConfig.

        Returns:
            The created OpNode.
        """
        if op_id in self._nodes:
            raise ValueError(f"Duplicate op_id: {op_id}")

        node = OpNode(
            id=op_id,
            spec=spec,
            placement=CorePlacement(cores, f"is_{op_id}_core"),
            ct_args=ct_args or {},
            rt_args=rt_args or {},
        )
        self._nodes[op_id] = node
        self._order.append(op_id)

        # Store CB configs for intermediate ports
        if cb_config:
            for port_name, config in cb_config.items():
                self._cb_configs[(op_id, port_name)] = config

        # Create edges from inputs
        if inputs:
            for dst_port, (src_id, src_port) in inputs.items():
                if src_id not in self._nodes:
                    raise ValueError(f"Unknown source node: {src_id}")
                src_node = self._nodes[src_id]
                transfer = _infer_transfer(
                    src_node.placement.core_range_set,
                    node.placement.core_range_set,
                )
                self._edges.append(DataEdge(src_id, src_port, op_id, dst_port, transfer))

        return node

    @property
    def nodes(self) -> List[OpNode]:
        return [self._nodes[nid] for nid in self._order]

    @property
    def edges(self) -> List[DataEdge]:
        return list(self._edges)

    def get_node(self, op_id: str) -> OpNode:
        return self._nodes[op_id]

    def get_schedule(self) -> List[str]:
        """Return topological order (insertion order for now)."""
        return list(self._order)

    def compile(
        self,
        external_ports: Optional[Set[Tuple[str, str]]] = None,
        l1_budget: int = 1048576,
    ) -> Tuple[str, List[str], "CBAllocator"]:
        """
        Compile the graph into a fused kernel source and CB allocations.

        Runs a 5-stage optimization pipeline:
          1. SDF Analysis     — repetition vectors, buffer bounds
          2. Polyhedral       — fusion legality, tile sizes
          3. ILP Scheduling   — optimal op ordering, makespan minimization
          4. Software Pipeline — MII, reader/compute/writer overlap
          5. Graph Coloring   — Chaitin-Briggs CB slot allocation

        Args:
            external_ports: Set of (op_id, port_name) backed by user tensors.
                If None, infers from port specs (sharded ports are external).
            l1_budget: Available L1 memory in bytes for intermediate buffers.

        Returns:
            (kernel_source, schedule, allocator) where allocator holds the
            full CB allocation state including liveness intervals.
        """
        # =====================================================================
        # Stage 1: SDF Analysis
        # =====================================================================
        sdf = SDFAnalyzer(self)
        self._sdf_result = {
            "repetition_vector": sdf.compute_repetition_vector(),
            "buffer_bounds": sdf.compute_buffer_bounds(),
            "double_buffering": sdf.suggest_double_buffering(),
        }

        # =====================================================================
        # Stage 2: Polyhedral Analysis
        # =====================================================================
        poly = PolyhedralAnalyzer(self)
        self._poly_result = {
            "domains": poly.build_iteration_domains(),
            "tile_sizes": poly.compute_tile_sizes(l1_budget),
        }
        # Verify all sequential pairs are fusion-legal
        schedule_order = self.get_schedule()
        for i in range(len(schedule_order) - 1):
            dep = poly.check_fusion_legality(schedule_order[i], schedule_order[i + 1])
            if not dep.is_legal:
                raise ValueError(
                    f"Fusion illegal between {schedule_order[i]} and " f"{schedule_order[i+1]}: {dep.reason}"
                )

        # =====================================================================
        # Stage 3: ILP Scheduling
        # =====================================================================
        ilp = ILPScheduler(self, sdf_result=sdf, poly_result=poly)
        ilp_result = ilp.solve(l1_budget=l1_budget)
        self._ilp_result = ilp_result

        # Use ILP-optimized schedule if it found a solution
        if ilp_result.solver_status in ("optimal", "feasible", "fallback"):
            schedule = ilp_result.schedule
        else:
            schedule = self.get_schedule()

        # =====================================================================
        # Stage 4: Software Pipelining
        # =====================================================================
        pipeliner = SoftwarePipeliner(self, ilp_result=ilp_result, sdf_result=sdf)
        self._pipeline_result = pipeliner.generate_schedule()

        # =====================================================================
        # Stage 5: Graph Coloring (Chaitin-Briggs) + Legacy Allocator
        # =====================================================================
        # Run Chaitin-Briggs for CB slot assignment validation
        coloring = ChaitinBriggsAllocator(self.nodes, self._edges, schedule)
        coloring_result = coloring.allocate()
        self._coloring_result = coloring_result

        # Use the existing CBAllocator for full allocation (L1 packing etc.)
        # but validate against Chaitin-Briggs result.
        # Disable index reuse: single-phase kernels have CB descriptors set once
        # before kernel_main(), so each unique CB needs its own physical index.
        allocator = CBAllocator(self.nodes, self._edges, schedule)
        allocator.allocate(external_ports=external_ports, allow_index_reuse=False)

        # =====================================================================
        # Post-allocation: Coalesced CB detection + derived ct_args
        # =====================================================================
        self._apply_coalescing_params(allocator, sdf)

        # Auto-generate CBConfig for internal ports that lack them.
        # Internal ports (e.g., reduce1.output chained to mul.in0) need CB
        # descriptors. If the user didn't provide a CBConfig, create a default
        # one using BF16 32x32 tile (2048 bytes).
        # For coalesced CBs, num_pages = number of producer outputs sharing the CB.
        for key, alloc in allocator._allocations.items():
            if not alloc.is_external and key not in self._cb_configs:
                # Only generate for one representative per CB index
                already_has = any(
                    k != key and c_alloc.index == alloc.index and k in self._cb_configs
                    for k, c_alloc in allocator._allocations.items()
                )
                if not already_has:
                    # Count how many output ports share this CB index
                    num_pages = sum(
                        1
                        for (oid, pn), a in allocator._allocations.items()
                        if a.index == alloc.index
                        and self._nodes[oid].spec.cb_ports.get(pn)
                        and self._nodes[oid].spec.cb_ports[pn].direction == CBDirection.OUTPUT
                    )
                    num_pages = max(num_pages, 1)
                    self._cb_configs[key] = CBConfig(
                        page_size=2048,  # BF16 32x32 tile
                        num_pages=num_pages,
                        data_format="bfloat16",
                        tile_height=32,
                        tile_width=32,
                    )

        # L1 packing for intermediate CBs
        cb_sizes = {key: config.total_size for key, config in self._cb_configs.items()}
        if cb_sizes:
            allocator.pack_l1(cb_sizes)

        # Generate unified kernel source
        codegen = UnifiedKernelCodegen(self, schedule, allocator._allocations)
        source = codegen.generate()

        return source, schedule, allocator

    def _apply_coalescing_params(self, allocator, sdf):
        """
        Detect coalesced input CBs and set derived ct_args on consumer nodes.

        When the sibling coalescing pass merges two producer outputs into one CB,
        the consumer op needs adjusted parameters:
        - in0_wait_tiles: total tiles to wait for (covers both producers)
        - in1_wait_tiles: same total (redundant wait is harmless)
        - in1_tile_offset: tile index offset for the second input

        This makes auto-fused output match hand-fused patterns, e.g.:
          cb_wait_front(intermed, 2)
          mul_tiles(intermed, intermed, 0, 1, 0)
        """
        allocs = allocator._allocations

        for node_id in self._order:
            node = self._nodes[node_id]
            # Find SAME_CORE input edges to this node
            input_edges = [e for e in self._edges if e.dst_node == node_id and e.transfer == TransferType.SAME_CORE]
            if len(input_edges) < 2:
                # No coalescing possible or needed — set defaults
                num_tiles = node.ct_args.get("num_tiles", 1)
                if "in0_wait_tiles" not in node.ct_args:
                    node.ct_args["in0_wait_tiles"] = num_tiles
                if "in1_wait_tiles" not in node.ct_args:
                    node.ct_args["in1_wait_tiles"] = num_tiles
                if "in1_tile_offset" not in node.ct_args:
                    node.ct_args["in1_tile_offset"] = 0
                continue

            # Check if any pair of input ports share a CB index
            port_to_cb = {}
            for edge in input_edges:
                dst_key = (edge.dst_node, edge.dst_port)
                if dst_key in allocs:
                    port_to_cb[edge.dst_port] = allocs[dst_key].index

            # Detect coalesced pairs (same CB index for different input ports)
            cb_to_ports = defaultdict(list)
            for port, cb_idx in port_to_cb.items():
                cb_to_ports[cb_idx].append(port)

            coalesced = {cb: ports for cb, ports in cb_to_ports.items() if len(ports) > 1}

            if not coalesced:
                num_tiles = node.ct_args.get("num_tiles", 1)
                node.ct_args.setdefault("in0_wait_tiles", num_tiles)
                node.ct_args.setdefault("in1_wait_tiles", num_tiles)
                node.ct_args.setdefault("in1_tile_offset", 0)
                continue

            # For each coalesced CB, compute the total pages and tile offsets.
            # Ports are ordered by their producer's schedule position.
            for cb_idx, ports in coalesced.items():
                # Find the producing edges, ordered by schedule position
                producing_edges = []
                for port in ports:
                    for edge in input_edges:
                        if edge.dst_port == port:
                            producing_edges.append(edge)
                producing_edges.sort(key=lambda e: self._order.index(e.src_node))

                # Each producer contributes num_tiles output tiles (default 1)
                tiles_per_producer = []
                for edge in producing_edges:
                    src_node = self._nodes[edge.src_node]
                    src_port_spec = src_node.spec.cb_ports.get(edge.src_port)
                    if src_port_spec and src_port_spec.sdf_rate:
                        rate = src_port_spec.sdf_rate
                        if rate.is_parametric:
                            # Resolve from the source node's ct_args
                            tiles = src_node.ct_args.get(rate.param_expr, 1)
                        else:
                            tiles = rate.tokens
                    else:
                        tiles = 1
                    tiles_per_producer.append(tiles)

                total_tiles = sum(tiles_per_producer)

                # The first port's tiles start at offset 0
                # The second port's tiles start at offset = first producer's count
                if len(producing_edges) >= 2:
                    offset = tiles_per_producer[0]
                    node.ct_args["in0_wait_tiles"] = total_tiles
                    node.ct_args["in1_wait_tiles"] = total_tiles
                    node.ct_args["in1_tile_offset"] = offset

    def build(
        self,
        device,
        io_tensors: Dict[Tuple[str, str], object],
        kernel_output_dir: Optional[str] = None,
    ):
        """
        Build the complete fused operation ready for execution.

        Args:
            device: ttnn device
            io_tensors: Map of (op_id, port_name) -> ttnn.Tensor for all
                        external inputs/outputs (sharded weight tensors, etc.)
            kernel_output_dir: If provided, write generated kernel to this dir

        Returns:
            FusedOp ready to execute via .run()
        """
        from models.demos.deepseek_v3_b1.auto_fusion.host_gen import HostGenerator

        # External ports are those with user-provided tensors
        external_ports = set(io_tensors.keys())
        source, schedule, allocator = self.compile(external_ports=external_ports)

        host_gen = HostGenerator(
            self,
            schedule,
            allocator,
            device,
            io_tensors,
            self._cb_configs,
        )
        fused_op = host_gen.build(source, kernel_output_dir)
        return fused_op
