# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Host-side descriptor generation for auto-fused kernels.

Generates the Python-side ProgramDescriptor (CB descriptors, kernel descriptors,
compile-time args, runtime args, semaphores) from a compiled FusionGraph.
This is the equivalent of what each micro-op's op.py does manually.

Key capability: L1 memory pool packing. Intermediate CBs (not backed by user
tensors) are packed into shared L1 pool tensors using the allocator's liveness
analysis. Two CBs with non-overlapping lifetimes share the same L1 memory,
matching the hand-optimized L1 reuse patterns in fused_ops/.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import ttnn
from models.demos.deepseek_v3_b1.auto_fusion.cb_allocator import CBAllocator
from models.demos.deepseek_v3_b1.auto_fusion.types import CBConfig
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


@dataclass
class FusedOp:
    """Result of building a fused operation. Ready for execution."""

    program_descriptor: object  # ttnn.ProgramDescriptor
    io_tensors: List[object]  # ttnn tensors in order for generic_op
    kernel_source: str  # Generated C++ source (for debugging)
    kernel_path: str  # Path to written kernel file
    pool_tensors: List[object] = None  # L1 pool tensors (kept alive for lifetime)
    l1_pool_size: int = 0  # Peak L1 pool usage in bytes

    def run(self):
        """Execute the fused operation via ttnn.generic_op."""
        return ttnn.generic_op(self.io_tensors, self.program_descriptor)


class HostGenerator:
    """
    Generates host-side descriptors for a compiled fusion graph.

    Translates the graph's CB allocations, compile-time args, and runtime args
    into the ttnn descriptor objects needed for ttnn.generic_op.
    """

    def __init__(
        self,
        graph,
        schedule: List[str],
        allocator: CBAllocator,
        device,
        io_tensors: Dict[Tuple[str, str], object],
        cb_configs: Dict[Tuple[str, str], CBConfig] = None,
    ):
        self._graph = graph
        self._schedule = schedule
        self._allocator = allocator
        self._cb_allocs = allocator._allocations
        self._device = device
        self._io_tensors = io_tensors  # (op_id, port_name) -> ttnn.Tensor
        self._cb_configs = cb_configs or {}
        self._nodes = {n.id: n for n in graph.nodes}
        self._pool_tensors = []  # Keep alive for program lifetime

    def build(self, kernel_source: str, output_dir: Optional[str] = None) -> FusedOp:
        """Build the complete FusedOp with program descriptor."""

        # Determine kernel output path.
        # The generated kernel uses relative includes like "../../../unified_kernels/..."
        # which assumes the kernel is 3 directory levels below models/demos/deepseek_v3_b1/.
        # We use auto_fusion/kernels/generated/ to match that depth.
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "kernels", "generated")
        os.makedirs(output_dir, exist_ok=True)
        kernel_path = os.path.join(output_dir, "auto_fused_kernel.cpp")

        # Write kernel source
        with open(kernel_path, "w") as f:
            f.write(kernel_source)

        # Create L1 pool tensor for intermediate CBs (if any)
        pool_size = self._create_l1_pool()

        # Build CB descriptors
        cb_descriptors = self._build_cb_descriptors()

        # Build compile-time args
        ncrisc_ct, brisc_ct, trisc_ct = self._build_compile_time_args()

        # Build common runtime args for TRISC
        trisc_common_rt = self._build_trisc_common_runtime_args()

        # Build core descriptors for role flags
        core_descs = self._build_core_descriptors()

        # Build per-core compile-time descriptors
        per_core_descs = self._build_per_core_descriptors()

        # Compute the union of all core ranges
        all_cores = self._get_all_cores()

        # Build semaphore list
        semaphores = self._build_semaphores()

        # Determine compute config from first op that has a non-noop trisc
        compute_config = self._build_compute_config()

        # Create UnifiedKernelDescriptor
        rel_kernel_path = os.path.relpath(kernel_path, self._find_tt_metal_root())

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source=rel_kernel_path,
            core_ranges=all_cores,
            ncrisc_named_compile_time_args=ncrisc_ct,
            brisc_named_compile_time_args=brisc_ct,
            trisc_named_compile_time_args=trisc_ct,
            trisc_common_runtime_args=trisc_common_rt,
            trisc_compute_config=compute_config,
            unified_compile_time_core_descriptors=core_descs,
            per_core_compile_time_descriptors=per_core_descs,
            noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
        )

        # Build ProgramDescriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=cb_descriptors,
            semaphores=semaphores,
        )

        # Build IO tensor list
        io_tensor_list = self._build_io_tensor_list()

        return FusedOp(
            program_descriptor=program_descriptor,
            io_tensors=io_tensor_list,
            kernel_source=kernel_source,
            kernel_path=kernel_path,
            pool_tensors=self._pool_tensors,
            l1_pool_size=pool_size,
        )

    # =========================================================================
    # L1 pool management
    # =========================================================================

    def _create_l1_pool(self) -> int:
        """
        No longer creates pool tensors. Internal CBs use standalone CBDescriptors.
        Returns 0 (no pool).
        """
        return 0

    # =========================================================================
    # CB descriptor generation
    # =========================================================================

    def _build_cb_descriptors(self) -> List:
        """
        Create CB descriptors from tensor bindings and CBConfig entries.

        External CBs: backed by user-provided sharded tensors.
        Internal CBs: standalone CBDescriptor with total_size (no pool tensor).
        """
        cb_descs = []
        seen_indices = set()

        # External CBs (backed by user tensors)
        for (op_id, port_name), alloc in self._cb_allocs.items():
            cb_idx = alloc.index
            if cb_idx in seen_indices:
                continue

            tensor_key = (op_id, port_name)
            if alloc.is_external and tensor_key in self._io_tensors:
                seen_indices.add(cb_idx)
                tensor = self._io_tensors[tensor_key]
                cb_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_idx, tensor)
                cb_descs.append(cb_desc)

        # Internal CBs (standalone CBDescriptor, like the hand-fused pattern)
        for (op_id, port_name), alloc in self._cb_allocs.items():
            cb_idx = alloc.index
            if cb_idx in seen_indices:
                continue
            if alloc.is_external:
                continue

            # Look for CBConfig for this port or any port sharing this CB index
            config = self._cb_configs.get((op_id, port_name))
            if config is None:
                # Check other ports sharing this CB index
                for (oid, pn), other_alloc in self._cb_allocs.items():
                    if other_alloc.index == cb_idx and (oid, pn) in self._cb_configs:
                        config = self._cb_configs[(oid, pn)]
                        break
            if config is None:
                continue

            seen_indices.add(cb_idx)

            # Determine core ranges for this internal CB.
            # Use the union of all node placements that reference this CB index.
            core_ranges = self._get_cb_core_ranges(cb_idx)

            data_format = self._resolve_data_format(config.data_format)
            tile = ttnn.Tile((config.tile_height, config.tile_width))
            tile_descriptor = ttnn.TileDescriptor(tile)

            fmt = ttnn.CBFormatDescriptor(
                buffer_index=cb_idx,
                data_format=data_format,
                page_size=config.page_size,
                tile=tile_descriptor,
            )
            cb_desc = ttnn.CBDescriptor(
                total_size=config.total_size,
                core_ranges=core_ranges,
                format_descriptors=[fmt],
            )
            cb_descs.append(cb_desc)

        return cb_descs

    def _get_cb_core_ranges(self, cb_idx: int):
        """Get the union of core ranges for all nodes that reference a given CB index."""
        seen = set()
        raw_ranges = []
        for (op_id, port_name), alloc in self._cb_allocs.items():
            if alloc.index == cb_idx:
                node = self._nodes[op_id]
                for cr in node.placement.core_range_set.ranges():
                    key = (cr.start.x, cr.start.y, cr.end.x, cr.end.y)
                    if key not in seen:
                        seen.add(key)
                        raw_ranges.append((key, cr))
        return self._filter_overlapping_ranges(raw_ranges)

    def _get_interpreted_tile(self, tensor) -> Tuple[object, int]:
        """
        Compute the interpreted tile and page_size for a tensor.

        Tensors may use narrow 1xN tiles for storage but the kernel interprets
        them as full 32x32 or half 16x32 tiles. This mirrors the logic in
        the standalone micro-op descriptors (e.g., rmsnorm/op.py).

        Returns:
            (interpreted_tile, tile_size) or (None, 0) if no reinterpretation needed.
        """
        FULL_32x32_TILE = ttnn.Tile((32, 32))
        HALF_16x32_TILE = ttnn.Tile((16, 32))

        try:
            tensor_shape = tensor.shape
            data_format = tensor.dtype
            width = tensor_shape[-1]

            # Determine if tiles should be reinterpreted
            is_16x32 = (width // FULL_32x32_TILE.tile_shape[1]) % FULL_32x32_TILE.tile_shape[0] != 0
            interpreted_tile = HALF_16x32_TILE if is_16x32 else FULL_32x32_TILE
            tile_size = interpreted_tile.get_tile_size(data_format)
            return interpreted_tile, tile_size
        except Exception:
            return None, 0

    def _resolve_data_format(self, fmt_name: str):
        """Convert a data format name to ttnn type."""
        fmt_map = {
            "bfloat16": ttnn.bfloat16,
            "float32": ttnn.float32,
            "bfloat8_b": ttnn.bfloat8_b,
            "bfloat4_b": ttnn.bfloat4_b,
        }
        return fmt_map.get(fmt_name, ttnn.bfloat16)

    # =========================================================================
    # Compile-time args
    # =========================================================================

    def _build_compile_time_args(self):
        """Build named compile-time args for each RISC."""
        ncrisc_ct = []
        brisc_ct = []
        trisc_ct = []

        for node_id in self._schedule:
            node = self._nodes[node_id]
            prefix = node_id
            spec = node.spec

            # Add CB index args for all ports
            for port_name, cb_idx in node.cb_bindings.items():
                cb_arg_name = f"{prefix}_{port_name}_cb"
                # All RISCs need CB indices
                ncrisc_ct.append((cb_arg_name, cb_idx))
                brisc_ct.append((cb_arg_name, cb_idx))
                trisc_ct.append((cb_arg_name, cb_idx))

            # Add num_pages args for sharded buffers
            for port_name in spec.ncrisc.setup_sharded:
                if port_name not in node.cb_bindings:
                    continue
                # Only add num_pages for external (sharded) ports
                key = (node_id, port_name)
                if key in self._cb_allocs and not self._cb_allocs[key].is_external:
                    continue
                num_pages_key = f"{port_name}_num_pages"
                if num_pages_key in node.ct_args:
                    ncrisc_ct.append((f"{prefix}_{port_name}_num_pages", node.ct_args[num_pages_key]))

            # Add op-specific CT args
            for key, val in node.ct_args.items():
                if key.startswith("_"):
                    continue  # Skip internal keys like _sender_cores
                arg_name = f"{prefix}_{key}"
                # NCRISC CT args
                if key in [a for a in spec.ncrisc.named_ct_args]:
                    ncrisc_ct.append((arg_name, self._to_ct_val(val)))
                # BRISC CT args
                if key in [a for a in spec.brisc.named_ct_args]:
                    brisc_ct.append((arg_name, self._to_ct_val(val)))
                # TRISC CT args
                if key in [a for a in spec.trisc.named_ct_args]:
                    trisc_ct.append((arg_name, self._to_ct_val(val)))

        return ncrisc_ct, brisc_ct, trisc_ct

    def _build_trisc_common_runtime_args(self) -> List:
        """Collect common runtime args for TRISC across all ops in schedule order."""
        args = []
        for node_id in self._schedule:
            node = self._nodes[node_id]
            for arg_name, c_type in node.spec.trisc.common_runtime_args:
                val = node.ct_args.get(arg_name, 0)
                args.append(int(val))
        return args

    # =========================================================================
    # Core descriptors and semaphores
    # =========================================================================

    def _build_core_descriptors(self) -> List:
        """Build UnifiedCompileTimeCoreDescriptor for role flags."""
        descs = []

        for node_id in self._schedule:
            node = self._nodes[node_id]
            spec = node.spec

            # Main role flag: is_<op_id>_core
            descs.append(
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg=node.placement.role_flag,
                    core_range=node.placement.core_range_set,
                    value=1,
                    other_value=0,
                )
            )

            # Sender/receiver flags for ops with split roles
            if "{is_sender}" in spec.op_template:
                sender_flag = f"is_{node_id}_sender"
                sender_cores = node.ct_args.get("_sender_cores", node.placement.core_range_set)
                descs.append(
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg=sender_flag,
                        core_range=sender_cores,
                        value=1,
                        other_value=0,
                    )
                )

            if "{is_receiver}" in spec.op_template:
                receiver_flag = f"is_{node_id}_receiver"
                receiver_cores = node.ct_args.get("_receiver_cores", node.placement.core_range_set)
                descs.append(
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg=receiver_flag,
                        core_range=receiver_cores,
                        value=1,
                        other_value=0,
                    )
                )

            # Custom bool flags in op_template that need per-core descriptors
            for ct_key, ct_val in node.ct_args.items():
                if ct_key.startswith("_"):
                    continue
                flag_placeholder = f"{{{ct_key}}}"
                if flag_placeholder in spec.op_template and isinstance(ct_val, bool):
                    custom_flag = f"{node_id}_{ct_key}"
                    # Get custom core range if provided, else use node's core range
                    custom_cores = node.ct_args.get(f"_{ct_key}_cores", node.placement.core_range_set)
                    descs.append(
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg=custom_flag,
                            core_range=custom_cores,
                            value=1 if ct_val else 0,
                            other_value=0 if ct_val else 1,
                        )
                    )

        return descs

    def _build_per_core_descriptors(self) -> List:
        """Build PerCoreCompileTimeDescriptor for per-core args like sender_idx."""
        descs = []

        for node_id in self._schedule:
            node = self._nodes[node_id]
            # Check for per-core descriptors stored as _per_core_<name>
            for key, val in node.ct_args.items():
                if key.startswith("_per_core_") and isinstance(val, list):
                    arg_name = key[len("_per_core_") :]
                    full_name = f"{node_id}_{arg_name}"
                    descs.append(
                        PerCoreCompileTimeDescriptor(
                            named_compile_time_arg=full_name,
                            core_values=val,  # List of (CoreCoord, value) tuples
                            other_value=node.ct_args.get(f"_{arg_name}_default", 0),
                        )
                    )

        return descs

    def _build_semaphores(self) -> List:
        """Build global semaphore allocations."""
        sem_count = 0
        for node_id in self._schedule:
            node = self._nodes[node_id]
            for key, val in node.ct_args.items():
                if key.startswith("_"):
                    continue
                if "semaphore" in key.lower() and isinstance(val, int):
                    sem_count = max(sem_count, val + 1)

        if sem_count == 0:
            return []

        # Use full device grid for semaphores (matches hand-fused pattern)
        full_grid = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(
                        self._device.compute_with_storage_grid_size().x - 1,
                        self._device.compute_with_storage_grid_size().y - 1,
                    ),
                )
            ]
        )
        return [ttnn.SemaphoreDescriptor(id=i, core_ranges=full_grid, initial_value=0) for i in range(sem_count)]

    def _build_compute_config(self):
        """Build compute config from the first op with non-noop TRISC.

        Respects per-node overrides:
        - math_fidelity: from ct_args or default LoFi
        - math_approx_mode: from ct_args (e.g., True for SiLU)
        - fp32_dest_acc_en: from ct_args
        """
        for node_id in self._schedule:
            node = self._nodes[node_id]
            if not node.spec.trisc.is_noop:
                fp32 = node.ct_args.get("fp32_dest_acc_en", False)
                math_approx = node.ct_args.get("math_approx_mode", False)
                fidelity_name = node.ct_args.get("math_fidelity", "LoFi")
                fidelity_map = {
                    "LoFi": ttnn.MathFidelity.LoFi,
                    "HiFi2": ttnn.MathFidelity.HiFi2,
                    "HiFi3": ttnn.MathFidelity.HiFi3,
                    "HiFi4": ttnn.MathFidelity.HiFi4,
                }
                fidelity = fidelity_map.get(fidelity_name, ttnn.MathFidelity.LoFi)
                return ttnn.ComputeConfigDescriptor(
                    math_fidelity=fidelity,
                    math_approx_mode=math_approx,
                    fp32_dest_acc_en=fp32,
                    dst_full_sync_en=fp32,
                )
        return ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _filter_overlapping_ranges(raw_ranges):
        """Filter out ranges fully contained by a larger range, then build CoreRangeSet.

        Prevents CoreRangeSet overlap errors when mixing rectangular grids
        (e.g., mcast (0,0)-(3,3)) with individual core ranges (e.g., (0,0)-(0,0)).
        """
        if not raw_ranges:
            return ttnn.CoreRangeSet([])
        filtered = []
        for i, (ki, ri) in enumerate(raw_ranges):
            contained = False
            for j, (kj, _rj) in enumerate(raw_ranges):
                if i == j:
                    continue
                if kj[0] <= ki[0] and kj[1] <= ki[1] and kj[2] >= ki[2] and kj[3] >= ki[3] and ki != kj:
                    contained = True
                    break
            if not contained:
                filtered.append(ri)
        return ttnn.CoreRangeSet(filtered if filtered else [r for _, r in raw_ranges])

    def _get_all_cores(self):
        """Get the union of all core ranges (deduplicated, no overlaps)."""
        seen = set()
        raw_ranges = []
        for node_id in self._schedule:
            node = self._nodes[node_id]
            for cr in node.placement.core_range_set.ranges():
                key = (cr.start.x, cr.start.y, cr.end.x, cr.end.y)
                if key not in seen:
                    seen.add(key)
                    raw_ranges.append((key, cr))
        return self._filter_overlapping_ranges(raw_ranges)

    def _build_io_tensor_list(self) -> List:
        """Build the ordered IO tensor list for generic_op."""
        tensors = []
        seen = set()
        for (op_id, port_name), tensor in self._io_tensors.items():
            tensor_id = id(tensor)
            if tensor_id not in seen:
                seen.add(tensor_id)
                tensors.append(tensor)
        return tensors

    def _to_ct_val(self, val) -> int:
        """Convert a Python value to a compile-time arg integer."""
        if isinstance(val, bool):
            return 1 if val else 0
        if isinstance(val, float):
            return int.from_bytes(struct.pack("f", val), byteorder="little")
        return int(val)

    def _find_tt_metal_root(self) -> str:
        """Find the tt-metal repository root."""
        path = os.path.abspath(__file__)
        while path != "/":
            if os.path.isdir(os.path.join(path, "tt_metal")):
                return path
            path = os.path.dirname(path)
        return os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        )
