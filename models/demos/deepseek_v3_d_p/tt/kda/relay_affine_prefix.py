# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The sequence-parallel prefix as a relay around the SP ring (ttnn.experimental.kda.relay_affine_prefix), in place of
all-gathering every rank's [A | B] summary and chaining all of them on every chip.

The relay's landing buffers are L1 tensors sharded one head per head core, at the same address on every device; they
outlive each call, since a rank's predecessor writes into them before the rank's own call starts. Layers run one after
another, so one relay per mesh and geometry is shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import ttnn

# Fabric configs that wrap a mesh axis (axis 0 = rows), as ttnn::ccl::get_usable_topology sees them.
_WRAPPED_AXES = {
    ttnn.FabricConfig.FABRIC_2D_TORUS_XY: (0, 1),
    ttnn.FabricConfig.FABRIC_1D_RING: (0, 1),
    ttnn.FabricConfig.FABRIC_2D_TORUS_X: (1,),
    ttnn.FabricConfig.FABRIC_2D_TORUS_Y: (0,),
}


def fabric_wraps_axis(axis: int) -> bool:
    """Whether the current fabric config wraps mesh axis `axis` into a ring."""
    return axis in _WRAPPED_AXES.get(ttnn.get_fabric_config(), ())


def persistent_semaphores(mesh_device, count: int) -> list:
    """Zeroed global semaphores on every worker core that persist across calls, created up front so they sit with the
    model's other long-lived L1 allocations. L1_SMALL, when the device has one, keeps a fabric mux from clobbering them
    (ttnn::ccl::prefer_l1_small_buffer_type)."""
    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    l1_small = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL)
    buffer_type = ttnn.BufferType.L1_SMALL if l1_small.total_bytes_per_bank > 0 else ttnn.BufferType.L1
    semaphores = [ttnn.create_global_semaphore(mesh_device, cores, 0, buffer_type) for _ in range(count)]
    # The zeroed counters must land before any device's program can increment a peer's.
    ttnn.synchronize_device(mesh_device)
    return semaphores


@dataclass(frozen=True)
class RelayOutputGate:
    """input @ weight run in the relay's dispatch on worker cores the relay leaves idle (ttnn.linear's 2D multicast
    matmul with program_config, on its cores):
    the output's first ``copied_columns`` tile columns are copied from ``copy_source`` (from tile column
    ``copy_source_column``) by ``copy_cores``, the projection's columns follow."""

    input: ttnn.Tensor
    weight: ttnn.Tensor
    program_config: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
    compute_config: ttnn.DeviceComputeKernelConfig
    memory_config: ttnn.MemoryConfig
    copy_source: ttnn.Tensor | None = None
    copy_source_column: int = 0
    copied_columns: int = 0
    copy_cores: list = field(default_factory=list)


class RelayAffinePrefix:
    """The SP relay's persistent landing buffers for one mesh and geometry."""

    _shared: dict = {}

    @classmethod
    def shared(cls, mesh_device, **kwargs) -> "RelayAffinePrefix":
        """One relay per mesh and geometry: layers run one after another, so they share its L1 landing buffers."""
        if kwargs.get("cores") is not None:
            kwargs["cores"] = tuple((c.x, c.y) if hasattr(c, "x") else tuple(c) for c in kwargs["cores"])
        key = (id(mesh_device), tuple(sorted(kwargs.items(), key=lambda item: item[0])))
        relay = cls._shared.get(key)
        if relay is None or relay.mesh_device is not mesh_device:
            relay = cls._shared[key] = cls(mesh_device, **kwargs)
        return relay

    def __init__(
        self,
        mesh_device,
        *,
        sequence_parallel_axis,
        batch_heads,
        key_dim,
        value_dim,
        num_links,
        local_rows,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        cores=None,
    ):
        if key_dim != value_dim:
            raise ValueError("relay prefix requires K == V")
        self.mesh_device = mesh_device
        self.axis = sequence_parallel_axis
        self.num_links = num_links
        self.local_rows = local_rows
        grid = mesh_device.compute_with_storage_grid_size()
        # One core per head, then one sender per link: the given cores (those another program merged into the relay
        # leaves idle), else the grid in row-major order.
        if cores is None:
            cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
        cores = [c if isinstance(c, ttnn.CoreCoord) else ttnn.CoreCoord(*c) for c in cores]
        if batch_heads + num_links > len(cores):
            raise ValueError(f"relay prefix needs {batch_heads + num_links} cores, has {len(cores)}")
        self.cores = cores[: batch_heads + num_links]
        self.head_cores = self.cores[:batch_heads]
        head_set = ttnn.CoreRangeSet([]).merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.head_cores]))
        shard = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(head_set, (key_dim, key_dim), ttnn.ShardOrientation.ROW_MAJOR),
        )
        alloc = lambda: ttnn.allocate_tensor_on_device(
            ttnn.Shape([batch_heads * key_dim, key_dim]), ttnn.float32, ttnn.TILE_LAYOUT, mesh_device, shard
        )
        self.entry_landing = alloc()
        self.state_staging = alloc()
        self.final_landing = alloc()
        # A peer can bump the entry and final arrival counters before this rank's call starts.
        self.semaphores = persistent_semaphores(mesh_device, 2)
        # The hop's matmul runs at the affine prefix's fidelity, accumulating in an FP32 DST.
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=math_fidelity, math_approx_mode=False, fp32_dest_acc_en=True
        )

    def __call__(self, packed, initial_state, actual_start, output_gate: RelayOutputGate | None = None):
        """Return this rank's entry states and the final state, both [BH, K, K] FP32, and the output gate's output
        (None without one)."""
        gate = {}
        if output_gate is not None:
            gate = dict(
                output_gate_input=output_gate.input,
                output_gate_weight=output_gate.weight,
                output_gate_program_config=output_gate.program_config,
                output_gate_compute_kernel_config=output_gate.compute_config,
                output_gate_memory_config=output_gate.memory_config,
                output_gate_copy_source=output_gate.copy_source,
                output_gate_copy_source_column=output_gate.copy_source_column,
                output_gate_copied_columns=output_gate.copied_columns,
                output_gate_copy_cores=list(output_gate.copy_cores),
            )
        return ttnn.experimental.kda.relay_affine_prefix(
            packed,
            initial_state,
            actual_start,
            self.entry_landing,
            self.state_staging,
            self.final_landing,
            self.semaphores,
            local_rows=self.local_rows,
            num_links=self.num_links,
            cores=self.cores,
            sequence_parallel_axis=self.axis,
            compute_kernel_config=self.compute_config,
            **gate,
        )
