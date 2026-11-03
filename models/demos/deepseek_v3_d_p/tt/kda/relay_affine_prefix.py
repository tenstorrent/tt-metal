# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The sequence-parallel prefix as a relay along the SP axis (ttnn.experimental.kda.relay_affine_prefix): each rank
applies its own [A | B] summary to the entry state its predecessor sends and forwards the result, and the last rank
hands the final state to the first. On a ring that hand-off is one hop; on a line it crosses the SP axis (2D routing,
or sp_size - 1 hops back on a 1D fabric).

The relay's landing buffer is an L1 tensor sharded one head per head core, at the same address on every device. A
rank's predecessor writes into it before the rank's own call starts, so a layer reserves it before its first program
(reserve) and the call frees it once the relay has consumed it: between layers that L1 is free for the other
attention layers (ring_mla's circular buffers span nearly all of L1 on the relay's head columns).

Load-bearing: the entry barrier of the layer's qkv convolution history exchange runs after the reservation and before
the relay, so no rank relays until every rank has reserved its landing buffer and finished the programs before it,
whose circular buffers may overlap that L1. A layer that drops that exchange needs another SP barrier there. Layers run
one after another, so one relay per mesh and geometry is shared; the final carry's SP-line barrier separates
consecutive calls.
"""

from __future__ import annotations

import ttnn


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


class RelayAffinePrefix:
    """The SP relay's landing buffer and semaphores for one mesh and geometry."""

    _shared: dict = {}

    @classmethod
    def shared(cls, mesh_device, **kwargs) -> RelayAffinePrefix:
        """One relay per mesh and geometry (mesh ids are never reused): layers run one after another, so they share
        its L1 landing buffer."""
        if kwargs.get("cores") is not None:
            kwargs["cores"] = tuple((c.x, c.y) if hasattr(c, "x") else tuple(c) for c in kwargs["cores"])
        key = (mesh_device.id(), tuple(sorted(kwargs.items())))
        if key not in cls._shared:
            cls._shared[key] = cls(mesh_device, **kwargs)
        return cls._shared[key]

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
        # One core per head, then one sender per link: the given cores, else the grid in row-major order.
        if cores is None:
            grid = mesh_device.compute_with_storage_grid_size()
            cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
        cores = [c if isinstance(c, ttnn.CoreCoord) else ttnn.CoreCoord(*c) for c in cores]
        if batch_heads + num_links > len(cores):
            raise ValueError(f"relay prefix needs {batch_heads + num_links} cores, has {len(cores)}")
        self.cores = cores[: batch_heads + num_links]
        head_set = ttnn.CoreRangeSet([]).merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores[:batch_heads]]))
        self._landing_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(head_set, (3 * key_dim, key_dim), ttnn.ShardOrientation.ROW_MAJOR),
        )
        # Three K x K states per head core (the entry state, the staged next state, the final state) while reserved.
        self._landing_shape = ttnn.Shape([3 * batch_heads * key_dim, key_dim])
        self._landing = None
        # A peer can bump the entry and final arrival counters before this rank's call starts.
        self.semaphores = persistent_semaphores(mesh_device, 2)
        # The hop's matmul runs at the affine prefix's fidelity, accumulating in an FP32 DST.
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=math_fidelity, math_approx_mode=False, fp32_dest_acc_en=True
        )
        # Where the entry states land for their reader (the chunk scan, or the entry correction's many cores).
        self.entry_memory_config = ttnn.DRAM_MEMORY_CONFIG

    def reserve(self):
        """Allocate the landing buffer for the next call, before any program of its layer (a no-op while reserved)."""
        if self._landing is None:
            self._landing = ttnn.allocate_tensor_on_device(
                self._landing_shape, ttnn.float32, ttnn.TILE_LAYOUT, self.mesh_device, self._landing_memory_config
            )

    def __call__(self, packed, initial_state, actual_start, entry_memory_config=None):
        """Return this rank's entry states (in entry_memory_config, default self.entry_memory_config) and the final
        state, both [BH, K, K] FP32. Frees the landing buffer once the relay has run."""
        # Reserving here suffices only when no program of this rank runs between its predecessor's relay and its own.
        self.reserve()
        result = ttnn.experimental.kda.relay_affine_prefix(
            packed,
            initial_state,
            actual_start,
            self._landing,
            self.semaphores,
            local_rows=self.local_rows,
            num_links=self.num_links,
            cores=self.cores,
            sequence_parallel_axis=self.axis,
            entry_memory_config=entry_memory_config or self.entry_memory_config,
            compute_kernel_config=self.compute_config,
        )
        # Every write into it is consumed within the call: the predecessor's entry columns by this rank's heads, and
        # the last rank's final state by the first rank's writer.
        ttnn.deallocate(self._landing)
        self._landing = None
        return result
