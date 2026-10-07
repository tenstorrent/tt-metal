# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Scratch ownership for sequential TP4 decoder layers and captured replay.

Pass one pool to every layer in a sequential stack on the same command queue.
Different logical batches retain separate scratch allocations. Warm all intended
buckets before capture and obey the trace allocator's lifetime rules. Release
traces before releasing this pool.
Independent overlapping command queues must use different pools.
"""

import ttnn


class DecodeCollectiveBuffers:
    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.tensors = {}

    def acquire(self, operation, x, output_memory):
        key = (operation, tuple(x.shape), str(x.dtype), str(output_memory))
        if key not in self.tensors:
            shape = list(x.shape)
            shape[-1] = shape[-1] * 4 if operation == "gather" else shape[-1] // 4
            output = ttnn.empty(
                shape,
                dtype=x.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=output_memory,
            )
            if operation == "gather":
                value = output
            elif operation == "reduce":
                intermediate, penult = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
                    x, dim=3, topology=ttnn.Topology.Ring
                )
                value = [intermediate, output, penult]
            else:
                raise ValueError(f"Unknown collective: {operation}")
            self.tensors[key] = value
        return self.tensors[key]

    def inventory(self):
        """Host metadata only; no device readback."""
        return [
            {
                "operation": key[0],
                "logical_input_shape": key[1],
                "buffers": [
                    {"padded_shape": list(t.padded_shape), "dtype": str(t.dtype), "memory": str(t.memory_config())}
                    for t in (value if isinstance(value, list) else [value])
                ],
            }
            for key, value in self.tensors.items()
        ]
