# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent scratch owned by one layer/replica, independent of request state."""


class DecodeWorkspace:
    def __init__(self, mesh, value_heads, *, shared_qk_heads=None):
        self.mesh = mesh
        self.value_heads = value_heads
        self.outputs = {}
        self.shared_qk_heads = shared_qk_heads
        self.shared_outputs = {}

    def prepare(self, batch):
        """Setup boundary only; never replace buffers referenced by existing traces."""
        import ttnn

        if not isinstance(batch, int) or isinstance(batch, bool) or not 1 <= batch <= 64:
            raise ValueError("Single-step GDN supports batches 1..64")
        # Serving compacts active users into 1/8/16 buckets. Native model calls
        # also need the explicitly allocated cache batch, including 2/4/32/64.
        for size in sorted({batch, *(b for b in (1, 8, 16) if b <= batch)}):
            if size not in self.outputs:
                self.outputs[size] = ttnn.allocate_tensor_on_device(
                    ttnn.Shape([size * self.value_heads, 128]),
                    ttnn.float32,
                    ttnn.ROW_MAJOR_LAYOUT,
                    self.mesh,
                    ttnn.DRAM_MEMORY_CONFIG,
                )
            # These batches passed the adapter's physical comparison. B1 was
            # slower; B2/B4 have not been qualified and retain fused Q/K prep.
            if self.shared_qk_heads is not None and size in (8, 16, 32, 64) and size not in self.shared_outputs:
                self.shared_outputs[size] = tuple(
                    ttnn.allocate_tensor_on_device(
                        ttnn.Shape([size * self.shared_qk_heads, 128]),
                        ttnn.float32,
                        ttnn.ROW_MAJOR_LAYOUT,
                        self.mesh,
                        ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for _ in range(2)
                )

    def shared_qk(self, batch):
        """Lookup only; trace replay must keep the same two scratch addresses."""
        if self.shared_qk_heads is None or batch not in (8, 16, 32, 64):
            return None
        try:
            return self.shared_outputs[batch]
        except KeyError:
            raise RuntimeError(
                "Prepare shared Q/K scratch during cache allocation before decode/trace capture"
            ) from None

    def output(self, batch):
        try:
            return self.outputs[batch]
        except KeyError:
            raise RuntimeError(
                "Prepare the GDN workspace during cache allocation before decode/trace capture"
            ) from None
