# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent scratch owned by one layer/replica, independent of request state."""

SINGLE_STEP_POLICIES = (
    "single_step",
    "single_step_shared_qk",
    "single_step_shared_qk_epilogue",
    "single_step_flat_prepare_epilogue",
)
SHARED_QK_POLICIES = SINGLE_STEP_POLICIES[1:]
EPILOGUE_POLICIES = SINGLE_STEP_POLICIES[2:]
EPILOGUE_BATCHES = (16, 32)


def uses_fused_epilogue(recurrence, batch):
    """Only opt in where the standalone physical comparison found a gain."""
    return recurrence in EPILOGUE_POLICIES and batch in EPILOGUE_BATCHES


class DecodeWorkspace:
    def __init__(self, mesh, value_heads, *, shared_qk_heads=None, fused_epilogue=False, flat_prepare=False):
        self.mesh = mesh
        self.value_heads = value_heads
        self.outputs = {}
        self.shared_qk_heads = shared_qk_heads
        self.shared_outputs = {}
        self.fused_epilogue = fused_epilogue
        self.epilogue_outputs = {}
        self.flat_prepare = flat_prepare
        self.prepared_outputs = {}

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
            if self.fused_epilogue and size in EPILOGUE_BATCHES and size not in self.epilogue_outputs:
                self.epilogue_outputs[size] = ttnn.allocate_tensor_on_device(
                    ttnn.Shape([size, 1, self.value_heads * 128]),
                    ttnn.bfloat16,
                    ttnn.TILE_LAYOUT,
                    self.mesh,
                    ttnn.DRAM_MEMORY_CONFIG,
                )
            if self.flat_prepare and size in EPILOGUE_BATCHES and size not in self.prepared_outputs:
                self.prepared_outputs[size] = tuple(
                    ttnn.allocate_tensor_on_device(
                        ttnn.Shape([size * self.value_heads, width]),
                        ttnn.float32,
                        ttnn.ROW_MAJOR_LAYOUT,
                        self.mesh,
                        ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for width in (128, 8)
                )

    def flat_outputs(self, batch):
        """Persistent value/gate buffers; small buckets retain their existing path."""
        if not self.flat_prepare or batch not in EPILOGUE_BATCHES:
            return None
        try:
            return self.prepared_outputs[batch]
        except KeyError:
            raise RuntimeError(
                "Prepare flat-input scratch during cache allocation before decode/trace capture"
            ) from None

    def epilogue_output(self, batch):
        """Caller-owned tiled output, retained for the complete trace lifetime."""
        try:
            return self.epilogue_outputs[batch]
        except KeyError:
            raise RuntimeError(
                "Prepare the fused epilogue during cache allocation before decode/trace capture"
            ) from None

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
