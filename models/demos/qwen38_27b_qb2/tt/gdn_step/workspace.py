# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent scratch independent of request state; compact L1 can be replica-shared."""

SINGLE_STEP_POLICIES = (
    "single_step",
    "single_step_shared_qk",
    "single_step_shared_qk_epilogue",
    "single_step_flat_prepare_epilogue",
    "single_step_compact_gdn",
)
SHARED_QK_POLICIES = SINGLE_STEP_POLICIES[1:]
EPILOGUE_POLICIES = SINGLE_STEP_POLICIES[2:]
EPILOGUE_BATCHES = (16, 32)


def uses_fused_epilogue(recurrence, batch):
    """Restrict fused policies to their explicitly targeted batch sizes."""
    return recurrence in EPILOGUE_POLICIES and batch in EPILOGUE_BATCHES


class CompactScratch:
    """Transient compact operands for one serialized CQ0 layer stack.

    Every layer consumes these tensors before the next layer overwrites them.
    Request history, recurrent state and each layer's other workspace remain
    private. Independent replicas or concurrently submitted stacks need their
    own pool. Allocate all batch shapes before capturing traces.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.outputs = {}


class DecodeWorkspace:
    def __init__(
        self,
        mesh,
        value_heads,
        *,
        shared_qk_heads=None,
        fused_epilogue=False,
        flat_prepare=False,
        compact_frontend=False,
        compact_pool=None,
    ):
        if compact_frontend and (shared_qk_heads != 4 or value_heads != 12 or not fused_epilogue or not flat_prepare):
            raise ValueError("Compact GDN needs TP4 heads, shared Q/K, flat preparation and fused epilogue")
        if compact_pool is not None and (not compact_frontend or compact_pool.mesh is not mesh):
            raise ValueError("Compact scratch pool must belong to the same mesh and compact policy")
        self.mesh = mesh
        self.value_heads = value_heads
        self.outputs = {}
        self.shared_qk_heads = shared_qk_heads
        self.shared_outputs = {}
        self.fused_epilogue = fused_epilogue
        self.epilogue_outputs = {}
        self.flat_prepare = flat_prepare
        self.prepared_outputs = {}
        self.compact_frontend = compact_frontend
        self.compact_outputs = compact_pool.outputs if compact_pool is not None else {}

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
            if self.compact_frontend and size in EPILOGUE_BATCHES and size not in self.compact_outputs:
                # Q/K/V plus normalized/gated output remain compact in L1.
                # Allocate once, before any trace can reference these addresses.
                self.compact_outputs[size] = tuple(
                    ttnn.allocate_tensor_on_device(
                        ttnn.Shape([1, size, width]),
                        ttnn.bfloat16,
                        ttnn.TILE_LAYOUT,
                        self.mesh,
                        ttnn.L1_MEMORY_CONFIG,
                    )
                    for width in (512, 512, 1536, 1536)
                )

    def compact_buffers(self, batch):
        try:
            return self.compact_outputs[batch]
        except KeyError:
            raise RuntimeError(
                "Prepare compact GDN scratch during cache allocation before decode/trace capture"
            ) from None

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
