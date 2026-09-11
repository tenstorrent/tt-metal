# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Measured explicit-ownership trace prototype, superseded by native GDN."""

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import _scan_matmul, _sequential_recurrence


class PrefillRecurrenceTrace:
    """Reserve every Python-visible scratch tensor before any decode capture.

    The model owns this object until teardown, including across slot resets.
    Only the common single-request S32 shape is traced; other shapes use the
    eager reference. Inputs and returned results never alias layer-owned caches.
    Construction requires a mesh with a reserved trace region.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.trace_id = None
        self.inputs = [
            ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh)
            for shape, dtype in [
                *(([12, 32, 1, 128], ttnn.bfloat16),) * 3,
                *(([12, 32, 1, 1], ttnn.bfloat16),) * 2,
                ([1, 12, 128, 128], ttnn.bfloat8_b),
            ]
        ]
        # Compile exact refresh and return-copy signatures before capture.
        for tensor in self.inputs:
            copied = ttnn.clone(tensor)
            ttnn.copy(copied, tensor)
            ttnn.deallocate(copied)
        self.scratch = []
        warm = self._forward()
        for tensor in warm:
            ttnn.deallocate(ttnn.clone(tensor))
        ttnn.synchronize_device(mesh)
        self.scratch.clear()
        self.trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        self.outputs = self._forward()
        ttnn.end_trace_capture(mesh, self.trace_id, cq_id=0)

    def _forward(self):
        # Explicit ownership replaces process-wide hooks/deallocation patches.
        # Keep the same rounding boundaries as _sequential_recurrence.
        def keep(tensor):
            self.scratch.append(tensor)
            return tensor

        query, key, value, beta, decay, initial = self.inputs
        state = keep(ttnn.typecast(initial, ttnn.bfloat16))
        state = keep(ttnn.reshape(state, (12, 1, 128, 128)))
        outputs = []
        for step in range(32):
            k_t = keep(key[:, step : step + 1])
            v_t = keep(value[:, step : step + 1])
            q_t = keep(query[:, step : step + 1])
            d_t = keep(decay[:, step : step + 1])
            b_t = keep(beta[:, step : step + 1])
            decayed = keep(ttnn.multiply(state, d_t))
            memory_value = keep(_scan_matmul(k_t, decayed))
            difference = keep(ttnn.subtract(v_t, memory_value))
            delta = keep(ttnn.multiply(difference, b_t))
            key_t = keep(ttnn.transpose(k_t, -2, -1))
            update = keep(_scan_matmul(key_t, delta))
            state = keep(ttnn.add(decayed, update))
            outputs.append(keep(_scan_matmul(q_t, state)))
        output = keep(ttnn.concat(outputs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        final_state = keep(ttnn.reshape(state, (1, 12, 128, 128)))
        return output, final_state

    def __call__(self, *values, **kwargs):
        sources = (*values, kwargs["initial_state"])
        if self.trace_id is None or any(
            tuple(source.shape) != tuple(destination.shape)
            or source.dtype != destination.dtype
            or source.memory_config() != destination.memory_config()
            for source, destination in zip(sources, self.inputs)
        ):
            return _sequential_recurrence(*values, **kwargs)
        for source, destination in zip(sources, self.inputs):
            ttnn.copy(source, destination)
        ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
        return tuple(ttnn.clone(tensor) for tensor in self.outputs)

    def close(self):
        if self.trace_id is not None:
            ttnn.synchronize_device(self.mesh)
            ttnn.release_trace(self.mesh, self.trace_id)
            self.trace_id = None
            self.outputs = ()
            self.scratch.clear()
            self.inputs.clear()
