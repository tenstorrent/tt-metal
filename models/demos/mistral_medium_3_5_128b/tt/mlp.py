# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""SiLU-gated dense MLP: ``down(silu(gate(x)) * up(x))``.

Structure from ``minimax_m3/tt/dense_mlp.py``: gate/up column-parallel (intermediate dim over TP),
down row-parallel, closed by a TP reduce-scatter straight into the sharded residual layout. The
activation is Ministral3's plain SiLU gate (M3's clamped swigluoai is not ported). Program configs are
left to ttnn's auto-selection at this model's shapes; compute is the HiFi4 / fp32-accumulate default.
"""

import ttnn

from .common import cache_name, compute_config, dtype_tag
from .precision import precision


def silu_gate(gate, up):
    """``silu(gate) * up`` on device; consumes neither input."""
    return ttnn.mul(
        gate,
        up,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class MLP:
    def __init__(
        self,
        mesh_device,
        mesh_config,
        ccl_manager,
        state_dict,
        *,
        gate_dtype=ttnn.bfloat8_b,
        up_dtype=ttnn.bfloat8_b,
        down_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        """``state_dict``: HF ``{gate,up,down}_proj.weight`` in ``[out, in]`` layout (empty when loading
        from ``tensor_cache_path``)."""
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.compute_kernel_config = compute_config(mesh_device)

        def load(name, dtype, mapper):
            w = state_dict.get(f"{name}.weight") if state_dict else None
            if w is not None:
                w = w.transpose(-2, -1).unsqueeze(0).unsqueeze(0)  # [1, 1, in, out]
            return ttnn.as_tensor(
                w,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
                cache_file_name=cache_name(tensor_cache_path, f"{name}_tp{mesh_config.tp}_{dtype_tag(dtype)}"),
            )

        self.gate_proj = load("gate_proj", gate_dtype, mesh_config.column_parallel(mesh_device))
        self.up_proj = load("up_proj", up_dtype, mesh_config.column_parallel(mesh_device))
        self.down_proj = load("down_proj", down_dtype, mesh_config.row_parallel(mesh_device))

    def _linear(self, x, w, dtype=ttnn.bfloat16):
        return ttnn.linear(
            x,
            w,
            dtype=dtype,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x):
        """``x`` full-width ``[1, 1, s_local, hidden]`` -> ``[1, 1, s_local, hidden/tp]`` (reduce-scattered)."""
        gate = self._linear(x, self.gate_proj)
        up = self._linear(x, self.up_proj)
        act = silu_gate(gate, up)
        gate.deallocate(True)
        up.deallocate(True)
        out = self._linear(act, self.down_proj, dtype=precision().proj_out_dtype)
        act.deallocate(True)
        if self.mesh_config.tp == 1:
            return out
        scattered = self.mesh_config.reduce_scatter(out, self.ccl_manager, axis=self.mesh_config.tp_axis, dim=3)
        out.deallocate(True)
        return scattered
