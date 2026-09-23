# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN SwiGLU MLP for Llama-3.1-8B-Instruct (tensor-parallel).

Mirrors the Hugging Face ``LlamaMLP``::

    down_proj(silu(gate_proj(x)) * up_proj(x))

Tensor-parallel scheme (TP = mesh size, 4 here)
-----------------------------------------------
hidden = 4096, intermediate = 14336.

* ``gate_proj`` and ``up_proj`` are COLUMN-parallel: their outputs feed the
  per-element ``silu(gate) * up``, so splitting the OUTPUT (intermediate)
  features gives every chip a disjoint 3584-wide slice that it can gate
  locally — no collective is needed in the middle of the block, because
  element ``i`` of the product only ever depends on element ``i`` of gate and up.
* ``down_proj`` is the projection that REDUCES back to the model dim, so it is
  ROW-parallel: its INPUT features are split on exactly the same 3584-wide
  boundaries, each chip computes a PARTIAL sum over the full hidden dim, and one
  ``all_reduce`` over the TP axis turns the partials into the replicated result.
* A ``down_proj`` bias would belong to the reduced output, so it stays
  REPLICATED and is added AFTER the reduce (adding it before would count it TP
  times). Llama-3.1 has no MLP biases; the general case is handled anyway.

The math is unchanged: the gathered output equals the single-device golden.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.tt_ccl import default_topology, get_num_links
from models.demos.llama_3_1_8b_instruct.tt._invocation import record
from models.demos.llama_3_1_8b_instruct._stubs.attention import (
    _compute_kernel_config,
    _num_devices,
    _tp_cluster_axis,
)


class TtLlamaMLP(LightweightModule):
    """Column-parallel gate/up -> silu*mul -> row-parallel down -> all_reduce.

    ``tp`` / ``compute_kernel_config`` / ``collective`` let an enclosing layer
    reuse its own mesh split and collective; left unset, the module derives them
    itself and owns its semaphores.
    """

    def __init__(self, mesh_device, torch_module, tp=None, compute_kernel_config=None, collective=None):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_devices = _num_devices(mesh_device)
        self.compute_kernel_config = compute_kernel_config or _compute_kernel_config(mesh_device)

        self.hidden_size = int(torch_module.gate_proj.weight.shape[1])
        self.intermediate_size = int(torch_module.gate_proj.weight.shape[0])

        if tp is None:
            tp = self.num_devices
            if tp > 1 and self.intermediate_size % tp:
                tp = 1  # replicate rather than split the intermediate dim unevenly
        self.tp = tp

        w_gate = torch_module.gate_proj.weight.detach().to(torch.float32).transpose(0, 1).contiguous()
        w_up = torch_module.up_proj.weight.detach().to(torch.float32).transpose(0, 1).contiguous()
        w_down = torch_module.down_proj.weight.detach().to(torch.float32).transpose(0, 1).contiguous()

        # gate/up: [hidden, intermediate] split on the INTERMEDIATE (output) axis.
        # down: [intermediate, hidden] split on the INTERMEDIATE (input) axis --
        # the same slices, so chip d's silu(gate)*up lines up with chip d's rows.
        self.w_gate = self._to_device(w_gate, shard_dim=-1)
        self.w_up = self._to_device(w_up, shard_dim=-1)
        self.w_down = self._to_device(w_down, shard_dim=0)

        self.b_gate = self._maybe_bias(torch_module.gate_proj, shard_dim=-1)
        self.b_up = self._maybe_bias(torch_module.up_proj, shard_dim=-1)
        self.b_down = self._maybe_bias(torch_module.down_proj, shard_dim=None)

        # --- Collective that closes the row-parallel down_proj ---------------
        self._external_collective = collective
        self.cluster_axis = _tp_cluster_axis(mesh_device)
        self.topology = ttnn.Topology.Linear
        self.num_links = 1
        self._ar_semaphores = None
        if collective is None and self.tp > 1:
            try:
                self.topology = default_topology(mesh_device) or ttnn.Topology.Linear
            except Exception:
                self.topology = ttnn.Topology.Linear
            try:
                self.num_links = max(int(get_num_links(mesh_device, self.cluster_axis)), 1)
            except Exception:
                self.num_links = 1
            self._ar_semaphores = self._make_all_reduce_semaphores()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _maybe_bias(self, proj, shard_dim):
        b = getattr(proj, "bias", None)
        if b is None:
            return None
        return self._to_device(b.detach().to(torch.float32).reshape(1, -1).contiguous(), shard_dim=shard_dim)

    def _to_device(self, host_tensor, shard_dim=None, dtype=ttnn.bfloat16):
        mapper = None
        if self.num_devices > 1:
            if shard_dim is None or self.tp == 1:
                mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
            else:
                mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=shard_dim)
        kwargs = dict(dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        if mapper is not None:
            kwargs["mesh_mapper"] = mapper
        return ttnn.from_torch(host_tensor.to(torch.bfloat16), **kwargs)

    def _make_all_reduce_semaphores(self):
        """all_reduce_async == reduce_scatter + all_gather: 2 barrier, 3 rs, 2 ag semaphores."""
        grid = self.mesh_device.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})

        def _new(count):
            return [ttnn.create_global_semaphore(self.mesh_device, cores, 0) for _ in range(count)]

        return {"barrier": _new(2), "rs": _new(3), "ag": _new(2)}

    def _all_reduce(self, tensor):
        """Sum the row-parallel partials over the TP axis; the result is replicated."""
        if self._external_collective is not None:
            return self._external_collective(tensor)
        if self.tp == 1 or self._ar_semaphores is None:
            return tensor
        sems = self._ar_semaphores
        reduced = ttnn.experimental.all_reduce_async(
            tensor,
            cluster_axis=self.cluster_axis,
            mesh_device=self.mesh_device,
            barrier_semaphores=sems["barrier"],
            rs_global_semaphores=sems["rs"],
            ag_global_semaphores=sems["ag"],
            math_op=ttnn.ReduceType.Sum,
            num_links=self.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.topology,
        )
        ttnn.deallocate(tensor)
        return reduced

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def __call__(self, x, **kwargs):
        return self.forward(x)

    def forward(self, x):
        record("m_l_p")  # Gate 2: proof of invocation, from INSIDE the real forward
        in_rank = len(x.shape)
        batch = int(x.shape[0]) if in_rank >= 3 else 1
        seq_len = int(x.shape[-2])
        hidden = int(x.shape[-1])
        if in_rank == 3:
            x = ttnn.reshape(x, (batch, 1, seq_len, hidden))

        gate = ttnn.linear(
            x,
            self.w_gate,
            bias=self.b_gate,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        up = ttnn.linear(
            x,
            self.w_up,
            bias=self.b_up,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        activated = ttnn.silu(gate)
        ttnn.deallocate(gate)
        inner = ttnn.mul(activated, up)
        ttnn.deallocate(activated)
        ttnn.deallocate(up)

        # Row-parallel: every chip holds a PARTIAL sum over the full hidden dim.
        out = ttnn.linear(
            inner,
            self.w_down,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(inner)

        out = self._all_reduce(out)
        if self.b_down is not None:
            out = ttnn.add(out, self.b_down)

        if in_rank == 3:
            out = ttnn.reshape(out, (batch, seq_len, hidden))
        return out


def build(device, torch_module):
    """Entry point used by the per-component PCC harness."""
    return TtLlamaMLP(device, torch_module)
