# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN Llama-3.1-8B decoder layer (tensor-parallel).

Mirrors the Hugging Face ``LlamaDecoderLayer``::

    h = x + self_attn(input_layernorm(x))
    out = h + mlp(post_attention_layernorm(h))

Tensor-parallel scheme (TP = mesh size, 4 here)
-----------------------------------------------
* Both RMSNorms are REPLICATED. They are per-element ops over the full model
  dim, so they shard in no scheme, and both of their inputs (the layer input and
  the post-attention residual) are already replicated on every chip.
* Self-attention re-uses the sharded ``TtLlamaAttention``: q/k/v column-parallel
  by HEAD, o_proj row-parallel, one all_reduce. It consumes a replicated
  activation and returns a replicated one, so the residual add needs no
  collective.
* MLP (SwiGLU): ``gate_proj`` and ``up_proj`` are COLUMN-parallel — their
  outputs feed the elementwise ``silu(gate) * up``, so each chip owns a disjoint
  slice of the 14336 intermediate features (3584 each) and the product is exact
  locally with no communication. ``down_proj`` REDUCES back to the model dim, so
  it is ROW-parallel: its input features are split the same way, each chip emits
  a PARTIAL sum over the full hidden dim, and one all_reduce over the TP axis
  makes the result replicated again.
* Residual adds happen on replicated tensors after each all_reduce.

Two collectives per layer (one per row-parallel projection). The math is
unchanged: the gathered output equals the single-device golden.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama_3_1_8b_instruct._stubs.attention import (
    TtLlamaAttention,
    _compute_kernel_config,
    _num_devices,
    _tp_cluster_axis,
)
from models.common.modules.tt_ccl import default_topology, get_num_links
from models.demos.llama_3_1_8b_instruct.tt._invocation import record
from models.demos.llama_3_1_8b_instruct._stubs.m_l_p import TtLlamaMLP


class TtLlamaDecoderLayer(LightweightModule):
    """input_norm -> sharded attention -> residual -> post_norm -> sharded MLP -> residual."""

    def __init__(self, mesh_device, torch_module):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_devices = _num_devices(mesh_device)
        self.compute_kernel_config = _compute_kernel_config(mesh_device)

        self.attention = TtLlamaAttention(mesh_device, torch_module.self_attn)
        # The attention module already picked the TP degree its head counts allow;
        # the MLP follows it so both halves of the layer agree on the mesh split.
        self.tp = self.attention.tp

        # --- Replicated norms -------------------------------------------------
        self.input_norm_weight = self._replicate(torch_module.input_layernorm.weight)
        self.input_norm_eps = float(getattr(torch_module.input_layernorm, "variance_epsilon", 1e-5))
        self.post_norm_weight = self._replicate(torch_module.post_attention_layernorm.weight)
        self.post_norm_eps = float(getattr(torch_module.post_attention_layernorm, "variance_epsilon", 1e-5))

        # --- Collective for the MLP's row-parallel down_proj ------------------
        self.cluster_axis = _tp_cluster_axis(mesh_device)
        self.topology = ttnn.Topology.Linear
        self.num_links = 1
        self._mlp_semaphores = None
        if self.tp > 1:
            try:
                self.topology = default_topology(mesh_device) or ttnn.Topology.Linear
            except Exception:
                self.topology = ttnn.Topology.Linear
            try:
                self.num_links = max(int(get_num_links(mesh_device, self.cluster_axis)), 1)
            except Exception:
                self.num_links = 1
            self._mlp_semaphores = self._make_all_reduce_semaphores()

        self.mlp = TtLlamaMLP(
            mesh_device,
            torch_module.mlp,
            self.tp,
            self.compute_kernel_config,
            self._mlp_all_reduce,
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _replicate(self, weight):
        host = weight.detach().to(torch.float32).reshape(1, -1).contiguous().to(torch.bfloat16)
        kwargs = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        if self.num_devices > 1:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.mesh_device)
        return ttnn.from_torch(host, **kwargs)

    def _make_all_reduce_semaphores(self):
        """all_reduce_async == reduce_scatter + all_gather: 2 barrier, 3 rs, 2 ag semaphores."""
        grid = self.mesh_device.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})

        def _new(count):
            return [ttnn.create_global_semaphore(self.mesh_device, cores, 0) for _ in range(count)]

        return {"barrier": _new(2), "rs": _new(3), "ag": _new(2)}

    def _mlp_all_reduce(self, tensor):
        """Sum the MLP's row-parallel partials over the TP axis; result is replicated."""
        if self.tp == 1 or self._mlp_semaphores is None:
            return tensor
        sems = self._mlp_semaphores
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
    def allocate_kv_cache(self, max_seq_len, batch=1):
        """Give this layer's attention its resident KV cache (see TtLlamaAttention)."""
        return self.attention.allocate_kv_cache(max_seq_len, batch)

    def __call__(self, hidden_states, **kwargs):
        return self.forward(hidden_states, **kwargs)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        mode="prefill",
        cur_pos=None,
        **kwargs,
    ):
        record("decoder_layer")  # Gate 2: proof of invocation, from INSIDE the real forward
        x = hidden_states
        in_rank = len(x.shape)
        batch = int(x.shape[0]) if in_rank >= 3 else 1
        seq_len = int(x.shape[-2])
        hidden = int(x.shape[-1])
        if in_rank == 3:
            x = ttnn.reshape(x, (batch, 1, seq_len, hidden))

        # --- Attention block --------------------------------------------------
        normed = ttnn.rms_norm(
            x,
            epsilon=self.input_norm_eps,
            weight=self.input_norm_weight,
            compute_kernel_config=self.compute_kernel_config,
        )
        attn_out = self.attention(
            normed,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            mode=mode,
            cur_pos=cur_pos,
        )
        residual = ttnn.add(x, attn_out)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(x)

        # --- Feed-forward block ----------------------------------------------
        normed2 = ttnn.rms_norm(
            residual,
            epsilon=self.post_norm_eps,
            weight=self.post_norm_weight,
            compute_kernel_config=self.compute_kernel_config,
        )
        mlp_out = self.mlp(normed2)
        ttnn.deallocate(normed2)
        out = ttnn.add(residual, mlp_out)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(residual)

        if in_rank == 3:
            out = ttnn.reshape(out, (batch, seq_len, hidden))
        return out


def build(device, torch_module):
    """Entry point used by the per-component PCC harness."""
    return TtLlamaDecoderLayer(device, torch_module)
