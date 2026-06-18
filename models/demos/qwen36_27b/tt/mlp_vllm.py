# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dense SwiGLU MLP for Qwen3.6-27B with tensor-parallel (TP) sharding.

TP scheme (mirrors p300x2/tp_model.py):
  * gate_proj / up_proj : column(output)-parallel. Fused into ONE matmul whose
    weight is [H, 2*I] sharded on the output dim (dim=3). Each chip computes its
    [.., 2*I/TP] slice; split into the per-chip gate (silu) and up halves.
  * down_proj           : row(input)-parallel. Weight [I, H] sharded on the input
    dim (dim=2). Each chip matmuls its [.., I/TP] activation; all_reduce sums the
    partial [.., H] outputs across the mesh.

When dense_tp is False this reduces to the plain single-device dense MLP.
"""

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule


class TtMLP(LightweightModule):
    def __init__(self, device, state_dict, layer_idx, config, dtype=ttnn.bfloat16, weights_dtype=None):
        super().__init__()
        self.device = device
        self.config = config
        self.dense_tp = getattr(config, "dense_tp", False)
        self.tp_size = getattr(config, "tp_size", 8)
        self.intermediate_size = config.intermediate_size
        if weights_dtype is None:
            weights_dtype = config.get_dense_dtype(getattr(config, "weights_dtype", ttnn.bfloat8_b))
        prefix = f"model.layers.{layer_idx}.mlp"

        gate_w = state_dict[f"{prefix}.gate_proj.weight"].T.contiguous()  # [H, I]
        up_w = state_dict[f"{prefix}.up_proj.weight"].T.contiguous()  # [H, I]
        down_w = state_dict[f"{prefix}.down_proj.weight"].T.contiguous()  # [I, H]

        if self.dense_tp:
            # Fuse gate|up into one column-parallel matmul: [H, 2I] sharded on dim 3.
            # MUST be PER-CHIP INTERLEAVED [gate_c | up_c] per chip so a contiguous
            # dim-3 shard gives chip c its own gate slice AND up slice. A plain
            # [all-gate | all-up] cat would put gate on chips 0..3 and up on 4..7,
            # and forward() (which slices each chip's output as [gate_c|up_c]) would
            # compute silu(gate)*gate and never use up — uncorrelated MLP output.
            # (Same fusion-ordering issue as the attention Wqkv shard.)
            Ip = self.intermediate_size // self.tp_size
            blocks = []
            for c in range(self.tp_size):
                blocks.append(torch.cat([gate_w[:, c * Ip:(c + 1) * Ip],
                                         up_w[:, c * Ip:(c + 1) * Ip]], dim=1))
            gu = torch.cat(blocks, dim=1)  # [H, 2I] in per-chip [gate_c|up_c] order
            self.gate_up_w = ttnn.from_torch(
                gu.unsqueeze(0).unsqueeze(0),
                dtype=weights_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=3),
            )
            # down: row(input)-parallel, shard input dim (dim 2 of [1,1,I,H]).
            self.down_proj_w = ttnn.from_torch(
                down_w.unsqueeze(0).unsqueeze(0),
                dtype=weights_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=2),
            )
        else:
            self.gate_proj_w = ttnn.from_torch(
                gate_w.unsqueeze(0).unsqueeze(0), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device
            )
            self.up_proj_w = ttnn.from_torch(
                up_w.unsqueeze(0).unsqueeze(0), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device
            )
            self.down_proj_w = ttnn.from_torch(
                down_w.unsqueeze(0).unsqueeze(0), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device
            )

    def forward(self, x):
        kcfg = self.config.matmul_kcfg()
        if self.dense_tp:
            Ip = self.intermediate_size // self.tp_size
            gu = ttnn.linear(x, self.gate_up_w, compute_kernel_config=kcfg)  # [.., 2*Ip]
            nd = len(gu.shape)
            lo = [0] * nd
            mid = list(gu.shape)
            mid[-1] = Ip
            hi = list(gu.shape)
            hi[-1] = 2 * Ip
            g = ttnn.silu(ttnn.slice(gu, lo, mid))
            u = ttnn.slice(gu, [0] * (nd - 1) + [Ip], hi)
            hidden = ttnn.mul(g, u)
            ttnn.deallocate(gu)
            # down: each chip already holds its [.., Ip] slice -> local matmul + all_reduce
            partial = ttnn.linear(hidden, self.down_proj_w, compute_kernel_config=kcfg)
            ttnn.deallocate(hidden)
            return ttnn.all_reduce(partial, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear)

        gate = ttnn.silu(ttnn.linear(x, self.gate_proj_w, compute_kernel_config=kcfg))
        up = ttnn.linear(x, self.up_proj_w, compute_kernel_config=kcfg)
        hidden = ttnn.mul(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        output = ttnn.linear(hidden, self.down_proj_w, compute_kernel_config=kcfg)
        ttnn.deallocate(hidden)
        return output
