# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS MoE router: logits = x @ Wr.T + br over the 128 experts, top-4 FIRST, softmax over the 4 chosen logits.

Every chip routes its own rows of the residual stream (plan.md step 5: [1, 1, S/8, 2880] per chip, replicated over the
TP columns, which compute the same), so there is no collective. Router weight [2880, 128] and bias replicated, bf16 by
default (ROUTER_WEIGHT_DTYPE). Logits in fp32 (HiFi4, fp32 dest): top-4 ties are precision-bound.

``route`` returns the sparse pair the experts step consumes (indices [.., S/8, 4] uint16 ROW_MAJOR, weights
[.., S/8, 4] fp32 TILE); ``__call__`` scatters it into the dense routing weights [1, 1, S/8, 128] the block graph
carries (0 off the chosen experts). ttnn.scatter has no fp32 TILE path, so the scatter runs in fp32 ROW_MAJOR onto a
zeros table built once at load.
"""

import torch

import ttnn

from .settings import settings


class TtRouter:
    def __init__(self, mesh, weight: torch.Tensor, bias: torch.Tensor, top_k: int):
        self.mesh, self.top_k = mesh, int(top_k)
        self.n_experts = int(weight.shape[0])
        wdtype = getattr(ttnn, settings.get("ROUTER_WEIGHT_DTYPE"))
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self.weight = ttnn.from_torch(
            weight.T.contiguous().float(),  # [hidden, experts]
            device=mesh,
            dtype=wdtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.bias = ttnn.from_torch(
            bias.reshape(1, -1).float(),
            device=mesh,
            dtype=wdtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, settings.get("ROUTER_FIDELITY")),
            math_approx_mode=False,
            fp32_dest_acc_en=settings.get("ROUTER_FP32_ACC"),
            packer_l1_acc=False,
        )
        self._zeros = {}  # rows per chip -> fp32 ROW_MAJOR zeros [1, 1, rows, experts] (prepare, at load)

    def prepare(self, rows: int):
        """Build the zeros table for `rows` local rows at load time (the forward never calls from_torch)."""
        if rows not in self._zeros:
            self._zeros[rows] = ttnn.from_torch(
                torch.zeros(1, 1, rows, self.n_experts),
                device=self.mesh,
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

    def route(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """x [1, 1, S, 2880] bf16 -> (indices [1, 1, S, k] uint16 ROW_MAJOR, weights [1, 1, S, k] fp32 TILE)."""
        logits = ttnn.linear(
            x,
            self.weight,
            bias=self.bias,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute,
        )
        vals, idx = ttnn.topk(logits, k=self.top_k, dim=-1, largest=True, sorted=True)
        logits.deallocate(True)
        weights = ttnn.softmax(vals, dim=-1, numeric_stable=True, compute_kernel_config=self.compute)
        vals.deallocate(True)
        idx_rm = ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT)
        idx.deallocate(True)
        return idx_rm, weights

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1, 1, S, 2880] bf16 -> dense routing weights [1, 1, S, 128] fp32 ROW_MAJOR."""
        idx, weights = self.route(x)
        w_rm = ttnn.to_layout(weights, ttnn.ROW_MAJOR_LAYOUT)
        weights.deallocate(True)
        rows = x.shape[-2]
        assert rows in self._zeros, f"TtRouter.prepare({rows}) was not called at load"
        out = ttnn.scatter(self._zeros[rows], -1, idx, w_rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        idx.deallocate(True)
        w_rm.deallocate(True)
        return out
