# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 damp gate on device (A1): x_k = x_{k-1} + beta * (h_k - x_{k-1}),
beta = sigmoid(RMSNorm(x_{k-1}; gain, eps) @ W^T + b).

Reference: ``reference.glm_5_4.ops.damp_gate_reference``. The gate reads the pass input x_{k-1} (A1c), has a bias
(A1d) and its width comes from the Linear weight W [width, hidden] (A1a): width 1 gives one beta per token, width
hidden one per channel. Both widths take one code path; only W's placement differs:

    x_{k-1} [S, H/TP] --RMSNorm(gain), distributed over TP--> --TP all_gather--> [S, H] @ W^T --> + b --> sigmoid
        --> beta [S, w]
    x_k = x_{k-1} + beta * (h - x_{k-1})          (fp32, one bf16 rounding; beta broadcasts when w = 1)

W^T [H, width] is replicated over TP when width is 1 (every TP chip computes the token's beta) and column-sharded
over TP when width is hidden (each chip computes the betas of its own H/TP channels, matching the TP-sharded
activations). Activations are [1, 1, S_local, H/TP], SP-sharded rows and TP-sharded hidden, as between blocks.

Weights come from host bf16 tensors (A8) and are not written to a TTNN weight cache: there is no GLM-5.4 cache
until the official weights are published (user decision 2026-11-23).
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import mesh_shard_dims
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm

GATE_COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


class TtDampGate(LightweightModule):
    """One loop pass's damp gate. ``weights``: ``{"norm_weight" [H], "proj_weight" [width, H], "proj_bias"
    [width]}`` (``synthetic_gates`` form), width 1 or H."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        weights: dict,
        *,
        eps: float,
        tp_axis: int,
        num_links: int,
        topology,
    ):
        super().__init__()
        norm_weight, proj_weight, proj_bias = (weights[k] for k in ("norm_weight", "proj_weight", "proj_bias"))
        hidden = norm_weight.shape[0]
        self.width = proj_weight.shape[0]
        if norm_weight.dim() != 1 or proj_weight.shape != (self.width, hidden) or self.width not in (1, hidden):
            raise ValueError(
                f"gate weight must be [1, {hidden}] or [{hidden}, {hidden}] with norm [{hidden}], got "
                f"{tuple(proj_weight.shape)} / {tuple(norm_weight.shape)}"
            )
        if proj_bias.shape != (self.width,):
            raise ValueError(f"gate bias must be [{self.width}], got {tuple(proj_bias.shape)}")
        self.mesh_device = mesh_device
        self.eps = eps
        self.tp_axis = tp_axis
        self.tp = mesh_device.shape[tp_axis]
        self.num_links = num_links
        self.topology = topology
        if self.width % self.tp and self.width != 1:
            raise ValueError(f"per-channel gate width {self.width} does not split over TP {self.tp}")

        mesh_shape = tuple(mesh_device.shape)
        # Width 1: every TP chip needs the whole column. Width H: chip t owns columns [t*H/TP, (t+1)*H/TP), the
        # channels of its activation shard.
        if self.width == 1:
            mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        else:
            mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=mesh_shard_dims(-1, tp_axis))
        self.proj_weight_t = ttnn.from_torch(
            proj_weight.to(torch.bfloat16).t().contiguous().reshape(1, 1, hidden, self.width),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        self.proj_bias = ttnn.from_torch(
            proj_bias.float().reshape(1, 1, 1, self.width),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        # The norm runs on the TP-sharded input, as the blocks' norms do (a full-width rms_norm does not fit L1).
        # No weight cache (see the module docstring).
        self.norm = TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=hidden,
            epsilon=eps,
            torch_weight=norm_weight.to(torch.bfloat16),
            cluster_axis=tp_axis,
            num_links=num_links,
            topology=topology,
        )

    def beta(self, x_prev: ttnn.Tensor) -> ttnn.Tensor:
        """beta [1, 1, S_local, width per chip] float32 for the pass input ``x_prev`` [1, 1, S_local, H/TP]."""
        normed = self.norm(x_prev)
        if self.tp > 1:
            gathered = ttnn.all_gather(
                normed, dim=-1, cluster_axis=self.tp_axis, num_links=self.num_links, topology=self.topology
            )
            ttnn.deallocate(normed)
            normed = gathered
        logits = ttnn.matmul(
            normed,
            self.proj_weight_t,
            dtype=ttnn.float32,
            compute_kernel_config=GATE_COMPUTE_CONFIG,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(normed)
        biased = ttnn.add(logits, self.proj_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(logits)
        beta = ttnn.sigmoid(biased, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(biased)
        return beta

    def forward(self, x_prev: ttnn.Tensor, h: ttnn.Tensor) -> ttnn.Tensor:
        """x_k [1, 1, S_local, H/TP] bf16 from the pass input ``x_prev`` and the pass output ``h`` (same shape)."""
        if tuple(x_prev.shape) != tuple(h.shape):
            raise ValueError(f"pass output {tuple(h.shape)} != pass input {tuple(x_prev.shape)}")
        beta = self.beta(x_prev)
        x32 = ttnn.typecast(x_prev, ttnn.float32)
        h32 = ttnn.typecast(h, ttnn.float32)
        diff = ttnn.subtract(h32, x32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(h32)
        step = ttnn.multiply(diff, beta, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(diff)
        ttnn.deallocate(beta)
        out32 = ttnn.add(x32, step, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x32)
        ttnn.deallocate(step)
        out = ttnn.typecast(out32, ttnn.bfloat16)
        ttnn.deallocate(out32)
        return out
