# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN implementation of the dots.ocr DotsVisionTransformer MLP block.

Reference: models/demos/rednote_hilab_dots.ocr/reference/functional.py
           :func:`vision_mlp_forward`

DotsSwiGLUFFN (no bias):

    h   = silu(fc1(x)) * fc3(x)      # gate=fc1, up=fc3
    out = fc2(h)                     # down

embed_dim 1536, intermediate_size 4224, use_bias = False.

The forward runs entirely with ttnn ops (linear / silu / mul); no host-side
matmul or activation. gate (fc1) and up (fc3) share the same input and output
width, so they are fused into a single [dim, 2*intermediate] linear and split on
device — one matmul instead of two — before applying SiLU to the gate half.

Reference TTNN impl this follows: models/demos/qwen25_vl/tt/vision_mlp.py
"""
import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtVisionMLP(LightweightModule):
    """dots.ocr vision SwiGLU FFN.

    Args:
        device: ttnn Device or MeshDevice.
        fc1_weight: torch.Tensor [intermediate, dim] (gate proj, no bias).
        fc3_weight: torch.Tensor [intermediate, dim] (up proj, no bias).
        fc2_weight: torch.Tensor [dim, intermediate] (down proj, no bias).
        dtype: activation/weight dtype (bf16).
    """

    def __init__(
        self,
        device,
        fc1_weight,
        fc3_weight,
        fc2_weight,
        dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None

        intermediate = fc1_weight.shape[0]
        self.intermediate = intermediate

        # Fuse gate (fc1) and up (fc3) into one [dim, 2*intermediate] linear.
        # ttnn.linear computes x @ W when W is [in, out]; pass the torch weight
        # transposed. Concatenate along the output dim: [gate | up].
        gate_t = fc1_weight.transpose(0, 1).contiguous()  # [dim, intermediate]
        up_t = fc3_weight.transpose(0, 1).contiguous()  # [dim, intermediate]
        gate_up = torch.cat([gate_t, up_t], dim=-1)  # [dim, 2*intermediate]
        self.gate_up_weight = ttnn.as_tensor(
            gate_up,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )

        self.fc2_weight = ttnn.as_tensor(
            fc2_weight.transpose(0, 1).contiguous(),  # [intermediate, dim]
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )

        # fp32 compute to match the reference float path.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [seq, dim] (TILE layout) -> [seq, dim]."""
        # Pin the gate/up split + SwiGLU elementwise chain to L1 at validation
        # grids (the [256, 4224] bf16 intermediate fits L1). At the model's real
        # document resolution (seq in the thousands) the [seq, intermediate]
        # buffer overflows L1, so fall back to DRAM above a tile-friendly
        # threshold.
        mem = ttnn.L1_MEMORY_CONFIG if x.shape[0] <= 1024 else ttnn.DRAM_MEMORY_CONFIG

        # Fused gate/up projection: [seq, dim] @ [dim, 2*intermediate].
        gate_up = ttnn.linear(
            x,
            self.gate_up_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=mem,
        )
        gate = ttnn.slice(
            gate_up,
            [0, 0],
            [gate_up.shape[0], self.intermediate],
            memory_config=mem,
        )  # fc1
        up = ttnn.slice(
            gate_up,
            [0, self.intermediate],
            [gate_up.shape[0], 2 * self.intermediate],
            memory_config=mem,
        )  # fc3

        # SwiGLU: silu(gate) * up.
        h = ttnn.mul(
            ttnn.silu(gate, memory_config=mem),
            up,
            memory_config=mem,
        )

        # Down projection: [seq, intermediate] @ [intermediate, dim].
        out = ttnn.linear(
            h,
            self.fc2_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=mem,
        )
        return out
