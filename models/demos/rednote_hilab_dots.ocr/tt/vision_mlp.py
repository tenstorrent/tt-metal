# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN vision MLP (SwiGLU FFN) for dots.ocr.

DotsSwiGLUFFN (modeling_dots_vision): ``fc2(silu(fc1(x)) * fc3(x))``,
1536 -> 4224 -> 1536, no biases. Identical SwiGLU pattern to reference_impl
models/demos/qwen25_vl/tt/vision_mlp.py (w1=gate/fc1, w3=up/fc3, w2=down/fc2).

TTNN mapping: two sibling ``ttnn.linear`` branches sharing the input, then a
single fused elementwise ``ttnn.mul(gate, up,
input_tensor_a_activations=[ttnn.UnaryOpType.SILU])`` (KB ttnn_mul_1, the
qwen25_vl reference idiom — silu computed inside the BinaryNg kernel), then
the down ``ttnn.linear``. Applied in the optimization phase: it removes the
standalone 110-core silu kernel pass measured at ~99 us (~14% of block kernel
time) at the production fp32 [1,1,896,1536] operating point.

Parallelism plan (ARCHITECTURE.md): vision tower placement=replicate — all
three weights are ``ReplicateTensorToMesh`` on the 1x4 mesh, activations stay
replicated, no CCL. On a single device the mesh_mapper degenerates gracefully.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtVisionMLP(LightweightModule):
    """dots.ocr vision SwiGLU FFN: fc2(silu(fc1(x)) * fc3(x)), no biases.

    Args:
        mesh_device: ttnn mesh device handle (weights replicated).
        state_dict: {"fc1.weight": [hidden, dim], "fc2.weight": [dim, hidden],
            "fc3.weight": [hidden, dim]} torch tensors (HF keys
            vision_tower.blocks.N.mlp.*).
        dtype: on-device weight dtype.
    """

    def __init__(self, mesh_device, state_dict, dtype=ttnn.bfloat16):
        super().__init__()
        self.mesh_device = mesh_device

        # Transpose [out, in] -> [in, out] for x @ W^T.
        as_weight = lambda name: ttnn.from_torch(
            state_dict[name].transpose(-2, -1).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.w1 = as_weight("fc1.weight")  # gate: [dim, hidden]
        self.w3 = as_weight("fc3.weight")  # up:   [dim, hidden]
        self.w2 = as_weight("fc2.weight")  # down: [hidden, dim]

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [..., dim] TILE_LAYOUT, replicated across the mesh.

        Returns: [..., dim], replicated.
        """
        gate = ttnn.linear(
            x,
            self.w1,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            x,
            self.w3,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Fused silu(gate) * up in one BinaryNg kernel (KB ttnn_mul_1).
        h = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(
            h,
            self.w2,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(h)
        return out
