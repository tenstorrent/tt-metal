# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN implementation of the dots.ocr language-model Qwen2 MLP block.

Reference: models/demos/rednote_hilab_dots.ocr/reference/functional.py
           :func:`mlp_forward`

Qwen2MLP SwiGLU (no bias):

    h   = silu(gate_proj(x)) * up_proj(x)
    out = down_proj(h)

hidden_size 1536, intermediate_size 8960, no bias.

The forward runs entirely with ttnn ops (linear / silu / mul); no host-side
matmul or activation. gate_proj and up_proj share the same input and output
width, so they are fused into a single [dim, 2*intermediate] linear and split on
device — one matmul instead of two — before applying SiLU to the gate half.

Reference TTNN impl this follows: models/demos/rednote_hilab_dots.ocr/tt/vision_mlp.py
"""
import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtMLP(LightweightModule):
    """dots.ocr LM Qwen2 SwiGLU MLP.

    Args:
        device: ttnn Device or MeshDevice.
        gate_weight: torch.Tensor [intermediate, dim] (gate proj, no bias).
        up_weight: torch.Tensor [intermediate, dim] (up proj, no bias).
        down_weight: torch.Tensor [dim, intermediate] (down proj, no bias).
        dtype: activation/weight dtype (bf16).
    """

    def __init__(
        self,
        device,
        gate_weight,
        up_weight,
        down_weight,
        dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None

        intermediate = gate_weight.shape[0]
        self.intermediate = intermediate

        # Fuse gate_proj and up_proj into one [dim, 2*intermediate] linear.
        # ttnn.linear computes x @ W when W is [in, out]; pass the torch weight
        # transposed. Concatenate along the output dim: [gate | up].
        gate_t = gate_weight.transpose(0, 1).contiguous()  # [dim, intermediate]
        up_t = up_weight.transpose(0, 1).contiguous()  # [dim, intermediate]
        gate_up = torch.cat([gate_t, up_t], dim=-1)  # [dim, 2*intermediate]
        self.gate_up_weight = ttnn.as_tensor(
            gate_up,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )

        self.down_weight = ttnn.as_tensor(
            down_weight.transpose(0, 1).contiguous(),  # [intermediate, dim]
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
        # Pin the gate/up split + SwiGLU elementwise chain to L1 for decode and
        # short prefills (the [seq, 8960] bf16 intermediate fits L1). At a large
        # prefill (e.g. a full-document vision prompt, seq in the thousands) that
        # intermediate is tens of MB and overflows L1, so fall back to DRAM above
        # a tile-friendly threshold. Decode (seq=1) keeps the L1 fast path.
        mem = ttnn.L1_MEMORY_CONFIG if x.shape[0] <= 1024 else ttnn.DRAM_MEMORY_CONFIG

        # Fused gate/up projection: [seq, dim] @ [dim, 2*intermediate].
        # bf8 intermediate: the [seq, 2*intermediate] gate_up tensor is the widest
        # activation in the block; emitting it bf8 halves its write AND every
        # downstream read (slice/silu/mul/down all flow bf8). The residual-stream
        # output (down, below) stays bf16. PCC headroom is ample (~0.9999).
        gate_up = ttnn.linear(
            x,
            self.gate_up_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat8_b,
            memory_config=mem,
        )
        gate = ttnn.slice(
            gate_up,
            [0, 0],
            [gate_up.shape[0], self.intermediate],
            memory_config=mem,
        )  # gate_proj
        up = ttnn.slice(
            gate_up,
            [0, self.intermediate],
            [gate_up.shape[0], 2 * self.intermediate],
            memory_config=mem,
        )  # up_proj

        # SwiGLU: silu(gate) * up.
        h = ttnn.mul(
            ttnn.silu(gate, memory_config=mem),
            up,
            memory_config=mem,
        )

        # Down projection: [seq, intermediate] @ [intermediate, dim].
        out = ttnn.linear(
            h,
            self.down_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=mem,
        )
        return out
