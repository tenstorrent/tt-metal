# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V2-VISION-V2 (3/N): qwen3.6 vision MLP with REAL TP=8 across mesh row axis.

Built on tt_dit primitives (`ColParallelLinear` + `RowParallelLinear`) — these
clamp the matmul core grid to BH Galaxy's allowed (11, 10) automatically and
pick a per-chip program config dynamically. So we avoid the qwen3_vl/tt/vision_mlp.py
trap (hardcoded Wormhole-grid matmul configs that fail on BH).

Topology on the BH GLX (8, 4) parent mesh:
  - TP=8 across cluster_axis=0 (rows): K dim of fc1/fc2 sharded across 8 chips
  - DP=4 across cluster_axis=1 (cols): one frame per col (optional; for V2
    PCC start with replicated on this axis, expand later)

The vision MLP shape:
  fc1: [hidden=1152, intermediate=4304], bias=True, GELU
  fc2: [intermediate=4304, hidden=1152], bias=True
  Activation: gelu_pytorch_tanh

The fc1 is ColParallelLinear: per-chip weight [1152, 4304/8=538]; output fractured.
The fc2 is RowParallelLinear: per-chip weight [4304/8=538, 1152]; output reduce-scattered to [B, S, 144].
After fc2: one all_gather on cluster_axis=0 restores [B, S, 1152] replicated for the next block.
"""

from __future__ import annotations

import torch

import ttnn
from models.tt_dit.layers.linear import ColParallelLinear, RowParallelLinear
from models.tt_dit.layers.module import Module
from models.tt_dit.parallel.manager import CCLManager


class Qwen36VisionMlpTP(Module):
    """Single qwen3.6 vision MLP layer with TP=8 on cluster_axis=0.

    Constructor takes the HF state-dict subset for one layer's MLP (4 tensors:
    linear_fc1.{weight,bias}, linear_fc2.{weight,bias}) and wires them into
    tt_dit's parallel linear layers.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        state_dict: dict[str, torch.Tensor],
        *,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        tp_mesh_axis: int = 0,
        dtype: ttnn.DataType = ttnn.bfloat8_b,
        state_dict_prefix: str = "",
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.tp_mesh_axis = tp_mesh_axis

        # fc1: 1152 -> 4304, GELU.  ColParallelLinear shards out_features (4304)
        # across cluster_axis=0 → per chip out=538. Input replicated on this axis.
        self.fc1 = ColParallelLinear(
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=True,
            activation_fn="gelu",  # fused via UnaryOpType.GELU
            dtype=dtype,
            mesh_device=mesh_device,
            mesh_axis=tp_mesh_axis,
        )
        # fc2: 4304 -> 1152.  RowParallelLinear shards in_features (4304)
        # across cluster_axis=0 → per chip in=538. Input MUST be fractured on
        # the matching axis (which it is, coming out of fc1).
        # Output is reduce_scattered along the OUTPUT dim. We expand back via
        # all_gather to give the next block a replicated input.
        self.fc2 = RowParallelLinear(
            in_features=intermediate_size,
            out_features=hidden_size,
            bias=True,
            dtype=dtype,
            mesh_device=mesh_device,
            mesh_axis=tp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        # Load weights. tt_dit's Module.load_state_dict takes care of the
        # mesh-sharded write via the Parameter abstraction. We strip the
        # provided prefix so the keys match "fc1.weight" / "fc2.weight" /
        # "fc1.bias" / "fc2.bias" expected by the Linear layers.
        clean: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            k2 = k[len(state_dict_prefix) :] if state_dict_prefix and k.startswith(state_dict_prefix) else k
            # HF qwen3_vl convention: "linear_fc1" / "linear_fc2"; tt_dit attribute names are fc1/fc2
            k2 = k2.replace("linear_fc1.", "fc1.")
            k2 = k2.replace("linear_fc2.", "fc2.")
            clean[k2] = v
        # tt_dit's Module.load_state_dict accepts nested {child.attr_name: tensor} (returns None or IncompatibleKeys).
        self.load_state_dict(clean)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Run the MLP. Input is replicated on tp_mesh_axis; output is replicated again.

        Internal flow (TP=8 example, hidden=1152 intermediate=4304):
          x         shape [B, S, 1152]   (replicated on rows)
          → fc1                          (per-chip weight [1152, 538], replicated input)
          → activations [B, S, 538]      (output fractured along rows)
          → fc2                          (per-chip weight [538, 1152], fractured input)
          → reduce_scatter → [B, S, 144] (fractured on output dim)
          → all_gather → [B, S, 1152]    (replicated again)
        """
        y = self.fc1.forward(x)  # GELU fused
        z = self.fc2.forward(y)
        ttnn.deallocate(y)

        # fc2 reduce-scattered output along output dim (dim=3 since outputs are 4D).
        # Restore replicated via all_gather on the same mesh_axis.
        needs_unsqueeze = len(z.shape) <= 3
        if needs_unsqueeze:
            z = ttnn.unsqueeze(z, 0)
        z = self.ccl_manager.all_gather_persistent_buffer(z, dim=3, mesh_axis=self.tp_mesh_axis)
        if needs_unsqueeze:
            z = ttnn.squeeze(z, 0)
        return z
