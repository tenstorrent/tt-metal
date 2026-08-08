# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
SwiGLU MLP for Qwen3.5: down(silu(gate(x)) * up(x)).

For HuggingFace Reference, refer to:
`Qwen3_5MLP` in `transformers.models.qwen3_5.modeling_qwen3_5`

"""
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_reduce

from .weights import load_mlp_weights


class Qwen35MLP(LightweightModule):
    """SwiGLU MLP for Qwen3.5: down(silu(gate(x)) * up(x))."""

    def __init__(self, mesh_device, state_dict, args, tensor_cache_path=None, tt_ccl=None):
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.num_devices = mesh_device.get_num_devices()
        self.weights = load_mlp_weights(mesh_device, state_dict, tensor_cache_path)

    def forward(self, x):
        """Input x is replicated (full hidden dim) on every device. On TP the
        output comes back fractured along the hidden dim (reduce-scatter); on a
        single device it keeps the full hidden dim."""

        tt_ccl, mesh_device, topology = self.tt_ccl, self.device, self.args.ccl_topology()
        w1, w2, w3 = self.weights.w1, self.weights.w2, self.weights.w3

        out = (ttnn.silu(x @ w1) * (x @ w3)) @ w2

        # On a (1,4) mesh tt_all_reduce reduce-scatters, leaving the output
        # fractured along the hidden dim (dim=-1).
        return tt_all_reduce(
            out,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            dim=-1,
            topology=topology,
        )
