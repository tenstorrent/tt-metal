# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU MLP for Qwen3.5: down(silu(gate(x)) * up(x)).

One forward serves both deployments:
* Tensor-parallel (27B on a (1,4) mesh): w1/w3 column-parallel, w2 row-parallel,
  then tt_all_reduce (a reduce-scatter on a mesh with a 1 in its shape, e.g.
  P150x4) leaves the output fractured along the hidden dim — matching the
  fractured-residual scheme used by models/demos/qwen35_27b.
* Single device (9B): the unit-mesh "shards" are the full weights and the
  reduce is skipped, so the output keeps the full hidden dim.
"""
import ttnn
from models.tt_transformers.tt.ccl import tt_all_reduce

from .weights import load_mlp_weights


class Qwen35MLP:
    """SwiGLU feed-forward network for Qwen3.5.

    Built per layer by tt/layer.py and directly by the MLP tests
    (tests/test_mlp_tp.py, tests/unit/test_mlp.py, tests/unit/test_component_pcc.py).
    args and tt_ccl are only needed on a multi-device mesh, where the reduce runs.
    """

    def __init__(self, mesh_device, state_dict, tensor_cache_path=None, args=None, tt_ccl=None):
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.num_devices = mesh_device.get_num_devices()
        self.weights = load_mlp_weights(mesh_device, state_dict, tensor_cache_path)
        # LoFi is enough fidelity here (the bfloat4_b/bfloat8_b weights dominate
        # the error budget) and fp32 dest accumulation keeps the long-k reductions
        # accurate. Decode additionally turns on packer L1 accumulation.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def forward(self, x):
        """Input x is replicated (full hidden dim) on every device. On TP the
        output comes back fractured along the hidden dim (reduce-scatter); on a
        single device it keeps the full hidden dim."""
        # Single-device activations are [B, seq, dim], so shape[1] is the seq
        # len and 1 means decode. TP activations are [1, 1, rows, dim], so the
        # decode config — validated for every TP shape — is always picked there.
        T = x.shape[1] if len(x.shape) >= 3 else 1
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config

        # Interleaved weights → let ttnn auto-select the matmul program (serves
        # both decode and prefill). SILU applied separately, then gate * up.
        mc = ttnn.DRAM_MEMORY_CONFIG
        w1_out = ttnn.linear(x, self.weights.w1, compute_kernel_config=ckc, memory_config=mc)
        w3_out = ttnn.linear(x, self.weights.w3, compute_kernel_config=ckc, memory_config=mc)
        w1_act = ttnn.silu(w1_out, memory_config=mc)
        ttnn.deallocate(w1_out)
        hidden = ttnn.mul(w1_act, w3_out, memory_config=mc)
        ttnn.deallocate(w1_act)
        ttnn.deallocate(w3_out)
        out = ttnn.linear(hidden, self.weights.w2, compute_kernel_config=ckc, memory_config=mc)
        ttnn.deallocate(hidden)

        # A unit mesh has nothing to reduce — and single-device callers (e.g.
        # tests/unit/test_component_pcc.py) construct the MLP without args, so
        # args.ccl_topology() below must not be evaluated.
        if self.num_devices == 1:
            return out

        # On a (1,4) mesh tt_all_reduce reduce-scatters, leaving the output
        # fractured along the hidden dim (dim=3).
        return tt_all_reduce(
            out,
            self.device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
