# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of FFN (Feed-Forward Network) module for DeepSeek V3 dense layers.

TtFPN (TP=4) module uses the shared expert architecture with DeepSeek 671B config dimensions.
"""

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import TtSharedExpert

COMPUTE_KERNEL_CONFIG_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# DeepSeek 671B FFN dimensions
EMB_DIM = 7168
HIDDEN_DIM = 18432


class TtFfn(TtSharedExpert):
    """
    FFN module for DeepSeek V3 dense layers.

    Inherits from TtSharedExpert with hardcoded dimensions from config_671B:
        - emb_dim: 7168 (dim)
        - hidden_dim: 18432 (inter_dim)
        - activations_dtype: bfloat16 (input/output)
    """

    def __init__(
        self,
        mesh_device,
        torch_weights: dict = None,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        weights_dtype=ttnn.bfloat8_b,  # explore bfp4 ?
        compute_kernel_config: ttnn.WormholeComputeKernelConfig = COMPUTE_KERNEL_CONFIG_HIFI2,  # explore COMPUTE_KERNEL_CONFIG_LOFI with bfp4
    ):
        """
        Initialize TtFfn module.

        Args:
            mesh_device: TTNN mesh device
            torch_weights: Optional dict with keys 'gate_proj', 'up_proj', 'down_proj'
            num_links: Number of ethernet links to use for CCL (default: 1)
            topology: CCL topology - Linear or Ring (default: Linear)
            weights_dtype: Data type for weights (default: bfloat4_b)
            compute_kernel_config: Compute kernel configuration
        """
        super().__init__(
            mesh_device=mesh_device,
            emb_dim=EMB_DIM,
            hidden_dim=HIDDEN_DIM,
            torch_weights=torch_weights,
            num_links=num_links,
            topology=topology,
            activations_dtype=ttnn.bfloat16,
            weights_dtype=weights_dtype,
            compute_kernel_config=compute_kernel_config,
        )
