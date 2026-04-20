# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of FFN (Feed-Forward Network) module for DeepSeek V3 dense layers.

TtFFN (TP=4) module uses the shared expert architecture with DeepSeek 671B config dimensions.
"""

from pathlib import Path
from typing import Optional

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

    Inherits from TtSharedExpert with DeepSeek-specific default dimensions:
        - emb_dim: 7168 (dim)
        - hidden_dim: 18432 (inter_dim)
        - weights_dtype: bfloat8_b
    """

    @staticmethod
    def check_cache_complete(cache_path: Path, cache_name_prefix: str) -> bool:
        """Check if dense FFN cache is complete (delegates to TtSharedExpert)."""
        return TtSharedExpert.check_cache_complete(cache_path, cache_name_prefix)

    @staticmethod
    def build_ttnn_cache(
        torch_weights: dict,
        mesh_device: ttnn.MeshDevice,
        cache_path: Path,
        cache_name_prefix: str,
        emb_dim: int = EMB_DIM,
        hidden_dim: int = HIDDEN_DIM,
        weights_dtype: ttnn.DataType = ttnn.bfloat8_b,
    ):
        """Build TTNN cache for dense FFN (delegates to TtSharedExpert)."""
        TtSharedExpert.build_ttnn_cache(
            torch_weights=torch_weights,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            mesh_device=mesh_device,
            weights_dtype=weights_dtype,
            cache_path=cache_path,
            cache_name_prefix=cache_name_prefix,
        )

    def __init__(
        self,
        mesh_device,
        torch_weights: dict = None,
        emb_dim: int = EMB_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        activations_dtype=ttnn.bfloat16,
        weights_dtype: ttnn.DataType = ttnn.bfloat8_b,
        compute_kernel_config: ttnn.WormholeComputeKernelConfig = COMPUTE_KERNEL_CONFIG_HIFI2,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: Optional[str] = None,
    ):
        """
        Initialize TtFfn module.

        Args:
            mesh_device: TTNN mesh device
            torch_weights: Optional dict with keys 'gate_proj', 'up_proj', 'down_proj'
            emb_dim: Embedding dimension (default: 7168)
            hidden_dim: Hidden dimension (default: 18432)
            num_links: Number of ethernet links to use for CCL (default: 1)
            topology: CCL topology - Linear or Ring (default: Linear)
            activations_dtype: Data type for activations (default: bfloat16)
            weights_dtype: Data type for weights (default: bfloat8_b)
            compute_kernel_config: Compute kernel configuration
            weight_cache_path: Optional path for caching TTNN weight tensors
            cache_name_prefix: Optional prefix for cache file names
        """
        super().__init__(
            mesh_device=mesh_device,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            torch_weights=torch_weights,
            num_links=num_links,
            topology=topology,
            activations_dtype=activations_dtype,
            weights_dtype=weights_dtype,
            compute_kernel_config=compute_kernel_config,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=cache_name_prefix,
        )
