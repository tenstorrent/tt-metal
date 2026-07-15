# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of FFN (Feed-Forward Network) module for DeepSeek V3 dense layers.

TtFfn (TP) inherits from TtSharedExpert for weight construction and caching, but
overrides `forward()` with the original simple flow (plain matmuls + ttnn.mul with
fused SiLU + reduce_scatter) so the dense FFN path is decoupled from the sub-device-
aware optimizations applied to TtSharedExpert on this branch.
"""

from pathlib import Path
from typing import Optional

from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import COMPUTE_KERNEL_CONFIG_HIFI2, TtSharedExpert

# DeepSeek 671B FFN dimensions
EMB_DIM = 7168
HIDDEN_DIM = 18432


class TtFfn(TtSharedExpert):
    """
    FFN module for DeepSeek V3 dense layers.

    Reuses TtSharedExpert's weight construction / caching but applies the original
    simple multi-chip forward flow (no sub-device coupling, no height-sharded
    matmul tuning):

        gate_out = x @ gate_proj
        up_out   = x @ up_proj
        activated = silu(gate_out) * up_out          (fused via ttnn.mul)
        output_full = activated @ down_proj
        output = reduce_scatter(output_full)         (only when TP > 1)
    """

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
        """Initialize TtFfn — same signature as before, no sub-device parameters."""
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
            # subdevice_id / subdevice_cores intentionally left as defaults (None) —
            # TtFfn's overridden forward() does not use them.
        )

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

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Original simple forward pass — plain matmuls, fused SiLU via ttnn.mul, and
        reduce_scatter over mesh columns. Independent of the sub-device-aware path
        in TtSharedExpert.forward().
        """
        batch_size = x.shape[0]
        logger.debug(f"Forward pass: input shape={x.shape}, batch_size={batch_size}")

        # Verify input is replicated (full emb_dim) when multiple mesh columns
        if self.mesh_device.shape[1] > 1:
            assert x.shape[-1] == self.emb_dim, (
                f"Input must be replicated (full emb_dim={self.emb_dim}), "
                f"but got sharded input with shape[-1]={x.shape[-1]}"
            )

        # Convert input to activations dtype if needed
        if x.dtype != self.activations_dtype:
            logger.warning(f"{x.dtype=} typecasting {self.activations_dtype}")
            x = ttnn.typecast(x, self.activations_dtype)

        assert (
            x.shape[-1] == self.gate_proj.shape[-2]
        ), f"Matmul shape mismatch: x[-1]={x.shape[-1]} != gate_proj[-2]={self.gate_proj.shape[-2]}"
        assert (
            x.shape[-1] == self.up_proj.shape[-2]
        ), f"Matmul shape mismatch: x[-1]={x.shape[-1]} != up_proj[-2]={self.up_proj.shape[-2]}"
        assert (
            self.gate_proj.shape[-1] == self.down_proj.shape[-2]
        ), f"Matmul shape mismatch: gate_proj[-1]={self.gate_proj.shape[-1]} != down_proj[-2]={self.down_proj.shape[-2]}"

        # Step 1: Gate projection
        gate_out = ttnn.matmul(x, self.gate_proj, compute_kernel_config=self.compute_kernel_config)
        logger.debug(f"After gate_proj matmul: {gate_out.shape}")

        # Step 2: Up projection
        up_out = ttnn.matmul(x, self.up_proj, compute_kernel_config=self.compute_kernel_config)
        logger.debug(f"After up_proj matmul: {up_out.shape}")

        # Step 3: SiLU activation and element-wise multiplication (fused)
        activated = ttnn.mul(
            gate_out,
            up_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )
        logger.debug(f"After SiLU fusion: {activated.shape}")

        # Step 4: Down projection
        output_full = ttnn.matmul(activated, self.down_proj, compute_kernel_config=self.compute_kernel_config)
        logger.debug(f"After down_proj matmul: {output_full.shape}")

        # Step 5: Reduce-scatter output across mesh columns when TP > 1
        if self.mesh_device.shape[1] > 1:
            output = ttnn.reduce_scatter(
                output_full,
                dim=-1,
                cluster_axis=1,
                num_links=self.num_links,
                topology=self.topology,
            )
        else:
            output = output_full
        logger.debug(f"After reduce_scatter: {output.shape}")

        return output
