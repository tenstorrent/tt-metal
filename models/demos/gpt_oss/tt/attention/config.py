# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import ttnn


@dataclass
class AttentionConfig:
    """Core attention configuration - model agnostic"""

    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    sliding_window: int | None = None
    scaling: float | None = None  # Computed if None

    def __post_init__(self):
        """Compute scaling factor if not provided"""
        if self.scaling is None:
            self.scaling = self.head_dim**-0.5


@dataclass
class ProgramConfig:
    """
    Base configuration for SDPA program configs.

    Models just need to specify chunk sizes and compute settings.
    The boilerplate SDPA config generation is handled automatically.
    """

    # Decode SDPA config
    decode_q_chunk_size: int = 0
    decode_k_chunk_size: int = 128

    # Prefill SDPA config
    prefill_q_chunk_size_small: int = 32
    prefill_k_chunk_size_small: int = 32
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048

    # Compute config
    math_fidelity: str = "HiFi4"
    math_approx_mode: bool = False
    fp32_dest_acc_en: bool = False
    packer_l1_acc: bool = False

    def __post_init__(self):
        """Validate configuration on creation"""
        if self.decode_q_chunk_size < 0 or self.decode_k_chunk_size <= 0:
            raise ValueError("Decode chunk sizes must be non-negative (q) and positive (k)")

        if self.prefill_q_chunk_size_small <= 0 or self.prefill_k_chunk_size_small <= 0:
            raise ValueError("Prefill small chunk sizes must be positive")

        if self.prefill_q_chunk_size_large <= 0 or self.prefill_k_chunk_size_large <= 0:
            raise ValueError("Prefill large chunk sizes must be positive")

        if self.prefill_threshold <= 0:
            raise ValueError("Prefill threshold must be positive")

        # Validate math_fidelity
        valid_fidelities = ["LoFi", "HiFi2", "HiFi3", "HiFi4"]
        if self.math_fidelity not in valid_fidelities:
            raise ValueError(f"math_fidelity must be one of {valid_fidelities}, got {self.math_fidelity}")

    def get_decode_sdpa_config(self, mesh_device) -> ttnn.SDPAProgramConfig:
        """Get SDPA config for decode mode"""
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=self.decode_q_chunk_size,
            k_chunk_size=self.decode_k_chunk_size,
            exp_approx_mode=False,
        )

    def get_prefill_sdpa_config(self, mesh_device, seq_len: int) -> ttnn.SDPAProgramConfig:
        """Get SDPA config for prefill mode based on sequence length"""
        if seq_len >= self.prefill_threshold:
            q_chunk = self.prefill_q_chunk_size_large
            k_chunk = self.prefill_k_chunk_size_large
        else:
            q_chunk = self.prefill_q_chunk_size_small
            k_chunk = self.prefill_k_chunk_size_small

        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            exp_approx_mode=False,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
        )

    def get_compute_kernel_config(self) -> ttnn.WormholeComputeKernelConfig:
        """Get compute kernel config"""
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.math_fidelity),
            math_approx_mode=self.math_approx_mode,
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
        )
