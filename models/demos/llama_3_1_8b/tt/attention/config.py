# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Llama 3.1 8B attention configuration (AttentionConfig + ProgramConfig).

Key points:
- **GQA**: 32 Q-heads / 8 KV-heads, head_dim 128 (num_kv_groups = 4).
- **Full rotary** (rotary_dim == head_dim == 128), **Llama3** frequency scaling — no partial rotary.
- **No attention sinks**, **no sliding window** — all layers are full-causal.
- **No QK-norm**, **no bias** (attention_bias: false).
- Uses HiFi4 with fp32_dest_acc_en=True as the default compute config.
"""

from dataclasses import dataclass

import ttnn


@dataclass
class AttentionConfig:
    """Core Llama 3.1 8B attention configuration."""

    hidden_size: int  # 4096
    num_heads: int  # 32 Q-heads
    num_kv_heads: int  # 8 KV-heads (GQA group = num_heads // num_kv_heads = 4)
    head_dim: int  # 128
    max_seq_len: int
    rms_norm_eps: float = 1e-5
    # softmax scale 1/sqrt(head_dim); computed if None.
    scaling: float | None = None

    # SP prefill path
    sequence_parallel: bool = False

    def __post_init__(self):
        if self.scaling is None:
            self.scaling = self.head_dim**-0.5

    @property
    def num_kv_groups(self) -> int:
        """Number of Q heads per KV head group."""
        return self.num_heads // self.num_kv_heads


@dataclass
class ProgramConfig:
    """SDPA + projection program configs + Blackhole compute kernel config.
    Uses HiFi4 with fp32_dest_acc_en=True for bring-up accuracy."""

    # Prefill SDPA chunking (seq-len dependent).
    prefill_q_chunk_size_small: int = 32
    prefill_k_chunk_size_small: int = 32
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048

    # Compute config — HiFi4 + fp32_dest_acc_en for bring-up accuracy.
    math_fidelity: str = "HiFi4"
    math_approx_mode: bool = False
    fp32_dest_acc_en: bool = True
    packer_l1_acc: bool = False

    def __post_init__(self):
        if (
            min(
                self.prefill_q_chunk_size_small,
                self.prefill_k_chunk_size_small,
                self.prefill_q_chunk_size_large,
                self.prefill_k_chunk_size_large,
                self.prefill_threshold,
            )
            <= 0
        ):
            raise ValueError("SDPA chunk sizes and threshold must be positive")
        valid_fidelities = ["LoFi", "HiFi2", "HiFi3", "HiFi4"]
        if self.math_fidelity not in valid_fidelities:
            raise ValueError(f"math_fidelity must be one of {valid_fidelities}, got {self.math_fidelity}")

    def get_prefill_sdpa_config(self, mesh_device, seq_len: int) -> ttnn.SDPAProgramConfig:
        if seq_len >= self.prefill_threshold:
            q_chunk, k_chunk = self.prefill_q_chunk_size_large, self.prefill_k_chunk_size_large
        else:
            q_chunk, k_chunk = self.prefill_q_chunk_size_small, self.prefill_k_chunk_size_small
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            exp_approx_mode=False,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
        )

    def get_compute_kernel_config(self):
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.math_fidelity),
            math_approx_mode=self.math_approx_mode,
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
        )
