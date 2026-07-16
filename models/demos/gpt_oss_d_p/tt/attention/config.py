# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS attention configuration. Mirrors ``minimax_m3/tt/attention/config.py``
(same AttentionConfig / ProgramConfig split) with the gpt-oss-specific deltas:

- **GQA**: 64 Q-heads / 8 KV-heads, head_dim 64.
- **Full rotary** (rotary_dim == head_dim), **YaRN** scaling — no partial rotary.
- **Attention sinks**: a learned per-head logit folded into the softmax denominator
  (held in AttentionWeights, pre-scaled by 1/sqrt(head_dim); see weights.py).
- **Sliding / full alternation**: ``sliding_window`` is set PER LAYER by the caller
  from ``hf_config.layer_types`` (None => full-causal for that layer).
- **No QK-norm** (unlike M3), **no MSA / sparse** path.
- **QKV/O projections carry bias** (``attention_bias: true`` in the HF config).
"""

from dataclasses import dataclass

import ttnn


@dataclass
class AttentionConfig:
    """Core gpt-oss attention configuration."""

    hidden_size: int  # 2880
    num_heads: int  # 64 Q-heads
    num_kv_heads: int  # 8 KV-heads (GQA group = num_heads // num_kv_heads = 8)
    head_dim: int  # 64
    max_seq_len: int

    # Per-layer sliding window. Set by the caller from hf_config.layer_types:
    #   "sliding_attention" -> sliding_window (128); "full_attention" -> None.
    sliding_window: int | None = None
    # Full rotary for gpt-oss (defaults to head_dim in __post_init__).
    rotary_dim: int | None = None
    rms_norm_eps: float = 1e-5
    # softmax scale 1/sqrt(head_dim); computed if None. Sinks are stored pre-divided by this so the
    # kernel's internal scaling reproduces HF (which doesn't scale the sink). See weights.py.
    scaling: float | None = None

    # P1 SP path: AllGather K/V + single-chip SDPA (sliding + sinks). Native ring SDPA swaps in at P6.
    sequence_parallel: bool = False

    def __post_init__(self):
        if self.scaling is None:
            self.scaling = self.head_dim**-0.5
        if self.rotary_dim is None:
            self.rotary_dim = self.head_dim

    @property
    def gqa_group_size(self) -> int:
        return self.num_heads // self.num_kv_heads


@dataclass
class ProgramConfig:
    """SDPA + projection program configs. Same shape as M3's; Blackhole compute
    kernel config. Models supply chunk sizes / core grids; boilerplate is here."""

    # Prefill SDPA chunking (seq-len dependent).
    prefill_q_chunk_size_small: int = 32
    prefill_k_chunk_size_small: int = 32
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048

    # Compute config.
    math_fidelity: str = "HiFi4"
    math_approx_mode: bool = False
    fp32_dest_acc_en: bool = False
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
