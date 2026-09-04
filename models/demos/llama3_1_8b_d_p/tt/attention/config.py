# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B attention configuration (AttentionConfig + ProgramConfig split).

Copied from ``gpt_oss_d_p/tt/attention/config.py``; the gpt-oss-specific features are removed
rather than defaulted off, so a caller cannot accidentally enable something Llama does not have:

- **GQA**: 32 Q-heads / 8 KV-heads (group 4), head_dim 128.
- **Full rotary** (rotary_dim == head_dim), **llama3** scaling — no partial rotary.
- **No attention sinks**, **no sliding window**, **no logit softcap**, **no QK-norm**, no MSA/sparse
  path — every layer is identical full-causal attention (spec ``attention.features`` is empty and
  ``layer_schedule`` is null).
- **No projection bias** (``attention_bias: false``) — unlike gpt-oss, which biases q/k/v/o.

SDPA chunk sizes are the tt_transformers Llama-family rule the spec records: 256/256 at or above a
2048 threshold, 64/64 below.
"""

from dataclasses import dataclass

import ttnn


@dataclass
class AttentionConfig:
    """Core Llama 3.1 8B attention configuration."""

    hidden_size: int  # 4096
    num_heads: int  # 32 Q-heads
    num_kv_heads: int  # 8 KV-heads (GQA group = 4)
    head_dim: int  # 128
    max_seq_len: int

    rotary_dim: int | None = None  # full rotary; defaults to head_dim
    rms_norm_eps: float = 1e-5
    # softmax scale 1/sqrt(head_dim); computed if None.
    scaling: float | None = None

    # SP prefill path: cache-backed RingJointSDPA over the block-cyclic KV cache.
    sequence_parallel: bool = False

    def __post_init__(self):
        if self.scaling is None:
            self.scaling = self.head_dim**-0.5
        if self.rotary_dim is None:
            self.rotary_dim = self.head_dim
        assert self.rotary_dim == self.head_dim, "Llama 3.1 is full-rotary; partial rotary is not supported here"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads"

    @property
    def gqa_group_size(self) -> int:
        return self.num_heads // self.num_kv_heads


@dataclass
class ProgramConfig:
    """SDPA program configs + Blackhole compute kernel config.

    Chunk sizes and the threshold are the spec's ``attention.q_chunk_size`` / ``k_chunk_size``
    (256/256 above 2048, 64/64 below — the tt_transformers Llama-family rule). Accumulation settings
    are the spec's ``numerics.accumulation``: HiFi4, fp32_dest_acc_en False, packer_l1_acc False.
    """

    prefill_q_chunk_size_small: int = 64
    prefill_k_chunk_size_small: int = 64
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048

    math_fidelity: str = "HiFi4"
    math_approx_mode: bool = False
    # fp32_dest_acc_en MUST stay False: the ring SDPA's streaming-softmax compute requires it.
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
