# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 attention configuration (AttentionConfig + ProgramConfig split).

Verified against ``transformers/models/ministral3/modular_ministral3.py`` (v5.12.1), not just the
config JSON:

- **GQA**: 96 Q-heads / 8 KV-heads, head_dim 128 (``n_q * head_dim == hidden_size``). At the TP=4
  target that is 24 Q + **2 KV** heads per chip — see ``config.py`` and ``kv_cache.py``.
- **Full rotary** (rotary_dim == head_dim == 128), **YaRN** scaling baked into the host cos/sin.
- **No projection bias** — ``Ministral3Attention`` inherits ``MistralAttention``, whose q/k/v/o
  ``nn.Linear`` are all ``bias=False``.
- **No QK-norm**, **no attention sinks**, **no sparse/MSA path**.
- **Dense causal on every layer**: ``sliding_window`` is ``null`` in the config, and
  ``Ministral3Attention`` forwards ``getattr(config, "sliding_window", None)`` straight through.
- ``scaling = head_dim ** -0.5``, exactly as ``MistralAttention.__init__``.

**One mechanism that no config field advertises.** ``Ministral3Attention.forward`` applies a
Llama-4 style position-dependent temperature to Q *after* RoPE::

    q *= 1 + beta * log(1 + floor(pos / original_max_position_embeddings))

with ``beta = rope_parameters["llama_4_scaling_beta"]``. This checkpoint ships ``beta = 0``, which
makes the factor exactly 1.0 — but the ``Ministral3Config`` class default is ``0.1``. If a refreshed
checkpoint enables it, this becomes a real per-position multiply on Q and NOT implementing it is
silently wrong. :attr:`llama4_scaling_beta` carries the value so the loader can assert it is zero;
see ``model_config.ModelArgs._assert_supported``.
"""

from dataclasses import dataclass

import ttnn


@dataclass
class AttentionConfig:
    """Core Mistral-Medium-3.5 attention configuration."""

    hidden_size: int  # 12288  (3072/chip at TP=4)
    num_heads: int  # 96 Q-heads  (24/chip at TP=4)
    num_kv_heads: int  # 8 KV-heads (GQA group = 12; 2/chip at TP=4)
    head_dim: int  # 128
    max_seq_len: int

    # Full rotary (defaults to head_dim in __post_init__). Present so a partial-rotary variant is a
    # config change rather than a code change; the forward path asserts they are equal.
    rotary_dim: int | None = None
    rms_norm_eps: float = 1e-5
    # softmax scale 1/sqrt(head_dim); computed if None.
    scaling: float | None = None
    # Llama-4 Q temperature beta; must be 0 for this checkpoint (see module docstring).
    llama4_scaling_beta: float = 0.0

    # SP prefill path: cache-backed RingJointSDPA over the block-cyclic SP KV cache.
    sequence_parallel: bool = False

    def __post_init__(self):
        if self.scaling is None:
            self.scaling = self.head_dim**-0.5
        if self.rotary_dim is None:
            self.rotary_dim = self.head_dim
        if self.rotary_dim != self.head_dim:
            raise NotImplementedError(
                f"mistral_medium_d_p implements FULL rotary only (rotary_dim == head_dim); "
                f"got rotary_dim={self.rotary_dim}, head_dim={self.head_dim}. A partial-rotary "
                "variant needs the slice/concat wrapper from minimax_m3/tt/attention/operations.py."
            )
        if self.num_heads % self.num_kv_heads:
            raise ValueError(f"num_heads {self.num_heads} not divisible by num_kv_heads {self.num_kv_heads}")

    @property
    def gqa_group_size(self) -> int:
        return self.num_heads // self.num_kv_heads


@dataclass
class ProgramConfig:
    """SDPA + compute-kernel program configs. Same structure as ``gpt_oss_d_p``; the chunk sizes are
    the only tuned numbers and are re-tunable per shape without touching the model code."""

    prefill_q_chunk_size_small: int = 32
    prefill_k_chunk_size_small: int = 32
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048

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
