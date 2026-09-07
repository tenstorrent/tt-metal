# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral-Medium-3.5 attention configuration (AttentionConfig + ProgramConfig split). Adapted from
``gpt_oss_d_p/tt/attention/config.py`` with everything gpt-oss has and Mistral does not removed.

- **Dense GQA**: 96 Q-heads / 8 KV-heads (group = 12), head_dim 128.
- **Full rotary** (rotary_dim == head_dim == 128), **YaRN** scaling baked into the cos/sin.
- **No attention sinks** — gpt-oss's learned per-head softmax-denominator logit has no counterpart
  in ``Ministral3Attention``, so there is no ``sinks`` tensor and no sink plumbing.
- **No sliding window** — ``config.sliding_window`` is ``null`` for Mistral, so EVERY layer is
  full-causal. gpt-oss's per-layer ``layer_types`` alternation is gone, and with it the dual
  gather-buffer sizing in ``dense_sp._gather_seq_len``: only the full-sequence branch remains.
- **No QK-norm**, **no MSA / sparse** path.
- **No projection bias** — ``Ministral3Attention`` builds q/k/v/o with ``bias=False``, unlike
  gpt-oss's ``attention_bias: true``, so the fused QKV bias and the o_proj bias are both dropped.
"""

from dataclasses import dataclass

import ttnn


@dataclass
class AttentionConfig:
    """Core Mistral-Medium-3.5 attention configuration."""

    hidden_size: int  # 12288
    num_heads: int  # 96 Q-heads
    num_kv_heads: int  # 8 KV-heads (GQA group = num_heads // num_kv_heads = 12)
    head_dim: int  # 128
    max_seq_len: int

    # Full rotary for Mistral (defaults to head_dim in __post_init__).
    rotary_dim: int | None = None
    rms_norm_eps: float = 1e-5
    # softmax scale 1/sqrt(head_dim); computed if None.
    scaling: float | None = None

    # SP prefill path: cache-backed RingJointSDPA from chunk 0 onward.
    sequence_parallel: bool = False

    def __post_init__(self):
        if self.scaling is None:
            self.scaling = self.head_dim**-0.5
        if self.rotary_dim is None:
            self.rotary_dim = self.head_dim
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({self.num_heads}) must be a multiple of num_kv_heads ({self.num_kv_heads})")

    @property
    def gqa_group_size(self) -> int:
        return self.num_heads // self.num_kv_heads


@dataclass
class ProgramConfig:
    """SDPA + projection program configs + the Blackhole compute kernel config.
    Models supply chunk sizes / core grids; boilerplate is here."""

    # Prefill SDPA chunking (seq-len dependent).
    prefill_q_chunk_size_small: int = 32
    prefill_k_chunk_size_small: int = 32
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048

    # Ring (cache-backed SP) SDPA chunking. Separate from the one-shot sizes above because the ring
    # op's Q slab is one chunk while its K/V slab is the whole cache.
    ring_q_chunk_size: int = 128
    ring_k_chunk_size: int = 128

    # Compute config. fp32 destination accumulation is ON by default, which is a change from the
    # gpt-oss donor's False. The donor's constraint is real but narrow: the RING cache-read op's
    # streaming online-softmax compute requires fp32_dest_acc_en=False, and
    # get_ring_compute_kernel_config hard-codes that. The PLAIN SDPA has no such constraint, and at
    # this model's sequence lengths the accumulation dtype is worth several points of KV PCC.
    # Measured, one-shot 5120 tokens, per-layer K vs the fp32 oracle:
    #     fp32_dest_acc_en=False   L0 0.99994  L1 0.99545  L2 0.98083  L3 0.95713
    #     fp32_dest_acc_en=True    L0 0.99994  L1 0.99800  L2 0.99101  L3 0.97771
    math_fidelity: str = "HiFi4"
    math_approx_mode: bool = False
    fp32_dest_acc_en: bool = True
    packer_l1_acc: bool = False

    def __post_init__(self):
        sizes = (
            self.prefill_q_chunk_size_small,
            self.prefill_k_chunk_size_small,
            self.prefill_q_chunk_size_large,
            self.prefill_k_chunk_size_large,
            self.prefill_threshold,
            self.ring_q_chunk_size,
            self.ring_k_chunk_size,
        )
        if min(sizes) <= 0:
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

    def get_ring_sdpa_config(self, mesh_device) -> ttnn.SDPAProgramConfig:
        """Program config for the cache-backed ring SDPA.

        The ring op requires its CCL workers and its compute cores to be disjoint, and the CCL
        workers live in the LAST compute column (see ``ccl.ring_attention_ccl_core_grid_offset``),
        so carve that column out of the compute grid here.
        """
        grid = mesh_device.compute_with_storage_grid_size()
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
            q_chunk_size=self.ring_q_chunk_size,
            k_chunk_size=self.ring_k_chunk_size,
            exp_approx_mode=False,
        )

    def get_compute_kernel_config(self):
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.math_fidelity),
            math_approx_mode=self.math_approx_mode,
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
        )

    def get_ring_compute_kernel_config(self, mesh_device):
        """Compute config for the ring SDPA. ``fp32_dest_acc_en`` MUST stay False here — the ring
        op's streaming online-softmax compute requires it — which is why it is hard-coded rather
        than read off the dataclass field (that field defaults to True for the plain SDPA)."""
        return ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, self.math_fidelity),
            math_approx_mode=self.math_approx_mode,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
