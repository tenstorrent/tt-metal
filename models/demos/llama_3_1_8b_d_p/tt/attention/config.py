# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention shape + program configuration for Llama-3.1-8B prefill.

Borrowed structurally from `minimax_m3/tt/attention/config.py` with the M3-specific fields removed
(partial rotary, QK-norm, Gemma norm, the whole MSA sparse block) — Llama has none of them.

## The numerics defaults are deliberately NOT the donor's

The donor's `ProgramConfig` defaults `fp32_dest_acc_en=False` and `packer_l1_acc=False`, and that is
a shape-tuned value, not a structural one: it encodes the donor's L1 budget and its op mix. The
bring-up default here is **HiFi4 with `fp32_dest_acc_en=True`** for every projection matmul and for
the plain (non-ring) SDPA, because the destination accumulation dtype dominates accuracy — bf16
accumulation over a deep contraction caps a decoder layer around PCC 0.989 while fp32 lifts it to
~0.996, and weight dtype and math fidelity barely matter next to it.

The one place `fp32_dest_acc_en=False` is kept is the **ring** SDPA
(`ring_joint_scaled_dot_product_attention`). That is local to the ring op — it is not a property of
SDPA in general, and a donor that sets it globally is over-broad — and here it was MEASURED rather
than inherited:

* On the cache-read path it is not a preference but a hard kernel constraint. Setting it True fails
  with `kv_actual_isl requires the ring-joint streaming compute path; the compute_common.hpp path
  selected by fp32_dest_acc_en=true is not supported` — i.e. fp32 dest accumulation selects a
  compute path that cannot do the block-cyclic KV pad rotation chunked prefill depends on.
* On the no-cache path it is merely allowed, and it buys nothing: whole-attention PCC moved
  0.989870 -> 0.989944, +7e-5. The accuracy that matters is elsewhere.

So `False` here costs nothing measurable and is required for P2 to work at all.
"""

from dataclasses import dataclass

import ttnn

RING_Q_CHUNK = 128
RING_K_CHUNK = 512
"""Ring SDPA chunking. Taken from the donor (`minimax3_gqa_causal_perf`) and **valid here for a
specific reason, not by inheritance**: the donor was measured at head_dim 128 and chunk_size 5120 on
an sp8 mesh, giving chunk_local = 5120/8 = 640 — and this spec's head_dim and chunk_size are the
same, so chunk_local is the same 640. Change either and these must be re-derived: a q_chunk that
does not divide the per-SP-row token count is the exact shape that stalls PCC around 0.91."""

RING_FP32_DEST_ACC = False
"""Destination accumulation for the ring SDPA only. MUST stay False: the cache-read path rejects
True outright (see the module docstring), and on the no-cache path it is worth +7e-5 PCC. Kept as a
named constant so the measurement is reproducible, not so it can be flipped."""


@dataclass
class AttentionConfig:
    """Llama-3.1-8B attention shapes.

    No `rotary_dim` (rotation is full-width: rotary_dim == head_dim == 128), no `use_qk_norm`, no
    `sliding_window`, no sparse-attention fields. Their absence is the point — each was a donor
    feature this model does not have, and carrying a dormant flag invites it being switched on.
    """

    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    chunk_size: int
    max_local_batch_size: int = 1
    users_row_sharded: bool = False
    scaling: float | None = None  # computed if None
    sequence_parallel: bool = True

    def __post_init__(self):
        if self.scaling is None:
            self.scaling = self.head_dim**-0.5
        assert self.hidden_size == self.num_heads * self.head_dim, (
            f"hidden {self.hidden_size} != num_heads {self.num_heads} * head_dim {self.head_dim}"
        )
        assert self.num_heads % self.num_kv_heads == 0, "GQA needs num_heads divisible by num_kv_heads"

    @property
    def num_key_value_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    def kv_heads_per_chip(self, tp: int) -> int:
        """**2 at TP=4** for this model. See `tt/attention/kv_cache.py` for why that matters."""
        assert self.num_kv_heads % tp == 0
        return self.num_kv_heads // tp

    def q_heads_per_chip(self, tp: int) -> int:
        assert self.num_heads % tp == 0
        return self.num_heads // tp


@dataclass
class ProgramConfig:
    """SDPA / matmul program configs and the compute settings.

    Matmul cores default to None, i.e. let ttnn auto-tune — the donor's choice too, and the right
    one for a bring-up: a hand-picked core grid is a perf decision measured against a profile, and
    the recipe puts perf tuning out of scope.
    """

    # Plain (non-ring) prefill SDPA chunking.
    prefill_q_chunk_size_small: int = 32
    prefill_k_chunk_size_small: int = 32
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048

    # Compute config. See the module docstring for why these are not the donor's defaults.
    math_fidelity: str = "HiFi4"
    math_approx_mode: bool = False
    fp32_dest_acc_en: bool = True
    packer_l1_acc: bool = True

    # Ring SDPA (both the first-chunk no-cache path and the cache-read path).
    ring_q_chunk_size: int = RING_Q_CHUNK
    ring_k_chunk_size: int = RING_K_CHUNK
    ring_fp32_dest_acc_en: bool = RING_FP32_DEST_ACC

    prefill_qkv_cores: tuple[int, int] | None = None
    prefill_out_cores: tuple[int, int] | None = None

    def __post_init__(self):
        for name in (
            "prefill_q_chunk_size_small",
            "prefill_k_chunk_size_small",
            "prefill_q_chunk_size_large",
            "prefill_k_chunk_size_large",
            "prefill_threshold",
            "ring_q_chunk_size",
            "ring_k_chunk_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        valid = ("LoFi", "HiFi2", "HiFi3", "HiFi4")
        if self.math_fidelity not in valid:
            raise ValueError(f"math_fidelity must be one of {valid}, got {self.math_fidelity}")

    def get_prefill_sdpa_config(self, mesh_device, seq_len: int) -> ttnn.SDPAProgramConfig:
        """Plain SDPA program config, chunked by whether the sequence clears `prefill_threshold`."""
        large = seq_len >= self.prefill_threshold
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            exp_approx_mode=False,
            q_chunk_size=self.prefill_q_chunk_size_large if large else self.prefill_q_chunk_size_small,
            k_chunk_size=self.prefill_k_chunk_size_large if large else self.prefill_k_chunk_size_small,
        )

    def get_ring_sdpa_config(self) -> ttnn.SDPAProgramConfig:
        """Ring SDPA program config. Chunk sizes are fixed, not sequence-length dependent: the ring
        op streams the gathered prefix, so what matters is the per-SP-row token count."""
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            exp_approx_mode=False,
            q_chunk_size=self.ring_q_chunk_size,
            k_chunk_size=self.ring_k_chunk_size,
        )

    def get_compute_kernel_config(self) -> ttnn.WormholeComputeKernelConfig:
        """HiFi4 + fp32 dest accumulation — matmuls and the plain SDPA.

        `WormholeComputeKernelConfig` is the name the API exposes; it is the Blackhole config too.
        """
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.math_fidelity),
            math_approx_mode=self.math_approx_mode,
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
        )

    def get_ring_compute_kernel_config(self) -> ttnn.WormholeComputeKernelConfig:
        """The ring SDPA's own compute config — the ONLY place dest accumulation is narrowed."""
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.math_fidelity),
            math_approx_mode=self.math_approx_mode,
            fp32_dest_acc_en=self.ring_fp32_dest_acc_en,
            packer_l1_acc=False,
        )


@dataclass
class LlamaAttentionProgramConfig(ProgramConfig):
    """The instance the decoder layer builds. Kept as a named subclass so a future per-model
    override has an obvious home, matching the donor's `MiniMaxM3AttentionProgramConfig`."""
