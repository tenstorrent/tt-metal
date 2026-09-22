# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shape and kernel configuration for the attention block. Ported from
``gpt_oss_d_p/tt/attention/config.py``.

``AttentionConfig`` is the small struct the attention functions read instead of the full model
config, so the op-level tests can build one directly at reduced width. Fields the source carries
that this model has no use for are **removed**, not defaulted:

* ``sliding_window`` — ``config.json`` has ``sliding_window: null`` and no ``layer_types``; every
  layer is full causal attention.
* ``rotary_dim`` — full rotary (``head_dim`` == rotary dim), no partial-rotary split.
* QK-norm fields — ``Ministral3Attention`` has no ``q_norm``/``k_norm``.
* attention sinks — no ``sinks`` parameter in the checkpoint.

``ProgramConfig`` holds the SDPA program configs and compute-kernel configs. The ring path's
config is separate from the plain prefill path's for two reasons kept from the source: the ring op
must be given a core grid that **excludes the CCL column** (``ring_attention_ccl_core_grid_offset``
in ``tt/ccl.py`` reserves ``x = grid.x - 1``), and it does not support ``fp32_dest_acc_en``.
"""

from dataclasses import dataclass

import ttnn


@dataclass
class AttentionConfig:
    """Dimensions and scaling for one attention block (global, not per-chip)."""

    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    rms_norm_eps: float = 1e-5
    scaling: float | None = None
    sequence_parallel: bool = False

    def __post_init__(self):
        """Default ``scaling`` to ``1/sqrt(head_dim)``.

        Mistral-Medium-3.5 uses the plain softmax scale — there is no YaRN ``mscale`` correction
        (``mscale``/``mscale_all_dim`` are 1.0/0.0) and no length-dependent Q scale
        (``llama_4_scaling_beta`` is 0.0). Both are asserted on the reference side.
        """
        if self.scaling is None:
            self.scaling = self.head_dim**-0.5
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be a multiple of num_kv_heads ({self.num_kv_heads}) " f"for GQA"
        )

    @property
    def gqa_group_size(self) -> int:
        """Q heads per KV head: 96 / 8 = 12."""
        return self.num_heads // self.num_kv_heads

    @property
    def rotary_dim(self) -> int:
        """Rotary dimension. Always the full ``head_dim`` for this model."""
        return self.head_dim


@dataclass
class ProgramConfig:
    """SDPA program configs and compute-kernel configs for the attention block."""

    prefill_q_chunk_size_small: int = 32
    prefill_k_chunk_size_small: int = 32
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048
    ring_q_chunk_size: int = 128
    ring_k_chunk_size: int = 128
    math_fidelity: str = "HiFi4"
    math_approx_mode: bool = False
    #: The recipe's bring-up default for the plain SDPA path (§2.3), matching ``tt/compute.py``.
    #: It is NOT applied to the ring cache-read op, which does not support it — see
    #: :meth:`get_ring_compute_kernel_config`. Keeping the exception local to the one op that has
    #: it is the whole point; this field used to default to False for everything.
    fp32_dest_acc_en: bool = True
    packer_l1_acc: bool = True

    def get_prefill_sdpa_config(self, mesh_device, seq_len: int):
        """Program config for the plain causal SDPA path.

        Chunk sizes step up past ``prefill_threshold``; below it the small chunks keep short
        diagnostic runs from tripping the ``seq_len % q_chunk_size`` constraint.
        """
        if seq_len >= self.prefill_threshold:
            q_chunk, k_chunk = self.prefill_q_chunk_size_large, self.prefill_k_chunk_size_large
        else:
            q_chunk, k_chunk = self.prefill_q_chunk_size_small, self.prefill_k_chunk_size_small
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            exp_approx_mode=False,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
        )

    def get_ring_sdpa_config(self, mesh_device):
        """Program config for ``ring_joint_scaled_dot_product_attention``.

        The core grid is narrowed to ``CoreCoord(grid.x - 1, grid.y)`` so the op does not collide
        with the CCL cores the ring gather runs on.
        """
        grid = mesh_device.compute_with_storage_grid_size()
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
            exp_approx_mode=False,
            q_chunk_size=self.ring_q_chunk_size,
            k_chunk_size=self.ring_k_chunk_size,
        )

    def get_compute_kernel_config(self):
        """Compute-kernel config for matmuls and the plain SDPA path."""
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=math_fidelity_from_name(self.math_fidelity),
            math_approx_mode=self.math_approx_mode,
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
        )

    def get_ring_compute_kernel_config(self, mesh_device):
        """Compute-kernel config for the ring SDPA op.

        The two accumulate flags are pinned off here rather than inherited from the fields above:
        ``ring_joint_scaled_dot_product_attention`` does not support ``fp32_dest_acc_en`` (the
        constraint this whole class used to apply globally), and its online-softmax accumulator is
        its own, so ``packer_l1_acc`` buys nothing while changing a path that already measures at
        target. Fidelity is still HiFi4.
        """
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=math_fidelity_from_name(self.math_fidelity),
            math_approx_mode=self.math_approx_mode,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def __post_init__(self):
        sizes = (
            self.prefill_q_chunk_size_small,
            self.prefill_k_chunk_size_small,
            self.prefill_q_chunk_size_large,
            self.prefill_k_chunk_size_large,
            self.ring_q_chunk_size,
            self.ring_k_chunk_size,
            self.prefill_threshold,
        )
        if min(sizes) <= 0:
            raise ValueError("SDPA chunk sizes and threshold must be positive")
        math_fidelity_from_name(self.math_fidelity)  # validates the name


def math_fidelity_from_name(name: str) -> "ttnn.MathFidelity":
    """``"HiFi4"`` -> ``ttnn.MathFidelity.HiFi4``. Keeps ``ProgramConfig`` a plain dataclass."""
    valid = ("LoFi", "HiFi2", "HiFi3", "HiFi4")
    if name not in valid:
        raise ValueError(f"math_fidelity must be one of {valid}, got {name!r}")
    return getattr(ttnn.MathFidelity, name)
