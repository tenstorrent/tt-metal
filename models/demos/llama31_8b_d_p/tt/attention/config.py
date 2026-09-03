# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B attention shape config + device program / compute-kernel config.

HF anchor: ``transformers.models.llama.modeling_llama.LlamaAttention``.
Template: ``models/demos/gpt_oss_d_p/tt/attention/config.py:23`` (``AttentionConfig``), ``:45``
(``__post_init__``), ``:52`` (``gqa_group_size``), ``:57`` (``ProgramConfig``), ``:90``
(``get_prefill_sdpa_config``), ``:102`` (``get_compute_kernel_config``).

Llama-3.1 shape: GQA **32 Q / 8 KV heads**, ``head_dim`` **128**, **full** rotary
(``rotary_dim == head_dim``), llama3-scaled RoPE (baked into the cos/sin by ``tt/rope.py``),
plain causal masking, ``attention_bias: false``.

**Deleted from the template**, because Llama has none of it (``00_MODEL_CARD.md`` §3;
``03_OUTLINE.md`` §3.7): ``sliding_window`` (``config.py:34``) and everything attention-sink
related (``:38-40``). ``scaling`` is still passed to SDPA **explicitly** rather than relying on the
kernel default, so the QK scale is visible at the call site.

Two corrections to the template, both measured:

* ``DEC-012`` / Appendix F.8 — ``sdpa_core_grid`` is an explicit named field pinned to **(8, 8)**
  and is *never* derived from ``mesh_device.compute_with_storage_grid_size()`` (measured **(12, 10)**
  on this Blackhole). :meth:`ProgramConfig.assert_sdpa_grid_fits` re-derives the constraint the
  ring-joint SDPA op asserts and is called at **module construction** time, so a bad grid fails in
  P5 instead of silently passing every single-card gate and failing at SP > 1 in P8.
* ``DEC-013`` — :meth:`ProgramConfig.get_compute_kernel_config` takes ``mesh_device`` and calls
  ``ttnn.init_device_compute_kernel_config`` instead of naming a class.
  ``ttnn.BlackholeComputeKernelConfig`` does not exist (``hasattr`` is ``False``;
  ``ttnn/ttnn/__init__.py:305`` exports only the Wormhole name) and where it *is* defined it is the
  same object (``ttnn/ttnn/types.py:61``), so an arch branch is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn

_VALID_MATH_FIDELITIES = ("LoFi", "HiFi2", "HiFi3", "HiFi4")


@dataclass
class AttentionConfig:
    """Llama-3.1 attention shape. Nothing device-specific lives here."""

    hidden_size: int  # 4096
    num_heads: int  # 32 Q heads
    num_kv_heads: int  # 8 KV heads (GQA group = 4)
    head_dim: int  # 128
    max_seq_len: int

    rms_norm_eps: float = 1e-5
    # Softmax scale 1/sqrt(head_dim); derived when None and then passed to SDPA explicitly.
    scaling: float | None = None
    # Full rotary for Llama (defaults to head_dim in __post_init__). A partial-rotary model would
    # need the slice/concat this package deliberately does not have.
    rotary_dim: int | None = None
    # SP prefill: cache-backed ring SDPA on every chunk. P8; see tt/attention/dense_sp.py.
    sequence_parallel: bool = False

    def __post_init__(self):
        if self.scaling is None:
            self.scaling = self.head_dim**-0.5
        if self.rotary_dim is None:
            self.rotary_dim = self.head_dim
        assert self.rotary_dim == self.head_dim, (
            f"Llama-3.1 is FULL rotary; got rotary_dim={self.rotary_dim} != head_dim={self.head_dim}. "
            f"Partial rotary is not implemented (00_MODEL_CARD.md §3)."
        )
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads {self.num_heads} % num_kv_heads {self.num_kv_heads} != 0; "
            f"ttnn.transformer.scaled_dot_product_attention asserts nqh % nkv == 0 "
            f"(sdpa_device_operation.cpp:98)"
        )
        assert (
            self.num_heads * self.head_dim == self.hidden_size
        ), f"num_heads*head_dim = {self.num_heads * self.head_dim} != hidden_size {self.hidden_size}"

    @property
    def gqa_group_size(self) -> int:
        """Q heads per KV head. 32 / 8 = 4 for Llama-3.1-8B."""
        return self.num_heads // self.num_kv_heads


@dataclass
class ProgramConfig:
    """SDPA program config + compute-kernel config. Only device knobs live here."""

    # Prefill SDPA chunking (seq-len dependent), carried verbatim from the template.
    prefill_q_chunk_size_small: int = 32
    prefill_k_chunk_size_small: int = 32
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048

    # DEC-012 / Appendix F.8: an EXPLICIT named field, not derived from the device grid.
    sdpa_core_grid: tuple = (8, 8)

    # SP ring-joint SDPA chunking (DEC-083). Seq-length independent, unlike the pair above: the ring
    # op's Q slab is one chunk's per-device rows for every chunk of a request, so there is no
    # threshold to cross. 128/128 are the template's ring values
    # (models/demos/gpt_oss_d_p/tt/attention/prefill.py:196-197).
    sp_ring_q_chunk_size: int = 128
    sp_ring_k_chunk_size: int = 128

    # Compute config. fp32_dest_acc_en defaults to True (DEC-031) — measured, not inherited: the
    # template ships False (gpt_oss config.py:71). The SP ring op is the one path that REQUIRES
    # False (gpt_oss prefill.py:200); it builds its own config, so it is unaffected.
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
        if self.math_fidelity not in _VALID_MATH_FIDELITIES:
            raise ValueError(f"math_fidelity must be one of {list(_VALID_MATH_FIDELITIES)}, got {self.math_fidelity}")
        if len(self.sdpa_core_grid) != 2 or min(self.sdpa_core_grid) <= 0:
            raise ValueError(f"sdpa_core_grid must be a positive (x, y) pair, got {self.sdpa_core_grid}")
        # The ring op asserts both chunk sizes are tile multiples
        # (`ring_joint_sdpa_device_operation.cpp:848`, `:853`); refuse here so a bad config fails at
        # construction rather than inside the first SP layer's forward.
        for name, value in (
            ("sp_ring_q_chunk_size", self.sp_ring_q_chunk_size),
            ("sp_ring_k_chunk_size", self.sp_ring_k_chunk_size),
        ):
            if value <= 0 or value % ttnn.TILE_SIZE != 0:
                raise ValueError(f"{name} must be a positive multiple of TILE_SIZE ({ttnn.TILE_SIZE}), got {value}")

    def assert_sdpa_grid_fits(self, mesh_device) -> None:
        """Fail at build time if the SDPA program grid would break the SP ring path (``DEC-012``).

        ``ring_joint_scaled_dot_product_attention`` asserts
        ``ccl_core_grid_offset.x >= program_config.compute_with_storage_grid_size.x``
        (``ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp:421``,
        reached because the SP path passes ``use_column_major_ccl=True`` —
        ``models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:134``), and the CCL offset this package
        uses is ``grid.x - 1``. Measured on this Blackhole: ``grid = (12, 10)``, so ``11 >= 8`` holds
        for the pinned 8x8 grid and ``11 >= 12`` would FAIL for a device-derived one.

        The check is deliberately unconditional (not gated on SP > 1): a grid choice that only
        breaks at SP > 1 would pass every P5/P7 single-card gate and fail two phases later.
        """
        grid = mesh_device.compute_with_storage_grid_size()
        assert self.sdpa_core_grid[0] <= grid.x - 1, (
            f"sdpa_core_grid.x = {self.sdpa_core_grid[0]} > grid.x - 1 = {grid.x - 1}: the SP "
            f"ring-joint SDPA asserts ccl_core_grid_offset.x >= sdpa_grid.x with the offset pinned "
            f"at grid.x - 1, so this config would fail at SP > 1 while passing every single-card "
            f"gate (Appendix F.8 / DEC-012). Keep sdpa_core_grid at (8, 8)."
        )
        assert (
            self.sdpa_core_grid[1] <= grid.y
        ), f"sdpa_core_grid.y = {self.sdpa_core_grid[1]} exceeds the device grid y = {grid.y}"

    def get_prefill_sdpa_config(self, mesh_device, seq_len: int) -> ttnn.SDPAProgramConfig:
        """The single-card / non-ring prefill SDPA program config."""
        self.assert_sdpa_grid_fits(mesh_device)
        if seq_len >= self.prefill_threshold:
            q_chunk, k_chunk = self.prefill_q_chunk_size_large, self.prefill_k_chunk_size_large
        else:
            q_chunk, k_chunk = self.prefill_q_chunk_size_small, self.prefill_k_chunk_size_small
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*self.sdpa_core_grid),
            exp_approx_mode=False,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
        )

    def get_ring_sdpa_config(self, mesh_device) -> ttnn.SDPAProgramConfig:
        """The **SP ring-joint** SDPA program config (``DEC-083``).

        Same pinned 8x8 grid as the single-card path — the ring op is the one that *asserts* it
        (``ring_joint_sdpa_device_operation.cpp:421``) — but its own q/k chunk sizes, which do not
        depend on the sequence length. The template derives its grid instead
        (``models/demos/gpt_oss_d_p/tt/attention/prefill.py:195``:
        ``CoreCoord(grid.x - 1, grid.y)`` = (11, 10) here), which also satisfies the assert at
        ``11 >= 11``; ``DEC-083`` keeps the pinned grid so exactly one grid rule holds in this
        package and ``assert_sdpa_grid_fits`` can be believed.
        """
        self.assert_sdpa_grid_fits(mesh_device)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*self.sdpa_core_grid),
            exp_approx_mode=False,
            q_chunk_size=self.sp_ring_q_chunk_size,
            k_chunk_size=self.sp_ring_k_chunk_size,
        )

    def get_compute_kernel_config(self, mesh_device):
        """``DEC-013``: the factory form, no arch branch, no class name."""
        return self._compute_kernel_config(mesh_device, fp32_dest_acc_en=self.fp32_dest_acc_en)

    def get_ring_compute_kernel_config(self, mesh_device):
        """The SP ring path's compute-kernel config: identical **except** ``fp32_dest_acc_en=False``.

        The ring op requires it (``models/demos/gpt_oss_d_p/tt/attention/prefill.py:200`` says so in
        as many words: "required by the ring op's streaming-sink compute"), so this path gives up the
        fp32 accumulator every other op in the package keeps under ``DEC-031``. That is a real,
        measured accuracy cost, not a formality — see ``DEC-084`` for the number — and it is a
        *separate method* rather than a mutated field so no other op can inherit it by accident.
        """
        return self._compute_kernel_config(mesh_device, fp32_dest_acc_en=False)

    def _compute_kernel_config(self, mesh_device, *, fp32_dest_acc_en: bool):
        return ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, self.math_fidelity),
            math_approx_mode=self.math_approx_mode,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
        )
