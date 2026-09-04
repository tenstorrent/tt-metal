# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`AttentionConfig` + `ProgramConfig` for Llama-3.1-8B prefill attention.

**HF anchor:** `transformers.models.llama.modeling_llama.LlamaAttention` (the shape half of it —
this file holds no math). **Template:**
`models/demos/gpt_oss_d_p/tt/attention/config.py:23` and `:57`.

**Dropped relative to the template**, because Llama has none of it
(`bringup_log/00_MODEL_CARD.md` §3):

* `sliding_window` and the `layer_types` plumbing the caller does — every Llama layer is
  full-causal, so there is no per-layer config copy and no `is_sliding`;
* `rotary_dim` — rotary_dim == head_dim == 128, full rotary, so a separate field would only be a
  second place for 128 to be wrong;
* the attention-sink field (`sinks` lives in the template's weights, but `scaling`'s pre-division
  comment at `models/demos/gpt_oss_d_p/tt/attention/config.py:38-39` goes with it).

**Inverted relative to the template:** `models/demos/gpt_oss_d_p/tt/attention/config.py:71` sets
`fp32_dest_acc_en: bool = False`, and carrying that forward costs 38.7x on an attention block at
bf8_b and 107.6x at bf16 (`BRINGUP_RECIPE.md:654-655`). Measured on this box at the MLP level it
costs 96x / 1168x (`bringup_log/06_GATES.md`, `G-MLP` A/B), so the default here is `True` and the
compute config is **not** rebuilt locally — it comes from the package's single factory
(`tt/config.py::default_compute_kernel_config`, `DEC-030`, `DEC-040`).

**The SDPA program grid is a pinned 8x8 and must never be derived from the device grid.** This
looks like a portability improvement and it is a P8-only landmine: the ring SDPA asserts
`ccl_core_grid_offset.x >= program_config.compute_with_storage_grid_size.x`
(`ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp:421`), and
the CCL offset is pinned at `grid.x - 1` = 11 on this (12,10) Blackhole (`tt/ccl.py`). With the
program grid at 8, `11 >= 8` holds; derived from the device grid it would be `11 >= 12` and fail —
**only at SP > 1, i.e. only in P8, after every single-card gate has passed**. `validate_grid()`
turns that into a construction-time failure instead.
"""

from dataclasses import dataclass

import ttnn

from ..config import default_compute_kernel_config


@dataclass
class AttentionConfig:
    """Llama GQA attention shapes. 32 Q heads / 8 KV heads / head_dim 128, full rotary, no biases."""

    hidden_size: int  # 4096
    num_heads: int  # 32 Q heads
    num_kv_heads: int  # 8 KV heads -> GQA group 4
    head_dim: int  # 128 (derived; `tt/config.py::derive_head_dim` is the one derivation)
    max_seq_len: int
    rms_norm_eps: float = 1e-5
    # Softmax scale `1/sqrt(head_dim)`; computed when None and always passed to SDPA explicitly, so
    # the value the module thinks it uses and the value the kernel uses cannot diverge.
    scaling: float | None = None
    # SP prefill (the ring path) — P8. False for every P5-P7 gate.
    sequence_parallel: bool = False

    def __post_init__(self):
        if self.scaling is None:
            self.scaling = self.head_dim**-0.5
        assert self.num_heads % self.num_kv_heads == 0, (
            f"GQA needs num_heads ({self.num_heads}) divisible by num_kv_heads "
            f"({self.num_kv_heads}); the SDPA op asserts the per-chip form of this at "
            f"ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:98"
        )
        assert (
            self.head_dim % ttnn.TILE_SIZE == 0
        ), f"head_dim {self.head_dim} is not tile-aligned; the KV cache DRAM shard is [1,1,32,head_dim]"

    @property
    def gqa_group_size(self) -> int:
        """32 // 8 = 4 globally, and 4 // 1 = 4 per chip at TP=8 — the group is TP-invariant."""
        return self.num_heads // self.num_kv_heads


@dataclass
class ProgramConfig:
    """SDPA program config + the compute-kernel config, for the dense prefill path only.

    The SP ring path builds its **own** program config (P8) rather than mutating this one: it needs
    a different grid (the CCL column carved out) and it is the one op in this model where
    `fp32_dest_acc_en=False` is mandatory (`bringup_log/03_OUTLINE.md` §2.7, `attention/dense_sp.py`).
    """

    # PINNED, never derived — see the module docstring. Same value as
    # `models/demos/gpt_oss_d_p/tt/attention/config.py:96`.
    sdpa_grid_x: int = 8
    sdpa_grid_y: int = 8

    # Seq-len-dependent SDPA chunking, template defaults
    # (`models/demos/gpt_oss_d_p/tt/attention/config.py:62-66`). Recipe §2.3 measured that sweeping
    # these over {32,128,256} moves the fused kernel's PCC by under 4%, so they are left at the
    # template's values and are not a tuning knob in a functional-first iteration
    # (`bringup_log/03_OUTLINE.md` §6).
    prefill_q_chunk_size_small: int = 32
    prefill_k_chunk_size_small: int = 32
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048

    # `exp_approx_mode=False`: recipe §2.3 measured it moves the SDPA PCC not at all, and the
    # package default for `math_approx_mode` is False, so the two agree.
    exp_approx_mode: bool = False

    # The ONLY compute-kernel field kept here. The other three live in
    # `tt/config.py::default_compute_kernel_config` (`DEC-030`, `DEC-040`); this one is exposed so
    # `G-ATTN` can A/B recipe §2.4's block-level 38.7x / 107.6x claim on this box. `True` is the
    # correct value; `False` is a measurement.
    fp32_dest_acc_en: bool = True

    def __post_init__(self):
        chunks = (
            self.prefill_q_chunk_size_small,
            self.prefill_k_chunk_size_small,
            self.prefill_q_chunk_size_large,
            self.prefill_k_chunk_size_large,
        )
        if min(*chunks, self.prefill_threshold) <= 0:
            raise ValueError("SDPA chunk sizes and threshold must be positive")
        for chunk in chunks:
            if chunk % ttnn.TILE_SIZE != 0:
                raise ValueError(
                    f"SDPA chunk size {chunk} must be a multiple of TILE_SIZE ({ttnn.TILE_SIZE}); the op "
                    f"asserts this at ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:108"
                )

    def validate_grid(self, mesh_device) -> None:
        """Fail at **construction** if the pinned SDPA grid cannot coexist with the CCL offset.

        `tt/ccl.py` puts the ring-attention CCL workers in compute column `grid.x - 1`, and
        `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp:421`
        requires that offset to be `>=` the SDPA program grid's `x`. Checking it here means a wrong
        grid fails when the `Attention` module is built, not two phases later at SP > 1 — the whole
        point of the landmine (`BRINGUP_RECIPE.md:1411-1419`).
        """
        grid = mesh_device.compute_with_storage_grid_size()
        ccl_offset_x = grid.x - 1
        if self.sdpa_grid_x > ccl_offset_x:
            raise ValueError(
                f"SDPA program grid x={self.sdpa_grid_x} exceeds the ring-attention CCL core offset "
                f"x={ccl_offset_x} (compute grid {grid.x}x{grid.y}); "
                f"ring_joint_sdpa_device_operation.cpp:421 asserts ccl_core_grid_offset.x >= sdpa_grid.x, "
                f"so this would pass every single-card gate and fail only at SP > 1. Keep the grid pinned at 8x8."
            )
        if self.sdpa_grid_y > grid.y:
            raise ValueError(f"SDPA program grid y={self.sdpa_grid_y} exceeds the compute grid y={grid.y}")

    def get_prefill_sdpa_config(self, mesh_device, seq_len: int) -> ttnn.SDPAProgramConfig:
        """The dense-prefill SDPA program config. Validates the pinned grid on every build."""
        self.validate_grid(mesh_device)
        if seq_len >= self.prefill_threshold:
            q_chunk, k_chunk = self.prefill_q_chunk_size_large, self.prefill_k_chunk_size_large
        else:
            q_chunk, k_chunk = self.prefill_q_chunk_size_small, self.prefill_k_chunk_size_small
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(self.sdpa_grid_x, self.sdpa_grid_y),
            exp_approx_mode=self.exp_approx_mode,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
        )

    def get_compute_kernel_config(self, mesh_device):
        """The package's single compute-kernel config (`DEC-030`), not a second definition.

        `models/demos/gpt_oss_d_p/tt/attention/config.py:102-108` builds its own from four local
        fields, which is exactly the "buried in an attention config" pattern
        `BRINGUP_RECIPE.md:1247-1250` blames for one package holding two different values of
        `fp32_dest_acc_en`. Only that one field is local here, and only so `G-ATTN` can measure it.
        """
        return default_compute_kernel_config(mesh_device, fp32_dest_acc_en=self.fp32_dest_acc_en)
