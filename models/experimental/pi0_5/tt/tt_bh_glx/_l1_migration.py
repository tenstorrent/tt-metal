# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Post-construction DRAM->L1 migration for the denoise stage.

The denoise loop runs N (5-10) Euler steps; on every step each of the 6
denoise chips re-reads its ~93 MB of expert weights. Those weights are
uploaded to DRAM by default (see expert_slice/suffix_slice/pipeline), so the
per-step matmuls stream them from DRAM each iteration. Moving the static
denoise weights into L1 once at pipeline construction lets the per-step
matmuls read them from on-chip L1 instead.

This fits: 3 expert layers/chip (~93 MB bf8 matmul + bf16 mod) sit inside the
~180 MB usable L1 per Blackhole chip with headroom for the (small) expert
matmul CB regions. The expert MLP (4096) is 4x smaller than the VLM MLP
(16384), so unlike the prefill stage the interleaved-L1 weights don't clash
with the kernel's static CB region.

Gated by PI0_GLX_DENOISE_L1 (default ON). Set =0 to keep weights in DRAM.
"""

from __future__ import annotations

import os
from typing import Optional

import ttnn


def denoise_l1_enabled() -> bool:
    """Whether to place denoise-stage weights in L1. Default ON."""
    return os.environ.get("PI0_GLX_DENOISE_L1", "1").lower() in ("1", "true", "yes", "on")


def _to_l1(t: Optional["ttnn.Tensor"]) -> Optional["ttnn.Tensor"]:
    """Move a tensor to L1 and free the DRAM source; idempotent on L1 tensors.

    When the tensor is already L1-resident, to_memory_config returns the same
    buffer reference, so deallocating it would dangle the returned handle.
    Guard against that by checking buffer_type first.
    """
    if t is None:
        return None
    if t.memory_config().buffer_type == ttnn.BufferType.L1:
        return t
    moved = ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(t)
    return moved


def _migrate_expert_block(block) -> None:
    """Move one AdaRMSGemmaBlockTTNN's weights+biases to L1."""
    attn = getattr(block, "attention", None)
    if attn is not None:
        attn.wqkv = _to_l1(attn.wqkv)
        attn.o_proj = _to_l1(attn.o_proj)
    mlp = getattr(block, "mlp", None)
    if mlp is not None:
        mlp.gate_proj = _to_l1(mlp.gate_proj)
        mlp.up_proj = _to_l1(mlp.up_proj)
        mlp.down_proj = _to_l1(mlp.down_proj)
    block.mod_weight = _to_l1(block.mod_weight)
    block.mod_bias = _to_l1(block.mod_bias)


def migrate_denoise_weights_to_l1(stage_denoise, suffix_slices, denoise_head) -> None:
    """Move every static denoise-stage weight/bias from DRAM to L1.

    Covers, per the denoise stage's owned tensors:
      - each expert chunk's blocks (QKV/O, MLP gate/up/down, adaRMS mod w+b)
      - each expert chunk's RoPE cos/sin tables (re-read every step)
      - each replicated suffix MLP's weights + biases
      - the final adaRMS-norm dense weight + bias (last denoise chip)

    Activations stay where the forward path already puts them (L1 in the
    denoise loop + suffix matmul outputs); only the DRAM-resident static
    tensors are migrated here.
    """
    for chunk in stage_denoise.chunks:
        for block in chunk.blocks:
            _migrate_expert_block(block)
        chunk.cos_meta = _to_l1(chunk.cos_meta)
        chunk.sin_meta = _to_l1(chunk.sin_meta)

    for sl in suffix_slices:
        weights = sl.suffix.weights
        for key in list(weights.keys()):
            weights[key] = _to_l1(weights[key])

    denoise_head.mod_weight = _to_l1(denoise_head.mod_weight)
    denoise_head.mod_bias = _to_l1(denoise_head.mod_bias)
