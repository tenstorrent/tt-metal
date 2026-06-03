# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Post-construction L1 weight migration for Option B (TP=8) blocks.

Walks every weight tensor on every block (VLM stages 1/2 and expert stage 3)
and moves it from DRAM to L1 via ttnn.to_memory_config(t, L1_MEMORY_CONFIG)
+ ttnn.deallocate(t). Same pattern as option_c/vision_slice.py's
_migrate_tower_weights_to_l1 — contained, reversible, no upstream class
changes.

Motivation: see ../docs/OPTION_B_L1_ASSESSMENT.md. TP=8 shards each
matmul output dim 8x, which both shrinks the per-chip weight load AND
shrinks the matmul kernel's static CB region (out_block_w drops from
~43 single-chip to ~6 at TP=8). The shrunk CB region leaves enough L1
headroom above it for the L1-resident weights to land cleanly — the
per-bank arithmetic that blocks Option C does not apply here.

Used by Pi0_5PipelineB(weights_l1=True, tp_shard=True). No-op when
either flag is False.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import ttnn

if TYPE_CHECKING:
    from .pipeline import Pi0_5PipelineB


def _to_l1(t: Optional["ttnn.Tensor"]) -> Optional["ttnn.Tensor"]:
    """Move a tensor to L1 and deallocate the DRAM source.

    Explicit deallocate keeps the per-step peak bounded by one buffer's
    worth of transient (DRAM source + L1 destination during the copy).
    """
    if t is None:
        return None
    new_t = ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(t)
    return new_t


def _migrate_tp_gemma_block_to_l1(block) -> None:
    """Pi0_5SubmeshTPGemmaBlock: VLM transformer block attrs.

    Attribute names verified against tt/option_b/tp_block.py:156-206.
    """
    block.q_proj = _to_l1(block.q_proj)
    block.kv_proj = _to_l1(block.kv_proj)
    block.o_proj = _to_l1(block.o_proj)
    block.gate_proj = _to_l1(block.gate_proj)
    block.up_proj = _to_l1(block.up_proj)
    block.down_proj = _to_l1(block.down_proj)
    if getattr(block, "input_layernorm", None) is not None:
        block.input_layernorm = _to_l1(block.input_layernorm)
    if getattr(block, "post_attention_layernorm", None) is not None:
        block.post_attention_layernorm = _to_l1(block.post_attention_layernorm)


def _migrate_tp_adarms_block_to_l1(block, migrate_mod_weight: bool = False) -> None:
    """Pi0_5SubmeshTPAdaRMSBlock: expert transformer block attrs.

    Attribute names verified against tt/option_b/tp_expert_block.py:65-118.
    Q/K/V/O + gate/up/down are sharded TP=8 → ~5 MB / chip / layer combined.
    Mod weight / bias are REPLICATED at bf8_b ([1024, 6144]) → ~6.7 MB / chip
    / layer × 18 layers = ~120 MB / chip just for mod_weight — that alone
    pushes expert L1 past the cap and triggers the CB clash on the last
    migration step.

    Default: keep mod_weight + mod_bias in DRAM (read once per Euler step per
    layer; DRAM is fine perf-wise) and only migrate the sharded matmul
    weights. Set migrate_mod_weight=True at your own risk; expert chip will
    blow past the 172.5 MB cap.
    """
    block.q_proj = _to_l1(block.q_proj)
    block.kv_proj = _to_l1(block.kv_proj)
    block.o_proj = _to_l1(block.o_proj)
    block.gate_proj = _to_l1(block.gate_proj)
    block.up_proj = _to_l1(block.up_proj)
    block.down_proj = _to_l1(block.down_proj)
    if migrate_mod_weight:
        block.mod_weight = _to_l1(block.mod_weight)
        if getattr(block, "mod_bias", None) is not None:
            block.mod_bias = _to_l1(block.mod_bias)


def _migrate_vlm_slice_to_l1(slice_obj) -> None:
    """Pi0_5SubmeshVLMSlice: walk vlm_blocks + final norm (when held)."""
    for block in slice_obj.vlm_blocks:
        _migrate_tp_gemma_block_to_l1(block)
    if getattr(slice_obj, "vlm_norm", None) is not None:
        slice_obj.vlm_norm = _to_l1(slice_obj.vlm_norm)


def _migrate_expert_slice_to_l1(slice_obj) -> None:
    """Pi0_5SubmeshExpertSlice: walk expert_blocks + final adarms-norm tensors."""
    for block in slice_obj.expert_blocks:
        _migrate_tp_adarms_block_to_l1(block)
    if getattr(slice_obj, "final_norm_mod_weight", None) is not None:
        slice_obj.final_norm_mod_weight = _to_l1(slice_obj.final_norm_mod_weight)
    if getattr(slice_obj, "final_norm_mod_bias", None) is not None:
        slice_obj.final_norm_mod_bias = _to_l1(slice_obj.final_norm_mod_bias)


def migrate_pipeline_weights_to_l1(pipe: "Pi0_5PipelineB") -> None:
    """Walk every Pi0_5PipelineB stage and migrate block weights to L1.

    Only meaningful when the pipeline was built with tp_shard=True (the
    replicated path puts ~125 MB / chip = too big to fit above the matmul
    CB region). Safe to call multiple times — the underlying _to_l1 walker
    deallocates the source after the move.
    """
    if pipe.stage_1 is not None and pipe.stage_1.slice is not None:
        _migrate_vlm_slice_to_l1(pipe.stage_1.slice)
    if pipe.stage_2 is not None and pipe.stage_2.slice is not None:
        _migrate_vlm_slice_to_l1(pipe.stage_2.slice)
    if pipe.stage_3 is not None and pipe.stage_3.slice is not None:
        _migrate_expert_slice_to_l1(pipe.stage_3.slice)
    # stage_0 (vision): host-resident SigLIP today, nothing to migrate.
    # suffix MLP on stage_3 is already L1-resident per stage_3_expert.py.
