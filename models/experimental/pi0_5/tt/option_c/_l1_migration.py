# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Post-construction L1 weight migration for Option C prefill TP=2.

Mirrors Option B's `_l1_migration.py` pattern: walk every TP block on the
prefill TP slice and move its matmul weights from DRAM to L1 via
`ttnn.to_memory_config(t, L1_MEMORY_CONFIG)` + `ttnn.deallocate(t)`.

The migration is only meaningful when `Pi0_5PipelineC(prefill_tp_size>1,
prefill_weights_l1=True)`. In replicated or layer-paired modes the per-chip
weight load is ~110 MB / chip → above the matmul kernel's CB region cap
(see L1_PLACEMENT_FINDINGS.md). TP=2 shrinks per-chip matmul shapes and
weight loads to ~55 MB / chip, which clears the threshold (analytical
0.46 MB / bank, validated experimentally on Option B at TP=8).

We intentionally do NOT migrate norm weights or biases — they're small
(~0.1 MB / chip) and the safer DRAM placement avoids edge cases for the
rms_norm kernel's CB region. Same default Option B uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import ttnn

if TYPE_CHECKING:
    from .pipeline import Pi0_5PipelineC


def _to_l1(t: Optional["ttnn.Tensor"]) -> Optional["ttnn.Tensor"]:
    """Move a tensor to L1 and deallocate the DRAM source.

    Explicit deallocate keeps the per-step peak bounded by one buffer's
    worth of transient (DRAM source + L1 destination during the copy).

    Idempotent: when the source is already L1-resident, `to_memory_config`
    is a no-op that returns the SAME buffer reference; calling
    `ttnn.deallocate(t)` on it would then free the underlying buffer that
    the returned tensor also references, producing a dangling reference and
    a "Tensor is not allocated" crash on the next op. Detect that case via
    `t.memory_config().buffer_type` and return `t` unchanged.
    """
    if t is None:
        return None
    # `memory_config` returns a MemoryConfig; .buffer_type is BufferType enum.
    if t.memory_config().buffer_type == ttnn.BufferType.L1:
        return t
    new_t = ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(t)
    return new_t


def _migrate_tp_gemma_block_to_l1(block, mlp_only: bool = False) -> None:
    """Pi0_5OptionCSubmeshTPGemmaBlock: VLM transformer block attrs.

    Attribute names match those in `tp_block.py`. Keep norms in DRAM (small
    overhead, but safer wrt the rms_norm kernel's CB region).

    When `mlp_only=True`, only the MLP weights (gate/up/down) are migrated
    to L1; Q/K/V/O stay in DRAM. This is the scope-tightened path needed
    when per-chip weight load otherwise overflows the CB-region headroom
    (e.g. depth=18 with 2 layers per (2,1) sub-mesh at TP=2 ≈ 126 MB / chip
    full migration, ≈ 100 MB / chip with MLP-only). The MLP weights are
    the dominant contributor (~92% of per-layer bytes) so the L1 hit ratio
    barely changes; the small Q/K/V/O matmuls still read from DRAM but
    are 3x smaller so the DRAM bandwidth cost is negligible.
    """
    if not mlp_only:
        block.q_proj = _to_l1(block.q_proj)
        block.kv_proj = _to_l1(block.kv_proj)
        block.o_proj = _to_l1(block.o_proj)
    block.gate_proj = _to_l1(block.gate_proj)
    block.up_proj = _to_l1(block.up_proj)
    block.down_proj = _to_l1(block.down_proj)


def _migrate_vlm_tp_slice_to_l1(slice_obj, mlp_only: bool = False) -> None:
    """Pi0_5OptionCVLMSliceTP: walk vlm_blocks. Final norm stays in DRAM
    (already DRAM-bounced on entry/exit of the rms_norm call site)."""
    for block in slice_obj.vlm_blocks:
        _migrate_tp_gemma_block_to_l1(block, mlp_only=mlp_only)


def migrate_prefill_weights_to_l1(pipe: "Pi0_5PipelineC", mlp_only: bool = False) -> None:
    """Walk the prefill stage's TP slice and migrate matmul weights to L1.

    No-op when prefill_tp_size == 1 or when the slice isn't TP type. Safe
    to call multiple times — the underlying `_to_l1` walker deallocates
    the source after the move, so a second call is a no-op for tensors
    already in L1 (the move just reads + writes the same region).

    `mlp_only=True` migrates only the (large) MLP weights to L1, keeping
    Q/K/V/O in DRAM. Use this when full migration would push past the L1
    headroom above the matmul kernel's static CB region.
    """
    if pipe.stage_1 is None or pipe.stage_1.slice is None:
        return
    # Only TP slice has weights worth migrating in this path.
    from .vlm_slice import Pi0_5OptionCVLMSliceTP

    if not isinstance(pipe.stage_1.slice, Pi0_5OptionCVLMSliceTP):
        return
    _migrate_vlm_tp_slice_to_l1(pipe.stage_1.slice, mlp_only=mlp_only)


# ---------------------------------------------------------------------------- #
# Denoise (expert + suffix) migration                                           #
# ---------------------------------------------------------------------------- #
#
# Mirror of `option_b._l1_migration._migrate_tp_adarms_block_to_l1` for the
# Option C denoise stage. Walks each `AdaRMSGemmaBlockTTNN` in the expert
# slice and moves the matmul weights (Q/K/V/O on `block.attention`, gate/up/
# down on `block.mlp`) + mod weight + LN-style weights from DRAM to L1.
#
# Unlike Option B's TP=8 path, Option C denoise runs at TP=1 (each expert
# layer's weights live on a single chip in the paired path, or fully
# replicated in the non-paired path). The matmul CB region size is small
# because the expert MLP is much smaller than the VLM MLP (mlp_dim=8192 vs
# 16384). Per OPTION_C_TP_WITHIN_STAGE_PLAN.md the dominant load is the
# adaRMS modulation Dense which is ~32 MB / chip / layer × 3 layers — see
# the prompt's denoise plan for the sharded mod_weight workaround.
#
# NO DRAM fallbacks: all migrated tensors land in L1. If a tensor doesn't
# fit, the migration helper will fail loudly rather than silently degrade
# to DRAM. The validation agent's probe will surface the first failure.


def _migrate_adarms_gemma_block_to_l1(block) -> None:
    """`AdaRMSGemmaBlockTTNN`: expert transformer block attrs.

    Attribute names verified against `tt/ttnn_gemma.py` (AdaRMSGemmaBlockTTNN
    __init__). Q/K/V/O live on `block.attention`; gate/up/down on `block.mlp`;
    `block.mod_weight` + `block.mod_bias` are at the block level.

    NO DRAM fallback: every migrated tensor lands in L1 or the migration
    raises. Per the prompt's "all weights+biases must be bf8_b" guidance,
    the uploads have already been recast to bf8_b in `expert_slice.py`; this
    helper just moves DRAM -> L1.

    Note: when `mod_sharded=True` at construction time, `block.mod_weight`
    is uploaded sharded along its 6144 axis with `_upload_l1_sharded`, which
    defaults to L1 — no migration needed. The `_to_l1` walker is idempotent
    so calling it on an already-L1 tensor is safe.
    """
    # Attention sub-block.
    if getattr(block, "attention", None) is not None:
        attn = block.attention
        if getattr(attn, "wqkv", None) is not None:
            attn.wqkv = _to_l1(attn.wqkv)
        if getattr(attn, "o_proj", None) is not None:
            attn.o_proj = _to_l1(attn.o_proj)
    # MLP sub-block.
    if getattr(block, "mlp", None) is not None:
        mlp = block.mlp
        if getattr(mlp, "gate_proj", None) is not None:
            mlp.gate_proj = _to_l1(mlp.gate_proj)
        if getattr(mlp, "up_proj", None) is not None:
            mlp.up_proj = _to_l1(mlp.up_proj)
        if getattr(mlp, "down_proj", None) is not None:
            mlp.down_proj = _to_l1(mlp.down_proj)
    # adaRMS modulation Dense + bias (may be sharded; _to_l1 is a no-op on L1).
    if getattr(block, "mod_weight", None) is not None:
        block.mod_weight = _to_l1(block.mod_weight)
    if getattr(block, "mod_bias", None) is not None:
        block.mod_bias = _to_l1(block.mod_bias)


def migrate_expert_slice_to_l1(slice_obj) -> None:
    """`Pi0_5OptionCExpertSlice` / `Pi0_5OptionCExpertSlicePaired`: walk all
    expert blocks + the final adaRMS-norm Dense / bias.

    Both slice classes expose `expert_blocks`, `final_norm_mod_weight`,
    `final_norm_mod_bias`. The paired path's blocks live on per-chip
    micro-submeshes; the replicated path's blocks live on the full 6-chip
    denoise submesh. `_to_l1` handles either case (the underlying tensor
    knows its own submesh).
    """
    for block in getattr(slice_obj, "expert_blocks", []):
        _migrate_adarms_gemma_block_to_l1(block)
    if getattr(slice_obj, "final_norm_mod_weight", None) is not None:
        slice_obj.final_norm_mod_weight = _to_l1(slice_obj.final_norm_mod_weight)
    if getattr(slice_obj, "final_norm_mod_bias", None) is not None:
        slice_obj.final_norm_mod_bias = _to_l1(slice_obj.final_norm_mod_bias)


def _migrate_suffix_slice_to_l1(suffix_obj) -> None:
    """`Pi0_5OptionCSuffixSlice`: walk action_in/out + time_mlp_in/out matmul
    weights and biases.

    The suffix MLP is uploaded with default `_upload_l1_replicated` which
    defaults to DRAM today (see `vlm_slice._upload_l1_replicated`); this
    walker is the explicit L1 migration. Weights are tiny (~2 MB total) so
    L1 fit is trivial.
    """
    for name in (
        "action_in_w",
        "action_in_b",
        "action_out_w",
        "action_out_b",
        "time_mlp_in_w",
        "time_mlp_in_b",
        "time_mlp_out_w",
        "time_mlp_out_b",
    ):
        t = getattr(suffix_obj, name, None)
        if t is not None:
            setattr(suffix_obj, name, _to_l1(t))


def migrate_pipeline_denoise_to_l1(pipe: "Pi0_5PipelineC") -> None:
    """Walk the denoise stage's expert + suffix slices and migrate to L1.

    Equivalent of `option_b._l1_migration.migrate_pipeline_weights_to_l1`
    but scoped to Option C's denoise stage only (vision/prefill have their
    own helpers). No-op when the denoise stage hasn't been initialized.

    Safe to call multiple times — `_to_l1` is idempotent for already-L1
    tensors.
    """
    if pipe.stage_2 is None:
        return
    if pipe.stage_2.slice is not None:
        migrate_expert_slice_to_l1(pipe.stage_2.slice)
    if pipe.stage_2.suffix is not None:
        _migrate_suffix_slice_to_l1(pipe.stage_2.suffix)
