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
    """
    if t is None:
        return None
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
