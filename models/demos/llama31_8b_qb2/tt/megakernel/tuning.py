# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, trace-stable controls for the second decode experiment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectionTuning:
    reader: str = "original"
    wide_subblocks: bool = False
    bounded_barrier: bool = False
    buffer_count: int = 2
    lookahead: int = 2
    hoist_pack_config: bool = False
    bank_vc: bool = False
    prefetch_gu_blocks: int = 0
    prefetch_down_blocks: int = 0
    alias_projection_cbs: bool = False
    prefetch_head_workers: bool = False
    head_prefetch_targets: str = "both"
    projection_placement: str = "row"
    coalesce_input: bool = False
    early_weight_blocks: int = 0
    early_weight_phases: int = 15
    share_qkv_workers: bool = False

    def __post_init__(self):
        if self.early_weight_blocks not in (0, 2, 3) or self.early_weight_blocks > self.buffer_count:
            raise ValueError("Early weight prefix must be0/2/3 blocks and fit its ring")
        if self.early_weight_phases not in (1, 2, 4, 8, 15):
            raise ValueError("Early weight phases must select QKV, O, GU, down or all")
        if self.early_weight_blocks and (self.reader == "original" or not self.alias_projection_cbs or self.share_qkv_workers or self.prefetch_head_workers or self.prefetch_gu_blocks or self.prefetch_down_blocks):
            raise ValueError("Early local reads currently require aliased separate-QKV workers without helper prefetch")
        if not 2 <= self.lookahead <= min(4, self.buffer_count):
            raise ValueError("Projection lookahead must be2..4 and fit the buffer count")
        if self.buffer_count > 3 and not self.alias_projection_cbs:
            raise ValueError("More than three projection buffers require phase aliases")
        if self.share_qkv_workers and (not self.alias_projection_cbs or self.reader == "original" or self.prefetch_head_workers or self.prefetch_gu_blocks or self.prefetch_down_blocks):
            raise ValueError("Shared QKV workers require static aliases and currently exclude helper prefetch")
        if self.head_prefetch_targets not in ("both", "qkv", "o"):
            raise ValueError("Head staging targets must be both, qkv or o")
        if self.head_prefetch_targets != "both" and not self.prefetch_head_workers:
            raise ValueError("Selected head staging targets require prefetch_head_workers")
        if self.prefetch_head_workers and (not self.alias_projection_cbs or self.buffer_count != 3 or self.reader == "original"):
            raise ValueError("Head-worker staging requires aliased three-buffer projection storage and tuned readers")
        if self.coalesce_input and self.reader == "original":
            raise ValueError("Contiguous activation reads require a tuned reader")
        if self.projection_placement not in ("row", "dram"):
            raise ValueError("Projection placement must be row or dram")
        if any(n not in (0, 2, 4, 6) for n in (self.prefetch_gu_blocks, self.prefetch_down_blocks)):
            raise ValueError("Prefetch depth must be zero, two, four or six blocks")
        if self.buffer_count not in (2, 3, 4, 5):
            raise ValueError("Projection buffer_count must be two to five")
        if self.reader not in ("original", "coalesced", "pipelined", "pipelined_rows"):
            raise ValueError(f"Unknown projection reader: {self.reader}")

    @property
    def buffers(self):
        return self.buffer_count

    @property
    def defines(self):
        return [
            ("EARLY_WEIGHT_BLOCKS", str(self.early_weight_blocks)),
            ("EARLY_WEIGHT_PHASES", str(self.early_weight_phases)),
            ("SHARED_QKV", str(int(self.share_qkv_workers))),
            ("PROJECTION_COALESCE_INPUT", str(int(self.coalesce_input))),
            ("DRAM_NEAR_PROJECTION", str(int(self.projection_placement == "dram"))),
            ("ALIAS_PROJECTION_CBS", str(int(self.alias_projection_cbs))),
            ("PROJECTION_READER", str(("original", "coalesced", "pipelined", "pipelined_rows").index(self.reader))),
            ("PROJECTION_HOIST_PACK", str(int(self.hoist_pack_config))),
            ("PROJECTION_BANK_VC", str(int(self.bank_vc))),
            ("GU_PREFETCH_BLOCKS", str(self.prefetch_gu_blocks)),
            ("DOWN_PREFETCH_BLOCKS", str(self.prefetch_down_blocks)),
            ("PROJECTION_BUFFERS", str(self.buffer_count)),
            ("PROJECTION_LOOKAHEAD", str(self.lookahead)),
            ("PROJECTION_WIDE", str(int(self.wide_subblocks))),
        ]
