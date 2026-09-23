# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, trace-stable controls for the second decode experiment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectionTuning:
    reader: str = "original"
    wide_subblocks: bool = False
    bounded_barrier: bool = False
    multicast_barrier: bool = False
    inline_cb_reset: bool = False
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
    scratch_init_once: str = "off"
    early_weight_blocks: int = 0
    early_weight_phases: int = 15
    share_qkv_workers: bool = False

    compact_activations: str = "off"
    attention_workers: int = 32
    attention_chunk: int = 256
    projection_full_dst: str = "off"
    norm_full_dst: bool = False
    norm_tile_height: int = 32
    projection_tile_height: int = 32
    custom_gu: bool = False
    qkv_custom_mm: bool = False
    qkv_buffers: int = 0
    qkv_early_blocks: int = -1
    head_early_blocks: int = 0
    head_placement: str = "row"
    profiler_phase: int = 0
    split_gu_bank_rows: bool = False
    batch_swiglu: bool = False

    def __post_init__(self):
        if self.custom_gu and (self.projection_tile_height != 16 or self.reader != "pipelined" or not self.wide_subblocks or self.projection_full_dst != "off" or self.compact_activations != "off" or self.share_qkv_workers):
            raise ValueError("Custom GU requires tiny16 wide ordinary pipelined projections")
        if self.qkv_custom_mm and (self.projection_tile_height != 16 or self.reader != "pipelined" or self.share_qkv_workers or self.compact_activations != "off" or self.prefetch_head_workers):
            raise ValueError("Custom QKV requires tiny16 ordinary separate-QKV pipelined readers")
        if self.head_early_blocks not in (0, 2, 3) or self.head_early_blocks > min(3, self.buffer_count):
            raise ValueError("Head prefix must be0/2/3 blocks and fit the head ring")
        if self.head_early_blocks and (self.reader != "pipelined" or self.prefetch_head_workers or self.share_qkv_workers):
            raise ValueError("Head own prefix requires separate QKV and pipelined readers without head helpers")
        if self.attention_workers not in (8, 16, 32) or self.attention_chunk not in (64, 128, 256):
            raise ValueError("Attention requires8/16/32 workers and64/128/256-token chunks")
        if self.norm_tile_height not in (16, 32):
            raise ValueError("Norm tile height must be sixteen or thirty-two")
        if self.projection_full_dst not in ("off", "mlp", "head", "all"):
            raise ValueError("Full destination mode must be off/mlp/head/all")
        if self.projection_tile_height not in (16, 32):
            raise ValueError("Projection tile height must be sixteen or thirty-two")
        if self.projection_tile_height == 16 and self.share_qkv_workers:
            raise ValueError("Tiny projection tiles currently require separate QKV workers")
        if self.qkv_buffers not in (0, 3, 4, 5, 6) or self.qkv_early_blocks not in (-1, 0, 2, 3, 4, 5, 6):
            raise ValueError("Unsupported independent QKV buffering/prefix")
        if self.qkv_early_blocks > (self.qkv_buffers or self.buffer_count):
            raise ValueError("QKV prefix must fit its independent ring")
        if (self.qkv_buffers or self.qkv_early_blocks >= 0) and (self.reader != "pipelined" or self.share_qkv_workers or self.prefetch_head_workers or self.compact_activations != "off"):
            raise ValueError("Independent QKV buffers require ordinary separate QKV pipelined readers without head staging or compact transport")
        if self.head_placement not in ("row", "order", "select"):
            raise ValueError("Head placement must be row/order/select")
        if self.compact_activations not in ("off", "norm", "all"):
            raise ValueError("Compact activation transport must select off/norm/all")
        if self.compact_activations != "off" and (self.reader != "pipelined" or self.buffer_count != 3 or not self.alias_projection_cbs or self.scratch_init_once != "all" or self.share_qkv_workers or self.prefetch_head_workers or self.prefetch_gu_blocks or self.prefetch_down_blocks):
            raise ValueError("Compact transport requires the separate-QKV aliased three-buffer pipeline with scratch-once all and no helper prefetch")
        if self.profiler_phase not in (0, 1, 2, 3):
            raise ValueError("Profiler phase must select main/O/GU/down")
        if self.split_gu_bank_rows and self.reader == "original":
            raise ValueError("Split GU bank rows require a tuned projection reader")
        if self.multicast_barrier and not self.bounded_barrier:
            raise ValueError("Multicast release currently requires the bounded layer barrier")
        if self.scratch_init_once not in ("off", "padding", "norm", "all"):
            raise ValueError("Scratch initialization must be off, padding, norm or all")
        if self.early_weight_blocks not in (0, 2, 3) or self.early_weight_blocks > self.buffer_count:
            raise ValueError("Early weight prefix must be0/2/3 blocks and fit its ring")
        if self.early_weight_phases not in (1, 2, 4, 7, 8, 15):
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
            ("CUSTOM_GU", str(int(self.custom_gu))),
            ("INLINE_CB_RESET", str(int(self.inline_cb_reset))),
            ("QKV_CUSTOM_MM", str(int(self.qkv_custom_mm))),
            ("HEAD_EARLY_BLOCKS", str(self.head_early_blocks)),
            ("ATTENTION_WORKERS", str(self.attention_workers)),
            ("FULL_DST_MLP", str(int(self.projection_full_dst in ("mlp", "all")))),
            ("FULL_DST_HEAD", str(int(self.projection_full_dst in ("head", "all")))),
            ("TINY_PROJECTION_M", str(int(self.projection_tile_height == 16))),
            ("COMPACT_ACTIVATIONS", str({"off":0, "norm":1, "all":3}[self.compact_activations])),
            ("PROFILER_PROJECTION_PHASE", str(self.profiler_phase)),
            ("GU_BANK_SPLIT", str(int(self.split_gu_bank_rows))),
            ("BATCH_SWIGLU", str(int(self.batch_swiglu))),
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
