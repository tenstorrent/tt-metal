# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE 5 Hz LM: narrow ``LMHead`` when audio-code columns span a vocab sub-range.

Audio-code token IDs are sparse in the full ~217k vocab, but often occupy a bounded
index band. When ``max(indices) - min(indices)`` is smaller than the full padded vocab,
``LMHead`` can skip matmul splits outside that band and slice columns from the reduced
concat output.

When the band still covers most splits, the wrapper falls back to the stock forward
and the experimental bridge gathers on host (PCC-safe).
"""

from __future__ import annotations

from typing import Any, Callable

import torch

import ttnn
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode


def ace_step_split_column_ranges(split_sizes: list[int]) -> list[tuple[int, int]]:
    off = 0
    ranges: list[tuple[int, int]] = []
    for size in split_sizes:
        size_i = int(size)
        ranges.append((off, off + size_i))
        off += size_i
    return ranges


def ace_step_narrow_column_band(indices: torch.Tensor) -> tuple[int, int] | None:
    """Inclusive min, exclusive max column for sorted-ish index tensor."""
    if indices is None or indices.numel() == 0:
        return None
    idx = indices.detach().to(dtype=torch.long, device="cpu").view(-1)
    return int(idx.min().item()), int(idx.max().item()) + 1


def ace_step_splits_for_band(split_sizes: list[int], col_lo: int, col_hi: int) -> list[int]:
    hits: list[int] = []
    for i, (start, end) in enumerate(ace_step_split_column_ranges(split_sizes)):
        if end > col_lo and start < col_hi:
            hits.append(i)
    return hits


def ace_step_patch_lm_head_narrow_forward(lm_head: Any) -> None:
    """Wrap ``lm_head.forward``; honors ``lm_head._ace_narrow_vocab_indices`` when set."""
    if getattr(lm_head, "_ace_narrow_forward_patched", False):
        return
    original: Callable = lm_head.forward

    def narrow_forward(x, debug_input_torch=None, debug_weight_torch=None):
        lm_head._ace_narrow_forward_used_band = False  # type: ignore[attr-defined]
        lm_head._ace_narrow_forward_col_band = None  # type: ignore[attr-defined]
        # TP guard: when the LMHead is vocab-sharded across devices (num_devices > 1, e.g. the
        # 5 Hz LM moved onto the mesh under TP), ``split_sizes`` tile only THIS device's
        # ``padded_vocab_size // num_devices`` shard, while the narrow band indices are GLOBAL.
        # Selecting/slicing splits by global columns against per-shard-local ranges is incorrect,
        # so bypass narrowing and let the stock forward do the correct sharded matmul + all-reduce.
        # No effect today: ACE-Step runs the LM on a 1×1 preprocess chip, so num_devices == 1.
        num_devices = int(getattr(getattr(lm_head, "args", None), "num_devices", 1) or 1)
        if num_devices > 1:
            return original(x, debug_input_torch=debug_input_torch, debug_weight_torch=debug_weight_torch)
        indices: torch.Tensor | None = getattr(lm_head, "_ace_narrow_vocab_indices", None)
        band = ace_step_narrow_column_band(indices) if indices is not None else None
        if band is None:
            return original(x, debug_input_torch=debug_input_torch, debug_weight_torch=debug_weight_torch)

        col_lo, col_hi = band
        use_prefetcher = lm_head.prefetcher is not None and lm_head.prefetcher.mode == Mode.DECODE
        split_sizes = lm_head.split_sizes_ring_mm if use_prefetcher else lm_head.split_sizes_dram_sharded
        padded_cols = sum(int(s) for s in split_sizes)
        split_hits = ace_step_splits_for_band(split_sizes, col_lo, col_hi)
        # Fall back when the band still touches every split (scattered across vocab).
        if len(split_hits) >= len(split_sizes) or (col_hi - col_lo) > padded_cols * 3 // 4:
            return original(x, debug_input_torch=debug_input_torch, debug_weight_torch=debug_weight_torch)

        outputs = []
        program_configs = [
            lm_head.args.get_lm_head_program_config(split_sizes[i], lm_head.prefetcher if use_prefetcher else None)
            for i in split_hits
        ]
        output_weights = lm_head.output_weights_ring_mm if use_prefetcher else lm_head.output_weights_dram_sharded
        out_mc = lm_head.args.get_lm_head_output_mem_config(
            Mode.DECODE if use_prefetcher else Mode.PREFILL,
            lm_head.prefetcher if use_prefetcher else None,
        )
        sharded_out_mc = lm_head.args.get_lm_head_sharded_output_mem_config(
            lm_head.prefetcher if use_prefetcher else None
        )
        for split_i, pc in zip(split_hits, program_configs):
            output = ttnn.linear(
                x,
                output_weights[split_i],
                compute_kernel_config=lm_head.compute_kernel_config,
                program_config=pc,
                memory_config=out_mc,
                dtype=lm_head.args.lm_head_dtype if hasattr(lm_head.args, "lm_head_dtype") else ttnn.bfloat8_b,
                sub_device_id=lm_head.prefetcher.worker_sub_device_id if lm_head.prefetcher else None,
            )
            output = ttnn.to_memory_config(output, memory_config=sharded_out_mc)
            outputs.append(output)
        ttnn.deallocate(x)
        output = ttnn.concat(
            outputs,
            dim=-1,
            memory_config=ttnn.L1_MEMORY_CONFIG if not use_prefetcher else ttnn.DRAM_MEMORY_CONFIG,
            sub_core_grids=lm_head.prefetcher.all_worker_cores_range_set if lm_head.prefetcher else None,
        )
        if use_prefetcher:
            output = ttnn.to_memory_config(
                output,
                memory_config=lm_head.args.get_lm_head_reshard_mem_config(lm_head.prefetcher),
            )
        output = tt_all_reduce(
            output,
            lm_head.mesh_device,
            lm_head.tt_ccl,
            cluster_axis=1,
            dim=3 if lm_head.args.is_galaxy else 0,
            memory_config=output.memory_config(),
            dtype=lm_head.args.ccl_dtype,
            sharded=False,
            use_composite=True,
            subdevice_id=lm_head.prefetcher.worker_sub_device_id if lm_head.prefetcher else None,
        )
        lm_head._ace_narrow_forward_used_band = True  # type: ignore[attr-defined]
        lm_head._ace_narrow_forward_col_band = (col_lo, col_hi)  # type: ignore[attr-defined]
        band_start = ace_step_split_column_ranges(split_sizes)[split_hits[0]][0]
        local_lo = max(0, col_lo - band_start)
        local_hi = col_hi - band_start
        return ttnn.slice(
            output,
            (0, 0, 0, local_lo),
            (output.shape[0], output.shape[1], output.shape[2], local_hi),
        )

    lm_head.forward = narrow_forward  # type: ignore[method-assign]
    lm_head._ace_narrow_forward_patched = True


__all__ = [
    "ace_step_narrow_column_band",
    "ace_step_patch_lm_head_narrow_forward",
    "ace_step_split_column_ranges",
    "ace_step_splits_for_band",
]
