# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np
import torch

from .metrics import metric_is_good
from .tile_utils import (
    COMPRESSED_BYTES_PER_ELEM,
    COMPRESSED_FORMATS,
    compressed_total_bytes,
    reconstruct_from_tiles,
    reshape_to_2d_with_padding,
    tile_metrics,
)


@dataclass
class CompressedTensorResult:
    """Result of compressed tensor format assignment.

    Attributes:
        assignment: (tiles_h, tiles_w) int8 array of format indices into COMPRESSED_FORMATS.
        tile_counts: Number of tiles assigned to each format, e.g. {"bfp8": 12, "bfp4": 988}.
        total_bytes: Estimated total byte cost of the compressed tensor tensor.
        quantized: Reconstructed float32 tensor after per-tile quantization (same shape as input).
    """

    assignment: np.ndarray
    tile_counts: dict[str, int] = field(default_factory=dict)
    total_bytes: float = 0.0
    quantized: Union[np.ndarray, torch.Tensor, None] = None


class CompressedTensorAssigner:
    """Assigns per-tile quantization formats using a threshold-based algorithm.

    For each 32x32 tile, tries formats from cheapest to most expensive and picks
    the first one that meets the quality threshold. Tiles that fail all cheap formats
    get assigned the most expensive candidate.

    Args:
        metric: Quality metric -- "pcc", "mae", or "atol".
        threshold: Quality threshold (for pcc: minimum correlation, for mae/atol: maximum error).
        formats: List of candidate format strings from COMPRESSED_FORMATS.
                 Defaults to ["bfp8", "bfp4"].
    """

    def __init__(
        self,
        metric: str = "pcc",
        threshold: float = 0.999,
        formats: list[str] | None = None,
        bfp0_mae_threshold: float = 0.01,
    ) -> None:
        if metric not in {"pcc", "mae", "atol"}:
            raise ValueError(f"Unsupported metric: {metric}")
        self.metric = metric
        self.threshold = threshold
        self.bfp0_mae_threshold = bfp0_mae_threshold
        self.formats = formats or ["bfp8", "bfp4"]
        for fmt in self.formats:
            if fmt not in COMPRESSED_FORMATS:
                raise ValueError(f"Unsupported format: {fmt}")

    def assign(
        self,
        weight_fp32: Union[np.ndarray, torch.Tensor],
        quantize_fn: Callable,
    ) -> CompressedTensorResult:
        """Run threshold-based compressed tensor assignment.

        Args:
            weight_fp32: Float32 weight tensor of any shape (numpy or torch).
            quantize_fn: Callable(tensor, fmt_str) -> quantized_tensor.
                         Accepts the same type as weight_fp32 (numpy or torch) and a format
                         string (e.g. "bfp8"). Must return the same type/shape (quantize-dequantize
                         round-trip).

        Returns:
            CompressedTensorResult with per-tile assignments, counts, byte cost, and reconstructed tensor.
            If input was a torch tensor, quantized will be a torch tensor.
        """
        input_is_torch = isinstance(weight_fp32, torch.Tensor)
        tile_hw = 32

        # Single-format short-circuit. With one candidate format the fallback
        # ``best_precision`` and the only entry of ``formats_by_cost`` are the
        # same, so every tile is assigned that format regardless of its score.
        # Skipping the per-format ``quantize_fn`` + ``tile_metrics`` + per-tile
        # assignment loop avoids the dominant host CPU cost when production uses
        # a single format (``formats=["bfp4"]``).  This also
        # bypasses the ``.float().cpu().numpy()`` copy and the padded-zeros
        # allocation -- only the input shape is needed.  ``quantized`` is left
        # as ``None`` because no current caller reads it; callers that need
        # the round-tripped tensor must use a multi-format assigner.
        if len(self.formats) == 1:
            shape = tuple(weight_fp32.shape)
            if 0 in shape or len(shape) == 0:
                # Empty / scalar inputs share the legacy 1x1 "empty" return
                # below; fall through to keep behavior identical for those.
                pass
            else:
                if len(shape) == 1:
                    h, w = 1, shape[0]
                else:
                    w = shape[-1]
                    h = 1
                    for d in shape[:-1]:
                        h *= d
                tiles_h = (h + tile_hw - 1) // tile_hw
                tiles_w = (w + tile_hw - 1) // tile_hw
                num_tiles = tiles_h * tiles_w
                fmt = self.formats[0]
                fmt_to_idx = {f: idx for idx, f in enumerate(COMPRESSED_FORMATS)}
                assignment = np.full((tiles_h, tiles_w), fmt_to_idx[fmt], dtype=np.int8)
                counts = {f: 0 for f in COMPRESSED_FORMATS}
                counts[fmt] = num_tiles
                return CompressedTensorResult(
                    assignment=assignment,
                    tile_counts=counts,
                    total_bytes=compressed_total_bytes(counts),
                    quantized=None,
                )

        if input_is_torch:
            input_device = weight_fp32.device
            xf = weight_fp32.detach().float().cpu().numpy()
        else:
            xf = np.asarray(weight_fp32, dtype=np.float32)

        if xf.size == 0:
            empty_q = torch.zeros_like(weight_fp32) if input_is_torch else xf.copy()
            return CompressedTensorResult(
                assignment=np.zeros((1, 1), dtype=np.int8),
                tile_counts={fmt: 0 for fmt in COMPRESSED_FORMATS},
                total_bytes=0.0,
                quantized=empty_q,
            )

        padded_ref, shape_info, pad_info = reshape_to_2d_with_padding(xf)
        tiles_h = pad_info[2] // tile_hw
        tiles_w = pad_info[3] // tile_hw
        tiles_ref = (
            padded_ref.reshape(tiles_h, tile_hw, tiles_w, tile_hw).transpose(0, 2, 1, 3).reshape(-1, tile_hw, tile_hw)
        )

        # Quantize entire tensor per format, then split into tiles and score
        tiles_by_fmt: dict[str, np.ndarray] = {}
        scores_by_fmt: dict[str, np.ndarray] = {}

        for fmt in self.formats:
            if input_is_torch:
                y_fmt = quantize_fn(weight_fp32, fmt)
                if isinstance(y_fmt, torch.Tensor):
                    y_fmt = y_fmt.detach().float().cpu().numpy()
            else:
                y_fmt = quantize_fn(xf, fmt)
            y_fmt = np.asarray(y_fmt, dtype=np.float32)
            padded_q, _, pad_info_q = reshape_to_2d_with_padding(y_fmt)
            if pad_info_q != pad_info:
                raise ValueError("Quantized tensor padding mismatch.")
            tiles_q = (
                padded_q.reshape(tiles_h, tile_hw, tiles_w, tile_hw).transpose(0, 2, 1, 3).reshape(-1, tile_hw, tile_hw)
            )
            tiles_by_fmt[fmt] = tiles_q
            # bfp0 always uses MAE (PCC is undefined for all-zeros vs signal)
            fmt_metric = "mae" if fmt == "bfp0" else self.metric
            scores_by_fmt[fmt] = tile_metrics(tiles_ref, tiles_q, fmt_metric)

        # Sort formats cheapest first; fallback is the most expensive
        fmt_to_idx = {fmt: idx for idx, fmt in enumerate(COMPRESSED_FORMATS)}
        formats_by_cost = sorted(self.formats, key=lambda f: COMPRESSED_BYTES_PER_ELEM.get(f, 0.0))
        best_precision = max(self.formats, key=lambda f: COMPRESSED_BYTES_PER_ELEM.get(f, 0.0))

        # Assign each tile: try cheapest first, fall back to most expensive
        num_tiles = tiles_ref.shape[0]
        assignments = np.full((num_tiles,), fmt_to_idx[best_precision], dtype=np.int8)
        for tile_idx in range(num_tiles):
            for fmt in formats_by_cost:
                score = scores_by_fmt[fmt][tile_idx]
                # bfp0 uses its own MAE threshold
                if fmt == "bfp0":
                    if metric_is_good(score, "mae", self.bfp0_mae_threshold):
                        assignments[tile_idx] = fmt_to_idx[fmt]
                        break
                elif metric_is_good(score, self.metric, self.threshold):
                    assignments[tile_idx] = fmt_to_idx[fmt]
                    break

        # Assemble output tiles
        tiles_out = tiles_ref.copy()
        for fmt in self.formats:
            tile_ids = np.where(assignments == fmt_to_idx[fmt])[0]
            if tile_ids.size > 0:
                tiles_out[tile_ids] = tiles_by_fmt[fmt][tile_ids]

        quantized = reconstruct_from_tiles(tiles_out, shape_info, pad_info, tile_hw=tile_hw)

        counts = {fmt: 0 for fmt in COMPRESSED_FORMATS}
        for fmt in self.formats:
            counts[fmt] = int(np.sum(assignments == fmt_to_idx[fmt]))

        if input_is_torch:
            quantized = torch.from_numpy(quantized).to(device=input_device)

        return CompressedTensorResult(
            assignment=assignments.reshape(tiles_h, tiles_w),
            tile_counts=counts,
            total_bytes=compressed_total_bytes(counts),
            quantized=quantized,
        )
