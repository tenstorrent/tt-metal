# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the compressed-matmul determinism tests.

The compressed matmul kernels take a per-tile format assignment, and the tests that hunt
for non-determinism need to state that assignment exactly rather than let
``CompressedTensorAssigner`` derive it from the data. These helpers build such an
assignment, quantize a host golden to match it, and localize a bitwise failure.

Used by the DRAM streaming tests (``test_dram_matmul_custom_compressed``), the L1
single-kernel tests (``test_matmul_custom_compressed``) and the expert matmul tests
(``per_core_allocation/test_matmul_expert``).
"""

import os

import numpy as np
import torch

from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import COMPRESSED_FORMATS, ttnn_quantize_fn

FMT_TO_IDX = {fmt: idx for idx, fmt in enumerate(COMPRESSED_FORMATS)}
IDX_TO_FMT = {idx: fmt for fmt, idx in FMT_TO_IDX.items()}

# Re-runs per determinism case. Raise for a soak:
#   COMPRESSED_MM_DETERMINISM_ITERS=500 pytest ...
DET_ITERS = int(os.environ.get("COMPRESSED_MM_DETERMINISM_ITERS", "50"))

# --- Kimi K2.7-Code production format mix ----------------------------------------------
#
# Measured over all 384 experts x 3 projections of layer 1 of the shipped
# target_3_5_32x32_native map (3.5 b/e): bfp4 54.95 %, bfp2 41.08 %, bfp0 3.96 %, bfp8 0 %.
# The uniform patterns below give bfp0 roughly eight times its production share, so a mix
# drawn at these ratios is the one that resembles what the model actually streams.
K27_FORMATS = ["bfp4", "bfp2", "bfp0"]
K27_FORMAT_RATIOS = [0.5495, 0.4108, 0.0396]
# Same mix as relative weights, for the helpers that take a {format: weight} dict.
K27_FORMAT_WEIGHTS = {"bfp4": 0.5495, "bfp2": 0.4108, "bfp0": 0.0396}

K27_HIDDEN = 7168
K27_MOE_INTERMEDIATE = 2048
K27_MOE_TP = 8
K27_PER_DEVICE_MOE_N = K27_MOE_INTERMEDIATE // K27_MOE_TP  # 256

PATTERNS = ["alternate", "pairs", "blocks", "random", "ratios"]


def build_stream_format_pattern(num_tiles, formats, pattern, period=2, seed=0, ratios=None):
    """Build a per-tile format code sequence, in the order the kernel consumes the tiles.

    The compute kernel reads tiles in pairs, so where a format change lands relative to a
    pair boundary is what ``pattern`` controls:

    - ``"alternate"``: the format changes at every tile, so one half of the changes fall
                       *inside* a pair.
    - ``"pairs"``:     the format changes every 2 tiles, so each pair holds one format and
                       changes only ever land on a pair boundary.
    - ``"blocks"``:    the format changes every ``period`` tiles (pass the subblock size to
                       keep each streamed subblock single-format).
    - ``"random"``:    a seeded draw from ``formats`` at equal probability.
    - ``"ratios"``:    a seeded draw at the probabilities in ``ratios`` (see
                       ``K27_FORMAT_RATIOS``).

    Args:
        num_tiles: length of the sequence.
        formats: format names to use, e.g. ``["bfp4", "bfp2"]``.
        pattern: one of ``PATTERNS``.
        period: run length for ``pattern="blocks"``.
        seed: RNG seed for the two drawn patterns.
        ratios: per-format probabilities for ``pattern="ratios"``, in the order of ``formats``.

    Returns:
        int8 array of length ``num_tiles`` holding COMPRESSED_FORMATS indices.
    """
    codes = np.array([FMT_TO_IDX[f] for f in formats], dtype=np.int8)
    i = np.arange(num_tiles)
    if pattern == "alternate":
        return codes[i % len(codes)]
    if pattern == "pairs":
        return codes[(i // 2) % len(codes)]
    if pattern == "blocks":
        assert period >= 1, f"period must be >= 1, got {period}"
        return codes[(i // period) % len(codes)]
    if pattern == "random":
        return codes[np.random.default_rng(seed).integers(0, len(codes), size=num_tiles)]
    if pattern == "ratios":
        assert ratios is not None, 'pattern="ratios" needs a ratios argument'
        assert len(ratios) == len(formats), f"ratios ({len(ratios)}) must match formats ({len(formats)})"
        p = np.asarray(ratios, dtype=np.float64)
        return codes[np.random.default_rng(seed).choice(len(codes), size=num_tiles, p=p / p.sum())]
    raise ValueError(f"Unknown pattern: {pattern!r}, expected one of {PATTERNS}")


def quantize_per_tile(tensor_f32, assignment, tile_w=32):
    """Host golden: quantize each tile with the format that ``assignment`` gives it.

    BFP block exponents are tile-aligned, so quantizing the whole tensor once per format and
    then selecting per tile matches quantizing tile by tile, and is far faster.
    """
    out = tensor_f32.clone()
    codes = np.repeat(np.repeat(assignment, tile_w, axis=0), tile_w, axis=1)
    codes = torch.from_numpy(codes.astype(np.int16))
    for fmt, code in FMT_TO_IDX.items():
        sel = codes == code
        if not bool(sel.any()):
            continue
        out = torch.where(sel, ttnn_quantize_fn(tensor_f32, fmt), out)
    return out


def assert_tile_counts(counts, assignment, formats):
    """Check that the packed tensor holds exactly the assignment's mix, tile for tile."""
    for fmt in FMT_TO_IDX:
        expected = int((assignment == FMT_TO_IDX[fmt]).sum())
        assert counts.get(fmt, 0) == expected, f"format {fmt}: packed {counts.get(fmt, 0)} tiles, expected {expected}"
    for fmt in formats:
        assert counts.get(fmt, 0) > 0, f"format {fmt} was requested but no tile uses it: {counts}"


def describe_mismatch(output, reference, assignment=None, per_core_n_tiles=None, tile_w=32):
    """Localize a determinism failure to cores, tile columns and tile formats."""
    diff = output != reference
    parts = [
        f"{int(diff.sum())} / {diff.numel()} elements differ",
        f"max |diff| = {(output.float() - reference.float()).abs().max().item()}",
    ]
    bad_cols = torch.nonzero(diff.any(dim=0).flatten()).flatten().tolist()
    bad_tile_cols = sorted({c // tile_w for c in bad_cols})
    if per_core_n_tiles:
        parts.append(f"core(s) {sorted({tc // per_core_n_tiles for tc in bad_tile_cols})}")
    if assignment is not None:
        fmts = sorted({IDX_TO_FMT[int(c)] for tc in bad_tile_cols for c in np.unique(assignment[:, tc])})
        parts.append(f"formats in affected columns: {fmts}")
    parts.append(f"first affected tile columns: {bad_tile_cols[:16]}")
    return "; ".join(parts)


def assert_bitwise_stable(run_once, iterations, *, context="", assignment=None, per_core_n_tiles=None, reference=None):
    """Call ``run_once()`` ``iterations`` times and require a bitwise equal result every time.

    ``run_once`` returns a torch tensor. ``reference`` seeds the comparison; when it is None
    the first call's result becomes the reference. Every divergence is collected, because how
    often it happens and whether it always hits the same tiles is the useful part.
    """
    divergences = []
    for i in range(iterations):
        output = run_once()
        if reference is None:
            reference = output.clone()
            continue
        if not torch.equal(output, reference):
            divergences.append(f"  iteration {i}: {describe_mismatch(output, reference, assignment, per_core_n_tiles)}")
    if divergences:
        detail = "\n".join(divergences[:10])
        raise AssertionError(
            f"Output is not bitwise stable over {iterations} runs"
            f"{f' ({context})' if context else ''}: {len(divergences)} differ.\n{detail}"
        )
    return reference
