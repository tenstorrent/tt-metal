# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Load BitSculpt BSPM files and remap codes to tt-metal COMPRESSED_FORMATS indices.

BitSculpt code ordering: 0=zero, 1=bfp2, 2=bfp4, 3=bfp8
tt-metal code ordering:  0=bfp8, 1=bfp4, 2=bfp2, 3=bfp0 (COMPRESSED_FORMATS)

This module provides functions to load BSPMs and transparently remap codes
so they can be fed directly into CompressedTensor.from_bspm() or __init__().

Usage:
    from models.demos.deepseek_v3_b1.compressed_tensor.bspm_loader import (
        load_bspm_for_layer, load_bspm_for_expert
    )

    # Full layer (all experts)
    data = load_bspm_for_layer("path/to/precision_map_B_3.5.bspm")
    ttnn_codes = data["codes"]  # (n_experts, 3, tiles_per_proj), tt-metal indices

    # Single expert projection → (tiles_h, tiles_w) for CompressedTensor
    assignment = load_bspm_for_expert("path/to/precision_map_B_3.5.bspm", expert_idx=0, proj_idx=0)

Prerequisites:
    Only the .bspm result files from BitSculpt are required — no BitSculpt Python
    package dependency.  The binary format is parsed directly here.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# BSPM binary format constants (must stay in sync with BitSculpt's export.py)
# ---------------------------------------------------------------------------

_BSPM_MAGIC = b"BSPM"
_BSPM_VERSION = 1
_BSPM_HEADER_SIZE = 64  # bytes; header is 48 bytes of fields + 16 bytes reserved
# struct layout: "<4sIIIIIIIIB3xII" = 48 bytes
#   4s  magic
#   I   version
#   I   layer_idx
#   I   n_experts
#   I   n_projections
#   I   tiles_per_proj
#   I   tile_rows
#   I   tile_cols
#   I   tile_size
#   B   variant_code   (A=0, B=1, C=2)
#   3x  padding
#   I   budget_millibits
#   I   actual_millibits
_BSPM_STRUCT = struct.Struct("<4sIIIIIIIIB3xII")
_VARIANT_NAMES = {0: "A", 1: "B", 2: "C"}

# ---------------------------------------------------------------------------
# Code remapping
# ---------------------------------------------------------------------------

# BitSculpt → tt-metal COMPRESSED_FORMATS index
# BitSculpt:  0=zero, 1=bfp2, 2=bfp4, 3=bfp8
# tt-metal:   0=bfp8, 1=bfp4, 2=bfp2, 3=bfp0
_BITSCULPT_TO_TTNN = np.array([3, 2, 1, 0], dtype=np.int8)

# tt-metal COMPRESSED_FORMATS index → BitSculpt code (inverse)
_TTNN_TO_BITSCULPT = np.array([3, 2, 1, 0], dtype=np.uint8)

# Readable names for validation/debugging
BITSCULPT_FORMATS = {0: "zero", 1: "bfp2", 2: "bfp4", 3: "bfp8"}
TTNN_FORMATS = {0: "bfp8", 1: "bfp4", 2: "bfp2", 3: "bfp0"}


def remap_bspm_to_ttnn(codes: np.ndarray) -> np.ndarray:
    """Remap BitSculpt BSPM codes to tt-metal COMPRESSED_FORMATS indices.

    Args:
        codes: uint8 array of any shape with values in {0, 1, 2, 3} (BitSculpt convention).

    Returns:
        int8 array of same shape with values in {0, 1, 2, 3} (tt-metal convention).
    """
    return _BITSCULPT_TO_TTNN[codes.astype(np.uint8)]


def remap_ttnn_to_bspm(codes: np.ndarray) -> np.ndarray:
    """Remap tt-metal COMPRESSED_FORMATS indices back to BitSculpt codes.

    Args:
        codes: int8/uint8 array with tt-metal format indices.

    Returns:
        uint8 array with BitSculpt codes.
    """
    return _TTNN_TO_BITSCULPT[codes.astype(np.uint8)]


# ---------------------------------------------------------------------------
# BSPM loading with remapping
# ---------------------------------------------------------------------------


def load_bspm_for_layer(
    bspm_path: str | Path,
    expected_n_experts: int | None = None,
) -> dict:
    """Load a BSPM file and remap all codes to tt-metal format indices.

    Args:
        bspm_path: Path to a .bspm file.
        expected_n_experts: Optional validation of expert count.

    Returns:
        Dict with all BSPM metadata plus:
            codes: np.ndarray (n_experts, 3, tiles_per_proj) with tt-metal format indices
            codes_bitsculpt: np.ndarray — original BitSculpt codes (for debugging)
    """
    with open(bspm_path, "rb") as f:
        raw = f.read(_BSPM_HEADER_SIZE)

    if len(raw) < 4:
        raise ValueError(f"File too small to be a BSPM: {bspm_path}")

    # Detect Git LFS pointer files before attempting to parse.
    # LFS pointers start with "version https://..." — reading the magic will
    # produce an invalid-magic error that is hard to diagnose.
    if raw[:4] == b"vers":
        raise RuntimeError(
            f"BSPM file appears to be a Git LFS pointer, not the actual data: {bspm_path}\n"
            "Run `git lfs install && git lfs pull` in the bit_sculpt repo to download the real files."
        )

    if len(raw) < _BSPM_HEADER_SIZE:
        raise ValueError(f"File too small for BSPM header: {len(raw)} bytes in {bspm_path}")

    (
        magic,
        version,
        layer_idx,
        n_experts,
        n_projections,
        tiles_per_proj,
        tile_rows,
        tile_cols,
        tile_size,
        variant_code,
        budget_millibits,
        actual_millibits,
    ) = _BSPM_STRUCT.unpack(raw[: _BSPM_STRUCT.size])

    if magic != _BSPM_MAGIC:
        raise ValueError(f"Invalid BSPM magic {magic!r} in {bspm_path}")
    if version != _BSPM_VERSION:
        raise ValueError(f"Unsupported BSPM version {version} in {bspm_path}")

    if expected_n_experts is not None and n_experts != expected_n_experts:
        raise ValueError(
            f"BSPM n_experts mismatch in {bspm_path}: "
            f"header={n_experts}, expected={expected_n_experts}. "
            "Delete the BSPM file and re-run allocation."
        )

    body_size = n_experts * n_projections * tiles_per_proj
    with open(bspm_path, "rb") as f:
        f.seek(_BSPM_HEADER_SIZE)
        body_bytes = f.read(body_size)

    if len(body_bytes) < body_size:
        raise ValueError(f"Truncated BSPM body in {bspm_path}: got {len(body_bytes)}, expected {body_size}")

    original_codes = np.frombuffer(body_bytes, dtype=np.uint8).reshape(n_experts, n_projections, tiles_per_proj)

    return {
        "magic": magic.decode("ascii"),
        "version": version,
        "layer_idx": layer_idx,
        "n_experts": n_experts,
        "n_projections": n_projections,
        "tiles_per_proj": tiles_per_proj,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "tile_size": tile_size,
        "variant": _VARIANT_NAMES.get(variant_code, "?"),
        "budget": budget_millibits / 1000.0,
        "actual_bits": actual_millibits / 1000.0,
        "codes_bitsculpt": original_codes,
        "codes": remap_bspm_to_ttnn(original_codes),
    }


def load_bspm_for_expert(
    bspm_path: str | Path,
    expert_idx: int,
    proj_idx: int,
    tile_rows: int | None = None,
    tile_cols: int | None = None,
) -> np.ndarray:
    """Load BSPM and return a 2D tt-metal assignment array for one expert projection.

    This is the primary interface for CompressedTensor integration. The returned
    array can be passed directly to CompressedTensor.__init__() or from_bspm().

    Args:
        bspm_path: Path to a .bspm file.
        expert_idx: Expert index (0 to n_experts-1).
        proj_idx: Projection index (0=gate, 1=up, 2=down).
        tile_rows: Number of tile rows. If None, inferred from BSPM header.
        tile_cols: Number of tile cols. If None, inferred from BSPM header.

    Returns:
        (tiles_h, tiles_w) int8 array with tt-metal COMPRESSED_FORMATS indices.
    """
    data = load_bspm_for_layer(str(bspm_path))
    codes = data["codes"]  # already remapped to tt-metal

    n_experts = data["n_experts"]
    tiles_per_proj = data["tiles_per_proj"]

    if expert_idx < 0 or expert_idx >= n_experts:
        raise IndexError(f"expert_idx={expert_idx} out of range [0, {n_experts})")
    if proj_idx < 0 or proj_idx >= 3:
        raise IndexError(f"proj_idx={proj_idx} out of range [0, 3)")

    flat_codes = codes[expert_idx, proj_idx]  # (tiles_per_proj,)

    # Determine tile grid shape
    tr = tile_rows if tile_rows is not None else data.get("tile_rows", 0)
    tc = tile_cols if tile_cols is not None else data.get("tile_cols", 0)

    if tr == 0 or tc == 0:
        # Try to infer from tiles_per_proj (square-ish factorization)
        # For R1: 14336 = 64 × 224 (2048/32 × 7168/32)
        # This is a heuristic; caller should provide tile_rows/tile_cols for correctness
        import math

        sqrt_n = int(math.isqrt(tiles_per_proj))
        for candidate in range(sqrt_n, 0, -1):
            if tiles_per_proj % candidate == 0:
                tr = candidate
                tc = tiles_per_proj // candidate
                break

    if tr * tc != tiles_per_proj:
        raise ValueError(
            f"tile_rows={tr} × tile_cols={tc} = {tr * tc} != tiles_per_proj={tiles_per_proj}. "
            f"Provide correct tile_rows and tile_cols."
        )

    return flat_codes.reshape(tr, tc)


# ---------------------------------------------------------------------------
# Batch loading for full model
# ---------------------------------------------------------------------------


def load_all_layer_bspms(
    bspm_dir: str | Path,
    model_short_name: str,
    variant: str,
    budget: float,
    layer_indices: list[int],
    expected_n_experts: int | None = None,
) -> dict[int, np.ndarray]:
    """Load BSPMs for multiple layers. Returns {layer_idx: remapped_codes}.

    Follows BitSculpt's file naming convention:
        {bspm_dir}/{model_short_name}/layer_{idx}/precision_eval/precision_map_{variant}_{budget:.1f}.bspm

    Args:
        bspm_dir: Root directory containing results (e.g., "results").
        model_short_name: Model short name (e.g., "deepseek-r1-0528").
        variant: Allocation variant (e.g., "B").
        budget: Bit budget (e.g., 3.5).
        layer_indices: List of layer indices to load.
        expected_n_experts: Optional validation.

    Returns:
        Dict mapping layer_idx → remapped codes array (n_experts, 3, tiles_per_proj).
        Missing layers are silently skipped.
    """
    bspm_dir = Path(bspm_dir)
    result = {}
    for layer_idx in layer_indices:
        bspm_path = (
            bspm_dir
            / model_short_name
            / f"layer_{layer_idx}"
            / "precision_eval"
            / f"precision_map_{variant}_{budget:.1f}.bspm"
        )
        if not bspm_path.exists():
            continue
        data = load_bspm_for_layer(bspm_path, expected_n_experts=expected_n_experts)
        result[layer_idx] = data["codes"]
    return result


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def summarize_bspm(bspm_path: str | Path) -> dict:
    """Load a BSPM and return summary statistics.

    Returns dict with:
        layer_idx, variant, budget, n_experts, tiles_per_proj,
        tile_counts (both BitSculpt and tt-metal names),
        avg_bits_per_element
    """
    data = load_bspm_for_layer(bspm_path)
    bs_codes = data["codes_bitsculpt"]

    # Count per BitSculpt format
    bs_counts = {BITSCULPT_FORMATS[i]: int((bs_codes == i).sum()) for i in range(4)}

    # Average bits per element
    bits_map = {0: 0.0, 1: 2.5, 2: 4.5, 3: 8.5}  # BitSculpt codes → b/e
    total_tiles = bs_codes.size
    total_bits = sum(bits_map[code] * count for code, count in enumerate(np.bincount(bs_codes.ravel(), minlength=4)))
    avg_be = total_bits / total_tiles if total_tiles > 0 else 0.0

    return {
        "layer_idx": data["layer_idx"],
        "variant": data["variant"],
        "budget": data["budget"],
        "actual_bits": data["actual_bits"],
        "n_experts": data["n_experts"],
        "tiles_per_proj": data["tiles_per_proj"],
        "tile_counts": bs_counts,
        "avg_bits_per_element": round(avg_be, 3),
        "total_tiles": total_tiles,
    }
