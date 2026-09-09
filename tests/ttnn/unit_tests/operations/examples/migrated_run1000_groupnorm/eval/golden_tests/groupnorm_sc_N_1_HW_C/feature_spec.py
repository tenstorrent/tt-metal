# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""groupnorm_sc_N_1_HW_C feature spec — TARGET universe + golden-test INPUTS.

TARGET is the ambition: every axis value the op should eventually support.
SUPPORTED (in the op file) is what works now. Each value in
TARGET[axis] - SUPPORTED[axis] is a refinement candidate.

INPUT_TAGGERS live with the op (in ttnn/operations/groupnorm_sc_N_1_HW_C/) —
they project per-shape scalars into the `num_groups` axis and the
`alignment` axis. test_golden.py imports INPUT_TAGGERS from the op
module, TARGET + INPUTS from here.

INPUTS structure: each entry is `(shape_tuple, num_groups)` — a 2-tuple
where the first element is the input shape `(N, 1, HW, C)` and the
second is the per-shape `num_groups` scalar. `(C / num_groups) % 32 == 0`
couples shape and num_groups, so num_groups travels with the shape
rather than cartesian-iterating over a TARGET axis (which would either
produce mostly-invalid cells or require a brittle INVALID dedupe).

Expected INPUT_TAGGERS in the op file:

- `tag_num_groups(inputs, axes) -> int`. Returns `inputs[1]` —
  the per-shape num_groups scalar. Declared first so later taggers
  can read `axes["num_groups"]` if they need to.

- `tag_alignment(inputs, axes) -> "tile_aligned" | "hw_non_aligned" | "c_non_aligned"`.
  Layout is `(N, 1, HW, C)`, so the last two dims are HW (-2) and C (-1).
    * tile_aligned     — HW % 32 == 0 and C % 32 == 0.
    * c_non_aligned    — C not divisible by 32 (priority bucket; the
                          channel-axis kernel path has different masking
                          than the spatial-reduce path, so this gets its
                          own bucket).
    * hw_non_aligned   — C aligned, HW not aligned.

Non-aligned shapes are NOT structurally invalid — they are legal inputs
the op should eventually support (common in real models). They live in
TARGET, get xfail-strict via SUPPORTED gating until a refinement enables
them, and never appear in INVALID.

Adding a derived axis (verifier hook):
INPUTS exposes every per-shape scalar to taggers. To declare a new
TARGET axis derived from those scalars, add a `(inputs, axes) -> value`
function to INPUT_TAGGERS in the op file — it runs once per cartesian
combo and can read sibling axis values from `axes`, so shape-coupled
axes don't need their underlying scalars cartesian-iterated.
"""

import ttnn


TARGET = {
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned", "hw_non_aligned", "c_non_aligned"],
    # gamma/beta presence. no_affine is canonicalized via INVALID dedupe
    # on (affine_dtype, affine_layout) so the no_affine cell doesn't blow
    # up 6× per shape.
    "affine": ["gamma_beta", "gamma_only", "no_affine"],
    # gamma and beta share dtype/layout — captures the mixed-precision
    # case (bf16 input + fp32 weights) without the 9× explosion of
    # independent per-weight axes. The no_affine cell is canonicalized to the
    # "none" sentinel (coupled to `affine` in INVALID below) — an honest "no
    # weight" marker, not a real-dtype placeholder (which would falsely read as
    # "mixed input/weight" on the Features tab).
    "affine_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, "none"],
    "affine_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "none"],
}


# Cells the test harness skips because they're structurally impossible.
# Single-tensor coupling: every multi-axis cell-dict couples axes
# describing the same tensor (or canonicalizes redundancy).
INVALID = [
    # bf8b is a block-quantized format; ROW_MAJOR has no blocks.
    # Single-tensor: both axes describe the activation tensor.
    {"dtype": ttnn.bfloat8_b, "layout": ttnn.ROW_MAJOR_LAYOUT},
    # Same impossibility on the affine (gamma/beta) tensors.
    {"affine_dtype": ttnn.bfloat8_b, "affine_layout": ttnn.ROW_MAJOR_LAYOUT},
    # no_affine canonicalization via the "none" sentinel: when affine=no_affine
    # all (affine_dtype, affine_layout) cells are observationally identical.
    # Collapse to a single canonical ("none","none") cell, coupling presence <->
    # sentinel both ways.
    #   affine present (gamma_beta/gamma_only) => real dtype/layout, never sentinel
    {"affine": "gamma_beta", "affine_dtype": "none"},
    {"affine": "gamma_beta", "affine_layout": "none"},
    {"affine": "gamma_only", "affine_dtype": "none"},
    {"affine": "gamma_only", "affine_layout": "none"},
    #   affine absent (no_affine) => sentinel only
    {"affine": "no_affine", "affine_dtype": ttnn.float32},
    {"affine": "no_affine", "affine_dtype": ttnn.bfloat16},
    {"affine": "no_affine", "affine_dtype": ttnn.bfloat8_b},
    {"affine": "no_affine", "affine_layout": ttnn.TILE_LAYOUT},
    {"affine": "no_affine", "affine_layout": ttnn.ROW_MAJOR_LAYOUT},
]


# Each entry: (shape, num_groups). shape = (N, 1, HW, C); num_groups is
# the per-shape coupled scalar. Read by tag_num_groups in the op file
# as inputs[1] and projected onto the num_groups axis.
#
# num_groups travels with the shape because (C/num_groups) % 32 == 0
# ties the two together — cartesian-iterating num_groups over a TARGET
# axis would either generate mostly-INVALID cells or require a brittle
# INVALID dedupe.
#
# Shape coverage:
#  - tile_aligned (HW % 32 == 0 and C % 32 == 0): the bulk; mix of N, HW, C scales
#  - tile_aligned, (C / num_groups) % 32 != 0: SD/SDXL U-Net regime
#    (num_groups=32 with C ∈ {320, 640, 960, 1280, 1920, 2560}). Different
#    kernel path from the standard (C/G) % 32 == 0 cells — exercises
#    intra-group masking along C.
#  - hw_non_aligned (HW not aligned, C aligned): refinement-candidate path
#  - c_non_aligned (C not aligned): refinement-candidate path. Includes
#    the hardest sub-case: C non-aligned AND num_groups > 1 AND
#    (C/num_groups) % 32 != 0 — both intra-group and trailing-tile
#    masking exercised at once.
#  - wide-C LLM-realistic: C ∈ {1024, 2048, 4096} for the multi-tile reduce path
#  - multi-batch (N > 1) to catch per-N partitioning bugs
INPUTS = [
    # --- tile_aligned, minimal ---
    ((1, 1, 32, 32), 1),
    ((1, 1, 64, 64), 1),
    ((1, 1, 64, 64), 2),
    # --- tile_aligned, C scaling (HW=32 fixed) ---
    ((1, 1, 32, 128), 4),
    ((1, 1, 32, 256), 8),
    ((1, 1, 32, 512), 16),
    ((1, 1, 32, 1024), 32),
    # --- tile_aligned, HW scaling (C=64 fixed) ---
    ((1, 1, 128, 64), 2),
    ((1, 1, 256, 64), 2),
    ((1, 1, 512, 64), 2),
    ((1, 1, 1024, 64), 2),
    # --- tile_aligned, square ---
    ((1, 1, 128, 128), 4),
    ((1, 1, 256, 256), 8),
    ((1, 1, 512, 512), 16),
    # --- tile_aligned, wide C (LLM-realistic) ---
    ((1, 1, 32, 2048), 32),
    ((1, 1, 64, 4096), 32),
    ((1, 1, 128, 1024), 32),
    ((1, 1, 128, 2048), 32),
    # --- tile_aligned, odd group counts (C/G must still % 32 == 0) ---
    ((1, 1, 64, 96), 3),
    ((1, 1, 64, 160), 5),
    ((1, 1, 64, 224), 7),
    ((2, 1, 64, 288), 9),
    # --- tile_aligned, multi-batch ---
    ((2, 1, 64, 128), 4),
    ((4, 1, 128, 256), 8),
    ((8, 1, 64, 64), 2),
    ((2, 1, 512, 128), 4),
    ((4, 1, 256, 256), 8),
    # --- tile_aligned, large (stress) ---
    ((1, 1, 2048, 128), 4),
    ((1, 1, 1024, 256), 8),
    ((2, 1, 512, 256), 8),
    # --- tile_aligned, (C / num_groups) % 32 != 0 (SD / SDXL U-Net) ---
    # HW and C are both multiples of 32 (so the alignment tagger reports
    # tile_aligned), but the per-group channel count is not. This is the
    # dominant GroupNorm regime in Stable Diffusion / SDXL: num_groups=32
    # with C in {320, 640, 960, 1280, 1920, 2560}, giving per-group widths
    # in {10, 20, 30, 40, 60, 80} — none of them tile-aligned. Different
    # kernel path from the (C/G) % 32 == 0 cases above (intra-group
    # masking along C, not just whole-tile reduction).
    # SD1.5 U-Net stages (latent 64x64 → 32x32 → 16x16 → 8x8)
    ((1, 1, 4096, 320), 32),  # C/G = 10
    ((1, 1, 1024, 640), 32),  # C/G = 20
    ((1, 1, 256, 1280), 32),  # C/G = 40
    ((1, 1, 64, 1280), 32),  # C/G = 40
    # SDXL U-Net stages (latent 128x128 → 64x64 → 32x32)
    ((1, 1, 16384, 320), 32),
    ((1, 1, 4096, 640), 32),
    ((1, 1, 1024, 1280), 32),
    # SD/SDXL up-path skip-concatenated channels
    ((1, 1, 1024, 960), 32),  # C/G = 30
    ((1, 1, 256, 1920), 32),  # C/G = 60
    ((1, 1, 1024, 1920), 32),
    ((1, 1, 256, 2560), 32),  # C/G = 80
    # Small-HW variants for fast iteration / kernel-path coverage at
    # minimal cost (same per-group-non-aligned regime as above)
    ((1, 1, 32, 320), 32),
    ((1, 1, 32, 640), 32),
    ((1, 1, 32, 1280), 32),
    ((1, 1, 64, 320), 32),
    # Multi-batch in the same regime
    ((2, 1, 1024, 640), 32),
    ((2, 1, 256, 1280), 32),
    # Non-32 group counts that still produce per-group non-aligned widths.
    # C is tile-aligned, num_groups divides C, but C/num_groups % 32 != 0.
    ((1, 1, 64, 192), 8),  # C/G = 24
    ((1, 1, 64, 384), 8),  # C/G = 48
    ((1, 1, 128, 448), 8),  # C/G = 56
    ((1, 1, 64, 160), 8),  # C/G = 20
    ((2, 1, 128, 192), 8),
    # --- hw_non_aligned (HW % 32 != 0, C aligned). num_groups=1 (always valid
    # numerically when C % G == 0). These all xfail under current SUPPORTED. ---
    ((1, 1, 17, 64), 1),
    ((1, 1, 50, 128), 1),
    ((1, 1, 47, 256), 1),
    ((2, 1, 100, 128), 1),
    # --- c_non_aligned (C % 32 != 0). num_groups=1 (no C/G alignment needed). ---
    ((1, 1, 64, 17), 1),
    ((1, 1, 64, 50), 1),
    ((1, 1, 128, 100), 1),
    ((2, 1, 64, 47), 1),
    # --- c_non_aligned AND (C / num_groups) % 32 != 0, with num_groups > 1.
    # The hardest sub-case: C itself is not tile-aligned, num_groups divides
    # C, AND the per-group width is also not tile-aligned. The kernel cannot
    # rely on either whole-C tile boundaries or whole-group tile boundaries
    # — both intra-group and trailing-tile masking are exercised at once.
    ((1, 1, 64, 48), 2),  # C/G = 24
    ((1, 1, 64, 80), 2),  # C/G = 40
    ((1, 1, 128, 48), 3),  # C/G = 16
    ((1, 1, 64, 80), 4),  # C/G = 20
    ((1, 1, 128, 144), 4),  # C/G = 36
    ((1, 1, 64, 200), 8),  # C/G = 25
    ((2, 1, 64, 48), 2),
]


# Hand-authored cases that don't fall out of TARGET × INPUTS cartesian.
# Each entry is one test — `inputs` mirrors the INPUTS-entry shape for
# this op, and every finite TARGET axis is pinned to one value. Shape-
# derived axes (those declared in the op's INPUT_TAGGERS) are NOT pinned
# — the test harness projects them via INPUT_TAGGERS so the resulting
# axes dict has identical shape to a cartesian-generated case and routes
# through is_supported / invalid_reason the same way.
#
# Optional reserved field per entry:
#   "extras": dict — loose-case-only runner overrides (precision
#     thresholds, input distribution, etc.). NOT part of the axes dict
#     and NOT used by is_supported. Threaded into the runner separately.
#     Defaults to {} when omitted.
#
# Use this list for: regression repros for specific bugs, model-derived
# shape profiles, and edge combinations the cartesian grid doesn't pass
# through. Add liberally — each entry costs one parametrize row, runs
# once per pytest invocation, and contributes to the same dashboard
# counts as generated cases.
LOOSE_CASES = []
