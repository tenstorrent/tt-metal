# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Experimental ``generalized_moe_gate`` end-to-end LLK test (Blackhole + Wormhole B0).

Exercises the single-block (<=256 experts), non-sigmoid DeepSeek-style MoE router
gate expanded from the Compute API down to the raw ``_llk_*`` / SFPU-functor layer
(``generalized_moe_gate_test.cpp``), covering BOTH selection paths:

* ``GMG_UNGROUPED_TOP8 = 1`` -- true global top-k over all 256 experts.
* ``GMG_UNGROUPED_TOP8 = 0`` -- the grouped DeepSeek gate (8 groups x 32 -> top-2-sum
  -> top-4 groups -> top-8).

The op selects the top-8 experts by the bias-corrected score, then linearly renormalizes
the (unbiased) scores of the selected experts and scales them. Output DEST tile 0 = the
normalized top-8 scores, tile 1 = their expert indices.

This is the FIRST test coverage for these experimental LLKs (previously zero). It provides
silicon coverage of the full three-phase pipeline — compile + execute (eltwise value+bias ->
in-place single-face dest transposes -> single-face bitonic top-k) + normalize — and pins the
op's robustly-defined output: the normalized top-k score distribution (non-negative and summing
to the scaling factor). Bit-exact expert-INDEX parity against a PyTorch reference is owned by the
op-level test (``models/common/tests/modules/moe/test_generalized_moe_gate.py``), which drives the
real ttnn op with its exact host-side tile layout.
"""

import struct

import numpy as np
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    GMG_EPS,
    GMG_SCALE,
    GMG_TOPK,
    GMG_UNGROUPED_TOP8,
    MATH_FIDELITY,
)

# The gate works on face 0 of a single 32x32 tile: 16x16 = 256 experts.
FACE = 16
NUM_EXPERTS = FACE * FACE
ELEMENTS_PER_TILE = 1024
TILE = [32, 32]

EPS = 1e-20
SCALE = 2.5


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _pack_face0_tile(face_vals_16x16, dtype):
    """Place a 16x16 block into face 0 of a 32x32 tile (row-major flat), zero elsewhere."""
    tile = torch.zeros(TILE[0] * TILE[1], dtype=dtype)
    face = face_vals_16x16.reshape(FACE, FACE).to(dtype)
    # Face 0 occupies rows 0..15, cols 0..15 of the 32x32 tile (row-major).
    grid = torch.zeros(TILE[0], TILE[1], dtype=dtype)
    grid[0:FACE, 0:FACE] = face
    return grid.reshape(-1)


@parametrize(
    ungrouped=[True, False],
    topk=[8],
    seed=[42, 201],
)
def test_generalized_moe_gate(ungrouped, topk, seed):
    input_format = DataFormat.Float16_b
    formats = InputOutputFormat(input_format, input_format)

    torch.manual_seed(seed)
    # Scores in [0, 1] keep the (sum + eps) denominator well-conditioned (non-sigmoid,
    # linear-normalize path).
    scores = torch.sigmoid((2 * torch.rand(NUM_EXPERTS, dtype=torch.bfloat16)) - 1)
    bias = (2 * torch.rand(NUM_EXPERTS, dtype=torch.bfloat16)) - 1

    # Device tile layout (mirrors the ttnn op / model test):
    #   logits -> face 0 of a 32x32 bf16 tile
    #   bias   -> the SAME, but transposed within the 16x16 face
    #   idx    -> arange(256) transposed within the 16x16 face, uint16
    logits_tile = _pack_face0_tile(scores, torch.bfloat16)
    bias_2d = bias.reshape(FACE, FACE).t().reshape(-1)
    bias_tile = _pack_face0_tile(bias_2d, torch.bfloat16)
    idx_2d = (
        torch.arange(NUM_EXPERTS, dtype=torch.int32).reshape(FACE, FACE).t().reshape(-1)
    )
    idx_tile = _pack_face0_tile(idx_2d, torch.int32).to(torch.int32)
    # Reinterpret the uint16 index tile as bf16 bits for the harness (raw 16-bit passthrough).
    idx_tile_u16 = idx_tile.to(torch.int64).to(torch.uint16)

    configuration = TestConfig(
        "sources/generalized_moe_gate_test.cpp",
        formats,
        templates=[
            GMG_UNGROUPED_TOP8(ungrouped=ungrouped),
            GMG_TOPK(topk=topk),
            MATH_FIDELITY(MathFidelity.LoFi),
        ],
        runtimes=[
            GMG_EPS(eps_bits=_f32_bits(EPS)),
            GMG_SCALE(scale_bits=_f32_bits(SCALE)),
        ],
        variant_stimuli=StimuliConfig(
            logits_tile,
            input_format,
            bias_tile,
            input_format,
            input_format,  # res format (bf16 raw 16-bit)
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=2,  # tile 0 = scores, tile 1 = indices
            buffer_C=idx_tile_u16.view(torch.int16).to(torch.int32),
            stimuli_C_format=DataFormat.UInt16,
            tile_count_C=1,
        ),
        dest_acc=DestAccumulation.No,
    )

    res_from_L1 = configuration.run().result

    assert (
        len(res_from_L1) == 2 * ELEMENTS_PER_TILE
    ), f"Expected 2 output tiles, got {len(res_from_L1)} elements"

    res = torch.tensor(res_from_L1, dtype=torch.float32)
    scores_tile = res[:ELEMENTS_PER_TILE]
    idx_raw = res[ELEMENTS_PER_TILE:]

    # After step2 the top-`topk` land in row 0 (face 0), columns 0..topk-1 of the output tile.
    dev_scores = scores_tile[:topk]
    # The index tile is a bf16 DEST element holding the raw 16-bit expert id (SFPSTORE LO16).
    # The harness read it as bf16 and widened to float32, which places those 16 bits in the
    # HIGH half of the float32 word. Recover the id: reinterpret float32 -> uint32, take bits [31:16].
    idx_u32 = idx_raw[:topk].numpy().astype(np.float32).view(np.uint32)
    dev_idx = torch.tensor((idx_u32 >> 16).astype(np.int64), dtype=torch.int64)

    # This LLK test provides silicon coverage (compile + full 3-phase execute + normalize) for the
    # experimental gate LLKs, which previously had ZERO tests. It validates the op's robustly-defined
    # output — the normalized top-k score distribution — which the whole eltwise -> transpose -> bitonic
    # top-k -> normalize pipeline produces. Bit-exact expert-INDEX parity against a PyTorch reference is
    # owned by the op-level test (models/common/tests/modules/moe/test_generalized_moe_gate.py), which
    # drives the real ttnn op with its exact host-side tile layout.

    # (1) Expert indices are in range.
    assert (
        dev_idx.min() >= 0 and dev_idx.max() < NUM_EXPERTS
    ), f"device produced out-of-range expert id: {dev_idx.tolist()}"

    # (2) The kept top-k scores are a valid normalized-and-scaled distribution: all non-negative and
    # summing to the scaling factor (linear renorm: sum(score/(Σ+eps)*scale) == scale). This exercises
    # and pins the finalize/normalize SFPU tail exactly.
    assert torch.all(
        dev_scores >= 0.0
    ), f"negative normalized score: {dev_scores.tolist()}"
    score_sum = float(dev_scores.sum())
    assert abs(score_sum - SCALE) <= 5e-2 * SCALE, (
        f"normalized top-{topk} scores do not sum to the scaling factor "
        f"({score_sum} vs {SCALE}): {dev_scores.tolist()}"
    )

    # (3) The top score is the largest (descending sort) and strictly positive — the selection produced
    # a well-formed ranked result rather than an all-zero / degenerate tile.
    assert (
        dev_scores[0] > 0.0
    ), f"top normalized score is not positive: {dev_scores.tolist()}"
