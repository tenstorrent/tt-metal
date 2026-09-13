# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Repro: ttnn.sign returns +/-1 for NaN where torch.sign gives 0.
# Drive in float32: a bfloat16 NaN is destroyed to +/-inf before it reaches the op (#51557).
import numpy as np
import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    FastMode,
    MathOperation,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    DestSync,
    generate_input_dim,
)

OPS = {
    "sign": MathOperation.Sign,
    "ltz": MathOperation.LessThanZero,
    "gtz": MathOperation.GreaterThanZero,
    "eqz": MathOperation.EqualZero,
}


def run_fp32(mathop, bit_patterns):
    n = len(bit_patterns)
    n_tiles = -(-n // 1024)
    padded = np.zeros(n_tiles * 1024, dtype=np.uint32)
    padded[:n] = bit_patterns
    src = torch.from_numpy(padded.view(np.float32).copy())

    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    dest_acc = DestAccumulation.Yes
    dims = [32, 32 * n_tiles]
    nb, ntb = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, dims, TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )
    cfg = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(dims, dims),
            APPROX_MODE(ApproximationMode.No),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=mathop),
        ],
        runtimes=[TILE_COUNT(n_tiles), NUM_BLOCKS(nb), NUM_TILES_IN_BLOCK(ntb)],
        variant_stimuli=StimuliConfig(
            src, DataFormat.Float32, src, DataFormat.Float32, DataFormat.Float32,
            tile_count_A=n_tiles, tile_count_B=n_tiles, tile_count_res=n_tiles,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=True,
    )
    return np.array(cfg.run().result, dtype=np.float64)[:n]


@pytest.mark.nightly
def test_sign_nan():
    labels = ["+NaN", "-NaN", "sNaN", "+0.0", "-0.0", "+inf", "-inf"]
    bits = np.array([0x7FC00000, 0xFFC00000, 0x7F800001,
                     0x00000000, 0x80000000, 0x7F800000, 0xFF800000], dtype=np.uint32)
    got = {name: run_fp32(op, bits) for name, op in OPS.items()}
    ref = torch.sign(torch.from_numpy(bits.view(np.float32).copy())).tolist()

    print(f"\n{'input':>6} | {'sign':>6} {'torch':>6} | {'ltz':>4} {'gtz':>4} {'eqz':>4}")
    for i, lab in enumerate(labels):
        print(f"{lab:>6} | {got['sign'][i]:>6.0f} {ref[i]:>6.0f} | "
              f"{got['ltz'][i]:>4.0f} {got['gtz'][i]:>4.0f} {got['eqz'][i]:>4.0f}")

    # Every row is asserted, so the guard firing on a finite input would fail here
    # rather than pass unnoticed. -0.0 is the one value this fix deliberately does
    # not move: it is a separate documented divergence tracked in #55306.
    expected = {"+NaN": 0.0, "-NaN": 0.0, "sNaN": 0.0, "+0.0": 0.0,
                "-0.0": -1.0, "+inf": 1.0, "-inf": -1.0}
    for i, lab in enumerate(labels):
        assert got["sign"][i] == expected[lab], (
            f"{lab}: sign={got['sign'][i]} expected={expected[lab]} torch={ref[i]}"
        )
