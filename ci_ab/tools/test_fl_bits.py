# Scratch bit dump (not for the PR): raw result bytes of one unary SFPU op over every bfloat16 pattern, and over
# fp32 words with every sign/exponent/top-mantissa pattern, written to $FL_BITS_OUT/<case>.npz for word compares.
import os

import numpy as np
import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    FastMode,
    MathOperation,
)
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
    generate_input_dim,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.golden_generators import TILE_DIMENSIONS

OUT = os.environ.get("FL_BITS_OUT", "/tmp/fl_bits")
OPS = {"identity": MathOperation.Identity, "softsign": MathOperation.Softsign}


def _inputs(kind):
    i = torch.arange(65536, dtype=torch.int64)
    if kind == "bf16":
        return (i << 16).to(torch.int32).view(torch.float32)
    lo = (i * 2654435761) & 0xFFFF
    return ((i << 16) | lo).to(torch.int32).view(torch.float32)


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("case", ["bf16_d16", "bf16_d32", "fp32_d32"])
def test_fl_bits(op, case):
    if case == "fp32_d32":
        formats, dest_acc, kind, utd = InputOutputFormat(DataFormat.Float32, DataFormat.Float32), DestAccumulation.Yes, "fp32", True
    else:
        formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
        dest_acc, kind, utd = (DestAccumulation.Yes if case == "bf16_d32" else DestAccumulation.No), "bf16", False
    src_A = _inputs(kind)
    dims = [256, 256]
    tile_cnt = 64
    nb, nt = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, dims, TILE_DIMENSIONS, BlocksCalculationAlgorithm.Standard
    )
    cfg = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(dims, dims),
            APPROX_MODE(ApproximationMode.No),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=OPS[op]),
        ],
        runtimes=[TILE_COUNT(tile_cnt), NUM_BLOCKS(nb), NUM_TILES_IN_BLOCK(nt)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            torch.zeros_like(src_A),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=tile_cnt,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=utd,
    )
    cfg.run()
    raw = cfg.variant_stimuli.collect_raw_result_bytes(TestConfig.TENSIX_LOCATION)
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/{op}_{case}.npz", res=np.frombuffer(raw, dtype=np.uint8), src=src_A.view(torch.int32).numpy())
