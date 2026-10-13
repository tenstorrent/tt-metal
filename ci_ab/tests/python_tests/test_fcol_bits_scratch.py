# Scratch (ra2, not committed): float column SUM/AVG result words, row 0 of every tile, for the head-against-main
# identity of the paired column groups. Float16_b inputs hold every bf16 pattern over four cases of 16 tiles; Float32
# inputs mix the fp32 special words with bf16-high and random words. Writes $FCOL_OUT/<case>.pt (raw row-0 words).
import os

import numpy as np
import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    MathOperation,
    ReducePool,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block

N = 16
SP = [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000, 0x7F7FFFFF,
      0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF,
      0x3F800000, 0xBF800000, 0x3F800001, 0x3F7FFFFF]


def bf16_case(k):
    bits = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16)
    perm = torch.randperm(bits.numel(), generator=torch.Generator().manual_seed(5))
    return bits[perm][k * N * 1024:(k + 1) * N * 1024].view(torch.bfloat16)


def fp32_case(k):
    g = torch.Generator().manual_seed(100 + k)
    n = N * 1024
    hi = (torch.randint(-(2**15), 2**15, (n // 2,), generator=g, dtype=torch.int32) << 16)
    sp = torch.tensor([v - (1 << 32) if v >= 2**31 else v for v in SP], dtype=torch.int64).repeat(64).to(torch.int32)
    rnd = torch.randint(-(2**31), 2**31 - 1, (n - hi.numel() - sp.numel(),), generator=g, dtype=torch.int64).to(torch.int32)
    x = torch.cat([hi, sp, rnd])
    return x[torch.randperm(n, generator=g)].view(torch.float32)


CASES = []
for dest in (DestAccumulation.No, DestAccumulation.Yes):
    for k in range(4):
        CASES.append((DataFormat.Float16_b, dest, k))
for k in range(4):
    CASES.append((DataFormat.Float32, DestAccumulation.Yes, k))


@pytest.mark.parametrize("pool", [ReducePool.Sum, ReducePool.Average], ids=["sum", "avg"])
@pytest.mark.parametrize("case", CASES, ids=[f"{f.name}-{d.name}-{k}" for f, d, k in CASES])
def test_fcol_bits(case, pool):
    fmt, dest, k = case
    formats = InputOutputFormat(fmt, fmt)
    src = bf16_case(k) if fmt == DataFormat.Float16_b else fp32_case(k)
    dims = [32, 32 * N]
    num_blocks, per_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest, formats, dims, TILE_DIMENSIONS, BlocksCalculationAlgorithm.Standard)
    src_A = tilize_block(src, dims, stimuli_format=fmt).flatten()
    cfg = TestConfig(
        "sources/sfpu_reduce_test.cpp",
        formats,
        templates=[generate_input_dim(dims, dims), APPROX_MODE(ApproximationMode.No),
                   MATH_OP(mathop=MathOperation.ReduceColumn, pool_type=pool)],
        runtimes=[NUM_BLOCKS(num_blocks), NUM_TILES_IN_BLOCK(per_block), TILE_COUNT(N)],
        variant_stimuli=StimuliConfig(src_A, fmt, torch.zeros_like(src_A), fmt, fmt,
                                      tile_count_A=N, tile_count_B=1, tile_count_res=N),
        dest_acc=dest,
        unpack_to_dest=True,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    cfg.run()
    out = os.environ.get("FCOL_OUT")
    if not out:
        return
    raw = cfg.variant_stimuli.collect_raw_result_bytes(TestConfig.TENSIX_LOCATION)
    w = np.frombuffer(raw, dtype=np.int16 if fmt == DataFormat.Float16_b else np.int32).reshape(N, 1024)
    row0 = np.concatenate([w[:, 0:16], w[:, 256:272]], axis=1)
    os.makedirs(out, exist_ok=True)
    torch.save(torch.from_numpy(row0.copy()), f"{out}/{fmt.name}-{dest.name}-{k}-{pool.name}.pt")
