# Scratch (fl round, not committed): row SUM/AVG/MAX/MIN over one row of several tiles, the result words of column 0,
# for the head-against-main identity of the tile-stepped row paths. Float16_b inputs hold every bf16 pattern over 16
# cases of four tiles (both Dest widths); Float32 inputs mix the fp32 special words with bf16-high and random words;
# both float formats also get finite-only cases; Int32 inputs mix random words with the extremes. Writes $ROWB_OUT/<case>.pt (raw column-0 words of tile 0).
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
from test_sfpu_reduce import use_int32_twos_complement

SP = [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000, 0x7F7FFFFF,
      0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF,
      0x3F800000, 0xBF800000, 0x3F800001, 0x3F7FFFFF]
IE = [0, 1, -1, 2**31 - 1, -(2**31), -(2**31) + 1, 2**30, -(2**30), 0x00FFFFFF, -0x00FFFFFF]


def to_i32(vals):
    return torch.tensor([v - (1 << 32) if v >= 2**31 else v for v in vals], dtype=torch.int64).to(torch.int32)


def bf16_case(k, n):
    bits = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16)
    perm = torch.randperm(bits.numel(), generator=torch.Generator().manual_seed(7))
    return bits[perm][(k * n) % bits.numel():(k * n) % bits.numel() + n].view(torch.bfloat16)


def fp32_case(k, n):
    g = torch.Generator().manual_seed(200 + k)
    hi = torch.randint(-(2**15), 2**15, (n // 2,), generator=g, dtype=torch.int32) << 16
    sp = to_i32(SP).repeat(8)
    rnd = torch.randint(-(2**31), 2**31 - 1, (n - hi.numel() - sp.numel(),), generator=g, dtype=torch.int64).to(torch.int32)
    x = torch.cat([hi, sp, rnd])
    return x[torch.randperm(n, generator=g)].view(torch.float32)


def finite_case(fmt, k, n):
    """Finite values only (no inf or NaN), magnitudes over about 2^-20 to 2^20, so every sum exercises the rounding."""
    g = torch.Generator().manual_seed(400 + k)
    x = torch.randn(n, generator=g) * torch.exp2(torch.randint(-20, 21, (n,), generator=g).float())
    return x.to(torch.bfloat16) if fmt == DataFormat.Float16_b else x.to(torch.float32)


def int32_case(k, n):
    g = torch.Generator().manual_seed(300 + k)
    ex = torch.tensor(IE, dtype=torch.int64).to(torch.int32).repeat(16)
    rnd = torch.randint(-(2**31), 2**31 - 1, (n - ex.numel(),), generator=g, dtype=torch.int64).to(torch.int32)
    x = torch.cat([ex, rnd])
    return x[torch.randperm(n, generator=g)]


CASES = []
for ct, ks in ((4, range(16)), (2, range(2))):
    for dest in (DestAccumulation.No, DestAccumulation.Yes):
        for k in ks:
            CASES.append((DataFormat.Float16_b, dest, ct, k))
for ct, ks in ((4, range(4)), (2, range(1))):
    for k in ks:
        CASES.append((DataFormat.Float32, DestAccumulation.Yes, ct, k))
        CASES.append((DataFormat.Int32, DestAccumulation.Yes, ct, k))
# finite inputs (case index 100+): Float16_b in both Dest widths and Float32
for k in range(100, 104):
    CASES.append((DataFormat.Float16_b, DestAccumulation.No, 4, k))
    CASES.append((DataFormat.Float16_b, DestAccumulation.Yes, 4, k))
    CASES.append((DataFormat.Float32, DestAccumulation.Yes, 4, k))
POOLS = [ReducePool.Sum, ReducePool.Average, ReducePool.Max, ReducePool.Min]


@pytest.mark.parametrize("pool", POOLS, ids=[p.name for p in POOLS])
@pytest.mark.parametrize("case", CASES, ids=[f"{f.name}-{d.name}-ct{c}-{k}" for f, d, c, k in CASES])
def test_rowct_bits(case, pool):
    fmt, dest, ct, k = case
    if fmt == DataFormat.Int32 and pool == ReducePool.Average:
        pytest.skip("integer row AVG is not implemented")
    formats = InputOutputFormat(fmt, fmt)
    n = ct * 1024
    if k >= 100:
        src = finite_case(fmt, k, n)
    elif fmt == DataFormat.Float16_b:
        src = bf16_case(k, n)
    elif fmt == DataFormat.Float32:
        src = fp32_case(k, n)
    else:
        src = int32_case(k, n)
    dims = [32, 32 * ct]
    num_blocks, per_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest, formats, dims, TILE_DIMENSIONS, BlocksCalculationAlgorithm.Standard)
    src_A = tilize_block(src, dims, stimuli_format=fmt).flatten()
    cfg = TestConfig(
        "sources/sfpu_reduce_test.cpp",
        formats,
        templates=[generate_input_dim(dims, dims), APPROX_MODE(ApproximationMode.No),
                   MATH_OP(mathop=MathOperation.ReduceRow, pool_type=pool)],
        runtimes=[NUM_BLOCKS(num_blocks), NUM_TILES_IN_BLOCK(per_block), TILE_COUNT(ct)],
        variant_stimuli=StimuliConfig(src_A, fmt, torch.zeros_like(src_A), fmt, fmt,
                                      tile_count_A=ct, tile_count_B=1, tile_count_res=ct,
                                      twos_complement=use_int32_twos_complement(formats, pool, MathOperation.ReduceRow)),
        dest_acc=dest,
        unpack_to_dest=True,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    cfg.run()
    out = os.environ.get("ROWB_OUT")
    if not out:
        return
    raw = cfg.variant_stimuli.collect_raw_result_bytes(TestConfig.TENSIX_LOCATION)
    w = np.frombuffer(raw, dtype=np.int16 if fmt == DataFormat.Float16_b else np.int32).reshape(ct, 1024)
    col0 = np.concatenate([w[0, 0:256:16], w[0, 512:768:16]])
    os.makedirs(out, exist_ok=True)
    torch.save(torch.from_numpy(col0.copy()), f"{out}/{fmt.name}-{dest.name}-ct{ct}-{k}-{pool.name}.pt")
