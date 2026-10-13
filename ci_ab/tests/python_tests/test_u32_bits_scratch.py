# Scratch (round r2, not committed): UInt32 MAX/MIN result words on rows and columns, for the head-against-main identity.
# Every ordered pair of 16 edge words (bit-31 neighbours among them) meets in its own lane, the second word at a varying
# position; random tiles mix the edge words with random words; multi-tile rows cover the cross-tile combine. Each case
# writes its result lanes and the unsigned golden to $U32_OUT/<case>-<pool>.pt and asserts equality with the golden.
import os

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.llk_params import MathOperation, ReducePool
from test_sfpu_reduce import _run_integer_reduce

T = 32
E = [0x00000000, 0x00000001, 0x00000002, 0x0000FFFF, 0x00010000, 0x3FFFFFFF, 0x40000000, 0x7FFFFFFE,
     0x7FFFFFFF, 0x80000000, 0x80000001, 0x8000FFFF, 0xBFFFFFFF, 0xC0000000, 0xFFFFFFFE, 0xFFFFFFFF]
PAIRS = [(a, b) for a in E for b in E]
ES = [0x00000000, 0x00000001, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF]
PAIRS_S = [(a, b) for a in ES for b in ES]


def rand_tile(shape, seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randint(0, 2**32, shape, generator=g, dtype=torch.int64)
    pick = torch.randint(0, len(E), shape, generator=g)
    use_e = torch.rand(shape, generator=g) < 0.5
    return torch.where(use_e, torch.tensor(E, dtype=torch.int64)[pick], v)


def pair_lanes(pairs, lanes, length, seed):
    """lanes x length: lane i holds pairs[i][0] everywhere and pairs[i][1] at one position; spare lanes are random."""
    g = torch.Generator().manual_seed(seed)
    out = rand_tile((lanes, length), seed + 1)
    for i, (a, b) in enumerate(pairs[:lanes]):
        out[i, :] = a
        out[i, (i * 7 + 3 + int(torch.randint(0, length, (1,), generator=g))) % length] = b
    return out


def cases():
    c = {}
    c["col_pairs"] = (MathOperation.ReduceColumn, pair_lanes(PAIRS, 256, T, 11).T.contiguous())  # [32, 256]
    c["col_rand"] = (MathOperation.ReduceColumn, rand_tile((T, 8 * T), 12))
    c["row_pairs_0"] = (MathOperation.ReduceRow, pair_lanes(PAIRS[:128], 128, T, 13))  # [128, 32]
    c["row_pairs_1"] = (MathOperation.ReduceRow, pair_lanes(PAIRS[128:], 128, T, 14))
    c["row_rand_4x1"] = (MathOperation.ReduceRow, rand_tile((4 * T, T), 15))
    c["row_rand_1x2"] = (MathOperation.ReduceRow, rand_tile((T, 2 * T), 16))
    c["row_rand_2x2"] = (MathOperation.ReduceRow, rand_tile((2 * T, 2 * T), 17))
    c["row_rand_1x4"] = (MathOperation.ReduceRow, rand_tile((T, 4 * T), 18))
    c["row_pairs_1x4_0"] = (MathOperation.ReduceRow, pair_lanes(PAIRS_S[:32], T, 4 * T, 19))  # cross-tile pairs
    c["row_pairs_1x4_1"] = (MathOperation.ReduceRow, pair_lanes(PAIRS_S[32:], T, 4 * T, 20))
    return c


CASES = cases()


@pytest.mark.parametrize("reduce_pool", [ReducePool.Max, ReducePool.Min], ids=["max", "min"])
@pytest.mark.parametrize("case", list(CASES))
def test_u32_bits(case, reduce_pool):
    mathop, values = CASES[case]
    axis = 0 if mathop == MathOperation.ReduceColumn else 1
    golden = values.max(dim=axis).values if reduce_pool == ReducePool.Max else values.min(dim=axis).values
    _, res = _run_integer_reduce(mathop, reduce_pool, injected_value=None, input_format=DataFormat.UInt32, tile=values)
    res = res.to(torch.int64) & 0xFFFFFFFF
    out = os.environ.get("U32_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        torch.save({"golden": golden, "device": res}, f"{out}/{case}-{reduce_pool.name}.pt")
    bad = (golden != res).nonzero().flatten().tolist()
    assert not bad, f"{len(bad)} lanes differ from the unsigned golden, first {bad[0]}: {int(golden[bad[0]]):#010x} vs {int(res[bad[0]]):#010x}"
