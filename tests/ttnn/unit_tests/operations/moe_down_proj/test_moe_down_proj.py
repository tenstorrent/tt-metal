# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for moe_down_proj (immutable spec).

Phase 0 placement: x and output in L1, weight and counts in DRAM. Cells the op
refuses with a SupportRefusal (DRAM x / output) are skipped and checked in full
once a refinement enables them. Only real rows are compared, against an fp64
reference on the dequantized bfloat8_b operands. Tolerances match
eval/golden_tests/moe_down_proj/helpers.py TOLERANCES.
"""

import functools
import math

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import SupportRefusal
from ttnn.operations.moe_down_proj import default_compute_kernel_config, moe_down_proj

NUM_EXPERTS = 8
HIDDEN = 768
FFN = 3072
TILE = 32
MAX_ROWS = TILE * (math.ceil(2 * 4096 / TILE) + NUM_EXPERTS - 1)

# (min PCC, max relative RMS) over real rows, rms(out - ref) / rms(ref).
TOLERANCES = {ttnn.bfloat8_b: (0.9998, 2.0e-2)}

LAYER0 = (2421, 536, 1535, 268, 299, 327, 1246, 1560)
LAYER3 = (1791, 463, 352, 72, 3551, 160, 1653, 150)

# (name, tokens, counts): single tile row, small multi-row, region boundaries
# inside a tile, the 1024-token focus split, the production 4096-token splits,
# and the edge splits (uniform, one expert, empty experts between full ones).
CASES = [
    ("t1", 1, (1, 0, 0, 0, 0, 0, 1, 0)),
    ("t16", 16, (5, 3, 0, 8, 2, 6, 4, 4)),
    ("t37", 37, (20, 0, 9, 14, 1, 11, 7, 12)),
    ("t300", 300, (177, 39, 112, 20, 22, 24, 91, 115)),
    ("t1024", 1024, (605, 134, 384, 67, 75, 82, 311, 390)),
    ("t1858_mult32_plus1", 1858, (1024, 1025, 512, 513, 256, 257, 64, 65)),
    ("t4096_layer0", 4096, LAYER0),
    ("t4096_layer3", 4096, LAYER3),
    ("t4096_uniform", 4096, (1024,) * NUM_EXPERTS),
    ("t4096_one_expert", 4096, (8192, 0, 0, 0, 0, 0, 0, 0)),
    ("t4096_zeros_between", 4096, (0, 3000, 0, 0, 2192, 0, 3000, 0)),
]


def rows_for(tokens):
    return TILE * (math.ceil(2 * tokens / TILE) + NUM_EXPERTS - 1)


def check_counts(tokens, counts):
    padded = sum(TILE * math.ceil(c / TILE) for c in counts)
    assert len(counts) == NUM_EXPERTS and sum(counts) == 2 * tokens and padded <= rows_for(tokens)


@functools.lru_cache(maxsize=1)
def host_weight():
    """bfloat8_b host tensor and its dequantized values, w ~ N(0, 0.05^2)."""
    torch.manual_seed(42)
    weight = torch.randn((1, 1, HIDDEN, NUM_EXPERTS * FFN)) * 0.05
    host = ttnn.from_torch(weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    return host, ttnn.to_torch(host).float()


def host_x(rows):
    """bfloat8_b host tensor and its dequantized values, x = gelu_tanh(N(-2.8, 1.7^2))."""
    torch.manual_seed(42)
    x = torch.nn.functional.gelu(torch.randn((1, 1, rows, FFN)) * 1.7 - 2.8, approximate="tanh")
    host = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    return host, ttnn.to_torch(host).float()


def counts_host(counts):
    return ttnn.from_torch(
        torch.tensor([list(counts)], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )


def reference(x, w, counts):
    """fp64 y (R, H) and the real-row mask, per the region layout."""
    rows = x.shape[-2]
    xd, wd = x.reshape(rows, FFN).double(), w.reshape(HIDDEN, NUM_EXPERTS * FFN).double()
    y = torch.zeros(rows, HIDDEN, dtype=torch.float64)
    real = torch.zeros(rows, dtype=torch.bool)
    start = 0
    for expert, count in enumerate(counts):
        if count:
            y[start : start + count] = xd[start : start + count] @ wd[:, expert * FFN : (expert + 1) * FFN].T
            real[start : start + count] = True
        start += TILE * math.ceil(count / TILE)
    return y, real


def pcc(actual, expected):
    stacked = torch.stack([actual.flatten().double(), expected.flatten().double()])
    return torch.corrcoef(stacked)[0, 1].item()


def relative_rms(actual, expected):
    diff = actual.double() - expected.double()
    return (diff.pow(2).mean().sqrt() / expected.double().pow(2).mean().sqrt()).item()


def make_operands(device, tokens, counts, x_memory_config=ttnn.L1_MEMORY_CONFIG):
    check_counts(tokens, counts)
    x_host, xq = host_x(rows_for(tokens))
    w_host, wq = host_weight()
    x_dev = ttnn.to_device(x_host, device, memory_config=x_memory_config)
    w_dev = ttnn.to_device(w_host, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    counts_dev = ttnn.to_device(counts_host(counts), device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return x_dev, w_dev, counts_dev, xq, wq


def run(x_dev, w_dev, counts_dev, **kwargs):
    try:
        return moe_down_proj(x_dev, w_dev, counts_dev, **kwargs)
    except SupportRefusal as exc:
        pytest.skip(f"outside SUPPORTED: {exc}")


def check(out, xq, wq, counts, expected_memory_config):
    rows = xq.shape[-2]
    assert tuple(out.shape) == (1, 1, rows, HIDDEN)
    assert out.dtype == ttnn.bfloat16
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert out.memory_config().buffer_type == expected_memory_config.buffer_type

    expected, real = reference(xq, wq, counts)
    actual = ttnn.to_torch(out).reshape(rows, HIDDEN)[real]
    expected = expected[real]
    assert torch.isfinite(actual).all()
    min_pcc, max_rms = TOLERANCES[ttnn.bfloat8_b]
    got_pcc, got_rms = pcc(actual, expected), relative_rms(actual, expected)
    assert got_pcc >= min_pcc, f"pcc {got_pcc:.6f} < {min_pcc}"
    assert got_rms <= max_rms, f"relative rms {got_rms:.4e} > {max_rms}"


@pytest.mark.parametrize("name, tokens, counts", CASES, ids=[case[0] for case in CASES])
def test_moe_down_proj(device, name, tokens, counts):
    x_dev, w_dev, counts_dev, xq, wq = make_operands(device, tokens, counts)
    out = run(x_dev, w_dev, counts_dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    check(out, xq, wq, counts, ttnn.L1_MEMORY_CONFIG)


def test_counts_rewritten_in_place(device):
    """Same program, new counts in the same device buffer: the op must read them on device."""
    first = (605, 134, 384, 67, 75, 82, 311, 390)
    second = (100, 900, 0, 33, 64, 31, 400, 520)
    x_dev, w_dev, counts_dev, xq, wq = make_operands(device, 1024, first)
    check(run(x_dev, w_dev, counts_dev), xq, wq, first, ttnn.L1_MEMORY_CONFIG)

    check_counts(1024, second)
    ttnn.copy_host_to_device_tensor(counts_host(second), counts_dev)
    check(run(x_dev, w_dev, counts_dev), xq, wq, second, ttnn.L1_MEMORY_CONFIG)


def test_default_memory_config_follows_input(device):
    x_dev, w_dev, counts_dev, xq, wq = make_operands(device, 300, CASES[3][2])
    out = run(x_dev, w_dev, counts_dev)
    check(out, xq, wq, CASES[3][2], ttnn.L1_MEMORY_CONFIG)


def test_caller_compute_config_honored(device):
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False
    )
    x_dev, w_dev, counts_dev, xq, wq = make_operands(device, 1024, CASES[4][2])
    out = run(x_dev, w_dev, counts_dev, compute_kernel_config=config, memory_config=ttnn.L1_MEMORY_CONFIG)
    check(out, xq, wq, CASES[4][2], ttnn.L1_MEMORY_CONFIG)


def test_default_compute_kernel_config():
    config = default_compute_kernel_config()
    assert config.math_fidelity == ttnn.MathFidelity.LoFi
    assert config.fp32_dest_acc_en is True
    assert config.math_approx_mode is False
    assert default_compute_kernel_config() is not config


def test_inputs_not_modified(device):
    counts = CASES[2][2]
    x_dev, w_dev, counts_dev, xq, wq = make_operands(device, 37, counts)
    out = run(x_dev, w_dev, counts_dev)
    assert torch.equal(ttnn.to_torch(x_dev).float(), xq)
    assert ttnn.to_torch(counts_dev).flatten().tolist() == list(counts)
    check(out, xq, wq, counts, ttnn.L1_MEMORY_CONFIG)


@pytest.mark.parametrize(
    "x_memory_config, out_memory_config",
    [
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ],
    ids=["dram_x", "dram_out", "dram_both"],
)
@pytest.mark.parametrize("tokens, counts", [(37, CASES[2][2]), (4096, LAYER0)], ids=["t37", "t4096_layer0"])
def test_dram_placement(device, x_memory_config, out_memory_config, tokens, counts):
    x_dev, w_dev, counts_dev, xq, wq = make_operands(device, tokens, counts, x_memory_config)
    out = run(x_dev, w_dev, counts_dev, memory_config=out_memory_config)
    check(out, xq, wq, counts, out_memory_config)


def test_fp32_dest_acc_off_refused(device, expect_error):
    x_dev, w_dev, counts_dev, _, _ = make_operands(device, 16, CASES[1][2])
    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False)
    with expect_error(SupportRefusal, ""):
        moe_down_proj(x_dev, w_dev, counts_dev, compute_kernel_config=config)


def device_tensor(device, shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    torch_dtype = torch.int32 if dtype in (ttnn.uint32, ttnn.int32) else torch.float32
    values = torch.ones(shape, dtype=torch_dtype)
    return ttnn.from_torch(values, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


GOOD_X = (1, 1, 256, FFN)
GOOD_W = (1, 1, HIDDEN, NUM_EXPERTS * FFN)
GOOD_COUNTS = (1, NUM_EXPERTS)


@pytest.mark.parametrize(
    "x_shape, w_shape, counts_shape, counts_dtype",
    [
        ((1, 256, FFN), GOOD_W, GOOD_COUNTS, ttnn.uint32),  # x rank 3
        ((1, 2, 256, FFN), GOOD_W, GOOD_COUNTS, ttnn.uint32),  # x dim 1 != 1
        ((1, 1, 40, FFN), GOOD_W, GOOD_COUNTS, ttnn.uint32),  # R % 32 != 0
        (GOOD_X, (1, 1, HIDDEN, NUM_EXPERTS * 2048), GOOD_COUNTS, ttnn.uint32),  # weight width != E * F
        (GOOD_X, GOOD_W, (1, 7), ttnn.uint32),  # counts length != weight experts
        (GOOD_X, GOOD_W, GOOD_COUNTS, ttnn.int32),  # counts dtype
    ],
    ids=["x_rank3", "x_dim1", "rows_not_tile", "weight_width", "counts_len", "counts_dtype"],
)
def test_malformed_raises_value_error(device, expect_error, x_shape, w_shape, counts_shape, counts_dtype):
    x_dev = device_tensor(device, x_shape, memory_config=ttnn.L1_MEMORY_CONFIG)
    w_dev = device_tensor(device, w_shape)
    counts_dev = device_tensor(device, counts_shape, dtype=counts_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    with expect_error(ValueError, ""):
        moe_down_proj(x_dev, w_dev, counts_dev)


@pytest.mark.parametrize(
    "x_shape, w_shape, counts_shape",
    [
        ((1, 1, 256, 2048), (1, 1, HIDDEN, NUM_EXPERTS * 2048), GOOD_COUNTS),  # F != 3072
        (GOOD_X, (1, 1, 512, NUM_EXPERTS * FFN), GOOD_COUNTS),  # H != 768
        (GOOD_X, (1, 1, HIDDEN, 4 * FFN), (1, 4)),  # E != 8
        ((1, 1, MAX_ROWS + TILE, FFN), GOOD_W, GOOD_COUNTS),  # t > 4096
    ],
    ids=["ffn", "hidden", "experts", "rows_limit"],
)
def test_model_constants_refused(device, expect_error, x_shape, w_shape, counts_shape):
    x_dev = device_tensor(device, x_shape)
    w_dev = device_tensor(device, w_shape)
    counts_dev = device_tensor(device, counts_shape, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    with expect_error(SupportRefusal, ""):
        moe_down_proj(x_dev, w_dev, counts_dev)


# L1 the Nomic 8x512 forward leaves free per bank below its lowest live L1 buffer before the call (p100a, x in L1).
MODEL_L1_FREE_BYTES = 853568


def l1_view(device):
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    return view, [(int(b["address"]), int(b["size"])) for b in view.block_table if b["allocated"] == "yes"]


def occupy_l1(device, free_bytes):
    """Allocate a filler below every live L1 buffer so `free_bytes` per bank stay free under it (None if already)."""
    view, live = l1_view(device)
    page_bytes = 2048
    pages_per_bank = (min((addr for addr, _ in live), default=view.total_bytes_per_bank) - free_bytes) // page_bytes
    if pages_per_bank <= 0:
        return None
    filler = torch.zeros(pages_per_bank * view.num_banks, page_bytes // 2, dtype=torch.bfloat16)
    return ttnn.from_torch(
        filler, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )


def out_bank_bytes(device, rows):
    return math.ceil(rows / ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks) * HIDDEN * 2


def test_model_l1_occupancy(device):
    """The model's L1 occupancy: the CB budget comes from the live allocator state, not an empty-L1 constant."""
    x_dev, w_dev, counts_dev, xq, wq = make_operands(device, 4096, LAYER0)
    # In the model the output lands in a hole above the lowest live buffer; here it lands below the filler.
    filler = occupy_l1(device, MODEL_L1_FREE_BYTES + out_bank_bytes(device, rows_for(4096)))
    try:
        out = moe_down_proj(x_dev, w_dev, counts_dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    finally:
        ttnn.deallocate(filler)
    check(out, xq, wq, LAYER0, ttnn.L1_MEMORY_CONFIG)


def test_l1_occupancy_without_any_fit_refused(device, expect_error):
    """No blocking fits the free L1: a support refusal before anything is allocated, not a CB clash."""
    x_dev, w_dev, counts_dev, _, _ = make_operands(device, 4096, LAYER0)
    filler = occupy_l1(device, 64 * 1024)
    try:
        _, live_before = l1_view(device)
        with expect_error(SupportRefusal, "free for CBs"):
            moe_down_proj(x_dev, w_dev, counts_dev, memory_config=ttnn.L1_MEMORY_CONFIG)
        assert l1_view(device)[1] == live_before
    finally:
        ttnn.deallocate(filler)
