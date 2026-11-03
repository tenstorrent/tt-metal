# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.nomic_embed_text_v2_moe.router_front_end against the stock ops it fuses.

The scores are softmax(input @ weight) over the weight's columns, bit for bit what ttnn.linear with K in one
block and ttnn.softmax return under the same compute config: the softmax masks the tile's padded columns as
ttnn.softmax does for a width off the tile grid. The rows are the input untilized, bit for bit.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device]

HIDDEN = 768
NUM_EXPERTS = 8
TILE = 32

# The router's budget for the scores' max-abs against float64; the op measures 1.6e-3 to 2.4e-3 at the
# model's router weights.
MAX_ABS = 5e-3

router_front_end = ttnn.experimental.nomic_embed_text_v2_moe.router_front_end


def kernel_config(device, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=False,
    )


def make_input(tokens: int, generator: torch.Generator, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    x = torch.randn(1, 1, tokens, HIDDEN, generator=generator).bfloat16()
    return x, ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def make_weight(experts: int, generator: torch.Generator, device):
    """Logits of about unit variance from unit-variance inputs."""
    weight = torch.randn(1, 1, HIDDEN, experts, generator=generator) / HIDDEN**0.5
    return weight, ttnn.from_torch(weight, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def stock_scores(x_tt, weight_tt, device):
    """ttnn.linear with K in one block, then ttnn.softmax, under the op's default config."""
    grid = device.compute_with_storage_grid_size()
    m_tiles = -(-x_tt.shape[-2] // TILE)
    per_core_m = -(-m_tiles // (grid.x * grid.y))
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=HIDDEN // TILE,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=per_core_m,
        out_block_w=1,
        per_core_M=per_core_m,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )
    config = kernel_config(device)
    logits = ttnn.linear(
        x_tt, weight_tt, program_config=program_config, compute_kernel_config=config, dtype=ttnn.float32
    )
    return ttnn.to_torch(ttnn.softmax(logits, dim=-1, compute_kernel_config=config))


def exact_scores(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.double() @ weight.double(), dim=-1)


def whole_tiles(scores) -> torch.Tensor:
    """The scores' tiles with their padding, (tokens rounded up to 32, 32)."""
    padded = list(scores.padded_shape)
    return ttnn.to_torch(ttnn.reshape(scores, padded, padded)).reshape(padded[-2], padded[-1])


def assert_same_bits(got: torch.Tensor, want: torch.Tensor):
    bits = torch.int32 if got.dtype == torch.float32 else torch.int16
    assert got.shape == want.shape
    assert torch.equal(got.contiguous().view(bits), want.contiguous().view(bits))


@pytest.mark.parametrize(
    "tokens, experts",
    [(1, 8), (37, 8), (1000, 8), (4096, 8), (32768, 8), (1000, 1), (1000, 32)],
    ids=["1", "37", "1000", "4096", "32768", "1000_1_expert", "1000_32_experts"],
)
def test_matches_the_stock_ops(device, tokens, experts):
    """Scores bit for bit those of ttnn.linear and ttnn.softmax, within the budget of float64, zero on the
    tile's padded columns; rows the input's bits. 37 tokens leave a partial block, 4096 give 18 cores two
    blocks, 32768 every core nine or ten; 32 experts fill the tile, so nothing is masked."""
    generator = torch.Generator().manual_seed(tokens + experts)
    x, x_tt = make_input(tokens, generator, device)
    weight, weight_tt = make_weight(experts, generator, device)

    scores, rows = router_front_end(x_tt, weight_tt)

    assert list(scores.shape) == [1, 1, tokens, experts]
    assert scores.dtype == ttnn.float32 and scores.layout == ttnn.TILE_LAYOUT
    assert list(rows.shape) == [1, 1, tokens, HIDDEN]
    assert rows.dtype == ttnn.bfloat16 and rows.layout == ttnn.ROW_MAJOR_LAYOUT
    got = ttnn.to_torch(scores)
    assert_same_bits(got, stock_scores(x_tt, weight_tt, device))
    assert (got.double() - exact_scores(x, weight)).abs().max() < MAX_ABS
    assert torch.equal(whole_tiles(scores)[:tokens, experts:], torch.zeros(tokens, TILE - experts))
    assert_same_bits(ttnn.to_torch(rows), x)


def split_layout(tokens: int, columns: int, block_rows: int) -> ttnn.MemoryConfig:
    """Block-sharded over columns x 8 cores, as the model's split layer norm leaves its output."""
    assert columns * block_rows * TILE >= tokens
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(columns - 1, 7))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (block_rows * TILE, HIDDEN // 8), ttnn.ShardOrientation.COL_MAJOR),
    )


@pytest.mark.parametrize(
    "tokens, input_memory_config",
    [
        (4096, ttnn.L1_MEMORY_CONFIG),
        (4096, split_layout(4096, columns=11, block_rows=12)),
        (1000, split_layout(1000, columns=11, block_rows=3)),
    ],
    ids=["4096_l1", "4096_split", "1000_split"],
)
def test_input_and_output_placement_change_no_bits(device, tokens, input_memory_config):
    """The reader addresses the input by page id and the writer the outputs, so an L1 or block-sharded input
    and L1 outputs give the DRAM call's bits; the split layout's blocks go to the cores of the columns holding
    them. At 4096 tokens its last column of cores holds 8 of its 12 tile rows."""
    generator = torch.Generator().manual_seed(tokens)
    x, x_tt = make_input(tokens, generator, device)
    _, weight_tt = make_weight(NUM_EXPERTS, generator, device)
    placed = ttnn.to_memory_config(x_tt, input_memory_config)

    scores, rows = router_front_end(x_tt, weight_tt)
    placed_scores, placed_rows = router_front_end(
        placed, weight_tt, scores_memory_config=ttnn.L1_MEMORY_CONFIG, rows_memory_config=ttnn.L1_MEMORY_CONFIG
    )

    assert placed_scores.memory_config() == ttnn.L1_MEMORY_CONFIG
    assert placed_rows.memory_config() == ttnn.L1_MEMORY_CONFIG
    assert_same_bits(whole_tiles(placed_scores)[:tokens], whole_tiles(scores)[:tokens])
    assert_same_bits(ttnn.to_torch(placed_rows), ttnn.to_torch(rows))
    assert_same_bits(ttnn.to_torch(placed_rows), x)


def test_program_cache_reuse(device):
    """A second call with another input and weight at new addresses reuses the program, multicasts the new
    weight and follows the new buffers."""
    tokens = 1000
    generator = torch.Generator().manual_seed(tokens)
    calls = []
    for _ in range(2):
        x, x_tt = make_input(tokens, generator, device)
        _, weight_tt = make_weight(NUM_EXPERTS, generator, device)
        calls.append((x, x_tt, weight_tt, router_front_end(x_tt, weight_tt)))
        if len(calls) == 1:
            entries = device.num_program_cache_entries()

    assert device.num_program_cache_entries() == entries
    (_, _, first_weight, (first_scores, _)), (_, _, second_weight, (second_scores, _)) = calls
    assert second_weight.buffer_address() != first_weight.buffer_address()
    assert second_scores.buffer_address() != first_scores.buffer_address()
    for x, x_tt, weight_tt, (scores, rows) in calls:
        assert_same_bits(ttnn.to_torch(scores), stock_scores(x_tt, weight_tt, device))
        assert_same_bits(ttnn.to_torch(rows), x)


@pytest.mark.parametrize(
    "tokens, input_memory_config",
    [(1000, ttnn.DRAM_MEMORY_CONFIG), (4096, split_layout(4096, columns=11, block_rows=12))],
    ids=["1000", "4096_split"],
)
def test_every_output_page_is_written(device, tokens, input_memory_config):
    """Every page of both outputs, the scores' padded columns and padding rows included, comes from the call,
    with the blocks split over the grid in row-major order or, from the split layout, by column.

    Both outputs are placed on DRAM freshly filled with NaN; the call before them ran other values through the
    same program, so its circular buffers hold other values where this call must write.
    """
    padded_tokens = -(-tokens // TILE) * TILE
    generator = torch.Generator().manual_seed(tokens)
    x, x_tt = make_input(tokens, generator, device)
    _, stale_tt = make_input(tokens, generator, device)
    _, weight_tt = make_weight(NUM_EXPERTS, generator, device)
    want_scores = stock_scores(x_tt, weight_tt, device).reshape(tokens, NUM_EXPERTS)
    placed, stale = (ttnn.to_memory_config(tensor, input_memory_config) for tensor in (x_tt, stale_tt))

    # Builds the program first: a cold call also allocates its kernel binaries in DRAM, which would move the
    # outputs off the poison.
    for call_input in (placed, stale):
        for output in router_front_end(call_input, weight_tt):
            ttnn.deallocate(output)
    # The scores' poison spans their whole tiles, so their padding starts as NaN too.
    poison = [
        ttnn.from_torch(
            torch.full((1, 1, padded_tokens, TILE), float("nan")),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        ),
        ttnn.from_torch(
            torch.full((1, 1, tokens, HIDDEN), float("nan")),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        ),
    ]
    poisoned_addresses = [tensor.buffer_address() for tensor in poison]
    ttnn.synchronize_device(device)
    for tensor in poison:
        ttnn.deallocate(tensor)

    scores, rows = router_front_end(placed, weight_tt)

    assert [scores.buffer_address(), rows.buffer_address()] == poisoned_addresses
    whole = whole_tiles(scores)
    assert not whole.isnan().any()
    assert_same_bits(whole[:tokens, :NUM_EXPERTS], want_scores)
    assert torch.equal(whole[:tokens, NUM_EXPERTS:], torch.zeros(tokens, TILE - NUM_EXPERTS))
    assert_same_bits(ttnn.to_torch(rows), x)


TOKENS = 64


def _tensor(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=None):
    """A factory the test calls with its device: the table below is built before any device exists."""
    return lambda device: ttnn.from_torch(
        torch.zeros(shape), dtype=dtype, layout=layout, device=device, memory_config=memory_config
    )


def _one_core_sharded(shape):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shape, ttnn.ShardOrientation.ROW_MAJOR),
    )


VALID = dict(
    input=_tensor((1, 1, TOKENS, HIDDEN)),
    weight=_tensor((HIDDEN, NUM_EXPERTS), dtype=ttnn.float32),
)

# case: (tensors replacing the valid ones, keyword arguments, error message)
INVALID_CALLS = {
    "input_dtype": (
        dict(input=_tensor((1, 1, TOKENS, HIDDEN), dtype=ttnn.float32)),
        {},
        "input must be BFLOAT16",
    ),
    "input_layout": (
        dict(input=_tensor((1, 1, TOKENS, HIDDEN), layout=ttnn.ROW_MAJOR_LAYOUT)),
        {},
        "input must be TILE layout",
    ),
    "input_batch": (
        dict(input=_tensor((2, 1, TOKENS, HIDDEN))),
        {},
        "input must be 1 x 1 x tokens x hidden",
    ),
    "input_rank": (
        dict(input=_tensor((1, TOKENS, HIDDEN))),
        {},
        "input must be 1 x 1 x tokens x hidden",
    ),
    "hidden_off_the_tile_grid": (
        dict(input=_tensor((1, 1, TOKENS, HIDDEN - 4))),
        {},
        "hidden must be whole tiles, got 764",
    ),
    "weight_dtype": (
        dict(weight=_tensor((HIDDEN, NUM_EXPERTS))),
        {},
        "weight must be FLOAT32",
    ),
    "weight_layout": (
        dict(weight=_tensor((HIDDEN, NUM_EXPERTS), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)),
        {},
        "weight must be TILE layout",
    ),
    "weight_sharded": (
        dict(
            weight=_tensor((HIDDEN, NUM_EXPERTS), dtype=ttnn.float32, memory_config=_one_core_sharded((HIDDEN, TILE)))
        ),
        {},
        "weight must be interleaved",
    ),
    "weight_hidden": (
        dict(weight=_tensor((HIDDEN // 2, NUM_EXPERTS), dtype=ttnn.float32)),
        {},
        "weight must be 768 x experts",
    ),
    "weight_leading_dimensions": (
        dict(weight=_tensor((2, HIDDEN, NUM_EXPERTS), dtype=ttnn.float32)),
        {},
        "weight's leading dimensions must be 1",
    ),
    "too_many_experts": (
        dict(weight=_tensor((HIDDEN, TILE + 1), dtype=ttnn.float32)),
        {},
        "experts must be 1 to 32, got 33",
    ),
    "scores_sharded": (
        {},
        dict(scores_memory_config=_one_core_sharded((TOKENS, TILE))),
        "scores_memory_config must be interleaved",
    ),
    "rows_sharded": (
        {},
        dict(rows_memory_config=_one_core_sharded((TOKENS, HIDDEN))),
        "rows_memory_config must be interleaved",
    ),
    "bfloat16_accumulation": (
        {},
        dict(compute_kernel_config=ttnn.WormholeComputeKernelConfig(fp32_dest_acc_en=False)),
        "compute_kernel_config must enable fp32_dest_acc_en",
    ),
}


@pytest.mark.parametrize("case", INVALID_CALLS.keys())
def test_invalid_inputs_raise(device, expect_error, case):
    tensor_overrides, kwargs, message = INVALID_CALLS[case]
    tensors = VALID | tensor_overrides

    with expect_error(RuntimeError, f"router_front_end: {message}"):
        router_front_end(tensors["input"](device), tensors["weight"](device), **kwargs)
