# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.nomic_embed_text_v2_moe.routed_combine against its definition.

Each output row is the bias plus each slot's row times its weight, accumulated in fp32 and rounded
to bf16 once. With one-hot weights the sum is a row plus the bias, exact in fp32, so the output must
equal it rounded to bf16 bit for bit. The packer breaks a tie away from zero, where torch rounds it
to even, so the reference rounds the same way.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device]

NUM_EXPERTS = 8
TOP_K = 2
HIDDEN = 768

routed_combine = ttnn.experimental.nomic_embed_text_v2_moe.routed_combine
routing_metadata = ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata


def bfloat16_ties_away(values: torch.Tensor) -> torch.Tensor:
    """float32 to bfloat16 rounding to nearest, ties away from zero, as the packer rounds."""
    bits = values.float().view(torch.int32)
    return ((bits + 0x8000) & ~0xFFFF).view(torch.float32).bfloat16()


def selection(tokens: int, values: torch.Tensor, generator: torch.Generator, device):
    """The window's routing metadata for a random top-2 selection with the given weights."""
    indices = torch.stack([torch.randperm(NUM_EXPERTS, generator=generator)[:TOP_K] for _ in range(tokens)])

    def tile(tensor, dtype):
        return ttnn.from_torch(tensor.reshape(1, 1, tokens, TOP_K), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    gather_index, combine_index, combine_weights, expert_counts = routing_metadata(
        tile(indices.to(torch.int32), ttnn.uint32), tile(values.float(), ttnn.float32), num_experts=NUM_EXPERTS
    )
    return gather_index.shape[-1], combine_index, combine_weights


def device_rows(rows: torch.Tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        rows.reshape(1, 1, -1, HIDDEN),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def make_bias(generator: torch.Generator, device):
    bias = torch.randn(1, 1, 1, HIDDEN, generator=generator).bfloat16()
    return bias, ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def padded_output(output, tokens: int):
    padded = (tokens + 31) // 32 * 32
    viewed = ttnn.reshape(output, (1, 1, padded, HIDDEN), (1, 1, padded, HIDDEN))
    return ttnn.to_torch(viewed).reshape(padded, HIDDEN)


@pytest.mark.parametrize(
    "tokens, memory_config",
    [(330, ttnn.DRAM_MEMORY_CONFIG), (1000, ttnn.DRAM_MEMORY_CONFIG), (4096, ttnn.L1_MEMORY_CONFIG)],
    ids=["330", "1000", "4096_l1"],
)
@pytest.mark.parametrize("slot", range(TOP_K))
def test_one_hot_weights_return_the_slot_row_plus_bias(device, tokens, memory_config, slot):
    """At weight 1 on one slot and 0 on the other, each token's output is that slot's row plus the
    bias, bit for bit. Every row that holds no slot is NaN, so a token reading one would show it.
    The padding tokens' weights are 0 on a real row, so the tile padding of the output is the bias.

    330 tokens split into three-tile chunks, 1000 and 4096 into eight-tile chunks, 4096 with four
    units on each core.
    """
    generator = torch.Generator().manual_seed(tokens + slot)
    values = torch.zeros(tokens, TOP_K)
    values[:, slot] = 1.0
    rows, combine_index, combine_weights = selection(tokens, values, generator, device)
    index = ttnn.to_torch(combine_index).long()
    expert_rows = torch.randn(rows, HIDDEN, generator=generator).bfloat16()
    used = torch.zeros(rows, dtype=torch.bool)
    used[index[:, :tokens].reshape(-1)] = True
    expert_rows[~used] = float("nan")
    bias, bias_device = make_bias(generator, device)

    output = routed_combine(
        device_rows(expert_rows, device, memory_config),
        combine_index,
        combine_weights,
        bias_device,
        num_tokens=tokens,
        memory_config=memory_config,
    )

    assert list(output.shape) == [1, 1, tokens, HIDDEN]
    got = padded_output(output, tokens)
    want = bfloat16_ties_away(expert_rows[index[slot, :tokens]].float() + bias.float().reshape(1, HIDDEN))
    assert torch.equal(got[:tokens].view(torch.int16), want.view(torch.int16))
    assert torch.equal(got[tokens:], bias.reshape(1, HIDDEN).expand(got.shape[0] - tokens, HIDDEN))


@pytest.mark.parametrize("tokens", [330, 4096])
def test_router_weights_are_no_less_precise_than_the_stock_combine(device, tokens):
    """With softmax-like weights, the error against a float64 sum of the same rows and weights is no
    worse than the stock path's: the embedding gather, attn_res_weighted_reduce_nc and a bias add."""
    generator = torch.Generator().manual_seed(tokens)
    values = torch.rand(tokens, TOP_K, generator=generator).sort(-1, descending=True).values * 0.5
    rows, combine_index, combine_weights = selection(tokens, values, generator, device)
    padded = combine_index.shape[-1]
    expert_rows = torch.randn(rows, HIDDEN, generator=generator).bfloat16()
    bias, bias_device = make_bias(generator, device)
    rows_device = device_rows(expert_rows, device)
    index = ttnn.to_torch(combine_index).long()[:, :tokens]
    weights = ttnn.to_torch(combine_weights).double().reshape(TOP_K, padded)[:, :tokens]
    want = (weights.unsqueeze(-1) * expert_rows.double()[index]).sum(0) + bias.double().reshape(1, HIDDEN)

    output = routed_combine(rows_device, combine_index, combine_weights, bias_device, num_tokens=tokens)
    gathered = ttnn.embedding(combine_index, rows_device, layout=ttnn.TILE_LAYOUT)
    summed = ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(
        ttnn.reshape(gathered, (1, TOP_K, padded, HIDDEN)), combine_weights, dim=1
    )
    stock = ttnn.add(summed, bias_device)

    def token_error(result):
        values = ttnn.to_torch(result).double().reshape(-1, HIDDEN)[:tokens]
        return (values - want).norm(dim=-1) / want.norm(dim=-1)

    combined_error, stock_error = token_error(output), token_error(stock)
    assert combined_error.max() <= stock_error.max()
    assert combined_error.mean() <= stock_error.mean()


def test_program_cache_reuse(device):
    """A second call with new routing at new addresses reuses the program and follows the new buffers."""
    tokens = 1000
    generator = torch.Generator().manual_seed(tokens)
    results = []
    for _ in range(2):
        values = torch.zeros(tokens, TOP_K)
        values[:, 0] = 1.0
        rows, combine_index, combine_weights = selection(tokens, values, generator, device)
        expert_rows = torch.randn(rows, HIDDEN, generator=generator).bfloat16()
        bias, bias_device = make_bias(generator, device)
        rows_device = device_rows(expert_rows, device)
        output = routed_combine(rows_device, combine_index, combine_weights, bias_device, num_tokens=tokens)
        results.append((output, rows_device, expert_rows, combine_index, bias))
        if len(results) == 1:
            entries = device.num_program_cache_entries()

    assert device.num_program_cache_entries() == entries
    assert results[1][0].buffer_address() != results[0][0].buffer_address()
    assert results[1][1].buffer_address() != results[0][1].buffer_address()
    for output, _, expert_rows, combine_index, bias in results:
        index = ttnn.to_torch(combine_index).long()
        want = bfloat16_ties_away(expert_rows[index[0, :tokens]].float() + bias.float().reshape(1, HIDDEN))
        assert torch.equal(padded_output(output, tokens)[:tokens].view(torch.int16), want.view(torch.int16))


def test_every_output_tile_is_written(device):
    """Every output tile comes from the call itself, not from what an earlier call left.

    The output is placed on DRAM freshly filled with NaN; the call before it combined other rows
    through the same program, so its circular buffers hold other values where this call must write.
    """
    tokens = 1000
    generator = torch.Generator().manual_seed(tokens)
    values = torch.zeros(tokens, TOP_K)
    values[:, 1] = 1.0
    rows, combine_index, combine_weights = selection(tokens, values, generator, device)
    expert_rows = torch.randn(rows, HIDDEN, generator=generator).bfloat16()
    stale_rows = device_rows(torch.randn(rows, HIDDEN, generator=generator).bfloat16(), device)
    bias, bias_device = make_bias(generator, device)
    rows_device = device_rows(expert_rows, device)

    # Builds the program first: a cold call also allocates its kernel binaries in DRAM, which would
    # move the output off the poison.
    ttnn.deallocate(routed_combine(rows_device, combine_index, combine_weights, bias_device, num_tokens=tokens))
    ttnn.deallocate(routed_combine(stale_rows, combine_index, combine_weights, bias_device, num_tokens=tokens))
    poison = ttnn.from_torch(
        torch.full((1, 1, tokens, HIDDEN), float("nan")), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    poisoned_address = poison.buffer_address()
    ttnn.synchronize_device(device)
    ttnn.deallocate(poison)

    output = routed_combine(rows_device, combine_index, combine_weights, bias_device, num_tokens=tokens)

    assert output.buffer_address() == poisoned_address
    index = ttnn.to_torch(combine_index).long()
    want = bfloat16_ties_away(expert_rows[index[1, :tokens]].float() + bias.float().reshape(1, HIDDEN))
    got = padded_output(output, tokens)
    assert torch.equal(got[:tokens].view(torch.int16), want.view(torch.int16))
    assert torch.equal(got[tokens:], bias.reshape(1, HIDDEN).expand(got.shape[0] - tokens, HIDDEN))


ROWS = 32 * NUM_EXPERTS
TOKENS = 64


def _tensor(shape, dtype, layout, torch_dtype=torch.float32):
    """A factory the test calls with its device: the table below is built before any device exists."""
    return lambda device: ttnn.from_torch(
        torch.zeros(shape, dtype=torch_dtype), dtype=dtype, layout=layout, device=device
    )


VALID = dict(
    expert_rows=_tensor((1, 1, ROWS, HIDDEN), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    combine_index=_tensor((TOP_K, TOKENS), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32),
    combine_weights=_tensor((1, TOP_K, TOKENS, 1), ttnn.float32, ttnn.TILE_LAYOUT),
    bias=_tensor((1, 1, 1, HIDDEN), ttnn.bfloat16, ttnn.TILE_LAYOUT),
)

# case: (tensors replacing the valid ones, keyword arguments replacing the valid ones, error message)
INVALID_CALLS = {
    "rows_layout": (
        dict(expert_rows=_tensor((1, 1, ROWS, HIDDEN), ttnn.bfloat16, ttnn.TILE_LAYOUT)),
        {},
        "expert_rows must be Layout::ROW_MAJOR layout",
    ),
    "rows_dtype": (
        dict(expert_rows=_tensor((1, 1, ROWS, HIDDEN), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)),
        {},
        "expert_rows must be DataType::BFLOAT16",
    ),
    "rows_rank": (
        dict(expert_rows=_tensor((ROWS, HIDDEN), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)),
        {},
        "expert_rows must be 1 x 1 x rows x hidden",
    ),
    "rows_too_few": (
        dict(expert_rows=_tensor((1, 1, TOP_K * TOKENS - 32, HIDDEN), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)),
        {},
        "expert_rows has 96 rows, fewer than the 128 that 64 tokens at top_k 2 fill",
    ),
    "index_dtype": (
        dict(combine_index=_tensor((TOP_K, TOKENS), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)),
        {},
        "combine_index must be DataType::UINT32",
    ),
    "index_rank": (
        dict(combine_index=_tensor((1, TOP_K, TOKENS), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)),
        {},
        "combine_index must be top_k x padded_tokens",
    ),
    "top_k": (
        dict(
            combine_index=_tensor((17, TOKENS), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32),
            combine_weights=_tensor((1, 17, TOKENS, 1), ttnn.float32, ttnn.TILE_LAYOUT),
        ),
        {},
        "top_k must be 1 to 16",
    ),
    "num_tokens": ({}, dict(num_tokens=TOKENS - 32), "num_tokens 32 does not match combine_index's 64 padded tokens"),
    "weights_shape": (
        dict(combine_weights=_tensor((1, TOP_K, TOKENS + 32, 1), ttnn.float32, ttnn.TILE_LAYOUT)),
        {},
        "combine_weights must be 1 x 2 x 64 x 1",
    ),
    "weights_dtype": (
        dict(combine_weights=_tensor((1, TOP_K, TOKENS, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT)),
        {},
        "combine_weights must be DataType::FLOAT32",
    ),
    "bias_shape": (
        dict(bias=_tensor((1, 1, 1, HIDDEN // 2), ttnn.bfloat16, ttnn.TILE_LAYOUT)),
        {},
        "bias must be 1 x 1 x 1 x 768",
    ),
    "memory_config": (
        {},
        dict(
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))}), (32, HIDDEN), ttnn.ShardOrientation.ROW_MAJOR
                ),
            )
        ),
        "memory_config must be interleaved",
    ),
    "bfloat16_accumulation": (
        {},
        dict(compute_kernel_config=ttnn.WormholeComputeKernelConfig(fp32_dest_acc_en=False)),
        "compute_kernel_config must enable fp32_dest_acc_en",
    ),
}


@pytest.mark.parametrize("case", INVALID_CALLS.keys())
def test_invalid_inputs_raise(device, expect_error, case):
    tensor_overrides, call_overrides, message = INVALID_CALLS[case]
    tensors = VALID | tensor_overrides
    kwargs = dict(num_tokens=TOKENS) | call_overrides

    with expect_error(RuntimeError, f"routed_combine: {message}"):
        routed_combine(
            tensors["expert_rows"](device),
            tensors["combine_index"](device),
            tensors["combine_weights"](device),
            tensors["bias"](device),
            **kwargs,
        )
