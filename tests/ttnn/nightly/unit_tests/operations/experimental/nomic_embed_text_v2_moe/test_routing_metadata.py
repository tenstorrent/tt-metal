# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata against its torch oracle.

The op is integer bookkeeping plus a 32-bit copy of the weights, so every output must equal the
oracle exactly.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.experimental.nomic_embed_text_v2_moe.routing_metadata_reference import (
    BLOCK_ROWS,
    assert_matches_reference,
)

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device]

NUM_EXPERTS = 8
TOP_K = 2
# A window holds at most MAX_SLOTS slots, top_k x num_tokens.
MAX_SLOTS = 65536
MAX_TOKENS = MAX_SLOTS // TOP_K
# Up to this many tokens the model's router leaves its selection in L1, and in DRAM above.
ROUTER_L1_MAX_TOKENS = 4096

# Stand-ins for what topk leaves in the tile padding. If a kernel read a padding row, the id would
# be clamped into the last expert and change its count; the weight would turn a combine weight
# non-finite.
POISON_ID = 1000
POISON_WEIGHT = float("nan")

routing_metadata = ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata


def selection(pattern: str, tokens: int, generator: torch.Generator) -> torch.Tensor:
    """(tokens, TOP_K) expert ids, distinct within each token as topk returns them."""
    if pattern == "uniform":
        weights = torch.ones(tokens, NUM_EXPERTS)
    elif pattern == "skewed":
        # The last expert is in four tokens of five, 3.2x the mean load (random token ids reach 3.45x
        # in the model), so its region spans many blocks.
        weights = (2.0 ** torch.arange(NUM_EXPERTS)).expand(tokens, NUM_EXPERTS)
    elif pattern == "same_pair":
        return torch.tensor([3, 5]).repeat(tokens, 1)
    elif pattern == "first_and_last":
        # Only the two outermost regions are used; every expert between them is empty.
        pairs = torch.tensor([[0, NUM_EXPERTS - 1], [NUM_EXPERTS - 1, 0]])
        return pairs[torch.randint(0, 2, (tokens,), generator=generator)]
    else:
        raise ValueError(pattern)
    return torch.multinomial(weights, TOP_K, replacement=False, generator=generator)


def selection_with_counts(counts: list[int], generator: torch.Generator) -> torch.Tensor:
    """(sum(counts) / TOP_K, TOP_K) ids routing exactly counts[e] slots to expert e.

    Sorted, the slots pair up as (s, s + tokens), distinct whenever no count exceeds tokens.
    """
    slots = torch.repeat_interleave(torch.arange(NUM_EXPERTS), torch.tensor(counts))
    tokens = slots.numel() // TOP_K
    assert max(counts) <= tokens
    indices = torch.stack([slots[:tokens], slots[tokens:]], dim=1)
    indices = indices[torch.randperm(tokens, generator=generator)]
    swap = torch.rand(tokens, generator=generator) < 0.5
    indices[swap] = indices[swap].flip(1)
    return indices


def device_selection(device, indices, values, memory_config=ttnn.L1_MEMORY_CONFIG):
    """(tokens, TOP_K) host tensors -> the router's (1, 1, tokens, TOP_K) selection, padding poisoned."""
    shape = (1, 1, *indices.shape)
    tt_indices = ttnn.from_torch(
        indices.reshape(shape),
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
        pad_value=POISON_ID,
    )
    tt_values = ttnn.from_torch(
        values.reshape(shape),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
        pad_value=POISON_WEIGHT,
    )
    return tt_indices, tt_values


@pytest.mark.parametrize("pattern", ["uniform", "skewed", "same_pair", "first_and_last"])
@pytest.mark.parametrize(
    "num_tokens", [1, 37, 128, 330, 1024, 2048, 2240, 4095, ROUTER_L1_MAX_TOKENS, ROUTER_L1_MAX_TOKENS + 1, MAX_TOKENS]
)
def test_matches_reference(device, num_tokens, pattern):
    """Inputs where the router leaves them: L1 up to ROUTER_L1_MAX_TOKENS tokens, DRAM above.

    The window's tile rows are split over a rectangle of cores. On Blackhole these sizes take 1, 2,
    4, 6 (one with a row less than the others), 32 (one row each), 32 (two rows each), 70 (one row
    each, the whole 7 x 10 block), 70, 70 (58 with two rows, 12 with one), 70 and 70 cores; the last
    is the longest window at top-2, MAX_SLOTS slots.
    """
    generator = torch.Generator().manual_seed(num_tokens)
    indices = selection(pattern, num_tokens, generator)
    values = torch.rand(num_tokens, TOP_K, generator=generator)

    memory_config = ttnn.L1_MEMORY_CONFIG if num_tokens <= ROUTER_L1_MAX_TOKENS else ttnn.DRAM_MEMORY_CONFIG

    outputs = routing_metadata(*device_selection(device, indices, values, memory_config), num_experts=NUM_EXPERTS)

    assert_matches_reference(outputs, indices, values, NUM_EXPERTS)


@pytest.mark.parametrize("window", ["short", "longest"])
@pytest.mark.parametrize("num_experts, top_k", [(1, 1), (3, 1), (5, 3), (16, 4), (64, 16)])
def test_other_expert_and_choice_counts(device, num_experts, top_k, window):
    """The validated range beyond the model's 8 experts and top-2.

    top_k 1, 3, 4 and 16 group the slots differently while counting and assigning; odd counts of
    experts leave the high half of the last packed count word empty; and the longest window of
    each top_k is MAX_SLOTS slots, the most the 16-bit packed count sums are bounded for. One
    expert at top-1 takes all 65536, the count that would carry into the next half.
    """
    num_tokens = 37 if window == "short" else MAX_SLOTS // top_k
    generator = torch.Generator().manual_seed(num_experts * 100 + top_k)
    indices = torch.multinomial(torch.ones(num_tokens, num_experts), top_k, replacement=False, generator=generator)
    values = torch.rand(num_tokens, top_k, generator=generator)

    outputs = routing_metadata(*device_selection(device, indices, values), num_experts=num_experts)

    assert_matches_reference(outputs, indices, values, num_experts)


@pytest.mark.parametrize(
    "counts",
    [
        [32] * NUM_EXPERTS,
        [1024] * NUM_EXPERTS,
        [1] * NUM_EXPERTS,
        [1025] * (NUM_EXPERTS - 1) + [993],
    ],
    ids=["128_tokens_multiples_of_32", "4096_tokens_multiples_of_32", "4_tokens_1_mod_32", "4084_tokens_1_mod_32"],
)
def test_region_alignment(device, counts):
    """Regions that fill their blocks exactly, and regions that waste 31 rows each.

    With every count 1 mod 32 the regions take every gathered row, the worst case num_blocks is
    sized for: one block more and the last expert would be written past the end.
    """
    generator = torch.Generator().manual_seed(len(counts) + sum(counts))
    indices = selection_with_counts(counts, generator)
    values = torch.rand(indices.shape[0], TOP_K, generator=generator)

    outputs = routing_metadata(*device_selection(device, indices, values), num_experts=NUM_EXPERTS)

    got = assert_matches_reference(outputs, indices, values, NUM_EXPERTS)
    used_rows = sum((count + BLOCK_ROWS - 1) // BLOCK_ROWS * BLOCK_ROWS for count in counts)
    if all(count % BLOCK_ROWS == 1 for count in counts):
        assert used_rows == got.gather_index.numel()
    else:
        assert used_rows == TOP_K * indices.shape[0]


@pytest.mark.parametrize("token_start, num_tokens", [(0, 2496), (2496, None), (32, 37)])
def test_token_windows(device, token_start, num_tokens):
    """Windows of a 5000-token selection in DRAM, where the router puts one above 4096 tokens.

    The window's ids are global, so gather_index can index the whole batch's activations.
    """
    tokens = 5000
    generator = torch.Generator().manual_seed(tokens)
    indices = selection("uniform", tokens, generator)
    values = torch.rand(tokens, TOP_K, generator=generator)
    tt_indices, tt_values = device_selection(device, indices, values, ttnn.DRAM_MEMORY_CONFIG)

    outputs = routing_metadata(
        tt_indices, tt_values, num_experts=NUM_EXPERTS, token_start=token_start, num_tokens=num_tokens
    )

    got = assert_matches_reference(outputs, indices, values, NUM_EXPERTS, token_start, num_tokens)
    end = tokens if num_tokens is None else token_start + num_tokens
    assert int(got.gather_index.min()) == token_start and int(got.gather_index.max()) == end - 1


def test_runs_shorter_than_a_block(device):
    """Each core sends the core assembling an expert's region its rows of that expert, written into
    L1 at their final place. With the last expert in one token of every 21, each core's run of it
    is 1 to 4 rows, so up to four cores write into the same 16-byte block at once. L1 applies a
    partial write as an atomic read-modify-write; were one lost, a row of the region would be wrong.
    Repeated, since a lost write would depend on timing.
    """
    tokens = 4096
    rare = torch.arange(tokens) % 21 == 0
    for seed in range(10):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.multinomial(torch.ones(tokens, NUM_EXPERTS - 1), TOP_K, generator=generator)
        indices[rare, torch.randint(0, TOP_K, (int(rare.sum()),), generator=generator)] = NUM_EXPERTS - 1
        values = torch.rand(tokens, TOP_K, generator=generator)

        outputs = routing_metadata(*device_selection(device, indices, values), num_experts=NUM_EXPERTS)

        got = assert_matches_reference(outputs, indices, values, NUM_EXPERTS)
        assert int(got.expert_counts[-1]) == int(rare.sum())


def test_out_of_range_ids_are_clamped(device):
    """An id at or above num_experts counts as the last expert instead of writing out of bounds."""
    tokens = 330
    generator = torch.Generator().manual_seed(tokens)
    indices = selection("uniform", tokens, generator)
    values = torch.rand(tokens, TOP_K, generator=generator)
    out_of_range = torch.rand(tokens, TOP_K, generator=generator) < 0.05
    bad_ids = torch.tensor([NUM_EXPERTS, NUM_EXPERTS + 1, 63, 64, 1000, 2**32 - 1])
    indices[out_of_range] = bad_ids[torch.randint(0, len(bad_ids), (int(out_of_range.sum()),), generator=generator)]

    outputs = routing_metadata(*device_selection(device, indices, values), num_experts=NUM_EXPERTS)

    got = assert_matches_reference(outputs, indices, values, NUM_EXPERTS)
    in_last_expert = int((indices == NUM_EXPERTS - 1).sum() + out_of_range.sum())
    assert int(got.expert_counts[-1]) == in_last_expert


def test_program_cache_reuse(device):
    """A second call with new routing at new addresses reuses the program and follows the new buffers."""
    tokens = 330
    generator = torch.Generator().manual_seed(tokens)
    first_indices = selection("uniform", tokens, generator)
    first_values = torch.rand(tokens, TOP_K, generator=generator)
    first_inputs = device_selection(device, first_indices, first_values)
    first_outputs = routing_metadata(*first_inputs, num_experts=NUM_EXPERTS)
    entries = device.num_program_cache_entries()

    second_indices = selection("skewed", tokens, generator)
    second_values = torch.rand(tokens, TOP_K, generator=generator)
    second_inputs = device_selection(device, second_indices, second_values)
    second_outputs = routing_metadata(*second_inputs, num_experts=NUM_EXPERTS)

    assert device.num_program_cache_entries() == entries
    assert second_inputs[0].buffer_address() != first_inputs[0].buffer_address()
    assert second_outputs[0].buffer_address() != first_outputs[0].buffer_address()
    assert_matches_reference(first_outputs, first_indices, first_values, NUM_EXPERTS)
    assert_matches_reference(second_outputs, second_indices, second_values, NUM_EXPERTS)


@pytest.mark.parametrize(
    "tokens, checked, stale",
    [
        (100, dict(token_start=32, num_tokens=37), dict(token_start=0, num_tokens=48)),
        (4096, dict(token_start=0, num_tokens=3990), dict(token_start=96, num_tokens=4000)),
    ],
    ids=["2_cores", "63_cores"],
)
def test_every_output_entry_is_written(device, tokens, checked, stale):
    """Every output entry comes from the call itself, not from what an earlier call left behind.

    Two kinds of memory outlive a call. Freed DRAM keeps its contents and the allocator hands the
    same addresses back first, so the outputs are placed on DRAM filled with poison. The kernels'
    L1 scratch sits at the same addresses in every program of the same sizes, so the call before is
    a window sized like the one checked (48 tokens for 37, 4000 for 3990, on the same cores): it
    leaves real rows and weights where the checked window's padding must be written as 0, and
    other rows where its regions and gaps lie.
    """
    generator = torch.Generator().manual_seed(tokens)
    indices = selection("uniform", tokens, generator)
    values = torch.rand(tokens, TOP_K, generator=generator)
    inputs = device_selection(device, indices, values, ttnn.DRAM_MEMORY_CONFIG)

    # Builds both programs first: a cold call also allocates its kernel binaries in DRAM, which
    # would move the outputs off the poison.
    specs = []
    for window in (checked, stale):
        outputs = routing_metadata(*inputs, num_experts=NUM_EXPERTS, **window)
        specs.append([(tuple(output.shape), output.dtype, output.layout) for output in outputs])
        for output in outputs:
            ttnn.deallocate(output)
    assert specs[0] == specs[1]

    poison = []
    for shape, dtype, layout in specs[0]:
        fill = torch.full(shape, float("nan")) if dtype == ttnn.float32 else torch.full(shape, 2**31 - 1)
        poison.append(ttnn.from_torch(fill, dtype=dtype, layout=layout, device=device))
    poisoned_addresses = {tensor.buffer_address() for tensor in poison}
    ttnn.synchronize_device(device)
    for tensor in poison:
        ttnn.deallocate(tensor)

    outputs = routing_metadata(*inputs, num_experts=NUM_EXPERTS, **checked)

    assert {output.buffer_address() for output in outputs} == poisoned_addresses
    assert_matches_reference(outputs, indices, values, NUM_EXPERTS, **checked)


SELECTION_SHAPE = (1, 1, 64, TOP_K)


def _tensor(shape, dtype, layout=ttnn.TILE_LAYOUT, **kwargs):
    """A factory the test calls with its device: the table below is built before any device exists."""
    return lambda device: ttnn.from_torch(torch.zeros(shape), dtype=dtype, layout=layout, device=device, **kwargs)


def _height_sharded(device):
    memory_config = ttnn.create_sharded_memory_config(
        (32, 32), ttnn.CoreGrid(y=1, x=2), ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
    )
    return _tensor(SELECTION_SHAPE, ttnn.float32, memory_config=memory_config)(device)


# case: (tensors replacing the valid ones, keyword arguments replacing the valid ones, error message)
INVALID_CALLS = {
    "indices_dtype": (dict(indices=_tensor(SELECTION_SHAPE, ttnn.int32)), {}, "indices must be DataType::UINT32"),
    "values_dtype": (dict(values=_tensor(SELECTION_SHAPE, ttnn.bfloat16)), {}, "values must be DataType::FLOAT32"),
    "row_major": (
        dict(indices=_tensor(SELECTION_SHAPE, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)),
        {},
        "indices must be TILE layout",
    ),
    "sharded": (dict(values=_height_sharded), {}, "values must be interleaved"),
    "tile_shape": (
        dict(indices=_tensor(SELECTION_SHAPE, ttnn.uint32, tile=ttnn.Tile((16, 32)))),
        {},
        "indices must use 32 x 32 tiles",
    ),
    "rank": (dict(indices=_tensor((2, 1, 32, TOP_K), ttnn.uint32)), {}, "indices must be 1 x 1 x tokens x top_k"),
    "shape_mismatch": (dict(values=_tensor((1, 1, 64, 3), ttnn.float32)), {}, "values must have the shape of indices"),
    "top_k": (
        dict(indices=_tensor((1, 1, 64, 17), ttnn.uint32), values=_tensor((1, 1, 64, 17), ttnn.float32)),
        {},
        "top_k must be 1 to 16",
    ),
    "num_experts": ({}, dict(num_experts=65), "num_experts must be 1 to 64"),
    "token_start_unaligned": ({}, dict(token_start=16), "token_start must be a multiple of 32"),
    "token_start_past_end": ({}, dict(token_start=64), "token_start must be below the token count"),
    "num_tokens_over_cap": (
        dict(
            indices=_tensor((1, 1, MAX_TOKENS + 32, TOP_K), ttnn.uint32),
            values=_tensor((1, 1, MAX_TOKENS + 32, TOP_K), ttnn.float32),
        ),
        dict(num_tokens=MAX_TOKENS + 1),
        "num_tokens must be 1 to 32768 at top_k 2, 65536 slots a window",
    ),
    "window_past_end": (
        {},
        dict(token_start=32, num_tokens=64),
        "the window of 64 tokens from token 32 ends past the token count",
    ),
}


@pytest.mark.parametrize("case", INVALID_CALLS.keys())
def test_invalid_inputs_raise(device, expect_error, case):
    tensor_overrides, call_overrides, message = INVALID_CALLS[case]
    tensors = dict(indices=_tensor(SELECTION_SHAPE, ttnn.uint32), values=_tensor(SELECTION_SHAPE, ttnn.float32))
    tensors |= tensor_overrides
    kwargs = dict(num_experts=NUM_EXPERTS) | call_overrides

    with expect_error(RuntimeError, f"routing_metadata: {message}"):
        routing_metadata(tensors["indices"](device), tensors["values"](device), **kwargs)
