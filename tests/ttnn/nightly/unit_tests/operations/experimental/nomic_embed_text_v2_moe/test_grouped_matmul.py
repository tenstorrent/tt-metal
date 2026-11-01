# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul against ttnn.matmul, expert by expert.

The op runs each block of rows through the same multiply-accumulate sequence as a stock 2D matmul of
that expert's rows with its weights and the same blocking (K in one block, 1x3 subblocks, fp32
accumulation), and the same packer-side GELU. So every row of every expert's region must equal the
stock result bit for bit; a wrong expert, block, column or K order shows as a mismatch.
"""

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.experimental.nomic_embed_text_v2_moe.routing_metadata_reference import (
    BLOCK_ROWS,
    region_starts,
)

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device]

NUM_EXPERTS = 8
HIDDEN = 768
INTERMEDIATE = 3072

grouped_matmul = ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul

# The model's two calls: w1 with its GELU, accurate (param 0) and fast (param 1), into bf8_b; w2 read
# transposed into bf16, in tiles and in rows for the combine.
PROJECTIONS = {
    "w1_accurate_gelu": dict(transposed=False, gelu=0.0, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT),
    "w1_fast_gelu": dict(transposed=False, gelu=1.0, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT),
    "w2_tile": dict(transposed=True, gelu=None, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
    "w2_row_major": dict(transposed=True, gelu=None, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
}

ROUTINGS = ["uniform", "one_expert", "first_and_last", "empty_between", "one_mod_32"]


def expert_counts(routing: str, tokens: int, generator: torch.Generator) -> torch.Tensor:
    """Slots per expert for a top-2 pass of `tokens` tokens."""
    slots = 2 * tokens
    counts = torch.zeros(NUM_EXPERTS, dtype=torch.int64)
    if routing == "uniform":
        picks = torch.stack([torch.randperm(NUM_EXPERTS, generator=generator)[:2] for _ in range(tokens)])
        counts = torch.bincount(picks.reshape(-1), minlength=NUM_EXPERTS)
    elif routing == "one_expert":
        # Every block in one region: lanes walk seven empty experts and one full one.
        counts[3] = slots
    elif routing == "first_and_last":
        counts[0], counts[-1] = slots // 2, slots - slots // 2
    elif routing == "empty_between":
        live = [expert for expert in range(NUM_EXPERTS) if expert not in (2, 5)]
        for index, expert in enumerate(live):
            counts[expert] = slots // len(live) + (1 if index < slots % len(live) else 0)
    elif routing == "one_mod_32":
        # Every region ends in a block holding a single row.
        counts[:] = (slots // NUM_EXPERTS - 1) // BLOCK_ROWS * BLOCK_ROWS + 1
    else:
        raise ValueError(routing)
    return counts


def gathered_rows(tokens: int) -> int:
    """routing_metadata's gathered row count for a top-2 pass, which holds any routing."""
    return BLOCK_ROWS * (math.ceil(2 * tokens / BLOCK_ROWS) + NUM_EXPERTS - 1)


def make_weight(device, generator: torch.Generator):
    """All experts' weights as the model stores them: (1, 1, 768, 8 * 3072), expert e in columns
    e * 3072 onwards. w1 reads it as (K, E * N); w2 as (N, E * K), transposed."""
    values = torch.randn(1, 1, HIDDEN, NUM_EXPERTS * INTERMEDIATE, generator=generator) * 0.03
    return ttnn.from_torch(values, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)


def make_input(device, transposed: bool, rows: int, generator: torch.Generator, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    if transposed:
        values = torch.randn(1, 1, rows, INTERMEDIATE, generator=generator) * 0.5
        return ttnn.from_torch(
            values, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
    values = torch.randn(1, 1, rows, HIDDEN, generator=generator)
    return ttnn.from_torch(
        values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )


def device_counts(device, counts: torch.Tensor):
    return ttnn.from_torch(
        counts.reshape(1, NUM_EXPERTS).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


def activation_of(projection: dict):
    if projection["gelu"] is None:
        return None
    return ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH, projection["gelu"])


def run(
    device, projection: dict, inputs, weight, counts, memory_config=ttnn.DRAM_MEMORY_CONFIG, per_expert_lanes=False
):
    fidelity = ttnn.MathFidelity.LoFi if projection["transposed"] else ttnn.MathFidelity.HiFi2
    return grouped_matmul(
        inputs,
        weight,
        counts,
        dtype=projection["dtype"],
        transpose_weight=projection["transposed"],
        activation=activation_of(projection),
        output_layout=projection["layout"],
        memory_config=memory_config,
        compute_kernel_config=compute_config(device, fidelity),
        per_expert_lanes=per_expert_lanes,
    )


def compute_config(device, fidelity):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


def assert_rows_match_stock(device, projection: dict, output, inputs, weight, counts: torch.Tensor):
    """Each expert's region of output against ttnn.matmul of the same rows and that expert's weights."""
    transposed = projection["transposed"]
    rows, k = inputs.shape[-2], inputs.shape[-1]
    n = HIDDEN if transposed else INTERMEDIATE
    got = ttnn.to_torch(output).float().reshape(rows, n)
    weight_values = ttnn.to_torch(weight).float().reshape(HIDDEN, NUM_EXPERTS * INTERMEDIATE)
    input_values = ttnn.to_torch(inputs).float().reshape(rows, k)
    grid = device.compute_with_storage_grid_size()
    per_core_n = math.ceil(n // 32 / grid.x)
    fidelity = ttnn.MathFidelity.LoFi if transposed else ttnn.MathFidelity.HiFi2
    starts = region_starts(counts)
    lengths = (counts + BLOCK_ROWS - 1) // BLOCK_ROWS * BLOCK_ROWS
    for expert in range(NUM_EXPERTS):
        begin, end = int(starts[expert]), int(starts[expert] + lengths[expert])
        if begin == end:
            continue
        columns = slice(expert * (k if transposed else n), (expert + 1) * (k if transposed else n))
        expert_weight = ttnn.from_torch(
            weight_values[:, columns].reshape(1, 1, HIDDEN, -1),
            dtype=weight.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        expert_rows = ttnn.from_torch(
            input_values[begin:end].reshape(1, 1, end - begin, k),
            dtype=inputs.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(math.ceil(n // 32 / per_core_n), grid.y),
            in0_block_w=k // 32,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=1,
            out_block_w=per_core_n,
            per_core_M=math.ceil((end - begin) // 32 / grid.y),
            per_core_N=per_core_n,
            transpose_mcast=False,
            fuse_batch=True,
            fused_activation=activation_of(projection),
        )
        want = ttnn.matmul(
            expert_rows,
            expert_weight,
            transpose_b=transposed,
            program_config=program_config,
            compute_kernel_config=compute_config(device, fidelity),
            dtype=projection["dtype"],
        )
        want_values = ttnn.to_torch(want).float().reshape(end - begin, n)
        mismatched = (got[begin:end] != want_values).any(-1).nonzero().reshape(-1)
        assert (
            mismatched.numel() == 0
        ), f"expert {expert}: rows {(mismatched + begin).tolist()[:8]} differ from ttnn.matmul"
        for tensor in (expert_weight, expert_rows, want):
            ttnn.deallocate(tensor)


CASES = [(name, tokens, routing) for name in PROJECTIONS for tokens in (1000, 4096) for routing in ROUTINGS] + [
    (name, tokens, "uniform") for name in PROJECTIONS for tokens in (352, 2048)
]


@pytest.mark.parametrize("projection_name, tokens, routing", CASES)
def test_rows_match_a_stock_matmul_per_expert(device, projection_name, tokens, routing):
    """Every row of every expert's region equals ttnn.matmul of that expert's rows, bit for bit.

    352 and 1000 tokens leave lanes with no block of some experts; 4096 is the model's largest pass.
    The routings cover one region, empty experts at both ends and between, and regions ending in a
    block of one real row.
    """
    projection = PROJECTIONS[projection_name]
    generator = torch.Generator().manual_seed(tokens)
    counts = expert_counts(routing, tokens, generator)
    weight = make_weight(device, generator)
    inputs = make_input(device, projection["transposed"], gathered_rows(tokens), generator)

    output = run(device, projection, inputs, weight, device_counts(device, counts))

    assert list(output.shape) == [1, 1, gathered_rows(tokens), HIDDEN if projection["transposed"] else INTERMEDIATE]
    assert output.layout == projection["layout"]
    assert_rows_match_stock(device, projection, output, inputs, weight, counts)


# per_expert_lanes, each core row multiplying one expert's blocks: the model's w1 and w2. At 32 tokens
# every lane has one block; at 352 the regions span several blocks and start past block 0, and
# one_expert puts every block on one lane and gives the other lanes none.
PER_EXPERT_CASES = [
    (name, tokens, routing)
    for name in ("w1_fast_gelu", "w2_row_major")
    for tokens, routing in ((32, "uniform"), (352, "uniform"), (352, "one_expert"))
]


@pytest.mark.parametrize("projection_name, tokens, routing", PER_EXPERT_CASES)
def test_per_expert_lanes_rows_match_a_stock_matmul_per_expert(device, projection_name, tokens, routing):
    """With per_expert_lanes every row still equals ttnn.matmul of its expert's rows, bit for bit.

    Seeded apart from test_rows_match_a_stock_matmul_per_expert, so an output placed on the freed
    buffer of an identical earlier call cannot pass for rows this one left unwritten.
    """
    projection = PROJECTIONS[projection_name]
    generator = torch.Generator().manual_seed(tokens + 1)
    counts = expert_counts(routing, tokens, generator)
    weight = make_weight(device, generator)
    inputs = make_input(device, projection["transposed"], gathered_rows(tokens), generator)

    output = run(device, projection, inputs, weight, device_counts(device, counts), per_expert_lanes=True)

    assert_rows_match_stock(device, projection, output, inputs, weight, counts)


def test_program_cache_reuse(device):
    """A second call with new routing at new addresses reuses the program and follows the new buffers."""
    projection = PROJECTIONS["w1_accurate_gelu"]
    tokens = 1000
    generator = torch.Generator().manual_seed(tokens)
    weight = make_weight(device, generator)
    first_counts = expert_counts("uniform", tokens, generator)
    first_inputs = make_input(device, False, gathered_rows(tokens), generator)
    first_output = run(device, projection, first_inputs, weight, device_counts(device, first_counts))
    entries = device.num_program_cache_entries()

    second_counts = expert_counts("empty_between", tokens, generator)
    second_inputs = make_input(device, False, gathered_rows(tokens), generator)
    second_output = run(device, projection, second_inputs, weight, device_counts(device, second_counts))

    assert device.num_program_cache_entries() == entries
    assert second_inputs.buffer_address() != first_inputs.buffer_address()
    assert second_output.buffer_address() != first_output.buffer_address()
    assert_rows_match_stock(device, projection, first_output, first_inputs, weight, first_counts)
    assert_rows_match_stock(device, projection, second_output, second_inputs, weight, second_counts)

    # per_expert_lanes is part of the program: the output is the same bit for bit, so only the
    # cache shows that the flag reaches the op.
    for _ in range(2):
        run(device, projection, second_inputs, weight, device_counts(device, second_counts), per_expert_lanes=True)
        assert device.num_program_cache_entries() == entries + 1


@pytest.mark.parametrize("per_expert_lanes", [False, True])
def test_every_used_row_is_written(device, per_expert_lanes):
    """Every row of every region comes from the call itself, not from what an earlier call left.

    Freed DRAM keeps its contents and the allocator hands the same addresses back first, so the
    output is placed on DRAM filled with NaN. The call before it runs a different routing through
    the same program, so its circular buffers hold other rows where the checked call must write.
    """
    projection = PROJECTIONS["w2_row_major"]
    tokens = 1000
    generator = torch.Generator().manual_seed(tokens)
    weight = make_weight(device, generator)
    inputs = make_input(device, True, gathered_rows(tokens), generator)
    counts = expert_counts("one_mod_32", tokens, generator)
    stale_counts = device_counts(device, expert_counts("uniform", tokens, generator))
    checked_counts = device_counts(device, counts)

    # Builds the program first: a cold call also allocates its kernel binaries in DRAM, which would
    # move the output off the poison.
    ttnn.deallocate(run(device, projection, inputs, weight, checked_counts, per_expert_lanes=per_expert_lanes))
    ttnn.deallocate(run(device, projection, inputs, weight, stale_counts, per_expert_lanes=per_expert_lanes))
    poison = ttnn.from_torch(
        torch.full((1, 1, gathered_rows(tokens), HIDDEN), float("nan")),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    poisoned_address = poison.buffer_address()
    ttnn.synchronize_device(device)
    ttnn.deallocate(poison)

    output = run(device, projection, inputs, weight, checked_counts, per_expert_lanes=per_expert_lanes)

    assert output.buffer_address() == poisoned_address
    assert_rows_match_stock(device, projection, output, inputs, weight, counts)


def regions_cut_at(counts: torch.Tensor, rows: int) -> torch.Tensor:
    """counts as the kernels apply them to an input of `rows` rows: the region crossing its end stops
    there, and the experts after it get none."""
    room = rows // BLOCK_ROWS
    cut = torch.zeros_like(counts)
    for expert, count in enumerate(counts.tolist()):
        blocks = min(math.ceil(count / BLOCK_ROWS), room)
        cut[expert] = count if blocks * BLOCK_ROWS >= count else blocks * BLOCK_ROWS
        room -= blocks
    return cut


def test_counts_past_the_input_write_nothing_past_the_output(device):
    """Counts that fill more blocks than the input holds, a larger window's say, stop at its end: the
    regions inside it are exact and nothing is written past the output.

    L1 buffers are allocated from the top down, so the output lands right below a guard allocated
    just before the call, where a write past the output's end would land.
    """
    projection = PROJECTIONS["w1_accurate_gelu"]
    tokens = 352
    generator = torch.Generator().manual_seed(tokens)
    weight = make_weight(device, generator)
    rows = gathered_rows(tokens)
    inputs = make_input(device, False, rows, generator)
    # A 640-token window's counts: 40 to 47 blocks against the input's 29.
    counts = expert_counts("uniform", 640, generator)
    window_counts = device_counts(device, counts)
    guard_values = torch.full((1, 1, 2048, HIDDEN), 7.0).bfloat16()
    guard = ttnn.from_torch(
        guard_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    output = run(device, projection, inputs, weight, window_counts, memory_config=ttnn.L1_MEMORY_CONFIG)

    grid = device.compute_with_storage_grid_size()
    output_tiles = rows // BLOCK_ROWS * INTERMEDIATE // 32
    bank_bytes = math.ceil(output_tiles / (grid.x * grid.y)) * ttnn.tile_size(ttnn.bfloat8_b)
    assert guard.buffer_address() == output.buffer_address() + bank_bytes
    assert torch.equal(ttnn.to_torch(guard), guard_values)
    assert_rows_match_stock(device, projection, output, inputs, weight, regions_cut_at(counts, rows))


ROWS = 32 * NUM_EXPERTS


def _input(shape=(1, 1, ROWS, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, **kwargs):
    """A factory the test calls with its device: the table below is built before any device exists."""
    return lambda device: ttnn.from_torch(torch.zeros(shape), dtype=dtype, layout=layout, device=device, **kwargs)


def _counts(shape=(1, NUM_EXPERTS), dtype=ttnn.uint32):
    return lambda device: ttnn.from_torch(
        torch.zeros(shape, dtype=torch.int32), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


def _width_sharded(device):
    memory_config = ttnn.create_sharded_memory_config(
        (ROWS, HIDDEN // 2), ttnn.CoreGrid(y=1, x=2), ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
    )
    return _input(memory_config=memory_config)(device)


GELU = ttnn.UnaryOpType.GELU_TANH

# case: (tensors replacing the valid ones, keyword arguments replacing the valid ones, error message)
INVALID_CALLS = {
    "input_dtype": (dict(input_tensor=_input(dtype=ttnn.float32)), {}, "input_tensor must be BFLOAT16 or BFLOAT8_B"),
    "input_row_major": (
        dict(input_tensor=_input(layout=ttnn.ROW_MAJOR_LAYOUT)),
        {},
        "input_tensor must be TILE layout",
    ),
    "input_sharded": (dict(input_tensor=_width_sharded), {}, "input_tensor must be interleaved"),
    "input_tile_shape": (
        dict(input_tensor=_input(tile=ttnn.Tile((16, 32)))),
        {},
        "input_tensor must use plain 32 x 32 tiles",
    ),
    "input_rank": (
        dict(input_tensor=_input(shape=(2, 1, ROWS, HIDDEN))),
        {},
        "input_tensor must be 1 x 1 x rows x columns",
    ),
    "k_mismatch": (dict(input_tensor=_input(shape=(1, 1, ROWS, 2 * HIDDEN))), {}, "input_tensor has K = 1536"),
    "experts_do_not_split": (
        dict(expert_counts=_counts(shape=(1, 5))),
        {},
        "the weight's 24576 columns do not split into 5 experts",
    ),
    "counts_dtype": (dict(expert_counts=_counts(dtype=ttnn.int32)), {}, "expert_counts must be UINT32"),
    "counts_shape": (dict(expert_counts=_counts(shape=(2, NUM_EXPERTS))), {}, "expert_counts must be 1 x num_experts"),
    "too_many_experts": (dict(expert_counts=_counts(shape=(1, 96))), {}, "num_experts must be 1 to 64"),
    "per_expert_lanes_too_many_experts": (
        dict(expert_counts=_counts(shape=(1, 16))),
        dict(per_expert_lanes=True),
        "per_expert_lanes gives each expert a core row",
    ),
    "output_dtype": ({}, dict(dtype=ttnn.float32), "dtype must be BFLOAT16 or BFLOAT8_B"),
    "row_major_bfloat8_b": (
        {},
        dict(dtype=ttnn.bfloat8_b, output_layout=ttnn.ROW_MAJOR_LAYOUT),
        "a ROW_MAJOR output must be BFLOAT16",
    ),
    "row_major_uneven_columns": (
        {},
        dict(dtype=ttnn.bfloat16, output_layout=ttnn.ROW_MAJOR_LAYOUT),
        "a ROW_MAJOR output needs N split evenly over the core columns",
    ),
    "activation_type": (
        {},
        dict(activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)),
        "activation must be GELU_TANH",
    ),
    "activation_parameter": ({}, dict(activation=ttnn.UnaryWithParam(GELU, 0.5)), "activation must be GELU_TANH"),
    "memory_config": (
        {},
        dict(
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 7))}),
                    (32, INTERMEDIATE),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
        ),
        "memory_config must be interleaved",
    ),
}


@pytest.mark.parametrize("case", INVALID_CALLS.keys())
def test_invalid_inputs_raise(device, expect_error, case):
    tensor_overrides, call_overrides, message = INVALID_CALLS[case]
    tensors = dict(
        input_tensor=_input(),
        weight=_input(shape=(1, 1, HIDDEN, NUM_EXPERTS * INTERMEDIATE), dtype=ttnn.bfloat8_b),
        expert_counts=_counts(),
    )
    tensors |= tensor_overrides
    kwargs = dict(dtype=ttnn.bfloat8_b) | call_overrides

    with expect_error(RuntimeError, f"grouped_matmul: {message}"):
        grouped_matmul(
            tensors["input_tensor"](device), tensors["weight"](device), tensors["expert_counts"](device), **kwargs
        )
