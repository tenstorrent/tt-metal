# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w1 against ttnn.matmul, expert by expert.

The op runs each block of rows through the same multiply-accumulate sequence as a stock 2D matmul of
that expert's rows with its weights and the same blocking (K in one block, 1x3 subblocks, fp32
accumulation), and the same packer-side GELU. So every row of every expert's region must equal the
stock result bit for bit; a wrong expert, block, column or K order shows as a mismatch. The plain
TILE weight and the chunked weight of prepare_grouped_matmul_weight are both held to it.
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

grouped_matmul = ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w1
prepare_grouped_matmul_weight = ttnn.experimental.nomic_embed_text_v2_moe.prepare_grouped_matmul_w1_weight

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


def chunked_weight(device, projection: dict, weight, chunk_tiles: int, bias=None, num_experts=NUM_EXPERTS):
    """weight (and bias) as prepare_grouped_matmul_weight lays them out, and the keyword arguments that go with it."""
    transposed = projection["transposed"]
    n = weight.shape[-2] if transposed else weight.shape[-1] // num_experts
    prepared = prepare_grouped_matmul_weight(
        weight, num_experts, transpose_weight=transposed, bias=bias, chunk_tiles=chunk_tiles, device=device
    )
    return prepared, dict(weight_chunk_tiles=chunk_tiles, weight_n=n, weight_has_bias=bias is not None)


def run(
    device,
    projection: dict,
    inputs,
    weight,
    counts,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    per_expert_lanes=False,
    chunk_tiles=0,
):
    """With chunk_tiles, the call reads weight as a chunked weight of that chunk size."""
    fidelity = ttnn.MathFidelity.LoFi if projection["transposed"] else ttnn.MathFidelity.HiFi2
    extra = {}
    if chunk_tiles:
        weight, extra = chunked_weight(device, projection, weight, chunk_tiles)
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
        **extra,
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


@pytest.mark.parametrize("per_expert_lanes, chunk_tiles", [(False, 0), (True, 0), (False, 60), (True, 60)])
def test_every_used_row_is_written(device, per_expert_lanes, chunk_tiles):
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
    lanes = dict(per_expert_lanes=per_expert_lanes)
    if chunk_tiles:
        # Prepared once, before the poison: a weight allocated after it would take its place.
        weight_used, extra = chunked_weight(device, projection, weight, chunk_tiles)
    else:
        weight_used, extra = weight, {}

    def call(counts_tensor):
        fidelity = ttnn.MathFidelity.LoFi
        return grouped_matmul(
            inputs,
            weight_used,
            counts_tensor,
            dtype=projection["dtype"],
            transpose_weight=True,
            output_layout=projection["layout"],
            compute_kernel_config=compute_config(device, fidelity),
            **lanes,
            **extra,
        )

    ttnn.deallocate(call(checked_counts))
    ttnn.deallocate(call(stale_counts))
    poison = ttnn.from_torch(
        torch.full((1, 1, gathered_rows(tokens), HIDDEN), float("nan")),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    poisoned_address = poison.buffer_address()
    ttnn.synchronize_device(device)
    ttnn.deallocate(poison)

    output = call(checked_counts)

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


GELU = ttnn.UnaryOpType.GELU_TANH

# case: (tensors replacing the valid ones, keyword arguments replacing the valid ones, error message)
INVALID_CALLS = {
    "input_dtype": (dict(input_tensor=_input(dtype=ttnn.float32)), {}, "input_tensor must be BFLOAT16 or BFLOAT8_B"),
    "input_row_major": (
        dict(input_tensor=_input(layout=ttnn.ROW_MAJOR_LAYOUT)),
        {},
        "input_tensor must be TILE layout",
    ),
    "input_tile_shape": (
        dict(input_tensor=_input(tile=ttnn.Tile((16, 32)))),
        {},
        "input_tensor must use plain 32 x 32 tiles",
    ),
    "input_rank": (
        dict(input_tensor=_input(shape=(1, 2, ROWS, HIDDEN))),
        {},
        "input_tensor must be B x 1 x rows x K",
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
    "row_major_batched_input": (
        dict(input_tensor=_input(shape=(2, 1, ROWS, INTERMEDIATE))),
        dict(dtype=ttnn.bfloat16, transpose_weight=True, output_layout=ttnn.ROW_MAJOR_LAYOUT),
        "a ROW_MAJOR output needs input_tensor 1 x 1 x rows x K in whole tiles",
    ),
    "row_major_ragged_input": (
        dict(input_tensor=_input(shape=(1, 1, ROWS - 28, INTERMEDIATE))),
        dict(dtype=ttnn.bfloat16, transpose_weight=True, output_layout=ttnn.ROW_MAJOR_LAYOUT),
        "a ROW_MAJOR output needs input_tensor 1 x 1 x rows x K in whole tiles",
    ),
    "activation_type": (
        {},
        dict(activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)),
        "activation must be GELU_TANH",
    ),
    "activation_parameter": ({}, dict(activation=ttnn.UnaryWithParam(GELU, 0.5)), "activation must be GELU_TANH"),
    "bias_dtype": (
        dict(bias=_input(shape=(1, 1, 1, NUM_EXPERTS * INTERMEDIATE))),
        {},
        "bias must be interleaved TILE of the weight's dtype",
    ),
    "bias_shape": (
        dict(bias=_input(shape=(1, 1, 1, INTERMEDIATE), dtype=ttnn.bfloat8_b)),
        {},
        "bias must be 1 x 1 x 1 x 24576",
    ),
    "bias_transposed_weight": (
        dict(
            input_tensor=_input(shape=(1, 1, ROWS, INTERMEDIATE)),
            bias=_input(shape=(1, 1, 1, NUM_EXPERTS * INTERMEDIATE), dtype=ttnn.bfloat8_b),
        ),
        dict(transpose_weight=True),
        "bias needs the weight untransposed",
    ),
    "bias_input_dtype": (
        dict(
            input_tensor=_input(dtype=ttnn.bfloat8_b),
            bias=_input(shape=(1, 1, 1, NUM_EXPERTS * INTERMEDIATE), dtype=ttnn.bfloat8_b),
        ),
        {},
        "bias needs a BFLOAT16 input",
    ),
    "chunked_with_bias_tensor": (
        dict(bias=_input(shape=(1, 1, 1, NUM_EXPERTS * INTERMEDIATE), dtype=ttnn.bfloat8_b)),
        dict(weight_chunk_tiles=60, weight_n=INTERMEDIATE),
        "a chunked weight carries its bias",
    ),
    "chunked_without_n": ({}, dict(weight_chunk_tiles=60), "a chunked weight needs weight_n"),
    "chunked_interleaved_weight": (
        {},
        dict(weight_chunk_tiles=60, weight_n=INTERMEDIATE),
        "a chunked weight must be TILE and WIDTH_SHARDED in DRAM",
    ),
    "has_bias_unchunked": ({}, dict(weight_has_bias=True), "weight_has_bias needs a chunked weight"),
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
    if "bias" in tensors:
        kwargs["bias"] = tensors["bias"](device)

    with expect_error(RuntimeError, f"grouped_matmul_w1: {message}"):
        grouped_matmul(
            tensors["input_tensor"](device), tensors["weight"](device), tensors["expert_counts"](device), **kwargs
        )


@pytest.mark.parametrize("chunk_tiles", [0, 2])
def test_every_expert_up_to_the_limit_is_read(device, chunk_tiles):
    """64 experts, the most the op takes: every expert's count reaches every kernel, so each region holds
    its own expert's rows. A small K and N keep it cheap; float32 matmuls of the same bfloat16 values are
    the oracle, within the bfloat16 output's rounding."""
    experts, k, n = 64, 64, 96
    generator = torch.Generator().manual_seed(experts)
    counts = torch.randint(0, 3 * BLOCK_ROWS, (experts,), generator=generator)
    counts[::7] = 0
    lengths = (counts + BLOCK_ROWS - 1) // BLOCK_ROWS * BLOCK_ROWS
    rows = int(lengths.sum())
    input_values = torch.randn(1, 1, rows, k, generator=generator).bfloat16().float()
    weight_values = (torch.randn(1, 1, k, experts * n, generator=generator) * 0.1).bfloat16().float()
    weight = ttnn.from_torch(weight_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    extra = {}
    if chunk_tiles:
        weight = prepare_grouped_matmul_weight(weight, experts, chunk_tiles=chunk_tiles, device=device)
        extra = dict(weight_chunk_tiles=chunk_tiles, weight_n=n)
    counts_tensor = ttnn.from_torch(
        counts.reshape(1, experts).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    output = grouped_matmul(
        ttnn.from_torch(input_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        weight,
        counts_tensor,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_config(device, ttnn.MathFidelity.HiFi4),
        **extra,
    )

    got = ttnn.to_torch(output).float().reshape(rows, n)
    begin = 0
    for expert in range(experts):
        end = begin + int(lengths[expert])
        want = input_values[0, 0, begin:end] @ weight_values[0, 0, :, expert * n : (expert + 1) * n]
        torch.testing.assert_close(got[begin:end], want, rtol=1e-2, atol=1e-2, msg=f"expert {expert}")
        begin = end


def bias_as_a_k_step(input_values: torch.Tensor, weight_values: torch.Tensor, bias_values: torch.Tensor):
    """The input and weight of the plain call a bias call amounts to: one more K tile, a tile whose rows
    hold 1 in their first column in the input's place, and one holding the bias in its first row in the
    weight's. The op runs its bias step after the K loop, in that order, so the two match bit for bit."""
    rows = input_values.shape[-2]
    ones = torch.zeros(1, 1, rows, 32)
    ones[..., 0] = 1.0
    bias_tile = torch.zeros(1, 1, 32, weight_values.shape[-1])
    bias_tile[..., 0, :] = bias_values.reshape(-1)
    return torch.cat((input_values, ones), dim=-1), torch.cat((weight_values, bias_tile), dim=-2)


# (experts, tokens, routing): the model's dense fc1, one expert over every row, and a routed w1.
BIAS_CASES = [(1, 1000, None), (NUM_EXPERTS, 1000, "uniform"), (NUM_EXPERTS, 352, "empty_between")]


@pytest.mark.parametrize("experts, tokens, routing", BIAS_CASES)
@pytest.mark.parametrize("gelu", [None, 2.0])
def test_bias_is_one_more_k_step(device, experts, tokens, routing, gelu):
    """A call with a bias equals, bit for bit, the plain call whose K takes the bias as one more step."""
    generator = torch.Generator().manual_seed(tokens + experts)
    rows = gathered_rows(tokens)
    counts = torch.tensor([rows]) if experts == 1 else expert_counts(routing, tokens, generator)
    input_values = torch.randn(1, 1, rows, HIDDEN, generator=generator)
    weight_values = torch.randn(1, 1, HIDDEN, experts * INTERMEDIATE, generator=generator) * 0.03
    bias_values = torch.randn(1, 1, 1, experts * INTERMEDIATE, generator=generator) * 0.5
    augmented_input, augmented_weight = bias_as_a_k_step(input_values, weight_values, bias_values)

    def call(inputs, weight, bias=None):
        tensors = [
            ttnn.from_torch(values, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
            for values, dtype in ((inputs, ttnn.bfloat16), (weight, ttnn.bfloat8_b))
        ]
        if bias is not None:
            bias = ttnn.from_torch(bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        out = grouped_matmul(
            *tensors,
            ttnn.from_torch(
                counts.reshape(1, experts).to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            ),
            dtype=ttnn.bfloat8_b,
            activation=None if gelu is None else ttnn.UnaryWithParam(GELU, gelu),
            compute_kernel_config=compute_config(device, ttnn.MathFidelity.HiFi2),
            bias=bias,
        )
        return ttnn.to_torch(out).float()

    got = call(input_values, weight_values, bias_values)
    want = call(augmented_input, augmented_weight)

    used = int(((counts + BLOCK_ROWS - 1) // BLOCK_ROWS * BLOCK_ROWS).sum())
    assert torch.equal(
        got[..., :used, :], want[..., :used, :]
    ), f"max abs {(got[..., :used, :] - want[..., :used, :]).abs().max():.3e} against the bias as a K step"


def test_batched_input_matches_its_flat_rows(device):
    """A (B, 1, S, K) input runs as its tile rows, each sequence's padding included: the output is
    (B, 1, S, N) and equals, bit for bit, the call on those rows laid out flat."""
    batch, seqlen = 3, 437
    padded = math.ceil(seqlen / 32) * 32
    generator = torch.Generator().manual_seed(seqlen)
    input_values = torch.randn(batch, 1, seqlen, HIDDEN, generator=generator)
    flat_values = torch.zeros(batch, 1, padded, HIDDEN)
    flat_values[:, :, :seqlen] = input_values
    weight = ttnn.from_torch(
        torch.randn(1, 1, HIDDEN, INTERMEDIATE, generator=generator) * 0.03,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    bias = ttnn.from_torch(
        torch.randn(1, 1, 1, INTERMEDIATE, generator=generator) * 0.5,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    counts = ttnn.from_torch(
        torch.tensor([[batch * padded]], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    def call(values):
        inputs = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        return grouped_matmul(
            inputs,
            weight,
            counts,
            dtype=ttnn.bfloat8_b,
            activation=ttnn.UnaryWithParam(GELU, 2.0),
            compute_kernel_config=compute_config(device, ttnn.MathFidelity.HiFi2),
            bias=bias,
        )

    batched = call(input_values)
    flat = call(flat_values.reshape(1, 1, batch * padded, HIDDEN))

    assert list(batched.shape) == [batch, 1, seqlen, INTERMEDIATE]
    got = ttnn.to_torch(batched).float()
    want = ttnn.to_torch(flat).float().reshape(batch, 1, padded, INTERMEDIATE)[:, :, :seqlen]
    assert torch.equal(got, want), f"max abs {(got - want).abs().max():.3e} against the flat rows"


@pytest.mark.parametrize("experts", [1, NUM_EXPERTS])
def test_polynomial_gelu_tracks_the_accurate_form(device, experts):
    """GELU_TANH parameter 2, the polynomial, against parameter 0, the accurate tanh form, both into
    bfloat16: within the two forms' own errors against exact GELU (1.3e-3 and 4.7e-4) plus a rounding
    of each output."""
    tokens = 1000
    generator = torch.Generator().manual_seed(tokens + experts)
    rows = gathered_rows(tokens)
    counts = torch.tensor([rows]) if experts == 1 else expert_counts("uniform", tokens, generator)
    inputs = ttnn.from_torch(
        torch.randn(1, 1, rows, HIDDEN, generator=generator),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    weight = ttnn.from_torch(
        torch.randn(1, 1, HIDDEN, experts * INTERMEDIATE, generator=generator) * 0.06,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    device_counts_ = ttnn.from_torch(
        counts.reshape(1, experts).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    def call(form):
        out = grouped_matmul(
            inputs,
            weight,
            device_counts_,
            dtype=ttnn.bfloat16,
            activation=ttnn.UnaryWithParam(GELU, form),
            compute_kernel_config=compute_config(device, ttnn.MathFidelity.HiFi2),
        )
        return ttnn.to_torch(out).float()

    polynomial, accurate = call(2.0), call(0.0)

    used = int(((counts + BLOCK_ROWS - 1) // BLOCK_ROWS * BLOCK_ROWS).sum())
    polynomial, accurate = polynomial[..., :used, :], accurate[..., :used, :]
    bound = 1.8e-3 + 2.0**-7 * accurate.abs()
    excess = (polynomial - accurate).abs() - bound
    assert excess.max() <= 0, f"{int((excess > 0).sum())} values past the bound, worst by {excess.max():.3e}"
    assert accurate.min() < -0.1 and accurate.max() > 1.0, "the inputs no longer reach both of GELU's regimes"


def block_sharded(columns: int, block_rows: int, width: int) -> ttnn.MemoryConfig:
    """Column-major block sharding over 8 core rows, as the model's split layer norm lays out its output."""
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(columns - 1, 7))})
    shard = ttnn.ShardSpec(grid, (block_rows * 32, width // 8), ttnn.ShardOrientation.COL_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard)


@pytest.mark.parametrize("batch, seqlen, columns, block_rows", [(4, 512, 11, 6), (3, 437, 7, 6)])
def test_sharded_input_matches_the_interleaved_call(device, batch, seqlen, columns, block_rows):
    """The reader addresses the input by page id, so a block-sharded input gives the interleaved
    call's output bit for bit, a sequence's tile padding included (3x437) and a last column of
    cores holding fewer rows (4x512: 64 tile rows on 11 columns of 6)."""
    generator = torch.Generator().manual_seed(seqlen)
    input_values = torch.randn(batch, 1, seqlen, HIDDEN, generator=generator)
    weight = make_weight(device, generator)
    counts = ttnn.from_torch(
        torch.tensor([[batch * math.ceil(seqlen / 32) * 32] + [0] * (NUM_EXPERTS - 1)], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    def call(memory_config):
        inputs = ttnn.from_torch(
            input_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
        return ttnn.to_torch(
            grouped_matmul(
                inputs,
                weight,
                counts,
                dtype=ttnn.bfloat8_b,
                activation=ttnn.UnaryWithParam(GELU, 2.0),
                compute_kernel_config=compute_config(device, ttnn.MathFidelity.HiFi2),
            )
        )

    assert torch.equal(call(block_sharded(columns, block_rows, HIDDEN)), call(ttnn.L1_MEMORY_CONFIG))


# The chunked weight against the same oracle. At 4096 tokens w2 has dedicated input senders east of its eight
# columns (a lane holds 26 blocks); at 1000 its east core sends. Chunks of 7 tiles leave a ragged last chunk in
# every slice.
CHUNKED_CASES = [
    (name, tokens, routing, 60)
    for name in PROJECTIONS
    for tokens in (1000, 4096)
    for routing in ("uniform", "one_expert", "empty_between", "one_mod_32")
] + [(name, 1000, "uniform", 7) for name in ("w1_fast_gelu", "w2_row_major")]


@pytest.mark.parametrize("projection_name, tokens, routing, chunk_tiles", CHUNKED_CASES)
def test_chunked_weight_rows_match_a_stock_matmul_per_expert(device, projection_name, tokens, routing, chunk_tiles):
    """With a chunked weight every row still equals ttnn.matmul of its expert's rows, bit for bit."""
    projection = PROJECTIONS[projection_name]
    generator = torch.Generator().manual_seed(tokens + 2)
    counts = expert_counts(routing, tokens, generator)
    weight = make_weight(device, generator)
    inputs = make_input(device, projection["transposed"], gathered_rows(tokens), generator)

    output = run(device, projection, inputs, weight, device_counts(device, counts), chunk_tiles=chunk_tiles)

    assert_rows_match_stock(device, projection, output, inputs, weight, counts)


@pytest.mark.parametrize("projection_name, tokens, routing", PER_EXPERT_CASES)
def test_per_expert_lanes_with_a_chunked_weight(device, projection_name, tokens, routing):
    """per_expert_lanes reads the chunked weight, each core its own expert's slice, bit for bit as ttnn.matmul."""
    projection = PROJECTIONS[projection_name]
    generator = torch.Generator().manual_seed(tokens + 3)
    counts = expert_counts(routing, tokens, generator)
    weight = make_weight(device, generator)
    inputs = make_input(device, projection["transposed"], gathered_rows(tokens), generator)

    output = run(
        device, projection, inputs, weight, device_counts(device, counts), per_expert_lanes=True, chunk_tiles=60
    )

    assert_rows_match_stock(device, projection, output, inputs, weight, counts)


def expected_chunked_tiles(weight_values, bias_values, num_experts, transposed, chunk_tiles, grid_x):
    """The chunked weight's tiles, (8 * shard tiles, 32, 32), from the TILE weight's values: each share's slice of
    each expert in slot order (each subblock's K rows, then its bias tiles), chunk j of share s and expert e in bank
    (s + j + e) % 8, written here independently of the op's own layout code."""
    rows, columns = weight_values.shape[-2:]
    k_tiles = columns // num_experts // 32 if transposed else rows // 32
    n_tiles = rows // 32 if transposed else columns // num_experts // 32
    width = math.ceil(n_tiles / grid_x)
    shares = math.ceil(n_tiles / width)
    widths = [width] * (shares - 1) + [n_tiles - (shares - 1) * width]
    subblock = max(s for s in range(1, 5) if all(w % s == 0 for w in widths))
    k_rows = k_tiles + (bias_values is not None)
    max_chunks = math.ceil(k_rows * width / chunk_tiles)
    sharers = math.ceil(shares / 8)
    shard_tiles = num_experts * max_chunks * sharers * chunk_tiles
    tiles = torch.zeros(8 * shard_tiles, 32, 32)

    def weight_tile(expert, k, n):
        row, column = (n, expert * k_tiles + k) if transposed else (k, expert * n_tiles + n)
        return weight_values[row * 32 : (row + 1) * 32, column * 32 : (column + 1) * 32]

    for share, share_width in enumerate(widths):
        first = share * width
        order = []
        for group in range(share_width // subblock):
            columns_of_group = range(first + group * subblock, first + (group + 1) * subblock)
            order += [(k, n) for k in range(k_tiles) for n in columns_of_group]
            if bias_values is not None:
                order += [("bias", n) for n in columns_of_group]
        for expert in range(num_experts):
            for position, (row, n) in enumerate(order):
                chunk, offset = divmod(position, chunk_tiles)
                bank = (share + chunk + expert) % 8
                index = (
                    bank * shard_tiles + ((expert * max_chunks + chunk) * sharers + share // 8) * chunk_tiles + offset
                )
                if row == "bias":
                    tiles[index, 0] = bias_values.reshape(-1)[
                        (expert * n_tiles + n) * 32 : (expert * n_tiles + n + 1) * 32
                    ]
                else:
                    tiles[index] = weight_tile(expert, row, n)
    return tiles


@pytest.mark.parametrize(
    "transposed, with_bias, chunk_tiles", [(False, False, 60), (True, False, 60), (False, True, 15), (True, False, 7)]
)
def test_prepared_weight_holds_each_tile_where_the_op_reads_it(device, transposed, with_bias, chunk_tiles):
    """Every tile of the chunked weight is the TILE weight's tile the layout puts there, exactly."""
    generator = torch.Generator().manual_seed(chunk_tiles)
    experts = 1 if with_bias else NUM_EXPERTS
    dtype = ttnn.bfloat16 if with_bias else ttnn.bfloat8_b
    weight = ttnn.from_torch(
        torch.randn(1, 1, HIDDEN, experts * INTERMEDIATE, generator=generator) * 0.03,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    bias = None
    if with_bias:
        bias = ttnn.from_torch(
            torch.randn(1, 1, 1, experts * INTERMEDIATE, generator=generator),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    prepared = prepare_grouped_matmul_weight(
        weight, experts, transpose_weight=transposed, bias=bias, chunk_tiles=chunk_tiles, device=device
    )

    weight_values = ttnn.to_torch(weight).float().reshape(HIDDEN, experts * INTERMEDIATE)
    bias_values = ttnn.to_torch(bias).float() if with_bias else None
    want = expected_chunked_tiles(
        weight_values, bias_values, experts, transposed, chunk_tiles, device.compute_with_storage_grid_size().x
    )
    got = ttnn.to_torch(prepared).float().reshape(32, -1, 32).permute(1, 0, 2)
    assert got.shape == want.shape, f"{got.shape} tiles against {want.shape}"
    assert torch.equal(got, want)


# One expert with a bias (the model's dense fc1): compute runs subblock-outer over a lane's whole set of blocks with
# the tail split at up to 4 blocks a lane (2 x 512), in groups of 2 above (8 x 512 and 3 x 437). Eight experts with a
# bias stay block by block.
CHUNKED_BIAS_CASES = [(1, (2, 512)), (1, (8, 512)), (1, (3, 437)), (NUM_EXPERTS, (1, 2016))]


@pytest.mark.parametrize("experts, input_shape", CHUNKED_BIAS_CASES)
def test_chunked_weight_with_bias_matches_the_tile_weight(device, experts, input_shape):
    """A chunked weight holding the bias gives the TILE weight's bias call bit for bit."""
    batch, seqlen = input_shape
    generator = torch.Generator().manual_seed(seqlen + experts)
    rows = batch * math.ceil(seqlen / 32) * 32
    counts = torch.tensor([rows]) if experts == 1 else expert_counts("uniform", rows // 2, generator)
    inputs = ttnn.from_torch(
        torch.randn(batch, 1, seqlen, HIDDEN, generator=generator),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    weight = ttnn.from_torch(
        torch.randn(1, 1, HIDDEN, experts * INTERMEDIATE, generator=generator) * 0.03,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    bias = ttnn.from_torch(
        torch.randn(1, 1, 1, experts * INTERMEDIATE, generator=generator) * 0.5,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    counts_tensor = ttnn.from_torch(
        counts.reshape(1, experts).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    common = dict(
        dtype=ttnn.bfloat8_b,
        activation=ttnn.UnaryWithParam(GELU, 2.0),
        compute_kernel_config=compute_config(device, ttnn.MathFidelity.HiFi2),
    )
    prepared, extra = chunked_weight(device, PROJECTIONS["w1_fast_gelu"], weight, 15, bias=bias, num_experts=experts)

    want = ttnn.to_torch(grouped_matmul(inputs, weight, counts_tensor, bias=bias, **common)).float()
    got = ttnn.to_torch(grouped_matmul(inputs, prepared, counts_tensor, **common, **extra)).float()

    used = min(rows, int(((counts + BLOCK_ROWS - 1) // BLOCK_ROWS * BLOCK_ROWS).sum()))
    got, want = got.reshape(-1, INTERMEDIATE), want.reshape(-1, INTERMEDIATE)
    if experts == 1:
        # (B, 1, S, N) holds only the logical rows of each sequence.
        assert torch.equal(got, want)
    else:
        assert torch.equal(got[:used], want[:used])


@pytest.mark.parametrize("batch, seqlen", [(1, 1024), (8, 512), (3, 300)])
def test_no_counts_is_one_expert_over_every_row(device, batch, seqlen):
    """expert_counts=None, one expert taking every row, equals passing that count, bit for bit (fc1's call)."""
    generator = torch.Generator().manual_seed(batch * seqlen)
    rows = batch * math.ceil(seqlen / 32) * 32
    inputs = ttnn.from_torch(
        torch.randn(batch, 1, seqlen, HIDDEN, generator=generator),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    weight = ttnn.from_torch(
        torch.randn(1, 1, HIDDEN, INTERMEDIATE, generator=generator) * 0.03,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    bias = ttnn.from_torch(
        torch.randn(1, 1, 1, INTERMEDIATE, generator=generator) * 0.5,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    prepared, extra = chunked_weight(device, PROJECTIONS["w1_fast_gelu"], weight, 15, bias=bias, num_experts=1)
    counts = ttnn.from_torch(
        torch.tensor([[rows]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    common = dict(
        dtype=ttnn.bfloat8_b,
        activation=ttnn.UnaryWithParam(GELU, 2.0),
        compute_kernel_config=compute_config(device, ttnn.MathFidelity.HiFi2),
        **extra,
    )

    want = ttnn.to_torch(grouped_matmul(inputs, prepared, counts, **common))
    got = ttnn.to_torch(grouped_matmul(inputs, prepared, None, **common))

    assert torch.equal(got, want)


def test_program_cache_reuse_with_a_chunked_weight(device):
    """A second call with new routing at new addresses reuses the program and follows the new buffers."""
    projection = PROJECTIONS["w2_row_major"]
    tokens = 4096
    generator = torch.Generator().manual_seed(tokens + 4)
    weight = make_weight(device, generator)
    prepared, extra = chunked_weight(device, projection, weight, 60)

    def call(inputs, counts):
        return grouped_matmul(
            inputs,
            prepared,
            device_counts(device, counts),
            dtype=projection["dtype"],
            transpose_weight=True,
            output_layout=projection["layout"],
            compute_kernel_config=compute_config(device, ttnn.MathFidelity.LoFi),
            **extra,
        )

    first_counts = expert_counts("uniform", tokens, generator)
    first_inputs = make_input(device, True, gathered_rows(tokens), generator)
    first_output = call(first_inputs, first_counts)
    entries = device.num_program_cache_entries()
    second_counts = expert_counts("first_and_last", tokens, generator)
    second_inputs = make_input(device, True, gathered_rows(tokens), generator)
    second_output = call(second_inputs, second_counts)

    assert device.num_program_cache_entries() == entries
    assert second_inputs.buffer_address() != first_inputs.buffer_address()
    assert_rows_match_stock(device, projection, first_output, first_inputs, weight, first_counts)
    assert_rows_match_stock(device, projection, second_output, second_inputs, weight, second_counts)


QKV = 3 * HIDDEN


def one_expert_call(device, batch, seqlen, n, gelu, seed):
    """A dense projection's call as the model makes it: one expert, a chunked weight with its bias, bfloat16 rows."""
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(batch, 1, seqlen, HIDDEN, generator=generator)
    weight_values = torch.randn(1, 1, HIDDEN, n, generator=generator) * 0.03
    bias_values = torch.randn(1, 1, 1, n, generator=generator) * 0.5
    inputs = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(weight_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(bias_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    prepared, extra = chunked_weight(device, PROJECTIONS["w1_fast_gelu"], weight, 15, bias=bias, num_experts=1)
    kwargs = dict(
        dtype=ttnn.bfloat16,
        activation=None if gelu is None else ttnn.UnaryWithParam(GELU, gelu),
        compute_kernel_config=compute_config(device, ttnn.MathFidelity.HiFi2),
        **extra,
    )
    return inputs, prepared, kwargs, (values, weight_values, bias_values)


# The tail split in groups (more than 4 blocks a lane, without counts): QKV's N of 72 tiles (shares of 7, 1 x 1
# subblocks) at 64 tile rows, 6 full blocks a lane and 4 tail blocks over 10 lanes; at 50, 5 and 0 tail blocks; fc1's
# N at 64 tile rows. With counts the same call deals its blocks whole, so the two must agree bit for bit.
GROUPED_TAIL_CASES = [(4, 512, QKV, None), (5, 300, QKV, None), (4, 512, INTERMEDIATE, 2.0)]


@pytest.mark.parametrize("batch, seqlen, n, gelu", GROUPED_TAIL_CASES)
def test_grouped_tail_split_matches_the_whole_block_deal(device, batch, seqlen, n, gelu):
    """Without counts the tail blocks' subblocks spread over the lanes; with counts they go whole: same bits."""
    inputs, prepared, kwargs, _ = one_expert_call(device, batch, seqlen, n, gelu, seed=batch * seqlen + n)
    rows = batch * math.ceil(seqlen / 32) * 32
    counts = ttnn.from_torch(
        torch.tensor([[rows]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    # The first output stays allocated and a NaN tensor of the same spec is freed where the second lands, so a tile
    # the second call never writes cannot pass on the first call's values.
    want_tensor = grouped_matmul(inputs, prepared, counts, **kwargs)
    want = ttnn.to_torch(want_tensor)
    ttnn.deallocate(grouped_matmul(inputs, prepared, None, **kwargs))
    poison = ttnn.from_torch(
        torch.full(tuple(want.shape), float("nan")), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn.deallocate(poison)
    got = ttnn.to_torch(grouped_matmul(inputs, prepared, None, **kwargs))
    assert not got.float().isnan().any(), f"{int(got.float().isnan().sum())} values never written"
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


# The whole-lane tail split, where every call splits: QKV at 32 tile rows (2 tail blocks of 7 one-tile subblocks over
# 10 lanes, 1 or 2 each), at 4 tile rows (fewer blocks than lanes), fc1's N at 32.
WHOLE_LANE_TAIL_CASES = [(2, 512, QKV, None), (1, 128, QKV, None), (2, 512, INTERMEDIATE, 2.0)]


@pytest.mark.parametrize("batch, seqlen, n, gelu", WHOLE_LANE_TAIL_CASES)
def test_whole_lane_tail_split_writes_every_row(device, batch, seqlen, n, gelu):
    """Every output tile is written (the output sits where a NaN tensor was) and matches torch."""
    inputs, prepared, kwargs, (values, weight_values, bias_values) = one_expert_call(
        device, batch, seqlen, n, gelu, seed=batch * seqlen + n + 1
    )

    def run():
        return grouped_matmul(inputs, prepared, None, **kwargs)

    ttnn.deallocate(run())
    poison = ttnn.from_torch(
        torch.full((batch, 1, seqlen, n), float("nan")), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn.deallocate(poison)
    got = ttnn.to_torch(run()).float()
    assert not got.isnan().any(), f"{int(got.isnan().sum())} values never written"

    expected = values @ weight_values + bias_values
    if gelu is not None:
        expected = torch.nn.functional.gelu(expected, approximate="tanh")
    pcc = torch.corrcoef(torch.stack([got.flatten(), expected.flatten()]))[0, 1].item()
    assert pcc >= 0.9999, f"PCC {pcc:.6f}"
