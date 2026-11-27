# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ROW_MAJOR bfloat16 -> TILE for g6_to_layout (regimes row_major_to_tile and packer_tilize).

Shapes are the production RM-input cases of eval/golden_tests/g6_to_layout/feature_spec.py plus a
few off-production shapes that cover the walk's corners. A bfloat16 output is graded bit for bit
(helpers.TOLERANCES[bfloat16] == ("exact", None)); a bfloat8_b output within one quantization step
of the host's bfloat8_b rendering (helpers.TOLERANCES[bfloat8_b] == ("block_step", 1)).
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_to_layout import g6_to_layout, select_plan

DRAM = ttnn.BufferType.DRAM
L1 = ttnn.BufferType.L1
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT

# (case id, shape, input buffer, output memory_config or None)
BF16_CASES = [
    (1989, (16, 1, 1536), DRAM, None),  # H = 1: every image is its own tile-row
    (1991, (16, 1, 512), DRAM, None),
    (2022, (8, 1, 1536), DRAM, None),
    (2024, (8, 1, 512), DRAM, None),
    (270, (1, 1, 1, 512), DRAM, None),
    (272, (1, 64, 128, 32), DRAM, None),  # one tile column
    (277, (1, 64, 128, 256), DRAM, None),
    (289, (1, 1, 1, 256), DRAM, None),
    (291, (1, 1, 1, 196), DRAM, None),  # last tile column 4 wide: left faces only
    (296, (1, 128, 256, 32), DRAM, None),
    (301, (1, 128, 256, 256), DRAM, None),
    (310, (1, 1, 1, 185), DRAM, None),  # 370-B rows
    (312, (1, 1, 1, 207), DRAM, None),
    (21, (1, 32, 64, 256), L1, ttnn.L1_MEMORY_CONFIG),
    (30, (1, 64, 128, 256), L1, ttnn.L1_MEMORY_CONFIG),
    (38, (1, 128, 256, 256), L1, ttnn.L1_MEMORY_CONFIG),
    (45, (1, 32, 128, 256), L1, None),  # the output keeps the input's L1 placement
    (49, (1, 128, 256, 128), L1, ttnn.L1_MEMORY_CONFIG),
    # Off production: H % 32 != 0 across image boundaries inside blocks, both dims ragged, a wide
    # row cut into column blocks, multi-tile-row blocks.
    ("h48", (3, 48, 100), DRAM, None),
    ("w_wide", (1, 40, 131072), DRAM, None),
    ("tall", (1, 4096, 2560), DRAM, None),
]

BF8B_CASES = [
    (16, (1, 1, 1, 256)),
    ("Q3", (1, 1, 32, 256)),
    ("w200", (1, 2, 70, 200)),  # W % 16 != 0: live groups padded with zeros
    ("tall", (1, 1024, 512)),
]

# -0.0, subnormals, +inf, -inf, NaN with payload
SPECIAL_BITS = (0x8000, 0x0001, 0x8001, 0x007F, 0x7F80, 0xFF80, 0x7FC1)
MODEL_L1_FREE_BYTES = 1347776
_FILLER_PAGE_BYTES = 2048
_GUARD_BITS = 0x5A5A


def _id(case):
    case_id, shape = case[0], case[1]
    return f"c{case_id}-{'x'.join(str(d) for d in shape)}"


def _randn(shape, seed=42):
    torch.manual_seed(seed)
    return torch.randn(shape, dtype=torch.bfloat16)


def _with_specials(shape):
    x = _randn(shape)
    flat = x.view(-1).view(torch.int16)
    table = torch.tensor([b - 0x10000 if b >= 0x8000 else b for b in SPECIAL_BITS], dtype=torch.int16)
    count = max(len(SPECIAL_BITS), flat.numel() // 100)
    positions = torch.randint(0, flat.numel(), (count,))
    flat[positions] = table[torch.randint(0, len(SPECIAL_BITS), (count,))]
    flat[: len(SPECIAL_BITS)] = table
    return x


def _group_scaled(shape):
    # randn times 2^k per group of 16 along W (k in [-8, 8]): every face row has its own exponent.
    torch.manual_seed(460)
    base = torch.randn(shape)
    k = torch.randint(-8, 9, (*shape[:-1], -(-shape[-1] // 16)))
    return (base * torch.pow(2.0, k).repeat_interleave(16, dim=-1)[..., : shape[-1]]).to(torch.bfloat16)


def _to_device(torch_input, device, buffer):
    return ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=RM,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer),
        preserve_nan_values=True,
    )


def _bits(t):
    return t.contiguous().view(torch.int16)


def _check_bf16(output, expected, buffer):
    assert list(output.shape) == list(expected.shape)
    assert output.dtype == ttnn.bfloat16
    assert output.layout == TILE
    memory = output.memory_config()
    assert memory.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert memory.buffer_type == buffer
    actual = ttnn.to_torch(output).reshape(expected.shape)
    differ = _bits(actual) != _bits(expected)
    count = int(differ.sum())
    assert (
        count == 0
    ), f"{count} of {differ.numel()} values differ bitwise, first at {tuple(differ.nonzero()[0].tolist())}"


def _host_bf8(values):
    packed = ttnn.from_torch(values, dtype=ttnn.bfloat8_b, layout=TILE)
    return ttnn.to_torch(packed).to(torch.bfloat16).reshape(values.shape)


def _group_exponent(values):
    width = values.shape[-1]
    padded = -(-width // 32) * 32
    v = values.to(torch.float32)
    if padded != width:
        v = torch.nn.functional.pad(v, (0, padded - width))
    exponent = (v.view(torch.int32) >> 23) & 0xFF
    groups = exponent.reshape(*v.shape[:-1], padded // 16, 16).amax(dim=-1)
    return groups.repeat_interleave(16, dim=-1)[..., :width]


def _check_block_step(output, torch_input):
    assert output.dtype == ttnn.bfloat8_b
    assert output.layout == TILE
    actual = ttnn.to_torch(output).reshape(torch_input.shape).to(torch.float64)
    rendering = _host_bf8(torch_input)
    step = torch.pow(2.0, (_group_exponent(rendering) - 133).to(torch.float64))
    bad = ~((actual - rendering.to(torch.float64)).abs() <= step)
    count = int(bad.sum())
    assert count == 0, f"{count} of {bad.numel()} values beyond one step, first at {tuple(bad.nonzero()[0].tolist())}"


@pytest.mark.parametrize("case", BF16_CASES, ids=_id)
def test_row_major_to_tile_bf16(device, case):
    _, shape, buffer, memory_config = case
    torch_input = _randn(shape)
    kwargs = {} if memory_config is None else {"memory_config": memory_config}
    output = g6_to_layout(_to_device(torch_input, device, buffer), TILE, **kwargs)
    _check_bf16(output, torch_input.clone(), buffer if memory_config is None else memory_config.buffer_type)


@pytest.mark.parametrize("shape", [(16, 1, 1536), (1, 1, 1, 185), (3, 48, 100)], ids=lambda s: "x".join(map(str, s)))
def test_row_major_to_tile_special_bit_patterns(device, shape):
    torch_input = _with_specials(shape)
    output = g6_to_layout(_to_device(torch_input, device, DRAM), TILE)
    _check_bf16(output, torch_input.clone(), DRAM)


@pytest.mark.parametrize("case", BF8B_CASES, ids=_id)
def test_row_major_to_tile_bf8b(device, case):
    _, shape = case
    torch_input = _group_scaled(shape)
    output = g6_to_layout(_to_device(torch_input, device, DRAM), TILE, dtype=ttnn.bfloat8_b)
    _check_block_step(output, torch_input)


@pytest.mark.parametrize("shape", [(1, 128, 256, 256), (1, 128, 256, 128)], ids=lambda s: "x".join(map(str, s)))
def test_row_major_to_tile_l1_output_within_model_budget(device, shape):
    # Cases 38/49: the L1 input lives outside the budget; the filler leaves exactly the model's free
    # bytes per bank, and guard pages next to them must survive the call.
    torch_input = _randn(shape)
    ttnn_input = _to_device(torch_input, device, L1)
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    banks = view.num_banks
    filler_pages = (view.largest_contiguous_bytes_free_per_bank - MODEL_L1_FREE_BYTES - _FILLER_PAGE_BYTES) // (
        _FILLER_PAGE_BYTES
    )
    held = []
    if filler_pages > 0:
        held.append(
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([filler_pages * banks, _FILLER_PAGE_BYTES // 2]),
                ttnn.bfloat16,
                RM,
                device,
                ttnn.L1_MEMORY_CONFIG,
            )
        )
    pattern = torch.full((banks, _FILLER_PAGE_BYTES // 2), _GUARD_BITS, dtype=torch.int16).view(torch.bfloat16)
    guard = ttnn.from_torch(pattern, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    output = g6_to_layout(ttnn_input, TILE, memory_config=ttnn.L1_MEMORY_CONFIG)

    assert int((_bits(ttnn.to_torch(guard)) != _GUARD_BITS).sum()) == 0, "guard pages beside the op were overwritten"
    _check_bf16(output, torch_input.clone(), L1)
    del held


@pytest.mark.parametrize(
    "shape,dtype,regime",
    [
        ((16, 1, 1536), None, "row_major_to_tile"),
        ((1, 128, 256, 256), ttnn.bfloat16, "row_major_to_tile"),
        ((1, 1, 1, 256), ttnn.bfloat8_b, "packer_tilize"),
    ],
    ids=["c1989", "c301", "c16"],
)
def test_row_major_to_tile_regime_selection(device, shape, dtype, regime):
    ttnn_input = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, RM, device, ttnn.DRAM_MEMORY_CONFIG
    )
    kwargs = {} if dtype is None else {"dtype": dtype}
    plan = select_plan(ttnn_input, TILE, **kwargs)
    assert plan["regime"] == regime
    assert plan["num_cores"] == len(plan["cores"]) > 1
