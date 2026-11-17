# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 27: the packed input form (z, b, a = None: mixed_qkv rows hold [x | z | a | b]) and per-token inputs in
L1. Both must return, to the bit, what the split DRAM call returns on the same data: on every kernel path, with the
tile padding of the packed rows poisoned, beside a low L1 input (the static buffers shrink), under a trace."""

import pytest
import torch

import ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from ttnn.operations.gated_delta_net import gated_delta_net_program_descriptor as gdn_pd
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs

TOKEN_INPUTS = ("mixed_qkv", "z", "b", "a")
NAMES = ("output", "conv_state_out", "recurrent_state_out")
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
CHIP = (12, 128, 128)


def chip(rows, batch=1):
    return ((1, rows, 2560), (batch, *CHIP))


def pack(tensors):
    """The model's projection row: x, z, then a BEFORE b."""
    return torch.cat([tensors[name] for name in ("mixed_qkv", "z", "a", "b")], dim=-1)


def upload(tensor, device, dtype, memory_config=DRAM, poison=None):
    t = ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    return t if poison is None else ttnn.fill_implicit_tile_padding(t, poison)


def device_args(tensors, device, token_dtype, *, packed=False, token_memory=DRAM, poison=None):
    """The ten arguments in signature order; packed: the packed rows, then None for z, b, a."""
    args = []
    for name in ORDER:
        if name == "mixed_qkv" and packed:
            args.append(upload(pack(tensors), device, token_dtype, token_memory, poison))
        elif name in TOKEN_INPUTS:
            args.append(None if packed else upload(tensors[name], device, token_dtype, token_memory))
        else:
            args.append(upload(tensors[name], device, ttnn.float32 if name == "recurrent_state" else token_dtype))
    return args


def run(args, **kwargs):
    kwargs.setdefault("memory_config", DRAM)
    returned = gated_delta_net(*args, compute_kernel_config=compute_config(), **kwargs)
    return [ttnn.to_torch(t) for t in returned]


def assert_same(got, want, label):
    for name, g, w in zip(NAMES, got, want):
        assert torch.equal(g, w), f"{label}: {name} differs from the split DRAM call"


SHAPES = [
    pytest.param(chip(1), ttnn.bfloat16, id="decode_b1"),
    pytest.param(chip(8, 8), ttnn.bfloat16, id="decode_b8"),
    pytest.param(chip(16, 16), ttnn.bfloat16, id="decode_b16"),
    pytest.param(chip(32, 32), ttnn.bfloat16, id="decode_b32"),
    pytest.param(chip(40, 40), ttnn.bfloat16, id="decode_b40"),
    pytest.param(chip(1), ttnn.float32, id="decode_b1_f32"),
    pytest.param(chip(33), ttnn.bfloat16, id="prefill_t33"),
    pytest.param(chip(128), ttnn.bfloat16, id="prefill_t128"),
    pytest.param(chip(200), ttnn.float32, id="prefill_t200_f32"),
    pytest.param(chip(2048), ttnn.bfloat16, id="prefill_t2048"),
    pytest.param(chip(4096), ttnn.bfloat16, id="prefill_t4096"),
    pytest.param(chip(80, 2), ttnn.bfloat16, id="batched_b2_t40"),
    pytest.param(((1, 3, 256), (3, 3, 32, 64)), ttnn.float32, id="small_decode_b3_f32"),
    pytest.param(((1, 100, 256), (1, 3, 32, 64)), ttnn.bfloat16, id="small_prefill_t100"),
    pytest.param(((1, 64, 4096), (1, 16, 128, 128)), ttnn.bfloat16, id="wide_hv16_t64"),
]


@pytest.mark.parametrize("inputs, token_dtype", SHAPES)
def test_packed_and_l1_match_split(device, inputs, token_dtype):
    torch.manual_seed(7)
    tensors = make_inputs(inputs)
    want = run(device_args(tensors, device, token_dtype))
    forms = {
        "packed": dict(packed=True),
        "packed, inf in the tile padding": dict(packed=True, poison=float("inf")),
        "split in L1": dict(token_memory=L1),
        "packed in L1": dict(packed=True, token_memory=L1),
        "packed in L1, inf in the tile padding": dict(packed=True, token_memory=L1, poison=float("inf")),
    }
    for label, form in forms.items():
        assert_same(run(device_args(tensors, device, token_dtype, **form)), want, label)


@pytest.mark.parametrize("inputs", [chip(8, 8), chip(2048)], ids=["decode_b8", "prefill_t2048"])
def test_l1_output(device, inputs):
    """The output follows mixed_qkv's memory config unless memory_config says otherwise."""
    torch.manual_seed(8)
    tensors = make_inputs(inputs)
    want = run(device_args(tensors, device, ttnn.bfloat16))
    args = device_args(tensors, device, ttnn.bfloat16, packed=True, token_memory=L1)
    returned = gated_delta_net(*args, compute_kernel_config=compute_config())
    assert returned[0].memory_config().buffer_type == ttnn.BufferType.L1
    assert_same([ttnn.to_torch(t) for t in returned], want, "packed in L1, output in L1")


@pytest.mark.parametrize(
    "inputs, low_kb",
    [
        pytest.param(chip(1), 600, id="decode_b1"),
        pytest.param(chip(16, 16), 600, id="decode_b16"),
        pytest.param(chip(32, 32), 600, id="decode_b32"),
        pytest.param(chip(128), 600, id="prefill_t128"),
        pytest.param(chip(2048), 300, id="prefill_t2048"),
        pytest.param(chip(2048), 600, id="prefill_t2048_low"),
        pytest.param(chip(4096), 300, id="prefill_t4096"),
    ],
)
def test_static_buffers_shrink_below_l1_input(device, monkeypatch, inputs, low_kb):
    """The packed rows allocated under low_kb KB of other L1 data per core, which is then freed: the input's address
    is far below L1_CB_REGION_END and the program's static buffers must end below it."""
    # The two calls run different plans and are compared to the bit: the state cores' norm of the last rows
    # (STATE_NORM_TAIL "auto") and the one-window gate factor of norm-split plans (PREP_ZW_FPU "auto") would be in
    # one of them only.
    monkeypatch.setattr(gdn_pd, "STATE_NORM_TAIL", 0)
    monkeypatch.setattr(gdn_pd, "PREP_ZW_FPU", False)
    torch.manual_seed(9)
    tensors = make_inputs(inputs)
    want = run(device_args(tensors, device, ttnn.bfloat16))
    grid = device.compute_with_storage_grid_size()
    pad_rows = low_kb * 1024 * grid.x * grid.y // (2 * 1024)
    pad = upload(torch.zeros(1, pad_rows, 1024), device, ttnn.bfloat16, L1)
    args = device_args(tensors, device, ttnn.bfloat16, packed=True, token_memory=L1)
    ttnn.deallocate(pad)
    l1_end = gdn_pd.l1_region_end([args[0]])
    assert l1_end is not None and l1_end < gdn_pd.L1_CB_REGION_END - 100_000
    assert_same(run(args), want, f"packed in L1 at {args[0].buffer_address()}")


def test_l1_bound_no_plan_meets(device, expect_error):
    """An L1 input so low that no plan fits is refused on the host, naming the bound."""
    tensors = make_inputs(chip(2048))
    grid = device.compute_with_storage_grid_size()
    pad = upload(torch.zeros(1, 1100 * grid.x * grid.y // 2, 1024), device, ttnn.bfloat16, L1)
    args = device_args(tensors, device, ttnn.bfloat16, packed=True, token_memory=L1)
    ttnn.deallocate(pad)
    with expect_error(RuntimeError, r"cores need \d+ B of static circular buffers.*below the call's L1 tensors"):
        gated_delta_net(*args, compute_kernel_config=compute_config(), memory_config=DRAM)


def test_refusals(device, expect_error):
    torch.manual_seed(0)
    tensors = make_inputs(chip(1))
    split = device_args(tensors, device, ttnn.bfloat16)
    packed = device_args(tensors, device, ttnn.bfloat16, packed=True)
    config = compute_config()
    for keep in ((1,), (2,), (3,), (1, 2), (1, 3), (2, 3)):
        args = [packed[0]] + [split[i] if i in keep else None for i in (1, 2, 3)] + split[4:]
        with expect_error(ValueError, "pass z, b and a as tensors .* or all three as None"):
            gated_delta_net(*args, compute_kernel_config=config)
    # Split-width rows without z, b, a; packed-width rows with them.
    with expect_error(ValueError, "gated_delta_net"):
        gated_delta_net(split[0], None, None, None, *split[4:], compute_kernel_config=config)
    with expect_error(ValueError, "gated_delta_net"):
        gated_delta_net(packed[0], *split[1:], compute_kernel_config=config)
    # Hv = 33: a and b do not share a tile column.
    wide = make_inputs(((1, 5, 1760), (1, 33, 32, 32)))
    with expect_error(ValueError, "packed form needs Hv <= 16"):
        gated_delta_net(*device_args(wide, device, ttnn.float32, packed=True), compute_kernel_config=config)


def test_cache_keys_do_not_collide(device):
    """Split, packed, L1 and DRAM calls of one shape alternate on a warm program cache."""
    torch.manual_seed(3)
    tensors = make_inputs(chip(8, 8))
    forms = [dict(), dict(packed=True), dict(token_memory=L1), dict(packed=True, token_memory=L1)]
    gdn_pd.clear_program_cache()
    want = run(device_args(tensors, device, ttnn.bfloat16))
    for rnd in range(3):
        for form in forms:
            assert_same(run(device_args(tensors, device, ttnn.bfloat16, **form)), want, f"round {rnd} {form}")
    assert gdn_pd.program_cache_size() == len(forms)


@pytest.mark.parametrize("batch", [1, 16], ids=["decode_b1", "decode_b16"])
def test_packed_width_sharded(device, batch):
    """The packed rows as a decode projection leaves them: one tile row, width sharded in L1 over 12 cores, the
    logical rows and columns a view of the padded shards."""
    torch.manual_seed(12)
    tensors = make_inputs(chip(batch, batch))
    want = run(device_args(tensors, device, ttnn.bfloat16))
    args = device_args(tensors, device, ttnn.bfloat16, packed=True)
    rows = pack(tensors)
    width, cores = rows.shape[-1], 12
    shard_w = -(-width // 32 // cores) * 32
    padded = torch.zeros(1, 1, 32, cores * shard_w)
    padded[0, 0, :batch, :width] = rows[0]
    grid = ttnn.num_cores_to_corerangeset(cores, device.compute_with_storage_grid_size(), row_wise=True)
    memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )
    sharded = ttnn.to_memory_config(upload(padded, device, ttnn.bfloat16), memory)
    args[0] = ttnn.reshape(sharded, [1, batch, width], [1, 32, cores * shard_w])
    assert_same(run(args), want, "packed, width sharded in L1")
