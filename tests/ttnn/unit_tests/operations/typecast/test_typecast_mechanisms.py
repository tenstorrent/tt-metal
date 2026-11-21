# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mechanism-pinned tests of typecast (op_design.md, Work Distribution: "Tests the plan needs" and the
regime-pinned table). The device comes from the conftest's module-scoped fixture."""

import pytest
import torch

import ttnn
from ttnn.operations.typecast import typecast
from ttnn.operations.typecast import typecast_program_artifacts as artifacts

TORCH = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
PAIRS = [(a, b) for a in (ttnn.bfloat16, ttnn.float32) for b in (ttnn.bfloat16, ttnn.float32)]
PAIR_IDS = [f"{a.name}-{b.name}" for a, b in PAIRS]


def _to_device(t, dtype, device):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _bits(t):
    """Bit patterns, so a comparison is exact (and NaN-safe)."""
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


def _assert_bit_exact(got, expected):
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    mismatched = (_bits(got) != _bits(expected)).sum().item()
    assert mismatched == 0, f"{mismatched} of {expected.numel()} elements differ"


def _run(device, torch_input, in_dtype, out_dtype, **knobs):
    """The op's program with its knobs turned (`threads`, `buffer_depth`, `block_tiles`)."""
    x = _to_device(torch_input, in_dtype, device)
    out = ttnn.allocate_tensor_on_device(ttnn.TensorSpec(x.shape, out_dtype, ttnn.TILE_LAYOUT), device)
    spec, run_args, idx = artifacts.create_program_artifacts(x, out, **knobs)
    return ttnn.to_torch(ttnn.generic_op([x, out], spec, run_args, idx))


# ---------------------------------------------------------------------------
# Thread plan
# ---------------------------------------------------------------------------


def test_thread_plan_on_the_program(device):
    """The kernels' thread counts are the plan's: read off the ProgramSpec the op builds."""
    x = _to_device(torch.randn(1, 1, 64, 64), ttnn.bfloat16, device)
    out = ttnn.allocate_tensor_on_device(ttnn.TensorSpec(x.shape, ttnn.float32, ttnn.TILE_LAYOUT), device)
    spec, _, _ = artifacts.create_program_artifacts(x, out)
    threads = {k.unique_id: k.num_threads for k in spec.kernels}
    assert (threads["reader"], threads["compute"], threads["writer"]) == artifacts.THREAD_PLAN


@pytest.mark.parametrize("threads", [(1, 2, 1), (1, 4, 1), (2, 2, 2), (4, 4, 2)], ids=lambda p: "R%dC%dW%d" % p)
@pytest.mark.parametrize("pair", PAIRS, ids=PAIR_IDS)
def test_other_thread_plans(device, threads, pair):
    """The plan is a live knob: other legal plans give the same bits (93 tiles: 2 or 3 per cluster, so the
    round-robin dealing stops anywhere)."""
    in_dtype, out_dtype = pair
    torch.manual_seed(1)
    t = torch.randn(1, 1, 96, 992, dtype=TORCH[in_dtype])
    _assert_bit_exact(_run(device, t, in_dtype, out_dtype, threads=threads), t.to(TORCH[out_dtype]))


# ---------------------------------------------------------------------------
# Numerics of the two unpack paths and the packer's narrowing
# ---------------------------------------------------------------------------


def test_fp32_to_bf16_rounds_to_nearest_even(device):
    """Ties (low 16 bits exactly 0x8000) with even and odd bf16 LSB, and their 0x7FFF / 0x8001 neighbours:
    bit-identical to torch's round-to-nearest-even."""
    torch.manual_seed(2)
    hi = torch.randint(0x3F80, 0x4100, (32 * 64,), dtype=torch.int32)  # 1.0 .. ~8.0, both LSB parities
    hi = torch.where(torch.rand(hi.shape) < 0.5, hi, hi | 0x8000)  # and negatives
    lows = torch.tensor([0x8000, 0x7FFF, 0x8001, 0x0000, 0xFFFF], dtype=torch.int32)
    low = lows[torch.arange(hi.numel()) % lows.numel()]
    t = ((hi << 16) | low).view(torch.float32).reshape(1, 1, 32, 64)
    _assert_bit_exact(ttnn.to_torch(typecast(_to_device(t, ttnn.float32, device), ttnn.bfloat16)), t.to(torch.bfloat16))


def test_fp32_to_fp32_keeps_every_bit(device):
    """Values with nonzero low 13 mantissa bits survive: the fp32 input is unpacked to DEST, not through SrcA."""
    torch.manual_seed(3)
    t = torch.randn(1, 1, 64, 128, dtype=torch.float32)
    t = (t.view(torch.int32) | 0x1FFF).view(torch.float32)
    _assert_bit_exact(ttnn.to_torch(typecast(_to_device(t, ttnn.float32, device), ttnn.float32)), t)


def test_bf16_to_fp32_is_exact(device):
    torch.manual_seed(4)
    t = torch.randn(1, 2, 64, 96, dtype=torch.bfloat16) * 1000
    _assert_bit_exact(ttnn.to_torch(typecast(_to_device(t, ttnn.bfloat16, device), ttnn.float32)), t.float())


# ---------------------------------------------------------------------------
# Blocking: several blocks per NEO, ragged last block, ring wrap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 1, 96, 992), (1, 1, 2048, 256)], ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("pair", PAIRS, ids=PAIR_IDS)
def test_multi_block(device, shape, pair):
    """93 tiles (3 per cluster: block 2, ragged last block) and 512 tiles (16 per cluster: four blocks of 4)."""
    in_dtype, out_dtype = pair
    torch.manual_seed(5)
    t = torch.randn(*shape, dtype=TORCH[in_dtype])
    got = ttnn.to_torch(typecast(_to_device(t, in_dtype, device), out_dtype))
    _assert_bit_exact(got, t.to(TORCH[out_dtype]))


@pytest.mark.parametrize("block_tiles,buffer_depth", [(1, 2), (3, 2), (3, 1), (4, 3), (4, 1), (2, 3)])
def test_block_knobs(device, block_tiles, buffer_depth):
    """Turned block / depth knobs (16 tiles per cluster): ragged tails (block 3) and rings that wrap."""
    torch.manual_seed(6)
    t = torch.randn(1, 1, 2048, 256, dtype=torch.float32)
    got = _run(device, t, ttnn.float32, ttnn.bfloat16, block_tiles=block_tiles, buffer_depth=buffer_depth)
    _assert_bit_exact(got, t.to(torch.bfloat16))


def test_block_tiles_rule():
    """The derived block: coarsest that keeps BUFFER_DEPTH blocks in flight, clamped by one DEST section."""
    d = artifacts.BUFFER_DEPTH
    assert artifacts.dest_tiles() == 4  # 32-bit DEST, half sync
    assert artifacts.choose_block_tiles(1, 2048, 4096, 1, d) == 1
    assert artifacts.choose_block_tiles(6, 2048, 4096, 1, d) == 3
    assert artifacts.choose_block_tiles(10, 2048, 4096, 1, d) == 4  # would be 5: clamped to DEST
    assert artifacts.choose_block_tiles(16, 2048, 4096, 4, d) == 2
    assert artifacts.choose_block_tiles(10_000, 4096, 4096, 1, d) == artifacts.dest_tiles()


@pytest.mark.parametrize("pair", PAIRS, ids=PAIR_IDS)
def test_ten_tiles_per_cluster(device, pair):
    """320 tiles: 10 per cluster, where the unclamped rule would pick a block of 5 (ring 10)."""
    in_dtype, out_dtype = pair
    torch.manual_seed(8)
    t = torch.randn(1, 1, 320, 1024, dtype=TORCH[in_dtype])
    _assert_bit_exact(ttnn.to_torch(typecast(_to_device(t, in_dtype, device), out_dtype)), t.to(TORCH[out_dtype]))


# ---------------------------------------------------------------------------
# One device session: after another program, before a tt-metal program
# ---------------------------------------------------------------------------


def test_device_session_order(device):
    """Another program first (a typecast of the other pair), then this op, then tt-metal's own untilize
    (`to_layout` to row-major, untagged NoC transfers): catches a sticky NoC tag left behind."""
    torch.manual_seed(7)
    a = torch.randn(1, 4, 32, 64, dtype=torch.bfloat16)
    b = torch.randn(1, 1, 128, 128, dtype=torch.float32)
    first = typecast(_to_device(b, ttnn.float32, device), ttnn.bfloat16)
    second = typecast(_to_device(a, ttnn.bfloat16, device), ttnn.float32)
    rm = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(second, ttnn.ROW_MAJOR_LAYOUT)))
    _assert_bit_exact(rm, a.float())
    _assert_bit_exact(ttnn.to_torch(first), b.to(torch.bfloat16))
    # and the op once more after tt-metal's program
    third = typecast(_to_device(b, ttnn.float32, device), ttnn.float32)
    _assert_bit_exact(ttnn.to_torch(third), b)


# ---------------------------------------------------------------------------
# Placement and refusals
# ---------------------------------------------------------------------------


def test_output_in_input_buffer(device):
    x = _to_device(torch.randn(1, 1, 32, 64), ttnn.bfloat16, device)
    out = typecast(x, dtype=ttnn.float32)
    mc = out.memory_config()
    assert mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert out.dtype == ttnn.float32 and out.layout == ttnn.TILE_LAYOUT


def test_refusals(device):
    x = _to_device(torch.randn(1, 1, 32, 64), ttnn.float32, device)
    with pytest.raises(TypeError):
        typecast(x, "bfloat16")
    with pytest.raises(NotImplementedError):
        typecast(x, ttnn.int32)
    with pytest.raises(NotImplementedError):  # rank 3
        typecast(_to_device(torch.randn(1, 32, 64), ttnn.float32, device), ttnn.bfloat16)
    with pytest.raises(NotImplementedError):  # L1 input: not in the first phase
        typecast(ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG), ttnn.bfloat16)


def test_sharded_input_refused_by_name(device):
    """R7: a sharded input is refused through the `sharding` axis (memory_axes), before any kernel work."""
    x = _to_device(torch.randn(1, 1, 64, 64), ttnn.float32, device)
    shard = ttnn.create_sharded_memory_config(
        (32, 64), ttnn.CoreGrid(x=2, y=1), ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
    )
    sharded = ttnn.to_memory_config(x, shard)
    with pytest.raises(NotImplementedError):
        typecast(sharded, ttnn.bfloat16)
    # Once L1 is supported (a refinement), the refusal must come from the `sharding` axis by name.
    from ttnn.operations.typecast import SUPPORTED as supported  # the op's own dict object

    saved = supported["memory_config"]
    supported["memory_config"] = sorted(set(saved) | {"L1"})
    try:
        with pytest.raises(NotImplementedError, match="sharding='height'"):
            typecast(sharded, ttnn.bfloat16)
    finally:
        supported["memory_config"] = saved
