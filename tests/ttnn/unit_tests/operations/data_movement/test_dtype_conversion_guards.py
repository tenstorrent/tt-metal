# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Datacopy ops (clone/copy/tilize/sharding/to_layout/to_memory_config) and matmul must reject
dtype= pairs they cannot convert through the unpack/pack packer, and must agree bit-exactly
with ttnn.typecast on the pairs they accept."""

import pytest
import torch
import ttnn


DT = {
    "float32": ttnn.float32,
    "bfloat16": ttnn.bfloat16,
    "int32": ttnn.int32,
    "uint32": ttnn.uint32,
    "uint16": ttnn.uint16,
    "uint8": ttnn.uint8,
}

TORCH_DT = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "uint32": torch.int64,
    "uint16": torch.int32,
    "uint8": torch.uint8,
}

# Pairs the datacopy kernel reinterprets as bits instead of converting. All must raise.
BAD_PAIRS = [
    ("float32", "int32"),
    ("float32", "uint32"),
    ("float32", "uint16"),
    ("bfloat16", "int32"),
    ("bfloat16", "uint32"),
    ("bfloat16", "uint16"),
    ("int32", "float32"),
    ("int32", "bfloat16"),
    ("uint32", "float32"),
    ("uint16", "float32"),
    ("uint16", "bfloat16"),
    ("int32", "uint16"),
    ("uint32", "uint16"),
    ("uint16", "int32"),
    ("uint16", "uint32"),
    # uint8 pairs not in the allowlist. uint8->uint16 is plausibly safe (uint8 is zero-extended
    # into a 32-bit Dest lane on BH), but unverified on silicon; keep as a rejection test so any
    # future widening of the allowlist is paired with an explicit on-device measurement.
    ("uint8", "uint16"),
    ("uint8", "bfloat16"),
    ("uint8", "float32"),
    ("int32", "uint8"),
    ("uint32", "uint8"),
    ("uint16", "uint8"),
    ("float32", "uint8"),
    ("bfloat16", "uint8"),
]

# Pairs the datacopy kernel preserves values on (same dtype, float<->float, INT32<->UINT32,
# UINT8->{INT32,UINT32}). Every op accepting dtype= must agree with ttnn.typecast on these.
SAFE_PAIRS = [
    ("float32", "bfloat16"),
    ("bfloat16", "float32"),
    ("int32", "uint32"),
    ("uint32", "int32"),
    ("uint8", "int32"),
    ("uint8", "uint32"),
]

GUARD_MSG = "datacopy cannot convert"
MATMUL_MSG = "integer output dtype"


def _mk(device, shape, dt):
    # Small non-negative integers so every value is exact in every dtype in the table.
    x = torch.randint(0, 100, shape).to(TORCH_DT[dt])
    return x, ttnn.from_torch(x, dtype=DT[dt], layout=ttnn.TILE_LAYOUT, device=device)


def _mk_signed_int32(device, shape, layout=ttnn.TILE_LAYOUT, memory_config=None):
    # int32 values that cover both halves of the range plus the sign boundary; exercises
    # that datacopy's int32<->uint32 bit reinterpret matches ttnn.typecast for negatives.
    x = torch.randint(-(2**31), 2**31 - 1, shape).to(torch.int32)
    kwargs = {"dtype": ttnn.int32, "layout": layout, "device": device}
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
    return x, ttnn.from_torch(x, **kwargs)


# ---------------------------------------------------------------------------
# Rejection tests. Match the new helper's message (not just "typecast") so a
# stray TT_FATAL that happens to mention typecast cannot mask the real guard.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("src,dst", BAD_PAIRS)
def test_clone_rejects(device, expect_error, src, dst):
    _, t = _mk(device, (1, 1, 32, 32), src)
    with expect_error(RuntimeError, GUARD_MSG):
        ttnn.clone(t, dtype=DT[dst])


@pytest.mark.parametrize("src,dst", BAD_PAIRS)
def test_copy_rejects(device, expect_error, src, dst):
    _, t = _mk(device, (1, 1, 32, 32), src)
    out = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32).to(TORCH_DT[dst]), dtype=DT[dst], layout=ttnn.TILE_LAYOUT, device=device
    )
    with expect_error(RuntimeError, GUARD_MSG):
        ttnn.copy(t, out)


@pytest.mark.parametrize("src,dst", BAD_PAIRS)
def test_tilize_rejects(device, expect_error, src, dst):
    x = torch.zeros(1, 1, 32, 32).to(TORCH_DT[src])
    t = ttnn.from_torch(x, dtype=DT[src], layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, GUARD_MSG):
        ttnn.tilize(t, dtype=DT[dst])


# to_layout(rm -> TILE, dtype=) on non-tile-aligned shapes routes through
# tilize_with_val_padding (to_layout_op.cpp). The 30x30 shape hits that path.
# tilize_with_val_padding rejects uint8 inputs upstream of our guard, so skip uint8 src.
@pytest.mark.parametrize("src,dst", [p for p in BAD_PAIRS if p[0] != "uint8"])
def test_to_layout_rejects(device, expect_error, src, dst):
    x = torch.zeros(1, 1, 30, 30).to(TORCH_DT[src])
    t = ttnn.from_torch(x, dtype=DT[src], layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, GUARD_MSG):
        ttnn.to_layout(t, ttnn.TILE_LAYOUT, dtype=DT[dst])


# to_memory_config(x, mc, dtype=) dispatches to prim::copy / sharded ops for the
# dtype-change path; the guard fires in whichever underlying op lands. The target mem
# config must differ from the input's so the early-return at to_memory_config_op.cpp:263
# doesn't bypass the dtype change.
@pytest.mark.parametrize("src,dst", BAD_PAIRS)
def test_to_memory_config_rejects(device, expect_error, src, dst):
    x = torch.randint(0, 100, (1, 1, 32, 32)).to(TORCH_DT[src])
    t = ttnn.from_torch(x, dtype=DT[src], layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    with expect_error(RuntimeError, GUARD_MSG):
        ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG, dtype=DT[dst])


@pytest.mark.parametrize("src,dst", BAD_PAIRS)
def test_interleaved_to_sharded_rejects(device, expect_error, src, dst):
    _, t = _mk(device, (1, 1, 64, 64), src)
    shard_mem = ttnn.create_sharded_memory_config(
        shape=(64, 64),
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    with expect_error(RuntimeError, GUARD_MSG):
        ttnn.interleaved_to_sharded(t, shard_mem, output_dtype=DT[dst])


@pytest.mark.parametrize("src,dst", BAD_PAIRS)
def test_sharded_to_interleaved_rejects(device, expect_error, src, dst):
    x, _ = _mk(device, (1, 1, 64, 64), src)
    shard_mem = ttnn.create_sharded_memory_config(
        shape=(64, 64),
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded = ttnn.from_torch(x, dtype=DT[src], layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard_mem)
    with expect_error(RuntimeError, GUARD_MSG):
        ttnn.sharded_to_interleaved(sharded, ttnn.DRAM_MEMORY_CONFIG, output_dtype=DT[dst])


# Partial sharding ops accept a dtype change on the same datacopy path and need the same reinterpret rejection.
@pytest.mark.parametrize("src,dst", [p for p in BAD_PAIRS if p[0] != "uint8"])
def test_interleaved_to_sharded_partial_rejects(device, expect_error, src, dst):
    _, t = _mk(device, (1, 1, 32, 32), src)
    with expect_error(RuntimeError, GUARD_MSG):
        ttnn.interleaved_to_sharded_partial(
            t,
            (1, 1),
            [32, 32],
            1,
            0,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
            output_dtype=DT[dst],
        )


@pytest.mark.parametrize("src,dst", [p for p in BAD_PAIRS if p[0] != "uint8"])
def test_sharded_to_interleaved_partial_rejects(device, expect_error, src, dst):
    x, _ = _mk(device, (1, 1, 32, 32), src)
    shard_mem = ttnn.create_sharded_memory_config(
        shape=(32, 32),
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded = ttnn.from_torch(x, dtype=DT[src], layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard_mem)
    cache = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32).to(TORCH_DT[dst]),
        dtype=DT[dst],
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    with expect_error(RuntimeError, GUARD_MSG):
        ttnn.sharded_to_interleaved_partial(
            sharded, cache, 1, 0, memory_config=ttnn.L1_MEMORY_CONFIG, output_dtype=DT[dst]
        )


# ---------------------------------------------------------------------------
# Safe-pair tests: the acceptance criterion is that no dtype pair accepted by
# these ops returns a value different from ttnn.typecast.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("src,dst", SAFE_PAIRS)
def test_clone_safe_pairs_match_typecast(device, src, dst):
    _, t = _mk(device, (1, 1, 32, 32), src)
    cloned = ttnn.to_torch(ttnn.clone(t, dtype=DT[dst])).to(torch.float64)
    oracle = ttnn.to_torch(ttnn.typecast(t, DT[dst])).to(torch.float64)
    assert torch.equal(cloned, oracle), f"clone {src}->{dst} drifted from typecast"


@pytest.mark.parametrize("src,dst", SAFE_PAIRS)
def test_copy_safe_pairs_match_typecast(device, src, dst):
    _, t = _mk(device, (1, 1, 32, 32), src)
    out_torch = torch.zeros(1, 1, 32, 32).to(TORCH_DT[dst])
    out = ttnn.from_torch(out_torch, dtype=DT[dst], layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.copy(t, out)
    got = ttnn.to_torch(out).to(torch.float64)
    oracle = ttnn.to_torch(ttnn.typecast(t, DT[dst])).to(torch.float64)
    assert torch.equal(got, oracle), f"copy {src}->{dst} drifted from typecast"


# tilize and the sharded ops have their own unpack/pack paths (e.g. uint8 widening); running
# the same allowlist through each surfaces a pair that only one of them actually handles.


@pytest.mark.parametrize("src,dst", SAFE_PAIRS)
def test_tilize_safe_pairs_match_typecast(device, src, dst):
    x, _ = _mk(device, (1, 1, 32, 32), src)
    rm = ttnn.from_torch(x, dtype=DT[src], layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    got = ttnn.to_torch(ttnn.tilize(rm, dtype=DT[dst])).to(torch.float64)
    tiled = ttnn.from_torch(x, dtype=DT[src], layout=ttnn.TILE_LAYOUT, device=device)
    oracle = ttnn.to_torch(ttnn.typecast(tiled, DT[dst])).to(torch.float64)
    assert torch.equal(got, oracle), f"tilize {src}->{dst} drifted from typecast"


# Negative int32 inputs: datacopy's int32<->uint32 path is a bit reinterpret, so -1 must round-trip
# as 0xFFFFFFFF both ways. If ttnn.typecast clamps or saturates the sign, that pair has to leave
# SAFE_PAIRS; this test is the canary.
@pytest.mark.parametrize("op", [ttnn.clone, ttnn.tilize])
def test_int32_to_uint32_preserves_negative_values(device, op):
    x, t = _mk_signed_int32(device, (1, 1, 32, 32))
    if op is ttnn.tilize:
        t = ttnn.from_torch(x, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    got = ttnn.to_torch(op(t, dtype=ttnn.uint32)).to(torch.int64) & 0xFFFFFFFF
    tiled = ttnn.from_torch(x, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    oracle = ttnn.to_torch(ttnn.typecast(tiled, ttnn.uint32)).to(torch.int64) & 0xFFFFFFFF
    assert torch.equal(got, oracle), f"{op.__name__} int32->uint32 negative-value mismatch vs typecast"


def test_uint32_to_int32_preserves_high_bit(device):
    # Sign-bit set via uint32 values >= 2^31; datacopy should keep the bit pattern, so typecast
    # must agree on the two's complement interpretation.
    x = torch.randint(2**31, 2**32 - 1, (1, 1, 32, 32), dtype=torch.int64).to(torch.int32)
    t = ttnn.from_torch(x, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(ttnn.clone(t, dtype=ttnn.int32)).to(torch.int64)
    oracle = ttnn.to_torch(ttnn.typecast(t, ttnn.int32)).to(torch.int64)
    assert torch.equal(got, oracle), "clone uint32->int32 high-bit mismatch vs typecast"


@pytest.mark.parametrize("src,dst", SAFE_PAIRS)
def test_interleaved_to_sharded_safe_pairs_match_typecast(device, src, dst):
    _, t = _mk(device, (1, 1, 64, 64), src)
    shard_mem = ttnn.create_sharded_memory_config(
        shape=(64, 64),
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    got = ttnn.to_torch(ttnn.interleaved_to_sharded(t, shard_mem, output_dtype=DT[dst])).to(torch.float64)
    oracle = ttnn.to_torch(ttnn.typecast(t, DT[dst])).to(torch.float64)
    assert torch.equal(got, oracle), f"interleaved_to_sharded {src}->{dst} drifted from typecast"


@pytest.mark.parametrize("src,dst", SAFE_PAIRS)
def test_sharded_to_interleaved_safe_pairs_match_typecast(device, src, dst):
    x, _ = _mk(device, (1, 1, 64, 64), src)
    shard_mem = ttnn.create_sharded_memory_config(
        shape=(64, 64),
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded = ttnn.from_torch(x, dtype=DT[src], layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard_mem)
    got = ttnn.to_torch(ttnn.sharded_to_interleaved(sharded, ttnn.DRAM_MEMORY_CONFIG, output_dtype=DT[dst])).to(
        torch.float64
    )
    tiled = ttnn.from_torch(x, dtype=DT[src], layout=ttnn.TILE_LAYOUT, device=device)
    oracle = ttnn.to_torch(ttnn.typecast(tiled, DT[dst])).to(torch.float64)
    assert torch.equal(got, oracle), f"sharded_to_interleaved {src}->{dst} drifted from typecast"


# ---------------------------------------------------------------------------
# Matmul: integer output dtype is reinterpreted float bits. Reject with the
# matmul-specific message; float output stays bit-exact for integer-valued
# inputs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dst", ["int32", "uint32", "uint16"])
def test_matmul_rejects_integer_output(device, expect_error, dst):
    a = torch.randint(0, 2, (1, 1, 32, 64)).float()
    w = torch.randint(0, 2, (1, 1, 64, 32)).float()
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, MATMUL_MSG):
        ttnn.matmul(ta, tw, dtype=DT[dst])


def test_matmul_float_output_bit_exact(device):
    a = torch.randint(0, 2, (1, 1, 32, 64)).float()
    w = torch.randint(0, 2, (1, 1, 64, 32)).float()
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.to_torch(ttnn.matmul(ta, tw, dtype=ttnn.float32)).double()
    ref = (a @ w).double()
    assert torch.equal(y, ref)


# ---------------------------------------------------------------------------
# experimental.typecast must perform a value conversion (SFPU typecast), not
# a bit reinterpretation through the datacopy packer.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src,dst",
    [("int32", "float32"), ("float32", "int32"), ("uint16", "float32"), ("bfloat16", "int32")],
)
def test_experimental_typecast_converts_values(device, src, dst):
    x, t = _mk(device, (1, 1, 32, 32), src)
    got = ttnn.to_torch(ttnn.experimental.typecast(t, DT[dst])).to(torch.float64)
    oracle = ttnn.to_torch(ttnn.typecast(t, DT[dst])).to(torch.float64)
    assert torch.equal(got, oracle), f"experimental.typecast {src}->{dst} drifted from typecast"


# Datacopy-safe pairs must keep the prim::copy path so existing callers (Falcon40B et al.) retain
# the sharded<->interleaved and layout-mismatched conversions that ttnn.typecast's SFPU kernel
# rejects.
def test_experimental_typecast_sharded_to_interleaved_safe_pair(device):
    x = torch.randn(1, 1, 64, 64, dtype=torch.float32)
    shard_mem = ttnn.create_sharded_memory_config(
        shape=(64, 64),
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard_mem)
    out = ttnn.experimental.typecast(sharded, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert out.dtype == ttnn.bfloat16
