# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Issue #58372: datacopy ops and matmul must reject dtype= pairs they cannot convert,
and still return bit-exact values (vs. ttnn.typecast) on the pairs they accept."""

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


# ---------------------------------------------------------------------------
# Safe-pair tests: assert the op's output is bit-exact to ttnn.typecast.
# Issue #58372 acceptance criteria is "no dtype pair accepted by these ops
# returns a value different from ttnn.typecast".
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
# experimental.typecast must perform the real conversion (route to SFPU),
# not the datacopy packer. Regression guard for the review of PR #66040.
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
