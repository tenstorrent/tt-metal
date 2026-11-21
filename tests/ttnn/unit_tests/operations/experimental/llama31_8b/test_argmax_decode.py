# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

VOCAB = 128256
CORES = 110


@pytest.fixture(autouse=True)
def p100_only(device):
    if device.dram_grid_size().x != 7:
        pytest.skip("Llama-3.1-8B P100 configs")


def make_logits(kind, vocab, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((1, 1, 32, vocab), generator=g) * 4
    if kind == "ties":
        x = torch.randint(-3, 4, (1, 1, 32, vocab), generator=g).float()
    elif kind == "special":
        n = vocab
        pick = lambda k: torch.randint(0, n, (k,), generator=g)
        x[0, 0, 0, pick(16)] = float("nan")
        x[0, 0, 1, pick(16)] = float("inf")
        x[0, 0, 2, :] = float("-inf")
        x[0, 0, 3, :] = 0.0
        x[0, 0, 3, pick(64)] = -0.0
        x[0, 0, 4, :] = -0.0
        x[0, 0, 5, :] = -torch.rand((n,), generator=g) - 1
        x[0, 0, 6, pick(4)] = -float("nan")
        x[0, 0, 7, pick(8)] = float("inf")
        x[0, 0, 7, pick(8)] = float("nan")
        x[0, 0, 8, :] = float("-inf")
        x[0, 0, 8, n - 1] = -1.0
        x[0, 0, 9, pick(32)] = float("inf")
        x[0, 0, 10, : n // 2] = float("-inf")
        x[0, 0, 11, n - 1] = 1e30
    x = x.to(torch.bfloat16)
    if kind == "special":
        bits = x.view(torch.int16)
        tiny = torch.randint(1, 0x7F, (vocab,), generator=g, dtype=torch.int16)
        bits[0, 0, 12, :] = tiny | torch.tensor(-0x8000, dtype=torch.int16)
        bits[0, 0, 12, pick(8)] = tiny[:8]
        bits[0, 0, 13, :] = 0
        bits[0, 0, 13, pick(1)] = 1
        bits[0, 0, 14, :] = tiny | torch.tensor(-0x8000, dtype=torch.int16)
    return x


def sharded_config(device, vocab, num_cores):
    grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, grid.x * grid.y)
    padded = -(-vocab // 32) * 32
    shard_w = -(-padded // num_cores)
    shard_w = -(-shard_w // 32) * 32
    used = -(-padded // shard_w)
    cores = ttnn.num_cores_to_corerangeset(used, grid, row_wise=True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, [32, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def width_sharded(device, t, num_cores):
    mem = sharded_config(device, t.shape[-1], num_cores)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)


def reference(device, x):
    ref_in = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return ttnn.to_torch(ttnn.argmax(ref_in, dim=-1, keepdim=False)).to(torch.int64)


@pytest.mark.parametrize("kind", ["randn", "ties", "special"])
def test_argmax_decode_matches_argmax(device, kind):
    x = make_logits(kind, VOCAB, seed=VOCAB + CORES)
    tt_in = width_sharded(device, x, CORES)
    assert ttnn.experimental.llama31_8b.argmax_decode_supported(tt_in)
    expected = reference(device, x)

    out = ttnn.experimental.llama31_8b.argmax_decode(tt_in)
    actual = ttnn.to_torch(out)
    assert list(out.shape) == [1, 1, 32]
    assert out.dtype == ttnn.uint32
    assert torch.equal(actual.to(torch.int64), expected), (actual, expected)

    preallocated = ttnn.from_torch(
        torch.zeros((1, 1, 32), dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    ttnn.experimental.llama31_8b.argmax_decode(tt_in, output_tensor=preallocated)
    assert torch.equal(ttnn.to_torch(preallocated).to(torch.int64), expected)


@pytest.mark.parametrize("vocab, num_cores", [(128256, 64), (128256, 16), (151936, 110), (1000, 3)])
def test_argmax_decode_rejects_shapes_outside_table(device, vocab, num_cores, expect_error):
    tt_in = width_sharded(device, make_logits("randn", vocab, seed=vocab + num_cores), num_cores)
    assert not ttnn.experimental.llama31_8b.argmax_decode_supported(tt_in)
    with expect_error(RuntimeError, "llama31_8b.argmax_decode"):
        ttnn.experimental.llama31_8b.argmax_decode(tt_in)


def test_argmax_decode_rejects_interleaved(device, expect_error):
    x = make_logits("randn", 1024, seed=0)
    tt_in = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    assert not ttnn.experimental.llama31_8b.argmax_decode_supported(tt_in)
    with expect_error(RuntimeError, "llama31_8b.argmax_decode"):
        ttnn.experimental.llama31_8b.argmax_decode(tt_in)


@pytest.mark.parametrize("kind", ["randn", "ties", "special"])
def test_argmax_decode_bf8_logits_match_untilize_argmax(device, kind):
    x = make_logits(kind, VOCAB, seed=VOCAB + CORES + 7)
    tt_bf8 = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    expected = ttnn.to_torch(ttnn.argmax(ttnn.untilize(tt_bf8, use_multicore=True), dim=-1, keepdim=False))
    tt_in = ttnn.to_memory_config(tt_bf8, sharded_config(device, VOCAB, CORES), dtype=ttnn.bfloat16)
    assert ttnn.experimental.llama31_8b.argmax_decode_supported(tt_in)
    actual = ttnn.to_torch(ttnn.experimental.llama31_8b.argmax_decode(tt_in))
    assert torch.equal(actual.to(torch.int64), expected.to(torch.int64)), (actual, expected)


def _bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


def _other_ops(device, a, b, logits_rm):
    return [
        ttnn.to_torch(ttnn.exp(a)),
        ttnn.to_torch(ttnn.matmul(a, b)),
        ttnn.to_torch(ttnn.add(a, b)),
        ttnn.to_torch(ttnn.argmax(logits_rm, dim=-1, keepdim=False)),
    ]


def test_argmax_decode_mixed_with_other_ops(device):
    g = torch.Generator().manual_seed(11)
    a_t = torch.randn((1, 1, 256, 256), generator=g).to(torch.bfloat16)
    b_t = torch.randn((1, 1, 256, 256), generator=g).to(torch.bfloat16)
    a = ttnn.from_torch(a_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(b_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x = make_logits("special", VOCAB, seed=3)
    logits_rm = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    expected = reference(device, x)
    alone = _other_ops(device, a, b, logits_rm)

    tt_in = width_sharded(device, x, CORES)
    for _ in range(2):
        first = ttnn.to_torch(ttnn.experimental.llama31_8b.argmax_decode(tt_in)).to(torch.int64)
        mixed = _other_ops(device, a, b, logits_rm)
        second = ttnn.to_torch(ttnn.experimental.llama31_8b.argmax_decode(tt_in)).to(torch.int64)
        assert torch.equal(first, expected) and torch.equal(second, expected)
        for got, want in zip(mixed, alone):
            assert torch.equal(_bits(got), _bits(want))


def _uint32_out(device, shape=(1, 1, 32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, on_device=True, mem=None):
    t = torch.zeros(shape, dtype=torch.int32)
    if not on_device:
        return ttnn.from_torch(t, dtype=dtype, layout=layout)
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=mem or ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("bad", ["int32", "short", "rank2", "tile", "host"])
def test_argmax_decode_rejects_bad_output_tensor(device, bad, expect_error):
    tt_in = width_sharded(device, make_logits("randn", VOCAB, seed=21), CORES)
    out = {
        "int32": lambda: _uint32_out(device, dtype=ttnn.int32),
        "short": lambda: _uint32_out(device, shape=(1, 1, 16)),
        "rank2": lambda: _uint32_out(device, shape=(1, 32)),
        "tile": lambda: _uint32_out(device, layout=ttnn.TILE_LAYOUT),
        "host": lambda: _uint32_out(device, on_device=False),
    }[bad]()
    assert not ttnn.experimental.llama31_8b.argmax_decode_supported(tt_in, output_tensor=out)
    with expect_error(RuntimeError, "llama31_8b.argmax_decode"):
        ttnn.experimental.llama31_8b.argmax_decode(tt_in, output_tensor=out)


def test_argmax_decode_rejects_sharded_memory_config(device, expect_error):
    tt_in = width_sharded(device, make_logits("randn", VOCAB, seed=22), CORES)
    mem = sharded_config(device, VOCAB, CORES)
    assert not ttnn.experimental.llama31_8b.argmax_decode_supported(tt_in, memory_config=mem)
    with expect_error(RuntimeError, "llama31_8b.argmax_decode"):
        ttnn.experimental.llama31_8b.argmax_decode(tt_in, memory_config=mem)


@pytest.mark.parametrize("out_kind", ["dram", "l1", "preallocated"])
def test_argmax_decode_program_cache(device, out_kind):
    xs = [make_logits(kind, VOCAB, seed=40 + i) for i, kind in enumerate(["randn", "special"])]
    expected = [reference(device, x) for x in xs]
    ins = [width_sharded(device, x, CORES) for x in xs]
    mem = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG, "preallocated": None}[out_kind]
    outs = [_uint32_out(device) for _ in xs] if out_kind == "preallocated" else [None, None]
    for t, o in zip(ins, outs):
        assert ttnn.experimental.llama31_8b.argmax_decode_supported(t, memory_config=mem, output_tensor=o)

    entries = []
    for t, o, want in zip(ins, outs, expected):
        if o is None:
            got = ttnn.experimental.llama31_8b.argmax_decode(t, memory_config=mem)
        else:
            ttnn.experimental.llama31_8b.argmax_decode(t, output_tensor=o)
            got = o
        entries.append(device.num_program_cache_entries())
        assert torch.equal(ttnn.to_torch(got).to(torch.int64), want)
    assert entries[1] == entries[0]


def test_argmax_decode_cache_hit_rejects_freed_output(device, expect_error):
    tt_in = width_sharded(device, make_logits("randn", VOCAB, seed=23), CORES)
    ttnn.experimental.llama31_8b.argmax_decode(tt_in, output_tensor=_uint32_out(device))
    entries = device.num_program_cache_entries()
    out = _uint32_out(device)
    ttnn.deallocate(out)
    with expect_error(RuntimeError, "llama31_8b.argmax_decode"):
        ttnn.experimental.llama31_8b.argmax_decode(tt_in, output_tensor=out)
    assert device.num_program_cache_entries() == entries
