# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The context-parallel KV cache (tt/kv_cache.py) on the device, tenstorrent/tt-metal#65886.

What is written is the reference latent (reference/attention.py Hy4Attention.latent, k_pass | k_rope)
and a 128-wide index key; what is read back must be that, at every position, after a round trip
through BlockCyclicLayout.merge:

* BF16 storage: bit-equal to the reference rounded to bf16, by both prefill paths and by decode.
* BFP8 storage: the host-striped path is bit-equal to the host's BFP8 packing of the reference. The
  device path and the decode append pack on the device, which rounds differently, so they are held
  to one BFP8 step of the reference (the step of each 16-value exponent group) and to a PCC.

Then the read path (stripe, filled, locate, gather), the cache fed by the device attention module,
and the measurements the issue asks for: stored bytes per token, the PCC cost of BFP8 storage on
latents and index keys from the checkpoint's layer 0-5 weights, the DRAM a chip has, and the wall
time of the cache's own writes and reads.

The cache stores rows as they are handed in. Only the tests that say so (the device attention's
latent, the checkpoint entries) use the device frame of tt/frames.py; the others write reference
rows in the HF order, which exercises the same storage.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.reference.attention import AttentionConfig, Hy4Attention, rms_norm, rope_cos_sin
from models.experimental.hy4_preview.reference.indexer import Hy4Indexer, IndexerConfig
from models.experimental.hy4_preview.tests.common import (
    GALAXY_DEVICE_PARAMS,
    GALAXY_MESH,
    SC1_HIDDEN_AXIS,
    check_pcc,
    hy4_tensors,
    pcc,
)
from models.experimental.hy4_preview.tt import frames
from models.experimental.hy4_preview.tt.attention import TtHy4Attention
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.kv_cache import (
    HY4_INDEXER_LAYERS,
    INDEX_KEY,
    LATENT,
    KvCacheConfig,
    TtHy4KvCache,
    selection_table,
)
from models.experimental.hy4_preview.tt.kv_layout import BlockCyclicLayout

HY4 = AttentionConfig()
# Hy4's latent (512 | 64) from a narrow hidden state, so long prompts stay cheap on the host. The
# hidden size still splits 4 ways and the heads 8 ways for the tests that run the device attention.
NARROW = AttentionConfig(
    hidden_size=768,
    num_heads=8,
    q_lora_rank=256,
    kv_lora_rank=512,
    qk_nope_head_dim=96,
    qk_rope_head_dim=64,
    v_head_dim=128,
)
DTYPES = ["bf16", "bfp8"]
TT_DTYPE = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}
ROUND = 32 * 128  # positions in one round of the 32-chip layout
BFP8_PCC = 0.9995

galaxy = pytest.mark.parametrize(
    "device_params, mesh_device", [(GALAXY_DEVICE_PARAMS, GALAXY_MESH)], indirect=True, ids=["8x4"]
)


def _reference(cfg, seed):
    """A reference attention layer with random parameters at the scale a trained layer has."""
    ref = Hy4Attention(cfg).eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name in Hy4Attention.PARAMETER_NAMES:
            p = getattr(ref, name)
            if p.dim() == 2:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[1] ** -0.5)
            elif name == "learnable_sink_param":
                p.copy_(torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.25 * torch.randn(p.shape, generator=g))
    return ref


_REF = {}


def _latents(length, seed, start=0):
    """Reference cache entries for positions ``start .. start + length``: (latent [length, 576],
    index key [length, 128]). The index key is random data of the right shape: the indexer's
    reference belongs to #65887 and the cache does not care what the 128 values mean."""
    if "narrow" not in _REF:
        _REF["narrow"] = _reference(NARROW, seed=1)
    ref = _REF["narrow"]
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(length, NARROW.hidden_size, generator=g)
    cos, sin = rope_cos_sin(torch.arange(length) + start, NARROW)
    with torch.no_grad():
        _, k_pass, k_rope = ref.latent(x, cos, sin)
    return torch.cat([k_pass, k_rope], dim=-1), torch.randn(length, 128, generator=g)


def _held(x, dtype):
    """What ``dtype`` storage holds of ``x`` [T, W] when the host packs it."""
    t = ttnn.from_torch(x.reshape(1, 1, *x.shape).float(), dtype=TT_DTYPE[dtype], layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(t).float().reshape(x.shape)


def _bfp8_step(x):
    """An upper bound of the BFP8 step at every value of ``x`` [T, W]: a group of 16 values in a row
    shares the exponent of its largest magnitude, and 7 mantissa bits leave a step of at most 1/64 of it."""
    groups = x.abs().reshape(x.shape[0], -1, 16).amax(dim=-1, keepdim=True)
    return (groups / 64).expand(-1, -1, 16).reshape(x.shape)


def _check(name, ref, got, dtype, exact):
    """got vs the reference rows it must hold. ``exact``: bit-equal to the host's packing."""
    assert got.shape == ref.shape, f"{name}: {tuple(got.shape)} vs {tuple(ref.shape)}"
    if exact:
        held = _held(ref, dtype)
        wrong = (got != held).any(dim=-1).nonzero().flatten()
        assert (
            wrong.numel() == 0
        ), f"{name}: {wrong.numel()} of {ref.shape[0]} rows differ, first at {wrong[:5].tolist()}"
        logger.info(f"{name}: {ref.shape[0]} rows bit-equal to the reference in {dtype}")
    else:
        bf16 = _held(ref, "bf16")
        over = ((got - bf16).abs() > _bfp8_step(bf16) * 1.001 + 1e-30).sum().item()
        check_pcc(name, ref, got, BFP8_PCC)
        assert over == 0, f"{name}: {over} values are more than one BFP8 step from the reference"


def _upload(mesh_device, x):
    """[T, W] -> device [1, 1, T, W] bf16, the same on every chip."""
    return ttnn.from_torch(
        x.reshape(1, 1, *x.shape).float().contiguous(), layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )


def _cache(mesh_device, dtype, max_users=2, max_context=3 * ROUND, layers=(0, 1), num_chips=32):
    cfg = KvCacheConfig(
        num_layers=2,
        indexer_layers=(0,),
        max_users=max_users,
        max_context=max_context,
        dtype=dtype,
        layout=BlockCyclicLayout(num_chips=num_chips),
    )
    return TtHy4KvCache(mesh_device, cfg, layers=layers)


def _append(cache, mesh_device, positions, rows):
    """One decode step: user u appends rows[u] = (latent row, index key row) at positions[u]."""
    users = cache.cfg.max_users
    latent = torch.zeros(users, cache.cfg.latent_dim)
    key = torch.zeros(users, cache.cfg.index_key_dim)
    for u, row in enumerate(rows):
        if row is not None:
            latent[u], key[u] = row
    indices = cache.append_indices(positions)
    cache.write_decode(0, indices, latent=_upload(mesh_device, latent), index_key=_upload(mesh_device, key))


# ---------------------------------------------------------------------------------------------
# prefill: both paths, lengths that do and do not fill whole blocks and rounds
# ---------------------------------------------------------------------------------------------
@galaxy
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("path", ["device", "host"])
def test_prefill_round_trip(mesh_device, dtype, path):
    layout = BlockCyclicLayout()
    # ROUND - 6 ends inside the last tile of a round: the pad to whole rounds adds no tile there
    for length in (1, 100, 128, 1000, ROUND - 6, ROUND, 5000):
        cache = _cache(mesh_device, dtype, layers=(0,))
        latent, key = _latents(length, seed=length)
        if path == "device":
            cache.write_prefill(0, 1, latent=_upload(mesh_device, latent), index_key=_upload(mesh_device, key))
        else:
            cache.write_prefill_host(0, 1, latent=latent, index_key=key)
        exact = dtype == "bf16" or path == "host"
        label = f"prefill {path} {dtype} T{length}"
        for kind, ref in ((LATENT, latent), (INDEX_KEY, key)):
            _check(f"{label} {kind}", ref, cache.to_torch(0, 1, length, kind), dtype, exact)
            stripes = cache.stripes_to_torch(0, kind)  # [32, users, slots, W]
            assert tuple(stripes.shape) == (32, 2, cache.cfg.slots, ref.shape[-1])
            assert (stripes[:, 0] == 0).all(), f"{label} {kind}: the other user changed"
            # every chip holds its own blocks and nothing else
            filled = layout.filled(length, cache.cfg.slots)
            assert (stripes[:, 1][~filled] == 0).all(), f"{label} {kind}: rows outside the prompt are not zero"
            if exact:
                want = layout.split(_held(ref, dtype), slots=cache.cfg.slots)
                assert torch.equal(stripes[:, 1], want), f"{label} {kind}: stripes differ from kv_layout.split"
        cache.deallocate()


# ---------------------------------------------------------------------------------------------
# decode: appends that cross a block, a chip, a mesh row and a round
# ---------------------------------------------------------------------------------------------
@galaxy
@pytest.mark.parametrize("dtype", DTYPES)
def test_decode_append(mesh_device, dtype):
    cache = _cache(mesh_device, dtype, layers=(0,))
    exact = dtype == "bf16"
    total = {0: 1030, 1: ROUND + 4}
    refs = {u: _latents(n, seed=10 + u) for u, n in total.items()}

    def prefill(user, length):
        cache.write_prefill(
            0,
            user,
            latent=_upload(mesh_device, refs[user][0][:length]),
            index_key=_upload(mesh_device, refs[user][1][:length]),
        )

    def verify(label, lengths):
        for user, length in lengths.items():
            for i, kind in enumerate((LATENT, INDEX_KEY)):
                got = cache.to_torch(0, user, length, kind)
                _check(f"append {dtype} {label} user {user} {kind}", refs[user][i][:length], got, dtype, exact)

    def row(user, position):
        return refs[user][0][position], refs[user][1][position]

    # user 0 over a block boundary, which is a chip boundary (127 | 128: chip 0 -> 1) while user 1
    # crosses a round boundary (4095 | 4096: chip 31 -> chip 0, second block of its stripe)
    prefill(0, 126)
    prefill(1, ROUND - 2)
    before = cache.stripes_to_torch(0, LATENT)
    for i in range(6):
        _append(cache, mesh_device, [126 + i, ROUND - 2 + i], [row(0, 126 + i), row(1, ROUND - 2 + i)])
    verify("block and round boundary", {0: 132, 1: ROUND + 4})
    after = cache.stripes_to_torch(0, LATENT)
    changed = (after != before).any(dim=-1)  # [32, users, slots]
    assert changed.sum().item() == 12, f"12 appended rows, {changed.sum().item()} rows changed"
    assert sorted(changed[:, 0].nonzero()[:, 0].unique().tolist()) == [0, 1]
    assert sorted(changed[:, 1].nonzero()[:, 0].unique().tolist()) == [0, 31]

    # user 0 alone over a mesh row boundary (1023 | 1024: chip 7 -> 8, the 8x4 mesh's row 1 -> 2); user 1
    # sits the steps out
    prefill(0, 1022)
    before_other = cache.stripes_to_torch(0, LATENT, user=1)
    for i in range(8):
        _append(cache, mesh_device, [1022 + i, None], [row(0, 1022 + i), None])
    verify("mesh row boundary", {0: 1030, 1: ROUND + 4})
    assert torch.equal(cache.stripes_to_torch(0, LATENT, user=1), before_other), "a skipped user changed"


# ---------------------------------------------------------------------------------------------
# prefill that continues a context: after decode, mid block, and chunk by chunk
# ---------------------------------------------------------------------------------------------
@galaxy
@pytest.mark.parametrize("dtype", DTYPES)
def test_prefill_continues_a_context(mesh_device, dtype):
    cache = _cache(mesh_device, dtype, layers=(0,))
    exact = dtype == "bf16"
    latent, key = _latents(3 * ROUND, seed=21)

    def write(start, end, host=False):
        if host:
            cache.write_prefill_host(0, 0, latent=latent[start:end], index_key=key[start:end], start=start)
        else:
            cache.write_prefill(
                0,
                0,
                latent=_upload(mesh_device, latent[start:end]),
                index_key=_upload(mesh_device, key[start:end]),
                start=start,
            )

    def verify(label, length):
        for kind, ref in ((LATENT, latent), (INDEX_KEY, key)):
            _check(f"continue {dtype} {label} {kind}", ref[:length], cache.to_torch(0, 0, length, kind), dtype, exact)

    write(0, 700)
    for p in (700, 701, 702, 703):
        _append(cache, mesh_device, [p, None], [(latent[p], key[p]), None])
    held = cache.to_torch(0, 0, 704, LATENT)
    write(704, 5003)  # starts mid block on chip 5 and runs into the second round
    verify("device from 704", 5003)
    assert torch.equal(cache.to_torch(0, 0, 704, LATENT), held), "the continuation changed earlier positions"
    held = cache.to_torch(0, 0, 5003, LATENT)
    write(5003, 6000, host=True)  # the host path from inside a tile
    verify("host from 5003", 6000)
    assert torch.equal(cache.to_torch(0, 0, 5003, LATENT), held), "the continuation changed earlier positions"
    held = cache.to_torch(0, 0, 6000, LATENT)
    write(6000, 6007)  # the device path from inside a tile (row 16 of its tile), a few rows
    write(6007, 2 * ROUND)  # and again (row 23), up to a round boundary
    verify("device from 6000 and 6007", 2 * ROUND)
    assert torch.equal(cache.to_torch(0, 0, 6000, LATENT), held), "the continuation changed earlier positions"
    write(2 * ROUND, 2 * ROUND + 300)  # a chunk that starts on a round boundary
    verify("device from a round boundary", 2 * ROUND + 300)
    held = cache.to_torch(0, 0, 2 * ROUND + 300, LATENT)
    write(2 * ROUND + 300, 3 * ROUND)  # from inside a tile of the last round up to the capacity
    verify("device from inside a tile to the capacity", 3 * ROUND)
    assert torch.equal(
        cache.to_torch(0, 0, 2 * ROUND + 300, LATENT), held
    ), "the continuation changed earlier positions"


# ---------------------------------------------------------------------------------------------
# read path: stripe, filled, locate + gather
# ---------------------------------------------------------------------------------------------
@galaxy
@pytest.mark.parametrize("dtype", DTYPES)
def test_read_path(mesh_device, dtype):
    cache = _cache(mesh_device, dtype, layers=(0,))
    layout, slots = cache.layout, cache.cfg.slots
    lengths = [300, 5000]
    refs = [_latents(n, seed=30 + n) for n in lengths]
    for user, (latent, key) in enumerate(refs):
        cache.write_prefill(0, user, latent=_upload(mesh_device, latent), index_key=_upload(mesh_device, key))

    def chips(t):
        return torch.stack([ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(t)])

    # the stripe is the storage; narrowed, it is the rows a context can reach
    assert tuple(cache.stripe(0).shape) == (2, 1, slots, 576)
    assert tuple(cache.stripe(0, INDEX_KEY).shape) == (2, 1, slots, 128)
    narrowed = cache.stripe(0, LATENT, user=1, length=5000)
    assert tuple(narrowed.shape) == (1, 1, layout.slots(5000), 576)
    assert torch.equal(chips(narrowed)[:, 0, 0], cache.stripes_to_torch(0, LATENT, user=1)[:, : layout.slots(5000)])

    # slot_position and filled, against kv_layout
    position = chips(cache.slot_position)[:, 0, 0, 0]  # [32, slots]
    assert torch.equal(position.long(), layout.position(torch.arange(32)[:, None], torch.arange(slots)[None, :]))
    for length in (1, 128, 300, 5000, cache.cfg.capacity):
        assert torch.equal(chips(cache.filled(length))[:, 0, 0, 0].bool(), layout.filled(length, slots)), length
    per_user = chips(cache.filled(lengths))  # [32, users, 1, 1, slots]
    for user, length in enumerate(lengths):
        assert torch.equal(per_user[:, user, 0, 0].bool(), layout.filled(length, slots)), f"user {user}"
    queries = torch.tensor([1.0, 129.0, 4097.0, 5000.0]).reshape(1, 1, 4, 1)  # query position + 1
    causal = chips(
        cache.filled(ttnn.from_torch(queries, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device))
    )
    for i, length in enumerate((1, 129, 4097, 5000)):
        assert torch.equal(causal[:, 0, 0, i].bool(), layout.filled(length, slots)), f"causal row {i}"
    logger.info(f"read path {dtype}: stripe, slot_position and filled agree with kv_layout")

    # gather hands the storage to ttnn.embedding as one table of users * slots rows: that reshape must
    # be a view, not a copy. (The embedding itself untilizes the whole table into a temporary on every
    # call; that cost is not measured here.)
    allocated = _dram(mesh_device)[0]
    table = ttnn.reshape(cache.stripe(0), [1, 1, 2 * slots, 576])
    assert _dram(mesh_device)[0] == allocated, "viewing the storage as a table allocated DRAM"
    assert tuple(table.shape) == (1, 1, 2 * slots, 576)

    # locate + gather: every chip returns K rows and flags its own; one chip owns each position
    g = torch.Generator().manual_seed(7)
    for user, length in enumerate(lengths):
        k = 64
        picks = torch.randperm(length, generator=g)[:k]
        picks[:4] = torch.tensor([0, length - 1, min(127, length - 1), min(128, length - 1)])
        d_picks = ttnn.from_torch(
            picks.float().reshape(1, 1, 1, k), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        rows, mine = cache.locate(d_picks, user=user)
        want_slot, want_mine = selection_table(layout, picks)
        got_mine = chips(mine)[:, 0, 0, 0].bool()
        assert torch.equal(got_mine, want_mine), f"user {user}: mine differs from kv_layout"
        assert (got_mine.sum(dim=0) == 1).all()
        got_rows = torch.stack([ttnn.to_torch(s).long() for s in ttnn.get_device_tensors(rows)])[:, 0]
        assert torch.equal(got_rows, want_slot + user * slots), f"user {user}: rows differ from kv_layout"
        for i, kind in enumerate((LATENT, INDEX_KEY)):
            gathered = cache.gather(0, rows, kind)
            assert tuple(gathered.shape) == (1, 1, k, cache.cfg.width(kind))
            got = chips(gathered)[:, 0, 0]  # [32, K, W]
            merged = (got * got_mine[..., None]).sum(dim=0)
            _check(f"gather {dtype} user {user} {kind}", refs[user][i][picks], merged, dtype, dtype == "bf16")
            assert torch.equal(merged, cache.to_torch(0, user, length, kind)[picks]), "gather differs from readback"


@galaxy
def test_locate_gather_queries(mesh_device):
    """locate and gather for a block of queries, the way the indexer hands a prefill selection over
    ([1, 1, S, K]), and for selection sizes that are not whole tiles (K = 1, 50)."""
    cache = _cache(mesh_device, "bf16", layers=(0,))
    layout, slots, length = cache.layout, cache.cfg.slots, 5000
    latent, key = _latents(length, seed=81)
    cache.write_prefill(0, 1, latent=_upload(mesh_device, latent), index_key=_upload(mesh_device, key))
    g = torch.Generator().manual_seed(82)
    for s, k in ((1, 1), (1, 50), (3, 64), (5, 50)):
        picks = torch.randint(0, length, (s, k), generator=g)
        picks[0, 0] = length - 1
        d_picks = ttnn.from_torch(
            picks.float().reshape(1, 1, s, k), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        rows, mine = cache.locate(d_picks, user=1)
        want_slot, want_mine = selection_table(layout, picks)  # [32, S, K]
        got_mine = torch.stack([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(mine)])[:, 0, 0].bool()
        got_rows = torch.stack([ttnn.to_torch(t).long() for t in ttnn.get_device_tensors(rows)])
        assert tuple(got_mine.shape) == tuple(got_rows.shape) == (32, s, k), f"S{s} K{k}: {tuple(got_rows.shape)}"
        assert torch.equal(got_mine, want_mine), f"S{s} K{k}: mine differs from kv_layout"
        assert torch.equal(got_rows, want_slot + slots), f"S{s} K{k}: rows differ from kv_layout"
        for kind, ref in ((LATENT, latent), (INDEX_KEY, key)):
            gathered = cache.gather(0, rows, kind)
            assert tuple(gathered.shape) == (1, s, k, cache.cfg.width(kind))
            got = torch.stack([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(gathered)])[:, 0]
            merged = (got * got_mine[..., None]).sum(dim=0).reshape(s * k, -1)
            _check(f"gather S{s} K{k} {kind}", ref[picks.flatten()], merged, "bf16", True)


# ---------------------------------------------------------------------------------------------
# the far end: the last tile row, the last slot, the last chip, the last user, and the capacity
# ---------------------------------------------------------------------------------------------
@galaxy
@pytest.mark.parametrize("dtype", DTYPES)
def test_far_end_of_the_stripes(mesh_device, dtype):
    """A context that runs up to the cache's capacity. The last 160 positions are the last 32 rows of
    chip 30's last block and the whole last block of chip 31. User 1 holds two decoded tokens right
    before them, then gets them by a prefill that stops one position short and a decode append into
    the very last row; user 0 gets them by one prefill that ends exactly at the capacity. Nothing
    before them may change, and nothing can be written past the end."""
    cache = _cache(mesh_device, dtype, layers=(0,))
    layout, slots, capacity = cache.layout, cache.cfg.slots, cache.cfg.capacity
    assert capacity == 3 * ROUND
    start = capacity - 160
    latent, key = _latents(160, seed=91, start=start)
    earlier = _latents(2, seed=92, start=start - 2)
    exact = dtype == "bf16"
    assert (layout.chip(start), layout.slot(start)) == (30, slots - 32)
    assert (layout.chip(capacity - 1), layout.slot(capacity - 1)) == (31, slots - 1)

    for i, p in enumerate((start - 2, start - 1)):
        _append(cache, mesh_device, [None, p], [None, (earlier[0][i], earlier[1][i])])
    held = {kind: cache.to_torch(0, 1, start, kind) for kind in (LATENT, INDEX_KEY)}
    cache.write_prefill(
        0, 1, latent=_upload(mesh_device, latent[:159]), index_key=_upload(mesh_device, key[:159]), start=start
    )
    _append(cache, mesh_device, [None, capacity - 1], [None, (latent[159], key[159])])
    cache.write_prefill(0, 0, latent=_upload(mesh_device, latent), index_key=_upload(mesh_device, key), start=start)
    for i, (kind, ref) in enumerate(((LATENT, latent), (INDEX_KEY, key))):
        for user in (0, 1):
            got = cache.to_torch(0, user, capacity, kind)
            _check(f"far end {dtype} user {user} {kind}", ref, got[start:], dtype, exact)
            if user == 0:
                assert (got[:start] == 0).all(), f"user 0 {kind}: positions before the write changed"
            else:
                assert torch.equal(got[:start], held[kind]), f"user 1 {kind}: positions before the write changed"
                assert (got[: start - 2] == 0).all()
                _check(f"far end {dtype} user 1 {kind} kept rows", earlier[i], got[start - 2 : start], dtype, exact)

    with pytest.raises(AssertionError, match="exceed"):  # allow-pytest.raises: host-side check
        cache.append_indices([capacity, None])
    with pytest.raises(AssertionError, match="exceed"):  # allow-pytest.raises: host-side check
        cache.write_prefill(0, 0, latent=_upload(mesh_device, latent[:64]), start=capacity - 32)
    with pytest.raises(AssertionError, match="exceed"):  # allow-pytest.raises: host-side check
        cache.write_prefill_host(0, 0, latent=latent[:64], start=capacity - 32)


# ---------------------------------------------------------------------------------------------
# one chip: the layout degenerates to a contiguous cache, the ops stay the same
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
def test_one_chip(device, dtype):
    cache = _cache(device, dtype, max_context=512, layers=(0,), num_chips=1)
    latent, key = _latents(300, seed=41)
    exact = dtype == "bf16"
    cache.write_prefill(0, 1, latent=_upload(device, latent[:190]), index_key=_upload(device, key[:190]))
    for p in (190, 191):
        _append(cache, device, [None, p], [None, (latent[p], key[p])])
    for start, end in ((192, 250), (250, 300)):  # from a tile row, then from inside a tile
        cache.write_prefill(
            0, 1, latent=_upload(device, latent[start:end]), index_key=_upload(device, key[start:end]), start=start
        )
    for kind, ref in ((LATENT, latent), (INDEX_KEY, key)):
        _check(f"one chip {dtype} {kind}", ref, cache.to_torch(0, 1, 300, kind), dtype, exact)
    assert (cache.stripes_to_torch(0, LATENT, user=0) == 0).all()


# ---------------------------------------------------------------------------------------------
# the device attention's latent goes straight into the cache
# ---------------------------------------------------------------------------------------------
@galaxy
@pytest.mark.parametrize("dtype", DTYPES)
def test_attention_latent_into_cache(mesh_device, dtype):
    """TtHy4Attention.latent leaves (k_pass, k_rope) whole on every chip; the cache stripes them on
    the device. Prefill a prompt, then append tokens one at a time, all from the device module."""
    cfg, length, steps = NARROW, 300, 3
    ref = _reference(cfg, seed=1)
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    attention = TtHy4Attention(plan, cfg, ref.weights(), weight_dtype=ttnn.bfloat16)
    cache = _cache(mesh_device, dtype, max_users=1, layers=(1,))

    x = torch.randn(length + steps, cfg.hidden_size, generator=torch.Generator().manual_seed(50))
    cos, sin = rope_cos_sin(torch.arange(length + steps), cfg)
    with torch.no_grad():
        _, k_pass, k_rope = ref.latent(x, cos, sin)
    # the cache holds the latent in the device frame: k_rope in interleaved pairs (tt/frames.py)
    want = torch.cat([k_pass, frames.rope_to_device(k_rope)], dim=-1)
    cos, sin = frames.rope_tables(cos, sin)

    def device_latent(rows):
        up = lambda t, mapper=None: ttnn.from_torch(  # noqa: E731
            t[rows].reshape(1, 1, -1, t.shape[-1]).contiguous(),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            mesh_mapper=mapper,
        )
        _, d_k_pass, d_k_rope = attention.latent(up(x, plan.shard_hidden(3)), up(cos), up(sin))
        return TtHy4KvCache.pack(d_k_pass, d_k_rope)

    cache.write_prefill(1, 0, latent=device_latent(slice(0, length)))
    for p in range(length, length + steps):
        cache.write_decode(1, cache.append_indices([p]), latent=device_latent(slice(p, p + 1)))
    got = cache.to_torch(1, 0, length + steps)
    check_pcc(f"attention latent into cache {dtype} prefill", want[:length], got[:length], 0.999)
    check_pcc(f"attention latent into cache {dtype} decode", want[length:], got[length:], 0.999)


# ---------------------------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------------------------
def _dram(mesh_device):
    """(bytes allocated, bytes free, largest free block) on one chip. The mesh allocates the same
    buffers on every chip, so one view stands for all 32."""
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    banks = view.num_banks
    return (
        view.total_bytes_allocated_per_bank * banks,
        view.total_bytes_free_per_bank * banks,
        view.largest_contiguous_bytes_free_per_bank * banks,
    )


@galaxy
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("max_context", [131072, 1048576], ids=["128K", "1M"])
def test_stored_bytes_per_token(mesh_device, dtype, max_context):
    """All 78 latent stripes and 21 index-key stripes of one user, allocated for real: DRAM taken per
    chip against the arithmetic, and so the bytes one token costs."""
    cfg = KvCacheConfig(max_users=1, max_context=max_context, dtype=dtype)
    assert cfg.values_per_token == 47616
    allocated_before, free_before, _ = _dram(mesh_device)
    start = time.time()
    cache = TtHy4KvCache(mesh_device, cfg)
    seconds = time.time() - start
    allocated, free, largest = _dram(mesh_device)
    per_chip = allocated - allocated_before
    per_token = per_chip * 32 / cfg.capacity
    logger.info(
        f"MEASURED kv {dtype} {max_context} positions: {per_chip / 2**20:.1f} MiB per chip "
        f"(arithmetic {cfg.bytes_per_chip() / 2**20:.1f} MiB), {per_token:.0f} bytes per token "
        f"(arithmetic {cfg.bytes_per_token:.0f}), {per_chip * 32 / 1e9:.2f} GB on the Galaxy, allocated in {seconds:.1f} s"
    )
    logger.info(
        f"MEASURED dram per chip: {free_before / 1e9:.3f} GB free before the cache "
        f"({allocated_before / 1e6:.1f} MB already allocated), {free / 1e9:.3f} GB free with it, "
        f"largest free block {largest / 1e9:.3f} GB"
    )
    # a write and a read at the far end of the largest stripe: the storage is real
    latent, key = _latents(200, seed=3)
    start_position = cfg.capacity - ROUND
    cache.write_prefill(
        77, 0, latent=_upload(mesh_device, latent), index_key=_upload(mesh_device, key), start=start_position
    )
    got = cache.layout.merge(
        torch.stack(
            [
                ttnn.to_torch(s).float()[0, 0]
                for s in ttnn.get_device_tensors(
                    ttnn.slice(cache.stripe(77), [0, 0, cfg.slots - 128, 0], [1, 1, cfg.slots, 576])
                )
            ]
        ),
        200,
    )
    _check(f"far end of a {max_context} stripe {dtype}", latent, got, dtype, dtype == "bf16")
    assert abs(per_chip - cfg.bytes_per_chip()) <= 0.01 * cfg.bytes_per_chip(), (per_chip, cfg.bytes_per_chip())
    cache.deallocate()


@galaxy
def test_bfp8_cost_on_checkpoint_latents(mesh_device):
    """PCC of BFP8 storage against BF16 storage and against the fp32 reference, on what the
    checkpoint's layer 0-5 weights put into the cache: the latent (kv_a_proj_with_mqa,
    kv_a_layernorm) and, on the full layers 0, 1 and 5, the index key (indexer wk, k_norm), both
    behind the layer's own input_layernorm and both in the frame the device stores (tt/frames.py:
    k_rope interleaved, the index key in the Hadamard basis). The index key is also compared after
    it is mapped back to the HF frame. The hidden states are random: a layer's real input needs the
    layers below it."""
    length = 2000
    caches = {dtype: _cache(mesh_device, dtype, max_users=1, max_context=ROUND, layers=(0,)) for dtype in DTYPES}
    ref = Hy4Attention(HY4).eval()
    indexer = Hy4Indexer(IndexerConfig()).eval()
    rope = HY4.qk_rope_head_dim

    def stored(kind, rows, label):
        """rows [T, W] through both caches -> what the BFP8 one holds; logs the PCCs."""
        got = {}
        for dtype, cache in caches.items():
            cache.write_prefill(0, 0, **{kind: _upload(mesh_device, rows)})
            got[dtype] = cache.to_torch(0, 0, length, kind)
        assert torch.equal(got["bf16"], _held(rows, "bf16")), f"{label}: the bf16 cache does not hold the reference"
        return got

    def summary(label, rows, got):
        per_token = min(pcc(rows[t], got[t]) for t in range(length))
        rel = (got - rows).norm() / rows.norm()
        return f"MEASURED bfp8 storage {label}: worst token pcc={per_token:.6f} relative error={rel:.5f}"

    for layer in range(6):
        prefix = f"model.layers.{layer}."
        kv_a, kv_norm, pre_norm = hy4_tensors(
            [
                prefix + "self_attn.kv_a_proj_with_mqa.weight",
                prefix + "self_attn.kv_a_layernorm.weight",
                prefix + "input_layernorm.weight",
            ]
        )
        cos, sin = rope_cos_sin(torch.arange(length), HY4)
        with torch.no_grad():
            ref.kv_a_proj_with_mqa.copy_(kv_a)
            ref.kv_a_layernorm.copy_(kv_norm)
            x = torch.randn(length, HY4.hidden_size, generator=torch.Generator().manual_seed(100 + layer))
            x = rms_norm(x, pre_norm, 1e-5)
            _, k_pass, k_rope = ref.latent(x, cos, sin)
        latent = frames.tail_to_device(torch.cat([k_pass, k_rope], dim=-1), rope)
        got = stored(LATENT, latent, f"L{layer} latent")
        for part, cols in (("k_pass", slice(0, 512)), ("k_rope", slice(512, 576)), ("latent", slice(0, 576))):
            check_pcc(f"bfp8 storage L{layer} {part} vs fp32", latent[:, cols], got["bfp8"][:, cols], 0.999)
            check_pcc(f"bfp8 storage L{layer} {part} vs bf16", got["bf16"][:, cols], got["bfp8"][:, cols], 0.999)
        logger.info(
            summary(f"L{layer} latent", latent, got["bfp8"])
            + f" k_pass rms={latent[:, :512].pow(2).mean().sqrt():.3f} k_rope rms={latent[:, 512:].pow(2).mean().sqrt():.3f}"
        )

        if layer not in HY4_INDEXER_LAYERS:
            continue
        wk, k_norm_weight, k_norm_bias = hy4_tensors(
            [prefix + f"self_attn.indexer.{name}" for name in ("wk.weight", "k_norm.weight", "k_norm.bias")]
        )
        with torch.no_grad():
            indexer.wk.copy_(wk)
            indexer.k_norm_weight.copy_(k_norm_weight)
            indexer.k_norm_bias.copy_(k_norm_bias)
            key_hf = indexer.keys(x, cos, sin)
        key = frames.index_to_device(key_hf, rope)
        got = stored(INDEX_KEY, key, f"L{layer} index key")
        check_pcc(f"bfp8 storage L{layer} index key vs fp32", key, got["bfp8"], 0.999)
        check_pcc(f"bfp8 storage L{layer} index key vs bf16", got["bf16"], got["bfp8"], 0.999)
        check_pcc(
            f"bfp8 storage L{layer} index key in the HF frame vs fp32",
            key_hf,
            frames.index_from_device(got["bfp8"], rope),
            0.999,
        )
        logger.info(summary(f"L{layer} index key", key, got["bfp8"]) + f" rms={key.pow(2).mean().sqrt():.3f}")
    for cache in caches.values():
        cache.deallocate()


def _timed(mesh_device, label, fn, repeats):
    """Run ``fn`` once (it compiles its programs), then ``repeats`` times against the clock. The clock
    stops after a read of every chip, queued behind the work, and what that read costs on an idle
    mesh is taken off. Logs the mean and returns the last result."""
    probe = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    def barrier():
        ttnn.synchronize_device(mesh_device)
        for shard in ttnn.get_device_tensors(probe):
            ttnn.to_torch(shard)

    fn()
    barrier()
    begin = time.perf_counter()
    barrier()
    idle = time.perf_counter() - begin
    begin = time.perf_counter()
    for _ in range(repeats):
        out = fn()
    ttnn.synchronize_device(mesh_device)
    synced = time.perf_counter() - begin
    barrier()
    total = time.perf_counter() - begin - idle
    logger.info(
        f"MEASURED kv timing {label}: {total * 1e3 / repeats:.2f} ms (warm, mean of {repeats}; "
        f"{synced * 1e3 / repeats:.2f} ms up to synchronize_device, barrier {idle * 1e3:.2f} ms)"
    )
    probe.deallocate(True)
    return out


@galaxy
@pytest.mark.parametrize("dtype", DTYPES)
def test_write_and_read_timing(mesh_device, dtype):
    """Wall time of the cache's own ops for one layer at a 131,072-position context (4,096 rows per
    chip): a 4,096-token prefill, a decode append, the per-step index upload, locate and gather of
    the indexer's 2,048 picks. Each is run once to compile and then timed warm, up to a read of every
    chip queued behind the work (minus what that read costs on an idle mesh). Numbers are logged, not
    asserted; what was written is checked at the end."""
    cfg = KvCacheConfig(num_layers=1, indexer_layers=(0,), max_users=1, max_context=131072, dtype=dtype)
    cache = TtHy4KvCache(mesh_device, cfg)
    layout, k = cache.layout, 2048
    assert cfg.slots == 4096
    start = cfg.capacity - 2 * ROUND  # the last two rounds: the far end of every stripe
    latent, key = _latents(ROUND + 1, seed=95, start=start)

    def timed(label, fn, repeats):
        return _timed(mesh_device, f"{dtype} {label}", fn, repeats)

    d_latent, d_key = _upload(mesh_device, latent[:ROUND]), _upload(mesh_device, key[:ROUND])
    timed("write_prefill 4096 tokens, latent", lambda: cache.write_prefill(0, 0, latent=d_latent, start=start), 10)
    timed("write_prefill 4096 tokens, index key", lambda: cache.write_prefill(0, 0, index_key=d_key, start=start), 10)

    # one decode step, its parts timed apart; every repeat appends the same token at the same position
    p = start + ROUND
    row_latent, row_key = _upload(mesh_device, latent[ROUND:]), _upload(mesh_device, key[ROUND:])
    indices = timed("append_indices (host table + upload, once per step)", lambda: cache.append_indices([p]), 50)
    timed("write_decode latent (per layer)", lambda: cache.write_decode(0, indices, latent=row_latent), 50)
    timed("write_decode index key (per full layer)", lambda: cache.write_decode(0, indices, index_key=row_key), 50)
    appended = ROUND + 1

    picks = start + torch.randperm(appended, generator=torch.Generator().manual_seed(96))[:k]
    picks[0] = p
    d_picks = ttnn.from_torch(
        picks.float().reshape(1, 1, 1, k), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    rows, mine = timed("locate 2048 picks (once per full layer)", lambda: cache.locate(d_picks), 50)
    gathered = timed("gather 2048 latent rows of a 4096-row stripe", lambda: cache.gather(0, rows), 100)
    timed("gather 2048 index-key rows of a 4096-row stripe", lambda: cache.gather(0, rows, INDEX_KEY), 100)

    exact = dtype == "bf16"
    got_mine = torch.stack([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(mine)])[:, 0, 0, 0].bool()
    assert torch.equal(got_mine, selection_table(layout, picks)[1]), "mine differs from kv_layout"
    got = torch.stack([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(gathered)])[:, 0, 0]
    merged = (got * got_mine[..., None]).sum(dim=0)
    _check(f"timing {dtype} gathered latent", latent[picks - start], merged, dtype, exact)
    tail = cache.to_torch(0, 0, cfg.capacity, INDEX_KEY)[start : start + appended]
    _check(f"timing {dtype} index key written", key[:appended], tail, dtype, exact)
    cache.deallocate()


@galaxy
def test_locate_gather_top_2048(mesh_device):
    """The read path at the indexer's real selection size: 2,048 positions out of a 12,288-position context."""
    cache = _cache(mesh_device, "bfp8", max_users=1, layers=(0,))
    layout, length, k = cache.layout, 3 * ROUND, 2048
    latent, key = _latents(length, seed=61)
    cache.write_prefill(0, 0, latent=_upload(mesh_device, latent), index_key=_upload(mesh_device, key))
    picks = torch.randperm(length, generator=torch.Generator().manual_seed(8))[:k]
    d_picks = ttnn.from_torch(
        picks.float().reshape(1, 1, 1, k), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    rows, mine = cache.locate(d_picks)
    want_slot, want_mine = selection_table(layout, picks)
    got_mine = torch.stack([ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(mine)])[:, 0, 0, 0].bool()
    got_rows = torch.stack([ttnn.to_torch(s).long() for s in ttnn.get_device_tensors(rows)])[:, 0]
    assert torch.equal(got_mine, want_mine) and torch.equal(got_rows, want_slot)
    for kind, ref in ((LATENT, latent), (INDEX_KEY, key)):
        got = torch.stack([ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(cache.gather(0, rows, kind))])
        merged = (got[:, 0, 0] * got_mine[..., None]).sum(dim=0)
        _check(f"gather top 2048 bfp8 {kind}", ref[picks], merged, "bfp8", False)
        assert torch.equal(merged, cache.to_torch(0, 0, length, kind)[picks]), "gather differs from readback"


@galaxy
def test_locate_gather_at_one_million(mesh_device):
    """The read path at the size the engine runs it: one layer's latent stripe of a 1,048,576-position
    context (32,768 rows per chip, BFP8, storage left uninitialised), written at both ends, and the
    indexer's 2,048 picks split between the first round and the last. Positions next to 2**20 need
    20 bits of the fp32 arithmetic in locate, and gather's table is the whole stripe.

    Run it in a job of its own: ttnn.embedding untilizes the [1, 1, 32768, 576] table on every call,
    the largest TILE -> row-major conversion this component makes. Warm locate and gather times at
    this size are logged."""
    cfg = KvCacheConfig(num_layers=1, indexer_layers=(), max_users=1, max_context=1 << 20, dtype="bfp8")
    cache = TtHy4KvCache(mesh_device, cfg, zero_init=False)
    layout, capacity, k = cache.layout, cfg.capacity, 2048
    assert (cfg.slots, capacity) == (32768, 1 << 20)
    first, _ = _latents(ROUND, seed=71)
    last, _ = _latents(ROUND, seed=72, start=capacity - ROUND)
    cache.write_prefill(0, 0, latent=_upload(mesh_device, first))
    cache.write_prefill(0, 0, latent=_upload(mesh_device, last), start=capacity - ROUND)

    g = torch.Generator().manual_seed(9)
    picks = torch.cat(
        [
            torch.randperm(ROUND, generator=g)[: k // 2],
            capacity - ROUND + torch.randperm(ROUND, generator=g)[: k // 2],
        ]
    )
    picks[:2] = torch.tensor([0, ROUND - 1])
    picks[-2:] = torch.tensor([capacity - ROUND, capacity - 1])
    picks = picks[torch.randperm(k, generator=g)]
    in_first = picks < ROUND
    ref = torch.where(
        in_first[:, None], first[picks.clamp(max=ROUND - 1)], last[(picks - (capacity - ROUND)).clamp(min=0)]
    )
    d_picks = ttnn.from_torch(
        picks.float().reshape(1, 1, 1, k), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    rows, mine = cache.locate(d_picks)
    want_slot, want_mine = selection_table(layout, picks)
    got_mine = torch.stack([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(mine)])[:, 0, 0, 0].bool()
    got_rows = torch.stack([ttnn.to_torch(t).long() for t in ttnn.get_device_tensors(rows)])[:, 0]
    assert torch.equal(got_mine, want_mine), "mine differs from kv_layout"
    assert torch.equal(got_rows, want_slot), "rows differ from kv_layout"
    start = time.perf_counter()
    gathered = cache.gather(0, rows)
    got = torch.stack([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(gathered)])[:, 0, 0]
    logger.info(
        f"MEASURED gather of {k} rows from a {cfg.slots}-row bfp8 stripe: {time.perf_counter() - start:.2f} s "
        "the first call, with compile and readback"
    )
    merged = (got * got_mine[..., None]).sum(dim=0)
    _check("gather at 1M bfp8 latent", ref, merged, "bfp8", False)
    gathered.deallocate(True)
    _timed(mesh_device, f"bfp8 locate {k} picks at 1M", lambda: cache.locate(d_picks), 20)
    _timed(mesh_device, f"bfp8 gather {k} latent rows of a {cfg.slots}-row stripe", lambda: cache.gather(0, rows), 20)
    cache.deallocate()
