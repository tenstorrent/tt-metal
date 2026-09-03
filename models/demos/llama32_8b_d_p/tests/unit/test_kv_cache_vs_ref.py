# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gate `G-KV` — `tt/attention/kv_cache.py` write + read-back, at the real `head_dim = 128`.

The KV cache **is** the output of prefill, so this file gates the point of the whole package. It
proves four separate things, and the gate needs all four (``03_OUTLINE.md`` §3.11, Appendix F.6):

1. **Round-trip PCC** — random K/V written into every ``(user, layer)`` slot comes back correct.
   ``PCC >= 0.99`` at ``bfloat8_b`` (``DEC-017``); the ``bfloat16`` number is *recorded*, since bf16
   cannot ship (``models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:77-81`` asserts a bf8_b cache
   for the chunked ring path).
2. **Exact block-cyclic read-back, not just PCC** — Appendix F.6's specific demand. Written with a
   bf16 cache and each token row carrying its own global index as its value, every row must read
   back **bit-exactly** at the right position. A PCC of 0.999 cannot distinguish "right values,
   right places" from "right values, two rows swapped".
3. **Written-region-only** — the three collateral-write failures a PCC test cannot see:
   the unwritten pad tail stays zero; an earlier chunk is bit-identical after a later chunk's write;
   and another ``(user, layer)`` slot is untouched. An out-of-region write is exactly the bug that
   would first surface as "one layer runs on garbage" three phases later.
4. **``head_dim = 128`` really is the only delta from the gpt-oss template.** gpt-oss runs 64. The
   DRAM shard spec is ``shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK(=32), head_dim]``
   (``models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:87``), so 128 **doubles the shard row**.
   ``test_dram_shard_geometry_at_head_dim_128`` pins the geometry and the bank count.

**Input distribution** (``DEC-026``): ``randn`` for the PCC cases; exact integer position indices
for the layout case. **Reference dtype policy:** the reference is the fp32 tensor that was written;
the ``bfloat8_b`` floor is the same tensor round-tripped through ttnn's own quantiser, so the
measured PCC is compared against the dtype's own limit rather than an absolute guess (``DEC-032``).

Single card ``(1,1)``: ``sp = 1``, ``tp = 1``, so one KV head per chip and the block-cyclic layout
degenerates to the identity. The layout *arithmetic* at ``sp = 4`` is therefore proved host-only in
``test_blockcyclic_positions_are_an_exact_inverse``, and the device path exercises the non-zero
``kv_actual`` chunk offsets.

Run:
    pytest models/demos/llama32_8b_d_p/tests/unit/test_kv_cache_vs_ref.py -x -q
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.common.prefill.runners.migration import get_num_dram_banks
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder, blockcyclic_positions
from models.demos.llama32_8b_d_p.tests.test_factory import TestFactory
from models.demos.llama32_8b_d_p.tests.unit.test_mlp_vs_ref import err_ratio, quantize_like_device
from models.demos.llama32_8b_d_p.tt.attention.kv_cache import (
    LLAMA_HEAD_DIM,
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    allocate_kv_cache,
    write_kv_chunk,
)

PCC_THRESHOLD = 0.99  # 03_OUTLINE.md §5, bf8_b
MAX_ERR_RATIO = 3.0  # DEC-032 / DEC-034
HEAD_DIM = LLAMA_HEAD_DIM  # 128 — the whole point of this gate (Appendix F.6)


def _to_chunk(mesh_device, nat, *, nkv, seq_len, head_dim, sp_axis, tp_axis, dtype=ttnn.bfloat16):
    """``[nkv, seq, head_dim]`` -> device ``[1, nkv, seq, head_dim]``, seq on SP rows, heads on TP cols."""
    rows, cols = tuple(mesh_device.shape)
    dims = [None, None]
    dims[sp_axis] = 2
    dims[tp_axis] = 1
    return ttnn.from_torch(
        nat.reshape(1, nkv, seq_len, head_dim),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=tuple(dims)),
    )


def _gather(mesh_device, cache_tensor, slot, col, positions, *, upto):
    """Read one ``(slot, head-col)`` back, invert the block-cyclic layout, return natural order."""
    rows, cols = tuple(mesh_device.shape)
    dts = ttnn.get_device_tensors(cache_tensor)
    dev = torch.cat([ttnn.to_torch(dts[r * cols + col])[slot, 0].float() for r in range(rows)], dim=0)
    nat = torch.empty_like(dev)
    nat[positions] = dev
    return nat[:upto]


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("num_users, num_layers", [(2, 2)], ids=["u2xl2"])
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["cbf8_b", "cbf16"])
@torch.no_grad()
def test_kv_cache_write_read_vs_ref(mesh_device, num_users, num_layers, seq_len, cache_dtype, reset_seeds):
    """Write GQA K/V into every ``(user, layer)`` slot at ``head_dim=128``, read back, PCC vs torch."""
    rows, cols = tuple(mesh_device.shape)
    sp, tp = rows, cols
    sp_axis, tp_axis = 0, 1  # matches MeshConfig(tp_axis=1) -> sp_axis=0
    nkv = tp  # one KV head per TP col (the per-chip cache head slot is 1)

    assert seq_len % (ttnn.TILE_SIZE * sp) == 0
    max_seq_len = seq_len  # one chunk == the whole cache => block-cyclic is the identity

    sent_k = torch.randn(num_users, num_layers, nkv, seq_len, HEAD_DIM)
    sent_v = torch.randn(num_users, num_layers, nkv, seq_len, HEAD_DIM)

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp_axis=sp_axis,
        num_users=num_users,
        head_dim=HEAD_DIM,
        cache_dtype=cache_dtype,
    )
    assert kv_cache.head_dim == HEAD_DIM == 128
    assert tuple(kv_cache.k.shape) == (num_users * num_layers, 1, max_seq_len // sp, HEAD_DIM)

    for u in range(num_users):
        for layer in range(num_layers):
            tt_k = _to_chunk(
                mesh_device,
                sent_k[u, layer],
                nkv=nkv,
                seq_len=seq_len,
                head_dim=HEAD_DIM,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
            )
            tt_v = _to_chunk(
                mesh_device,
                sent_v[u, layer],
                nkv=nkv,
                seq_len=seq_len,
                head_dim=HEAD_DIM,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
            )
            write_kv_chunk(kv_cache, tt_k, tt_v, slot_idx=u, layer_idx=layer, kv_actual=0, sp_axis=sp_axis)
            tt_k.deallocate(True)
            tt_v.deallocate(True)
    ttnn.synchronize_device(mesh_device)

    positions = blockcyclic_positions(sp, seq_len, max_seq_len)
    worst = 1.0
    for u in range(num_users):
        for layer in range(num_layers):
            slot = u * num_layers + layer
            host_k = torch.stack(
                [_gather(mesh_device, kv_cache.k, slot, c, positions, upto=seq_len) for c in range(nkv)], dim=0
            )
            host_v = torch.stack(
                [_gather(mesh_device, kv_cache.v, slot, c, positions, upto=seq_len) for c in range(nkv)], dim=0
            )
            ok_k, pcc_k = comp_pcc(sent_k[u, layer], host_k, PCC_THRESHOLD)
            ok_v, pcc_v = comp_pcc(sent_v[u, layer], host_v, PCC_THRESHOLD)
            # The floor: the same tensor through ttnn's own quantiser. The cache stores K/V and
            # nothing else, so the dtype IS the whole error budget — the measured PCC should sit on
            # top of this, and a gap means the write/read placement, not the dtype (DEC-032).
            _, floor_k = comp_pcc(
                sent_k[u, layer],
                quantize_like_device(sent_k[u, layer].reshape(1, nkv, seq_len, HEAD_DIM), cache_dtype).reshape(
                    nkv, seq_len, HEAD_DIM
                ),
                0.0,
            )
            ratio = err_ratio(pcc_k, floor_k)
            worst = min(worst, float(pcc_k), float(pcc_v))
            logger.info(
                f"[G-KV] {cache_dtype} (user={u}, layer={layer}) slot={slot}: K PCC = {pcc_k} | "
                f"V PCC = {pcc_v} | dtype floor (K) = {floor_k} | err ratio = {ratio:.2f}x | "
                f"threshold {PCC_THRESHOLD}"
            )
            assert ok_k, f"[G-KV] K cache mismatch (user={u}, layer={layer}): {pcc_k}"
            assert ok_v, f"[G-KV] V cache mismatch (user={u}, layer={layer}): {pcc_v}"
            assert ratio <= MAX_ERR_RATIO, (
                f"[G-KV] K at {cache_dtype} slot {slot} sits {ratio:.1f}x off the dtype floor "
                f"{floor_k}; the cache only stores values, so a gap is a placement bug (DEC-034)"
            )
    logger.info(f"[G-KV] {cache_dtype}: {num_users}x{num_layers} slots, worst PCC = {worst}, head_dim = {HEAD_DIM}")


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("chunk, n_chunks", [(64, 4)], ids=["c64x4"])
@torch.no_grad()
def test_kv_cache_readback_is_positionally_exact(mesh_device, chunk, n_chunks, reset_seeds):
    """Appendix F.6's demand: assert the **block-cyclic read-back**, not just a PCC.

    Each token row carries its own global position as its value, and the cache is ``bfloat16`` so
    small integers are exact. Every row must then read back **bit-identically** at the position it
    was written to. A high PCC survives a permutation of nearby rows; this does not — and a row
    permutation is precisely what a wrong ``head_dim``-sized shard stride would produce.

    ``max_seq_len`` is capped at **256** on purpose: ``bfloat16`` carries 8 significant bits, so
    integers above 256 are **not** exactly representable (measured — position 257 reads back as
    256, greatest relative difference 1/257). Four chunks of 64 keep every position exact while
    still exercising four distinct ``kv_actual`` offsets (0, 64, 128, 192). Head identity goes in
    its own lane block rather than being added to the position, for the same reason.
    """
    rows, cols = tuple(mesh_device.shape)
    sp, tp = rows, cols
    sp_axis, tp_axis = 0, 1
    nkv = tp
    max_seq_len = chunk * n_chunks

    assert max_seq_len <= 256, "bfloat16 represents integers exactly only up to 256 (see docstring)"
    # Lanes [0, 64) hold the global position p; lanes [64, 128) hold the head index. Both stay
    # small enough to be exact in bf16, and a head mix-up shows up in the second block.
    pos = torch.arange(max_seq_len, dtype=torch.float32).reshape(1, max_seq_len, 1)
    head = torch.arange(nkv, dtype=torch.float32).reshape(nkv, 1, 1)
    half = HEAD_DIM // 2
    sent = torch.cat(
        [pos.repeat(nkv, 1, half), head.repeat(1, max_seq_len, half)], dim=-1
    )  # [nkv, max_seq_len, HEAD_DIM]

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=max_seq_len,
        sp_axis=sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
        cache_dtype=ttnn.bfloat16,
    )
    for i in range(n_chunks):
        lo = i * chunk  # kv_actual: 0, chunk, 2*chunk — all tile-aligned
        tt = _to_chunk(
            mesh_device,
            sent[:, lo : lo + chunk],
            nkv=nkv,
            seq_len=chunk,
            head_dim=HEAD_DIM,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
        )
        write_kv_chunk(kv_cache, tt, tt, slot_idx=0, layer_idx=0, kv_actual=lo, sp_axis=sp_axis)
        tt.deallocate(True)
    ttnn.synchronize_device(mesh_device)

    positions = blockcyclic_positions(sp, chunk, max_seq_len)
    for c in range(nkv):
        got = _gather(mesh_device, kv_cache.k, 0, c, positions, upto=max_seq_len)
        torch.testing.assert_close(got, sent[c], rtol=0.0, atol=0.0)
    logger.info(
        f"[G-KV] positional read-back EXACT (rtol=atol=0) for {n_chunks} x {chunk} tokens at "
        f"head_dim={HEAD_DIM}: every one of {max_seq_len} rows is at its own position"
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_writes_touch_only_their_own_region(mesh_device, reset_seeds):
    """No collateral writes: the pad tail, an earlier chunk, and another slot all stay put.

    This is the half of ``G-KV`` a PCC cannot see. All three failures below present identically —
    "a later layer reads garbage" — and only at ``G-MESH-KV``, three phases downstream.
    """
    rows, cols = tuple(mesh_device.shape)
    sp, tp = rows, cols
    sp_axis, tp_axis = 0, 1
    nkv = tp
    chunk = 128
    num_layers = 2
    max_seq_len = chunk * 3  # write 2 chunks, leave one chunk of pad tail

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp_axis=sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
        cache_dtype=ttnn.bfloat16,
    )

    def _raw(slot):
        """The raw per-chip cache rows for one slot, in device (block-cyclic) order."""
        dts = ttnn.get_device_tensors(kv_cache.k)
        return torch.cat([ttnn.to_torch(dts[r * cols])[slot, 0].float() for r in range(rows)], dim=0)

    sent = torch.randn(nkv, max_seq_len, HEAD_DIM)

    def _write(lo, layer_idx):
        tt = _to_chunk(
            mesh_device,
            sent[:, lo : lo + chunk],
            nkv=nkv,
            seq_len=chunk,
            head_dim=HEAD_DIM,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
        )
        write_kv_chunk(kv_cache, tt, tt, slot_idx=0, layer_idx=layer_idx, kv_actual=lo, sp_axis=sp_axis)
        tt.deallocate(True)
        ttnn.synchronize_device(mesh_device)

    # --- chunk 0 into layer 0 ---
    _write(0, layer_idx=0)
    after_first = _raw(0).clone()
    positions = blockcyclic_positions(sp, chunk, max_seq_len)

    # (a) another (user, layer) slot is still exactly zero.
    other = _raw(1)
    assert other.abs().max().item() == 0.0, (
        f"writing (slot 0, layer 0) changed slot 1: max|v| = {other.abs().max().item()}; "
        f"slot = user_id*num_layers + layer_idx addressing is wrong"
    )

    # --- chunk 1 into layer 0 ---
    _write(chunk, layer_idx=0)
    after_second = _raw(0)

    # (b) the rows chunk 0 wrote are bit-identical after chunk 1's write.
    first_rows = torch.as_tensor([p for p in positions.tolist() if p < chunk])
    torch.testing.assert_close(after_second[first_rows], after_first[first_rows], rtol=0.0, atol=0.0)

    # (c) the never-written pad tail [2*chunk, max_seq_len) is still exactly zero.
    tail_rows = torch.as_tensor([i for i, p in enumerate(positions.tolist()) if p >= 2 * chunk])
    assert len(tail_rows) == max_seq_len - 2 * chunk
    tail_max = after_second[tail_rows].abs().max().item()
    assert tail_max == 0.0, (
        f"the unwritten pad tail is not zero (max|v| = {tail_max}); a write ran past kv_actual + "
        f"chunk, which at head_dim=128 is exactly what a stale 64-wide shard stride would do"
    )

    # And the second chunk really did land (else (b) and (c) would pass on an all-zero cache).
    got = _gather(mesh_device, kv_cache.k, 0, 0, positions, upto=2 * chunk)
    _, pcc = comp_pcc(sent[0, : 2 * chunk], got, PCC_THRESHOLD)
    logger.info(
        f"[G-KV] written-region-only: other slot exactly 0; chunk 0 bit-identical after chunk 1's "
        f"write; pad tail [{2 * chunk}, {max_seq_len}) exactly 0; both chunks readable at "
        f"PCC = {pcc}"
    )
    assert float(pcc) >= PCC_THRESHOLD, f"the two written chunks did not read back: {pcc}"


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_dram_shard_geometry_at_head_dim_128(mesh_device, reset_seeds, expect_error):
    """``head_dim`` 64 -> 128 is the only delta from the template, and it doubles the shard row.

    Pins the three numbers P10's producer-side packed-K/V reader depends on
    (``bringup_log/08_PREFILL_INTEGRATION.md``): the 32-token contiguous DRAM block, the shard row
    ``[1, 1, 32, head_dim]``, and the bank count ``mesh_device.dram_grid_size().x``
    (``models/demos/common/prefill/runners/migration.py:338`` — measured **8** on this Blackhole).
    Also checks that the two divisibility constraints refuse rather than round.
    """
    banks = get_num_dram_banks(mesh_device)
    assert NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 32, "P10's reader assumes a 32-token DRAM block"
    assert LLAMA_HEAD_DIM == 128 and LLAMA_HEAD_DIM % ttnn.TILE_SIZE == 0
    logger.info(
        f"[G-KV] DRAM shard geometry: shard_shape = [1, 1, {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}, "
        f"{LLAMA_HEAD_DIM}] = {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK * LLAMA_HEAD_DIM} values "
        f"({LLAMA_HEAD_DIM // ttnn.TILE_SIZE} tiles wide, 2x gpt-oss's 64); DRAM banks = {banks}"
    )

    sp = mesh_device.shape[0]
    with expect_error(AssertionError, "multiple of TILE_SIZE\\*sp"):
        allocate_kv_cache(mesh_device, num_layers=1, max_seq_len=100, head_dim=HEAD_DIM)
    with expect_error(AssertionError, "head_dim"):
        allocate_kv_cache(mesh_device, num_layers=1, max_seq_len=ttnn.TILE_SIZE * sp, head_dim=100)

    kv_cache = allocate_kv_cache(mesh_device, num_layers=1, max_seq_len=128, head_dim=HEAD_DIM)
    tt = _to_chunk(mesh_device, torch.randn(1, 128, 64), nkv=1, seq_len=128, head_dim=64, sp_axis=0, tp_axis=1)
    with expect_error(AssertionError, "head_dim"):
        write_kv_chunk(kv_cache, tt, tt, slot_idx=0, layer_idx=0, kv_actual=0, sp_axis=0)
    with expect_error(AssertionError, "tile-aligned"):
        good = _to_chunk(
            mesh_device,
            torch.randn(1, 128, HEAD_DIM),
            nkv=1,
            seq_len=128,
            head_dim=HEAD_DIM,
            sp_axis=0,
            tp_axis=1,
        )
        write_kv_chunk(kv_cache, good, good, slot_idx=0, layer_idx=0, kv_actual=17, sp_axis=0)
    with expect_error(AssertionError, "layer_idx"):
        good2 = _to_chunk(
            mesh_device,
            torch.randn(1, 128, HEAD_DIM),
            nkv=1,
            seq_len=128,
            head_dim=HEAD_DIM,
            sp_axis=0,
            tp_axis=1,
        )
        write_kv_chunk(kv_cache, good2, good2, slot_idx=0, layer_idx=5, kv_actual=0, sp_axis=0)


@torch.no_grad()
def test_blockcyclic_positions_are_an_exact_inverse():
    """Host-only: the SP layout arithmetic the ``(1,1)`` device tests degenerate away.

    At ``sp = 1`` the block-cyclic reorder is the identity, so this is the **only** coverage of the
    layout the SP-sharded write/read actually relies on. Mirrors
    ``models/demos/gpt_oss_d_p/tests/unit/test_kv_cache_vs_ref.py:114``, at ``head_dim = 128``.
    """
    sp = TestFactory.TARGET_SP  # 4
    chunk_local = ttnn.TILE_SIZE  # 32 — tile-aligned per-chip block
    chunk_size_global = chunk_local * sp  # 128
    seq_len = 3 * chunk_size_global

    nat = torch.arange(seq_len).reshape(seq_len, 1).repeat(1, HEAD_DIM)
    reordered = block_cyclic_reorder(nat, chunk_local, sp, seq_dim=0)
    positions = blockcyclic_positions(sp, chunk_size_global, seq_len)

    assert torch.equal(torch.sort(positions).values, torch.arange(seq_len)), "not a permutation"
    assert not torch.equal(positions, torch.arange(seq_len)), "expected a non-identity reorder at sp>1"
    assert torch.equal(reordered[:, 0], positions), "block_cyclic_reorder and blockcyclic_positions disagree"

    recovered = torch.empty_like(nat)
    recovered[positions] = reordered
    assert torch.equal(recovered, nat), "the block-cyclic scatter/gather did not round-trip"
    logger.info(
        f"[G-KV] block-cyclic layout exact-inverse at sp={sp}, chunk={chunk_size_global}, "
        f"seq_len={seq_len}, head_dim={HEAD_DIM}"
    )
