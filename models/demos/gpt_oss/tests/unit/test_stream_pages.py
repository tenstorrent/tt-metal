# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the routed prefill page tables (tt/experts/stream_pages.py): reading tile (expert, column, k)
through the table returns that weight tile's bytes in the decode expert streams, for full and round-only segments, the
bank rotation, the gate|up slot order and the down columns, and in the batch-32 dense layout.

The streams are packed with the real mxfp4_split.pack_columns; the bank / slot ordering and the round-only cut
replicate experts/weights.py _load_split_decode_expert_weights (that function needs ttnn). Both modules
are loaded from their files, so no ttnn is imported: runs on any host with torch.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

_TT = Path(__file__).resolve().parents[2] / "tt"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sp = _load("gpt_oss_stream_pages", _TT / "experts" / "stream_pages.py")
ms = _load("gpt_oss_mxfp4_split", _TT / "fused_decode" / "mxfp4_split.py")

TILE = 32


def _mxfp4(E, K, N, gen):
    """Random exact MXFP4 values [E, K, N]: one power-of-two scale per 32 K values of a column."""
    codes = torch.tensor([0.0, 0.5, 1, 1.5, 2, 3, 4, 6])
    mag = codes[torch.randint(0, 8, (E, K, N), generator=gen)]
    sign = torch.where(torch.rand(E, K, N, generator=gen) < 0.5, -1.0, 1.0)
    scale = torch.pow(2.0, torch.randint(-8, -2, (E, K // TILE, 1, N), generator=gen).float())
    return (sign * mag).reshape(E, K // TILE, TILE, N).mul(scale).reshape(E, K, N)


def _pack_device(w, b, wr, kind, banks, rot, gate_up):
    """One device's bank byte streams, as experts/weights.py builds them (pack -> slot order -> bank rotation -> cut)."""
    p4, p2, r4 = ms.pack_columns(w, b, True, wr)
    cols = torch.cat([p4, p2, r4], -1)  # [E, n_tiles, segment bytes]
    if gate_up:
        half = cols.shape[1] // 2
        P = half // banks
        order = [c for bank in range(banks) for j in range(bank * P, (bank + 1) * P) for c in (j, half + j)]
        cols = cols[:, order]
    C = cols.shape[1] // banks
    full = [cols[:, L * C : (L + 1) * C].reshape(-1) for L in ((bb + rot) % banks for bb in range(banks))]
    seg, part4 = cols.shape[-1], r4.shape[-1]
    if kind == sp.STREAM_ROUND_ONLY:
        return [v.reshape(-1, seg)[:, -part4:].reshape(-1) for v in full]
    return full


def _expected_tiles(wr):
    """[E, n_tiles, k_tiles, 32, 32] the weight tiles of wr [E, K, N] (tile (k, n) of each expert, not transposed)."""
    E, K, N = wr.shape
    return wr.reshape(E, K // TILE, TILE, N // TILE, TILE).permute(0, 3, 1, 2, 4)


# Small shapes with the model's structure: inter_pad / 32 = 2 * banks pairs -> P = 2 slots pairs per bank; hidden
# not a multiple of 32 * banks, so the down stream has padded columns.
E, H, I_PAD, BANKS, ROT = 3, 160, 256, 4, 3


@pytest.fixture(scope="module")
def weights():
    gen = torch.Generator().manual_seed(0)
    gate_up = _mxfp4(E, H, 2 * I_PAD, gen)  # [gate shard | up shard] of one device
    gate_up_round = ms.round_k(torch.randn(E, H, 2 * I_PAD, generator=gen) * 0.02)  # a K-grouped bf4 rounding
    gu_bias = torch.randn(E, 2 * I_PAD, generator=gen) * 0.1
    down = _mxfp4(E, I_PAD, H, gen)
    down_round = ms.round_k(torch.randn(E, I_PAD, H, generator=gen) * 0.02)
    dn_bias = torch.randn(E, H, generator=gen) * 0.1
    npad = -(-H // TILE // BANKS) * BANKS * TILE - H
    pad = lambda t: torch.nn.functional.pad(t, (0, npad))
    return dict(
        gu=(gate_up, gu_bias, gate_up_round),
        dn=(pad(down), pad(dn_bias), pad(down_round)),
        dn_bias=dn_bias,
        gu_bias=gu_bias,
    )


@pytest.mark.parametrize("kind", [sp.STREAM_FULL, sp.STREAM_ROUND_ONLY])
@pytest.mark.parametrize("kind_dn", [sp.STREAM_FULL, sp.STREAM_ROUND_ONLY])
def test_stream_page_table_reads_rounded_tiles(weights, kind, kind_dn):
    gu_banks = _pack_device(*weights["gu"], kind, BANKS, ROT, gate_up=True)
    dn_banks = _pack_device(*weights["dn"], kind_dn, BANKS, ROT, gate_up=False)
    gu_l = sp.split_stream_layout("gate_up", kind, E, H, I_PAD, BANKS, ROT)
    dn_l = sp.split_stream_layout("down", kind_dn, E, H, I_PAD, BANKS, ROT)
    assert all(len(b) == gu_l.bank_bytes for b in gu_banks)
    assert all(len(b) == dn_l.bank_bytes for b in dn_banks)
    n_gu, n_dn = I_PAD // TILE, H // TILE
    table = sp.stream_page_table(gu_l, dn_l, n_gu, n_dn)
    assert table.shape == (E, 2 * (2 * n_gu + n_dn))
    gate_e, up_e, down_e = sp.table_entries(table, n_gu, n_dn)

    gu_tiles = _expected_tiles(weights["gu"][2])  # [E, 2 * n_gu, kx, 32, 32]
    dn_tiles = _expected_tiles(weights["dn"][2])
    for k in range(H // TILE):
        for entries, cols in ((gate_e, gu_tiles[:, :n_gu]), (up_e, gu_tiles[:, n_gu:])):
            raw = sp.read_tiles(gu_banks, entries, k, sp.BFP4_BYTES, sp.BFP4_BYTES)
            # the bytes of the packed tile, and its values: the stored tile is the weight tile transposed
            want = ms.pack_bfp(cols[:, :, k].transpose(-1, -2).reshape(-1, TILE, TILE), 3)
            assert torch.equal(raw.reshape(-1, sp.BFP4_BYTES), want)
            got = ms.unpack_bfp(raw.reshape(-1, sp.BFP4_BYTES), 3).reshape(E, n_gu, TILE, TILE)
            assert torch.equal(got.transpose(-1, -2), cols[:, :, k].float())
    for k in range(I_PAD // TILE):
        raw = sp.read_tiles(dn_banks, down_e, k, sp.BFP4_BYTES, sp.BFP4_BYTES)
        got = ms.unpack_bfp(raw.reshape(-1, sp.BFP4_BYTES), 3).reshape(E, n_dn, TILE, TILE)
        assert torch.equal(got.transpose(-1, -2), dn_tiles[:, :n_dn, k].float())

    # biases: the decode value (the bias tile's two residual terms), within BFP4 two-term error of the bf16 bias
    gu_bias = weights["gu_bias"]
    decode_gu = ms.bias_tiles_t(gu_bias.reshape(E, -1, TILE)).sum(-1).reshape(E, -1)
    gate_b = sp.stream_column_biases(gu_banks, gate_e, H // TILE, ms.unpack_bfp).reshape(E, -1)
    up_b = sp.stream_column_biases(gu_banks, up_e, H // TILE, ms.unpack_bfp).reshape(E, -1)
    assert torch.equal(gate_b, decode_gu[:, :I_PAD]) and torch.equal(up_b, decode_gu[:, I_PAD:])
    assert torch.allclose(gate_b, gu_bias[:, :I_PAD], atol=0, rtol=1 / 32)
    down_b = sp.stream_column_biases(dn_banks, down_e, I_PAD // TILE, ms.unpack_bfp).reshape(E, -1)
    assert torch.equal(down_b, ms.bias_tiles_t(weights["dn_bias"].reshape(E, -1, TILE)).sum(-1).reshape(E, -1))


def test_bank_rotation_and_slots():
    """Entries follow the documented placement: gate column n = slot 2 (n % P) of logical bank n // P on DRAM bank
    (n // P - rot) % banks; up the next slot; down column n = segment n % Cd of logical bank n // Cd."""
    lay = sp.split_stream_layout("gate_up", sp.STREAM_ROUND_ONLY, 32, 2880, 768, 8, 7)
    P = 3
    for e in (0, 5, 31):
        for n in range(24):
            b, o = sp.gate_up_entry(lay, e, n, False)
            assert b == (n // P - 7) % 8 and o == (e * 2 * P + 2 * (n % P)) * lay.part_bytes
            bu, ou = sp.gate_up_entry(lay, e, n, True)
            assert bu == b and ou == o + lay.part_bytes
    dn = sp.split_stream_layout("down", sp.STREAM_FULL, 32, 2880, 768, 8, 7)
    for e in (0, 31):
        for n in range(90):
            b, o = sp.down_entry(dn, e, n)
            assert b == (n // 12 - 7) % 8
            assert o == (e * 12 + n % 12) * dn.seg_bytes + dn.seg_bytes - dn.part_bytes


def test_model_layout_numbers():
    """gpt-oss-20b at TP 4 on 8 banks: the numbers of experts/weights.py _load_split_decode_expert_weights."""
    E, H, I_pad, banks = 32, 2880, 768, 8
    kx, kd, P, Cd = 90, 24, 3, 12
    gu_col = (kx + 1) * 576 * 2 + kx * 320
    dn_seg = (kd + 1) * 576 * 2 + kd * 320
    full_gu = sp.split_stream_layout("gate_up", sp.STREAM_FULL, E, H, I_pad, banks, 7)
    full_dn = sp.split_stream_layout("down", sp.STREAM_FULL, E, H, I_pad, banks, 7)
    ro_gu = sp.split_stream_layout("gate_up", sp.STREAM_ROUND_ONLY, E, H, I_pad, banks, 7)
    ro_dn = sp.split_stream_layout("down", sp.STREAM_ROUND_ONLY, E, H, I_pad, banks, 7)
    assert (full_gu.seg_bytes, full_gu.part_bytes, full_gu.segs_per_expert) == (gu_col, (kx + 1) * 576, 2 * P)
    assert (full_dn.seg_bytes, full_dn.part_bytes, full_dn.segs_per_expert) == (dn_seg, (kd + 1) * 576, Cd)
    assert full_gu.bank_bytes == E * 2 * P * gu_col and full_dn.bank_bytes == E * Cd * dn_seg
    assert ro_gu.bank_bytes == full_gu.bank_bytes // gu_col * (kx + 1) * 576
    assert ro_dn.bank_bytes == full_dn.bank_bytes // dn_seg * (kd + 1) * 576
    for lay in (full_gu, full_dn, ro_gu, ro_dn):
        assert lay.bank_bytes % 64 == 0  # whole 64-byte pages of the uint8 stream tensor
    assert ms.BFP4_BYTES == sp.BFP4_BYTES and ms.BFP2_BYTES == sp.BFP2_BYTES
    table = sp.stream_page_table(full_gu, full_dn, 24, 90)
    assert table.shape == (32, 2 * (48 + 90))
    assert int(table[..., 1::2].max()) < full_gu.bank_bytes and int(table.max()) < (1 << 31)
    for kind in (sp.STREAM_FULL, sp.STREAM_ROUND_ONLY):  # the op needs 64 B aligned addresses
        gu = sp.split_stream_layout("gate_up", kind, E, H, I_pad, banks, 7)
        dn = sp.split_stream_layout("down", kind, E, H, I_pad, banks, 7)
        assert (sp.stream_page_table(gu, dn, 24, 90)[..., 1::2] % 64 == 0).all()


def test_bank_bytes_from_host():
    """The [rows, banks * 64] host copy of a uint8 stream tensor -> each bank's contiguous bytes."""
    banks, rows = 4, 5
    per_bank = [torch.randint(0, 256, (rows * 64,), dtype=torch.uint8) for _ in range(banks)]
    host = torch.cat([b.reshape(rows, 64) for b in per_bank], dim=-1)
    got = sp.bank_bytes_from_host(host, banks)
    assert all(torch.equal(g, b) for g, b in zip(got, per_bank))


def test_dense_page_table():
    """Batch-32 dense layout (gate / up [H, E * il], down [E * il, H], DRAM width-sharded): the table entry + k * stride
    is tile (k, column) of the expert's slice, tracked here by tile ids in a simulated sharded buffer."""
    E, il, H, banks = 4, 64, 96, 4
    tb = 8  # bytes per simulated tile
    gu_w = -(-(E * il) // (TILE * banks))  # shard width in tiles (_dram_width_sharded)
    dn_w = -(-H // (TILE * banks))

    def sharded(K, N, W, base):
        """Per bank bytes of a [K, N] tile grid width-sharded with W tile columns per bank; each tile holds its id
        base + (k * n_tiles + n) as a little-endian int64."""
        kt, nt = K // TILE, N // TILE
        bufs = []
        for b in range(banks):
            ids = torch.full((kt, W), -1, dtype=torch.int64)
            for c in range(W):
                if b * W + c < nt:
                    ids[:, c] = base + torch.arange(kt) * nt + b * W + c
            bufs.append(ids.reshape(-1).view(torch.uint8))
        return bufs

    gate = sharded(H, E * il, gu_w, 0)
    up_delta = 1 << 20
    up = sharded(H, E * il, gu_w, 10**6)
    down = sharded(E * il, H, dn_w, 2 * 10**6)
    # gate and up in one address space per bank: up at gate + up_delta
    gu = [torch.cat([g, torch.zeros(up_delta - len(g), dtype=torch.uint8), u]) for g, u in zip(gate, up)]
    table = sp.dense_page_table(E, il, H, gu_w, dn_w, tb, up_delta)
    n_gu, n_dn = il // TILE, H // TILE
    ge, ue, de = sp.table_entries(table, n_gu, n_dn)
    tile_id = lambda bufs, entries, k, stride: sp.read_tiles(bufs, entries, k, stride, tb).view(torch.int64)[..., 0]
    for k in range(H // TILE):
        want = torch.tensor([[k * (E * il // TILE) + e * n_gu + n for n in range(n_gu)] for e in range(E)])
        assert torch.equal(tile_id(gu, ge, k, gu_w * tb), want)
        assert torch.equal(tile_id(gu, ue, k, gu_w * tb), want + 10**6)
    for k in range(il // TILE):
        want = torch.tensor([[(e * n_gu + k) * n_dn + n for n in range(n_dn)] for e in range(E)]) + 2 * 10**6
        assert torch.equal(tile_id(down, de, k, dn_w * tb), want)


def test_dense_page_table_needs_whole_tiles():
    with pytest.raises(AssertionError, match="not whole tiles"):
        sp.dense_page_table(32, 720, 2880, 90, 12, 576)


def test_as_uint32_words():
    t = torch.tensor([[0, 1, (1 << 31) + 5, (1 << 32) - 1]])
    w = sp.as_uint32_words(t)
    assert w.dtype == torch.int32 and torch.equal(w.to(torch.int64) & 0xFFFFFFFF, t)
