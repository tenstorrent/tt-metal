# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Page tables through which the routed prefill experts (prefill_routed.py, TP mode) read the decode experts' weights.

unified_routed_expert_moe with weight_pages reads tile k of column n of local expert e at byte offset + k * k_stride
in DRAM bank `bank`, where (bank, offset) is entry n of e's gate / up / down block of the table:
    table[e] = [gate n = 0..n_gu-1][up n = 0..n_gu-1][down n = 0..n_dn-1], each entry two uint32 (bank, offset),
offsets relative to the weight source tensor's buffer address (the same address in every bank of a DRAM-sharded buffer).

Decode expert streams (experts/weights.py _load_split_decode_expert_weights, fused_decode/mxfp4_split.py), per device:
  * bank b holds logical bank L = (b + rot) % banks, its bytes contiguous (uint8 width-sharded, 64-byte pages);
  * gate|up bank bytes: for expert e, for slot s in 0..2P-1 (slot 2r = gate column L * P + r, slot 2r + 1 = up column
    L * P + r): one segment; down bank bytes: for expert e, for c in 0..Cd-1 (output column L * Cd + c): one segment;
  * a full segment is [BFP4 exact part][BFP2 part][rounded part], a round-only segment only the rounded part;
  * the rounded part is (k_tiles + 1) BFP4 tiles: k = 0..k_tiles-1 weight tiles (stored transposed within the tile),
    then the bias tile (row n holds output column n's two residual terms at K positions 0 and 16).

Pure torch, no ttnn: host tests (tests/unit/test_stream_pages.py) load this file directly.
"""

from dataclasses import dataclass

import torch

TILE = 32
BFP4_BYTES = 576  # bfloat4_b tile: 64 exponent bytes + 512 mantissa bytes (mxfp4_split.BFP4_BYTES)
BFP2_BYTES = 320  # BFP2 tile (mxfp4_split.BFP2_BYTES)
BFP8_BYTES = 1088
# Stream kinds of a decode expert stream with a rounded part (experts/weights.py _load_split_decode_expert_weights).
STREAM_FULL, STREAM_ROUND_ONLY = "full", "round_only"
# K positions of the bias tile's residual terms (mxfp4_split.bias_tiles_t)
BIAS_TERM_COLUMNS = (0, 16)


@dataclass(frozen=True)
class SplitStreamLayout:
    """Byte layout of one decode expert stream on one device: num_experts experts, segs_per_expert segments per
    expert in every bank, and in each segment the rounded part of k_tiles weight tiles + 1 bias tile."""

    kind: str
    num_experts: int
    banks: int
    rot: int
    segs_per_expert: int
    k_tiles: int

    @property
    def part_bytes(self):
        return (self.k_tiles + 1) * BFP4_BYTES

    @property
    def seg_bytes(self):
        """Bytes of one segment as packed: [BFP4 + bias][BFP2][rounded BFP4 + bias], or the rounded part alone."""
        if self.kind == STREAM_ROUND_ONLY:
            return self.part_bytes
        return 2 * self.part_bytes + self.k_tiles * BFP2_BYTES

    @property
    def segs_per_bank(self):
        return self.num_experts * self.segs_per_expert

    @property
    def bank_bytes(self):
        return self.segs_per_bank * self.seg_bytes

    def part_offset(self, expert, seg):
        """Byte offset in its bank of the rounded part (weight tile k = 0) of segment `seg` of `expert`."""
        assert 0 <= expert < self.num_experts and 0 <= seg < self.segs_per_expert, (expert, seg)
        i = expert * self.segs_per_expert + seg
        return i * self.seg_bytes + self.seg_bytes - self.part_bytes

    def physical_bank(self, logical):
        """DRAM bank holding logical bank `logical` (bank b holds (b + rot) % banks)."""
        return (logical - self.rot) % self.banks


def split_stream_layout(stream, kind, num_experts, hidden, inter_pad, banks, rot):
    """SplitStreamLayout of a decode gate|up ("gate_up") or down ("down") stream of `kind` (STREAM_*), for one TP shard
    of inter_pad (padded) intermediate columns."""
    assert kind in (STREAM_FULL, STREAM_ROUND_ONLY), kind
    assert hidden % TILE == 0 and inter_pad % TILE == 0, (hidden, inter_pad)
    if stream == "gate_up":
        pairs = inter_pad // TILE
        assert pairs % banks == 0, f"{pairs} gate|up column pairs do not split over {banks} banks"
        return SplitStreamLayout(kind, num_experts, banks, rot, 2 * (pairs // banks), hidden // TILE)
    assert stream == "down", stream
    cols = -(-hidden // TILE // banks)  # fused_decode/stream.py columns_per_bank
    return SplitStreamLayout(kind, num_experts, banks, rot, cols, inter_pad // TILE)


def gate_up_entry(layout, expert, n, up):
    """(bank, offset) of gate (up=False) or up (up=True) tile column n of `expert` in a gate|up stream."""
    P = layout.segs_per_expert // 2
    L, r = divmod(n, P)
    assert L < layout.banks, (n, P, layout.banks)
    return layout.physical_bank(L), layout.part_offset(expert, 2 * r + int(up))


def down_entry(layout, expert, n):
    """(bank, offset) of output tile column n of `expert` in a down stream."""
    L, c = divmod(n, layout.segs_per_expert)
    assert L < layout.banks, (n, layout.segs_per_expert, layout.banks)
    return layout.physical_bank(L), layout.part_offset(expert, c)


def stream_page_table(gate_up, down, n_gate_up, n_down):
    """[E, 2 * (2 * n_gate_up + n_down)] int64 page table of the decode streams (layouts gate_up and down): per expert
    the gate, up and down blocks of (bank, offset) entries (module docstring). k strides: BFP4_BYTES for both."""
    assert gate_up.num_experts == down.num_experts
    assert n_gate_up <= gate_up.segs_per_expert // 2 * gate_up.banks, n_gate_up
    assert n_down <= down.segs_per_expert * down.banks, n_down
    rows = []
    for e in range(gate_up.num_experts):
        row = [gate_up_entry(gate_up, e, n, False) for n in range(n_gate_up)]
        row += [gate_up_entry(gate_up, e, n, True) for n in range(n_gate_up)]
        row += [down_entry(down, e, n) for n in range(n_down)]
        rows.append([v for entry in row for v in entry])
    return torch.tensor(rows, dtype=torch.int64)


def table_entries(table, n_gate_up, n_down):
    """Split a page table into its (gate, up, down) blocks of [E, n, 2] (bank, offset) entries."""
    t = table.reshape(table.shape[0], -1, 2)
    return t[:, :n_gate_up], t[:, n_gate_up : 2 * n_gate_up], t[:, 2 * n_gate_up : 2 * n_gate_up + n_down]


def bank_bytes_from_host(host, banks, page=64):
    """One device's host copy [rows, banks * page] of a uint8 width-sharded stream tensor -> each bank's bytes."""
    return [host[:, b * page : (b + 1) * page].reshape(-1) for b in range(banks)]


def read_tiles(bank_bytes, entries, k, k_stride, tile_bytes):
    """Bytes of tile k of every entry: bank_bytes[b] is DRAM bank b's uint8 bytes; entries [..., 2] -> [..., tile_bytes]."""
    flat = entries.reshape(-1, 2)
    out = torch.stack(
        [bank_bytes[int(b)][int(o) + k * k_stride : int(o) + k * k_stride + tile_bytes] for b, o in flat.tolist()]
    )
    return out.reshape(*entries.shape[:-1], tile_bytes)


def stream_column_biases(bank_bytes, entries, k_tiles, unpack_bfp):
    """The bias decode adds to every column of entries [..., 2]: its bias tile (tile k_tiles of the rounded part)
    unpacked with mxfp4_split.unpack_bfp, row n's terms at K 0 and 16 summed. Returns [..., 32] float32 (column n of the
    tile column at index n)."""
    raw = read_tiles(bank_bytes, entries, k_tiles, BFP4_BYTES, BFP4_BYTES)
    tiles = unpack_bfp(raw.reshape(-1, BFP4_BYTES), 3)  # [m, 32, 32] stored (transposed): row n = output column n
    vals = sum(tiles[:, :, c] for c in BIAS_TERM_COLUMNS)
    return vals.reshape(*entries.shape[:-1], TILE).float()


def width_sharded_tile_entry(n_tile, k_tile, shard_tiles, tile_bytes):
    """(bank, offset) of tile (k_tile, n_tile) of a TILE tensor [K, N] DRAM width-sharded over the banks with shard
    width shard_tiles tiles (ROW_MAJOR shard grid (0, 0)..(banks - 1, 0): bank b holds tile columns
    [b * shard_tiles, (b + 1) * shard_tiles), its tiles row-major in the shard with row stride shard_tiles, a partial
    last shard included: tt_metal buffer page mapping). The k stride of such a column is shard_tiles * tile_bytes."""
    b, c = divmod(n_tile, shard_tiles)
    return b, (k_tile * shard_tiles + c) * tile_bytes


def dense_page_table(num_experts, inter_pad, hidden, gu_shard_tiles, down_shard_tiles, tile_bytes, up_delta=0):
    """[E, 2 * (2 * n_gu + n_dn)] int64 page table of the batch-32 dense decode expert weights (experts/weights.py
    load_dense_decode_expert_weights, experts/dense_layout.py: gate / up [H, E * inter_pad] with expert e in columns
    [e * inter_pad, (e + 1) * inter_pad), down [E * inter_pad, H] with expert e in rows [e * inter_pad,
    (e + 1) * inter_pad), all DRAM width-sharded, normal (not transposed) tiles). Gate and up are two tensors: the up
    entries carry up_delta = up buffer address - gate buffer address (mod 2^32) so both read from the gate source.
    n_gu = inter_pad / 32, n_dn = ceil(hidden / 32); k strides: gu_shard_tiles * tile_bytes (gate / up, K = hidden)
    and down_shard_tiles * tile_bytes (down, K = inter_pad).

    Only whole-tile experts can be addressed: inter_pad must be a multiple of 32 (gpt-oss-20b at TP 4: 720 columns
    padded to 736, dense_layout.padded_expert_width)."""
    assert (
        inter_pad % TILE == 0
    ), f"expert width {inter_pad} is not whole tiles: the dense layout is not addressable per expert"
    n_gu, n_dn, k_dn = inter_pad // TILE, -(-hidden // TILE), inter_pad // TILE
    assert gu_shard_tiles > 0 and down_shard_tiles > 0
    rows = []
    for e in range(num_experts):
        gate = [width_sharded_tile_entry(e * n_gu + n, 0, gu_shard_tiles, tile_bytes) for n in range(n_gu)]
        up = [(b, (o + up_delta) % (1 << 32)) for b, o in gate]
        down = [width_sharded_tile_entry(n, e * k_dn, down_shard_tiles, tile_bytes) for n in range(n_dn)]
        rows.append([v for entry in gate + up + down for v in entry])
    return torch.tensor(rows, dtype=torch.int64)


def as_uint32_words(table):
    """int64 table (values in [0, 2^32)) -> int32 tensor with the same 32-bit words, for a ttnn uint32 tensor."""
    assert int(table.min()) >= 0 and int(table.max()) < (1 << 32), "page table entries must fit uint32"
    return torch.where(table >= (1 << 31), table - (1 << 32), table).to(torch.int32)
