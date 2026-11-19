# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only: the addressing of the streamed MoE program, kernel formula by kernel formula, against the weight
streams of fused/moe/layout.py. Every tile of the streams carries a code (expert, gate / up / down, K tile, column
tile); the test walks the union as the kernels do (kernels/stream_reader.cpp, gate_up_compute.cpp,
gate_up_writer.cpp, finish_writer.cpp, down_compute.cpp, with the arguments fused/moe/program.py gives them) and
checks that

* a gate|up core's matmul meets, for every K tile it reads of x, the gate and the up tile of that K tile and of the
  right pair of the right expert, and every (unit, K tile) is covered exactly once (twice for a two-term plain slot);
* every unit's partials land in distinct entries of its finisher's buffer, S of them, and the finisher's unit order
  matches the entries;
* a down core's block holds, column-major, its columns of the slot's expert for all P K tiles, and all 192 columns
  are covered once per slot.
"""

import itertools

import pytest
import torch

from models.experimental.hy4_preview.fused.moe import layout as L

GATE, UP, DOWN = 0, 1, 2
KB = 48  # program.py GATE_UP_KB


def code(e, kind, k, col):
    return float(((e * 3 + kind) * 256 + k) * 256 + col)


def coded_streams(share, entries):
    """The two streams with every tile filled with its code; as [bank][tile] code tables."""
    P = share.P
    e, k, p = torch.meshgrid(torch.arange(entries), torch.arange(L.K_TILES), torch.arange(P), indexing="ij")
    gate_t = ((e * 3 + GATE) * 256 + k) * 256 + p  # [E, K, P]
    up_t = ((e * 3 + UP) * 256 + k) * 256 + p
    expand_gu = lambda t: t.permute(0, 2, 1).repeat_interleave(32, dim=1).repeat_interleave(32, dim=2).float()
    gate, up = expand_gu(gate_t), expand_gu(up_t)  # [E, 32 P, 6144]
    e, c, k = torch.meshgrid(torch.arange(entries), torch.arange(L.K_TILES), torch.arange(P), indexing="ij")
    down_t = ((e * 3 + DOWN) * 256 + k) * 256 + c  # [E, col, k]
    down = down_t.repeat_interleave(32, dim=1).repeat_interleave(32, dim=2).float()  # [E, 6144, 32 P]
    table = lambda s: s.reshape(-1, 32, L.BANKS, 32)[:, 0, :, 0].T.contiguous()  # [bank][tile]
    gu, dn = L.gate_up_stream(gate, up, share), L.down_stream(down, share)
    assert (gu.reshape(-1, 32, L.BANKS, 32) == gu.reshape(-1, 32, L.BANKS, 32)[:, :1, :, :1]).all()
    return table(gu), table(dn)


CASES = [
    ("tp32", L.tp_share(32), dict(G=6, D=6, NF=6), 2),
    ("tp32", L.tp_share(32), dict(G=8, D=4, NF=6), 2),
    ("tp8", L.tp_share(8), dict(G=6, D=6, NF=6), 2),
    ("tp8", L.tp_share(8), dict(G=5, D=3, NF=7), 1),
    ("ep", L.ep_share(), dict(G=6, D=6, NF=6), 1),
]


@pytest.mark.parametrize(
    "name, share, cores, terms", CASES, ids=[f"{c[0]}-g{c[2]['G']}d{c[2]['D']}t{c[3]}" for c in CASES]
)
def test_addressing(name, share, cores, terms):
    P, S, pb = share.P, share.S, share.pb
    G, D, NF = cores["G"], cores["D"], cores["NF"]
    entries = 12
    gu, dn = coded_streams(share, entries)
    tile = L.BFP4_TILE_BYTES
    # a union: 5 routed slots and (with terms == 2) one plain slot stored as entries (8, 9)
    ids = [7, 2, 11, 0, 5] + ([8] if terms == 2 else [])
    U = len(ids)
    first_plain = U - (1 if terms == 2 else 0)
    cols = L.COLS_PER_BANK // D

    # ---- gate | up: stream_reader.cpp + gate_up_compute.cpp + gate_up_writer.cpp
    item_blocks = share.piece_k // KB
    block_tiles = KB * 2
    covered = {}  # (unit, k) -> count
    entries_seen = {}  # finisher -> set of partial entries
    for b, j in itertools.product(range(L.BANKS), range(G)):
        k0 = (b % S) * share.piece_k  # program.py: the compute kernel's runtime arg
        total = U * pb
        for v in range(j, total, G):
            slot = v // pb
            t = v - slot * pb
            src = ids[slot] * share.gate_up_stride + t * share.piece_bytes  # reader (core_offset 0)
            n_blocks = terms * item_blocks if slot >= first_plain else item_blocks
            g = t * L.BANKS + b  # writer
            p, q = divmod(g, S)
            u = slot * P + p
            for blk in range(n_blocks):
                if blk > 0 and blk % item_blocks == 0:
                    src += share.gate_up_stride - share.piece_bytes
                term = blk // item_blocks
                for k in range(KB):  # compute: custom_mm reads tile k * ct + c of the block for output tile c
                    kx = k0 + (blk % item_blocks) * KB + k  # the x tile it multiplies with
                    for c, kind in ((0, GATE), (1, UP)):
                        got = gu[b][src // tile + k * 2 + c].item()
                        assert got == code(ids[slot] + term, kind, kx, p), (name, b, j, v, blk, k, c)
                    covered[(u, kx)] = covered.get((u, kx), 0) + 1
                src += block_tiles * tile
            entry = ((slot // NF) * P + p) * S + q
            seen = entries_seen.setdefault(slot % NF, set())
            assert (entry, u) not in seen
            seen.add((entry, u))
    for slot in range(U):
        want = terms if slot >= first_plain else 1
        for p in range(P):
            assert all(covered.get((slot * P + p, k), 0) == want for k in range(L.K_TILES)), (name, slot, p)
    assert len(covered) == U * P * L.K_TILES
    # ---- finishers: finish_writer.cpp's unit n is pair n % P of slot f + (n // P) * NF, entries n * S .. n * S + S - 1
    for f in range(NF):
        units = [slot * P + p for slot in range(f, U, NF) for p in range(P)]
        got = sorted(entries_seen.get(f, set()))
        assert got == sorted((n * S + q, u) for n, u in enumerate(units) for q in range(S)), (name, f)

    # ---- down: stream_reader.cpp (pb = 1, one block per term) + down_compute.cpp (read_transposed)
    seen_cols = {}
    for b, d in itertools.product(range(L.BANKS), range(D)):
        block = cols * P * tile
        col0 = b * L.COLS_PER_BANK + d * cols  # the writer's first output tile
        for slot in range(U):
            src = ids[slot] * share.down_stride + d * block
            for term in range(terms if slot >= first_plain else 1):
                if term > 0:
                    src += share.down_stride - block
                for c, k in itertools.product(range(cols), range(P)):
                    got = dn[b][src // tile + c * P + k].item()  # tile c * kt + k feeds output tile c with act tile k
                    assert got == code(ids[slot] + term, DOWN, k, col0 + c), (name, b, d, slot, term, c, k)
                src += block
            for c in range(cols):
                seen_cols[(slot, col0 + c)] = seen_cols.get((slot, col0 + c), 0) + 1
    assert all(seen_cols.get((slot, c), 0) == 1 for slot in range(U) for c in range(L.K_TILES))
