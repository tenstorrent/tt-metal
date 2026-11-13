# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Expert placement of the Hy4-Preview routed experts (CPU only): 256 experts over the 4x8 mesh, 8 per chip.

Decode (``moe_compute`` via TTMoEDecode, ``expert_mapping: sequential``, dispatch ``cluster_axis`` 0 = mesh rows):
    expert e -> mesh column c = e // 32 (8 groups of 32 experts), mesh row r = (e % 32) // 8, local slot e % 8,
    linear chip id r * 8 + c  (the "column-major" convention of tt_moe_decode_config: adjacent experts straddle rows).
Same rule as the DSV4 tree with 384 -> 256 (DSV4: ``((e % 48) // 12) * 8 + e // 48``).
The unified prefill (``ExpertMapping`` of deepseek_v3_d_p) uses the same map: group = column, chip = row.
"""
NE, NROWS, NCOLS = 256, 4, 8
NCHIP = NROWS * NCOLS
EPC = NE // NCHIP  # 8
GROUP = NE // NCOLS  # 32 experts per mesh column


def chip_of(e: int) -> int:
    return ((e % GROUP) // EPC) * NCOLS + e // GROUP


def sequential():
    return [chip_of(e) for e in range(NE)]


def row_col(e: int):
    c = chip_of(e)
    return divmod(c, NCOLS)


def local_slot(e: int) -> int:
    return e % EPC


def experts_of_chip(row: int, col: int):
    return [col * GROUP + row * EPC + j for j in range(EPC)]


def validate(a=None):
    a = a or sequential()
    assert len(a) == NE and all(0 <= c < NCHIP for c in a)
    cnt = [0] * NCHIP
    for c in a:
        cnt[c] += 1
    assert all(n == EPC for n in cnt), sorted(set(cnt))
    return a
