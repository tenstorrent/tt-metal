# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drafter taps of the PREFILL: the inputs the DSpark drafter needs for the last 128 positions of every prompt, captured while the prefill computes them.

The drafter's ring K/V of a position is a function of ``hidden`` = concat of the MEAN OVER THE 4 mHC STREAMS at the INPUT of backbone layers 37 / 38 / 39 (after the layer's Engram
forward, before the layer; ``tt/spec_decoder.py`` taps the same tensors in every verify round). The prefill computes exactly those streams for every prompt token, so the speculative
launch does not have to replay the prompt tail through the verify trace to seed the drafter (``SpecRunner.seed``: ~32 full-trunk rounds): ``PrefillTaps.write`` (inside the traced
prefill chunk, before ``pl.forward`` of layers 37 / 38 / 39) scatters the mean-over-streams rows of the positions [S - 128, S) of every user into a persistent per-user stash (bf16),
and ``DSparkDrafter.seed_from_taps`` (tt/mtp.py, eager, after the prefill) turns the stash into the drafter's ring rows.

Stash layout (per mesh row, bf16 row-major rows of 512 elements, so the row scatter of the paged KV hand-off serves it: one token row of 5120 = 10 rows): for the layer index li = 0, 1, 2
and the decode user ``ud`` of the row (in-row index), slot = position % 128:  row ``li * S10 + 10 * stash_row(ud, slot) + c`` (c = 0..9), ``stash_row = ((ud // Uc) * 128 + slot) * Uc + ud % Uc``
with ``Uc`` the users per chunk of the chunked drafter: the rows of one drafter chunk and one slot are contiguous (slot-major), which is the order the eager seeding consumes.
Nothing is allocated inside a trace: the stash and the per-chunk index tensors (``PagedStateSink.bind``) are persistent, the indices are refreshed before every replay like the hand-off's.
"""

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as P

TAP_LAYERS = (37, 38, 39)  # dspark_target_layer_ids
TAP_ROWS = 128  # attention window of the drafter = positions kept per user
D_MODEL = 5120
ROW_W = 512  # width of a stash row (HEAD_DIM: the width of the row-scatter kernel)
PIECES = D_MODEL // ROW_W
GROUP_TOK = 1024  # token rows per scatter call (the kernel keeps every index of a call in L1)
SKIP = 0xFFFFFFFF
BLOCK = 5  # draft rows per user (dspark_block_size, tt/mtp.py)


def drafter_chunk(U):
    """Users per mesh row of one drafter chunk (the drafter's token rows, 5 per user, stay <= 32 per mesh row): see ``SpecRunner``."""
    return U if BLOCK * U <= 32 else 4


def taps_possible(layer_ids):
    return all(L in layer_ids for L in TAP_LAYERS)


class PrefillTaps:
    def __init__(self, md, users_per_row):
        """``users_per_row``: DECODE users per mesh row (the pool's / the drafter's user index)."""
        self.md = md
        self.rows, self.cols = tuple(md.shape)
        self.Ud = users_per_row
        self.Uc = drafter_chunk(users_per_row)
        assert self.Ud % self.Uc == 0
        self.S10 = self.Ud * TAP_ROWS * PIECES
        self.shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(self.rows, self.cols))
        self.stash = ttnn.from_torch(
            torch.zeros(self.rows, 1, 3 * self.S10, ROW_W),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.shard,
        )

    # ---- host side -------------------------------------------------------------------------------------------------------------------
    def stash_row(self, ud, slot):
        return ((ud // self.Uc) * TAP_ROWS + slot) * self.Uc + ud % self.Uc

    @staticmethod
    def groups(rtok):
        """(token rows per scatter call, calls) for ``rtok`` token rows per mesh row."""
        g = min(rtok, GROUP_TOK)
        while rtok % g:
            g -= 32
        return g, rtok // g

    def host_ids(self, s0, C, U, user_of, lens):
        """Scatter indices of the chunk of positions [s0, s0 + C) of ``U`` prefill slots per mesh row -> int64 [rows * G, 10 * g]: stash row of every piece of every token row
        (SKIP: not a tap row: the user is not in this replay / the position is not among its last 128). ``user_of(r, u)`` = global decode user of prefill slot u of row r (None: empty).
        """
        rtok = U * C
        pos = torch.arange(s0, s0 + C)
        tok = torch.full((self.rows, rtok), SKIP, dtype=torch.int64)
        for r in range(self.rows):
            for u in range(U):
                b = user_of(r, u)
                S = 0 if b is None else int(lens[b])
                if S <= 0:
                    continue
                ud = b % self.Ud
                ok = (pos < S) & (pos >= S - TAP_ROWS)
                row = ((ud // self.Uc) * TAP_ROWS + pos % TAP_ROWS) * self.Uc + ud % self.Uc
                tok[r, u * C : (u + 1) * C] = torch.where(ok, row, torch.full_like(pos, SKIP))
        ids = tok.unsqueeze(-1) * PIECES + torch.arange(PIECES)
        ids = torch.where(tok.unsqueeze(-1) == SKIP, torch.full_like(ids, SKIP), ids)
        g, G = self.groups(rtok)
        return ids.reshape(self.rows * G, g * PIECES)

    def upload(self, ids, host=False):
        t = ttnn.from_torch(
            ids.to(torch.int32).contiguous(),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.shard,
            **({} if host else dict(device=self.md, memory_config=ttnn.DRAM_MEMORY_CONFIG)),
        )
        return t

    # ---- in the traced chunk ---------------------------------------------------------------------------------------------------------
    @staticmethod
    def own_rows(xs):
        """streams of this device's own token rows (list of [32,1,4,D] chunks, or a PkList holding [1,1,N,4D]) -> mean over the 4 streams, bf16 tile [1,1,N,D]. Does not free ``xs``."""
        if getattr(xs, "packed", False):
            x = xs[0]
            N, D = int(x.shape[2]), int(x.shape[3]) // 4
            q = [ttnn.slice(x, [0, 0, 0, i * D], [1, 1, N, (i + 1) * D]) for i in range(4)]
            s = ttnn.add(ttnn.add(q[0], q[1]), ttnn.add(q[2], q[3]))
            m = ttnn.multiply(s, 0.25)
            for t in q + [s]:
                ttnn.deallocate(t)
        else:
            parts = []
            for x in xs:
                mc = ttnn.mean(x, dim=2, keepdim=True)  # [32,1,1,D] fp32
                parts.append(ttnn.reshape(mc, [1, 1, x.shape[0], x.shape[3]]))
            m = ttnn.concat(parts, dim=2) if len(parts) > 1 else parts[0]
        b = ttnn.typecast(m, ttnn.bfloat16)
        if b is not m:
            ttnn.deallocate(m)
        return b

    def write(self, li, xs, mesh_config, ccl, colsplit, ids_dev):
        """Trace-safe: scatter the tap rows of layer index ``li`` (0..2) of this chunk into the stash. Column split: every column owns every 8th 32-token chunk, so the own rows are
        gathered over the columns first (the decode-side streams, hence the drafter, are replicated over the columns).
        """
        own = self.own_rows(xs)
        if colsplit:
            n8 = int(own.shape[2]) // 32
            hg = mesh_config.allgather(
                ttnn.reshape(own, [n8, 1, 32, D_MODEL]), ccl, axis=1, dim=1
            )  # [n8, 8, 32, D]: (group, column, token) = token order
            ttnn.deallocate(own)
            own = ttnn.reshape(hg, [1, 1, n8 * self.cols * 32, D_MODEL])
        R = int(own.shape[2])
        g, G = self.groups(R)
        rm = ttnn.to_layout(own, ttnn.ROW_MAJOR_LAYOUT)
        for gi in range(G):
            src = ttnn.reshape(
                ttnn.slice(rm, [0, 0, gi * g, 0], [1, 1, (gi + 1) * g, D_MODEL]), [1, 1, g * PIECES, ROW_W]
            )
            ids = ttnn.slice(ids_dev, [gi, 0], [gi + 1, g * PIECES])
            P.paged_scatter_rows(self.stash, src, ids, base_offset=li * self.S10)
            ttnn.deallocate(src)
            ttnn.deallocate(ids)
        ttnn.deallocate(rm)
        if own is not rm:
            ttnn.deallocate(own)

    # ---- eager readers ---------------------------------------------------------------------------------------------------------------
    def chunk_hidden(self, c):
        """bf16 tile [1,1,TAP_ROWS * Uc, 15360] of drafter chunk ``c`` (rows slot-major: slot * Uc + user, users c * Uc ..): the concat of the 3 layers' stash rows."""
        M = TAP_ROWS * self.Uc
        parts = []
        for li in range(3):
            lo = li * self.S10 + c * M * PIECES
            rows = ttnn.slice(self.stash, [0, 0, lo, 0], [1, 1, lo + M * PIECES, ROW_W])
            full = ttnn.reshape(rows, [1, 1, M, D_MODEL])
            parts.append(ttnn.to_layout(full, ttnn.TILE_LAYOUT))
        return ttnn.concat(parts, dim=3)
