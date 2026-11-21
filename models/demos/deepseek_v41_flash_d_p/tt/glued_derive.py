# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.7C attempt 3 (OWNER 2026-11-20 19:50, OV-67, PACKED_GLUED_DYNAMIC=on): a glued chunk's per-chunk tables derived
ON THE DEVICE from the chunk header, which arrives in the same upload as the token ids.

The header (tt-d-gen's G1 fields, drafts/tt_d_gen_packed_ask.md): HDR_WORDS uint32 words
``[n, (slot, actual_start, actual_end, global_write_start) x GLUED_SEGS, 0...]``, appended to every SP row's shard of
the engine input (``[SP, 1, W + HDR_WORDS]``: the ids, then the header; tt/packed_device.make_input). Nothing else
about the layout is uploaded: every table below is made by a prepared program (one kernel source, one section a
program, the chip's index baked in at build, the header's byte offset a per-run value) that reads the header and
writes a persistent output (allocated once, pinned). A program's cache key holds shapes and section parameters only
(the indexer's chunk / ratio / capacity, a compressed row count, a segment count): no layout is a key.

    keep    each chip's rows below the used rows: 1.0 / 0.0 ([1, 1, R, 1] bf16 TILE a chip; GluedGeo.keep_rows), and
            the MoE head's actual_end word of the chip's SP row (r W + its real rows; tt/packed.real_prefix)
    region  each row's window index list in the glued region ([1, 1, R, 128] uint32 RM a chip; GluedGeo.region_lists;
            a per-run mask of the segments whose carry is masked: class 0)
    ckeep   the chunk's n compressed rows below the used rows ([1, 1, n, 1] bf16 TILE, every chip; GluedCall._ckeep)
    idx     per pass p < GLUED_SEGS: the top-k valid length of each chip ([1, 1, 1, 1] uint32), and for the candidate
            indexer the pool's valid length and its -inf cap row ([1, 1, 1, ctopk] bf16 TILE); the ChipRows piece
            tables of the passes (phase rolls, keeps, candidate keeps; GluedCall._index_dynamic's Call pieces) in one
            [1, 1, 1, N_TABLES x TABLE_WORDS] uint32 page a chip, at fixed table slots (``slot``)
    srv     the server-order index rows of the segments that write ((d + off) mod S: GluedCall._srv), concatenated
            ([1, 1, 1, n W] uint32 a chip, its SP row's; per-run values: which segments)

The host keeps what decides the op sequence (passes, which calls run, which segments write: from the descriptor it
holds), never a table. tests/packed/test_glued_derive_ops.py checks every output against the host functions it
replaces (GluedGeo / GluedCall / Call) on random layouts.
"""

from __future__ import annotations

import torch

import ttnn

from .ccl import pin
from .common import DRAM, SP_AXIS, TP_AXIS

HDR_WORDS = 32  # header words: n + 4 x GLUED_SEGS fields, padded to 128 bytes
GLUED_SEGS = 6
N_TABLES = 64  # (tt/packed_glued: ChipRows tables a chunk, words a table)
TABLE_WORDS = 32
WINDOW = 128
REGION_CORES = 16  # region rows split over this many cores a chip (data movement only)


def slot(kind: str, p: int, phased: bool = False, long_: bool = False) -> int:
    """The ChipRows table slot of the idx program's piece tables: 'phase' (pass p's rows rolled to the alone chip
    rows), 'keep' (pass p's top-k rows -> its segment's rows; phased or not), 'cand' (the candidate pool's keep rows of
    the chips whose pool is long / short)."""
    if kind == "phase":
        return p
    if kind == "keep":
        return 6 + 6 * int(phased) + p
    assert kind == "cand", kind
    return 18 + 12 * int(long_) + 6 * int(phased) + p


def header_words(desc) -> list:
    """The chunk header of a glued chunk (tt/packed.GluedChunk) as HDR_WORDS ints (tt-d-gen's G1 fields)."""
    segs = list(desc.segments)
    assert 1 <= len(segs) <= GLUED_SEGS, len(segs)
    w = [len(segs)]
    for s in segs:
        w += [int(s.slot), int(s.start), int(s.end), int(s.gws)]
    return w + [0] * (HDR_WORDS - len(w))


_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CHUNK = get_compile_time_arg_val(0);
constexpr uint32_t SP = get_compile_time_arg_val(1);
constexpr uint32_t TP = get_compile_time_arg_val(2);
constexpr uint32_t R = get_compile_time_arg_val(3);
constexpr uint32_t BLOCK = get_compile_time_arg_val(4);  // decode's kv block (PREFILL_END=cap: cap = end // BLOCK * BLOCK)
constexpr uint32_t NC = get_compile_time_arg_val(5);
constexpr uint32_t A0 = get_compile_time_arg_val(6);
constexpr uint32_t A1 = get_compile_time_arg_val(7);
constexpr uint32_t A2 = get_compile_time_arg_val(8);
constexpr uint32_t A3 = get_compile_time_arg_val(9);
constexpr uint32_t A4 = get_compile_time_arg_val(10);
constexpr uint32_t A5 = get_compile_time_arg_val(11);
constexpr uint32_t WIN = 128;
constexpr uint32_t SENT = 0xFFFFFFFFu;
constexpr uint32_t NSEG = 6;
constexpr auto h_args = TensorAccessorArgs<12>();
@ACCESSORS@

#ifndef ALIGN_ROWS
#define ALIGN_ROWS 0
#endif

struct Seg {
    uint32_t start, rows, off, lane, gws;
};

FORCE_INLINE void zero_words(uint32_t addr, uint32_t words) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < words; ++i) {
        p[i] = 0;
    }
}

void kernel_main() {
    const uint32_t hdr = get_arg_val<uint32_t>(0);
    const uint32_t q = get_arg_val<uint32_t>(1 + NOUT);
    const uint32_t k = get_arg_val<uint32_t>(2 + NOUT);
    (void)k;
    const uint32_t hoff = get_arg_val<uint32_t>(3 + NOUT);
    cb_reserve_back(0, 1);
    const uint32_t base = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t buf = base + 128;
    const auto H = TensorAccessor(h_args, hdr, HPAGE);
    noc_async_read(H.get_noc_addr(0, hoff), base, 128);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* hw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    Seg s[NSEG];
    uint32_t ns = 0, used = 0;
    const uint32_t n = hw[0] < NSEG ? hw[0] : NSEG;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t st = hw[2 + 4 * i], en = hw[3 + 4 * i], gw = hw[4 + 4 * i];
        const uint32_t cap = en / BLOCK * BLOCK;
        const uint32_t rows = cap > st ? cap - st : 0;
        // PB.9G (ALIGN_ROWS = rows a chip): a segment owns whole chips (its length rounded up, at least one chip)
        const uint32_t len = en > st ? en - st : 1;
        constexpr uint32_t AR = ALIGN_ROWS ? ALIGN_ROWS : 1;  // (no division by a zero constant)
        const uint32_t ext = ALIGN_ROWS ? (len + AR - 1) / AR * AR : rows;
        if (rows) {
            s[ns].start = st;
            s[ns].rows = rows;
            s[ns].off = used;
            s[ns].lane = i;
            s[ns].gws = gw;
            ++ns;
        }
        used += ext;
    }
    // this chip's entries (pass p = its p-th): segment, local rows [a, b)
    const uint32_t q0 = q * R;
    uint32_t ne = 0, ej[NSEG], ea[NSEG], eb[NSEG];
    for (uint32_t j = 0; j < ns; ++j) {
        const uint32_t lo = s[j].off, hi = s[j].off + s[j].rows;
        if (lo < q0 + R && hi > q0) {
            ej[ne] = j;
            ea[ne] = (lo > q0 ? lo : q0) - q0;
            eb[ne] = (hi < q0 + R ? hi : q0 + R) - q0;
            ++ne;
        }
    }
    volatile tt_l1_ptr uint32_t* u = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);

#if D_KEEP
    {
        const auto O0 = TensorAccessor(o0_args, get_arg_val<uint32_t>(1), P0);
        const auto O1 = TensorAccessor(o1_args, get_arg_val<uint32_t>(2), P1);
        volatile tt_l1_ptr uint16_t* t16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf);
        for (uint32_t t = 0; t < R / 32; ++t) {
            zero_words(buf, 512);
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t i = 32 * t + r;  // (a row of a segment's computed rows; PB.9G: not its chip's pad)
                bool in = false;
                for (uint32_t e = 0; e < ne; ++e) {
                    in = in || (ea[e] <= i && i < eb[e]);
                }
                if (in) {
                    t16[((r >> 4) << 9) + ((r & 15) << 4)] = 0x3F80;
                }
            }
            noc_async_write(buf, O0.get_noc_addr(t), 2048);
            noc_async_write_barrier();
        }
        constexpr uint32_t W = CHUNK / SP;
        const uint32_t rr = q / TP;
        uint32_t nr = 0;  // (the SP row's real rows as a prefix: its last real row + 1, tt/packed.real_prefix)
        for (uint32_t j = 0; j < ns; ++j) {
            const uint32_t lo = s[j].off, hi = s[j].off + s[j].rows;
            if (lo < (rr + 1) * W && hi > rr * W) {
                const uint32_t e = (hi < (rr + 1) * W ? hi : (rr + 1) * W) - rr * W;
                nr = e > nr ? e : nr;
            }
        }
        u[0] = rr * W + nr;
        noc_async_write(buf, O1.get_noc_addr(0), 4);
        noc_async_write_barrier();
    }
#endif

#if D_REGION
    {
        const auto O0 = TensorAccessor(o0_args, get_arg_val<uint32_t>(1), P0);
        const uint32_t mask = get_arg_val<uint32_t>(4 + NOUT);
        const uint32_t lo = R * k / NC, hi = R * (k + 1) / NC;
        constexpr uint32_t BR = 8;
        for (uint32_t i0 = lo; i0 < hi; i0 += BR) {
            const uint32_t nb = (hi - i0) < BR ? (hi - i0) : BR;
            for (uint32_t b = 0; b < nb; ++b) {
                const uint32_t i = i0 + b;
                volatile tt_l1_ptr uint32_t* row = u + b * WIN;
                int32_t jj = -1;
                for (uint32_t e = 0; e < ne; ++e) {
                    if (ea[e] <= i && i < eb[e]) {
                        jj = static_cast<int32_t>(ej[e]);
                    }
                }
                if (jj < 0) {
                    row[0] = WIN + i;
                    for (uint32_t x = 1; x < WIN; ++x) {
                        row[x] = SENT;
                    }
                    continue;
                }
                const Seg& g = s[jj];
                const int32_t d = static_cast<int32_t>(q0 + i - g.off);
                const int32_t cls = ((mask >> jj) & 1u) ? 0 : static_cast<int32_t>(g.start < WIN ? g.start : WIN);
                int32_t x0 = d - static_cast<int32_t>(WIN - 1);
                if (x0 < -cls) {
                    x0 = -cls;
                }
                const int32_t carry0 = static_cast<int32_t>(WIN + R + WIN * g.lane + WIN);
                const int32_t own0 = static_cast<int32_t>(WIN) - static_cast<int32_t>(q0) + static_cast<int32_t>(g.off);
                for (uint32_t x = 0; x < WIN; ++x) {
                    const int32_t X = x0 + static_cast<int32_t>(x);
                    if (X > d) {
                        row[x] = SENT;
                    } else if (X < 0) {
                        row[x] = static_cast<uint32_t>(carry0 + X);
                    } else {
                        // own rows (mx >= q0): 128 + mx - q0; the previous chip's: mx - (q0 - 128): the same formula
                        row[x] = static_cast<uint32_t>(own0 + X);
                    }
                }
            }
            for (uint32_t b = 0; b < nb; ++b) {
                noc_async_write(buf + b * WIN * 4, O0.get_noc_addr(i0 + b), WIN * 4);
            }
            noc_async_write_barrier();
        }
    }
#endif

#if D_CKEEP
    {
        const auto O0 = TensorAccessor(o0_args, get_arg_val<uint32_t>(1), P0);
        volatile tt_l1_ptr uint16_t* t16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf);
        constexpr uint32_t RAT = CHUNK / A0;  // (model rows a compressed row)
        for (uint32_t t = 0; t < A0 / 32; ++t) {
            zero_words(buf, 512);
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t m = (32 * t + r) * RAT;  // (a compressed row of a segment's computed rows)
                bool in = false;
                for (uint32_t j = 0; j < ns; ++j) {
                    in = in || (s[j].off <= m && m < s[j].off + s[j].rows);
                }
                if (in) {
                    t16[((r >> 4) << 9) + ((r & 15) << 4)] = 0x3F80;
                }
            }
            noc_async_write(buf, O0.get_noc_addr(t), 2048);
            noc_async_write_barrier();
        }
    }
#endif

#if D_IDX
    {
        // A0 the indexer's chunk, A1 its ratio, A2 the key capacity, A3 its block, A4 ctopk
        uint32_t lens[NSEG], nbv[NSEG];
        for (uint32_t j = 0; j < ns; ++j) {
            uint32_t v = (s[j].start + A0) / A1;
            v = (v + 31) / 32 * 32;
            lens[j] = v < A2 ? v : A2;
            nbv[j] = lens[j] / A3;
        }
        (void)nbv;
        for (uint32_t p = 0; p < NSEG; ++p) {
            const bool e = p < ne;
            const uint32_t j = e ? ej[p] : 0;
            u[0] = e ? lens[j] : 32;
            noc_async_write(buf, TensorAccessor(vl0_args, get_arg_val<uint32_t>(1 + p), 4).get_noc_addr(0), 4);
            noc_async_write_barrier();
#if D_CAND
            const uint32_t v = e ? nbv[j] : 0;
            u[0] = (e && v >= A4) ? v : A4;
            noc_async_write(buf, TensorAccessor(vlc0_args, get_arg_val<uint32_t>(7 + p), 4).get_noc_addr(0), 4);
            noc_async_write_barrier();
            const uint32_t cv = (e && v < A4) ? v : 0;
            volatile tt_l1_ptr uint16_t* t16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf);
            const auto C = TensorAccessor(cap0_args, get_arg_val<uint32_t>(13 + p), 2048);
            for (uint32_t t = 0; t < A4 / 32; ++t) {
                zero_words(buf, 512);
                for (uint32_t c = 0; c < 32; ++c) {
                    t16[((c >> 4) << 8) + (c & 15)] = (32 * t + c < cv) ? 0x7F80 : 0xFF80;
                }
                noc_async_write(buf, C.get_noc_addr(t), 2048);
                noc_async_write_barrier();
            }
#endif
        }
        // the ChipRows piece tables (tt/glued_derive.slot): [count, (source row, destination row, rows) x count]
        zero_words(buf, 64 * 32);
        auto put = [&](uint32_t t, uint32_t np, const uint32_t* w) {
            volatile tt_l1_ptr uint32_t* tb = u + t * 32;
            tb[0] = np;
            for (uint32_t x = 0; x < 3 * np; ++x) {
                tb[1 + x] = w[x];
            }
        };
        for (uint32_t p = 0; p < NSEG; ++p) {
            const bool e = p < ne;
            const uint32_t j = e ? ej[p] : 0;
            const uint32_t a = e ? ea[p] : 0, b = e ? eb[p] : 0;
            const uint32_t offr = e ? s[j].off % R : 0;
            uint32_t w[6];
            const uint32_t c = e ? (R - offr) % R : 0;
            if (c) {
                w[0] = 0; w[1] = c; w[2] = R - c; w[3] = R - c; w[4] = 0; w[5] = c;
                put(p, 2, w);
            } else {
                w[0] = 0; w[1] = 0; w[2] = R;
                put(p, 1, w);
            }
            uint32_t k0[3] = {a, a, b - a};
            const uint32_t t = (a + R - offr) % R;
            const uint32_t first = (b - a) < (R - t) ? (b - a) : (R - t);
            uint32_t k1[6] = {t, a, first, 0, a + first, b - a - first};
            const uint32_t n1 = first < b - a ? 2 : 1;
            if (e) {
                put(6 + p, 1, k0);
                put(12 + p, n1, k1);
            }
#if D_CAND
            if (e) {
                const uint32_t L = nbv[j] >= A4 ? 1 : 0;
                put(18 + 12 * L + p, 1, k0);
                put(18 + 12 * L + 6 + p, n1, k1);
            }
#endif
        }
        const auto T = TensorAccessor(tab_args, get_arg_val<uint32_t>(TAB_ARG), 64 * 32 * 4);
        noc_async_write(buf, T.get_noc_addr(0), 64 * 32 * 4);
        noc_async_write_barrier();
    }
#endif

#if D_SRV
    {
        // A0 segments written; per-run values: their segment indices
        constexpr uint32_t W = CHUNK / SP;
        const auto O0 = TensorAccessor(o0_args, get_arg_val<uint32_t>(1), P0);
        const uint32_t rr = q / TP;
        for (uint32_t x = 0; x < A0; ++x) {
            const uint32_t j = get_arg_val<uint32_t>(4 + NOUT + x);
            // PB.9G: an encoder replay segment (gws > start) writes [gws, cap) as a chunk at gws: gws's server order,
            // its rows from gws - start on (TtCompressor._write_from's shifted rows)
            const uint32_t lo = s[j].gws > s[j].start ? s[j].gws : s[j].start;
            const uint32_t s0 = lo % CHUNK;
            const uint32_t intra = s0 % W, cst = (s0 / W) % SP;
            const uint32_t kk = (rr + SP - cst) % SP;
            const uint32_t off = s[j].off + (lo - s[j].start);
            for (uint32_t jj = 0; jj < W; ++jj) {
                const uint32_t d = kk == 0 ? (jj < W - intra ? jj : CHUNK - W + jj) : kk * W - intra + jj;
                u[jj] = (d + off) % CHUNK;
            }
            noc_async_write(buf, O0.get_noc_addr(0, x * W * 4), W * 4);
            noc_async_write_barrier();
        }
    }
#endif
}
"""


def _page_bytes(t: ttnn.Tensor) -> int:
    if t.layout == ttnn.TILE_LAYOUT:
        return int(ttnn.tile_size(t.dtype))
    eb = {ttnn.uint32: 4, ttnn.bfloat16: 2, ttnn.int32: 4, ttnn.float32: 4}[t.dtype]
    return int(t.shape[-1]) * eb


class Derive:
    """The per-chunk derive programs of a mesh (persistent outputs; ``begin`` a chunk, then each output on first use)."""

    def __init__(self, mesh, chunk: int, block: int, align: int = 0):
        self.mesh = mesh
        self.align = int(align)  # PB.9G: rows a chip of chip-aligned lanes (0: back to back)
        self.sp, self.tp = int(mesh.shape[SP_AXIS]), int(mesh.shape[TP_AXIS])
        self.n = self.sp * self.tp
        self.chunk, self.block = int(chunk), int(block)
        self.R = self.chunk // self.n
        self.W = self.chunk // self.sp
        assert self.R % 32 == 0 and self.R >= WINDOW, (self.chunk, self.n)
        self._pps: dict = {}
        self._outs: dict = {}
        self.hdr = None  # (tensor, byte offset) of this chunk's header
        self._up = None  # (the fallback upload of this chunk's header)
        self._ran: dict = {}
        self.stats = {"chunks": 0, "header_uploads": 0, "runs": 0}

    # ------------------------------------------------------------------ the chunk
    def begin(self, desc, inp=None) -> None:
        """A new chunk: its header is in ``inp`` (the engine input [SP, 1, W + HDR_WORDS], tt/packed_device.make_input)
        or, without one (tests driving run_chunk with host ids), uploaded here as the only per-chunk upload."""
        self.end()
        self.stats["chunks"] += 1
        if inp is not None and int(inp.shape[-1]) == self.W + HDR_WORDS:
            self.hdr = (inp, self.W * 4)
        else:
            w = torch.tensor(header_words(desc), dtype=torch.int64).to(torch.int32).view(1, 1, 1, HDR_WORDS)
            self._up = ttnn.from_torch(
                w,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            self.stats["header_uploads"] += 1
            self.hdr = (self._up, 0)

    def end(self) -> None:
        self._ran = {}
        self.hdr = None
        if self._up is not None:
            if self._up.is_allocated():
                ttnn.deallocate(self._up)
            self._up = None

    # ------------------------------------------------------------------ outputs
    def _alloc(self, shape, dtype, layout, per_chip: bool) -> ttnn.Tensor:
        """A persistent output: ``shape`` a chip ([SP, TP, ...] sharded when per_chip, replicated otherwise)."""
        if per_chip:
            host = torch.zeros(self.sp, self.tp, *shape[2:])
            mapper = ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(0, 1))
        else:
            host = torch.zeros(*shape)
            mapper = ttnn.ReplicateTensorToMesh(self.mesh)
        if dtype == ttnn.uint32:
            host = host.to(torch.int32)
        t = ttnn.from_torch(host, dtype=dtype, layout=layout, device=self.mesh, memory_config=DRAM, mesh_mapper=mapper)
        pin(self.mesh, t)
        return t

    def _outputs(self, kind: str, params: tuple) -> list:
        key = (kind, params)
        hit = self._outs.get(key)
        if hit is not None:
            return hit
        R, U32, BF, RM, TL = self.R, ttnn.uint32, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
        if kind == "keep":
            outs = [
                self._alloc([self.sp, self.tp, R, 1], BF, TL, True),
                self._alloc([self.sp, self.tp, 1, 1], U32, RM, True),
            ]
        elif kind == "region":
            outs = [self._alloc([self.sp, self.tp, R, WINDOW], U32, RM, True)]
        elif kind == "ckeep":
            outs = [self._alloc([1, 1, int(params[0]), 1], BF, TL, False)]
        elif kind == "idx":
            ctopk, cand = int(params[4]), bool(params[5])
            outs = [self._alloc([self.sp, self.tp, 1, 1], U32, RM, True) for _ in range(GLUED_SEGS)]
            if cand:
                outs += [self._alloc([self.sp, self.tp, 1, 1], U32, RM, True) for _ in range(GLUED_SEGS)]
                outs += [self._alloc([self.sp, self.tp, 1, ctopk], BF, TL, True) for _ in range(GLUED_SEGS)]
            outs.append(self._alloc([self.sp, self.tp, 1, N_TABLES * TABLE_WORDS], U32, RM, True))
        elif kind == "srv":
            outs = [self._alloc([self.sp, self.tp, 1, int(params[0]) * self.W], U32, RM, True)]
        else:
            raise AssertionError(kind)
        self._outs[key] = outs
        return outs

    # ------------------------------------------------------------------ programs
    def _source(self, kind: str, params: tuple, outs: list) -> tuple:
        acc, defs = [], []
        prev = "h_args"
        names = []
        if kind == "idx":
            cand = bool(params[5])
            names = [f"vl{p}" for p in range(GLUED_SEGS)]
            if cand:
                names += [f"vlc{p}" for p in range(GLUED_SEGS)] + [f"cap{p}" for p in range(GLUED_SEGS)]
            names.append("tab")
        else:
            names = [f"o{i}" for i in range(len(outs))]
        for nm in names:
            acc.append(f"constexpr auto {nm}_args = TensorAccessorArgs<{prev}.next_compile_time_args_offset()>();")
            prev = f"{nm}_args"
        for i, t in enumerate(outs):
            defs.append(f"constexpr uint32_t P{i} = {_page_bytes(t)};")
        flags = {"D_KEEP": 0, "D_REGION": 0, "D_CKEEP": 0, "D_IDX": 0, "D_SRV": 0, "D_CAND": 0}
        flags[{"keep": "D_KEEP", "region": "D_REGION", "ckeep": "D_CKEEP", "idx": "D_IDX", "srv": "D_SRV"}[kind]] = 1
        if kind == "idx" and bool(params[5]):
            flags["D_CAND"] = 1
        head = "".join(f"#define {k} {v}\n" for k, v in flags.items())
        head += f"#define NOUT {len(outs)}\n"
        head += f"#define ALIGN_ROWS {self.align}\n"
        if kind == "idx":
            head += f"#define TAB_ARG {len(outs)}\n"
        body = _KERNEL.replace("@ACCESSORS@", "\n".join(acc + defs + [f"constexpr uint32_t HPAGE = HPAGE_BYTES;"]))
        return head, body

    def _build(self, kind: str, params: tuple, hdr_t: ttnn.Tensor, hpage: int, outs: list):
        a = (list(params) + [0] * 6)[:6]
        ct = [self.chunk, self.sp, self.tp, self.R, self.block, 0] + [int(x) for x in a]
        nc = REGION_CORES if kind == "region" else 1
        ct[5] = nc
        ct += list(ttnn.TensorAccessorArgs(hdr_t).get_compile_time_args())
        for t in outs:
            ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        head, body = self._source(kind, params, outs)
        src = head + f"#define HPAGE_BYTES {int(hpage)}\n" + body
        cx = min(nc, 8)  # (cores (0, 0) .. (cx - 1, nc / cx - 1): inside every box's worker grid)
        cy = nc // cx
        assert cx * cy == nc, nc
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cx - 1, cy - 1))])
        size = 64 + 128 + 16384
        nvals = 1 + (1 if kind == "region" else GLUED_SEGS if kind == "srv" else 0)
        addrs = [int(hdr_t.buffer_address())] + [int(t.buffer_address()) for t in outs]
        md = ttnn.MeshProgramDescriptor()
        for q in range(self.n):
            r, c = divmod(q, self.tp)
            prog = ttnn.ProgramDescriptor()
            prog.cbs = [
                ttnn.CBDescriptor(
                    total_size=size, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, size)]
                )
            ]
            rt = ttnn.RuntimeArgs()
            for kk in range(nc):
                rt[kk % cx][kk // cx] = addrs + [q, kk] + [0] * nvals
            prog.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=src,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=crs,
                    compile_time_args=ct,
                    runtime_args=rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                    ),
                )
            ]
            md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = prog
        first = 1 + len(outs) + 2
        return md, [(0, first + v) for v in range(nvals)]

    def run(self, kind: str, params: tuple = (), values: tuple = ()) -> list:
        """The outputs of ``kind`` for this chunk (made once a chunk per (kind, params, values))."""
        params, values = tuple(params), tuple(int(v) for v in values)
        okey = (kind, params if kind != "region" else (int(values[0]),))
        outs = self._outputs(*okey)
        if self._ran.get(okey) == values:  # (made this chunk with these values; another value set rewrites it)
            return outs
        assert self.hdr is not None, "glued_derive: no chunk header (Derive.begin)"
        hdr_t, hoff = self.hdr
        hpage = _page_bytes(hdr_t)
        io = [hdr_t] + outs
        key = (kind, params, hpage, tuple(ttnn.TensorAccessorArgs(hdr_t).get_compile_time_args()))
        pp = self._pps.get(key)
        if pp is None:
            md, slots = self._build(kind, params, hdr_t, hpage, outs)
            pp = self._pps[key] = (ttnn.bringup.prepare_program(md, io, slots), len(slots))
        vals = [int(hoff)]
        if kind == "region":
            vals.append(int(values[0]))
        elif kind == "srv":
            vals += list(values) + [0] * (GLUED_SEGS - len(values))
        assert len(vals) == pp[1], (kind, vals, pp[1])
        ttnn.bringup.run_prepared(pp[0], io, vals)
        self._ran[okey] = values
        self.stats["runs"] += 1
        return outs

    def moe_span(self, real: int) -> dict | None:
        """PB.4's moe head span of this chunk (tt/packed_device.MoEPadSpan.span for a glued chunk: each SP row's rows
        up to its last real row; PB.9G: pad rows of a segment's last chip inside that prefix stay routed, row-local):
        None when every row is real (``real`` rows of the chunk's), else the zero start words and the derived end
        words."""
        if int(real) >= self.chunk:
            return None
        if getattr(self, "_zero", None) is None:
            self._zero = self._alloc([self.sp, self.tp, 1, 1], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, True)
        return dict(actual_start=self._zero, actual_end=self.run("keep")[1])


_BY: dict = {}


def derive(mesh, chunk: int, block: int, align: int = 0) -> Derive:
    key = (id(mesh), int(chunk), int(block), int(align))
    hit = _BY.get(key)
    if hit is None or hit.mesh is not mesh:
        hit = _BY[key] = Derive(mesh, chunk, block, align)
    return hit
