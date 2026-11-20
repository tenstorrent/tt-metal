# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fused iHC site of Hy4-preview decode: the pre-mix, the post-mix and the two back to back, for 8 token rows, as
kernels on the Tensix cores of ONE chip (fused/README.md, decision 3; kernels/ihc_site.cpp).

    pre        streams -> y [8, C] (the collapsed sublayer input), 1 / RMS(y) per row, the site's coefficients
    post       F [8, C] (the sublayer output) + streams -> streams, in place
    post_pre   post of one sublayer and pre of the next as one program: post(attention) + pre(MLP), and
               post(MLP) + pre(next layer's attention). The same binaries serve every site of every layer: what
               differs between sites is data (fn^T and the page of constants).

Mirrors reference/ihc.py IHCWrap (hc_pre / hc_post) and the RMSNorm statistic of reference.attention.rms_norm.

Interface for the layer. Everything is L1 of the K site cores (a rectangle of cores, row major), width sharded: core
k owns hidden columns [k C / K, (k + 1) C / K) of every row and stream. A sharded tensor has the same L1 address on
each of its cores (tensor.buffer_address()), so a fragment of the layer program reaches core k's part at that
address on core k.

    streams   fp32 [1, 1, 32, C]   layout.streams_to_pages. 786 KB for C = 6144, 786 / K KB per core. Lives here for
              the whole decode step; post updates it in place; nothing else reads it.
    y         what pre leaves for the broadcast: fp32 or BF16 [1, 1, 32, C / 4], layout.rows_to_pages. Core k's
              shard is the bytes of the [8, C / K] slice as [8, 32] tiles (two faces of [8, 16]) back to back:
              4 (or 2) * 8 * C / K bytes, 196.6 KB (98.3 KB BF16) over the K cores. Not normalised: the norm is
              deferred into the consumers, which take
    stat      1 / RMS(y) per token row, fp32, on site core 0 only (self.stat's first page): token row r is the
              fp32 word at byte 64 r of that page (and the 7 lanes after it, layout.SFPU_LANES). 8 scalars.
    F         what post expects back: the sublayer output in the layout and sharding of y, fp32 or BF16 (BF16 is
              widened exactly), complete in the site cores' L1 before the program starts.
    coef      fp32 [1, 1, 32, 32 K], one page per core (every core the same): the pre and post coefficients of
              the last pre, each on all lanes of its own cell. The state post reads; it never leaves the chip.
    fn^T      BF16 [1, 1, 32, C] per site, layout.fn_to_pages; const: fp32, one page per site, layout.const_page
              (scale, base, magnitude, the epsilons, 1 / (n C), 1 / C as SFPU lane vectors). Either L1 tensors of
              one site (site_params: 393 KB + 4 K KB) or, for many sites, two DRAM tensors of all of them
              (site_table / SiteRef): the reader of each core then fetches its pages of the site named by its
              runtime arguments, and one program serves every site.

IhcSite.program returns (io tensors, ProgramDescriptor): the kernels, CBs (ids 0..8 and 16..21) and semaphores
(0..2) of the site cores, to run alone with ttnn.generic_op (pre / post / post_pre) or to merge into a layer program.

Precision. Streams, both mixes, the coefficients and both sums of squares are fp32 in the SFPU. The fn projection
runs either in the FPU (HiFi4, fp32 accumulation; its operands are 19-bit, as in tt/ihc.py: 5e-4 relative on the
mixes; fn BF16, the checkpoint's dtype) or, with sfpu_proj, in the SFPU as well (fp32 throughout; fn as fp32 pages,
four times the streams' size).
"""

from __future__ import annotations

from dataclasses import dataclass

import os

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)
from models.experimental.hy4_preview.fused.ihc_site import layout
from models.experimental.hy4_preview.reference.ihc import IHCConfig, IHCWrap

KERNEL = "models/experimental/hy4_preview/fused/ihc_site/kernels/ihc_site.cpp"
PAGE = 4096  # fp32 page

# CB ids
CB_X, CB_XMM, CB_W, CB_F, CB_COEF, CB_CONST, CB_XCHG, CB_XCHG2, CB_COEF2 = 0, 1, 2, 3, 4, 5, 6, 7, 8
CB_XOUT, CB_Y, CB_PART, CB_COEF_OUT, CB_STAT, CB_DBG = 16, 17, 18, 19, 20, 21
CB_STAT_IN, CB_GAMMA, CB_XN = 9, 10, 22  # norm_out
SEM_PART, SEM_STAT, SEM_COEF = 0, 1, 2

_BYTES = {ttnn.float32: 4, ttnn.bfloat16: 2}


def _run(io, descriptor):
    """Run a program; on the mock Galaxy, compile and size-check it instead (nothing executes there)."""
    if os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
        ttnn.experimental.prepare_generic_op(io, descriptor)
    else:
        ttnn.generic_op(io, descriptor)


@dataclass
class SiteParams:
    """One site's data: fn^T and the page of constants, on device."""

    w: object
    const: object


@dataclass
class SiteTable:
    """Every site's data in two DRAM tensors (IhcSite.site_table): fn^T of site s is pages [s C / 32, (s + 1) C / 32)
    of ``w``, its constants page s of ``const``. A program takes the site as runtime arguments, so one program serves
    all of them."""

    w: object
    const: object
    sites: int
    gamma: object = (
        None  # the norms' weights, one per site (site_table(gammas=...)); None: NormOut.gamma is an L1 tensor
    )


@dataclass
class SiteRef:
    """Site ``index`` of a SiteTable, in place of a SiteParams."""

    table: SiteTable
    index: int


@dataclass
class NormOut:
    """The layer program's form of a pre-mix (IhcSite.program norm_out): the site cores also apply the sublayer's
    RMSNorm, x = y * (1 / RMS(y) of the row) * gamma, fp32 in the SFPU, leave it as BF16 in ``xn`` and hand it to the
    chip's sender core: core k writes its shard (the bytes of the [8, C / K] slice as [8, 32] BF16 row tiles) to
    ``staging + k * shard bytes`` on the sender core, and site core 0, once all K are there, adds ``ready_bit`` to
    the word at ``ready`` on the sender core. ``gather`` / ``status``: two persistent host-zeroed L1 words on site
    core 0 (the count of shards; a code when the bounded wait for them ran out)."""

    # fp32 rows pages of the norm's weight (IhcSite.gamma_to_device), or None: the site table holds it
    # (IhcSite.site_table(gammas=...)) and every site core fetches its pages with fn^T
    gamma: object
    xn: object  # BF16 rows tensor (IhcSite.empty_rows(ttnn.bfloat16))
    sender: int  # NoC x | y << 16
    staging: int
    ready: int
    ready_bit: int
    gather: int
    status: int
    wait_limit: int = 30_000_000
    stamps: int = 0  # measurement: two L1 words on site core 0 for (writer start, op made ready), wall clock; 0: none


def site_cores(num_cores: int, origin=(0, 0), per_row: int = 8):
    """K site cores as a rectangle, row major: core k is the k-th shard of every site tensor."""
    assert num_cores % min(num_cores, per_row) == 0, f"{num_cores} cores do not fill rows of {per_row}"
    cols = min(num_cores, per_row)
    rows = num_cores // cols
    start = ttnn.CoreCoord(origin[0], origin[1])
    end = ttnn.CoreCoord(origin[0] + cols - 1, origin[1] + rows - 1)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(start, end)])
    cores = [ttnn.CoreCoord(origin[0] + c, origin[1] + r) for r in range(rows) for c in range(cols)]
    return grid, cores


class IhcSite:
    """The site on one chip. ``num_cores`` must divide C / 128 (a core owns whole rows pages).

    Cores, measured (traced, 8 rows, C = 6144, us per post + pre program, of which 5 to 6 are the program's launch):
    SFPU projection 32.3 / 21.1 / 17.6 / 15.6 on 8 / 16 / 24 / 48 cores; FPU projection 46.7 / 26.9 / 17.7 / 13.6 /
    12.6 / 12.9 on 2 / 4 / 8 / 16 / 24 / 48. Default: 24 cores (three rows of 8), SFPU projection."""

    def __init__(
        self,
        device,
        cfg: IHCConfig = None,
        num_cores: int = 24,
        origin=(0, 0),
        y_dtype=ttnn.float32,
        f_dtype=ttnn.float32,
        lanes: str = None,
        block_unpack: bool = True,
        root_exchange: bool = None,
        low_unpack: bool = False,
        mm32: bool = False,
        sfpu_proj: bool = True,
        per_row: int = 8,
        cb_base: int = 0,
        sem_base: int = 0,
    ):
        """per_row: site cores per grid row (the site is a rectangle of cores, row major). cb_base / sem_base: added
        to every CB id (0..10, 16..22) and semaphore id (0..2) of the site's programs, for a program that has other
        kernels' buffers on the site cores."""
        self.device = device
        self.cb_base, self.sem_base = cb_base, sem_base
        # The 4 streams pages of a batch in one unpacker run (16 faces) rather than one run per page: bit-equal
        # and 5 to 16 % faster on the board (tests/test_ihc_site.py test_latency).
        self.block_unpack = block_unpack
        # fallback: unpack into DEST slots 0..3 only (probe_device: whether the unpacker reaches the upper slots)
        self.low_unpack = low_unpack
        assert not (low_unpack and num_cores > 32), "low_unpack: the root core takes at most 4 exchange pages"
        # fallback: the projection as one full [32, 32] x [32, 32] matmul per page (layout.fn_to_pages mm32)
        self.mm32 = mm32
        # The fn projection in the SFPU (fp32 throughout) rather than the FPU (19-bit operands): fn as fp32 pages,
        # four per streams page (layout.fn_to_row_pages, 3.1 MB per site for C = 6144). On the board, 8 rows, the
        # checkpoint's layers 0-5: coefficients within 6e-7 of float64 against 3e-3 with the FPU (the ttnn form:
        # 2e-3), for 7.5 us more per program on 16 cores, 5.0 on 24, 2.7 on 48 (tests/test_ihc_site.py
        # test_precision, test_latency_sfpu_projection). False: the FPU projection (needs 1 / 8 of the fn bytes).
        self.sfpu_proj = sfpu_proj
        assert not (sfpu_proj and (low_unpack or mm32)), "the SFPU projection has neither fallback"
        self.cfg = cfg or IHCConfig()
        assert self.cfg.n == 4, "the page layout holds 4 streams"
        self.C = self.cfg.dim
        self.K = num_cores
        assert self.C % (128 * num_cores) == 0, f"{num_cores} cores do not split C = {self.C} into whole rows pages"
        assert num_cores <= 48, "the root core takes 6 exchange pages of 8 cores' partials"
        # From 16 cores the partials go to the root core only, which multicasts the coefficients; below, every
        # core sends its partial to every core (kernels/ihc_site.cpp). On the board the two are level at 8 cores
        # and the root exchange is 0.5 us ahead at 16.
        self.root_exchange = num_cores >= 16 if root_exchange is None else root_exchange
        assert self.root_exchange or num_cores <= 16, "the all-to-all exchange holds 16 cores"
        self.px = self.C // 32 // num_cores  # streams pages per core
        self.py = self.px // 4
        # a core's streams, its y and F shards and one site's fn pages against its L1 (about 1.4 MB)
        core_bytes = self.px * (PAGE + 2 * 1024) + self.px * (4 * PAGE if sfpu_proj else 2048)
        assert core_bytes < 1300 * 1024, f"{num_cores} cores: {core_bytes >> 10} KB per core do not fit L1; use more"
        self.grid, self.cores = site_cores(num_cores, origin, per_row)
        self.noc = [device.worker_core_from_logical_core(c) for c in self.cores]
        self.y_dtype, self.f_dtype = y_dtype, f_dtype
        self.lanes = lanes or layout.SFPU_LANES
        self.coef = self._pages(torch.zeros(32, 32 * num_cores), ttnn.float32)
        self.stat = self._pages(torch.zeros(32, 32 * num_cores), ttnn.float32)
        self.dbg = None  # tests: every core's scratch page after a pre run with debug=True

    # ------------------------------------------------------------------ tensors
    def _memory_config(self, width):
        spec = ttnn.ShardSpec(self.grid, (32, width // self.K), ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)

    def _pages(self, matrix: torch.Tensor, dtype):
        """A [32, W] page matrix -> the [1, 1, 32, W] device tensor, W / K columns per site core."""
        return ttnn.from_torch(
            matrix.float().reshape(1, 1, *matrix.shape).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=self._memory_config(matrix.shape[-1]),
        )

    @staticmethod
    def _matrix(tensor) -> torch.Tensor:
        return ttnn.to_torch(tensor).float()[0, 0]

    def streams_to_device(self, x: torch.Tensor):
        """[T, 4, C] -> device streams."""
        return self._pages(layout.streams_to_pages(x), ttnn.float32)

    def streams_to_torch(self, streams, T: int = layout.ROWS) -> torch.Tensor:
        return layout.pages_to_streams(self._matrix(streams), T)

    def rows_to_device(self, v: torch.Tensor, dtype=None):
        """[T, C] -> device rows tensor (a sublayer output F)."""
        return self._pages(layout.rows_to_pages(v), dtype or self.f_dtype)

    def rows_to_torch(self, rows, T: int = layout.ROWS) -> torch.Tensor:
        return layout.pages_to_rows(self._matrix(rows), T)

    def empty_rows(self, dtype=None):
        return self._pages(torch.zeros(32, self.C // 4), dtype or self.y_dtype)

    def site_params(self, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor) -> SiteParams:
        """One site's fn [8, 4 C], base [8], scale [2] -> its device data."""
        w = self._pages(*self._fn_pages(fn))
        page = layout.const_page(base, scale, self.cfg, self.lanes)
        return SiteParams(w=w, const=self._pages(page.repeat(1, self.K), ttnn.float32))

    def _fn_pages(self, fn: torch.Tensor):
        """(page matrix, dtype) of one site's fn for this site's projection."""
        if self.sfpu_proj:
            return layout.fn_to_row_pages(fn, self.cfg.n, self.lanes), ttnn.float32
        return layout.fn_to_pages(fn, self.cfg.n, self.lanes, self.mm32), ttnn.bfloat16

    def site_table(self, sites, gammas=None) -> SiteTable:
        """[(fn, base, scale), ...] -> the DRAM tensors of all those sites; site s of it is SiteRef(table, s).
        gammas: per site the weight [C] of the RMSNorm that follows its pre-mix (NormOut(gamma=None) then takes it
        from here: no L1 tensor per norm)."""
        w_dtype = self._fn_pages(sites[0][0])[1]
        w = torch.cat([self._fn_pages(fn)[0] for fn, _, _ in sites], dim=1)
        const = torch.cat([layout.const_page(base, scale, self.cfg, self.lanes) for _, base, scale in sites], dim=1)

        def dram(matrix, dtype):
            return ttnn.from_torch(
                matrix.float().reshape(1, 1, *matrix.shape).contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        gamma = None
        if gammas is not None:
            assert len(gammas) == len(sites)
            rows = [layout.rows_to_pages(g.float().reshape(1, -1).expand(layout.ROWS, -1)) for g in gammas]
            gamma = dram(torch.cat(rows, dim=1), ttnn.float32)
        return SiteTable(w=dram(w, w_dtype), const=dram(const, ttnn.float32), sites=len(sites), gamma=gamma)

    def gamma_to_device(self, weight: torch.Tensor):
        """An RMSNorm weight [C] -> its fp32 rows pages (NormOut.gamma)."""
        return self._pages(layout.rows_to_pages(weight.float().reshape(1, -1).expand(layout.ROWS, -1)), ttnn.float32)

    def coef_to_torch(self, T: int = layout.ROWS, core: int = 0):
        """(pre [T, 4], post [T, 4]) as core ``core`` holds them."""
        page = self._matrix(self.coef)[:, 32 * core : 32 * core + 32]
        return layout.coef_from_page(page, T, self.lanes)

    def coef_to_device(self, pre: torch.Tensor, post: torch.Tensor):
        """Overwrite the coefficient state (tests of post alone)."""
        page = layout.coef_to_page(pre, post, self.lanes)
        self.coef = self._pages(page.repeat(1, self.K), ttnn.float32)

    def stat_to_torch(self, T: int = layout.ROWS) -> torch.Tensor:
        """1 / RMS(y) per token row [T]."""
        return layout.stat_from_page(self._matrix(self.stat)[:, :32], T)

    # ------------------------------------------------------------------ programs
    def _cb(self, index, tensor, tile=(32, 32), dtype=ttnn.float32):
        cb = ttnn.cb_descriptor_from_sharded_tensor(index, tensor)
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=index,
            data_format=dtype,
            page_size=tile[0] * tile[1] * _BYTES[dtype],
            tile=ttnn.TileDescriptor(ttnn.Tile(list(tile))),
        )
        cb.format_descriptors = [fmt]
        return cb

    def _scratch_cb(self, index, pages):
        fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32, page_size=PAGE)
        return ttnn.CBDescriptor(total_size=pages * PAGE, core_ranges=self.grid, format_descriptors=[fmt])

    def program(
        self,
        streams,
        f=None,
        params: SiteParams = None,
        y=None,
        probe=False,
        stop_after=0,
        debug=False,
        empty=False,
        norm_out: NormOut = None,
    ):
        """The program descriptor and its io tensors. f: post-mix; params (and y): pre-mix. debug (tests): every
        core also leaves its scratch page in self.dbg (layout.debug_from_page). norm_out: the pre-mix also applies
        the norm and hands the result to the chip's sender core (NormOut)."""
        do_post, do_pre = f is not None, params is not None
        assert do_post or do_pre
        K = self.K
        b = self.cb_base
        CB_X, CB_XMM, CB_W, CB_F, CB_COEF, CB_CONST, CB_XCHG, CB_XCHG2, CB_COEF2, CB_STAT_IN, CB_GAMMA = (
            b + i for i in range(11)
        )
        CB_XOUT, CB_Y, CB_PART, CB_COEF_OUT, CB_STAT, CB_DBG, CB_XN = (b + i for i in range(16, 23))
        SEM_PART, SEM_STAT, SEM_COEF = (self.sem_base + i for i in range(3))
        site_dram = isinstance(params, SiteRef)
        table = params.table if site_dram else None
        gamma_dram = do_pre and norm_out is not None and norm_out.gamma is None
        assert not gamma_dram or (
            site_dram and table.gamma is not None
        ), "NormOut(gamma=None) needs site_table(gammas=...)"
        io = [streams, self.coef, self.stat]
        cbs = [self._cb(CB_X, streams), self._cb(CB_COEF_OUT, self.coef), self._cb(CB_STAT, self.stat)]
        fp32_copies = [CB_X]
        f_fp32 = True
        if do_post:
            f_fp32 = f.dtype == ttnn.float32
            cbs += [self._cb(CB_F, f, dtype=f.dtype), self._cb(CB_COEF, self.coef), self._cb(CB_XOUT, streams)]
            io.append(f)
            fp32_copies += [CB_COEF] + ([CB_F] if f_fp32 else [])
        if do_pre:
            w_tile = (32, 32) if (self.mm32 or self.sfpu_proj) else (32, 16)
            w_dtype = ttnn.float32 if self.sfpu_proj else ttnn.bfloat16
            w_bytes = self.px * (4 * PAGE if self.sfpu_proj else 2048)  # fn bytes of one core
            if self.sfpu_proj:
                fp32_copies.append(CB_W)
            else:
                cbs.append(self._cb(CB_XMM, streams, tile=(32, 32) if self.mm32 else (16, 32)))
            cbs += [
                self._scratch_cb(CB_XCHG, (K + 7) // 8),
                self._scratch_cb(CB_XCHG2, (K + 15) // 16),
                self._scratch_cb(CB_PART, 1),
            ]
            if site_dram:
                # the reader fills these from the DRAM tensors of all sites
                w_fmt = ttnn.CBFormatDescriptor(
                    buffer_index=CB_W,
                    data_format=w_dtype,
                    page_size=w_tile[0] * w_tile[1] * _BYTES[w_dtype],
                    tile=ttnn.TileDescriptor(ttnn.Tile(list(w_tile))),
                )
                cbs += [
                    ttnn.CBDescriptor(total_size=w_bytes, core_ranges=self.grid, format_descriptors=[w_fmt]),
                    self._scratch_cb(CB_CONST, 1),
                ]
                io += [table.w, table.const]
            else:
                cbs += [self._cb(CB_W, params.w, tile=w_tile, dtype=w_dtype), self._cb(CB_CONST, params.const)]
                io += [params.w, params.const]
            fp32_copies += [CB_CONST, CB_XCHG, CB_XCHG2]
            if self.root_exchange:
                cbs.append(self._cb(CB_COEF2, self.coef))
                fp32_copies.append(CB_COEF2)
            if not probe:
                cbs.append(self._cb(CB_Y, y, dtype=y.dtype))
                io.append(y)
            if norm_out is not None:
                assert self.root_exchange and not probe and stop_after == 0 and not empty
                assert y.dtype == ttnn.float32, "the norm reads y back as fp32"
                cbs += [
                    self._cb(CB_STAT_IN, self.stat),
                    self._scratch_cb(CB_GAMMA, self.py) if gamma_dram else self._cb(CB_GAMMA, norm_out.gamma),
                    self._cb(CB_XN, norm_out.xn, dtype=ttnn.bfloat16),
                ]
                io += [table.gamma if gamma_dram else norm_out.gamma, norm_out.xn]
                fp32_copies += [CB_STAT_IN, CB_GAMMA, CB_Y]
            if debug:
                if self.dbg is None:
                    self.dbg = self._pages(torch.zeros(32, 32 * K), ttnn.float32)
                cbs.append(self._cb(CB_DBG, self.dbg))
                io.append(self.dbg)
        ct = [
            ("cb_x", CB_X),
            ("cb_xmm", CB_XMM),
            ("cb_w", CB_W),
            ("cb_f", CB_F),
            ("cb_coef", CB_COEF),
            ("cb_const", CB_CONST),
            ("cb_xchg", CB_XCHG),
            ("cb_xchg2", CB_XCHG2),
            ("cb_xout", CB_XOUT if do_post else CB_COEF_OUT),
            ("cb_y", CB_Y),
            ("cb_part", CB_PART),
            ("cb_coef_out", CB_COEF_OUT),
            ("cb_stat", CB_STAT),
            ("cb_dbg", CB_DBG),
            ("cb_coef2", CB_COEF2),
            ("num_cores", K),
            ("x_pages", self.px),
            ("do_post", int(do_post)),
            ("do_pre", int(do_pre)),
            ("probe", int(probe)),
            ("f_fp32", int(f_fp32)),
            ("sem_part", SEM_PART),
            ("sem_stat", SEM_STAT),
            ("sem_coef", SEM_COEF),
            ("root_exchange", int(self.root_exchange)),
            ("stop_after", stop_after),
            ("debug", int(debug)),
            ("block_unpack", int(self.block_unpack)),
            ("low_unpack", int(self.low_unpack)),
            ("mm32", int(self.mm32)),
            ("site_dram", int(site_dram)),
            ("sfpu_proj", int(self.sfpu_proj)),
            ("empty_program", int(empty)),
            ("norm_out", int(norm_out is not None)),
            ("gamma_dram", int(gamma_dram)),
            ("cb_stat_in", CB_STAT_IN),
            ("cb_gamma", CB_GAMMA),
            ("cb_xn", CB_XN),
            ("norm_wait_limit", norm_out.wait_limit if norm_out is not None else 0),
        ]
        config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=True,
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for cb in fp32_copies:
            modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
        root, first, last = self.noc[0], self.noc[0], self.noc[-1]
        peers = [v for p in self.noc for v in (p.x, p.y)]
        reader_ct, site_args = [], lambda k: []
        if site_dram:
            assert 0 <= params.index < table.sites
            reader_ct = list(ttnn.TensorAccessorArgs(table.w).get_compile_time_args())
            reader_ct += list(ttnn.TensorAccessorArgs(table.const).get_compile_time_args())
            if gamma_dram:
                reader_ct += list(ttnn.TensorAccessorArgs(table.gamma).get_compile_time_args())
            per_page = 4 if self.sfpu_proj else 1  # fn pages per streams page
            pages = per_page * self.C // 32  # fn pages of one site
            g_pages = self.C // 4 // 32  # rows pages of one site's gamma
            site_args = lambda k: [
                table.w.buffer_address(),
                table.const.buffer_address(),
                params.index * pages + k * per_page * self.px,
                params.index,
            ] + ([table.gamma.buffer_address(), params.index * g_pages + k * self.py] if gamma_dram else [])
        shard = self.py * 2048  # a core's BF16 shard of the rows
        norm_args = lambda k: (
            []
            if norm_out is None or not do_pre
            else [
                norm_out.sender,
                norm_out.staging + k * shard,
                norm_out.ready,
                norm_out.ready_bit,
                norm_out.gather,
                norm_out.status,
                norm_out.stamps,
            ]
        )
        per_core = PerCoreRuntimeArgsDescriptor(
            ncrisc_args=[
                (c, [int(k == 0), first.x, first.y, last.x, last.y] + site_args(k)) for k, c in enumerate(self.cores)
            ],
            brisc_args=[(c, [k, root.x, root.y] + peers + norm_args(k)) for k, c in enumerate(self.cores)],
            trisc_args=[(c, [int(k == 0)]) for k, c in enumerate(self.cores)],
        )
        kernel = UnifiedKernelDescriptor(
            kernel_source=KERNEL,
            core_ranges=self.grid,
            ncrisc_compile_time_args=reader_ct,
            ncrisc_named_compile_time_args=ct,
            brisc_named_compile_time_args=ct,
            trisc_named_compile_time_args=ct,
            trisc_compute_config=config,
            per_core_runtime_args_descriptor=per_core,
        )
        semaphores = [
            ttnn.SemaphoreDescriptor(int(SEM_PART), ttnn.CoreType.WORKER, self.grid, 0),
            ttnn.SemaphoreDescriptor(int(SEM_STAT), ttnn.CoreType.WORKER, self.grid, 0),
            ttnn.SemaphoreDescriptor(int(SEM_COEF), ttnn.CoreType.WORKER, self.grid, 0),
        ]
        descriptor = ttnn.ProgramDescriptor(
            kernels=kernel.get_kernel_descriptors().kernels, semaphores=semaphores, cbs=cbs
        )
        return io, descriptor

    def pre(self, streams, params: SiteParams, y=None, stop_after=0, debug=False):
        """streams -> y (returned), with self.stat and self.coef updated."""
        y = y if y is not None else self.empty_rows()
        _run(*self.program(streams, params=params, y=y, stop_after=stop_after, debug=debug))
        return y

    def empty(self, streams, f, params, y):
        """The post + pre program with kernels that return at once (its launch cost on the site cores)."""
        _run(*self.program(streams, f=f, params=params, y=y, empty=True))

    def post(self, streams, f):
        """streams += post * F with the coefficients of the last pre (self.coef), in place."""
        _run(*self.program(streams, f=f))
        return streams

    def post_pre(self, streams, f, params: SiteParams, y=None, stop_after=0, debug=False):
        """post of the sublayer whose output is ``f``, then pre of the site ``params``. Returns y."""
        y = y if y is not None else self.empty_rows()
        _run(*self.program(streams, f=f, params=params, y=y, stop_after=stop_after, debug=debug))
        return y

    def debug_to_torch(self, core: int = 0) -> dict:
        """Core ``core``'s intermediates of the last pre run with debug=True (layout.debug_from_page)."""
        page = self._matrix(self.dbg)[:, 32 * core : 32 * core + 32]
        out = layout.debug_from_page(page, self.lanes)
        if self.mm32 or self.sfpu_proj:
            out["proj"] = out["part_mix"]  # the projection is accumulated outside the scratch page
        if self.root_exchange:
            # the root core keeps the mixes and pre | post in the free half of its coefficient page
            coef = self._matrix(self.coef)[:, :32]
            out["mix"] = layout._cells(coef, 16 + 0, 1, self.lanes)
            out["g"] = layout._cells(coef, 16 + 2, 1, self.lanes)
        return out

    def probe(self, page: torch.Tensor, const: torch.Tensor):
        """Run the SFPU probe (kernels/ihc_sfpu.hpp probe_cells) on one page per core; returns core 0's
        (coefficient page, statistic page)."""
        x = self._pages(page.repeat(1, self.C // 32), ttnn.float32)
        params = SiteParams(
            w=self._pages(torch.zeros(32, self.C), ttnn.bfloat16),
            const=self._pages(const.repeat(1, self.K), ttnn.float32),
        )
        _run(*self.program(x, params=params, probe=True))
        return self._matrix(self.coef)[:, :32], self._matrix(self.stat)[:, :32]


def probe_pages():
    """The probe's input pages: distinct values, small enough for rsqrt and sigmoid to be checked on them."""
    index = (32 * torch.arange(32).reshape(32, 1) + torch.arange(32).reshape(1, 32)).float()
    return 0.5 + index / 256.0, 100.0 + index


def probe_device(device) -> dict:
    """What the SFPU probe finds on the device:
        lanes        which columns one SFPU register loads: "half" or "evenodd" (layout.SFPU_LANES)
        unpack_high  whether the unpacker writes fp32 pages into DEST slots above 3 (if not: IhcSite(low_unpack=True))
    On the mock Galaxy nothing runs: the documented values."""
    site = IhcSite(device, IHCConfig(dim=128), num_cores=1, sfpu_proj=False)
    page, const = probe_pages()
    out, _ = site.probe(page, const)
    if os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
        return {"lanes": layout.SFPU_LANES, "unpack_high": True}
    same = tuple(c for c in range(16) if torch.equal(out[0:4, c], page[0:4, c]))
    lanes = {tuple(range(8)): "half", tuple(range(0, 16, 2)): "evenodd"}.get(same)
    assert lanes is not None, f"SFPU probe: unit 0 came back on columns {same}"
    high = torch.equal(layout.unit_get(out, 5, lanes), layout.unit_get(const, 0, lanes))
    return {"lanes": lanes, "unpack_high": high}


def probe_projection(device, lanes: str, low_unpack: bool) -> dict:
    """The fn projection on the device against the host, on one core and 128 columns, for both forms of the
    matmul: {"mm16": relative error of the [16, 32] x [32, 16] form, "mm32": of the full-tile form}."""
    cfg = IHCConfig(dim=128)
    g = torch.Generator().manual_seed(77)
    fn = (torch.randn(8, cfg.n * cfg.dim, generator=g) * 0.02).to(torch.bfloat16).float()
    x = torch.randn(layout.ROWS, cfg.n, cfg.dim, generator=g)
    ref = x.reshape(layout.ROWS, -1).double() @ fn.double().t()
    found = {}
    for name, mm32 in (("mm16", False), ("mm32", True)):
        site = IhcSite(device, cfg, num_cores=1, lanes=lanes, low_unpack=low_unpack, mm32=mm32, sfpu_proj=False)
        params = site.site_params(fn, torch.zeros(8), torch.ones(2))
        site.pre(site.streams_to_device(x), params, debug=True)
        if os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
            found[name] = 0.0
            continue
        dev = site.debug_to_torch(0)["part_mix"].double()
        found[name] = ((dev - ref).norm() / ref.norm()).item()
    return found


def detect_lanes(device) -> str:
    return probe_device(device)["lanes"]


# ---------------------------------------------------------------------- golden
def golden_pre(cfg: IHCConfig, fn, base, scale, x: torch.Tensor):
    """reference.ihc.IHCWrap.hc_pre on x [T, 4, C] plus the RMSNorm statistic of its result.

    Returns (y [T, C], pre [T, 4], post [T, 4], inv_rms [T]), fp32."""
    wrap = IHCWrap(cfg)
    wrap.fn.data, wrap.base.data, wrap.scale.data = fn.float(), base.float(), scale.float()
    with torch.no_grad():
        pre, post = wrap.parametrize(x.float().unsqueeze(0))
        y, _ = wrap.hc_pre(x.float().unsqueeze(0))
    y = y[0]
    inv_rms = torch.rsqrt(y.pow(2).mean(-1) + cfg.norm_eps)
    return y, pre[0], post[0], inv_rms


def golden_post(cfg: IHCConfig, x: torch.Tensor, f: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
    """reference.ihc.IHCWrap.hc_post: x [T, 4, C], f [T, C], post [T, 4] -> [T, 4, C]."""
    wrap = IHCWrap(cfg)
    with torch.no_grad():
        return wrap.hc_post(f.float().unsqueeze(0), x.float().unsqueeze(0), post.float().unsqueeze(0))[0]
