# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Fused Sol-Attn routing pre-pass: two custom device programs (``ttnn.generic_op``) around the two pooled all-gathers.

Same outputs and semantics as :class:`sol_prepass.SolPrepass` (its ``torch_reference`` is the reference for both):

* ``sol_block_pool`` (kernels ``sol_kernels/sol_pool_*.cpp``): per-64-token-block means of the local Q/K/V shards,
  one program for the three tensors, every core busy; replaces the three [R x NBP] pooling matmuls (8.8 ms at 10 s).
* ``sol_route`` (``sol_kernels/sol_route_*.cpp``): moments, thresholds, proxy scores, Q-chunk union, K-chunk union,
  uint32 mask and the column-bias tiles in one program; replaces ~56 small ttnn ops.

Geometry constants are built on host once per geometry (cached per mesh, like SolPrepass) and the program
descriptors are cached with them; per call only the buffer addresses are patched (generic_op re-applies runtime
args on a program-cache hit). Plan and measurements: ~/sol_attn_work/PREPASS_FUSED_PLAN.md.
"""

from __future__ import annotations

import os
import struct
import time

import torch

import ttnn

from .sol_prepass import LOG2E, NEG, TILE, VAR_EPS, SolGeometry, _pad32, build_constants

KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sol_kernels")
TILE_BYTES = 2048
L1_CB_BUDGET = 1_200_000  # bytes of circular buffers per core we allow (Blackhole L1 = 1.5 MB)
SCRATCH_PAGES = 8


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def build_fused_constants(
    geo: SolGeometry, *, band: int, sink_tokens: int, force_first_chunk: bool, scale: float, dense_prefix_queries: bool
) -> dict:
    """Host constants of the route program (torch fp32), derived from sol_prepass.build_constants.

    const_or [S,1,nqc_pad,n_blocks_pad]: the threshold-independent part of block01 (band | sink | dense-prefix rows,
      OR-ed over the Q blocks of each chunk, | forced block 0 per shard) & full blocks -- exactly what
      torch_reference produces when no score passes the threshold.
    colconst [1,1,3*S*32,nb_pad]: for kind in (full, valid, len_bias) and shard s, 32 identical rows of that
      shard's column constant (a tile operand for the compute kernel).
    qsel [1,1,nqc_pad,nb_pad]: 1 where the Q block belongs to the Q chunk. ksel [1,1,nb_pad,nkcl]: 1 where the
      local block belongs to the local K chunk (geometry only). erow [1,1,1024,32]: tile r has column r = 1
      (E_r @ T copies row r of T into every row). ones [1,1,32,32]: row 0 = 1.
    """
    c = build_constants(
        geo,
        band=band,
        sink_tokens=sink_tokens,
        force_first_chunk=force_first_chunk,
        scale=scale,
        dense_prefix_queries=dense_prefix_queries,
    )
    S, nbp, nqcp = geo.shards, geo.nb_pad, geo.nqc_pad
    m = ((c["band"] + c["sink_row"]) > 0.5).float()  # [S,1,nbp,nbg]
    m2 = (m.transpose(-2, -1) @ c["qc_pool_t"]).transpose(-2, -1)  # [S,1,nqcp,nbg]
    const_or = ((m2 + c["force_row"]) > 0.5).float() * c["full_row"]
    colconst = torch.zeros(1, 1, 3 * S * TILE, nbp)
    for kind, row in enumerate((c["full_row"], c["valid_row"], c["len_bias_row"])):
        for s in range(S):
            colconst[0, 0, (kind * S + s) * TILE : (kind * S + s + 1) * TILE, :] = row[0, 0, 0, s * nbp : (s + 1) * nbp]
    qsel = c["qc_pool_t"][0, 0].T.contiguous().view(1, 1, nqcp, nbp)
    nkcl = _pad32(geo.nkc)
    ksel = torch.zeros(1, 1, nbp, nkcl)
    for b in range(geo.nb):
        ksel[0, 0, b, (b * geo.block) // geo.k_chunk] = 1.0
    erow = torch.zeros(1, 1, 32 * TILE, TILE)
    for r in range(TILE):
        erow[0, 0, r * TILE : (r + 1) * TILE, r] = 1.0
    ones = torch.zeros(1, 1, TILE, TILE)
    ones[0, 0, 0, :] = 1.0
    out = {
        "const_or": const_or,
        "colconst": colconst,
        "qsel": qsel,
        "ksel": ksel,
        "erow": erow,
        "ones": ones,
        "n_valid_blocks": c["n_valid_blocks"],
        "nkcl": nkcl,
    }
    out.update(
        {
            "kc_pool": c["kc_pool"],
            "kc_pool_t": c["kc_pool_t"],
            "valid_row": c["valid_row"],
            "len_bias_row": c["len_bias_row"],
        }
    )
    return out


def torch_reference_chunk(ref: dict, consts: dict) -> dict:
    """Chunk-mode outputs from torch_reference's gather-mode block mask (mirror of SolPrepass._finish_chunk_mode)."""
    block01 = ref["block_mask"]  # [S,H,nqc_pad,nbg]
    chunk01 = ((block01 @ consts["kc_pool"]) > 0.5).float()  # [S,H,nqc_pad,nkc_pad]
    kept = ((chunk01 @ consts["kc_pool_t"]) > 0.5).float() * consts["valid_row"]
    bias = consts["len_bias_row"] + kept * NEG
    return {"block_mask": chunk01, "bias": bias, "kept_blk": kept}


class SolPrepassFused:
    """Drop-in replacement for SolPrepass (same constructor keywords, prepare() and __call__())."""

    _shared: dict = {}
    _generation: int = 0

    @classmethod
    def new_generation(cls) -> None:
        """Drop the shared per-geometry state (uploaded constants, program descriptors) so the next call re-uploads it.

        E13 (2026-10-20): constants uploaded once and kept across generations lose their device contents between two
        generations (the VAE decode / prompt encoder run in between; see TTNN_SHARDED_UPLOAD_BUG.md), which made every
        sparse generation after the first of a process wrong at 5 s / 15 s (E12: PSNR 23.5 -> 16.6, mask 70 % denser).
        Re-uploading per generation restores the first generation bit for bit at no measurable cost. Called by
        MiniMaxH3Attention when the caller sets `sol_logical_n` for a request.
        """
        cls._shared.clear()
        cls._generation += 1

    def __init__(
        self,
        *,
        mesh_device,
        ccl_manager,
        sp_axis: int,
        tau: float,
        thresh_type: str = "diag",
        band: int = 1,
        force_first_chunk: bool = True,
        dense_prefix_queries: bool = True,
        mode: str = "chunk",
        emit_bias: bool | None = None,
        math_fidelity=None,
        emit_chunk_union: bool = False,
        debug_moments: bool = False,
        units_per_shard: bool | None = None,
        mask_reuse: int = 1,
    ):
        # "compact" (tier 3, DESIGN_BLOCK_COMPACT.md) consumes the gather-form block mask: same outputs as "gather".
        # emit_chunk_union=True additionally writes the (Q chunk x K chunk) union row as "chunk_union" (dormant option).
        assert mode in ("gather", "chunk", "compact"), mode
        assert thresh_type == "diag", "fused pre-pass implements the diag threshold (H3 policy)"
        self.mode = mode
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.sp_axis = sp_axis
        self.tau = float(tau)
        self.thresh_type = thresh_type
        self.band = band
        # compact mode keeps the gather-mode convention: block 0 of every shard is forced (a Q chunk computing zero
        # chunks in a ring iteration deadlocks the accumulator staging protocol; sol-vid2, 2026-10-16).
        self.force_first_chunk = force_first_chunk
        self.dense_prefix_queries = dense_prefix_queries
        # Same convention as SolPrepass: None -> TT_SOL_EMIT_BIAS (default 1 until the ring op generates the bias).
        self.emit_bias = (os.environ.get("TT_SOL_EMIT_BIAS", "1") == "1") if emit_bias is None else bool(emit_bias)
        self.emit_chunk_union = bool(emit_chunk_union) and mode != "chunk"
        self.debug_moments = bool(debug_moments)  # route program writes mean/E2/var tiles into the bias tensor (tests)
        # E1 (sol_attn_next_optimizations item 2): route-program unit = (head, shard) instead of (head, Q-chunk tile row),
        # H*S units instead of H*NQCT (14 at NQCT = 1) over the 120-core grid. None -> SOL_ROUTE_UNITS_PER_SHARD (default
        # 1); 0 reproduces the committed program. Applies to the gather-form mask (MODE 1) only, see _route_program.
        self.route_units_per_shard = (
            (os.environ.get("SOL_ROUTE_UNITS_PER_SHARD", "1") == "1")
            if units_per_shard is None
            else bool(units_per_shard)
        )
        self.route_units: tuple[int, int] | None = None  # (unit count, units_per_shard) of the last built route program
        self.math_fidelity = math_fidelity if math_fidelity is not None else ttnn.MathFidelity.HiFi4
        # E10 mask reuse: N > 1 keeps the route outputs (block mask, bias, union) of a call and reuses them for the next
        # N-1 calls, skipping the route program; the pool program and the two gathers still run every call because the
        # ring op's correction term needs the current block means. 1 (default) = route every call (committed behaviour).
        self.mask_reuse = int(mask_reuse)
        self._held_route: tuple | None = None  # (mask, bias, union) of the last routed call, while mask_reuse > 1
        self._reuse_left = 0  # calls the held route may still serve
        self.last_mask_reused = False  # whether the last call reused the held mask (diagnostics)
        self._state: dict | None = None
        self._key = None
        self.last_density: float | None = None
        self.host_times: dict = {}  # last call's host-side timing of allocation vs launch, per program (seconds)

    # ------------------------------------------------------------------ constants / descriptors

    def _upload(self, t: torch.Tensor, *, shard: bool, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
        shape = tuple(self.mesh_device.shape)
        if shard:
            mapper = ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=[None, 0] if self.sp_axis == 1 else [0, None], mesh_shape=shape
            )
        else:
            mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.mesh_device, mesh_mapper=mapper)

    def prepare(self, geo: SolGeometry, sink_tokens: int, scale: float) -> None:
        key = (geo, sink_tokens, scale)
        if key == self._key:
            return
        shared_key = (
            id(self.mesh_device),
            self.sp_axis,
            self.mode,
            self.band,
            self.force_first_chunk,
            self.dense_prefix_queries,
            self.tau,
            self.emit_bias,
            self.emit_chunk_union,
            self.debug_moments,
            self.route_units_per_shard,
            str(self.math_fidelity),
            key,
        )
        st = SolPrepassFused._shared.get(shared_key)
        if st is None:
            c = build_fused_constants(
                geo,
                band=self.band,
                sink_tokens=sink_tokens,
                force_first_chunk=self.force_first_chunk,
                scale=scale,
                dense_prefix_queries=self.dense_prefix_queries,
            )
            st = {
                "geo": geo,
                "scale": scale,
                "n_valid_blocks": c["n_valid_blocks"],
                "nkcl": c["nkcl"],
                "const_or": self._upload(c["const_or"], shard=True),
                "colconst": self._upload(c["colconst"], shard=False),
                "qsel": self._upload(c["qsel"], shard=False),
                "ksel": self._upload(c["ksel"], shard=False),
                "erow": self._upload(c["erow"], shard=False),
                "ones": self._upload(c["ones"], shard=False),
                "geo": self._upload(
                    torch.tensor(
                        [
                            [[[max(0, min(geo.rows, geo.logical_n - s * geo.rows))] + [0] * 31]]
                            for s in range(geo.shards)
                        ],
                        dtype=torch.int32,
                    ),
                    shard=True,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                "pool_prog": None,
                "route_prog": None,
            }
            SolPrepassFused._shared[shared_key] = st
        self._state = st
        self.geo = geo
        self._key = key
        self.reset_mask_reuse()  # a held mask belongs to the previous geometry

    def reset_mask_reuse(self) -> None:
        """Drop the held route outputs: the next call routes afresh (call at the start of every generation)."""
        self._held_route = None
        self._reuse_left = 0

    def _cores(self):
        g = self.mesh_device.compute_with_storage_grid_size()
        nx, ny = g.x, g.y
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))])
        return nx, ny, ranges

    @staticmethod
    def _split_units(n_units: int, nx: int, ny: int) -> list[tuple[ttnn.CoreCoord, int, int]]:
        n_cores = nx * ny
        out = []
        for ci in range(n_cores):
            x, y = ci % nx, ci // nx
            start = ci * n_units // n_cores
            end = (ci + 1) * n_units // n_cores
            out.append((ttnn.CoreCoord(x, y), start, end - start))
        return out

    @staticmethod
    def _cb(idx: int, pages: int, cores, page_size: int = TILE_BYTES, dtype=ttnn.bfloat16) -> ttnn.CBDescriptor:
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page_size)
        return ttnn.CBDescriptor(total_size=pages * page_size, core_ranges=cores, format_descriptors=[fmt])

    @staticmethod
    def _kernel(path: str, cores, ct_args: list[int], rt_args, common: list[int], config) -> ttnn.KernelDescriptor:
        return ttnn.KernelDescriptor(
            kernel_source=os.path.join(KERNEL_DIR, path),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=list(ct_args),
            runtime_args=rt_args,
            common_runtime_args=list(common),
            config=config,
        )

    def _compute_config(self, full_sync: bool):
        # fp32 DST holds 4 tiles per section in half-sync mode and 8 in full-sync mode (tests/ttnn/unit_tests/kernel_lib/
        # test_chain_oob.py). The route kernel addresses DST tiles 0..7, so it must run full-sync; the pool kernel
        # uses DT (<= 4) tiles and keeps the math/pack overlap of half-sync.
        return ttnn.ComputeConfigDescriptor(
            math_fidelity=self.math_fidelity, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=full_sync
        )

    def _sp_ranges(self) -> list[ttnn.MeshCoordinateRange]:
        rows, cols = tuple(self.mesh_device.shape)
        out = []
        for s in range(self.geo.shards):
            if self.sp_axis == 1:
                out.append(ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, s), ttnn.MeshCoordinate(rows - 1, s)))
            else:
                out.append(ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(s, 0), ttnn.MeshCoordinate(s, cols - 1)))
        return out

    # ------------------------------------------------------------------ pool program

    def _pool_program(self, x: ttnn.Tensor, geo_t: ttnn.Tensor, outs: list[ttnn.Tensor]) -> ttnn.ProgramDescriptor:
        geo = self.geo
        _, H, R, D = x.shape
        assert R == geo.rows and R % TILE == 0 and D % TILE == 0
        RT, DT, NBPT = R // TILE, D // TILE, geo.nb_pad // TILE
        nx, ny, cores = self._cores()
        ct = [H, RT, DT, NBPT, R, nx, ny, 3 * H * NBPT]  # grid + unit count: kernels derive their unit range per core
        acc_in = [a for _ in range(3) for a in ttnn.TensorAccessorArgs(x).get_compile_time_args()]
        acc_in += list(ttnn.TensorAccessorArgs(geo_t).get_compile_time_args())
        acc_out = [a for o in outs for a in ttnn.TensorAccessorArgs(o).get_compile_time_args()]
        cbs = [
            self._cb(0, 32, cores),
            self._cb(2, 8, cores),
            self._cb(3, 1, cores),
            self._cb(4, 1, cores, page_size=128, dtype=ttnn.uint32),
            self._cb(16, 2 * DT, cores),
        ]
        reader = self._kernel(
            "sol_pool_reader.cpp", cores, ct + acc_in, [], [0, 0, 0, 0], ttnn.ReaderConfigDescriptor()
        )
        writer = self._kernel("sol_pool_writer.cpp", cores, ct + acc_out, [], [0, 0, 0], ttnn.WriterConfigDescriptor())
        compute = self._kernel("sol_pool_compute.cpp", cores, ct, [], [], self._compute_config(full_sync=False))
        return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    def _run_pool(self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor) -> list[ttnn.Tensor]:
        st = self._state
        _, H, _, D = q.shape
        t0 = time.perf_counter()
        outs = [
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, H, self.geo.nb_pad, D]),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                self.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(3)
        ]
        t1 = time.perf_counter()
        if st["pool_prog"] is None:
            st["pool_prog"] = self._pool_program(q, st["geo"], outs)
        prog = st["pool_prog"]
        prog.kernels[0].common_runtime_args = [
            q.buffer_address(),
            k.buffer_address(),
            v.buffer_address(),
            st["geo"].buffer_address(),
        ]
        prog.kernels[1].common_runtime_args = [o.buffer_address() for o in outs]
        t2 = time.perf_counter()
        ttnn.generic_op([q, k, v, st["geo"], *outs], prog)
        t3 = time.perf_counter()
        self.host_times["pool"] = {"alloc": t1 - t0, "patch": t2 - t1, "launch": t3 - t2}
        return outs

    # ------------------------------------------------------------------ route program

    def _route_program(self, q_bar, kc, mask, bias, union) -> ttnn.ProgramDescriptor:
        st, geo = self._state, self.geo
        _, H, _, D = q_bar.shape
        S, DT, NBPT, NQCT = geo.shards, D // TILE, geo.nb_pad // TILE, geo.nqc_pad // TILE
        NKCLT = st["nkcl"] // TILE
        NBGT = S * NBPT
        assert NBPT * DT <= 8 * 32 and DT <= 4 and NKCLT <= 7, (NBPT, DT, NKCLT)
        mode = 0 if self.mode == "chunk" else (2 if self.emit_chunk_union else 1)
        log2_scale = st["scale"] * LOG2E
        TQ = geo.q_chunk // geo.block  # Q-block tile rows per Q-chunk tile row
        assert geo.q_chunk % geo.block == 0 and 1 <= TQ <= 8, geo.q_chunk
        ct = [
            H,
            S,
            NBPT,
            DT,
            NQCT,
            NKCLT,
            geo.nqc,
            geo.nkc,
            geo.nkc_pad,
            mode,
            1 if self.emit_bias else 0,
            _f32_bits(self.tau),
            _f32_bits(VAR_EPS / (log2_scale * log2_scale)),
            _f32_bits(NEG),
            _f32_bits(1.0 / st["n_valid_blocks"]),
            TQ,
            1 if self.debug_moments else 0,
        ]
        ins = [q_bar, kc, st["const_or"], st["colconst"], st["qsel"], st["ksel"], st["erow"], st["ones"]]
        acc_in = [a for t in ins for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        acc_out = [a for t in (mask, bias, union) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        nx, ny, cores = self._cores()
        # Unit = (head, shard): H*S units. The chunk-form outputs (MODE 0 mask, MODE 2 union row) assemble whole rows from
        # every shard in the writer's L1 row buffer (their per-shard word ranges are not NoC-aligned), so those modes keep
        # the (head, Q-chunk tile row) unit with H*NQCT units; the kernels static_assert the same rule.
        units_per_shard = 1 if (self.route_units_per_shard and mode == 1) else 0
        units = H * S if units_per_shard else H * NQCT
        self.route_units = (units, units_per_shard)
        ct += [nx, ny, units, units_per_shard]  # grid + unit count + unit kind; kernels derive their range per core
        mask_t = NKCLT if mode == 0 else NBPT  # per shard
        sizes = {
            0: NBGT * DT,
            1: NBPT * DT,
            2: 1,
            3: NQCT * NBPT,
            4: NBPT * NKCLT,
            5: 32,
            6: NBPT,
            7: 3 * NBPT,
            8: 8,
            9: 8,
            10: TQ * DT,
            11: TQ,
            12: TQ,
            13: TQ * NBPT,
            14: NBPT,
            15: NKCLT,
            16: mask_t,
            17: NBPT,
            18: 16,
            20: DT,
            21: NKCLT,
        }
        cbs = [self._cb(i, n, cores) for i, n in sizes.items()]
        row_bytes = geo.nkc_pad * 4  # chunk-form row buffer: 32 complete rows
        cbs.append(self._cb(19, 32, cores, page_size=row_bytes, dtype=ttnn.uint32))
        cbs.append(self._cb(22, SCRATCH_PAGES, cores, page_size=geo.nb_pad * 4, dtype=ttnn.uint32))  # gather-form ring
        total = sum(n for n in sizes.values()) * TILE_BYTES + 32 * row_bytes + SCRATCH_PAGES * geo.nb_pad * 4
        if total > L1_CB_BUDGET:
            raise ValueError(f"sol_route circular buffers need {total} B of L1 (> {L1_CB_BUDGET}); geometry too large")
        reader = self._kernel("sol_route_reader.cpp", cores, ct + acc_in, [], [0] * 8, ttnn.ReaderConfigDescriptor())
        writer = self._kernel("sol_route_writer.cpp", cores, ct + acc_out, [], [0, 0, 0], ttnn.WriterConfigDescriptor())
        compute = self._kernel("sol_route_compute.cpp", cores, ct, [], [], self._compute_config(full_sync=True))
        return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    def _run_route(
        self, q_bar: ttnn.Tensor, kc: ttnn.Tensor
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None, ttnn.Tensor | None]:
        st, geo = self._state, self.geo
        _, H, _, _ = q_bar.shape
        width = geo.nkc_pad if self.mode == "chunk" else geo.n_blocks_pad
        t0 = time.perf_counter()
        mask = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, H, geo.nqc_pad, width]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.emit_bias:
            bias = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, H, geo.nqc_pad * TILE, geo.n_blocks_pad]),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                self.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            bias = st["ones"]  # unused accessor target
        if self.emit_chunk_union:
            union = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, H, geo.nqc_pad, geo.nkc_pad]),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                self.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            union = mask  # unused accessor target
        t1 = time.perf_counter()
        if st["route_prog"] is None:
            st["route_prog"] = self._route_program(q_bar, kc, mask, bias, union)
        prog = st["route_prog"]
        ins = [q_bar, kc, st["const_or"], st["colconst"], st["qsel"], st["ksel"], st["erow"], st["ones"]]
        prog.kernels[0].common_runtime_args = [t.buffer_address() for t in ins]
        prog.kernels[1].common_runtime_args = [mask.buffer_address(), bias.buffer_address(), union.buffer_address()]
        t2 = time.perf_counter()
        ttnn.generic_op([*ins, mask, bias, union], prog)
        t3 = time.perf_counter()
        self.host_times["route"] = {"alloc": t1 - t0, "patch": t2 - t1, "launch": t3 - t2}
        return mask, (bias if self.emit_bias else None), (union if self.emit_chunk_union else None)

    # ------------------------------------------------------------------ forward

    def pool(self, q, k, v) -> dict:
        """Block means only (q_bar, kc_local, vc_local), for unit tests."""
        qb, kcl, vcl = self._run_pool(q, k, v)
        return {"q_bar": qb, "kc_local": kcl, "vc_local": vcl}

    def __call__(
        self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, *, scale: float, report_density: bool = False
    ) -> dict:
        assert abs(scale - self._state["scale"]) < 1e-12, "scale is baked into the prepared constants"
        geo = self.geo
        q_bar, kc_local, vc_local = self._run_pool(q, k, v)
        kc = self.ccl_manager.all_gather(kc_local, dim=2, mesh_axis=self.sp_axis, use_hyperparams=False)
        vc = self.ccl_manager.all_gather(vc_local, dim=2, mesh_axis=self.sp_axis, use_hyperparams=False)
        ttnn.deallocate(kc_local)
        ttnn.deallocate(vc_local)
        reuse = self.mask_reuse > 1 and self._held_route is not None and self._reuse_left > 0
        if reuse:
            mask, bias, union = self._held_route
            self._reuse_left -= 1
            self.host_times["route"] = {"alloc": 0.0, "patch": 0.0, "launch": 0.0}
        else:
            mask, bias, union = self._run_route(q_bar, kc)
            if self.mask_reuse > 1:
                self._held_route = (mask, bias, union)  # the references keep the DRAM buffers alive across calls
                self._reuse_left = self.mask_reuse - 1
        self.last_mask_reused = reuse
        if report_density:
            host = ttnn.to_torch(
                mask,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device,
                    dims=[0, 1] if self.sp_axis == 0 else [1, 0],
                    mesh_shape=tuple(self.mesh_device.shape),
                ),
            )
            sub = host[:, : host.shape[1] // tuple(self.mesh_device.shape)[1 - self.sp_axis], : geo.nqc].float()
            if self.mode == "chunk":
                cols = torch.cat([torch.arange(s * geo.nkc, s * geo.nkc + geo.nkc) for s in range(geo.shards)])
            else:
                cols = torch.cat([torch.arange(s * geo.nb_pad, s * geo.nb_pad + geo.nb) for s in range(geo.shards)])
            self.last_density = float(sub[..., cols].mean())
        ttnn.deallocate(q_bar)
        out = {"block_mask": mask, "pooled_k": kc, "pooled_v": vc, "bias": bias}
        if union is not None:
            out["chunk_union"] = union
        return out
