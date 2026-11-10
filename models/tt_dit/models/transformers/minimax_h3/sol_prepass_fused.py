# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Fused Sol-Attn routing pre-pass: two custom device programs (``ttnn.generic_op``) around the two pooled all-gathers.

Same outputs and semantics as :class:`sol_prepass.SolPrepass` (its ``torch_reference`` is the reference for both):

* ``sol_block_pool`` (kernels ``sol_kernels/sol_pool_*.cpp``): per-64-token-block means of the local Q/K/V shards,
  one program for the three tensors, every core busy; replaces the three [R x NBP] pooling matmuls.
* ``sol_route`` (``sol_kernels/sol_route_*.cpp``): moments, thresholds, proxy scores, Q-chunk union, K-chunk union,
  uint32 mask and the column-bias tiles in one program; replaces the chain of small ttnn ops in SolPrepass.

Geometry constants are built on host once per geometry (cached per mesh, like SolPrepass) and the program
descriptors are cached with them; per call only the buffer addresses are patched (generic_op re-applies runtime
args on a program-cache hit).
"""

from __future__ import annotations

import json
import os
import struct

import torch

import ttnn

from .sol_prepass import (
    LOG2E,
    NEG,
    TILE,
    VAR_EPS,
    SolGeometry,
    _pad32,
    build_constants,
    compute_row_max,
    row_max_constants,
)

KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sol_kernels")
TILE_BYTES = 2048
BFP8_TILE_BYTES = 1088
DEFAULT_GATHER_KWARGS: dict = {"num_workers_per_link": 1, "barrier": False}
DEFAULT_GATHER_OWN_SEMS = True
DEFAULT_GATHER_PERSISTENT = False
DEFAULT_PERSISTENT_INTERNAL = True
L1_CB_BUDGET = 1_200_000
SCRATCH_PAGES = 8


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def build_fused_constants(
    geo: SolGeometry,
    *,
    band: int,
    sink_tokens: int,
    force_first_chunk: bool,
    scale: float,
    dense_prefix_queries: bool,
    block_map: tuple[int, ...] | None = None,
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
        block_map=block_map,
    )
    S, nbp, nqcp = geo.shards, geo.nb_pad, geo.nqc_pad
    m = ((c["band"] + c["sink_row"]) > 0.5).float()
    m2 = (m.transpose(-2, -1) @ c["qc_pool_t"]).transpose(-2, -1)
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
    assert S in (4, 8), S
    shsel = torch.zeros(1, 1, 8 * TILE, TILE)
    for n in range(8):
        for q in range(min(4, S)):
            shsel[0, 0, n * TILE, q * 8 + n] = 1.0
    out = {
        "const_or": const_or,
        "colconst": colconst,
        "qsel": qsel,
        "ksel": ksel,
        "erow": erow,
        "ones": ones,
        "shsel": shsel,
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
    block01 = ref["block_mask"]
    chunk01 = ((block01 @ consts["kc_pool"]) > 0.5).float()
    kept = ((chunk01 @ consts["kc_pool_t"]) > 0.5).float() * consts["valid_row"]
    bias = consts["len_bias_row"] + kept * NEG
    return {"block_mask": chunk01, "bias": bias, "kept_blk": kept}


class SolPrepassFused:
    """Drop-in replacement for SolPrepass (same constructor keywords, prepare() and __call__())."""

    _shared: dict = {}
    _generation: int = 0
    pooled_v_bfp8: bool = False
    keep_across_generations: bool = False

    @classmethod
    def new_generation(cls) -> None:
        """Drop the shared per-geometry state (uploaded constants, program descriptors) so the next call re-uploads it.

        The device contents of constants uploaded once are not preserved across generations (other models, such as
        the VAE decoder and the prompt encoder, run in between), so the constants are re-uploaded per generation.
        Called by MiniMaxH3Attention when the caller sets `sol_logical_n` for a request.

        With `keep_across_generations` the state stays (same buffers) and `prepare()` rewrites the constants' contents
        from their host copies on the first call of the new generation.
        """
        if not cls.keep_across_generations:
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
        row_max: bool = False,
        units_per_shard: bool | None = None,
        mask_reuse: int = 1,
        head_weights: bool | None = None,
        gather_kwargs: dict | None = None,
        gather_persistent: bool = False,
    ):
        assert mode in ("gather", "chunk", "compact", "pack"), mode
        assert thresh_type == "diag", "fused pre-pass implements the diag threshold (H3 policy)"
        self.mode = mode
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.sp_axis = sp_axis
        self.tau = float(tau)
        self.thresh_type = thresh_type
        self.band = band
        self.force_first_chunk = force_first_chunk
        self.dense_prefix_queries = dense_prefix_queries
        self.emit_bias = True if emit_bias is None else bool(emit_bias)
        self.emit_chunk_union = bool(emit_chunk_union) and mode != "chunk"
        self.route_units_per_shard = (
            (os.environ.get("SOL_ROUTE_UNITS_PER_SHARD", "1") == "1")
            if units_per_shard is None
            else bool(units_per_shard)
        )
        self.route_q_shared = os.environ.get("SOL_ROUTE_Q_SHARED", "1") == "1"
        self.route_units: tuple[int, int] | None = None
        self.head_weights = (
            (os.environ.get("SOL_HEAD_WEIGHTS", "0") == "1") if head_weights is None else bool(head_weights)
        )
        self._route_hw: ttnn.Tensor | None = None
        self._route_hw_buf: ttnn.Tensor | None = None
        self.math_fidelity = math_fidelity if math_fidelity is not None else ttnn.MathFidelity.HiFi4
        self.row_max = bool(row_max)
        self._row_max_consts: dict | None = None
        self._row_max_ckc = (
            ttnn.init_device_compute_kernel_config(
                mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
            )
            if self.row_max
            else None
        )
        self.mask_reuse = int(mask_reuse)
        self.gather_kwargs = dict(DEFAULT_GATHER_KWARGS if gather_kwargs is None else gather_kwargs)
        if os.environ.get("SOL_AG_KW"):
            self.gather_kwargs.update(json.loads(os.environ["SOL_AG_KW"]))
        self.gather_persistent = bool(gather_persistent or DEFAULT_GATHER_PERSISTENT)
        self.gather_own_sems = os.environ.get("SOL_AG_OWN_SEMS", "1" if DEFAULT_GATHER_OWN_SEMS else "0") == "1"
        self._ag_sems: list | None = None
        self._ag_idx = 0
        self.persistent_internal = (
            os.environ.get("SOL_PERSISTENT_INTERNAL", "1" if DEFAULT_PERSISTENT_INTERNAL else "0") == "1"
        )
        self._held_route: tuple | None = None
        self._reuse_left = 0
        self.last_mask_reused = False
        self._state: dict | None = None
        self._key = None
        self.last_density: float | None = None
        self.last_block_density: float | None = None

    def _upload(
        self, t: torch.Tensor, *, shard: bool, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, on_device: bool = True
    ) -> ttnn.Tensor:
        shape = tuple(self.mesh_device.shape)
        if shard:
            mapper = ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=[None, 0] if self.sp_axis == 1 else [0, None], mesh_shape=shape
            )
        else:
            mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        device = self.mesh_device if on_device else None
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, mesh_mapper=mapper)

    def _geo_host(self, geo: SolGeometry) -> torch.Tensor:
        """Valid rows per shard, the pool kernel's `geo` operand (one int per shard in a 32-int row)."""
        return torch.tensor(
            [[[[max(0, min(geo.rows, geo.logical_n - s * geo.rows))] + [0] * 31]] for s in range(geo.shards)],
            dtype=torch.int32,
        )

    _CONST_SPECS = {
        "const_or": (True, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "colconst": (False, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "qsel": (False, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "ksel": (False, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "erow": (False, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "ones": (False, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "shsel": (False, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "geo": (True, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
    }

    def _refresh_constants(self, st: dict) -> None:
        """Rewrite every uploaded constant in place (same buffers, so captured traces stay valid)."""
        for name, host in st["host_consts"].items():
            shard, dtype, layout = self._CONST_SPECS[name]
            ttnn.copy_host_to_device_tensor(
                self._upload(host, shard=shard, dtype=dtype, layout=layout, on_device=False), st[name]
            )

    # Prefix spread (prefix_spread.py): the logical block of each packed block, set by the attention; None = identity.
    block_map: tuple[int, ...] | None = None

    def prepare(self, geo: SolGeometry, sink_tokens: int, scale: float) -> None:
        key = (geo, sink_tokens, scale) if self.block_map is None else (geo, sink_tokens, scale, self.block_map)
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
            self.route_units_per_shard,
            str(self.math_fidelity),
            key,
        )
        st = SolPrepassFused._shared.get(shared_key)
        if st is not None and SolPrepassFused.keep_across_generations:
            if st.get("generation") != SolPrepassFused._generation:
                self._refresh_constants(st)
                st["generation"] = SolPrepassFused._generation
        if st is None:
            c = build_fused_constants(
                geo,
                band=self.band,
                sink_tokens=sink_tokens,
                force_first_chunk=self.force_first_chunk,
                scale=scale,
                dense_prefix_queries=self.dense_prefix_queries,
                block_map=self.block_map,
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
                "shsel": self._upload(c["shsel"], shard=False),
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
                "pool_prog": {},
                "route_prog": None,
                "pool_outs": None,
            }
            if SolPrepassFused.keep_across_generations:
                st["host_consts"] = {name: c[name] for name in self._CONST_SPECS if name != "geo"}
                st["host_consts"]["geo"] = self._geo_host(geo)
                st["generation"] = SolPrepassFused._generation
            SolPrepassFused._shared[shared_key] = st
        old = getattr(self, "_state", None)
        if old is not None and old is not st and old.get("pool_outs"):
            for t in {id(t): t for t in old["pool_outs"]}.values():
                ttnn.deallocate(t)
            old["pool_outs"] = None
        self._state = st
        self.geo = geo
        self._key = key
        self.reset_mask_reuse()

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
    def _kernel(
        path: str, cores, ct_args: list[int], rt_args, common: list[int], config, defines=()
    ) -> ttnn.KernelDescriptor:
        return ttnn.KernelDescriptor(
            kernel_source=os.path.join(KERNEL_DIR, path),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=list(ct_args),
            runtime_args=rt_args,
            common_runtime_args=list(common),
            config=config,
            defines=list(defines),
        )

    def _compute_config(self, full_sync: bool):
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

    def _pool_program(self, x: ttnn.Tensor, geo_t: ttnn.Tensor, outs: list[ttnn.Tensor]) -> ttnn.ProgramDescriptor:
        geo = self.geo
        _, H, R, D = x.shape
        assert R == geo.rows and R % TILE == 0 and D % TILE == 0
        RT, DT, NBPT = R // TILE, D // TILE, geo.nb_pad // TILE
        nx, ny, cores = self._cores()
        ct = [H, RT, DT, NBPT, R, nx, ny, 3 * H * NBPT]
        acc_in = [a for _ in range(3) for a in ttnn.TensorAccessorArgs(x).get_compile_time_args()]
        acc_in += list(ttnn.TensorAccessorArgs(geo_t).get_compile_time_args())
        acc_out = [a for o in outs for a in ttnn.TensorAccessorArgs(o).get_compile_time_args()]
        x_dtype = x.get_dtype()
        if x_dtype == ttnn.bfloat8_b:
            x_page, x_defines = BFP8_TILE_BYTES, [("SOL_POOL_X_BFP8", "1")]
        elif x_dtype == ttnn.bfloat16:
            x_page, x_defines = TILE_BYTES, []
        else:
            raise ValueError(f"Sol pre-pass: q / k / v must be bfloat16 or bfloat8_b, got {x_dtype}")
        v_bfp8 = outs[2].get_dtype() == ttnn.bfloat8_b
        v_defines = [("SOL_POOL_V_BFP8", "1")] if v_bfp8 else []
        cbs = [
            self._cb(0, 3 * 4 * DT, cores, page_size=x_page, dtype=x_dtype),
            self._cb(2, 3 * 4, cores),
            self._cb(3, 1, cores),
            self._cb(4, 1, cores, page_size=128, dtype=ttnn.uint32),
            self._cb(5, 32, cores),
            self._cb(16, 2 * DT, cores),
        ]
        if v_bfp8:
            cbs.append(self._cb(17, 2 * DT, cores, page_size=BFP8_TILE_BYTES, dtype=ttnn.bfloat8_b))
        reader = self._kernel(
            "sol_pool_reader.cpp", cores, ct + acc_in, [], [0, 0, 0, 0], ttnn.ReaderConfigDescriptor(), x_defines
        )
        writer = self._kernel(
            "sol_pool_writer.cpp", cores, ct + acc_out, [], [0, 0, 0], ttnn.WriterConfigDescriptor(), v_defines
        )
        compute = self._kernel(
            "sol_pool_compute.cpp", cores, ct, [], [], self._compute_config(full_sync=False), x_defines + v_defines
        )
        return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    def _run_pool(
        self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, persistent: bool | None = None
    ) -> list[ttnn.Tensor]:
        st = self._state
        _, H, _, D = q.shape
        if not (q.get_dtype() == k.get_dtype() == v.get_dtype()):
            raise ValueError(
                f"Sol pre-pass: q / k / v must share a dtype (cb 0 holds all three), got {q.get_dtype()} / "
                f"{k.get_dtype()} / {v.get_dtype()}"
            )
        persistent = self.persistent_internal if persistent is None else bool(persistent)
        v_out_dtype = ttnn.bfloat8_b if self.pooled_v_bfp8 else ttnn.bfloat16
        held = st["pool_outs"]
        if not (persistent and held is not None and held[2].get_dtype() == v_out_dtype):
            held = [
                ttnn.allocate_tensor_on_device(
                    ttnn.Shape([1, H, self.geo.nb_pad, D]),
                    v_out_dtype if i == 2 else ttnn.bfloat16,
                    ttnn.TILE_LAYOUT,
                    self.mesh_device,
                    ttnn.DRAM_MEMORY_CONFIG,
                )
                for i in range(3)
            ]
        outs = held
        if persistent:
            st["pool_outs"] = outs
        key = (q.get_dtype(), v_out_dtype)
        if key not in st["pool_prog"]:
            st["pool_prog"][key] = self._pool_program(q, st["geo"], outs)
        prog = st["pool_prog"][key]
        prog.kernels[0].common_runtime_args = [
            q.buffer_address(),
            k.buffer_address(),
            v.buffer_address(),
            st["geo"].buffer_address(),
        ]
        prog.kernels[1].common_runtime_args = [o.buffer_address() for o in outs]
        ttnn.generic_op([q, k, v, st["geo"], *outs], prog)
        return outs

    def _route_program(self, q_bar, kc, mask, bias, union) -> ttnn.ProgramDescriptor:
        st, geo = self._state, self.geo
        _, H, _, D = q_bar.shape
        S, DT, NBPT, NQCT = geo.shards, D // TILE, geo.nb_pad // TILE, geo.nqc_pad // TILE
        NKCLT = st["nkcl"] // TILE
        NBGT = S * NBPT
        assert NBPT * DT <= 8 * 32 and DT <= 4 and NKCLT <= 7, (NBPT, DT, NKCLT)
        mode = 0 if self.mode == "chunk" else (2 if self.emit_chunk_union else 1)
        log2_scale = st["scale"] * LOG2E
        TQ = geo.q_chunk // geo.block
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
        ]
        ins = [q_bar, kc, st["const_or"], st["colconst"], st["qsel"], st["ksel"], st["erow"], st["ones"]]
        acc_in = [a for t in ins for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        acc_out = [a for t in (mask, bias, union) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        nx, ny, cores = self._cores()
        hw = self._route_hw if self._route_hw is not None else mask
        hw_on = int(self._route_hw is not None)
        acc_out += [hw_on, geo.nb] + list(ttnn.TensorAccessorArgs(hw).get_compile_time_args())
        units_per_shard = 1 if (self.route_units_per_shard and mode == 1) else 0
        units = H * S if units_per_shard else H * NQCT
        self.route_units = (units, units_per_shard)
        ct += [nx, ny, units, units_per_shard]
        mask_t = NKCLT if mode == 0 else NBPT
        core_xy = self._core_noc_xy(nx, ny) if units_per_shard else []
        st["route_core_xy"] = core_xy
        const_mcast = os.environ.get("SOL_ROUTE_CONST_MCAST", "0") == "1"
        rects: list[int] = []
        if units_per_shard and not const_mcast:
            rects = [0]
        elif units_per_shard:
            assert H * S < nx * ny, "the constants sender must be a core without a unit"
            xs = sorted({core_xy[lx] & 0xFF for lx in range(nx)})
            ys = sorted({(core_xy[ly * nx] >> 8) & 0xFF for ly in range(ny)})
            assert ys == list(range(ys[0], ys[0] + ny)), ys
            runs: list[list[int]] = []
            for x in xs:
                if runs and x == runs[-1][-1] + 1:
                    runs[-1].append(x)
                else:
                    runs.append([x])
            for run in runs:
                rects += [run[0], ys[0], run[-1], ys[-1], len(run) * ny]
            rects = [len(runs)] + rects
        st["route_rects"] = rects
        sizes = {
            0: (NBPT if units_per_shard else NBGT) * DT,
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
        mask_rows_bytes = 32 * geo.nb_pad * 4
        if units_per_shard:
            sizes.update({23: (S + 3) // 4, 24: 8, 29: 8})
            del sizes[16]
        cbs = [self._cb(i, n, cores) for i, n in sizes.items()]
        if units_per_shard:
            cbs.append(self._cb(16, 2, cores, page_size=mask_rows_bytes, dtype=ttnn.uint32))
        row_bytes = geo.nkc_pad * 4
        cbs.append(self._cb(19, 32, cores, page_size=row_bytes, dtype=ttnn.uint32))
        cbs.append(self._cb(22, SCRATCH_PAGES, cores, page_size=geo.nb_pad * 4, dtype=ttnn.uint32))
        total = sum(n for n in sizes.values()) * TILE_BYTES + 32 * row_bytes + SCRATCH_PAGES * geo.nb_pad * 4
        total += 2 * mask_rows_bytes if units_per_shard else 0
        if total > L1_CB_BUDGET:
            raise ValueError(f"sol_route circular buffers need {total} B of L1 (> {L1_CB_BUDGET}); geometry too large")
        acc_extra = [a for t in (mask, st["shsel"]) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        reader = self._kernel(
            "sol_route_reader.cpp",
            cores,
            ct + acc_in + acc_extra + [int(units_per_shard and self.route_q_shared)],
            [],
            [0] * 10 + core_xy + rects,
            ttnn.ReaderConfigDescriptor(),
        )
        writer = self._kernel("sol_route_writer.cpp", cores, ct + acc_out, [], [0, 0, 0], ttnn.WriterConfigDescriptor())
        compute = self._kernel("sol_route_compute.cpp", cores, ct, [], [], self._compute_config(full_sync=True))
        sems = (
            [ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in (0, 1, 2)]
            if units_per_shard
            else []
        )
        return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=sems, cbs=cbs)

    def _core_noc_xy(self, nx: int, ny: int) -> list[int]:
        """NoC coordinates (x | y << 8) of every logical worker core, index y*nx + x. One program with common runtime
        args runs on every chip of the mesh, so the coordinates must agree across chips (virtual worker coordinates do);
        verified against every mesh coordinate when the experimental per-chip query is available."""
        md = self.mesh_device
        logical = [(x, y) for y in range(ny) for x in range(nx)]
        xy = [md.worker_core_from_logical_core(ttnn.CoreCoord(x, y)) for x, y in logical]
        per_chip = getattr(getattr(getattr(ttnn, "_ttnn", None), "multi_device", None), "experimental", None)
        per_chip = getattr(per_chip, "worker_core_from_logical_core", None)
        if per_chip is not None:
            rows, cols = tuple(md.shape)
            for r in range(rows):
                for c in range(cols):
                    mc = ttnn.MeshCoordinate(r, c)
                    for (x, y), w0 in zip(logical, xy):
                        w = per_chip(md, mc, ttnn.CoreCoord(x, y))
                        if (w.x, w.y) != (w0.x, w0.y):
                            raise ValueError(
                                f"worker core ({x},{y}) maps to NoC ({w.x},{w.y}) on chip {mc} but ({w0.x},{w0.y}) on "
                                "chip (0,0): the route program's common core coordinates would be wrong"
                            )
        return [w.x | (w.y << 8) for w in xy]

    def _run_route(
        self, q_bar: ttnn.Tensor, kc: ttnn.Tensor
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None, ttnn.Tensor | None]:
        st, geo = self._state, self.geo
        _, H, _, _ = q_bar.shape
        width = geo.nkc_pad if self.mode == "chunk" else geo.n_blocks_pad
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
            bias = st["ones"]
        if self.emit_chunk_union:
            union = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, H, geo.nqc_pad, geo.nkc_pad]),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                self.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            union = mask
        hw_ok = self.head_weights and self.mode != "chunk" and self.route_units_per_shard and geo.nqc_pad // TILE <= 8
        if self.head_weights and not hw_ok:
            print("SolPrepassFused: SOL_HEAD_WEIGHTS needs (head, shard) units, the block-form mask and NQCT <= 8: off")
        hw_shape = [1, 1, H * geo.shards, 8]
        if not hw_ok:
            self._route_hw = None
        elif self._route_hw_buf is None or list(self._route_hw_buf.shape) != hw_shape:
            self._route_hw_buf = ttnn.allocate_tensor_on_device(
                ttnn.Shape(hw_shape),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                self.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            self._route_hw = self._route_hw_buf
        else:
            self._route_hw = self._route_hw_buf
        if st["route_prog"] is None:
            st["route_prog"] = self._route_program(q_bar, kc, mask, bias, union)
        prog = st["route_prog"]
        ins = [q_bar, kc, st["const_or"], st["colconst"], st["qsel"], st["ksel"], st["erow"], st["ones"]]
        prog.kernels[0].common_runtime_args = (
            [t.buffer_address() for t in ins]
            + [mask.buffer_address(), st["shsel"].buffer_address()]
            + st["route_core_xy"]
            + st["route_rects"]
        )
        hw = self._route_hw if self._route_hw is not None else mask
        prog.kernels[1].common_runtime_args = [
            mask.buffer_address(),
            bias.buffer_address(),
            union.buffer_address(),
            hw.buffer_address(),
        ]
        ttnn.generic_op([*ins, st["shsel"], mask, bias, union] + ([hw] if self._route_hw is not None else []), prog)
        return mask, (bias if self.emit_bias else None), (union if self.emit_chunk_union else None)

    def _gather(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """All-gather a pooled [1, H, nb_pad, D] tensor along dim 2 over the SP axis with the configured arguments."""
        ccl = self.ccl_manager
        if not self.gather_kwargs and not self.gather_persistent:
            return ccl.all_gather(t, dim=2, mesh_axis=self.sp_axis, use_hyperparams=False)
        kw = dict(self.gather_kwargs)
        topo = kw.pop("topology", None)
        barrier = kw.pop("barrier", True) and not self.gather_persistent
        if isinstance(topo, str):
            topo = {"ring": ttnn.Topology.Ring, "linear": ttnn.Topology.Linear}[topo.lower()]
        if self.gather_own_sems:
            if self._ag_sems is None:
                self._ag_sems = [
                    [ttnn.create_global_semaphore(self.mesh_device, ccl.ccl_cores, 0) for _ in range(2)]
                    for _ in range(2)
                ]
            sems = self._ag_sems[self._ag_idx]
            self._ag_idx ^= 1
        else:
            sems = ccl.get_ag_ping_pong_semaphore(self.sp_axis)
        return ttnn.experimental.all_gather_async(
            t,
            persistent_output_buffer=(
                ccl.get_ag_ping_pong_buffer(t.shape, 2, self.sp_axis, dtype=t.get_dtype())
                if self.gather_persistent
                else None
            ),
            dim=2,
            multi_device_global_semaphore=sems,
            num_links=kw.pop("num_links", ccl.num_links),
            topology=topo if topo is not None else ccl.topology,
            cluster_axis=self.sp_axis,
            barrier_semaphore=ccl.get_barrier_semaphore(self.sp_axis) if barrier else None,
            **kw,
        )

    def pool(self, q, k, v) -> dict:
        """Block means only (q_bar, kc_local, vc_local), for unit tests."""
        qb, kcl, vcl = self._run_pool(q, k, v, persistent=False)
        return {"q_bar": qb, "kc_local": kcl, "vc_local": vcl}

    def __call__(
        self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, *, scale: float, report_density: bool = False
    ) -> dict:
        assert abs(scale - self._state["scale"]) < 1e-12, "scale is baked into the prepared constants"
        geo = self.geo
        row_max = None
        if self.row_max:
            if self._row_max_consts is None:
                self._row_max_consts = {
                    name: self._upload(t, shard=False) for name, t in row_max_constants(int(q.shape[3]), scale).items()
                }
            row_max = compute_row_max(
                q,
                k,
                consts=self._row_max_consts,
                ccl_manager=self.ccl_manager,
                sp_axis=self.sp_axis,
                compute_kernel_config=self._row_max_ckc,
            )
        q_bar, kc_local, vc_local = self._run_pool(q, k, v)
        kc = self._gather(kc_local)
        vc = self._gather(vc_local)
        if not self.persistent_internal:
            ttnn.deallocate(kc_local)
            ttnn.deallocate(vc_local)
        reuse = self.mask_reuse > 1 and self._held_route is not None and self._reuse_left > 0
        if reuse:
            mask, bias, union, self._route_hw = self._held_route
            self._reuse_left -= 1
        else:
            mask, bias, union = self._run_route(q_bar, kc)
            if self.mask_reuse > 1:
                self._held_route = (mask, bias, union, self._route_hw)
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
            self.last_block_density = None if self.mode == "chunk" else self.last_density
        if not self.persistent_internal:
            ttnn.deallocate(q_bar)
        out = {"block_mask": mask, "pooled_k": kc, "pooled_v": vc, "bias": bias, "row_max": row_max}
        if union is not None:
            out["chunk_union"] = union
        if self._route_hw is not None:
            out["head_weights"] = self._route_hw
        return out
