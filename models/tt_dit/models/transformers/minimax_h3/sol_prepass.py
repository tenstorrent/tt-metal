# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Sol-Attn routing pre-pass for the ring-joint SDPA path: two custom device programs (``ttnn.generic_op``) around
the two pooled all-gathers.

* ``sol_block_pool`` (kernels ``sol_kernels/sol_pool_*.cpp``): per-64-token-block means of the local Q/K/V shards,
  one program for the three tensors, every core busy.
* two all-gathers of the pooled K and V along the SP ring (``ttnn.experimental.all_gather_async``).
* ``sol_route`` (``sol_kernels/sol_route_*.cpp``): moments, thresholds, proxy scores, Q-chunk union, K-chunk union,
  uint32 mask and the column-bias tiles in one program.

Per device it produces, from the local Q/K/V shards [1, H_local, R, D]:

* ``block_mask``  uint32 row-major [1, H, n_q_chunks_pad, n_blocks_pad]: 1 = this (local Q chunk, global
  64-token block) pair is attended exactly (the kernel gathers kept blocks into dense K/V chunks), 0 = the
  block goes through the pooled correction. Only full 64-token blocks inside logical_n are ever exact;
  block 0 of every shard is forced exact (one gathered chunk per ring iteration). In "chunk" mode the mask is
  per K chunk instead: [1, H, n_q_chunks_pad, n_k_chunks_pad].
* ``pooled_k`` / ``pooled_v``  bf16 tile [1, H, n_blocks_pad, D]: per-64-token block mean K / mean V
  for the WHOLE sequence (all-gathered along SP), per-shard blocking.
The pooled-column bias (ln(block token count) / scale for approximated blocks, a large negative value for exact and
padding blocks) is not produced here: the ring op's reader generates the tile rows on device from the mask row and
the block geometry (sol_bias_gen.hpp). Interface contract: SOL_ATTN_INTERFACE.md next to this file.

Geometry: blocks are 64 tokens inside each SP shard (the last block of a shard may be 32 tokens);
K chunks are ``k_chunk`` tokens inside each shard; global K chunk index = shard * k_chunks_per_shard
+ local index. Every tensor dimension is padded to a tile multiple.

Geometry constants are built on host once per geometry (cached per mesh) together with the program descriptors;
per call only the buffer addresses are patched (generic_op re-applies runtime args on a program-cache hit).
The fp32 torch mirror of this arithmetic lives next to the unit test (test_sol_prepass_minimax_h3.py).
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import torch

import ttnn

from ....layers.module import Module

LOG2E = 1.4426950408889634
VAR_EPS = 1.0e-6
NEG = -30000.0  # finite "minus infinity" for bf16 bias/mask arithmetic (exp2 underflows to exactly 0)
BLOCK = 64
TILE = 32
TILE_BYTES = 2048
KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sol_kernels")
L1_CB_BUDGET = 1_200_000  # circular-buffer bytes per core the route program may use (Blackhole L1 = 1.5 MB)
SCRATCH_PAGES = 8


def _pad32(n: int) -> int:
    return -(-n // TILE) * TILE


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


@dataclass(frozen=True)
class SolGeometry:
    shards: int
    rows: int  # rows per shard (padded sequence / shards)
    logical_n: int
    q_chunk: int
    k_chunk: int
    block: int = BLOCK

    @property
    def nb(self) -> int:  # blocks per shard
        return -(-self.rows // self.block)

    @property
    def nb_pad(self) -> int:
        return _pad32(self.nb)

    @property
    def n_blocks_pad(self) -> int:  # gathered pooled length (keeps per-shard padding)
        return self.shards * self.nb_pad

    @property
    def nqc(self) -> int:
        return -(-self.rows // self.q_chunk)

    @property
    def nqc_pad(self) -> int:
        return _pad32(self.nqc)

    @property
    def nkc(self) -> int:  # K chunks per shard
        return -(-self.rows // self.k_chunk)

    @property
    def nkc_all(self) -> int:
        return self.shards * self.nkc

    @property
    def nkc_pad(self) -> int:
        return _pad32(self.nkc_all)

    def block_lengths(self) -> torch.Tensor:
        """[shards, nb_pad] valid tokens per (padded) block, clipped to logical_n."""
        out = torch.zeros(self.shards, self.nb_pad)
        for s in range(self.shards):
            for b in range(self.nb):
                start = s * self.rows + b * self.block
                end = min(start + self.block, s * self.rows + self.rows, self.logical_n)
                out[s, b] = max(0, end - start)
        return out


# ----------------------------------------------------------------------------- host constants


def build_constants(
    geo: SolGeometry,
    *,
    band: int,
    sink_tokens: int,
    force_first_chunk: bool,
    scale: float,
    dense_prefix_queries: bool = True,
) -> dict:
    """Host-built constant matrices (torch fp32) of the routing rule; all tile-padded. Shared by the torch
    reference and by the route program's constants (build_route_constants).

    The bias is in pre-scale logit units: the SDPA compute kernel evaluates exp((QK + bias - max) * scale),
    so a column bias of ln(len)/scale weights the pooled column by exactly `len`.
    """
    S, nb, nbp, nbg = geo.shards, geo.nb, geo.nb_pad, geo.n_blocks_pad
    lengths = geo.block_lengths()  # [S, nbp]
    # Pooling matrix transposed, per shard: [S, 1, rows, nb_pad], entry 1/len for a block's valid rows.
    pool_t = torch.zeros(S, 1, geo.rows, nbp)
    for s in range(S):
        for b in range(nb):
            length = int(lengths[s, b])
            if length:
                pool_t[s, 0, b * geo.block : b * geo.block + length, b] = 1.0 / length
    # Global block index of each padded block: g = s * nb_pad + b. Band on real indices (s*nb + b) so the
    # neighbourhood crosses shard boundaries like the reference's contiguous blocking.
    gidx = torch.full((nbg,), -1)
    for s in range(S):
        gidx[s * nbp : s * nbp + nb] = torch.arange(nb) + s * nb
    valid_row = (lengths.reshape(-1) > 0).float()  # [nbg]
    full_row = (lengths.reshape(-1) >= geo.block).float()  # exact-eligible: full blocks only
    # Band per shard: [S, 1, nb_pad(local q blocks), nbg]
    band_m = torch.zeros(S, 1, nbp, nbg)
    for s in range(S):
        qg = torch.full((nbp,), -(10**9))
        qg[:nb] = torch.arange(nb) + s * nb
        band_m[s, 0] = ((qg[:, None] - gidx[None, :]).abs() <= band).float() * valid_row[None, :]
        if dense_prefix_queries:
            # Prefix (text + condition) QUERY rows attend densely, as the reference implementation recomputes them
            # outside the sparse kernel: every Q block whose first token lies before sink_tokens keeps all valid K
            # blocks. Folded into the band constant so it costs no extra op. (The ragged last block stays pooled:
            # the kernel only gathers full blocks.)
            for b in range(nb):
                if s * geo.rows + b * geo.block < sink_tokens:
                    band_m[s, 0, b] = valid_row
    # Sink columns: blocks whose first token lies before sink_tokens.
    sink_row = torch.zeros(1, 1, 1, nbg)
    for s in range(S):
        for b in range(nb):
            if s * geo.rows + b * geo.block < sink_tokens and lengths[s, b] > 0:
                sink_row[0, 0, 0, s * nbp + b] = 1.0
    # Block -> global K chunk pooling: [1, 1, nbg, nkc_pad]
    kc_pool = torch.zeros(1, 1, nbg, geo.nkc_pad)
    for s in range(S):
        for b in range(nb):
            if lengths[s, b] > 0:
                kc_pool[0, 0, s * nbp + b, s * geo.nkc + (b * geo.block) // geo.k_chunk] = 1.0
    # Local Q block -> Q chunk pooling (transposed): [S, 1, nb_pad, nqc_pad]  (same for every shard)
    qc_pool_t = torch.zeros(1, 1, nbp, geo.nqc_pad)
    for b in range(nb):
        qc_pool_t[0, 0, b, (b * geo.block) // geo.q_chunk] = 1.0
    # Forced columns: block 0 of every shard (one gathered chunk per ring iteration)
    force_row = torch.zeros(1, 1, 1, nbg)
    if force_first_chunk:
        for s in range(S):
            force_row[0, 0, 0, s * nbp] = 1.0
    # Valid K chunks (hold at least one real token)
    valid_chunk_row = torch.zeros(1, 1, 1, geo.nkc_pad)
    for s in range(S):
        for c in range(geo.nkc):
            if s * geo.rows + c * geo.k_chunk < geo.logical_n:
                valid_chunk_row[0, 0, 0, s * geo.nkc + c] = 1.0
    len_bias_row = torch.full((1, 1, 1, nbg), NEG)
    pos = lengths.reshape(-1) > 0
    len_bias_row[0, 0, 0, pos] = torch.log(lengths.reshape(-1)[pos]) / scale
    return {
        "pool_t": pool_t,
        "band": band_m,
        "sink_row": sink_row,
        "valid_row": valid_row.view(1, 1, 1, nbg),
        "full_row": full_row.view(1, 1, 1, nbg),
        "kc_pool": kc_pool,
        "kc_pool_t": kc_pool.transpose(-2, -1).contiguous(),
        "qc_pool_t": qc_pool_t,
        "force_row": force_row,
        "valid_chunk_row": valid_chunk_row,
        "len_bias_row": len_bias_row,
        "n_valid_blocks": float(pos.sum()),
    }


def build_route_constants(
    geo: SolGeometry, *, band: int, sink_tokens: int, force_first_chunk: bool, scale: float, dense_prefix_queries: bool
) -> dict:
    """Host constants of the route program (torch fp32), derived from build_constants.

    const_or [S,1,nqc_pad,n_blocks_pad]: the threshold-independent part of block01 (band | sink | dense-prefix rows,
      OR-ed over the Q blocks of each chunk, | forced block 0 per shard) & full blocks -- exactly what
      the routing rule produces when no score passes the threshold.
    colconst [1,1,S*32,nb_pad]: per shard s, 32 identical rows of its full-block column constant (a tile operand
      for the compute kernel).
    qsel [1,1,nqc_pad,nb_pad]: 1 where the Q block belongs to the Q chunk. ksel [1,1,nb_pad,nkcl]: 1 where the
      local block belongs to the local K chunk (geometry only). ones [1,1,32,32]: row 0 = 1.
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
    colconst = torch.zeros(1, 1, S * TILE, nbp)
    for s in range(S):
        colconst[0, 0, s * TILE : (s + 1) * TILE, :] = c["full_row"][0, 0, 0, s * nbp : (s + 1) * nbp]
    qsel = c["qc_pool_t"][0, 0].T.contiguous().view(1, 1, nqcp, nbp)
    nkcl = _pad32(geo.nkc)
    ksel = torch.zeros(1, 1, nbp, nkcl)
    for b in range(geo.nb):
        ksel[0, 0, b, (b * geo.block) // geo.k_chunk] = 1.0
    ones = torch.zeros(1, 1, TILE, TILE)
    ones[0, 0, 0, :] = 1.0
    # Shard-row selectors for the route moment exchange ((head, shard) units): gather tile t holds shard ps = 4t + q at
    # rows q*8 + n (n = moment tile 0..7: DT sums, DT sums of squares); selector tile n has ones at row 0, columns
    # q*8 + n, so selector[n] @ gather[t] puts the sum over that tile's shards of moment n in row 0, accumulated over t.
    assert S in (4, 8), S  # other counts leave never-written rows in the gather tiles (0 * stale NaN = NaN)
    shsel = torch.zeros(1, 1, 8 * TILE, TILE)
    for n in range(8):
        for q in range(min(4, S)):
            shsel[0, 0, n * TILE, q * 8 + n] = 1.0
    return {
        "const_or": const_or,
        "colconst": colconst,
        "qsel": qsel,
        "ksel": ksel,
        "ones": ones,
        "shsel": shsel,
        "n_valid_blocks": c["n_valid_blocks"],
        "nkcl": nkcl,
        "kc_pool": c["kc_pool"],
        "kc_pool_t": c["kc_pool_t"],
        "valid_row": c["valid_row"],
        "len_bias_row": c["len_bias_row"],
    }


# ----------------------------------------------------------------------------- device pre-pass


class SolPrepass(Module):
    """Device-side routing pre-pass: pool program, two pooled all-gathers, route program.

    No weights. The constants of a geometry are built on host and uploaded by `prepare()`; the attention module
    creates one instance per layer and per request (the logical sequence length is set per request), so that is
    one upload per layer and request.
    """

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
        mode: str = "gather",
        mask_reuse: int = 1,
    ):
        # "gather": block mask [1,H,nqc_pad,n_blocks_pad], the reader gathers kept 64-token blocks (chains off).
        # "chunk": K-chunk mask [1,H,nqc_pad,nkc_pad] (any kept block in the chunk keeps the chunk); the reader
        #          streams densely with chains and compute skips dropped chunks.
        # "compact": the gather-form block mask (full blocks only; block 0 of every shard forced kept so every
        #          (Q chunk, shard) computes at least one block), streamed through the chains; the reader compacts
        #          each core's kept blocks in L1. "pack" is compact plus the op's cross-chunk packing: same output.
        assert mode in ("gather", "chunk", "compact", "pack"), mode
        assert thresh_type == "diag", "the pre-pass implements the diag threshold (H3 policy)"
        super().__init__()
        self.mode = mode
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.sp_axis = sp_axis
        self.tau = float(tau)
        self.band = band
        # block 0 of every shard is forced kept: a Q chunk computing zero chunks in a ring iteration deadlocks the
        # ring op's accumulator staging protocol.
        self.force_first_chunk = force_first_chunk
        self.dense_prefix_queries = dense_prefix_queries
        # Mask reuse: N > 1 keeps the route output (block mask) of a call and reuses them for the next N-1
        # calls, skipping the route program; the pool program and the two gathers still run every call because the
        # ring op's correction term needs the current block means. 1 (default) = route every call.
        self.mask_reuse = int(mask_reuse)
        # The two pooled gathers run on their own global semaphores (two sets, alternated per gather) instead of the
        # CCL manager's all-gather ping-pong pool, which the ring attention op's fused K/V gather shares and whose
        # reader zeroes its semaphore at exit (ring_attention_all_gather_reader.cpp).
        self._ag_sems: list | None = None
        self._ag_idx = 0
        self.route_units: tuple[int, int] | None = None  # (unit count, units_per_shard) of the built route program
        self._held_mask: ttnn.Tensor | None = None  # the last routed mask, while mask_reuse > 1
        self._reuse_left = 0  # calls the held route may still serve
        self.last_mask_reused = False  # whether the last call reused the held mask (diagnostics)
        self._state: dict | None = None
        self._key = None
        # Density of the mask the op skips at (report_density=True; host readback, off in production). In chunk mode
        # that is the (q_chunk x k_chunk) density, so the block-level density is only known in the block-form modes.
        self.last_density: float | None = None
        self.last_block_density: float | None = None

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
        """Build and upload the constants of a geometry; a no-op while geometry, sink and scale are unchanged."""
        key = (geo, sink_tokens, scale)
        if key == self._key:
            return
        self._release_state()
        c = build_route_constants(
            geo,
            band=self.band,
            sink_tokens=sink_tokens,
            force_first_chunk=self.force_first_chunk,
            scale=scale,
            dense_prefix_queries=self.dense_prefix_queries,
        )
        self._state = {
            "scale": scale,
            "n_valid_blocks": c["n_valid_blocks"],
            "nkcl": c["nkcl"],
            "const_or": self._upload(c["const_or"], shard=True),
            "colconst": self._upload(c["colconst"], shard=False),
            "qsel": self._upload(c["qsel"], shard=False),
            "ksel": self._upload(c["ksel"], shard=False),
            "ones": self._upload(c["ones"], shard=False),
            "shsel": self._upload(c["shsel"], shard=False),
            "geo": self._upload(
                torch.tensor(
                    [[[[max(0, min(geo.rows, geo.logical_n - s * geo.rows))] + [0] * 31]] for s in range(geo.shards)],
                    dtype=torch.int32,
                ),
                shard=True,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            "pool_prog": None,
            "route_prog": None,
            "pool_outs": None,
        }
        self.geo = geo
        self._key = key
        self.reset_mask_reuse()  # a held mask belongs to the previous geometry

    def _release_state(self) -> None:
        """Free the uploaded constants and the persistent pool outputs of the current geometry."""
        st = self._state
        if st is None:
            return
        for name in ("const_or", "colconst", "qsel", "ksel", "ones", "shsel", "geo"):
            ttnn.deallocate(st[name])
        for t in st["pool_outs"] or []:
            ttnn.deallocate(t)
        self._state = None
        self._key = None

    def reset_mask_reuse(self) -> None:
        """Drop the held mask: the next call routes afresh (call at the start of every generation)."""
        self._held_mask = None
        self._reuse_left = 0

    def _cores(self):
        g = self.mesh_device.compute_with_storage_grid_size()
        nx, ny = g.x, g.y
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))])
        return nx, ny, ranges

    @staticmethod
    def _cb(idx: int, pages: int, cores, page_size: int = TILE_BYTES, dtype=ttnn.bfloat16) -> ttnn.CBDescriptor:
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page_size)
        return ttnn.CBDescriptor(total_size=pages * page_size, core_ranges=cores, format_descriptors=[fmt])

    @staticmethod
    def _kernel(path: str, cores, ct_args: list[int], common: list[int], config) -> ttnn.KernelDescriptor:
        return ttnn.KernelDescriptor(
            kernel_source=os.path.join(KERNEL_DIR, path),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=list(ct_args),
            runtime_args=[],
            common_runtime_args=list(common),
            config=config,
        )

    @staticmethod
    def _compute_config(full_sync: bool):
        # fp32 DST holds 4 tiles per section in half-sync mode and 8 in full-sync mode
        # (tests/ttnn/unit_tests/kernel_lib/test_chain_oob.py). The route kernel addresses DST tiles 0..7, so it must
        # run full-sync; the pool kernel uses DT (<= 4) tiles and keeps the math/pack overlap of half-sync.
        return ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=full_sync,
        )

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
        # Reader pipeline (sol_pool_reader.cpp): batches of 4 tile rows, three batches resident in cb 0 (data) and cb 2
        # (pooling tiles) so two can be in flight on the NoC while compute consumes the third; cb 5 holds the 32
        # standard pooling tiles built once per core.
        cbs = [
            self._cb(0, 3 * 4 * DT, cores),
            self._cb(2, 3 * 4, cores),
            self._cb(3, 1, cores),
            self._cb(4, 1, cores, page_size=128, dtype=ttnn.uint32),
            self._cb(5, 32, cores),
            self._cb(16, 2 * DT, cores),
        ]
        reader = self._kernel("sol_pool_reader.cpp", cores, ct + acc_in, [0, 0, 0, 0], ttnn.ReaderConfigDescriptor())
        writer = self._kernel("sol_pool_writer.cpp", cores, ct + acc_out, [0, 0, 0], ttnn.WriterConfigDescriptor())
        compute = self._kernel("sol_pool_compute.cpp", cores, ct, [], self._compute_config(full_sync=False))
        return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    def _run_pool(self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, persistent: bool = True) -> list[ttnn.Tensor]:
        """Block means of the local shards: [q_bar, kc_local, vc_local], each [1, H, nb_pad, D].

        persistent (the forward path): the three outputs are allocated once per geometry and reused; on the in-order
        command queue the next call's pool program runs only after the previous call's gathers and route have
        consumed them. Saves three mesh allocations and three deallocations of host time per call.
        """
        st = self._state
        _, H, _, D = q.shape
        if persistent and st["pool_outs"] is not None:
            outs = st["pool_outs"]
        else:
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
            if persistent:
                st["pool_outs"] = outs
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
        ttnn.generic_op([q, k, v, st["geo"], *outs], prog)
        return outs

    # ------------------------------------------------------------------ route program

    def _route_program(self, q_bar, kc, mask) -> ttnn.ProgramDescriptor:
        st, geo = self._state, self.geo
        _, H, _, D = q_bar.shape
        S, DT, NBPT, NQCT = geo.shards, D // TILE, geo.nb_pad // TILE, geo.nqc_pad // TILE
        NKCLT = st["nkcl"] // TILE
        NBGT = S * NBPT
        assert NBPT * DT <= 8 * 32 and DT <= 4 and NKCLT <= 7, (NBPT, DT, NKCLT)
        mode = 0 if self.mode == "chunk" else 1  # kernel MODE: 0 = K-chunk mask, 1 = block mask
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
            _f32_bits(self.tau),
            _f32_bits(VAR_EPS / (log2_scale * log2_scale)),
            _f32_bits(NEG),
            _f32_bits(1.0 / st["n_valid_blocks"]),
            TQ,
        ]
        ins = [q_bar, kc, st["const_or"], st["colconst"], st["qsel"], st["ksel"], st["ones"]]
        acc_in = [a for t in ins for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        # The writer's second accessor (the chunk-union output of an earlier variant) is kept pointed at the mask.
        acc_out = [a for t in (mask, mask) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        nx, ny, cores = self._cores()
        # Unit = (head, shard): H*S units, which keeps the core grid filled when NQCT is small. The chunk-form mask
        # (MODE 0) assembles whole rows from every shard in the writer's L1 row buffer (their per-shard word ranges
        # are not NoC-aligned), so it keeps the (head, Q-chunk tile row) unit with H*NQCT units; the kernels
        # static_assert the same rule.
        units_per_shard = int(mode == 1 and H * S <= nx * ny) # True if the block mask units fit the core grid else false for Wormhole
        units = H * S if units_per_shard else H * NQCT
        self.route_units = (units, units_per_shard)
        ct += [nx, ny, units, units_per_shard]  # grid + unit count + unit kind; kernels derive their range per core
        mask_t = NKCLT if mode == 0 else NBPT  # per shard
        # (head, shard) units read only their shard's pooled K and exchange partial moments between the S cores of
        # a head (sol_route_reader.cpp): the reader needs every logical core's NoC coordinates (common runtime args
        # 8..) and a semaphore; cb 23 receives the peers' partial moments (gather tiles), 24 ships this unit's,
        # 26 holds the reader's copy of the mask tiles and 29 the shard-row selectors. The head's pooled Q is read
        # from DRAM by its shard-0 core and shipped to the other S-1 cores over the NoC (reader semaphore 1).
        core_xy = self._core_noc_xy(nx, ny) if units_per_shard else []
        st["route_core_xy"] = core_xy
        sizes = {
            0: (NBPT if units_per_shard else NBGT) * DT,
            1: NBPT * DT,
            2: 1,
            3: NQCT * NBPT,
            4: NBPT * NKCLT,
            6: NBPT,
            7: NBPT,
            8: 8,
            9: 8,
            10: TQ * DT,
            11: TQ,
            12: TQ,
            13: TQ * NBPT,
            14: NBPT,
            16: mask_t,
            20: DT,
            21: NKCLT,
        }
        mask_rows_bytes = (
            32 * geo.nb_pad * 4
        )  # (head, shard) units: one page = the 32 row-major uint32 mask rows of (it, sp)
        if units_per_shard:  # 23: gather tiles, 24: this shard's reduced tiles, 29: shsel; 16 becomes the row page
            sizes.update({23: (S + 3) // 4, 24: 8, 29: 8})
            del sizes[16]
        cbs = [self._cb(i, n, cores) for i, n in sizes.items()]
        if units_per_shard:
            cbs.append(self._cb(16, 2, cores, page_size=mask_rows_bytes, dtype=ttnn.uint32))
        row_bytes = geo.nkc_pad * 4  # chunk-form row buffer: 32 complete rows
        cbs.append(self._cb(19, 32, cores, page_size=row_bytes, dtype=ttnn.uint32))
        cbs.append(self._cb(22, SCRATCH_PAGES, cores, page_size=geo.nb_pad * 4, dtype=ttnn.uint32))  # gather-form ring
        total = sum(n for n in sizes.values()) * TILE_BYTES + 32 * row_bytes + SCRATCH_PAGES * geo.nb_pad * 4
        total += 2 * mask_rows_bytes if units_per_shard else 0
        if total > L1_CB_BUDGET:
            raise ValueError(f"sol_route circular buffers need {total} B of L1 (> {L1_CB_BUDGET}); geometry too large")
        acc_extra = list(ttnn.TensorAccessorArgs(st["shsel"]).get_compile_time_args())
        reader = self._kernel(
            "sol_route_reader.cpp",
            cores,
            ct + acc_in + acc_extra + [units_per_shard],  # last arg: pooled Q shared over the head's cores
            [0] * 8 + core_xy,
            ttnn.ReaderConfigDescriptor(),
        )
        writer = self._kernel("sol_route_writer.cpp", cores, ct + acc_out, [0, 0], ttnn.WriterConfigDescriptor())
        compute = self._kernel("sol_route_compute.cpp", cores, ct, [], self._compute_config(full_sync=True))
        sems = (
            [ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in (0, 1)]
            if units_per_shard
            else []
        )
        return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=sems, cbs=cbs)

    def _core_noc_xy(self, nx: int, ny: int) -> list[int]:
        """NoC coordinates (x | y << 8) of every logical worker core, index y*nx + x. One program with common runtime
        args runs on every chip of the mesh; the virtual worker coordinates are the same on every chip."""
        xy = [
            self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y)) for y in range(ny) for x in range(nx)
        ]
        return [w.x | (w.y << 8) for w in xy]

    def _run_route(self, q_bar: ttnn.Tensor, kc: ttnn.Tensor) -> ttnn.Tensor:
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
        if st["route_prog"] is None:
            st["route_prog"] = self._route_program(q_bar, kc, mask)
        prog = st["route_prog"]
        ins = [q_bar, kc, st["const_or"], st["colconst"], st["qsel"], st["ksel"], st["ones"]]
        prog.kernels[0].common_runtime_args = (
            [t.buffer_address() for t in ins] + [st["shsel"].buffer_address()] + st["route_core_xy"]
        )
        prog.kernels[1].common_runtime_args = [mask.buffer_address(), mask.buffer_address()]
        ttnn.generic_op([*ins, st["shsel"], mask, mask], prog)
        return mask

    def _gather(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """All-gather a pooled [1, H, nb_pad, D] tensor along dim 2 over the SP ring.

        One worker per link direction (more workers go through a mux core that serialises them) and no ring-wide
        barrier semaphore before the gather (measured 2026-10-28: 130 -> 72 us per gather at 5 s).
        """
        ccl = self.ccl_manager
        if self._ag_sems is None:
            self._ag_sems = [
                [ttnn.create_global_semaphore(self.mesh_device, ccl.ccl_cores, 0) for _ in range(2)] for _ in range(2)
            ]
        sems = self._ag_sems[self._ag_idx]
        self._ag_idx ^= 1
        return ttnn.experimental.all_gather_async(
            t,
            persistent_output_buffer=None,
            dim=2,
            multi_device_global_semaphore=sems,
            num_links=ccl.num_links,
            topology=ccl.topology,
            cluster_axis=self.sp_axis,
            barrier_semaphore=None,
            num_workers_per_link=1,
        )

    # ------------------------------------------------------------------ forward

    def forward(
        self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, *, scale: float, report_density: bool = False
    ) -> dict:
        assert abs(scale - self._state["scale"]) < 1e-12, "scale is baked into the prepared constants"
        geo = self.geo
        q_bar, kc_local, vc_local = self._run_pool(q, k, v)
        kc = self._gather(kc_local)
        vc = self._gather(vc_local)
        # deallocate() defaults to force=True, so a caller freeing the returned mask also releases
        # _held_mask. Reuse only while that allocation is still live; otherwise route again.
        reuse = (
            self.mask_reuse > 1
            and self._held_mask is not None
            and self._held_mask.is_allocated()
            and self._reuse_left > 0
        )
        if reuse:
            mask = self._held_mask
            self._reuse_left -= 1
        else:
            mask = self._run_route(q_bar, kc)
            if self.mask_reuse > 1:
                self._held_mask = mask
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
        return {"block_mask": mask, "pooled_k": kc, "pooled_v": vc}
