# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Sol-Attn routing pre-pass for the ring-joint SDPA path: two custom device programs (``ttnn.generic_op``) around
the two pooled all-gathers.

* ``sol_block_pool`` (kernels ``sol_kernels/sol_pool_*.cpp``): per-64-token-block means of the local Q/K/V shards,
  one program for the three tensors, every core busy.
* two all-gathers of the pooled K and V along the SP ring (``ttnn.experimental.all_gather_async``).
* ``sol_route`` (``sol_kernels/sol_route_*.cpp``): moments, thresholds, proxy scores and the uint32 block mask in
  one program.

Per device it produces, from the local Q/K/V shards [1, H_local, R, D]:

* ``block_mask``  uint32 row-major [1, H, nb_pad, n_blocks_pad]: one row per local 64-token Q block, one word per
  global padded 64-token K block. 1 = this (Q block, K block) pair is attended exactly, 0 = the K block goes through
  the pooled correction. This is the mask of the ring op's resident mode, whose query unit is the 64-row block.
  Only full 64-token K blocks inside logical_n are ever exact; block 0 of every shard is forced exact (every Q block
  computes at least one block per ring iteration). Rows of padding Q blocks are zero.
* ``pooled_k`` / ``pooled_v``  tile [1, H, n_blocks_pad, D] in K's dtype (bf16 or bf8; ``pooled_dtype`` overrides):
  per-64-token block mean K / mean V for the WHOLE sequence (all-gathered along SP), per-shard blocking.

The outputs are the inputs of ``ttnn.transformer.sparse_ring_sdpa`` (block_mask, pooled_k, pooled_v). Q, K and V may
be bf16 or bf8; K and V share a dtype, Q may differ (e.g. bf16 Q with bf8 K/V).
The pooled-column bias (ln(block token count) / scale for approximated blocks, a large negative value for exact and
padding blocks) is not produced here: the ring op's reader generates it on device from the mask row and the block
geometry.

Geometry: blocks are 64 tokens inside each SP shard (the last block of a shard may be 32 tokens); global padded block
index = shard * nb_pad + local index. Every tensor dimension is padded to a tile multiple.

Geometry constants are built on host once per geometry (cached per mesh) together with the program descriptors;
per call only the buffer addresses are patched (generic_op re-applies runtime args on a program-cache hit).
The packed length is data only (const_or, valid_blocks, geo), never a compile-time arg, and the layers share one set
of route constants per padded geometry (SolRouteConstants): a new length inside it moves that one set in place, and
a captured trace stays valid.
The fp32 torch mirror of this arithmetic lives next to the unit test (test_sol_prepass_minimax_h3.py).
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import torch

import ttnn

from ....layers.module import Module
from ....utils.tracing import live_tracers

LOG2E = 1.4426950408889634
VAR_EPS = 1.0e-6
NEG = -30000.0  # finite "minus infinity" for bf16 bias/mask arithmetic (exp2 underflows to exactly 0)
BLOCK = 64
TILE = 32
TILE_BYTES = 2048
# Tile bytes per dtype the pool program reads its inputs in (bf8: 1024 mantissa bytes + 64 shared exponents).
DATA_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088}
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
    force_first_block: bool,
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
    # Forced columns: block 0 of every shard (every Q block computes at least one block per ring iteration)
    force_row = torch.zeros(1, 1, 1, nbg)
    if force_first_block:
        for s in range(S):
            force_row[0, 0, 0, s * nbp] = 1.0
    # Mask rows that belong to a real local Q block (the same on every shard): [1, 1, nb_pad, 1]
    q_row = torch.zeros(1, 1, nbp, 1)
    q_row[0, 0, :nb] = 1.0
    len_bias_row = torch.full((1, 1, 1, nbg), NEG)
    pos = lengths.reshape(-1) > 0
    len_bias_row[0, 0, 0, pos] = torch.log(lengths.reshape(-1)[pos]) / scale
    return {
        "pool_t": pool_t,
        "band": band_m,
        "sink_row": sink_row,
        "valid_row": valid_row.view(1, 1, 1, nbg),
        "full_row": full_row.view(1, 1, 1, nbg),
        "force_row": force_row,
        "q_row": q_row,
        "len_bias_row": len_bias_row,
        "n_valid_blocks": float(pos.sum()),
    }


def build_route_constants(
    geo: SolGeometry, *, band: int, sink_tokens: int, force_first_block: bool, scale: float, dense_prefix_queries: bool
) -> dict:
    """Host constants of the route program (torch fp32), derived from build_constants.

    const_or [S,1,nb_pad,n_blocks_pad]: the threshold-independent part of the block mask (band | sink | dense-prefix
      rows | forced block 0 per shard) & full blocks -- exactly what the routing rule produces when no score passes
      the threshold.
    valid_blocks [1,1,nb_pad,n_blocks_pad]: 1 where the row is a real Q block and the column a full K block. The
      route kernel multiplies the mask by it, so partial and padding K blocks and padding Q rows end up 0.
    ones [1,1,32,32]: row 0 = 1.
    """
    c = build_constants(
        geo,
        band=band,
        sink_tokens=sink_tokens,
        force_first_block=force_first_block,
        scale=scale,
        dense_prefix_queries=dense_prefix_queries,
    )
    const_or = ((c["band"] + c["sink_row"] + c["force_row"]) > 0.5).float() * c["full_row"]  # [S,1,nbp,nbg]
    valid_blocks = c["q_row"] * c["full_row"]  # [1,1,nbp,nbg]
    ones, shsel = _route_selectors(geo.shards)
    return {
        "const_or": const_or,
        "valid_blocks": valid_blocks,
        "ones": ones,
        "shsel": shsel,
        "n_valid_blocks": c["n_valid_blocks"],
        "valid_row": c["valid_row"],
        "len_bias_row": c["len_bias_row"],
    }


def _route_selectors(S: int) -> tuple[torch.Tensor, torch.Tensor]:
    """ones [1,1,32,32] (row 0 = 1) and the shard-row selectors shsel [1,1,256,32] of the route program."""
    ones = torch.zeros(1, 1, TILE, TILE)
    ones[0, 0, 0, :] = 1.0
    # Shard-row selectors for the route moment exchange ((head, shard) units): gather tile t holds shard 4t + q at
    # rows q*8 + n (n = moment tile 0..7: head_dim_tiles sums, head_dim_tiles sums of squares); selector tile n has
    # ones at row 0, columns q*8 + n, so selector[n] @ gather[t] puts the sum over that tile's shards of moment n in
    # row 0, accumulated over t. Each gather tile holds 4 shards, so S must fill those tiles (a partial tile's unread
    # rows are stale: 0 * NaN = NaN). The exchange itself runs only for 4 or 8 shards. 32 shards (a 4x32 mesh) are a
    # full 8 tiles, but that mesh uses the all-shard layout instead; this selector is uploaded and not read there.
    assert S in (4, 8, 32), S
    shsel = torch.zeros(1, 1, 8 * TILE, TILE)
    for n in range(8):
        for q in range(min(4, S)):
            shsel[0, 0, n * TILE, q * 8 + n] = 1.0
    return ones, shsel


def route_factors(
    geo: SolGeometry, *, band: int, sink_tokens: int, force_first_block: bool, dense_prefix_queries: bool
) -> dict:
    """The route constants as factors, so a new packed length or sink moves three vectors, not two matrices.

    The route kernel computes mask = (hits + const_or > 0.5) * valid_blocks, so const_or = base + sink_cols +
    prefix_rows and valid_blocks = q_full * full_cols give the mask of build_route_constants. base [S,1,nb_pad,nbg]
    (band | forced block 0 of every shard) and q_full [1,1,nb_pad,nbg] (real Q rows) depend on the padded geometry
    only; full_cols and sink_cols [1,1,1,nbg], prefix_rows [S,1,nb_pad,1] and n_valid_blocks on the request.
    """
    S, nb, nbp, nbg = geo.shards, geo.nb, geo.nb_pad, geo.n_blocks_pad
    lengths = geo.block_lengths()  # [S, nbp]
    shard, local = torch.arange(S)[:, None], torch.arange(nbp)[None, :]
    real = (local < nb).expand(S, nbp)
    index = shard * nb + local  # real block index: the band crosses shard boundaries
    base = ((index[:, :, None] - index.reshape(-1)).abs() <= band) & real[:, :, None] & real.reshape(-1)
    if force_first_block:
        base[:, :, ::nbp] = True
    in_prefix = real & (shard * geo.rows + local * geo.block < sink_tokens)
    return {
        "base": base.float()[:, None],
        "q_full": real[:1].float().reshape(1, 1, nbp, 1).expand(1, 1, nbp, nbg).contiguous(),
        "full_cols": (lengths >= geo.block).float().reshape(1, 1, 1, nbg),
        "sink_cols": in_prefix.float().reshape(1, 1, 1, nbg),
        "prefix_rows": (in_prefix & dense_prefix_queries).float().reshape(S, 1, nbp, 1),
        "n_valid_blocks": float((lengths > 0).sum()),
    }


# ----------------------------------------------------------------------------- device constants


class SolRouteConstants:
    """Device constants of the pool and route programs for one padded geometry (shards, rows per shard, block).

    They are the same in every layer, so the transformer keeps one set per padded length and every layer's pre-pass
    reads it. `set()` moves the set to a packed length and sink in place: it rewrites the geometry page and three
    vectors and composes const_or and valid_blocks on device (route_factors), so buffers, programs and captured
    traces stay.
    """

    def __init__(
        self,
        mesh_device,
        sp_axis: int,
        geo: SolGeometry,
        *,
        band: int = 1,
        force_first_block: bool = True,
        dense_prefix_queries: bool = True,
    ):
        self.mesh_device = mesh_device
        self.sp_axis = sp_axis
        self.padded = (geo.shards, geo.rows, geo.block)
        self.rule = (band, force_first_block, dense_prefix_queries)
        self.tensors: dict[str, ttnn.Tensor] = {}
        self._request = None
        self._vectors = None  # host copies of the uploaded full_cols, sink_cols, prefix_rows
        # A trace replay may overwrite buffers allocated after its capture: such a set is rewritten at every set().
        self._trusted = not any(tracer.trace_captured for tracer in live_tracers())

    def _upload(self, name: str, t: torch.Tensor, *, shard: bool, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        """Upload a host constant; a tensor of that name is rewritten in place (same spec)."""
        if shard:
            mapper = ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=[None, 0] if self.sp_axis == 1 else [0, None],
                mesh_shape=tuple(self.mesh_device.shape),
            )
        else:
            mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        if name in self.tensors:
            host = ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=mapper)
            ttnn.copy_host_to_device_tensor(host, self.tensors[name])
        else:
            self.tensors[name] = ttnn.from_torch(
                t, dtype=dtype, layout=layout, device=self.mesh_device, mesh_mapper=mapper
            )

    def set(self, geo: SolGeometry, sink_tokens: int) -> None:
        """Move the constants to this packed length and sink; a no-op while both are unchanged."""
        assert (geo.shards, geo.rows, geo.block) == self.padded, (geo, self.padded)
        request = (geo.logical_n, sink_tokens)
        if request == self._request and self._trusted:
            return
        band, force_first_block, dense_prefix_queries = self.rule
        f = route_factors(
            geo,
            band=band,
            sink_tokens=sink_tokens,
            force_first_block=force_first_block,
            dense_prefix_queries=dense_prefix_queries,
        )
        t = self.tensors
        rewrite = not t or not self._trusted
        if rewrite:
            ones, shsel = _route_selectors(geo.shards)
            self._upload("base", f["base"], shard=True)
            self._upload("q_full", f["q_full"], shard=False)
            self._upload("ones", ones, shard=False)
            self._upload("shsel", shsel, shard=False)
        if not t.keys() >= {"const_or", "valid_blocks"}:
            self._upload("const_or", torch.zeros_like(f["base"]), shard=True)
            self._upload("valid_blocks", torch.zeros_like(f["q_full"]), shard=False)
        # geo page per device: word 0 the shard's valid tokens (pool reader), word 1 the fp32 bits of
        # 1 / number of non-padding K blocks (route compute).
        inv_valid_blocks = _f32_bits(1.0 / f["n_valid_blocks"])
        geo_words = torch.tensor(
            [
                [[[max(0, min(geo.rows, geo.logical_n - s * geo.rows)), inv_valid_blocks] + [0] * 30]]
                for s in range(geo.shards)
            ],
            dtype=torch.int32,
        )
        self._upload("geo", geo_words, shard=True, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        vectors = (f["full_cols"], f["sink_cols"], f["prefix_rows"])
        if rewrite or any(not torch.equal(a, b) for a, b in zip(vectors, self._vectors)):
            self._upload("full_cols", f["full_cols"], shard=False)
            self._upload("sink_cols", f["sink_cols"], shard=False)
            self._upload("prefix_rows", f["prefix_rows"], shard=True)
            ttnn.add(t["base"], t["sink_cols"], output_tensor=t["const_or"])
            ttnn.add(t["const_or"], t["prefix_rows"], output_tensor=t["const_or"])
            ttnn.multiply(t["q_full"], t["full_cols"], output_tensor=t["valid_blocks"])
            self._vectors = vectors
        self._request = request

    def release(self) -> None:
        for t in self.tensors.values():
            ttnn.deallocate(t)
        self.tensors = {}
        self._request = None


# ----------------------------------------------------------------------------- device pre-pass


class SolPrepass(Module):
    """Device-side routing pre-pass: pool program, two pooled all-gathers, route program.

    No weights. `prepare()` binds the route constants of a geometry (SolRouteConstants); the attention module keeps
    one instance per layer and per padded length (rows per device). A new packed length or sink inside that padded
    length keeps buffers and programs.
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
        force_first_block: bool = True,
        dense_prefix_queries: bool = True,
        mask_reuse: int = 1,
        route_unit: str = "auto",
        pooled_dtype=None,
    ):
        # pooled_dtype: dtype of the returned pooled_k / pooled_v (the sparse ring op wants K's dtype). None: the dtype
        # of the k passed to forward. The route program always scores in bf16; bf8 copies are cast for the caller.
        # route_unit: how the route program splits its work. "head_shard": one core per (head, K shard), the cores of a
        # head exchange partial K moments; "head_q_tile": one core per (head, 32 Q blocks), reading every shard's K;
        # "auto" picks head_shard when it fits (at most 8 shards and heads * shards <= cores). Tests force each kind.
        assert route_unit in ("auto", "head_shard", "head_q_tile"), route_unit
        assert thresh_type == "diag", "the pre-pass implements the diag threshold (H3 policy)"
        super().__init__()
        self.route_unit = route_unit
        self.pooled_dtype = pooled_dtype
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.sp_axis = sp_axis
        self.tau = float(tau)
        self.band = band
        # block 0 of every shard is forced kept: a query computing zero blocks in a ring iteration deadlocks the
        # ring op's accumulator staging protocol.
        self.force_first_block = force_first_block
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
        # Kept fraction of the real (Q block, K block) cells (report_density=True; host readback, off in production).
        self.last_density: float | None = None

    # ------------------------------------------------------------------ constants / descriptors

    def prepare(
        self, geo: SolGeometry, sink_tokens: int, scale: float, constants: SolRouteConstants | None = None
    ) -> None:
        """Bind the route constants of a geometry.

        `constants`: the set the layers share, which its owner has set to this request; None keeps a private set and
        sets it here. Programs and pool outputs are kept while the constants and the scale stay the same, so a new
        packed length or sink inside a padded geometry changes no address a captured trace holds.
        """
        st = self._state
        own = constants is None
        if own:
            keep = st is not None and st["own"] and st["scale"] == scale
            keep = keep and st["constants"].padded == (geo.shards, geo.rows, geo.block)
            constants = (
                st["constants"]
                if keep
                else SolRouteConstants(
                    self.mesh_device,
                    self.sp_axis,
                    geo,
                    band=self.band,
                    force_first_block=self.force_first_block,
                    dense_prefix_queries=self.dense_prefix_queries,
                )
            )
        assert constants.rule == (self.band, self.force_first_block, self.dense_prefix_queries), constants.rule
        if st is None or st["constants"] is not constants or st["scale"] != scale:
            self._release_state()
            self._state = {"constants": constants, "own": own, "scale": scale}
            self._state.update(pool_prog=None, route_prog=None, pool_outs=None)
        if own:
            constants.set(geo, sink_tokens)
        self._state.update(constants.tensors)
        self.geo = geo
        self.reset_mask_reuse()  # a held mask belongs to the previous request

    def _release_state(self) -> None:
        """Free the persistent pool outputs of the current geometry, and its constants when they are private."""
        st = self._state
        if st is None:
            return
        if st["own"]:
            st["constants"].release()
        for t in st["pool_outs"] or []:
            ttnn.deallocate(t)
        self._state = None

    def reset_mask_reuse(self) -> None:
        """Drop the held mask: the next call routes afresh (call at the start of every generation)."""
        self._held_mask = None
        self._reuse_left = 0

    def _cores(self):
        grid = self.mesh_device.compute_with_storage_grid_size()
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        return grid.x, grid.y, ranges

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
        # run full-sync; the pool kernel uses head_dim_tiles (<= 4) tiles and keeps the math/pack overlap of half-sync.
        return ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=full_sync,
        )

    # ------------------------------------------------------------------ pool program

    def _pool_program(
        self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, geo_t: ttnn.Tensor, outs: list[ttnn.Tensor]
    ) -> ttnn.ProgramDescriptor:
        geo = self.geo
        _, num_heads, shard_rows, head_dim = q.shape
        assert shard_rows == geo.rows and shard_rows % TILE == 0 and head_dim % TILE == 0
        for x in (q, k, v):
            assert x.dtype in DATA_TILE_BYTES, f"the pool program reads bf16 or bf8 inputs, got {x.dtype}"
        assert k.dtype == v.dtype, ("K and V must share a dtype", k.dtype, v.dtype)
        q_own_cb = int(q.dtype != k.dtype)  # Q's data goes through its own CB (cb 1) in Q's format
        shard_row_tiles, head_dim_tiles = shard_rows // TILE, head_dim // TILE
        shard_block_tiles = geo.nb_pad // TILE
        grid_x, grid_y, cores = self._cores()
        num_units = 3 * num_heads * shard_block_tiles  # (tensor, head, group) units
        # Compile-time args, named as in sol_pool_units.hpp; kernels derive their unit range per core from the grid.
        ct = [num_heads, shard_row_tiles, head_dim_tiles, shard_block_tiles, shard_rows, grid_x, grid_y, num_units]
        ct.append(q_own_cb)  # 8 Q_OWN_CB
        acc_in = [a for x in (q, k, v, geo_t) for a in ttnn.TensorAccessorArgs(x).get_compile_time_args()]
        acc_out = [a for o in outs for a in ttnn.TensorAccessorArgs(o).get_compile_time_args()]
        # Reader pipeline (sol_pool_reader.cpp): batches of 4 tile rows, three batches resident in cb 0 (data; cb 1 for Q
        # when its dtype differs) and cb 2 (pooling tiles) so two can be in flight on the NoC while compute consumes the
        # third; cb 5 holds the 32 standard pooling tiles built once per core.
        data_pages = 3 * 4 * head_dim_tiles
        cbs = [
            self._cb(0, data_pages, cores, page_size=DATA_TILE_BYTES[k.dtype], dtype=k.dtype),
            self._cb(2, 3 * 4, cores),
            self._cb(3, 1, cores),
            self._cb(4, 1, cores, page_size=128, dtype=ttnn.uint32),
            self._cb(5, 32, cores),
            self._cb(16, 2 * head_dim_tiles, cores),
        ]
        if q_own_cb:
            cbs.append(self._cb(1, data_pages, cores, page_size=DATA_TILE_BYTES[q.dtype], dtype=q.dtype))
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
        dtypes = (q.dtype, k.dtype, v.dtype)
        if st["pool_prog"] is None or st.get("pool_dtypes") != dtypes:  # the CB formats follow the input dtypes
            st["pool_prog"] = self._pool_program(q, k, v, st["geo"], outs)
            st["pool_dtypes"] = dtypes
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
        _, num_heads, _, head_dim = q_bar.shape
        # Names match the compile-time args of sol_route_compute.cpp.
        num_shards, head_dim_tiles = geo.shards, head_dim // TILE
        shard_block_tiles = geo.nb_pad // TILE  # blocks per shard, padded, / 32
        all_block_tiles = num_shards * shard_block_tiles
        assert shard_block_tiles * head_dim_tiles <= 8 * 32 and head_dim_tiles <= 4, (shard_block_tiles, head_dim_tiles)
        log2_scale = st["scale"] * LOG2E
        grid_x, grid_y, cores = self._cores()
        # Unit = (head, shard): num_heads * num_shards units, which keeps the core grid filled. The partial-moment
        # exchange packs at most 8 shards, so a 32-shard mesh (4x32) uses (head, Q-block tile) units, which read every
        # shard's pooled K.
        fits = num_shards <= 8 and num_heads * num_shards <= grid_x * grid_y
        if self.route_unit == "head_shard":
            assert fits, (num_shards, num_heads, grid_x * grid_y)
        units_per_shard = int(fits and self.route_unit != "head_q_tile")
        num_units = num_heads * num_shards if units_per_shard else num_heads * shard_block_tiles
        self.route_units = (num_units, units_per_shard)
        ct = [
            num_heads,  # 0 NUM_HEADS
            num_shards,  # 1 NUM_SHARDS
            shard_block_tiles,  # 2 SHARD_BLOCK_TILES
            head_dim_tiles,  # 3 HEAD_DIM_TILES
            _f32_bits(self.tau),  # 4 TAU
            _f32_bits(VAR_EPS / (log2_scale * log2_scale)),  # 5 VARIANCE_EPSILON
            0,  # 6 unused: 1 / valid K blocks is read at run time from the geo page (word 1)
            grid_x,  # 7 GRID_X
            grid_y,  # 8 GRID_Y
            num_units,  # 9 NUM_UNITS
            units_per_shard,  # 10 UNITS_PER_SHARD; kernels derive their unit range per core
        ]
        ins = [q_bar, kc, st["const_or"], st["valid_blocks"], st["ones"]]
        acc_in = [a for t in ins for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        acc_out = list(ttnn.TensorAccessorArgs(mask).get_compile_time_args())
        # (head, shard) units read only their shard's pooled K and exchange partial moments between the num_shards
        # cores of a head (sol_route_reader.cpp): the reader needs every logical core's NoC coordinates (common runtime
        # args 7..) and a semaphore; cb 23 receives the peers' partial moments (gather tiles), 24 ships this unit's and
        # 29 holds the shard-row selectors. The head's pooled Q is read from DRAM by its shard-0 core and shipped to the
        # other num_shards - 1 cores over the NoC (reader semaphore 1).
        core_xy = self._core_noc_xy(grid_x, grid_y) if units_per_shard else []
        st["route_core_xy"] = core_xy
        sizes = {
            0: (shard_block_tiles if units_per_shard else all_block_tiles) * head_dim_tiles,  # k_pooled
            1: shard_block_tiles * head_dim_tiles,  # q_pooled
            2: 1,  # ones
            6: shard_block_tiles,  # const_or
            7: shard_block_tiles,  # valid_blocks
            8: 8,  # partial_moments
            9: 8,  # moments
            10: head_dim_tiles,  # q_squared
            11: 1,  # threshold_col
            12: 1,  # threshold
            13: shard_block_tiles,  # block_hits
            20: head_dim_tiles,  # variance
        }
        if units_per_shard:  # 14: block_mask, 23: gather tiles, 24: this shard's reduced tiles, 29: shard_select
            sizes.update({14: shard_block_tiles, 23: (num_shards + 3) // 4, 24: 8, 29: 8})
        else:  # 16: mask_out tiles
            sizes[16] = shard_block_tiles
        cbs = [self._cb(i, n, cores) for i, n in sizes.items()]
        total = sum(sizes.values()) * TILE_BYTES
        block_row_bytes = geo.nb_pad * 4  # one shard's words of a mask row
        if units_per_shard:  # 16: two pages of the 32 row-major uint32 mask rows of (Q-block tile, shard)
            cbs.append(self._cb(16, 2, cores, page_size=32 * block_row_bytes, dtype=ttnn.uint32))
            total += 2 * 32 * block_row_bytes
        else:  # 22: the writer's ring of row buffers
            cbs.append(self._cb(22, SCRATCH_PAGES, cores, page_size=block_row_bytes, dtype=ttnn.uint32))
            total += SCRATCH_PAGES * block_row_bytes
        cbs.append(self._cb(3, 1, cores, page_size=128, dtype=ttnn.uint32))  # 3: the geo page
        total += 128
        if total > L1_CB_BUDGET:
            raise ValueError(f"sol_route circular buffers need {total} B of L1 (> {L1_CB_BUDGET}); geometry too large")
        acc_extra = [a for t in (st["shsel"], st["geo"]) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        reader = self._kernel(
            "sol_route_reader.cpp",
            cores,
            ct + acc_in + acc_extra + [units_per_shard],  # last arg: pooled Q shared over the head's cores
            [0] * 7 + core_xy,
            ttnn.ReaderConfigDescriptor(),
        )
        writer = self._kernel("sol_route_writer.cpp", cores, ct + acc_out, [0], ttnn.WriterConfigDescriptor())
        compute = self._kernel("sol_route_compute.cpp", cores, ct, [], self._compute_config(full_sync=True))
        sems = (
            [ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in (0, 1)]
            if units_per_shard
            else []
        )
        return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=sems, cbs=cbs)

    def _core_noc_xy(self, grid_x: int, grid_y: int) -> list[int]:
        """NoC coordinates (x | y << 8) of every logical worker core, index y * grid_x + x. One program with common runtime
        args runs on every chip of the mesh; the virtual worker coordinates are the same on every chip."""
        xy = [
            self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            for y in range(grid_y)
            for x in range(grid_x)
        ]
        return [w.x | (w.y << 8) for w in xy]

    def _run_route(self, q_bar: ttnn.Tensor, kc: ttnn.Tensor) -> ttnn.Tensor:
        st, geo = self._state, self.geo
        _, H, _, _ = q_bar.shape
        mask = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, H, geo.nb_pad, geo.n_blocks_pad]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        if st["route_prog"] is None:
            st["route_prog"] = self._route_program(q_bar, kc, mask)
        prog = st["route_prog"]
        ins = [q_bar, kc, st["const_or"], st["valid_blocks"], st["ones"]]
        ins += [st["shsel"], st["geo"]]
        prog.kernels[0].common_runtime_args = [t.buffer_address() for t in ins] + st["route_core_xy"]
        prog.kernels[1].common_runtime_args = [mask.buffer_address()]
        ttnn.generic_op([*ins, mask], prog)
        return mask

    def _gather(self, t: ttnn.Tensor, *, barrier: bool = False) -> ttnn.Tensor:
        """All-gather a pooled [1, H, nb_pad, D] tensor along dim 2 over the SP ring.

        One worker per link direction (more workers go through a mux core that serialises them).

        barrier: every chip waits until the whole ring has reached this gather before it sends. The K gather needs
        it: it is the first cross-chip write of a call and lands in a fresh buffer, which the allocator may place
        where the previous call's mask and pooled K/V were (the caller frees them as soon as the sparse op is
        enqueued). Without it a chip that finished that sparse op early overwrites them under a chip still running
        it, and the output of a 10 s clip changes from run to run. The V gather runs without one (130 -> 72 us per
        gather at 5 s): a chip finishes the K gather only once every chip has started it.
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
            barrier_semaphore=ccl.get_barrier_semaphore(self.sp_axis) if barrier else None,
            num_workers_per_link=1,
        )

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        *,
        scale: float,
        report_density: bool = False,
        pooled: bool = True,
    ) -> dict:
        """`pooled=False` returns the mask alone, for the sparse op's kept-blocks-only mode: pooled K is gathered
        for the routing only and pooled V not at all. The routing keeps full blocks only, so the tokens of a partly
        filled last block (`logical_n % 64` of them) are then keys for no query; Sol's own `tail=False` can still
        route to that block."""
        assert abs(scale - self._state["scale"]) < 1e-12, "scale is baked into the prepared constants"
        geo = self.geo
        q_bar, kc_local, vc_local = self._run_pool(q, k, v)
        out_dtype = k.dtype if self.pooled_dtype is None else self.pooled_dtype
        cast = out_dtype != ttnn.bfloat16
        kc = self._gather(kc_local, barrier=True)  # bf16: the route program scores against it
        if not pooled:
            vc = None
        elif cast:  # V is only returned: cast the local shard, so the gather moves the smaller tiles
            vc_local_cast = ttnn.typecast(vc_local, out_dtype)
            vc = self._gather(vc_local_cast)
            ttnn.deallocate(vc_local_cast)
        else:
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
            sub = host[:, : host.shape[1] // tuple(self.mesh_device.shape)[1 - self.sp_axis], : geo.nb].float()
            cols = torch.cat([torch.arange(s * geo.nb_pad, s * geo.nb_pad + geo.nb) for s in range(geo.shards)])
            self.last_density = float(sub[..., cols].mean())
        if not pooled:  # the route program has been enqueued on kc; the in-order queue keeps it alive for it
            ttnn.deallocate(kc)
            return {"block_mask": mask}
        if cast:  # the route program has been enqueued on the bf16 copy; the in-order queue keeps it alive for it
            kc_cast = ttnn.typecast(kc, out_dtype)
            ttnn.deallocate(kc)
            kc = kc_cast
        return {"block_mask": mask, "pooled_k": kc, "pooled_v": vc}
