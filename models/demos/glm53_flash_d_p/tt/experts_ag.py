# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 routed experts on the MiMo all-gather MoE ops (GLM_EXPERTS_MODE=ag): no dispatch / combine.

Port of models/demos/mimo_v2_d_p/tt/moe_ag.py (MoeAgBlock, branch mstaletovic/mimo-v2-dp) onto the ttnn.bringup ops
(INDEX.md: moe_ag, flat_routed_expert_ttnn, fabric_all_gather_ttnn, fabric_reduce_scatter_ttnn), for a mesh with two
rows (the dispatch axis 0, SP = 2) and any number of columns (TP); more rows (a Galaxy 4x4) skip the fused send-back:

    x, top-k (idx, w) of mesh row r's S/2 tokens (the same on every chip of the row)
      -> fabric_all_gather over axis 0 -> every chip holds the chunk's T = S tokens (row r's at r S/2)
         (replicated input: x already holds all S tokens, no gather)
      -> moe_ag_route_plan: this chip's local experts' (token, k) pairs as flat rows (counts / regions / token_index /
         y_slot), from the gathered top-k and the chip's local-slot map
      -> flat_routed_expert in indexed mode (flat row i reads gathered row token_index[i]; clamped_silu = silu(min(g,
         10)) * clamp(u, +-10), as GLM): y [rows, H] bf16 row major
      -> moe_ag_local_reduce, fused send-back: phase 1 the other row's tokens' partial, fabric_all_gather axis 0, phase
         2 this row's tokens' partial + the peer's -> [S/2, H] bf16 tiles: the column's sum for row r's tokens
         (R > 2 rows: moe_ag_local_reduce phase 0 over all T tokens as tiles, fabric_reduce_scatter axis 0 -> the same)
      -> reduce over the columns: split (the model's default residual layout) fabric_reduce_scatter axis 1 -> the
         chip's [S/(2 C), H] rows r S/2 + c S/(2 C) ..; replicated: reduce_scatter + all_gather on axis 1, then
         all_gather axis 0 -> [S, H] on every chip.

Expert placement: device d = r C + c (row-major) holds global experts d E/n .. (d + 1) E/n - 1. Weights: one laid-out
flat cache per layer (the flat op's own layout, keyed by dtype and plan hash) under
generated/glm53_flash_d_p/tt_cache/flat/<rows>x<cols>; a miss dequantizes the layer's fp8 experts (reference/weights.py).
Precision (tests/test_flat_stage_probe.py, test_bfp8_rounding.py): the packer rounds bf16 -> bfp8 ties away from zero
(+0.28% per pack); with fp32 down accumulation and stochastic packer rounding the flat expert is unbiased against fp32
math on its own bfp4 weights (coef 1.0005). Partial sums are bf16 (+0.1%); the unified path adds them in fp32.
Knobs: GLM_AG_GATHER_OP (fabric | high_bw), GLM_AG_RS_OP (fabric | ttnn), GLM_MOE_LINKS (links of every collective),
GLM_AG_DOWN_FP32 (default 1: fp32 DEST down projection; 0: bf16 DEST, ~3-4% output gain with bfp4 weights),
GLM_AG_PACK_SRND (default 0; 1: stochastic rounding in the flat expert's packers, analysis only),
GLM_AG_XBF16 / GLM_AG_HBF16 (default 0: the flat expert's x / h as bfp8 tiles; 1: bf16).
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.glm53_flash_d_p.tt.experts import CACHE_ROOT, LazyExpertWeights

NONE = 0xFFFFFFFF
# more than two mesh rows, split output: one full-mesh reduce-scatter of the [T, H] partials (a snake line over every
# chip, chip d gets rows d T/n) instead of axis 0 then axis 1. 4x4 Galaxy, chunk 5120: 852 vs 906 us device, but 15
# bf16 roundings of the running sum instead of 3 + 3 (max abs err 0.37 vs 0.19 on unit partials); off by default
RS_FULL_MESH = os.environ.get("GLM_MOE_RS_FULL_MESH", "0") == "1"


def _up(v, a):
    return -(-v // a) * a


def _dram(mesh, shape, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, layout, mesh, ttnn.DRAM_MEMORY_CONFIG)


def _per_device(mesh, per_dev, dtype):
    """per_dev: torch [rows, cols, ...] -> each device its [1, 1, ...] slice (row major, DRAM)."""
    rows, cols = tuple(mesh.shape)
    return ttnn.from_torch(
        per_dev.reshape(rows, cols, 1, -1),
        device=mesh,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(0, 1)),
    )


def flat_rows(tokens, k, experts_per_chip):
    """Flat expert rows that hold any routing without dropping a pair: a token puts at most min(k, epc) pairs on a chip
    and each active local expert pads its region to 32 rows (moe_ag_route_plan requires at least this many)."""
    pairs = tokens * min(k, experts_per_chip)
    return _up(pairs, 32) + 32 * (min(pairs, experts_per_chip) - 1)


def default_gids(n_dev, epc):
    """gids[d]: the global expert ids of row-major device d, in local order (contiguous blocks)."""
    return [list(range(d * epc, (d + 1) * epc)) for d in range(n_dev)]


def local_map(mesh, gids, n_global):
    """gids[d] -> per device [1, 1, n_global] uint32: the global id's local slot, or NONE."""
    lmap = torch.full((len(gids), n_global), NONE, dtype=torch.int64)
    for d, gl in enumerate(gids):
        for l, g in enumerate(gl):
            lmap[d, g] = l
    return _per_device(mesh, lmap, ttnn.uint32)


def chip_info(mesh, s):
    """Per device [1, 16] uint32: word 0 its mesh row, word 1 the other row's block start in the 2-row gather
    ((1 - r) S/2: where the peer's partials for this chip's tokens land), word 2 its own block start (r S/2)."""
    rows, cols = tuple(mesh.shape)
    t = torch.zeros(rows, cols, 16, dtype=torch.int64)
    for r in range(rows):
        t[r, :, 0] = r
        t[r, :, 1] = (rows - 1 - r) * s
        t[r, :, 2] = r * s
    return _per_device(mesh, t, ttnn.uint32)


def flat_cache_dir(mesh):
    rows, cols = tuple(mesh.shape)
    return CACHE_ROOT / "flat" / f"{rows}x{cols}"


_SEMS = {}


def _sems(mesh):
    """Caller-owned fabric_all_gather semaphores (zero, left zero by every call): one pair per mesh."""
    hit = _SEMS.get(id(mesh))
    if hit is None or hit[0] is not mesh:
        g = mesh.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
        sems = [ttnn.create_global_semaphore(mesh, crs, 0, ttnn.BufferType.L1_SMALL) for _ in range(2)]
        hit = _SEMS[id(mesh)] = (mesh, sems)
    return hit[1]


def all_gather_rows(x, out, axis, links):
    """x [1, 1, s, W] DRAM interleaved -> out [1, 1, G s, W] (chip p's rows at p s) over mesh axis ``axis``."""
    if os.environ.get("GLM_AG_GATHER_OP", "fabric") == "high_bw":
        return ttnn.experimental.high_bw_all_gather(x, dim=2, output_tensor=out, cluster_axis=axis, num_links=links)
    s0, s1 = _sems(x.device())
    return ttnn.bringup.fabric_all_gather(
        x,
        dim=2,
        output_tensor=out,
        cluster_axis=axis,
        num_links=links,
        ready_semaphore=s0,
        data_valid_semaphore=s1,
    )


def reduce_scatter_rows(x, axis, links):
    """Sum of [1, 1, G s, H] bf16 TILE partials over mesh axis ``axis``; chip p keeps rows p s .. (a fresh tensor)."""
    if os.environ.get("GLM_AG_RS_OP", "fabric") == "fabric":
        return ttnn.bringup.fabric_reduce_scatter(x, cluster_axis=axis, num_links=links)
    return ttnn.reduce_scatter(x, dim=2, cluster_axis=axis, num_links=links, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class _Block:
    """Persistent buffers of the all-gather data movement for one mesh and chunk (S/2 tokens per mesh row), shared by
    every MoE layer (they run one after another)."""

    def __init__(self, mesh, *, s, hidden, k, n_global, gids, links):
        rows, cols = tuple(mesh.shape)
        assert rows >= 2, f"the all-gather MoE needs at least two mesh rows, got {tuple(mesh.shape)}"
        self.mesh, self.S, self.H, self.K, self.links = mesh, s, hidden, k, links
        # two rows: the fused send-back into the persistent ``own``; more: a reduce-scatter over axis 0 (fresh output)
        self.fused = rows == 2
        self.T = T = rows * s
        self.epc = len(gids[0])
        self.rows = flat_rows(T, k, self.epc)
        self.gx = _dram(mesh, [1, 1, T, hidden], ttnn.bfloat16, ttnn.TILE_LAYOUT)  # gathered x, tiles (shared expert)
        self.gidx_t = _dram(mesh, [1, 1, T, k], ttnn.uint16, ttnn.TILE_LAYOUT)
        self.gw_t = _dram(mesh, [1, 1, T, k], ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.lmap = local_map(mesh, gids, n_global)
        self.counts = _dram(mesh, [1, n_global], ttnn.uint32)
        self.regions = _dram(mesh, [1, n_global], ttnn.uint32)
        # zero-initialized once: the plan writes only the used regions
        self.token_index = ttnn.from_torch(
            torch.zeros(1, self.rows, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.y_slot = _dram(mesh, [1, T * k], ttnn.uint32)
        self.info = chip_info(mesh, s)
        if self.fused:
            self.other = _dram(mesh, [1, 1, s, hidden])
            self.own = _dram(mesh, [1, 1, s, hidden], ttnn.bfloat16, ttnn.TILE_LAYOUT)
            self.g_sp = _dram(mesh, [1, 1, 2 * s, hidden])

    def gather(self, x, idx, w, replicated):
        """x [1, 1, s, H] bf16 TILE (s = S/2, mesh row r's tokens; replicated: all S tokens), idx / w [1, 1, s, K] TILE
        -> gathered x [1, 1, T, H] row major, idx [T, K] uint16 and w [T, K] bf16 row major (fresh: free them)."""
        if replicated:
            gx = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            gi = ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            gw = ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return gx, gi, gw, True
        # x gathered as tiles (the shared expert reuses them: TtExpertsAg.gathered_x), then row major for the flat op
        all_gather_rows(x, self.gx, 0, self.links)
        all_gather_rows(idx, self.gidx_t, 0, self.links)
        all_gather_rows(w, self.gw_t, 0, self.links)
        gx = ttnn.to_layout(self.gx, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gi = ttnn.to_layout(self.gidx_t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gw = ttnn.to_layout(self.gw_t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return gx, gi, gw, True

    def gather_full(self, x, idx, w):
        """Full-mesh form of gather (split, GLM_MOE_FULL_MESH): x [1, 1, S/n, H] TILE, idx / w [1, 1, S/n, K] TILE, this
        chip's own rows of the split residual (the router ran on them) -> the same gathered x / idx / w as gather: one
        cluster_axis=None gather per tensor, row-major chip order = chip d's rows at d S/n (MiMo's gather_full)."""
        if not hasattr(self, "_mesh_dim2"):
            # the split inputs declare the mesh shards on dims 0 / 1 (split_from_host's reshape); the bytes are dim 2
            # split over all chips in row-major order, which is what the full-mesh gather validates
            rows, cols = tuple(self.mesh.shape)
            dist = ttnn.MeshShape(rows, cols)
            coords = [ttnn.MeshCoordinate([c[i] for i in range(c.dims())]) for c in ttnn.MeshCoordinateRange(dist)]
            self._mesh_dim2 = ttnn.TensorTopology(dist, [ttnn.PlacementShard(2), ttnn.PlacementShard(2)], coords)
        for t in (x, idx, w):
            t.update_tensor_topology(self._mesh_dim2)
        all_gather_rows(x, self.gx, None, self.links)
        all_gather_rows(idx, self.gidx_t, None, self.links)
        all_gather_rows(w, self.gw_t, None, self.links)
        gx = ttnn.to_layout(self.gx, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gi = ttnn.to_layout(self.gidx_t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gw = ttnn.to_layout(self.gw_t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return gx, gi, gw, True

    def plan(self, gidx, lmap):
        ttnn.bringup.moe_ag_route_plan(
            gidx, lmap, self.epc, self.rows, outputs=[self.counts, self.regions, self.token_index, self.y_slot]
        )

    def reduce(self, y, gw):
        """y [rows, H] bf16 row major -> [1, 1, S/R, H] bf16 TILE: this mesh column's sum over its R chips' experts for
        this mesh row's tokens (two rows: the persistent ``own``; more: a fresh tensor, see ``fused``)."""
        if not self.fused:
            (part,) = ttnn.bringup.moe_ag_local_reduce(y, self.y_slot, gw, self.info, self.S, phase=0, tiled=True)
            col = reduce_scatter_rows(part, 0, self.links)
            ttnn.deallocate(part)
            return col
        ttnn.bringup.moe_ag_local_reduce(y, self.y_slot, gw, self.info, self.S, phase=1, outputs=[self.other])
        all_gather_rows(self.other, self.g_sp, 0, self.links)
        ttnn.bringup.moe_ag_local_reduce(
            y, self.y_slot, gw, self.info, self.S, phase=2, peer=self.g_sp, tiled=True, outputs=[self.own]
        )
        return self.own


_BLOCKS = {}
# debug hooks (tests/test_moe_routing.py): ROUTE_LOG a list -> each call appends its layer's route-plan token counts per
# global expert (every chip's copy); TIME_FLAT -> profiler signposts right around the flat op, so its device time per
# chip is charged to section "flat<layer>" (the profiler must be enabled)
ROUTE_LOG = None
TIME_FLAT = False


class TtExpertsAg:
    """GLM routed experts on the all-gather MoE ops; the same call as tt/experts.py:TtExperts."""

    def __init__(
        self,
        mesh,
        layer: int,
        torch_weights,
        num_experts: int,
        emb_dim: int,
        hidden_dim: int,
        top_k: int = 8,
        limit: float = 10.0,
        max_seq_len: int = 8192,
        num_links: int = 1,
        weights_dtype=ttnn.bfloat4_b,
        cache: bool = True,
    ):
        """torch_weights: sequence (len E) of HF expert weight dicts ((out, in) layout), indexed lazily; only read on a
        weight cache miss. max_seq_len: the longest chunk (all S tokens)."""
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

        rows, cols = tuple(mesh.shape)
        n = rows * cols
        assert rows >= 2 and num_experts % n == 0, tuple(mesh.shape)
        self.mesh_rows = rows
        assert float(limit) == 10.0, f"clamped_silu is fixed at limit 10, got {limit}"
        self.mesh, self.layer, self.links = mesh, layer, num_links
        self.E, self.K, self.H, self.I = num_experts, top_k, emb_dim, hidden_dim
        self.epc = num_experts // n
        self.max_seq_len = max_seq_len
        self.gids = default_gids(n, self.epc)
        # fp32 DEST for the down projection: bf16 DEST inflates the bfp4 experts' output by ~3-4% (flat expert worklog)
        self.down_fp32 = os.environ.get("GLM_AG_DOWN_FP32", "1") != "0"
        # unbiased packer rounding: the default rounds bf16 -> bfp8 ties away from zero (x tilize, h, y packs), +1.25%
        # on the experts' output at layer 4 (tests/test_flat_stage_probe.py); 56k top-1 0.870 -> 0.884 (unified 0.881)
        self.pack_srnd = os.environ.get("GLM_AG_PACK_SRND", "0") == "1"
        # x / h tile formats in the flat expert (default bfp8 / bfp8): bf16 x + h halves the experts' error vs fp32
        # math at LoFi (rel L2 0.019 -> 0.010, layer 4) at ~+0-10% expert time at <= 256 rows per expert, +36% at 2048
        self.x_bf16 = os.environ.get("GLM_AG_XBF16", "0") == "1"
        self.h_bf16 = os.environ.get("GLM_AG_HBF16", "0") == "1"
        # split call: the gathered x, all S rows as tiles (persistent, valid until the next MoE layer's call); the
        # model's shared expert reads it instead of all-gathering the same rows again
        self.gathered_x = None
        wdtype = {ttnn.bfloat4_b: "bf4", ttnn.bfloat8_b: "bf8"}[weights_dtype]
        prefix = None
        fake = isinstance(torch_weights, str)  # "fake": uninitialised weights (perf runs, reference/fake_weights.py)
        if cache and not fake:
            d = flat_cache_dir(mesh)
            d.mkdir(parents=True, exist_ok=True)
            prefix = str(d / f"layer_{layer}")

        def weights():
            t = lambda w: w.T.contiguous()
            out = []
            for gl in self.gids:
                dev = []
                for g in gl:
                    w = torch_weights[g]
                    dev.append((t(w["gate_proj"]), t(w["up_proj"]), t(w["down_proj"])))
                out.append(dev)
            return out

        # m: the per-expert token cap = the whole chunk (a hot expert takes most of the tokens: test_c_*_experts)
        self.flat = FlatRoutedExpert(
            mesh,
            torch_weights if fake else weights,
            m=max_seq_len,
            H=emb_dim,
            I=hidden_dim,
            gids=self.gids,
            n_global=num_experts,
            wdtype=wdtype,
            act="clamped_silu",
            pin=1,
            cache_prefix=prefix,
            x_bf16=self.x_bf16,
            h_bf16=self.h_bf16,
        )

    def _block(self, s):
        key = (id(self.mesh), s, self.H, self.K, self.E, self.epc, self.links)
        hit = _BLOCKS.get(key)
        if hit is None or hit.mesh is not self.mesh:
            assert (
                self.mesh_rows * s <= self.max_seq_len
            ), f"chunk {self.mesh_rows * s} > max_seq_len {self.max_seq_len}"
            hit = _BLOCKS[key] = _Block(
                self.mesh, s=s, hidden=self.H, k=self.K, n_global=self.E, gids=self.gids, links=self.links
            )
        return hit

    def topk_from_dense(self, dense):
        """dense [1,1,s,E] (exactly K nonzeros per row) -> (idx uint16 [1,1,s,K], weights bf16 [1,1,s,K]) TILE."""
        if dense.dtype != ttnn.bfloat16:
            dense = ttnn.typecast(dense, ttnn.bfloat16)
        wts, idx = ttnn.topk(dense, k=self.K, dim=-1, largest=True, sorted=True)
        return idx, wts

    def __call__(self, x, dense=None, idx=None, wts=None, split=False, full_mesh=False):
        """x [1,1,S,H] replicated; routing as dense [1,1,S,E] or (idx, wts) [1,1,S,K] -> experts_out [1,1,S,H] bf16
        replicated. split: x and the routing are mesh row r's share [r S/R, (r+1) S/R) (on every chip of the row); the
        output is this chip's rows [r S/R + c S/(R C), + S/(R C)) (the split residual layout). full_mesh (with split):
        x and the routing are this chip's own rows only (S/n); one full-mesh gather replaces the row's axis-1 gather and
        the axis-0 gather (GLM_MOE_FULL_MESH)."""
        tmp = []
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
            tmp.append(x)
        if idx is None:
            idx, wts = self.topk_from_dense(dense)
            tmp += [idx, wts]
        else:
            if idx.dtype != ttnn.uint16:
                idx = ttnn.typecast(idx, ttnn.uint16)
                tmp.append(idx)
            if wts.dtype != ttnn.bfloat16:
                wts = ttnn.typecast(wts, ttnn.bfloat16)
                tmp.append(wts)
        if idx.layout != ttnn.TILE_LAYOUT:
            idx = ttnn.to_layout(idx, ttnn.TILE_LAYOUT)
            tmp.append(idx)
        if wts.layout != ttnn.TILE_LAYOUT:
            wts = ttnn.to_layout(wts, ttnn.TILE_LAYOUT)
            tmp.append(wts)
        S = x.shape[-2]
        s = S if split else S // self.mesh_rows
        if full_mesh:
            assert split, "full_mesh needs the split layout"
            s = S * tuple(self.mesh.shape)[1]  # the mesh row's tokens, as the axis-1 gather would have made them
        blk = self._block(s)
        if full_mesh:
            gx, gi, gw, own_gx = blk.gather_full(x, idx, wts)
        else:
            gx, gi, gw, own_gx = blk.gather(x, idx, wts, replicated=not split)
        self.gathered_x = blk.gx if split else None
        for t in tmp:
            ttnn.deallocate(t)
        blk.plan(gi, blk.lmap)
        if ROUTE_LOG is not None:
            ROUTE_LOG.append(
                {
                    "layer": self.layer,
                    "counts": [ttnn.to_torch(t).reshape(-1).tolist() for t in ttnn.get_device_tensors(blk.counts)],
                    "gids": [list(g) for g in self.gids],
                    "T": blk.T,
                }
            )
        gx2 = ttnn.reshape(gx, (blk.T, self.H))
        if TIME_FLAT:
            from models.demos.common.bringup.testing import profiler

            profiler.signpost(f"flat{self.layer}")
        y = self.flat(
            gx2,
            blk.counts,
            blk.regions,
            token_index=blk.token_index,
            y_row_major=True,
            down_fp32=self.down_fp32,
            pack_stochastic_rounding=self.pack_srnd,
        )
        if TIME_FLAT:
            profiler.signpost(f"after_flat{self.layer}")
        if own_gx:
            ttnn.deallocate(gx)
        ttnn.deallocate(gi)
        if split and not blk.fused and RS_FULL_MESH:  # the [T, H] partials straight to the chip's own rows
            (part,) = ttnn.bringup.moe_ag_local_reduce(y, blk.y_slot, gw, blk.info, blk.S, phase=0, tiled=True)
            ttnn.deallocate(y)
            ttnn.deallocate(gw)
            out = reduce_scatter_rows(part, None, self.links)
            ttnn.deallocate(part)
            return out
        col = blk.reduce(y, gw)
        ttnn.deallocate(y)
        ttnn.deallocate(gw)
        if split:
            out = reduce_scatter_rows(col, 1, self.links)
            if not blk.fused:
                ttnn.deallocate(col)
            return out
        rs = ttnn.reduce_scatter(
            col, dim=2, cluster_axis=1, num_links=self.links, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if not blk.fused:
            ttnn.deallocate(col)
        half = ttnn.all_gather(rs, dim=2, cluster_axis=1, num_links=self.links, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(rs)
        out = ttnn.all_gather(half, dim=2, cluster_axis=0, num_links=self.links, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(half)
        return out


def build_experts_ag(mesh, loader, cfg, layer: int, max_chunk: int, weights_dtype=ttnn.bfloat4_b) -> TtExpertsAg:
    return TtExpertsAg(
        mesh,
        layer,
        "fake" if getattr(loader, "fake", False) else LazyExpertWeights(loader, layer, cfg.n_routed_experts),
        num_experts=cfg.n_routed_experts,
        emb_dim=cfg.hidden_size,
        hidden_dim=cfg.moe_intermediate_size,
        top_k=cfg.num_experts_per_tok,
        limit=cfg.swiglu_limit,
        max_seq_len=max_chunk,
        weights_dtype=weights_dtype,
        num_links=int(os.environ.get("GLM_MOE_LINKS", "1")),
        cache=os.environ.get("GLM_EXPERTS_CACHE", "1") != "0",
    )
