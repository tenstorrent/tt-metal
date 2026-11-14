"""Hy4 decode attention block on a 1x8 mesh row (task decode-merge): q all-gather -> Hy4 split kernel over the column's own keys -> all_to_all of (o, m, l) -> online-softmax merge (+sink) in the TP8 head layout.
Per-column inputs are sharded over mesh dim 0 (column c = device c). Variants are chosen with env/args, see Block.
"""
import os
import sys

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/container_app_user/tt-metal")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tt"))
from dk_dev import SCALE, ckc_of, load_split, pcc, timed_traced  # noqa: E402

NC = 8  # columns
HL = 8  # heads per column (TP8)
D, V = 576, 512


def open_row():
    """the DSV4 way (full 4x8 mesh with 1D ring fabric) then a 1x8 submesh = mesh row 0. (Opening MeshShape(1,8) directly fails the fabric router handshake on a partial galaxy.)"""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    full = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=16384, trace_region_size=200_000_000)
    md = full.create_submesh(ttnn.MeshShape(1, 8), ttnn.MeshCoordinate(0, 0))
    return full, md


def close_row(full, md):
    ttnn.close_mesh_device(md)
    ttnn.close_mesh_device(full)


def to_mesh(md, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    """t [8, ...]: slice c goes to column c (per-device shape [1, ...] with the leading 8 split)"""
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dtype,
        layout=layout,
        mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def from_mesh(md, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))


class Block:
    """One layer's decode attention for S query rows (users) per column.
    q_pad  [1,32,S,576] bf16 RM per column (8 real heads in rows 0..7, zero pad)
    kv     per column shard (bf16 [1,1,T,576] RM or SCALED_FP8 packed)
    idx    uint32 [1,1,S,2048] per column: LOCAL rows of the column's own selected keys, sentinel tail (seqshard.split_ids)
    sink   [1,1,8*S,1] TILE bf16 per column: sink logit (scaled domain) of the column's own head h for rows (h,s)
    -> [1,1,8*S,512] per column, rows (h,s): the merged attention output of the column's 8 heads (TP8 layout for o_proj)
    """

    def __init__(
        self,
        md,
        S,
        G=None,
        kc=None,
        num_links=2,
        topo=ttnn.Topology.Ring,
        fid="hifi4",
        v2=True,
        tile=False,
        softmax=False,
        safe=False,
    ):
        self.safe = safe  # merge reference = true max over the sources and the sink (+18 us: ttnn.max over dim 0) instead of the sink
        self.tile, self.softmax = (
            tile,
            softmax,
        )  # tile: kernel writes TILE tiles in the owner layout [8j,S,8,544] (no tilize op); softmax: merge via ttnn.softmax over the sources of the lse
        self.v2 = v2  # v2: kernel writes the owner layout directly (no slice/permute), tails = [scaled max, sum, 1.0], sink as a 9th source row
        from models.demos.gpt_oss.tt.ccl import CCLManager

        self.md, self.S, self.topo, self.nl = md, S, topo, num_links
        self.ccl = CCLManager(md, num_links=num_links, topology=topo)
        self.mod = load_split()
        R = NC * S
        if G is None:
            G, kc = (8, 128) if R * 8 <= 120 else (4, 256)
        self.G, self.kc = G, kc
        self.ckc = ckc_of(md, fid)
        self.R = R

    # --- stages -------------------------------------------------------------------------------------------------
    def idx_rows(self, idx):
        """[1,1,S,K] -> [1,1,8S,K]: row (j,s) = idx[s] for every owner j (j-major). Once per index layer (shared layers reuse it)."""
        if self.tile and os.environ.get("SS_IDXMOD", "0") == "1":
            return idx
        return ttnn.repeat(idx, ttnn.Shape([1, 1, NC, 1]))

    def gather_q(self, q_pad):
        return ttnn.experimental.all_gather_async(
            q_pad,
            dim=2,
            cluster_axis=1,
            mesh_device=self.md,
            topology=self.topo,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            num_links=self.nl,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
        )

    def kernel(self, qg, kv, idx8):
        return self.mod.sparse_sdpa_split(
            qg,
            kv,
            idx8,
            V,
            SCALE,
            self.kc,
            self.G,
            self.ckc,
            None,
            emit_stats=3 if self.tile else 2,
            owner_s=self.S if self.v2 else 0,
        )  # v1 [1,32,8S,544] ; v2 [8j,1,8S,544] owner layout

    def to_owner(self, out):
        """[1,32,8S,544] RM (h, (j,s)) -> [8j,1,8S,544] TILE rows (h,s)"""
        S = self.S
        if self.tile:
            return out
        if self.v2:
            return ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        o8 = ttnn.slice(out, [0, 0, 0, 0], [1, HL, NC * S, V + 32])
        o8 = ttnn.reshape(o8, [HL, NC, S, V + 32])
        o8 = ttnn.permute(o8, (1, 0, 2, 3))
        o8 = ttnn.reshape(o8, [NC, 1, HL * S, V + 32])
        return ttnn.to_layout(o8, ttnn.TILE_LAYOUT)

    def a2a(self, x):
        return ttnn.experimental.all_to_all_async_generic(
            x, in_dim=1, out_dim=0, num_links=self.nl, topology=self.topo, cluster_axis=1
        )  # [1,8src,8S,544]

    def merge(self, x, sink):
        """x [1,8src,8S,544] TILE (cols 0..511 o normalised by the source's local sum, col 512 raw max, col 513 local sum); sink [1,1,8S,1]. -> [1,1,8S,512]"""
        R = HL * self.S
        if self.tile:
            return self.merge4(x, sink) if self.softmax else self.merge3(x, sink)
        if self.v2:
            return self.merge2(x, sink)
        o = ttnn.slice(x, [0, 0, 0, 0], [1, NC, R, V])
        m = ttnn.slice(x, [0, 0, 0, V], [1, NC, R, V + 1])
        l = ttnn.slice(x, [0, 0, 0, V + 1], [1, NC, R, V + 2])
        M = ttnn.max(m, dim=1, keepdim=True)
        d = ttnn.multiply(ttnn.subtract(m, M), SCALE)
        w = ttnn.multiply(l, ttnn.exp(d))
        num = ttnn.sum(ttnn.multiply(o, w), dim=1, keepdim=True)
        den = ttnn.sum(w, dim=1, keepdim=True)
        den = ttnn.add(den, ttnn.exp(ttnn.subtract(sink, ttnn.multiply(M, SCALE))))
        return ttnn.multiply(num, ttnn.reciprocal(den))

    def merge2(self, x, sinkrow):
        """x [1,8src,8S,544] TILE: cols 0..511 o (normalised by the source's local sum), 512 scaled max, 513 local sum, 514 = 1.0; sinkrow [1,1,8S,544]: the sink as a 9th source (max = sink logit, sum = 1, o = 0)."""
        R = HL * self.S
        x9 = ttnn.concat([x, sinkrow], dim=1)  # [1,9,R,544]
        m = ttnn.slice(x9, [0, 0, 0, V], [1, NC + 1, R, V + 1])
        l = ttnn.slice(x9, [0, 0, 0, V + 1], [1, NC + 1, R, V + 2])
        w = ttnn.multiply(l, ttnn.exp(ttnn.subtract(m, ttnn.max(m, dim=1, keepdim=True))))
        acc = ttnn.sum(ttnn.multiply(x9, w), dim=1, keepdim=True)  # [1,1,R,544]: cols 0..511 numerator, 514 denominator
        den = ttnn.slice(acc, [0, 0, 0, V + 2], [1, 1, R, V + 3])
        return ttnn.multiply(ttnn.slice(acc, [0, 0, 0, 0], [1, 1, R, V]), ttnn.reciprocal(den))

    def merge3(self, x, sinkcol):
        """tile output: x [1,8*S,32,576] (src-major, then s) TILE: o cols 0..511, tail tiles ms @512, l @544 (col 0 of each; rows = heads, 8 real + 24 pad); sinkcol [1,S,32,1] (sink logit per head row, scaled domain) -> [1,S,32,512].
        The softmax reference is the SINK instead of the max over the sources (ttnn.max over dim 0 costs 18 us, sum 2 us): w_c = l_c exp(ms_c - sink), den = sum_c w_c + 1 (the sink term e^0), no overflow while ms - sink < 88.
        Reductions over the source dim are 4-10x faster with H=32 rows than with 8 (the H=8 shape takes the padded slow path), slices are fast only on tile-aligned columns.
        """
        S = self.S
        W = V + 64
        xr = ttnn.reshape(x, [NC, S, 32, W])
        m = ttnn.slice(xr, [0, 0, 0, V], [NC, S, 32, V + 1])
        l = ttnn.slice(xr, [0, 0, 0, V + 32], [NC, S, 32, V + 33])
        if self.safe:
            M = ttnn.maximum(ttnn.max(m, dim=0, keepdim=True), sinkcol)  # [1,S,32,1]
            w = ttnn.multiply(l, ttnn.exp(ttnn.subtract(m, M)))
            den = ttnn.add(ttnn.sum(w, dim=0, keepdim=True), ttnn.exp(ttnn.subtract(sinkcol, M)))
        else:
            w = ttnn.multiply(l, ttnn.exp(ttnn.subtract(m, sinkcol)))  # [8,S,32,1]
            den = ttnn.add(ttnn.sum(w, dim=0, keepdim=True), 1.0)
        f = ttnn.multiply(w, ttnn.reciprocal(den))
        acc = ttnn.sum(ttnn.multiply(xr, f), dim=0, keepdim=True)  # [1,S,32,576]
        return ttnn.slice(acc, [0, 0, 0, 0], [1, S, 32, V])

    def merge4(self, x, sinkrow):
        S = self.S
        x9 = ttnn.concat([ttnn.reshape(x, [NC, S, HL, V + 32]), sinkrow], dim=0)
        m = ttnn.slice(x9, [0, 0, 0, V], [NC + 1, S, HL, V + 1])
        l = ttnn.slice(x9, [0, 0, 0, V + 1], [NC + 1, S, HL, V + 2])
        f = ttnn.softmax(ttnn.add(m, ttnn.log(l)), dim=0)
        o = ttnn.slice(x9, [0, 0, 0, 0], [NC + 1, S, HL, V])
        return ttnn.sum(ttnn.multiply(o, f), dim=0, keepdim=True)

    def run(self, q_pad, kv, idx8, sink, upto=99):
        st = {}
        st["qg"] = self.gather_q(q_pad)
        if upto == 1:
            return st["qg"]
        st["out"] = self.kernel(st["qg"], kv, idx8)
        if upto == 2:
            return st["out"]
        st["own"] = self.to_owner(st["out"])
        if upto == 3:
            return st["own"]
        st["x"] = self.a2a(st["own"])
        if upto == 4:
            return st["x"]
        return self.merge(st["x"], sink)
