# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-position tables for ttnn.experimental.deepseek_prefill.rotary_embedding_indexed, built once at load.

The tables cover every position of the served cache in its block-cyclic order (keyed by the per-row chunk
S/8 = 640) and are SP-sharded over axis 0, replicated over axis 1: the op derives each chip's rows from one
``kv_actual_global`` on the device, the same math update_padded_kv_cache writes the cache with, so any 32-aligned
chunk start works without a host slice (as deepseek_v3_d_p/tt/mla/rope.py:get_rope_tensors_indexed).

The op computes x * cos + (x @ trans) * sin per row, so it serves two tables:
    rope_tables     YaRN (factor 2048 over 512) x LeRoPE exp(alpha) inverse frequencies, interleaved: cos / sin
                    duplicated per pair [t1, t1, t2, t2, ...]; trans maps (x[2i], x[2i+1]) -> (-x[2i+1], x[2i])
    scale_tables    a per-position, per-column factor f[p, c] (cos = f, sin = 0): x * f
"""

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import seqbuf
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.layout import SP_AXIS, TP_AXIS


def block_cyclic_reorder(t: torch.Tensor, chunk_local: int, sp: int) -> torch.Tensor:
    """[1, 1, T, W] natural positions -> the order whose plain SP shard gives chip c blocks c, c + sp, ..."""
    blocks = list(torch.split(t, chunk_local, dim=2))
    assert len(blocks) % sp == 0, f"{len(blocks)} blocks of {chunk_local} do not split over sp {sp}"
    return torch.cat([blocks[b] for c in range(sp) for b in range(c, len(blocks), sp)], dim=2)


def rot_matrix() -> torch.Tensor:
    m = torch.zeros(1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE)
    m[..., torch.arange(0, ttnn.TILE_SIZE, 2), torch.arange(1, ttnn.TILE_SIZE, 2)] = 1
    m[..., torch.arange(1, ttnn.TILE_SIZE, 2), torch.arange(0, ttnn.TILE_SIZE, 2)] = -1
    return m


class IndexedTables:
    """cos / sin [T, W] (natural positions, host fp32) and trans, on the mesh in the indexed op's layout."""

    def __init__(self, mesh, cos: torch.Tensor, sin, chunk: int):
        """sin may be a device table already in this layout (one zero sin table shared by every layer's scaling
        table)."""
        sp = mesh.shape[SP_AXIS]
        max_seq = cos.shape[0]
        assert chunk % (ttnn.TILE_SIZE * sp) == 0 and max_seq % chunk == 0, (max_seq, chunk, sp)
        self.mesh, self.chunk_local, self.width = mesh, chunk // sp, cos.shape[1]
        self.cos = self._put(cos)
        self.sin = sin if isinstance(sin, ttnn.Tensor) else self._put(sin)
        self.trans = ttnn.from_torch(
            rot_matrix(),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    @classmethod
    def table(cls, mesh, t: torch.Tensor, chunk: int) -> ttnn.Tensor:
        """One [T, W] host table on the mesh in this class's layout (no trans, no object)."""
        obj = cls.__new__(cls)
        obj.mesh, obj.chunk_local, obj.width = mesh, chunk // mesh.shape[SP_AXIS], t.shape[1]
        return obj._put(t)

    def _put(self, t: torch.Tensor) -> ttnn.Tensor:
        t = block_cyclic_reorder(t.float().reshape(1, 1, *t.shape), self.chunk_local, self.mesh.shape[SP_AXIS])
        dims = [None, None]
        dims[SP_AXIS] = 2
        return ttnn.from_torch(
            t,
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=dims),
        )


class RowTables:
    """An fp32 per-position table [T, W] (natural positions, host) for composed fp32 ops, which
    rotary_embedding_indexed (bf16 only) cannot apply. ``rows(start)`` is fp32 [1, 1, n, W] per chip: the rows of the
    positions the chip holds in a chunk at ``start`` (any tile-aligned start), the ones rotary_embedding_indexed and
    update_padded_kv_cache give it.

    Chip (r, t) keeps SP row r's block-cyclic shard (blocks r, r + sp, ...), moved on by ``sub`` * t rows when the chip
    holds the t-th ``sub`` rows of its SP row (query rows split over TP). Where its rows start in the shard depends on
    the start, as rotary_embedding_indexed's update_idxt: the slab plus the in-block offset on the boundary chip (the
    one holding ``start``), the next slab on the chips before it, the slab on those after. ttnn.slice takes one start
    for every chip, so each candidate is sliced and every chip keeps its own (ttnn.where on its SP index): exact. A
    chunk-aligned start is one slice."""

    def __init__(self, mesh, t: torch.Tensor, chunk: int, n: int, sub: int = 0):
        sp, tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
        self.sp, self.chunk, self.block, self.n, self.width = sp, chunk, chunk // sp, n, t.shape[1]
        shards = block_cyclic_reorder(t.float().reshape(1, 1, *t.shape), self.block, sp).reshape(sp, -1, self.width)

        def moved(s, k):
            return torch.cat([s[k:], s.new_zeros(k, self.width)]) if k else s

        per = [[moved(shards[r], sub * c) for c in range(tp)] for r in range(sp)]  # [sp][tp]: tensor dims 0, 1
        dims = [None, None]
        dims[SP_AXIS], dims[TP_AXIS] = 0, 1

        def put(x):
            return ttnn.from_torch(
                x.contiguous(),
                device=mesh,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
            )

        self.table = put(torch.stack([torch.stack(row) for row in per]))
        self.chip = put(torch.arange(sp).float().view(sp, 1, 1, 1).expand(sp, tp, n, self.width))  # chip's SP index

    def _at(self, a: int) -> ttnn.Tensor:
        return ttnn.slice(self.table, [0, 0, a, 0], [1, 1, a + self.n, self.width])

    def rows(self, start: int) -> ttnn.Tensor:
        """Slices may be views of the table, so only the selections made here are deallocated."""
        assert start % ttnn.TILE_SIZE == 0, f"chunk start {start} is not tile-aligned"
        slab = start // self.chunk * self.block
        b, off = (start // self.block) % self.sp, start % self.block
        out, owned = self._at(slab), False  # the chips after the boundary chip (every chip at a chunk-aligned start)
        for cmp, keep, a in ((ttnn.eq, off, slab + off), (ttnn.lt, b, slab + self.block)):
            if not keep:
                continue
            pick = cmp(self.chip, float(b))  # the boundary chip; the chips before it
            sel = ttnn.where(pick, self._at(a), out)
            ttnn.deallocate(pick)
            if owned:
                ttnn.deallocate(out)
            out, owned = sel, True
        return out


def log_positions(max_seq: int, width: int = ttnn.TILE_SIZE) -> torch.Tensor:
    """log(p + 1) of every position, fp32 [max_seq, width] (the column repeated): the softmax scaling's
    f[p, h] = s_h log(p + 1) + b_h is computed from it on the device (SSMAX_TABLE=shared_log)."""
    return torch.arange(1, max_seq + 1).float().log()[:, None].expand(max_seq, width).contiguous()


def log_rows(mesh, max_seq: int, chunk: int, n: int, sub: int = 0) -> "RowTables":
    """The log(p + 1) RowTables of n rows per chip (sub: the TP split of the rows), one per mesh and shape."""
    return seqbuf.get(
        mesh, ("log_rows", max_seq, chunk, n, sub), lambda: RowTables(mesh, log_positions(max_seq), chunk, n, sub)
    )


def rope_tables(mesh, inv_freq: torch.Tensor, max_seq: int, chunk: int) -> IndexedTables:
    """SEQ_BUFFERS=shared: one copy per (max_seq, chunk, inverse frequencies), every RoPE layer's (seqbuf.py)."""
    return seqbuf.get(
        mesh,
        ("rope", max_seq, chunk, seqbuf.tensor_key(inv_freq)),
        lambda: _rope_tables(mesh, inv_freq, max_seq, chunk),
    )


def _rope_tables(mesh, inv_freq: torch.Tensor, max_seq: int, chunk: int) -> IndexedTables:
    freqs = torch.arange(max_seq)[:, None].float() * inv_freq[None, :].float()  # fp32, as the reference
    cos = torch.stack((freqs.cos(), freqs.cos()), dim=-1).flatten(-2)
    sin = torch.stack((freqs.sin(), freqs.sin()), dim=-1).flatten(-2)
    return IndexedTables(mesh, cos, sin, chunk)


def scale_tables(mesh, f: torch.Tensor, chunk: int) -> IndexedTables:
    return IndexedTables(mesh, f, torch.zeros_like(f), chunk)


class SplitTables:
    """An fp32 per-position table [T, W] (natural positions, host) selected on the device at a device-held chunk
    start (ChunkMeta.start), exactly: the trace-safe form of RowTables.rows. rotary_embedding_indexed (bf16 only) picks
    a chip's rows of a table at a start tensor; the fp32 table is split into three bf16 tables hi + mid + lo whose fp32
    sum is the table (checked here), each picked with cos = part, sin = 0 on ones (1 x part, exact at HiFi4), cast to
    fp32 (exact) and summed in fp32 (exact: every partial sum is representable). ``select`` gives the same [1, 1, n, W]
    fp32 rows per chip as RowTables(mesh, t, chunk, n, sub).rows(start), sub = n with ``subshard`` (rows over TP)."""

    def __init__(self, mesh, t: torch.Tensor, chunk: int):
        t = t.float().contiguous()
        hi = t.bfloat16().float()
        r1 = t - hi
        mid = r1.bfloat16().float()
        lo = (r1 - mid).bfloat16().float()
        assert torch.equal((hi + mid) + lo, t), "fp32 table is not hi + mid + lo"
        tiny = (lo != 0) & (lo.abs() < 2.0**-126)
        assert not bool(tiny.any()), "a lo part is subnormal (the device flushes it)"
        sp = mesh.shape[SP_AXIS]
        self.mesh, self.width = mesh, t.shape[1]
        base = IndexedTables.__new__(IndexedTables)
        base.mesh, base.chunk_local, base.width = mesh, chunk // sp, t.shape[1]
        self.parts = [base._put(p) for p in (hi, mid, lo)]
        self.zero = base._put(torch.zeros_like(t))
        self.trans = ttnn.from_torch(
            rot_matrix(),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.ones = {}
        self.cfg = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def prepare(self, n: int) -> None:
        """Allocate the ones input of n rows (outside a trace capture)."""
        if n not in self.ones:
            self.ones[n] = ttnn.from_torch(
                torch.ones(1, 1, n, self.width),
                device=self.mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

    def select(self, start: ttnn.Tensor, n: int, subshard: bool) -> ttnn.Tensor:
        assert n in self.ones, "SplitTables.prepare(n) first (outside the capture)"
        acc = None
        for p in self.parts:
            b = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
                self.ones[n],
                p,
                self.zero,
                self.trans,
                start,
                cluster_axis=SP_AXIS,
                seq_subshard_axis=TP_AXIS if subshard else None,
                compute_kernel_config=self.cfg,
            )
            f = ttnn.typecast(b, ttnn.float32)
            ttnn.deallocate(b)
            if acc is None:
                acc = f
            else:
                s = ttnn.add(acc, f, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(acc)
                ttnn.deallocate(f)
                acc = s
        return acc


_SPLIT = {}  # (id(mesh), chunk, digest) -> SplitTables: layers with the same table share one


def split_tables(mesh, t: torch.Tensor, chunk: int) -> SplitTables:
    import hashlib

    t = t.float().contiguous()
    key = (id(mesh), chunk, tuple(t.shape), hashlib.sha256(t.numpy().tobytes()).hexdigest())
    if key not in _SPLIT:
        _SPLIT[key] = SplitTables(mesh, t, chunk)
    return _SPLIT[key]


def log_split(mesh, max_seq: int, chunk: int) -> SplitTables:
    """The log(p + 1) SplitTables (SSMAX_TABLE=shared_log, the metadata path), one per mesh, max_seq and chunk; keyed
    without hashing the table (it is a function of max_seq)."""
    key = (id(mesh), chunk, (max_seq, ttnn.TILE_SIZE), "log(p + 1)")
    if key not in _SPLIT:
        _SPLIT[key] = SplitTables(mesh, log_positions(max_seq), chunk)
    return _SPLIT[key]


def select_split(meta, t: SplitTables, n: int, subshard: bool) -> ttnn.Tensor:
    """fp32 [1, 1, n, W] per chip of split table ``t`` at the chunk start meta holds, computed once per chunk and shared
    by every layer reading the same table (meta.derived; freed by meta.unpack / free_derived, not by the caller)."""
    key = ("split", id(t), n, subshard)
    if key not in meta.derived:
        meta.derived[key] = t.select(meta.start, n, subshard)
    return meta.derived[key]


def release_split(mesh) -> None:
    """Deallocates the mesh's SplitTables (TtModel.resize builds them again at the new max_seq)."""
    for k in [k for k in _SPLIT if k[0] == id(mesh)]:
        t = _SPLIT.pop(k)
        for x in [*t.parts, t.zero, t.trans, *t.ones.values()]:
            if x.is_allocated():
                ttnn.deallocate(x)
