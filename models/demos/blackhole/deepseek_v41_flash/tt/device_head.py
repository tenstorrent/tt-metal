# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device embedding and LM head for the DeepSeek-V4.1-Flash decode (4x8 mesh, batch rows x users_per_row).

Layouts (same as the layers): tokens are sharded over mesh rows (``T`` users per row) and replicated over columns.

  * ``DSV41DeviceEmbedding``: token ids [T,1] uint32 -> the 4 identical fp32 residual streams [T,1,4,D] and the
    one-hot ``pre`` [T,1,1,4] the first layer collapses with. The table [129280, 5120] bf16 is replicated.
  * ``DSV41DeviceHead``: last layer's streams [T,1,4,D] + ``pre`` -> logits. hc collapse, RMSNorm and the vocab
    projection; the vocab (129280 = 8 x 16160) is sharded over the mesh columns, so every device returns the logits of
    its 16160 vocabulary slice for its T users. ``argmax`` reduces them on the host (8 (value, index) pairs per user).

Both load their weights through ``ttnn.as_tensor`` with a cache file, like the rest of the repo's model code, so only
the first run reads the checkpoint.
"""

import os
from pathlib import Path

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.ckpt_stamp import ensure_stamp

WEIGHT_CACHE = os.environ.get("DSV41_WEIGHT_CACHE", "/mnt/tt-data/ssinghal/dsv4-weight-cache")
VOCAB, DIM, HC = 129280, 5120, 4


def _cache(name):
    if not WEIGHT_CACHE or WEIGHT_CACHE == "0":
        return None
    ensure_stamp(WEIGHT_CACHE)
    Path(WEIGHT_CACHE).mkdir(parents=True, exist_ok=True)
    return str(Path(WEIGHT_CACHE) / name)


class ShardedEmbeddingTable:
    """DSV41_EMBED_SHARD=1 (default off): the embedding table split over the 8 mesh columns along the hidden dim ([vocab, 5120/8] bf16 per chip, 19.7 instead of 157.8 MiB/bank,
    -138 MiB/bank; measured in tests/test_embedding_shard.py of wt/embed_shard). Every column looks the same token ids up in its slice, then ONE all_gather over the
    columns restores the full row (a copy, bit-identical to the replicated lookup). Used wherever the replicated table was passed to ttnn.embedding (``embed_lookup``).
    """

    def __init__(self, mesh_device, embed_weight=None):
        rows, cols = tuple(mesh_device.shape)
        assert DIM % (cols * 32) == 0
        self.md, self.cols = mesh_device, cols
        self.weight = ttnn.as_tensor(
            embed_weight.to(torch.bfloat16) if embed_weight is not None else None,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 1), mesh_shape=(rows, cols)),
            cache_file_name=_cache(f"embed_bf16_rm_dshard{cols}"),
        )

    def set_ccl(self, mesh_config, ccl):
        """the model's mesh config / CCL manager: the gather then runs as ``all_gather_async`` on the model's (lean) ping-pong semaphores (20-25 us instead of 50-80 us of the sync op)."""
        self.mc, self.ccl = mesh_config, ccl

    def _ag(self, x):
        """all_gather of a 4D tensor over the mesh columns along dim 3 (model CCL when set_ccl was called, else the sync op)."""
        mc = getattr(self, "mc", None)
        if mc is None:
            return ttnn.all_gather(x, dim=3, cluster_axis=1, num_links=1, topology=ttnn.Topology.Ring)
        kw = {"lean_ag": True} if getattr(mc, "_dsv41_lean", False) else {}
        return mc.allgather(x, self.ccl, axis=1, dim=3, **kw)

    def _own_idx(self, T):
        """[T, 1] uint32 per device: column c holds the row numbers c*T .. (c+1)*T - 1 (its own tokens inside the gathered [cols*T] block)."""
        cache = self.__dict__.setdefault("_own", {})
        if T not in cache:
            rows, cols = tuple(self.md.shape)
            cache[T] = ttnn.from_torch(
                torch.arange(cols * T, dtype=torch.int32).reshape(cols * T, 1),
                device=self.md,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(None, 0), mesh_shape=(rows, cols)),
            )
        return cache[T]

    def lookup_rm_colsplit(self, tokens):
        """Like ``lookup_rm`` for token ids that DIFFER per mesh column (column-split prefill: column c owns the 32-token chunks i % 8 == c). Every column holds only a D / cols slice of
        the table, so (1) the ids of all columns are gathered, (2) every column looks all of them up in its slice, (3) the slices are gathered into full rows of all ``cols * T``
        tokens, (4) each column picks its own T rows with a row gather (``_own_idx``). Rows are bit-identical to the replicated lookup.
        """
        T = tokens.shape[0]
        rows, cols = tuple(self.md.shape)
        ids = self._ag(ttnn.reshape(tokens, [1, 1, 1, T]))  # [1, 1, 1, cols * T], column-major blocks of T ids
        e = ttnn.embedding(
            ttnn.reshape(ids, [1, cols * T]), self.weight, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
        )  # [1, cols*T, D/cols]
        g = self._ag(ttnn.reshape(e, [1, 1, cols * T, e.shape[-1]]))  # [1, 1, cols*T, D]
        tab = ttnn.reshape(g, [cols * T, g.shape[-1]])
        return ttnn.embedding(self._own_idx(T), tab, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)  # [T, 1, D]

    def lookup_rm(self, tokens, colsplit=False):
        """token ids [T, 1] -> the full rows [T, 1, D] bf16 ROW_MAJOR (local lookup of the D / cols slice, then one all_gather over the columns).
        ``colsplit``: the ids differ per mesh column (column-split prefill): see ``lookup_rm_colsplit``."""
        if colsplit:
            return self.lookup_rm_colsplit(tokens)
        T = tokens.shape[0]
        e = ttnn.embedding(tokens, self.weight, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)  # [T, 1, D / cols]
        mc = getattr(self, "mc", None)
        if mc is None:
            g = ttnn.all_gather(e, dim=len(e.shape) - 1, cluster_axis=1, num_links=1, topology=ttnn.Topology.Ring)
            return g
        e = ttnn.reshape(e, [1, 1, T, e.shape[-1]])
        kw = {"lean_ag": True} if getattr(mc, "_dsv41_lean", False) else {}
        g = mc.allgather(e, self.ccl, axis=1, dim=3, **kw)
        return ttnn.reshape(g, [T, 1, g.shape[-1]])

    def lookup(self, tokens, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, colsplit=False):
        g = self.lookup_rm(tokens, colsplit)
        return g if layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(g, layout)


def embed_shard_on():
    return os.environ.get("DSV41_EMBED_SHARD", "0") == "1"


def embed_rm_on():
    """DSV41_EMBED_RM=1 (also implied by DSV41_EMBED_SHARD=1): DSV41DeviceEmbedding.forward builds the 4 fp32 streams from a ROW_MAJOR lookup (concat of 4 copies + one tilize) instead of the
    tile lookup + typecast + repeat: bit-identical, 240 -> 80 us at T=16, 361 -> 112 us at T=24, 19.7 -> 6.7 ms for a 2048-token/row prefill chunk (tests/test_embedding_dshard_bench.py).
    """
    return os.environ.get("DSV41_EMBED_RM", "0") == "1" or embed_shard_on()


def embed_lookup(table, tokens, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, colsplit=False):
    """ttnn.embedding on a replicated table, or the sharded lookup (+ all_gather) of a ``ShardedEmbeddingTable`` (``colsplit``: the token ids differ per mesh column)."""
    if isinstance(table, ShardedEmbeddingTable):
        return table.lookup(tokens, layout, dtype, colsplit)
    return ttnn.embedding(tokens, table, layout=layout, dtype=dtype)


class DSV41DeviceEmbedding:
    def __init__(self, mesh_device, embed_weight=None, users_per_row=4):
        """``embed_weight``: [vocab, dim] torch tensor; may be None when the cache file already exists."""
        self.md = mesh_device
        if embed_shard_on():
            self.weight = ShardedEmbeddingTable(mesh_device, embed_weight)
        else:
            self.weight = ttnn.as_tensor(
                embed_weight.to(torch.bfloat16) if embed_weight is not None else None,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=_cache("embed_bf16_rm"),
            )
        # the one-hot pre-mix is a constant: upload it once (host writes are not allowed inside a trace)
        self.pre = ttnn.from_torch(
            torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(users_per_row, 1, 1, 1),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def rebatch(self, users_per_row):
        """A new embedding for another users-per-row that shares this one's (batch independent) table: only the one-hot ``pre`` is rebuilt."""
        import copy

        new = copy.copy(self)
        new.pre = ttnn.from_torch(
            torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(users_per_row, 1, 1, 1),
            device=self.md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.md),
        )
        return new

    def upload_tokens(self, token_ids: torch.Tensor):
        """token_ids [B] int -> device tensor [B,1] uint32 sharded over mesh rows (replicated over columns)."""
        rows, cols = tuple(self.md.shape)
        return ttnn.from_torch(
            token_ids.reshape(-1, 1).to(torch.int32),
            device=self.md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(rows, cols)),
        )

    def forward(self, tokens, override=None, keep=None):
        """tokens [T,1] uint32 -> (streams [T,1,4,D] fp32, pre [T,1,1,4] fp32).
        ``override`` [1,1,T,D] bf16 tile + ``keep`` [1,1,T,1] (1 = keep the table row, 0 = take the override row; the override row is zero where keep = 1): image positions
        of a prompt take the vision encoder's rows instead of the table row of the image token id (``emb * keep + override`` is exact: keep is 0 / 1).
        """
        T = tokens.shape[0]
        src = getattr(
            self, "colsplit_src", None
        )  # the column-split prefill model (its embedding is a copy sharing the table): the ids differ per mesh column
        cs = bool(src is not None and getattr(src, "cs", False))
        if override is None and embed_rm_on():
            e = ttnn.reshape(
                embed_lookup(self.weight, tokens, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, colsplit=cs),
                [T, 1, 1, DIM],
            )
            return ttnn.to_layout(ttnn.concat([e] * HC, dim=2), ttnn.TILE_LAYOUT, dtype=ttnn.float32), self.pre
        emb = embed_lookup(self.weight, tokens, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, colsplit=cs)  # [T,1,D]
        if override is not None:
            e2 = ttnn.add(ttnn.multiply(ttnn.reshape(emb, [1, 1, T, emb.shape[-1]]), keep), override)
            emb = ttnn.reshape(e2, [T, 1, emb.shape[-1]])
        h = ttnn.typecast(ttnn.reshape(emb, [T, 1, 1, emb.shape[-1]]), ttnn.float32)
        streams = ttnn.repeat(h, [1, 1, HC, 1])
        return streams, self.pre


class DSV41DeviceHead:
    def __init__(self, mesh_device, norm_weight, head_weight=None, norm_eps=1e-6, dtype=None):
        """``norm_weight`` [dim] fp32; ``head_weight`` [vocab, dim] torch (None when the cache file exists)."""
        self.md = mesh_device
        rows, cols = tuple(mesh_device.shape)
        assert VOCAB % (cols * 32) == 0, f"vocab {VOCAB} must split into whole tiles over {cols} columns"
        self.cols, self.eps = cols, norm_eps
        self._col_ids = None
        dtype = dtype or {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("DSV41_HEAD_DTYPE", "bfp8")]
        self.dtype = dtype
        self.norm_w = ttnn.from_torch(
            norm_weight.reshape(1, 1, 1, -1).float(),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        tag = "bfp8" if dtype == ttnn.bfloat8_b else "bf16"
        self.head_w = ttnn.as_tensor(  # [1,1,dim,vocab], vocab sharded over the mesh columns
            head_weight.t().contiguous().reshape(1, 1, DIM, VOCAB).to(torch.bfloat16)
            if head_weight is not None
            else None,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=(rows, cols)),
            cache_file_name=_cache(f"head_{tag}_vshard{cols}"),
        )
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.ckc32 = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def forward(self, streams, pre, collapse_l1=False):
        """streams [T,1,4,D] fp32, pre [T,1,1,4] fp32 -> logits shard [1,1,T,vocab/cols] fp32 (this column's vocab slice)."""
        T = streams.shape[0]
        y = ttnn.matmul(
            pre,
            streams,
            compute_kernel_config=self.ckc32,
            memory_config=ttnn.L1_MEMORY_CONFIG if collapse_l1 else ttnn.DRAM_MEMORY_CONFIG,
        )  # hc_pre collapse -> [T,1,1,D]
        y = ttnn.reshape(y, [1, 1, T, DIM])
        x = ttnn.typecast(ttnn.typecast(y, ttnn.bfloat16), ttnn.float32)  # reference rounds to bf16 first
        ms = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True)
        x = ttnn.typecast(
            ttnn.multiply(ttnn.multiply(x, ttnn.rsqrt(ttnn.add(ms, self.eps))), self.norm_w), ttnn.bfloat16
        )
        return ttnn.matmul(x, self.head_w, compute_kernel_config=self.ckc, dtype=ttnn.float32)

    def sample(self, logits, mesh_config, ccl):
        """Greedy sampling INSIDE the trace: per-column (max, index) of this column's vocab slice, all-gathered over the 8 mesh
        columns -> [1,1,T,16] fp32 = (max_0, idx_0, max_1, idx_1, ...). ``combine`` finishes it on the host from 16 numbers.
        """
        mx = ttnn.max(logits, dim=-1, keepdim=True)  # [1,1,T,1]
        idx = ttnn.argmax(
            ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True
        )  # local index in this slice
        idxf = ttnn.typecast(ttnn.to_layout(idx, ttnn.TILE_LAYOUT), ttnn.float32)
        return mesh_config.allgather(ttnn.concat([mx, idxf], dim=-1), ccl, axis=1, dim=3)

    def sample_global(self, logits, mesh_config, ccl):
        """Greedy sampling entirely on the device: the global argmax over the vocab -> token ids [T,1] uint32 row-major, identical on
        every device. Per column (max, local index) -> all-gathered over the 8 columns -> best column -> token = col * shard + idx.
        """
        if os.environ.get("DSV41_SAMPLE_V2", "1") == "1":
            return self.sample_global_rm(ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), mesh_config, ccl)
        cols = self.cols
        shard = VOCAB // cols
        mx = ttnn.max(logits, dim=-1, keepdim=True)  # [1,1,T,1]
        idx = ttnn.argmax(ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True)
        idxf = ttnn.typecast(ttnn.to_layout(idx, ttnn.TILE_LAYOUT), ttnn.float32)
        # ONE gather of the (max, local index) PAIR instead of two width-1 gathers: a width-1 gather of this shape is a
        # broadcast + concat pair of launches (~55 us) and the pair rides in the very same two launches
        pair = mesh_config.allgather(
            ttnn.concat([mx, idxf], dim=-1), ccl, axis=1, dim=3
        )  # [1,1,T,2*cols] = (max_0, idx_0, max_1, idx_1, ...)
        if getattr(self, "_pair_ids", None) is None:  # [1,1,1,2*cols] constants, built once outside any trace
            mk = lambda t: ttnn.from_torch(  # noqa: E731
                t.reshape(1, 1, 1, 2 * cols),
                device=self.md,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.md),
            )
            hide = torch.zeros(2 * cols, dtype=torch.float32)
            hide[1::2] = -1e30  # the index lanes can never win the argmax over the interleaved row
            self._pair_ids = (mk(hide), mk(torch.arange(2 * cols, dtype=torch.float32) - 1.0))
        hide, lane = self._pair_ids
        best = ttnn.argmax(
            ttnn.to_layout(ttnn.add(pair, hide), ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True
        )  # = 2 * best column; first max wins, like torch
        bestf = ttnn.typecast(ttnn.to_layout(best, ttnn.TILE_LAYOUT), ttnn.float32)  # [1,1,T,1]
        sel = ttnn.sum(
            ttnn.multiply(ttnn.eq(bestf, lane), pair), dim=-1, keepdim=True
        )  # lane 2c+1 (lane[j] = j-1) of the best column c = its local index
        tok = ttnn.add(ttnn.multiply(bestf, float(shard) / 2.0), sel)  # 2c * shard/2 + idx, exact in fp32 (< 2^24)
        T = tok.shape[2]
        return ttnn.reshape(ttnn.to_layout(ttnn.typecast(tok, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [T, 1])

    @property
    def shard(self):
        return VOCAB // self.cols

    def sample_pair_rm(self, rm, mesh_config, ccl, tile_pairs=False):
        """First half of ``sample_global_rm``: the all-gathered (max_c, local index_c) pairs of the 8 columns, fp32 TILE [1,1,T,2*cols] (reduced by pair_select.py).
        tile_pairs=True: max and index as two tile-aligned columns [1,1,T,64] (col 0 of each tile) and the one-worker all-gather: [1,1,T,64*cols]; the 2-wide gather is two programs of ~25 us,
        the tile-aligned one is one program (45.9 -> 20.2 us, 14.2 us lean: tests/test_pair_gather_bench.py)."""
        idx = ttnn.argmax(rm, dim=-1, keepdim=True)
        mx = ttnn.gather(rm, 3, idx)
        if tile_pairs:
            pad = [(0, 0), (0, 0), (0, 0), (0, 31)]
            both = ttnn.to_layout(
                ttnn.concat([ttnn.pad(mx, pad, 0.0), ttnn.pad(ttnn.typecast(idx, ttnn.float32), pad, 0.0)], dim=3),
                ttnn.TILE_LAYOUT,
            )  # [1,1,T,64] row = [max, 0 x 31, idx, 0 x 31]
            kw = {"lean_ag": True} if getattr(mesh_config, "_dsv41_lean", False) else {}
            return mesh_config.allgather(both, ccl, axis=1, dim=3, **kw)
        pair_in = ttnn.to_layout(ttnn.concat([mx, ttnn.typecast(idx, ttnn.float32)], dim=3), ttnn.TILE_LAYOUT)
        return mesh_config.allgather(pair_in, ccl, axis=1, dim=3)

    def sample_global_rm(self, rm, mesh_config, ccl):
        """``sample_global`` from the ROW-MAJOR fp32 logits [1,1,T,V/cols] of this column: the same tokens (exact: the max is the gathered element at the argmax index, ties -> lowest index like torch),
        in 12 ops instead of 30: the width-16160 ``max`` (3 single-core programs, 123 us) is a 3 us ``gather``, the (max, index) pair is built and the 8 pairs are reduced in row-major with strided slices.
        """
        cols = self.cols
        shard = VOCAB // cols
        T = rm.shape[2]
        idx = ttnn.argmax(rm, dim=-1, keepdim=True)  # [1,1,T,1] uint32, local index in this slice
        mx = ttnn.gather(rm, 3, idx)  # [1,1,T,1] fp32 = max
        pair_in = ttnn.to_layout(ttnn.concat([mx, ttnn.typecast(idx, ttnn.float32)], dim=3), ttnn.TILE_LAYOUT)
        pair = mesh_config.allgather(pair_in, ccl, axis=1, dim=3)  # [1,1,T,2*cols] = (max_0, idx_0, max_1, idx_1, ...)
        pr = ttnn.to_layout(pair, ttnn.ROW_MAJOR_LAYOUT)
        mxs = ttnn.slice(pr, [0, 0, 0, 0], [1, 1, T, 2 * cols], [1, 1, 1, 2])  # the 8 column maxima
        ids = ttnn.slice(pr, [0, 0, 0, 1], [1, 1, T, 2 * cols], [1, 1, 1, 2])  # their local indices
        c = ttnn.argmax(mxs, dim=-1, keepdim=True)  # best column, first max wins
        il = ttnn.gather(ids, 3, c)
        tok = ttnn.add(
            ttnn.multiply(ttnn.typecast(c, ttnn.float32), float(shard)), il
        )  # col * shard + idx, exact in fp32 (< 2^24)
        return ttnn.reshape(ttnn.typecast(tok, ttnn.uint32), [T, 1])

    def combine(self, sampled):
        """Host side of ``sample``: read ONE device per mesh row (the gather made every column identical) -> tokens [B]."""
        rows, cols = tuple(self.md.shape)
        shard = VOCAB // cols
        devs = ttnn.get_device_tensors(
            ttnn.from_device(sampled) if os.environ.get("DSV41_COMBINE", "1") == "1" else sampled
        )  # one mesh read, then host views
        v = torch.cat(
            [ttnn.to_torch(devs[r * cols]).float().reshape(-1, cols, 2)[:, :, :] for r in range(rows)]
        )  # [B, cols, 2]
        best = v[:, :, 0].argmax(-1)
        return v[torch.arange(v.shape[0]), best, 1].long() + best * shard

    def gather_logits(self, logits):
        """Host copy of the full logits [B, vocab] (diagnostics / PCC; the decode loop uses ``argmax``)."""
        rows, cols = tuple(self.md.shape)
        t = ttnn.to_torch(
            logits, mesh_composer=ttnn.ConcatMesh2dToTensor(self.md, dims=(2, 3), mesh_shape=(rows, cols))
        )
        return t.reshape(-1, VOCAB).float()  # rows concatenate users, columns concatenate vocab

    def argmax(self, logits):
        """Greedy token per user [B]: per-column (max, index) on device, 8-way combine on the host."""
        rows, cols = tuple(self.md.shape)
        shard = VOCAB // cols
        rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        idx = ttnn.argmax(rm, dim=-1, keepdim=True)  # local index in this column's slice
        mx = ttnn.max(logits, dim=-1, keepdim=True)
        comp = lambda t, d: ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMesh2dToTensor(self.md, dims=d, mesh_shape=(rows, cols))
        )
        idx_h, mx_h = comp(idx, (2, 3)).reshape(-1, cols).long(), comp(mx, (2, 3)).reshape(-1, cols).float()
        best_col = mx_h.argmax(-1)
        return idx_h.gather(1, best_col[:, None])[:, 0] + best_col * shard
