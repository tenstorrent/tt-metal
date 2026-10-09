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

WEIGHT_CACHE = os.environ.get("DSV41_WEIGHT_CACHE", "/mnt/tt-data/ssinghal/dsv4-weight-cache")
VOCAB, DIM, HC = 129280, 5120, 4


def _cache(name):
    if not WEIGHT_CACHE or WEIGHT_CACHE == "0":
        return None
    Path(WEIGHT_CACHE).mkdir(parents=True, exist_ok=True)
    return str(Path(WEIGHT_CACHE) / name)


class DSV41DeviceEmbedding:
    def __init__(self, mesh_device, embed_weight=None, users_per_row=4):
        """``embed_weight``: [vocab, dim] torch tensor; may be None when the cache file already exists."""
        self.md = mesh_device
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

    def forward(self, tokens):
        """tokens [T,1] uint32 -> (streams [T,1,4,D] fp32, pre [T,1,1,4] fp32)."""
        T = tokens.shape[0]
        emb = ttnn.embedding(tokens, self.weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [T,1,D]
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

    def forward(self, streams, pre):
        """streams [T,1,4,D] fp32, pre [T,1,1,4] fp32 -> logits shard [1,1,T,vocab/cols] fp32 (this column's vocab slice)."""
        T = streams.shape[0]
        y = ttnn.matmul(pre, streams, compute_kernel_config=self.ckc32)  # hc_pre collapse -> [T,1,1,D]
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
        cols = self.cols
        shard = VOCAB // cols
        mx = ttnn.max(logits, dim=-1, keepdim=True)  # [1,1,T,1]
        idx = ttnn.argmax(ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True)
        idxf = ttnn.typecast(ttnn.to_layout(idx, ttnn.TILE_LAYOUT), ttnn.float32)
        mx_all = mesh_config.allgather(mx, ccl, axis=1, dim=3)  # [1,1,T,cols]
        ix_all = mesh_config.allgather(idxf, ccl, axis=1, dim=3)
        best = ttnn.argmax(
            ttnn.to_layout(mx_all, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True
        )  # first max wins, like torch
        bestf = ttnn.typecast(ttnn.to_layout(best, ttnn.TILE_LAYOUT), ttnn.float32)  # [1,1,T,1]
        if getattr(self, "_col_ids", None) is None:  # [1,1,1,cols] = 0..cols-1, built once outside any trace
            self._col_ids = ttnn.from_torch(
                torch.arange(cols, dtype=torch.float32).reshape(1, 1, 1, cols),
                device=self.md,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.md),
            )
        sel = ttnn.sum(
            ttnn.multiply(ttnn.eq(bestf, self._col_ids), ix_all), dim=-1, keepdim=True
        )  # index inside the best column
        tok = ttnn.add(ttnn.multiply(bestf, float(shard)), sel)  # exact in fp32 (< 2^24)
        T = tok.shape[2]
        return ttnn.reshape(ttnn.to_layout(ttnn.typecast(tok, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [T, 1])

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
