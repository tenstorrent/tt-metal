# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash image encoder on the 4x8 mesh: ViT (32 blocks, 2D RoPE, full attention inside one image) + 3x3 aligner.

Reference: ``inference/vision.py`` (``ViT``, ``Aligner``) and ``inference/model.py`` (``encode_image`` / ``merge_image_embeddings``) of the checkpoint.

Layout: tensor parallel over the 8 mesh columns (2 of the 16 heads, 1/8 of the MLP and of the aligner per column); the 4 mesh rows encode DIFFERENT images at the same time (row r the
r-th image of a wave: inputs sharded over the rows, the all-reduces run along each row). Activations are bf16 [1,1,n,dim] per device. The weights cost ~140 MiB per chip (~17 MiB per DRAM
bank), the persistent input buffers ~40 MiB.

Everything after the patch upload runs on the device: ViT, the 3x3 unfold (a row gather of the final features: ``ttnn.embedding`` over an index table, the aligner's first weight is stored in
the gather's (kernel position, channel) order) and the aligner, so one image costs one upload and one read back and the whole thing can be captured as a trace (``capture_traces``).

``encode_many(images)`` / ``encode(patches, n_h, n_w)`` -> [n_tokens, 5120] bf16 host rows (one per LLM token of an image's IMAGE slots, reading order);
``image_rows(...)`` adds the learned start / newline / end vectors and returns the whole span [h*(w+1)+2, 5120] of ``image_processor.image_token_types``.
"""

import os
from pathlib import Path

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.vision_args import VisionArgs  # noqa: F401 (re-exported)

WEIGHT_CACHE = os.environ.get("DSV41_WEIGHT_CACHE", "/mnt/tt-data/ssinghal/dsv4-weight-cache")
TILE = 32
SDPA_CHUNK = 128  # q / k chunk of the flash attention; image lengths are padded to a multiple of it
# padded lengths of the tower (patches) and of the aligner (tokens): a few buckets instead of every multiple of the chunk, so that ``warmup`` can compile every program at startup (a
# new shape compiles for ~10 s); the padded tail is fenced off by cu_window_seqlens. Above the last bucket: multiples of the chunk.
TRACE_MAX = int(
    os.environ.get("DSV41_VISION_TRACE_MAX", "3072")
)  # patch buckets above this run eager: the tower is device bound there (a trace gains 1.7-1.9x up to 2048 patches, 1.2-1.4x at 3072, nothing beyond)
VIT_BUCKETS = [
    int(x) for x in os.environ.get("DSV41_VIT_BUCKETS", "256,512,1024,2048,3072,4096,6144,9216").split(",")
]  # (an image of n patches uses the first bucket >= n + 1: one zero row to gather from)
AL_BUCKETS = [
    int(x) for x in os.environ.get("DSV41_ALIGN_BUCKETS", "64,128,192,256,384,512,640,768,896,1024").split(",")
]


def _bucket(n, buckets, mult):
    for b in buckets:
        if n <= b:
            return b
    return -(-n // mult) * mult


def _cache(name):
    if not WEIGHT_CACHE or WEIGHT_CACHE == "0":
        return None
    Path(WEIGHT_CACHE).mkdir(parents=True, exist_ok=True)
    return str(Path(WEIGHT_CACHE) / name)


def rope_tables(n_h, n_w, head_dim, theta):
    """2D RoPE of ``vision.get_vision_cos_sin`` -> cos, sin [n, head_dim] in the INTERLEAVED pair order of ``rotary_embedding_llama``
    (pair j of the checkpoint's split-half rotation sits at channels 2j, 2j+1)."""
    dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = (torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq).flatten(1)  # [n, head_dim/2]
    return freqs.cos().repeat_interleave(2, dim=-1), freqs.sin().repeat_interleave(2, dim=-1)


def _interleave_perm(head_dim):
    half = head_dim // 2
    perm = torch.empty(head_dim, dtype=torch.long)
    perm[0::2] = torch.arange(half)
    perm[1::2] = torch.arange(half, head_dim)
    return perm


class DSV41Vision:
    def __init__(self, mesh_device, mesh_config, ccl, args=None, shards=None, log=print):
        self.md, self.mc, self.ccl = mesh_device, mesh_config, ccl
        self.a = args or VisionArgs()
        self.rows, self.cols = tuple(mesh_device.shape)
        a = self.a
        assert a.n_heads % self.cols == 0 and a.inter % (self.cols * TILE) == 0 and a.dim % (self.cols * TILE) == 0
        self.hl = a.n_heads // self.cols  # heads per column
        self.il = a.inter // self.cols
        self.log = log
        self.sh = shards or _Shards()
        self.shard_rows = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=(self.rows, self.cols))
        self.traces = {}  # n_pad -> (trace id, trunk output [n_pad, vdim] row-major)
        self.use_trace = False  # set by ``capture_traces``
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.sdpa_ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        from models.tt_transformers.tt.common import get_rot_transformation_mat

        self.trans_mat = self._rep(get_rot_transformation_mat(), ttnn.bfloat16)
        self._load()
        self._alloc_inputs()

    # ---- weights ------------------------------------------------------------------------------------------------------------
    def _rep(self, t, dtype, name=None, layout=ttnn.TILE_LAYOUT):
        return ttnn.as_tensor(
            t,
            dtype=dtype,
            layout=layout,
            device=self.md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.md),
            cache_file_name=_cache(name) if name else None,
        )

    def _col(self, t, name, dtype=ttnn.bfloat16):
        """t [1,1,K,cols*N] -> per column [1,1,K,N] (replicated over the rows)."""
        return ttnn.as_tensor(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(None, 3), mesh_shape=(self.rows, self.cols)),
            cache_file_name=_cache(name),
        )

    def _rowp(self, t, name, dtype=ttnn.bfloat16):
        """t [1,1,cols*K,N] -> per column [1,1,K,N]."""
        return ttnn.as_tensor(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(None, 2), mesh_shape=(self.rows, self.cols)),
            cache_file_name=_cache(name),
        )

    def _load(self):
        a, sh, C, hd = self.a, self.sh, self.cols, self.a.head_dim
        tag = f"vision_tp{C}"
        g = lambda k: sh.get(k)
        w = g("vision.patch_embed.proj.weight").float()  # [1024, 588]
        kp = -(-w.shape[1] // TILE) * TILE
        wp = torch.zeros(kp, w.shape[0])
        wp[: w.shape[1]] = w.t()
        self.k_patch = kp
        self.patch_w = self._rep(wp.reshape(1, 1, kp, -1), ttnn.bfloat16, f"{tag}_patch_w")
        self.patch_b = self._rep(
            g("vision.patch_embed.proj.bias").float().reshape(1, 1, 1, -1), ttnn.bfloat16, f"{tag}_patch_b"
        )
        perm = _interleave_perm(hd)
        self.blocks = []
        for L in range(a.n_layers):
            p = f"vision.blocks.{L}."
            n = f"{tag}_L{L}"
            wqkv = g(p + "attn.wqkv.weight").float()  # [3*vdim, vdim], rows q | k | v, head-major
            bqkv = g(p + "attn.wqkv.bias").float()

            def pack(
                t,
            ):  # [3*vdim, ...] -> [..., C * 3 * hl * hd] with per column (q heads | k heads | v heads); q, k channels interleaved
                parts = list(t.chunk(3, dim=0))
                for i in (0, 1):
                    parts[i] = (
                        parts[i]
                        .reshape(a.n_heads, hd, *t.shape[1:])
                        .index_select(1, perm)
                        .reshape(a.vdim, *t.shape[1:])
                    )
                cols = [
                    torch.cat([pt[c * self.hl * hd : (c + 1) * self.hl * hd] for pt in parts], dim=0) for c in range(C)
                ]
                return torch.cat(cols, dim=0)

            qkv_w = pack(wqkv).t().contiguous()  # [vdim, C*3*hl*hd]
            qkv_b = pack(bqkv)
            w1 = g(p + "mlp.w1.weight").float()  # [2*inter, vdim]: gate | up
            gate_w, up_w = w1[: a.inter], w1[a.inter :]
            blk = {
                "norm1": self._rep(g(p + "norm1.weight").float().reshape(1, 1, 1, -1), ttnn.bfloat16, f"{n}_norm1"),
                "norm2": self._rep(g(p + "norm2.weight").float().reshape(1, 1, 1, -1), ttnn.bfloat16, f"{n}_norm2"),
                "wqkv": self._col(qkv_w.reshape(1, 1, *qkv_w.shape), f"{n}_wqkv"),
                "bqkv": self._col(qkv_b.reshape(1, 1, 1, -1), f"{n}_bqkv"),
                "wo": self._rowp(
                    g(p + "attn.wo.weight").float().t().contiguous().reshape(1, 1, a.vdim, a.vdim), f"{n}_wo"
                ),
                "bo": self._rep(g(p + "attn.wo.bias").float().reshape(1, 1, 1, -1), ttnn.bfloat16, f"{n}_bo"),
                "w_gate": self._col(gate_w.t().contiguous().reshape(1, 1, a.vdim, a.inter), f"{n}_wgate"),
                "w_up": self._col(up_w.t().contiguous().reshape(1, 1, a.vdim, a.inter), f"{n}_wup"),
                "w2": self._rowp(
                    g(p + "mlp.w2.weight").float().t().contiguous().reshape(1, 1, a.inter, a.vdim), f"{n}_w2"
                ),
            }
            blk["idx"] = L
            self.blocks.append(blk)
        self.norm = self._rep(g("vision.norm.weight").float().reshape(1, 1, 1, -1), ttnn.bfloat16, f"{tag}_norm")
        r2 = a.vdim * a.ratio**2
        w1 = g(
            "aligner.w1.weight"
        ).float()  # [dim, vdim * 9], input index = channel * 9 + kernel position (F.unfold order)
        w1k = (
            w1.view(a.dim, a.vdim, a.ratio**2).permute(0, 2, 1).reshape(a.dim, r2)
        )  # input index = kernel position * vdim + channel (the row gather's order)
        self.al_w1 = self._col(w1k.t().contiguous().reshape(1, 1, r2, a.dim), f"{tag}_al_w1k")
        self.al_b1 = self._col(g("aligner.w1.bias").float().reshape(1, 1, 1, -1), f"{tag}_al_b1")
        self.al_w2 = self._rowp(
            g("aligner.w2.weight").float().t().contiguous().reshape(1, 1, a.dim, a.dim), f"{tag}_al_w2"
        )
        self.al_b2 = self._rep(g("aligner.w2.bias").float().reshape(1, 1, 1, -1), ttnn.bfloat16, f"{tag}_al_b2")
        self.delim = {k: g(k).to(torch.bfloat16) for k in ("image_start", "image_end", "image_newline")}

    # ---- forward ------------------------------------------------------------------------------------------------------------
    # ---- persistent input buffers (allocated before any trace of the model is captured: a persistent tensor allocated under a live trace can be clobbered by its replays) --------------
    def _host_rows(self, t, dtype, layout):
        """host tensor [rows, ...] -> the sharded host tensor of an upload (row r -> mesh row r, replicated over the columns)."""
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, mesh_mapper=self.shard_rows)

    def _dev_rows(self, shape, dtype, layout):
        return ttnn.from_torch(
            torch.zeros(*shape),
            dtype=dtype,
            layout=layout,
            device=self.md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.shard_rows,
        )

    def _alloc_inputs(self):
        R, a = self.rows, self.a
        self.in_n, self.in_nt = {}, {}
        for n_pad in VIT_BUCKETS:
            # row-major: tilized on the device (a host tilize of the 45 MB patch tensor of a 4 x 9216 patch wave costs ~40 ms)
            self.in_n[n_pad] = {
                "x": self._dev_rows((R, 1, n_pad, self.k_patch), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                "cos": self._dev_rows((R, 1, n_pad, a.head_dim), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                "sin": self._dev_rows((R, 1, n_pad, a.head_dim), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                "mask": self._dev_rows((R, 1, n_pad, 1), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                "cu": self._dev_rows((R, 3), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            }

        for nt_pad in AL_BUCKETS:
            self.in_nt[nt_pad] = self._dev_rows((R, nt_pad * a.ratio**2), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    # ---- forward ------------------------------------------------------------------------------------------------------------
    def _allreduce(self, x):
        return self.mc.allreduce(x, self.ccl, axis=1)

    def _block(self, x, blk, cos, sin, cu_window, n_pad):
        a = self.a
        h = ttnn.rms_norm(x, weight=blk["norm1"], epsilon=a.eps, compute_kernel_config=self.ckc)
        qkv = ttnn.linear(h, blk["wqkv"], bias=blk["bqkv"], compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
        ttnn.deallocate(h)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.hl, num_kv_heads=self.hl, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)
        q = ttnn.experimental.rotary_embedding_llama(
            q, cos, sin, self.trans_mat, is_decode_mode=False, compute_kernel_config=self.sdpa_ckc
        )
        k = ttnn.experimental.rotary_embedding_llama(
            k, cos, sin, self.trans_mat, is_decode_mode=False, compute_kernel_config=self.sdpa_ckc
        )
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.md.compute_with_storage_grid_size(),
            q_chunk_size=SDPA_CHUNK,
            k_chunk_size=SDPA_CHUNK,
            exp_approx_mode=False,
        )
        o = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=a.head_dim**-0.5,
            program_config=pc,
            compute_kernel_config=self.sdpa_ckc,
            cu_window_seqlens=cu_window,
        )
        for t in (q, k, v):
            ttnn.deallocate(t)
        o = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,n,hl*hd]
        part = ttnn.linear(o, blk["wo"], compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
        ttnn.deallocate(o)
        red = self._allreduce(part)
        att = ttnn.add(red, blk["bo"])
        ttnn.deallocate(red)
        x = ttnn.add(x, att)
        ttnn.deallocate(att)
        h = ttnn.rms_norm(x, weight=blk["norm2"], epsilon=a.eps, compute_kernel_config=self.ckc)
        gate = ttnn.linear(h, blk["w_gate"], compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
        up = ttnn.linear(h, blk["w_up"], compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
        ttnn.deallocate(h)
        act = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        part = ttnn.linear(act, blk["w2"], compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
        ttnn.deallocate(act)
        red = self._allreduce(part)
        x = ttnn.add(x, red)
        ttnn.deallocate(red)
        return x

    def _trunk(self, n_pad):
        """ViT on the persistent inputs of the bucket -> the final features, rows >= n zeroed, [n_pad, vdim] row-major per device (row r: its image)."""
        a, bn = self.a, self.in_n[n_pad]
        tile = lambda t: ttnn.to_layout(t, ttnn.TILE_LAYOUT)
        cu = ttnn.reshape(bn["cu"], [3])
        cos, sin, mask, xin = tile(bn["cos"]), tile(bn["sin"]), tile(bn["mask"]), tile(bn["x"])
        x = ttnn.linear(xin, self.patch_w, bias=self.patch_b, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
        ttnn.deallocate(xin)
        for blk in self.blocks:
            x = self._block(x, blk, cos, sin, cu, n_pad)
        for t in (cos, sin):
            ttnn.deallocate(t)
        x = ttnn.rms_norm(x, weight=self.norm, epsilon=a.eps, compute_kernel_config=self.ckc)
        x = ttnn.multiply(
            x, mask
        )  # rows >= n are zero: the gather reads the zero row n for every cell of the 3x3 windows outside the image
        ttnn.deallocate(mask)
        xr = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [n_pad, a.vdim])
        ttnn.deallocate(x)
        return xr

    def _aligner_dev(self, xr, n_pad, nt_pad):
        """3x3 unfold (a row gather of the features) + aligner -> [1,1,nt_pad,dim] per device (row r: the tokens of its image)."""
        a = self.a
        gidx = ttnn.reshape(self.in_nt[nt_pad], [1, nt_pad * a.ratio**2])
        g = ttnn.embedding(gidx, xr, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, nt_pad * 9, vdim]
        u = ttnn.to_layout(ttnn.reshape(g, [1, 1, nt_pad, a.vdim * a.ratio**2]), ttnn.TILE_LAYOUT)
        h1 = ttnn.linear(u, self.al_w1, bias=self.al_b1, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
        ttnn.deallocate(u)
        h1 = ttnn.gelu(h1, fast_and_approximate_mode=False)
        part = ttnn.linear(h1, self.al_w2, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
        ttnn.deallocate(h1)
        red = self._allreduce(part)
        out = ttnn.add(red, self.al_b2)
        ttnn.deallocate(red)
        return out

    def _run(self, n_pad, nt_pad):
        xr = self._trunk(n_pad)
        out = self._aligner_dev(xr, n_pad, nt_pad)
        ttnn.deallocate(xr)
        return out

    # ---- host side of a wave of images ---------------------------------------------------------------------------------------
    def _gather_index(self, n_h, n_w, nt_pad):
        """[nt_pad * 9] int: for LLM token (i, j) and kernel position k = a * 3 + b the ViT row (3i + a) * n_w + (3j + b), n (= the zero row) outside the image / for the padded tokens."""
        r, n = self.a.ratio, n_h * n_w
        th, tw = -(-n_h // r), -(-n_w // r)
        rr = r * torch.arange(th).reshape(th, 1, 1, 1) + torch.arange(r).reshape(1, 1, r, 1)
        cc = r * torch.arange(tw).reshape(1, tw, 1, 1) + torch.arange(r).reshape(1, 1, 1, r)
        idx = (
            torch.where((rr < n_h) & (cc < n_w), rr * n_w + cc, torch.full((), n))
            .expand(th, tw, r, r)
            .reshape(th * tw, r * r)
        )
        out = torch.full((nt_pad, r * r), n, dtype=torch.long)
        out[: th * tw] = idx
        return out.reshape(-1)

    def _upload_wave(self, imgs, n_pad, nt_pad):
        """Host tensors of the wave (one image per mesh row) -> the persistent inputs of the buckets."""
        R, a = self.rows, self.a
        bn = self.in_n[n_pad]
        xin = torch.zeros(R, 1, n_pad, self.k_patch, dtype=torch.bfloat16)
        cos = torch.ones(R, 1, n_pad, a.head_dim)
        sin = torch.zeros(R, 1, n_pad, a.head_dim)
        mask = torch.zeros(R, 1, n_pad, 1)
        cu = torch.zeros(R, 3, dtype=torch.int32)
        gidx = torch.zeros(R, nt_pad * a.ratio**2, dtype=torch.int32)
        for r, (patches, n_h, n_w) in enumerate(imgs):
            n = n_h * n_w
            xin[r, 0, :n, : patches[0].numel()] = patches.reshape(n, -1)
            c, s = rope_tables(n_h, n_w, a.head_dim, a.theta)
            cos[r, 0, :n], sin[r, 0, :n] = c, s
            mask[r, 0, :n] = 1.0
            cu[r] = torch.tensor(
                [0, n, n_pad], dtype=torch.int32
            )  # the zero-padded tail is its own window: it never mixes into the image
            gidx[r] = self._gather_index(n_h, n_w, nt_pad).to(torch.int32)
        up = lambda t, dev, dt, lay: ttnn.copy_host_to_device_tensor(self._host_rows(t, dt, lay), dev)
        up(xin, bn["x"], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        up(cos, bn["cos"], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        up(sin, bn["sin"], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        up(mask, bn["mask"], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        up(cu, bn["cu"], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        up(gidx, self.in_nt[nt_pad], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    def _buckets(self, imgs):
        r = self.a.ratio
        n_pad = _bucket(max(h * w for _, h, w in imgs) + 1, VIT_BUCKETS, SDPA_CHUNK)
        nt = max(-(-h // r) * -(-w // r) for _, h, w in imgs)
        return n_pad, _bucket(nt, AL_BUCKETS, TILE)

    def _encode_wave(self, imgs):
        """<= rows images -> their aligner rows (host). The missing rows repeat the last image (their result is dropped)."""
        R, r = self.rows, self.a.ratio
        real = len(imgs)
        imgs = list(imgs) + [imgs[-1]] * (R - real)
        n_pad, nt_pad = self._buckets(imgs)
        assert (
            n_pad in self.in_n
        ), f"image of {max(h * w for _, h, w in imgs)} patches is beyond the buckets {VIT_BUCKETS}"
        self._upload_wave(imgs, n_pad, nt_pad)
        tr = self.traces.get(n_pad) if self.use_trace else None
        if tr is not None:  # the trunk (ViT) replays its trace, the aligner (~10 ops) runs eager
            ttnn.execute_trace(self.md, tr[0], cq_id=0, blocking=False)
            out = self._aligner_dev(tr[1], n_pad, nt_pad)
        else:
            out = self._run(n_pad, nt_pad)
        devs = ttnn.get_device_tensors(out)
        rows = []
        for i in range(real):
            _, h, w = imgs[i]
            nt = -(-h // r) * -(-w // r)
            rows.append(ttnn.to_torch(devs[i * self.cols])[0, 0, :nt].to(torch.bfloat16))
        ttnn.synchronize_device(
            self.md
        )  # every device done (the readback waits for the devices it reads only) before anything else runs
        ttnn.deallocate(out)
        return rows

    def encode_many(self, images):
        """images [(patches [n, 3, p, p] bf16, n_h, n_w)] -> list of aligner rows [n_llm_h * n_llm_w, dim] bf16 (host), in order. Waves of ``rows`` images of similar size run in parallel on the mesh rows."""
        out = [None] * len(images)
        order = sorted(range(len(images)), key=lambda i: images[i][1] * images[i][2])
        for w0 in range(0, len(order), self.rows):
            wave = order[w0 : w0 + self.rows]
            for i, rows in zip(wave, self._encode_wave([images[i] for i in wave])):
                out[i] = rows
        return out

    def encode(self, patches, n_h, n_w):
        """One image -> aligner rows [n_llm_h * n_llm_w, dim] bf16 (host)."""
        return self.encode_many([(patches, n_h, n_w)])[0]

    def image_rows(self, patches, n_h, n_w, types):
        """The whole image span [len(types), dim] bf16: aligner rows at the IMAGE slots, the learned vectors at start / newline / end
        (``types`` = ``image_processor.image_token_types``)."""
        from models.demos.blackhole.deepseek_v41_flash.tt.vision_prompt import (
            IMAGE,
            IMAGE_END,
            IMAGE_NEW_LINE,
            IMAGE_START,
        )

        out = torch.empty(types.numel(), self.a.dim, dtype=torch.bfloat16)
        out[types == IMAGE_START] = self.delim["image_start"]
        out[types == IMAGE_END] = self.delim["image_end"]
        out[types == IMAGE_NEW_LINE] = self.delim["image_newline"]
        out[types == IMAGE] = self.encode(patches, n_h, n_w)
        return out

    def image_rows_many(self, items):
        """items [(patches, n_h, n_w, types)] -> list of whole spans (see ``image_rows``), the images encoded together over the mesh rows."""
        from models.demos.blackhole.deepseek_v41_flash.tt.vision_prompt import (
            IMAGE,
            IMAGE_END,
            IMAGE_NEW_LINE,
            IMAGE_START,
        )

        encoded = self.encode_many([(p, h, w) for p, h, w, _ in items])
        outs = []
        for (_, _, _, types), rows in zip(items, encoded):
            out = torch.empty(types.numel(), self.a.dim, dtype=torch.bfloat16)
            out[types == IMAGE_START] = self.delim["image_start"]
            out[types == IMAGE_END] = self.delim["image_end"]
            out[types == IMAGE_NEW_LINE] = self.delim["image_newline"]
            out[types == IMAGE] = rows
            outs.append(out)
        return outs

    def warm_pairs(self, n_pad):
        """The aligner buckets of a patch bucket: the one a normal (near square) aspect ratio gives (n / 9 tokens plus the border of the 3x3 grid), and the next larger one."""
        nt = -(-(n_pad - 1) // self.a.ratio**2) + int(0.7 * n_pad**0.5) + 2
        first = _bucket(
            min(nt, AL_BUCKETS[-1]), AL_BUCKETS, TILE
        )  # (an image has at most max_image_tokens = 1024 tokens)
        return [first] + [b for b in AL_BUCKETS if b > first][:1]

    def _synthetic_wave(self, n_pad):
        """<= rows images of n_pad - 1 patches (3 x n): every program of the bucket runs on them."""
        n_h = 3 * 8
        n_w = (n_pad - 1) // n_h
        return [(torch.zeros(n_h * n_w, 3, self.a.patch, self.a.patch, dtype=torch.bfloat16), n_h, n_w)] * self.rows

    def warmup(self):
        """Compile every program of the tower on synthetic waves: every patch bucket x its aligner buckets."""
        for n_pad in VIT_BUCKETS:
            imgs = self._synthetic_wave(n_pad)
            for nt_pad in self.warm_pairs(n_pad):
                self._upload_wave(imgs, n_pad, nt_pad)
                ttnn.deallocate(self._run(n_pad, nt_pad))
                ttnn.synchronize_device(self.md)

    def capture_traces(self):
        """Capture the ViT trunk as one trace per patch bucket up to ``TRACE_MAX`` patches: AFTER every other trace of the process was captured and every persistent tensor allocated (the inputs, weights and the CCL
        semaphores of this object exist since ``__init__``): the trace's intermediates are allocated and freed during the capture, and a persistent tensor allocated after a capture could sit
        on them. ``_encode_wave`` replays the trace of the wave's bucket; the aligner (a few ops, nt_pad dependent) runs eager. The trunk outputs stay allocated (the traces write them).
        """
        for n_pad in [b for b in VIT_BUCKETS if b <= TRACE_MAX]:
            self._upload_wave(self._synthetic_wave(n_pad), n_pad, self.warm_pairs(n_pad)[0])
            ttnn.synchronize_device(self.md)
            tid = ttnn.begin_trace_capture(self.md, cq_id=0)
            xr = self._trunk(n_pad)
            ttnn.end_trace_capture(self.md, tid, cq_id=0)
            ttnn.synchronize_device(self.md)
            self.traces[n_pad] = (tid, xr)
        self.use_trace = True
