# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-preview device embedding + LM head (copy + parametrisation of deepseek_v41_flash/tt/{device_head.py}) on a TP8 mesh row (1 x 8, or any rows x 8).

Layouts: tokens / streams are replicated over the 8 columns (every device holds the full 6144 residual) and sharded over mesh rows (``T`` users per row).
  * ``Hy4DeviceEmbedding``: ids [T,1] uint32 -> streams [T,1,4,6144] fp32 (hidden repeated into the 4 iHC channels, like DSV4). The table [120832, 6144] bf16 is NOT replicated (1.38 GiB/chip):
      mode "dshard": hidden dim sharded over the 8 columns ([120832, 768] per chip) + all_gather of the [T,1,768] lookups   (0.173 GiB/chip)
      mode "vshard": vocab sharded ([15104, 6144] per chip), ids shifted into the local range, out-of-range rows zeroed, all_reduce(sum)   (0.173 GiB/chip)
  * ``Hy4DeviceHead``: streams [T,1,4,6144] -> hc_head (hc 4 -> 1, fp32) -> RMSNorm (bf16 out) -> vocab matmul. Vocab 120832 = 8 x 15104 (472 tiles per column) -> logits [1,1,T,15104] per column.
      variants: head weight dtype bf16 / bfp8 / bfp4 / fp32 (weights), out dtype fp32 (vLLM enable_lm_head_fp32 = fp32 accumulate + fp32 logits, weights stay bf16) or bf16, math fidelity, fp32 dest acc.
"""
import os
from pathlib import Path

import torch

import ttnn

VOCAB, DIM, HC = 120832, 6144, 4
RMS_EPS, HC_EPS = 1e-5, 1e-6
WEIGHT_CACHE = os.environ.get("HY4_WEIGHT_CACHE", "")

_DT = {
    "bf16": ttnn.bfloat16,
    "bfp8": ttnn.bfloat8_b,
    "bfp4": ttnn.bfloat4_b,
    "fp32": ttnn.float32,
}
# bytes per element on device (bfp8_b / bfp4_b: 1 + 1/16 exponent byte per 16 values = 1.0625 / 0.5625 incl. shared exponents)
BYTES = {"bf16": 2.0, "bfp8": 1.0625, "bfp4": 0.5625, "fp32": 4.0}


def head_bytes_per_chip(dtype="bf16", cols=8, vocab=VOCAB, dim=DIM):
    return vocab // cols * dim * BYTES[dtype]


def _cache(name):
    if not WEIGHT_CACHE or WEIGHT_CACHE == "0":
        return None
    Path(WEIGHT_CACHE).mkdir(parents=True, exist_ok=True)
    return str(Path(WEIGHT_CACHE) / name)


def _rep(md, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


class Hy4DeviceEmbedding:
    def __init__(self, md, mc, ccl, embed_weight, mode="dshard"):
        self.md, self.mc, self.ccl, self.mode = md, mc, ccl, mode
        rows, cols = tuple(md.shape)
        self.rows, self.cols = rows, cols
        w = embed_weight.to(torch.bfloat16)
        assert w.shape == (VOCAB, DIM)
        if mode == "dshard":
            mapper = ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=(rows, cols))
            w = w.reshape(1, 1, VOCAB, DIM)
        else:
            mapper = ttnn.ShardTensor2dMesh(md, dims=(None, 2), mesh_shape=(rows, cols))
            w = w.reshape(1, 1, VOCAB, DIM)
        self.weight = ttnn.from_torch(
            w,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        self.shard = VOCAB // cols
        # this device's vocab offset (vshard): [1,1,1,1] int32 per column
        off = (torch.arange(cols, dtype=torch.float32) * self.shard).reshape(1, cols, 1, 1)
        self.off = ttnn.from_torch(
            off,
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 1), mesh_shape=(rows, cols)),
        )

    def upload_tokens(self, ids):
        rows, cols = self.rows, self.cols
        return ttnn.from_torch(
            ids.reshape(-1, 1).to(torch.int32),
            device=self.md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(rows, cols)),
        )

    def lookup(self, tokens):
        """tokens [T,1] uint32 -> hidden [T,1,6144] bf16 TILE (full width on every device)"""
        T = tokens.shape[0]
        if self.mode == "dshard":
            e = ttnn.embedding(tokens, self.weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [T,1,768]
            e = ttnn.reshape(e, [1, 1, T, DIM // self.cols])
            g = self.mc.allgather(e, self.ccl, axis=1, dim=3)  # [1,1,T,6144]
            return ttnn.reshape(g, [T, 1, DIM])
        # vshard: local id = id - off; rows outside [0, shard) are zeroed
        tf = ttnn.typecast(ttnn.reshape(ttnn.to_layout(tokens, ttnn.TILE_LAYOUT), [1, 1, T, 1]), ttnn.float32)
        loc = ttnn.subtract(tf, self.off)
        inr = ttnn.multiply(ttnn.ge(loc, 0.0), ttnn.lt(loc, float(self.shard)))  # [1,1,T,1] 1/0
        locc = ttnn.multiply(loc, inr)  # clamp to 0 outside
        lid = ttnn.reshape(ttnn.to_layout(ttnn.typecast(locc, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [T, 1])
        e = ttnn.embedding(lid, self.weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [T,1,6144]
        e = ttnn.multiply(ttnn.reshape(e, [1, 1, T, DIM]), ttnn.typecast(inr, ttnn.bfloat16))
        g = self.mc.allgather(e, self.ccl, axis=1, dim=2)  # [1,1,cols*T,6144]  (sum of the 8 partials; one is non-zero)
        g = ttnn.reshape(g, [1, self.cols, T, DIM])
        return ttnn.reshape(ttnn.sum(g, dim=1, keepdim=True), [T, 1, DIM])

    def forward(self, tokens):
        T = tokens.shape[0]
        h = self.lookup(tokens)
        h = ttnn.typecast(ttnn.reshape(h, [T, 1, 1, DIM]), ttnn.float32)
        return ttnn.repeat(h, [1, 1, HC, 1])


class Hy4DeviceHead:
    def __init__(
        self, md, mc, ccl, W, head_dtype="bf16", out_dtype="fp32", fidelity="HiFi2", fp32_acc=True, packer_l1=True
    ):
        """W: dict(lm_head [V,D] bf16, final_norm [D], hc_head.fn [4, 4D] fp32, hc_head.base [4], hc_head.scale [1])"""
        self.md, self.mc, self.ccl = md, mc, ccl
        rows, cols = tuple(md.shape)
        assert VOCAB % (cols * 32) == 0
        self.rows, self.cols, self.shard = rows, cols, VOCAB // cols
        self.head_dtype, self.out_dtype = head_dtype, out_dtype
        # constants of the shipped path (hidden3): fn rows in the tall layout + base + the bf16 norm weight (~0.4 MB per chip in all); the constants of the alternative paths are built lazily (__getattr__)
        self._W = W
        self.scale = float(W["hc_head.scale"].float().reshape(-1)[0])
        # fn_tall: 4 x [1,1,32,768]: flat index k = s'*6144 + d = row*768 + c  (row = 32 per user = (s', d // 768))
        self.fn_tall = [_rep(md, W["hc_head.fn"].float()[i].reshape(1, 1, 32, 768)) for i in range(HC)]
        self.base_col = _rep(md, W["hc_head.base"].float().reshape(1, 1, 1, HC))
        self.norm_w_rm = _rep(
            md,
            W["final_norm"].to(torch.bfloat16).reshape(1, 1, DIM // 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        wt = W["lm_head"].t().contiguous().reshape(1, 1, DIM, VOCAB).to(torch.bfloat16)
        ttdt = _DT[head_dtype]
        self.head_w = ttnn.from_torch(
            wt.float() if head_dtype == "fp32" else wt,
            dtype=ttdt,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=(rows, cols)),
        )
        self.out_tt = ttnn.float32 if out_dtype == "fp32" else ttnn.bfloat16
        fid = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}[
            fidelity
        ]
        self.ckc = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=fid,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_acc,
            packer_l1_acc=packer_l1,
        )
        self.ckc32 = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.fp32_in = head_dtype == "fp32"

    _LAZY = ("norm_w", "fn_bmm", "fn_t_bf", "fn_t", "base_row")

    def __getattr__(self, name):
        if name not in Hy4DeviceHead._LAZY:
            raise AttributeError(name)
        W, md = self.__dict__["_W"], self.__dict__["md"]
        fn_t = torch.nn.functional.pad(W["hc_head.fn"].float().t().contiguous(), (0, 32 - HC)).reshape(
            1, 1, HC * DIM, 32
        )
        if name == "norm_w":
            v = _rep(md, W["final_norm"].float().reshape(1, 1, 1, DIM))
        elif name == "fn_bmm":  # [32,1,768,32]: [r, 0, c, s] = fn[s].reshape(32,768)[r, c] (columns 4..31 zero)
            v = _rep(
                md,
                torch.nn.functional.pad(
                    W["hc_head.fn"].float().reshape(HC, 32, 768).permute(1, 2, 0), (0, 32 - HC)
                ).reshape(32, 1, 768, 32),
            )
        elif name == "fn_t_bf":
            v = _rep(md, fn_t, dtype=ttnn.bfloat16)
        elif name == "fn_t":
            v = _rep(md, fn_t)
        else:
            v = _rep(md, torch.nn.functional.pad(W["hc_head.base"].float().reshape(1, 1, 1, HC), (0, 32 - HC)))
        setattr(self, name, v)
        return v

    def hidden(self, streams):
        """streams [T,1,4,D] fp32 -> normed hidden [1,1,T,D] bf16: hc_head (4 -> 1, fp32) then RMSNorm, in vLLM order (the streams are rounded to bf16 first: vLLM keeps them bf16 between layers)."""
        T = streams.shape[0]
        stop = getattr(self, "stop", None)
        x = ttnn.typecast(ttnn.typecast(streams, ttnn.bfloat16), ttnn.float32)
        if stop == "round":
            return x
        xr = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [1, 1, T, HC * DIM])
        xf = ttnn.to_layout(xr, ttnn.TILE_LAYOUT)  # [1,1,T,4D] fp32
        if stop == "flat":
            return xf
        ms = ttnn.mean(ttnn.multiply(xf, xf), dim=-1, keepdim=True)  # [1,1,T,1]
        rs = ttnn.rsqrt(ttnn.add(ms, RMS_EPS))
        if stop == "ms":
            return rs
        mixes = ttnn.matmul(xf, self.fn_t, compute_kernel_config=self.ckc32)  # [1,1,T,32] (first 4 columns used)
        if stop == "mixes":
            return mixes
        z = ttnn.add(ttnn.multiply(ttnn.multiply(mixes, rs), self.scale), self.base_row)  # [1,1,T,32]
        pre = ttnn.add(ttnn.sigmoid(z), HC_EPS)  # [1,1,T,32]
        if stop == "pre":
            return pre
        y = None
        for s in range(HC):
            xs = ttnn.slice(xf, [0, 0, 0, s * DIM], [1, 1, T, (s + 1) * DIM])
            ps = ttnn.slice(pre, [0, 0, 0, s], [1, 1, T, s + 1])
            t = ttnn.multiply(xs, ps)
            y = t if y is None else ttnn.add(y, t)
        if stop == "y":
            return y
        y = ttnn.typecast(ttnn.typecast(y, ttnn.bfloat16), ttnn.float32)  # y.to(bf16) in vLLM
        ms2 = ttnn.mean(ttnn.multiply(y, y), dim=-1, keepdim=True)
        h = ttnn.multiply(ttnn.multiply(y, ttnn.rsqrt(ttnn.add(ms2, RMS_EPS))), self.norm_w)
        return ttnn.typecast(h, ttnn.float32 if self.fp32_in else ttnn.bfloat16)

    def hidden2(self, streams, mix_bf16=True, norm_op=True):
        """Faster hidden: ttnn.rms_norm for the 4D-wide statistic (x * rs), bf16 matmul for the mixes, batched matmul for the channel sum, ttnn.rms_norm for the final norm."""
        stop = getattr(self, "stop", None)
        T = streams.shape[0]
        xb = ttnn.typecast(streams, ttnn.bfloat16)  # the bf16 rounding of vLLM's inter-layer streams
        xr = ttnn.reshape(ttnn.to_layout(xb, ttnn.ROW_MAJOR_LAYOUT), [1, 1, T, HC * DIM])
        xf = ttnn.to_layout(xr, ttnn.TILE_LAYOUT)  # [1,1,T,4D] bf16
        if stop == "flat":
            return xf
        xn = ttnn.rms_norm(xf, epsilon=RMS_EPS, compute_kernel_config=self.ckc32)  # x * rsqrt(mean(x^2) + eps), bf16
        if stop == "rms":
            return xn
        if mix_bf16:
            mixes = ttnn.matmul(xn, self.fn_t_bf, compute_kernel_config=self.ckc32)
        else:
            mixes = ttnn.matmul(ttnn.typecast(xn, ttnn.float32), self.fn_t, compute_kernel_config=self.ckc32)
        if stop == "mixes":
            return mixes
        z = ttnn.add(ttnn.multiply(ttnn.typecast(mixes, ttnn.float32), self.scale), self.base_row)
        pre = ttnn.add(ttnn.sigmoid(z), HC_EPS)  # [1,1,T,32] fp32
        pre4 = ttnn.reshape(ttnn.slice(pre, [0, 0, 0, 0], [1, 1, T, HC]), [T, 1, 1, HC])
        if stop == "pre":
            return pre4
        xs = ttnn.typecast(xb, ttnn.float32)
        y = ttnn.matmul(pre4, xs, compute_kernel_config=self.ckc32)  # [T,1,1,D]
        y = ttnn.reshape(y, [1, 1, T, DIM])
        if stop == "y":
            return y
        yb = ttnn.typecast(y, ttnn.bfloat16)
        if norm_op:
            return ttnn.rms_norm(yb, epsilon=RMS_EPS, weight=self.norm_w_rm, compute_kernel_config=self.ckc32)
        y = ttnn.typecast(yb, ttnn.float32)
        ms2 = ttnn.mean(ttnn.multiply(y, y), dim=-1, keepdim=True)
        return ttnn.typecast(
            ttnn.multiply(ttnn.multiply(y, ttnn.rsqrt(ttnn.add(ms2, RMS_EPS))), self.norm_w), ttnn.bfloat16
        )

    def hidden3(self, streams, norm="manual"):
        """Tall layout: the 4*6144 values of a user are viewed as 32 rows of 768 ([T,1,32,768]: one tile row per user, 24 tiles wide) so the reductions run over a few tiles on T cores instead of 768 tiles on one:
        sumsq and the 4 mixes are elementwise products with the fn rows ([1,4,32,768] broadcast) + two small sums; the channel sum is one batched matmul on the original [T,1,4,D] layout.
        """
        stop = getattr(self, "stop", None)
        T = streams.shape[0]
        xb = ttnn.typecast(streams, ttnn.bfloat16)  # inter-layer streams are bf16 in vLLM
        xr = ttnn.reshape(ttnn.to_layout(xb, ttnn.ROW_MAJOR_LAYOUT), [T, 1, 32, 768])
        x = ttnn.typecast(ttnn.to_layout(xr, ttnn.TILE_LAYOUT), ttnn.float32)
        if stop == "flat":
            return x
        ms = ttnn.sum(ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True), dim=2, keepdim=True)  # [T,1,1,1]
        rs = ttnn.rsqrt(ttnn.add(ttnn.multiply(ms, 1.0 / (HC * DIM)), RMS_EPS))
        if stop == "ms":
            return rs
        mixes = ttnn.concat(
            [ttnn.sum(ttnn.sum(ttnn.multiply(x, f), dim=-1, keepdim=True), dim=2, keepdim=True) for f in self.fn_tall],
            dim=3,
        )  # [T,1,1,4]
        if stop == "mixes":
            return mixes
        z = ttnn.add(ttnn.multiply(ttnn.multiply(mixes, rs), self.scale), self.base_col)  # [T,1,1,4]
        pre4 = ttnn.add(ttnn.sigmoid(z), HC_EPS)
        if stop == "pre":
            return pre4
        y = ttnn.matmul(pre4, ttnn.typecast(xb, ttnn.float32), compute_kernel_config=self.ckc32)  # [T,1,1,D]
        y = ttnn.reshape(y, [1, 1, T, DIM])
        if stop == "y":
            return y
        yb = ttnn.typecast(y, ttnn.bfloat16)
        if norm == "rms_default":
            return ttnn.rms_norm(yb, epsilon=RMS_EPS, weight=self.norm_w_rm)
        y = ttnn.typecast(yb, ttnn.float32)
        ms2 = ttnn.mean(ttnn.multiply(y, y), dim=-1, keepdim=True)
        return ttnn.typecast(
            ttnn.multiply(ttnn.multiply(y, ttnn.rsqrt(ttnn.add(ms2, RMS_EPS))), self.norm_w), ttnn.bfloat16
        )

    def hidden4(self, streams, ymode="ew", mmode="loop"):
        """hidden3 with the channel sum as an elementwise product + sum over the 4 rows of the tile (ymode "ew") and optionally the mixes as a batched matmul over the 32 row chunks (mmode "bmm")."""
        stop = getattr(self, "stop", None)
        T = streams.shape[0]
        xb = ttnn.typecast(streams, ttnn.bfloat16)
        xr = ttnn.reshape(ttnn.to_layout(xb, ttnn.ROW_MAJOR_LAYOUT), [T, 1, 32, 768])
        x = ttnn.typecast(ttnn.to_layout(xr, ttnn.TILE_LAYOUT), ttnn.float32)
        if stop == "flat":
            return x
        ms = ttnn.sum(ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True), dim=2, keepdim=True)
        rs = ttnn.rsqrt(ttnn.add(ttnn.multiply(ms, 1.0 / (HC * DIM)), RMS_EPS))
        if stop == "ms":
            return rs
        if mmode == "loop":
            mixes = ttnn.concat(
                [
                    ttnn.sum(ttnn.sum(ttnn.multiply(x, f), dim=-1, keepdim=True), dim=2, keepdim=True)
                    for f in self.fn_tall
                ],
                dim=3,
            )
        else:  # bmm: [T,1,32,768] -> permute -> [32,1,T,768] @ [32,1,768,32]
            xp = ttnn.permute(x, (2, 1, 0, 3))
            m = ttnn.matmul(xp, self.fn_bmm, compute_kernel_config=self.ckc32)  # [32,1,T,32]
            mixes = ttnn.slice(ttnn.sum(m, dim=0, keepdim=True), [0, 0, 0, 0], [1, 1, T, HC])  # [1,1,T,4]
            mixes = ttnn.reshape(mixes, [T, 1, 1, HC])
        if stop == "mixes":
            return mixes
        z = ttnn.add(ttnn.multiply(ttnn.multiply(mixes, rs), self.scale), self.base_col)
        pre4 = ttnn.add(ttnn.sigmoid(z), HC_EPS)  # [T,1,1,4]
        if stop == "pre":
            return pre4
        xs = ttnn.typecast(xb, ttnn.float32)  # [T,1,4,D]
        if ymode == "ew":
            pc = ttnn.transpose(pre4, 2, 3)  # [T,1,4,1]
            y = ttnn.sum(ttnn.multiply(xs, pc), dim=2, keepdim=True)  # [T,1,1,D]
        else:
            y = ttnn.matmul(pre4, xs, compute_kernel_config=self.ckc32)
        y = ttnn.reshape(y, [1, 1, T, DIM])
        if stop == "y":
            return y
        yb = ttnn.typecast(y, ttnn.bfloat16)
        return ttnn.rms_norm(yb, epsilon=RMS_EPS, weight=self.norm_w_rm)

    def hid(self, streams):
        v = os.environ.get("HY4_HEAD_V", "3")
        if self.fp32_in or v == "1":
            return self.hidden(streams)
        if v == "4":
            return self.hidden4(streams, ymode=os.environ.get("HY4_Y", "ew"), mmode=os.environ.get("HY4_M", "loop"))
        return self.hidden3(streams, norm=os.environ.get("HY4_NORM", "manual")) if v == "3" else self.hidden2(streams)

    def forward(self, streams):
        """streams [T,1,4,D] fp32 -> logits shard [1,1,T,15104] (this column's vocab slice), fp32 or bf16."""
        h = self.hid(streams)
        return ttnn.matmul(h, self.head_w, compute_kernel_config=self.ckc, dtype=self.out_tt)

    def matmul_only(self, h):
        return ttnn.matmul(h, self.head_w, compute_kernel_config=self.ckc, dtype=self.out_tt)

    # sampling / readback ---------------------------------------------------------------------------------------------
    def sample_global(self, logits):
        """Greedy: global argmax over the vocab -> token ids [T,1] uint32 identical on every device (per column (max, idx), all-gather, best column)."""
        cols, shard = self.cols, self.shard
        lg = logits if self.out_dtype == "fp32" else ttnn.typecast(logits, ttnn.float32)
        rm = ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT)
        idx = ttnn.argmax(rm, dim=-1, keepdim=True)
        mx = ttnn.to_layout(
            ttnn.gather(rm, -1, index=idx), ttnn.TILE_LAYOUT
        )  # max = value at the argmax (gather 10 us; ttnn.max over the row 120 us)
        idxf = ttnn.typecast(ttnn.to_layout(idx, ttnn.TILE_LAYOUT), ttnn.float32)
        g = self.mc.allgather(
            ttnn.concat([mx, idxf], dim=-1), self.ccl, axis=1, dim=3
        )  # [1,1,T,cols*2] = (max_0, idx_0, max_1, idx_1, ...)
        T = g.shape[2]
        g = ttnn.reshape(g, [1, T, cols, 2])
        mx_all = ttnn.reshape(ttnn.slice(g, [0, 0, 0, 0], [1, T, cols, 1]), [1, 1, T, cols])
        ix_all = ttnn.reshape(ttnn.slice(g, [0, 0, 0, 1], [1, T, cols, 2]), [1, 1, T, cols])
        best = ttnn.argmax(ttnn.to_layout(mx_all, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True)
        bestf = ttnn.typecast(ttnn.to_layout(best, ttnn.TILE_LAYOUT), ttnn.float32)
        if getattr(self, "_col_ids", None) is None:
            self._col_ids = _rep(self.md, torch.arange(cols, dtype=torch.float32).reshape(1, 1, 1, cols))
        sel = ttnn.sum(ttnn.multiply(ttnn.eq(bestf, self._col_ids), ix_all), dim=-1, keepdim=True)
        tok = ttnn.add(ttnn.multiply(bestf, float(shard)), sel)
        T = tok.shape[2]
        return ttnn.reshape(ttnn.to_layout(ttnn.typecast(tok, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [T, 1])

    def gather_logits(self, logits):
        """Host copy of the full logits [B, V] fp32 (diagnostics / PCC)."""
        rows, cols = tuple(self.md.shape)
        t = ttnn.to_torch(
            logits, mesh_composer=ttnn.ConcatMesh2dToTensor(self.md, dims=(2, 3), mesh_shape=(rows, cols))
        )
        return t.reshape(-1, VOCAB).float()
