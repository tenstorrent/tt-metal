# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batched Gated-DeltaNet (linear attention) for multi-user serving.

Replaces the per-row/per-token exact-emulation path with standard batched TTNN ops:
  * one matmul for the input projections of all rows/tokens,
  * native ttnn.transformer.chunk_gated_delta_rule (WY chunked) for prefill,
  * native ttnn.transformer.fused_recurrent_gated_delta_rule (in-place ring mode) for decode.

Per chip (TP4): 4 key heads, 8 value heads, head dim 128. Local channel order of the conv input is
[q 512 | k 512 | v 1024]. State storage (per layer, per chip):
  * recurrent: SEGMENTS tensors [SEG*8, 128, 128] fp32; slot s lives in segment s // SEG, blocks
    (s % SEG) * 8 ... +8.  A decode step updates whole segments in place; rows that must not change
    get beta = 0 and g = 0, which makes the delta-rule update an exact identity.
  * conv: 3 tensors [1, 1, 32, 2048] bf16 TILE (x_{t-3}, x_{t-2}, x_{t-1}); row s = slot s.
"""

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG


def _dbg(mesh, what):
    import os

    if os.environ.get("QWEN_FAST_DEBUG_SYNC", "0") == "1":
        print("DBG before-sync", what, flush=True)
        ttnn.synchronize_device(mesh)
        print("DBG ok", what, flush=True)


SEG = 8
SEGMENTS = 4
OPT = __import__("os").environ.get("QWEN_FAST_OPT", "1") == "1"
KH, VH, D = 4, 8, 128  # local heads per chip


class FastGDN:
    def __init__(self, layer, sd, mesh, *, weight_dtype=ttnn.bfloat16):
        self.layer = layer
        self.mesh = mesh
        self.eps = layer.config.rms_norm_eps
        qkv = sd["linear_attn.in_proj_qkv.weight"]  # [8192, 2048]
        q, k, v = qkv.split((2048, 2048, 4096), dim=0)
        z = sd["linear_attn.in_proj_z.weight"]  # [4096, 2048]
        a, b = sd["linear_attn.in_proj_a.weight"], sd["linear_attn.in_proj_b.weight"]  # [32, 2048]
        taps = sd["linear_attn.conv1d.weight"][:, 0, :]  # [8192, 4]
        tq, tk, tv = taps.split((2048, 2048, 4096), dim=0)
        pieces, conv = [], []
        for r in range(4):
            ks, vs, hs = slice(r * 512, (r + 1) * 512), slice(r * 1024, (r + 1) * 1024), slice(r * 8, (r + 1) * 8)
            pad = lambda t: torch.nn.functional.pad(t, (0, 0, 0, 24))
            # local columns: q 0:512 | k 512:1024 | v 1024:2048 | z 2048:3072 | a 3072:3080 (pad 3104) | b 3104:3112
            pieces.append(torch.cat((q[ks], k[ks], v[vs], z[vs], pad(a[hs]), pad(b[hs])), dim=0).T)
            conv.append(torch.cat((tq[ks], tk[ks], tv[vs]), dim=0))  # [2048, 4]
        shard = lambda t, dim, dtype, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=layout,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        )
        self.w_in = shard(torch.cat(pieces, dim=1).reshape(1, 1, 2048, 4 * 3136), 3, weight_dtype)
        conv = torch.cat(conv, dim=0)  # [8192, 4] rank-major
        self.taps = [shard(conv[:, j].reshape(1, 1, 1, 8192), 3, ttnn.bfloat16) for j in range(4)]
        self.neg_a = shard((-sd["linear_attn.A_log"].float().exp()).reshape(1, 1, 1, 32), 3, ttnn.float32)
        self.dt_bias = shard(sd["linear_attn.dt_bias"].float().reshape(1, 1, 1, 32), 3, ttnn.float32)
        rep = lambda t, dtype: ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.norm_w = rep(sd["linear_attn.norm.weight"].float().reshape(1, 1, 1, 128), ttnn.bfloat16)
        # l2norm(x) = rms_norm(x, eps/128) / sqrt(128): fold the 1/sqrt(128) into the weight.
        self.l2_w = rep(torch.full((1, 1, 1, 128), 128**-0.5), ttnn.float32)
        self.w_out = shard(sd["linear_attn.out_proj.weight"].T.contiguous().reshape(1, 1, 4096, 2048), 2, weight_dtype)
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
        )
        self.mm_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False
        )

    # ------------------------------------------------------------------ state
    @staticmethod
    def new_state(mesh):
        z = lambda shape, dtype: ttnn.zeros(
            shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM
        )
        return {
            "rec": [z((SEG * VH, D, D), ttnn.float32) for _ in range(SEGMENTS)],
            "conv": [z((1, 1, 32, 2048), ttnn.bfloat16) for _ in range(3)],
        }

    @staticmethod
    def reset_state(state):
        for t in state["rec"] + state["conv"]:
            ttnn.fill(t, 0.0, output_tensor=t)

    @staticmethod
    def slot_view(state, slot):
        seg = state["rec"][slot // SEG]
        return ttnn.reshape(seg, (SEG, VH, D, D)), slot % SEG

    # ------------------------------------------------------------------ shared pieces
    def _project(self, x):
        """x [1, 1, R, 2048] bf16 -> projections [1, 1, R, 3136] bf16 (local)."""
        return ttnn.linear(x, self.w_in, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_compute, memory_config=DRAM)

    def _gates(self, proj):
        """-> g, beta [1, 1, R, 8] fp32."""
        a = ttnn.typecast(proj[:, :, :, 3072:3080], ttnn.float32)
        b = ttnn.typecast(proj[:, :, :, 3104:3112], ttnn.float32)
        g = ttnn.multiply(ttnn.softplus(ttnn.add(a, self.dt_bias)), self.neg_a)
        beta = ttnn.sigmoid(b)
        return g, beta

    def _gated_norm(self, o, zz, rows):
        """o, zz [rows*8, 128] tiles -> [1, 1, rows, 1024] bf16."""
        n = ttnn.rms_norm(o, epsilon=self.eps, weight=self.norm_w, compute_kernel_config=self.compute)
        y = ttnn.multiply(n, ttnn.silu(ttnn.typecast(zz, ttnn.float32)), dtype=ttnn.bfloat16)
        return ttnn.reshape(y, (1, 1, rows, 1024))

    def _out(self, y):
        # local partial sum [1,1,R,2048] fp32; caller all-reduces
        return ttnn.linear(y, self.w_out, dtype=ttnn.float32, compute_kernel_config=self.mm_compute, memory_config=DRAM)

    # ------------------------------------------------------------------ decode
    def decode(self, x, state, *, active_f, inactive_f, segments, ring_idx, active_f32=None):
        """Batched single-token step for 32 physical rows (row = slot).

        x [1, 1, 32, 2048] normalized input. active_f / inactive_f: [1, 1, 32, 1] bf16 0/1 masks.
        segments: number of 8-slot state segments to update (covers the active extent).
        Returns the local out-projection partial [1, 1, 32, 2048] fp32.
        """
        proj = self._project(x)
        xt = proj[:, :, :, 0:2048]
        s0, s1, s2 = state["conv"]
        w0, w1, w2, w3 = self.taps
        acc = ttnn.multiply(s0, w0, dtype=ttnn.float32)
        for src, w in ((s1, w1), (s2, w2), (xt, w3)):
            acc = ttnn.add(acc, ttnn.multiply(src, w, dtype=ttnn.float32))
        conv = ttnn.silu(acc)
        conv = ttnn.typecast(conv, ttnn.bfloat16)
        if OPT:
            # exact select of the shifted history for active rows (ttnn.where: no arithmetic on the values)
            for dst, new in ((s0, s1), (s1, s2), (s2, xt)):
                ttnn.copy(ttnn.where(active_f, new, dst), dst)
            a = proj[:, :, :, 3072:3080]
            b = proj[:, :, :, 3104:3112]
            g = ttnn.multiply(ttnn.softplus(ttnn.add(a, self.dt_bias, dtype=ttnn.float32)), self.neg_a)
            g = ttnn.multiply(g, active_f32)
            beta = ttnn.multiply(ttnn.sigmoid(b), active_f32, dtype=ttnn.float32)
        else:
            # shift conv history only for active rows (exact select: m*a + (1-m)*b with m in {0,1})
            for dst, new in ((s0, s1), (s1, s2), (s2, xt)):
                mixed = ttnn.add(ttnn.multiply(new, active_f), ttnn.multiply(dst, inactive_f))
                ttnn.copy(mixed, dst)
            g, beta = self._gates(proj)
            maskf = ttnn.typecast(active_f, ttnn.float32)
            g = ttnn.multiply(g, maskf)
            beta = ttnn.multiply(beta, maskf)
        rows = segments * SEG
        q = ttnn.reshape(conv[:, :, :rows, 0:512], (rows, 1, KH, D))
        k = ttnn.reshape(conv[:, :, :rows, 512:1024], (rows, 1, KH, D))
        v = ttnn.reshape(conv[:, :, :rows, 1024:2048], (rows, 1, VH, D))
        q = ttnn.rms_norm(q, epsilon=self.eps / D, weight=self.l2_w, compute_kernel_config=self.compute)
        k = ttnn.rms_norm(k, epsilon=self.eps / D, weight=self.l2_w, compute_kernel_config=self.compute)
        g = ttnn.reshape(g[:, :, :rows, :], (rows, 1, VH))
        beta = ttnn.reshape(beta[:, :, :rows, :], (rows, 1, VH))
        outs = []
        for j in range(segments):
            sl = slice(j * SEG, (j + 1) * SEG)
            o, _ = ttnn.transformer.fused_recurrent_gated_delta_rule(
                q[sl],
                k[sl],
                v[sl],
                g[sl],
                beta[sl],
                scale=D**-0.5,
                initial_state=state["rec"][j],
                initial_state_block_idx=ring_idx,
                output_per_token_state=True,
                compute_kernel_config=self.compute,
                exact_state=True,
            )
            outs.append(o)  # [SEG, 1, 8, 128]
        o = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=0)
        o = ttnn.reshape(o, (rows * VH, D))
        zz = ttnn.reshape(proj[:, :, :rows, 2048:3072], (rows * VH, D))
        y = self._gated_norm(o, zz, rows)
        if rows < 32:
            y = ttnn.pad(y, [(0, 0), (0, 0), (0, 32 - rows), (0, 0)], 0.0)
        return self._out(y)

    # ------------------------------------------------------------------ speculative verify
    def verify(self, x, spec, li, *, users, steps):
        """Multi-token step for speculative verification (see fast_spec.py). 32 rows = `users` x `steps` candidate
        tokens, row r = user * steps + column. Conv history comes from the persistent per-layer verify buffers via
        exact 0/1 selection matmuls (host-built `spec.sel`, constant `spec.tapmat`); the recurrent state is the
        per-layer ring [steps * users * VH, K, V] in the native kernel's ring mode (per-token states written in place,
        initial block per head = `spec.ring_idx`). Same conv/gate arithmetic as `decode` (masks: spec.active_*)."""
        proj = self._project(x)
        xt = proj[:, :, :, 0:2048]
        hist_buf, xprev = spec.vhist[li], spec.xprev[li]
        e = ttnn.concat([hist_buf, xprev], dim=2)  # [1,1,64,2048]
        hist = ttnn.matmul(spec.sel, e, dtype=ttnn.bfloat16, compute_kernel_config=self.compute)  # exact select
        taps = ttnn.matmul(
            spec.tapmat, ttnn.concat([hist, xt], dim=2), dtype=ttnn.bfloat16, compute_kernel_config=self.compute
        )  # [1,1,128,2048]: the 4 conv inputs of every row (exact select)
        ttnn.copy(hist, hist_buf)
        ttnn.copy(xt, xprev)
        w0, w1, w2, w3 = self.taps
        acc = ttnn.multiply(taps[:, :, 0:32, :], w0, dtype=ttnn.float32)
        for j, w in ((1, w1), (2, w2), (3, w3)):
            acc = ttnn.add(acc, ttnn.multiply(taps[:, :, 32 * j : 32 * j + 32, :], w, dtype=ttnn.float32))
        conv = ttnn.typecast(ttnn.silu(acc), ttnn.bfloat16)
        a = proj[:, :, :, 3072:3080]
        b = proj[:, :, :, 3104:3112]
        g = ttnn.multiply(ttnn.softplus(ttnn.add(a, self.dt_bias, dtype=ttnn.float32)), self.neg_a)
        g = ttnn.multiply(g, spec.active_f32)
        beta = ttnn.multiply(ttnn.sigmoid(b), spec.active_f32, dtype=ttnn.float32)
        rows = users * steps
        q = ttnn.reshape(conv[:, :, :rows, 0:512], (rows, 1, KH, D))
        k = ttnn.reshape(conv[:, :, :rows, 512:1024], (rows, 1, KH, D))
        v = ttnn.reshape(conv[:, :, :rows, 1024:2048], (rows, 1, VH, D))
        q = ttnn.rms_norm(q, epsilon=self.eps / D, weight=self.l2_w, compute_kernel_config=self.compute)
        k = ttnn.rms_norm(k, epsilon=self.eps / D, weight=self.l2_w, compute_kernel_config=self.compute)
        # (rows, 1, H, D) -> (users, steps, H, D): identical tile memory (heads pad to 32 rows), so pass the padded
        # shape and reshape is a free view (plain reshape untilizes + retilizes: ~43 us each)
        q = ttnn.reshape(q, (users, steps, KH, D), (users, steps, 32, D))
        k = ttnn.reshape(k, (users, steps, KH, D), (users, steps, 32, D))
        v = ttnn.reshape(v, (users, steps, VH, D), (users, steps, 32, D))
        g = ttnn.reshape(g[:, :, :rows, :], (users, steps, VH))
        beta = ttnn.reshape(beta[:, :, :rows, :], (users, steps, VH))
        o, _ = ttnn.transformer.fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=D**-0.5,
            initial_state=spec.ring[li],
            initial_state_block_idx=spec.ring_idx,
            output_per_token_state=True,
            compute_kernel_config=self.compute,
            exact_state=True,
        )  # o [users, steps, VH, D]
        o = ttnn.reshape(o, (rows * VH, D))
        zz = ttnn.reshape(proj[:, :, :rows, 2048:3072], (rows * VH, D))
        y = self._gated_norm(o, zz, rows)
        if rows < 32:
            y = ttnn.pad(y, [(0, 0), (0, 0), (0, 32 - rows), (0, 0)], 0.0)
        return self._out(y)

    # ------------------------------------------------------------------ prefill
    def prefill(self, x, state, *, slot, start, length, consts):
        """Prefill `length` tokens of one request. x [1, 1, T32, 2048] normalized input (rows >= length are
        padding). Updates conv/recurrent state of `slot` and returns the local out partial [1,1,T32,2048] fp32."""
        t32 = x.shape[-2]
        proj = self._project(x)
        xt = proj[:, :, :, 0:2048]
        s0, s1, s2 = state["conv"]
        if start == 0:
            hist = ttnn.zeros((1, 1, 3, 2048), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh)
        else:
            rows = [ttnn.to_layout(t[:, :, slot : slot + 1, :], ttnn.ROW_MAJOR_LAYOUT) for t in (s0, s1, s2)]
            hist = ttnn.concat(rows, dim=2)
        xp = ttnn.concat([hist, ttnn.to_layout(xt, ttnn.ROW_MAJOR_LAYOUT)], dim=2)  # [1,1,T32+3,2048] RM
        acc = None
        for j in range(4):
            tap = ttnn.to_layout(xp[:, :, j : j + t32, :], ttnn.TILE_LAYOUT)
            prod = ttnn.multiply(tap, self.taps[j], dtype=ttnn.float32)
            acc = prod if acc is None else ttnn.add(acc, prod)
        conv = ttnn.typecast(ttnn.silu(acc), ttnn.bfloat16)
        # new conv history = last 3 real tokens (xp rows length .. length+2)
        for j, dst in enumerate((s0, s1, s2)):
            row = ttnn.to_layout(xp[:, :, length + j : length + j + 1, :], ttnn.TILE_LAYOUT)
            write_row(dst, row, slot, consts)
        g, beta = self._gates(proj)
        if length < t32:
            valid = consts["row_valid"](length, t32)  # [1,1,T32,1] fp32
            g = ttnn.multiply(g, valid)
            beta = ttnn.multiply(beta, valid)
        view, index = self.slot_view(state, slot)
        if start == 0:
            init = consts["zero_state"]
        else:
            init = ttnn.reshape(view[index : index + 1], (1, VH, D, D))
        o, fs = ttnn.transformer.chunk_gated_delta_rule(
            ttnn.reshape(conv[:, :, :, 0:512], (1, t32, 512)),
            ttnn.reshape(conv[:, :, :, 512:1024], (1, t32, 512)),
            ttnn.reshape(conv[:, :, :, 1024:2048], (1, t32, 1024)),
            ttnn.reshape(g, (1, t32, VH)),
            ttnn.reshape(beta, (1, t32, VH)),
            scale=D**-0.5,
            initial_state=init,
            output_final_state=True,
            chunk_size=32,
            eye=consts["eye"],
            tril=consts["tril"],
            ones=consts["ones"],
            masks=consts["masks"],
            num_key_heads=KH,
            output_head_major=False,
            exact_state=True,
        )
        ttnn.fill_cache(view, fs, index)
        o = ttnn.to_layout(ttnn.reshape(o, (t32 * VH, D)), ttnn.TILE_LAYOUT)
        zz = ttnn.reshape(proj[:, :, :, 2048:3072], (t32 * VH, D))
        return self._out(self._gated_norm(o, zz, t32))


def write_row(dst, row, slot, consts):
    """dst[..., slot, :] = row[..., 0, :] for a [1, 1, 32, W] TILE tensor; exact (one-hot select)."""
    onehot, rest = consts["onehot"][slot]
    row = ttnn.repeat(row[:, :, 0:1, :], (1, 1, 32, 1)) if row.shape[-2] == 1 else row
    mixed = ttnn.add(ttnn.multiply(dst, rest), ttnn.multiply(row, onehot))
    ttnn.copy(mixed, dst)


def make_prefill_consts(mesh):
    rep = lambda t: ttnn.from_torch(
        t.contiguous(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    c = 32
    ii, jj = torch.arange(32).unsqueeze(1), torch.arange(32).unsqueeze(0)
    masks = torch.cat(
        [((ii < 16) & (jj < 16)).float(), ((ii >= 16) & (jj >= 16)).float(), ((ii >= 16) & (jj < 16)).float()], 1
    )
    consts = dict(
        eye=rep(torch.eye(c).reshape(1, 1, c, c)),
        tril=rep(torch.tril(torch.ones(c, c)).reshape(1, 1, c, c)),
        ones=rep(torch.ones(1, 1, c, c)),
        masks=rep(masks.reshape(1, 1, 32, 96)),
        zero_state=ttnn.zeros(
            (1, VH, D, D), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM
        ),
    )
    consts["onehot"] = []
    for s in range(32):
        e = torch.zeros(1, 1, 32, 1)
        e[0, 0, s, 0] = 1
        up = lambda t: ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        consts["onehot"].append((up(e), up(1 - e)))

    def row_valid(length, t32):
        # Transient per call: a tensor cached after decode-trace capture could alias trace scratch memory.
        m = (torch.arange(t32) < length).float().reshape(1, 1, t32, 1)
        return rep(m)

    consts["row_valid"] = row_valid
    return consts
