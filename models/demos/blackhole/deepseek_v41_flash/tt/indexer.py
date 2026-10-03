# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash decode-time lightning indexer of ONE index-source layer (4x8 Blackhole mesh, users on the mesh rows).

Reference (checkpoint ``model.py`` ``Indexer.forward``, one query token per user)::

    q   = fp4_sim(rope(wq_b(qr)))                        [32 heads, 128]   rope on the last 64 dims, per-32 fp4 (e2m1, e8m0 scale) simulation
    w   = weights_proj(x) * (128**-0.5 * 32**-0.5)        [32]
    s_t = sum_h relu(q_h . k_t) * w_h                     t over the N = (pos + 1) // ratio compressed entries (keys k_t: RoPE'd, fp4-simulated)
    out = the 512 best t (sorted by position)             [-1 where fewer than 512 entries exist]

Device mapping (``backend="fused"``):
    w   = linear(x, weights_proj / 64)                              [1,1,U,32]
    q   = linear(qr, wq_b)  -> heads -> RoPE (x*C + (x@P)*S)        [U,32,1,128]  (heads = Hi, one query row)
    s   = ttnn.experimental.indexer_score_dsa(q, K, w)              [U,1,1,T] bf16 row-major, K = key cache [U,1,T,128] (bf16 / bfp8_b tiles)
    ids = ttnn.experimental.topk_large_indices(s, k=512, valid_length_tensor=N)   [U,1,1,512] uint32, sorted by descending score

``valid_length_tensor`` is read on the device, so one captured trace serves every step while the cache fills; the score op always covers the
allocated length ``T`` (scalar runtime args are frozen in a trace), so capture one trace per ``T`` bucket of the context range.

Shared position: all users of a call share the valid length (the single ``valid_length_tensor``); users at different lengths need a per-user
additive mask on the scores first (``mask`` argument of ``select``).
"""

import torch

import ttnn

HEADS = 32
DIM = 128
ROPE = 64
TOPK = 512


def _pair_swap_matrix(n: int = DIM, rope: int = ROPE) -> torch.Tensor:
    """P with (x @ P)[2i] = -x[2i+1], (x @ P)[2i+1] = x[2i] on the last ``rope`` dims, 0 elsewhere (adjacent-pair rotation)."""
    P = torch.zeros(n, n)
    for i in range((n - rope) // 2, n // 2):
        P[2 * i + 1, 2 * i] = -1.0
        P[2 * i, 2 * i + 1] = 1.0
    return P


def rope_rows(freqs_cis: torch.Tensor, positions: torch.Tensor):
    """complex freqs [S, 32], positions [B] -> C, S [B, 128]: full-width tables (C = 1, S = 0 outside the rotated dims)."""
    f = freqs_cis[positions]
    cos, sin = f.real.repeat_interleave(2, dim=-1), f.imag.repeat_interleave(2, dim=-1)
    B = positions.shape[0]
    C, S = torch.ones(B, DIM), torch.zeros(B, DIM)
    C[:, DIM - ROPE :], S[:, DIM - ROPE :] = cos, sin
    return C, S


class DSV41DecodeIndexer:
    def __init__(
        self,
        mesh_device,
        w: dict,
        freqs_cis,
        users_per_row=4,
        n_alloc=2080,
        ratio=2,
        topk=TOPK,
        key_dtype=ttnn.bfloat16,
        q_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat8_b,
        backend="fused",
        k_chunk=128,
        head_group=0,
        fp4_q=False,
    ):
        """``w``: ``wq_b`` [4096, 1280] and ``weights_proj`` [32, 5120] (dequantised bf16 [out, in]); ``freqs_cis``: complex RoPE table of the layer.
        ``n_alloc``: allocated key length per user (multiple of 32, >= max entries + 32: row 0 of the query tile may only see keys <= n_alloc - 32).
        """
        assert n_alloc % 32 == 0 and topk % 16 == 0
        self.md, self.T, self.ratio, self.topk = mesh_device, users_per_row, ratio, topk
        self.rows, self.cols = tuple(mesh_device.shape)
        self.B = self.rows * self.T
        self.n_alloc, self.key_dtype, self.q_dtype, self.backend, self.fp4_q = (
            n_alloc,
            key_dtype,
            q_dtype,
            backend,
            fp4_q,
        )
        self.freqs_cis = freqs_cis
        md = mesh_device
        rep = ttnn.ReplicateTensorToMesh(md)
        up = lambda t, dtype=ttnn.bfloat16: ttnn.from_torch(
            t.contiguous(),
            device=md,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self._up = up
        scale = DIM**-0.5 * HEADS**-0.5
        self.wq_b = up(w["wq_b"].T.reshape(1, 1, -1, HEADS * DIM).float(), weight_dtype)
        self.wproj = up((w["weights_proj"].T.float() * scale).reshape(1, 1, -1, HEADS))
        self.P = up(_pair_swap_matrix().reshape(1, 1, DIM, DIM))
        self.ckc = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.cfg = ttnn.IndexerScoreProgramConfig(
            q_chunk_size=32, k_chunk_size=min(k_chunk, n_alloc), head_group_size=head_group
        )
        self.k_cache = None

    # ---- state ---------------------------------------------------------------------------------------------------------------
    def load_keys(self, index_k: torch.Tensor):
        """index_k [B, N, 128] (RoPE'd, fp4-simulated keys of every user, row-major user order) -> device cache [U, 1, n_alloc, 128] per mesh row."""
        B, N, _ = index_k.shape
        assert B == self.B and N + 32 <= self.n_alloc
        full = torch.zeros(B, 1, self.n_alloc, DIM)
        full[:, 0, :N] = index_k.float()
        self.k_cache = ttnn.from_torch(
            full,
            device=self.md,
            dtype=self.key_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(self.rows, self.cols)),
        )

    def step_inputs(self, positions: torch.Tensor, st=None):
        """positions [B] (decode position of every user, row-major). The valid length is shared: all users must have the same position."""
        p = int(positions[0])
        assert bool((positions == p).all()), "one valid_length_tensor per call: all users share the position"
        C, S = rope_rows(self.freqs_cis, positions)
        rs = ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(self.rows, self.cols))
        host = lambda t, dtype, layout: ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, mesh_mapper=rs)
        vals = {
            "C": host(C.reshape(self.B, 1, 1, DIM).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "S": host(S.reshape(self.B, 1, 1, DIM).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        }
        st = {} if st is None else st
        for k, v in vals.items():
            if k in st:
                ttnn.copy_host_to_device_tensor(v, st[k])
            else:
                st[k] = ttnn.to_device(v, self.md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        n_valid = max(
            (p + 1) // self.ratio, self.topk
        )  # topk_large_indices needs valid_length >= k (entries < topk: see docs)
        vl = ttnn.from_torch(
            torch.full((1, 1, 1, 1), n_valid, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.md),
        )
        if "valid" in st:
            ttnn.copy_host_to_device_tensor(vl, st["valid"])
        else:
            st["valid"] = ttnn.to_device(vl, self.md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return st

    # ---- stages --------------------------------------------------------------------------------------------------------------
    def _fp4_sim(self, q):
        """In-place-style fp4 (e2m1, per-32 e8m0 scale) quantise-dequantise of q [U,1,32,128] exactly like the reference ``fp4_act_quant``:
        scale = 2**ceil(log2(amax / 6)) per block of 32, value = round-to-grid(x / scale) * scale. The e2m1 grid {0,.5,1,1.5,2,3,4,6} is a
        round-half-even rounding of |y| / step with step 0.5 (|y| < 2), 1 (< 4), 2 (>= 4)."""
        U = self.T
        x = ttnn.reshape(ttnn.typecast(q, ttnn.float32), (U, 1, HEADS * 4, 32))  # row = head * 4 + block
        amax = ttnn.max(ttnn.abs(x), dim=-1, keepdim=True)
        scale = ttnn.exp2(ttnn.ceil(ttnn.log2(ttnn.multiply(ttnn.clamp(amax, min=6 * 2.0**-126), 1.0 / 6.0))))
        y = ttnn.clamp(ttnn.divide(x, scale), min=-6.0, max=6.0)
        mag = ttnn.abs(y)
        step = ttnn.add(ttnn.multiply(ttnn.ge(mag, 2.0), 0.5), ttnn.add(ttnn.multiply(ttnn.ge(mag, 4.0), 1.0), 0.5))
        r = ttnn.floor(ttnn.add(ttnn.divide(mag, step), 0.5))
        out = ttnn.multiply(ttnn.multiply(ttnn.multiply(r, step), ttnn.sign(y)), scale)
        return ttnn.typecast(ttnn.reshape(out, (U, 1, HEADS, DIM)), ttnn.bfloat16)

    def project(self, x, qr, st):
        """x [1,1,U,5120], qr [1,1,U,1280] (bf16 tiles, replicated over the columns) -> (q [U,32,1,128], w [U,1,1,32])."""
        U = self.T
        w = ttnn.linear(x, self.wproj, compute_kernel_config=self.ckc, core_grid=ttnn.CoreGrid(y=2, x=8))  # [1,1,U,32]
        q = ttnn.linear(
            qr, self.wq_b, compute_kernel_config=self.ckc, core_grid=ttnn.CoreGrid(y=2, x=8)
        )  # [1,1,U,4096]
        q = ttnn.reshape(q, (U, 1, HEADS, DIM))  # heads on the tile rows
        q = ttnn.addcmul(
            ttnn.multiply(q, st["C"]), ttnn.matmul(q, self.P, compute_kernel_config=self.ckc), st["S"]
        )  # RoPE on the last 64 dims
        if self.fp4_q:
            q = self._fp4_sim(q)
        if self.backend == "fused":
            q = ttnn.permute(q, (0, 2, 1, 3))  # [U,32,1,128]: heads = Hi
            q = ttnn.pad(
                q, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0
            )  # the score op needs Sq % 32 == 0: row 0 is the query, 31 padding rows
            if self.q_dtype != ttnn.bfloat16:
                q = ttnn.typecast(q, self.q_dtype)
        w = ttnn.reshape(w, (U, 1, 1, HEADS))
        if self.backend == "fused":
            w = ttnn.pad(w, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0)
        return q, w

    def score(self, q, w):
        """fused: list (one per user) of scores [1,1,32,n_alloc] bf16 row-major (row 0 is the query; slots >= the valid length are garbage and never
        ranked). ``indexer_score_dsa`` takes ONE user per call (k batch must be 1), so user u is a call with ``cache_batch_idx=u`` on the
        [U,1,T,128] key cache. matmul backend: one batched result [U,1,1,T]."""
        if self.backend == "fused":
            out = []
            for u in range(self.T):
                qu = ttnn.slice(q, [u, 0, 0, 0], [u + 1, HEADS, 32, DIM])
                wu = ttnn.slice(w, [u, 0, 0, 0], [u + 1, 1, 32, HEADS])
                out.append(
                    ttnn.experimental.indexer_score_dsa(
                        qu,
                        self.k_cache,
                        wu,
                        chunk_start_idx=self.n_alloc - 32,
                        kv_len=self.n_alloc,
                        program_config=self.cfg,
                        cache_batch_idx=u,
                    )
                )
            return out
        s = ttnn.matmul(
            q, self.k_cache, transpose_b=True, activation="relu", compute_kernel_config=self.ckc
        )  # [U,1,32,T]
        s = ttnn.matmul(w, s, compute_kernel_config=self.ckc)  # [U,1,1(32),T]
        return ttnn.to_layout(ttnn.slice(s, [0, 0, 0, 0], [self.T, 1, 1, self.n_alloc]), ttnn.ROW_MAJOR_LAYOUT)

    def select(self, scores, st):
        """top-``topk`` entry ids: uint32 [U,1,1,topk] (descending score, 0xFFFFFFFF where fewer valid entries), ONE ``topk_large_indices`` call
        for all users (rows are ranked in parallel, so the cost is that of one user): fused backend concatenates row 0 (the query row) of every
        user's score tile. ``st["valid"]``: valid length (device tensor)."""
        if self.backend == "fused":
            rows = [ttnn.slice(s, [0, 0, 0, 0], [1, 1, 1, self.n_alloc]) for s in scores]
            scores = rows[0] if len(rows) == 1 else ttnn.concat(rows, dim=0)  # [U,1,1,T] row-major
        return ttnn.experimental.topk_large_indices(scores, k=self.topk, valid_length_tensor=st["valid"])

    def forward(self, x, qr, st, return_scores=False):
        q, w = self.project(x, qr, st)
        s = self.score(q, w)
        ids = self.select(s, st)
        return (ids, s) if return_scores else ids
