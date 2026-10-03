# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash attention, decode, TP over heads (``tp_heads``), for window-only layers (ratio 0).

Mesh 4x8: rows = users (``users_per_row`` each), columns = TP. Per device (row r, column c):

    x_norm [1,1,T,5120] (replicated over columns)
      q  = rms_norm(x @ wq_a) @ wq_b[:, heads 8c..8c+7]     8 of 64 query heads, head_dim 512
      kv = rms_norm(x @ wkv)                                  1 KV head, K == V, replicated on every column
      RoPE on the last 64 dims of q and kv (adjacent-pair rotation)
      window cache <- kv ; SDPA decode (sink, window 128) -> o [T, 8, 512]
      inverse RoPE on o's last 64 dims
      o @ wo_a[group c] (4096 -> 1024) @ wo_b[rows of group c] (1024 -> 5120)  -> partial sums
      all-reduce over the 8 columns -> [1,1,T,5120] replicated

The 8 output groups of the checkpoint's grouped projection coincide with the 8 columns (group g = heads 8g..8g+7).
RoPE is done as ``x * cos + (x @ P) * sin`` on the 64-wide slice, P the pair-swap/sign matrix.
"""

import torch

import ttnn

HEAD_DIM = 512
ROPE_DIM = 64
N_HEADS = 64
N_GROUPS = 8
O_LORA = 1024
Q_LORA = 1280
DIM = 5120
WINDOW = 128
PAD_HEADS = 32  # local heads (8) padded to a tile for SDPA decode


def rope_tables(freqs_cis: torch.Tensor):
    """complex [S, 32] -> cos, sin [S, 64] with each value repeated for its adjacent pair."""
    return freqs_cis.real.repeat_interleave(2, dim=-1), freqs_cis.imag.repeat_interleave(2, dim=-1)


def pair_swap_matrix():
    """P with (x @ P)[2i] = -x[2i+1], (x @ P)[2i+1] = x[2i]."""
    P = torch.zeros(ROPE_DIM, ROPE_DIM)
    for i in range(ROPE_DIM // 2):
        P[2 * i + 1, 2 * i] = -1.0
        P[2 * i, 2 * i + 1] = 1.0
    return P


class DSV41Attention:
    def __init__(self, mesh_device, mesh_config, ccl_manager, w: dict, freqs_cis, users_per_row=4, max_seq=256):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl = ccl_manager
        self.rows, self.cols = tuple(mesh_device.shape)
        assert self.cols == N_GROUPS
        self.T = users_per_row
        self.max_seq = max_seq
        self.scale = HEAD_DIM**-0.5
        md = mesh_device
        bf = ttnn.bfloat16
        shape = (self.rows, self.cols)

        def up(t, dtype=bf, mapper=None, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t.contiguous(),
                device=md,
                dtype=dtype,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper if mapper is not None else ttnn.ReplicateTensorToMesh(md),
            )

        col_shard = lambda dim: ttnn.ShardTensor2dMesh(md, dims=(None, dim), mesh_shape=shape)
        # replicated projections: weights stored [in, out]
        self.wq_a = up(w["wq_a"].T.reshape(1, 1, DIM, Q_LORA))
        self.wkv = up(w["wkv"].T.reshape(1, 1, DIM, HEAD_DIM))
        self.q_norm = up(w["q_norm"].reshape(1, 1, Q_LORA // 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.kv_norm = up(w["kv_norm"].reshape(1, 1, HEAD_DIM // 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
        # column-parallel over heads: wq_b^T [1280, 64*512], local [1280, 8*512]
        self.wq_b = up(w["wq_b"].T.reshape(1, 1, Q_LORA, N_HEADS * HEAD_DIM), mapper=col_shard(3))
        # grouped o-projection: group g = column g
        wo_a = w["wo_a"].reshape(N_GROUPS, O_LORA, N_HEADS * HEAD_DIM // N_GROUPS)  # [g, r, d]
        self.wo_a = up(
            wo_a.permute(2, 0, 1).reshape(1, 1, -1, N_GROUPS * O_LORA), mapper=col_shard(3)
        )  # [4096, 8*1024]
        self.wo_b = up(w["wo_b"].T.reshape(1, 1, N_GROUPS * O_LORA, DIM), mapper=col_shard(2))  # [8*1024, 5120]
        # attention sink, pre-divided by the softmax scale (the kernel multiplies sinks by `scale`)
        sink = (w["attn_sink"].float() / self.scale).reshape(N_GROUPS, N_HEADS // N_GROUPS)
        sinks = torch.zeros(N_GROUPS, PAD_HEADS, 32)
        sinks[:, : N_HEADS // N_GROUPS, 0] = sink
        self.sinks = up(
            sinks.reshape(N_GROUPS * PAD_HEADS, 32), mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 0), mesh_shape=shape)
        )  # 2-D [32, 32] per device
        self.P = up(pair_swap_matrix().reshape(1, 1, ROPE_DIM, ROPE_DIM))
        self.cos_tab, self.sin_tab = rope_tables(freqs_cis)
        # window KV cache [T, 1, max_seq, 512]; K and V are the same vector, one tensor serves both
        self.cache = up(torch.zeros(self.T, 1, max_seq, HEAD_DIM))
        self.ckc = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.eps = 1e-20

    # ---- step inputs (positions of the users in each row) -------------------------------------------
    def step_inputs(self, positions: torch.Tensor):
        """positions: [rows * T] int, global user order (row-major). Returns device tensors for forward()."""
        md = self.mesh_device
        rows_shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(self.rows, self.cols))
        pos = ttnn.from_torch(positions.to(torch.int32), device=md, dtype=ttnn.int32, mesh_mapper=rows_shard)
        cos = self.cos_tab[positions].reshape(-1, 1, 1, ROPE_DIM)  # [rows*T, 1, 1, 64]
        sin = self.sin_tab[positions].reshape(-1, 1, 1, ROPE_DIM)
        up = lambda t: ttnn.from_torch(
            t.reshape(-1, 1, 1, ROPE_DIM).to(torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rows_shard,
        )
        return {"pos": pos, "cos": up(cos), "sin": up(sin), "nsin": up(-sin)}

    def load_window(self, kv_rows: torch.Tensor):
        """Host cache seed: kv_rows [rows*T, S, 512] (positions 0..S-1) -> every row's cache slice."""
        rows_shard = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, None), mesh_shape=(self.rows, self.cols))
        S = kv_rows.shape[1]
        full = torch.zeros(self.rows * self.T, 1, self.max_seq, HEAD_DIM)
        full[:, 0, :S] = kv_rows
        self.cache = ttnn.from_torch(
            full,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rows_shard,
        )

    # ---- helpers --------------------------------------------------------------------------------------
    def _rope(self, x, cos, sin):
        """x [..., 512]: rotate the last 64 dims; cos/sin broadcast against [1?, T, H, 64]."""
        lead = list(x.shape)[:-1]
        nope = ttnn.slice(x, [0] * len(lead) + [0], lead + [HEAD_DIM - ROPE_DIM])
        rp = ttnn.slice(x, [0] * len(lead) + [HEAD_DIM - ROPE_DIM], lead + [HEAD_DIM])
        rot = ttnn.add(ttnn.multiply(rp, cos), ttnn.multiply(ttnn.matmul(rp, self.P), sin))
        return ttnn.concat([nope, rot], dim=-1)

    def _qkv(self, x, st):
        """x [1,1,T,5120] bf16 -> q [1,T,32,512] (RoPE'd, heads padded), kv [1,T,1,512] (RoPE'd), tile layout."""
        T, H = self.T, N_HEADS // N_GROUPS
        cos, sin = st["cos"].reshape(1, T, 1, ROPE_DIM), st["sin"].reshape(1, T, 1, ROPE_DIM)
        qr = ttnn.rms_norm(
            ttnn.linear(x, self.wq_a, compute_kernel_config=self.ckc), weight=self.q_norm, epsilon=self.eps
        )
        q = ttnn.linear(qr, self.wq_b, compute_kernel_config=self.ckc)  # [1,1,T,8*512]
        q = ttnn.reshape(ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT), [1, T, H, HEAD_DIM])
        q = ttnn.pad(ttnn.to_layout(q, ttnn.TILE_LAYOUT), [(0, 0), (0, 0), (0, PAD_HEADS - H), (0, 0)], 0.0)
        q = self._rope(q, cos, sin)
        kv = ttnn.rms_norm(
            ttnn.linear(x, self.wkv, compute_kernel_config=self.ckc), weight=self.kv_norm, epsilon=self.eps
        )
        kv = ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(kv, ttnn.ROW_MAJOR_LAYOUT), [1, T, 1, HEAD_DIM]), ttnn.TILE_LAYOUT
        )
        kv = self._rope(kv, cos, sin)
        return q, kv

    def _write_cache(self, cache, row, idx):
        """cache [T,1,L,512] <- row [1,T,1,512] at per-user index idx (int32 [T])."""
        T = self.T
        pad = ttnn.pad(row, [(0, 0), (0, 0), (0, PAD_HEADS - 1), (0, 0)], 0.0)
        cfg = ttnn.create_sharded_memory_config(
            shape=(PAD_HEADS, HEAD_DIM),
            core_grid=ttnn.num_cores_to_corerangeset(T, ttnn.CoreCoord(8, 8), row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        ttnn.experimental.paged_update_cache(
            cache, ttnn.to_memory_config(pad, cfg), update_idxs_tensor=idx, page_table=None
        )

    def _finish(self, o, st):
        """o [1,T,32,512] attention output -> inverse RoPE, grouped output projection, all-reduce -> [1,1,T,5120]."""
        T, H = self.T, N_HEADS // N_GROUPS
        o = ttnn.slice(o, [0, 0, 0, 0], [1, T, H, HEAD_DIM])
        o = self._rope(o, st["cos"].reshape(1, T, 1, ROPE_DIM), st["nsin"].reshape(1, T, 1, ROPE_DIM))  # inverse
        o = ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT), [1, 1, T, H * HEAD_DIM]), ttnn.TILE_LAYOUT
        )
        part = ttnn.linear(
            ttnn.linear(o, self.wo_a, compute_kernel_config=self.ckc), self.wo_b, compute_kernel_config=self.ckc
        )
        return self.mesh_config.allreduce(part, self.ccl, axis=1)

    def forward(self, x, st):
        """x: [1,1,T,5120] bf16 tile, normed attention input. st: step_inputs(). -> [1,1,T,5120] replicated."""
        T = self.T
        q, kv = self._qkv(x, st)
        self._write_cache(self.cache, kv, st["pos"])
        grid = ttnn.num_cores_to_corerangeset(T, ttnn.CoreCoord(8, 8), row_wise=True).bounding_box().grid_size()
        prog = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            q_chunk_size=0,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        o = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            self.cache,
            self.cache,
            cur_pos_tensor=st["pos"],
            sliding_window_size=WINDOW,
            attention_sink=self.sinks,
            scale=self.scale,
            program_config=prog,
            compute_kernel_config=self.ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, T, 32, 512]
        return self._finish(o, st)


class DSV41CompressedAttention(DSV41Attention):
    """Layers with ``compress_ratio > 0`` that own a compressor (kv source layers): window ring + compressed cache.

    Attention is composite (the SDPA decode op cannot join a 128-slot ring with a separate compressed cache):

        scores = [ q @ K_window^T , q @ K_comp^T ] * scale + mask ;  softmax over [scores, sink] ; o = P @ [K_window; K_comp]

    Valid only while every compressed position is selected, i.e. compress_len <= index_topk (512), so the
    indexer's top-k is the identity; ``step_inputs`` asserts this. All users must share the step position
    (the compressor's group-complete decision is made on the host).
    """

    def __init__(
        self,
        mesh_device,
        mesh_config,
        ccl_manager,
        w,
        freqs_cis,
        ratio,
        comp_w,
        users_per_row=4,
        max_comp=128,
        index_topk=512,
        source=None,
    ):
        """``comp_w`` is None and ``source`` the owning ``DSV41CompressedAttention`` for layers that only READ the
        compressed cache of the last kv-source layer (they keep their own window cache)."""
        super().__init__(
            mesh_device, mesh_config, ccl_manager, w, freqs_cis, users_per_row=users_per_row, max_seq=WINDOW
        )
        assert ratio in (1, 2) and max_comp % 32 == 0
        self.ratio, self.max_comp, self.index_topk = ratio, max_comp, index_topk
        self.source = source
        if source is not None:
            assert comp_w is None and source.ratio == ratio and source.max_comp == max_comp
        md, T = mesh_device, users_per_row
        rep = ttnn.ReplicateTensorToMesh(md)
        up = lambda t, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT, mp=None: ttnn.from_torch(
            t.contiguous(),
            device=md,
            dtype=dt,
            layout=lay,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mp if mp is not None else rep,
        )
        if comp_w is not None:
            self.c_wkv = up(comp_w["wkv"].T.reshape(1, 1, DIM, HEAD_DIM))
            self.c_norm = up(comp_w["norm"].reshape(1, 1, HEAD_DIM // 32, 32), lay=ttnn.ROW_MAJOR_LAYOUT)
            if ratio > 1:
                self.c_wgate = up(comp_w["wgate"].T.reshape(1, 1, DIM, HEAD_DIM))
            self.comp_cache = up(torch.zeros(T, 1, max_comp, HEAD_DIM))
        self.kv_state = [None] * ratio
        self.score_state = [None] * ratio
        # sink as a score column: raw sink in column 0, -1e30 elsewhere, one [T, 1, 32, 32] block per device
        sink = w["attn_sink"].float().reshape(N_GROUPS, N_HEADS // N_GROUPS)
        blk = torch.full((N_GROUPS, PAD_HEADS, 32), -1e30)
        blk[:, : N_HEADS // N_GROUPS, 0] = sink
        col = ttnn.ShardTensor2dMesh(md, dims=(None, 0), mesh_shape=(self.rows, self.cols))
        sb = up(blk.reshape(N_GROUPS * PAD_HEADS, 32), mp=col)
        self.sink_blk = ttnn.repeat(ttnn.reshape(sb, [1, 1, PAD_HEADS, 32]), [T, 1, 1, 1])

    # ---- state seeding / step inputs ---------------------------------------------------------------------
    def load_state(self, window_kv, comp_kv, kv_state, score_state):
        """window_kv [B,128,512] ring, comp_kv [B,Lc,512], kv_state/score_state [B,ratio,512] (reference buffers)."""
        md = self.mesh_device
        rs = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(self.rows, self.cols))
        up = lambda t, dt, lay: ttnn.from_torch(
            t.contiguous(), device=md, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rs
        )
        B = window_kv.shape[0]
        self.cache = up(window_kv.reshape(B, 1, WINDOW, HEAD_DIM).float(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        if self.source is not None:  # reads the owner's compressed cache; nothing else to seed
            return
        comp = torch.zeros(B, 1, self.max_comp, HEAD_DIM)
        comp[:, 0, : comp_kv.shape[1]] = comp_kv.float()
        self.comp_cache = up(comp, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        if self.ratio > 1:
            for i in range(self.ratio):
                st_up = lambda t: ttnn.from_torch(
                    t.reshape(1, 1, B, HEAD_DIM).float().contiguous(),
                    device=md,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(self.rows, self.cols)),
                )
                self.kv_state[i], self.score_state[i] = st_up(kv_state[:, i]), st_up(score_state[:, i])

    def step_inputs(self, positions: torch.Tensor):
        st = super().step_inputs(positions)
        p = int(positions[0])
        assert bool((positions == p).all()), "compressed attention assumes all users share the step position"
        r, md = self.ratio, self.mesh_device
        comp_len = (p + 1) // r
        assert comp_len <= min(self.max_comp, self.index_topk), "compressed length exceeds the indexer-free regime"
        rs = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(self.rows, self.cols))
        B = self.rows * self.T
        i32 = lambda v: ttnn.from_torch(
            torch.full((B,), v, dtype=torch.int32), device=md, dtype=ttnn.int32, mesh_mapper=rs
        )
        st["pos_ring"], st["comp_idx"] = i32(p % WINDOW), i32(p // r)
        g = max(p + 1 - r, 0)  # RoPE position of a freshly pooled latent (first token of its group)
        mk = lambda t: ttnn.from_torch(
            t.reshape(1, 1, 1, ROPE_DIM).repeat(B, 1, 1, 1).to(torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rs,
        )
        st["cos_g"], st["sin_g"] = mk(self.cos_tab[g]), mk(self.sin_tab[g])
        wm = torch.zeros(B, 1, 1, WINDOW)
        wm[..., p + 1 :] = -1e9  # window slots not filled yet (ring not wrapped)
        cm = torch.zeros(B, 1, 1, self.max_comp)
        cm[..., comp_len:] = -1e9
        mm = lambda t: ttnn.from_torch(
            t.to(torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rs,
        )
        st["wmask"], st["cmask"] = mm(wm), mm(cm)
        st["slot"], st["complete"] = p % r, (p + 1) % r == 0
        return st

    # ---- compressor --------------------------------------------------------------------------------------
    def _compress_step(self, x, st):
        """One decode token through the compressor -> [1,T,1,512] RoPE'd latent if a group completes, else None."""
        T = self.T
        if self.ratio == 1:
            lat = ttnn.rms_norm(
                ttnn.linear(x, self.c_wkv, compute_kernel_config=self.ckc), weight=self.c_norm, epsilon=self.eps
            )
        else:
            kv_new = ttnn.linear(x, self.c_wkv, dtype=ttnn.float32, compute_kernel_config=self.ckc)
            sc_new = ttnn.linear(x, self.c_wgate, dtype=ttnn.float32, compute_kernel_config=self.ckc)
            self.kv_state[st["slot"]], self.score_state[st["slot"]] = kv_new, sc_new
            if not st["complete"]:
                return None
            m = self.score_state[0]
            for s in self.score_state[1:]:
                m = ttnn.maximum(m, s)
            e = [ttnn.exp(ttnn.subtract(s, m)) for s in self.score_state]
            z = e[0]
            for t in e[1:]:
                z = ttnn.add(z, t)
            pooled = ttnn.multiply(self.kv_state[0], e[0])
            for kvs, t in zip(self.kv_state[1:], e[1:]):
                pooled = ttnn.add(pooled, ttnn.multiply(kvs, t))
            pooled = ttnn.divide(pooled, z)
            lat = ttnn.rms_norm(ttnn.typecast(pooled, ttnn.bfloat16), weight=self.c_norm, epsilon=self.eps)
        lat = ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(lat, ttnn.ROW_MAJOR_LAYOUT), [1, T, 1, HEAD_DIM]), ttnn.TILE_LAYOUT
        )
        return self._rope(lat, st["cos_g"].reshape(1, T, 1, ROPE_DIM), st["sin_g"].reshape(1, T, 1, ROPE_DIM))

    # ---- attention ---------------------------------------------------------------------------------------
    def _attend(self, q, st):
        T = self.T
        qb = ttnn.permute(q, (1, 0, 2, 3))  # [T,1,32,512]
        sw = ttnn.matmul(qb, self.cache, transpose_b=True, compute_kernel_config=self.ckc)  # [T,1,32,128]
        sc = ttnn.matmul(qb, self.comp_cache, transpose_b=True, compute_kernel_config=self.ckc)  # [T,1,32,Mc]
        sw = ttnn.add(ttnn.multiply(sw, self.scale), st["wmask"])
        sc = ttnn.add(ttnn.multiply(sc, self.scale), st["cmask"])
        p = ttnn.softmax(ttnn.concat([sw, sc, self.sink_blk], dim=-1), dim=-1, numeric_stable=True)
        pw = ttnn.slice(p, [0, 0, 0, 0], [T, 1, PAD_HEADS, WINDOW])
        pc = ttnn.slice(p, [0, 0, 0, WINDOW], [T, 1, PAD_HEADS, WINDOW + self.max_comp])
        o = ttnn.add(
            ttnn.matmul(pw, self.cache, compute_kernel_config=self.ckc),
            ttnn.matmul(pc, self.comp_cache, compute_kernel_config=self.ckc),
        )
        return ttnn.permute(o, (1, 0, 2, 3))  # [1,T,32,512]

    def forward(self, x, st):
        q, kv = self._qkv(x, st)
        self._write_cache(self.cache, kv, st["pos_ring"])
        if self.source is None:
            lat = self._compress_step(x, st)
            if lat is not None:
                self._write_cache(self.comp_cache, lat, st["comp_idx"])
        else:
            self.comp_cache = self.source.comp_cache  # the owner has written this step's latent already
        return self._finish(self._attend(q, st), st)
