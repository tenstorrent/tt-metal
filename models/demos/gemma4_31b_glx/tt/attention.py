# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 sliding-window attention on the SP8 x TP4 Galaxy, one layer, over the served ring KV cache.

Adapted from models/demos/gemma4_d_p/tt/attention (ring_prefill.py, operations.py, weights.py) and tt/ccl.py.

Per chunk (C = 5120 tokens, L = C / SP = 640 rows per SP row, 160 per chip between blocks):
    all_gather rows over TP -> [1, 1, 640, 5376]
    fused QKV linear (column-parallel, per TP column [q 8 heads | k 4 heads | v 4 heads] x 256)
    nlp_create_qkv_heads -> q [1, 8, 640, 256], k / v [1, 4, 640, 256]
    q_norm(w), k_norm(w), v_norm (no weight), per head over 256, in fp32 (ATTN_HEAD_NORM)
    RoPE (theta 1e4, all 256 dims) on q and k, adjacent-pair lane order, in fp32 (ATTN_ROPE: elementwise or
    rotary_embedding_llama)
    update_padded_kv_cache: k, v (fp32, ATTN_KV_FP32_WRITE) typecast to bfp8 into the served cache
    ring_joint SDPA over the cache, causal, window 1024, scale 1.0 (CP ring along mesh axis 0)
    nlp_concat_heads, o_proj (row-parallel), reduce_scatter rows over TP -> [1, 1, 160, 5376]

Q/K weight rows and the q/k norm weights are permuted from the checkpoint's NeoX (rotate-half) lane order to
adjacent-pair order (sliding_kv_indices), so RoPE is a pair rotation and K lands in the cache in the served
order (serving_contract.md: sliding K channels in adjacent-pair order, V in HF order).

The served cache per chip is [slots, 4 heads, max_seq / SP, 256], block-cyclic: chunk k's 640 rows on SP row r are
positions k * C + r * L + [0, L). Positions and the cache-write offset of every chunk are device tensors built at load
time; the forward only selects them by chunk index.
"""

import torch

import ttnn

from .settings import settings

TILE = ttnn.TILE_SIZE
SP_AXIS, TP_AXIS = 0, 1


def sliding_kv_indices(head_dim: int) -> torch.Tensor:
    """NeoX lanes in adjacent-pair order: [0, h, 1, h + 1, ...], h = head_dim / 2."""
    half = head_dim // 2
    return torch.stack((torch.arange(half), torch.arange(half, head_dim)), dim=1).reshape(-1)


def rot_transformation_mat(size: int = TILE) -> torch.Tensor:
    """The adjacent-pair rotation x -> (-x1, x0, -x3, x2, ...) as x @ m: rotary_embedding_llama's one tile, or the
    full head_dim for the elementwise RoPE."""
    m = torch.zeros(1, 1, size, size)
    m[..., torch.arange(0, size, 2), torch.arange(1, size, 2)] = 1
    m[..., torch.arange(1, size, 2), torch.arange(0, size, 2)] = -1
    return m


def sliding_rope_tables(cfg: dict, max_seq: int):
    """cos / sin [max_seq, head_dim] (fp32, lanes in adjacent-pair order) of the sliding layers' RoPE."""
    d = cfg["head_dim"]
    theta = cfg["rope_parameters"]["sliding_attention"]["rope_theta"]
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.int64).float() / d))
    freqs = torch.arange(max_seq, dtype=torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)[:, sliding_kv_indices(d)]
    return emb.cos(), emb.sin()


def _compute_config(mesh, fidelity, fp32_acc):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=False,
    )


def matmul_grid():
    """MATMUL_CORE_GRID as a linear's core_grid (None: ttnn's default)."""
    g = settings.get("MATMUL_CORE_GRID")
    return None if g == "auto" else ttnn.CoreGrid(y=int(g.split("x")[0]), x=int(g.split("x")[1]))


def _ring_cache_memory_config(mesh, width):
    """Served-cache layout: one 32-token record per round-robin DRAM shard over all banks."""
    banks = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(mesh.dram_grid_size().x)]
    )
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[1, 1, TILE, width],
            grid=banks,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


def _allocate_ring_cache(mesh, shape, width):
    """A zeroed, replicated-topology bfp8 cache, allocated on device (no host staging tensor)."""
    from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill

    cache = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh, _ring_cache_memory_config(mesh, width)
    )
    DRAMZeroFill.op(cache)
    dist = ttnn.MeshShape(*tuple(mesh.shape))
    coords = [ttnn.MeshCoordinate([c[i] for i in range(c.dims())]) for c in ttnn.MeshCoordinateRange(dist)]
    cache.update_tensor_topology(
        ttnn.TensorTopology(dist, [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], coords)
    )
    return cache


class SlidingAttention:
    def __init__(self, mesh, w: dict, cfg: dict, max_seq: int, chunk: int, chunk_tables: bool = True):
        """w: checkpoint tensors q_proj, k_proj, v_proj, o_proj, q_norm, k_norm (HF names without the prefix).
        cfg: the text config. max_seq / chunk: the served geometry (cache capacity and chunk size).
        chunk_tables: build this layer's own RoPE tables and per-chunk positions / offsets (chunk-aligned starts,
        the component test). The device model passes False: it hands every call a ChunkMeta (tt/model.py) with the
        chunk's RoPE values and on-device scalars for any tile-aligned start."""
        self.mesh = mesh
        sp, tp = tuple(mesh.shape)
        self.sp, self.tp = sp, tp
        self.d = d = cfg["head_dim"]
        self.n_q, self.n_kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.n_q_local, self.n_kv_local = self.n_q // tp, self.n_kv // tp
        self.eps = cfg["rms_norm_eps"]
        self.window = cfg["sliding_window"]
        self.chunk, self.max_seq = chunk, max_seq
        assert max_seq % chunk == 0 and chunk % (sp * tp * TILE) == 0, (max_seq, chunk)
        self.slab = chunk // sp

        replicate = ttnn.ReplicateTensorToMesh(mesh)
        order = sliding_kv_indices(d)

        # Fused QKV, column-parallel: TP column t holds [q heads 8t.. | k heads 4t.. | v heads 4t..].
        hidden = w["q_proj"].shape[-1]
        q = w["q_proj"].reshape(self.n_q, d, hidden)[:, order].reshape(self.n_q, d, hidden)
        k = w["k_proj"].reshape(self.n_kv, d, hidden)[:, order]
        v = w["v_proj"].reshape(self.n_kv, d, hidden)
        cols = []
        for t in range(tp):
            qs = slice(t * self.n_q_local, (t + 1) * self.n_q_local)
            ks = slice(t * self.n_kv_local, (t + 1) * self.n_kv_local)
            cols += [q[qs].reshape(-1, hidden), k[ks].reshape(-1, hidden), v[ks].reshape(-1, hidden)]
        wqkv = torch.cat(cols, dim=0).transpose(0, 1).contiguous()  # [hidden, tp * local_qkv]
        self.wqkv = ttnn.from_torch(
            wqkv,
            device=mesh,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, (sp, tp), dims=(None, -1)),
        )
        # o_proj^T [n_q * d, hidden], row-parallel: TP column t holds the rows of its q heads.
        self.wo = ttnn.from_torch(
            w["o_proj"].transpose(0, 1).contiguous(),
            device=mesh,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, (sp, tp), dims=(None, -2)),
        )

        self.head_norm = settings.get("ATTN_HEAD_NORM")
        # K / V reach the cache from fp32: the device's bf16 -> bfp8 conversion rounds row norms up (~+0.3 %).
        self.kv_dtype = ttnn.float32 if settings.get("ATTN_KV_FP32_WRITE") else ttnn.bfloat16
        self.rope = settings.get("ATTN_ROPE")

        def norm_weight(x):
            fp32 = self.head_norm == "fp32"
            return ttnn.from_torch(
                x[order].reshape(1, 1, 1, d) if fp32 else x[order].reshape(1, 1, d // TILE, TILE).to(torch.bfloat16),
                device=mesh,
                dtype=ttnn.float32 if fp32 else ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT if fp32 else ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
            )

        self.q_norm_w, self.k_norm_w = norm_weight(w["q_norm"]), norm_weight(w["k_norm"])

        # RoPE tables [max_seq, d], lanes in adjacent-pair order, looked up per chunk with ttnn.embedding.
        if chunk_tables:
            self.cos, self.sin = (
                ttnn.from_torch(
                    t.to(torch.bfloat16),  # ttnn.embedding takes bf16 tables only
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=replicate,
                )
                for t in sliding_rope_tables(cfg, max_seq)
            )
        self.trans_mat = ttnn.from_torch(
            rot_transformation_mat(d if self.rope == "elementwise" else TILE),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )

        # Per chunk index: the SP-sharded absolute positions ([1, L] per chip, SP row r: k*C + r*L + [0, L)) and the
        # cache-write offset (kv_actual_global = k*C, a replicated [1, 1, 1, 1] uint32 scalar).
        def scalar(value):
            return ttnn.from_torch(
                torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
                device=mesh,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
            )

        n_chunks = max_seq // chunk if chunk_tables else 0
        self.positions = [
            ttnn.from_torch(
                torch.arange(c * chunk, (c + 1) * chunk, dtype=torch.int64).reshape(1, chunk),
                device=mesh,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, (sp, tp), dims=(1, None)),
            )
            for c in range(n_chunks)
        ]
        self.kv_actual = [scalar(c * chunk) for c in range(n_chunks)]
        self.slot = scalar(0)

        # The served cache, one layer: [slots, n_kv_local, max_seq / SP, d] bfp8 K and V.
        self.slots = settings.get("KV_SLOTS")
        shape = [self.slots, self.n_kv_local, max_seq // sp, d]
        self.cache_k = _allocate_ring_cache(mesh, shape, d)
        self.cache_v = _allocate_ring_cache(mesh, shape, d)

        # Ring SDPA: CCL workers on the last compute column, SDPA on the rest.
        grid = mesh.compute_with_storage_grid_size()
        all_cores = ttnn.num_cores_to_corerangeset(grid.x * grid.y, grid, row_wise=True)
        self.ring_sems = [ttnn.create_global_semaphore(mesh, all_cores, 0) for _ in range(3)]
        self.ccl_offset = ttnn.CoreCoord(grid.x - 1, 0)
        q_chunk, k_chunk = settings.get("ATTN_SDPA_Q_CHUNK"), settings.get("ATTN_SDPA_K_CHUNK")
        sdpa_cores = (grid.x - 1) * grid.y
        # Split K three ways when the cores hold three bands of every (head, Q chunk) unit (gemma4_d_p).
        k_splits = settings.get("ATTN_SDPA_K_SPLITS")
        if k_splits == "auto":
            k_splits = 3 if self.n_q_local * -(-self.slab // q_chunk) * 3 <= sdpa_cores else 1
        self.k_splits = int(k_splits)
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
            max_k_splits=self.k_splits,
        )
        # Compact halo receive buffers: the window rounded up to whole k chunks (a full-capacity one is rejected).
        # A chunk whose start is not on a slab boundary packs Q from two ring groups on one device: the op then needs
        # two predecessor-halo slots (ring_joint_sdpa_device_operation.cpp, chunked_q_wraps).
        halo = max(-(-(self.window - 1) // k_chunk) * k_chunk, TILE) * (1 if chunk_tables else 2)
        self.halo_k, self.halo_v = (
            ttnn.zeros(
                [1, self.n_kv_local, halo, d],
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=c.memory_config(),
            )
            for c in (self.cache_k, self.cache_v)
        )
        self.num_links = settings.get("ATTN_RING_NUM_LINKS")

        self.proj_cfg = _compute_config(mesh, settings.get("ATTN_PROJ_FIDELITY"), settings.get("ATTN_PROJ_FP32_ACC"))
        self.grid = matmul_grid()
        self.sdpa_cfg = _compute_config(mesh, settings.get("ATTN_SDPA_FIDELITY"), settings.get("ATTN_SDPA_FP32_ACC"))
        self.norm_cfg = _compute_config(mesh, "HiFi4", True)

    def chunk_index(self, start: int) -> int:
        assert start % self.chunk == 0 and 0 <= start < self.max_seq, start
        return start // self.chunk

    def write_kv(self, k, v, start: int, meta=None):
        """Append one chunk's K and V ([1, n_kv_local, L, d] per chip, SP row r holding its slab) to slot 0. fp32 (or
        bfp8) inputs: the device's bf16 -> bfp8 typecast rounds up on average (row norms +0.28 %), fp32 -> bfp8 does
        not. meta (ChunkMeta): its on-device slot, start and actual_end; rows from ceil32(actual_end) on are not
        written."""
        for cache, x in ((self.cache_k, k), (self.cache_v, v)):
            xb = ttnn.typecast(x, ttnn.bfloat8_b) if x.dtype != ttnn.bfloat8_b else x
            if meta is None:
                ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                    cache=cache,
                    input=xb,
                    slot_idx=self.slot,
                    kv_actual_global=self.kv_actual[self.chunk_index(start)],
                    layer_idx=0,
                    num_layers=1,
                    cluster_axis=SP_AXIS,
                )
            else:
                ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                    cache=cache,
                    input=xb,
                    slot_idx=meta.slot,
                    kv_actual_global=meta.kv_actual,
                    layer_idx=0,
                    num_layers=1,
                    cluster_axis=SP_AXIS,
                    valid_global=meta.valid,
                )
            if xb is not x:
                ttnn.deallocate(xb)

    def load_rows(self, k, v, start: int, end: int):
        """Harness / state loading (not the forward): chunk-aligned start, rows [start, ceil32(end)) of slot 0 from
        k / v laid out as write_kv's (host scalars)."""
        for cache, x in ((self.cache_k, k), (self.cache_v, v)):
            xb = ttnn.typecast(x, ttnn.bfloat8_b) if x.dtype != ttnn.bfloat8_b else x
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache=cache,
                input=xb,
                slot_idx=0,
                layer_idx=0,
                num_layers=1,
                kv_actual_global=start,
                cluster_axis=SP_AXIS,
                valid_global=end,
            )
            if xb is not x:
                ttnn.deallocate(xb)

    def _head_norm(self, x, weight, dtype=ttnn.bfloat16):
        """Per-head RMS norm over d, x [1, h, L, d]. ATTN_HEAD_NORM fp32: x * rsqrt(mean(x^2) + eps) * w with every
        op in fp32 (weight [1, 1, 1, d] fp32 TILE); rms_norm: ttnn.rms_norm (weight ROW_MAJOR [1, 1, d / 32, 32])."""
        if self.head_norm == "fp32":
            x32 = ttnn.typecast(x, ttnn.float32)
            sq = ttnn.multiply(x32, x32, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ms = ttnn.mean(
                sq, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.norm_cfg
            )
            ttnn.deallocate(sq)
            r = ttnn.rsqrt(ttnn.add(ms, self.eps, dtype=ttnn.float32), fast_and_approximate_mode=False)
            ttnn.deallocate(ms)
            y = ttnn.multiply(x32, r, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for t in (x32, r):
                ttnn.deallocate(t)
            if weight is not None:
                yw = ttnn.multiply(y, weight, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(y)
                y = yw
            if dtype != ttnn.float32:
                yd = ttnn.typecast(y, dtype)
                ttnn.deallocate(y)
                y = yd
            return y
        _, h, s, d = x.shape
        y = ttnn.rms_norm(
            ttnn.reshape(x, (1, 1, h * s, d)),
            weight=weight,
            epsilon=self.eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.norm_cfg,
        )
        return ttnn.reshape(y, (1, h, s, d))

    def _rope(self, x, cos, sin, dtype=ttnn.bfloat16):
        """x [1, h, L, d] (fp32 for elementwise) -> dtype (elementwise only; llama returns bf16)."""
        if self.rope == "llama":
            return ttnn.experimental.rotary_embedding_llama(
                x, cos, sin, self.trans_mat, is_decode_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        # x * cos + (x @ R) * sin in fp32; R is the exact +-1 pair rotation.
        rot = ttnn.matmul(
            x,
            self.trans_mat,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.norm_cfg,
            core_grid=self.grid,
        )
        xc = ttnn.multiply(x, cos, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rs = ttnn.multiply(rot, sin, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(rot)
        y = ttnn.add(xc, rs, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(xc)
        ttnn.deallocate(rs)
        return y

    def __call__(self, x, start: int, meta=None):
        """x: the residual chunk [1, 1, L / TP, hidden] bf16 per chip. Returns the same layout.
        meta: the device model's ChunkMeta (any tile-aligned start, pad rows); None: chunk-aligned start, own tables."""
        xg = ttnn.all_gather(x, dim=2, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qkv = ttnn.linear(
            xg,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.proj_cfg,
            core_grid=self.grid,
        )
        ttnn.deallocate(xg)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.n_q_local,
            num_kv_heads=self.n_kv_local,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        # q / k stay fp32 into the elementwise RoPE
        qk_dtype = ttnn.float32 if self.rope == "elementwise" else ttnn.bfloat16
        q_n, k_n = self._head_norm(q, self.q_norm_w, qk_dtype), self._head_norm(k, self.k_norm_w, qk_dtype)
        v_n = self._head_norm(v, None, self.kv_dtype)
        for t in (q, k, v):
            ttnn.deallocate(t)

        if meta is None:
            c = self.chunk_index(start)
            cos, sin = (
                ttnn.unsqueeze_to_4D(ttnn.embedding(self.positions[c], t, layout=ttnn.TILE_LAYOUT))
                for t in (self.cos, self.sin)
            )
            if self.rope == "elementwise":
                cos, sin = (ttnn.typecast(t, ttnn.float32) for t in (cos, sin))
            slot, kv_actual = self.slot, self.kv_actual[c]
        else:
            cos, sin = meta.sliding_rope  # looked up once per chunk, shared by every sliding layer
            slot, kv_actual = meta.slot, meta.kv_actual
        q_r, k_r = self._rope(q_n, cos, sin), self._rope(k_n, cos, sin, self.kv_dtype)
        for t in (q_n, k_n) + ((cos, sin) if meta is None else ()):
            ttnn.deallocate(t)

        if meta is not None and meta.keep is not None:
            # Pad rows (positions >= actual_end) reach the cache as exact zeros: the last 32-token record ships whole.
            k_m, v_m = (
                ttnn.multiply(t, meta.keep, dtype=t.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in (k_r, v_n)
            )
            ttnn.deallocate(k_r)
            ttnn.deallocate(v_n)
            k_r, v_n = k_m, v_m
        self.write_kv(k_r, v_n, start, meta)
        ttnn.deallocate(k_r)
        ttnn.deallocate(v_n)

        attn, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q_r,
            self.cache_k,
            self.cache_v,
            None,
            None,
            None,
            persistent_output_buffer_k=self.halo_k,
            persistent_output_buffer_v=self.halo_v,
            joint_strategy="rear",
            logical_n=self.max_seq,
            program_config=self.sdpa_program_config,
            scale=1.0,
            compute_kernel_config=self.sdpa_cfg,
            dim=2,
            multi_device_global_semaphore=self.ring_sems,
            num_links=self.num_links,
            cluster_axis=SP_AXIS,
            mesh_device=self.mesh,
            topology=ttnn.Topology.Linear,
            ccl_core_grid_offset=self.ccl_offset,
            use_column_major_ccl=True,
            is_causal=True,
            is_balanced=False,
            slot_id=slot,
            kv_actual_isl_tensor=kv_actual,
            kv_cache_num_layers=1,
            kv_cache_layer_idx=0,
            sliding_window_size=self.window,
        )
        ttnn.deallocate(q_r)
        heads = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        partial = ttnn.linear(
            heads,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.proj_cfg,
            core_grid=self.grid,
        )
        ttnn.deallocate(heads)
        out = ttnn.reduce_scatter(partial, dim=2, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(partial)
        return out
