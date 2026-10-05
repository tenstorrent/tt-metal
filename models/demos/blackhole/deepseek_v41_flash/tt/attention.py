# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash attention, decode, TP over heads (``tp_heads``): window-only layers and compressed layers.

Mesh 4x8: rows = users (``users_per_row`` each), columns = TP. Per device (row r, column c):

    x_norm [1,1,T,5120] (replicated over columns)
      q  = rms_norm(x @ wq_a) @ wq_b[:, heads 8c..8c+7]     8 of 64 query heads, head_dim 512
      kv = rms_norm(x @ wkv)                                  1 KV head, K == V, replicated on every column
      RoPE on the last 64 dims of q and kv (adjacent-pair rotation)
      window cache <- kv ; SDPA decode (sink) -> o [T, 8, 512]
      inverse RoPE on o's last 64 dims
      o @ wo_a[group c] (4096 -> 1024) @ wo_b[rows of group c] (1024 -> 5120)  -> partial sums
      all-reduce over the 8 columns -> [1,1,T,5120] replicated

The 8 output groups of the checkpoint's grouped projection coincide with the 8 columns (group g = heads 8g..8g+7).

Op-count optimisations (decode is launch-bound: ~25 ops, the small ones cost ~5 us each inside a trace):
  * wq_a and wkv are ONE matmul (``wqkv``); every projection has an explicit core grid / 1D program config (the default
    config runs these single-tile-row matmuls on a handful of cores: 2-4x slower) and bfp8 weights (DRAM bound).
  * RoPE is ``x * C + (x @ P) * S`` with full-width per-user tables (C = 1, S = 0 off the rotated dims), 3 ops, no
    slice/concat. The kv vector rides in the q tile as an extra head (tile row 0, the q heads are rows 1..8) so ONE
    rope serves q and kv; ``paged_update_cache`` then reads row 0. The extra head is carried through SDPA and the
    head concat (wo_a has 512 zero rows for it) -- it costs nothing and saves a separate kv rope.
  * heads are split with ``nlp_create_qkv_heads_decode`` on ``[kv | q | latent | latent]`` (the K slot carries the
    compressed latent for the compressed-cache write), the head concat is ``nlp_concat_heads_decode``.
  * compressed layers keep window ring and compressed cache in ONE ``[T, 1, WINDOW + max_comp, 512]`` tensor and
    use ONE SDPA decode call (non-causal, additive mask, sink) instead of a ~15 op composite.
  * every L1 sharded temporary is freed before SDPA runs: it statically needs ~1.45 MB of the L1 on its cores.
"""

import os

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import attn_fused as AF

HEAD_DIM = 512
ROPE_DIM = 64
N_HEADS = 64
N_GROUPS = 8
O_LORA = 1024
Q_LORA = 1280
DIM = 5120
WINDOW = 128
PAD_HEADS = 32  # local heads (8) padded to a tile for SDPA decode
LOCAL_HEADS = N_HEADS // N_GROUPS
NH = LOCAL_HEADS + 1  # tile rows used per user: row 0 carries the kv vector (see _qkv), rows 1..8 the q heads


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


def full_pair_swap():
    """[512, 512]: the pair swap on the last 64 dims, zero elsewhere."""
    P = torch.zeros(HEAD_DIM, HEAD_DIM)
    P[HEAD_DIM - ROPE_DIM :, HEAD_DIM - ROPE_DIM :] = pair_swap_matrix()
    return P


def _kv_dtype():
    """Storage dtype of the attention KV caches (window ring + compressed latents): env DSV41_KV_DTYPE=bf16 (default) | bfp8 | bfp4."""
    return {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "split_emu": ttnn.bfloat16}[
        os.environ.get("DSV41_KV_DTYPE", "bf16")
    ]


def _kv_emu():
    """DSV41_KV_DTYPE=split_emu: ACCURACY EMULATION of the split (window ring bfp8, compressed latents bfp4) in bf16 storage. A real split
    needs two SDPA calls whose softmax partials cannot be merged (SDPA decode returns no LSE), so it is only emulated: values are
    rounded through the block format on every write and at load time; no bytes or time are saved."""
    return os.environ.get("DSV41_KV_DTYPE", "bf16") == "split_emu"


def _emu_host(t, dtype):
    """torch [..., L, 512] -> the same values rounded through a block-float tile format."""
    return (
        ttnn.to_torch(ttnn.from_torch(t.float().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT))
        .reshape(t.shape)
        .float()
    )


def _emu_dev(row, dtype):
    return ttnn.typecast(ttnn.typecast(row, dtype), ttnn.bfloat16, memory_config=row.memory_config())


def _wdtype():
    return ttnn.bfloat8_b if os.environ.get("DSV41_ATTN_WDT", "bfp8") == "bfp8" else ttnn.bfloat16


def _cfg_1d(K, N, max_cores):
    """1D multicast program config for a single tile-row activation: N split over <= max_cores cores."""
    Kt, Nt = K // 32, N // 32
    pcn = -(-Nt // max_cores)
    n = -(-Nt // pcn)
    gx, gy = min(n, 8), -(-n // 8)
    sw = max(d for d in range(1, min(pcn, 4) + 1) if pcn % d == 0)
    ibw = max(d for d in range(1, 9) if Kt % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=sw,
        per_core_M=1,
        per_core_N=pcn,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _ucfg32():
    return ttnn.create_sharded_memory_config(
        shape=(PAD_HEADS, HEAD_DIM),
        core_grid=ttnn.num_cores_to_corerangeset(32, ttnn.CoreCoord(8, 8), row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _proj_kwargs(name, K, N):
    """Matmul placement per projection (measured in-trace on BH, tests/test_attn_probe_matmul.py): the default config
    runs these single-tile-row matmuls on a handful of cores (K=5120: 80 us); a 2x8 grid / 1D config is 2-4x faster.
    Override with DSV41_ATTN_CFG_<NAME>=cg:y,x | 1d:max_cores | default (probe knob)."""
    v = os.environ.get(f"DSV41_ATTN_CFG_{name}")
    if v is None:
        v = {"QKV": "1d:64", "CP": "cg:2,8", "OA": "cg:2,8", "QB": "1d:32", "OB": "1d:32", "SWO": "cg:1,8"}.get(
            name, "default"
        )
    if v == "default":
        return {}
    kind, arg = v.split(":")
    if kind == "cg":
        y, x = (int(t) for t in arg.split(","))
        return {"core_grid": ttnn.CoreGrid(y=y, x=x)}
    return {"program_config": _cfg_1d(K, N, int(arg))}


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
        wdt = _wdtype()
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

        self._up = up
        col_shard = lambda dim: ttnn.ShardTensor2dMesh(md, dims=(None, dim), mesh_shape=shape)
        # replicated projections: weights stored [in, out]
        self.wqkv = up(
            torch.cat([w["wq_a"].T, w["wkv"].T], dim=1).reshape(1, 1, DIM, Q_LORA + HEAD_DIM), dtype=wdt
        )  # [wq_a | wkv]
        # tile-layout gammas: the row-major gamma costs a tilize inside every rms_norm call (20 us vs 13 us)
        self.q_norm = up(w["q_norm"].reshape(1, 1, 1, Q_LORA))
        self.kv_norm = up(w["kv_norm"].reshape(1, 1, 1, HEAD_DIM))
        # column-parallel over heads: wq_b^T [1280, 64*512], local [1280, 8*512]
        self.wq_b = up(w["wq_b"].T.reshape(1, 1, Q_LORA, N_HEADS * HEAD_DIM), dtype=wdt, mapper=col_shard(3))
        # grouped o-projection: group g = column g
        wo_a = w["wo_a"].reshape(N_GROUPS, O_LORA, N_HEADS * HEAD_DIM // N_GROUPS)  # [g, r, d]
        wo_a = wo_a.permute(2, 0, 1).reshape(-1, N_GROUPS * O_LORA)  # [4096, 8*1024]
        # 512 zero rows in front: the concatenated heads carry the kv row as "head 0" (see _qkv)
        self.wo_a = up(
            torch.cat([torch.zeros(HEAD_DIM, wo_a.shape[1]), wo_a.float()]).reshape(1, 1, -1, N_GROUPS * O_LORA),
            dtype=wdt,
            mapper=col_shard(3),
        )
        self.wo_b = up(
            w["wo_b"].T.reshape(1, 1, N_GROUPS * O_LORA, DIM), dtype=wdt, mapper=col_shard(2)
        )  # [8*1024, 5120]
        # attention sink, pre-divided by the softmax scale (the kernel multiplies sinks by `scale`)
        sink = (w["attn_sink"].float() / self.scale).reshape(N_GROUPS, LOCAL_HEADS)
        sinks = torch.zeros(N_GROUPS, PAD_HEADS, 32)
        sinks[:, 1:NH, 0] = sink  # row 0 is the kv "head"
        self.sinks = up(
            sinks.reshape(N_GROUPS * PAD_HEADS, 32), mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 0), mesh_shape=shape)
        )  # 2-D [32, 32] per device
        self.Pf = up(full_pair_swap().reshape(1, 1, HEAD_DIM, HEAD_DIM))
        self.cos_tab, self.sin_tab = rope_tables(freqs_cis)
        # KV cache [T, 1, max_seq, 512]; K and V are the same vector, one tensor serves both
        self.cache = up(torch.zeros(self.T, 1, max_seq, HEAD_DIM), dtype=_kv_dtype())
        self.ckc = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.ckc_sdpa = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("DSV41_ATTN_SDPA_FID", "HiFi4")),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.eps = 1e-20
        self._ucfg = ttnn.create_sharded_memory_config(
            shape=(PAD_HEADS, HEAD_DIM),
            core_grid=ttnn.num_cores_to_corerangeset(self.T, ttnn.CoreCoord(8, 8), row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self._proj = {
            "QKV": _proj_kwargs("QKV", DIM, Q_LORA + HEAD_DIM),
            "SWO": _proj_kwargs("SWO", HEAD_DIM, HEAD_DIM),
            "QB": _proj_kwargs("QB", Q_LORA, LOCAL_HEADS * HEAD_DIM),
            "OA": _proj_kwargs("OA", NH * HEAD_DIM, O_LORA),
            "OB": _proj_kwargs("OB", O_LORA, DIM),
            "SW": _proj_kwargs("SW", HEAD_DIM, HEAD_DIM),
            "CP": _proj_kwargs("CP", DIM, 2 * HEAD_DIM),
        }

    # ---- step inputs (positions of the users in each row) -------------------------------------------
    def _rows_up(self, t, dim, dtype=ttnn.bfloat16):
        md = self.mesh_device
        return ttnn.from_torch(
            t.to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32),
            device=md,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(dim, None), mesh_shape=(self.rows, self.cols)),
        )

    def _put(self, st, key, t, dim, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        """Create ``st[key]`` on first use; afterwards overwrite the SAME device buffer in place (a trace replays with
        the buffer addresses it captured, so per-step values must be copied into persistent tensors)."""
        md = self.mesh_device
        tdt = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.int32: torch.int32}[dtype]
        host = ttnn.from_torch(
            t.to(tdt).contiguous(),
            dtype=dtype,
            layout=layout,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(dim, None), mesh_shape=(self.rows, self.cols)),
        )
        if key in st:
            ttnn.copy_host_to_device_tensor(host, st[key])
        else:
            st[key] = ttnn.to_device(host, md, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _rope_inputs(self, positions: torch.Tensor):
        """Full-width rope tables for a position per user: row layout [1,1,B,w] (C = 1 / S = 0 off the rotated dims)."""
        B = positions.shape[0]
        cos, sin = self.cos_tab[positions], self.sin_tab[positions]  # [B, 64]
        c = torch.ones(B, HEAD_DIM)
        s = torch.zeros(B, HEAD_DIM)
        c[:, HEAD_DIM - ROPE_DIM :], s[:, HEAD_DIM - ROPE_DIM :] = cos, sin
        return c, s

    def step_inputs(self, positions: torch.Tensor, st=None):
        """positions: [rows * T] int, global user order (row-major). Returns device tensors for forward().
        With ``st`` (a dict returned by an earlier call) the tensors are refreshed IN PLACE (same device buffers), which is
        what a captured trace needs; layers of the same kind can share one ``st``."""
        st = {} if st is None else st
        B = positions.shape[0]
        c, s = self._rope_inputs(positions)
        self._put(st, "pos", positions, 0, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self._put(st, "Ch", c.reshape(1, B, 1, HEAD_DIM), 1)
        self._put(st, "Sh", s.reshape(1, B, 1, HEAD_DIM), 1)
        self._put(st, "nSh", (-s).reshape(1, B, 1, HEAD_DIM), 1)
        return st

    def load_window(self, kv_rows: torch.Tensor):
        """Host cache seed: kv_rows [rows*T, S, 512] (positions 0..S-1) -> every row's cache slice."""
        rows_shard = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, None), mesh_shape=(self.rows, self.cols))
        S = kv_rows.shape[1]
        full = torch.zeros(self.rows * self.T, 1, self.max_seq, HEAD_DIM)
        full[:, 0, :S] = kv_rows
        self.cache = ttnn.from_torch(
            full,
            device=self.mesh_device,
            dtype=_kv_dtype(),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rows_shard,
        )

    # ---- helpers --------------------------------------------------------------------------------------
    def _lin(self, x, w, name, **kw):
        if (
            int(x.shape[-2]) > 32
        ):  # spec verify with > 32 rows (DSV41_SPEC_ROWS): the tuned 1D configs are per_core_M = 1 (<= 32 rows)
            return ttnn.linear(x, w, compute_kernel_config=self.ckc, core_grid=ttnn.CoreGrid(y=8, x=8), **kw)
        return ttnn.linear(x, w, compute_kernel_config=self.ckc, **self._proj[name], **kw)

    def _rope_rows(self, x, c, s, name="SW"):
        """x [..., 512] (any leading dims): x * C + (x @ P) * S with full-width tables broadcast over rows."""
        if AF.flag("DSV41_ATTN_FUSED_ROPE", "1"):
            return AF.rope_inplace(x, c, s, 1, rows_layout=True)
        return ttnn.addcmul(ttnn.multiply(x, c), self._lin(x, self.Pf, name), s)

    def _qkv(self, x, st, lat=None):
        """x [1,1,T,5120] bf16 -> (q, kv, lat_row) with q [1,T,9,512] (DRAM; row 0 = RoPE'd kv vector, rows 1..8 = RoPE'd q
        heads), kv [1,T,9,512] the same values L1 height-sharded (row 0 is what ``paged_update_cache`` reads) and lat_row
        [1,T,1,512] (L1 sharded) the compressed latent (``lat`` [1,1,T,512], already RoPE'd) when given.

        The kv vector rides in the q tile as an extra head so one RoPE (3 ops) serves q and kv; the latent travels in the
        K slot of ``nlp_create_qkv_heads_decode``."""
        T = self.T
        y = self._lin(x, self.wqkv, "QKV")  # [1,1,T,1792] = [wq_a | wkv]
        qr = ttnn.rms_norm(ttnn.slice(y, [0, 0, 0, 0], [1, 1, T, Q_LORA]), weight=self.q_norm, epsilon=self.eps)
        self._last_qr = qr  # the indexer of index-source layers consumes it (paged path)
        q = self._lin(qr, self.wq_b, "QB")
        kv = ttnn.rms_norm(
            ttnn.slice(y, [0, 0, 0, Q_LORA], [1, 1, T, Q_LORA + HEAD_DIM]), weight=self.kv_norm, epsilon=self.eps
        )
        tail = kv if lat is None else lat
        cat = ttnn.concat([kv, q, tail, tail], dim=3)
        if (
            T > 32
        ):  # spec verify with > 32 rows (DSV41_SPEC_ROWS): nlp_create_qkv_heads_decode takes <= 32 users per call
            assert T % 32 == 0
            cfg32 = _ucfg32()
            qs, ks = [], []
            for o in range(0, T, 32):
                qh, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                    ttnn.slice(cat, [0, 0, o, 0], [1, 1, o + 32, cat.shape[3]]),
                    num_heads=NH,
                    num_kv_heads=1,
                    memory_config=cfg32,
                )
                ttnn.deallocate(v)
                qs.append(ttnn.to_memory_config(qh, ttnn.DRAM_MEMORY_CONFIG))
                ttnn.deallocate(qh)
                ks.append(k)
            for k2 in ks[1:]:
                ttnn.deallocate(k2)
            q_rot = self._rope_heads(ttnn.concat(qs, dim=1), st["Ch"], st["Sh"], fused_users=T)
            return q_rot, ttnn.clone(q_rot), ks[0]
        qh, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            cat, num_heads=NH, num_kv_heads=1, memory_config=self._ucfg
        )
        ttnn.deallocate(v)
        # leave L1 now: SDPA at head_dim 512 needs almost all of L1 on its cores, which are the user cores of q/k/v
        q_d = ttnn.to_memory_config(qh, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(qh)
        q_rot = self._rope_heads(q_d, st["Ch"], st["Sh"], fused_users=T)
        return q_rot, ttnn.to_memory_config(q_rot, self._ucfg), k

    def _fused_pre(self):
        return AF.flag("DSV41_ATTN_FUSED_PRE", "1") and self.T <= 16 and _kv_dtype() == ttnn.bfloat16 and not _kv_emu()

    def _qkv_fused(self, x, st, lat, pos_key, comp_key=None):
        """_qkv + both cache writes in ONE program (tt/attn_pre.py) -> q_rot [1,T,32,512] DRAM (row 0 kv, rows 1..8 q heads)."""
        from models.demos.blackhole.deepseek_v41_flash.tt.attn_pre import attn_pre

        T = self.T
        y = self._lin(x, self.wqkv, "QKV")
        qr = ttnn.rms_norm(ttnn.slice(y, [0, 0, 0, 0], [1, 1, T, Q_LORA]), weight=self.q_norm, epsilon=self.eps)
        q = self._lin(qr, self.wq_b, "QB")
        kv = ttnn.rms_norm(
            ttnn.slice(y, [0, 0, 0, Q_LORA], [1, 1, T, Q_LORA + HEAD_DIM]), weight=self.kv_norm, epsilon=self.eps
        )
        return attn_pre(kv, q, lat, self.cache, st[pos_key], st[comp_key] if comp_key else None, st["Ch"], st["Sh"], T)

    def _write_cache(self, cache, row, idx, comp=False):
        """cache [T,1,L,512] <- row [1,T,*,512] (L1 sharded, tile row 0 valid) at per-user index idx (int32 [T])."""
        if _kv_emu():
            row = _emu_dev(row, ttnn.bfloat4_b if comp else ttnn.bfloat8_b)
        ttnn.experimental.paged_update_cache(cache, row, update_idxs_tensor=idx, page_table=None)

    def _sdpa_cfg(self, k_chunk):
        # One core per (user, kv head). Two cores per head are ~7 us faster but grow the SDPA static CB region from
        # 1.453 MB to 1.490 MB, which no longer fits the contiguous L1 left by the MoE decode ops (resident decode
        # clashes), so the default is 1 (tests/test_attn_probe4.py).
        mc = int(os.environ.get("DSV41_ATTN_SDPA_CORES", "1"))
        n = (
            self.T * mc
        )  # one core per (user, kv head); wrap into rows of 8 like the sharded q/k (the BH worker grid is 13 wide)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(min(n, 8), (n + 7) // 8),
            q_chunk_size=0,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
            max_cores_per_head_batch=mc,
        )

    def _finish(self, o, st):
        """o [1,T,9,512] attention output -> inverse RoPE, grouped output projection, all-reduce -> [1,1,T,5120]."""
        if self.T > 32:  # spec verify with > 32 rows: nlp_concat_heads_decode takes <= 32 users per call
            o = self._rope_heads(o, st["Ch"], st["nSh"], fused_users=self.T)
            cs = []
            for r in range(0, self.T, 32):
                oc = ttnn.to_memory_config(ttnn.slice(o, [0, r, 0, 0], [1, r + 32, o.shape[2], o.shape[3]]), _ucfg32())
                cs.append(
                    ttnn.to_memory_config(
                        ttnn.experimental.nlp_concat_heads_decode(oc, num_heads=NH), ttnn.DRAM_MEMORY_CONFIG
                    )
                )
            c = ttnn.concat(cs, dim=2)
            part = self._lin(self._lin(c, self.wo_a, "OA"), self.wo_b, "OB")
            out = self.mesh_config.allreduce(part, self.ccl, axis=1)
            return ttnn.reshape(out, (1, 1, self.T, DIM))
        o = self._rope_inv(o, st)
        c = ttnn.experimental.nlp_concat_heads_decode(o, num_heads=NH)  # [1,1,32,9*512], L1 width sharded
        c = ttnn.to_memory_config(c, ttnn.DRAM_MEMORY_CONFIG)  # wo_a is 2x faster from interleaved DRAM
        part = self._lin(self._lin(c, self.wo_a, "OA"), self.wo_b, "OB")
        out = self.mesh_config.allreduce(part, self.ccl, axis=1)
        return ttnn.reshape(
            out, (1, 1, self.T, DIM), (1, 1, 32, DIM)
        )  # nlp_concat_heads_decode pads the rows to 32: view back to T

    def _rope_heads(self, x, c, s, memory_config=None, fused_users=None):
        """x [1,T,32,512] (DRAM): x * C + (x @ P) * S, tables [1,T,1,512] broadcast over the head rows."""
        if fused_users is not None and fused_users > 32 and AF.flag("DSV41_ATTN_FUSED_ROPE", "1"):
            # the rope kernel uses 2 cores per user: <= 32 users per call (spec verify with > 32 rows)
            parts = []
            for r in range(0, fused_users, 32):
                sl = lambda t: ttnn.slice(t, [0, r, 0, 0], [1, r + 32, t.shape[2], t.shape[3]])
                parts.append(AF.rope_inplace(sl(x), sl(c), sl(s), 32))
            x = ttnn.concat(parts, dim=1)
            return x if memory_config is None else ttnn.to_memory_config(x, memory_config)
        if fused_users is not None and AF.flag("DSV41_ATTN_FUSED_ROPE", "1"):
            AF.rope_inplace(x, c, s, fused_users)  # in place on the DRAM tensor
            return x if memory_config is None else ttnn.to_memory_config(x, memory_config)
        return ttnn.addcmul(ttnn.multiply(x, c), self._lin(x, self.Pf, "SWO"), s, memory_config=memory_config)

    def _rope_inv(self, o, st):
        return self._rope_heads(o, st["Ch"], st["nSh"], memory_config=self._ucfg, fused_users=self.T)

    def forward(self, x, st):
        """x: [1,1,T,5120] bf16 tile, normed attention input. st: step_inputs(). -> [1,1,T,5120] replicated."""
        if self._fused_pre():
            q = self._qkv_fused(x, st, None, "pos")
            return self._sdpa_finish(q, st)
        q, kv, k = self._qkv(x, st)
        self._write_cache(self.cache, kv, st["pos"])
        # SDPA at head_dim 512 statically reserves ~1.39 MB of L1 on its cores (which are also the user cores of the
        # sharded kv): free them before it runs.
        ttnn.deallocate(kv)
        ttnn.deallocate(k)
        o = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            self.cache,
            self.cache,
            cur_pos_tensor=st["pos"],
            sliding_window_size=WINDOW,
            attention_sink=self.sinks,
            scale=self.scale,
            program_config=self._sdpa_cfg(128),
            compute_kernel_config=self.ckc_sdpa,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, T, 9(32), 512]
        return self._finish(o, st)

    def _sdpa_finish(self, q, st):
        o = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            self.cache,
            self.cache,
            cur_pos_tensor=st["pos"],
            sliding_window_size=WINDOW,
            attention_sink=self.sinks,
            scale=self.scale,
            program_config=self._sdpa_cfg(128),
            compute_kernel_config=self.ckc_sdpa,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return self._finish(o, st)


class DSV41CompressedAttention(DSV41Attention):
    """Layers with ``compress_ratio > 0`` that own a compressor (kv source layers): window ring + compressed cache.

    One cache tensor ``[T, 1, WINDOW + max_comp, 512]``: slots [0, WINDOW) the window ring, [WINDOW, ..) the compressed
    latents. SDPA decode runs non-causal over all slots with an additive mask (unfilled ring slots / unfilled latent
    slots -> -1e9) and the sink, so the whole composite attention is one op.

    Valid only while every compressed position is selected, i.e. compress_len <= index_topk (512), so the
    indexer's top-k is the identity; ``step_inputs`` asserts this. All users must share the step position
    (the compressor's group-complete decision is made on the host).

    Layers that only READ the compressed cache of the kv-source layer (``source``) keep their own copy of the latents
    in their own cache tensor: the owner publishes this step's latent (``last_lat``) and every reader writes it too.
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
            mesh_device, mesh_config, ccl_manager, w, freqs_cis, users_per_row=users_per_row, max_seq=WINDOW + max_comp
        )
        assert ratio in (1, 2) and max_comp % 32 == 0
        self.ratio, self.max_comp, self.index_topk = ratio, max_comp, index_topk
        self.source = source
        self.last_lat = None
        if source is not None:
            assert comp_w is None and source.ratio == ratio and source.max_comp == max_comp
        md, T = mesh_device, users_per_row
        up = self._up
        if comp_w is not None:
            wdt = _wdtype()
            self.c_norm = up(comp_w["norm"].reshape(1, 1, 1, HEAD_DIM))
            if ratio > 1:  # [wkv | wgate] fused, fp32 outputs
                self.c_wcat = up(
                    torch.cat([comp_w["wkv"].T, comp_w["wgate"].T], dim=1).reshape(1, 1, DIM, 2 * HEAD_DIM), dtype=wdt
                )
            else:
                self.c_wkv = up(comp_w["wkv"].T.reshape(1, 1, DIM, HEAD_DIM), dtype=wdt)
        self.cs_state = [None] * ratio  # per slot: [1,1,T,1024] fp32 = [kv | score]
        # step-independent mode (``load_state(..., start_pos=p)``): ratio 2 keeps only the PREVIOUS token's [kv | score] in a
        # persistent buffer and pools (previous, current) at EVERY step; the pooled latent is written to the slot of the group
        # in progress, which the mask hides until the group completes and the real latent overwrites it.
        self.prev_cs = None
        k_chunk = next(c for c in (256, 128, 64, 32) if (WINDOW + max_comp) % c == 0)
        self._k_chunk = k_chunk

    def snapshot_state(self):
        """Copy of the step-carried compressor state (the previous token's [kv | score]); None if there is none."""
        return None if self.prev_cs is None else ttnn.clone(self.prev_cs)

    def restore_state(self, snap):
        if snap is not None:
            ttnn.copy(snap, self.prev_cs)

    @property
    def comp_cache(self):
        return ttnn.slice(self.cache, [0, 0, WINDOW, 0], [self.T, 1, WINDOW + self.max_comp, HEAD_DIM])

    # ---- state seeding / step inputs ---------------------------------------------------------------------
    def load_state(self, window_kv, comp_kv, kv_state, score_state, start_pos=None):
        """window_kv [B,128,512] ring, comp_kv [B,Lc,512], kv_state/score_state [B,ratio,512] (reference buffers)."""
        md = self.mesh_device
        rs = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(self.rows, self.cols))
        up = lambda t, dt, lay: ttnn.from_torch(
            t.contiguous(), device=md, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rs
        )
        B = window_kv.shape[0]
        if _kv_emu():
            window_kv = _emu_host(window_kv.reshape(B, 1, WINDOW, HEAD_DIM), ttnn.bfloat8_b)
            comp_kv = _emu_host(comp_kv.reshape(B, 1, -1, HEAD_DIM), ttnn.bfloat4_b) if self.source is None else comp_kv
        win = up(window_kv.reshape(B, 1, WINDOW, HEAD_DIM).float(), _kv_dtype(), ttnn.TILE_LAYOUT)
        if self.source is not None:  # reads the owner's compressed cache: start from a copy of its latents
            comp = ttnn.slice(self.source.cache, [0, 0, WINDOW, 0], [self.T, 1, WINDOW + self.max_comp, HEAD_DIM])
            self.cache = ttnn.concat([win, comp], dim=2)
            return
        comp = torch.zeros(B, 1, self.max_comp, HEAD_DIM)
        comp[:, 0, : comp_kv.shape[-2]] = comp_kv.reshape(B, -1, HEAD_DIM).float()
        self.cache = up(
            torch.cat([window_kv.reshape(B, 1, WINDOW, HEAD_DIM).float(), comp], dim=2), _kv_dtype(), ttnn.TILE_LAYOUT
        )
        if self.ratio > 1:
            for i in range(self.ratio):
                cs = torch.cat([kv_state[:, i], score_state[:, i]], dim=-1).float().reshape(1, 1, B, 2 * HEAD_DIM)
                self.cs_state[i] = ttnn.from_torch(
                    cs.contiguous(),
                    device=md,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(self.rows, self.cols)),
                )
            if start_pos is not None:  # the state of the previous position is slot (start_pos - 1) % ratio
                prev = (start_pos - 1) % self.ratio
                cs = torch.cat([kv_state[:, prev], score_state[:, prev]], dim=-1).float().reshape(1, 1, B, 2 * HEAD_DIM)
                self.prev_cs = ttnn.from_torch(
                    cs.contiguous(),
                    device=md,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(self.rows, self.cols)),
                )

    def step_inputs(self, positions: torch.Tensor, st=None):
        st = super().step_inputs(positions, st)
        p = int(positions[0])
        assert bool((positions == p).all()), "compressed attention assumes all users share the step position"
        r = self.ratio
        comp_len = (p + 1) // r
        assert comp_len <= min(self.max_comp, self.index_topk), "compressed length exceeds the indexer-free regime"
        B = positions.shape[0]
        i32 = lambda v: torch.full((B,), v, dtype=torch.int32)
        self._put(st, "pos_ring", i32(p % WINDOW), 0, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self._put(st, "comp_idx", i32(WINDOW + p // r), 0, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        g = max(p + 1 - r, 0)  # RoPE position of a freshly pooled latent (first token of its group)
        c, s = self._rope_inputs(torch.full((B,), g))
        self._put(st, "Cg", c.reshape(1, 1, B, HEAD_DIM), 2)
        self._put(st, "Sg", s.reshape(1, 1, B, HEAD_DIM), 2)
        mask = torch.zeros(B, 1, 1, WINDOW + self.max_comp)
        mask[..., p + 1 : WINDOW] = -1e9  # window slots not filled yet (ring not wrapped)
        mask[..., WINDOW + comp_len :] = -1e9
        self._put(st, "mask", mask.expand(B, 1, NH, -1).contiguous(), 0)
        st["slot"], st["complete"] = p % r, (p + 1) % r == 0  # only used by the legacy (host-branching) compressor path
        return st

    # ---- compressor --------------------------------------------------------------------------------------
    def _compress_step(self, x, st):
        """One decode token through the compressor -> [1,1,T,512] RoPE'd latent (rows) if a group completes, else None."""
        if self.ratio == 1:
            lat = ttnn.rms_norm(self._lin(x, self.c_wkv, "CP"), weight=self.c_norm, epsilon=self.eps)
        else:
            cs = self._lin(x, self.c_wcat, "CP", dtype=ttnn.float32)
            if self.prev_cs is not None:  # step-independent: pool (previous token, this token) at every step
                (a, b) = self.prev_cs, cs
            else:  # legacy: slot-indexed state and a host decision whether the group completed
                self.cs_state[st["slot"]] = cs
                if not st["complete"]:
                    return None
                (a, b) = self.cs_state
            sl = lambda t, lo: ttnn.slice(t, [0, 0, 0, lo], [1, 1, self.T, lo + HEAD_DIM])
            # softmax over the two slots: pooled = kv_b + (kv_a - kv_b) * sigmoid(score_a - score_b), on [kv | score]
            d = ttnn.subtract(a, b)
            pooled = ttnn.addcmul(sl(b, 0), sl(d, 0), sl(ttnn.sigmoid(d), HEAD_DIM))
            lat = ttnn.rms_norm(ttnn.typecast(pooled, ttnn.bfloat16), weight=self.c_norm, epsilon=self.eps)
            if self.prev_cs is not None:
                ttnn.copy(cs, self.prev_cs)  # this token becomes the "previous" one of the next step
        return self._rope_rows(lat, st["Cg"], st["Sg"])

    def forward(self, x, st):
        lat = self._compress_step(x, st) if self.source is None else self.source.last_lat
        self.last_lat = lat
        if self._fused_pre():
            q = self._qkv_fused(x, st, lat, "pos_ring", "comp_idx")
            return self._sdpa_finish_c(q, st)
        q, kv, k = self._qkv(x, st, lat)
        self._write_cache(self.cache, kv, st["pos_ring"])
        if lat is not None:
            self._write_cache(self.cache, k, st["comp_idx"], comp=True)
        ttnn.deallocate(kv)  # free L1 on the SDPA cores (see DSV41Attention.forward)
        ttnn.deallocate(k)
        o = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            self.cache,
            self.cache,
            is_causal=False,
            attn_mask=st["mask"],
            attention_sink=self.sinks,
            scale=self.scale,
            program_config=self._sdpa_cfg(self._k_chunk),
            compute_kernel_config=self.ckc_sdpa,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return self._finish(o, st)

    def _sdpa_finish_c(self, q, st):
        o = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            self.cache,
            self.cache,
            is_causal=False,
            attn_mask=st["mask"],
            attention_sink=self.sinks,
            scale=self.scale,
            program_config=self._sdpa_cfg(self._k_chunk),
            compute_kernel_config=self.ckc_sdpa,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return self._finish(o, st)
