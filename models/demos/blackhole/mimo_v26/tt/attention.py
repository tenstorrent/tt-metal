# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2.6-Flash attention (SWA + global) on a Blackhole 4x8 mesh.

Mapping: mesh columns (axis 1, 8 devices) = tensor parallel over heads, mesh rows (axis 0, 4 devices) = independent user groups (DP).
Per device: 8 Q heads (64/8), 1 KV head (SWA: 8 KV heads -> head c; global: 4 KV heads -> head c//2, i.e. replicated on 2 columns).
Decisions (LOG E261007-0114-2df + coordinator): V is zero-padded to 192 inside the weights so the KV cache and every op see head_dim 192; V*0.707 is folded
into the V weights; o_proj rows for the 64 pad dims are zero. RoPE = x*cos + (x@R)*sin over the full 192 (R = rotate_half on the first 64 dims, cos=1/sin=0 on
the pass-through dims). Sink input to the SDPA ops is sink/scale. SWA KV is a ring of `ring_blocks` pages (decode: cache_position_modulo, chunked prefill:
page table aliasing logical block i -> ring block i % ring_blocks); global KV is a plain paged cache.
"""
from dataclasses import dataclass

import torch

import ttnn

HIDDEN, NQ, DQK, DV, ROT = 4096, 64, 192, 128, 64
SCALE = DQK**-0.5
VSCALE = 0.707
WINDOW = 128
BLOCK = 64
TP = 8


@dataclass(frozen=True)
class LayerCfg:
    kind: str  # "swa" | "ga"
    nkv: int
    theta: float
    sink: bool
    window: int  # 0 = none


SWA = LayerCfg("swa", 8, 1e4, True, WINDOW)
GA = LayerCfg("ga", 4, 1e7, False, 0)


def rope_tables(theta, max_pos):
    """cos/sin [max_pos, 192]: first 64 dims rotary (HF cat(freqs,freqs) layout), cos=1/sin=0 on the 128 pass-through dims."""
    inv = theta ** (-torch.arange(0, ROT, 2).float() / ROT)
    f = torch.arange(max_pos).float()[:, None] * inv[None]
    emb = torch.cat([f, f], -1)
    cos = torch.cat([emb.cos(), torch.ones(max_pos, DQK - ROT)], -1)
    sin = torch.cat([emb.sin(), torch.zeros(max_pos, DQK - ROT)], -1)
    return cos, sin


def rot_matrix():
    """x @ R == rotate_half on dims [0,64): out[j] = -x[j+32] (j<32), out[j] = x[j-32] (32<=j<64), 0 elsewhere."""
    R = torch.zeros(DQK, DQK)
    for j in range(ROT // 2):
        R[j + ROT // 2, j] = -1.0
        R[j, j + ROT // 2] = 1.0
    return R


def convert_weights(cfg: LayerCfg, q_w, k_w, v_w, o_w, sink=None):
    """HF-layout torch weights -> per-column host tensors.
    q_w [64*192, 4096], k_w [nkv*192, 4096], v_w [nkv*128, 4096], o_w [4096, 64*128], sink [64] or None.
    returns wqkv [8, 4096, 1920], wo [8, 1536, 4096], sink_dec [8*32, 32] (sink/scale), sink_pre [1, 64, 1, 1] (sink/scale).
    """
    nqc = NQ // TP
    wq, wo = [], []
    wqkv = []
    for c in range(TP):
        kvh = c * cfg.nkv // TP
        q = q_w[c * nqc * DQK : (c + 1) * nqc * DQK]
        k = k_w[kvh * DQK : (kvh + 1) * DQK]
        v = v_w[kvh * DV : (kvh + 1) * DV] * VSCALE
        v = torch.cat([v, torch.zeros(DQK - DV, HIDDEN)], 0)
        wqkv.append(torch.cat([q, k, v], 0).T.contiguous())
        o = torch.zeros(nqc * DQK, HIDDEN)
        for h in range(nqc):
            o[h * DQK : h * DQK + DV] = o_w[:, (c * nqc + h) * DV : (c * nqc + h + 1) * DV].T
        wo.append(o)
    out = dict(wqkv=torch.stack(wqkv), wo=torch.stack(wo))
    if sink is not None:
        s = sink.float() / SCALE
        sd = torch.zeros(TP, 32, 32)
        sd[:, :nqc, 0] = s.reshape(TP, nqc)
        out["sink_dec"] = sd.reshape(TP * 32, 32)
        out["sink_pre"] = s.reshape(1, NQ, 1, 1)
    return out


def _pad8(t):
    """pad the last dim to a multiple of 8 int32 entries (page table stick size must be a multiple of 32 bytes)"""
    n = t.shape[-1]
    return torch.nn.functional.pad(t, (0, (-n) % 8))


def _k_hifi4():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )


def _k_hifi2():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )


def _k_mm():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


def _row_mapper(mesh):
    return ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(0, None))


def decode_pos_io(mesh, slots=32):
    """Device-resident per-row decode position tensors shared by all layers: cur [rows*slots] sharded -> int32 [slots] per device, idx uint32 [1,slots]."""
    rows = mesh.shape[0]
    m = _row_mapper(mesh)
    mk = lambda t, dt: ttnn.from_torch(t, device=mesh, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=m)
    return dict(
        cur=mk(torch.zeros(rows * slots, dtype=torch.int32), ttnn.int32),
        idx=mk(torch.zeros(rows, slots, dtype=torch.int32), ttnn.uint32),
    )


def decode_pos_set(mesh, io, positions):
    """Write per-row/per-slot positions into the persistent device tensors. Returns the HOST tensors: the host->device copy is asynchronous, the caller must keep
    them alive (and synchronize) until the copy has executed, otherwise the device may read freed host memory (row-wide glitches seen as run-to-run nondeterminism).
    """
    rows = mesh.shape[0]
    pos = torch.as_tensor(positions, dtype=torch.int32)
    if pos.dim() == 1:
        pos = pos.reshape(1, -1).expand(rows, -1)
    assert pos.shape[0] == rows
    m = _row_mapper(mesh)
    h = lambda t, dt: ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=m)
    hosts = [h(pos.reshape(-1), ttnn.int32), h(pos, ttnn.uint32)]
    ttnn.copy_host_to_device_tensor(hosts[0], io["cur"])
    ttnn.copy_host_to_device_tensor(hosts[1], io["idx"])
    return hosts


class MiMoAttention:
    def __init__(
        self,
        mesh,
        cfg: LayerCfg,
        host_w: dict,
        page_table: torch.Tensor,
        max_pos: int,
        ring_blocks: int = 0,
        users: int = 32,
        wdtype=ttnn.bfloat8_b,
    ):
        """page_table: [users, nb] int32 physical block ids. GA: nb = max_pos/BLOCK. SWA: nb = ring_blocks (ring)."""
        self.mesh, self.cfg, self.users = mesh, cfg, users
        self.ring_blocks = ring_blocks if cfg.window else 0
        self.max_pos = max_pos
        self.page_table = page_table.to(torch.int32)
        assert (not cfg.window) or page_table.shape[1] == ring_blocks
        self.nlog = max_pos // BLOCK
        rows, cols = mesh.shape
        self.rows = rows
        col_shard = lambda d: ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(None, d))
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def up(t, dtype, mapper, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t, device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
            )

        self.wqkv = up(host_w["wqkv"].reshape(1, TP, HIDDEN, -1), wdtype, col_shard(1))
        self.wo = up(host_w["wo"].reshape(1, TP, -1, HIDDEN), wdtype, col_shard(1))
        self.sink_dec = up(host_w["sink_dec"], ttnn.bfloat16, col_shard(0)) if cfg.sink else None
        self.sink_pre = up(host_w["sink_pre"], ttnn.bfloat16, col_shard(1)) if cfg.sink else None
        cos, sin = rope_tables(cfg.theta, max_pos)
        self.cos_tab = up(cos.reshape(1, 1, max_pos, DQK), ttnn.bfloat16, rep)
        self.sin_tab = up(sin.reshape(1, 1, max_pos, DQK), ttnn.bfloat16, rep)
        self.rot = up(rot_matrix().reshape(1, 1, DQK, DQK), ttnn.bfloat16, rep)
        nblocks = int(self.page_table.max()) + 1
        zeros = torch.zeros(nblocks, 1, BLOCK, DQK)
        self.k_cache = up(zeros, ttnn.bfloat16, rep)
        self.v_cache = up(zeros, ttnn.bfloat16, rep)
        self.pt_dec = up(_pad8(self.page_table), ttnn.int32, rep, ttnn.ROW_MAJOR_LAYOUT)
        grid = mesh.compute_with_storage_grid_size()
        self.dec_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False
        )
        self.pre_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=256, k_chunk_size=256, exp_approx_mode=False
        )
        kv_cores = ttnn.num_cores_to_corerangeset(users, ttnn.CoreCoord(8, 8), row_wise=True)
        self.kv_mem = ttnn.create_sharded_memory_config(
            shape=(32, DQK),
            core_grid=kv_cores,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.att_mem = ttnn.create_sharded_memory_config(
            shape=(32, DQK),
            core_grid=kv_cores,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self._rep = rep
        self._up = up
        self._dec_states = {}

    # ------------------------------------------------------------------ helpers
    def _rope(self, x, cos, sin):
        xr = ttnn.matmul(x, self.rot, compute_kernel_config=_k_hifi4())
        a = ttnn.multiply(x, cos)
        b = ttnn.multiply(xr, sin)
        ttnn.deallocate(xr)
        out = ttnn.add(a, b)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        return out

    def _all_reduce(self, x):
        return ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def logical_page_row(self, user):
        row = self.page_table[user]
        if self.ring_blocks:
            return torch.stack([row[i % self.ring_blocks] for i in range(self.nlog)]).to(torch.int32)
        return row[: self.nlog]

    # ------------------------------------------------------------------ host -> persistent device inputs (trace friendly)
    def _host(self, t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._rep)

    def _persist(self, t, dtype):
        return ttnn.from_torch(t, device=self.mesh, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._rep)

    def prefill_io(self, S: int):
        """Persistent device tensors a prefill chunk of S tokens reads (update them with prefill_set before each call / trace replay)."""
        assert S % 128 == 0
        qc = min(256, S)
        return dict(
            S=S,
            cfg=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
                q_chunk_size=qc,
                k_chunk_size=qc,
                exp_approx_mode=False,
            ),
            rope_idx=self._persist(torch.zeros(1, S, dtype=torch.int32), ttnn.uint32),
            fill_pt=self._persist(torch.zeros(1, S // BLOCK, dtype=torch.int32), ttnn.int32),
            sdpa_pt=self._persist(
                torch.zeros(1, _pad8(torch.zeros(1, self.nlog)).shape[-1], dtype=torch.int32), ttnn.int32
            ),
            start=self._persist(torch.zeros(1, dtype=torch.int32), ttnn.int32),
        )

    def prefill_set(self, io, chunk_start: int, user: int):
        S = io["S"]
        assert chunk_start % min(256, S) == 0 and chunk_start + S <= self.max_pos
        full = self.logical_page_row(user)
        nb0 = chunk_start // BLOCK
        host = dict(
            rope_idx=self._host(
                torch.arange(chunk_start, chunk_start + S, dtype=torch.int32).reshape(1, S), ttnn.uint32
            ),
            fill_pt=self._host(full[nb0 : nb0 + S // BLOCK].reshape(1, -1).to(torch.int32), ttnn.int32),
            sdpa_pt=self._host(_pad8(full.reshape(1, -1).to(torch.int32)), ttnn.int32),
            start=self._host(torch.tensor([chunk_start], dtype=torch.int32), ttnn.int32),
        )
        for k, h in host.items():
            ttnn.copy_host_to_device_tensor(h, io[k])

    def prefill_chunk(self, x, io):
        """x: [1,1,S,4096] bf16 TILE per device (replicated over columns). All position dependent state comes from `io` (device tensors), so the call
        can be captured in a trace and replayed with prefill_set. Returns [1,1,S,4096] (all-reduced over columns)."""
        S = io["S"]
        qkv = ttnn.linear(
            x, self.wqkv, dtype=ttnn.bfloat16, compute_kernel_config=_k_mm(), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=NQ // TP, num_kv_heads=1, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)
        cos = ttnn.reshape(ttnn.embedding(io["rope_idx"], self.cos_tab, layout=ttnn.TILE_LAYOUT), (1, 1, S, DQK))
        sin = ttnn.reshape(ttnn.embedding(io["rope_idx"], self.sin_tab, layout=ttnn.TILE_LAYOUT), (1, 1, S, DQK))
        qr = self._rope(q, cos, sin)
        kr = self._rope(k, cos, sin)
        for t in (q, k, cos, sin):
            ttnn.deallocate(t)
        ttnn.experimental.paged_fill_cache(self.k_cache, kr, io["fill_pt"], batch_idx=0)
        ttnn.experimental.paged_fill_cache(self.v_cache, v, io["fill_pt"], batch_idx=0)
        ttnn.deallocate(kr)
        ttnn.deallocate(v)
        kw = dict(chunk_start_idx_tensor=io["start"], program_config=io["cfg"], compute_kernel_config=_k_hifi2())
        if self.cfg.window:
            kw.update(sliding_window_size=self.cfg.window, attention_sink=self.sink_pre)
        o = ttnn.transformer.chunked_scaled_dot_product_attention(qr, self.k_cache, self.v_cache, io["sdpa_pt"], **kw)
        ttnn.deallocate(qr)
        oc = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,S,8*192]
        ttnn.deallocate(o)
        part = ttnn.linear(
            oc, self.wo, dtype=ttnn.bfloat16, compute_kernel_config=_k_mm(), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(oc)
        return self._all_reduce(part)

    # ------------------------------------------------------------------ decode (32 user slots per row)
    def _dec_state(self, B):
        """(page table device tensor [B, nb], kv shard config, attention-output shard config) for B decode slots; built lazily, freed by reconfigure."""
        if B not in self._dec_states:
            cores = ttnn.num_cores_to_corerangeset(B, ttnn.CoreCoord(8, 8), row_wise=True)
            mk = lambda: ttnn.create_sharded_memory_config(
                shape=(32, DQK),
                core_grid=cores,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            pt = self._up(_pad8(self.page_table[:B]), ttnn.int32, self._rep, ttnn.ROW_MAJOR_LAYOUT)
            self._dec_states[B] = (pt, mk(), mk())
        return self._dec_states[B]

    def reconfigure(self, page_table):
        """New KV pool for a new user count / context: free the caches and the per-B decode states, allocate new caches for page_table ([slots, nb] int32).
        Weights, rope tables, sinks are kept."""
        for t in (self.k_cache, self.v_cache, self.pt_dec):
            ttnn.deallocate(t)
        for pt, _, _ in self._dec_states.values():
            ttnn.deallocate(pt)
        self._dec_states = {}
        self.page_table = page_table.to(torch.int32)
        nblocks = int(self.page_table.max()) + 1
        zeros = torch.zeros(nblocks, 1, BLOCK, DQK)
        self.k_cache = self._up(zeros, ttnn.bfloat16, self._rep)
        self.v_cache = self._up(zeros, ttnn.bfloat16, self._rep)
        self.pt_dec = self._up(_pad8(self.page_table), ttnn.int32, self._rep, ttnn.ROW_MAJOR_LAYOUT)

    def decode_io(self):
        """Persistent decode inputs, PER MESH ROW: cur_pos int32 RM [32] and rope_idx uint32 RM [1,32] on every device of row r (positions of row r's 32 slots).
        Refresh with decode_set (positions [32] = same for all rows, or [rows,32])."""
        return decode_pos_io(self.mesh)

    def decode_set(self, io, positions):
        self._keep = decode_pos_set(self.mesh, io, positions)
        ttnn.synchronize_device(self.mesh)

    def decode_inputs(self, positions):
        io = self.decode_io()
        self.decode_set(io, positions)
        return io["cur"], io["idx"]

    def decode(self, x, cur_pos, rope_idx):
        """x: [1,1,B,4096] bf16 TILE per device, B = decode slots per mesh row (1..32). Returns [1,1,B,4096]."""
        B = x.shape[2]
        pt_dec, kv_mem, att_mem = self._dec_state(B)
        qkv = ttnn.linear(
            x,
            self.wqkv,
            dtype=ttnn.bfloat16,
            compute_kernel_config=_k_mm(),
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=NQ // TP, num_kv_heads=1, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)
        cos = ttnn.embedding(rope_idx, self.cos_tab, layout=ttnn.TILE_LAYOUT)  # [1,B,192]
        sin = ttnn.embedding(rope_idx, self.sin_tab, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.reshape(cos, (1, B, 1, DQK))
        sin = ttnn.reshape(sin, (1, B, 1, DQK))
        cosq = ttnn.repeat(cos, [1, 1, NQ // TP, 1])
        sinq = ttnn.repeat(sin, [1, 1, NQ // TP, 1])
        qr = self._rope(q, cosq, sinq)
        kr = self._rope(k, cos, sin)
        for t in (q, k, cos, sin, cosq, sinq):
            ttnn.deallocate(t)
        kr = ttnn.to_memory_config(kr, kv_mem)
        v = ttnn.to_memory_config(v, kv_mem)
        extra = {"cache_position_modulo": self.ring_blocks * BLOCK} if self.ring_blocks else {}
        ttnn.experimental.paged_update_cache(self.k_cache, kr, update_idxs_tensor=cur_pos, page_table=pt_dec, **extra)
        ttnn.experimental.paged_update_cache(self.v_cache, v, update_idxs_tensor=cur_pos, page_table=pt_dec, **extra)
        ttnn.deallocate(kr)
        ttnn.deallocate(v)
        kw = dict(
            cur_pos_tensor=cur_pos,
            scale=SCALE,
            program_config=self.dec_cfg,
            compute_kernel_config=_k_hifi4(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.cfg.window:
            kw.update(sliding_window_size=self.cfg.window, attention_sink=self.sink_dec, **extra)
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            qr, self.k_cache, self.v_cache, page_table_tensor=pt_dec, **kw
        )
        ttnn.deallocate(qr)
        o = ttnn.to_memory_config(o, att_mem)
        oc = ttnn.experimental.nlp_concat_heads_decode(o, num_heads=NQ // TP)
        ttnn.deallocate(o)
        oc = ttnn.to_memory_config(oc, ttnn.DRAM_MEMORY_CONFIG)
        part = ttnn.linear(
            oc, self.wo, dtype=ttnn.bfloat16, compute_kernel_config=_k_mm(), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(oc)
        return self._all_reduce(part)
