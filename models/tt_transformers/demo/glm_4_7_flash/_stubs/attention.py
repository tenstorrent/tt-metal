# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN MLA attention for GLM-4.7-Flash (TP-sharded).

GLM-4.7-Flash uses Multi-head Latent Attention (MLA):
  Q path:  x → q_a_proj (latent compress) → RMSNorm → q_b_proj (up-project, column-parallel)
  KV path: x → kv_a_proj (latent compress + rope) → split → RMSNorm → kv_b_proj (column-parallel)
  Output:  attention_output → o_proj (row-parallel) → all_reduce

Interleaved RoPE is handled by pre-permuting the rope output columns of
q_b_proj and kv_a_proj so the output is in split-half order, then applying
a standard split-half rotation with element-wise ttnn ops.
"""
from __future__ import annotations

import os

import torch

import ttnn
from models.tt_transformers.tt.attention import Attention  # kept per ADAPT requirement


class TtAttention:
    def __init__(self, device, torch_module) -> None:
        self.device = device

        self.num_heads = 20
        self.qk_nope_head_dim = 192
        self.qk_rope_head_dim = 64
        self.qk_head_dim = 256
        self.v_head_dim = 256
        self.q_lora_rank = 768
        self.kv_lora_rank = 512
        self.hidden_size = 2048
        self.scale = self.qk_head_dim**-0.5

        sd = torch_module.state_dict()

        # Interleaved → split-half permutation for 64-dim rope
        half = self.qk_rope_head_dim // 2
        rope_perm = torch.empty(self.qk_rope_head_dim, dtype=torch.long)
        for i in range(half):
            rope_perm[i] = 2 * i
            rope_perm[half + i] = 2 * i + 1

        # q_b_proj: permute rope columns within each head
        q_b_w = sd["q_b_proj.weight"].float()  # [5120, 768]
        q_b_w = q_b_w.view(self.num_heads, self.qk_head_dim, self.q_lora_rank)
        q_b_rope = q_b_w[:, self.qk_nope_head_dim :, :]  # [20, 64, 768]
        q_b_w = torch.concat([q_b_w[:, : self.qk_nope_head_dim, :], q_b_rope[:, rope_perm, :]], dim=1)
        Wq_b = q_b_w.reshape(self.num_heads * self.qk_head_dim, self.q_lora_rank).t().contiguous()

        # kv_a_proj: permute rope portion (last 64 output rows)
        kv_a_w = sd["kv_a_proj_with_mqa.weight"].float()  # [576, 2048]
        kv_a_latent = kv_a_w[: self.kv_lora_rank, :]  # [512, 2048]
        kv_a_rope = kv_a_w[self.kv_lora_rank :, :]  # [64, 2048]
        kv_a_w = torch.concat([kv_a_latent, kv_a_rope[rope_perm, :]], dim=0)
        Wkv_a = kv_a_w.t().contiguous()  # [2048, 576]

        # Other weights (standard transpose for ttnn.matmul: W^T so x @ W^T = x @ stored)
        Wq_a = sd["q_a_proj.weight"].float().t().contiguous()  # [2048, 768]
        Wkv_b = sd["kv_b_proj.weight"].float().t().contiguous()  # [512, 8960]
        Wo = sd["o_proj.weight"].float().t().contiguous()  # [5120, 2048]

        # RMS norm weights
        q_ln_w = sd["q_a_layernorm.weight"].float()
        kv_ln_w = sd["kv_a_layernorm.weight"].float()

        # TP config
        try:
            _is_mesh = isinstance(device, ttnn.MeshDevice)
        except AttributeError:
            _is_mesh = False
        _mesh_shape = list(device.shape) if _is_mesh else [1, 1]
        shard = bool(os.environ.get("TT_HW_PLANNER_SHARD_RUN")) and _is_mesh
        TP = _mesh_shape[-1] if shard else 1
        shard = shard and TP > 1 and (self.num_heads % TP == 0)
        self._shard = shard
        self._TP = TP
        self._tp_axis = len(_mesh_shape) - 1
        self._mesh_shape = _mesh_shape
        self._local_heads = self.num_heads // TP if shard else self.num_heads

        if shard:
            self._w_q_b = self._shd(Wq_b, 1, dtype=ttnn.bfloat8_b)
            self._w_kv_b = self._shd(Wkv_b, 1, dtype=ttnn.bfloat8_b)
            self._w_o = self._shd(Wo, 0, dtype=ttnn.bfloat8_b)
        else:
            self._w_q_b = self._rep(Wq_b, dtype=ttnn.bfloat8_b)
            self._w_kv_b = self._rep(Wkv_b, dtype=ttnn.bfloat8_b)
            self._w_o = self._rep(Wo, dtype=ttnn.bfloat8_b)

        self._w_q_a = self._rep(Wq_a, dtype=ttnn.bfloat8_b)
        self._w_kv_a = self._rep(Wkv_a, dtype=ttnn.bfloat8_b)

        self._q_ln_w = self._rep_1d(q_ln_w)
        self._kv_ln_w = self._rep_1d(kv_ln_w)

        # Pre-build causal masks at init time (avoid torch ops in forward)
        self._causal_masks = {}
        for _T in (32, 64, 128, 256, 512, 1024):
            _neg = torch.triu(torch.full((_T, _T), -1e9, dtype=torch.float32), diagonal=1)
            self._causal_masks[_T] = self._rep(_neg.reshape(1, 1, _T, _T))

        self._cached_rope = {}
        # HiFi4 + fp32 accumulation: the default (approx) rms_norm config under-scales by ~1%.
        self._norm_ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.ckc_proj = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    # ---- helpers ----

    def _is_mesh(self):
        try:
            return isinstance(self.device, ttnn.MeshDevice)
        except AttributeError:
            return hasattr(self.device, "get_device_ids")

    def _rep(self, t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32):
        if self._is_mesh():
            return ttnn.from_torch(
                t,
                dtype=dtype,
                layout=layout,
                device=self.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.device)

    def _shd(self, t, dim, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=self.device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.device, mesh_shape=self._mesh_shape, dims=(None, dim)),
        )

    def _rep_1d(self, w):
        w4d = w.reshape(1, 1, -1, 32)
        if self._is_mesh():
            return ttnn.from_torch(
                w4d,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
        return ttnn.from_torch(
            w4d,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _fp32(self, t):
        if isinstance(t, ttnn.Tensor):
            if t.dtype != ttnn.float32:
                return ttnn.typecast(t, ttnn.float32)
            return t
        return self._rep(t.float())

    def _to_heads(self, t, B, T, n, d):
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        rm = ttnn.reshape(rm, [B, T, n, d])
        rm = ttnn.permute(rm, (0, 2, 1, 3))
        return ttnn.to_layout(rm, ttnn.TILE_LAYOUT)

    def _from_heads(self, t, B, T, n, d):
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        rm = ttnn.permute(rm, (0, 2, 1, 3))
        rm = ttnn.reshape(rm, [B, T, n * d])
        return ttnn.to_layout(rm, ttnn.TILE_LAYOUT)

    def _get_causal_mask(self, T):
        m = self._causal_masks.get(T)
        if m is None:
            neg = torch.triu(torch.full((T, T), -1e9, dtype=torch.float32), diagonal=1)
            m = self._rep(neg.reshape(1, 1, T, T))
            self._causal_masks[T] = m
        return m

    def _prepare_rope_tensors(self, pos_emb):
        cos_t, sin_t = pos_emb
        if isinstance(cos_t, torch.Tensor):
            key = id(cos_t)
            cached = self._cached_rope.get(key)
            if cached is not None:
                return cached
            half_d = cos_t.shape[-1] // 2
            cos_half = cos_t[..., :half_d].float()
            sin_half = sin_t[..., :half_d].float()
            while cos_half.ndim < 4:
                cos_half = cos_half.unsqueeze(0)
                sin_half = sin_half.unsqueeze(0)
            cos_tt = self._rep(cos_half.contiguous())
            sin_tt = self._rep(sin_half.contiguous())
            self._cached_rope[key] = (cos_tt, sin_tt)
            return cos_tt, sin_tt
        return cos_t, sin_t

    def _rms_norm_manual(self, x, weight):
        x_sq = ttnn.multiply(x, x)
        mean_sq = ttnn.mean(x_sq, dim=-1, keepdim=True)
        ttnn.deallocate(x_sq)
        eps_t = ttnn.full_like(mean_sq, 1e-6)
        mean_sq_eps = ttnn.add(mean_sq, eps_t)
        ttnn.deallocate(mean_sq)
        ttnn.deallocate(eps_t)
        rms_inv = ttnn.rsqrt(mean_sq_eps)
        ttnn.deallocate(mean_sq_eps)
        normed = ttnn.multiply(x, rms_inv)
        ttnn.deallocate(rms_inv)
        return ttnn.multiply(normed, weight)

    def _apply_rope_split_half(self, x, cos_half, sin_half):
        shape = list(x.shape)
        D = shape[-1]
        half_d = D // 2

        x1 = ttnn.slice(x, [0, 0, 0, 0], [shape[0], shape[1], shape[2], half_d])
        x2 = ttnn.slice(x, [0, 0, 0, half_d], [shape[0], shape[1], shape[2], D])

        r1 = ttnn.subtract(
            ttnn.matmul(x1, cos_half) if False else ttnn.multiply(x1, cos_half),
            ttnn.multiply(x2, sin_half),
        )
        r2 = ttnn.add(
            ttnn.multiply(x2, cos_half),
            ttnn.multiply(x1, sin_half),
        )
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)

        rm1 = ttnn.to_layout(r1, ttnn.ROW_MAJOR_LAYOUT)
        rm2 = ttnn.to_layout(r2, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(r1)
        ttnn.deallocate(r2)
        result = ttnn.concat([rm1, rm2], dim=-1)
        ttnn.deallocate(rm1)
        ttnn.deallocate(rm2)
        return ttnn.to_layout(result, ttnn.TILE_LAYOUT)

    # ---- forward ----

    # ---- KV-cached path (prefill writes the cache, decode reads + appends) ----
    #
    # Token layout is [1, 1, N, hidden] with N = B*T (prefill, row b*T+t) or N = B (decode).
    # kv_cache = (k, v), each [B, local_heads, C, 256] bf16, persistent on device.
    # cos/sin come from the rotary_embedding stub: [B, T, 64] (prefill) or [B, 1, 64] (decode).

    def _rms(self, x, w, eps):
        try:
            return ttnn.rms_norm(x, weight=w, epsilon=eps, compute_kernel_config=self._norm_ckc)
        except Exception:
            xb = ttnn.typecast(x, ttnn.bfloat16)
            nb = ttnn.rms_norm(xb, weight=w, epsilon=eps, compute_kernel_config=self._norm_ckc)
            return ttnn.typecast(nb, ttnn.float32)

    def _mla_project(self, hs, N):
        q_c = ttnn.matmul(hs, self._w_q_a, compute_kernel_config=self.ckc_proj)
        q_c = self._rms(q_c, self._q_ln_w, 1e-6)
        q = ttnn.matmul(q_c, self._w_q_b, compute_kernel_config=self.ckc_proj, dtype=ttnn.bfloat16)
        ttnn.deallocate(q_c)
        kv_c = ttnn.matmul(hs, self._w_kv_a, compute_kernel_config=self.ckc_proj)
        k_lat = ttnn.slice(kv_c, [0, 0, 0, 0], [1, 1, N, self.kv_lora_rank])
        k_pe = ttnn.slice(kv_c, [0, 0, 0, self.kv_lora_rank], [1, 1, N, self.kv_lora_rank + self.qk_rope_head_dim])
        ttnn.deallocate(kv_c)
        k_lat = self._rms(k_lat, self._kv_ln_w, 1e-6)
        kv = ttnn.matmul(k_lat, self._w_kv_b, compute_kernel_config=self.ckc_proj, dtype=ttnn.bfloat16)
        ttnn.deallocate(k_lat)
        return q, kv, ttnn.typecast(k_pe, ttnn.bfloat16)

    def _rope_tile(self, x, cos, sin):
        # split-half rotation (weights were permuted from interleaved at build); cos/sin are the
        # HF 64-wide tables whose halves are equal, so the first half is the per-pair angle.
        s = list(x.shape)
        hd = self.qk_rope_head_dim // 2
        c = ttnn.slice(cos, [0, 0, 0, 0], [cos.shape[0], cos.shape[1], cos.shape[2], hd])
        sn = ttnn.slice(sin, [0, 0, 0, 0], [sin.shape[0], sin.shape[1], sin.shape[2], hd])
        x1 = ttnn.slice(x, [0, 0, 0, 0], [s[0], s[1], s[2], hd])
        x2 = ttnn.slice(x, [0, 0, 0, hd], [s[0], s[1], s[2], 2 * hd])
        r1 = ttnn.subtract(ttnn.multiply(x1, c), ttnn.multiply(x2, sn))
        r2 = ttnn.add(ttnn.multiply(x2, c), ttnn.multiply(x1, sn))
        return ttnn.concat([r1, r2], dim=-1)

    def _rm_reshape(self, t, shape):
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        rm = ttnn.reshape(rm, shape)
        return ttnn.to_layout(rm, ttnn.TILE_LAYOUT)

    def _kv_update_memcfg(self, B):
        mc = getattr(self, "_kv_upd_mc", {}).get(B)
        if mc is None:
            grid = self.device.compute_with_storage_grid_size()
            crs = ttnn.num_cores_to_corerangeset(B, grid, row_wise=True)
            mc = ttnn.create_sharded_memory_config(
                shape=(32, self.qk_head_dim),
                core_grid=crs,
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            )
            self._kv_upd_mc = {**getattr(self, "_kv_upd_mc", {}), B: mc}
        return mc

    def attend_cached(self, q, kv, k_pe, cos, sin, kv_cache, mode, batch, seq_len, cur_pos=None):
        """MLA core on projected q/kv/k_pe ([1,1,N,*] bf16). Returns [1,1,N,H*v_head] bf16."""
        B, T, H = batch, seq_len, self._local_heads
        nope, rope, vd, qd = self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim, self.qk_head_dim
        k_cache, v_cache = kv_cache
        if mode == "prefill":
            qh = ttnn.permute(self._rm_reshape(q, [B, T, H, qd]), (0, 2, 1, 3))  # [B,H,T,256]
            kvh = ttnn.permute(self._rm_reshape(kv, [B, T, H, nope + vd]), (0, 2, 1, 3))  # [B,H,T,448]
            kpe = self._rm_reshape(k_pe, [B, 1, T, rope])
            cos4 = self._rm_reshape(cos, [B, 1, T, rope])
            sin4 = self._rm_reshape(sin, [B, 1, T, rope])
        else:
            qh = self._rm_reshape(q, [1, B, H, qd])  # [1,B,H,256]
            kvh = self._rm_reshape(kv, [1, B, H, nope + vd])
            kpe = self._rm_reshape(k_pe, [1, B, 1, rope])
            cos4 = self._rm_reshape(cos, [1, B, 1, rope])
            sin4 = self._rm_reshape(sin, [1, B, 1, rope])
        d0, d1, d2 = list(qh.shape)[:3]
        q_nope = ttnn.slice(qh, [0, 0, 0, 0], [d0, d1, d2, nope])
        q_rope = ttnn.slice(qh, [0, 0, 0, nope], [d0, d1, d2, qd])
        k_nope = ttnn.slice(kvh, [0, 0, 0, 0], [d0, d1, d2, nope])
        v = ttnn.slice(kvh, [0, 0, 0, nope], [d0, d1, d2, nope + vd])
        q_rope = self._rope_tile(q_rope, cos4, sin4)
        kpe = self._rope_tile(kpe, cos4, sin4)
        kpe = ttnn.repeat(kpe, ttnn.Shape([1, H, 1, 1]) if mode == "prefill" else ttnn.Shape([1, 1, H, 1]))
        Q = ttnn.concat([q_nope, q_rope], dim=-1)
        K = ttnn.concat([k_nope, kpe], dim=-1)
        if mode == "prefill":
            out = ttnn.transformer.scaled_dot_product_attention(
                Q, K, v, is_causal=True, scale=self.scale, compute_kernel_config=self.ckc
            )  # [B,H,T,256]
            C = k_cache.shape[2]
            Kc = K if T == C else ttnn.pad(K, [(0, 0), (0, 0), (0, C - T), (0, 0)], 0.0)
            Vc = v if T == C else ttnn.pad(v, [(0, 0), (0, 0), (0, C - T), (0, 0)], 0.0)
            ttnn.copy(Kc, k_cache)
            ttnn.copy(Vc, v_cache)
            out = ttnn.permute(out, (0, 2, 1, 3))  # [B,T,H,256]
            return self._rm_reshape(out, [1, 1, B * T, H * vd])
        mc = self._kv_update_memcfg(B)
        ttnn.experimental.paged_update_cache(k_cache, ttnn.to_memory_config(K, mc), update_idxs_tensor=cur_pos)
        ttnn.experimental.paged_update_cache(v_cache, ttnn.to_memory_config(v, mc), update_idxs_tensor=cur_pos)
        out = ttnn.transformer.scaled_dot_product_attention_decode(
            Q, k_cache, v_cache, cur_pos_tensor=cur_pos, scale=self.scale, compute_kernel_config=self.ckc
        )  # [1,B,H,256]
        return self._rm_reshape(out, [1, 1, B, H * vd])

    def forward_cached(self, hidden_states, position_embeddings, kv_cache, mode, batch, seq_len, cur_pos=None):
        hs = hidden_states
        if isinstance(hs, torch.Tensor):
            hs = ttnn.from_torch(
                hs.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        if hs.dtype != ttnn.bfloat16:
            hs = ttnn.typecast(hs, ttnn.bfloat16)
        if hs.layout != ttnn.TILE_LAYOUT:
            hs = ttnn.to_layout(hs, ttnn.TILE_LAYOUT)
        N = hs.shape[-2]
        q, kv, k_pe = self._mla_project(hs, N)
        cos, sin = position_embeddings
        attn = self.attend_cached(q, kv, k_pe, cos, sin, kv_cache, mode, batch, seq_len, cur_pos)
        out = ttnn.matmul(attn, self._w_o, compute_kernel_config=self.ckc_proj, dtype=ttnn.bfloat16)
        ttnn.deallocate(attn)
        if self._shard:
            out = ttnn.all_reduce(out, cluster_axis=self._tp_axis, topology=ttnn.Topology.Linear)
        return out

    def __call__(self, hidden_states, **kwargs):
        if kwargs.get("kv_cache") is not None:
            return self.forward_cached(
                hidden_states,
                kwargs["position_embeddings"],
                kwargs["kv_cache"],
                kwargs["mode"],
                kwargs["batch"],
                kwargs["seq_len"],
                kwargs.get("cur_pos"),
            )
        hs = self._fp32(hidden_states)
        if hs.layout != ttnn.TILE_LAYOUT:
            hs = ttnn.to_layout(hs, ttnn.TILE_LAYOUT)
        shape = list(hs.shape)

        if len(shape) == 3:
            B, T, _H = shape
        else:
            B, T, _H = shape[0], shape[-2], shape[-1]

        H = self._local_heads
        nope_d = self.qk_nope_head_dim
        rope_d = self.qk_rope_head_dim
        v_d = self.v_head_dim

        # ==== Q path ====
        q_c = ttnn.matmul(hs, self._w_q_a, compute_kernel_config=self.ckc)
        try:
            q_c = ttnn.rms_norm(q_c, weight=self._q_ln_w, epsilon=1e-6, compute_kernel_config=self._norm_ckc)
        except Exception:
            q_c_bf = ttnn.typecast(q_c, ttnn.bfloat16)
            ttnn.deallocate(q_c)
            q_c_bf = ttnn.rms_norm(q_c_bf, weight=self._q_ln_w, epsilon=1e-6, compute_kernel_config=self._norm_ckc)
            q_c = ttnn.typecast(q_c_bf, ttnn.float32)
            ttnn.deallocate(q_c_bf)

        q = ttnn.matmul(q_c, self._w_q_b, compute_kernel_config=self.ckc)
        ttnn.deallocate(q_c)

        q_heads = self._to_heads(q, B, T, H, self.qk_head_dim)
        ttnn.deallocate(q)

        q_nope = ttnn.slice(q_heads, [0, 0, 0, 0], [B, H, T, nope_d])
        q_rope = ttnn.slice(q_heads, [0, 0, 0, nope_d], [B, H, T, nope_d + rope_d])
        ttnn.deallocate(q_heads)

        # ==== KV path ====
        kv_c = ttnn.matmul(hs, self._w_kv_a, compute_kernel_config=self.ckc)
        ttnn.deallocate(hs)

        k_latent = ttnn.slice(kv_c, [0, 0, 0], [B, T, self.kv_lora_rank])
        k_pe_raw = ttnn.slice(kv_c, [0, 0, self.kv_lora_rank], [B, T, self.kv_lora_rank + rope_d])
        ttnn.deallocate(kv_c)

        try:
            k_latent = ttnn.rms_norm(k_latent, weight=self._kv_ln_w, epsilon=1e-6, compute_kernel_config=self._norm_ckc)
        except Exception:
            k_lat_bf = ttnn.typecast(k_latent, ttnn.bfloat16)
            ttnn.deallocate(k_latent)
            k_lat_bf = ttnn.rms_norm(k_lat_bf, weight=self._kv_ln_w, epsilon=1e-6, compute_kernel_config=self._norm_ckc)
            k_latent = ttnn.typecast(k_lat_bf, ttnn.float32)
            ttnn.deallocate(k_lat_bf)

        kv = ttnn.matmul(k_latent, self._w_kv_b, compute_kernel_config=self.ckc)
        ttnn.deallocate(k_latent)

        kv_heads = self._to_heads(kv, B, T, H, nope_d + v_d)
        ttnn.deallocate(kv)

        k_nope = ttnn.slice(kv_heads, [0, 0, 0, 0], [B, H, T, nope_d])
        v_heads = ttnn.slice(kv_heads, [0, 0, 0, nope_d], [B, H, T, nope_d + v_d])
        ttnn.deallocate(kv_heads)

        # ==== RoPE ====
        rm_kpe = ttnn.to_layout(k_pe_raw, ttnn.ROW_MAJOR_LAYOUT)
        rm_kpe = ttnn.reshape(rm_kpe, [B, 1, T, rope_d])
        k_pe_4d = ttnn.to_layout(rm_kpe, ttnn.TILE_LAYOUT)
        ttnn.deallocate(k_pe_raw)

        pos_emb = kwargs.get("position_embeddings")
        if pos_emb is not None:
            cos_tt, sin_tt = self._prepare_rope_tensors(pos_emb)
            q_rope = self._apply_rope_split_half(q_rope, cos_tt, sin_tt)
            k_pe_4d = self._apply_rope_split_half(k_pe_4d, cos_tt, sin_tt)

        # Expand k_pe from [B,1,T,64] to [B,H,T,64]
        k_pe_rm = ttnn.to_layout(k_pe_4d, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(k_pe_4d)
        k_pe_parts = [k_pe_rm] * H
        k_pe_exp = ttnn.concat(k_pe_parts, dim=1)
        ttnn.deallocate(k_pe_rm)
        k_pe_exp = ttnn.to_layout(k_pe_exp, ttnn.TILE_LAYOUT)

        # Assemble full Q and K
        q_nope_rm = ttnn.to_layout(q_nope, ttnn.ROW_MAJOR_LAYOUT)
        q_rope_rm = ttnn.to_layout(q_rope, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(q_nope)
        ttnn.deallocate(q_rope)
        q_full_rm = ttnn.concat([q_nope_rm, q_rope_rm], dim=-1)
        ttnn.deallocate(q_nope_rm)
        ttnn.deallocate(q_rope_rm)
        q_full = ttnn.to_layout(q_full_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(q_full_rm)

        k_nope_rm = ttnn.to_layout(k_nope, ttnn.ROW_MAJOR_LAYOUT)
        k_pe_exp_rm = ttnn.to_layout(k_pe_exp, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(k_nope)
        ttnn.deallocate(k_pe_exp)
        k_full_rm = ttnn.concat([k_nope_rm, k_pe_exp_rm], dim=-1)
        ttnn.deallocate(k_nope_rm)
        ttnn.deallocate(k_pe_exp_rm)
        k_full = ttnn.to_layout(k_full_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(k_full_rm)

        # ==== Attention ====
        Kt = ttnn.transpose(k_full, -2, -1)
        ttnn.deallocate(k_full)
        scores = ttnn.matmul(q_full, Kt, compute_kernel_config=self.ckc)
        ttnn.deallocate(q_full)
        ttnn.deallocate(Kt)
        scores = ttnn.multiply(scores, self.scale)
        scores = ttnn.add(scores, self._get_causal_mask(T))
        probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.ckc, numeric_stable=True)
        ttnn.deallocate(scores)

        attn_out = ttnn.matmul(probs, v_heads, compute_kernel_config=self.ckc)
        ttnn.deallocate(probs)
        ttnn.deallocate(v_heads)

        # ==== O projection ====
        attn_flat = self._from_heads(attn_out, B, T, H, v_d)
        ttnn.deallocate(attn_out)

        out = ttnn.matmul(attn_flat, self._w_o, compute_kernel_config=self.ckc)
        ttnn.deallocate(attn_flat)
        if self._shard:
            out = ttnn.all_reduce(out, cluster_axis=self._tp_axis, topology=ttnn.Topology.Linear)
        return ttnn.typecast(out, ttnn.bfloat16)


def build(device, torch_module=None):
    return TtAttention.build(device, torch_module)


def attention(device, torch_module=None):
    return TtAttention.build(device, torch_module)
