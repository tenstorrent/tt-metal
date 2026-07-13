# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `image3_decoder_layer` (HunyuanImage3DecoderLayer) of
tencent/HunyuanImage-3.0.

A full decoder block:

    residual = x
    x = input_layernorm(x)                    # RMSNorm
    x = self_attn(x, custom_pos_emb)          # fused-QKV GQA + HF rope + qk-norm + SDPA + o_proj
    x = residual + x
    residual = x
    x = post_attention_layernorm(x)           # RMSNorm
    x = mlp(x)                                 # MoE: shared SwiGLU + top-8 gate + 64 expert SwiGLUs
    x = residual + x

Everything runs in TTNN ops on device; weights are extracted from the HF
reference module at `build()` time. No torch reference is invoked at forward
time.

Tensor-parallel (TP) shard path
-------------------------------
When `build()` is handed a `ttnn.MeshDevice` (the `run_component(mode='shard')`
harness supplies a `MeshShape(1, TP)` mesh) the layer graduates DIRECTLY
tensor-parallel. The math is byte-for-byte the single-device math, only the
placement changes; the gathered output equals the single-device golden:

  * attention QKV is COLUMN-parallel by kv-group -- device d owns kv-head(s)
    `[d*kv_per_dev : (d+1)*kv_per_dev]` and their q-heads. The fused-qkv rows
    are re-ordered into per-device `[q..|k..|v..]` blocks so a contiguous
    `ShardTensorToMesh(dim=-1)` split hands each device exactly its heads;
    SDPA runs locally on the per-device head count.
  * attention O-proj is ROW-parallel -- its INPUT features are split
    `ShardTensorToMesh(dim=0)` (the contiguous split lines up with the local
    head order), each device produces a partial sum, and an all-reduce
    (all_gather + sum) reassembles the full projection.
  * the MoE routed experts are EXPERT-parallel -- a disjoint `num_experts/TP`
    subset of experts per device (`ShardTensorToMesh(dim=0)` over a stacked
    expert-weight tensor). The router is computed REPLICATED (softmax/top-k
    need all experts); each device selects ITS expert columns from the
    replicated router via a sharded one-hot selection matmul, combines its
    experts, and an all-reduce sums the per-device partials.
  * the shared expert, router `wg`, both RMSNorms, the qk-norms and the rope
    cos/sin tables stay REPLICATED (elementwise / lookup / small-matmul roles).

The single-device path (no mesh) is unchanged and still composes the graduated
`mo_e` stub.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.vision.generative.hunyuanimage_3_0._stubs import mo_e as _mo_e

HF_MODEL_ID = "tencent/HunyuanImage-3.0"

# Max sequence the decode KV cache is allocated for (prefill_len + generated).
# Tile-aligned. Tiny on device (per-chip [B, 1_kv_head, max_seq, 128] bf16).
_DECODE_MAX_SEQ = 512

# Decode matmul compute config. Decode is overhead/latency-bound at batch=1, and
# the default (HiFi4, 4-pass) is the slowest; LoFi (1-pass) is a large step-time
# win (tt_transformers runs decode matmuls at reduced fidelity). PCC/token-gated.
_DECODE_MM_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


def _to_ttnn(t: torch.Tensor, device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t.to(torch.float32),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _linear_weight(w: torch.Tensor, device, dtype=ttnn.bfloat16):
    """nn.Linear stores weight as [out, in]; ttnn.linear(x, W) computes x @ W,
    so upload the transpose [in, out]."""
    return _to_ttnn(w.t().contiguous(), device, dtype=dtype)


# ----------------------------------------------------------------------------
# mesh helpers (TP shard path)
# ----------------------------------------------------------------------------
def _is_mesh_device(device) -> bool:
    try:
        if isinstance(device, ttnn.MeshDevice):
            return True
    except AttributeError:
        pass
    return hasattr(device, "get_num_devices") and hasattr(device, "get_device_ids")


class _TtDecoderLayer:
    def __init__(self, device, torch_module):
        self.device = device
        self.is_mesh = _is_mesh_device(device)
        self.num_devices = int(device.get_num_devices()) if self.is_mesh else 1

        cfg = torch_module.self_attn

        self.num_heads = int(cfg.num_heads)
        self.num_kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(cfg.head_dim)
        self.groups = self.num_heads // self.num_kv_heads
        self.hidden_size = int(cfg.hidden_size)
        self.use_qk_norm = bool(getattr(cfg, "use_qk_norm", False))
        self.use_rope = bool(getattr(cfg, "use_rotary_pos_emb", False))
        self.scale = 1.0 / (self.head_dim**0.5)

        self.eps = float(torch_module.input_layernorm.variance_epsilon)

        # Gate 2 real-invocation counter.
        self.num_calls = 0

        if self.is_mesh:
            # This 6U Blackhole Galaxy only brings the fabric up on the FULL
            # physical mesh, so the harness opens MeshShape(rows, cols) (e.g.
            # (8, 4)) rather than a flat TP-device sub-mesh. Run TP across the
            # mesh axis whose length divides the head/expert counts (the length-
            # 8 axis) and DP-REPLICATE across the other axis. Collectives are
            # confined to the TP axis via cluster_axis; replicated DP columns
            # each compute the identical full result, so the gathered device-0
            # output equals the single-device golden.
            self.mesh_shape = tuple(int(x) for x in device.shape)
            self.tp_axis = self._pick_tp_axis()
            self.tp = int(self.mesh_shape[self.tp_axis])
            self._build_sharded(torch_module)
        else:
            self._build_single(torch_module)

    # ------------------------------------------------------------------
    def _pick_tp_axis(self) -> int:
        """Pick the mesh axis to run tensor-parallel over: the longest axis
        whose length (>1) divides num_kv_heads, so head-parallel attention (and,
        since num_experts is a multiple of num_kv_heads here, expert-parallel
        MoE) split cleanly across it."""
        best = None
        for ax, sz in enumerate(self.mesh_shape):
            if sz > 1 and self.num_kv_heads % sz == 0:
                if best is None or sz > self.mesh_shape[best]:
                    best = ax
        return 0 if best is None else best

    # ------------------------------------------------------------------
    # mesh weight-upload helpers (2D-mesh aware: shard across the TP axis,
    # replicate across the DP axis)
    # ------------------------------------------------------------------
    def _repl(self, t, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
        """Upload REPLICATED onto every device of the mesh."""
        return ttnn.from_torch(
            t.to(torch.float32),
            dtype=dtype,
            layout=layout,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

    def _shard(self, t, dim, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
        """Upload SHARDED: split tensor `dim` across the TP mesh axis, replicate
        across the other (DP) mesh axis."""
        dims = [None, None]
        dims[self.tp_axis] = dim
        return ttnn.from_torch(
            t.to(torch.float32),
            dtype=dtype,
            layout=layout,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.device, dims=tuple(dims), mesh_shape=self.mesh_shape),
        )

    def _shard_linear(self, w, dim, dtype=ttnn.bfloat16):
        """nn.Linear weight [out, in] -> upload transpose [in, out] sharded on
        `dim` of the TRANSPOSED tensor (dim=0 = input feats / row-parallel,
        dim=-1 = output feats / column-parallel)."""
        return self._shard(w.t().contiguous(), dim, dtype=dtype)

    # ------------------------------------------------------------------
    # single-device build (unchanged)
    # ------------------------------------------------------------------
    def _build_single(self, torch_module):
        device = self.device
        cfg = torch_module.self_attn

        # --- norms ---
        self.input_ln_w = _to_ttnn(torch_module.input_layernorm.weight.reshape(1, 1, 1, -1), device)
        self.post_ln_w = _to_ttnn(torch_module.post_attention_layernorm.weight.reshape(1, 1, 1, -1), device)
        if self.use_qk_norm:
            self.q_norm_w = _to_ttnn(cfg.query_layernorm.weight.reshape(1, 1, 1, -1), device)
            self.k_norm_w = _to_ttnn(cfg.key_layernorm.weight.reshape(1, 1, 1, -1), device)

        # --- attention: permute fused-QKV rows so output is [q | k | v] contiguous ---
        # HF layout: qkv[b,q,6144] -> reshape [b,q,kv=8,groups+2=6,hd=128], split [4,1,1].
        # flat row for (kv, slot, d) = kv*768 + slot*128 + d.  q uses slots 0..3, k=4, v=5.
        # q head index = kv*groups + group (matches HF reshape->transpose order).
        nKV, g, hd = self.num_kv_heads, self.groups, self.head_dim
        block = (g + 2) * hd
        q_rows, k_rows, v_rows = [], [], []
        for h in range(self.num_heads):
            kv = h // g
            grp = h % g
            base = kv * block + grp * hd
            q_rows.extend(range(base, base + hd))
        for kv in range(nKV):
            base = kv * block
            k_rows.extend(range(base + g * hd, base + g * hd + hd))
            v_rows.extend(range(base + (g + 1) * hd, base + (g + 1) * hd + hd))
        perm = q_rows + k_rows + v_rows
        qkv_w = cfg.qkv_proj.weight[perm, :].contiguous()  # [6144, 4096]
        self.qkv_w = _linear_weight(qkv_w, device)
        self.o_w = _linear_weight(cfg.o_proj.weight, device)

        # --- MoE: composed graduated HunyuanMoE port ---
        # HunyuanImage3DecoderLayer.mlp == HunyuanMoE. Delegate the whole MoE
        # (shared SwiGLU + top-8 gate + 64 routed expert SwiGLUs) to the
        # graduated `mo_e` stub, which itself composes the graduated
        # `top_k_gate`. This is the real HF module nesting.
        self.moe = _mo_e.build(device, torch_module.mlp)

    # ------------------------------------------------------------------
    # tensor-parallel build (mesh)
    # ------------------------------------------------------------------
    def _build_sharded(self, torch_module):
        tp = self.tp
        cfg = torch_module.self_attn
        g, hd = self.groups, self.head_dim

        assert (
            self.num_kv_heads % tp == 0
        ), f"TP={tp} must divide num_kv_heads={self.num_kv_heads} for head-parallel attention"
        kv_per_dev = self.num_kv_heads // tp
        q_per_dev = self.num_heads // tp
        self.tp_num_heads = q_per_dev
        self.tp_num_kv_heads = kv_per_dev

        # --- norms / qk-norms / (rope tables handled at forward) -> REPLICATED ---
        self.input_ln_w = self._repl(torch_module.input_layernorm.weight.reshape(1, 1, 1, -1))
        self.post_ln_w = self._repl(torch_module.post_attention_layernorm.weight.reshape(1, 1, 1, -1))
        if self.use_qk_norm:
            self.q_norm_w = self._repl(cfg.query_layernorm.weight.reshape(1, 1, 1, -1))
            self.k_norm_w = self._repl(cfg.key_layernorm.weight.reshape(1, 1, 1, -1))

        # --- QKV column-parallel: reorder the fused-qkv rows into per-device
        #     [q(q_per_dev) | k(kv_per_dev) | v(kv_per_dev)] blocks so a
        #     contiguous dim=-1 shard hands device d exactly its heads. ---
        block = (g + 2) * hd
        sharded_rows = []
        for d in range(tp):
            kvs = range(d * kv_per_dev, (d + 1) * kv_per_dev)
            for kv in kvs:  # q heads of this device's kv heads
                for grp in range(g):
                    base = kv * block + grp * hd
                    sharded_rows.extend(range(base, base + hd))
            for kv in kvs:  # k heads
                base = kv * block + g * hd
                sharded_rows.extend(range(base, base + hd))
            for kv in kvs:  # v heads
                base = kv * block + (g + 1) * hd
                sharded_rows.extend(range(base, base + hd))
        qkv_w = cfg.qkv_proj.weight[sharded_rows, :].contiguous()  # [6144, hidden]
        self.qkv_w = self._shard_linear(qkv_w, dim=-1)  # column-parallel (output feats)

        # O-proj row-parallel: split INPUT features. The concat-heads output
        # feature order is head-major, and device d holds the contiguous block
        # [d*q_per_dev*hd : (d+1)*q_per_dev*hd], so a plain dim=0 (input) shard
        # of the transposed weight lines up exactly.
        self.o_w = self._shard_linear(cfg.o_proj.weight, dim=0)  # row-parallel (input feats)

        # --- decode KV cache (incremental single-token decode) ---
        # Full [B, num_kv_heads, max_seq, head_dim] sharded on the head dim across
        # the TP axis -> each chip owns its 1 local KV head: local [B,1,max_seq,hd].
        # Matches how the QKV weights are head-sharded, so the decode SDPA runs
        # purely on local shards. Allocated once; None until a decode run seeds it.
        self.kv_batch = 1  # single-user decode (batch=1)
        k0 = torch.zeros(self.kv_batch, self.num_kv_heads, _DECODE_MAX_SEQ, hd)
        self.k_cache = self._shard(k0, dim=1, dtype=ttnn.bfloat16)
        self.v_cache = self._shard(k0.clone(), dim=1, dtype=ttnn.bfloat16)

        # --- MoE: composed graduated HunyuanMoE port (expert-parallel) ---
        # Exactly as the single-device path: HunyuanImage3DecoderLayer.mlp ==
        # HunyuanMoE. Delegate the whole MoE to the graduated `mo_e` stub, which
        # (on a mesh) shards its 64 routed experts expert-parallel and itself
        # composes the graduated `top_k_gate`. This keeps both mo_e AND
        # top_k_gate on the one real sharded forward path (Gate 2) and avoids a
        # duplicate ~10 GB expert-weight set on device.
        self.moe = _mo_e.build(self.device, torch_module.mlp)

    # ------------------------------------------------------------------
    def _mesh_reduce(self, x):
        """All-reduce (sum) a per-device partial across the TP mesh axis:
        all_gather along the TP axis (cluster_axis) then sum, so every device on
        that axis holds the full result. Confined to the TP axis so the
        DP-replicated columns each keep their own identical full result."""
        g = ttnn.all_gather(x, dim=0, cluster_axis=self.tp_axis, num_links=1, topology=ttnn.Topology.Linear)
        r = ttnn.sum(g, dim=0, keepdim=True)
        ttnn.deallocate(g)
        return r

    def _swiglu(self, x, gu_w, down_w, inter):
        gu = ttnn.linear(x, gu_w)
        x1 = ttnn.slice(gu, [0, 0, 0], [gu.shape[0], gu.shape[1], inter])
        x2 = ttnn.slice(gu, [0, 0, inter], [gu.shape[0], gu.shape[1], 2 * inter])
        ttnn.deallocate(gu)
        act = ttnn.multiply(x1, ttnn.silu(x2))
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        out = ttnn.linear(act, down_w)
        ttnn.deallocate(act)
        return out

    def _rope_qk(self, q, k, custom_pos_emb, *, replicate):
        cos_t, sin_t = custom_pos_emb
        if isinstance(cos_t, ttnn.Tensor):
            cos, sin = cos_t, sin_t
        elif replicate:
            cos = self._repl(cos_t.reshape(1, 1, cos_t.shape[-2], cos_t.shape[-1]))
            sin = self._repl(sin_t.reshape(1, 1, sin_t.shape[-2], sin_t.shape[-1]))
        else:
            cos = _to_ttnn(cos_t.reshape(1, 1, cos_t.shape[-2], cos_t.shape[-1]), self.device)
            sin = _to_ttnn(sin_t.reshape(1, 1, sin_t.shape[-2], sin_t.shape[-1]), self.device)
        q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=False)
        k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)
        return q, k

    # ------------------------------------------------------------------
    def _attention(self, x, custom_pos_emb):
        S = x.shape[1]
        qkv = ttnn.linear(x, self.qkv_w)  # [1, S, 6144] = q|k|v
        # nlp_create_qkv_heads expects a 4D [B, 1, S, dim] fused-qkv tensor.
        qkv = ttnn.reshape(qkv, [1, 1, S, qkv.shape[-1]])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        # q: [1, num_heads, S, hd]; k,v: [1, num_kv_heads, S, hd]

        # HF rope (rotate_half) THEN qk-norm, matching the reference order.
        if self.use_rope and custom_pos_emb is not None:
            q, k = self._rope_qk(q, k, custom_pos_emb, replicate=False)

        if self.use_qk_norm:
            q = ttnn.rms_norm(q, epsilon=self.eps, weight=self.q_norm_w)
            k = ttnn.rms_norm(k, epsilon=self.eps, weight=self.k_norm_w)

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        # attn: [1, num_heads, S, hd] -> concat heads -> [1, 1, S, hidden]
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.reshape(attn, [1, S, self.hidden_size])
        out = ttnn.linear(attn, self.o_w)
        ttnn.deallocate(attn)
        return out

    # ------------------------------------------------------------------
    def _attention_sharded(self, x, custom_pos_emb):
        S = x.shape[1]
        qkv = ttnn.linear(x, self.qkv_w)  # [1, S, tp_qkv] per device
        qkv = ttnn.reshape(qkv, [1, 1, S, qkv.shape[-1]])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.tp_num_heads,
            num_kv_heads=self.tp_num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)

        if self.use_rope and custom_pos_emb is not None:
            q, k = self._rope_qk(q, k, custom_pos_emb, replicate=True)

        if self.use_qk_norm:
            q = ttnn.rms_norm(q, epsilon=self.eps, weight=self.q_norm_w)
            k = ttnn.rms_norm(k, epsilon=self.eps, weight=self.k_norm_w)

        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.scale)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.reshape(attn, [1, S, self.tp_num_heads * self.head_dim])
        out_partial = ttnn.linear(attn, self.o_w)  # [1, S, hidden] partial sum
        ttnn.deallocate(attn)
        out = self._mesh_reduce(out_partial)  # all-reduce -> full o_proj, replicated
        ttnn.deallocate(out_partial)
        return out

    # ------------------------------------------------------------------
    def _attention_decode(self, x, cos, sin, current_pos):
        """Single-token (S=1) incremental-KV decode attention on the mesh.
        x: [1, B, hidden]; cos/sin: [1,1,B,hd] position-indexed (replicated);
        current_pos: INT32 [B] device tensor (write index, shared with SDPA).
        Runs on local shards: q=[1,B,tp_num_heads,hd], k/v=[1,B,1,hd], KV cache
        local [B,1,max_seq,hd]. GQA (4 q : 1 kv) handled inside sdpa_decode."""
        B = x.shape[-2]
        qkv = ttnn.linear(x, self.qkv_w, compute_kernel_config=_DECODE_MM_CFG)  # [1, B, tp_qkv]
        qkv = ttnn.reshape(qkv, [1, 1, B, qkv.shape[-1]])  # decode create-heads layout
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=self.tp_num_heads,
            num_kv_heads=self.tp_num_kv_heads,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # q [1,B,tp_num_heads,hd], k/v [1,B,1,hd] (height-sharded)
        ttnn.deallocate(qkv)
        # rope, KV-write and sdpa_decode all want HEIGHT-SHARDED inputs; only
        # rms_norm needs interleaved. Capture the create_heads sharded configs so
        # we can deshard for the norm then re-shard back. v needs neither rope nor
        # norm -> stays sharded for the write.
        q_cfg, k_cfg = q.memory_config(), k.memory_config()

        # rope (decode-mode, sharded) THEN qk-norm (HunyuanImage3 order)
        if self.use_rope and cos is not None:
            q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=True)
            k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=True)
        if self.use_qk_norm:
            q = ttnn.rms_norm(
                ttnn.sharded_to_interleaved(q, ttnn.DRAM_MEMORY_CONFIG), epsilon=self.eps, weight=self.q_norm_w
            )
            k = ttnn.rms_norm(
                ttnn.sharded_to_interleaved(k, ttnn.DRAM_MEMORY_CONFIG), epsilon=self.eps, weight=self.k_norm_w
            )
            q = ttnn.interleaved_to_sharded(q, q_cfg)
            k = ttnn.interleaved_to_sharded(k, k_cfg)

        # write new K/V at current_pos into the resident cache (sharded input).
        # non-fused (separate K,V): the fused op requires K,V on non-overlapping
        # cores, which fails at batch=1 (both on a single core).
        ttnn.experimental.paged_update_cache(self.k_cache, k, update_idxs_tensor=current_pos)
        ttnn.experimental.paged_update_cache(self.v_cache, v, update_idxs_tensor=current_pos)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # single-token attention over cache[0..cur_pos]. Bounded 8x8=64-core grid:
        # the default (full 120-core) grid exceeds the SDPA tree-reduction limit
        # (max 64 cores/head).
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            self.k_cache,
            self.v_cache,
            cur_pos_tensor=current_pos,
            scale=self.scale,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                q_chunk_size=32,
                k_chunk_size=128,
                exp_approx_mode=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, B, tp_num_heads, hd]
        ttnn.deallocate(q)

        # concat heads: sdpa output is [1,B,tp_num_heads,hd] DRAM (GQA requires DRAM
        # output; nlp_concat_heads_decode wants sharded input) -> plain reshape merges
        # the head dim head-major, matching the row-parallel o-proj input order.
        attn = ttnn.reshape(attn, [1, B, self.tp_num_heads * self.head_dim])
        out_partial = ttnn.linear(attn, self.o_w, compute_kernel_config=_DECODE_MM_CFG)
        ttnn.deallocate(attn)
        out = self._mesh_reduce(out_partial)
        ttnn.deallocate(out_partial)
        return out

    # ------------------------------------------------------------------
    def __call__(self, hidden_states, custom_pos_emb=None, return_l_aux=False, **kwargs):
        self.num_calls += 1
        if self.is_mesh:
            return self._forward_sharded(hidden_states, custom_pos_emb, return_l_aux)

        residual = hidden_states
        x = ttnn.rms_norm(hidden_states, epsilon=self.eps, weight=self.input_ln_w)
        attn = self._attention(x, custom_pos_emb)
        ttnn.deallocate(x)
        hidden = ttnn.add(residual, attn)
        ttnn.deallocate(attn)

        residual2 = hidden
        x2 = ttnn.rms_norm(hidden, epsilon=self.eps, weight=self.post_ln_w)
        # Composed graduated MoE (which composes the graduated gate).
        moe, l_aux = self.moe(x2, return_l_aux=True)
        ttnn.deallocate(x2)
        out = ttnn.add(residual2, moe)
        ttnn.deallocate(moe)
        if return_l_aux:
            return out, l_aux
        ttnn.deallocate(l_aux)
        return out

    # ------------------------------------------------------------------
    def _forward_sharded(self, hidden_states, custom_pos_emb=None, return_l_aux=False):
        residual = hidden_states
        x = ttnn.rms_norm(hidden_states, epsilon=self.eps, weight=self.input_ln_w)
        attn = self._attention_sharded(x, custom_pos_emb)
        ttnn.deallocate(x)
        hidden = ttnn.add(residual, attn)
        ttnn.deallocate(attn)

        residual2 = hidden
        x2 = ttnn.rms_norm(hidden, epsilon=self.eps, weight=self.post_ln_w)
        # Composed graduated MoE (which composes the graduated gate), sharded.
        moe, l_aux = self.moe(x2, return_l_aux=True)
        ttnn.deallocate(x2)
        out = ttnn.add(residual2, moe)
        ttnn.deallocate(moe)
        if return_l_aux:
            return out, l_aux
        if l_aux is not None:
            ttnn.deallocate(l_aux)
        return out

    # ------------------------------------------------------------------
    def forward_decode(self, hidden_states, cos, sin, current_pos, return_l_aux=False):
        """Single-token decode forward (same pre-norm/residual structure as
        _forward_sharded, but incremental-KV attention). hidden_states: [1,B,hidden]."""
        residual = hidden_states
        x = ttnn.rms_norm(hidden_states, epsilon=self.eps, weight=self.input_ln_w)
        attn = self._attention_decode(x, cos, sin, current_pos)
        ttnn.deallocate(x)
        hidden = ttnn.add(residual, attn)
        ttnn.deallocate(attn)

        residual2 = hidden
        x2 = ttnn.rms_norm(hidden, epsilon=self.eps, weight=self.post_ln_w)
        moe, l_aux = self.moe(x2, return_l_aux=True)
        ttnn.deallocate(x2)
        out = ttnn.add(residual2, moe)
        ttnn.deallocate(moe)
        if return_l_aux:
            return out, l_aux
        if l_aux is not None:
            ttnn.deallocate(l_aux)
        return out


def build(device, torch_module):
    return _TtDecoderLayer(device, torch_module)


def image3_decoder_layer(*args, **kwargs):  # pragma: no cover - build() is the entry point
    raise RuntimeError(
        "image3_decoder_layer must be constructed via build(device, torch_module); "
        "the bare module-level callable is not supported for this native port."
    )
