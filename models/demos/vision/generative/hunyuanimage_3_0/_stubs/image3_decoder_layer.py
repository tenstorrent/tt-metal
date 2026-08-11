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

import os as _os
import time as _time

import torch

import ttnn
from models.demos.vision.generative.hunyuanimage_3_0._stubs import mo_e as _mo_e

HF_MODEL_ID = "tencent/HunyuanImage-3.0"

# --- optional tracy-free stage profiler (HUNYUAN_STAGE_PROFILE=1) -------------
# Accumulates per-stage device wall-time (attention vs MoE) across a forward via
# ttnn.synchronize_device, to find the gen_image bottleneck WITHOUT a tracy build.
# No-op (zero overhead) unless enabled -> production/trace paths are unaffected.
# The synchronize serializes stages so absolute ms are inflated vs the pipelined
# run, but the attn:moe:other RATIO is the signal.
_STAGE_PROF = {"on": _os.environ.get("HUNYUAN_STAGE_PROFILE") == "1", "attn_ms": 0.0, "moe_ms": 0.0, "layers": 0}


class _StageTimer:
    def __init__(self, dev, key):
        self.dev, self.key, self.t = dev, key, None

    def __enter__(self):
        if _STAGE_PROF["on"]:
            try:
                ttnn.synchronize_device(self.dev)
            except Exception:
                pass
            self.t = _time.time()
        return self

    def __exit__(self, *a):
        if _STAGE_PROF["on"] and self.t is not None:
            try:
                ttnn.synchronize_device(self.dev)
            except Exception:
                pass
            _STAGE_PROF[self.key] += (_time.time() - self.t) * 1000.0


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


def _host_of(dtype):
    # P1 host pre-cast: cast bf16 weights on HOST before from_torch so the
    # fp32->bf16 cast happens once at build (host side), not as a recurring
    # on-device Tilize+Typecast on the lazily-materialized fp32 upload. bfloat8_b/
    # bfloat4_b have no host torch equivalent -> stay fp32 (packed on device at
    # build). Mirrors the optimized mo_e.py (_host_of).
    return torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32


def _to_ttnn(t: torch.Tensor, device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t.to(_host_of(dtype)),
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
    # A 1-device mesh (or a plain single device) is treated as single-chip: the
    # TP/EP collectives below then no-op, so the model runs fabric-free on one
    # chip (matches the hunyuan-image3-bringup path; unblocks single-chip PCC when
    # the inter-chip fabric is unavailable). Real multi-chip (>1 device) unchanged.
    try:
        if isinstance(device, ttnn.MeshDevice):
            return device.get_num_devices() > 1
    except AttributeError:
        pass
    return hasattr(device, "get_num_devices") and hasattr(device, "get_device_ids") and device.get_num_devices() > 1


class _TtDecoderLayer:
    def __init__(self, device, torch_module, ccl_manager=None):
        self.device = device
        self.ccl_manager = ccl_manager  # SP Step 1: shared mesh CCLManager (used from Step 2)
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
            # FSDP (HUNYUAN_FSDP=1, OFF by default): the attention qkv/o weights are
            # TP-sharded on the tp_axis but 4x-REPLICATED across the other (DP) mesh
            # axis (dead replication -- a single image has no batch on that axis).
            # FSDP shards them across the DP axis too and all-gathers each weight
            # back at forward (see _shard_linear_fsdp / _fsdp_gather). Cuts per-chip
            # attention-weight footprint by dp. Needs exactly a 2D mesh with dp>1.
            self.dp_axis = (1 - self.tp_axis) if len(self.mesh_shape) == 2 else None
            self.dp = int(self.mesh_shape[self.dp_axis]) if self.dp_axis is not None else 1
            self.is_fsdp = _os.environ.get("HUNYUAN_FSDP", "0") == "1" and self.dp > 1
            # SP Step 1 (HUNYUAN_SP): the residual/activation SEQUENCE dim is
            # sharded across the DP axis (axis 1), so that axis carries S/dp tokens
            # per device. Attention then all-gathers K,V over this axis so each
            # device's local heads see all tokens (see _attention_sharded). Needs a
            # real 2D mesh with dp>1. OFF by default -> self.sp False, no change.
            self.sp = _mo_e._sp_on() and self.dp_axis is not None and self.dp > 1
            self.sp_axis = self.dp_axis if self.sp else None
            # SP Step 2 (HUNYUAN_SP_FUSED): H-shard the residual on the TP axis, distributed
            # RMSNorm, AG+MM (col-parallel) + MM/reduce_scatter (row-parallel). Requires SP.
            self.sp_fused = _mo_e._sp_fused_on() and self.sp
            self._build_sharded(torch_module)
        else:
            self.sp = False
            self.sp_axis = None
            self.sp_fused = False
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
            t.to(_host_of(dtype)),
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
            t.to(_host_of(dtype)),
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

    def _shard_fsdp(self, t, tp_dim, fsdp_dim, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
        """FSDP 2D-shard: split `tp_dim` across the TP axis AND `fsdp_dim` across the
        DP axis, so the otherwise-DP-replicated weight becomes a 1/dp shard. The
        DP-sharded dim is all-gathered back at forward (see _fsdp_gather)."""
        dims = [None, None]
        dims[self.tp_axis] = tp_dim
        dims[self.dp_axis] = fsdp_dim
        return ttnn.from_torch(
            t.to(_host_of(dtype)),
            dtype=dtype,
            layout=layout,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.device, dims=tuple(dims), mesh_shape=self.mesh_shape),
        )

    def _shard_linear_fsdp(self, w, tp_dim, fsdp_dim, dtype=ttnn.bfloat16):
        """nn.Linear weight [out, in] -> transpose [in, out], then FSDP 2D-shard
        (tp_dim/fsdp_dim index the TRANSPOSED tensor: 0=in feats, -1=out feats)."""
        return self._shard_fsdp(w.t().contiguous(), tp_dim, fsdp_dim, dtype=dtype)

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
        self.moe = _mo_e.build(device, torch_module.mlp, ccl_manager=self.ccl_manager)

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

        # --- norms / qk-norms / (rope tables handled at forward) ---
        # SP Step 2: the residual is H-sharded on the TP axis, so the two hidden-dim
        # RMSNorm weights are H-sharded too (each device holds its H/tp slice; the norm
        # is DISTRIBUTED -- variance all_reduced over the TP axis at forward). Stored 3D
        # [1,1,H/tp] to broadcast over S/sp. The qk-norms operate on head_dim (NOT sharded
        # -- heads are sharded, head_dim is full) so they stay REPLICATED.
        if self.sp_fused:
            self.input_ln_w = self._shard(torch_module.input_layernorm.weight.reshape(1, 1, -1), dim=-1)
            self.post_ln_w = self._shard(torch_module.post_attention_layernorm.weight.reshape(1, 1, -1), dim=-1)
        else:
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
        if self.is_fsdp:
            # column-parallel out on tp_axis (dim=-1) + FSDP shard in on dp_axis (dim=0)
            self.qkv_w = self._shard_linear_fsdp(qkv_w, tp_dim=-1, fsdp_dim=0)
        else:
            self.qkv_w = self._shard_linear(qkv_w, dim=-1)  # column-parallel (output feats)

        # O-proj row-parallel: split INPUT features. The concat-heads output
        # feature order is head-major, and device d holds the contiguous block
        # [d*q_per_dev*hd : (d+1)*q_per_dev*hd], so a plain dim=0 (input) shard
        # of the transposed weight lines up exactly.
        if self.is_fsdp:
            # row-parallel in on tp_axis (dim=0) + FSDP shard out on dp_axis (dim=-1)
            self.o_w = self._shard_linear_fsdp(cfg.o_proj.weight, tp_dim=0, fsdp_dim=-1)
        else:
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
        self.moe = _mo_e.build(self.device, torch_module.mlp, ccl_manager=self.ccl_manager)

    # ------------------------------------------------------------------
    def _mesh_reduce(self, x):
        """All-reduce (sum) a per-device partial across the TP mesh axis. Fused ring
        all_reduce (cluster_axis=TP) -- same math as the old all_gather(dim=0)+sum but
        it does NOT touch dim 0, so it is batch-safe (a [bsz,S,H] partial stays
        [bsz,S,H]). Required for CFG-parallel (bsz=2); equivalent at bsz=1."""
        if not self.is_mesh:
            return x  # single chip: no TP axis to reduce over; partial is complete
        return ttnn.all_reduce(
            x, cluster_axis=self.tp_axis, num_links=_mo_e._ccl_links(), topology=_mo_e._sp_topology()
        )

    def _norm(self, x, weight):
        """RMSNorm on the residual stream. SP Step 2: the hidden dim is TP-sharded, so
        the variance spans devices -> DISTRIBUTED RMSNorm (local sum-of-squares all_reduced
        over the TP axis; validated PCC 1.000001). Otherwise the plain fused rms_norm."""
        if self.sp_fused:
            return _mo_e._dist_rmsnorm(x, weight, self.tp_axis, self.eps, self.hidden_size)
        return ttnn.rms_norm(x, epsilon=self.eps, weight=weight)

    def _fsdp_gather(self, w, dim):
        """All-gather an FSDP-sharded weight back over the DP axis just before its
        matmul (mirrors flux2 all_gather_persistent_buffer(weight, dim=2/3)). `dim`
        is the 4D gather dim: 2 = in/K (column-parallel qkv), 3 = out/N (row-parallel
        o_proj). No-op unless FSDP is on. Weight is static, but re-gathered each call
        (flux2-faithful); if this dominates, cache the gathered weight at setup."""
        if not self.is_fsdp:
            return w
        w4 = ttnn.unsqueeze_to_4D(w)
        g = ttnn.all_gather(
            w4, dim=dim, cluster_axis=self.dp_axis, num_links=_mo_e._ccl_links(), topology=_mo_e._sp_topology()
        )
        return ttnn.reshape(g, [g.shape[-2], g.shape[-1]])

    def _swiglu(self, x, gu_w, down_w, inter):
        gu = ttnn.linear(x, gu_w)
        x1 = ttnn.slice(gu, [0, 0, 0], [gu.shape[0], gu.shape[1], inter])
        x2 = ttnn.slice(gu, [0, 0, inter], [gu.shape[0], gu.shape[1], 2 * inter])
        ttnn.deallocate(gu)
        # SwiGLU: fuse silu into the multiply (silu(x2) * x1 in one op).
        act = ttnn.multiply(x2, x1, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
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
    def _attention(self, x, custom_pos_emb, attn_mask=None):
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

        # gen_image threads a per-sample additive block mask (0 attend / -inf
        # masked) here; text-prefix is causal, image tokens attend bidirectionally.
        # attn_mask=None (gen_text / graduated path) => full non-causal SDPA (unchanged).
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
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
    def _attention_sharded(self, x, custom_pos_emb, attn_mask=None):
        bsz = x.shape[0]  # 1, or 2 under CFG-parallel
        S = x.shape[1]
        if self.sp_fused:
            # AG+MM: x is H-sharded [1, S/sp, H/tp]; all_gather H over the TP axis then
            # matmul the col-parallel qkv weight [H, tp_qkv] -> [1, S/sp, tp_qkv].
            qkv = _mo_e._agmm(
                x, self.qkv_w, self.device, self.ccl_manager, self.tp_axis, compute_kernel_config=_mo_e._mm_cfg()
            )
        else:
            qkv = _mo_e._minmm(
                x,
                self._fsdp_gather(self.qkv_w, 2),
                compute_kernel_config=_mo_e._mm_cfg(),
                core_grid=_mo_e._mm_grid(self.device),
                fallback=ttnn.linear,
            )  # [1, S, tp_qkv] per device
        qkv = ttnn.reshape(qkv, [bsz, 1, S, qkv.shape[-1]])
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

        if self.sp:
            # SP: Q stays sequence-sharded (this device holds S/dp local query
            # tokens); all-gather K,V over the SP axis so the local heads attend to
            # ALL tokens (S/dp -> S on the sequence dim=2). rope + qk-norm were
            # already applied per-position while K,V were S/dp, so the gathered K
            # carries the correct per-position rotation. SDPA output length ==
            # query length, so it stays S/dp-sharded, matching the residual stream.
            # Explicit KV-gather (flux2 Linear-topology fallback), NOT a fused op.
            k = ttnn.all_gather(
                k, dim=2, cluster_axis=self.sp_axis, num_links=_mo_e._ccl_links(), topology=_mo_e._sp_topology()
            )
            v = ttnn.all_gather(
                v, dim=2, cluster_axis=self.sp_axis, num_links=_mo_e._ccl_links(), topology=_mo_e._sp_topology()
            )

        _sdpa_kw = {"attn_mask": attn_mask, "is_causal": False, "scale": self.scale}
        _ck = _mo_e._mm_cfg()  # only pass fidelity when set, so the default path is untouched
        if _ck is not None:
            _sdpa_kw["compute_kernel_config"] = _ck
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, **_sdpa_kw)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.reshape(attn, [bsz, S, self.tp_num_heads * self.head_dim])
        if _mo_e._sp_ring_fusedmm_on():
            # SP_RING: FUSED O-proj matmul + reduce_scatter (overlaps the MM with the RS
            # on the ring fabric). attn [1,S,hd_local] @ o_w [hd_local,H] -> RS over TP ->
            # [1, S/sp, H/tp], matching the H-sharded residual.
            out = _mo_e._mmrs_last(
                attn,
                self.o_w,
                self.device,
                self.ccl_manager,
                self.tp_axis,
                compute_kernel_config=_mo_e._mm_cfg(),
            )
            ttnn.deallocate(attn)
            return out
        out_partial = _mo_e._minmm(
            attn,
            self.o_w if self.sp_fused else self._fsdp_gather(self.o_w, 3),
            compute_kernel_config=_mo_e._mm_cfg(),
            core_grid=_mo_e._mm_grid(self.device),
            fallback=ttnn.linear,
        )  # [1, S, hidden] partial sum
        ttnn.deallocate(attn)
        if self.sp_fused:
            # MM + reduce_scatter: sum the per-device head partials AND re-scatter H ->
            # [1, S/sp, H/tp], matching the H-sharded residual (no full-H all_reduce).
            out = _mo_e._reduce_scatter_last(out_partial, self.ccl_manager, self.tp_axis)
        else:
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
        qkv = ttnn.linear(x, self._fsdp_gather(self.qkv_w, 2), compute_kernel_config=_DECODE_MM_CFG)  # [1, B, tp_qkv]
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
        out_partial = ttnn.linear(attn, self._fsdp_gather(self.o_w, 3), compute_kernel_config=_DECODE_MM_CFG)
        ttnn.deallocate(attn)
        out = self._mesh_reduce(out_partial)
        ttnn.deallocate(out_partial)
        return out

    # ------------------------------------------------------------------
    def __call__(self, hidden_states, custom_pos_emb=None, return_l_aux=False, **kwargs):
        self.num_calls += 1
        attn_mask = kwargs.get("attn_mask")
        if self.is_mesh:
            return self._forward_sharded(hidden_states, custom_pos_emb, return_l_aux, attn_mask=attn_mask)

        residual = hidden_states
        x = ttnn.rms_norm(hidden_states, epsilon=self.eps, weight=self.input_ln_w)
        attn = self._attention(x, custom_pos_emb, attn_mask=attn_mask)
        ttnn.deallocate(x)
        hidden = ttnn.add(residual, attn)
        ttnn.deallocate(attn)

        residual2 = hidden
        x2 = ttnn.rms_norm(hidden, epsilon=self.eps, weight=self.post_ln_w)
        # Composed graduated MoE (which composes the graduated gate).
        if return_l_aux:
            moe, l_aux = self.moe(x2, return_l_aux=True)
        else:
            # inference: skip l_aux (mo_e returns a single tensor).
            moe = self.moe(x2, return_l_aux=False)
            l_aux = None
        ttnn.deallocate(x2)
        out = ttnn.add(residual2, moe)
        ttnn.deallocate(moe)
        if return_l_aux:
            return out, l_aux
        if l_aux is not None:
            ttnn.deallocate(l_aux)
        return out

    # ------------------------------------------------------------------
    def _forward_sharded(self, hidden_states, custom_pos_emb=None, return_l_aux=False, attn_mask=None):
        residual = hidden_states
        x = self._norm(hidden_states, self.input_ln_w)
        with _StageTimer(self.device, "attn_ms"):
            attn = self._attention_sharded(x, custom_pos_emb, attn_mask=attn_mask)
        ttnn.deallocate(x)
        hidden = ttnn.add(residual, attn)
        ttnn.deallocate(attn)

        residual2 = hidden
        x2 = self._norm(hidden, self.post_ln_w)
        # Composed graduated MoE (which composes the graduated gate), sharded.
        with _StageTimer(self.device, "moe_ms"):
            if return_l_aux:
                moe, l_aux = self.moe(x2, return_l_aux=True)
            else:
                # inference: skip l_aux (mo_e returns a single tensor).
                moe = self.moe(x2, return_l_aux=False)
                l_aux = None
        if _STAGE_PROF["on"]:
            _STAGE_PROF["layers"] += 1
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
        if return_l_aux:
            moe, l_aux = self.moe(x2, return_l_aux=True)
        else:
            # inference: skip l_aux (mo_e returns a single tensor).
            moe = self.moe(x2, return_l_aux=False)
            l_aux = None
        ttnn.deallocate(x2)
        out = ttnn.add(residual2, moe)
        ttnn.deallocate(moe)
        if return_l_aux:
            return out, l_aux
        if l_aux is not None:
            ttnn.deallocate(l_aux)
        return out


def build(device, torch_module, ccl_manager=None):
    return _TtDecoderLayer(device, torch_module, ccl_manager=ccl_manager)


def image3_decoder_layer(*args, **kwargs):  # pragma: no cover - build() is the entry point
    raise RuntimeError(
        "image3_decoder_layer must be constructed via build(device, torch_module); "
        "the bare module-level callable is not supported for this native port."
    )
