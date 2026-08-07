# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Multichip TTNN decoder for poolside/Laguna-XS-2.1 (Blackhole p300c ×4, 1×4 mesh).

Parallelizes the single-chip optimized decoder (``tt/optimized_decoder.py``,
``OptimizedDecoder``) across a 1×4 Blackhole mesh with **1D tensor parallelism (TP=4)**
for the dense/attention path and **expert parallelism (EP=4)** for the routed MoE. The
optimized decoder is the single-chip baseline: this class subclasses it and reuses every
optimized helper (precision policy, packed-QKV split, RMSNorm, RoPE, SDPA config, the
DRAM-sharded matmul helper, and the sparse-expert program configs). Only weight placement
(mesh sharding at load time) and the collectives the scheme needs are added here.

Scheme:
  * Residual stream is **replicated** (BF16). Both RMSNorms see the full hidden and are
    exact locally — no distributed norm needed.
  * Attention: WQKV column-parallel (packed, reordered so device d owns Q heads
    [d·lqh:(d+1)·lqh) and KV heads [2d:2d+2)); per-head QK-norm/RoPE/SDPA local; KV cache
    holds the device-local 2 KV heads; softplus gate g_proj column-parallel; WO row-parallel
    → partial → one all_reduce.
  * Dense MLP (layer 0): gate/up column-parallel, down row-parallel → partial → all_reduce.
  * MoE (layers 1-39): router runs replicated; expert weights EP-sharded (64 experts/device);
    a mesh-sharded selection-matmul turns the replicated 256-wide router output into device d's
    contiguous 64-wide scores/sparsity; ``ttnn.sparse_matmul`` (nnz=None, Blackhole-required)
    over the local experts; shared expert TP; routed_local + shared_partial → one all_reduce.

Collectives: exactly **2 all_reduce / layer** (attn out, MLP/MoE out), cluster_axis=1,
Ring topology, 2 links. Decoder-layer input and output share the replicated ``[1,1,B,H]`` /
``[1,seq,H]`` layout, so layers stack with no inter-layer reshard.

Public API matches OptimizedDecoder/FunctionalDecoder (``from_state_dict``, ``alloc_kv_cache``,
``make_page_table``, ``prefill_forward``, ``decode_forward``) — the tensors must live on the
1×4 mesh (replicated inputs). Set ``FABRIC_1D_RING`` before ``open_mesh_device``.
"""
from __future__ import annotations

import math

import torch

import ttnn

from .optimized_decoder import (
    TILE,
    LayerConfig,
    OptimizedDecoder,
    PrecisionPolicy,
    _cached_device_tensor,
    _dram_weight_memcfg,
    _hf_rope_tables,
    _sparse_pc,
    weight_cache_key,
)


class MultichipDecoder(OptimizedDecoder):
    """Multichip TTNN decoder for Laguna-XS-2.1 (Blackhole p300c ×4, 1×4 mesh): TP=4 attention/dense +
    EP=4 routed MoE, replicated BF16 residual, exactly two ring ``all_reduce``s per layer (comm-optimal for
    this small hidden H=2048).

    ``PACK_GATE_UP`` (default True) enables the decode gate/up matmul-packing optimization: the two
    same-input gate/up projections (routed-expert MoE, dense MLP, shared expert) are concatenated at load
    into one wide weight so a single matmul replaces two dispatches (decode is dispatch/latency-bound). PCC
    is preserved exactly (SiLU is applied after the on-device split), so it is a pure device-time win
    (measured ~−5.4% layer decode). Set ``PACK_GATE_UP=False`` for the unpacked baseline (A/B evidence).
    Previously this lived in a separate ``OptimizedMultichipDecoder`` subclass; folded in here as a flag.
    """

    # Inherited runtime chunk knobs (MOE_PREFILL_CHUNK, PIPE_CHUNK, PREFILL_SDPA_CHUNK).
    PACK_GATE_UP = True

    def __init__(self, cfg, weights, cos_table, sin_table, mesh_device, policy, meta):
        super().__init__(cfg, weights, cos_table, sin_table, mesh_device, policy, meta)
        self.D = meta["mesh_devices"]
        self.global_experts = meta["global_experts"]
        self.local_experts = meta["local_experts"]
        # 1×4 Blackhole (P150x4): TP/EP live on cluster_axis 1; 2 ethernet links/axis.
        self.tp_axis = 1
        self.ccl_topology = ttnn.Topology.Ring
        self.num_links = meta.get("num_links", 2)

    # ---- collective ------------------------------------------------------- #
    def _reduce(self, x):
        """Ring all_reduce over the TP axis to turn a row-parallel partial into the
        replicated residual layout. Traceable (verified).

        The all_reduce payload dtype is ``policy.ccl`` (default BF16 == the replicated
        residual, so the path is byte-identical to the stage-06 baseline). A lower CCL
        dtype casts the partial before the collective and casts the reduced result back to
        BF16 for the residual add — swept as a yes/no switch in the datatype sweep."""
        ccl = getattr(self.policy, "ccl", ttnn.bfloat16)
        if ccl == ttnn.bfloat16:
            return ttnn.all_reduce(x, cluster_axis=self.tp_axis, topology=self.ccl_topology, num_links=self.num_links)
        xin = ttnn.typecast(x, ccl) if x.dtype != ccl else x
        out = ttnn.all_reduce(xin, cluster_axis=self.tp_axis, topology=self.ccl_topology, num_links=self.num_links)
        return ttnn.typecast(out, ttnn.bfloat16) if out.dtype != ttnn.bfloat16 else out

    # ---- construction ------------------------------------------------------ #
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, max_seq_len, policy=None, **kwargs):
        cfg = LayerConfig.from_hf(hf_config, layer_idx)
        policy = policy or PrecisionPolicy()
        dev = mesh_device
        D = mesh_device.get_num_devices()
        dram_cores = mesh_device.dram_grid_size().x

        replicate = ttnn.ReplicateTensorToMesh(dev)

        def shard(dim):
            return ttnn.ShardTensorToMesh(dev, dim=dim)

        def g(name):
            return state_dict[name].float()

        # ---- device-weight disk cache (plan Stage 0; see optimized_decoder) ----
        # The mesh tag encodes the mapping (replicate / shard-dim-N) AND the device
        # count D, so a multichip weight never collides with the single-chip ("sc")
        # copy or a different-D mesh. ``build`` is a thunk so a cache hit skips the
        # source build (critical for the 256-expert stack).
        def rep_tt(name, build, dtype, layout=ttnn.TILE_LAYOUT):
            return _cached_device_tensor(
                build,
                device=dev,
                dtype=dtype,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
                cache_key=weight_cache_key(name, layer_idx, f"rep_d{D}"),
            )

        def shard_tt(name, build, dim, dtype, layout=ttnn.TILE_LAYOUT):
            return _cached_device_tensor(
                build,
                device=dev,
                dtype=dtype,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=shard(dim),
                cache_key=weight_cache_key(name, layer_idx, f"sh{dim}_d{D}"),
            )

        # ---- global shapes ----
        H = cfg.hidden
        hd = cfg.head_dim
        GQ = cfg.num_heads  # 48 (full) / 64 (sliding)
        GKV = cfg.num_kv_heads  # 8
        assert GQ % D == 0 and GKV % D == 0, f"heads must divide mesh {D}: Q={GQ} KV={GKV}"
        lqh = GQ // D
        lkv = GKV // D
        local_q_w = lqh * hd
        local_kv_w = lkv * hd
        local_qkv_w = local_q_w + 2 * local_kv_w

        w = {}

        def store_shard(key, w_in_full, k_local, n_local, dtype, mesh_dim):
            """Store a mesh-sharded interleaved copy (``key``) and a mesh-sharded +
            per-device DRAM-width-sharded copy (``key+"_ds"``). ``w_in_full`` is the full
            (unsharded) weight already in ttnn [in, out] orientation; ``mesh_dim`` is the
            tensor axis sharded across the mesh; ``k_local``/``n_local`` are the resulting
            PER-DEVICE [in, out] dims for the DRAM shard spec. Only the interleaved
            copy is disk-cached; the ``_ds`` copy is derived on-device via
            to_memory_config (a width-sharded memory_config is not preserved through
            the ttnn tensor cache — see optimized_decoder._cached_device_tensor)."""
            w[key] = shard_tt(key, lambda: w_in_full, mesh_dim, dtype)
            w[key + "_ds"] = ttnn.to_memory_config(w[key], _dram_weight_memcfg(k_local, n_local, dram_cores))

        # --- packed QKV (column-parallel), reordered into per-device [Q_d|K_d|V_d] blocks --- #
        wq = g("self_attn.q_proj.weight").t().contiguous()  # [H, GQ*hd]
        wk = g("self_attn.k_proj.weight").t().contiguous()  # [H, GKV*hd]
        wv = g("self_attn.v_proj.weight").t().contiguous()  # [H, GKV*hd]
        blocks = []
        for d in range(D):
            blocks.append(wq[:, d * local_q_w : (d + 1) * local_q_w])
            blocks.append(wk[:, d * local_kv_w : (d + 1) * local_kv_w])
            blocks.append(wv[:, d * local_kv_w : (d + 1) * local_kv_w])
        wqkv = torch.cat(blocks, dim=1).contiguous()  # [H, D*local_qkv_w]; shard(dim1) -> per-dev block
        store_shard("wqkv", wqkv, H, local_qkv_w, policy.attn_qkv, mesh_dim=1)
        # WO row-parallel: device d owns contiguous Q-head rows [d*local_q_w:(d+1)*local_q_w] -> plain shard(dim0)
        wo = g("self_attn.o_proj.weight").t().contiguous()  # [GQ*hd, H]
        store_shard("wo", wo, local_q_w, H, policy.attn_o, mesh_dim=0)
        # softplus gate g_proj [H, GQ] column-parallel -> device d gate for its Q heads
        w["wg"] = shard_tt("wg", lambda: g("self_attn.g_proj.weight").t().contiguous(), 1, policy.attn_gate)
        w["q_norm"] = rep_tt("q_norm", lambda: g("self_attn.q_norm.weight").reshape(1, 1, 1, hd), policy.qk_norm)
        w["k_norm"] = rep_tt("k_norm", lambda: g("self_attn.k_norm.weight").reshape(1, 1, 1, hd), policy.qk_norm)
        w["input_ln"] = rep_tt("input_ln", lambda: g("input_layernorm.weight").reshape(1, 1, 1, H), ttnn.bfloat16)
        w["post_ln"] = rep_tt(
            "post_ln", lambda: g("post_attention_layernorm.weight").reshape(1, 1, 1, H), ttnn.bfloat16
        )

        global_experts = cfg.num_experts
        local_experts = global_experts // D if cfg.is_moe else 0

        if cfg.is_moe:
            E, I = global_experts, cfg.moe_intermediate
            # router replicated (full 256-wide logits/top-k on every device)
            w["gate_w"] = rep_tt("gate_w", lambda: g("mlp.gate.weight").t().contiguous(), policy.router)
            w["e_bias"] = rep_tt(
                "e_bias", lambda: g("mlp.experts.e_score_correction_bias").reshape(1, 1, 1, E), ttnn.bfloat16
            )
            # mesh-sharded selection matrix: identity[E,E] shard(dim3) -> device d gets [1,1,E,local_E]
            # so matmul(scores[1,1,T,E], sel[1,1,E,local_E]) == scores[..., 64d:64d+64] (SPMD-safe slice).
            w["ep_sel"] = shard_tt("ep_sel", lambda: torch.eye(E).reshape(1, 1, E, E), 3, ttnn.bfloat16)

            # expert weights EP-sharded on the expert dim (dim1): device d holds experts [64d:64d+64].
            # The 256-expert torch.stack is the dominant boot cost, so build it lazily
            # inside the cache thunk — a HIT skips the stack (and the safetensors
            # upcast) entirely, not just the tilize/shard.
            def _stack(proj):
                return torch.stack([g(f"mlp.experts.{i}.{proj}.weight") for i in range(E)])

            w["exp_gate"] = shard_tt(
                "exp_gate", lambda: _stack("gate_proj").transpose(1, 2).reshape(1, E, H, I), 1, policy.moe_ff13
            )
            w["exp_up"] = shard_tt(
                "exp_up", lambda: _stack("up_proj").transpose(1, 2).reshape(1, E, H, I), 1, policy.moe_ff13
            )
            w["exp_down"] = shard_tt(
                "exp_down", lambda: _stack("down_proj").transpose(1, 2).reshape(1, E, I, H), 1, policy.moe_ff2
            )
            # shared expert TP (col/col/row); down produces a partial folded into the routed all_reduce
            gsh = cfg.shared_intermediate
            local_sh = gsh // D
            store_shard(
                "sh_gate",
                g("mlp.shared_expert.gate_proj.weight").t().contiguous(),
                H,
                local_sh,
                policy.shared_ff13,
                mesh_dim=1,
            )
            store_shard(
                "sh_up",
                g("mlp.shared_expert.up_proj.weight").t().contiguous(),
                H,
                local_sh,
                policy.shared_ff13,
                mesh_dim=1,
            )
            store_shard(
                "sh_down",
                g("mlp.shared_expert.down_proj.weight").t().contiguous(),
                local_sh,
                H,
                policy.shared_ff2,
                mesh_dim=0,
            )
            cfg.shared_intermediate = local_sh  # local for _glu_mlp
        else:
            II = cfg.intermediate
            local_II = II // D
            store_shard(
                "mlp_gate", g("mlp.gate_proj.weight").t().contiguous(), H, local_II, policy.dense_ff13, mesh_dim=1
            )
            store_shard("mlp_up", g("mlp.up_proj.weight").t().contiguous(), H, local_II, policy.dense_ff13, mesh_dim=1)
            store_shard(
                "mlp_down", g("mlp.down_proj.weight").t().contiguous(), local_II, H, policy.dense_ff2, mesh_dim=0
            )
            cfg.intermediate = local_II  # local for _glu_mlp

        # item 2.4: build the (cos,sin) RoPE tables once PER ATTENTION KIND, not once per layer then
        # dedup. LagunaModel.from_pretrained threads a shared ``rope_tables`` dict; the first layer of
        # each kind builds+caches, later layers of that kind reuse the SAME device tensors. This removes
        # the ~4.7 GB transient peak (all 40 pairs alive before _dedup_rope) and cuts the per-layer
        # trust-remote-code load (get_class_from_dynamic_module runs twice — one per kind — not 40x).
        # Tables depend only on (attention_type, max_seq_len, config), so sharing is exact / bit-identical.
        rope_tables = kwargs.get("rope_tables")
        kind = cfg.attention_type
        if rope_tables is not None and kind in rope_tables:
            cos_2d, sin_2d = rope_tables[kind]
        else:
            cos, sin = _hf_rope_tables(hf_config, kind, max_seq_len)
            cos_2d = ttnn.from_torch(
                cos, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, mesh_mapper=replicate
            )
            sin_2d = ttnn.from_torch(
                sin, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, mesh_mapper=replicate
            )
            if rope_tables is not None:
                rope_tables[kind] = (cos_2d, sin_2d)

        # ---- mutate cfg to LOCAL head counts so all inherited attention code runs per-device ----
        cfg.num_heads = lqh
        cfg.num_kv_heads = lkv
        cfg.num_kv_groups = lqh // lkv

        meta = {
            "dram_cores": dram_cores,
            "q_w": local_q_w,
            "kv_w": local_kv_w,
            "qkv_w": local_qkv_w,
            "mesh_devices": D,
            "global_experts": global_experts,
            "local_experts": local_experts,
            "num_links": 2,
        }
        dec = cls(cfg, w, cos_2d, sin_2d, dev, policy, meta)
        if cls.PACK_GATE_UP:
            dec._pack_gate_up()
        return dec

    def _pack_gate_up(self):
        """Concat the same-input gate/up weights (interleaved + DRAM-width-sharded ``_ds`` copies) into one
        packed weight and free the separate copies. Pure on-device concat/reshard of the already-loaded
        (and disk-cached) tensors, so it inherits the weight cache. The packed copies are intentionally NOT
        separately disk-cached: they are cheap to derive on device, and the ttnn tensor cache cannot
        round-trip the ``_ds`` copy's custom width-sharded memory_config anyway."""
        w = self.w
        dram_cores = self.meta["dram_cores"]
        H = self.cfg.hidden

        def pack_dram_pair(gk, uk, out, n_local):
            """Concat two DRAM-width-sharded gate/up weights (interleaved + _ds copies) into one
            [H, 2*n_local] weight; rebuild the DRAM shard spec for the doubled width; free originals."""
            il = ttnn.concat([w[gk], w[uk]], dim=-1)  # [H, 2*n_local] interleaved DRAM
            ds = ttnn.to_memory_config(il, _dram_weight_memcfg(H, 2 * n_local, dram_cores))
            w[out] = il
            w[out + "_ds"] = ds
            for k in (gk, uk):
                ttnn.deallocate(w[k])
                ttnn.deallocate(w[k + "_ds"])
                del w[k]
                del w[k + "_ds"]

        if self.cfg.is_moe:
            # routed experts: interleaved BFP4 [1,64,H,512] each -> [1,64,H,1024]
            w["exp_gate_up"] = ttnn.concat([w["exp_gate"], w["exp_up"]], dim=-1)
            ttnn.deallocate(w["exp_gate"])
            ttnn.deallocate(w["exp_up"])
            del w["exp_gate"]
            del w["exp_up"]
            pack_dram_pair("sh_gate", "sh_up", "sh_gate_up", self.cfg.shared_intermediate)
        else:
            pack_dram_pair("mlp_gate", "mlp_up", "mlp_gate_up", self.cfg.intermediate)

    # ---- KV cache / page table (replicated init; each device holds its own local KV heads) ---- #
    def alloc_kv_cache(self, max_users, max_seq_len, block_size=32, dtype=None):
        dtype = dtype or self.policy.kv_cache
        blocks_per_user = int(math.ceil(max_seq_len / block_size))
        max_num_blocks = blocks_per_user * max_users
        shape = (max_num_blocks, self.cfg.num_kv_heads, block_size, self.cfg.head_dim)  # local KV heads
        k = ttnn.from_torch(
            torch.zeros(shape),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        v = ttnn.from_torch(
            torch.zeros(shape),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        return {"k": k, "v": v, "block_size": block_size, "blocks_per_user": blocks_per_user, "dtype": dtype}

    def make_page_table(self, num_users, blocks_per_user):
        pt = torch.arange(num_users * blocks_per_user, dtype=torch.int32).reshape(num_users, blocks_per_user)
        return ttnn.from_torch(
            pt,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

    # ---- MoE (Expert Parallel) --------------------------------------------- #
    def _moe(self, ln_flat, m, sharded):
        """Routed-expert MoE (EP=4). With PACK_GATE_UP, one packed gate+up sparse_matmul (split + SwiGLU on
        device); else the unpacked two-matmul baseline (``_moe_separate``). Both end in a ring all_reduce."""
        if not self.PACK_GATE_UP:
            return self._moe_separate(ln_flat, m, sharded)
        cfg = self.cfg
        LE = self.local_experts
        H, I, K = cfg.hidden, cfg.moe_intermediate, cfg.top_k
        T = ln_flat.shape[2]
        logits = ttnn.linear(ln_flat, self.w["gate_w"], compute_kernel_config=self._ck_router)
        scores = ttnn.sigmoid(logits)
        sel = ttnn.add(scores, self.w["e_bias"])
        _, idx = ttnn.topk(ttnn.typecast(sel, ttnn.bfloat16), k=K, dim=-1, sorted=True)
        wsel = ttnn.gather(scores, dim=3, index=idx)
        if cfg.norm_topk_prob:
            wsel = ttnn.div(wsel, ttnn.sum(wsel, dim=3, keepdim=True))
        if cfg.routed_scaling != 1.0:
            wsel = ttnn.multiply(wsel, cfg.routed_scaling)
        dense = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=idx, src=wsel)
        dense_local = ttnn.matmul(dense, self.w["ep_sel"], compute_kernel_config=self._ck_router)
        union = ttnn.sum(dense_local, dim=2, keepdim=True)
        sparsity = ttnn.to_layout(union, ttnn.ROW_MAJOR_LAYOUT)
        a = ttnn.reshape(ln_flat, (1, 1, T, H))
        moe_mem = ttnn.L1_MEMORY_CONFIG if sharded else ttnn.DRAM_MEMORY_CONFIG
        otile = ttnn.Tile([TILE, TILE])
        gu_pc = _sparse_pc(2 * I, T, H)  # packed gate+up, N = 2*I
        gu = ttnn.sparse_matmul(
            a,
            self.w["exp_gate_up"],
            sparsity=sparsity,
            program_config=gu_pc,
            compute_kernel_config=self._ck_moe,
            memory_config=moe_mem,
            output_tile=otile,
        )
        gu = ttnn.reshape(gu, (1, LE, T, 2 * I))
        gate_o = ttnn.slice(gu, [0, 0, 0, 0], [1, LE, T, I])
        up_o = ttnn.slice(gu, [0, 0, 0, I], [1, LE, T, 2 * I])
        glu = ttnn.mul(ttnn.silu(gate_o), up_o)
        dn_pc = _sparse_pc(H, T, I)
        down_o = ttnn.sparse_matmul(
            glu,
            self.w["exp_down"],
            sparsity=sparsity,
            is_input_a_sparse=True,
            program_config=dn_pc,
            compute_kernel_config=self._ck_moe,
            memory_config=moe_mem,
            output_tile=otile,
        )
        wv = ttnn.reshape(dense_local, (1, T, LE))
        wv = ttnn.permute(wv, (0, 2, 1))
        wv = ttnn.reshape(wv, (1, LE, T, 1))
        weighted = ttnn.mul(down_o, wv)  # [1, LE, T, H]
        if self._use_fused_reduce:  # gated (TT_LAGUNA_FUSED_REDUCE=1); PCC-validate before enabling
            (reduced,) = ttnn.experimental.deepseek_moe_fast_reduce_nc(
                weighted, dim=1, split_size=H, output_memory_config=moe_mem, compute_kernel_config=self._ck_moe
            )
            routed_local = ttnn.reshape(reduced, (1, 1, T, H))
        else:
            routed_local = ttnn.reshape(ttnn.sum(weighted, dim=1), (1, 1, T, H))
        shared_partial = self._glu_mlp(ln_flat, "sh", cfg.hidden, cfg.shared_intermediate, self._ck_shared, sharded)
        combined = ttnn.add(routed_local, ttnn.reshape(shared_partial, (1, 1, T, H)))
        return self._reduce(combined)

    # ---- MoE baseline (unpacked gate/up: two separate sparse_matmuls) ------ #
    def _moe_separate(self, ln_flat, m, sharded):
        cfg = self.cfg
        GE, LE = self.global_experts, self.local_experts
        H, I, K = cfg.hidden, cfg.moe_intermediate, cfg.top_k
        T = ln_flat.shape[2]
        # --- router (replicated: identical full-256 selection on every device) ---
        logits = ttnn.linear(ln_flat, self.w["gate_w"], compute_kernel_config=self._ck_router)  # [1,1,T,GE]
        scores = ttnn.sigmoid(logits)
        sel = ttnn.add(scores, self.w["e_bias"])
        _, idx = ttnn.topk(ttnn.typecast(sel, ttnn.bfloat16), k=K, dim=-1, sorted=True)
        wsel = ttnn.gather(scores, dim=3, index=idx)
        if cfg.norm_topk_prob:
            wsum = ttnn.sum(wsel, dim=3, keepdim=True)
            wsel = ttnn.div(wsel, wsum)
        if cfg.routed_scaling != 1.0:
            wsel = ttnn.multiply(wsel, cfg.routed_scaling)
        dense = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=idx, src=wsel)  # [1,1,T,GE] replicated
        # --- EP selection: replicated 256-wide -> device-local contiguous 64-wide (SPMD-safe matmul) ---
        dense_local = ttnn.matmul(dense, self.w["ep_sel"], compute_kernel_config=self._ck_router)  # [1,1,T,LE]
        union = ttnn.sum(dense_local, dim=2, keepdim=True)  # [1,1,1,LE]
        sparsity = ttnn.to_layout(union, ttnn.ROW_MAJOR_LAYOUT)
        a = ttnn.reshape(ln_flat, (1, 1, T, H))
        moe_mem = ttnn.L1_MEMORY_CONFIG if sharded else ttnn.DRAM_MEMORY_CONFIG
        otile = ttnn.Tile([TILE, TILE])
        gu_pc = _sparse_pc(I, T, H)
        gate_o = ttnn.sparse_matmul(
            a,
            self.w["exp_gate"],
            sparsity=sparsity,
            program_config=gu_pc,
            compute_kernel_config=self._ck_moe,
            memory_config=moe_mem,
            output_tile=otile,
        )
        up_o = ttnn.sparse_matmul(
            a,
            self.w["exp_up"],
            sparsity=sparsity,
            program_config=gu_pc,
            compute_kernel_config=self._ck_moe,
            memory_config=moe_mem,
            output_tile=otile,
        )
        gate_o = ttnn.reshape(gate_o, (1, LE, T, I))
        up_o = ttnn.reshape(up_o, (1, LE, T, I))
        glu = ttnn.mul(ttnn.silu(gate_o), up_o)
        dn_pc = _sparse_pc(H, T, I)
        down_o = ttnn.sparse_matmul(
            glu,
            self.w["exp_down"],
            sparsity=sparsity,
            is_input_a_sparse=True,
            program_config=dn_pc,
            compute_kernel_config=self._ck_moe,
            memory_config=moe_mem,
            output_tile=otile,
        )  # [1,LE,T,H]
        wv = ttnn.reshape(dense_local, (1, T, LE))
        wv = ttnn.permute(wv, (0, 2, 1))
        wv = ttnn.reshape(wv, (1, LE, T, 1))
        weighted = ttnn.mul(down_o, wv)
        routed_local = ttnn.reshape(ttnn.sum(weighted, dim=1), (1, 1, T, H))  # partial (local experts)
        shared_partial = self._glu_mlp(ln_flat, "sh", cfg.hidden, cfg.shared_intermediate, self._ck_shared, sharded)
        combined = ttnn.add(routed_local, ttnn.reshape(shared_partial, (1, 1, T, H)))
        return self._reduce(combined)  # all_reduce -> total routed + shared (replicated)

    # ---- dense / shared SwiGLU MLP: one packed gate+up matmul, split, SwiGLU (else unpacked base) ---- #
    def _glu_mlp(self, x, key, H, I, ck, sharded):
        if not self.PACK_GATE_UP:
            return super()._glu_mlp(x, key, H, I, ck, sharded)
        guk, dk = {"mlp": ("mlp_gate_up", "mlp_down"), "sh": ("sh_gate_up", "sh_down")}[key]
        w = self.w
        if sharded and self.use_dram_sharded:
            gu = self._dram_mm(x, w[guk], w[guk + "_ds"], H, 2 * I, ck)  # [.,.,M,2I] width-sharded
            gu = ttnn.sharded_to_interleaved(gu, ttnn.L1_MEMORY_CONFIG)
            shp = list(gu.shape)
            g = ttnn.slice(gu, [0] * len(shp), shp[:-1] + [I])
            u = ttnn.slice(gu, [0] * (len(shp) - 1) + [I], shp[:-1] + [2 * I])
            gg = ttnn.mul(ttnn.silu(g), u)
            out = self._dram_mm(gg, w[dk], w[dk + "_ds"], I, H, ck)
            return ttnn.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG)
        # prefill: interleaved packed gate+up linear, split, SwiGLU, down
        gu = ttnn.linear(x, w[guk], compute_kernel_config=ck)  # [.,seq,2I]
        shp = list(gu.shape)
        g = ttnn.slice(gu, [0] * len(shp), shp[:-1] + [I])
        u = ttnn.slice(gu, [0] * (len(shp) - 1) + [I], shp[:-1] + [2 * I])
        gg = ttnn.mul(ttnn.silu(g), u)
        return ttnn.linear(gg, w[dk], compute_kernel_config=ck)

    # ---- dense/shared MLP: gate/up column, down row -> caller reduces ------ #
    def _mlp(self, ln, T, sharded):
        cfg = self.cfg
        ln_flat = ttnn.reshape(ln, (1, 1, T, cfg.hidden))
        if not cfg.is_moe:
            partial = self._glu_mlp(ln_flat, "mlp", cfg.hidden, cfg.intermediate, self._ck_dense, sharded)
            return self._reduce(ttnn.reshape(partial, (1, 1, T, cfg.hidden)))
        if T <= self.MOE_PREFILL_CHUNK:
            return self._moe(ln_flat, T, sharded)
        outs = []
        for s in range(0, T, self.MOE_PREFILL_CHUNK):
            e = min(s + self.MOE_PREFILL_CHUNK, T)
            chunk = ttnn.slice(ln_flat, [0, 0, s, 0], [1, 1, e, cfg.hidden])
            outs.append(self._moe(chunk, e - s, sharded))
        return ttnn.concat(outs, dim=2)

    # ---- prefill (single shot): reuse optimized body + all_reduce after WO --- #
    def prefill_forward(self, x_BSH, kv_cache, page_table, *, user_id=0, start_pos=0, rope_mats=None):
        seq = x_BSH.shape[-2]
        if seq > self.PIPE_CHUNK:
            return self._prefill_pipelined(x_BSH, kv_cache, page_table, user_id, start_pos)
        cfg = self.cfg
        residual = x_BSH
        ln = self._rms(x_BSH, self.w["input_ln"])
        q, k, v = self._qkv_roped(ln, seq, start_pos, rope=rope_mats)
        cdt = kv_cache["dtype"]
        # paged_fill_cache writes k/v into the blocks listed in the page table
        # starting at its first column, so the table must be offset to the block
        # containing start_pos. Cold prefill (start_pos=0) => col0=0 (unchanged);
        # a prefix-cache suffix prefill (start_pos>0) must NOT write at position 0
        # (that would clobber the cached prefix's first blocks). Mirrors the
        # per-chunk slicing in _prefill_pipelined.
        bs = kv_cache["block_size"]
        col0 = start_pos // bs
        ncol = (seq + bs - 1) // bs
        fill_pt = ttnn.slice(page_table, [0, col0], [page_table.shape[0], col0 + ncol])
        ttnn.experimental.paged_fill_cache(kv_cache["k"], self._cast_fill(k, cdt), fill_pt, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(kv_cache["v"], self._cast_fill(v, cdt), fill_pt, batch_idx=user_id)
        attn = self._prefill_attention(q, k, v, kv_cache, page_table, user_id, start_pos, seq)
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # item 2.5
        attn = ttnn.reshape(attn, (1, seq, cfg.num_heads * cfg.head_dim))
        attn = self._gate(attn, ln)
        o = ttnn.linear(attn, self.w["wo"], compute_kernel_config=self._ck_o)  # row-parallel partial
        o = self._reduce(o)
        h = ttnn.add(residual, o)
        ln2 = self._rms(h, self.w["post_ln"])
        mlp_out = ttnn.reshape(self._mlp(ln2, seq, sharded=False), (1, seq, cfg.hidden))
        return ttnn.add(h, mlp_out)

    def _prefill_pipelined(self, x_BSH, kv_cache, page_table, user_id, start_pos):
        cfg = self.cfg
        seq = x_BSH.shape[-2]
        bs = kv_cache["block_size"]
        cdt = kv_cache["dtype"]
        CH = (self._prefill_pipe_chunk // bs) * bs  # env-gated outer chunk (TT_LAGUNA_PREFILL_FAST)
        win = cfg.sliding_window
        outs = []
        for c in range(0, seq, CH):
            ch = min(CH, seq - c)
            gpos = start_pos + c
            xc = ttnn.slice(x_BSH, [0, c, 0], [1, c + ch, cfg.hidden])
            residual = xc
            ln = self._rms(xc, self.w["input_ln"])
            q, k, v = self._qkv_roped(ln, ch, gpos)
            col0 = gpos // bs
            ncol = (ch + bs - 1) // bs
            chunk_pt = ttnn.slice(page_table, [0, col0], [page_table.shape[0], col0 + ncol])
            ttnn.experimental.paged_fill_cache(kv_cache["k"], self._cast_fill(k, cdt), chunk_pt, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(kv_cache["v"], self._cast_fill(v, cdt), chunk_pt, batch_idx=user_id)
            user_pt = ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[1]])
            if not cfg.is_sliding:
                attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q,
                    kv_cache["k"],
                    kv_cache["v"],
                    user_pt,
                    chunk_start_idx=gpos,
                    compute_kernel_config=self._sdpa_compute,
                )
            else:
                # Sliding layers read from the paged window cache (not a local
                # k_tail), so a cached prefix (prefix caching) is visible: the
                # window can span back into positions filled by an earlier
                # prefill. ttnn chunked SDPA now composes with sliding_window_size
                # (see memory laguna-ttnn-incremental-rebuild; PoC PCC 1.0 vs the
                # dense sliding op). page_table is full-width with null-padded
                # out-of-window blocks (vLLM SlidingWindowManager), so absolute
                # chunk_start_idx=gpos indexing stays correct against the smaller
                # physical pool.
                attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q,
                    kv_cache["k"],
                    kv_cache["v"],
                    user_pt,
                    chunk_start_idx=gpos,
                    sliding_window_size=win,
                    compute_kernel_config=self._sdpa_compute,
                )
            attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # item 2.5
            attn = ttnn.reshape(attn, (1, ch, cfg.num_heads * cfg.head_dim))
            attn = self._gate(attn, ln)
            o = ttnn.linear(attn, self.w["wo"], compute_kernel_config=self._ck_o)
            o = self._reduce(o)
            h = ttnn.add(residual, o)
            ln2 = self._rms(h, self.w["post_ln"])
            mlp_out = ttnn.reshape(self._mlp(ln2, ch, sharded=False), (1, ch, cfg.hidden))
            outs.append(ttnn.add(h, mlp_out))
        return ttnn.concat(outs, dim=1)

    # ---- decode: reuse optimized body + all_reduce after WO --------------- #
    def decode_forward(self, x_1BH, cur_pos, rope_idx, page_table, kv_cache, sequential_kv_write=False, rope_mats=None):
        cfg = self.cfg
        B = x_1BH.shape[-2]
        residual = x_1BH
        ln = self._rms(x_1BH, self.w["input_ln"])
        qkv = self._dram_mm(ln, self.w["wqkv"], self.w["wqkv_ds"], cfg.hidden, self.meta["qkv_w"], self._ck_qkv)
        if self.use_dram_sharded:
            qkv = ttnn.sharded_to_interleaved(qkv, ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = self._split_qkv(qkv, B)
        q = self._per_head_norm(q, self.w["q_norm"])
        k = self._per_head_norm(k, self.w["k_norm"])
        # item 2.3: share the DRAM cos/sin gather across layers of a kind (rope_mats); shard to L1
        # PER LAYER (an L1-sharded cos_sh cannot be hoisted — scratch, clobbered by later layers).
        if rope_mats is None:  # local fallback (layer PCC tests / direct callers)
            cos = self._rope_decode(rope_idx, B)
            sin = self._rope_decode(rope_idx, B, sin=True)
        else:
            cos, sin = rope_mats
        if self._use_fused_rope:  # gated (TT_LAGUNA_FUSED_ROPE=1): fused HF rotate_half — PCC-validate first
            cos_sh = self._shard_cossin(cos, B, cfg.rotary_dim)
            sin_sh = self._shard_cossin(sin, B, cfg.rotary_dim)
            q = self._fused_rope_decode(q, cos_sh, sin_sh, cfg.num_heads, B)
            k = self._fused_rope_decode(k, cos_sh, sin_sh, cfg.num_kv_heads, B)
        else:
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)
        k_sh = self._shard_kv(k, B)
        v_sh = self._shard_kv(v, B)
        if sequential_kv_write and B > 1:
            # Spec-decode VERIFY: the B candidate rows share ONE user's blocks; with BLOCK_SIZE==TILE==32
            # consecutive positions land in the same tile, so a batched paged_update_cache RMW-races and
            # corrupts KV. Serialize the tiny per-row writes (matmuls + SDPA above/below still run batched,
            # so the verify stays fast). Mirrors gemma4/tt/attention/decode.py:147.
            self._seq_kv_write(kv_cache["k"], k_sh, cur_pos, page_table, B)
            self._seq_kv_write(kv_cache["v"], v_sh, cur_pos, page_table, B)
        else:
            ttnn.experimental.paged_update_cache(kv_cache["k"], k_sh, update_idxs_tensor=cur_pos, page_table=page_table)
            ttnn.experimental.paged_update_cache(kv_cache["v"], v_sh, update_idxs_tensor=cur_pos, page_table=page_table)
        sdpa_kwargs = {
            "cur_pos_tensor": cur_pos,
            "page_table_tensor": page_table,
            "scale": cfg.scaling,
            "compute_kernel_config": self._sdpa_compute,
        }
        if cfg.is_sliding:
            sdpa_kwargs["sliding_window_size"] = cfg.sliding_window
        if sequential_kv_write:  # spec-decode verify. k_chunk via TT_LAGUNA_VERIFY_K (default 64 = accurate SDPA)
            # ACCURACY FIX (2026-08-06): this is the LIVE served decode_forward (overrides OptimizedDecoder's);
            # it previously ran _sdpa_pc (k128) — the SAME config proven LOSSY on normal decode (teacher top1
            # 0.95→0.58). Committed spec tokens ARE this verify's argmax, so k128 made spec-decode inherit the
            # lossy trajectory. _sdpa_pc_verify (TT_LAGUNA_VERIFY_K, default 64) aligns it with the accurate
            # normal-decode SDPA. The disproven "verify REQUIRES k128" claim is dropped (k64 verify runs
            # correctly on device — standalone driver + served HumanEval). See tt/optimized_decoder.py.
            sdpa_kwargs["program_config"] = self._sdpa_pc_verify
            sdpa_kwargs["num_kv_heads"] = cfg.num_kv_heads
        elif self._decode_use_sdpa_pc:
            # Stage 2 fix: accuracy-safe fast NORMAL-decode config — k_chunk=64 (not 128) keeps the
            # max_cores=16 parallel-KV-scan long-context speed but drops the k128 last-partial-chunk masking
            # that was LOSSY (teacher top1 0.95→0.58, layer PCC -0.016 at low non-aligned cur_pos).
            # See tt/optimized_decoder.py (decode SDPA program config).
            sdpa_kwargs["program_config"] = self._sdpa_pc_decode
            sdpa_kwargs["num_kv_heads"] = cfg.num_kv_heads
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q, kv_cache["k"], kv_cache["v"], **sdpa_kwargs
        )
        attn = ttnn.reshape(attn, (1, 1, B, cfg.num_heads * cfg.head_dim))
        attn = self._gate(attn, ln)
        q_w = self.meta["q_w"]
        o = self._dram_mm(attn, self.w["wo"], self.w["wo_ds"], q_w, cfg.hidden, self._ck_o)
        if self.use_dram_sharded:
            o = ttnn.sharded_to_interleaved(o, ttnn.L1_MEMORY_CONFIG)
        o = self._reduce(o)  # row-parallel partial -> replicated
        h = ttnn.add(residual, o)
        ln2 = self._rms(h, self.w["post_ln"])
        mlp_out = self._mlp(ln2, B, sharded=True)
        return ttnn.add(h, mlp_out)
