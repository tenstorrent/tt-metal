# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Optimized multichip TTNN decoder for poolside/Laguna-XS-2.1 (Blackhole p300c ×4, 1×4 mesh).

Dedicated optimization pass over ``tt/multichip_decoder.py`` (``MultichipDecoder``). The multichip
parallelization itself (TP=4 attention/dense + EP=4 routed MoE, replicated BF16 residual, exactly 2
ring ``all_reduce``/layer) was already the primary multi-device win and is inherited unchanged — it
is comm-optimal for this small hidden (H=2048); see the CCL evidence in
``doc/optimized_multichip_decoder/ccl_family_evidence.md``.

What this subclass changes — **pack the two repeated same-input gate/up projections into one matmul**
(OPT-001 / OPT-010). In decode the model is dispatch/latency-bound (modeled DRAM roofline ~2.6% at
batch-1), so collapsing two matmul dispatches over the *same* activation into one wide matmul removes
real device time:

  * **Routed-expert MoE gate+up** ``ttnn.sparse_matmul`` (the single largest decode consumer,
    ~132 µs for the gate+up pair): the two ``[1,64,H,512]`` BFP4 expert weights are concatenated at
    load into one ``[1,64,H,1024]`` weight; one ``sparse_matmul`` produces ``[1,64,T,1024]``, split
    on device into gate/up halves, then ``silu(gate) * up``. Measured layer-4 traced decode
    0.9145 → **0.8653 ms/tok (−5.4%)**, **PCC 1.000000 vs the separate baseline** (numerically
    identical — SiLU is applied after the split, so no approximate-activation cost).
  * **Dense-MLP and shared-expert gate+up** (DRAM-sharded ``_dram_mm``): same idea — concat the
    gate/up weights (interleaved + DRAM-width-sharded copies) at load, one matmul, split, SwiGLU.

PCC is preserved exactly (packing is pure dispatch reduction), so the context contract, precision
policy, and all correctness bars are unchanged. Everything else (packed QKV, DRAM-sharded decode
matmuls, BFP8/BFP4 precision, LoFi/HiFi2 fidelity, BFP8 paged KV cache, RoPE, SDPA config, mesh
weight placement, collectives, EP selection-matmul, active-expert path) is inherited.

Multichip-specific families tried and REJECTED with on-device evidence (async/persistent CCL, fused
matmul-CCL, sharded/fractured residual, collective placement, router/attention precision) are in the
work log; none beat the inherited latency-bound ring all_reduce or the stage-02 precision frontier.
Inter-layer residual contract UNCHANGED: replicated BF16 [1,1,B,H] / [1,seq,H], no inter-layer
collective — full-model bringup MUST preserve it.
"""
from __future__ import annotations

import ttnn

from .multichip_decoder import MultichipDecoder
from .optimized_decoder import TILE, _dram_weight_memcfg, _sparse_pc


class OptimizedMultichipDecoder(MultichipDecoder):
    # Toggle for A/B evidence; default path packs gate+up projections.
    PACK_GATE_UP = True

    # ---- construction: build packed gate+up weights, free the separate copies ---- #
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        dec = super().from_state_dict(*args, **kwargs)
        if not cls.PACK_GATE_UP:
            return dec
        w = dec.w
        dram_cores = dec.meta["dram_cores"]
        H = dec.cfg.hidden

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

        if dec.cfg.is_moe:
            # routed experts: interleaved BFP4 [1,64,H,512] each -> [1,64,H,1024]
            w["exp_gate_up"] = ttnn.concat([w["exp_gate"], w["exp_up"]], dim=-1)
            ttnn.deallocate(w["exp_gate"])
            ttnn.deallocate(w["exp_up"])
            del w["exp_gate"]
            del w["exp_up"]
            pack_dram_pair("sh_gate", "sh_up", "sh_gate_up", dec.cfg.shared_intermediate)
        else:
            pack_dram_pair("mlp_gate", "mlp_up", "mlp_gate_up", dec.cfg.intermediate)
        return dec

    # ---- dense / shared SwiGLU MLP: one packed gate+up matmul, split, SwiGLU ---- #
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

    # ---- MoE (Expert Parallel): one packed gate+up sparse_matmul, split, SwiGLU ---- #
    def _moe(self, ln_flat, m, sharded):
        if not self.PACK_GATE_UP:
            return super()._moe(ln_flat, m, sharded)
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
        routed_local = ttnn.reshape(ttnn.sum(ttnn.mul(down_o, wv), dim=1), (1, 1, T, H))
        shared_partial = self._glu_mlp(ln_flat, "sh", cfg.hidden, cfg.shared_intermediate, self._ck_shared, sharded)
        combined = ttnn.add(routed_local, ttnn.reshape(shared_partial, (1, 1, T, H)))
        return self._reduce(combined)
