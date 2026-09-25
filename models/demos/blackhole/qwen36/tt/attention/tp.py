# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel full-attention for Qwen3.5 (validated 64k+ on 27B).

Q/K-norm: HF-correct (1+weight) uniformly at prefill and decode.
Keep Q bf16 into SDPA unless bf8 mode (QWEN_SDPA_BF8=1).
Weights interleaved per device; x replicated in, output reduce-scattered on dim=3.
"""
import os

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention.rope_tp import (
    apply_partial_rope_decode,
    apply_partial_rope_prefill,
    shard_rope_tables_decode,
)
from models.tt_transformers.tt.ccl import tt_all_reduce


def load_attention_weights_tp(mesh, state_dict, args, cache_dir=None):
    """Shard one full-attention layer's weights across the mesh."""
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    def c(n):
        return str(cache_dir / n) if cache_dir is not None else None

    tw = {}
    # Column-parallel q/k/v: fused [q+gate|k|v] per device, or separate DRAM-sharded weights.
    # Distinct cache names — as_tensor reload ignores requested memcfg.
    fused_qkv = getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None
    # De-interleave [q,gate] per head → contiguous q/gate slices (avoids ~5.3ms relayout).
    qg_deint = fused_qkv

    # When TP > n_kv_heads there is no whole KV head per device, so pre-expand K/V; a no-op when tp <= n_kv_heads.
    kv_rep = lambda w: tpc.replicate_kv_weight(w, args.n_kv_heads, args.num_devices, args.head_dim)
    k_proj, v_proj = kv_rep(state_dict["k_proj.weight"]), kv_rep(state_dict["v_proj.weight"])

    if fused_qkv:
        if qg_deint:
            fused = tpc.prepare_attn_qkv_deint(
                state_dict["q_proj.weight"],
                k_proj,
                v_proj,
                args.n_local_heads,
                args.head_dim,
                args.n_local_kv_heads * args.head_dim,
                args.num_devices,
            )
        else:
            fused = tpc.prepare_attn_qkv(
                state_dict["q_proj.weight"],
                k_proj,
                v_proj,
                args.n_local_heads * args.head_dim * 2,
                args.n_local_kv_heads * args.head_dim,
                args.num_devices,
            )
        # proj_1d_decode: interleaved weight (fast small-grid 1D decode matmul; prefill AGMM verified
        # bit-identical on interleaved — test_agmm_accepts_interleaved_weight). Distinct cache suffix.
        _proj1d = getattr(args, "proj_1d_decode", False)
        _base = "wqkv_fused_qkvg" if qg_deint else "wqkv_fused"
        tw["wqkv_fused"] = tpc.shard_w(
            fused,
            mesh,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if _proj1d else args.attn_qkv_fused_weight_memcfg,
            cache_path=c(_base + (".il" if _proj1d else ".dramshard")),
            dtype=ttnn.bfloat8_b,
        )
    else:
        qkv_sharded = getattr(args, "attn_qg_weight_memcfg", None) is not None
        qg_mc = args.attn_qg_weight_memcfg if qkv_sharded else ttnn.DRAM_MEMORY_CONFIG
        k_mc = args.attn_k_weight_memcfg if qkv_sharded else ttnn.DRAM_MEMORY_CONFIG
        v_mc = args.attn_v_weight_memcfg if qkv_sharded else ttnn.DRAM_MEMORY_CONFIG
        tag = ".dramshard" if qkv_sharded else ""
        tw["wqkv"] = tpc.shard_w(
            state_dict["q_proj.weight"],
            mesh,
            dim=-1,
            memory_config=qg_mc,
            cache_path=c("wqkv" + tag),
            dtype=ttnn.bfloat8_b,
        )
        # k_proj/v_proj are the KV-replicated weights: shard_w splits tp*head_dim rows evenly, so
        # each device lands on its GQA-assigned head instead of a fraction of one.
        tw["wk"] = tpc.shard_w(
            k_proj,
            mesh,
            dim=-1,
            memory_config=k_mc,
            cache_path=c("wk" + tag),
            dtype=ttnn.bfloat8_b,
        )
        tw["wv"] = tpc.shard_w(
            v_proj,
            mesh,
            dim=-1,
            memory_config=v_mc,
            cache_path=c("wv" + tag),
            dtype=ttnn.bfloat8_b,
        )
    # Row-parallel wo (reduce-scatter after): DRAM-width-sharded like the in-proj — decode tput win.
    wo_sharded = getattr(args, "attn_wo_weight_memcfg", None) is not None
    tw["wo"] = tpc.shard_w(
        state_dict["o_proj.weight"],
        mesh,
        dim=0,
        memory_config=args.attn_wo_weight_memcfg if wo_sharded else ttnn.DRAM_MEMORY_CONFIG,
        cache_path=c("wo.dramshard" if wo_sharded else "wo"),
        dtype=ttnn.bfloat8_b,
    )
    # QK norms: HF-correct zero-centered (1+weight), used uniformly at prefill AND decode.
    # ROW_MAJOR gamma layout -> folded into ttnn.rms_norm's `weight=` (one kernel, not norm+multiply).
    tw["q_norm"] = tpc.replicate_rmsnorm_weight(state_dict["q_norm.weight"].to(torch.float32) + 1.0, mesh, None)
    tw["k_norm"] = tpc.replicate_rmsnorm_weight(state_dict["k_norm.weight"].to(torch.float32) + 1.0, mesh, None)
    return tw


class TPAttention:
    """Standalone TP full-attention with internal per-head KV caches (decode)."""

    def __init__(self, mesh, args, tw, tt_ccl):
        self.mesh = mesh
        self.args = args
        self.tw = tw
        self.tt_ccl = tt_ccl
        self.B = args.max_batch_size
        self._kv_shard_cfg_cache = {}  # active-width B -> KV-update height shard cfg (bucketed decode)
        self.NH = args.n_local_heads
        self.NKV = args.n_local_kv_heads
        self.HD = args.head_dim
        self.scale = self.HD**-0.5
        self.rope_dim = args.rope_head_dim
        self.compute_cfg = tpc.COMPUTE_HIFI2
        # bf8 SDPA (QWEN_SDPA_BF8=1): bf8 Q + bf8 KV; keeps HiFi2 (HiFi4 was slower)
        self._sdpa_bf8 = os.environ.get("QWEN_SDPA_BF8", "0") == "1"
        # Must match load_attention_weights_tp gates
        self._dram_sharded = getattr(args, "attn_qg_weight_memcfg", None) is not None
        self._wo_sharded = getattr(args, "attn_wo_weight_memcfg", None) is not None
        self._fused_qkv = getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None
        self._qg_deint = self._fused_qkv
        # Fuse prefill norm-allgather + fused-QKV in-proj (all_gather_minimal_matmul_async).
        # Norm's prefill post-AG disabled in layer.py; decode path unchanged.
        self._fuse_agmm = self._fused_qkv
        self.k_caches = None
        self.v_caches = None
        # External paged KV cache (vLLM/contract path); internal caches kept for demo fallback
        self.paged_k = None
        self.paged_v = None
        self.use_paged = False

    def set_paged_kv_cache(self, k_cache, v_cache):
        """Attach an externally-allocated paged KV cache (one call after allocate_kv_caches)."""
        self.paged_k = k_cache
        self.paged_v = v_cache
        self.use_paged = True

    def _qkv(self, x):
        """Q+gate/K/V projections → (qg, kp, vp). Fused path: one matmul, then slice."""
        tw = self.tw
        if not self._fused_qkv:
            return (
                self._col_proj(x, tw["wqkv"], self.args.attn_qg_progcfg),
                self._col_proj(x, tw["wk"], self.args.attn_k_progcfg),
                self._col_proj(x, tw["wv"], self.args.attn_v_progcfg),
            )
        # Prefill gathers x back to full K; long chunks use the FUSED all-gather+matmul, short ones gather separately. Output stays DRAM.
        if self._fuse_agmm and x.shape[-2] > tpc.TILE_SIZE:
            if x.shape[-2] >= tpc.AGMM_MIN_SEQ:
                qkv = tpc.all_gather_matmul_prefill(
                    x, tw["wqkv_fused"], self.tt_ccl, self.compute_cfg, self.args.ccl_topology()
                )
            else:
                x_full = tpc.all_gather_prefill(x, self.tt_ccl, self.args.ccl_topology())
                # Short prefill: the 2D kernel is core-starved at this M (tpc.short_prefill_1d_progcfg).
                pc_1d = tpc.short_prefill_1d_progcfg(
                    x_full.shape[-2],
                    self.args.dim,
                    tw["wqkv_fused"].shape[-1],
                    num_cores=40,
                    grid_w=self.args.decode_grid_w,
                )
                if pc_1d is not None:
                    qkv = ttnn.linear(
                        x_full,
                        tw["wqkv_fused"],
                        compute_kernel_config=self.compute_cfg,
                        program_config=pc_1d,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                else:
                    qkv = self._col_proj(x_full, tw["wqkv_fused"], self.args.attn_qkv_fused_progcfg)
                ttnn.deallocate(x_full)
        elif getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= tpc.TILE_SIZE:
            # Decode: small-grid 1D matmul (interleaved weight). Output DRAM so _make_heads_decode's
            # to_memory_config(.,L1) stays a real copy before it deallocates the source.
            qkv = tpc.matmul_1d_decode(
                x,
                tw["wqkv_fused"],
                self.args.attn_qkv_decode_1d_progcfg,
                self.compute_cfg,
                out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            qkv = self._col_proj(x, tw["wqkv_fused"], self.args.attn_qkv_fused_progcfg)
        # Fused weight is [q|k|v|gate] (prepare_attn_qkv_deint): the q|k|v block is contiguous, so
        # return it whole (no gate wedged between q and k → no re-concat in _make_heads*). Gate is
        # the trailing block. Sentinel: vp=None flags the fused/contiguous layout to _make_heads*.
        qkv3_dim = self.NH * self.HD + 2 * self.NKV * self.HD
        gate_dim = self.NH * self.HD
        sh = list(qkv.shape)
        # qkv3 short-lived (split by _make_heads then freed) -> L1 in PREFILL only; decode keeps DRAM
        # (L1 qkv3 breaks the decode trace). gate lives across SDPA (post-concat) -> always DRAM.
        _qkv3_mc = ttnn.L1_MEMORY_CONFIG if sh[2] > tpc.TILE_SIZE else ttnn.DRAM_MEMORY_CONFIG
        qkv3 = ttnn.slice(qkv, (0, 0, 0, 0), (sh[0], sh[1], sh[2], qkv3_dim), memory_config=_qkv3_mc)
        gate = ttnn.slice(qkv, (0, 0, 0, qkv3_dim), (sh[0], sh[1], sh[2], qkv3_dim + gate_dim))
        ttnn.deallocate(qkv)
        return qkv3, gate, None

    def _col_proj(self, x, weight, decode_progcfg):
        """Column-parallel projection; DRAM-sharded decode matmul when enabled."""
        if not self._dram_sharded:
            return ttnn.linear(x, weight, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            self.compute_cfg,
            decode_progcfg,
            self.args.act_shard_hidden,
            self.args.prefill_progcfg,
            self.args.dim,
        )

    def _wo_proj(self, x, weight):
        """Row-parallel output projection: DRAM-sharded decode/prefill matmul (K=attn_out_dim_tp),
        matching the in-proj. Falls back to plain interleaved when no sharded memcfg."""
        if getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= tpc.TILE_SIZE:
            # Decode: tuned ~32-core 1D matmul (interleaved weight) -> DRAM for the reduce-scatter.
            return tpc.matmul_1d_decode(
                x,
                weight,
                self.args.attn_wo_decode_1d_progcfg,
                self.compute_cfg,
                out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if not self._wo_sharded:
            if x.shape[-2] > tpc.TILE_SIZE:
                # Prefill: FPU-tuned 2D config beats ttnn-auto; short prefill switches to 1D, where the 2D kernel is core-starved.
                pc = tpc.short_prefill_1d_progcfg(
                    x.shape[-2],
                    weight.shape[-2],
                    weight.shape[-1],
                    num_cores=48,
                    grid_w=getattr(self.args, "decode_grid_w", 8),
                ) or tpc.create_prefill_mlp_matmul_program_config(
                    x.shape[-2],
                    weight.shape[-2],
                    weight.shape[-1],
                    max_cols=getattr(self.args, "decode_grid_w", 8),
                    tuning=getattr(self.args, "prefill_tuning", None),
                )
                return ttnn.linear(
                    x,
                    weight,
                    compute_kernel_config=self.compute_cfg,
                    program_config=pc,
                    # L1 output on BH only; WH has no room beside the CBs (tpc.prefill_l1_output_ok).
                    memory_config=ttnn.L1_MEMORY_CONFIG if tpc.prefill_l1_output_ok() else ttnn.DRAM_MEMORY_CONFIG,
                )
            return ttnn.linear(x, weight, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            self.compute_cfg,
            self.args.attn_wo_progcfg,
            self.args.act_shard_attn_out,
            self.args.prefill_progcfg,
            self.args.attn_out_dim_tp,
        )

    def _make_heads(self, qg, kp, vp, S):
        """Split qg into heads; returns (q, gate_flat, k, v) via fused nlp_create_qkv_heads.

        gate_flat stays flat [1,1,S,NH*HD] (col h*HD+d = head h, dim d), matching nlp_concat_heads'
        column order. Gate is applied AFTER concat_heads (see forward_prefill*), so no head-major
        reshape/transpose is needed; bit-identical to per-head gating, saves ~1 ms/attn-layer at S=2048.
        """
        NH, NKV, HD = self.NH, self.NKV, self.HD
        if vp is None:
            # Fused [q|k|v|gate] weight (_qkv sentinel vp=None): qg is the contiguous [q|k|v] block,
            # kp is the gate. Slice q and (already-contiguous) kv directly — no concat needed.
            gate_flat = kp
            # q_flat, kv feed nlp_create_qkv_heads then free immediately -> L1 (short-lived, no clash).
            q_flat = ttnn.slice(qg, (0, 0, 0, 0), (1, 1, S, NH * HD), memory_config=ttnn.L1_MEMORY_CONFIG)
            kv = ttnn.slice(
                qg, (0, 0, 0, NH * HD), (1, 1, S, NH * HD + 2 * NKV * HD), memory_config=ttnn.L1_MEMORY_CONFIG
            )
            ttnn.deallocate(qg)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                q_flat,
                kv,
                num_heads=NH,
                num_kv_heads=NKV,
                transpose_k_heads=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(q_flat)
            ttnn.deallocate(kv)
            return q, gate_flat, k, v
        # Interleaved qg: split [q;gate] per head; gate flattened to [1,1,S,NH*HD] (applied post-concat).
        qg = ttnn.reshape(qg, (1, S, NH, 2 * HD))
        q_part, gate_part = ttnn.chunk(qg, 2, dim=-1)
        ttnn.deallocate(qg)
        gate_flat = ttnn.reshape(gate_part, (1, 1, S, NH * HD))
        ttnn.deallocate(gate_part)
        q_flat = ttnn.reshape(q_part, (1, 1, S, NH * HD))
        ttnn.deallocate(q_part)
        kv = ttnn.concat([kp, vp], dim=-1)
        ttnn.deallocate(kp)
        ttnn.deallocate(vp)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            q_flat,
            kv,
            num_heads=NH,
            num_kv_heads=NKV,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_flat)
        ttnn.deallocate(kv)
        return q, gate_flat, k, v

    def _concat_heads(self, gated):
        """Prefill concat-heads via nlp_concat_heads (post-gate). L1 output: short-lived post-SDPA temp,
        no kernel-CB clash."""
        return ttnn.experimental.nlp_concat_heads(gated, memory_config=ttnn.L1_MEMORY_CONFIG)

    def _make_heads_decode(self, qg, kp, vp, B):
        """Decode head-split via nlp_create_qkv_heads_decode (the batched-decode idiom).

        Returns (q, gate_flat, k, v): q [1,B,NH,HD] and k/v [1,B,NKV,HD] HEIGHT-SHARDED one user per
        core (heads padded to a tile), gate_flat [1,1,B,NH*HD] flat. The kernel only shuffles a fused
        Q|K|V, so the gate half of qg is split off first and applied post-SDPA. The fused tensor is
        kept in L1 to dodge the Blackhole interleaved-reader bug (tt-metal #16667: DRAM input zeros
        odd-indexed Q rows).

        The head tensors stay SHARDED: that layout is exactly what paged_update_cache, the decode-mode
        rope and SDPA-decode want, so v reaches the KV cache with no data movement at all and q/k only
        round-trip through L1-interleaved for the rms_norm (which has no height-sharded kernel).
        gate_flat is left flat — the gate is applied after nlp_concat_heads_decode, whose column order
        is the same h*HD+d, so no head-major reshape is needed (bit-identical, saves a relayout).
        """
        NH, NKV, HD = self.NH, self.NKV, self.HD
        _L1 = ttnn.L1_MEMORY_CONFIG
        if vp is None:
            # Fused [q|k|v|gate] weight: qg is already the contiguous [q|k|v] the head-split wants. qkv MUST be L1 (tt-metal #16667 zeros odd Q rows from DRAM).
            qkv = ttnn.to_memory_config(qg, _L1)
            ttnn.deallocate(qg)
            gate_flat = kp
        else:
            # Interleaved qg: [q;gate] per head -> split then re-flatten to [1,1,B,NH*HD].
            qg_r = ttnn.reshape(qg, (1, B, NH, 2 * HD), memory_config=_L1)
            ttnn.deallocate(qg)
            q_part = ttnn.slice(qg_r, (0, 0, 0, 0), (1, B, NH, HD), memory_config=_L1)
            gate_part = ttnn.slice(qg_r, (0, 0, 0, HD), (1, B, NH, 2 * HD), memory_config=_L1)
            ttnn.deallocate(qg_r)
            q_flat = ttnn.reshape(q_part, (1, 1, B, NH * HD), memory_config=_L1)
            ttnn.deallocate(q_part)
            gate_flat = ttnn.reshape(gate_part, (1, 1, B, NH * HD), memory_config=_L1)
            ttnn.deallocate(gate_part)
            qkv = ttnn.concat([q_flat, kp, vp], dim=-1, memory_config=_L1)
            ttnn.deallocate(q_flat)
            ttnn.deallocate(kp)
            ttnn.deallocate(vp)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=NH, num_kv_heads=NKV, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)
        return q, gate_flat, k, v

    def _qk_norm_decode(self, x, weight):
        """(1+w) RMS norm over the head dim, returned in the head-shard layout it came in with.

        ttnn.rms_norm has no HEIGHT_SHARDED program factory (layernorm rejects it outright), so the
        head tensor round-trips through L1-interleaved. The gain is folded into the norm kernel via
        ``weight=`` (ROW_MAJOR gamma) instead of a separate broadcast multiply.
        """
        _L1 = ttnn.L1_MEMORY_CONFIG
        mc = x.memory_config()
        x_i = ttnn.sharded_to_interleaved(x, _L1)
        ttnn.deallocate(x)
        x_n = ttnn.rms_norm(x_i, epsilon=1e-6, weight=weight, memory_config=_L1)
        ttnn.deallocate(x_i)
        out = ttnn.to_memory_config(x_n, mc)
        ttnn.deallocate(x_n)
        return out

    def _concat_heads_decode(self, attn, head_mc, B):
        """Decode concat-heads via nlp_concat_heads_decode. attn [1,B,NH,HD] -> [1,1,B,NH*HD] L1.

        The op wants a height-sharded input ([1,B,heads-padded-to-32,HD], one core per user) — the
        very layout nlp_create_qkv_heads_decode produced and SDPA-decode wrote back into, so the
        reshard below is a no-op except on the multi-KV-head configs where SDPA can't emit sharded.
        Output is width-sharded, then returned to L1-interleaved for the gate + o_proj matmul.
        """
        NH, HD = self.NH, self.HD
        _L1 = ttnn.L1_MEMORY_CONFIG
        if attn.memory_config() != head_mc:
            attn_sh = ttnn.to_memory_config(attn, head_mc)
            ttnn.deallocate(attn)
            attn = attn_sh
        out_sh = ttnn.experimental.nlp_concat_heads_decode(attn, num_heads=NH)
        ttnn.deallocate(attn)
        out = ttnn.sharded_to_interleaved(out_sh, _L1)  # [1, 1, 32, NH*HD] (batch padded to 32)
        ttnn.deallocate(out_sh)
        # nlp_concat_heads_decode always emits batch padded to 32; slice back to the real B (a no-op
        # at B=32, required for B<32 e.g. the B=1 demo/vLLM path).
        if out.shape[-2] != B:
            out_b = ttnn.slice(out, (0, 0, 0, 0), (1, 1, B, NH * HD), memory_config=_L1)
            ttnn.deallocate(out)
            out = out_b
        return out

    def reset_state(self):
        def z():
            return ttnn.from_torch(
                torch.zeros(self.B, 1, self.args.max_seq_len, self.HD, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

        self.k_caches = [z() for _ in range(self.NKV)]
        self.v_caches = [z() for _ in range(self.NKV)]

    def forward_prefill(self, x, cos_tt, sin_tt):
        """Causal prefill. x [1,1,S,dim]: K-sharded (dim/tp per device) when the fused in-proj
        AG-matmul path is active (``_fuse_agmm`` and S>TILE — the norm skips its post-AG); replicated
        otherwise. Output reduce-scattered on dim=3."""
        tw, NH, NKV, HD = self.tw, self.NH, self.NKV, self.HD
        S = x.shape[-2]

        qg, kp, vp = self._qkv(x)

        q, gate_flat, k, v = self._make_heads(qg, kp, vp, S)

        q = ttnn.rms_norm(q, epsilon=1e-6, weight=tw["q_norm"], memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.rms_norm(k, epsilon=1e-6, weight=tw["k_norm"], memory_config=ttnn.L1_MEMORY_CONFIG)
        q = apply_partial_rope_prefill(q, cos_tt, sin_tt, NH, self.rope_dim)
        k = apply_partial_rope_prefill(k, cos_tt, sin_tt, NKV, self.rope_dim)

        # Fill per-head KV cache for decode (stateful path only)
        if self.k_caches is not None:
            # Don't deallocate slices — for NKV==1 they alias k/v used by SDPA
            for h in range(NKV):
                ttnn.fill_cache(self.k_caches[h], ttnn.slice(k, (0, h, 0, 0), (1, h + 1, S, HD)), 0)
                ttnn.fill_cache(self.v_caches[h], ttnn.slice(v, (0, h, 0, 0), (1, h + 1, S, HD)), 0)

        q8, k8, v8 = q, k, v
        padded = max(32, ((S + 31) // 32) * 32)
        # SDPA flash chunk: 128 for S>=2048, 64 below. 256 wins in isolation but its CBs clash with the resident attn input in the full model.
        ch = min(128 if S >= 2048 else 64, padded)
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=ch, k_chunk_size=ch
        )
        attn = ttnn.transformer.scaled_dot_product_attention(
            q8, k8, v8, is_causal=True, scale=self.scale, memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=sdpa_cfg
        )
        ttnn.deallocate(q8)
        ttnn.deallocate(k8)
        ttnn.deallocate(v8)

        # Concat heads first, then gate: concat col h*HD+d == gate_flat col h*HD+d, so this is
        # bit-identical to per-head gating but skips the gate reshape+transpose to head-major.
        attn = self._concat_heads(attn)
        # concat(attn)+sigmoid(gate) in L1; gated stays DRAM (feeds the wo matmul_reduce_scatter — an L1
        # CCL activation risks clashing with its CBs).
        gated = ttnn.multiply(
            attn, ttnn.sigmoid(gate_flat, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(attn)
        ttnn.deallocate(gate_flat)
        partial = self._wo_proj(gated, tw["wo"])
        ttnn.deallocate(gated)
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _kv_shard_cfg(self, B):
        """Height shard for paged_update_cache (one user per core), sized to the ACTIVE width B.
        Returns the precomputed max-batch config unchanged when B==self.B (byte-identical prod path);
        builds a width-B config (B cores) for bucketed decode. Mirrors model_config.kv_update_shard_cfg."""
        if B == self.B:
            return self.args.kv_update_shard_cfg
        cfg = self._kv_shard_cfg_cache.get(B)
        if cfg is None:
            cols = next(c for c in range(min(8, B), 0, -1) if B % c == 0)
            cfg = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, self.HD),
                core_grid=ttnn.CoreGrid(x=cols, y=B // cols),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self._kv_shard_cfg_cache[B] = cfg
        return cfg

    def forward_decode(self, x, cur_pos_tt, cos_tt, sin_tt, page_table=None):
        tw, NH, NKV, HD = self.tw, self.NH, self.NKV, self.HD
        # Active decode width from the input; under bucketed decode everything below runs at this width and SDPA reads only these B users' pages.
        B = x.shape[-2]
        _L1 = ttnn.L1_MEMORY_CONFIG  # keep decode head-prep + attn output L1-resident
        use_paged = self.use_paged and page_table is not None
        if not use_paged and self.k_caches is None:
            self.reset_state()

        qg, kp, vp = self._qkv(x)

        # q/k/v come out HEIGHT-SHARDED, one user per core, heads padded to a tile — the layout the
        # rope, the KV-cache update and SDPA-decode all take, so they stay in it end to end.
        q, gate_flat, k, v = self._make_heads_decode(qg, kp, vp, B)
        head_mc = q.memory_config()

        # QK norm — (1+w), matching prefill/HF, folded into the rms_norm kernel.
        q = self._qk_norm_decode(q, tw["q_norm"])
        k = self._qk_norm_decode(k, tw["k_norm"])

        # Decode cos/sin arrive [1,B,1,rope_dim] interleaved; height-shard them once onto the head
        # grid (q and k share the pair) for the op's native decode mode.
        cos_s, sin_s = shard_rope_tables_decode(cos_tt, sin_tt, head_mc)
        q = apply_partial_rope_decode(q, cos_s, sin_s, NH, B, self.rope_dim)
        k = apply_partial_rope_decode(k, cos_s, sin_s, NKV, B, self.rope_dim)
        ttnn.deallocate(cos_s)
        ttnn.deallocate(sin_s)

        # SDPA-decode uses the real device grid, not a hardcoded 64: cores_per_head = grid/B, so the extra cores pay off at long context.
        _sdpa_grid = self.mesh.compute_with_storage_grid_size()
        sdpa_dec_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(_sdpa_grid.x, _sdpa_grid.y),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )
        # Emit SDPA straight into the head shard so nlp_concat_heads_decode needs no reshard. The op
        # refuses a sharded output for a multi-KV-head cache (is_gqa), which keeps the interleaved one.
        sdpa_out_mc = head_mc if NKV == 1 else _L1
        if use_paged:
            # External paged KV: update at cur_pos, then paged SDPA-decode. k/v are already the
            # one-user-per-core height shard paged_update_cache requires — no pad, no reshard.
            keys, values = self.paged_k, self.paged_v
            # paged_update_cache takes bf16/fp32 and casts to bf8 cache; decode K/V stay bf16 (prefill fill needs bf8)
            ttnn.experimental.paged_update_cache(keys, k, update_idxs_tensor=cur_pos_tt, page_table=page_table)
            ttnn.experimental.paged_update_cache(values, v, update_idxs_tensor=cur_pos_tt, page_table=page_table)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                keys,
                values,
                page_table_tensor=page_table,
                cur_pos_tensor=cur_pos_tt,
                scale=self.scale,
                program_config=sdpa_dec_cfg,
                memory_config=sdpa_out_mc,
            )
            ttnn.deallocate(q)
        else:
            # Internal per-head KV caches (test / generate_tp oracle path).
            if NKV == 1:
                ttnn.experimental.paged_update_cache(self.k_caches[0], k, update_idxs_tensor=cur_pos_tt)
                ttnn.experimental.paged_update_cache(self.v_caches[0], v, update_idxs_tensor=cur_pos_tt)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
            else:
                # One cache per KV head: slice the heads apart, which needs an interleaved tensor.
                k_i = ttnn.sharded_to_interleaved(k, _L1)
                ttnn.deallocate(k)
                v_i = ttnn.sharded_to_interleaved(v, _L1)
                ttnn.deallocate(v)
                for h in range(NKV):
                    k_h = ttnn.slice(k_i, (0, 0, h, 0), (1, B, h + 1, HD))
                    v_h = ttnn.slice(v_i, (0, 0, h, 0), (1, B, h + 1, HD))
                    # pad_and_free, NOT pad + deallocate: this pad aliases its input (tpc.pad_and_free).
                    k_hp = tpc.pad_and_free(k_h, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
                    v_hp = tpc.pad_and_free(v_h, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
                    _kv_cfg = self._kv_shard_cfg(B)
                    k_sh = ttnn.to_memory_config(k_hp, _kv_cfg)
                    v_sh = ttnn.to_memory_config(v_hp, _kv_cfg)
                    ttnn.deallocate(k_hp)
                    ttnn.deallocate(v_hp)
                    ttnn.experimental.paged_update_cache(self.k_caches[h], k_sh, update_idxs_tensor=cur_pos_tt)
                    ttnn.experimental.paged_update_cache(self.v_caches[h], v_sh, update_idxs_tensor=cur_pos_tt)
                    ttnn.deallocate(k_sh)
                    ttnn.deallocate(v_sh)
                ttnn.deallocate(k_i)
                ttnn.deallocate(v_i)

            if NKV == 1:
                k_full, v_full = self.k_caches[0], self.v_caches[0]
            else:
                k_full = ttnn.concat(self.k_caches, dim=1)
                v_full = ttnn.concat(self.v_caches, dim=1)

            # Non-paged oracle path only: bound the K-chunk, or the full-cache SDPA's CBs grow with max_seq_len and overrun the CCL semaphores.
            nonpaged_sdpa_cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=128
            )
            attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                k_full,
                v_full,
                cur_pos_tensor=cur_pos_tt,
                scale=self.scale,
                program_config=nonpaged_sdpa_cfg,
                memory_config=sdpa_out_mc,
            )
            ttnn.deallocate(q)

        # Merge heads FIRST, then gate: nlp_concat_heads_decode puts head h dim d at column h*HD+d,
        # which is exactly gate_flat's column order, so this is bit-identical to gating per head and
        # skips the gate's head-major relayout.
        out = self._concat_heads_decode(attn_out, head_mc, B)  # consumes attn_out
        gated_flat = ttnn.multiply(out, ttnn.sigmoid(gate_flat, memory_config=_L1), memory_config=_L1)
        ttnn.deallocate(out)
        ttnn.deallocate(gate_flat)
        wo_partial = self._wo_proj(gated_flat, tw["wo"])
        ttnn.deallocate(gated_flat)
        wo_partial = ttnn.reshape(wo_partial, (1, 1, B, wo_partial.shape[-1]))
        return tt_all_reduce(
            wo_partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward_prefill_paged(
        self,
        x,
        cos_tt,
        sin_tt,
        page_table,
        chunk_page_table=None,
        chunk_start_idx=0,
        chunk_start_idx_tensor=None,
        user_id=0,
    ):
        """Paged-KV prefill for one chunk: fill cache + chunked SDPA over prior chunks.

        x is K-sharded when the fused in-proj path is active (same contract as ``forward_prefill``).
        chunk_start_idx_tensor: optional device offset for FLEXIBLE chunked SDPA (one program
        per trace/bucket). chunk_start_idx (int) still sizes the page table host-side.
        """
        assert self.use_paged and self.paged_k is not None, "forward_prefill_paged requires a bound paged KV cache"
        tw, NH, NKV, HD = self.tw, self.NH, self.NKV, self.HD
        if chunk_start_idx is None:
            chunk_start_idx = 0
        S = x.shape[-2]

        qg, kp, vp = self._qkv(x)

        q, gate_flat, k, v = self._make_heads(qg, kp, vp, S)

        q = ttnn.rms_norm(q, epsilon=1e-6, weight=tw["q_norm"], memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.rms_norm(k, epsilon=1e-6, weight=tw["k_norm"], memory_config=ttnn.L1_MEMORY_CONFIG)
        q = apply_partial_rope_prefill(q, cos_tt, sin_tt, NH, self.rope_dim)
        k = apply_partial_rope_prefill(k, cos_tt, sin_tt, NKV, self.rope_dim)

        # bf8 SDPA: paged_fill_cache doesn't cast — cast K/V to cache dtype before fill
        if self._sdpa_bf8:
            _k8 = ttnn.typecast(k, ttnn.bfloat8_b)
            ttnn.deallocate(k)
            k = _k8
            _v8 = ttnn.typecast(v, ttnn.bfloat8_b)
            ttnn.deallocate(v)
            v = _v8

        # Fill this chunk into the paged cache
        k_paged, v_paged = self.paged_k, self.paged_v
        block_size = k_paged.shape[2]
        fill_page_table = chunk_page_table if chunk_page_table is not None else page_table
        page_len = fill_page_table.shape[1] * block_size
        if page_len < S:
            k_fill = ttnn.slice(k, (0, 0, 0, 0), (1, NKV, page_len, HD))
            v_fill = ttnn.slice(v, (0, 0, 0, 0), (1, NKV, page_len, HD))
        else:
            k_fill, v_fill = k, v
        ttnn.experimental.paged_fill_cache(k_paged, k_fill, fill_page_table, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(v_paged, v_fill, fill_page_table, batch_idx=user_id)
        if page_len < S:
            ttnn.deallocate(k_fill)
            ttnn.deallocate(v_fill)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Chunked SDPA over paged cache; keep Q bf16 unless bf8 mode (QWEN_SDPA_BF8=1), which also
        # makes the KV cache bf8 -> full bf8 matmul
        if self._sdpa_bf8:
            q8 = ttnn.typecast(q, dtype=ttnn.bfloat8_b)
            ttnn.deallocate(q)
        else:
            q8 = q

        # chunk_start_idx % q_chunk_size == 0; FLEXIBLE path uses one program per trace.
        # q/k_chunk=128 is valid (chunk_start always divisible by 2048) and faster than 64/256.
        if chunk_start_idx_tensor is not None:
            qk_chunk = 128
        else:
            cap = 128 if S >= 2048 else 64  # 128 beats 256
            qk_chunk = cap if not chunk_start_idx else min(cap, chunk_start_idx & -chunk_start_idx)
        # Full BH grid for SDPA perf (bit-identical to 8×8; see test_tp_chunked_prefill_pcc_sweep)
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
            exp_approx_mode=False,
            q_chunk_size=qk_chunk,
            k_chunk_size=qk_chunk,
        )

        # Pad page table to cover Q+offset and satisfy stick-size % 32 (extra blocks masked by causality)
        sdpa_page_table = page_table
        needed_blocks = (S + chunk_start_idx + block_size - 1) // block_size
        target_blocks = max(needed_blocks, page_table.shape[-1])
        target_blocks = ((target_blocks + 31) // 32) * 32
        if page_table.shape[-1] < target_blocks:
            zeros_pad = ttnn.zeros(
                (page_table.shape[0], target_blocks - page_table.shape[-1]),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            sdpa_page_table = ttnn.concat([page_table, zeros_pad], dim=-1)
            ttnn.deallocate(zeros_pad)

        if chunk_start_idx_tensor is not None:
            attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=q8,
                input_tensor_k=k_paged,
                input_tensor_v=v_paged,
                page_table_tensor=sdpa_page_table,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
                compute_kernel_config=self.compute_cfg,
                program_config=sdpa_cfg,
            )
        else:
            attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=q8,
                input_tensor_k=k_paged,
                input_tensor_v=v_paged,
                page_table_tensor=sdpa_page_table,
                chunk_start_idx=chunk_start_idx,
                compute_kernel_config=self.compute_cfg,
                program_config=sdpa_cfg,
            )
        if sdpa_page_table is not page_table:
            ttnn.deallocate(sdpa_page_table)
        ttnn.deallocate(q8)

        # Concat heads first, then gate (flat gate matches concat column order); see forward_prefill.
        attn = self._concat_heads(attn)
        # concat(attn)+sigmoid(gate) in L1; gated stays DRAM (feeds the wo matmul_reduce_scatter — an L1
        # CCL activation risks clashing with its CBs).
        gated = ttnn.multiply(
            attn, ttnn.sigmoid(gate_flat, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(attn)
        ttnn.deallocate(gate_flat)
        partial = self._wo_proj(gated, tw["wo"])
        ttnn.deallocate(gated)
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
