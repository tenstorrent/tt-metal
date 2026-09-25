# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel full-attention for Qwen3.5 (validated 64k+ on 27B).

Q/K-norm: HF-correct (1+weight) uniformly at prefill and decode.
Keep Q bf16 into SDPA unless bf8 mode (QWEN_SDPA_BF8=1).
Weights interleaved per device; x replicated in, output reduce-scattered on dim=3.
"""
import hashlib
import os

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention import rope_tp
from models.demos.blackhole.qwen36.tt.attention.rope_tp import (
    apply_partial_rope_decode,
    apply_partial_rope_prefill,
    apply_rope_full_decode,
    apply_rope_full_prefill,
    shard_rot_mats_decode,
)
from models.tt_transformers.tt.ccl import tt_all_reduce


def _t3k_wh(args):
    """T3K (8-chip Wormhole). Same predicate as tp_common.wh_t3k."""
    return tpc.wh_t3k(args)


def _kv_no_pad(args):
    """Unpadded KV reshard. Mutually exclusive with permuted RoPE (paged_update_cache segfault). N150 stays padded."""
    return _t3k_wh(args)


def _rp_cache_tag(source_tensor, args):
    """Cache-name hash of the pre-permutation weight, ROPE_PERM_VERSION, and the perm dims."""
    h = hashlib.sha256()
    h.update(rope_tp.ROPE_PERM_VERSION.encode())
    h.update(repr((tuple(source_tensor.shape), str(source_tensor.dtype), args.head_dim, args.rope_head_dim)).encode())
    # .view(uint8) reinterprets raw bytes, including bfloat16.
    h.update(source_tensor.contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:16]


def load_attention_weights_tp(mesh, state_dict, args, cache_dir=None):
    """Shard one full-attention layer's weights across the mesh."""
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    def c(n):
        return str(cache_dir / n) if cache_dir is not None else None

    tw = {}
    # GQA: expand k/v so every device owns one whole KV head.
    k_proj = tpc.replicate_kv_weight(state_dict["k_proj.weight"], args.n_kv_heads, args.num_devices, args.head_dim)
    v_proj = tpc.replicate_kv_weight(state_dict["v_proj.weight"], args.n_kv_heads, args.num_devices, args.head_dim)

    # Fold rope_channel_perm into the weights; no runtime op.
    rope_permuted = getattr(args, "rope_permuted_enabled", False)
    q_proj = state_dict["q_proj.weight"]
    q_norm_w = state_dict["q_norm.weight"].to(torch.float32) + 1.0
    k_norm_w = state_dict["k_norm.weight"].to(torch.float32) + 1.0
    rp_q = rp_k = rp_fused = ""
    if rope_permuted:
        hd, rd = args.head_dim, args.rope_head_dim
        q_tag = _rp_cache_tag(q_proj, args)
        k_tag = _rp_cache_tag(k_proj, args)
        rp_q = f".rp.{q_tag}"
        rp_k = f".rp.{k_tag}"
        rp_fused = f".rp.{q_tag}.{k_tag}"
        q_proj = rope_tp.permute_rope_channels(q_proj, hd, rd, mesh, stride=2 * hd)
        k_proj = rope_tp.permute_rope_channels(k_proj, hd, rd, mesh)
        q_norm_w = rope_tp.permute_rope_channels(q_norm_w, hd, rd, mesh)
        k_norm_w = rope_tp.permute_rope_channels(k_norm_w, hd, rd, mesh)

    # Column-parallel q/k/v: fused [q+gate|k|v] per device, or separate DRAM-sharded weights.
    # Distinct cache names — as_tensor reload ignores requested memcfg.
    fused_qkv = getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None
    # De-interleave [q,gate] per head → contiguous q/gate slices (avoids ~5.3ms relayout).
    qg_deint = fused_qkv

    if fused_qkv:
        if qg_deint:
            fused = tpc.prepare_attn_qkv_deint(
                q_proj,
                k_proj,
                v_proj,
                args.n_local_heads,
                args.head_dim,
                args.n_local_kv_heads * args.head_dim,
                args.num_devices,
            )
        else:
            fused = tpc.prepare_attn_qkv(
                q_proj,
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
            cache_path=c(_base + (".il" if _proj1d else ".dramshard") + rp_fused),
            dtype=ttnn.bfloat8_b,
        )
    else:
        qkv_sharded = getattr(args, "attn_qg_weight_memcfg", None) is not None
        qg_mc = args.attn_qg_weight_memcfg if qkv_sharded else ttnn.DRAM_MEMORY_CONFIG
        k_mc = args.attn_k_weight_memcfg if qkv_sharded else ttnn.DRAM_MEMORY_CONFIG
        v_mc = args.attn_v_weight_memcfg if qkv_sharded else ttnn.DRAM_MEMORY_CONFIG
        tag = ".dramshard" if qkv_sharded else ""
        tw["wqkv"] = tpc.shard_w(
            q_proj,
            mesh,
            dim=-1,
            memory_config=qg_mc,
            cache_path=c("wqkv" + tag + rp_q),
            dtype=ttnn.bfloat8_b,
        )
        # k_proj/v_proj are the KV-replicated weights: shard_w splits tp*head_dim rows evenly, so
        # each device lands on its GQA-assigned head instead of a fraction of one.
        tw["wk"] = tpc.shard_w(
            k_proj,
            mesh,
            dim=-1,
            memory_config=k_mc,
            cache_path=c("wk" + tag + rp_k),
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
    # QK norms: HF-correct zero-centered (1+weight), used uniformly at prefill AND decode
    tw["q_norm"] = tpc.replicate(q_norm_w, mesh, None)
    tw["k_norm"] = tpc.replicate(k_norm_w, mesh, None)
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
        self._kv_fused_shard_cfg_cache = {}  # active-width B -> (K, V) disjoint fused-write shard cfgs
        self._spec_sdpa_cfg_cache = {}  # T -> fused spec-verify SDPA (progcfg, groups, tiles); None = no fit
        self.NH = args.n_local_heads
        self.NKV = args.n_local_kv_heads
        self.HD = args.head_dim
        self.scale = self.HD**-0.5
        self.rope_dim = args.rope_head_dim
        self.compute_cfg = tpc.COMPUTE_HIFI2
        # bf8 Q/KV + HiFi2. QWEN_SDPA_BF8 overrides; see tp_common.sdpa_bf8_enabled.
        self._sdpa_bf8 = tpc.sdpa_bf8_enabled(args)
        # Must match load_attention_weights_tp gates
        self._dram_sharded = getattr(args, "attn_qg_weight_memcfg", None) is not None
        self._wo_sharded = getattr(args, "attn_wo_weight_memcfg", None) is not None
        self._fused_qkv = getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None
        self._qg_deint = self._fused_qkv
        # Fuse prefill norm-allgather + fused-QKV in-proj (all_gather_minimal_matmul_async).
        # Norm's prefill post-AG disabled in layer.py; decode path unchanged.
        self._fuse_agmm = self._fused_qkv and tpc.is_blackhole()
        # Decode head split/merge via nlp_create/concat_heads_decode (the batched-decode idiom).
        self._use_nlp_decode_heads = True
        # Must match load_attention_weights_tp: the permutation lives in the weights.
        self._rope_permuted = getattr(args, "rope_permuted_enabled", False)
        # None uses the ttnn default of 16. Only the MTP drafter sets this.
        self.decode_sdpa_max_cores = None
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
        # Prefill: x is K-sharded (norm skipped its AG) -> fused all-gather + QKV matmul. Output stays
        # DRAM: L1 clashes with a downstream matmul's CBs (verified; full-attn has more L1 pressure here).
        if self._fuse_agmm and x.shape[-2] > tpc.TILE_SIZE:
            qkv = tpc.all_gather_matmul_prefill(
                x, tw["wqkv_fused"], self.tt_ccl, self.compute_cfg, self.args.ccl_topology()
            )
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
            qkv = self._col_proj(
                x,
                tw["wqkv_fused"],
                self.args.attn_qkv_fused_progcfg,
                prefill_progcfg_fn=getattr(self.args, "attn_qkv_fused_prefill_progcfg", None),
            )
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

    def _kv_update_shard_cfg(self, n, HD):
        """HEIGHT-sharded n-row paged_update_cache input: one 32-row tile per core."""
        cache = getattr(self, "_kvu_cfg_cache", None)
        if cache is None:
            cache = self._kvu_cfg_cache = {}
        if n not in cache:
            gx = self.mesh.compute_with_storage_grid_size().x
            assert n <= gx, f"exact-KV write needs n({n}) <= grid width({gx}); use a smaller draft len"
            cache[n] = ttnn.create_sharded_memory_config(
                shape=(tpc.TILE_SIZE, HD),
                core_grid=ttnn.CoreGrid(x=n, y=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        return cache[n]

    def _col_proj(self, x, weight, decode_progcfg, prefill_progcfg_fn=None):
        """Column-parallel projection. prefill_progcfg_fn overrides only the prefill branch."""
        if not self._dram_sharded:
            return ttnn.linear(x, weight, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            self.compute_cfg,
            decode_progcfg,
            self.args.act_shard_hidden,
            prefill_progcfg_fn or self.args.prefill_progcfg,
            self.args.dim,
            # LoFi on the kpass1 path: the BFP8 mantissa dominates the error.
            prefill_compute_cfg=tpc.COMPUTE_LOFI_NO_FP32_ACC if prefill_progcfg_fn is not None else None,
        )

    def _qk_norm(self, x, weight, memory_config):
        """Fuse the q/k-norm scale into rms_norm on Wormhole; Blackhole stays two ops."""
        if tpc.is_blackhole():
            return ttnn.multiply(
                ttnn.rms_norm(x, epsilon=1e-6, memory_config=memory_config), weight, memory_config=memory_config
            )
        return ttnn.rms_norm(x, weight=weight, epsilon=1e-6, memory_config=memory_config)

    def _rope_decode(self, q, k, cos_tt, sin_tt, B):
        """Do not keep sharded cos/sin across a layer boundary: they clash with GDN L1."""
        if not self._rope_permuted:
            q = apply_partial_rope_decode(q, cos_tt, sin_tt, self.NH, B, self.rope_dim)
            k = apply_partial_rope_decode(k, cos_tt, sin_tt, self.NKV, B, self.rope_dim)
            return q, k
        cfg = self.args.rope_k_shard_cfg
        cos_sh, sin_sh = shard_rot_mats_decode(cos_tt, sin_tt, cfg)

        def _rope(x):
            x_sh = ttnn.to_memory_config(x, cfg)
            ttnn.deallocate(x)
            out = apply_rope_full_decode(x_sh, cos_sh, sin_sh)
            ttnn.deallocate(x_sh)
            return out

        q, k = _rope(q), _rope(k)
        ttnn.deallocate(cos_sh)
        ttnn.deallocate(sin_sh)
        return q, k

    def _rope_prefill(self, q, k, cos_tt, sin_tt):
        """Prefill RoPE for Q and K: one op each when permuted, else slice/rotate/slice/concat."""
        if not self._rope_permuted:
            return (
                apply_partial_rope_prefill(q, cos_tt, sin_tt, self.NH, self.rope_dim),
                apply_partial_rope_prefill(k, cos_tt, sin_tt, self.NKV, self.rope_dim),
            )
        return apply_rope_full_prefill(q, cos_tt, sin_tt), apply_rope_full_prefill(k, cos_tt, sin_tt)

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
                # Prefill: FPU-tuned 2D config beats ttnn-auto's 1x1 stall; L1 output (gated stays DRAM)
                # max_cols is the device width; L1 output feeds the separate reduce-scatter.
                _wo_kpass1 = getattr(self.args, "attn_wo_prefill_progcfg", None)
                if _wo_kpass1 is not None:
                    pc = _wo_kpass1(x.shape[-2], weight.shape[-2], weight.shape[-1])
                    # LoFi once weight and activation are BFP8; see COMPUTE_LOFI_NO_FP32_ACC.
                    ck = tpc.COMPUTE_LOFI_NO_FP32_ACC
                else:
                    pc = tpc.create_prefill_mlp_matmul_program_config(
                        x.shape[-2],
                        weight.shape[-2],
                        weight.shape[-1],
                        max_cols=getattr(self.args, "decode_grid_w", 8),
                        tuning=getattr(self.args, "prefill_tuning", None),
                    )
                    ck = self.compute_cfg
                _wo_bf8 = self.args.dim > 4096 and not tpc.is_blackhole()
                return ttnn.linear(
                    x,
                    weight,
                    compute_kernel_config=ck,
                    program_config=pc,
                    # L1 while the output fits; DRAM once it would crowd the matmul CBs.
                    memory_config=tpc.prefill_out_memory_config(x.shape[-2], weight.shape[-1]),
                    **({"dtype": ttnn.bfloat8_b} if _wo_bf8 else {}),
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
            # Pass the contiguous [q|k|v] block; nlp_create_qkv_heads splits in-kernel.
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qg,
                num_heads=NH,
                num_kv_heads=NKV,
                transpose_k_heads=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(qg)
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

    def _make_heads_decode(self, qg, kp, vp, B, skip_v_reshard=False):
        """Decode head-split via nlp_create_qkv_heads_decode (the batched-decode idiom).

        The kernel only shuffles a fused Q|K|V, so the gate half of qg is split off first and applied
        post-SDPA exactly like the reshape path. The fused tensor is kept in L1 to dodge the Blackhole
        interleaved-reader bug (DRAM input zeros odd-indexed Q rows, tt-metal #16667).
        """
        NH, NKV, HD = self.NH, self.NKV, self.HD
        _L1 = ttnn.L1_MEMORY_CONFIG
        if vp is None:
            # Fused [q|k|v|gate] weight (_qkv sentinel vp=None): qg is already the contiguous [q|k|v]
            # the decode head-split wants — feed it directly, no concat. kp is the gate. qkv must be
            # L1 (tt-metal #16667: DRAM input zeros odd Q rows); one to_memory_config replaces the
            # old 3-way concat (which had also served to land qkv in L1).
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
        q = ttnn.sharded_to_interleaved(q, _L1)
        k = ttnn.sharded_to_interleaved(k, _L1)
        if skip_v_reshard and v.memory_config() == self.args.kv_update_shard_cfg:
            pass  # v stays in its native sharded layout for the paged-cache write
        else:
            v = ttnn.sharded_to_interleaved(v, _L1)
        return q, gate_flat, k, v

    def _concat_heads_decode(self, attn_out, B, gate_flat=None):
        """Decode concat-heads via nlp_concat_heads_decode. attn_out [1,B,NH,HD] L1 -> [1,B,NH*HD] L1.

        The op wants a height-sharded input ([1,B,heads-padded-to-32,HD], one core per user), so the
        returned to L1-interleaved so the downstream o_proj matmul is unchanged.
        ``gate_flat`` is applied here: concat output is already the flat layout the gate has.
        """
        from models.tt_transformers.tt.model_config import num_to_corerange

        NH, HD = self.NH, self.HD
        _L1 = ttnn.L1_MEMORY_CONFIG
        grid = self.mesh.compute_with_storage_grid_size()
        gx = min(B, grid.x)
        if B >= gx and B % gx != 0:
            # nlp_concat_heads_decode needs a rectangular core set; a prime B larger than grid.x cannot form one.
            fits = [x for x in range(gx, 0, -1) if B % x == 0 and B // x <= grid.y]
            assert fits, (
                f"decode batch B={B} cannot be laid out one-user-per-core on this "
                f"{grid.x}x{grid.y} grid: B has no divisor <= {grid.x} whose quotient is <= {grid.y} "
                f"(B is prime and > {grid.x}), and the core set must be rectangular. Use a batch "
                f"width that factors into the grid; for spec decode that means a draft length K "
                f"whose K+1 does (K=10 -> T=11 and K=12 -> T=13 do NOT, K=6 -> T=7 and K=11 -> T=12 do)."
            )
            gx = max(fits)
        core_grid = ttnn.CoreRangeSet({num_to_corerange(B, grid_x=gx, grid_y=grid.y)})
        shard_cfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, HD),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        gated_sh = ttnn.to_memory_config(attn_out, shard_cfg)
        ttnn.deallocate(attn_out)
        out_sh = ttnn.experimental.nlp_concat_heads_decode(gated_sh, num_heads=NH)
        ttnn.deallocate(gated_sh)
        out = ttnn.sharded_to_interleaved(out_sh, _L1)  # [1, 1, 32, NH*HD] (batch padded to 32)
        ttnn.deallocate(out_sh)
        # nlp_concat_heads_decode always emits batch padded to 32; slice back to the real B before
        # the reshape (a no-op at B=32, required for B<32 e.g. the B=1 demo/vLLM path).
        if out.shape[-2] != B:
            out = ttnn.slice(out, (0, 0, 0, 0), (1, 1, B, NH * HD), memory_config=_L1)
        if gate_flat is not None:
            out = self._apply_gate(out, gate_flat, _L1)
        return ttnn.reshape(out, (1, B, NH * HD), memory_config=_L1)

    def _apply_gate(self, x, gate, memory_config):
        """Fused sigmoid-multiply. N150 and Blackhole stay unfused."""
        if tpc.wh_9b_n300(self.args) or _t3k_wh(self.args):
            out = ttnn.multiply(
                x, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID], memory_config=memory_config
            )
        else:
            out = ttnn.multiply(x, ttnn.sigmoid(gate, memory_config=memory_config), memory_config=memory_config)
        ttnn.deallocate(x)
        ttnn.deallocate(gate)
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

        q = self._qk_norm(q, tw["q_norm"], ttnn.L1_MEMORY_CONFIG)
        k = self._qk_norm(k, tw["k_norm"], ttnn.L1_MEMORY_CONFIG)
        q, k = self._rope_prefill(q, k, cos_tt, sin_tt)

        # Fill per-head KV cache for decode (stateful path only)
        if self.k_caches is not None:
            # Don't deallocate slices — for NKV==1 they alias k/v used by SDPA
            for h in range(NKV):
                ttnn.fill_cache(self.k_caches[h], ttnn.slice(k, (0, h, 0, 0), (1, h + 1, S, HD)), 0)
                ttnn.fill_cache(self.v_caches[h], ttnn.slice(v, (0, h, 0, 0), (1, h + 1, S, HD)), 0)

        q8, k8, v8 = q, k, v
        padded = max(32, ((S + 31) // 32) * 32)
        # SDPA flash chunk: 128 for S>=2048, 64 below. (256 wins in ISOLATION at S=3072/4096
        # -- test_sdpa_prefill_opt -- but in the full model its larger CBs clash with the resident
        # attn-input L1 buffer during a single-pass prefill of S>2048 (prefill_tp/generate_tp;
        # program.cpp "circular buffers ... clash with L1 buffers"). Production serving chunks
        # prefill at <=2048, so this path never sees S>2048 and 256 has no reachable win.)
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
            attn,
            ttnn.sigmoid(gate_flat, memory_config=ttnn.L1_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **({"dtype": ttnn.bfloat8_b} if (self.args.dim > 4096 and not tpc.is_blackhole()) else {}),
        )
        ttnn.deallocate(attn)
        ttnn.deallocate(gate_flat)
        partial = self._wo_proj(gated, tw["wo"])
        ttnn.deallocate(gated)
        # Pass prefill_ccl_tuning, matching the mlp and gdn reduce-scatters.
        _ccl_kw = {}
        if S > tpc.TILE_SIZE:
            _cps, _wpl = tpc.prefill_ccl_tuning()
            _ccl_kw = {"chunks_per_sync": _cps, "num_workers_per_link": _wpl}
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **_ccl_kw,
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

    def _kv_fused_shard_cfgs(self, B):
        """Disjoint K/V grids sized to active width B. A core past B reads the page table out of range."""
        if B == self.B:
            return self.args.kv_cache_write_k_shard_cfg, self.args.kv_cache_write_v_shard_cfg
        cfgs = self._kv_fused_shard_cfg_cache.get(B)
        if cfgs is None:
            cols = next(c for c in range(min(8, B), 0, -1) if B % c == 0)
            rows = B // cols
            k_cfg = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, self.HD),
                core_grid=ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, rows), ttnn.CoreCoord(cols - 1, 2 * rows - 1))}
                ),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            v_cfg = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, self.HD),
                core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))}),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            cfgs = (k_cfg, v_cfg)
            self._kv_fused_shard_cfg_cache[B] = cfgs
        return cfgs

    # Tg sizes the SDPA L1. A T with no listed split takes the legacy B=T path.
    _SPEC_SDPA_L1_FIT = {
        4: (1, 64, 0),  # Tg=4, one group, 64 cores/head
        8: (2, 55, 0),  # two groups of 4
        12: (
            3,
            36,
            0,
        ),  # three groups of 4
    }

    # Wormhole: same Tg=4 splits, cores-per-head sized to a 64-core grid.
    _SPEC_SDPA_L1_FIT_WH = {
        4: (1, 64, 0),  # one group, whole 64-core grid on one reduction group
        8: (2, 32, 0),  # two groups of 4, 32 cores/head -> 64 active
        12: (3, 21, 0),  # three groups of 4, 21 cores/head -> 63 active (1 idle)
        # Capability table only; the K policy lives in demo/text_demo.py.
        16: (4, 16, 0),  # four groups of 4, 16 cores/head -> 64 active
        20: (5, 12, 0),  # five groups of 4, 12 cores/head -> 60 active (4 idle)
    }

    def _spec_sdpa_plan(self, T):
        """SDPA program config for the fused spec-verify path, or None when T has no L1-fitting split."""
        if T not in self._spec_sdpa_cfg_cache:
            plan = None
            # Pick the table for this arch; a Blackhole entry over-subscribes a Wormhole grid.
            fit = (self._SPEC_SDPA_L1_FIT if tpc.is_blackhole() else self._SPEC_SDPA_L1_FIT_WH).get(T)
            if fit is not None:
                groups, max_cores, k_chunk = fit
                grid = self.mesh.compute_with_storage_grid_size()
                # Clamp to the grid actually present (harvesting can leave <64 on Wormhole).
                max_cores = max(1, min(max_cores, (grid.x * grid.y) // groups))
                plan = (
                    ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=(grid.x, grid.y),
                        exp_approx_mode=False,
                        q_chunk_size=0,
                        k_chunk_size=k_chunk,
                        max_cores_per_head_batch=max_cores,
                    ),
                    groups,
                    T // groups,
                )
            self._spec_sdpa_cfg_cache[T] = plan
        return self._spec_sdpa_cfg_cache[T]

    def spec_sdpa_enabled(self, T):
        """Whether the fused spec-verify SDPA is used at T candidates."""
        if T <= 1 or os.environ.get("QWEN36_SPEC_FUSED_SDPA", "1") == "0":
            return False
        return self._spec_sdpa_plan(T) is not None

    def spec_sdpa_groups(self, T):
        """Batch rows the fused spec-verify SDPA wants at T; 1 when that path is off."""
        return self._spec_sdpa_plan(T)[1] if self.spec_sdpa_enabled(T) else 1

    def forward_decode(
        self,
        x,
        cur_pos_tt,
        cos_tt,
        sin_tt,
        page_table=None,
        alias_kv_write=False,
        spec_verify_mode=False,
        spec_page_table=None,
    ):
        """alias_kv_write: the B rows share one page table. spec_verify_mode is never inferred from B."""
        tw, NH, NKV, HD = self.tw, self.NH, self.NKV, self.HD
        # Active decode width, taken from the input (x is [1,1,B,dim_frac]). Normally == self.B.
        # BUCKETED decode: a request feeds B<self.B users; every shape/reshape/rope/head-split and
        # the KV-update shard config below run at this width, and the paged SDPA reads only these B
        # users' pages via the width-B page_table. The B==self.B path is byte-identical to before.
        B = x.shape[-2]
        _L1 = ttnn.L1_MEMORY_CONFIG  # keep decode head-prep + attn output L1-resident
        use_paged = self.use_paged and page_table is not None
        if not use_paged and self.k_caches is None:
            self.reset_state()

        qg, kp, vp = self._qkv(x)

        if self._use_nlp_decode_heads:
            # Do not enable skip_v_reshard on T3K: the guard compares the wrong shard spec and the cache write segfaults.
            q, gate, k, v = self._make_heads_decode(
                qg, kp, vp, B, skip_v_reshard=use_paged and tpc.wh_9b_n300(self.args)
            )
            # Gate-after-concat is elementwise-identical. Other meshes reshape the flat gate first.
            gate_is_flat = tpc.wh_9b_n300(self.args) or _t3k_wh(self.args)
            if not gate_is_flat:
                gate_r = ttnn.reshape(gate, (1, B, NH, HD), memory_config=_L1)
                ttnn.deallocate(gate)
                gate = gate_r
        elif vp is None:
            # Fused [q|k|v|gate] weight (_qkv sentinel vp=None): qg is contiguous [q|k|v], kp is gate.
            # Slice q/k/v heads directly from qg; gate is the separate block.
            q = ttnn.reshape(
                ttnn.slice(qg, (0, 0, 0, 0), (1, 1, B, NH * HD), memory_config=_L1), (1, B, NH, HD), memory_config=_L1
            )
            k = ttnn.reshape(
                ttnn.slice(qg, (0, 0, 0, NH * HD), (1, 1, B, NH * HD + NKV * HD), memory_config=_L1),
                (1, B, NKV, HD),
                memory_config=_L1,
            )
            v = ttnn.reshape(
                ttnn.slice(qg, (0, 0, 0, NH * HD + NKV * HD), (1, 1, B, NH * HD + 2 * NKV * HD), memory_config=_L1),
                (1, B, NKV, HD),
                memory_config=_L1,
            )
            ttnn.deallocate(qg)
            gate = ttnn.reshape(kp, (1, B, NH, HD), memory_config=_L1)
            ttnn.deallocate(kp)
            gate_is_flat = False
        else:
            qg_r = ttnn.reshape(qg, (1, B, NH, HD * 2), memory_config=_L1)
            ttnn.deallocate(qg)
            q = ttnn.slice(qg_r, (0, 0, 0, 0), (1, B, NH, HD), memory_config=_L1)
            gate = ttnn.slice(qg_r, (0, 0, 0, HD), (1, B, NH, HD * 2), memory_config=_L1)
            ttnn.deallocate(qg_r)
            k = ttnn.reshape(kp, (1, B, NKV, HD), memory_config=_L1)
            ttnn.deallocate(kp)
            v = ttnn.reshape(vp, (1, B, NKV, HD), memory_config=_L1)
            ttnn.deallocate(vp)
            gate_is_flat = False

        # QK norm — (1+w), matching prefill/HF (the prior "flat" no-+1 decode band-aided the reshape scramble).
        q = self._qk_norm(q, tw["q_norm"], _L1)
        k = self._qk_norm(k, tw["k_norm"], _L1)

        q, k = self._rope_decode(q, k, cos_tt, sin_tt, B)

        # SDPA-decode grid: use the real device grid (11x10=110 cores on P150x4), not a
        # hardcoded 64. cores_per_head = grid_total/B (sdpa_decode_program_factory.cpp), so a
        # bigger grid gives each batch row more parallel cores for its KV-reduction. At SHORT
        # context (~4k) the reduction is shallow enough that fixed per-core overhead dominates
        # and this makes ~no difference (B=1: flat; B=8: ~3% worse, both within noise). At LONG
        # context (~64k) the reduction is deep enough that the extra cores are a real win:
        # SdpaDecodeDeviceOperation duration B=8: 1569.9us -> 1396.2us (-11%); B=1: 220.8us ->
        # 215.5us (-2.4%, no regression). Using the full grid unconditionally since it never hurts
        # and helps significantly at long context, where batched decode is otherwise slowest.
        # max_cores_per_head_batch defaults to 16; decode_sdpa_max_cores lifts it per instance.
        _sdpa_grid = self.mesh.compute_with_storage_grid_size()
        sdpa_dec_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(_sdpa_grid.x, _sdpa_grid.y),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
            **({} if self.decode_sdpa_max_cores is None else {"max_cores_per_head_batch": self.decode_sdpa_max_cores}),
        )
        if use_paged:
            # External paged KV: update at cur_pos, then paged SDPA-decode
            keys, values = self.paged_k, self.paged_v
            # N300-9B drops the pad; every other config keeps pad-then-reshard.
            _kv_cfg = self._kv_shard_cfg(B)
            if alias_kv_write and B > 1:
                # Aliased page-table rows must be written one row at a time. Slice the interleaved tensor, not the sharded one.
                k_p = ttnn.pad(k, [1, B, 32, HD], [0, 0, 0, 0], 0.0, memory_config=_L1)
                v_p = ttnn.pad(v, [1, B, 32, HD], [0, 0, 0, 0], 0.0, memory_config=_L1)
                _sc1 = self._kv_update_shard_cfg(1, HD)
                _nb = page_table.shape[-1]
                for i in range(B):
                    # A full-span ttnn.slice aliases its input and must not be deallocated.
                    pos_i = ttnn.slice(cur_pos_tt, (i,), (i + 1,))
                    pt_i = ttnn.slice(page_table, (i, 0), (i + 1, _nb))
                    for _cache, _src in ((keys, k_p), (values, v_p)):
                        row = ttnn.slice(_src, (0, i, 0, 0), (1, i + 1, 32, HD))
                        row_sh = ttnn.to_memory_config(row, _sc1)
                        ttnn.deallocate(row)
                        ttnn.experimental.paged_update_cache(_cache, row_sh, update_idxs_tensor=pos_i, page_table=pt_i)
                        ttnn.deallocate(row_sh)
                    ttnn.deallocate(pos_i)
                    ttnn.deallocate(pt_i)
                ttnn.deallocate(k_p)
                ttnn.deallocate(v_p)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                k_sh = v_sh = None  # this arm writes per row; nothing left for the tail dealloc
            elif not tpc.wh_9b_n300(self.args):
                # Do not switch this back to is_blackhole(); T3K/N150 stay on pad-then-reshard.
                if _kv_no_pad(self.args):
                    # Reshard the unpadded k/v; to_memory_config already shards the padded height.
                    k_sh = ttnn.to_memory_config(k, _kv_cfg)
                    v_sh = ttnn.to_memory_config(v, _kv_cfg)
                else:
                    k_p = ttnn.pad(k, [1, B, 32, HD], [0, 0, 0, 0], 0.0, memory_config=_L1)
                    v_p = ttnn.pad(v, [1, B, 32, HD], [0, 0, 0, 0], 0.0, memory_config=_L1)
                    k_sh = ttnn.to_memory_config(k_p, _kv_cfg)
                    v_sh = ttnn.to_memory_config(v_p, _kv_cfg)
                    ttnn.deallocate(k_p)
                    ttnn.deallocate(v_p)
                # Free the pad input only after the reshard has consumed the pad output.
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                # This branch must still write k_sh/v_sh.
                ttnn.experimental.paged_update_cache(keys, k_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table)
                ttnn.experimental.paged_update_cache(values, v_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table)
            elif getattr(self.args, "kv_cache_write_fused_enabled", False):
                # Fused cache write needs K and V on disjoint shard grids.
                _fused_k_cfg, _fused_v_cfg = self._kv_fused_shard_cfgs(B)
                if k.memory_config() == _fused_k_cfg:
                    k_sh = k
                else:
                    k_sh = ttnn.to_memory_config(k, _fused_k_cfg)
                    ttnn.deallocate(k)
                if v.memory_config() == _fused_v_cfg:
                    v_sh = v  # already on the natural half straight from the head split; no reshard
                else:
                    v_sh = ttnn.to_memory_config(v, _fused_v_cfg)
                    ttnn.deallocate(v)
                ttnn.experimental.paged_fused_update_cache(
                    keys, k_sh, values, v_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table
                )
            else:
                if k.memory_config() == _kv_cfg:
                    k_sh = k  # permuted RoPE already emitted K here (see the fused branch above)
                else:
                    k_sh = ttnn.to_memory_config(k, _kv_cfg)
                    ttnn.deallocate(k)
                if v.memory_config() == _kv_cfg:
                    # _make_heads_decode's skip_v_reshard already left v exactly here; no-op.
                    v_sh = v
                else:
                    v_sh = ttnn.to_memory_config(v, _kv_cfg)
                    ttnn.deallocate(v)
                # Decode K/V dtype must match the cache; paged_update_cache rejects a mismatch.
                ttnn.experimental.paged_update_cache(keys, k_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table)
                ttnn.experimental.paged_update_cache(values, v_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table)
            if k_sh is not None:
                ttnn.deallocate(k_sh)
                ttnn.deallocate(v_sh)
            # Paged SDPA batch comes from the page-table row count and is not checked against Q.
            assert B == page_table.shape[0] == cur_pos_tt.shape[-1], (
                f"decode SDPA batch mismatch: q rows {B}, page_table rows {page_table.shape[0]}, "
                f"cur_pos len {cur_pos_tt.shape[-1]}"
            )
            _spec_plan = self._spec_sdpa_plan(B) if (spec_verify_mode and self.spec_sdpa_enabled(B)) else None
            if _spec_plan is not None:
                _spec_cfg, _spec_groups, _spec_tiles = _spec_plan
                # Reshape is a metadata alias: same bytes, tile order matches the fused SDPA index.
                assert spec_page_table is not None and spec_page_table.shape[0] == _spec_groups, (
                    f"spec_verify_mode at T={B} needs a {_spec_groups}-row page table (one aliased row "
                    f"per candidate group), got {None if spec_page_table is None else spec_page_table.shape}"
                )
                _spec_rows = _spec_tiles * ttnn.TILE_SIZE
                _spec_shape = ttnn.Shape([1, _spec_groups, _spec_rows, HD])
                q = ttnn.reshape(q, _spec_shape, _spec_shape)
                attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q,
                    keys,
                    values,
                    page_table_tensor=spec_page_table,
                    cur_pos_tensor=cur_pos_tt,
                    scale=self.scale,
                    program_config=_spec_cfg,
                    memory_config=_L1,
                    spec_multi_pos_tiles=_spec_tiles,
                )
                ttnn.deallocate(q)
                # Output is byte-identical to the legacy [1,B,32,HD]; view it back (same alias trick).
                attn_out = ttnn.reshape(attn_out, ttnn.Shape([1, B, NH, HD]), ttnn.Shape([1, B, ttnn.TILE_SIZE, HD]))
            else:
                attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q,
                    keys,
                    values,
                    page_table_tensor=page_table,
                    cur_pos_tensor=cur_pos_tt,
                    scale=self.scale,
                    program_config=sdpa_dec_cfg,
                    # Emit to L1: consumed by the L1 sigmoid-gate multiply next (output-only, doesn't
                    # change the SDPA reduction), before the wo matmul + all-reduce re-materialize to DRAM.
                    memory_config=_L1,
                )
                ttnn.deallocate(q)
        else:
            # Wormhole reshards the single-head slice directly; Blackhole pads to 32.
            if k.is_sharded():
                k = ttnn.sharded_to_interleaved(k, _L1)
            if v.is_sharded():
                v = ttnn.sharded_to_interleaved(v, _L1)
            for h in range(NKV):
                k_h = ttnn.slice(k, (0, 0, h, 0), (1, B, h + 1, HD))
                v_h = ttnn.slice(v, (0, 0, h, 0), (1, B, h + 1, HD))
                _kv_cfg = self._kv_shard_cfg(B)
                if tpc.is_blackhole():
                    k_hp = ttnn.pad(k_h, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
                    v_hp = ttnn.pad(v_h, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
                    ttnn.deallocate(k_h)
                    ttnn.deallocate(v_h)
                    k_sh = ttnn.to_memory_config(k_hp, _kv_cfg)
                    v_sh = ttnn.to_memory_config(v_hp, _kv_cfg)
                    ttnn.deallocate(k_hp)
                    ttnn.deallocate(v_hp)
                else:
                    k_sh = ttnn.to_memory_config(k_h, _kv_cfg)
                    v_sh = ttnn.to_memory_config(v_h, _kv_cfg)
                    ttnn.deallocate(k_h)
                    ttnn.deallocate(v_h)
                # See the bf8-cache impossibility note at the paged call site above.
                ttnn.experimental.paged_update_cache(self.k_caches[h], k_sh, update_idxs_tensor=cur_pos_tt)
                ttnn.experimental.paged_update_cache(self.v_caches[h], v_sh, update_idxs_tensor=cur_pos_tt)
                ttnn.deallocate(k_sh)
                ttnn.deallocate(v_sh)
            ttnn.deallocate(k)
            ttnn.deallocate(v)

            if NKV == 1:
                k_full, v_full = self.k_caches[0], self.v_caches[0]
            else:
                k_full = ttnn.concat(self.k_caches, dim=1)
                v_full = ttnn.concat(self.v_caches, dim=1)

            # Non-paged oracle path (test/generate_tp only): the full-cache SDPA-decode's static CBs
            # grow with max_seq_len and, unbounded (k_chunk_size=0), overrun into the persistent CCL
            # semaphore buffers at the top of L1. Bound the K-chunk to cap the CB footprint (the paged
            # production path reads bounded blocks, so it keeps the auto config).
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
                # Emit to L1: consumed by the L1 sigmoid-gate multiply next (output-only, doesn't
                # change the SDPA reduction), before the wo matmul + all-reduce re-materialize to DRAM.
                memory_config=_L1,
            )
            ttnn.deallocate(q)

        # A flat gate is applied inside _concat_heads_decode.
        if gate_is_flat:
            gated_flat = self._concat_heads_decode(attn_out, B, gate_flat=gate)
        else:
            gated = self._apply_gate(attn_out, gate, _L1)
            if self._use_nlp_decode_heads:
                gated_flat = self._concat_heads_decode(gated, B)  # consumes + deallocates gated
            else:
                gated_flat = ttnn.reshape(gated, (1, B, NH * HD))
                ttnn.deallocate(gated)
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
        exact_kv_pos=None,
        exact_kv_pt=None,
    ):
        """Paged-KV prefill for one chunk: fill cache + chunked SDPA over prior chunks.

        exact_kv_pos writes each row at its absolute index; paged_fill_cache is block-aligned.
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

        q = self._qk_norm(q, tw["q_norm"], ttnn.L1_MEMORY_CONFIG)
        k = self._qk_norm(k, tw["k_norm"], ttnn.L1_MEMORY_CONFIG)
        q, k = self._rope_prefill(q, k, cos_tt, sin_tt)

        k_paged, v_paged = self.paged_k, self.paged_v
        block_size = k_paged.shape[2]
        if exact_kv_pos is not None:
            # Unaligned chunk_start must use paged_update_cache; paged_fill_cache writes at block start.
            n = exact_kv_pos.shape[-1]
            k_d = ttnn.permute(ttnn.slice(k, (0, 0, 0, 0), (1, NKV, n, HD)), (0, 2, 1, 3))
            v_d = ttnn.permute(ttnn.slice(v, (0, 0, 0, 0), (1, NKV, n, HD)), (0, 2, 1, 3))
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            k_p = ttnn.pad(k_d, [1, n, 32, HD], [0, 0, 0, 0], 0.0)
            v_p = ttnn.pad(v_d, [1, n, 32, HD], [0, 0, 0, 0], 0.0)
            ttnn.deallocate(k_d)
            ttnn.deallocate(v_d)
            # Identical page-table rows must be written one row at a time. Slice the interleaved tensor.
            _sc1 = self._kv_update_shard_cfg(1, HD)
            for i in range(n):
                if n == 1:
                    # A full-span slice aliases the input; leave its lifetime to the caller.
                    pos_i = exact_kv_pos
                    pt_i = exact_kv_pt
                else:
                    pos_i = ttnn.slice(exact_kv_pos, (i,), (i + 1,))
                    pt_i = ttnn.slice(exact_kv_pt, (i, 0), (i + 1, exact_kv_pt.shape[-1]))
                for _cache, _src in ((k_paged, k_p), (v_paged, v_p)):
                    row = ttnn.slice(_src, (0, i, 0, 0), (1, i + 1, 32, HD))
                    row_sh = ttnn.to_memory_config(row, _sc1)
                    ttnn.deallocate(row)
                    ttnn.experimental.paged_update_cache(_cache, row_sh, update_idxs_tensor=pos_i, page_table=pt_i)
                    ttnn.deallocate(row_sh)
                if n > 1:
                    ttnn.deallocate(pos_i)
                    ttnn.deallocate(pt_i)
            ttnn.deallocate(k_p)
            ttnn.deallocate(v_p)
        else:
            # bf8 SDPA: paged_fill_cache doesn't cast — cast K/V to cache dtype before fill
            if self._sdpa_bf8:
                _k8 = ttnn.typecast(k, ttnn.bfloat8_b)
                ttnn.deallocate(k)
                k = _k8
                _v8 = ttnn.typecast(v, ttnn.bfloat8_b)
                ttnn.deallocate(v)
                v = _v8

            # Fill this chunk into the paged cache
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
        # bf8 mode also makes the KV cache bf8.
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
        # The chunk must not exceed the query length, which also keeps the asymmetric k-chunk off verify.
        qk_chunk = min(qk_chunk, S)
        # Asymmetric k-chunk is 27B on Wormhole only.
        _kc = (
            qk_chunk * 2
            if (
                self.args.dim > 4096 and not tpc.is_blackhole() and qk_chunk == 128 and S <= tpc.PREFILL_FULL_GRID_MAX_M
            )
            else qk_chunk
        )
        # Full BH grid for SDPA perf (bit-identical to 8×8; see test_tp_chunked_prefill_pcc_sweep)
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
            exp_approx_mode=False,
            q_chunk_size=qk_chunk,
            k_chunk_size=_kc,
        )

        # Pad the page table so Q+offset and the stick size are multiples of 32.
        sdpa_page_table = page_table
        needed_blocks = (S + chunk_start_idx + block_size - 1) // block_size
        target_blocks = max(needed_blocks, page_table.shape[-1])
        target_blocks = ((target_blocks + 31) // 32) * 32
        if page_table.shape[-1] < target_blocks:
            sdpa_page_table = ttnn.pad(page_table, [(0, 0), (0, target_blocks - page_table.shape[-1])], value=0)

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
            attn,
            ttnn.sigmoid(gate_flat, memory_config=ttnn.L1_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **({"dtype": ttnn.bfloat8_b} if (self.args.dim > 4096 and not tpc.is_blackhole()) else {}),
        )
        ttnn.deallocate(attn)
        ttnn.deallocate(gate_flat)
        partial = self._wo_proj(gated, tw["wo"])
        ttnn.deallocate(gated)
        # Pass prefill_ccl_tuning, matching the mlp and gdn reduce-scatters.
        _ccl_kw = {}
        if S > tpc.TILE_SIZE:
            _cps, _wpl = tpc.prefill_ccl_tuning()
            _ccl_kw = {"chunks_per_sync": _cps, "num_workers_per_link": _wpl}
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **_ccl_kw,
        )
