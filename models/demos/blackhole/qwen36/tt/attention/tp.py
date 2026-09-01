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

_WH_KV_PAD_NOTE = """Why Wormhole decode skips the ttnn.pad before the KV-cache reshard.

The decode cache write used to be, for both the paged and the per-head branch:

    p    = ttnn.pad(t, [1, B, 32, HD], ...)      # t is [1,B,NKV,HD] or [1,B,1,HD]
    ttnn.deallocate(t)                           # free the pad's INPUT
    sh   = ttnn.to_memory_config(p, kv_update_shard_cfg)
    ttnn.deallocate(p)
    ttnn.experimental.paged_update_cache(cache, sh, ...)

The pad moves no data: in TILE layout the head dim is already padded to 32 physically. But
freeing its input immediately hands that L1 back to the allocator while the pad's write is still
pending, and the shard allocation on the next line reuses the same region and clobbers it. At
B<=8 the footprint is too small for the reused region to collide; at B=32 (4x the L1) it does,
which is why the corruption was NON-DETERMINISTIC and arrived in whole 32-column tile blocks.

Controlled 2x2 on N300, 5 reps, correct user rows out of B:

    pad + dealloc early   B=8 8/8    B=32 10-13/32   <-- the original sequence
    pad + dealloc late    B=8 8/8    B=32 32/32
    NO pad (Wormhole now) B=8 8/8    B=32 32/32      <-- and one fewer op per tensor

Resharding the unpadded tensor is equivalent: to_memory_config shards on the PADDED height
(B*32), which is exactly paged_update_cache's documented "height sharded on B cores" input, and
num_kv_heads still comes from the cache. Verified identical shard specs (grid/shape/orientation)
against the native nlp_create_qkv_heads_decode output at B=8/16/32.

Downstream effect of the fix on N300, B=32: attention pos0 PCC 0.42-0.91 -> 0.99993-0.99996;
test_model_tp batched decode/chunked-prefill 0.6162-0.9697 -> 0.9993-0.9996.

Blackhole deliberately stays on the original path: it has 1.84x the L1 (140 cores vs 80), so the
collision was likely never reachable there, and the fix is unverified on P150 from this bring-up
host. The same removal should be applied there once it can be tested.
"""


def _rp_cache_tag(source_tensor, args):
    """Content hash for a permuted-RoPE cache file name, over the PRE-permutation source weight
    plus rope_tp.ROPE_PERM_VERSION plus the dims that shape the permutation. Two runs land on the
    same tag iff both the checkpoint weight bytes and the permutation construction code are
    unchanged, so a cache built by an older/different construction can never be silently reused --
    ttnn.as_tensor just misses under the new name and rebuilds. See README-N300-9B.md's "warm
    weight cache hides..." known limitation for the blind spot this closes."""
    h = hashlib.sha256()
    h.update(rope_tp.ROPE_PERM_VERSION.encode())
    h.update(repr((tuple(source_tensor.shape), str(source_tensor.dtype), args.head_dim, args.rope_head_dim)).encode())
    # .view(torch.uint8) reinterprets raw bytes regardless of dtype (bfloat16 included, unlike
    # .numpy() which rejects it directly) -- exactly the bytes ttnn.as_tensor is about to consume.
    h.update(source_tensor.contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:16]


def load_attention_weights_tp(mesh, state_dict, args, cache_dir=None):
    """Shard one full-attention layer's weights across the mesh."""
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    def c(n):
        return str(cache_dir / n) if cache_dir is not None else None

    tw = {}
    # GQA KV replication (TP > n_kv_heads, e.g. TP=8 on T3K with 4 KV heads): expand k/v from
    # [n_kv_heads*head_dim, in] to [TP*head_dim, in] so every device owns ONE WHOLE KV head instead
    # of a fraction of one. Devices sharing a KV head recompute identical K/V into their own local
    # cache (caches are per-device and never gathered), so this is correct, not just convenient.
    # No-op when TP <= n_kv_heads, so the validated TP<=n_kv_heads layouts are byte-for-byte
    # unchanged. Device d lands on KV head (d*n_kv_heads)//TP, which matches HF's q-head grouping
    # (q head h -> kv head h//(n_heads/n_kv_heads)) because q heads shard contiguously.
    k_proj = tpc.replicate_kv_weight(state_dict["k_proj.weight"], args.n_kv_heads, args.num_devices, args.head_dim)
    v_proj = tpc.replicate_kv_weight(state_dict["v_proj.weight"], args.n_kv_heads, args.num_devices, args.head_dim)

    # Permuted-head_dim RoPE (attention/rope_tp.py's rope_channel_perm): fold the head_dim
    # channel permutation into the weights FEEDING the rotary; no runtime op.
    # q_proj is [q_head0 | gate_head0 | q_head1 | ...], hence stride=2*head_dim to permute only each
    # head's q half; the gate stays in HF order, multiplying the unpermuted
    # output. k_proj is plain per-head blocks. V and o_proj are deliberately untouched: nothing about
    # V is rotated and the output must reach o_proj in HF channel order. q/k NORM
    # weights are per-channel over head_dim, so they permute with it.
    # Cache files get a ".rp.<hash>" tag: <hash> is a content hash of the PRE-permutation source
    # weight plus rope_tp.ROPE_PERM_VERSION (see _rp_cache_tag), so these differ from the HF-order
    # tensors AND self-invalidate if the checkpoint weights or the permutation construction code
    # change -- a stale cache is never served from (or written to) the same entry as a fresh one.
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
        self.NH = args.n_local_heads
        self.NKV = args.n_local_kv_heads
        self.HD = args.head_dim
        self.scale = self.HD**-0.5
        self.rope_dim = args.rope_head_dim
        self.compute_cfg = tpc.COMPUTE_HIFI2
        # bf8 SDPA: bf8 Q + bf8 KV, HiFi2 (HiFi4 was slower). Default ON for N300,
        # override with QWEN_SDPA_BF8=0/1 — see tp_common.sdpa_bf8_enabled for the measurements.
        # Must agree with model.py's allocate_kv_caches (same helper), which sets the paged cache's
        # dtype to match what this flag casts K/V to before the fill.
        self._sdpa_bf8 = tpc.sdpa_bf8_enabled(args)
        # Must match load_attention_weights_tp gates
        self._dram_sharded = getattr(args, "attn_qg_weight_memcfg", None) is not None
        self._wo_sharded = getattr(args, "attn_wo_weight_memcfg", None) is not None
        self._fused_qkv = getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None
        self._qg_deint = self._fused_qkv
        # Fuse prefill norm-allgather + fused-QKV in-proj (all_gather_minimal_matmul_async).
        # Norm's prefill post-AG disabled in layer.py; decode path unchanged.
        # BH-only: all_gather_matmul_prefill's grid assumes BH's taller (9-10 row) compute grid; WH
        # tops out at 8 rows, so this fusion is unvalidated there. Must match layer.py's
        # _fuse_norm_agmm gate. Falls back to the unfused AG + matmul path on WH.
        self._fuse_agmm = self._fused_qkv and tpc.is_blackhole()
        # Decode head split/merge via nlp_create/concat_heads_decode (the batched-decode idiom).
        self._use_nlp_decode_heads = True
        # Permuted-head_dim full-width RoPE (rope_tp.rope_channel_perm). Must agree with
        # load_attention_weights_tp's gate: the permutation lives in the weights, so turning this on
        # at runtime without permuted weights (or vice versa) is silently wrong, not just slow.
        self._rope_permuted = getattr(args, "rope_permuted_enabled", False)
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

    def _col_proj(self, x, weight, decode_progcfg, prefill_progcfg_fn=None):
        """Column-parallel projection; DRAM-sharded decode matmul when enabled.

        prefill_progcfg_fn: WH-only one-K-pass override for the PREFILL branch only (paired with
        COMPUTE_HIFI2_NO_FP32_ACC, mirroring gdn/tp.py's _col_proj); None keeps the shared halved-block
        self.args.prefill_progcfg + self.compute_cfg, unchanged from before. Decode is unaffected
        either way — decode_progcfg/self.compute_cfg keep going through sharded_decode_matmul's M<=32
        branch untouched.
        """
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
            # LoFi (not HiFi2) on the kpass1 path: the BFP8 weight's mantissa already dominates this
            # matmul's error, so HiFi2's extra passes bought ~7e-5 PCC for 12% cost.
            # See tp_common.COMPUTE_LOFI_NO_FP32_ACC for the measurements.
            prefill_compute_cfg=tpc.COMPUTE_LOFI_NO_FP32_ACC if prefill_progcfg_fn is not None else None,
        )

    def _qk_norm(self, x, weight, memory_config):
        """RMS q/k-norm then scale by (1+weight) -- tw["q_norm"]/tw["k_norm"] are pre-offset, so this
        is the HF zero-centered convention.

        WORMHOLE ONLY: fuses the scale into ttnn.rms_norm's own weight argument -- mathematically
        the same op (rms_norm already documents its weight as a post-normalize multiply, and
        models/common/rmsnorm.py's framework RMSNorm already calls it this way) -- instead of a
        separate ttnn.multiply, removing one op per q/k-norm call. MEASURED (device kernel duration,
        N150, S=2048, HD=256, prefill head-count shapes):
            q (NH=8) unfused 182.0us -> fused 78.3us  (-57%, -104us)
            k (NH=2) unfused  55.1us -> fused 25.1us  (-54%, -30us)
        Matches the profile's BinaryNg costs for these ops (~105us/~34us) almost exactly. Output
        differs from the unfused sequence by bf16 rounding only (max abs diff 0.0625 on a
        N(0,1)-scale input) -- the same class of noise already accepted for the matmul kpass1
        changes above, not a new source of error.

        Kept as the original two-op sequence on Blackhole: unverified on that hardware from this
        host and out of scope here, so it stays untouched -- is_blackhole() gates it exactly like
        the matmul progcfg overrides above.
        """
        if tpc.is_blackhole():
            return ttnn.multiply(
                ttnn.rms_norm(x, epsilon=1e-6, memory_config=memory_config), weight, memory_config=memory_config
            )
        return ttnn.rms_norm(x, weight=weight, epsilon=1e-6, memory_config=memory_config)

    def _rope_decode(self, q, k, cos_tt, sin_tt, B):
        """Decode RoPE for Q and K. Permuted single-op path when enabled, else the partial chain.

        SIX device ops replace fourteen: shard cos, shard sin, reshard Q, rotate Q, reshard K,
        rotate K. K then still pays one Reshard onto the KV-write grid (sharded->sharded now, where
        it used to be an InterleavedToSharded), so the section is 7 device ops against 15.

        Q and K share ONE grid (``rope_k_shard_cfg``) so that a single cos/sin pair serves both --
        the sharded rotary lays its kernels, cos/sin CBs included, on the INPUT's grid, so two grids
        would mean two pairs. Making that grid the KV-write's shifted half (which would have made
        K's rotary output land in the write layout for free, since the rotary copies its output
        shard spec from its input) was measured WRONG -- see model_config.rope_k_shard_cfg.

        MEASURED end to end on a whole full-attention decode layer, device profiler, N300
        (tests/perf/test_attn_rope_permuted_sweep.py): 46 -> 38 programs and 704.5 -> 687.4 us at
        B=1; 49 -> 37 programs and 907.2 -> 778.1 us (-14.2%) at B=32.

        EVERYTHING IS FREED BEFORE RETURNING, and that is not just tidiness. An earlier version
        memoised the sharded cos/sin across layers (they are identical for every layer of a decode
        step, so it saved 2 ops per layer) and it BROKE THE MODEL: those shards stay resident in L1
        for the whole step, including while the GDN layers run, and GDN's conv1d then failed with
        "Statically allocated circular buffers in program N clash with L1 buffers on core range
        [0-0 - 4-0]" -- the same class of L1-placement conflict apply_partial_rope_prefill documents
        against SDPA's CBs. Attention must not hold L1 across a layer boundary. Two ops per layer is
        the price of that, and it is the right price.
        """
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
                # feeds the separate RS. max_cols = device width (11 on BH): wide grid (~10-wide) + the
                # existing L1-out. See test_mlp_matmul_sweep_prefill.
                #
                # WORMHOLE ONLY: attn_wo_prefill_progcfg (None on BH) swaps in the one-K-pass factory,
                # same fix as the fused-QKV in-proj above. At K=attn_out_dim_tp=2048, N=dim=4096, 8
                # cols -> per_core_N=16: fp32 dest acc on caps out_subblock_w at 4 (16%4==0, blk_w=8,
                # TWO K passes); off, the cap rises to 8 (16%8==0, blk_w=16) -- ONE pass.
                # MEASURED (N300, M=2048 K=2048 N=4096; tests/perf/test_attn_qkv_inproj_sweep.py):
                #     baseline fp32T/pkT in0=DRAM   558.6us   <- today (matches the 531-536us layer profile)
                #     kpass1   fp32F/pkT in0=DRAM   517.2us   -7.4%   pcc=0.99993 (vs 0.99997 baseline)
                # in0=L1 measured within noise (-0.6pp more), not worth the added CB-clash risk for it.
                _wo_kpass1 = getattr(self.args, "attn_wo_prefill_progcfg", None)
                if _wo_kpass1 is not None:
                    pc = _wo_kpass1(x.shape[-2], weight.shape[-2], weight.shape[-1])
                    # LoFi: with a BFP8 weight AND (post-bf8-SDPA) a BFP8 activation, HiFi2 was worth
                    # ~6e-5 PCC for 18% of the op. See tp_common.COMPUTE_LOFI_NO_FP32_ACC.
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
                # MODEL-GATED (27B on Wormhole). Emit bf8 so the row-parallel reduce-scatter that
                # consumes this carries half the bytes -- as the matmul's OUTPUT DTYPE, not a
                # typecast afterwards, which would be a whole extra pass over [1,S,dim].
                #
                # Third instance of the same win in this layer -- all three reduce-scatter a
                # [1,1,2048,5120] row-parallel partial, and bf8 lands them at the same ~550-590us
                # floor. MEASURED (T3K TP=8, 27B, seq 2048): wo matmul 419 -> 277us, reduce-scatter
                # 1021 -> 548us = -610us/layer. Accuracy measured, not assumed, since this op has a
                # dtype cliff on Blackhole at TP=4 (see gdn/tp.py): test_attention_tp_prefill
                # 0.9993245 -> 0.9991838, matching the GDN out-proj's cost.
                #
                # Gated to the 27B because that cliff proves the safe dtype here is TP- and
                # model-dependent, and TP=8/dim=5120 is the only configuration measured.
                _wo_bf8 = self.args.dim > 4096 and not tpc.is_blackhole()
                return ttnn.linear(
                    x,
                    weight,
                    compute_kernel_config=ck,
                    program_config=pc,
                    # L1 while it fits; DRAM once the [seq,dim] output would crowd out this matmul's
                    # own circular buffers on WH (see tp_common.prefill_out_memory_config).
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
            # SINGLE-TENSOR nlp_create_qkv_heads: hand it the contiguous [q|k|v] block whole and let
            # it do the split in-kernel, instead of slicing q and kv out first to feed the two-tensor
            # (input, input_kv) overload. The op's fused form handles the GQA asymmetry -- it derives
            # the q/k/v boundaries from num_heads/num_kv_heads, so a (NH + 2*NKV)*HD = 1280-wide input
            # is accepted even though the docstring's shape reads 3*head_dim*num_heads.
            #
            # VERIFIED BIT-EXACT against the two-slice form: q/k/v all torch.equal, max|diff| == 0.0
            # (NH=3, NKV=1, HD=256, S=128 -- the 27B TP=8 head geometry).
            # MEASURED (T3K TP=8, 27B, seq 2048, layer3_fullattn single-layer profile): removes the
            # two slices, 20.8us (1280->768, q) + 15.8us (1280->512, k|v) = -36.6us and -2 device ops.
            # Both were running at ~100-130 GB/s, i.e. at bandwidth -- so this is not a slow-op fix,
            # it is one fewer full pass over the tensor.
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

        Returns (q, gate_flat, k, v): q [1,B,NH,HD], gate_flat [1,1,B,NH*HD], k/v [1,B,NKV,HD].

        GATE STAYS FLAT. It used to be reshaped to [1,B,NH,HD] here purely to match the post-SDPA
        ``attn_out`` the sigmoid-multiply consumed, and that reshape was a REAL op (a 7us/8-core
        ReshapeViewDeviceOperation in the decode capture): NH < TILE_SIZE, so [1,1,B,NH*HD] ->
        [1,B,NH,HD] moves the head axis into the tiled plane and needs a data rewrite (measured in
        test_attn_head_split_copy_reshape_sweep.py, which established it could not be made cheaper
        IN PLACE -- this removes the need for it instead). ``forward_decode`` now applies the gate
        AFTER ``_concat_heads_decode`` instead, where the attention output is already back in the
        flat [1,1,B,NH*HD] layout the gate has had all along, so no reshape is needed on either side.
        Bit-identical: concat-heads is a pure permutation of elements and the gate is laid out
        head-major to match it, so an elementwise multiply commutes with it exactly.

        q/k are
        always L1-interleaved (q/k feed rms_norm, which categorically rejects HEIGHT_SHARDED input --
        ttnn's layernorm_device_operation.cpp -- so this reshard is not optional there). v is
        interleaved too UNLESS ``skip_v_reshard``: v has no norm/RoPE consumer, its only use is the
        paged-cache write, and nlp_create_qkv_heads_decode's native HEIGHT_SHARDED output already
        uses the same one-user-per-core grid/shape as ``self.args.kv_update_shard_cfg`` (both derive
        it from B the same way) -- so for the paged decode path (see forward_decode) the
        sharded_to_interleaved here immediately followed by a to_memory_config back to
        kv_update_shard_cfg is a pure round trip. VERIFIED byte-identical (nlp_create_qkv_heads_decode
        v output memory_config == kv_update_shard_cfg for B=32, both HD=128 and HD=256) and re-checked
        at runtime below (falls back to the reshard if the configs ever diverge, e.g. B != max_batch_
        size). WH-only: gated off on Blackhole, which pads v before its own reshard (_WH_KV_PAD_NOTE)
        and has not been checked against this shortcut from this host.

        The kernel only shuffles a fused Q|K|V, so the gate half of qg is split off first and applied
        post-SDPA exactly like the reshape path. The fused tensor is kept in L1 to dodge the Blackhole
        interleaved-reader bug (tt-metal #16667: DRAM input zeros odd-indexed Q rows).
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
            pass  # v feeds straight into the paged-cache write in its native sharded layout
        else:
            v = ttnn.sharded_to_interleaved(v, _L1)
        return q, gate_flat, k, v

    def _concat_heads_decode(self, attn_out, B, gate_flat=None):
        """Decode concat-heads via nlp_concat_heads_decode. attn_out [1,B,NH,HD] L1 -> [1,B,NH*HD] L1.

        The op wants a height-sharded input ([1,B,heads-padded-to-32,HD], one core per user), so the
        SDPA output is resharded across `B` cores first (a grid-width-aligned rectangle — a ragged
        core set is rejected by the height-sharded mem config). Output is width-sharded, then
        returned to L1-interleaved so the downstream o_proj matmul is unchanged.

        ``gate_flat`` [1,1,B,NH*HD] (optional): the sigmoid gate, applied HERE rather than on
        [1,B,NH,HD] before the call. This is where the flat gate becomes free -- ``out`` below is
        already [1,1,B,NH*HD], the gate's own natural layout straight from the QKV projection, so
        neither side needs the [1,1,B,NH*HD] <-> [1,B,NH,HD] relayout that used to cost a real
        7us ReshapeViewDeviceOperation per layer in ``_make_heads_decode``. nlp_concat_heads_decode
        lays head h at columns [h*HD, (h+1)*HD), which is exactly the head-major order the gate
        block already has, so multiplying after the concat is elementwise-identical to multiplying
        before it (concat-heads is a pure permutation; an elementwise multiply commutes with it when
        both operands are permuted the same way). Also cheaper in its own right at batch: the
        [1,B,NH,HD] multiply padded NH=8 up to a 32-row tile, the flat one does not.
        """
        from models.tt_transformers.tt.model_config import num_to_corerange

        NH, HD = self.NH, self.HD
        _L1 = ttnn.L1_MEMORY_CONFIG
        grid = self.mesh.compute_with_storage_grid_size()
        gx = min(B, grid.x)
        if B >= gx and B % gx != 0:
            gx = max(x for x in range(gx, 0, -1) if B % x == 0 and B // x <= grid.y)
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
        """Sigmoid-gate multiply, shape-agnostic (x and gate must have the same layout).

        Fused sigmoid-multiply (input_tensor_b_activations): one kernel instead of two. Unlike the
        GDN module's silu-gate (gdn/tp.py's _silu_mul, kept UNFUSED because fusing silu overflowed
        to NaN for large-magnitude z), sigmoid is bounded to [0,1] for any input magnitude -- no
        overflow mode to inherit. VERIFIED (gate values swept to +/-50, well past realistic
        magnitudes): zero NaN, output bit-identical to the unfused sequence (same bf16 rounding
        noise vs an fp32 reference either way). MEASURED (device kernel duration): 9.86us -> 8.64us
        (-12.4%).

        SCOPED TO WORMHOLE 9B ON N300 (tpc.wh_9b_n300) like the other decode changes, even though
        this one is shape-agnostic and would very likely be safe everywhere: the NaN sweep that
        justifies it was run on this config only, and the GDN precedent right next door shows this
        exact fusion mechanism CAN blow up for a different activation. Widening the scope wants its
        own numerical check per config, not an assumption.
        """
        if tpc.wh_9b_n300(self.args):
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
        # MODEL-GATED (27B on Wormhole). Emit `gated` as bf8 so the wo matmul is BFP8 x BFP8.
        # `gated` is attn * sigmoid(gate) -- an independent tensor with no KV-cache involvement, so
        # unlike the attention_norm gather (see layer.py's ATTEMPTED AND REJECTED note) nothing
        # blocks narrowing it.
        # MEASURED (T3K TP=8, 27B, seq 2048, layer3_fullattn):
        #     wo matmul 2048x768x5120   275 -> 261us   (-14)
        #     gated multiply             36 ->  31us   (-5)
        #                                              = -22us/layer
        #     PCC (test_attention_tp_prefill, S=64)  0.9991838 -> 0.9991558
        # SMALL, and the reason is worth recording: the MLP's down-proj runs the same BFP8 x BFP8
        # shape at 50.1% of peak against this op's 25.1%, which suggests a much bigger prize -- but
        # wo's K is 768 (24 tiles) against the down-proj's 2176 (68), so it is load/store-bound and
        # halving in0's bytes cannot buy what it buys there. Do not extrapolate FLOPs% between
        # matmuls with very different K.
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
        # PREFILL CCL tuning (chunks_per_sync/num_workers_per_link): mlp.py's and gdn/tp.py's
        # reduce-scatters already pass tpc.prefill_ccl_tuning() here; this call site never did --
        # an oversight, not an architecture split (that helper's tuning already ships unconditionally
        # on both WH and BH for MLP/GDN). See tp_common.prefill_ccl_tuning for the measurements
        # (~-230us on this op's mean, and it tightens the run-to-run spread ~6x).
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
        """Disjoint K/V height-shard grids for the fused paged-cache write, sized to the ACTIVE
        width B. Returns the precomputed max-batch configs unchanged when B==self.B (byte-identical
        prod path); builds width-B disjoint grids for bucketed decode. Mirrors _kv_shard_cfg and
        model_config.kv_cache_write_{k,v}_shard_cfg -- same natural/shifted-half split (V on rows
        0..rows-1, K on rows..2*rows-1), just re-cut to B's cols/rows instead of max_batch_size's.

        Without this, the fused write used max_batch_size's grid at every active B: every core in
        that grid runs (ttnn's paged_fused_update_cache marks the whole supplied grid active), and a
        core past the true active width reads update_idxs_tensor/page_table out of its real range --
        an out-of-bounds read that can write cache data to an arbitrary location. The non-fused path
        (the branch below) already avoids this via _kv_shard_cfg(B); this mirrors it for the fused one.

        The grid-fits-both-halves precondition that gates kv_cache_write_fused_enabled at max B
        holds at any B <= max B too (fewer rows needed), so it is not re-checked here.
        """
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

    def forward_decode(self, x, cur_pos_tt, cos_tt, sin_tt, page_table=None):
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
            # The per-head test/generate_tp oracle path below (use_paged=False, k_caches loop) slices
            # v per-KV-head, which wants it interleaved -- only the paged production path can take it
            # natively sharded straight into the cache write. SCOPED TO WORMHOLE 9B ON N300
            # (tpc.wh_9b_n300): the shard-spec equality this relies on was verified on that config,
            # and _make_heads_decode's runtime guard makes a mismatch fall back safely anyway.
            # gate comes back FLAT ([1,1,B,NH*HD]) here; it is applied after _concat_heads_decode,
            # which is where that layout is already the right one (see _make_heads_decode).
            q, gate, k, v = self._make_heads_decode(
                qg, kp, vp, B, skip_v_reshard=use_paged and tpc.wh_9b_n300(self.args)
            )
            # SCOPED TO WORMHOLE 9B ON N300 (tpc.wh_9b_n300) like the other decode changes: applying
            # the gate after the concat instead of before is elementwise-identical (concat-heads is a
            # pure permutation and the gate is head-major to match), and prefill's _make_heads has
            # done exactly this for longer, but was only measured and PCC-checked on
            # config, so every other mesh/model keeps the original pre-concat multiply by reshaping
            # the flat gate back to [1,B,NH,HD] here.
            gate_is_flat = tpc.wh_9b_n300(self.args)
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

        # REVERTED Q+K merged RoPE (was here briefly): an isolated microbenchmark measured -2.3%
        # (172.3us -> 168.4us) for concatenating Q+K before rotating once. A REAL full-layer Tracy
        # capture (test_profile_single_layer_attention_decode.py, tt-perf-report) caught what the
        # isolated test missed: NH=16 + NKV=4 = 20 heads is no clean tile multiple,
        # so the concat forces Untilize -> Concat -> TilizeWithValPadding round-trips on BOTH sides of
        # the merge (6+5+1+18+9+22 = 61us of new ops) that the synthetic benchmark's tensors never
        # hit. In the real layer this section went 46us -> 98us (+113%), swamping
        # every other win this session found. Lesson: an isolated op-shape microbenchmark cannot
        # decide a change like this -- the exact same caution mlp.py's _CKC_MLP_KPASS1 comment already
        # gives ("the isolated sweep... CANNOT decide this... trust the full-layer capture"). Confirm
        # any future attempt at this fusion against a full-layer Tracy capture, not just the isolated
        # op timing.
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
        _sdpa_grid = self.mesh.compute_with_storage_grid_size()
        sdpa_dec_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(_sdpa_grid.x, _sdpa_grid.y),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )
        if use_paged:
            # External paged KV: update at cur_pos, then paged SDPA-decode
            keys, values = self.paged_k, self.paged_v
            # N300-9B drops the pad to [1,B,32,HD] (see _WH_KV_PAD_NOTE); everyone else (Blackhole,
            # and now T3K/N150 too) keeps the original pad-then-reshard sequence verbatim.
            #
            # DELIBERATE narrowing (Wormhole gating audit, item 8): this used to be tpc.is_blackhole(),
            # so T3K/N150 rode the no-pad path along with N300. _WH_KV_PAD_NOTE's own measurements are
            # "Controlled 2x2 on N300" and explicitly unverified beyond that config, so T3K/N150 are
            # narrowed back to the safe pad-then-reshard branch on purpose -- not because they're
            # Blackhole, but because the no-pad fix has no validation there. Don't revert this to
            # is_blackhole() without measuring the no-pad path on T3K/N150 first.
            _kv_cfg = self._kv_shard_cfg(B)
            if not tpc.wh_9b_n300(self.args):
                k_p = ttnn.pad(k, [1, B, 32, HD], [0, 0, 0, 0], 0.0, memory_config=_L1)
                v_p = ttnn.pad(v, [1, B, 32, HD], [0, 0, 0, 0], 0.0, memory_config=_L1)
                k_sh = ttnn.to_memory_config(k_p, _kv_cfg)
                v_sh = ttnn.to_memory_config(v_p, _kv_cfg)
                ttnn.deallocate(k_p)
                ttnn.deallocate(v_p)
                # Free the pad's INPUT only after the reshard has consumed the pad's output --
                # _WH_KV_PAD_NOTE's "dealloc late" ordering. Freeing k/v immediately after the pad
                # hands that L1 back while the pad's write is still in flight, which measured
                # B=32 10-13/32 users correct (B=8 was unaffected, which is why it hid here).
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                # This branch PREPARES k_sh/v_sh; it must still write them. Both sibling branches
                # below issue their own write, so omitting it here silently skipped the KV update
                # on every config except N300-9B -- decode then attended over a cache holding only
                # the prefill tokens, with the new token's slot left at zero.
                ttnn.experimental.paged_update_cache(keys, k_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table)
                ttnn.experimental.paged_update_cache(values, v_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table)
            elif getattr(self.args, "kv_cache_write_fused_enabled", False):
                # Fused K+V cache write (paged_fused_update_cache): needs K and V on DISJOINT shard
                # grids. V is pointed at the NATURAL half -- the grid nlp_create_qkv_heads_decode
                # already emits -- so the equality guard below short-circuits and V pays NO reshard,
                # exactly as before the fused write existed; K takes the SHIFTED half,
                # free for K because it arrives interleaved from RoPE and owes a reshard either way.
                # See model_config.py's kv_cache_write_{k,v}_shard_cfg for why this assignment (and
                # not the reverse) is the cheap one, and for the NaN footgun in making
                # head split emit onto the shifted half instead.
                # VERIFIED (test_kv_cache_sdpa_decode_sweep.py + test_attn_head_split_v_reshard_sweep.py
                # ::test_kv_write_grid_swap_removes_v_reshard, N300, per-user-distinct paged cache):
                # bit-identical cache contents, 20.3us (2 separate writes) -> 18.3us fused (-9.9%),
                # then 26.7us/4 programs -> 26.0us/3 programs (-3.0%) from this grid assignment.
                # With permuted RoPE, K is ALREADY on this grid -- the sharded rotary copies its
                # output shard spec from its input, and _rope_decode fed it rope_k_shard_cfg (this
                # config) precisely so the write needs no reshard. Same guard style as V's below.
                #
                # Re-cut to the ACTIVE width B (bucketed decode), not max_batch_size: the fused op
                # activates every core in the supplied grid, so at the precomputed max-batch grid a
                # smaller active B leaves cores past B reading update_idxs_tensor/page_table out of
                # range. _kv_fused_shard_cfgs(B) mirrors _kv_shard_cfg(B) below for this fused path.
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
                # DECODE K/V MUST BE bf16/fp32 HERE, AND MUST MATCH THE CACHE -- which is what makes a
                # bf8 KV cache impossible on this op. paged_update_cache asserts BOTH
                #   input.dtype == FLOAT32 || input.dtype == BFLOAT16
                #       (paged_update_cache_device_operation.cpp:296)
                #   input.dtype == cache.dtype  ("Input and cache tensors must have same dtype!")
                # so against a bf8 cache a bf16 input fails the second and a bf8 input fails the first.
                # There is no dtype that satisfies both; a cast does not help (tried, it just moves
                # which assert fires). paged_fill_cache, which PREFILL uses, has the LOOSER contract
                # (input==FP32 || input==BF16 || cache==BFP8 || cache==BFP4) -- which is exactly why a
                # bf8 cache passes every prefill-only profile, every module test and the 64k demo while
                # failing 5 of 20 test_model_tp cases (every one that decodes after a prefill).
                # Enabling a bf8 KV cache needs a C++ change to paged_update_cache, not a Python cast.
                ttnn.experimental.paged_update_cache(keys, k_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table)
                ttnn.experimental.paged_update_cache(values, v_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table)
            ttnn.deallocate(k_sh)
            ttnn.deallocate(v_sh)
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
            # Internal per-head KV caches. Wormhole reshards the [1,B,1,HD] single-head slice
            # directly; Blackhole keeps the original pad-to-32 sequence verbatim (_WH_KV_PAD_NOTE).
            #
            # This oracle path slices K/V per KV-head, which needs them interleaved -- permuted RoPE
            # leaves K height-sharded (that layout is what the PAGED write wants), so undo it here.
            # Costs one op on a test/demo-only path rather than on the production paged one.
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

        # Sigmoid gate. When the head split left it FLAT ([1,1,B,NH*HD], as it
        # the QKV projection), it is applied INSIDE _concat_heads_decode -- after the concat, where
        # the attention output is back in that same flat layout. That removes the [1,1,B,NH*HD] ->
        # [1,B,NH,HD] gate reshape entirely (a real 7us/8-core op per layer) instead of paying it just
        # to line the two operands up. Elementwise-identical either side of the concat; see
        # _concat_heads_decode. The non-flat branches keep the original pre-concat multiply.
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
        # DECODE reduce-scatter tuning: checked wpl in {1,2,4,8} at this shape
        # (test_attn_output_reduce_sweep.py). wpl=1 and wpl=8 are consistently worse across repeated
        # runs; wpl=2 (upstream default) vs wpl=4 FLIPS direction run-to-run (+-4-7% either way across
        # 5 repeated measurements) -- that comparison is noise, not signal, so it stays at upstream's
        # default (wpl=2, implicit) rather than "fixing" something that isn't broken.
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

        q = self._qk_norm(q, tw["q_norm"], ttnn.L1_MEMORY_CONFIG)
        k = self._qk_norm(k, tw["k_norm"], ttnn.L1_MEMORY_CONFIG)
        q, k = self._rope_prefill(q, k, cos_tt, sin_tt)

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
        # makes the KV cache bf8 -> full bf8 matmul. TESTED mixed dtype (bf16 Q vs
        # cache/K/V): PASSES correctness (PCC actually a hair better, 0.99979 vs 0.99969) but SDPA
        # itself measures ~24us SLOWER (507us vs 483-485us) — full bf8 both operands is faster on
        # this hardware, matching the same BF16-activation-vs-BFP8-both pattern seen on the other
        # matmuls (e.g. wo_proj). Keeping the typecast: it's a one-time ~46us cost per chunk against
        # a per-call SDPA win, not a wash to remove.
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
        # ASYMMETRIC K-CHUNK (MODEL-GATED, 27B on Wormhole). Every config here had set
        # q_chunk_size == k_chunk_size; decoupling them is worth -7.2% on this op. The Q chunk is what
        # bounds the CB footprint (q_chunk >= 256 fails outright -- see below), but K can be doubled
        # for free because the K/V blocks stream.
        # MEASURED (T3K TP=8, 27B, seq 2048, layer3_fullattn single-layer profile, device kernel time):
        #     q=128 k=128  490.1us   <- was (and what the "128 beats 64/256" note above compared)
        #     q=128 k=256  455.0us   -35.1us  <- used
        #     q= 64 k=128  532.5us
        #     q=256 k=256 / q=256 k=128 / q=512 k=128   all FAIL (CB overflow)
        #     q=128 k=512 / k=1024 / k=2048             all FAIL (CB overflow)
        # exp_approx_mode was swept at the same time and is a WASH -- 488.9us at k=128 and 455.5us at
        # k=256, both within noise of exp_approx_mode=False -- so it stays off and costs no accuracy.
        # Gated because the CB headroom that makes k=256 fit is specific to this head geometry
        # (NH=3, NKV=1, HD=256 at TP=8) and this chunk length.
        # LENGTH-CAPPED at the production chunk: at S=4096 the doubled K chunk overflows CBs
        # (layer3_fullattn-seq4096 fails), same shape of limit as tp_common.PREFILL_FULL_GRID_MAX_M.
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

        # Pad page table for Q+offset and stick-size % 32 (extra blocks masked).
        # ttnn.pad instead of zeros+concat: same result (verified via ttnn.to_torch, byte-identical),
        # 1 program dispatch instead of 4 -- ttnn's ROW_MAJOR concat path for this shape decomposes
        # into a Concat plus TWO internal Permute-kernel dispatches (visible in the build log as
        # writer/reader_permute_interleaved_rm_blocked_generic), which is exactly where the profile's
        # tiny PermuteDeviceOperation/ConcatDeviceOperation rows on the INT32 page table came from.
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
        # MODEL-GATED (27B on Wormhole). Emit `gated` as bf8 so the wo matmul is BFP8 x BFP8.
        # `gated` is attn * sigmoid(gate) -- an independent tensor with no KV-cache involvement, so
        # unlike the attention_norm gather (see layer.py's ATTEMPTED AND REJECTED note) nothing
        # blocks narrowing it.
        # MEASURED (T3K TP=8, 27B, seq 2048, layer3_fullattn):
        #     wo matmul 2048x768x5120   275 -> 261us   (-14)
        #     gated multiply             36 ->  31us   (-5)
        #                                              = -22us/layer
        #     PCC (test_attention_tp_prefill, S=64)  0.9991838 -> 0.9991558
        # SMALL, and the reason is worth recording: the MLP's down-proj runs the same BFP8 x BFP8
        # shape at 50.1% of peak against this op's 25.1%, which suggests a much bigger prize -- but
        # wo's K is 768 (24 tiles) against the down-proj's 2176 (68), so it is load/store-bound and
        # halving in0's bytes cannot buy what it buys there. Do not extrapolate FLOPs% between
        # matmuls with very different K.
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
        # PREFILL CCL tuning (chunks_per_sync/num_workers_per_link): mlp.py's and gdn/tp.py's
        # reduce-scatters already pass tpc.prefill_ccl_tuning() here; this call site never did --
        # an oversight, not an architecture split (that helper's tuning already ships unconditionally
        # on both WH and BH for MLP/GDN). See tp_common.prefill_ccl_tuning for the measurements
        # (~-230us on this op's mean, and it tightens the run-to-run spread ~6x).
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
