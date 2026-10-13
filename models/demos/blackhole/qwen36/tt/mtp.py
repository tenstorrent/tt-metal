# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5/3.6 MTP (multi-token prediction) head — the speculative-decode drafter.
Every checkpoint ships a single-layer MTP head (``mtp.*``) that reuses the main token embedding
and LM head. Structure (DeepSeek-V3 MTP2D): h' = fc(concat[enorm(embed(token)), hnorm(hidden)]);
h'' = DecoderLayer(h') (1 full-attention layer); logits = LMHead(norm(h'')) (shared head).
``enorm``/``hnorm``/``fc``/``DecoderLayer``/``norm`` = mtp.pre_fc_norm_embedding / pre_fc_norm_hidden /
fc (eh_proj, [dim, 2*dim]) / layers.0 (qwen36 full-attn layer verbatim) / mtp.norm. Own paged KV
cache, separate from the base. forward_decode returns ``(logits, next_hidden)``; ``next_hidden`` is
fed back as ``hidden`` for the next chained draft (EAGLE-style K>1). Spec feed contract
(Qwen36Model.spec_feed_rows) is POST-NORM: the base feeds its final-norm output (fractured to dim/tp)
and the chain feeds back mtp.norm's output, re-fractured to dim/tp, so hnorm sees the same kind of tensor at every step."""
import os

import torch
from loguru import logger

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.tt_transformers.tt.common import Mode


def argmax_last(logits):
    """argmax over the vocab dim for ONE row of drafter logits -> [1,1,1] uint32 ROW_MAJOR.

    ttnn.argmax needs ROW_MAJOR input: a TILE tensor takes a single-core internal-untilize path that
    is catastrophically slow on a 151k-wide vocab, so untilize multicore first. The GATHERED form:
    ``logits`` is the full replicated [1,1,1,vocab] row. Reached through draft_argmax, which is what
    both the eager draft step (SpeculativeDecoder._draft_argmax) and the traced chain
    (Qwen36Model._draft_body) call, so the two pick their ids through byte-identical ops.
    """
    u = ttnn.untilize(logits, use_multicore=True)
    out = ttnn.argmax(u, dim=-1, keepdim=False)  # [1,1,1] uint32 RM
    ttnn.deallocate(u)
    return out


def draft_argmax(mtp, logits):
    """Drafter logits -> token id, for whichever LM-head form the flag selected.

    THE shared drafter pick: the eager chain (SpeculativeDecoder._draft/_draft_warmup, via
    _draft_argmax) and the traced one (Qwen36Model._draft_body) both come through here, so the two
    dispatch on one value and switch together — the same reason argmax_last was hoisted to module
    level in the first place. Both forms take the drafter's own logits and return [1,1,1] uint32
    ROW_MAJOR, and neither frees ``logits``: the caller owns it.

    QWEN36_DRAFT_SHARDED_ARGMAX (resolved once in Qwen36MTP.__init__) selects the sharded form, in
    which ``logits`` is THIS device's [1,1,1,vocab/tp] fp32 shard and the vocab all-gather never
    happened; off, ``logits`` is the gathered full-vocab row and this is exactly argmax_last.
    """
    if mtp._sharded_argmax:
        return mtp._argmax_sharded(logits)  # per-shard argmax + device-only winner pick
    return argmax_last(logits)  # full-vocab gathered row: untilize + argmax


class Qwen36MTP:
    """Single-layer MTP drafter head. Reuses the parent model's embedding + LM head."""

    # --- per-shard drafter argmax (QWEN36_DRAFT_SHARDED_ARGMAX) constants ------------------- #
    # Lanes each device contributes to the two winner-pick all-gathers. 32 = one tile column, so the
    # dim=3 gather is a whole-tile concat and never a sub-tile write; a 1-wide TILE tensor would ask
    # the CCL to splice 32-column tiles at column granularity.
    _LANES = 32
    # Pad/floor for lanes that must never win a value argmax. Same number the plain greedy demo path
    # uses (demo/text_demo.py _maxval_dev): far below any real logit and far from the fp32/bf16
    # overflow edge. Only ever compared, never added to.
    _NEG_FLOOR = -1.0e30
    # Added to every LOSING lane's id before the int32 min, so the min returns the WINNING lane's id.
    # Must exceed the largest global token id and stay exact in bf16 (2**24 has mantissa 1.0) and in
    # int32 (max id + sentinel = 2**24 + 248319, far below 2**31). Same constant and same role as
    # TIEBREAK_INDEX_SENTINEL in models/common/sampling/tt_sampling.py.
    _SENTINEL = 2**24

    def __init__(self, mesh_device, args, state_dict, parent, tensor_cache_path=None, tt_ccl=None):
        self.args = args
        self.device = mesh_device
        self.mesh_device = mesh_device
        self.num_devices = getattr(args, "num_devices", 1)
        self.tt_ccl = tt_ccl

        # Shared (no new weights): the main embedding + LM head + final-norm reuse.
        self.embd = parent.embd
        self._lm_head = parent._lm_head
        self.lm_head_weight = parent.lm_head_weight

        mtp_cache = (tensor_cache_path / "mtp") if tensor_cache_path is not None else None

        # Two pre-fc norms + the post-block norm, keyed under "mtp." (mtp.pre_fc_norm_embedding,
        # mtp.pre_fc_norm_hidden, mtp.norm). Built exactly like Qwen36DecoderLayer._make_norm.
        self.pre_fc_norm_embedding = self._make_norm(state_dict, "pre_fc_norm_embedding", mtp_cache, "attn")
        self.pre_fc_norm_hidden = self._make_norm(state_dict, "pre_fc_norm_hidden", mtp_cache, "attn")
        self.head_norm = self._make_norm(state_dict, "norm", mtp_cache, "lm_head")

        # fc (eh_proj): torch weight [dim, 2*dim] -> concat(token_emb, hidden)[..,2*dim] -> hidden.
        fc_w = state_dict["mtp.fc.weight"]
        assert fc_w.shape == (args.dim, 2 * args.dim), f"unexpected mtp.fc shape {tuple(fc_w.shape)}"
        if self.num_devices > 1:
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            self._fc_compute_cfg = tpc.COMPUTE_HIFI2
            # Column-parallel: transpose to [2*dim, dim] then shard dim=-1 -> [2*dim, dim/tp] per device.
            self.fc = tpc.shard_w(
                fc_w,
                mesh_device,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=(mtp_cache / "fc" if mtp_cache is not None else None),
                dtype=ttnn.bfloat8_b,
            )
        else:
            self._fc_compute_cfg = None
            self.fc = ttnn.as_tensor(
                fc_w.T.contiguous(),  # [2*dim, dim]
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=(str(mtp_cache / "fc") if mtp_cache is not None else None),
            )

        # Reuse the full-attention decoder layer for mtp.layers.0. Remap the checkpoint keys to a
        # full-attention layer index L so is_full_attention_layer(L) is True and the substate loader
        # finds layers.{L}.self_attn.* / .mlp.* / .{input,post_attention}_layernorm.
        L = next((i for i, t in enumerate(args.attention_type_list) if t == "full_attention"), None)
        assert (
            L is not None
        ), f"checkpoint attention_type_list has no full_attention layer (len={len(args.attention_type_list)})"
        assert args.is_full_attention_layer(L), f"MTP host layer {L} is not full attention"
        self.mtp_host_layer = L
        prefix = f"layers.{L}."
        mtp_layer_sd = {
            prefix + k[len("mtp.layers.0.") :]: v for k, v in state_dict.items() if k.startswith("mtp.layers.0.")
        }
        # Dedicated cache root (.../mtp/) so the reused layer's sharded weights never collide with
        # the real layer L's cache.
        self.decoder = Qwen36DecoderLayer(
            mesh_device, args, mtp_layer_sd, layer_num=L, tensor_cache_path=mtp_cache, tt_ccl=tt_ccl
        )
        # KV accessor for allocate_kv_caches / rollback.
        self.attention = self.decoder.attention
        # Drafter-only decode SDPA width. Shared decode program config leaves max_cores_per_head_batch
        # at ttnn's default of 16, which at B=1 and 1 local KV head puts 16 of 110 cores on the KV
        # reduction. Every one of the K draft steps is a B=1 decode that rescans the WHOLE prompt KV,
        # so SDPA is reduction-bound. 64 is the kernel ceiling (tree reduction capped at
        # MAX_TREE_REDUCTION_ROUNDS=6 = 2^6 cores/head). Set on this MTP TPAttention only, so the
        # base model's full-attention layers keep their config. Batched reseed (B=K+1 through this
        # same instance) is unaffected: at B=11 both 16 and 64 resolve to min(110, max*B)/B = 10
        # cores/head. Changes the drafter's reduction order, so bf16 near-ties can draft a different
        # token — that only shifts acceptance, never correctness: every draft is arbitrated by verify.
        self.attention.decode_sdpa_max_cores = 64

        # QWEN36_DRAFT_SHARDED_ARGMAX=1: the DRAFTER stops all-gathering its full-vocab fp32 logit
        # row. Each device argmaxes its own [1,1,1,vocab/tp] shard and the global winner is picked
        # on device from two 32-lane gathers (_argmax_sharded). Default off; the gathered
        # _lm_head(out_dtype=float32) + argmax_last path stays the fallback and is what runs
        # whenever this resolves to False. The attributes are always defined so draft_argmax can
        # test `mtp._sharded_argmax` without a getattr dance.
        self._sharded_argmax = False
        self._shard_off = None  # [1,1,1,_LANES] int32, per device: d * vocab_shard in every lane
        self._sel_lane = None  # [1,1,1,_LANES*num_devices] int32 replicated: lane j holds j
        self._fp32_cfg = None
        self._vocab_shard = 0
        self._maxval_c = 0
        self._maxval_r = 0
        # On by default; QWEN36_DRAFT_SHARDED_ARGMAX=0 opts out. Changes the DRAFTER's argmax
        # reduction order — a bf16 near-tie can draft a different token, shifting acceptance only
        # (verify arbitrates every draft, so correctness holds); measured acceptance was identical.
        if bool(int(os.environ.get("QWEN36_DRAFT_SHARDED_ARGMAX", "1"))):
            self._init_sharded_argmax(parent)

    def _init_sharded_argmax(self, parent):
        """Resolve QWEN36_DRAFT_SHARDED_ARGMAX and allocate the two persistent constants it needs.

        Runs at model-build time, i.e. long before any begin_trace_capture, which is what makes the
        constants usable from inside the traced draft body: the trace bakes in their ADDRESSES and
        nothing ever reallocates them. Refuses (and stays off) on any shape the pick cannot handle,
        so the flag is safe to set unconditionally.
        """
        nd = self.num_devices
        vocab = int(self.args.vocab_size)
        lanes = self._LANES
        if nd < 2 or not getattr(parent, "_lmhead_vocab_sharded", False):
            logger.warning("QWEN36_DRAFT_SHARDED_ARGMAX ignored: the LM head is not vocab-sharded")
            return
        if vocab % nd != 0:
            logger.warning(f"QWEN36_DRAFT_SHARDED_ARGMAX ignored: vocab {vocab} not divisible by {nd} devices")
            return
        if vocab >= self._SENTINEL:
            logger.warning(f"QWEN36_DRAFT_SHARDED_ARGMAX ignored: vocab {vocab} >= sentinel {self._SENTINEL}")
            return
        shard = vocab // nd
        self._vocab_shard = shard
        # Multi-core max over one vocab shard: ttnn.max(dim=-1) parallelises over tile ROWS, so fold
        # the shard into a tall/narrow [R, C=32] grid (R/32 cores) instead of reducing one 1-row,
        # shard-wide tensor on a single core. Verbatim from demo/text_demo.py's greedy path.
        self._maxval_c = 32
        self._maxval_r = (((shard + self._maxval_c - 1) // self._maxval_c) + 31) // 32 * 32
        # Exactly the config Qwen36Model._lm_head builds for its fp32 branch: HiFi2 matches the
        # bfloat8_b weight and fp32 dest accumulation is what actually keeps the extra bits. It is
        # ALSO what selects the accurate SFPU reduce inside ttnn.max for a FLOAT32 input (see
        # _maxval_dev), which is why the same handle is reused there.
        self._fp32_cfg = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # Device d's vocab shard covers [d*shard, (d+1)*shard): the LM-head weight is sharded with
        # ShardTensorToMesh(dim=-1) (Qwen36Model.__init__) and ShardTensorToMesh(dim=0) here walks
        # the mesh in the same order, so row d lands on the device that owns vocab block d. The
        # offset is broadcast across all lanes so the id side never needs a ttnn.pad.
        off = (torch.arange(nd, dtype=torch.int32) * shard).reshape(nd, 1, 1, 1).repeat(1, 1, 1, lanes)
        self._shard_off = ttnn.from_torch(
            off.contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
        )
        # Lane index of the gathered row, replicated: compared against the winning lane to build the
        # select mask. int32 so that compare is integer-exact (see _argmax_sharded).
        self._sel_lane = ttnn.from_torch(
            torch.arange(nd * lanes, dtype=torch.int32).reshape(1, 1, 1, nd * lanes),
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        self._sharded_argmax = True
        logger.info(
            f"MTP drafter: per-shard argmax enabled (vocab {vocab} = {nd} x {shard}, "
            f"maxval grid {self._maxval_r}x{self._maxval_c})"
        )

    def _maxval_dev(self, shard):
        """Max over ONE device's [1,1,1,vocab/tp] fp32 logit shard -> [1,1,1,1] fp32 TILE.

        Ported from the plain greedy demo path (demo/text_demo.py _maxval_dev) with one addition:
        compute_kernel_config. fp32_dest_acc_en=True is what selects the ACCURATE SFPU reduce for a
        FLOAT32 input (ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp,
        `fp32_sfpu_eligible`); without it ttnn.max takes the FPU path, which truncates its sources
        to TF32 (10 mantissa bits) and would hand the cross-device compare a max ~2**13 coarser than
        the fp32 logits it exists to rank. The demo path passes no config because its logits are
        bfloat16, which survives that truncation intact.
        """
        R, C = self._maxval_r, self._maxval_c
        cfg = self._fp32_cfg
        padded = ttnn.pad(shard, [(0, 0), (0, 0), (0, 0), (0, R * C - self._vocab_shard)], value=self._NEG_FLOOR)
        grid = ttnn.reshape(padded, (1, 1, R, C))
        part = ttnn.max(grid, dim=-1, compute_kernel_config=cfg)
        part_row = ttnn.reshape(part, (1, 1, 1, R))
        val = ttnn.max(part_row, dim=-1, keepdim=True, compute_kernel_config=cfg)
        for t in (padded, grid, part, part_row):
            ttnn.deallocate(t)
        return val

    def _argmax_sharded(self, logits_shard):
        """Global vocab argmax from this device's LOGIT SHARD -> [1,1,1] uint32 ROW_MAJOR.

        Same contract as the gathered argmax_last — same output shape, dtype, layout and tie-break —
        but the full vocab row is never materialised. Each device argmaxes its own
        [1,1,1,vocab/tp] fp32 shard; the winner is then picked from two _LANES-wide all-gathers
        (2 x 32 x 4 B per device) instead of one vocab-wide fp32 one (248320 x 4 B = 993 KB).
        Every step is a fixed-shape device op with no host readback, so this runs in the eager
        chain and captures inside the traced draft body alike.

        B=1 only (all three need_logits=True callers are single-row draft steps).
        The caller owns ``logits_shard``; this never frees it.

        Exactness, step by step:
          * both argmaxes compare RAW fp32 bits in the reader kernel (float32_greater), so the
            ranking is exact fp32 and ties go to the LOWEST index — the same rule, and therefore
            the same token, that a single argmax over the gathered row returns
            (ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/argmax_common.hpp,
            process_value_comparison: strict > to advance, std::min on equality);
          * every id lives in INT32 from the moment it leaves argmax. ttnn.min/max on INT32 take the
            SFPU reduce path and int32 eltwise is integer-exact, where FLOAT32 (and UINT32) go to
            the FPU, which truncates to TF32 — an 11-bit mantissa cannot hold a 248320-range id, so
            a float id reduce returns a number that is no token at all. Same reasoning and the same
            op sequence as _adjust_values_for_tiebreak in models/common/sampling/tt_sampling.py.
        """
        from models.tt_transformers.tt.ccl import tt_all_gather

        lanes = self._LANES
        topo = self.args.ccl_topology()

        # 1. Local argmax over this device's shard. ttnn.argmax needs ROW_MAJOR for the multicore
        #    last-dim path (a TILE input falls back to a single-core internal untilize).
        rm = ttnn.to_layout(logits_shard, ttnn.ROW_MAJOR_LAYOUT)
        lidx = ttnn.argmax(rm, dim=-1, keepdim=False)  # [1,1,1] uint32 ROW_MAJOR, LOCAL index
        ttnn.deallocate(rm)

        # 2. This shard's max VALUE — the only thing the other devices need in order to rank it.
        val = self._maxval_dev(logits_shard)  # [1,1,1,1] fp32 TILE

        # 3. Candidate GLOBAL id = d * vocab_shard + local index, int32, in every lane. The reshape
        #    is a metadata ALIAS (the last dim stays 1), so lidx and lidx4 are one buffer and
        #    exactly one of them may be deallocated.
        lidx4 = ttnn.reshape(lidx, (1, 1, 1, 1))
        lidx_t = ttnn.to_layout(lidx4, ttnn.TILE_LAYOUT)  # uint32 TILE [1,1,1,1]
        ttnn.deallocate(lidx4)
        lidx_i = ttnn.typecast(lidx_t, ttnn.int32)
        ttnn.deallocate(lidx_t)
        gid = ttnn.add(self._shard_off, lidx_i)  # [1,1,1,lanes] int32, broadcast over W
        ttnn.deallocate(lidx_i)

        # 4. One tile-wide value block: lane 0 carries the real max, the rest a floor that can never
        #    win. ttnn.pad is pure data movement, so the fp32 max survives bit-exact.
        valp = ttnn.pad(val, [(0, 0), (0, 0), (0, 0), (0, lanes - 1)], value=self._NEG_FLOOR)
        ttnn.deallocate(val)

        # 5. The two gathers, issued back to back so they overlap. Mirrors Qwen36Model._lm_head's
        #    call exactly (same helper, same tt_ccl, same topology). dtype is passed explicitly
        #    because tt_all_gather defaults its CCL dtype to bfloat16 and TYPECASTS anything else on
        #    the way in — which would put the bf16 ties straight back into the values and destroy
        #    the ids outright.
        vg = tt_all_gather(
            valp, self.mesh_device, self.tt_ccl, cluster_axis=None, dim=3, topology=topo, dtype=ttnn.float32
        )
        gg = tt_all_gather(
            gid, self.mesh_device, self.tt_ccl, cluster_axis=None, dim=3, topology=topo, dtype=ttnn.int32
        )
        ttnn.deallocate(valp)
        ttnn.deallocate(gid)

        # 6. Winning LANE. Device d's block starts at lane d*lanes and only that lane holds a real
        #    value, so the argmax's lowest-index tie-break resolves a cross-device tie to the lowest
        #    device, hence the lowest global id — what the gathered path returns for the same tie.
        vg_rm = ttnn.to_layout(vg, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(vg)
        cw = ttnn.argmax(vg_rm, dim=-1, keepdim=False)  # [1,1,1] uint32 ROW_MAJOR, winning lane
        ttnn.deallocate(vg_rm)
        cw4 = ttnn.reshape(cw, (1, 1, 1, 1))
        cw_t = ttnn.to_layout(cw4, ttnn.TILE_LAYOUT)
        ttnn.deallocate(cw4)
        cw_i = ttnn.typecast(cw_t, ttnn.int32)
        ttnn.deallocate(cw_t)

        # 7. Read the winning lane's id out of gg without ever leaving int32: push every LOSING lane
        #    past every real id by the sentinel, then take the row min. Exactly one lane is left
        #    alone. The 0/1 mask goes through bfloat16 for the multiply (2**24 is exact there, and
        #    it is the combination tt_sampling.py validated on device) and back to int32 for the add.
        nw = ttnn.ne(self._sel_lane, cw_i)  # [1,1,1,lanes*nd] int32 1/0, broadcast over W
        ttnn.deallocate(cw_i)
        nw_b = ttnn.typecast(nw, ttnn.bfloat16)
        ttnn.deallocate(nw)
        off = ttnn.multiply(nw_b, float(self._SENTINEL))
        ttnn.deallocate(nw_b)
        off_i = ttnn.typecast(off, ttnn.int32)
        ttnn.deallocate(off)
        masked = ttnn.add(gg, off_i)
        ttnn.deallocate(gg)
        ttnn.deallocate(off_i)
        win = ttnn.min(masked, dim=-1, keepdim=True)  # [1,1,1,1] int32, SFPU reduce, exact
        ttnn.deallocate(masked)

        # 8. Back to the argmax_last contract: [1,1,1] uint32 ROW_MAJOR, ready for the chain's
        #    reshape(idx, (1,1)) into the next step's embedding lookup.
        win_u = ttnn.typecast(win, ttnn.uint32)
        ttnn.deallocate(win)
        out4 = ttnn.to_layout(win_u, ttnn.ROW_MAJOR_LAYOUT)  # [1,1,1,1] uint32 ROW_MAJOR
        ttnn.deallocate(win_u)
        return ttnn.reshape(out4, (1, 1, 1))  # metadata alias of out4: ONE buffer, ONE deallocate

    def _make_norm(self, state_dict, weight_key, cache, ag_key):
        """RMSNorm (zero-centered) wrapped in DistributedNorm under TP; plain RMSNorm otherwise."""
        norm = RMSNorm(
            device=self.device,
            dim=self.args.dim,
            state_dict=state_dict,
            weight_key=weight_key,
            state_dict_prefix="mtp.",
            weight_cache_path=cache,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=self.args.norm_eps,
            **(
                dict(
                    is_distributed=self.args.is_distributed_norm,
                    ccl_topology=self.args.ccl_topology(),
                    tt_ccl=self.tt_ccl,
                )
                if self.num_devices > 1
                else {}
            ),
        )
        if self.num_devices > 1:
            from models.tt_transformers.tt.distributed_norm import DistributedNorm

            return DistributedNorm(norm, self.args, tt_ccl=self.tt_ccl, TG=self.args.is_galaxy, ag_config_key=ag_key)
        return norm

    def _fuse(self, token_emb, hidden_states, mode):
        """enorm(token_emb) ⊕ hnorm(hidden) -> fc -> fractured hidden [1,1,B|S,dim/tp]."""
        # DECODE gather-then-norm needs a norm_config for the all-gather's (sharded) output memcfg;
        # PREFILL takes the distributed-norm branch and works with the default (None).
        # The pre-fc norms must GATHER their fractured input to full dim (the concat + fc need it),
        # like the model's final norm. Use the "lm_head" norm config (the gather-and-norm path proven
        # in decode by _final_norm_decode), NOT "attn" (which assumes the fused in-proj gathers).
        # The pre-fc norms must GATHER their fractured input to full dim (concat + fc need it), like
        # the model's final norm — use the "lm_head" gather-then-norm config (proven in decode by
        # _final_norm_decode), not "attn" (which assumes the fused in-proj gathers).
        nc = None
        if self.num_devices > 1 and mode == Mode.DECODE:
            nc = dict(self.args.get_norm_config("lm_head", Mode.DECODE))
            nc["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG
        e = self.pre_fc_norm_embedding(token_emb, mode=mode, norm_config=nc)  # -> full [1,1,*,dim]
        h = self.pre_fc_norm_hidden(hidden_states, mode=mode, norm_config=nc)  # -> full [1,1,*,dim]
        cat = ttnn.concat([e, h], dim=-1)  # [1,1,*,2*dim]  (order: [embedding, hidden])
        ttnn.deallocate(e)
        ttnn.deallocate(h)
        kw = dict(compute_kernel_config=self._fc_compute_cfg) if self._fc_compute_cfg is not None else {}
        fused = ttnn.linear(cat, self.fc, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)  # [1,1,*,dim/tp]
        ttnn.deallocate(cat)
        return fused

    def forward_decode(
        self,
        hidden_states,
        token_ids,
        position_idxs,
        cos,
        sin,
        page_table,
        sharded_lm_head=False,
        need_logits=True,
        alias_kv_write=False,
    ):
        """One MTP draft step, or ONE batched KV-maintenance step over B rows.
        ``hidden_states`` [1,1,B,dim/tp] fractured: the base's drafter feed (spec_feed_rows:
        fractured final-norm output), or the previous step's next_hidden when chaining.
        ``token_ids`` [B,1] uint32: token just before what we predict. ``position_idxs`` [B] int32:
        KV write index (base cur_pos + step). ``cos``/``sin``: partial-RoPE. ``page_table`` [B, blocks]
        for the MTP layer's own paged KV. ``alias_kv_write``: B rows belong to ONE sequence at
        consecutive positions (batched reseed), so KV writes share physical blocks and must go row
        by row (TPAttention.forward_decode). Returns (logits, next_hidden), both fractured
        [1,1,B,dim/tp]. need_logits=True: next_hidden is mtp.norm's output re-fractured to dim/tp
        (chain value). need_logits=False skips the head norm and returns RAW block output — KV maintenance only (reseed / catch-up); callers discard it, so it is not a valid chain value.
        """
        mode = Mode.DECODE
        tok_emb = self.embd(token_ids)  # [B,1,dim/tp]
        tok_emb = ttnn.reshape(tok_emb, (1, 1, tok_emb.shape[0] * tok_emb.shape[1], tok_emb.shape[-1]))
        fused = self._fuse(tok_emb, hidden_states, mode)
        ttnn.deallocate(tok_emb)

        next_hidden = self.decoder.forward(
            fused,
            cos=cos,
            sin=sin,
            mode="decode",
            position_tensor=position_idxs,
            page_table=page_table,
            alias_kv_write=alias_kv_write,
        )
        ttnn.deallocate(fused)

        if not need_logits:
            # KV-maintenance step (reseed / catch-up): the caller only needs the drafter's KV written
            # at this slot and throws the logits away, so skip the head norm AND the 151k-vocab LM
            # head (plus its vocab all-gather) entirely. ~half the MTP steps per spec iteration.
            return None, next_hidden

        hnc = None
        if self.num_devices > 1:
            hnc = dict(self.args.get_norm_config("lm_head", Mode.DECODE))
            hnc["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG
        normed = self.head_norm(next_hidden, mode=mode, norm_config=hnc)  # -> full [1,1,B,dim]
        # Chain: the next step is fused from mtp.norm's output, not the raw block output. The DECODE
        # head norm gathers to the full dim (replicated), so slice each device's own dim/tp span back
        # out — the inverse of that all-gather — to restore the fractured [1,1,B,dim/tp] every
        # consumer expects. `normed` itself stays alive for the LM head.
        ttnn.deallocate(next_hidden)
        if self.num_devices > 1:
            next_hidden = ttnn.mesh_partition(normed, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            next_hidden = ttnn.clone(normed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self._sharded_argmax and not sharded_lm_head:
            # Per-shard drafter argmax (QWEN36_DRAFT_SHARDED_ARGMAX): keep this device's
            # [1,1,B,vocab/tp] fp32 shard and skip the vocab all-gather entirely — _argmax_sharded
            # turns it into the global id with two 32-lane gathers instead of one 993 KB one. Same
            # matmul math as the _lm_head fp32 branch below (HiFi2 + fp32 dest accumulation), minus
            # the collective, so the drafter keeps the fp32 tie-breaking it was given fp32 for.
            logits = ttnn.linear(normed, self.lm_head_weight, dtype=ttnn.float32, compute_kernel_config=self._fp32_cfg)
        elif sharded_lm_head or getattr(self, "_ondev_argmax", False):
            logits = ttnn.linear(normed, self.lm_head_weight)  # vocab-sharded shard
        else:
            # fp32 for the DRAFTER only (the shared base/verify call keeps its default bf16 output —
            # losslessness is defined by the base argmax). The drafter's argmax consumes these
            # directly, so bf16 ties that used to discard a good draft are broken correctly.
            logits = self._lm_head(normed, out_dtype=ttnn.float32)
        ttnn.deallocate(normed)
        return logits, next_hidden

    def forward_prefill(self, hidden_states, token_ids, cos, sin, page_table, chunk_page_table=None, chunk_start_idx=0):
        """Warm the MTP paged KV cache over the prompt (one forward, all positions).

        hidden_states : [1,1,S,dim/tp] fractured — the base's per-position drafter feed
                        (Qwen36Model.spec_feed_rows: the fractured final-norm output).
        token_ids     : [1,S] uint32 device — MTP input tokens for the prompt.
        page_table / chunk_page_table : the MTP layer's own paged KV page table.
        Returns the fractured decoder-block output [1,1,S,dim/tp] (raw, before mtp.norm — only the
        KV write matters here; callers free it).
        """
        S = token_ids.shape[-1]
        tok_emb = self.embd(token_ids)  # [1,S,dim/tp]
        tok_emb = ttnn.reshape(tok_emb, (1, 1, S, tok_emb.shape[-1]))
        tok_emb = ttnn.to_memory_config(tok_emb, ttnn.DRAM_MEMORY_CONFIG)
        fused = self._fuse(tok_emb, hidden_states, Mode.PREFILL)
        ttnn.deallocate(tok_emb)
        out = self.decoder.forward(
            fused,
            cos=cos,
            sin=sin,
            mode="prefill",
            page_table=page_table,
            chunk_page_table=chunk_page_table if chunk_page_table is not None else page_table,
            chunk_start_idx=chunk_start_idx,
        )
        ttnn.deallocate(fused)
        return out
