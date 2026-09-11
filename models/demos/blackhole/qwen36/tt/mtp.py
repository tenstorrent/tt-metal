# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5/3.6 MTP (multi-token prediction) head — the speculative-decode drafter.

Every Qwen3.5/3.6 checkpoint ships a single-layer MTP head (the ``mtp.*`` tensors) that
reuses the main model's token embedding and LM head. Structure (mirrors DeepSeek-V3 MTP2D):

    h'  = fc( concat[ enorm(embed(token)), hnorm(hidden) ] )     # fuse token + hidden
    h'' = DecoderLayer(h')                                        # 1 full-attention layer
    logits = LMHead( norm(h'') )                                  # shared head

``enorm``  = mtp.pre_fc_norm_embedding, ``hnorm`` = mtp.pre_fc_norm_hidden,
``fc``     = mtp.fc (eh_proj, [dim, 2*dim]), ``DecoderLayer`` = mtp.layers.0 (reuses the
qwen36 full-attention decoder layer verbatim), ``norm`` = mtp.norm. The head maintains its
OWN paged KV cache (mtp.layers.0.self_attn has its own k/v_proj), separate from the base.

forward_decode returns ``(logits, next_hidden)``: ``next_hidden`` is fed back as ``hidden`` for the
next chained draft step (EAGLE-style K>1). WHAT it is follows the spec feed contract the parent
model selects (Qwen36Model.spec_feed_rows, env QWEN36_SPEC_POSTNORM):

* V3 (default; QWEN36_SPEC_POSTNORM unset or 1): the base feeds the OUTPUT of its final norm
  (fractured to dim/tp), and the chain feeds back the OUTPUT of mtp.norm, re-fractured to dim/tp —
  so the drafter's hnorm sees the same kind of tensor at every step.
* V0 (QWEN36_SPEC_POSTNORM=0): the base feeds its residual stream BEFORE the final norm, and the
  chain feeds back the decoder-block output BEFORE mtp.norm.

Shapes and dtypes are identical in both contracts.

This module is torch-free: every buffer is filled on device (ttnn.zeros) and every host->device
staging tensor is built straight from python ints/floats via ttnn.Tensor, which is bit-identical to
the from_torch + ReplicateTensorToMesh path it replaces (verified including tile padding).
"""

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.tt_transformers.tt.common import Mode


class Qwen36MTP:
    """Single-layer MTP drafter head. Reuses the parent model's embedding + LM head."""

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
        # Feed contract, decided by the parent (see module docstring). V3 makes forward_decode chain
        # mtp.norm's output instead of the raw block output.
        self.spec_postnorm = bool(getattr(parent, "spec_postnorm", False))

        # Shard-argmax greedy pick: keep the drafter's logits vocab-sharded and reduce 8 scalars
        # across the mesh instead of all-gathering the whole fp32 vocab row (that gather was 1.47 ms
        # of a ~3.4 ms drafter leg on T3K/27B; a traced leg is 3986 -> 2684 us). Byte-identical to
        # the gathered argmax, and measured to leave the drafted ids and the acceptance rate
        # unchanged -- see tp_common.greedy_pick. Needs an evenly fractured head; env escape hatch
        # Scoped to T3K (tpc.wh_t3k), the config it was measured and gated on.
        self.shard_argmax = tpc.wh_t3k(args) and args.vocab_size % self.num_devices == 0
        # Replicated offset constant for the combine. Built here so it predates any trace capture.
        self._argmax_offsets = (
            tpc.vocab_shard_offsets(mesh_device, self.num_devices, args.vocab_size // self.num_devices)
            if self.shard_argmax
            else None
        )

        mtp_cache = (tensor_cache_path / "mtp") if tensor_cache_path is not None else None

        # Two pre-fc norms + the post-block norm, keyed under "mtp." (mtp.pre_fc_norm_embedding,
        # mtp.pre_fc_norm_hidden, mtp.norm). Built exactly like Qwen36DecoderLayer._make_norm.
        self.pre_fc_norm_embedding = self._make_norm(state_dict, "pre_fc_norm_embedding", mtp_cache, "attn")
        self.pre_fc_norm_hidden = self._make_norm(state_dict, "pre_fc_norm_hidden", mtp_cache, "attn")
        self.head_norm = self._make_norm(state_dict, "norm", mtp_cache, "lm_head")

        # fc (eh_proj): torch weight [dim, 2*dim] -> concat(token_emb, hidden)[..,2*dim] -> hidden.
        fc_w = state_dict["mtp.fc.weight"]
        assert fc_w.shape == (args.dim, 2 * args.dim), f"unexpected mtp.fc shape {tuple(fc_w.shape)}"
        # A DRAFTER-ONLY bfloat4_b copy of the LM head (T3K). That matmul is
        # 943 us -- 54% of a traced drafter leg -- and it is weight-streaming bound at ~60% of DRAM
        # peak, so config tuning cannot move it (auto is optimal; in0 in L1 costs +1577 us) and the
        # only lever left is bytes: bfp4 halves the 169 MB read. The base/verify head is untouched,
        # so losslessness is unaffected by construction -- the drafter only PROPOSES and every
        # proposal is still checked against the bf16 base argmax. What it can cost is ACCEPTANCE,
        # which is why this is opt-in and measured rather than assumed: it only pays if the drafted
        # tokens survive verification at nearly the old rate.
        #
        # MEASURED on T3K/27B, and acceptance does NOT suffer:
        #   traced drafter leg   2513.2 -> 2108.4 us   (-16.1%, the LM head 943 -> ~540)
        #   demo ISL 128, K=6    45.84 -> 47.26 tok/s  (+3.1%, 2 reps: 47.23 / 47.28)
        #   acceptance           4.00/6 -> 4.10/6      (UP, or level within 10-iteration sampling)
        #   spec lossless        unchanged -- same divergence point, same near-tie flip, 2.69/3
        # Costs ~89 MB/device for the second copy (the bf16 base/verify head stays).
        self._lm_head_bfp4 = None
        if self.shard_argmax and "output.weight" in state_dict:
            self._lm_head_bfp4 = ttnn.as_tensor(
                state_dict["output.weight"].T.contiguous(),  # [dim, vocab], as the parent builds it
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=(str(tensor_cache_path / "output.weight.vshard.bfp4") if tensor_cache_path else None),
                **(
                    dict(mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1))
                    if getattr(parent, "_lmhead_vocab_sharded", False)
                    else {}
                ),
            )
        self._fc_decode_pc = (
            tpc.fc_decode_program_config(mesh_device, 2 * args.dim, args.dim // self.num_devices)
            if tpc.wh_t3k(args)
            else None
        )
        if self.num_devices > 1:
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
        # Drafter-only decode SDPA width. The shared decode program config leaves
        # max_cores_per_head_batch at ttnn's default of 16, which at B=1 and 1 local KV head puts 16
        # of the grid's 110 cores on the KV reduction. The drafter is exactly the shape that hurts:
        # every one of the K draft steps is a B=1 decode that rescans the WHOLE prompt-length KV, so
        # its SDPA is reduction-bound and scales with the core count. 64 is the ceiling the kernel
        # allows (tree reduction is capped at MAX_TREE_REDUCTION_ROUNDS=6 rounds = 2^6 cores/head).
        #
        # Set on the MTP's own TPAttention instance only, so the base model's 16 full-attention
        # layers keep the config they have. The batched reseed (B=K+1 rows through this same
        # instance) is unaffected: at B=11 both 16 and 64 resolve to min(110, max*B)/B = 10
        # cores/head. It DOES change the drafter's reduction order, so bf16 near-ties can round the
        # other way and a different token gets drafted — which only shifts acceptance, never
        # correctness: every draft is arbitrated by the base model's verify.
        self.attention.decode_sdpa_max_cores = 64
        # Traced draft window (init_draft_window / capture_draft_window / draft_leg).
        self._dw = None
        self._rw = None
        # Width of the cos/sin this drafter's attention expects. Under permuted full-width RoPE
        # (wh_9b_n300) the permutation is folded into the drafter's own q/k at load time, so it needs
        # the widened tables -- exactly like every other consumer on this branch. rope_permuted is
        # off elsewhere, where this collapses to rope_head_dim and nothing changes.
        self._rope_full_head_dim = parent.rope.full_head_dim
        self.rope_width = parent.rope.rope_width

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
        # NEGATIVE, do not retry: the two gathers these norms do (46 + 42 us, the biggest non-matmul
        # cost in the prologue) look like one gather of the two inputs ROW-STACKED, since all_gather
        # on dim 3 gathers each row independently. Both forms were built and measured on T3K/27B:
        # a sharded-output gather is rejected outright (the 2-row stack is physical height 2, and
        # the norm's width-shard spec requires tile-sized shards), and the DRAM-output form works
        # and is bit-identical (same drafted ids) but costs +90 us/leg -- the interleaved gather is
        # slower than the sharded one, and the stack/slice/reshard needed to split the rows back out
        # adds five ops. Two cheap gathers beat one awkward one here.
        e = self.pre_fc_norm_embedding(token_emb, mode=mode, norm_config=nc)  # -> full [1,1,*,dim]
        h = self.pre_fc_norm_hidden(hidden_states, mode=mode, norm_config=nc)  # -> full [1,1,*,dim]
        # L1 for the fc's in0 (640 KB at 27B/TP=8): the tt-perf-report hint "place input 0 in L1"
        # is worth -2.1 us here (72.5 -> 70.4). It is NOT worth taking on the LM head, where the same
        # move costs +1577 us (962 -> 2539) -- see tp_common.fc_decode_program_config.
        cat_mc = ttnn.L1_MEMORY_CONFIG if self._fc_decode_pc is not None else None
        cat = ttnn.concat([e, h], dim=-1, **({"memory_config": cat_mc} if cat_mc else {}))  # [1,1,*,2*dim]
        ttnn.deallocate(e)
        ttnn.deallocate(h)
        kw = dict(compute_kernel_config=self._fc_compute_cfg) if self._fc_compute_cfg is not None else {}
        # Swept program config for the 1-tile-tall form (every draft leg, and the batched reseed at
        # B<=32). It is per_core_M=1 + fuse_batch, so anything taller -- prefill -- must keep auto or
        # work_split trips. 2.3x on this matmul; see tp_common.fc_decode_program_config.
        if self._fc_decode_pc is not None and int(cat.shape[-2]) <= 32:
            kw["program_config"] = self._fc_decode_pc
        fused = ttnn.linear(cat, self.fc, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)  # [1,1,*,dim/tp]
        ttnn.deallocate(cat)
        return fused

    def _argmax_last(self, logits):
        """Greedy pick over the vocab dim for ONE row -> [1,1,1] uint32 ROW_MAJOR.

        Takes the shard-combine path when forward_decode left the logits vocab-sharded (the default
        on a mesh), the plain untilize+argmax when they were gathered. Both live in
        tp_common.greedy_pick, which documents why they return the same id.
        """
        return tpc.greedy_pick(
            logits,
            self.mesh_device,
            self.tt_ccl,
            self.args.ccl_topology(),
            shard_offsets=self._argmax_offsets if self.shard_argmax else None,
            vocab_size=self.args.vocab_size,
        )

    # ── traced draft window ──────────────────────────────────────────────────
    # A draft chain is K sequential legs, and MEASURED on T3K/27B each eager leg costs ~19 ms of
    # HOST time (rope build + uploads ~1.8, the ~40-op forward ~14, argmax ~3) against 0.5 ms of
    # device time -- the whole chain drains in 2.8 ms. So the chain is dispatch-bound, not silicon-
    # bound, and the fix is to stop walking python per leg: capture ONE trace per window index and
    # replay it.
    #
    # Everything that varies rides persistent buffers: `pos`/`cos`/`sin` are staged ONCE per
    # iteration for the whole window (one host upload each), and each leg's trace slices its own
    # row at a STATIC offset. The chain itself never touches the host -- each trace ends by copying
    # its own argmax id and next hidden into the `tok`/`h` buffers the next leg reads, so a leg is
    # exactly one execute_trace. Only the K ids are read back, once, at the end.

    def init_draft_window(self, w_max, hidden_dim_frac, hidden_dtype, page_table):
        """Allocate the window buffers. Must run before any trace is captured."""
        if self._dw is not None:
            return
        mesh = self.device
        rd = self.rope_width
        rm, tile = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

        def dev(shape, dtype, layout):
            # Filled on device and replicated across the mesh -- no host tensor to build or upload.
            return ttnn.zeros(shape, dtype=dtype, layout=layout, device=mesh)

        self._dw = {
            "w_max": w_max,
            "pos": dev([w_max], ttnn.int32, rm),
            "cos": dev([1, w_max, 1, rd], ttnn.bfloat16, tile),
            "sin": dev([1, w_max, 1, rd], ttnn.bfloat16, tile),
            "tok": dev([1, 1], ttnn.uint32, rm),
            "h": dev([1, 1, 1, hidden_dim_frac], hidden_dtype, tile),
            "pt": page_table,
            "traces": {},
            "compiled": False,
        }

    def stage_draft_window(self, start_pos, width, rope_delta=0):
        """Upload pos/cos/sin for drafter positions [start_pos, start_pos+width). Once per iteration.

        cos/sin come from the SAME rot_mats_decode the eager leg called, just for the whole window in
        one call instead of once per leg, so a traced chain drafts exactly what the eager one did.
        It hands back device tensors, so they are copied straight into the persistent buffers.
        """
        from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode

        dw = self._dw
        assert width == dw["w_max"], f"draft window must be staged at full width {dw['w_max']}, got {width}"
        pos = list(range(start_pos, start_pos + width))
        src = ttnn.Tensor(pos, [width], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(src, dw["pos"])
        cos_tt, sin_tt = rot_mats_decode(
            self.device,
            self.args.rope_head_dim,
            self.args.max_seq_len,
            self.args.rope_theta,
            [p + rope_delta for p in pos],
            full_head_dim=self._rope_full_head_dim,
        )
        ttnn.copy(cos_tt, dw["cos"])
        ttnn.copy(sin_tt, dw["sin"])
        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

    def _draft_win_body(self, k):
        """Leg k's graph: static window slices in, argmax id out, chain written back in-graph."""
        dw = self._dw
        rd = self.rope_width
        cur_pos = ttnn.slice(dw["pos"], [k], [k + 1])
        cos = ttnn.slice(dw["cos"], [0, k, 0, 0], [1, k + 1, 1, rd])
        sin = ttnn.slice(dw["sin"], [0, k, 0, 0], [1, k + 1, 1, rd])
        logits, next_hidden = self.forward_decode(dw["h"], dw["tok"], cur_pos, cos, sin, dw["pt"], need_logits=True)
        idx = self._argmax_last(logits)  # [1,1,1] uint32 ROW_MAJOR
        ttnn.deallocate(logits)
        # Chain IN-GRAPH: the next leg reads these same buffers, so it needs no host step at all.
        ttnn.copy(ttnn.reshape(idx, (1, 1)), dw["tok"])
        ttnn.copy(next_hidden, dw["h"])
        ttnn.deallocate(next_hidden)
        return idx

    def compile_draft_window(self):
        """Compile every leg's programs EAGERLY. Must precede ANY trace capture: a program that
        first compiles while a trace is parked writes its kernel binaries over that trace."""
        dw = self._dw
        if dw["traces"] or dw["compiled"]:
            return
        for k in range(dw["w_max"]):
            ttnn.deallocate(self._draft_win_body(k))
        ttnn.synchronize_device(self.device)
        dw["compiled"] = True

    def capture_draft_window(self):
        """Capture one trace per window index. Compiles nothing (compile_draft_window ran)."""
        dw = self._dw
        if dw["traces"]:
            return
        self.compile_draft_window()
        for k in range(dw["w_max"]):
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            idx = self._draft_win_body(k)
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
            dw["traces"][k] = {"id": tid, "idx": idx}

    def draft_leg(self, k):
        """Replay leg k. Returns its persistent argmax-id tensor (read it back later, in bulk)."""
        tr = self._dw["traces"][k]
        ttnn.execute_trace(self.device, tr["id"], cq_id=0, blocking=False)
        return tr["idx"]

    def seed_draft_window(self, tok_tt, hidden):
        """Load the chain's first token + anchor hidden into the window buffers (device copies)."""
        ttnn.copy(tok_tt, self._dw["tok"])
        ttnn.copy(hidden, self._dw["h"])

    def release_draft_window(self):
        if self._dw is None:
            return
        for tr in self._dw["traces"].values():
            ttnn.release_trace(self.device, tr["id"])
        self._dw["traces"] = {}

    # ── traced reseed ────────────────────────────────────────────────────────
    # The batched reseed is ONE fixed-shape B=T forward, and MEASURED it costs ~35 ms of host to
    # enqueue against 0.8 ms of device -- the same dispatch-bound shape the draft chain had, made
    # worse by alias_kv_write issuing a per-row paged_update_cache pair. Everything that varies (the
    # tokens, their positions, the per-row page table, the RoPE rows) is data, and the row count is
    # fixed with padding rows aimed at the scratch block, so one trace covers every iteration.

    def init_reseed_window(self, T, num_blocks):
        if self._rw is not None:
            return
        mesh = self.device
        rd = self.rope_width
        rm, tile = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

        def dev(shape, dtype, layout):
            return ttnn.zeros(shape, dtype=dtype, layout=layout, device=mesh)

        self._rw = {
            "T": T,
            "num_blocks": num_blocks,
            "tok": dev([T, 1], ttnn.uint32, rm),
            "pos": dev([T], ttnn.int32, rm),
            "pt": dev([T, num_blocks], ttnn.int32, rm),
            "cos": dev([1, T, 1, rd], ttnn.bfloat16, tile),
            "sin": dev([1, T, 1, rd], ttnn.bfloat16, tile),
            "id": None,
            "compiled": False,
        }

    def stage_reseed_window(self, tok, pos, pt, cos_tt, sin_tt):
        """Refresh the reseed inputs. ``tok``/``pos``/``pt`` are FLAT row-major python int lists, so
        their host tensors are built straight from ints; ``cos_tt``/``sin_tt`` are already ON DEVICE
        (gathered off the resident rope table by _rope_tp_cos_sin_decode_rows) and are copied
        device-to-device, so no rope value crosses the bus. The caller owns them."""
        rw = self._rw
        T, nb = rw["T"], rw["num_blocks"]
        rm = ttnn.ROW_MAJOR_LAYOUT
        for data, shape, dtype, dst in (
            (tok, [T, 1], ttnn.uint32, "tok"),
            (pos, [T], ttnn.int32, "pos"),
            (pt, [T, nb], ttnn.int32, "pt"),
        ):
            ttnn.copy_host_to_device_tensor(ttnn.Tensor(data, shape, dtype, rm), rw[dst])
        ttnn.copy(cos_tt, rw["cos"])
        ttnn.copy(sin_tt, rw["sin"])

    def _reseed_body(self, vhidden):
        rw = self._rw
        _, h_next = self.forward_decode(
            vhidden,
            rw["tok"],
            rw["pos"],
            rw["cos"],
            rw["sin"],
            rw["pt"],
            need_logits=False,
            alias_kv_write=True,
        )
        ttnn.deallocate(h_next)

    def compile_reseed_window(self, vhidden):
        """Compile the reseed programs eagerly, before ANY capture. Its KV writes land wherever the
        staged (zero) page table points -- stage the scratch block everywhere first."""
        if self._rw["compiled"] or self._rw["id"] is not None:
            return
        self._reseed_body(vhidden)
        ttnn.synchronize_device(self.device)
        self._rw["compiled"] = True

    def capture_reseed_window(self, vhidden):
        if self._rw["id"] is not None:
            return
        self.compile_reseed_window(vhidden)
        tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._reseed_body(vhidden)
        ttnn.end_trace_capture(self.device, tid, cq_id=0)
        self._rw["id"] = tid

    def reseed_replay(self):
        ttnn.execute_trace(self.device, self._rw["id"], cq_id=0, blocking=False)

    def release_reseed_window(self):
        if self._rw is not None and self._rw["id"] is not None:
            ttnn.release_trace(self.device, self._rw["id"])
            self._rw["id"] = None

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

        hidden_states : [1,1,B,dim/tp] fractured — the base's drafter feed for the row
                        (Qwen36Model.spec_feed_rows: pre-final-norm residual under V0, fractured
                        final-norm output under V3), or the previous MTP step's next_hidden when
                        chaining.
        token_ids     : [B,1] uint32 device — the token at the position just before what we predict.
        position_idxs : [B] int32 device — KV write index into the MTP cache (base cur_pos + step).
        cos, sin      : partial-RoPE tables for position_idxs (+ rope_delta).
        page_table    : [B, blocks] int32 for the MTP layer's own paged KV cache.
        alias_kv_write: the B rows belong to ONE sequence at consecutive positions (the batched
                        reseed), so their KV writes share physical blocks and must go row by row —
                        see TPAttention.forward_decode.

        Returns (logits, next_hidden), both fractured [1,1,B,dim/tp]. With need_logits=True,
        next_hidden is the chain value: the decoder-block output before mtp.norm (V0) or mtp.norm's
        output re-fractured to dim/tp (V3). With need_logits=False the head norm is skipped and
        next_hidden is ALWAYS the raw block output regardless of contract: that path exists for KV
        maintenance only (reseed / catch-up) and every caller discards the returned hidden, so it is
        not a valid chain value under V3.
        """
        mode = Mode.DECODE
        # T3K takes tpc.decode_embed rather than self.embd: at the drafter's B=1 the plain call's
        # internal TilizeWithValPadding runs on ONE core (55.6 us for a [1,1,1,dim/tp] row, the
        # leg's largest data-movement op). decode_embed splits it; bit-identical either way.
        # Every other config keeps the plain call, so nothing about 9B/N300 changes here.
        tok_emb = (
            tpc.decode_embed(self.embd, token_ids, self.args) if tpc.wh_t3k(self.args) else self.embd(token_ids)
        )  # [B,1,dim/tp]
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
        if self.spec_postnorm:
            # V3 chain: the next step is fused from mtp.norm's output, not the raw block output. The
            # DECODE head norm gathers to the full dim (replicated), so slice each device's own
            # dim/tp span back out — the inverse of that all-gather — to restore the fractured
            # [1,1,B,dim/tp] every consumer expects. `normed` itself stays alive for the LM head.
            ttnn.deallocate(next_hidden)
            if self.num_devices > 1:
                next_hidden = ttnn.mesh_partition(normed, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                next_hidden = ttnn.clone(normed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if sharded_lm_head or getattr(self, "_ondev_argmax", False):
            logits = ttnn.linear(normed, self.lm_head_weight)  # vocab-sharded shard
        else:
            # fp32 for the DRAFTER only (the shared base/verify call keeps its default bf16 output —
            # losslessness is defined by the base argmax). The drafter's argmax consumes these
            # directly, so bf16 ties that used to discard a good draft are broken correctly.
            #
            # gather=False under shard_argmax: _argmax_last reduces the shards itself, so the fp32
            # vocab all-gather is skipped entirely. The matmul is unchanged either way.
            if self._lm_head_bfp4 is not None and self.shard_argmax:
                logits = ttnn.linear(
                    normed,
                    self._lm_head_bfp4,
                    dtype=ttnn.float32,
                    compute_kernel_config=ttnn.init_device_compute_kernel_config(
                        self.device.arch(),
                        math_fidelity=ttnn.MathFidelity.LoFi,  # LoFi matches the bfp4 weight
                        fp32_dest_acc_en=True,
                        packer_l1_acc=False,
                    ),
                )
            else:
                logits = self._lm_head(normed, out_dtype=ttnn.float32, gather=not self.shard_argmax)
        ttnn.deallocate(normed)
        return logits, next_hidden

    def forward_prefill(self, hidden_states, token_ids, cos, sin, page_table, chunk_page_table=None, chunk_start_idx=0):
        """Warm the MTP paged KV cache over the prompt (one forward, all positions).

        hidden_states : [1,1,S,dim/tp] fractured — the base's per-position drafter feed
                        (Qwen36Model.spec_feed_rows: pre-final-norm under V0, fractured final-norm
                        output under V3).
        token_ids     : [1,S] uint32 device — MTP input tokens for the prompt.
        page_table / chunk_page_table : the MTP layer's own paged KV page table.
        Returns the fractured decoder-block output [1,1,S,dim/tp] (raw, before mtp.norm, under both
        contracts — only the KV write matters here; callers free it).
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
