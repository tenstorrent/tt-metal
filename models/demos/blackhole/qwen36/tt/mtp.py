# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5/3.6 single-layer MTP drafter for speculative decode.
Reuses the parent embedding and LM head, and keeps its own paged KV cache.
"""

import os

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.tt_transformers.tt.common import Mode


class Qwen36MTP:
    """Single-layer MTP drafter. Reuses the parent embedding and LM head."""

    def __init__(self, mesh_device, args, state_dict, parent, tensor_cache_path=None, tt_ccl=None):
        self.args = args
        self.device = mesh_device
        self.mesh_device = mesh_device
        self.num_devices = getattr(args, "num_devices", 1)
        self.tt_ccl = tt_ccl

        self.embd = parent.embd
        self._lm_head = parent._lm_head
        self.lm_head_weight = parent.lm_head_weight
        # V3 chains mtp.norm's output; V0 chains the raw block output.
        self.spec_postnorm = bool(getattr(parent, "spec_postnorm", False))

        # T3K shard-argmax: vocab-sharded logits, even fracture only; same id as the gathered argmax.
        self.shard_argmax = tpc.wh_t3k(args) and args.vocab_size % self.num_devices == 0
        # QWEN36_MTP_HIPREC_DRAFT=1: bf16 gathered argmax. Do not pair a bf16 head with shard_argmax (hangs).
        self._hiprec_draft = os.environ.get("QWEN36_MTP_HIPREC_DRAFT", "0") == "1"
        if self._hiprec_draft:
            self.shard_argmax = False
        # Replicated offset constant for the combine. Built here so it predates any trace capture.
        self._argmax_offsets = (
            tpc.vocab_shard_offsets(mesh_device, self.num_devices, args.vocab_size // self.num_devices)
            if self.shard_argmax
            else None
        )

        mtp_cache = (tensor_cache_path / "mtp") if tensor_cache_path is not None else None

        self.pre_fc_norm_embedding = self._make_norm(state_dict, "pre_fc_norm_embedding", mtp_cache, "attn")
        self.pre_fc_norm_hidden = self._make_norm(state_dict, "pre_fc_norm_hidden", mtp_cache, "attn")
        self.head_norm = self._make_norm(state_dict, "norm", mtp_cache, "lm_head")

        # fc (eh_proj): torch weight [dim, 2*dim] -> concat(token_emb, hidden)[..,2*dim] -> hidden.
        fc_w = state_dict["mtp.fc.weight"]
        assert fc_w.shape == (args.dim, 2 * args.dim), f"unexpected mtp.fc shape {tuple(fc_w.shape)}"
        # Drafter-only bfp4 LM head. The base/verify head stays bf16, so verify losslessness is unchanged.
        self._lm_head_bfp4 = None
        if self.shard_argmax and not self._hiprec_draft and "output.weight" in state_dict:
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

        # Remap mtp.layers.0.* onto a full-attention layer index so the decoder loader finds its keys.
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
        # Dedicated cache root so the reused layer's weights never collide with the real layer L.
        self.decoder = Qwen36DecoderLayer(
            mesh_device, args, mtp_layer_sd, layer_num=L, tensor_cache_path=mtp_cache, tt_ccl=tt_ccl
        )
        # KV accessor for allocate_kv_caches / rollback.
        self.attention = self.decoder.attention
        # Drafter-only. 64 is the kernel core ceiling; a different reduction order can change drafts, not verify.
        self.attention.decode_sdpa_max_cores = 64
        self._dw = None
        self._rw = None
        # Permuted full-width RoPE needs the widened tables; otherwise this is rope_head_dim.
        self._rope_full_head_dim = parent.rope.full_head_dim
        self.rope_width = parent.rope.rope_width

    def _make_norm(self, state_dict, weight_key, cache, ag_key):
        """Zero-centered RMSNorm; DistributedNorm under TP."""
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
        """Concat enorm(token) and hnorm(hidden), then fc, to fractured [1, 1, *, dim/tp]."""
        # DECODE gather-then-norm needs a norm_config; pre-fc norms gather to full dim via the lm_head config.
        # Do not fuse the two pre-fc gathers into one row-stacked gather.
        nc = None
        if self.num_devices > 1 and mode == Mode.DECODE:
            nc = dict(self.args.get_norm_config("lm_head", Mode.DECODE))
            nc["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG
        e = self.pre_fc_norm_embedding(token_emb, mode=mode, norm_config=nc)  # -> full [1,1,*,dim]
        h = self.pre_fc_norm_hidden(hidden_states, mode=mode, norm_config=nc)  # -> full [1,1,*,dim]
        # L1 for the fc in0 when the decode program config is set. Do not do this for the LM head.
        cat_mc = ttnn.L1_MEMORY_CONFIG if self._fc_decode_pc is not None else None
        cat = ttnn.concat([e, h], dim=-1, **({"memory_config": cat_mc} if cat_mc else {}))  # [1,1,*,2*dim]
        ttnn.deallocate(e)
        ttnn.deallocate(h)
        kw = dict(compute_kernel_config=self._fc_compute_cfg) if self._fc_compute_cfg is not None else {}
        # per_core_M=1 config; taller than 32 rows must not use it or work_split trips.
        _m = int(cat.shape[-2])
        if self._fc_decode_pc is not None and _m <= 32:
            kw["program_config"] = self._fc_decode_pc
        elif _m >= 512:
            # Prefill matmul config for M>=512 only; smaller buckets stay on auto. Fall back to auto on error.
            try:
                _g = self.device.compute_with_storage_grid_size()
                kw["program_config"] = tpc.create_prefill_matmul_program_config(
                    _m, cat.shape[-1], self.fc.shape[-1], grid_size=(_g.x, _g.y)
                )
            except Exception:
                pass  # auto
        fused = ttnn.linear(cat, self.fc, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)  # [1,1,*,dim/tp]
        ttnn.deallocate(cat)
        return fused

    def _argmax_last(self, logits):
        """Greedy pick for one row; shard-combine or untilize+argmax, same id either way."""
        return tpc.greedy_pick(
            logits,
            self.mesh_device,
            self.tt_ccl,
            self.args.ccl_topology(),
            shard_offsets=self._argmax_offsets if self.shard_argmax else None,
            vocab_size=self.args.vocab_size,
        )

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
        """Stage the whole window's rope once; same rot_mats_decode the eager path uses."""
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
        """Compile every leg before any trace capture; a later compile clobbers a parked trace."""
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
        ttnn.copy(tok_tt, self._dw["tok"])
        ttnn.copy(hidden, self._dw["h"])

    def release_draft_window(self):
        if self._dw is None:
            return
        for tr in self._dw["traces"].values():
            ttnn.release_trace(self.device, tr["id"])
        self._dw["traces"] = {}

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
        """Stage flat host int lists and on-device cos/sin into the persistent reseed buffers."""
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
        """Compile before any capture; stage the scratch block first so throwaway KV misses the sequence."""
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
        """One draft or reseed step. alias_kv_write is one sequence; need_logits=False is not a V3 chain value."""
        mode = Mode.DECODE
        # T3K: decode_embed splits the B=1 tilize. Bit-identical; other configs keep embd.
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
            return None, next_hidden

        hnc = None
        if self.num_devices > 1:
            hnc = dict(self.args.get_norm_config("lm_head", Mode.DECODE))
            hnc["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG
        normed = self.head_norm(next_hidden, mode=mode, norm_config=hnc)  # -> full [1,1,B,dim]
        if self.spec_postnorm:
            # V3: chain mtp.norm's output, re-fractured to dim/tp. `normed` stays for the LM head.
            ttnn.deallocate(next_hidden)
            if self.num_devices > 1:
                next_hidden = ttnn.mesh_partition(normed, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                next_hidden = ttnn.clone(normed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if sharded_lm_head or getattr(self, "_ondev_argmax", False):
            logits = ttnn.linear(normed, self.lm_head_weight)  # vocab-sharded shard
        else:
            # Drafter logits are fp32; base/verify stays bf16. shard_argmax skips the vocab all-gather.
            if self._lm_head_bfp4 is not None and self.shard_argmax:
                logits = ttnn.linear(
                    normed,
                    self._lm_head_bfp4,
                    dtype=ttnn.float32,
                    compute_kernel_config=ttnn.init_device_compute_kernel_config(
                        self.device.arch(),
                        math_fidelity=ttnn.MathFidelity.LoFi,  # matches the bfp4 weight
                        packer_l1_acc=True,  # output unchanged; fp32 dest accumulation stays on
                    ),
                )
            else:
                logits = self._lm_head(normed, out_dtype=ttnn.float32, gather=not self.shard_argmax)
        ttnn.deallocate(normed)
        return logits, next_hidden

    def forward_prefill(self, hidden_states, token_ids, cos, sin, page_table, chunk_page_table=None, chunk_start_idx=0):
        """Warm MTP paged KV over the prompt. Returns the raw block output; only the KV write matters."""
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
