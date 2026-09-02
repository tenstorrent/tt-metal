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

forward_decode returns ``(logits, next_hidden)``: ``next_hidden`` is the decoder-block output
(pre-mtp.norm), fed back as ``hidden`` for the next chained draft step (EAGLE-style K>1).
"""
import ttnn
from models.common.rmsnorm import RMSNorm
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
        L = next(i for i, t in enumerate(args.attention_type_list) if t == "full_attention")
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

        hidden_states : [1,1,B,dim/tp] fractured — base last-decoder-layer output (pre final norm),
                        or the previous MTP step's next_hidden when chaining.
        token_ids     : [B,1] uint32 device — the token at the position just before what we predict.
        position_idxs : [B] int32 device — KV write index into the MTP cache (base cur_pos + step).
        cos, sin      : partial-RoPE tables for position_idxs (+ rope_delta).
        page_table    : [B, blocks] int32 for the MTP layer's own paged KV cache.
        alias_kv_write: the B rows belong to ONE sequence at consecutive positions (the batched
                        reseed), so their KV writes share physical blocks and must go row by row —
                        see TPAttention.forward_decode.

        Returns (logits, next_hidden). next_hidden is the decoder-block output (fractured, pre norm).
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
        if sharded_lm_head or getattr(self, "_ondev_argmax", False):
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

        hidden_states : [1,1,S,dim/tp] fractured — base per-position last-hidden (pre final norm).
        token_ids     : [1,S] uint32 device — MTP input tokens for the prompt.
        page_table / chunk_page_table : the MTP layer's own paged KV page table.
        Returns the fractured decoder-block output [1,1,S,dim/tp] (for optional last-row logits).
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
