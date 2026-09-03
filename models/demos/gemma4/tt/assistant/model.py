# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 it-assistant drafter model (EAGLE / Multi-Token-Prediction).

The drafter is a tiny Gemma4 text model (4 layers, hidden 1024) that proposes K
candidate tokens from a single backbone position. Each step:

    inputs_embeds = cat(target_embed(last_token), last_hidden)       # [.., 2*backbone] (scaled embed)
    h = pre_projection(inputs_embeds)                                # [.., hidden]
    for layer in 4 decoder layers:                                   # cross-attend
        h = layer(h, kv = target's last {sliding,full} layer KV)     #   into target KV
    h = norm(h)
    logits      = lm_head(h)                                         # next draft token (argmax)
    next_hidden = post_projection(h)                                 # recurrent hidden

The decoder layers are ordinary ``Gemma4DecoderLayer``s (MoE disabled) run in
decode mode with ``is_kv_shared=True``: they compute only Q (the K/V weights are
synthesized as zeros and discarded) and the SDPA attends into the *target's* KV
cache for that layer type. ``position_ids`` and the target KV are held fixed
across the K drafter steps — matching HF's
``SinglePositionMultiTokenCandidateGenerator``.

Reference: transformers ``Gemma4AssistantForCausalLM.forward`` and
``generation/candidate_generator.py:SinglePositionMultiTokenCandidateGenerator``.

Constraints (first cut):
  * batch = 1
  * the target must use UNBOUNDED sliding KV caches (``bounded_sliding_kv_cache``
    off) so the drafter's cross-attention reads absolute cache positions without
    a circular-buffer modulo (the assistant attention config doesn't carry one).
"""

import os

import torch

import ttnn
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4.tt.ccl import ccl_allgather
from models.demos.gemma4.tt.dram_sharded import decode_1d_matmul_config, decode_in0_l1_enabled, lm_head_decode_config
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.rms_norm import RMSNorm
from models.demos.gemma4.utils.general_utils import get_cache_file_name
from models.demos.gemma4.utils.substate import substate


def _inject_zero_kv_weights(state_dict, text_args):
    """Add zero k_proj/v_proj and a unit k_norm for every assistant layer.

    The assistant checkpoint stores no K/V projections (its layers are all
    KV-shared), but ``Gemma4DecoderLayer``'s attention loader expects a full
    fused QKV. We inject zeros for the K/V columns (the split discards them under
    ``is_kv_shared=True``) and a unit k_norm so the loader's unconditional
    ``k_norm.weight`` read succeeds. Mutates and returns ``state_dict``.
    """
    hidden = text_args.hidden_size
    for i in range(text_args.num_hidden_layers):
        cfg = Gemma4AttentionConfig(text_args, i)
        kv_size = cfg.num_key_value_heads * cfg.head_dim
        prefix = f"model.layers.{i}.self_attn"
        if f"{prefix}.k_proj.weight" not in state_dict:
            state_dict[f"{prefix}.k_proj.weight"] = torch.zeros((kv_size, hidden), dtype=torch.bfloat16)
        # Sliding (non-global) layers load a separate v_proj; global layers tie V=K.
        if not cfg.use_kv_tying and f"{prefix}.v_proj.weight" not in state_dict:
            state_dict[f"{prefix}.v_proj.weight"] = torch.zeros((kv_size, hidden), dtype=torch.bfloat16)
        if f"{prefix}.k_norm.weight" not in state_dict:
            state_dict[f"{prefix}.k_norm.weight"] = torch.ones((cfg.head_dim,), dtype=torch.bfloat16)
    return state_dict


class Gemma4AssistantModel:
    def __init__(
        self,
        mesh_device,
        assistant_args,
        target_model,
        state_dict,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=None,
        max_local_batch_size=1,
        precision=None,
    ):
        self.mesh_device = mesh_device
        self.max_local_batch_size = max_local_batch_size
        self.args = assistant_args
        self.text_args = assistant_args.text_args
        self.target = target_model
        self.ccl_manager = ccl_manager
        self.mesh_config = mesh_config
        self.backbone_hidden_size = assistant_args.backbone_hidden_size
        self.hidden_size = self.text_args.hidden_size
        self.vocab_size = self.text_args.vocab_size
        self.layer_types = list(self.text_args.layer_types)

        if assistant_args.use_ordered_embeddings:
            raise NotImplementedError(
                "use_ordered_embeddings (centroid masked embedding) is not supported; "
                "31B/12B assistants set it False."
            )

        tp = mesh_config.tp if mesh_config else 1
        is_mesh = hasattr(mesh_device, "shape")
        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

        # The drafter shares the target's per-layer-type RoPE caches (identical
        # head_dim / theta), so its Q is RoPE'd consistently with the cached K.
        self.rope_caches_2d = target_model.rope_caches_2d

        state_dict = _inject_zero_kv_weights(dict(state_dict), self.text_args)

        # Per-module dtype overrides from precision_overrides.json (see
        # Gemma4Model for the target-model equivalent). Without this the
        # drafter's attention/mlp weights silently stayed at ``dtype`` (bf16)
        # regardless of what the table says for this checkpoint, since the
        # assistant is a separate checkpoint keyed independently of the target.
        from models.demos.gemma4.tt.precision import Gemma4Precision

        if precision is None:
            precision = Gemma4Precision()
        shared_mlp_dtype = precision.get("shared_mlp", dtype)
        attention_dtype = precision.get("attention", dtype)
        lm_head_dtype = precision.get("lm_head", dtype)

        # Decoder layers (reuse the target's layer, MoE disabled, KV-shared).
        self.layers = []
        for i in range(self.text_args.num_hidden_layers):
            layer = Gemma4DecoderLayer(
                mesh_device=mesh_device,
                hf_config=self.text_args,
                state_dict=state_dict,
                layer_idx=i,
                ccl_manager=ccl_manager,
                dtype=dtype,
                shared_mlp_dtype=shared_mlp_dtype,
                attention_dtype=attention_dtype,
                tensor_cache_path=f"{tensor_cache_path}/layer_{i}" if tensor_cache_path else None,
                mesh_config=mesh_config,
                max_seq_len=self.text_args.max_seq_len,
                max_local_batch_size=max_local_batch_size,
            )
            self.layers.append(layer)

        # Final norm (model.norm)
        self.norm = RMSNorm(
            mesh_device=mesh_device,
            hf_config=self.text_args,
            state_dict=substate(state_dict, "model.norm"),
            tensor_cache_path=f"{tensor_cache_path}/final_norm" if tensor_cache_path else None,
            mesh_config=mesh_config,
        )

        # pre_projection (2*backbone -> hidden) and post_projection (hidden ->
        # backbone) are small and kept replicated so hidden stays full-width
        # across TP (matching the layer norms / attention which expect full
        # hidden). lm_head (hidden -> vocab) is column-parallel on vocab and
        # all-gathered, mirroring the target.
        col_mapper = mesh_config.column_parallel(mesh_device) if tp > 1 else None

        def _linear(key, mapper, transpose=True, dtype_override=None, cache_suffix=""):
            w = state_dict.get(key)
            if w is None:
                return None
            wt = w.transpose(-2, -1) if transpose else w
            wt = wt.unsqueeze(0).unsqueeze(0)
            return ttnn.as_tensor(
                wt,
                device=mesh_device,
                dtype=dtype_override if dtype_override is not None else dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mapper if mapper is not None else (replicate if is_mesh else None),
                cache_file_name=get_cache_file_name(tensor_cache_path, key.replace(".", "_") + cache_suffix),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # pre_projection is DRAM-bandwidth-bound at decode (M=1 padded to a
        # single tile, K=2*backbone, N=hidden): its whole cost is reading the
        # weight, so bfp8 halves the bytes read vs the model-wide default
        # (bf16). Opt out with GEMMA4_PREPROJ_BFP8=0. The cache filename gets
        # a distinct suffix whenever this differs from the model-wide dtype so
        # flipping the flag can never load a stale wrong-dtype tensorbin.
        preproj_bfp8 = os.environ.get("GEMMA4_PREPROJ_BFP8", "1").lower() not in ("0", "false", "no")
        preproj_dtype = ttnn.bfloat8_b if preproj_bfp8 else dtype
        preproj_suffix = "_bfp8" if preproj_dtype != dtype else ""
        self.pre_projection = _linear(
            "pre_projection.weight", None, dtype_override=preproj_dtype, cache_suffix=preproj_suffix
        )
        # post_projection (hidden -> backbone) is pre_projection's output-side
        # mirror and is just as DRAM-bandwidth-bound at decode (M=1 padded to a
        # tile) -- bfp8 halves its weight-read bytes too. Its N (backbone,
        # e.g. 5376) is wide relative to the compute grid, unlike
        # pre_projection's N=hidden, so it does NOT get a decode_1d_matmul_config
        # below: decode_1d_matmul_config's own docstring says wide-N decode
        # matmuls are already at the DRAM-bandwidth ceiling under auto and a
        # forced grid only costs time there.
        postproj_bfp8 = os.environ.get("GEMMA4_POSTPROJ_BFP8", "1").lower() not in ("0", "false", "no")
        postproj_dtype = ttnn.bfloat8_b if postproj_bfp8 else dtype
        postproj_suffix = "_bfp8" if postproj_dtype != dtype else ""
        self.post_projection = _linear(
            "post_projection.weight", None, dtype_override=postproj_dtype, cache_suffix=postproj_suffix
        )
        # lm_head tied to the assistant's own embed_tokens when a separate
        # lm_head.weight isn't stored. Follows precision_overrides.json's
        # "lm_head" entry (bfp8 for the shipped variants), same override the
        # target model's own lm_head gets.
        lm_key = "lm_head.weight" if "lm_head.weight" in state_dict else "model.embed_tokens.weight"
        lm_head_suffix = "_bfp8" if lm_head_dtype != dtype else ""
        self.lm_head = _linear(lm_key, col_mapper, dtype_override=lm_head_dtype, cache_suffix=lm_head_suffix)
        if self.pre_projection is None or self.post_projection is None or self.lm_head is None:
            raise ValueError("Assistant checkpoint missing pre_projection / post_projection / lm_head weights")

        # Narrow-N decode program config for pre_projection (K=2*backbone,
        # N=hidden e.g. 10752x1024 on the 31B assistant): ttnn auto spreads N
        # one tile per core on this shape and collapses the output subblock to
        # 1x1, stalling the pipeline. Same tuned family as the target's fused
        # QKV matmul (see attention/weights.py); NOT bit-exact vs auto (it
        # re-chooses the blocking / accumulation order). Opt out independently
        # of QKV's flag with GEMMA4_PREPROJ_DECODE_PROGCFG=0 (note
        # decode_1d_matmul_config itself also honors GEMMA4_QKV_DECODE_PROGCFG).
        preproj_progcfg = os.environ.get("GEMMA4_PREPROJ_DECODE_PROGCFG", "1").lower() not in ("0", "false", "no")
        self._pre_proj_decode_config = (
            decode_1d_matmul_config(mesh_device, 2 * self.backbone_hidden_size, self.hidden_size)
            if preproj_progcfg
            else None
        )

        # lm_head decode config: same tuned helper the TARGET model already
        # uses for its own last-token lm_head (Gemma4Model.compute_logits ->
        # lm_head_decode_config), just never wired up for the assistant's
        # separate lm_head call. M is always one padded tile here (assistant
        # decode is always single-token, per-layer position doesn't change the
        # shape -- see the pre_projection config above), so this is computed
        # once and reused for every step() regardless of sequence position.
        self._lm_head_decode_config = lm_head_decode_config(
            mesh_device, ttnn.TILE_SIZE, self.hidden_size, int(self.lm_head.shape[-1])
        )

    def _raw_token_embed(self, token_tt):
        """Target token embedding of a single token id -> [1,1,1,backbone] TILE.

        Uses the *scaled* embedding (``embed_tokens``; device table has
        ``sqrt(hidden)`` baked in at load). HF's ``embed_tokens`` is a
        ``Gemma4TextScaledWordEmbedding`` that applies the ``sqrt(hidden)``
        normalizer inside its forward, so the drafter input
        ``cat(get_input_embeddings()(token), hidden)`` carries the *scaled*
        embedding. Feeding the unscaled table (~62x too small) starves the
        ``pre_projection`` token branch and collapses drafter acceptance
        (measured 0.19 unscaled -> 1.44 scaled, matching the HF reference).

        Requests TILE layout from ``embed_tokens`` directly (it tilizes inside
        the embedding kernel) instead of a standalone ``ttnn.to_layout`` after
        the lookup + TP all-gather — see ``Gemma4Model.embed_tokens``'s own
        docstring: dropping that separate ``TilizeDeviceOperation`` measured
        faster with bit-identical output on the main decode path, and RoPE's
        cos/sin cache lookups already use the same ``layout=`` pattern.
        """
        emb = self.target.embed_tokens(token_tt, layout=ttnn.TILE_LAYOUT)
        if len(emb.shape) == 3:
            emb = ttnn.unsqueeze_to_4D(emb)
        return emb

    def step(
        self,
        token_tt,
        target_hidden,
        shared_kv,
        page_tables,
        pos_uint32,
        pos_int32,
        return_logits=True,
        gather_logits=True,
    ):
        """One drafter step.

        Args:
            token_tt: [1,1] uint32 last token id.
            target_hidden: [1,1,1,backbone] TILE — the recurrent hidden (target's
                last-token hidden on the first step, then this method's previous
                ``next_hidden``).
            shared_kv: {layer_type: [k_cache, v_cache]} target caches.
            page_tables: {layer_type: page_table} (or a single page_table reused
                for both types in the simple unbounded case).
            pos_uint32: [1,32] uint32 fixed position for RoPE lookup.
            pos_int32: [1] int32 fixed position for SDPA cur_pos.
            return_logits: when False, skip the lm_head + its TP all-gather and
                return ``(None, next_hidden)`` (used to isolate the lm_head/CCL
                cost in timing harnesses).
            gather_logits: when False (and ``return_logits``), skip the vocab
                all-gather and return TP-sharded logits. Greedy fused decode
                reduces those with a local argmax + tiny gather instead of
                moving the full 262k row.

        Returns:
            (logits [1,1,1,vocab or vocab/tp], next_hidden [1,1,1,backbone]).
        """
        # Decode activations/outputs are tiny (one padded tile) at every matmul
        # in this method -- landing them in L1 removes a DRAM round-trip on
        # each side without touching any matmul's own program config. Same
        # tradeoff as dram_sharded.decode_in0_l1_enabled elsewhere in the
        # decode path; one flag covers all of them here.
        decode_l1 = decode_in0_l1_enabled()
        decode_out_memcfg = ttnn.L1_MEMORY_CONFIG if decode_l1 else ttnn.DRAM_MEMORY_CONFIG

        tok_embed = self._raw_token_embed(token_tt)
        inp = ttnn.concat([tok_embed, target_hidden], dim=-1)
        tok_embed.deallocate(True)

        if decode_l1 and inp.memory_config().buffer_type != ttnn.BufferType.L1:
            inp_l1 = ttnn.to_memory_config(inp, ttnn.L1_MEMORY_CONFIG)
            inp.deallocate(True)
            inp = inp_l1

        if self._pre_proj_decode_config is not None:
            program_config, compute_kernel_config = self._pre_proj_decode_config
            h = ttnn.linear(
                inp,
                self.pre_projection,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=decode_out_memcfg,
            )
        else:
            h = ttnn.linear(inp, self.pre_projection, memory_config=decode_out_memcfg)
        inp.deallocate(True)

        for i, layer in enumerate(self.layers):
            lt = self.layer_types[i]
            pt = page_tables[lt] if isinstance(page_tables, dict) else page_tables
            h = layer(
                h,
                rope_mats=self.rope_caches_2d[lt],
                position_idx=pos_uint32,
                page_table=pt,
                kv_cache=shared_kv[lt],
                is_decode=True,
                token_index=None,
                is_kv_shared=True,
                position_idx_cache=pos_int32,
            )

        # normed feeds BOTH lm_head and post_projection below -- land it in L1
        # once here rather than letting each matmul's default (DRAM) round-trip
        # it separately.
        normed = self.norm.forward(h, interleaved_memory_config=decode_out_memcfg if decode_l1 else None)
        h.deallocate(True)

        logits = None
        if return_logits:
            if self._lm_head_decode_config is not None:
                program_config, out_memcfg, compute_kernel_config = self._lm_head_decode_config
                logits = ttnn.linear(
                    normed,
                    self.lm_head,
                    program_config=program_config,
                    memory_config=out_memcfg,
                    compute_kernel_config=compute_kernel_config,
                )
            else:
                logits = ttnn.linear(normed, self.lm_head, memory_config=decode_out_memcfg)
            if gather_logits and self.mesh_config is not None and self.mesh_config.tp > 1:
                logits = ccl_allgather(logits, self.mesh_config, self.ccl_manager)

        next_hidden = ttnn.linear(normed, self.post_projection, memory_config=decode_out_memcfg)
        normed.deallocate(True)
        return logits, next_hidden
