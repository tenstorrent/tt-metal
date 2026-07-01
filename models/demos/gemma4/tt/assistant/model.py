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

import torch

import ttnn
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4.tt.ccl import ccl_allgather
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
    ):
        self.mesh_device = mesh_device
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
                tensor_cache_path=f"{tensor_cache_path}/layer_{i}" if tensor_cache_path else None,
                mesh_config=mesh_config,
                max_seq_len=self.text_args.max_seq_len,
                max_local_batch_size=1,
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

        def _linear(key, mapper, transpose=True):
            w = state_dict.get(key)
            if w is None:
                return None
            wt = w.transpose(-2, -1) if transpose else w
            wt = wt.unsqueeze(0).unsqueeze(0)
            return ttnn.as_tensor(
                wt,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mapper if mapper is not None else (replicate if is_mesh else None),
                cache_file_name=get_cache_file_name(tensor_cache_path, key.replace(".", "_")),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.pre_projection = _linear("pre_projection.weight", None)
        self.post_projection = _linear("post_projection.weight", None)
        # lm_head tied to the assistant's own embed_tokens when a separate
        # lm_head.weight isn't stored.
        lm_key = "lm_head.weight" if "lm_head.weight" in state_dict else "model.embed_tokens.weight"
        self.lm_head = _linear(lm_key, col_mapper)
        if self.pre_projection is None or self.post_projection is None or self.lm_head is None:
            raise ValueError("Assistant checkpoint missing pre_projection / post_projection / lm_head weights")

    def _raw_token_embed(self, token_tt):
        """Target token embedding of a single token id -> [1,1,1,backbone] TILE.

        Uses the *scaled* embedding (``embed_tokens`` = raw table * sqrt(hidden)).
        HF's ``embed_tokens`` is a ``Gemma4TextScaledWordEmbedding`` that applies
        the ``sqrt(hidden)`` normalizer inside its forward, so the drafter input
        ``cat(get_input_embeddings()(token), hidden)`` carries the *scaled*
        embedding. Feeding the unscaled table (~62x too small) starves the
        ``pre_projection`` token branch and collapses drafter acceptance
        (measured 0.19 unscaled -> 1.44 scaled, matching the HF reference).
        """
        emb = self.target.embed_tokens(token_tt)
        if len(emb.shape) == 3:
            emb = ttnn.unsqueeze_to_4D(emb)
        return ttnn.to_layout(emb, ttnn.TILE_LAYOUT)

    def step(self, token_tt, target_hidden, shared_kv, page_tables, pos_uint32, pos_int32, return_logits=True):
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

        Returns:
            (logits [1,1,1,vocab], next_hidden [1,1,1,backbone]).
        """
        tok_embed = self._raw_token_embed(token_tt)
        inp = ttnn.concat([tok_embed, target_hidden], dim=-1)
        tok_embed.deallocate(True)

        h = ttnn.linear(inp, self.pre_projection)
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

        normed = self.norm.forward(h)
        h.deallocate(True)

        logits = None
        if return_logits:
            logits = ttnn.linear(normed, self.lm_head)
            if self.mesh_config is not None and self.mesh_config.tp > 1:
                logits = ccl_allgather(logits, self.mesh_config, self.ccl_manager)

        next_hidden = ttnn.linear(normed, self.post_projection)
        normed.deallocate(True)
        return logits, next_hidden
