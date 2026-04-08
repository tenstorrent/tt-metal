# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Model — full on-device implementation following gpt-oss pattern.

Architecture:
- 30 decoder layers with [5 sliding, 1 full] x 5 pattern
- Two RoPE configs: sliding (head_dim=256, theta=10k) and global (head_dim=512, theta=1M)
- Embedding scaled by sqrt(hidden_size)
- final_logit_softcapping = 30.0
- tie_word_embeddings = True

Supports both prefill and decode modes with paged attention.
Compatible with tt_transformers Generator interface.
"""

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.rms_norm import RMSNorm
from models.demos.gemma4.utils.general_utils import get_cache_file_name
from models.demos.gemma4.utils.substate import substate


def create_rope_caches(mesh_device, hf_config, max_seq_len):
    """Create HF-format cos/sin caches for both sliding and global layer types.

    Returns:
        caches_4d: dict mapping layer_type -> (cos_tt, sin_tt) [1,1,max_seq_len,head_dim] for prefill
        caches_2d: dict mapping layer_type -> (cos_tt, sin_tt) [max_seq_len,head_dim] for decode embedding lookup
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    is_mesh = hasattr(mesh_device, "shape")
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    rope = Gemma4TextRotaryEmbedding(hf_config)
    x_dummy = torch.randn(1, max_seq_len, hf_config.hidden_size)
    pos_ids = torch.arange(max_seq_len).unsqueeze(0)

    caches_4d = {}
    caches_2d = {}
    for layer_type in set(hf_config.layer_types):
        cos, sin = rope(x_dummy, pos_ids, layer_type=layer_type)
        # cos, sin: [1, max_seq_len, head_dim]

        # 4D for prefill: [1, 1, max_seq_len, head_dim]
        cos_4d = ttnn.from_torch(
            cos.unsqueeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )
        sin_4d = ttnn.from_torch(
            sin.unsqueeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )
        caches_4d[layer_type] = (cos_4d, sin_4d)

        # 2D for decode embedding lookup: [max_seq_len, head_dim]
        cos_2d = ttnn.from_torch(
            cos.squeeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )
        sin_2d = ttnn.from_torch(
            sin.squeeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )
        caches_2d[layer_type] = (cos_2d, sin_2d)

    return caches_4d, caches_2d


class Gemma4Model:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=None,
        max_seq_len=131072,
        max_local_batch_size=1,
        num_layers=None,
        paged_attention_config=None,
        create_kv_cache=True,
        # Legacy parameters — ignored
        transformation_mats=None,
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.mesh_config = mesh_config
        self.hidden_size = hf_config.hidden_size
        self.vocab_size = hf_config.vocab_size
        self.final_logit_softcapping = hf_config.final_logit_softcapping
        self.embed_scale = hf_config.hidden_size**0.5
        self.ccl_manager = ccl_manager
        self.max_seq_len = max_seq_len
        self.hidden_size_per_layer_input = getattr(hf_config, "hidden_size_per_layer_input", 0) or 0
        n_layers = num_layers or hf_config.num_hidden_layers

        # KV sharing map: layers after (full_n_layers - num_kv_shared_layers) share KV
        # from the last non-shared layer of the same type
        full_n_layers = hf_config.num_hidden_layers
        num_kv_shared = getattr(hf_config, "num_kv_shared_layers", 0) or 0
        first_shared_idx = full_n_layers - num_kv_shared
        self.kv_shared_layer_map = {}  # layer_idx -> source_layer_idx
        if num_kv_shared > 0 and first_shared_idx < n_layers:
            prev_layers = hf_config.layer_types[:first_shared_idx]
            for i in range(first_shared_idx, n_layers):
                lt = hf_config.layer_types[i]
                if lt in prev_layers:
                    source = len(prev_layers) - 1 - list(prev_layers)[::-1].index(lt)
                    if source < n_layers:  # Source must be within our layer range
                        self.kv_shared_layer_map[i] = source
            if self.kv_shared_layer_map:
                logger.info(f"KV sharing enabled: {len(self.kv_shared_layer_map)} layers share KV from earlier layers")

        # RoPE caches per layer type (sliding vs global)
        # Needs real HF text config (set by create_tt_model via _hf_text_config)
        hf_text_config = getattr(hf_config, "_hf_text_config", None)
        if hf_text_config is not None:
            self.rope_caches, self.rope_caches_2d = create_rope_caches(mesh_device, hf_text_config, max_seq_len)
        else:
            # Fallback: no automatic RoPE — caller must pass rope_mats explicitly
            self.rope_caches = {}
            self.rope_caches_2d = {}

        # Embedding
        is_mesh = hasattr(mesh_device, "shape")
        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

        if state_dict and "model.language_model.embed_tokens.weight" in state_dict:
            embed_key = "model.language_model.embed_tokens.weight"
        elif state_dict and "model.embed_tokens.weight" in state_dict:
            embed_key = "model.embed_tokens.weight"
        else:
            embed_key = None

        if embed_key and state_dict:
            embed_weight = state_dict[embed_key]
            self.embedding_weight = ttnn.as_tensor(
                embed_weight.unsqueeze(0).unsqueeze(0),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=replicate,
                cache_file_name=get_cache_file_name(tensor_cache_path, "embed_tokens.weight"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # LM head (tied with embeddings)
            lm_head_weight = embed_weight.transpose(0, 1).unsqueeze(0).unsqueeze(0)
            self.lm_head_weight = ttnn.as_tensor(
                lm_head_weight,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=replicate,
                cache_file_name=get_cache_file_name(tensor_cache_path, "lm_head.weight"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.embedding_weight = None
            self.lm_head_weight = None

        # Per-layer input embeddings (E2B/E4B) — kept as CPU torch tensors for computation
        # Also store embedding weight reference for decode per-layer input
        self._embed_weight_cpu = None
        if embed_key and state_dict:
            self._embed_weight_cpu = state_dict[embed_key]
        self.per_layer_input_weights = {}
        if self.hidden_size_per_layer_input and state_dict:
            pli_size = self.hidden_size_per_layer_input
            # Try both key formats
            for prefix in ["model.language_model.", "model."]:
                pli_embed_key = f"{prefix}embed_tokens_per_layer.weight"
                pli_proj_key = f"{prefix}per_layer_model_projection.weight"
                pli_norm_key = f"{prefix}per_layer_projection_norm.weight"
                if pli_embed_key in state_dict:
                    self.per_layer_input_weights = {
                        "embed_tokens_per_layer": state_dict[pli_embed_key],  # [vocab_pli, n_layers * pli_size]
                        "per_layer_model_projection": state_dict[pli_proj_key],  # [n_layers * pli_size, hidden]
                        "per_layer_projection_norm": state_dict[pli_norm_key],  # [pli_size]
                    }
                    self.per_layer_input_scale = 2.0**-0.5
                    self.per_layer_model_projection_scale = hf_config.hidden_size**-0.5
                    self.per_layer_embed_scale = pli_size**0.5

                    # Device-side PLI weights for on-device decode (trace-compatible)
                    # Shard the large embedding on dim=-1 across TP devices to halve per-device DRAM
                    tp = mesh_config.tp if mesh_config else 1
                    pli_col_mapper = mesh_config.column_parallel(mesh_device) if tp > 1 else replicate
                    # Use bfloat8_b for the large PLI embedding to reduce per-device DRAM
                    self.pli_embed_weight_tt = ttnn.as_tensor(
                        state_dict[pli_embed_key],
                        device=mesh_device,
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=pli_col_mapper,
                        cache_file_name=get_cache_file_name(tensor_cache_path, f"pli_embed_tokens_per_layer_tp{tp}"),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    self.pli_tp = tp  # Track TP for all-gather after embedding lookup
                    # Projection weight also sharded on output dim (column-parallel)
                    pli_proj_w = state_dict[pli_proj_key].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
                    self.pli_proj_weight_tt = ttnn.as_tensor(
                        pli_proj_w,
                        device=mesh_device,
                        dtype=dtype,
                        layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=pli_col_mapper,
                        cache_file_name=get_cache_file_name(tensor_cache_path, f"pli_model_projection_tp{tp}"),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    # Norm weight is small — replicate
                    pli_norm_w = state_dict[pli_norm_key].reshape(1, 1, -1, ttnn.TILE_SIZE)
                    self.pli_norm_weight_tt = ttnn.as_tensor(
                        pli_norm_w,
                        device=mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=replicate,
                        cache_file_name=get_cache_file_name(tensor_cache_path, "pli_projection_norm"),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )

                    logger.info(f"Per-layer input embeddings loaded on device (pli_size={pli_size})")
                    break

        # Decoder layers (each creates its own KV cache if requested)
        self.layers = []
        for i in range(n_layers):
            layer = Gemma4DecoderLayer(
                mesh_device=mesh_device,
                hf_config=hf_config,
                state_dict=state_dict,
                layer_idx=i,
                ccl_manager=ccl_manager,
                dtype=dtype,
                tensor_cache_path=tensor_cache_path,
                mesh_config=mesh_config,
                max_seq_len=max_seq_len,
                max_local_batch_size=max_local_batch_size,
            )
            # Create KV cache for non-shared layers only
            # Shared layers will use their source layer's KV cache
            if create_kv_cache and i not in self.kv_shared_layer_map:
                from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache

                attn_cfg = Gemma4AttentionConfig(hf_config, i)
                kv_cache = init_kv_cache(
                    mesh_device=mesh_device,
                    config=attn_cfg,
                    max_batch_size=max_local_batch_size,
                    max_seq_len=max_seq_len,
                    paged_attention_config=paged_attention_config,
                    cache_dtype=ttnn.bfloat16,
                )
                layer.self_attn.kv_cache = kv_cache
            self.layers.append(layer)

        # Extract KV caches for external access (Generator interface)
        # Shared layers point to their source layer's cache
        self.tt_kv_cache = []
        for i, layer in enumerate(self.layers):
            if i in self.kv_shared_layer_map:
                source_idx = self.kv_shared_layer_map[i]
                self.tt_kv_cache.append(self.layers[source_idx].self_attn.kv_cache)
            else:
                self.tt_kv_cache.append(layer.self_attn.kv_cache)

        # Final norm
        if state_dict and "model.language_model.norm.weight" in state_dict:
            norm_state = substate(state_dict, "model.language_model.norm")
        elif state_dict and "model.norm.weight" in state_dict:
            norm_state = substate(state_dict, "model.norm")
        else:
            norm_state = {}

        self.norm = RMSNorm(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=norm_state,
            tensor_cache_path=f"{tensor_cache_path}/final_norm" if tensor_cache_path else None,
            mesh_config=mesh_config,
        )

    def _compute_per_layer_inputs(self, input_ids_torch, embeds_torch):
        """Compute per-layer input embeddings on CPU (E2B/E4B).

        Returns list of [1, seq_len, pli_size] tensors, one per layer, or None.
        Returns None if input_ids_torch or embeds_torch are not provided (e.g. trace mode).
        """
        if not self.hidden_size_per_layer_input or not self.per_layer_input_weights:
            return None
        if input_ids_torch is None or embeds_torch is None:
            return None

        import torch.nn.functional as F

        w = self.per_layer_input_weights
        pli_size = self.hidden_size_per_layer_input
        n_layers = len(self.layers)
        # The per-layer embedding weight has ALL layers baked in
        # Infer full layer count from the weight shape, not the (possibly overridden) config
        embed_w = w["embed_tokens_per_layer"]  # [vocab_pli, full_n_layers * pli_size]
        full_n_layers = embed_w.shape[-1] // pli_size

        # 1. Per-layer token embedding: embed_tokens_per_layer(input_ids)
        pli_embed = F.embedding(input_ids_torch.long(), embed_w) * self.per_layer_embed_scale
        pli_embed = pli_embed.reshape(*input_ids_torch.shape, full_n_layers, pli_size)

        # 2. Projection from main embeddings
        proj_w = w["per_layer_model_projection"]  # [full_n_layers * pli_size, hidden]
        pli_proj = F.linear(embeds_torch.float(), proj_w.float()) * self.per_layer_model_projection_scale
        pli_proj = pli_proj.reshape(*embeds_torch.shape[:-1], full_n_layers, pli_size)

        # 3. Norm the projection
        norm_w = w["per_layer_projection_norm"]  # [pli_size]
        eps = self.hf_config.rms_norm_eps
        pli_proj_f = pli_proj.float()
        var = pli_proj_f.pow(2).mean(-1, keepdim=True)
        pli_proj = (pli_proj_f * torch.rsqrt(var + eps) * norm_w.float()).to(pli_proj.dtype)

        # 4. Combine: (projection + embed) * scale
        per_layer_inputs = (pli_proj + pli_embed.float()) * self.per_layer_input_scale

        # Return as list of per-layer tensors
        return [per_layer_inputs[:, :, i, :].to(torch.bfloat16) for i in range(n_layers)]

    def _compute_pli_device(self, tokens_tt, input_embeds_tt):
        """Compute per-layer input embeddings entirely on device (decode only, trace-compatible).

        Args:
            tokens_tt: [1, 1] uint32 token tensor on device
            input_embeds_tt: [1, 1, 1, hidden_size] embedded input on device

        Returns:
            Tensor [1, 1, n_layers, pli_size] on device, or None if PLI not used.
        """
        if not self.hidden_size_per_layer_input or not hasattr(self, "pli_embed_weight_tt"):
            return None

        pli_size = self.hidden_size_per_layer_input
        n_layers = len(self.layers)
        tp = getattr(self, "pli_tp", 1)

        # Infer full_n_layers from weight shape (accounting for TP sharding)
        local_out_dim = self.pli_embed_weight_tt.shape[-1]
        full_out_dim = local_out_dim * tp
        full_n_layers_pli = full_out_dim // pli_size

        # 1. Per-layer token embedding (column-parallel sharded if TP > 1)
        pli_embed = ttnn.embedding(tokens_tt, self.pli_embed_weight_tt, layout=ttnn.TILE_LAYOUT)
        pli_embed = ttnn.mul(pli_embed, self.per_layer_embed_scale)

        # 2. Projection from main embeddings (column-parallel sharded if TP > 1)
        pli_proj = ttnn.linear(input_embeds_tt, self.pli_proj_weight_tt)
        pli_proj = ttnn.mul(pli_proj, self.per_layer_model_projection_scale)

        # Match shapes: pli_embed is 3D [1, 1, local_out] → 4D
        pli_embed = ttnn.unsqueeze_to_4D(pli_embed)

        # All-gather if TP > 1 to reconstruct full output dim
        if tp > 1:
            pli_embed = ttnn.all_gather(
                pli_embed,
                num_links=1,
                dim=-1,
                topology=ttnn.Topology.Ring,
                cluster_axis=self.mesh_config.tp_axis,
            )
            pli_proj = ttnn.all_gather(
                pli_proj,
                num_links=1,
                dim=-1,
                topology=ttnn.Topology.Ring,
                cluster_axis=self.mesh_config.tp_axis,
            )

        # 3. Reshape to [1, 1, full_n_layers, pli_size] for per-vector RMSNorm
        pli_proj = ttnn.reshape(pli_proj, (1, 1, full_n_layers_pli, pli_size))
        pli_embed = ttnn.reshape(pli_embed, (1, 1, full_n_layers_pli, pli_size))

        # 4. RMSNorm on projection (norms last dim = pli_size)
        pli_proj = ttnn.rms_norm(pli_proj, weight=self.pli_norm_weight_tt, epsilon=self.hf_config.rms_norm_eps)

        # 5. Combine: (projection + embed) * scale
        combined = ttnn.add(pli_proj, pli_embed)
        pli_embed.deallocate(True)
        pli_proj.deallocate(True)
        combined = ttnn.mul(combined, self.per_layer_input_scale)

        # Slice to n_layers if model uses fewer than full
        if n_layers < full_n_layers_pli:
            combined = combined[:, :, :n_layers, :]

        return combined  # [1, 1, n_layers, pli_size]

    def _get_rope_mats(self, layer_idx, seq_len=None, for_decode=False):
        """Get (cos, sin) for a given layer.

        Args:
            seq_len: If set, slice 4D cache to this length (prefill).
            for_decode: If True, return 2D caches [max_seq_len, head_dim] for embedding lookup.
        """
        layer_type = self.hf_config.layer_types[layer_idx]
        if for_decode:
            return self.rope_caches_2d[layer_type]
        cos, sin = self.rope_caches[layer_type]
        if seq_len is not None:
            cos = cos[:, :, :seq_len, :]
            sin = sin[:, :, :seq_len, :]
        return (cos, sin)

    def __call__(
        self,
        hidden_states,
        rope_mats=None,
        position_idx=None,
        page_table=None,
        kv_caches=None,
        is_decode=True,
        token_index=None,
        input_ids_torch=None,
        embeds_torch=None,
        pli_device_tensors=None,
        position_idx_cache=None,
        tokens_tt=None,
    ):
        """
        Forward pass through decoder layers + final norm + lm_head + softcapping.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device (post-embedding)
            rope_mats: (cos, sin) override, or dict {layer_type: (cos, sin)} for pre-sliced decode
            position_idx: decode position tensor ([1,32] uint32 for embedding RoPE, or [1] int32 legacy)
            page_table: paged attention table
            kv_caches: list of [k, v] per layer, or None (uses self.tt_kv_cache)
            is_decode: True for decode, False for prefill
            token_index: int for decode RoPE slicing (None when using embedding-based RoPE)
            input_ids_torch: CPU tensor of input_ids for per-layer input computation (E2B)
            embeds_torch: CPU tensor of embeddings for per-layer input projection (E2B)
            pli_device_tensors: optional list of pre-computed PLI device tensors (trace mode)
            position_idx_cache: optional [batch] int32 tensor for KV cache update (when position_idx is uint32)
        """
        seq_len = hidden_states.shape[2]
        caches = kv_caches or self.tt_kv_cache

        # Compute per-layer inputs (E2B/E4B)
        # Decode: compute on device via ttnn.embedding (trace-compatible)
        # Prefill: compute on CPU (multi-token PLI)
        pli_combined_tt = None
        per_layer_inputs = None
        if pli_device_tensors is not None:
            pass  # Pre-computed device tensors provided externally
        elif is_decode and tokens_tt is not None and self.hidden_size_per_layer_input:
            pli_combined_tt = self._compute_pli_device(tokens_tt, hidden_states)
        else:
            per_layer_inputs = self._compute_per_layer_inputs(input_ids_torch, embeds_torch)

        is_mesh = hasattr(self.mesh_device, "shape")

        # Determine which layers are KV sources (their K/V will be shared)
        kv_source_indices = set(self.kv_shared_layer_map.values()) if not is_decode else set()
        # Store K/V from source layers for sharing during prefill
        shared_kv_store = {}  # source_layer_idx -> (tt_k, tt_v) kept alive on device

        for i, layer in enumerate(self.layers):
            # Per-layer RoPE: sliding and global layers have different cos/sin
            if rope_mats is not None:
                if isinstance(rope_mats, dict):
                    # Dict mapping layer_type -> (cos, sin) — pre-sliced for trace decode
                    layer_type = self.hf_config.layer_types[i]
                    layer_rope = rope_mats[layer_type]
                else:
                    layer_rope = rope_mats  # Single (cos, sin) override (backward compat / tests)
            elif is_decode:
                # Decode: return 2D caches for on-device embedding lookup
                layer_rope = self._get_rope_mats(i, for_decode=True)
            else:
                layer_rope = self._get_rope_mats(i, seq_len=seq_len)

            # Convert per-layer input to device tensor if available
            pli_tt = None
            if pli_combined_tt is not None:
                # On-device decode: slice layer i from combined [1, 1, n_layers, pli_size]
                pli_tt = pli_combined_tt[:, :, i : i + 1, :]
            elif pli_device_tensors is not None and i < len(pli_device_tensors):
                # Pre-computed device tensors (legacy trace mode)
                pli_tt = pli_device_tensors[i]
            elif per_layer_inputs is not None and i < len(per_layer_inputs):
                pli_4d = per_layer_inputs[i].unsqueeze(0).unsqueeze(0)  # [1, 1, seq, pli_size]
                pli_tt = ttnn.from_torch(
                    pli_4d,
                    device=self.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh else None,
                )

            kv_cache = caches[i] if caches else None

            # KV sharing: determine if this layer shares or provides K/V
            shared_kv = None
            keep_kv = False
            is_kv_shared = i in self.kv_shared_layer_map
            if not is_decode and is_kv_shared:
                source_idx = self.kv_shared_layer_map[i]
                shared_kv = shared_kv_store.get(source_idx)
            elif not is_decode and i in kv_source_indices:
                keep_kv = True

            hidden_states = layer(
                hidden_states,
                rope_mats=layer_rope,
                position_idx=position_idx,
                page_table=page_table,
                kv_cache=kv_cache,
                is_decode=is_decode,
                token_index=token_index,
                per_layer_input=pli_tt,
                shared_kv=shared_kv,
                keep_kv=keep_kv,
                is_kv_shared=is_kv_shared,
                position_idx_cache=position_idx_cache,
            )

            # For KV source layers during prefill, capture the K/V from the attention
            # The K/V are kept alive on device (not deallocated) when keep_kv=True
            if keep_kv and layer.self_attn._last_kv is not None:
                shared_kv_store[i] = layer.self_attn._last_kv

        # Deallocate any stored shared K/V tensors
        for kv_pair in shared_kv_store.values():
            if kv_pair is not None:
                kv_pair[0].deallocate(True)
                kv_pair[1].deallocate(True)

        # Final norm
        hidden_states = self.norm.forward(hidden_states)

        # LM head
        if self.lm_head_weight is not None:
            logits = ttnn.linear(hidden_states, self.lm_head_weight)
            hidden_states.deallocate(True)
        else:
            logits = hidden_states

        # Softcapping: tanh(logits / cap) * cap
        if self.final_logit_softcapping and self.final_logit_softcapping > 0:
            cap = self.final_logit_softcapping
            logits = ttnn.mul(logits, 1.0 / cap)
            logits = ttnn.tanh(logits)
            logits = ttnn.mul(logits, cap)

        return logits

    def embed_tokens(self, tokens):
        """Embed input tokens and scale by sqrt(hidden_size)."""
        if self.embedding_weight is None:
            raise RuntimeError("Embedding weights not loaded")
        embeds = ttnn.embedding(tokens, self.embedding_weight, dtype=ttnn.bfloat16)
        embeds = ttnn.mul(embeds, self.embed_scale)
        return embeds

    # ── Generator-compatible interface ────────────────────────────────────

    def ttnn_prefill_forward(
        self,
        x,
        user_id=0,
        page_table=None,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
        input_ids_torch=None,
        embeds_torch=None,
    ):
        """Prefill forward — matches tt_transformers Generator interface."""
        seq_len = x.shape[-2]
        logits = self(
            hidden_states=x,
            position_idx=None,
            page_table=page_table,
            kv_caches=kv_cache,
            is_decode=False,
            input_ids_torch=input_ids_torch,
            embeds_torch=embeds_torch,
        )

        # Extract last token tile for next-token prediction
        if get_last_token != -1:
            logits = ttnn.slice(
                logits,
                (0, 0, get_last_token, 0),
                (1, 1, get_last_token + 32, logits.shape[-1]),
            )

        return logits

    def ttnn_decode_forward(
        self,
        tokens,
        current_pos,
        rot_mat_idxs=None,
        page_table=None,
        kv_cache=None,
        input_ids_torch=None,
        embeds_torch=None,
        rope_mats=None,
        pli_device_tensors=None,
        position_idx_cache=None,
    ):
        """Decode forward — matches tt_transformers Generator interface.

        Args:
            rope_mats: Optional dict {layer_type: (cos_tt, sin_tt)} of pre-sliced RoPE device tensors.
                       When provided (trace mode), token_index=0 is used. When None, token_index is
                       extracted from current_pos and full RoPE caches are used.
            pli_device_tensors: Optional list of pre-computed PLI device tensors for trace mode.
            position_idx_cache: Optional [batch] int32 tensor for KV cache/SDPA (when current_pos is uint32).
        """
        input_embeds = self.embed_tokens(tokens)
        input_embeds = ttnn.reshape(input_embeds, (1, 1, tokens.shape[-1], self.hidden_size))
        input_embeds = ttnn.to_layout(input_embeds, ttnn.TILE_LAYOUT)

        # Compute embeds_torch for per-layer input if needed
        if embeds_torch is None and self.hidden_size_per_layer_input and input_ids_torch is not None:
            if self._embed_weight_cpu is not None:
                embeds_torch = (
                    torch.nn.functional.embedding(input_ids_torch.long(), self._embed_weight_cpu).float()
                    * self.embed_scale
                )

        # Get position as int for token_index (only needed for legacy 4D RoPE path)
        # When using 2D rope caches (default decode path), position is handled via
        # on-device embedding lookup and token_index is unused.
        if rope_mats is not None:
            token_index = 0  # Pre-sliced rope from caller
        elif self.rope_caches_2d:
            token_index = None  # On-device embedding lookup handles position
        elif isinstance(current_pos, ttnn.Tensor):
            is_mesh = hasattr(self.mesh_device, "shape")
            pos_cpu = ttnn.get_device_tensors(current_pos)[0] if is_mesh else current_pos
            token_index = int(ttnn.to_torch(pos_cpu).item())
        else:
            token_index = int(current_pos)

        logits = self(
            hidden_states=input_embeds,
            position_idx=current_pos,
            page_table=page_table,
            kv_caches=kv_cache,
            input_ids_torch=input_ids_torch,
            embeds_torch=embeds_torch,
            is_decode=True,
            token_index=token_index,
            rope_mats=rope_mats,
            pli_device_tensors=pli_device_tensors,
            position_idx_cache=position_idx_cache,
            tokens_tt=tokens,
        )

        return logits, None
