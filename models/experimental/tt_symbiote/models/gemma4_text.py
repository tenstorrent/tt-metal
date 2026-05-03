# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN Gemma4 Text Model and CausalLM implementations.

TTNNGemma4TextModel replaces Gemma4TextModel to:
- Handle input_ids -> embedding on device (trace-safe)
- Iterate decoder layers without ModuleList slicing
  (HF's self.layers[:N] reconstructs a ModuleList, failing for TTNNModule)
- Keep rotary embeddings and causal masks on host (unchanged)

TTNNGemma4ForCausalLM wraps Gemma4ForConditionalGeneration to:
- Handle lm_head + logit soft-capping entirely on device
- Convert TTNN output to plain torch tensor before returning
- Prevent TorchTTNNTensor dispatch corruption on slice/div/tanh/mul ops
"""

from typing import Optional

import torch
import ttnn
from transformers.modeling_outputs import BaseModelOutputWithPast

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.modules.gemma4_modules import TTNNGemma4LayerStack, TTNNGemma4ScaledEmbedding
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm


class TTNNGemma4TextModel(TTNNModule):
    """Replaces Gemma4TextModel (the language_model inside Gemma4Model).

    Follows the same pattern as TTNNBailingMoeV2Model: stores a reference to
    the original HF model, overrides ``call`` to run the forward pass with
    TTNN-replaced children, and handles the embedding input conversion that
    would otherwise break trace capture.
    """

    @staticmethod
    def from_torch(model):
        new_model = TTNNGemma4TextModel()
        new_model.model = model
        new_model._decode_cache_position = None
        new_model._decode_rope_position_idx = None

        # Bypass tensor wrapping/unwrapping for decoder layers.
        # These sit under the HF Gemma4TextModel (nn.Module), so
        # set_device() would give them _bypass_tensor_wrapping=False.
        # Bypassing is safe: no PyTorch ops touch hidden_states between
        # layer calls, and each layer's forward already works with raw
        # ttnn.Tensor objects.
        for layer in model.layers:
            if isinstance(layer, TTNNModule):
                layer._bypass_tensor_wrapping = True
        ttnn_norm = TTNNDistributedRMSNorm.from_torch(model.norm)
        ttnn_norm._bypass_tensor_wrapping = True
        object.__setattr__(model, "norm", ttnn_norm)

        ttnn_layers = [layer for layer in model.layers if isinstance(layer, TTNNModule)]
        num_layers = model.config.num_hidden_layers
        layer_types = list(model.config.layer_types[:num_layers])
        new_model.layer_stack = TTNNGemma4LayerStack(ttnn_layers[:num_layers], layer_types)
        new_model.layer_stack._bypass_tensor_wrapping = True
        new_model.unique_layer_types = set(layer_types)

        # Ensure embed_tokens is a TTNN module so call() can pass a ttnn.Tensor to it.
        # If Pass 1 already replaced it (test_gemma4.py flow), skip; otherwise install it here.
        if not isinstance(model.embed_tokens, TTNNModule):
            ttnn_embed = TTNNGemma4ScaledEmbedding.from_torch(model.embed_tokens)
            object.__setattr__(model, "embed_tokens", ttnn_embed)
        new_model._ttnn_embed_tokens = model.embed_tokens

        # Save references to the original PyTorch per-layer input modules before
        # any subsequent replacement passes can overwrite them.  The PLI computation
        # runs on CPU in call() and needs the original torch modules.
        pli_size = getattr(model.config, "hidden_size_per_layer_input", 0) or 0
        new_model._pli_enabled = bool(pli_size)
        if new_model._pli_enabled:
            # Save original torch versions of PLI modules.  Pass 1 may have already
            # replaced Gemma4TextScaledWordEmbedding → TTNNGemma4ScaledEmbedding and
            # Gemma4RMSNorm → TTNNDistributedRMSNorm.  In those cases, extract the
            # original torch layer from _fallback_torch_layer so CPU PLI computation
            # still works.
            pli_embed = model.embed_tokens_per_layer
            if isinstance(pli_embed, TTNNModule):
                pli_embed = pli_embed._fallback_torch_layer
            new_model._pli_embed = pli_embed

            # per_layer_model_projection: nn.Linear (not affected by Pass 1)
            pli_proj = model.per_layer_model_projection
            if isinstance(pli_proj, TTNNModule):
                pli_proj = pli_proj._fallback_torch_layer
            new_model._pli_model_projection = pli_proj
            new_model._pli_model_projection_scale = model.per_layer_model_projection_scale

            # per_layer_projection_norm: Gemma4RMSNorm(pli_dim)
            pli_norm = model.per_layer_projection_norm
            if isinstance(pli_norm, TTNNModule):
                pli_norm = pli_norm._fallback_torch_layer
            new_model._pli_projection_norm = pli_norm

            new_model._pli_input_scale = model.per_layer_input_scale
            new_model._pli_num_layers = model.config.num_hidden_layers
            new_model._pli_dim = pli_size

        return new_model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def to_device(self, device):
        super().to_device(device)
        self.layer_stack.to_device(device)
        self._create_rope_caches()
        return self

    def _create_rope_caches(self):
        """Create HF-format cos/sin caches for both prefill (4D) and decode (2D).

        Uses HF's own Gemma4TextRotaryEmbedding for exact numerical match.
        Stores two dicts mapping layer_type -> (cos_tt, sin_tt):
        - _rope_caches_4d: [1, 1, max_seq_len, head_dim] for prefill
        - _rope_caches_2d: [max_seq_len, head_dim] for decode embedding lookup
        """
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

        config = self.model.config
        max_seq_len = min(getattr(config, "max_position_embeddings", 8192), 16384)
        mesh_device = self.device
        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None

        rope = Gemma4TextRotaryEmbedding(config)
        x_dummy = torch.randn(1, max_seq_len, config.hidden_size)
        pos_ids = torch.arange(max_seq_len).unsqueeze(0)

        self._rope_caches_4d = {}
        self._rope_caches_2d = {}
        for layer_type in set(config.layer_types):
            cos, sin = rope(x_dummy, pos_ids, layer_type=layer_type)
            # cos, sin: [1, max_seq_len, head_dim]

            # 4D for prefill: [1, 1, max_seq_len, head_dim]
            cos_4d = ttnn.from_torch(
                cos.unsqueeze(0).to(torch.bfloat16),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
            )
            sin_4d = ttnn.from_torch(
                sin.unsqueeze(0).to(torch.bfloat16),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
            )
            self._rope_caches_4d[layer_type] = (cos_4d, sin_4d)

            # 2D for decode embedding lookup: [max_seq_len, head_dim]
            cos_2d = ttnn.from_torch(
                cos.squeeze(0).to(torch.bfloat16),
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
            )
            sin_2d = ttnn.from_torch(
                sin.squeeze(0).to(torch.bfloat16),
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
            )
            self._rope_caches_2d[layer_type] = (cos_2d, sin_2d)

        # Verify cos/sin width matches expected head_dim for each layer type
        for layer_type in set(config.layer_types):
            cos_w = self._rope_caches_4d[layer_type][0].shape[-1]
            expected = config.global_head_dim if layer_type == "full_attention" else config.head_dim
            assert cos_w == expected, f"cos cache width {cos_w} != expected {expected} for {layer_type}"

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped HF model for HF compatibility.

        HF code may access config, embed_tokens.weight, etc. on the language_model.
        """
        # Check own __dict__ first (set by TTNNModule.__init__ and from_torch)
        try:
            return self.__dict__[name]
        except KeyError:
            pass
        # Delegate to the wrapped HF model
        return getattr(self.__dict__["model"], name)

    def call(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        ttnn_object = self
        self = self.model
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            # Convert input_ids to UINT32 on device for TTNN embedding lookup.
            # This is the host→device transfer that prevents embed_tokens from
            # being @trace_enabled. By doing it here in the model wrapper
            # (outside any trace boundary) we keep the decoder layer traces clean.
            input_ids_tt = ttnn.from_torch(
                input_ids.cpu().to(torch.int32),
                device=ttnn_object.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_object.device),
            )
            inputs_embeds = self.embed_tokens(input_ids_tt)

        # --- Per-layer input (PLI) computation ---
        # Gemma4 31B uses per-layer inputs: a combination of a dedicated per-layer
        # embedding and a projection of the main embeddings, conditioned per layer.
        # This uses the original PyTorch PLI modules (saved in from_torch as _pli_*)
        # on CPU — they are lightweight compared to the decoder layers.  The result
        # is converted to a list of ttnn.Tensors (one per layer) for the on-device
        # decoder layer PLI block.
        if getattr(ttnn_object, "_pli_enabled", False) and input_ids is not None:
            with torch.no_grad():
                cpu_input_ids = input_ids.cpu()

                # Step 1: Per-layer embedding from the dedicated vocab table.
                # embed_tokens_per_layer(input_ids) produces (B, S, num_layers * pli_dim),
                # reshaped to (B, S, num_layers, pli_dim).
                if per_layer_inputs is None:
                    pli_embedded = ttnn_object._pli_embed(cpu_input_ids).reshape(
                        *cpu_input_ids.shape,
                        ttnn_object._pli_num_layers,
                        ttnn_object._pli_dim,
                    )
                else:
                    pli_embedded = per_layer_inputs

                # Step 2: Project main embeddings → per-layer space.
                # Recompute torch inputs_embeds from the original HF embedding to avoid
                # an expensive device-to-host transfer of the TTNN inputs_embeds tensor.
                torch_embed_fn = getattr(self.embed_tokens, "_fallback_torch_layer", None)
                if torch_embed_fn is not None:
                    torch_inputs_embeds = torch_embed_fn(cpu_input_ids)
                else:
                    # Fallback: convert TTNN embeddings back to torch (expensive D2H)
                    torch_inputs_embeds = ttnn.to_torch(
                        inputs_embeds,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(ttnn_object.device, ttnn_object.device.shape, (0, -1)),
                    ).to(torch.float32)

                pli_projection = ttnn_object._pli_model_projection(torch_inputs_embeds)
                pli_projection = pli_projection * ttnn_object._pli_model_projection_scale
                pli_projection = pli_projection.reshape(
                    *torch_inputs_embeds.shape[:-1],
                    ttnn_object._pli_num_layers,
                    ttnn_object._pli_dim,
                )
                pli_projection = ttnn_object._pli_projection_norm(pli_projection)

                # Step 3: Combine projected embeddings with per-layer embeddings.
                per_layer_inputs = (pli_projection + pli_embedded) * ttnn_object._pli_input_scale

            # per_layer_inputs: (B, S, num_layers, pli_dim) torch.Tensor
            # Pre-split into a list of ttnn.Tensors (one per layer) for the layer stack.
            mesh_mapper_pli = (
                ttnn.ReplicateTensorToMesh(ttnn_object.device) if ttnn_object.device.get_num_devices() > 1 else None
            )
            pli_list = []
            for layer_i in range(ttnn_object._pli_num_layers):
                pli_slice = per_layer_inputs[:, :, layer_i, :].contiguous()
                pli_tt = ttnn.from_torch(
                    pli_slice.to(torch.bfloat16),
                    device=ttnn_object.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=mesh_mapper_pli,
                )
                pli_list.append(pli_tt)
            per_layer_inputs = pli_list

        if use_cache and past_key_values is None:
            from transformers import DynamicCache

            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device="cpu") + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # Build causal mask mapping (full_attention + sliding_attention)
        # For decode (seq_len == 1), masks are unused by decoder layers
        # (TTNNGemma4DecoderLayer.forward passes attention_mask=None when
        # is_decode).  Skip creation to avoid unnecessary DRAM allocations.
        if inputs_embeds.shape[1] == 1:
            causal_mask_mapping = {"full_attention": None, "sliding_attention": None}
        elif not isinstance(attention_mask, dict):
            from transformers.models.gemma4.modeling_gemma4 import (
                create_causal_mask,
                create_sliding_window_causal_mask,
            )

            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
        else:
            causal_mask_mapping = attention_mask

        # Pre-convert causal masks to ttnn.Tensor for bypass-enabled decoder layers.
        # With _bypass_tensor_wrapping=True, fast_unwrap_to_device passes torch.Tensor
        # unchanged, but TTNNSDPAAttention needs ttnn.Tensor for on-device SDPA.
        mesh_mapper = (
            ttnn.ReplicateTensorToMesh(ttnn_object.device) if ttnn_object.device.get_num_devices() > 1 else None
        )
        for mask_key, mask_val in causal_mask_mapping.items():
            if isinstance(mask_val, torch.Tensor):
                # Convert HF boolean mask to additive format before TTNN conversion.
                # HF returns bool masks at seq_len >= 1024 (sliding window boundary):
                # True means "attend", False means "mask out".
                # TTNN SDPA expects an additive mask: attend → 0.0, mask out → -inf.
                if mask_val.dtype == torch.bool:
                    additive = torch.zeros_like(mask_val, dtype=torch.bfloat16)
                    additive = additive.masked_fill(~mask_val, float("-inf"))
                    mask_val = additive
                causal_mask_mapping[mask_key] = ttnn.from_torch(
                    mask_val,
                    device=ttnn_object.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=mesh_mapper,
                )

        # Persist mask tensors to prevent GC between generate() calls.
        # The trace captures device addresses of these tensors during capture;
        # if freed before replay, the trace reads from stale/reused addresses.
        if inputs_embeds.shape[1] != 1:
            ttnn_object._persistent_prefill_masks = causal_mask_mapping

        hidden_states = inputs_embeds

        # Pre-convert cache_position to an on-device TTNN tensor (trace-safe).
        # This runs OUTSIDE any trace boundary, so ttnn.from_torch is allowed.
        #
        # IMPORTANT: For decode (single-token steps), we use a PERSISTENT device
        # buffer allocated once and updated in-place via ttnn.copy(). This prevents
        # the trace allocator from aliasing the buffer's device address with trace
        # intermediates. Without this, layer 0's trace replay can overwrite the
        # cache_position buffer, corrupting it for layers 1-59.
        # See PLAN_gemma4_traced_mode_root_cause.md for full analysis.
        cache_position = kwargs.get("cache_position")
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], dtype=torch.int32
            )
        is_decode = inputs_embeds.shape[1] == 1
        if not isinstance(cache_position, ttnn.Tensor):
            cp = cache_position
            if hasattr(cp, "cpu"):
                cp = cp.cpu()
            if isinstance(cp, torch.Tensor):
                cp = cp.to(torch.int32)
            else:
                cp = torch.tensor(cp, dtype=torch.int32)
            mesh_mapper = (
                ttnn.ReplicateTensorToMesh(ttnn_object.device) if ttnn_object.device.get_num_devices() > 1 else None
            )
            if is_decode and ttnn_object._decode_cache_position is not None:
                # Subsequent decode steps: copy new value into persistent buffer.
                cp_temp = ttnn.from_torch(
                    cp,
                    device=ttnn_object.device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=mesh_mapper,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.copy(cp_temp, ttnn_object._decode_cache_position)
                cache_position = ttnn_object._decode_cache_position
            else:
                cache_position = ttnn.from_torch(
                    cp,
                    device=ttnn_object.device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=mesh_mapper,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if is_decode:
                    # First decode step: save as persistent buffer.
                    # This allocation happens BEFORE trace capture, so the
                    # trace allocator knows this address is in use.
                    ttnn_object._decode_cache_position = cache_position
        kwargs["cache_position"] = cache_position

        # Provide HF-format cos/sin caches to attention layers.
        if is_decode:
            # Decode: provide 2D caches for ttnn.embedding lookup
            position_embeddings = {}
            for layer_type in self.unique_layer_types:
                position_embeddings[layer_type] = ttnn_object._rope_caches_2d[layer_type]

            # Create [1, 32] uint32 position tensor for ttnn.embedding lookup.
            # Extract scalar position from the torch cp tensor (still in scope).
            _cp_local2 = locals().get("cp")
            if isinstance(_cp_local2, torch.Tensor):
                cur_pos_int = int(_cp_local2.flatten()[0].item())
            else:
                cur_pos_int = int(locals().get("past_seen_tokens", 0))
            pos_padded = torch.nn.functional.pad(
                torch.tensor([cur_pos_int], dtype=torch.int32).reshape(1, 1),
                (0, 31),
                "constant",
                0,
            )
            mesh_mapper_rope = (
                ttnn.ReplicateTensorToMesh(ttnn_object.device) if ttnn_object.device.get_num_devices() > 1 else None
            )
            if ttnn_object._decode_rope_position_idx is None:
                ttnn_object._decode_rope_position_idx = ttnn.from_torch(
                    pos_padded,
                    device=ttnn_object.device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=mesh_mapper_rope,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                pos_temp = ttnn.from_torch(
                    pos_padded,
                    device=ttnn_object.device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=mesh_mapper_rope,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.copy(pos_temp, ttnn_object._decode_rope_position_idx)
            kwargs["rope_position_idx"] = ttnn_object._decode_rope_position_idx
        else:
            # Prefill: provide 4D caches sliced to seq_len.
            # Cache sliced results as persistent model attributes to prevent GC
            # between trace capture and replay.  The trace bakes in device
            # addresses of these sliced tensors; if freed, replay reads garbage.
            seq_len = inputs_embeds.shape[1]
            if not hasattr(ttnn_object, "_rope_caches_4d_prefill"):
                ttnn_object._rope_caches_4d_prefill = {}
            if seq_len not in ttnn_object._rope_caches_4d_prefill:
                sliced = {}
                for layer_type in self.unique_layer_types:
                    cos_4d, sin_4d = ttnn_object._rope_caches_4d[layer_type]
                    sliced[layer_type] = (
                        cos_4d[:, :, :seq_len, :],
                        sin_4d[:, :, :seq_len, :],
                    )
                ttnn_object._rope_caches_4d_prefill[seq_len] = sliced
            position_embeddings = ttnn_object._rope_caches_4d_prefill[seq_len]

        hidden_states = ttnn_object.layer_stack(
            hidden_states,
            attention_mask=causal_mask_mapping,
            position_embeddings=position_embeddings,
            per_layer_inputs=per_layer_inputs,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=TorchTTNNTensor(hidden_states),
            past_key_values=past_key_values,
        )


class TTNNGemma4ForCausalLM(TTNNModule):
    """Wraps Gemma4ForConditionalGeneration to handle lm_head + soft-capping on device.

    HF's Gemma4ForConditionalGeneration.forward() applies slicing, division, tanh,
    and multiplication on the TorchTTNNTensor returned by the inner model. The
    TorchTTNNTensor dispatcher uses *logical* shape coordinates for ttnn.slice but
    operates on the *physical* per-device tensor, corrupting data.

    This wrapper intercepts forward(), runs lm_head and soft-capping entirely in
    TTNN, then converts to a plain torch.Tensor before returning to HF.

    Usage: instantiate via from_torch(), then monkey-patch the HF model's forward:
        wrapper = TTNNGemma4ForCausalLM.from_torch(model)
        wrapper.to_device(mesh_device)
        model.forward = wrapper.forward
    """

    @classmethod
    def from_torch(cls, hf_model):
        """Create from a Gemma4ForConditionalGeneration instance.

        Must be called AFTER all sub-module replacements (decoder layers, linear,
        TTNNGemma4TextModel) so that hf_model.lm_head is already a
        TTNNLinearIColShardedWRowSharded.
        """
        new = cls()
        new._fallback_torch_layer = hf_model
        new.config = hf_model.config
        # Store references -- sub-modules are already TTNN-replaced by earlier passes
        new.model = hf_model.model  # Gemma4Model (wraps language_model)
        new.lm_head = hf_model.lm_head  # TTNNLinearIColShardedWRowSharded
        text_config = hf_model.config.get_text_config()
        new.vocab_size = text_config.vocab_size
        new.final_logit_softcapping = text_config.final_logit_softcapping
        return new

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
        # Step 1: Call the language_model (TTNNGemma4TextModel) DIRECTLY,
        # bypassing HF's Gemma4Model.forward().
        #
        # HF's Gemma4Model.forward() embeds input_ids via get_input_embeddings()
        # which dispatches through TTNNModule with the default DistributedTensorConfig.
        # That config uses ShardTensor2dMesh(dims=(0,-1)), sharding the seq dimension
        # across 8 devices (112 tokens / 8 = 14 per device) instead of replicating.
        # TTNNGemma4TextModel.call() correctly uses ReplicateTensorToMesh for input_ids.
        outputs = self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        # Step 2: Extract raw TTNN tensor from TorchTTNNTensor wrapper.
        hidden_states = outputs.last_hidden_state
        if hasattr(hidden_states, "to_ttnn"):
            hidden_states_ttnn = hidden_states.to_ttnn
        elif isinstance(hidden_states, ttnn.Tensor):
            hidden_states_ttnn = hidden_states
        else:
            hidden_states_ttnn = hidden_states

        if isinstance(hidden_states_ttnn, ttnn.Tensor):
            # Step 3: Slice to keep only the last logits_to_keep tokens.
            # logits_to_keep=0 means "keep all" (HF default), so skip slicing.
            # Use PHYSICAL shape to avoid the logical/physical mismatch bug.
            phys_shape = [int(s) for s in hidden_states_ttnn.shape]
            if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                seq_len = phys_shape[1]  # seq dim (not sharded across devices)
                start_idx = max(0, seq_len - logits_to_keep)
                hidden_states_ttnn = ttnn.slice(
                    hidden_states_ttnn,
                    [0, start_idx, 0],
                    [phys_shape[0], seq_len, phys_shape[2]],
                )

            # Step 4: Run lm_head on device (TTNNLinearIColShardedWRowSharded).
            logits_ttnn = self.lm_head(hidden_states_ttnn)
            if hasattr(logits_ttnn, "to_ttnn"):
                logits_ttnn = logits_ttnn.to_ttnn

            # Step 5+6: Convert to torch FIRST, then apply softcapping on CPU
            # in float32 for exact tanh precision (ttnn.tanh has ~13% of elements
            # off by 1 ULP from torch.tanh, which can flip argmax on tight logits).
            mesh_device = self.device
            ttnn.synchronize_device(mesh_device)
            logits = ttnn.to_torch(
                logits_ttnn,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_device.shape, (0, -1)),
            )
            logits = logits[:1]
            logits = logits.to(torch.float32)

            if self.final_logit_softcapping is not None:
                cap = self.final_logit_softcapping
                logits = logits / cap
                logits = torch.tanh(logits)
                logits = logits * cap
        else:
            # Torch fallback (CPU mode, no TTNN tensors).
            if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                slice_indices = slice(-logits_to_keep, None)
            else:
                slice_indices = slice(None)
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            if self.final_logit_softcapping is not None:
                logits = logits / self.final_logit_softcapping
                logits = torch.tanh(logits)
                logits = logits * self.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self._fallback_torch_layer.loss_function(logits, labels, self.vocab_size, **kwargs)

        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

    def decode_one_step(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=True,
        logits_to_keep=0,
        **kwargs,
    ):
        """Single greedy decode/prefill step with on-device slice and argmax.

        Returns (token_id_host, token_id_tt, past_key_values).
        """
        outputs = self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if hasattr(hidden_states, "to_ttnn"):
            hidden_states_ttnn = hidden_states.to_ttnn
        elif isinstance(hidden_states, ttnn.Tensor):
            hidden_states_ttnn = hidden_states
        else:
            raise RuntimeError("decode_one_step requires TTNN tensor output from language model")

        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            phys_shape = [int(s) for s in hidden_states_ttnn.shape]
            seq_len = phys_shape[1]
            start_idx = max(0, seq_len - logits_to_keep)
            hidden_states_ttnn = ttnn.slice(
                hidden_states_ttnn,
                [0, start_idx, 0],
                [phys_shape[0], seq_len, phys_shape[2]],
            )

        logits_ttnn = self.lm_head(hidden_states_ttnn)
        if hasattr(logits_ttnn, "to_ttnn"):
            logits_ttnn = logits_ttnn.to_ttnn

        logits_ttnn = ttnn.reshape(logits_ttnn, [1, 1, 1, -1])

        logits_gathered = ttnn.all_gather(
            logits_ttnn,
            dim=3,
            num_links=1,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
        )

        logits_untilized = ttnn.untilize(logits_gathered, use_multicore=True)

        token_id_tt = ttnn.argmax(logits_untilized, dim=3, keepdim=True, use_multicore=True)

        token_id_torch = ttnn.to_torch(ttnn.get_device_tensors(token_id_tt)[0])
        token_id_host = int(token_id_torch.flatten()[0].item())

        ttnn.deallocate(logits_gathered)
        ttnn.deallocate(logits_untilized)

        return token_id_host, token_id_tt, outputs.past_key_values
