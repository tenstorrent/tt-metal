# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeRotaryEmbedding

import ttnn
from models.demos.glm_45.config import MeshConfig
from models.demos.glm_45.utils.general_utils import get_cache_file_name, get_decode_mask
from models.demos.glm_45.utils.substate import substate
from models.tt_transformers.tt.common import copy_host_to_device

from .layer import DecoderLayer
from .rms_norm import RMSNorm
from .rope import ApplyRotaryPosEmb


class Model:
    """
    GLM-4.5 TTNN Model Implementation

    Implements GLM-4.5 using TTNN tensors and operations, supporting both prefill and
    decode passes. The model includes MoE (Mixture of Experts) layers and uses GLM’s
    rotary embeddings and masking. It is compatible with the tt_transformers generator.

    Key Features:
    - MoE (Mixture of Experts) architecture with router and experts
    - Optional sliding window attention depending on config
    - Paged attention support for memory efficiency
    - Compatible with tt_transformers generator interface
    """

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        paged_attention_config=None,
        mesh_config=None,
        create_kv_cache=True,
    ):
        """
        Initialize GLM-4.5 model

        Args:
            mesh_device: TTNN mesh device for computation
            hf_config: HuggingFace model configuration
            state_dict: Model weights dictionary
            ccl_manager: Collective communication manager
            dtype: Data type for tensors (default: bfloat16)
            tensor_cache_path: Path for tensor caching
            paged_attention_config: Configuration for paged attention
            mesh_config: Mesh configuration for parallelization
        """
        self.mesh_device = mesh_device
        self.vocab_size = hf_config.vocab_size
        self.hf_config = hf_config

        # Provide safe defaults for optional config fields
        if not hasattr(self.hf_config, "sliding_window"):
            # Default to no sliding window; full causal attention only
            setattr(self.hf_config, "sliding_window", 0)

        # Use MeshConfig for clean parallelization
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])

        # Initialize rope embeddings for generator compatibility (GLM-4.5)
        self.rope_embeddings = Glm4MoeRotaryEmbedding(hf_config)
        self.apply_rope = ApplyRotaryPosEmb(hf_config)
        embedding_weight = substate(state_dict, "model.embed_tokens")["weight"]
        embedding_weight = embedding_weight.unsqueeze(0).unsqueeze(0)
        self.embedding_weight = ttnn.as_tensor(
            embedding_weight,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "model.embed_tokens.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.layers = [
            DecoderLayer(
                mesh_device,
                hf_config,
                substate(state_dict, f"model.layers.{layer_idx}"),
                layer_idx,
                ccl_manager,
                dtype=dtype,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, f"model.layers.{layer_idx}"),
                paged_attention_config=paged_attention_config,
                mesh_config=self.mesh_config,
                create_kv_cache=create_kv_cache,
            )
            for layer_idx in range(hf_config.num_hidden_layers)
        ]
        self.norm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "model.norm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "norm"),
            mesh_config=self.mesh_config,
        )
        self.lm_head_weight = ttnn.as_tensor(
            substate(state_dict, "lm_head")["weight"].transpose(0, 1),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "lm_head.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Initialize attention masks and rope embeddings storage for decode
        self.device_decode_sliding_mask = get_decode_mask(0, self.hf_config.sliding_window)
        self._current_rope_mats = None
        self.rotary_setup_decode = None
        # Static decode batch matches attention's static batch
        try:
            self.decode_batch = self.layers[0].self_attn.decode_batch
        except Exception:
            self.decode_batch = 2

    @classmethod
    def create_transformer_compatible(
        cls,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        attention_class=None,
        rope_setup_class=None,
        mesh_config=None,
        create_kv_cache=True,
    ):
        """Constructor compatible with tt_transformers.Transformer interface"""
        # Create a CCL manager for GLM-4.5
        from models.demos.glm_45.tt.ccl import CCLManager

        ccl_manager = CCLManager(mesh_device)

        # Create instance using direct initialization
        instance = cls.__new__(cls)
        instance.__init__(
            mesh_device=mesh_device,
            hf_config=args.hf_config,
            state_dict=state_dict,
            ccl_manager=ccl_manager,
            dtype=dtype,
            tensor_cache_path=weight_cache_path,
            paged_attention_config=paged_attention_config,
            mesh_config=mesh_config,
            create_kv_cache=create_kv_cache,
        )

        # Add tt_transformers compatible attributes
        instance.args = args
        instance.vocab_size = args.vocab_size
        instance.n_layers = args.n_layers
        instance.dtype = dtype

        return instance

    def __call__(
        self,
        input_ids,
        attention_masks,
        position_embeddings,
        position_idx=None,
    ):
        input_embeds = ttnn.embedding(input_ids, self.embedding_weight, layout=ttnn.TILE_LAYOUT)

        hidden_states = input_embeds
        for decoder_layer in self.layers:
            mask = attention_masks[decoder_layer.attention_type]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=mask,
                position_embeddings=position_embeddings,
                position_idx=position_idx,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = ttnn.matmul(hidden_states, self.lm_head_weight)
        return hidden_states

    def _create_rope_embeddings(self, seq_len_or_pos):
        """Create rope embeddings for prefill (seq) or decode (pos), with static shapes for decode.

        - Prefill: returns DRAM TILE tensors for apply_rope path
        - Decode: returns HEIGHT_SHARDED tensors across static decode batch, matching attention memconfigs
        """
        # Prefill path (seq_len > 1): keep original behavior
        if isinstance(seq_len_or_pos, int) and seq_len_or_pos > 1:
            position_ids = torch.arange(seq_len_or_pos).unsqueeze(0)
            rope_temp_tensor = torch.randn(1)
            cos, sin = self.rope_embeddings(rope_temp_tensor, position_ids)
            tt_cos = ttnn.from_torch(
                cos.unsqueeze(-2), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )
            tt_sin = ttnn.from_torch(
                sin.unsqueeze(-2), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )
            return (self.apply_rope, tt_cos, tt_sin)

        # Decode path (single position): return DRAM TILE cos/sin compatible with functional RoPE
        pos_val = self._to_scalar_pos(seq_len_or_pos)
        position_ids = torch.tensor([pos_val]).unsqueeze(0)
        rope_temp_tensor = torch.randn(1)
        cos, sin = self.rope_embeddings(rope_temp_tensor, position_ids)
        tt_cos = ttnn.from_torch(
            cos.unsqueeze(-2), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        tt_sin = ttnn.from_torch(
            sin.unsqueeze(-2), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        return (self.apply_rope, tt_cos, tt_sin)

    def ttnn_decode_forward(
        self, tokens, current_pos, rot_mat_idxs=None, page_table=None, kv_cache=None, argmax_on_device=False
    ):
        """Decode forward pass - processes single tokens"""
        # Embed tokens
        input_embeds = ttnn.embedding(tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT)

        # Ensure proper shape for decoder layers: [B, 1, H]
        # Preserve batch B from tokens/current_pos; avoid collapsing to batch 1
        if len(input_embeds.shape) == 4:
            # Expected [1, 1, B, H]; move B to leading dim and drop the extra singleton
            hs = ttnn.transpose(input_embeds, 0, 2)  # [B, 1, 1, H]
            hs = ttnn.squeeze(hs, dim=2)  # [B, 1, H]
            hidden_states = hs
        else:
            # If embedding returns [B, H], add seq_len dim -> [B, 1, H]
            if len(input_embeds.shape) == 2:
                hidden_states = ttnn.unsqueeze(input_embeds, 1)
            else:
                hidden_states = input_embeds

        # Inputs are pre-padded on host to static decode batch size

        # Use pre-prepared rope embeddings and attention masks
        rope_mats = self._current_rope_mats
        attention_masks = self.device_decode_sliding_mask

        # Process through decoder layers
        from loguru import logger

        try:
            logger.info(
                f"GLM decode: tokens_shape={tokens.shape} hidden={hidden_states.shape} pos={getattr(current_pos,'shape',None)} page_table_shape={getattr(page_table,'shape',None)}"
            )
        except Exception:
            pass
        for i, decoder_layer in enumerate(self.layers):
            try:
                logger.info(f"GLM decode: layer {i} start: hidden={hidden_states.shape}")
            except Exception:
                pass
            layer_mask = attention_masks[decoder_layer.attention_type]
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None
            try:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_embeddings=rope_mats,
                    position_idx=current_pos,
                    page_table=page_table,
                    kv_cache=layer_kv_cache,
                )
            except Exception as e:
                try:
                    logger.exception(f"GLM decode: failure in layer {i}")
                except Exception:
                    pass
                raise
            try:
                logger.info(f"GLM decode: layer {i} done: hidden={hidden_states.shape}")
            except Exception:
                pass

        # Final norm and lm_head
        hidden_states = self.norm(hidden_states)
        logits = ttnn.matmul(hidden_states, self.lm_head_weight)
        return logits

    def ttnn_prefill_forward(
        self,
        x,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
    ):
        """Prefill forward pass - processes full sequences"""
        # Ensure proper shape for decoder layers
        if len(x.shape) == 4:
            hidden_states = ttnn.squeeze(x, dim=1)
        else:
            hidden_states = x

        # Use provided rotation matrices or create new ones
        seq_len = hidden_states.shape[-2]
        if rot_mats_global is not None:
            rope_mats = rot_mats_global
        else:
            rope_mats = self._create_rope_embeddings(seq_len)

        # Create attention masks
        mask = torch.triu(torch.full((1, 1, seq_len, seq_len), -float("inf")), diagonal=1)
        sliding_mask = mask + torch.tril(
            torch.full((1, 1, seq_len, seq_len), -float("inf")),
            diagonal=-self.hf_config.sliding_window,
        )

        tt_mask = ttnn.from_torch(mask, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_sliding_mask = ttnn.from_torch(
            sliding_mask, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        attention_masks = {"full_attention": tt_mask, "sliding_attention": tt_sliding_mask}

        # Process through decoder layers
        for i, decoder_layer in enumerate(self.layers):
            layer_mask = attention_masks[decoder_layer.attention_type]
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_embeddings=rope_mats,
                position_idx=None,
                page_table=page_table,
                kv_cache=layer_kv_cache,
            )

        # Handle last token slicing for efficiency
        if get_last_token != -1:
            if len(hidden_states.shape) == 3:
                hidden_states = ttnn.unsqueeze(hidden_states, dim=1)
            hidden_states = ttnn.slice(
                hidden_states, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, hidden_states.shape[-1])
            )
            if len(hidden_states.shape) == 4 and hidden_states.shape[1] == 1:
                hidden_states = ttnn.squeeze(hidden_states, dim=1)

        # Final norm and lm_head
        hidden_states = self.norm(hidden_states)
        logits = ttnn.matmul(hidden_states, self.lm_head_weight)
        return logits

    def prepare_inputs_decode(self, tokens, current_pos, page_table=None):
        """
        Prepare inputs for decode mode - matches tt_transformers interface (4 values)
        """
        host_inputs = self.prepare_decode_inputs_host(tokens, current_pos, page_table)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
        self.update_attention_masks(current_pos)
        # Return 4 values to match tt_transformers interface:
        # tokens, current_pos, rope_idxs, page_table
        return (
            device_inputs[0],  # tokens
            device_inputs[1],  # current_pos
            device_inputs[2],  # rope_idxs (list of cos, sin)
            device_inputs[3],  # page_table
        )

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """Prepare decode inputs on host before transferring to device"""
        # Convert tokens to shape [B] and pad to static decode batch size
        if tokens.dim() == 2 and tokens.shape[-1] == 1:
            tokens = tokens.squeeze(-1)
        tokens = tokens.contiguous().view(-1)  # [B]
        B = int(tokens.shape[0])
        if B < self.decode_batch:
            pad = torch.zeros(self.decode_batch - B, dtype=tokens.dtype)
            tokens = torch.cat([tokens, pad], dim=0)
        tokens = ttnn.from_torch(tokens, device=None, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

        # Prepare current position and pad to static batch size
        pos = current_pos.reshape(-1)
        if pos.shape[0] < self.decode_batch:
            # Repeat last index to fill
            fill_val = pos[-1] if pos.numel() > 0 else torch.tensor(0, dtype=pos.dtype)
            pad = fill_val.repeat(self.decode_batch - pos.shape[0])
            pos = torch.cat([pos, pad], dim=0)
        current_pos_tt = ttnn.from_torch(pos, device=None, dtype=ttnn.int32)

        # Prepare page table if provided and pad to static batch size
        if page_table is not None:
            pt = page_table
            if pt.dim() >= 1 and pt.shape[0] < self.decode_batch:
                last = pt[pt.shape[0] - 1 : pt.shape[0]]
                pad = last.repeat(self.decode_batch - pt.shape[0], *([1] * (pt.dim() - 1)))
                pt = torch.cat([pt, pad], dim=0)
            page_table = ttnn.from_torch(pt, device=None, dtype=ttnn.int32)

        # Create rope index tensor placeholder (not used in GLM attention decode)
        rope_idxs = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        return tokens, current_pos_tt, rope_idxs, page_table

    def update_attention_masks(self, current_pos):
        """Update sliding window attention mask and RoPE position for decode mode (static batch)."""
        # Create DRAM TILE RoPE matrices for current positions (functional RoPE expects DRAM TILE)
        pos_scalar = self._to_scalar_pos(current_pos)
        position_ids = torch.tensor([pos_scalar]).unsqueeze(0)
        rope_temp_tensor = torch.randn(1)
        cos, sin = self.rope_embeddings(rope_temp_tensor, position_ids)
        tt_cos = ttnn.from_torch(
            cos.unsqueeze(-2), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        tt_sin = ttnn.from_torch(
            sin.unsqueeze(-2), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        self._current_rope_mats = (self.apply_rope, tt_cos, tt_sin)

        pos_idx = self._to_scalar_pos(current_pos)
        sliding_mask = get_decode_mask(pos_idx, self.hf_config.sliding_window)
        sliding_mask = sliding_mask.repeat(
            1, self.mesh_config.shard_size(self.hf_config.num_attention_heads), 1, 1
        ).transpose(1, 2)

        tt_sliding_mask = ttnn.from_torch(
            sliding_mask, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=self.mesh_device
        )
        attention_masks = {"full_attention": None, "sliding_attention": tt_sliding_mask}
        self.device_decode_sliding_mask = attention_masks

    def prepare_inputs_prefill(self, tokens, start_pos=0, page_table=None, chunk_page_table=None):
        """Prepare inputs for prefill mode"""
        # Embed the tokens
        if tokens.dim() == 2:
            tokens = tokens.reshape(1, 1, 1, -1)

        tokens = ttnn.from_torch(tokens, device=self.mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        tokens_embd = ttnn.embedding(tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT)

        # Ensure proper 4D shape
        if len(tokens_embd.shape) == 3:
            tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)

        # Prepare rotation matrices
        seq_len = tokens_embd.shape[-2] if len(tokens_embd.shape) == 4 else tokens_embd.shape[-2]
        rope_mats = self._create_rope_embeddings(seq_len)
        rot_mats_global = rope_mats
        rot_mats_local = None

        # Prepare page tables if provided
        tt_page_table = None
        tt_chunk_page_table = None
        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table, device=self.mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table, device=self.mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )

        return tokens_embd, rot_mats_global, rot_mats_local, tt_page_table, tt_chunk_page_table

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False):
        """Process decode output and convert to torch tensors"""
        concat_out = self.concat_device_output(tt_out)

        if is_tokens:
            return concat_out[:B, 0]  # [batch_size]

        torch_out = concat_out[:, 0, : self.vocab_size]  # [batch, vocab_size]
        return torch_out.unsqueeze(1).view(B, S, -1)

    def concat_device_output(self, tt_out):
        """Convert multi-device tensor to torch tensor"""
        tt_output_tensor = ttnn.get_device_tensors(tt_out)[0]
        tt_output_tensor = tt_output_tensor.cpu(blocking=True, cq_id=0)
        return ttnn.to_torch(tt_output_tensor)

    def process_output_prefill(self, tt_out, last_token_idx):
        """Process prefill output and extract last token logits"""
        tt_output_tensor = ttnn.get_device_tensors(tt_out)[0]
        torch_output = ttnn.to_torch(tt_output_tensor)
        result = torch_output[:, last_token_idx, : self.vocab_size]
        return result

    # Internal helpers
    def _to_scalar_pos(self, pos):
        """Safely convert position tensor/number into a single scalar index.

        Handles cases where `pos` is a tensor with multiple elements (e.g.,
        padded batches or tiled vectors). Returns the first non-negative
        position if available; otherwise returns 0.
        """
        # Fast path for Python ints
        if isinstance(pos, int):
            return int(pos)

        # Torch tensor or numpy-like with .item
        if hasattr(pos, "numel"):
            try:
                n = int(pos.numel())
            except Exception:
                n = 0
            if n == 1:
                return int(pos.item())
            if n > 1:
                flat = pos.reshape(-1)
                # Prefer first non-negative value (skip paddings like -1)
                try:
                    non_neg = flat[flat >= 0]
                    if non_neg.numel() > 0:
                        return int(non_neg[0].item())
                except Exception:
                    pass
                # Fallback to the first element
                try:
                    return int(flat[0].item())
                except Exception:
                    return 0

        # Generic fallback: attempt .item(), else cast to int, else 0
        try:
            return int(pos.item())
        except Exception:
            try:
                return int(pos)
            except Exception:
                return 0
