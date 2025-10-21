# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

import ttnn
from models.demos.gpt_oss.config import MeshConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name, get_decode_mask
from models.demos.gpt_oss.utils.substate import substate
from models.tt_transformers.tt.common import copy_host_to_device

from .layer import DecoderLayer
from .rms_norm import RMSNorm
from .rope import ApplyRotaryPosEmb


class Model:
    """
    GPT-OSS TTNN Model Implementation

    This class implements the GPT-OSS model using TTNN tensors and operations.
    It supports both prefill and decode modes with sliding window attention.

    Key Features:
    - MoE (Mixture of Experts) architecture with router and experts
    - Sliding window attention for efficient long sequences
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
        Initialize GPT-OSS model

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

        # Use MeshConfig for clean parallelization
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])

        # Initialize rope embeddings for generator compatibility
        self.rope_embeddings = GptOssRotaryEmbedding(hf_config)
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
        sliding_mask = get_decode_mask(0, self.hf_config.sliding_window)
        sliding_mask = sliding_mask.repeat(
            1, self.mesh_config.shard_size(self.hf_config.num_attention_heads), 1, 1
        ).transpose(1, 2)

        tt_sliding_mask = ttnn.from_torch(
            sliding_mask, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=self.mesh_device
        )
        self.device_decode_sliding_mask = {"full_attention": None, "sliding_attention": tt_sliding_mask}
        self._current_rope_mats = self._create_rope_embeddings(0, self.mesh_device)

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
        # Create a dummy CCL manager for GPT-OSS
        from models.demos.gpt_oss.tt.ccl import CCLManager

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

    def _create_rope_embeddings(self, seq_len_or_pos, device):
        """Create rope embeddings for sequence length or specific position"""
        if isinstance(seq_len_or_pos, int) and seq_len_or_pos > 1:
            # Sequence mode - create for full sequence
            position_ids = torch.arange(seq_len_or_pos).unsqueeze(0)
        else:
            # Position mode - create for specific position
            pos_val = seq_len_or_pos.item() if hasattr(seq_len_or_pos, "item") else seq_len_or_pos
            position_ids = torch.tensor([pos_val]).unsqueeze(0)

        rope_temp_tensor = torch.randn(1)
        cos, sin = self.rope_embeddings(rope_temp_tensor, position_ids)

        tt_cos = ttnn.from_torch(cos.unsqueeze(-2), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_sin = ttnn.from_torch(sin.unsqueeze(-2), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

        return (self.apply_rope, tt_cos, tt_sin)

    def ttnn_decode_forward(
        self, tokens, current_pos, rot_mat_idxs=None, page_table=None, kv_cache=None, argmax_on_device=False
    ):
        """Decode forward pass - processes single tokens"""
        # Embed tokens
        input_embeds = ttnn.embedding(tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT)

        # Ensure proper shape for decoder layers
        if len(input_embeds.shape) == 4:
            hidden_states = ttnn.squeeze(input_embeds, dim=1)
        else:
            hidden_states = input_embeds

        # Use pre-prepared rope embeddings and attention masks
        rope_mats = self._current_rope_mats
        attention_masks = self.device_decode_sliding_mask

        # Process through decoder layers
        for i, decoder_layer in enumerate(self.layers):
            layer_mask = attention_masks[decoder_layer.attention_type]
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_embeddings=rope_mats,
                position_idx=current_pos,
                page_table=page_table,
                kv_cache=layer_kv_cache,
            )

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
            rope_mats = self._create_rope_embeddings(seq_len, self.mesh_device)

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
        # Convert tokens to proper format
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        # Convert to TTNN tensors on host
        tokens = ttnn.from_torch(
            tokens.unsqueeze(0).unsqueeze(0),  # Convert to 4D
            device=None,  # Keep on host
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Prepare current position
        current_pos_tt = ttnn.from_torch(current_pos, device=None, dtype=ttnn.int32)

        # Prepare page table if provided
        if page_table is not None:
            page_table = ttnn.from_torch(page_table, device=None, dtype=ttnn.int32)

        # Prepare attention masks
        pos_idx = current_pos.item() if hasattr(current_pos, "item") else current_pos

        # Create rope index tensor
        rope_idxs = ttnn.from_torch(
            torch.tensor([pos_idx], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return tokens, current_pos_tt, rope_idxs, page_table

    def update_attention_masks(self, current_pos):
        """Update sliding window attention mask and RoPE position for decode mode"""
        # Update RoPE for the new position
        updated_current_rope_mats = self._create_rope_embeddings(current_pos, None)
        ttnn.copy_host_to_device_tensor(updated_current_rope_mats[1], self._current_rope_mats[1])
        ttnn.copy_host_to_device_tensor(updated_current_rope_mats[2], self._current_rope_mats[2])

        pos_idx = current_pos.item() if hasattr(current_pos, "item") else current_pos
        sliding_mask = get_decode_mask(pos_idx, self.hf_config.sliding_window)
        sliding_mask = sliding_mask.repeat(
            1, self.mesh_config.shard_size(self.hf_config.num_attention_heads), 1, 1
        ).transpose(1, 2)

        tt_sliding_mask = ttnn.from_torch(sliding_mask, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=None)
        ttnn.copy_host_to_device_tensor(tt_sliding_mask, self.device_decode_sliding_mask["sliding_attention"])

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
        rope_mats = self._create_rope_embeddings(seq_len, self.mesh_device)
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
