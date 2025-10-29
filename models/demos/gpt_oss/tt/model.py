# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.gpt_oss.config import MeshConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss.utils.substate import substate
from models.tt_transformers.tt.common import copy_host_to_device, gather_cos_sin, precompute_freqs
from ttnn import replicate_tensor_to_mesh_mapper

from .layer import DecoderLayer
from .rms_norm import RMSNorm
from .rope import RotarySetup


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
        self.core_grid = mesh_device.compute_with_storage_grid_size()
        self.head_dim = hf_config.head_dim

        # Use MeshConfig for clean parallelization
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])

        # Setup RoPE using tt-transformers RotarySetup (handles cos/sin matrices and transformation matrices)
        # Force datatype to bfloat16 since rotary_embedding_llama requires bfloat16
        max_seq_len = getattr(hf_config, "max_position_embeddings", 131072)
        self.rope_setup = RotarySetup(
            device=mesh_device,
            batch_size=1,
            head_dim=hf_config.head_dim,
            max_seq_len=max_seq_len,
            rope_theta=getattr(hf_config, "rope_theta", 10000.0),
            rope_scaling=None,
            datatype=ttnn.bfloat16,
        )

        # Keep references for compatibility
        self.cos_matrix = self.rope_setup.cos_matrix
        self.sin_matrix = self.rope_setup.sin_matrix
        self.transformation_mats = self.rope_setup.get_both_trans_mats()

        # Keep host versions for prefill mode
        cos_cached_host, sin_cached_host = precompute_freqs(
            dim=hf_config.head_dim,
            end=max_seq_len * 2,
            theta=getattr(hf_config, "rope_theta", 10000.0),
            scale_factor=None,
            orig_context_len=131072,
        )
        self.cos_cached = cos_cached_host
        self.sin_cached = sin_cached_host

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
                transformation_mats=self.transformation_mats,
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
            dtype=ttnn.bfloat8_b,
            cache_file_name=get_cache_file_name(tensor_cache_path, "lm_head.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

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
        """
        Create rope embeddings for sequence length or specific position using Meta format.
        Returns [cos, sin] to match tt-transformers interface.

        For decode mode (single position), cos/sin are HEIGHT_SHARDED to match Q/K/V from nlp_create_qkv_heads_decode.
        For prefill mode (sequence), cos/sin are interleaved.
        """

        position_ids = torch.arange(seq_len_or_pos)

        # Use gather_cos_sin to get cos/sin in Meta format (pairwise duplicated)
        cos, sin = gather_cos_sin(position_ids, self.cos_cached, self.sin_cached)
        # gather_cos_sin returns [1, 1, len(position_ids), head_dim]

        tt_cos = ttnn.from_torch(cos, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_sin = ttnn.from_torch(sin, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        return [tt_cos, tt_sin]

    def get_rot_idxs(self, position_idxs, on_host=False):
        """Delegate to RotarySetup.get_rot_idxs"""
        return self.rope_setup.get_rot_idxs(position_idxs, on_host=on_host)

    def get_rot_mats(self, position_idxs):
        """Delegate to RotarySetup.get_rot_mats"""
        return self.rope_setup.get_rot_mats(position_idxs)

    def ttnn_decode_forward(
        self, tokens, current_pos, rot_mat_idxs=None, page_table=None, kv_cache=None, argmax_on_device=False
    ):
        """
        Decode forward pass - processes single tokens.
        Matches tt-transformers interface where rot_mat_idxs are used for on-device RoPE lookup.
        """
        # Embed tokens
        input_embeds = ttnn.embedding(tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT)

        # Ensure proper shape for decoder layers
        if len(input_embeds.shape) == 4:
            hidden_states = ttnn.squeeze(input_embeds, dim=1)
        else:
            hidden_states = input_embeds

        # Get RoPE embeddings via on-device embedding lookup (matches tt-transformers)
        rope_mats = self.get_rot_mats(rot_mat_idxs)

        # Process through decoder layers
        for i, decoder_layer in enumerate(self.layers):
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=None,
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
            sliding_mask, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b
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
        Prepare inputs for decode mode - matches tt_transformers interface (4 values).
        Returns: tokens, current_pos, rope_idxs, page_table

        Note: rope_idxs are position indices that will be used with get_rot_mats()
        for on-device RoPE embedding lookup.
        """
        host_inputs = self.prepare_decode_inputs_host(tokens, current_pos, page_table)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
        # Return 4 values to match tt_transformers interface:
        # tokens, current_pos, rope_idxs, page_table
        return (
            device_inputs[0],  # tokens
            device_inputs[1],  # current_pos
            device_inputs[2],  # rope_idxs - position indices for embedding lookup
            device_inputs[3],  # page_table
        )

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """
        Prepare decode inputs on host before transferring to device.
        Matches tt-transformers Transformer.prepare_decode_inputs_host (model.py lines 204-252).

        Args:
            tokens: torch.Tensor of shape [batch] with token ids
            current_pos: torch.Tensor of shape [batch] with current positions
            page_table: Optional page table for paged attention

        Returns:
            Tuple of (tokens, current_pos_tt, rope_idxs, page_table) all as ttnn tensors on host
        """
        B = tokens.shape[0]
        if current_pos.dim() == 0:
            current_pos = current_pos.unsqueeze(0)
        assert current_pos.shape[0] == B, "Batch size mismatch"

        # Convert tokens to TTNN format
        tokens = ttnn.from_torch(
            tokens,
            device=None,
            dtype=ttnn.uint32,
            mesh_mapper=replicate_tensor_to_mesh_mapper(self.mesh_device),
        )
        tokens = ttnn.unsqueeze_to_4D(tokens)

        # Ensure position indices are non-negative (matches tt-transformers)
        rot_current_pos = torch.maximum(current_pos, torch.tensor(0, dtype=torch.int64))
        rope_idxs = self.get_rot_idxs(rot_current_pos, on_host=True)

        # Prepare current position tensor
        current_pos_tt = ttnn.from_torch(
            current_pos,
            device=None,
            dtype=ttnn.int32,
            mesh_mapper=replicate_tensor_to_mesh_mapper(self.mesh_device),
        )

        # Prepare page table if provided
        if page_table is not None:
            page_table = ttnn.from_torch(
                page_table,
                device=None,
                dtype=ttnn.int32,
                mesh_mapper=replicate_tensor_to_mesh_mapper(self.mesh_device),
            )

        return tokens, current_pos_tt, rope_idxs, page_table

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
