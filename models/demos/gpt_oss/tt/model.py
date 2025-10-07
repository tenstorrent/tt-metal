import torch

import ttnn
from models.demos.gpt_oss.moe import MeshConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name, get_decode_mask
from models.experimental.tt_dit.utils.substate import substate
from models.tt_transformers.tt.common import copy_host_to_device

from ..reference.modeling_gpt_oss import GptOssRotaryEmbedding
from .layer import DecoderLayer
from .rms_norm import RMSNorm
from .rope import ApplyRotaryPosEmb


class Model:
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
    ):
        """Original GPT-OSS constructor"""
        self._init_gpt_oss(
            mesh_device,
            hf_config,
            state_dict,
            ccl_manager,
            dtype,
            tensor_cache_path,
            paged_attention_config,
            mesh_config,
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
    ):
        """Constructor compatible with tt_transformers.Transformer interface"""
        # Create a dummy CCL manager for GPT-OSS
        from models.demos.gpt_oss.tt.ccl import CCLManager

        ccl_manager = CCLManager(mesh_device)

        # Create instance using original constructor
        instance = cls.__new__(cls)
        instance._init_gpt_oss(
            mesh_device=mesh_device,
            hf_config=args.hf_config,
            state_dict=state_dict,
            ccl_manager=ccl_manager,
            dtype=dtype,
            tensor_cache_path=weight_cache_path,
            paged_attention_config=paged_attention_config,
            mesh_config=mesh_config,  # Pass MeshConfig through
        )

        # Add tt_transformers compatible attributes
        instance.args = args
        instance.vocab_size = args.vocab_size
        instance.n_layers = args.n_layers
        instance.dtype = dtype

        return instance

    def _init_gpt_oss(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        paged_attention_config=None,
        mesh_config=None,
    ):
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
        self._current_rope_stuff = None

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

    def _create_rope_stuff(self, seq_len):
        """
        Create rope embeddings for the given sequence length
        """
        rope_temp_tensor = torch.randn(1)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        cos, sin = self.rope_embeddings(rope_temp_tensor, position_ids)

        tt_cos = ttnn.from_torch(
            cos.unsqueeze(-2), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        tt_sin = ttnn.from_torch(
            sin.unsqueeze(-2), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

        return (self.apply_rope, tt_cos, tt_sin)

    def _create_rope_for_position(self, current_pos):
        """Create rope embeddings for specific position (for decode) - matches test_demo.py exactly"""
        pos_val = current_pos.item() if hasattr(current_pos, "item") else current_pos
        rope_temp_tensor = torch.randn(1)
        # EXACTLY like original test: torch.tensor([cur_pos]).unsqueeze(0)
        position_ids = torch.tensor([pos_val]).unsqueeze(0)
        cos, sin = self.rope_embeddings(rope_temp_tensor, position_ids)

        tt_cos = ttnn.from_torch(
            cos.unsqueeze(-2), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        tt_sin = ttnn.from_torch(
            sin.unsqueeze(-2), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

        return (self.apply_rope, tt_cos, tt_sin)

    def ttnn_decode_forward(
        self,
        tokens,
        current_pos,
        rot_mat_idxs=None,  # Add this parameter to match generator signature
        page_table=None,
        kv_cache=None,
        argmax_on_device=False,
    ):
        """
        Decode forward pass - processes single tokens
        """
        x = tokens

        # Use the page_table parameter directly since it's passed as a keyword argument
        # No need to extract from inputs since the generator passes it directly

        # For decode mode, we expect single token input
        input_embeds = ttnn.embedding(x, self.embedding_weight, layout=ttnn.TILE_LAYOUT)

        # Ensure the right shape for decoder layers (remove extra dimensions if 4D)
        if len(input_embeds.shape) == 4:
            # Convert from [1, 1, seq_len, hidden_size] to [1, seq_len, hidden_size]
            hidden_states = ttnn.squeeze(input_embeds, dim=1)
        else:
            hidden_states = input_embeds

        # Use pre-prepared rope embeddings (stored in instance during prepare_decode_inputs_host)
        rope_stuff = self._current_rope_stuff

        # Use pre-prepared attention masks (stored in instance during prepare_decode_inputs_host)
        attention_masks = self.device_decode_sliding_mask

        for i, decoder_layer in enumerate(self.layers):
            # Each layer picks its appropriate attention mask based on layer type
            layer_mask = attention_masks[decoder_layer.attention_type]

            # Get layer-specific kv_cache if provided (like tt-transformers)
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_embeddings=rope_stuff,
                position_idx=current_pos,
                page_table=page_table,
                kv_cache=layer_kv_cache,
            )

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
        """
        Prefill forward pass - processes full sequences
        """
        # x is already embedded input tokens from prepare_inputs_prefill
        # Keep track of original shape for slice operation
        is_4d_input = len(x.shape) == 4

        # Ensure the right shape for decoder layers (remove extra dimensions if 4D)
        if is_4d_input:
            # Convert from [1, 1, seq_len, hidden_size] to [1, seq_len, hidden_size]
            hidden_states = ttnn.squeeze(x, dim=1)
        else:
            hidden_states = x

        # Use provided rotation matrices or create new ones
        seq_len = hidden_states.shape[-2]
        if rot_mats_global is not None:
            rope_stuff = rot_mats_global
        else:
            rope_stuff = self._create_rope_stuff(seq_len)

        # Create both full attention and sliding window attention masks (exact original test pattern)
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

        for i, decoder_layer in enumerate(self.layers):
            # Each layer picks its appropriate attention mask based on layer type
            layer_mask = attention_masks[decoder_layer.attention_type]

            # Get layer-specific kv_cache if provided (like tt-transformers)
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_embeddings=rope_stuff,
                position_idx=None,
                page_table=page_table,
                kv_cache=layer_kv_cache,
            )

        # If get_last_token is specified, slice before norm and lm_head for efficiency
        if get_last_token != -1:
            # Need to work with the original 4D tensor for slicing to match tt_transformers
            # Convert back to 4D if we squeezed it
            if len(hidden_states.shape) == 3:
                # Convert [batch, seq_len, hidden] -> [batch, 1, seq_len, hidden]
                hidden_states = ttnn.unsqueeze(hidden_states, dim=1)

            # Now slice as 4D tensor (matching tt_transformers exactly)
            # Slicing the tensor to the nearest ceiling/floor multiples of 32 for the prefill_len, to get the last token
            hidden_states = ttnn.slice(
                hidden_states, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, hidden_states.shape[-1])
            )

            # Squeeze back to 3D for norm/lm_head if needed
            if len(hidden_states.shape) == 4 and hidden_states.shape[1] == 1:
                hidden_states = ttnn.squeeze(hidden_states, dim=1)

        hidden_states = self.norm(hidden_states)
        logits = ttnn.matmul(hidden_states, self.lm_head_weight)

        return logits

    def prepare_inputs_decode(self, tokens, current_pos, page_table=None):
        """
        Prepare inputs for decode mode - matches tt_transformers interface (4 values)
        """
        host_inputs = self.prepare_decode_inputs_host(tokens, current_pos, page_table)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)

        # Return 4 values to match tt_transformers interface:
        # tokens, current_pos, rope_idxs, page_table
        return (
            device_inputs[0],  # tokens
            device_inputs[1],  # current_pos
            device_inputs[2],  # rope_idxs (list of cos, sin)
            device_inputs[3],  # page_table
        )

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """
        Prepare decode inputs on host before transferring to device - matches tt_transformers interface
        """

        # Convert tokens to proper format
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        # For decode mode, we expect single tokens - NO padding needed
        # (Padding is only needed for prefill mode with longer sequences)

        # Convert to TTNN tensors on host (not on device) - they will be moved to device later
        tokens = ttnn.from_torch(
            tokens.unsqueeze(0).unsqueeze(0),  # Convert to 4D
            device=None,  # Keep on host
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Prepare current position
        current_pos_tt = ttnn.from_torch(
            current_pos,
            device=None,  # Keep on host
            dtype=ttnn.int32,
        )

        # Prepare page table if provided
        if page_table is not None:
            page_table = ttnn.from_torch(
                page_table,
                device=None,  # Keep on host
                dtype=ttnn.int32,
            )

        # Prepare attention masks on host (following original test pattern exactly)
        pos_idx = current_pos.item() if hasattr(current_pos, "item") else current_pos
        sliding_mask = get_decode_mask(pos_idx, self.hf_config.sliding_window)
        sliding_mask = sliding_mask.repeat(
            1, self.mesh_config.shard_size(self.hf_config.num_attention_heads), 1, 1
        ).transpose(1, 2)

        # Pad to tile alignment (TTNN TILE_LAYOUT requires dimensions to be multiples of 32)
        # current_h = sliding_mask.shape[2]  # heads_per_device
        # if current_h % 32 != 0:
        #     pad_h = 32 - (current_h % 32)
        #     sliding_mask = torch.nn.functional.pad(sliding_mask, (0, 0, 0, pad_h), value=-float("inf"))

        tt_mask = None  # No causal mask needed in decode mode
        # Convert sliding_mask to TTNN tensor on host first
        tt_sliding_mask_host = ttnn.from_torch(
            sliding_mask,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        # Create RoPE embeddings on host for the current position (matching test_demo.py pattern)
        rope_temp_tensor = torch.randn(1)
        # EXACTLY like original test: torch.tensor([cur_pos]).unsqueeze(0)
        position_ids = torch.tensor([pos_idx]).unsqueeze(0)
        tt_cos, tt_sin = self.rope_embeddings(rope_temp_tensor, position_ids)
        tt_cos = ttnn.from_torch(tt_cos.unsqueeze(-2), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_sin = ttnn.from_torch(tt_sin.unsqueeze(-2), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        # Use sliding window mask for sliding attention layers
        attention_masks = {"full_attention": None, "sliding_attention": tt_sliding_mask_host}

        # Store references to the DEVICE tensors - these will be automatically updated during trace execution
        self.device_decode_sliding_mask = attention_masks
        # Store the rope setup and cos/sin tensors for use in the model
        # Use self.apply_rope (ApplyRotaryPosEmb) not self.rope_embeddings (GptOssRotaryEmbedding)
        self._current_rope_stuff = (self.apply_rope, tt_cos, tt_sin)

        # Return 4 values to match tt_transformers interface:
        # tokens, current_pos_tt, rope_idxs, page_table
        # rope_idxs should be a single TTNN tensor, not a list
        # For GPT-OSS, we need to create a single rope index tensor
        rope_idxs = ttnn.from_torch(
            torch.tensor([pos_idx], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return tokens, current_pos_tt, rope_idxs, page_table

    def update_attention_masks(self, current_pos):
        """
        Update sliding window attention mask for decode mode - matches tt_transformers interface
        """
        tt_sliding_mask_device = copy_host_to_device(
            [self.device_decode_sliding_mask["sliding_attention"]], mesh_device=self.mesh_device
        )
        tt_cos_device = copy_host_to_device([self._current_rope_stuff[1]], mesh_device=self.mesh_device)
        tt_sin_device = copy_host_to_device([self._current_rope_stuff[2]], mesh_device=self.mesh_device)
        rope_stuff = (self.apply_rope, tt_cos_device[0], tt_sin_device[0])
        attention_masks = {"full_attention": None, "sliding_attention": tt_sliding_mask_device[0]}

        # Store references to the DEVICE tensors - these will be automatically updated during trace execution
        self.device_decode_sliding_mask = attention_masks
        self._current_rope_stuff = rope_stuff

    def prepare_inputs_prefill(self, tokens, start_pos=0, page_table=None, chunk_page_table=None):
        """
        Prepare inputs for prefill mode - matches Gemma3 interface (5 values)
        """

        # Embed the tokens
        if tokens.dim() == 2:
            tokens = tokens.reshape(1, 1, 1, -1)

        tokens = ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        tokens_embd = ttnn.embedding(tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT)

        # Ensure proper 4D shape for embedding output
        if len(tokens_embd.shape) == 3:
            tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)

        # Prepare rotation matrices - create actual rope stuff for compatibility
        seq_len = tokens_embd.shape[-2] if len(tokens_embd.shape) == 4 else tokens_embd.shape[-2]
        rope_stuff = self._create_rope_stuff(seq_len)
        rot_mats_global = rope_stuff  # Pass the rope_stuff as rot_mats_global
        rot_mats_local = None

        # Prepare page table if provided
        tt_page_table = None
        tt_chunk_page_table = None
        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        # Return 5 values to match Gemma3 interface:
        return tokens_embd, rot_mats_global, rot_mats_local, tt_page_table, tt_chunk_page_table

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False):
        """
        Process decode output and convert to torch tensors
        """

        # Follow original test pattern for multi-device tensors
        concat_out = self.concat_device_output(tt_out)

        if is_tokens:
            # For token output, return the token indices
            result = concat_out[:B, 0]  # [batch_size]
            return result

        # Original test uses: ttnn.to_torch(tt_output_tensor)[:, 0, :]
        # For decode, we get position 0 in sequence dimension
        torch_out = concat_out[:, 0, : self.vocab_size]  # [batch, vocab_size]

        # Reshape to match expected output format [batch, seq=1, vocab_size]
        return torch_out.unsqueeze(1).view(B, S, -1)

    def concat_device_output(self, tt_out):
        """
        Convert multi-device tensor to torch tensor following original test pattern.
        """

        # Follow the original test pattern for multi-device tensors
        # Get tensor from first device, move to CPU, then convert to torch
        tt_output_tensor = ttnn.get_device_tensors(tt_out)[0]

        tt_output_tensor = tt_output_tensor.cpu(blocking=True, cq_id=0)

        torch_tensor = ttnn.to_torch(tt_output_tensor)

        return torch_tensor

    def process_output_prefill(self, tt_out, last_token_idx):
        # last_token_idx = 92
        """
        Input is ttnn device tensor of logits. Output is torch logits tensor.
        Matches original test_demo.py pattern exactly:
        tt_output_tensor = ttnn.get_device_tensors(tt_output)[0]
        prefill_out = ttnn.to_torch(tt_output_tensor)[:, decode_start_pos - 1, :]
        """

        # EXACT original test_demo.py pattern:
        # Step 1: Get device tensor from first device
        tt_output_tensor = ttnn.get_device_tensors(tt_out)[0]

        # Step 2: Convert to torch (original test does this directly without cpu())
        torch_output = ttnn.to_torch(tt_output_tensor)

        # Step 3: Extract token at last_token_idx (original: [:, decode_start_pos - 1, :])
        result = torch_output[:, last_token_idx, : self.vocab_size]

        # Check if this matches expected shape - original test shows [1, vocab_size] 2D tensor

        # Get the token ID and decode it for comparison
        # token_id = torch.argmax(result[0].float(), dim=-1)

        return result  # Return 2D tensor [1, vocab_size] like original test shows

    def _transform_decode_inputs_device(self, tokens):
        """
        Transform decode inputs on device (e.g., embedding)
        """
        tt_tokens = ttnn.embedding(tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT)

        # Ensure proper 4D shape for embedding output
        if len(tt_tokens.shape) == 3:
            tt_tokens = ttnn.unsqueeze_to_4D(tt_tokens)

        return tt_tokens

    def _increment_decode_positions_device(self, current_pos, rot_mat_idxs_global, rot_mat_idxs_local):
        """
        Increment position indices on device for next decode step
        """
        # Simple increment operation - can be made more sophisticated
        current_pos_tiled = ttnn.to_layout(current_pos, layout=ttnn.TILE_LAYOUT)
        # Update only active positions (current_pos != -1)
        predicate = ttnn.ne(current_pos_tiled, -1)
        result = ttnn.where(
            predicate,
            ttnn.add(current_pos_tiled, 1),
            current_pos_tiled,
        )
        ttnn.copy(ttnn.to_layout(result, layout=ttnn.ROW_MAJOR_LAYOUT), current_pos)

        if rot_mat_idxs_global is not None:
            ttnn.plus_one(rot_mat_idxs_global)
        if rot_mat_idxs_local is not None:
            ttnn.plus_one(rot_mat_idxs_local)
