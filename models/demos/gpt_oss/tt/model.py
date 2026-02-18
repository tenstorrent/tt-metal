# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import torch

import ttnn
from models.common.utility_functions import nearest_32
from models.demos.gpt_oss.config import MeshConfig, Mode, ModeConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss.utils.substate import substate
from models.tt_transformers.tt.common import copy_host_to_device, rope_scaling_model_factory
from models.tt_transformers.tt.rope import RotarySetup

from .layer import DecoderLayer
from .rms_norm import RMSNorm


def create_rope_setup(
    mesh_device,
    hf_config,
    max_local_batch_size=1,
    users_row_sharded=False,
    datatype=ttnn.bfloat16,
    shard_batch_to_mesh_dim=0,
):
    """
    Create and return a RotarySetup instance for the GPT-OSS model.

    This function extracts the rope setup logic from the Model class to allow
    for independent testing and comparison with HuggingFace reference implementations.

    Args:
        mesh_device: TTNN mesh device for computation
        hf_config: HuggingFace model configuration containing rope_scaling, rope_theta, etc.
        max_local_batch_size: Maximum local batch size (default: 1)
        users_row_sharded: Whether users are row-sharded across devices (default: False)
        datatype: TTNN data type for tensors (default: ttnn.bfloat16)
        shard_batch_to_mesh_dim: Mesh dimension to shard batch to (default: 0)

    Returns:
        RotarySetup: Configured rotary setup instance with cos/sin matrices
    """
    max_seq_len = getattr(hf_config, "max_position_embeddings", 131072)
    rope_scaling = rope_scaling_model_factory(hf_config.rope_scaling)
    batch_size = max_local_batch_size * mesh_device.shape[0] if users_row_sharded else max_local_batch_size

    rope_setup = RotarySetup(
        device=mesh_device,
        batch_size=batch_size,
        head_dim=hf_config.head_dim,
        max_seq_len=max_seq_len,
        rope_theta=getattr(hf_config, "rope_theta", 150000.0),
        rope_scaling=rope_scaling,
        datatype=datatype,
        shard_batch_to_mesh_dim=shard_batch_to_mesh_dim,
    )

    return rope_setup


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
        max_local_batch_size=1,
        users_row_sharded=False,
        use_throughput_experts=False,
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
        # hf_config.num_hidden_layers = 1
        self.core_grid = ttnn.CoreCoord(8, 8)
        self.head_dim = hf_config.head_dim
        self.max_local_batch_size = max_local_batch_size
        self.users_row_sharded = users_row_sharded

        self.ccl_manager = ccl_manager

        # Use mode-aware MeshConfig (stores separate configs for prefill and decode)
        # Decode: EP=rows for expert parallelism, SP=1
        # Prefill: EP=1, SP=rows for sequence parallelism (auto-defaults)
        self.mesh_config = mesh_config or MeshConfig(
            mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1], ep=mesh_device.shape[0], sp=1)
        )

        # Setup RoPE using tt-transformers RotarySetup (handles cos/sin matrices and transformation matrices)
        # Force datatype to bfloat16 since rotary_embedding_llama requires bfloat16
        self.rope_setup = create_rope_setup(
            mesh_device=mesh_device,
            hf_config=hf_config,
            max_local_batch_size=max_local_batch_size,
            users_row_sharded=users_row_sharded,
            datatype=ttnn.bfloat16,
            shard_batch_to_mesh_dim=0,
        )

        # Keep references for compatibility
        self.cos_matrix = self.rope_setup.cos_matrix
        self.sin_matrix = self.rope_setup.sin_matrix
        self.transformation_mats = self.rope_setup.get_both_trans_mats()

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
                max_local_batch_size=max_local_batch_size,
                users_row_sharded=users_row_sharded,
                use_throughput_experts=use_throughput_experts,
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
            cache_file_name=get_cache_file_name(tensor_cache_path, "lm_head_sharded.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_config.column_parallel(mesh_device),
        )

    @classmethod
    def create_transformer_compatible(
        cls,
        args,
        dtype,
        mesh_device,
        state_dict,
        tensor_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        attention_class=None,
        rope_setup_class=None,
        mesh_config=None,
        create_kv_cache=True,
        users_row_sharded=False,
        use_throughput_experts=False,
    ):
        """Constructor compatible with tt_transformers.Transformer interface"""
        # Create a dummy CCL manager for GPT-OSS
        from models.demos.gpt_oss.tt.ccl import CCLManager

        ccl_manager = CCLManager(mesh_device, num_links=4 if mesh_device.shape[0] > 1 else 1)

        # Create instance using direct initialization
        instance = cls.__new__(cls)
        instance.__init__(
            mesh_device=mesh_device,
            hf_config=args.hf_config,
            state_dict=state_dict,
            ccl_manager=ccl_manager,
            dtype=dtype,
            tensor_cache_path=tensor_cache_path,
            paged_attention_config=paged_attention_config,
            mesh_config=mesh_config,
            create_kv_cache=create_kv_cache,
            max_local_batch_size=args.max_local_batch_size,
            users_row_sharded=users_row_sharded,
            use_throughput_experts=use_throughput_experts,
        )

        # Add tt_transformers compatible attributes
        instance.args = args
        instance.vocab_size = args.vocab_size
        instance.n_layers = args.n_layers
        instance.dtype = dtype

        return instance

    def switch_mode(self, mode: Mode):
        # No-op; required by tt_transformers generator interface.
        return None

    def _forward_layers_and_head(
        self,
        hidden_states,
        rope_mats,
        current_pos,
        page_table,
        kv_cache,
        get_last_token=-1,
        is_decode=True,
        user_id=0,
        batch_size=1,
    ):
        """
        Shared forward pass through decoder layers and final projection.

        Args:
            hidden_states: Input tensor
            rope_mats: RoPE rotation matrices [cos, sin]
            current_pos: Current position (for decode) or None (for prefill)
            page_table: Page table for paged attention
            kv_cache: KV cache list per layer

        Returns:
            logits: Output logits
        """
        # Determine mode based on current_pos presence
        mode = Mode.DECODE if current_pos is not None else Mode.PREFILL

        # Process through decoder layers
        for i, decoder_layer in enumerate(self.layers):
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=rope_mats,
                position_idx=current_pos,
                page_table=page_table,
                kv_cache=layer_kv_cache,
                is_decode=is_decode,
                user_id=user_id,
                batch_size=batch_size,
            )
        logits = hidden_states

        if get_last_token != -1:
            if len(logits.shape) == 3:
                logits = ttnn.unsqueeze(logits, dim=1)
            if batch_size > 1:
                # Batch>1: tokens are concatenated [1,1,B*S,H]. Extract each user's 32-token tile.
                per_user_seq = logits.shape[2] // batch_size
                tiles = []
                for b in range(batch_size):
                    start = b * per_user_seq + get_last_token
                    tile = ttnn.slice(logits, (0, 0, start, 0), (1, 1, start + 32, logits.shape[-1]))
                    tiles.append(tile)
                logits.deallocate(True)
                logits = ttnn.concat(tiles, dim=2)  # [1, 1, B*32, H]
                for t in tiles:
                    t.deallocate(True)
            else:
                logits_sliced = ttnn.slice(
                    logits, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, logits.shape[-1])
                )
                logits.deallocate(True)
                logits = logits_sliced
            hidden_states = logits

        # Final norm and lm_head
        hidden_states = self.norm(hidden_states)
        logits = ttnn.matmul(hidden_states, self.lm_head_weight, dtype=ttnn.bfloat8_b)
        hidden_states.deallocate(True)
        # TP all-gather if using tensor parallelism
        config = self.mesh_config.get_config(mode)
        if config.tp > 1:
            logits_gathered = self.mesh_config.allgather(
                logits, self.ccl_manager, axis=self.mesh_config.tp_axis, dim=-1
            )
            logits.deallocate(True)
            logits = logits_gathered

        return logits

    def ttnn_decode_forward(
        self,
        tokens,
        current_pos,
        rot_mat_idxs=None,
        page_table=None,
        kv_cache=None,
        sampling_on_device=False,
        capture_sampling_trace=False,
    ):
        """
        Decode forward pass - processes single tokens.
        Matches tt-transformers interface where rot_mat_idxs are used for on-device RoPE lookup.
        """
        # Embed tokens
        input_embeds = ttnn.embedding(tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        input_embeds = ttnn.unsqueeze(input_embeds, 0)
        # Get RoPE embeddings via on-device embedding lookup (matches tt-transformers)
        rope_mats = self.rope_setup.get_rot_mats(self.get_tt_pos_idx(rot_mat_idxs))

        # Forward through layers and head (shared with prefill)
        out = self._forward_layers_and_head(
            hidden_states=input_embeds,
            rope_mats=rope_mats,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=True,
        )
        # Return logits and None for log-probs for compatibility with generator interface
        # TODO: Add log-probs return value once sampling_on_device is supported
        return out, None

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
        batch_size=1,
    ):
        """Prefill forward pass - processes full sequences"""
        # Use provided rotation matrices or slice from rope_setup (matches tt-transformers)
        seq_len = x.shape[-2]
        if rot_mats_global is not None:
            rope_mats = rot_mats_global
        else:
            # Slice cos/sin matrices for prefill sequence length (matches tt-transformers model.py lines 156-159)
            rope_mats = [
                self.rope_setup.cos_matrix_prefill[:, :, :seq_len, :],
                self.rope_setup.sin_matrix_prefill[:, :, :seq_len, :],
            ]

        # Forward through layers and head (shared with decode)
        logits = self._forward_layers_and_head(
            hidden_states=x,
            rope_mats=rope_mats,
            current_pos=None,  # No current_pos for prefill
            page_table=page_table,
            kv_cache=kv_cache,
            get_last_token=get_last_token,
            is_decode=False,
            user_id=user_id,
            batch_size=batch_size,
        )

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

    def get_tt_pos_idx(self, current_pos):
        if isinstance(current_pos, ttnn.Tensor):
            return current_pos
        else:
            # Ensure position indices are non-negative (matches tt-transformers)
            B = current_pos.shape[0]
            rot_current_pos = torch.maximum(current_pos, torch.tensor(0, dtype=torch.int64))
            rot_current_pos = rot_current_pos.reshape(1, B)  # [1, batch]
            assert rot_current_pos.shape == (1, B), "rot_current_pos must be a [1, batch] tensor"
            assert torch.min(rot_current_pos) >= 0, "rot_current_pos must be non-negative"
            # Add padding if needed
            pad_size = nearest_32(B) - B
            rot_current_pos = torch.nn.functional.pad(rot_current_pos, (0, pad_size), "constant", 0)
            mesh_mapper = (
                ttnn.ShardTensor2dMesh(self.mesh_device, dims=(-1, None), mesh_shape=self.mesh_device.shape)
                if self.users_row_sharded
                else ttnn.ReplicateTensorToMesh(self.mesh_device)
            )
            rope_idxs = ttnn.as_tensor(
                rot_current_pos,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mesh_mapper,
            )

            return rope_idxs

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
        if self.users_row_sharded:
            mesh_mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, None), mesh_shape=self.mesh_device.shape)
        else:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        tokens = ttnn.from_torch(tokens.squeeze(), device=None, dtype=ttnn.uint32, mesh_mapper=mesh_mapper)
        tokens = ttnn.unsqueeze_to_4D(tokens)

        rope_idxs = self.get_tt_pos_idx(current_pos)

        # Prepare current position tensor
        current_pos_tt = ttnn.from_torch(current_pos, device=None, dtype=ttnn.int32, mesh_mapper=mesh_mapper)

        # Prepare page table if provided
        if page_table is not None:
            page_table = ttnn.from_torch(page_table, device=None, dtype=ttnn.int32, mesh_mapper=mesh_mapper)

        return tokens, current_pos_tt, rope_idxs, page_table

    def prepare_inputs_prefill_trace(
        self, tokens, start_pos=0, page_table=None, chunk_page_table=None, last_token_idx=None
    ):
        """Prepare inputs on host so we later send them to device"""
        host_inputs = self.prepare_inputs_prefill(
            tokens,
            start_pos=start_pos,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            trace_enabled=True,
            last_token_idx=last_token_idx,
        )
        return host_inputs

    def transform_and_embed_prefill_inputs_device(self, tokens, tt_page_table, tt_chunk_page_table):
        """Transform and embed tokens on device"""
        tokens_embd = ttnn.embedding(tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        tokens.deallocate(True)
        if len(tokens_embd.shape) == 3:
            tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)
        return tokens_embd, tt_page_table, tt_chunk_page_table

    def prepare_inputs_prefill(
        self,
        tokens,
        start_pos=0,
        page_table=None,
        chunk_page_table=None,
        trace_enabled=False,
        last_token_idx=None,
        global_user_id=None,
        batched_prefill=False,
    ):
        """Prepare inputs for prefill mode

        Args:
            batched_prefill: If True, tokens is [num_rows, seq_len] and will be
                sharded across mesh rows. Each row processes a different user.
        """
        # Embed the tokens
        device = None if trace_enabled else self.mesh_device

        if batched_prefill:
            # Row-parallel batched prefill: tokens is [num_rows, seq_len]
            # Shard across mesh rows so each row gets one user's tokens [1, seq_len]
            num_rows = tokens.shape[0]
            seq_len_per_user = tokens.shape[1]
            tokens = tokens.reshape(num_rows, 1, 1, seq_len_per_user)
            tokens = ttnn.from_torch(
                tokens,
                device=device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, None), mesh_shape=self.mesh_device.shape),
            )
        else:
            if tokens.dim() == 2:
                tokens = tokens.reshape(1, 1, 1, -1)
            tokens = ttnn.from_torch(tokens, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

        if not trace_enabled:
            tokens_embd = ttnn.embedding(tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
            tokens.deallocate(True)

            # Ensure proper 4D shape
            if len(tokens_embd.shape) == 3:
                tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)

        # Prepare rotation matrices (slice from rope_setup like tt-transformers model.py lines 156-159)
        seq_len = self.args.max_seq_len if trace_enabled else tokens_embd.shape[-2]
        rot_mats_global = [
            self.rope_setup.cos_matrix_prefill[:, :, :seq_len, :],
            self.rope_setup.sin_matrix_prefill[:, :, :seq_len, :],
        ]
        rot_mats_local = None

        # Prepare page tables if provided
        tt_page_table = None
        tt_chunk_page_table = None
        if page_table is not None:
            if self.users_row_sharded and page_table.shape[0] > 1:
                # Multi-user prefill: shard page table across rows
                tt_page_table = ttnn.from_torch(
                    page_table,
                    device=device,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.mesh_device, dims=(0, None), mesh_shape=self.mesh_device.shape
                    ),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            elif self.users_row_sharded and page_table.shape[0] == 1:
                # Single-user prefill with row-sharding: create page table with valid entries
                # only on the target row to prevent KV cache corruption on other rows
                assert (
                    global_user_id is not None
                ), "global_user_id is required for single-user row-sharded prefill to target the correct mesh row"
                num_rows = self.mesh_device.shape[0]
                users_per_row = getattr(self.args, "max_local_batch_size", self.args.max_batch_size // num_rows)
                target_row = global_user_id // users_per_row

                # Create page table with -1 (invalid) for all rows except target
                full_page_table = torch.full((num_rows, page_table.shape[1]), -1, dtype=page_table.dtype)
                full_page_table[target_row] = page_table[0]

                tt_page_table = ttnn.from_torch(
                    full_page_table,
                    device=device,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.mesh_device, dims=(0, None), mesh_shape=self.mesh_device.shape
                    ),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            else:
                # Single-user prefill or non-row-sharded: replicate page table
                tt_page_table = ttnn.from_torch(
                    page_table,
                    device=device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )

        return (
            tokens if trace_enabled else tokens_embd,
            rot_mats_global,
            rot_mats_local,
            tt_page_table,
            tt_chunk_page_table,
        )

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False):
        """Process decode output and convert to torch tensors"""
        concat_out = self.concat_device_output(tt_out)
        if is_tokens:
            return concat_out[:B, 0]  # [batch_size]

        torch_out = concat_out[:, 0, :, :]  # [1, 1, B, vocab_size]
        # TODO: this view is dangerous, forces bad tensor shapes to work but we get garbage outputs if they're wrong
        return torch_out.view(B, S, -1)

    def concat_device_output(self, tt_out):
        """Convert multi-device tensor to torch tensor"""
        if self.users_row_sharded:
            tt_output_tensor = ttnn.get_device_tensors(tt_out)[:: self.mesh_device.shape[1]]
            return torch.concat([ttnn.to_torch(t) for t in tt_output_tensor], dim=-2)
        else:
            tt_output_tensor = ttnn.get_device_tensors(tt_out)[0]
            tt_output_tensor = tt_output_tensor.cpu(blocking=True, cq_id=0)
            return ttnn.to_torch(tt_output_tensor)

    def process_output_prefill(self, tt_out, last_token_idx):
        """Process prefill output and extract last token logits"""
        tt_output_tensor = ttnn.get_device_tensors(tt_out)[0]
        torch_output = ttnn.to_torch(tt_output_tensor)
        result = torch_output[..., last_token_idx, : self.vocab_size]
        return result

    def process_output_prefill_batched(self, tt_out, last_token_idxs, users_per_row=1, seq_len_per_user=None):
        """Process row-parallel batched prefill output.

        Extracts logits from one device per row (first device of each row).
        Supports multiple users per row when users_per_row > 1.

        Args:
            tt_out: Multi-device output tensor
            last_token_idxs: List of last_token_idx per user (length = num_rows * users_per_row)
            users_per_row: Number of users per mesh row per iteration
            seq_len_per_user: Per-user sequence length (required when users_per_row > 1)

        Returns:
            List of per-user logit tensors (one per user)
        """
        num_cols = self.mesh_device.shape[1]
        device_tensors = ttnn.get_device_tensors(tt_out)
        results = []
        num_rows = self.mesh_device.shape[0]
        for row in range(num_rows):
            device_idx = row * num_cols  # First device of each row
            torch_output = ttnn.to_torch(device_tensors[device_idx])
            for u in range(users_per_row):
                user_flat_idx = row * users_per_row + u
                last_idx = last_token_idxs[user_flat_idx] if isinstance(last_token_idxs, list) else last_token_idxs
                if users_per_row > 1:
                    # Tokens are concatenated: user u's last token is at offset u*seq_len_per_user + last_idx
                    global_idx = u * seq_len_per_user + last_idx
                    result = torch_output[..., global_idx, : self.vocab_size]
                else:
                    result = torch_output[..., last_idx, : self.vocab_size]
                results.append(result)
        return results
