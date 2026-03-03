# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger

import ttnn
from models.common.sampling.generator import SamplingGenerator
from models.common.utility_functions import nearest_32
from models.demos.gpt_oss.config import MeshConfig, Mode, ModeConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss.utils.substate import substate
from models.tt_transformers.tt.common import rope_scaling_model_factory
from models.tt_transformers.tt.rope import RotarySetup

from .layer import DecoderLayer
from .rms_norm import RMSNorm


def create_rope_setup(
    mesh_devices,
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
        mesh_devices: List of TTNN mesh devices for computation
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
    batch_size = max_local_batch_size * mesh_devices[0].shape[0] if users_row_sharded else max_local_batch_size

    rope_setup = [
        RotarySetup(
            device=device,
            batch_size=batch_size,
            head_dim=hf_config.head_dim,
            max_seq_len=max_seq_len,
            rope_theta=getattr(hf_config, "rope_theta", 150000.0),
            rope_scaling=rope_scaling,
            datatype=datatype,
            shard_batch_to_mesh_dim=shard_batch_to_mesh_dim,
        )
        for device in mesh_devices
    ]

    return rope_setup


def worker_for_task(t, N, M):
    q = N // M
    r = N % M

    if t < (q + 1) * r:
        return t // (q + 1)
    else:
        return r + (t - (q + 1) * r) // q


class ModelWithMP:
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
        mesh_shape=None,
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
        print("Loading Model MP")
        self.mesh_device = mesh_device
        self.mesh_shape = mesh_shape if mesh_shape is not None else mesh_shape
        self.vocab_size = hf_config.vocab_size
        self.hf_config = hf_config
        # hf_config.num_hidden_layers = 1
        self.core_grid = ttnn.CoreCoord(8, 8)
        self.head_dim = hf_config.head_dim
        self.max_local_batch_size = max_local_batch_size
        self.users_row_sharded = users_row_sharded

        self.ccl_manager = ccl_manager

        self.mesh_config = MeshConfig(
            self.mesh_shape,
            decode=ModeConfig(mp=self.mesh_shape[1], ep=self.mesh_shape[0], sp=1, tp=1),
            mp_enabled=True,
        )
        print("Initialized ModelWithMP with mesh_config: ", self.mesh_config)
        self.mp_submeshes = self.ccl_manager.mp_submeshes

        print("Submeshes = ", self.mp_submeshes, "model_config = ", self.mesh_config)
        # Setup RoPE using tt-transformers RotarySetup (handles cos/sin matrices and transformation matrices)
        # Force datatype to bfloat16 since rotary_embedding_llama requires bfloat16

        num_layers_per_submesh = hf_config.num_hidden_layers // self.mesh_shape[1]
        num_layers_per_submesh_rem = hf_config.num_hidden_layers % self.mesh_shape[1]
        print(
            "num_layers_per_submesh = ",
            num_layers_per_submesh,
            "num_layers_per_submesh_rem = ",
            num_layers_per_submesh_rem,
        )

        self.rope_setup = create_rope_setup(
            mesh_devices=self.mp_submeshes,  # Run on first submesh
            hf_config=hf_config,
            max_local_batch_size=max_local_batch_size,
            users_row_sharded=users_row_sharded,
            datatype=ttnn.bfloat16,
            shard_batch_to_mesh_dim=0,
        )

        # Keep references for compatibility
        self.cos_matrix = [x.cos_matrix for x in self.rope_setup]
        self.sin_matrix = [x.sin_matrix for x in self.rope_setup]
        self.transformation_mats = [x.get_both_trans_mats() for x in self.rope_setup]

        if state_dict:
            embedding_weight = substate(state_dict, "model.embed_tokens")["weight"]
            embedding_weight = embedding_weight.unsqueeze(0).unsqueeze(0)
        else:
            embedding_weight = None

        self.embedding_weight = ttnn.as_tensor(
            embedding_weight,
            dtype=ttnn.bfloat16,
            device=self.mp_submeshes[0],  # Place on first submesh for embedding lookup
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "model.embed_tokens.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.layers = []
        for layer_idx in range(hf_config.num_hidden_layers):
            submesh_id = worker_for_task(layer_idx, hf_config.num_hidden_layers, self.mesh_shape[1])
            print("Assigning layer ", layer_idx, " to submesh ", submesh_id)
            self.layers.append(
                DecoderLayer(
                    self.mp_submeshes[submesh_id],
                    hf_config,
                    substate(state_dict, f"model.layers.{layer_idx}"),
                    layer_idx,
                    ccl_manager,
                    dtype=dtype,
                    tensor_cache_path=get_cache_file_name(tensor_cache_path, f"model.layers.{layer_idx}"),
                    paged_attention_config=paged_attention_config,
                    mesh_config=self.mesh_config,
                    create_kv_cache=create_kv_cache,
                    transformation_mats=self.transformation_mats[submesh_id],
                    max_local_batch_size=max_local_batch_size,
                    users_row_sharded=users_row_sharded,
                    use_throughput_experts=use_throughput_experts,
                )
            )
        self.norm = RMSNorm(
            self.mp_submeshes[-1],
            hf_config,
            substate(state_dict, "model.norm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "norm"),
            mesh_config=self.mesh_config,
        )

        # Pad lm_head vocab dimension to padded_vocab_size BEFORE column-parallel sharding.
        # TTSampling._create_indices_tensors uses padded_per_device as the stride for device
        # offset calculation: global_idx = device_id * padded_per_device + local_idx.
        # If we shard at unpadded boundaries and pad after, the offsets are wrong for devices 1+.
        # Pre-sharding padding ensures device shard boundaries match the offset stride.
        sampling_splits = 1  # mesh_shape[1]
        per_device_padded = (((self.vocab_size + sampling_splits - 1) // sampling_splits + 31) // 32) * 32
        padded_vocab_size = per_device_padded * sampling_splits

        if state_dict:
            lm_head_weight = substate(state_dict, "lm_head")["weight"].transpose(0, 1)  # [hidden, vocab]
            if lm_head_weight.shape[1] < padded_vocab_size:
                lm_head_weight = torch.nn.functional.pad(
                    lm_head_weight, (0, padded_vocab_size - lm_head_weight.shape[1]), "constant", 0
                )
        else:
            lm_head_weight = None

        self.lm_head_weight = ttnn.as_tensor(
            lm_head_weight,
            device=self.mp_submeshes[-1],
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            cache_file_name=get_cache_file_name(tensor_cache_path, "lm_head_padded.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Initialize on-device sampling (supported when per-device vocab fits in 64K)
        self._supports_on_device_sampling = True  # self.vocab_size // sampling_splits <= 64 * 1024
        self._prefill_sampling_active = False
        # sampling_dp: number of independent sampling groups (one per mesh row for row-sharded users)
        self.sampling_dp = mesh_shape[0] if users_row_sharded else 1
        if self._supports_on_device_sampling:
            # tt_ccl=None makes TTSampling fall back to ttnn.all_gather() which works on [4,8] meshes
            self.sampling = SamplingGenerator(
                args=self.args if hasattr(self, "args") else self._make_sampling_args(hf_config, self.mp_submeshes[-1]),
                mesh_device=self.mp_submeshes[-1],
                tt_ccl=None,
                enable_internal_trace=False,
            )
            # Hook reset_sampling_params to set prefill flag — Generator calls this
            # before prefill forward; tells _forward_layers_and_head to skip TP all-gather
            _orig_reset = self.sampling.reset_sampling_params

            def _reset_with_flag(params, _orig=_orig_reset):
                _orig(params)
                self._prefill_sampling_active = True

            self.sampling.reset_sampling_params = _reset_with_flag
            logger.info(f"On-device sampling initialized (vocab_size={self.vocab_size}, splits={sampling_splits})")
        else:
            self.sampling = None

    def _make_sampling_args(self, hf_config, mesh_device):
        """Create a minimal args object for SamplingGenerator/TTSampling."""

        class _SamplingArgs:
            pass

        args = _SamplingArgs()
        args.vocab_size = hf_config.vocab_size
        num_tp = 1
        # padded_vocab_size: per-device vocab must be tile-aligned (multiple of 32)
        # for TTPenalties scatter operations.
        # The lm_head weight is padded to this size BEFORE column-parallel sharding,
        # so device shard boundaries align with TTSampling device offset strides.
        per_device_vocab = ((args.vocab_size + num_tp - 1) // num_tp + 31) // 32 * 32
        args.padded_vocab_size = per_device_vocab * num_tp
        args.cluster_shape = (1, 1)
        args.sampling_all_gather_axis = 1
        args.num_devices = mesh_device.get_num_devices()
        args.is_galaxy = self.mesh_shape[0] > 1
        args.model_config = {}  # No SAMPLING_AG_CONFIG → regular sampling path always used
        # sampling_dp: number of independent sampling groups (one per mesh row)
        # Only use row-sharded sampling when users_row_sharded is active
        args.sampling_dp = self.sampling_dp
        return args

    def _increment_decode_positions_device(self, current_pos, rot_mat_idxs):
        """On-device position increment for traced decode loops with sampling."""
        for pos in current_pos:
            ttnn.plus_one(pos, skip_negative_entries=True)
        for idx in rot_mat_idxs:
            ttnn.plus_one(idx)

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
        use_model_parallelism=False,
        ccl_manager=None,
        mesh_shape=None,
    ):
        """Constructor compatible with tt_transformers.Transformer interface"""
        # Create a dummy CCL manager for GPT-OSS
        from models.demos.gpt_oss.tt.ccl import CCLManager

        if ccl_manager is None:
            ccl_manager = CCLManager(
                mesh_device, num_links=4 if mesh_shape[0] > 1 else 1, use_model_parallelism=use_model_parallelism
            )
        else:
            print("Got CCL Manager")
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
            mesh_shape=mesh_shape,
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

    def _copy_hidden_states_between_submeshes(self, hidden_states, from_submesh_id, to_submesh_id):
        """
        Copy hidden_states from one submesh to another using the pre-created socket pair.
        Sockets are created in the constructor and reused for every forward pass.
        """
        to_submesh = self.mp_submeshes[to_submesh_id]
        output_tensor = ttnn.allocate_tensor_on_device(hidden_states.spec, to_submesh)
        sender_socket, receiver_socket = self.ccl_manager.submesh_socket_pairs[(from_submesh_id, to_submesh_id)]
        ttnn.experimental.send_async(hidden_states, sender_socket)
        ttnn.experimental.recv_async(output_tensor, receiver_socket)
        ttnn.synchronize_device(self.mp_submeshes[from_submesh_id])
        ttnn.synchronize_device(to_submesh)
        hidden_states.deallocate(True)
        return output_tensor

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
        sampling_on_device=False,
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
        seq_len = hidden_states.shape[-2]

        num_submeshes = self.mesh_shape[1]
        # hidden_states starts on submesh 0 (embedding is on first submesh)
        current_submesh_id = 0

        # Process through decoder layers
        for i, decoder_layer in enumerate(self.layers):
            # Layer i is on submesh given by worker_for_task (same mapping as in __init__)
            next_submesh_id = worker_for_task(i, self.hf_config.num_hidden_layers, num_submeshes)
            if next_submesh_id != current_submesh_id:
                # logger.info(
                #     f"Copying hidden_states from submesh {current_submesh_id}, {hidden_states.device().id()} to {next_submesh_id} for layer {i}"
                # )
                # Copy hidden_states from current submesh to the submesh that runs this layer
                hidden_states = self._copy_hidden_states_between_submeshes(
                    hidden_states, current_submesh_id, next_submesh_id
                )
                # logger.info(f"Copied hidden_states with device id {hidden_states.device().id()} for layer {i}")
                current_submesh_id = next_submesh_id

            layer_kv_cache = kv_cache[i] if kv_cache is not None else None
            this_rope_mats = rope_mats[current_submesh_id]
            # logger.info(f"Running layer {i} on submesh {current_submesh_id}, kv_device = {layer_kv_cache[0].device().id()}, rope_device = {this_rope_mats[0].device().id()}, position_device = {current_pos[current_submesh_id].device().id() if current_pos is not None else None}")
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=this_rope_mats,
                position_idx=current_pos[current_submesh_id] if current_pos is not None else None,
                page_table=page_table,
                kv_cache=layer_kv_cache,
                is_decode=is_decode,
                user_id=user_id,
                batch_size=batch_size,
            )
            ttnn.ReadDeviceProfiler(self.mp_submeshes[current_submesh_id])
            current_submesh_id = next_submesh_id
        logits = hidden_states

        # Norm and lm_head are on the last submesh; copy there if we are not already
        last_submesh_id = num_submeshes - 1
        if current_submesh_id != last_submesh_id:
            copied = self._copy_hidden_states_between_submeshes(hidden_states, current_submesh_id, last_submesh_id)
            hidden_states.deallocate(True)
            hidden_states = copied
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
        # Skip TP all-gather when sampling is active — TTSampling handles its own all-gather
        skip_gather = sampling_on_device or self._prefill_sampling_active
        self._prefill_sampling_active = False
        # No post-matmul padding needed: the lm_head weight is pre-padded to
        # padded_vocab_size before column-parallel sharding, so each device's
        # matmul output is already tile-aligned (per_device_padded width).
        # logger.info("Synchronzing last submesh")
        ttnn.synchronize_device(self.mp_submeshes[-1])
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
        # For non-row-sharded b<32, token buffer is padded to 32 — only embed real tokens
        actual_batch = current_pos[0].shape[-1]
        if not self.users_row_sharded and tokens.shape[-1] > actual_batch:
            tokens_for_embed = tokens[:, :, :, :actual_batch]
        else:
            tokens_for_embed = tokens
        input_embeds = ttnn.embedding(
            tokens_for_embed, self.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b
        )
        input_embeds = ttnn.unsqueeze(input_embeds, 0)
        # Get RoPE embeddings via on-device embedding lookup (matches tt-transformers)self.rope_setup.get_rot_mats(self.get_tt_pos_idx(rot_mat_idxs))
        rope_mats = []
        for i in range(len(self.mp_submeshes)):
            rope_mats.append(
                self.rope_setup[i].get_rot_mats(self.get_tt_pos_idx(rot_mat_idxs[i], self.mp_submeshes[i]))
            )

        # Forward through layers and head (shared with prefill)
        out = self._forward_layers_and_head(
            hidden_states=input_embeds,
            rope_mats=rope_mats,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=True,
            sampling_on_device=sampling_on_device,
        )

        if sampling_on_device and self.sampling is not None:
            # Pad logits batch to 32 (TTSampling requirement) before split-trace or sampling
            batch_dim = out.shape[-2]
            if batch_dim < 32:
                out = ttnn.pad(out, padding=[(0, 0), (0, 0), (0, 32 - batch_dim), (0, 0)], value=0.0)
            self._increment_decode_positions_device(current_pos, rot_mat_idxs)
            if capture_sampling_trace:
                return out
            tt_toks, tt_log_probs = self.sampling.sample(out, tt_out_tok=tokens, enable_trace=False)
            return tt_toks, tt_log_probs

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
            rope_mats = self.rope_setup

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

    def process_logits_after_prefill_trace(self, logits, last_token_idx):
        """
        Post-process traced prefill output to the 32-token tile containing `last_token_idx`.

        Unlike tt_transformers `Transformer`, GPT-OSS `ttnn_prefill_forward` already
        applies final norm + lm_head, so this method only slices logits.
        """
        get_last_token = (last_token_idx // 32) * 32
        logits = ttnn.slice(
            logits,
            (0, 0, get_last_token, 0),
            (1, 1, get_last_token + 32, logits.shape[-1]),
        )
        return logits

    # def prepare_inputs_decode(self, tokens, current_pos, page_table=None):
    #     """
    #     Prepare inputs for decode mode - matches tt_transformers interface (4 values).
    #     Returns: tokens, current_pos, rope_idxs, page_table

    #     Note: rope_idxs are position indices that will be used with get_rot_mats()
    #     for on-device RoPE embedding lookup.
    #     """
    #     host_inputs = self.prepare_decode_inputs_host(tokens, current_pos, page_table)
    #     device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
    #     # Return 4 values to match tt_transformers interface:
    #     # tokens, current_pos, rope_idxs, page_table
    #     return (
    #         device_inputs[0],  # tokens
    #         device_inputs[1],  # current_pos
    #         device_inputs[2],  # rope_idxs - position indices for embedding lookup
    #         device_inputs[3],  # page_table
    #     )

    def get_tt_pos_idx(self, current_pos, device):
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
            rope_idxs = ttnn.to_device(
                ttnn.as_tensor(
                    rot_current_pos,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                device=device,
            )

            return rope_idxs

    def prepare_inputs_decode(self, tokens, current_pos, page_table=None):
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

        # Pad token buffer to 32 for non-row-sharded b<32 (TTSampling requirement)
        if not self.users_row_sharded and tokens.view(-1).shape[-1] < 32:
            tokens = torch.nn.functional.pad(tokens.view(-1), (0, 32 - len(tokens.view(-1))), "constant", 0)

        tokens = ttnn.from_torch(tokens.squeeze(), device=self.mp_submeshes[0], dtype=ttnn.uint32)
        tokens = ttnn.unsqueeze_to_4D(tokens)

        rope_idxs = [self.get_tt_pos_idx(current_pos, device=submesh) for submesh in self.mp_submeshes]

        # Prepare current position tensor
        current_pos_tt = [
            ttnn.from_torch(current_pos, device=submesh, dtype=ttnn.int32) for submesh in self.mp_submeshes
        ]

        # Prepare page table if provided
        if page_table is not None:
            page_table = ttnn.from_torch(page_table, device=self.mp_submeshes[0], dtype=ttnn.int32)

        return tokens, current_pos_tt, rope_idxs, page_table

    def prepare_prefill_inputs_trace(
        self, tokens, start_pos=0, page_table=None, chunk_page_table=None, last_token_idx=None, **kwargs
    ):
        """Prepare inputs on host so we later send them to device"""
        host_inputs = self.prepare_inputs_prefill(
            tokens,
            start_pos=start_pos,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            trace_enabled=True,
            last_token_idx=last_token_idx,
            **kwargs,
        )
        return host_inputs

    def transform_and_embed_prefill_inputs_device(self, tokens, tt_page_table, tt_chunk_page_table):
        """Transform and embed tokens on device"""
        tokens_embd = ttnn.embedding(tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        # Keep `tokens` allocated: trace replay updates this same device buffer via copy_host_to_device.
        # Deallocating it here breaks prefill trace replay with "Buffer must be allocated on device".
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
        device = None if trace_enabled else self.mp_submeshes[0]

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
                mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=(0, None), mesh_shape=device.shape),
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
            [
                x.cos_matrix_prefill[:, :, :seq_len, :],
                x.sin_matrix_prefill[:, :, :seq_len, :],
            ]
            for x in self.rope_setup
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
                    mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=(0, None), mesh_shape=device.shape),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            elif self.users_row_sharded and page_table.shape[0] == 1:
                # Single-user prefill with row-sharding: create page table with valid entries
                # only on the target row to prevent KV cache corruption on other rows
                assert (
                    global_user_id is not None
                ), "global_user_id is required for single-user row-sharded prefill to target the correct mesh row"
                num_rows = device.shape[0]
                users_per_row = getattr(self.args, "max_local_batch_size", self.args.max_batch_size // num_rows)
                target_row = global_user_id // users_per_row

                # Create page table with -1 (invalid) for all rows except target
                full_page_table = torch.full((num_rows, page_table.shape[1]), -1, dtype=page_table.dtype)
                full_page_table[target_row] = page_table[0]

                tt_page_table = ttnn.from_torch(
                    full_page_table,
                    device=device,
                    mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=(0, None), mesh_shape=device.shape),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            else:
                # Single-user prefill or non-row-sharded: replicate page table
                tt_page_table = ttnn.from_torch(
                    page_table,
                    device=device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(device),
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

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False, is_log_probs=False):
        """Process decode output and convert to torch tensors"""
        concat_out = self.concat_device_output(tt_out)
        if is_tokens or is_log_probs:
            # Token IDs or log probs: shape [1, 1, B] or [1, 1, 1, B] -> [B]
            return concat_out.reshape(-1)[:B]

        torch_out = concat_out[:, 0, :, :]  # [1, 1, B, vocab_size]
        # TODO: this view is dangerous, forces bad tensor shapes to work but we get garbage outputs if they're wrong
        return torch_out.view(B, S, -1)

    def concat_device_output(self, tt_out):
        """Convert multi-device tensor to torch tensor"""
        if self.users_row_sharded:
            tt_output_tensor = ttnn.get_device_tensors(tt_out)[:: self.mesh_shape[1]]
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
        num_cols = self.mesh_shape[1]
        device_tensors = ttnn.get_device_tensors(tt_out)
        results = []
        num_rows = self.mesh_shape[0]
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
