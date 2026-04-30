# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger

import ttnn
from models.common.sampling.generator import SamplingGenerator
from models.common.utility_functions import nearest_32
from models.demos.gpt_oss.config import MeshConfig, Mode, ModeConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss.utils.substate import substate
from models.tt_transformers.tt.common import copy_host_to_device, rope_scaling_model_factory
from models.tt_transformers.tt.rope import RotarySetup

from .layer import DecoderLayer
from .rms_norm import RMSNorm


def compute_per_device_vocab(vocab_size, num_tp):
    """Compute per-device vocab width: tile-aligned then rounded to next power of 2.

    The power-of-2 rounding enables ttnn.topk's multi-core path (bitonic sort
    requires power-of-2 width). Without it, topk falls back to single-core.

    This must be used consistently for both lm_head weight padding and sampling
    args so device shard boundaries match TTSampling device offset strides.
    """
    per_device = (((vocab_size + num_tp - 1) // num_tp + 31) // 32) * 32
    return 1 << (per_device - 1).bit_length()  # next power of 2


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

        if state_dict:
            embedding_weight = substate(state_dict, "model.embed_tokens")["weight"]
            embedding_weight = embedding_weight.unsqueeze(0).unsqueeze(0)
        else:
            embedding_weight = None

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
                tokens_per_device=max_local_batch_size,
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
        # Pad lm_head vocab dimension to padded_vocab_size BEFORE column-parallel sharding.
        # TTSampling._create_indices_tensors uses padded_per_device as the stride for device
        # offset calculation: global_idx = device_id * padded_per_device + local_idx.
        # If we shard at unpadded boundaries and pad after, the offsets are wrong for devices 1+.
        # Pre-sharding padding ensures device shard boundaries match the offset stride.
        # Round per-device width to next power of 2 so ttnn.topk can use its multi-core path
        # (bitonic sort requires power-of-2 width). Without this, topk falls back to single-core
        # and takes ~14ms instead of being parallelized across many cores.
        sampling_splits = mesh_device.shape[1]
        per_device_padded = compute_per_device_vocab(self.vocab_size, sampling_splits)
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
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            cache_file_name=get_cache_file_name(tensor_cache_path, "lm_head_padded_pow2.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_config.column_parallel(mesh_device),
        )

        # Initialize on-device sampling (supported when padded per-device vocab fits in 64K)
        self._supports_on_device_sampling = per_device_padded <= 64 * 1024
        self._prefill_sampling_active = False
        # sampling_dp: number of independent sampling groups (one per mesh row for row-sharded users)
        self.sampling_dp = mesh_device.shape[0] if users_row_sharded else 1
        if self._supports_on_device_sampling:
            # tt_ccl=None makes TTSampling fall back to ttnn.all_gather() which works on [4,8] meshes
            self.sampling = SamplingGenerator(
                args=self.args if hasattr(self, "args") else self._make_sampling_args(hf_config, mesh_device),
                mesh_device=mesh_device,
                tt_ccl=None,
                enable_internal_trace=False,
            )
            # Hook reset_sampling_params to set prefill flag — Generator calls this
            # before prefill forward; tells _forward_layers_and_head to skip TP all-gather
            _orig_reset = self.sampling.reset_sampling_params

            def _reset_with_flag(params, _orig=_orig_reset, **kwargs):
                _orig(params, **kwargs)
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
        num_tp = mesh_device.shape[1]
        per_device_vocab = compute_per_device_vocab(args.vocab_size, num_tp)
        args.padded_vocab_size = per_device_vocab * num_tp
        args.cluster_shape = tuple(mesh_device.shape)
        args.sampling_all_gather_axis = 1
        args.num_devices = mesh_device.get_num_devices()
        args.is_galaxy = mesh_device.shape[0] > 1
        args.model_config = {}  # No SAMPLING_AG_CONFIG → regular sampling path always used
        # sampling_dp: number of independent sampling groups (one per mesh row)
        # Only use row-sharded sampling when users_row_sharded is active
        args.sampling_dp = self.sampling_dp
        args.use_topk_logprobs = True
        return args

    def _increment_decode_positions_device(self, current_pos, rot_mat_idxs):
        """On-device position increment for traced decode loops with sampling."""
        ttnn.plus_one(current_pos, skip_negative_entries=True)
        ttnn.plus_one(rot_mat_idxs)

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
        sampling_on_device=False,
        batch_size=1,
        skip_lm_head=False,
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

        if skip_lm_head:
            return hidden_states

        # Final norm and lm_head
        hidden_states = self.norm(hidden_states)
        logits = ttnn.matmul(hidden_states, self.lm_head_weight, dtype=ttnn.bfloat8_b)
        hidden_states.deallocate(True)
        self._prefill_sampling_active = False
        # TP all-gather is deferred to process_output_prefill / process_output_decode
        # (outside trace capture) since all_gather_async writes to device,
        # which is forbidden during trace capture.

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
        actual_batch = current_pos.shape[-1]
        if not self.users_row_sharded and tokens.shape[-1] > actual_batch:
            tokens_for_embed = tokens[:, :, :, :actual_batch]
        else:
            tokens_for_embed = tokens
        input_embeds = ttnn.embedding(
            tokens_for_embed, self.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b
        )
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
        skip_lm_head=False,
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
            skip_lm_head=skip_lm_head,
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

    def prepare_row_sharded_prefill_iter(
        self, tokens, page_table, prompt_lens, iter_idx, max_padded_len, max_num_blocks, users_per_row_per_iter=1
    ):
        """Prepare one iteration of row-sharded batched prefill.

        Packs users_per_row_per_iter (upr) users per mesh row into the seq dim so a
        single forward pass processes num_rows * upr users. tokens tensor comes back
        shaped (num_rows, upr * max_padded_len); page_table is stacked at batch dim
        as (num_rows * upr, max_num_blocks) for per-user paged_fill_cache.
        """
        num_rows = self.mesh_device.shape[0]
        users_per_row_total = len(prompt_lens) // num_rows
        user_indices = []
        for row in range(num_rows):
            for u in range(users_per_row_per_iter):
                flat_uid = row * users_per_row_total + iter_idx * users_per_row_per_iter + u
                user_indices.append(flat_uid if flat_uid < len(prompt_lens) else row * users_per_row_total)
        tokens_list, pt_list, last_idxs = [], [], []
        for uid in user_indices:
            plen = int(prompt_lens[uid])
            toks = torch.cat(
                [tokens[uid : uid + 1, :plen], torch.zeros(1, max_padded_len - plen, dtype=tokens.dtype)], dim=-1
            )
            tokens_list.append(toks)
            pt_list.append(page_table[uid : uid + 1, :max_num_blocks])
            last_idxs.append(plen - 1)
        # tokens: (num_rows*upr, max_padded_len) -> (num_rows, upr*max_padded_len) packed along seq
        tokens_packed = torch.cat(tokens_list, dim=0).reshape(num_rows, -1)
        pt_stacked = torch.cat(pt_list, dim=0)
        return tokens_packed, pt_stacked, last_idxs, user_indices

    def run_row_sharded_prefill_forward(
        self, tokens_iter, pt_iter, kv_cache, fixed_glt, skip_lm_head=False, batch_size=1
    ):
        """Run one non-traced row-sharded prefill iteration."""
        host_out = self.prepare_inputs_prefill(tokens_iter, page_table=pt_iter, batched_prefill=True)
        return self.ttnn_prefill_forward(
            host_out[0],
            rot_mats_global=host_out[1],
            rot_mats_local=host_out[2],
            user_id=0,
            page_table=host_out[3],
            get_last_token=fixed_glt,
            kv_cache=kv_cache,
            batch_size=batch_size,
            skip_lm_head=skip_lm_head,
        )

    def extract_prefill_logits_to_host(
        self, tt_logits, last_idxs, user_indices, fixed_glt, output_tensor, users_per_row_per_iter=1
    ):
        """Extract per-row TP-gathered logits to host output_tensor.

        When users_per_row_per_iter > 1, each row's device tensor holds upr users'
        logits concatenated along seq. With get_last_token != -1 the forward already
        sliced out a 32-token tile per user, so user u's logit sits at index
        u*32 + (last_idx % 32); with get_last_token == -1 no slicing was done and
        user u's logit is at u*max_padded_len + last_idx.
        """
        device_tensors = ttnn.get_device_tensors(tt_logits)
        nc = self.mesh_device.shape[1]
        num_rows = self.mesh_device.shape[0]
        for row_idx in range(num_rows):
            device_base = row_idx * nc
            row_logits = [ttnn.to_torch(device_tensors[device_base + col]) for col in range(nc)]
            torch_output = torch.cat(row_logits, dim=-1)
            per_user_stride = 32 if fixed_glt != -1 else torch_output.shape[-2] // users_per_row_per_iter
            for u in range(users_per_row_per_iter):
                flat_idx = row_idx * users_per_row_per_iter + u
                uid = user_indices[flat_idx]
                if uid >= output_tensor.shape[0]:
                    continue
                pos_within = last_idxs[flat_idx] % 32 if fixed_glt != -1 else last_idxs[flat_idx]
                global_pos = u * per_user_stride + pos_within
                output_tensor[uid, 0] = torch_output[..., global_pos, : self.vocab_size].view(-1)

    def clear_kv_caches(self):
        """Clear all KV caches (guard against None for vLLM)."""
        for layer_obj in self.layers:
            lp = getattr(layer_obj.self_attn, "layer_past", None)
            if lp is not None:
                ttnn.mul(lp[0], 0, output_tensor=lp[0])
                ttnn.mul(lp[1], 0, output_tensor=lp[1])

    def row_sharded_batched_prefill(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        prefill_seq_lens,
        enable_trace=True,
        sampling_params=None,
        model_args=None,
        trace_cache=None,
    ):
        """Row-parallel batched prefill: 1 user per row per iteration."""
        from models.tt_transformers.tt.common import copy_host_to_device, get_block_size, num_blocks_in_seq

        mesh_device = self.mesh_device
        num_rows = mesh_device.shape[0]
        batch_size = len(prompt_lens)
        actual_batch_size = batch_size
        # upr (users-per-row-per-iter): pack 8 short-prompt users per row per
        # iter to amortise trace cost; long-prompt batches already saturate the
        # device so packing would exhaust DRAM. Below 16 users total it is not
        # worth padding up to 32 just to use upr=8.
        max_seq = max(prefill_seq_lens) if prefill_seq_lens else 128
        upr = 8 if (max_seq <= 128 and batch_size > 16) else 1
        # Pad batch up to multiple of (num_rows * upr) so num_iters is integer.
        align = num_rows * upr
        if batch_size % align != 0:
            pad_count = align - (batch_size % align)
            tokens = torch.cat([tokens, torch.zeros(pad_count, tokens.shape[1], dtype=tokens.dtype)], dim=0)
            prompt_lens = list(prompt_lens) + [int(prompt_lens[0])] * pad_count
            prefill_seq_lens = list(prefill_seq_lens) + [prefill_seq_lens[0]] * pad_count
            if page_table is not None:
                page_table = torch.cat([page_table, page_table[:1].expand(pad_count, -1)], dim=0)
            batch_size += pad_count
        users_per_row = batch_size // num_rows
        num_iters = users_per_row // upr

        max_padded_len = max(prefill_seq_lens)
        block_size = get_block_size(kv_cache)
        max_num_blocks = num_blocks_in_seq(max_padded_len, block_size)
        all_last_idxs = [int(prompt_lens[uid]) - 1 for uid in range(batch_size)]
        fixed_glt = (min(all_last_idxs) // 32) * 32
        if (max(all_last_idxs) // 32) * 32 != fixed_glt:
            fixed_glt = -1
        skip_lm = sampling_params is not None

        vocab_size = model_args.vocab_size if model_args else self.vocab_size
        output_tensor = torch.zeros(batch_size, 1, vocab_size)
        trace_key = "rsbp_" + str(max_padded_len) + "_upr" + str(upr) + ("_nolm" if skip_lm else "_lm")
        # Only trace short sequences; longer prefills are one-shot per call so trace
        # capture cost is not amortised, and capture itself can blow trace memory.
        enable_trace_current = (
            enable_trace and max_padded_len <= 4096 and model_args.can_enable_trace(max_padded_len, 0)
        )

        tc_ids = trace_cache["ids"]
        tc_inputs = trace_cache["inputs"]
        tc_outputs = trace_cache["outputs"]

        if enable_trace_current:
            if tc_ids[trace_key] is None:
                # Compile
                t0, p0, l0, u0 = self.prepare_row_sharded_prefill_iter(
                    tokens, page_table, prompt_lens, 0, max_padded_len, max_num_blocks, users_per_row_per_iter=upr
                )
                ho = self.prepare_inputs_prefill(t0, page_table=p0, trace_enabled=True, batched_prefill=True)
                rot_g, rot_l = ho[1], ho[2]
                hi = (ho[0], ho[3], ho[4])
                di = copy_host_to_device(hi, mesh_device=mesh_device)
                tr = self.transform_and_embed_prefill_inputs_device(*di)
                tt_out = self.ttnn_prefill_forward(
                    tr[0],
                    rot_mats_global=rot_g,
                    rot_mats_local=rot_l,
                    user_id=0,
                    page_table=tr[1],
                    get_last_token=fixed_glt,
                    kv_cache=kv_cache,
                    batch_size=upr,
                    skip_lm_head=skip_lm,
                )
                if not skip_lm:
                    self.extract_prefill_logits_to_host(
                        tt_out, l0, u0, fixed_glt, output_tensor, users_per_row_per_iter=upr
                    )
                ttnn.synchronize_device(mesh_device)
                self.clear_kv_caches()
                # Capture trace
                ho = self.prepare_inputs_prefill(t0, page_table=p0, trace_enabled=True, batched_prefill=True)
                hi = (ho[0], ho[3], ho[4])
                di = copy_host_to_device(hi, mesh_device=mesh_device)
                tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                tr = self.transform_and_embed_prefill_inputs_device(*di)
                tt_out = self.ttnn_prefill_forward(
                    tr[0],
                    rot_mats_global=rot_g,
                    rot_mats_local=rot_l,
                    user_id=0,
                    page_table=tr[1],
                    get_last_token=fixed_glt,
                    kv_cache=kv_cache,
                    batch_size=upr,
                    skip_lm_head=skip_lm,
                )
                ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
                ttnn.synchronize_device(mesh_device)
                tc_ids[trace_key] = tid
                tc_inputs[trace_key] = di
                tc_outputs[trace_key] = tt_out
                self.clear_kv_caches()

            for iter_idx in range(num_iters):
                ti, pi, li, ui = self.prepare_row_sharded_prefill_iter(
                    tokens,
                    page_table,
                    prompt_lens,
                    iter_idx,
                    max_padded_len,
                    max_num_blocks,
                    users_per_row_per_iter=upr,
                )
                ho = self.prepare_inputs_prefill(ti, page_table=pi, trace_enabled=True, batched_prefill=True)
                hi = (ho[0], ho[3], ho[4])
                copy_host_to_device(hi, device_tensors=tc_inputs[trace_key], mesh_device=mesh_device)
                ttnn.execute_trace(mesh_device, tc_ids[trace_key], cq_id=0, blocking=False)
                if not skip_lm:
                    self.extract_prefill_logits_to_host(
                        tc_outputs[trace_key], li, ui, fixed_glt, output_tensor, users_per_row_per_iter=upr
                    )
        else:
            for iter_idx in range(num_iters):
                ti, pi, li, ui = self.prepare_row_sharded_prefill_iter(
                    tokens,
                    page_table,
                    prompt_lens,
                    iter_idx,
                    max_padded_len,
                    max_num_blocks,
                    users_per_row_per_iter=upr,
                )
                tt_out = self.run_row_sharded_prefill_forward(
                    ti, pi, kv_cache, fixed_glt, skip_lm_head=skip_lm, batch_size=upr
                )
                if not skip_lm:
                    self.extract_prefill_logits_to_host(
                        tt_out, li, ui, fixed_glt, output_tensor, users_per_row_per_iter=upr
                    )

        ttnn.synchronize_device(mesh_device)
        if sampling_params is not None:
            # GPT-OSS always emits <|channel|> (token 200005) as the first generated token
            # regardless of prompt content. Skipping lm_head and returning this hardcoded
            # token saves ~100ms/user by avoiding the 201K-vocab matmul. vLLM device
            # sampling accepts this as a pre-sampled token ID.
            CHANNEL_TOKEN_ID = 200005
            return torch.full((actual_batch_size, 1), CHANNEL_TOKEN_ID, dtype=torch.int64), torch.zeros(
                actual_batch_size, 1, dtype=torch.float32
            )
        return output_tensor[:actual_batch_size]

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

        # Pad token buffer to 32 for non-row-sharded b<32 (TTSampling requirement)
        if not self.users_row_sharded and tokens.view(-1).shape[-1] < 32:
            tokens = torch.nn.functional.pad(tokens.view(-1), (0, 32 - len(tokens.view(-1))), "constant", 0)
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
        batch_size=1,
        user_id=0,
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

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False, is_log_probs=False):
        """Process decode output and convert to torch tensors.

        Host-side TP gather for logits: the generator moves output to CPU
        before calling this method, so on-device allgather is not possible.
        """
        if is_tokens or is_log_probs:
            concat_out = self.concat_device_output(tt_out)
            # Token IDs or log probs: shape [1, 1, B] or [1, 1, 1, B] -> [B]
            return concat_out.reshape(-1)[:B]

        # Host-side TP gather: concatenate TP shards per row, then DP rows.
        config = self.mesh_config.get_config(Mode.DECODE)
        if config.tp > 1:
            device_tensors = ttnn.get_device_tensors(tt_out)
            tp = config.tp
            if self.users_row_sharded:
                # TP gather per row, then DP gather across rows (rows carry different users)
                num_rows = len(device_tensors) // tp
                rows = []
                for r in range(num_rows):
                    row_tensors = device_tensors[r * tp : (r + 1) * tp]
                    row_out = torch.cat([ttnn.to_torch(t) for t in row_tensors], dim=-1)
                    rows.append(row_out)
                torch_out = torch.cat(rows, dim=-2) if num_rows > 1 else rows[0]
            else:
                # Rows are EP replicas with identical data; TP-gather first row only
                row_tensors = device_tensors[:tp]
                torch_out = torch.cat([ttnn.to_torch(t) for t in row_tensors], dim=-1)
        else:
            torch_out = self.concat_device_output(tt_out)
        torch_out = torch_out[:, 0, :, :]  # [1, 1, B, padded_vocab_size]
        torch_out = torch_out.view(B, S, -1)
        # Truncate to vocab_size — lm_head is padded to padded_vocab_size for
        # on-device sampling (pow2 topk), but callers expect vocab_size width.
        if torch_out.shape[-1] > self.vocab_size:
            torch_out = torch_out[:, :, : self.vocab_size]
        return torch_out

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
        """Process prefill output and extract last token logits.

        Host-side TP gather: the generator moves logits to CPU before calling
        this method, so on-device allgather is not possible here.
        """
        config = self.mesh_config.get_config(Mode.PREFILL)
        if config.tp > 1:
            device_tensors = ttnn.get_device_tensors(tt_out)
            tp = config.tp
            torch_output = torch.cat([ttnn.to_torch(device_tensors[i]) for i in range(tp)], dim=-1)
        else:
            torch_output = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
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
            device_idx = row * num_cols
            dev_out = device_tensors[device_idx]
            if users_per_row > 1:
                # Batch all user slices on device, single D2H per row
                slices = []
                for u in range(users_per_row):
                    user_flat_idx = row * users_per_row + u
                    last_idx = last_token_idxs[user_flat_idx] if isinstance(last_token_idxs, list) else last_token_idxs
                    global_idx = u * seq_len_per_user + last_idx
                    sl = ttnn.slice(dev_out, (0, 0, global_idx, 0), (1, 1, global_idx + 1, dev_out.shape[-1]))
                    slices.append(sl)
                batched = ttnn.concat(slices, dim=2)
                for sl in slices:
                    sl.deallocate(True)
                torch_out = ttnn.to_torch(batched)
                batched.deallocate(True)
                for u in range(users_per_row):
                    results.append(torch_out[..., u, : self.vocab_size])
            else:
                last_idx = last_token_idxs[row] if isinstance(last_token_idxs, list) else last_token_idxs
                token_logit = ttnn.slice(dev_out, (0, 0, last_idx, 0), (1, 1, last_idx + 1, dev_out.shape[-1]))
                result = ttnn.to_torch(token_logit)[..., : self.vocab_size]
                token_logit.deallocate(True)
                results.append(result)
        return results
