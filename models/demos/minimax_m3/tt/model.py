# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger

import ttnn
from models.common.sampling.generator import SamplingGenerator
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.utils.general_utils import get_cache_file_name
from models.demos.minimax_m3.utils.substate import substate
from models.tt_transformers.tt.common import rope_scaling_model_factory
from models.tt_transformers.tt.rope import RotarySetup

from .layer import DecoderLayer
from .parallel_embedding import EMBED_CACHE_NAME, TtParallelEmbedding
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
    Create and return a RotarySetup instance for the MiniMax-M3 model.

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
    # MiniMax-M3 has no rope_scaling; rope_scaling_model_factory requires a dict,
    # so only build it when params are present, else pass None (plain RoPE).
    rope_scaling_params = getattr(hf_config, "rope_scaling", None)
    rope_scaling = rope_scaling_model_factory(rope_scaling_params) if rope_scaling_params else None
    batch_size = max_local_batch_size * mesh_device.shape[0] if users_row_sharded else max_local_batch_size

    # MiniMax-M3 uses PARTIAL rotary: only the first `rotary_dim` (64) of each
    # 128-wide head is rotated. Build the cos/sin matrices at rotary_dim width;
    # the attention layer rotates [..., :rotary_dim] and passes [..., rotary_dim:]
    # through unchanged (see attention/operations.py:apply_rope).
    rotary_dim = getattr(hf_config, "rotary_dim", hf_config.head_dim)

    rope_setup = RotarySetup(
        device=mesh_device,
        batch_size=batch_size,
        head_dim=rotary_dim,
        max_seq_len=max_seq_len,
        rope_theta=getattr(hf_config, "rope_theta", 150000.0),
        rope_scaling=rope_scaling,
        datatype=datatype,
        shard_batch_to_mesh_dim=shard_batch_to_mesh_dim,
    )

    return rope_setup


class Model:
    """
    MiniMax-M3 TTNN Model Implementation

    This class implements the MiniMax-M3 model using TTNN tensors and operations.
    It supports both prefill and decode modes.

    Key Features:
    - MoE (Mixture of Experts): sigmoid+bias router (top-8), SiLU-SwiGLU experts
    - Full causal GQA attention (no sliding window, no sinks) with partial RoPE + QK-norm
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
        mesh_config=None,
        max_local_batch_size=1,
        users_row_sharded=False,
        use_throughput_experts=False,
        use_ep_moe=False,
        ep_seq_len_per_chip=1024,
        expert_weight_dtype=ttnn.bfloat4_b,
        sequence_parallel=False,
    ):
        """
        Initialize MiniMax-M3 model

        Args:
            mesh_device: TTNN mesh device for computation
            hf_config: HuggingFace model configuration
            state_dict: Model weights dictionary
            ccl_manager: Collective communication manager
            dtype: Data type for tensors (default: bfloat16)
            tensor_cache_path: Path for tensor caching
            mesh_config: Mesh configuration for parallelization
        """
        self.mesh_device = mesh_device
        self.vocab_size = hf_config.vocab_size
        self.hf_config = hf_config
        self.core_grid = ttnn.CoreCoord(8, 8)
        self.head_dim = hf_config.head_dim
        self.max_local_batch_size = max_local_batch_size
        self.users_row_sharded = users_row_sharded
        self.sequence_parallel = sequence_parallel

        self.ccl_manager = ccl_manager

        # Prefill mesh parallelization: TP over the cols; SP and the EP MoE follow from the rows.
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])

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

        # Parallel (sharded) token embedding: the [vocab, hidden] table is sharded on hidden across the
        # TP cols and replicated across the SP rows, so each device stores only [vocab, hidden/tp] —
        # ~1.85 GB/device less than a fully-replicated table. forward() does the local emb-dim-slice
        # lookup then a managed TP all-gather to rebuild the full-hidden, TP-replicated residual stream
        # the decoder layers expect (kept bf16 so the residual stream keeps full dynamic range).
        if state_dict:
            embedding_weight = substate(state_dict, "model.embed_tokens")["weight"]  # [vocab, hidden]
        else:
            embedding_weight = None  # cache-only: the sharded .tensorbin is loaded on a cache hit
        self.embedding = TtParallelEmbedding(
            mesh_device,
            self.vocab_size,
            hf_config.hidden_size,
            self.mesh_config,
            self.ccl_manager,
            torch_weight=embedding_weight,
            cache_file_name=get_cache_file_name(tensor_cache_path, EMBED_CACHE_NAME),
            dtype=ttnn.bfloat16,
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
                mesh_config=self.mesh_config,
                transformation_mats=self.transformation_mats,
                max_local_batch_size=max_local_batch_size,
                users_row_sharded=users_row_sharded,
                use_ep_moe=use_ep_moe,
                ep_seq_len_per_chip=ep_seq_len_per_chip,
                expert_weight_dtype=expert_weight_dtype,
                sequence_parallel=sequence_parallel,
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

    def _forward_layers_and_head(
        self,
        hidden_states,
        rope_mats,
        current_pos,
        get_last_token=-1,
        user_id=0,
        sampling_on_device=False,
        batch_size=1,
        skip_lm_head=False,
        on_layer_complete=None,
        kv_cache=None,
        cached_len=0,
        indexed_rope=False,
    ):
        """
        Prefill forward pass through decoder layers and final projection.

        on_layer_complete: optional callback ``fn(layer_idx)`` invoked after each
            decoder layer finishes. This is the SEAM for per-layer KV migration in
            the prefill/decode-disaggregation pipeline (see PREFILL_PROPOSAL.md §7).
            Default None = no-op (the validated single-process path is unchanged).

        Args:
            hidden_states: Input tensor
            rope_mats: RoPE rotation matrices [cos, sin]
            current_pos: Current position (for decode) or None (for prefill)
            kv_cache: Externally-owned MiniMaxKVCache (packed K/V/index_k), shared across layers.
            cached_len: valid prefix length already in the cache before this chunk (0 = first/only chunk).

        Returns:
            logits: Output logits
        """
        # Process through decoder layers
        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=rope_mats,
                position_idx=current_pos,
                kv_cache=kv_cache,
                user_id=user_id,
                batch_size=batch_size,
                cached_len=cached_len,
                indexed_rope=indexed_rope,
            )
            # Per-layer migration seam (no-op unless a pipeline supplies a callback).
            if on_layer_complete is not None:
                on_layer_complete(i)
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
        # TP all-gather is deferred to process_output_prefill
        # (outside trace capture) since all_gather_async writes to device,
        # which is forbidden during trace capture.

        return logits

    def prefill_forward(
        self,
        x,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
        skip_lm_head=False,
        on_layer_complete=None,
        cached_len=0,
        indexed_rope=False,
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
            kv_cache=kv_cache,
            get_last_token=get_last_token,
            user_id=user_id,
            batch_size=batch_size,
            skip_lm_head=skip_lm_head,
            on_layer_complete=on_layer_complete,
            cached_len=cached_len,
            indexed_rope=indexed_rope,
        )

        return logits

    def process_logits_after_prefill_trace(self, logits, last_token_idx):
        """
        Post-process traced prefill output to the 32-token tile containing `last_token_idx`.

        Unlike tt_transformers `Transformer`, MiniMax-M3 `prefill_forward` already
        applies final norm + lm_head, so this method only slices logits.
        """
        get_last_token = (last_token_idx // 32) * 32
        logits = ttnn.slice(
            logits,
            (0, 0, get_last_token, 0),
            (1, 1, get_last_token + 32, logits.shape[-1]),
        )
        return logits

    def prepare_inputs_prefill(
        self,
        tokens,
        start_pos=0,
        trace_enabled=False,
        last_token_idx=None,
        batch_size=1,
        user_id=0,
        batched_prefill=False,
        **kwargs,
    ):
        """Prepare inputs for prefill mode

        Args:
            batched_prefill: If True, tokens is [num_rows, seq_len] and will be
                sharded across mesh rows. Each row processes a different user.
        """
        # Embed the tokens
        device = None if trace_enabled else self.mesh_device

        if self.sequence_parallel:
            # Sequence-parallel prefill: ONE prompt of seq_len tokens, sharded by SEQUENCE across the
            # SP rows (row r -> tokens [r*s_local:(r+1)*s_local]) and replicated across the TP cols.
            # Each device embeds its 1/sp seq-shard; the residual stream then stays SP-sharded through
            # every layer (attention gathers across the SP ring internally, MoE routes per-row).
            if tokens.dim() == 1:
                tokens = tokens.reshape(1, -1)
            seq_total = tokens.shape[-1]
            sp = self.mesh_device.shape[self.mesh_config.sp_axis]
            assert seq_total % sp == 0, f"SP prefill needs seq_len ({seq_total}) divisible by sp ({sp})"
            tokens = tokens.reshape(1, 1, 1, seq_total)
            tdims = [None, None]
            tdims[self.mesh_config.sp_axis] = 3  # seq dim across SP rows
            tokens = ttnn.from_torch(
                tokens,
                device=device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=tuple(tdims), mesh_shape=self.mesh_device.shape
                ),
            )
        elif batched_prefill:
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
            # bf16 (not bf8) so the residual stream it seeds keeps full dynamic range — bf8's per-tile
            # shared exponent crushes small channels once massive activations appear (see layer.py adds).
            # Sharded lookup + managed TP all-gather -> full-hidden, TP-replicated embedding.
            tokens_embd = self.embedding(tokens)
            tokens.deallocate(True)

            # Ensure proper 4D shape
            if len(tokens_embd.shape) == 3:
                tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)

        # Prepare rotation matrices (slice from rope_setup like tt-transformers model.py lines 156-159)
        seq_len = self.args.max_seq_len if trace_enabled else tokens_embd.shape[-2]
        if self.sequence_parallel and not trace_enabled:
            # Per-row RoPE: the global cos/sin are computed for positions [0:seq_total]; re-shard them
            # across the SP rows so row r rotates its OWN positions [r*s_local:(r+1)*s_local] (matching
            # its token shard), replicated across the TP cols. Re-use the model's own (format-exact)
            # prefill cos/sin rather than rebuilding the Meta-swizzled tables.
            sp = self.mesh_device.shape[self.mesh_config.sp_axis]
            seq_total = seq_len * sp
            rdims = [None, None]
            rdims[self.mesh_config.sp_axis] = 2  # seq dim across SP rows

            def _reshard_rope(dev_tensor):
                full = ttnn.to_torch(ttnn.get_device_tensors(dev_tensor)[0])[:, :, :seq_total, :]
                return ttnn.from_torch(
                    full,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.mesh_device, dims=tuple(rdims), mesh_shape=self.mesh_device.shape
                    ),
                )

            rot_mats_global = [
                _reshard_rope(self.rope_setup.cos_matrix_prefill),
                _reshard_rope(self.rope_setup.sin_matrix_prefill),
            ]
        elif self.sequence_parallel and trace_enabled:
            # Tripwire (do NOT silently fall through to replicated rope): under sequence-parallel the
            # rows are SEQUENCE shards, so row r must rotate positions [r*s_local:(r+1)*s_local]. The
            # reshard above uses a host round-trip and is gated `not trace_enabled`, so tracing an SP
            # prefill would otherwise hit the `else` and rotate EVERY row from position 0 -> global
            # garbage. SP prefill is currently run eagerly; if SP+trace is ever needed, build the
            # SP-sharded cos/sin as a persistent device tensor instead of removing this guard.
            raise NotImplementedError(
                "Sequence-parallel prefill RoPE must be resharded per SP row, but that is not "
                "implemented on the trace path (would replicate position-0 angles to all rows). "
                "Run SP prefill eagerly (trace_enabled=False), or add a persistent SP-sharded cos/sin."
            )
        else:
            # Non-SP, or batched_prefill where each mesh row is a different USER at the SAME positions
            # [0:seq_len]: replicated cos/sin is correct (rows are users, not sequence shards).
            rot_mats_global = [
                self.rope_setup.cos_matrix_prefill[:, :, :seq_len, :],
                self.rope_setup.sin_matrix_prefill[:, :, :seq_len, :],
            ]
        rot_mats_local = None

        return (
            tokens if trace_enabled else tokens_embd,
            rot_mats_global,
            rot_mats_local,
        )

    def process_output_prefill(self, tt_out, last_token_idx):
        """Process prefill output and extract last token logits.

        Host-side TP gather: the generator moves logits to CPU before calling
        this method, so on-device allgather is not possible here.
        """
        tp = self.mesh_config.tp
        if tp > 1:
            device_tensors = ttnn.get_device_tensors(tt_out)
            torch_output = torch.cat([ttnn.to_torch(device_tensors[i]) for i in range(tp)], dim=-1)
        else:
            torch_output = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
        result = torch_output[..., last_token_idx, : self.vocab_size]
        return result
