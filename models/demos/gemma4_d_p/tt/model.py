# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 Galaxy prefill model with context-parallel ring attention."""


import torch

import ttnn
from models.common.tensor_utils import get_rot_transformation_mat
from models.demos.gemma4_d_p.tt.attention.global_kv_cache import pack_global_rope_device, pack_sliding_rope_device
from models.demos.gemma4_d_p.tt.attention.ring_prefill import ring_cache_capacity
from models.demos.gemma4_d_p.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4_d_p.tt.rms_norm import RMSNorm
from models.demos.gemma4_d_p.utils.general_utils import cast_host_for_ttnn, get_cache_file_name
from models.demos.gemma4_d_p.utils.substate import substate


def _get_lm_head_program_config(mesh_device, m: int, k: int, n: int):
    """Distribute a token-tile projection across the compute grid with vocabulary shards."""
    tile_size = 32
    grid = mesh_device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y

    m_tiles = max(1, (m + tile_size - 1) // tile_size)
    k_tiles = max(1, k // tile_size)
    n_tiles = max(1, n // tile_size)

    if m_tiles > 1 or n > 64 * 1024:
        return None

    per_core_n = max(1, (n_tiles + num_cores - 1) // num_cores)

    in0_block_w = 32
    while in0_block_w > 1 and k_tiles % in0_block_w != 0:
        in0_block_w //= 2

    out_subblock_w = min(per_core_n, 4)
    while out_subblock_w > 1 and per_core_n % out_subblock_w != 0:
        out_subblock_w -= 1

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _cp_chunk_major_row_order(max_seq_len, cp, chunk_size):
    """Row permutation putting each CP rank's positions in chunk order.

    Multi-chunk CP prefill has a problem the single-chunk case hides. For chunk ``n``
    rank ``r`` owns global positions ``[n*C + r*L, +L)`` with ``L = C/cp``. If the
    RoPE cache is sharded by position, rank ``r`` holds ``[r*max/cp, ...)``, so the
    local index it needs is ``n*C - r*(C - L)`` — rank-dependent, and the model
    slices with a mesh-wide scalar that cannot vary per device.

    Permuting fixes it. Lay row ``m`` out as::

        m = r*(max_seq_len/cp) + n*L + j   holding global position   n*C + r*L + j

    so that a contiguous shard across the CP axis hands rank ``r`` exactly its own
    positions, ordered by chunk. The slice for chunk ``n`` is then ``[n*L, +L)`` on
    every rank — a uniform scalar, which is ``chunk_start_idx // cp``.

    Returns the index array to gather rows by, or None when there is nothing to do.
    """
    if cp <= 1 or not chunk_size:
        return None
    slab = chunk_size // cp
    if slab == 0 or max_seq_len % chunk_size != 0 or chunk_size % cp != 0:
        return None
    num_chunks = max_seq_len // chunk_size
    order = torch.empty(max_seq_len, dtype=torch.long)
    for rank in range(cp):
        for chunk in range(num_chunks):
            local_base = rank * (max_seq_len // cp) + chunk * slab
            global_base = chunk * chunk_size + rank * slab
            order[local_base : local_base + slab] = torch.arange(global_base, global_base + slab)
    return order


def create_rope_caches(mesh_device, hf_config, max_seq_len, mesh_config=None, prefill_chunk_size=None):
    """Create chunk-major CP-sharded RoPE tables and replicated tables for traced position lookup."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.demos.gemma4_d_p.tt.ccl import cp_degree

    is_mesh = hasattr(mesh_device, "shape")
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    cp = cp_degree(mesh_config) if (is_mesh and mesh_config is not None) else 1
    row_order = None
    if cp > 1:
        assert max_seq_len % cp == 0, f"max_seq_len {max_seq_len} must be divisible by CP degree {cp}"
        shard_dims = (-2, None) if mesh_config.sp_axis == 0 else (None, -2)
        prefill_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=shard_dims)
        # Multi-chunk needs the rows reordered so one scalar slice serves every rank;
        # single-chunk (max_seq_len == chunk) is already correct without it.
        row_order = _cp_chunk_major_row_order(max_seq_len, cp, prefill_chunk_size)
    else:
        prefill_mapper = replicate

    rope = Gemma4TextRotaryEmbedding(hf_config)
    x_dummy = torch.randn(1, max_seq_len, hf_config.hidden_size)
    pos_ids = torch.arange(max_seq_len).unsqueeze(0)

    caches_4d = {}
    caches_2d = {}
    for layer_type in set(hf_config.layer_types):
        cos, sin = rope(x_dummy, pos_ids, layer_type=layer_type)
        # cos, sin: [1, max_seq_len, head_dim]
        # Cast to bfloat16 on host so from_torch's requested dtype matches the
        # source: a dtype conversion inside from_torch queries tile metadata on
        # the row-major host intermediate and emits the #18536 warning.
        cos = cos.to(torch.bfloat16)
        sin = sin.to(torch.bfloat16)

        # 4D for prefill: [1, 1, max_seq_len, head_dim].
        # Sharded along positions under CP (see docstring), replicated otherwise.
        cos_prefill, sin_prefill = cos, sin
        if row_order is not None:
            cos_prefill = cos[:, row_order, :]
            sin_prefill = sin[:, row_order, :]
        cos_4d = ttnn.from_torch(
            cos_prefill.unsqueeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=prefill_mapper,
        )
        sin_4d = ttnn.from_torch(
            sin_prefill.unsqueeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=prefill_mapper,
        )
        caches_4d[layer_type] = (cos_4d, sin_4d)

        # Replicated 2D tables support per-rank position lookup inside traces.
        # Row-major weights let embedding gather only the requested positions.
        cos_2d = ttnn.from_torch(
            cos.squeeze(0),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )
        sin_2d = ttnn.from_torch(
            sin.squeeze(0),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )
        caches_2d[layer_type] = (cos_2d, sin_2d)

    return caches_4d, caches_2d


class Gemma4Model:
    """Galaxy prefill model with ring-cache outputs for disaggregation."""

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
        precision=None,
        # Global prefill chunk size. Only needed under context parallelism with more
        # than one chunk: it sets the RoPE cache's chunk-major row order and sizes the
        # ring KV cache slabs. None means single-chunk prefill.
        prefill_chunk_size=None,
        ring_kv_caches=None,
        prefill_weights_only=False,
    ):
        from models.demos.gemma4_d_p.config import validate_galaxy_mesh

        validate_galaxy_mesh(mesh_device.shape)
        if mesh_config is None or mesh_config.mesh_shape != tuple(mesh_device.shape):
            raise ValueError("Galaxy prefill requires a matching mesh_config")
        if prefill_chunk_size is None:
            prefill_chunk_size = min(8192, max_seq_len)
        if max_seq_len <= 0 or prefill_chunk_size <= 0:
            raise ValueError("sequence and chunk lengths must be positive")
        if max_seq_len % prefill_chunk_size or prefill_chunk_size % (mesh_config.prefill.sp * ttnn.TILE_SIZE):
            raise ValueError("prefill chunks must divide max_seq_len and contain whole CP-local tiles")
        if prefill_chunk_size < 1024 * mesh_config.prefill.sp:
            raise ValueError("prefill chunk size must cover the sliding window on each CP rank")
        self.prefill_weights_only = prefill_weights_only
        self.lm_head_weight = None
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.prefill_chunk_size = prefill_chunk_size
        self.ring_cache_max_seq_len = ring_cache_capacity(max_seq_len, prefill_chunk_size)
        self.mesh_config = mesh_config
        self.hidden_size = hf_config.hidden_size
        self.vocab_size = hf_config.vocab_size
        self.final_logit_softcapping = hf_config.final_logit_softcapping
        self.embed_scale = hf_config.hidden_size**0.5
        self.ccl_manager = ccl_manager
        self._rope_prefill_positions = None
        self._packed_global_rope_trans_mat = None
        if mesh_config is not None and mesh_config.prefill.sp > 1:
            self._packed_global_rope_trans_mat = ttnn.from_torch(
                get_rot_transformation_mat(),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        # When True the caller refreshes the ring metadata itself, outside any trace.
        self._ring_metadata_external = False
        self._prefill_trace_controller = None
        self.max_seq_len = max_seq_len
        n_layers = num_layers or hf_config.num_hidden_layers

        # Per-module dtype resolution. ``precision`` (Gemma4Precision) holds
        # any overrides loaded from precision_overrides.json; modules without
        # an override fall back to ``dtype`` (the model-wide default). Dtypes
        # are then threaded explicitly through DecoderLayer / used directly
        # for embedding + lm_head, so each weight loads at the right precision
        # and lands in a cache file tagged with that dtype.
        from models.demos.gemma4_d_p.tt.precision import Gemma4Precision

        if precision is None:
            precision = Gemma4Precision()
        mlp_dtype = precision.get("shared_mlp", dtype)
        attention_dtype = precision.get("attention", dtype)
        embedding_dtype = precision.get("embedding", dtype)
        lm_head_dtype = precision.get("lm_head", dtype)
        # Paged K/V storage, not a weight: it sizes with context rather than with the model,
        # so it is the one tensor whose precision trades against how long a prompt fits.
        kv_cache_dtype = precision.get("kv_cache", dtype)

        # RoPE caches per layer type (sliding vs global)
        # Needs real HF text config (set by create_tt_model via _hf_text_config)
        hf_text_config = getattr(hf_config, "_hf_text_config", None)
        if hf_text_config is not None:
            self.rope_caches, self.rope_caches_2d = create_rope_caches(
                mesh_device,
                hf_text_config,
                max_seq_len,
                mesh_config=self.mesh_config,
                prefill_chunk_size=prefill_chunk_size,
            )
        else:
            # Fallback: no automatic RoPE — caller must pass rope_mats explicitly
            self.rope_caches = {}
            self.rope_caches_2d = {}

        # Embedding
        is_mesh = hasattr(mesh_device, "shape")
        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
        tp = mesh_config.tp if mesh_config else 1
        tp_suffix = f"_tp{tp}" if tp > 1 else ""

        from models.demos.gemma4_d_p.tt.precision import dtype_to_str

        if state_dict and "model.language_model.embed_tokens.weight" in state_dict:
            embed_key = "model.language_model.embed_tokens.weight"
        elif state_dict and "model.embed_tokens.weight" in state_dict:
            embed_key = "model.embed_tokens.weight"
        else:
            embed_key = None

        if embed_key and state_dict:
            embed_weight = state_dict[embed_key]

            # Embedding: column-parallel (shard hidden dim across TP devices)
            # Each device holds [vocab, hidden/TP]; all-gather after lookup.
            if tp > 1:
                embed_mapper = mesh_config.column_parallel(mesh_device)
            else:
                embed_mapper = replicate
            embed_suffix = f"_{dtype_to_str(embedding_dtype)}"
            self.embedding_weight = ttnn.as_tensor(
                cast_host_for_ttnn(embed_weight.unsqueeze(0).unsqueeze(0), embedding_dtype),
                device=mesh_device,
                dtype=embedding_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=embed_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, f"embed_tokens.weight{tp_suffix}{embed_suffix}"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            if not prefill_weights_only:
                # LM head (tied with embeddings): column-parallel (shard vocab dim)
                # Each device holds [hidden, vocab/TP]; all-gather logits after softcapping.
                # Default is bfloat16 — bfloat8_b is generally too lossy for 262k-vocab
                # argmax, but the override is exposed for systems that genuinely
                # need the DRAM relief and can tolerate the precision loss.
                lm_head_weight = embed_weight.transpose(0, 1).unsqueeze(0).unsqueeze(0)
                if tp > 1:
                    lm_mapper = mesh_config.column_parallel(mesh_device)
                else:
                    lm_mapper = replicate
                lm_head_suffix = f"_{dtype_to_str(lm_head_dtype)}"
                self.lm_head_weight = ttnn.as_tensor(
                    lm_head_weight,
                    device=mesh_device,
                    dtype=lm_head_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=lm_mapper,
                    cache_file_name=get_cache_file_name(
                        tensor_cache_path, f"lm_head.weight{tp_suffix}{lm_head_suffix}"
                    ),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
        else:
            self.embedding_weight = None
            self.lm_head_weight = None

        # Each layer owns a ring cache unless the caller supplies one.
        self.layers = []
        if ring_kv_caches is not None and len(ring_kv_caches) != n_layers:
            raise ValueError(f"expected {n_layers} external ring caches, got {len(ring_kv_caches)}")
        for i in range(n_layers):
            layer = Gemma4DecoderLayer(
                mesh_device=mesh_device,
                hf_config=hf_config,
                state_dict=state_dict,
                layer_idx=i,
                ccl_manager=ccl_manager,
                dtype=dtype,
                mlp_dtype=mlp_dtype,
                attention_dtype=attention_dtype,
                tensor_cache_path=tensor_cache_path,
                mesh_config=mesh_config,
                max_seq_len=self.ring_cache_max_seq_len,
                max_local_batch_size=max_local_batch_size,
                ring_kv_cache=(ring_kv_caches[i] if ring_kv_caches is not None else None),
            )
            self.layers.append(layer)

        self.tt_kv_cache = [layer.self_attn.ring_kv_cache for layer in self.layers]

        self.norm = None
        if not prefill_weights_only:
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

    def _get_rope_mats(self, layer_idx, seq_len=None, start_pos=0):
        """Slice chunk-major RoPE caches using a CP-local row offset."""
        cos, sin = self.rope_caches[self.hf_config.layer_types[layer_idx]]
        if seq_len is not None:
            cos = cos[:, :, start_pos : start_pos + seq_len, :]
            sin = sin[:, :, start_pos : start_pos + seq_len, :]
        return cos, sin

    def set_prefill_trace_controller(self, controller):
        """Attach the segmented trace controller used for per-layer migration acks."""
        self._prefill_trace_controller = controller

    def set_prefill_rope_positions(self, position_idx):
        """Set the CP-sharded absolute positions that the caller updates before each replay."""
        self._rope_prefill_positions = position_idx

    def __call__(
        self,
        hidden_states,
        rope_mats=None,
        user_id=0,
        chunk_start_idx=0,
        on_layer_complete=None,
        d2h_service=None,
        metadata_msg=None,
    ):
        """Prefill one user's chunk and return hidden states.

        With prefill_weights_only, omit the output norm and LM-head weights.
        Otherwise the caller runs the LM head on the final token after the last
        chunk. Migration acknowledgements follow each layer's KV writes.
        """
        seq_len = hidden_states.shape[2]
        if hidden_states.shape[0] != 1 or hidden_states.shape[1] != 1:
            raise ValueError("Ring prefill processes one user per call")
        if d2h_service is not None and metadata_msg is None:
            raise ValueError("metadata_msg is required for D2H layer acknowledgements")
        if not self._ring_metadata_external:
            self.ccl_manager.set_ring_metadata(slot_idx=user_id, kv_actual_global=chunk_start_idx)

        gathered_rope = {}
        if rope_mats is None and self._rope_prefill_positions is not None:
            for layer_type in set(self.hf_config.layer_types[: len(self.layers)]):
                cos, sin = self.rope_caches_2d[layer_type]
                gathered_rope[layer_type] = (
                    ttnn.unsqueeze_to_4D(ttnn.embedding(self._rope_prefill_positions, cos, layout=ttnn.TILE_LAYOUT)),
                    ttnn.unsqueeze_to_4D(ttnn.embedding(self._rope_prefill_positions, sin, layout=ttnn.TILE_LAYOUT)),
                )

        packed_rope_by_type = {}
        for i, layer in enumerate(self.layers):
            layer_type = self.hf_config.layer_types[i]
            if rope_mats is not None:
                layer_rope = rope_mats[layer_type] if isinstance(rope_mats, dict) else rope_mats
            elif gathered_rope:
                layer_rope = gathered_rope[layer_type]
            else:
                layer_rope = self._get_rope_mats(
                    i, seq_len=seq_len, start_pos=chunk_start_idx // self.mesh_config.prefill.sp
                )
            if layer_type not in packed_rope_by_type and self._packed_global_rope_trans_mat is not None:
                pack_rope = pack_global_rope_device if layer_type == "full_attention" else pack_sliding_rope_device
                packed_rope_by_type[layer_type] = (*pack_rope(*layer_rope), self._packed_global_rope_trans_mat)
            packed_rope = packed_rope_by_type.get(layer_type)

            hidden_states = layer(
                hidden_states,
                rope_mats=layer_rope,
                chunk_start_idx=chunk_start_idx,
                packed_global_rope=packed_rope if layer_type == "full_attention" else None,
                packed_sliding_rope=packed_rope if layer_type == "sliding_attention" else None,
            )
            if self._ring_metadata_external:
                from models.demos.gemma4_d_p.tt.attention.ring_prefill import zero_ring_cache_padding

                zero_ring_cache_padding(
                    self.tt_kv_cache[i], self.ccl_manager, self.mesh_config, self.prefill_chunk_size
                )
            if d2h_service is not None:
                ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2h_service, metadata=metadata_msg)
            elif on_layer_complete is not None:
                if self._prefill_trace_controller is not None:
                    self._prefill_trace_controller.layer_ack(i)
                else:
                    ttnn.synchronize_device(self.mesh_device)
                    on_layer_complete(i)
        return hidden_states if self.prefill_weights_only else self.norm.forward(hidden_states)

    def _cp_gather_prefill_sequence(self, hidden_states):
        """Gather a chunk across CP ranks without freeing the caller-owned hidden states."""
        from models.demos.gemma4_d_p.tt.ccl import cp_degree

        if cp_degree(self.mesh_config) <= 1:
            return hidden_states
        return ttnn.all_gather(
            hidden_states,
            dim=2,
            cluster_axis=self.mesh_config.sp_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _apply_lm_head(self, hidden_states):
        """Project a token tile to logits, apply softcapping, and gather the vocabulary."""
        from models.demos.gemma4_d_p.tt.ccl import ccl_allgather

        if self.lm_head_weight is None:
            raise RuntimeError("LM head weights not loaded")
        program_config = _get_lm_head_program_config(
            self.mesh_device, m=hidden_states.shape[2], k=self.hidden_size, n=self.lm_head_weight.shape[-1]
        )
        logits = ttnn.linear(hidden_states, self.lm_head_weight, program_config=program_config)
        hidden_states.deallocate(True)
        if self.final_logit_softcapping and self.final_logit_softcapping > 0:
            cap = self.final_logit_softcapping
            logits = ttnn.mul(logits, 1.0 / cap)
            logits = ttnn.tanh(logits)
            logits = ttnn.mul(logits, cap)
        return ccl_allgather(logits, self.mesh_config, self.ccl_manager)

    def embed_tokens(self, tokens):
        """Embed input tokens and scale by sqrt(hidden_size).

        Embedding is column-parallel (hidden dim sharded across TP devices).
        All-gather reconstructs full hidden dim after lookup.
        """
        if self.embedding_weight is None:
            raise RuntimeError("Embedding weights not loaded")
        embeds = ttnn.embedding(tokens, self.embedding_weight, dtype=ttnn.bfloat16)
        embeds = ttnn.mul(embeds, self.embed_scale)

        # All-gather sharded hidden dim back to full hidden
        if self.mesh_config is not None and self.mesh_config.tp > 1:
            embeds = ttnn.unsqueeze_to_4D(embeds)
            from models.demos.gemma4_d_p.tt.ccl import ccl_allgather

            embeds = ccl_allgather(embeds, self.mesh_config, self.ccl_manager)
        return embeds

    # ── Generator-compatible interface ────────────────────────────────────

    def _reshape_prefill_embeds(self, tt_embeds, seq_len):
        if len(tt_embeds.shape) == 3:
            return ttnn.reshape(tt_embeds, (1, 1, seq_len, self.hidden_size))
        if tt_embeds.shape[2] != seq_len:
            return ttnn.reshape(tt_embeds, (1, 1, seq_len, self.hidden_size))
        return tt_embeds

    def transform_and_embed_prefill_inputs_device(self, tokens):
        """Embed CP-sharded tokens into tiled hidden states inside the prefill trace."""
        seq_len = tokens.shape[-1]
        if len(tokens.shape) == 4:
            tokens = ttnn.reshape(tokens, (1, seq_len))
        embeds = self._reshape_prefill_embeds(self.embed_tokens(tokens), seq_len)
        return ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)

    def process_output_prefill(self, tt_out, last_token_idx):
        """Read prefill logits to host and slice to the last token's vocab row.

        Under TP, Gemma4 all-gathers logits inside the model so a single
        device tensor already holds the full vocab.
        """
        if self.mesh_config is not None and self.mesh_config.tp > 1:
            torch_output = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
        else:
            torch_output = ttnn.to_torch(tt_out)
        return torch_output[..., last_token_idx, : self.vocab_size]

    def process_logits_after_prefill_trace(self, hidden_states, last_token_idx, *, chunk_start_idx=None):
        """Project a token tile without freeing the trace output.

        By default last_token_idx is a physical chunk row. With chunk_start_idx,
        it is an absolute token position in a rotated chunk.
        """
        if chunk_start_idx is not None:
            from models.demos.common.prefill.chunk_layout import chunk_row_for_position

            last_token_idx = chunk_row_for_position(
                last_token_idx, chunk_start_idx, self.prefill_chunk_size, self.mesh_config.prefill.sp
            )
        gathered = self._cp_gather_prefill_sequence(hidden_states)
        tile_start = (last_token_idx // 32) * 32
        sliced = ttnn.slice(gathered, (0, 0, tile_start, 0), (1, 1, tile_start + 32, gathered.shape[-1]))
        if gathered is not hidden_states and gathered is not sliced:
            gathered.deallocate(True)
        return self._apply_lm_head(sliced)
