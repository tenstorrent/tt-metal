# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 Galaxy prefill model with context-parallel ring attention."""

import torch

import ttnn
from models.common.tensor_utils import get_rot_transformation_mat
from models.demos.gemma4_d_p.tt.attention.global_kv_cache import packed_rope_columns
from models.demos.gemma4_d_p.tt.attention.ring_prefill import ring_cache_capacity, ring_sdpa_chunk_sizes
from models.demos.gemma4_d_p.tt.ccl import ccl_allgather, ccl_partition_rows
from models.demos.gemma4_d_p.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4_d_p.tt.precision import dtype_to_str
from models.demos.gemma4_d_p.tt.prefill_metadata import PrefillMetadata
from models.demos.gemma4_d_p.utils.general_utils import get_cache_file_name


def create_packed_rope_tables(mesh_device, hf_config, max_seq_len):
    """Replicated RoPE tables per layer type, for looking up a chunk's positions inside a trace.

    Each table's columns are already in one of the packed RoPE lane orders (packed_rope_columns), so a chunk looks
    them up with ttnn.embedding instead of gathering columns per chunk. Global: Q cos, Q sin, K cos, K sin. Sliding:
    cos, sin. Row-major so that embedding reads only the requested positions.
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    rope = Gemma4TextRotaryEmbedding(hf_config)
    # The rotary module reads only x's device and dtype.
    x_dummy = torch.empty(1, 1, 1)
    pos_ids = torch.arange(max_seq_len).unsqueeze(0)

    tables_by_type = {}
    for layer_type in set(hf_config.layer_types):
        # cos, sin: [1, max_seq_len, head_dim]. Cast to bfloat16 on host so from_torch's requested dtype matches the
        # source: a dtype conversion inside from_torch queries tile metadata on the row-major host intermediate and
        # emits the #18536 warning.
        cos, sin = (t.to(torch.bfloat16).squeeze(0) for t in rope(x_dummy, pos_ids, layer_type=layer_type))
        tables_by_type[layer_type] = tuple(
            ttnn.from_torch(
                table[:, columns].contiguous(),
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
            )
            for columns in packed_rope_columns(layer_type, int(cos.shape[-1]))
            for table in (cos, sin)
        )
    return tables_by_type


def prefill_chunk_geometry_error(prefill_chunk_size, cp_degree, max_seq_len, *, tp_degree):
    """Reason this chunk geometry is unusable, or None. The ring SDPA validates the rest at compile."""
    if max_seq_len <= 0 or prefill_chunk_size <= 0:
        return "sequence and chunk lengths must be positive"
    if max_seq_len % prefill_chunk_size:
        return "prefill chunks must divide max_seq_len"
    # Each chunk is split across CP, then across TP.
    # Each device keeps chunk_size / (CP * TP) token rows,
    # which must be a multiple of 32 so each shard contains whole tiles.
    if prefill_chunk_size % (cp_degree * tp_degree * ttnn.TILE_SIZE):
        return (
            f"prefill chunk {prefill_chunk_size} must be a multiple of CP x TP x {ttnn.TILE_SIZE} = "
            f"{cp_degree * tp_degree * ttnn.TILE_SIZE} for the sequence-parallel residual"
        )
    # Sliding layers step through the per-rank slab in whole K chunks. The chunked sliding SDPA accepts only a
    # 128-token K chunk (ring_joint_sdpa validation), so a slab that is not a multiple of it cannot run.
    slab = prefill_chunk_size // cp_degree
    sliding_k_chunk = ring_sdpa_chunk_sizes(slab, sliding=True)[1]
    if slab % sliding_k_chunk:
        return (
            f"prefill chunk {prefill_chunk_size} gives a {slab}-token slab per CP rank; sliding attention needs a "
            f"multiple of its {sliding_k_chunk}-token K chunk (chunk a multiple of {cp_degree * sliding_k_chunk})"
        )
    return None


class Gemma4Model:
    """Galaxy prefill model with ring-cache outputs for disaggregation."""

    def __init__(
        self,
        mesh_config,
        hf_config,
        state_dict,
        ccl_manager,
        prefill_chunk_size,
        precision,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        max_seq_len=262144,
        max_local_batch_size=1,
        num_layers=None,
        ring_kv_caches=None,
    ):
        assert state_dict and any(
            key.startswith("model.language_model.") for key in state_dict
        ), "Expected a multimodal Gemma4 state_dict with model.language_model.* keys"
        mesh_device = mesh_config.device

        geometry_error = prefill_chunk_geometry_error(
            prefill_chunk_size, mesh_config.cp_degree, max_seq_len, tp_degree=mesh_config.tp_degree
        )
        if geometry_error:
            raise ValueError(geometry_error)

        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.prefill_chunk_size = prefill_chunk_size
        self.ring_cache_max_seq_len = ring_cache_capacity(max_seq_len, prefill_chunk_size)
        self.mesh_config = mesh_config
        self.hidden_size = hf_config.hidden_size
        self.vocab_size = hf_config.vocab_size
        self.embed_scale = hf_config.hidden_size**0.5
        self.ccl_manager = ccl_manager
        self._rope_prefill_positions = None
        self._packed_global_rope_trans_mat = ttnn.from_torch(
            get_rot_transformation_mat(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # When True the caller refreshes the ring metadata itself, outside any trace.
        self.prefill_metadata = PrefillMetadata(mesh_config)
        self._prefill_metadata_external = False
        self._prefill_trace_controller = None
        self.max_seq_len = max_seq_len
        n_layers = num_layers or hf_config.num_hidden_layers

        mlp_dtype = precision.get("shared_mlp", dtype)
        attention_dtype = precision.get("attention", dtype)
        embedding_dtype = precision.get("embedding", dtype)
        kv_cache_dtype = precision.get("kv_cache", dtype)

        # RoPE caches per layer type (sliding vs global)
        # Needs real HF text config (set by create_tt_model via _hf_text_config)
        hf_text_config = getattr(hf_config, "_hf_text_config", None)
        self.packed_rope_tables = (
            create_packed_rope_tables(mesh_device, hf_text_config, max_seq_len) if hf_text_config is not None else {}
        )

        # Embedding
        is_mesh = hasattr(mesh_device, "shape")
        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
        tp = mesh_config.tp_degree
        tp_suffix = f"_tp{tp}" if tp > 1 else ""

        embed_key = "model.language_model.embed_tokens.weight"
        if embed_key in state_dict:
            embed_weight = state_dict[embed_key]

            # Embedding: column-parallel (shard hidden dim across TP devices)
            # Each device holds [vocab, hidden/TP]; all-gather after lookup.
            if tp > 1:
                embed_mapper = mesh_config.column_parallel()
            else:
                embed_mapper = replicate
            embed_suffix = f"_{dtype_to_str(embedding_dtype)}"
            self.embedding_weight = ttnn.as_tensor(
                embed_weight.unsqueeze(0).unsqueeze(0),
                device=mesh_device,
                dtype=embedding_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=embed_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, f"embed_tokens.weight{tp_suffix}{embed_suffix}"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Don't load LM head
        else:
            self.embedding_weight = None

        # Each layer owns a ring cache unless the caller supplies one.
        self.layers = []
        if ring_kv_caches is not None and len(ring_kv_caches) != n_layers:
            raise ValueError(f"expected {n_layers} external ring caches, got {len(ring_kv_caches)}")
        for i in range(n_layers):
            layer = Gemma4DecoderLayer(
                mesh_config=mesh_config,
                hf_config=hf_config,
                state_dict=state_dict,
                layer_idx=i,
                ccl_manager=ccl_manager,
                dtype=dtype,
                mlp_dtype=mlp_dtype,
                attention_dtype=attention_dtype,
                tensor_cache_path=tensor_cache_path,
                max_seq_len=self.ring_cache_max_seq_len,
                max_local_batch_size=max_local_batch_size,
                ring_kv_cache=ring_kv_caches[i] if ring_kv_caches is not None else None,
            )
            self.layers.append(layer)

        self.tt_kv_cache = [layer.self_attn.ring_kv_cache for layer in self.layers]

        # Skip final norm

    def lookup_packed_rope(self, layer_type):
        """This chunk's packed RoPE lanes for layer_type, looked up at the staged positions, with the
        transformation matrix last (the order the attention layer unpacks)."""
        if self._rope_prefill_positions is None:
            raise RuntimeError("prefill needs the chunk's RoPE positions: call set_prefill_rope_positions first")
        if layer_type not in self.packed_rope_tables:
            raise RuntimeError(f"no RoPE tables for {layer_type}: the model was built without _hf_text_config")
        rope_tensors = []
        for table in self.packed_rope_tables[layer_type]:
            values = ttnn.embedding(self._rope_prefill_positions, table, layout=ttnn.TILE_LAYOUT)
            rope_tensors.append(ttnn.unsqueeze_to_4D(values))
        rope_tensors.append(self._packed_global_rope_trans_mat)
        return tuple(rope_tensors)

    def set_prefill_trace_controller(self, controller):
        """Attach the segmented trace controller used for per-layer migration acks."""
        self._prefill_trace_controller = controller

    def set_prefill_rope_positions(self, position_idx):
        """Set the CP-sharded absolute positions that the caller updates before each replay."""
        self._rope_prefill_positions = position_idx

    def __call__(
        self,
        hidden_states,
        user_id=0,
        chunk_start_idx=0,
        on_layer_complete=None,
        d2h_service=None,
        metadata_msg=None,
    ):
        """Prefill one user's chunk and return its final decoder hidden states.

        ``hidden_states`` holds this TP device's 1/TP of the chunk's rows, as
        ``transform_and_embed_prefill_inputs_device`` returns them. The caller owns
        trace staging. Migration acknowledgements follow each layer's KV writes.
        """
        if hidden_states.shape[0] != 1 or hidden_states.shape[1] != 1:
            raise ValueError("Ring prefill processes one user per call")
        if d2h_service is not None and metadata_msg is None:
            raise ValueError("metadata_msg is required for D2H layer acknowledgements")
        if not self._prefill_metadata_external:
            self.prefill_metadata.update(slot_idx=user_id, kv_actual_global=chunk_start_idx)

        # Each layer type's RoPE lanes are looked up once per chunk and shared by all its layers.
        packed_rope_by_type = {
            layer_type: self.lookup_packed_rope(layer_type)
            for layer_type in set(self.hf_config.layer_types[: len(self.layers)])
        }
        for i, layer in enumerate(self.layers):
            layer_type = self.hf_config.layer_types[i]
            packed_rope = packed_rope_by_type[layer_type]
            hidden_states = layer(
                hidden_states,
                prefill_metadata=self.prefill_metadata,
                chunk_start_idx=chunk_start_idx,
                packed_global_rope=packed_rope if layer_type == "full_attention" else None,
                packed_sliding_rope=packed_rope if layer_type == "sliding_attention" else None,
            )
            if d2h_service is not None:
                ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2h_service, metadata=metadata_msg)
            elif on_layer_complete is not None:
                if self._prefill_trace_controller is not None:
                    self._prefill_trace_controller.layer_ack(i)
                else:
                    ttnn.synchronize_device(self.mesh_device)
                    on_layer_complete(i)
        if hidden_states.is_sharded():
            # The layers keep the residual block-sharded between norms.
            sharded = hidden_states
            hidden_states = ttnn.sharded_to_interleaved(sharded, ttnn.DRAM_MEMORY_CONFIG)
            sharded.deallocate(True)
        hidden_states = ccl_allgather(hidden_states, self.mesh_config, self.ccl_manager, dim=2)
        return hidden_states

    def embed_tokens(self, tokens):
        """Embed input tokens and scale by sqrt(hidden_size).

        Embedding is column-parallel (hidden dim sharded across TP devices).
        All-gather reconstructs full hidden dim after lookup; then the tiled
        result keeps this TP device's 1/TP of the rows, which the layers carry.
        """
        if self.embedding_weight is None:
            raise RuntimeError("Embedding weights not loaded")
        # Tile layout out of the lookup: the caller wants tiles, and this skips a separate tilize per chunk.
        embeds = ttnn.embedding(tokens, self.embedding_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        embeds = ttnn.mul(embeds, self.embed_scale)

        # All-gather sharded hidden dim back to full hidden
        if self.mesh_config is not None and self.mesh_config.tp_degree > 1:
            embeds = ttnn.unsqueeze_to_4D(embeds)
            from models.demos.gemma4_d_p.tt.ccl import ccl_allgather

            embeds = ccl_allgather(embeds, self.mesh_config, self.ccl_manager)
        return ccl_partition_rows(embeds, self.mesh_config)

    def transform_and_embed_prefill_inputs_device(self, tokens):
        """Embed CP-sharded tokens into tiled hidden states, keeping this TP device's 1/TP of the rows."""
        assert (
            len(tokens.shape) == 2 and tokens.shape[0] == 1
        ), f"Expected tokens shaped [1, sequence_length], got {tokens.shape}"
        return self.embed_tokens(tokens)
