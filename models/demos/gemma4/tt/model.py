# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Model — full on-device implementation following gpt-oss pattern.

Architecture:
- 30 decoder layers with [5 sliding, 1 full] x 5 pattern
- Two RoPE configs: sliding (head_dim=256, theta=10k) and global (head_dim=512, theta=1M)
- Embedding scaled by sqrt(hidden_size)
- final_logit_softcapping = 30.0
- tie_word_embeddings = True

Supports both prefill and decode modes with paged attention.
Compatible with tt_transformers Generator interface.
"""


import os

import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.sampling.generator import SamplingGenerator
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig, flush_deferred_bounded_fills
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.rms_norm import RMSNorm
from models.demos.gemma4.utils.general_utils import cast_host_for_ttnn, get_cache_file_name
from models.demos.gemma4.utils.substate import substate

# Tracy signpost headers — paired begin/end with the same name. The
# ``models/tt_transformers/scripts/op_perf_results.py --signpost <NAME>``
# tool consumes these to filter the op CSV to a single region. Targets
# from issue #44953: lm_head + sampling ≤ 10% of decode step time and
# sampling alone < 5%. On-device sampling itself runs in the
# tt_transformers Generator (the model returns logits in sampling layout),
# so it is profiled there / by op name (SamplingDeviceOperation, TopK).
LM_HEAD_SIGNPOST = "gemma4_lm_head"


def _compute_per_device_vocab(vocab_size, num_tp):
    """Per-device vocab width: tile-aligned then rounded to next power of 2.

    Power-of-2 rounding enables ttnn.topk's multi-core bitonic sort.
    Must match between LM head weight padding and sampling args.
    """
    per_device = (((vocab_size + num_tp - 1) // num_tp + 31) // 32) * 32
    return 1 << (per_device - 1).bit_length()


def _get_lm_head_program_config(mesh_device, m: int, k: int, n: int):
    """Build a 1D-mcast matmul program config for the LM head.

    LM head shape is [B, 1, H] x [H, V_per_dev] with B padded to 32 for
    decode (so M_tiles=1). Primary target is gemma-4-31B-it on T3K (1x8):
    H=5376 (K_tiles=168), vocab=262144, V_per_dev=32768 (N_tiles=1024).
    Layout: the activation tile is small,
    the weight slab is large — mcast in0 across the whole compute grid and
    split N evenly across cores.

    Picks per_core_N = ceil(N_tiles / num_cores) — the kernel pads the
    trailing core when num_cores doesn't divide N_tiles (e.g. 80 BH cores
    against 1024 tiles → 13 per core with a partial tail). in0_block_w is
    the largest power of 2 dividing K_tiles, capped at 32 so the in0 CB
    stays small. out_subblock_w stays <=4 to fit the dest register file.
    """
    tile_size = 32
    grid = mesh_device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y

    m_tiles = max(1, (m + tile_size - 1) // tile_size)
    k_tiles = max(1, k // tile_size)
    n_tiles = max(1, n // tile_size)

    # Scope the explicit 1D-mcast config to the regime it is tuned for:
    # the sharded-vocab decode / last-token shape [B<=32, 1, H] x [H, V_per_dev]
    # with M_tiles==1 and a per-device vocab shard bounded at 64K (the same
    # width at which on-device sampling stays enabled, i.e. TP shards the 262144
    # vocab down to <=64K). Outside that regime this config overruns L1:
    #   - tp==1 (e.g. E2B on a single WH N150) keeps the full 262144-wide vocab
    #     on one chip, so per_core_N explodes and the in1 CB grows to ~8 MB,
    #     well past the ~1.4 MB L1 (program.cpp circular-buffer validation throw);
    #   - a non-last-token prefill slice (get_last_token==-1 -> M==seq_len) scales
    #     the output CB by M_tiles.
    # In those cases return None so ttnn.linear falls back to its default matmul
    # heuristic, which blocks N/M to fit L1 (the pre-tuning behaviour).
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
    """Create HF-format cos/sin caches for both sliding and global layer types.

    Returns:
        caches_4d: dict mapping layer_type -> (cos_tt, sin_tt) [1,1,max_seq_len,head_dim] for prefill
        caches_2d: dict mapping layer_type -> (cos_tt, sin_tt) [max_seq_len,head_dim] for decode embedding lookup

    Under context parallelism the 4D prefill caches are sharded along the position
    axis instead of replicated, so rank ``r`` holds positions
    ``[r*max_seq_len/cp, (r+1)*max_seq_len/cp)`` — exactly the tokens it owns. This
    is how each rank gets its true absolute positions: ``_get_rope_mats`` slices
    ``[0:seq_len]`` of the *local* shard with a mesh-wide scalar index, and a scalar
    cannot differ per device, so the per-rank offset has to come from the sharding.

    That alignment holds only when ``max_seq_len`` equals the prefill chunk, which
    is the default in the prefill harness; the caller is expected to check.

    The 2D decode caches stay replicated — decode is not context-parallel.
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.demos.gemma4.tt.ccl import cp_degree

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

        # 2D for decode embedding lookup: [max_seq_len, head_dim].
        # ROW_MAJOR is the layout ttnn.embedding needs for its weight; storing
        # these TILE forced an Untilize of the whole [max_seq_len, head_dim]
        # cache on *every* per-layer RoPE lookup (240 Untilize ops / decode,
        # ~25 us each). ROW_MAJOR storage drops that conversion entirely — the
        # embedding op gathers the position rows and tilizes only the small
        # [1, 32, head_dim] result.
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


def _inject_missing_kv_shared_attention_weights(state_dict, hf_config, kv_shared_layer_map):
    """Add placeholder K/V tensors for checkpoint-omitted kv-shared layers.

    Gemma4 E2B/E4B checkpoints can omit K/V projections for layers that reuse a
    source layer's KV cache. The runtime correctly skips K/V work for those
    layers, but the constructor still builds a fused QKV tensor before that
    runtime flag is known. Zero K/V placeholders make weight loading complete;
    they are discarded under ``is_kv_shared=True``.
    """
    if not state_dict or not kv_shared_layer_map:
        return

    for layer_idx in kv_shared_layer_map:
        cfg = Gemma4AttentionConfig(hf_config, layer_idx)
        kv_size = cfg.num_key_value_heads * cfg.head_dim
        for prefix in ("model.language_model.", "model."):
            attn_prefix = f"{prefix}layers.{layer_idx}.self_attn"
            q_key = f"{attn_prefix}.q_proj.weight"
            if q_key not in state_dict:
                continue

            weight_dtype = state_dict[q_key].dtype
            norm_dtype = state_dict.get(f"{attn_prefix}.q_norm.weight", state_dict[q_key]).dtype
            state_dict.setdefault(
                f"{attn_prefix}.k_proj.weight",
                torch.zeros((kv_size, hf_config.hidden_size), dtype=weight_dtype),
            )
            if not cfg.use_kv_tying:
                state_dict.setdefault(
                    f"{attn_prefix}.v_proj.weight",
                    torch.zeros((kv_size, hf_config.hidden_size), dtype=weight_dtype),
                )
            state_dict.setdefault(
                f"{attn_prefix}.k_norm.weight",
                torch.ones((cfg.head_dim,), dtype=norm_dtype),
            )


class Gemma4Model:
    # Generator-interface flag. PLI models (E2B/E4B) recompute host per-layer
    # inputs from the token every step, so their decode-trace input buffers
    # must be restaged every replay. Non-PLI models (12B/26B/31B) keep this
    # False and rely on on-device token feedback + position plus_one — required
    # for async scheduling (#51186); host restage under async lag re-processes
    # the previous token ("TheThe user user...").
    # Overridden in ``__init__`` from ``hidden_size_per_layer_input``.
    _tt_vllm_always_refresh_decode_trace_inputs = True
    # Sampling writes a tile-aligned [1,1,1,32] token vector; decode embeds only
    # the active batch. Non-PLI prepare_decode pads tokens to this width so the
    # sampled ids can be written straight back into the trace input buffer.
    _DECODE_TOKEN_FEEDBACK_WIDTH = 32
    # NOTE: This is a runtime capability (depends on mesh shape / per-device vocab).
    # It is set during __init__ after the sampling module is constructed.
    _supports_on_device_sampling = False
    # On-device greedy at B=sampling_max (#48037, mirrors qwen3_vl / qwen25_vl):
    # Gemma4 only captures the sampling *trace* at sampling_max (B=32). Replaying
    # that trace freezes ``all_gather_async`` semaphores from capture time, so the
    # gather corrupts from the 2nd decode step. B=1 already sampled eagerly
    # (batch != sampling_max) and stayed correct. Run sampling eagerly so each
    # step re-acquires a fresh semaphore. Non-PLI keeps device token feedback
    # (``_tt_vllm_always_refresh_decode_trace_inputs=False``) for async; the
    # eagerly sampled id is still written into the padded feedback buffer.
    _tt_disable_sampling_trace = True

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
        paged_attention_config=None,
        create_kv_cache=True,
        precision=None,
        bounded_sliding_kv_cache: bool = False,
        # Global prefill chunk size. Only needed under context parallelism with more
        # than one chunk: it sets the RoPE cache's chunk-major row order and sizes the
        # ring KV cache slabs. None means single-chunk prefill.
        prefill_chunk_size=None,
        # Legacy parameters — ignored
        transformation_mats=None,
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.prefill_chunk_size = prefill_chunk_size
        self.mesh_config = mesh_config
        self.hidden_size = hf_config.hidden_size
        self.vocab_size = hf_config.vocab_size
        self.final_logit_softcapping = hf_config.final_logit_softcapping
        self.embed_scale = hf_config.hidden_size**0.5
        self.ccl_manager = ccl_manager
        # Pinned RoPE slices for traced chunked prefill; see _refresh_rope_prefill.
        # Empty and inactive unless a caller opts in, so the untraced path is unchanged.
        self._rope_prefill_buffers = {}
        self._rope_prefill_pinned = False
        # When True the caller refreshes the ring metadata itself, outside any trace.
        self._ring_metadata_external = False
        self.max_seq_len = max_seq_len
        self.hidden_size_per_layer_input = getattr(hf_config, "hidden_size_per_layer_input", 0) or 0
        # Host restage every step only when PLI must be recomputed from the token.
        # Non-PLI keeps device token/pos continuity for async decode (#51186).
        # Debug/kill-switch: GEMMA4_ALWAYS_REFRESH_DECODE=1 forces host restage
        # (disables async-safe continuity; platform will also disable async).
        force_refresh = os.environ.get("GEMMA4_ALWAYS_REFRESH_DECODE", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self._tt_vllm_always_refresh_decode_trace_inputs = bool(self.hidden_size_per_layer_input) or force_refresh
        n_layers = num_layers or hf_config.num_hidden_layers

        # Per-module dtype resolution. ``precision`` (Gemma4Precision) holds
        # any overrides loaded from precision_overrides.json; modules without
        # an override fall back to ``dtype`` (the model-wide default). Dtypes
        # are then threaded explicitly through DecoderLayer / used directly
        # for embedding + lm_head, so each weight loads at the right precision
        # and lands in a cache file tagged with that dtype.
        from models.demos.gemma4.tt.precision import Gemma4Precision

        if precision is None:
            precision = Gemma4Precision()
        shared_mlp_dtype = precision.get("shared_mlp", dtype)
        attention_dtype = precision.get("attention", dtype)
        experts_dtype = precision.get("experts", dtype)
        router_dtype = precision.get("router", dtype)
        embedding_dtype = precision.get("embedding", dtype)
        lm_head_dtype = precision.get("lm_head", dtype)

        # KV sharing map: layers after (full_n_layers - num_kv_shared_layers) share KV
        # from the last non-shared layer of the same type
        full_n_layers = hf_config.num_hidden_layers
        num_kv_shared = getattr(hf_config, "num_kv_shared_layers", 0) or 0
        first_shared_idx = full_n_layers - num_kv_shared
        self.kv_shared_layer_map = {}  # layer_idx -> source_layer_idx
        if num_kv_shared > 0 and first_shared_idx < n_layers:
            prev_layers = hf_config.layer_types[:first_shared_idx]
            for i in range(first_shared_idx, n_layers):
                lt = hf_config.layer_types[i]
                if lt in prev_layers:
                    source = len(prev_layers) - 1 - list(prev_layers)[::-1].index(lt)
                    if source < n_layers:  # Source must be within our layer range
                        self.kv_shared_layer_map[i] = source
            if self.kv_shared_layer_map:
                logger.info(f"KV sharing enabled: {len(self.kv_shared_layer_map)} layers share KV from earlier layers")

        _inject_missing_kv_shared_attention_weights(state_dict, hf_config, self.kv_shared_layer_map)

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

        # Device tensors for traced multi-chunk RoPE slicing (chunk_start_idx tensor
        # → [start, start+chunk) without leaving the captured graph). Lazy-init
        # the per-head-dim ends buffers; zeros are shared.
        self._tt_slice_start_zeros_4 = None
        self._tt_seq_len_buffer_by_hd = {}

        # Embedding
        is_mesh = hasattr(mesh_device, "shape")
        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
        tp = mesh_config.tp if mesh_config else 1
        tp_suffix = f"_tp{tp}" if tp > 1 else ""

        from models.demos.gemma4.tt.precision import dtype_to_str

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
                cache_file_name=get_cache_file_name(tensor_cache_path, f"lm_head.weight{tp_suffix}{lm_head_suffix}"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.embedding_weight = None
            self.lm_head_weight = None

        # Per-layer input embeddings (E2B/E4B) — kept as CPU torch tensors for computation
        # Also store embedding weight reference for decode per-layer input
        self._embed_weight_cpu = None
        if embed_key and state_dict:
            self._embed_weight_cpu = state_dict[embed_key]

        # Stash for prefill inputs computed in ``prepare_inputs_prefill``.
        # The Generator interface splits prefill into prepare→forward, but
        # forward's signature doesn't carry the host-side input_ids/embeds
        # the model needs for per-layer inputs, so we cache them here.
        # Direct callers (text_demo, unit tests) pass them explicitly to
        # ``ttnn_prefill_forward`` and bypass this stash.
        self._prefill_input_ids_torch = None
        self._prefill_embeds_torch = None

        # Stash for the per-layer-input (PLI) device tensor produced in
        # ``prepare_decode_inputs_host``. ``Generator``'s decode path
        # unpacks only the first 4 elements of ``prepare_inputs_decode``'s
        # return tuple, dropping the trailing PLI tensor (E2B/E4B), so we
        # cache it here and have ``ttnn_decode_forward`` fall back to it
        # when the explicit kwarg is None — same pattern as the prefill
        # stash above.
        self._decode_pli_combined = None
        self.per_layer_input_weights = {}
        if self.hidden_size_per_layer_input and state_dict:
            pli_size = self.hidden_size_per_layer_input
            # Try both key formats
            for prefix in ["model.language_model.", "model."]:
                pli_embed_key = f"{prefix}embed_tokens_per_layer.weight"
                pli_proj_key = f"{prefix}per_layer_model_projection.weight"
                pli_norm_key = f"{prefix}per_layer_projection_norm.weight"
                if pli_embed_key in state_dict:
                    self.per_layer_input_weights = {
                        "embed_tokens_per_layer": state_dict[pli_embed_key],  # [vocab_pli, n_layers * pli_size]
                        "per_layer_model_projection": state_dict[pli_proj_key],  # [n_layers * pli_size, hidden]
                        "per_layer_projection_norm": state_dict[pli_norm_key],  # [pli_size]
                    }
                    self.per_layer_input_scale = 2.0**-0.5
                    self.per_layer_model_projection_scale = hf_config.hidden_size**-0.5
                    self.per_layer_embed_scale = pli_size**0.5

                    logger.info(f"Per-layer input embeddings loaded (pli_size={pli_size})")
                    break

        # Decoder layers (each creates its own KV cache if requested)
        self.bounded_sliding_kv_cache = bounded_sliding_kv_cache
        self.layers = []
        for i in range(n_layers):
            layer = Gemma4DecoderLayer(
                mesh_device=mesh_device,
                hf_config=hf_config,
                state_dict=state_dict,
                layer_idx=i,
                ccl_manager=ccl_manager,
                dtype=dtype,
                shared_mlp_dtype=shared_mlp_dtype,
                attention_dtype=attention_dtype,
                experts_dtype=experts_dtype,
                router_dtype=router_dtype,
                tensor_cache_path=tensor_cache_path,
                mesh_config=mesh_config,
                max_seq_len=max_seq_len,
                max_local_batch_size=max_local_batch_size,
                bounded_sliding_kv_cache=bounded_sliding_kv_cache,
                ring_prefill_chunk_size=prefill_chunk_size,
            )
            # Create KV cache for non-shared layers only
            # Shared layers will use their source layer's KV cache
            if create_kv_cache and i not in self.kv_shared_layer_map:
                from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache

                attn_cfg = Gemma4AttentionConfig(hf_config, i)
                # Bounded SlidingWindowSpec allocation for sliding layers: only enough
                # physical blocks to cover one sliding-window-sized region per user,
                # instead of one max_seq_len-sized region. Mirrors vLLM's hybrid
                # kv_cache_groups. Full-attention layers keep the existing allocation.
                max_num_blocks_override = None
                if (
                    bounded_sliding_kv_cache
                    and attn_cfg.is_sliding
                    and attn_cfg.sliding_window is not None
                    and paged_attention_config is not None
                ):
                    sliding_blocks_per_seq = attn_cfg.sliding_window // paged_attention_config.block_size
                    max_num_blocks_override = sliding_blocks_per_seq * max_local_batch_size
                max_num_blocks_override = self._cp_block_pool_override(
                    max_num_blocks_override, paged_attention_config, bounded_sliding_kv_cache
                )
                kv_cache = init_kv_cache(
                    mesh_device=mesh_device,
                    config=attn_cfg,
                    max_batch_size=max_local_batch_size,
                    max_seq_len=max_seq_len,
                    paged_attention_config=paged_attention_config,
                    cache_dtype=ttnn.bfloat16,
                    max_num_blocks_override=max_num_blocks_override,
                )
                layer.self_attn.kv_cache = kv_cache
            self.layers.append(layer)

        # Extract KV caches for external access (Generator interface)
        # Shared layers point to their source layer's cache
        self.tt_kv_cache = []
        for i, layer in enumerate(self.layers):
            if i in self.kv_shared_layer_map:
                source_idx = self.kv_shared_layer_map[i]
                self.tt_kv_cache.append(self.layers[source_idx].self_attn.kv_cache)
            else:
                self.tt_kv_cache.append(layer.self_attn.kv_cache)

        # Last layer index of each attention type — these are the layers whose
        # KV the Gemma4 *it-assistant* drafter cross-attends into (HF
        # ``shared_kv_states`` exposes "the last layer of each layer_type"). Used
        # by speculative decoding (see tt/assistant/model.py + tt/spec_decode.py).
        self.last_kv_layer_by_type = {}
        for i in range(n_layers):
            self.last_kv_layer_by_type[hf_config.layer_types[i]] = i

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

        # sampling_dp: number of independent sampling groups (one per mesh row).
        # This is 1 for standard TP-only meshes (e.g. 1x8), and >1 for multi-row
        # meshes where each row samples users independently (e.g. Galaxy 4x8).
        #
        # tt_transformers' Generator reads this attribute via _get_sampling_contract.
        self.sampling_dp = mesh_device.shape[0] if is_mesh else 1

        # On-device sampling (greedy/top-k/top-p) — avoids reading full vocab logits to CPU
        self.sampling = None
        if is_mesh and tp > 1:
            per_device_padded = _compute_per_device_vocab(hf_config.vocab_size, tp)
            if per_device_padded <= 64 * 1024:
                sampling_args = self._make_sampling_args(hf_config, mesh_device, tp)
                # Match sampling all-gather topology to Gemma4 CCLManager (Ring on
                # BH≥8, Linear elsewhere / GEMMA4_CCL_TOPOLOGY). Without this,
                # TTSampling defaults to Linear while model collectives use Ring.
                #
                # allow_force_argmax=True (#48037): greedy (temp=0 → k=1,p=0,temp=1)
                # must take the single-gather argmax path. With it disabled (the
                # previous Gemma4 default), greedy went through the heavy
                # top-k/top-p multi-gather pipeline that corrupts at B=32.
                # Requires a real TT_CCL (semaphores for force-argmax
                # all_gather_async) — passing tt_ccl=None made force-argmax
                # unusable, which is why it was previously forced off.
                from models.tt_transformers.tt.ccl import TT_CCL

                if ccl_manager is not None:
                    sampling_args.model_config["SAMPLING_AG_CONFIG"] = {
                        "allow_force_argmax": True,
                        "num_links": ccl_manager.num_links,
                        "topology": ccl_manager.topology,
                    }
                sampling_tt_ccl = TT_CCL(mesh_device)
                self.sampling = SamplingGenerator(
                    args=sampling_args,
                    mesh_device=mesh_device,
                    tt_ccl=sampling_tt_ccl,
                )
                _apply_gemma4_single_untilize_override(self.sampling.tt_sampling)
                topo = getattr(self.sampling.tt_sampling, "ag_topology", None)
                topo_name = "Ring" if topo == ttnn.Topology.Ring else "Linear"
                logger.info(
                    f"On-device sampling initialized (vocab={hf_config.vocab_size}, "
                    f"per_device={per_device_padded}, ag_topology={topo_name}, "
                    f"force_argmax=1, disable_sampling_trace={int(self._tt_disable_sampling_trace)})"
                )
        # Generator/vLLM entry points gate on this flag (and sampling != None).
        self._supports_on_device_sampling = self.sampling is not None

        # Trace-safe bounded-fill cap: one persistent 1-element int32 device
        # tensor shared by every sliding layer. Traced prefill runs with
        # get_last_token=-1 (lm_head deferred), so the host-side valid_seq_len
        # slice is skipped; the generator refreshes this tensor out-of-trace
        # and paged_fill_cache's writer reads it at runtime to skip padding
        # tiles that would otherwise wrap the circular KV window.
        self.prefill_valid_len_dev = None
        if bounded_sliding_kv_cache:
            self._init_prefill_valid_len_dev()

    def _init_prefill_valid_len_dev(self):
        """Allocate the persistent valid_seq_len tensor and stash it on every
        bounded sliding layer's attention config (``prefill_valid_len_dev``).
        """
        is_mesh = hasattr(self.mesh_device, "shape")
        self.prefill_valid_len_dev = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int32),
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh else None,
        )
        for layer in self.layers:
            cfg = getattr(getattr(layer, "self_attn", None), "config", None)
            if cfg is not None and getattr(cfg, "cache_position_modulo", None) is not None:
                cfg.prefill_valid_len_dev = self.prefill_valid_len_dev

    def update_prefill_valid_seq_len(self, valid_seq_len: int):
        """Refresh the persistent valid_seq_len device tensor (out of trace).

        ``valid_seq_len`` is the real (unpadded) token count for the current
        prefill chunk. The writer kernel block-aligns it; callers pass the raw
        length (``last_token_idx - num_cached_tokens + 1``).
        """
        if self.prefill_valid_len_dev is None:
            return
        if valid_seq_len is None or int(valid_seq_len) <= 0:
            return
        is_mesh = hasattr(self.mesh_device, "shape")
        host = ttnn.from_torch(
            torch.tensor([int(valid_seq_len)], dtype=torch.int32),
            device=None,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh else None,
        )
        ttnn.copy_host_to_device_tensor(host, self.prefill_valid_len_dev)

    @staticmethod
    def _make_sampling_args(hf_config, mesh_device, tp):
        """Create minimal args object for SamplingGenerator/TTSampling."""

        class _Args:
            pass

        args = _Args()
        args.vocab_size = hf_config.vocab_size
        per_device_vocab = _compute_per_device_vocab(args.vocab_size, tp)
        args.padded_vocab_size = per_device_vocab * tp
        args.cluster_shape = tuple(mesh_device.shape)
        args.sampling_all_gather_axis = 1  # gather across TP (column) axis
        args.sampling_dp = mesh_device.shape[0]
        args.num_devices = mesh_device.get_num_devices()
        args.is_galaxy = mesh_device.shape[0] > 1
        args.model_config = {}
        args.use_topk_logprobs = False
        return args

    def _compute_per_layer_inputs(self, input_ids_torch, embeds_torch):
        """Compute per-layer input embeddings on CPU (E2B/E4B).

        Returns list of [1, seq_len, pli_size] tensors, one per layer, or None
        if the model is not configured with per-layer inputs.

        Raises ValueError if the model has PLI configured but input_ids_torch or
        embeds_torch are missing — silently dropping PLI produces garbage decode
        output without any other failure signal.
        """
        if not self.hidden_size_per_layer_input or not self.per_layer_input_weights:
            return None
        if input_ids_torch is None or embeds_torch is None:
            raise ValueError(
                "Model has per-layer inputs configured but input_ids_torch/embeds_torch "
                "are missing. Pass pli_combined (decode) or pli_device_tensors instead, "
                "or supply input_ids_torch and embeds_torch."
            )

        import torch.nn.functional as F

        w = self.per_layer_input_weights
        pli_size = self.hidden_size_per_layer_input
        n_layers = len(self.layers)
        # The per-layer embedding weight has ALL layers baked in
        # Infer full layer count from the weight shape, not the (possibly overridden) config
        embed_w = w["embed_tokens_per_layer"]  # [vocab_pli, full_n_layers * pli_size]
        full_n_layers = embed_w.shape[-1] // pli_size

        # 1. Per-layer token embedding: embed_tokens_per_layer(input_ids)
        pli_embed = F.embedding(input_ids_torch.long(), embed_w) * self.per_layer_embed_scale
        pli_embed = pli_embed.reshape(*input_ids_torch.shape, full_n_layers, pli_size)

        # 2. Projection from main embeddings
        proj_w = w["per_layer_model_projection"]  # [full_n_layers * pli_size, hidden]
        pli_proj = F.linear(embeds_torch.float(), proj_w.float()) * self.per_layer_model_projection_scale
        pli_proj = pli_proj.reshape(*embeds_torch.shape[:-1], full_n_layers, pli_size)

        # 3. Norm the projection
        norm_w = w["per_layer_projection_norm"]  # [pli_size]
        eps = self.hf_config.rms_norm_eps
        pli_proj_f = pli_proj.float()
        var = pli_proj_f.pow(2).mean(-1, keepdim=True)
        pli_proj = (pli_proj_f * torch.rsqrt(var + eps) * norm_w.float()).to(pli_proj.dtype)

        # 4. Combine: (projection + embed) * scale
        per_layer_inputs = (pli_proj + pli_embed.float()) * self.per_layer_input_scale

        # Return as list of per-layer tensors
        return [per_layer_inputs[:, :, i, :].to(torch.bfloat16) for i in range(n_layers)]

    def _ensure_rope_slice_bufs(self, head_dim: int):
        """Allocate shared start-zeros / ends buffers for device RoPE slicing."""
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        if self._tt_slice_start_zeros_4 is None:
            self._tt_slice_start_zeros_4 = ttnn.from_torch(
                torch.tensor([0, 0, 0, 0], dtype=torch.int32),
                device=self.mesh_device,
                mesh_mapper=mesh_mapper,
            )
        if head_dim not in self._tt_seq_len_buffer_by_hd:
            self._tt_seq_len_buffer_by_hd[head_dim] = ttnn.from_torch(
                torch.tensor([1, 1, self.max_seq_len, head_dim], dtype=torch.int32),
                device=self.mesh_device,
                mesh_mapper=mesh_mapper,
            )

    def _slice_prefill_rot_mats(self, rot_mats, chunk_start_idx, seq_len):
        """Device-slice full RoPE mats to ``[chunk_start_idx, chunk_start_idx+seq_len)``.

        Uses the tensor-args ``ttnn.slice`` path (``slice_dim`` + ``num_devices``)
        so the start offset can be refreshed out-of-trace while the graph stays
        fixed. ``num_devices`` is overloaded as ``max_seq_len // seq_len`` so the
        fixed output length equals the prefill chunk (same pattern as Galaxy /
        tt_transformers traced APC RoPE).
        """
        if rot_mats is None or chunk_start_idx is None or not isinstance(chunk_start_idx, ttnn.Tensor):
            return rot_mats
        full_rot_cos, full_rot_sin = rot_mats
        if full_rot_cos.shape[2] == seq_len:
            return rot_mats
        if self.max_seq_len % seq_len != 0:
            raise ValueError(
                f"Traced multi-chunk RoPE requires max_seq_len ({self.max_seq_len}) "
                f"divisible by chunk seq_len ({seq_len})"
            )
        num_parts = self.max_seq_len // seq_len
        head_dim = int(full_rot_cos.shape[3])
        self._ensure_rope_slice_bufs(head_dim)
        z = self._tt_slice_start_zeros_4
        tt_slice_starts = ttnn.concat([z[0:2], chunk_start_idx, z[3:4]], dim=0)
        ends = self._tt_seq_len_buffer_by_hd[head_dim]
        rot_cos_slice = ttnn.slice(
            input_tensor=full_rot_cos,
            starts=tt_slice_starts,
            ends=ends,
            slice_dim=2,
            num_devices=num_parts,
        )
        rot_sin_slice = ttnn.slice(
            input_tensor=full_rot_sin,
            starts=tt_slice_starts,
            ends=ends,
            slice_dim=2,
            num_devices=num_parts,
        )
        return (rot_cos_slice, rot_sin_slice)

    def _get_rope_mats(self, layer_idx, seq_len=None, for_decode=False, start_pos=0):
        """Get (cos, sin) for a given layer.

        Args:
            seq_len: If set, slice 4D cache to this length (prefill).
            for_decode: If True, return 2D caches [max_seq_len, head_dim] for embedding lookup.
            start_pos: Absolute position of the first token in this prefill call.
                Non-zero only for generator-level multi-chunk prefill (chunk N starts
                at ``N*chunk_size``); the RoPE slice must cover
                ``[start_pos, start_pos+seq_len)`` so chunk tokens get their true
                positions instead of restarting at 0. When ``start_pos`` is a device
                tensor, returns the *full* cache so ``_slice_prefill_rot_mats`` can
                cut it inside the traced graph.
        """
        layer_type = self.hf_config.layer_types[layer_idx]
        if for_decode:
            return self.rope_caches_2d[layer_type]
        cos, sin = self.rope_caches[layer_type]
        if isinstance(start_pos, ttnn.Tensor):
            return (cos, sin)
        if seq_len is not None:
            if self._rope_prefill_pinned:
                # Traced chunked prefill: hand back a buffer at a FIXED address whose
                # contents were refreshed for this chunk (see _refresh_rope_prefill).
                # Slicing here would allocate a new tensor per chunk, and a trace records
                # the address it saw at capture — every replay would then re-read chunk
                # N's RoPE rows regardless of which chunk is running.
                return self._rope_prefill_buffers[(layer_type, seq_len)]
            cos = cos[:, :, start_pos : start_pos + seq_len, :]
            sin = sin[:, :, start_pos : start_pos + seq_len, :]
        return (cos, sin)

    def _refresh_rope_prefill(self, seq_len, start_pos):
        """Point the pinned RoPE buffers at ``[start_pos, start_pos+seq_len)``.

        Runs on the host side of a chunk, outside any traced region: it copies the
        chunk's slice into buffers whose addresses stay put, so a captured trace reads
        the right positions on every replay. Buffers are allocated on first use, one
        pair per (layer_type, seq_len) — with two layer types and one prefill chunk
        size that is two pairs for the whole model.
        """
        for layer_type, (cos, sin) in self.rope_caches.items():
            key = (layer_type, seq_len)
            if key not in self._rope_prefill_buffers:
                self._rope_prefill_buffers[key] = (
                    ttnn.clone(cos[:, :, 0:seq_len, :]),
                    ttnn.clone(sin[:, :, 0:seq_len, :]),
                )
            cos_buf, sin_buf = self._rope_prefill_buffers[key]
            ttnn.copy(cos[:, :, start_pos : start_pos + seq_len, :], cos_buf)
            ttnn.copy(sin[:, :, start_pos : start_pos + seq_len, :], sin_buf)
        self._rope_prefill_pinned = True

    def __call__(
        self,
        hidden_states,
        rope_mats=None,
        position_idx=None,
        page_table=None,
        kv_caches=None,
        is_decode=True,
        token_index=None,
        input_ids_torch=None,
        embeds_torch=None,
        pli_device_tensors=None,
        position_idx_cache=None,
        pli_combined=None,
        get_last_token=-1,
        page_tables_per_layer=None,
        batch_size=1,
        user_id=0,
        return_hidden=False,
        sequential_kv_write=False,
        packed=None,
        chunk_start_idx=None,
        chunk_page_table=None,
        valid_seq_lens=None,
        keep_sharded_for_sampling=False,
    ):
        """
        Forward pass through decoder layers + final norm + lm_head + softcapping.

        ``return_hidden`` (decode only): also return the post-norm hidden states
        ``[1,1,B,hidden]`` alongside logits, as ``(logits, hidden)``. The Gemma4
        it-assistant drafter consumes the target's last-token hidden state, and
        the multi-token verify forward (``ttnn_verify_forward``) needs the hidden
        states for every verified position to seed the next drafter iteration.

        ``keep_sharded_for_sampling``: when True (decode + on-device sampling),
        leave lm_head logits TP-sharded. Host sampling / full-vocab reads must
        leave this False so decode all-gathers the 262k vocab.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device (post-embedding)
            rope_mats: (cos, sin) override, or dict {layer_type: (cos, sin)} for pre-sliced decode
            position_idx: decode position tensor ([1,32] uint32 for embedding RoPE, or [1] int32 legacy)
            page_table: paged attention table
            kv_caches: list of [k, v] per layer, or None (uses self.tt_kv_cache)
            is_decode: True for decode, False for prefill
            token_index: int for decode RoPE slicing (None when using embedding-based RoPE)
            input_ids_torch: CPU tensor of input_ids for per-layer input computation (E2B)
            embeds_torch: CPU tensor of embeddings for per-layer input projection (E2B)
            pli_device_tensors: optional list of pre-computed PLI device tensors (trace mode)
            position_idx_cache: optional [batch] int32 tensor for KV cache update (when position_idx is uint32)
            pli_combined: optional [1,1,n_layers,pli_size] device tensor of pre-computed PLI (decode)
            page_tables_per_layer: optional list of per-layer page tables, one
                entry per decoder layer. When set, each layer's attention
                receives ``page_tables_per_layer[i]`` instead of ``page_table``.
                The vLLM hybrid kv-cache manager produces this list so
                sliding-window layers can index a smaller paged pool than
                full-attention layers (KV cache groups).
            packed: optional packed-verify dict (decode only). Carries the
                P-position packed attention inputs; layer-type-specific entries
                (attn_mask_full / attn_mask_sliding, embed_idx_full /
                embed_idx_sliding, rope_packed per type) are selected per layer
                here and routed to ``packed_decode_forward``.
        """
        seq_len = hidden_states.shape[2]
        rope_seq_len = seq_len // batch_size if (not is_decode and batch_size > 1) else seq_len
        caches = kv_caches or self.tt_kv_cache

        # Real (unpadded) prefill length for KV fill cap. Scalar from
        # ``get_last_token`` (B=1 / uniform), or per-slot list for batched
        # prefill with hetero actual lengths (``valid_seq_lens``). Batched
        # path keeps ``get_last_token=-1`` so lm_head stays deferred.
        prefill_valid_len = None
        if not is_decode and get_last_token is not None and get_last_token >= 0:
            prefill_valid_len = get_last_token + 1
        elif not is_decode and valid_seq_lens is not None:
            prefill_valid_len = valid_seq_lens

        if page_tables_per_layer is not None and len(page_tables_per_layer) != len(self.layers):
            raise ValueError(
                f"page_tables_per_layer has {len(page_tables_per_layer)} entries "
                f"but model has {len(self.layers)} layers"
            )

        # Compute per-layer inputs (E2B/E4B)
        # Decode: PLI pre-computed on host (pli_combined); main embed on device
        # Prefill: computed on CPU from input_ids_torch / embeds_torch
        pli_combined_tt = None
        per_layer_inputs = None
        if pli_combined is not None:
            pli_combined_tt = pli_combined
        elif pli_device_tensors is not None:
            # Pre-computed device tensors provided externally (legacy trace mode).
            # For PLI models every layer must receive its per-layer input: a short
            # list would silently run the remaining layers with pli_tt=None, which
            # drops PLI and produces bad output with no other failure signal. The
            # normal _compute_per_layer_inputs path treats missing PLI as a hard
            # error, so enforce the same invariant at this boundary.
            if self.hidden_size_per_layer_input and len(pli_device_tensors) != len(self.layers):
                raise ValueError(
                    f"pli_device_tensors has {len(pli_device_tensors)} entries "
                    f"but PLI model has {len(self.layers)} layers"
                )
        else:
            per_layer_inputs = self._compute_per_layer_inputs(input_ids_torch, embeds_torch)

        is_mesh = hasattr(self.mesh_device, "shape")

        # Determine which layers are KV sources (their K/V will be shared)
        kv_source_indices = set(self.kv_shared_layer_map.values()) if not is_decode else set()
        # Store K/V from source layers for sharing during prefill
        shared_kv_store = {}  # source_layer_idx -> (tt_k, tt_v) kept alive on device

        # Decode RoPE: slice cos/sin ONCE per layer_type and share across all layers.
        # There are only two layer_types (sliding / global), so the position-gather
        # (ttnn.embedding) runs twice per decode step instead of once per layer. The
        # gathered [1, 1, batch_pad, head_dim] tensors are passed down with
        # rope_presliced=True and freed after the layer loop. Only taken on the
        # internal-cache decode path (rope_mats override paths keep their behavior).
        decode_rope_presliced = {}
        if is_decode and rope_mats is None and self.rope_caches_2d and position_idx is not None:
            used_types = {self.hf_config.layer_types[i] for i in range(len(self.layers))}
            for lt in used_types:
                if lt not in self.rope_caches_2d:
                    continue
                cos_2d, sin_2d = self.rope_caches_2d[lt]
                cos_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, cos_2d, layout=ttnn.TILE_LAYOUT))
                sin_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, sin_2d, layout=ttnn.TILE_LAYOUT))
                decode_rope_presliced[lt] = (cos_pos, sin_pos)

        # Publish this chunk's per-chunk scalars once, before any layer runs. Every layer
        # reads the same two metadata tensors, and they must be written from the host
        # outside any traced region — a trace captures addresses, not values, which is the
        # whole point of routing these through DRAM instead of runtime args.
        # Skipped when a traced caller owns the update: copy_host_to_device_tensor has to
        # run outside the captured region, or the trace would replay a stale write.
        if not is_decode and chunk_start_idx is not None and not self._ring_metadata_external:
            from models.demos.gemma4.tt.ccl import cp_degree as _cp_degree

            if _cp_degree(self.mesh_config) > 1:
                self.ccl_manager.set_ring_metadata(slot_idx=user_id or 0, kv_actual_global=int(chunk_start_idx))

        for i, layer in enumerate(self.layers):
            # Per-layer RoPE: sliding and global layers have different cos/sin
            rope_presliced = False
            if rope_mats is not None:
                if isinstance(rope_mats, dict):
                    # Dict mapping layer_type -> (cos, sin) — pre-sliced for trace decode
                    layer_type = self.hf_config.layer_types[i]
                    layer_rope = rope_mats[layer_type]
                else:
                    layer_rope = rope_mats  # Single (cos, sin) override (backward compat / tests)
            elif is_decode and decode_rope_presliced:
                # Decode: use the per-layer-type cos/sin gathered once before the loop.
                layer_rope = decode_rope_presliced[self.hf_config.layer_types[i]]
                rope_presliced = True
            elif is_decode:
                # Decode fallback: return 2D caches for on-device embedding lookup
                layer_rope = self._get_rope_mats(i, for_decode=True)
            else:
                # Generator-level multi-chunk prefill: chunk N's tokens occupy
                # absolute positions [chunk_start_idx, chunk_start_idx+seq_len);
                # offset the RoPE slice so they aren't re-encoded from 0.
                # Device-tensor offsets stay inside the traced graph.
                if isinstance(chunk_start_idx, ttnn.Tensor):
                    layer_rope = self._slice_prefill_rot_mats(
                        self._get_rope_mats(i, start_pos=chunk_start_idx),
                        chunk_start_idx,
                        rope_seq_len,
                    )
                else:
                    rope_start_pos = int(chunk_start_idx) if chunk_start_idx is not None else 0
                    # CP RoPE rows are chunk-major per rank, so scalar host slices
                    # advance by a per-rank slab rather than the global chunk.
                    from models.demos.gemma4.tt.ccl import cp_degree

                    cp = cp_degree(self.mesh_config)
                    if cp > 1:
                        rope_start_pos //= cp
                    layer_rope = self._get_rope_mats(i, seq_len=rope_seq_len, start_pos=rope_start_pos)

            # Convert per-layer input to device tensor if available
            pli_tt = None
            if pli_combined_tt is not None:
                # On-device decode: slice layer i from combined [1, 1, n_layers, pli_size]
                pli_tt = pli_combined_tt[:, :, i : i + 1, :]
            elif pli_device_tensors is not None:
                # Pre-computed device tensors (legacy trace mode). Length was
                # validated to match len(self.layers) for PLI models above.
                pli_tt = pli_device_tensors[i]
            elif per_layer_inputs is not None and i < len(per_layer_inputs):
                pli_layer = per_layer_inputs[i]
                if batch_size > 1 and pli_layer.dim() == 3:
                    pli_4d = pli_layer.reshape(1, 1, -1, pli_layer.shape[-1])
                else:
                    pli_4d = pli_layer.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, pli_size]
                pli_tt = ttnn.from_torch(
                    pli_4d,
                    device=self.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh else None,
                )

            kv_cache = caches[i] if caches else None

            # KV sharing: determine if this layer shares or provides K/V
            shared_kv = None
            keep_kv = False
            is_kv_shared = i in self.kv_shared_layer_map
            if not is_decode and is_kv_shared:
                source_idx = self.kv_shared_layer_map[i]
                shared_kv = shared_kv_store.get(source_idx)
            elif not is_decode and i in kv_source_indices:
                keep_kv = True

            layer_page_table = page_tables_per_layer[i] if page_tables_per_layer is not None else page_table

            layer_packed = None
            if packed is not None:
                lt = self.hf_config.layer_types[i]
                sliding = lt == "sliding_attention"
                rope_packed = packed.get("rope_packed") or {}
                layer_packed = {
                    "packed_p": packed["packed_p"],
                    "position_idx": packed["position_idx"],
                    "kv_write_idxs": packed.get("kv_write_idxs"),
                    "attn_mask": packed["attn_mask_sliding"] if sliding else packed["attn_mask_full"],
                    "rope_packed": rope_packed.get(lt),
                    "embed_idx": packed.get("embed_idx_sliding") if sliding else packed.get("embed_idx_full"),
                    "hot_pt": packed.get("hot_pt"),
                }

            hidden_states = layer(
                hidden_states,
                rope_mats=layer_rope,
                position_idx=position_idx,
                page_table=layer_page_table,
                kv_cache=kv_cache,
                is_decode=is_decode,
                token_index=token_index,
                per_layer_input=pli_tt,
                shared_kv=shared_kv,
                keep_kv=keep_kv,
                is_kv_shared=is_kv_shared,
                position_idx_cache=position_idx_cache,
                batch_size=batch_size,
                user_id=user_id,
                valid_seq_len=prefill_valid_len,
                sequential_kv_write=sequential_kv_write,
                rope_presliced=rope_presliced,
                packed=layer_packed,
                chunk_start_idx=chunk_start_idx,
                chunk_page_table=chunk_page_table,
            )

            # For KV source layers during prefill, capture the K/V from the attention
            # The K/V are kept alive on device (not deallocated) when keep_kv=True
            if keep_kv and layer.self_attn._last_kv is not None:
                shared_kv_store[i] = layer.self_attn._last_kv

        # Free the per-layer-type decode RoPE tensors shared across the loop.
        for cos_pos, sin_pos in decode_rope_presliced.values():
            cos_pos.deallocate(True)
            sin_pos.deallocate(True)

        # Deallocate any stored shared K/V tensors
        for kv_pair in shared_kv_store.values():
            if kv_pair is not None:
                kv_pair[0].deallocate(True)
                kv_pair[1].deallocate(True)

        # Single-user intermediate generator-level chunks (get_last_token=-1 with
        # a chunk_page_table, not in prefill-trace mode) only need the KV fill
        # from the layer loop above — their logits are discarded by the chunk
        # loop, so skip the expensive full-sequence lm_head.
        # Gate on chunk_page_table: get_last_token defaults to -1 for all direct
        # ttnn_prefill_forward callers (unit tests, demos), which still need logits.
        if (
            not is_decode
            and get_last_token == -1
            and batch_size == 1
            and chunk_page_table is not None
            and not getattr(self, "_prefill_trace_mode", False)
        ):
            # Intermediate generator chunk: do not flush (last chunk owns the ring).
            return None

        # Final norm (must run before batched early-return). Generator then
        # defers lm_head per slot via ``process_logits_after_prefill_trace``,
        # which expects *post-norm* hidden (same contract as traced prefill).
        # Returning pre-norm here skipped RMSNorm and corrupted next-token
        # logits → decode garbage on metal batch-32 / hetero batched prefill.
        hidden_states = self.norm.forward(hidden_states)

        if not is_decode and get_last_token == -1 and batch_size > 1:
            self._flush_deferred_bounded_fills_if_needed()
            return hidden_states

        # Speculative decoding seed: the it-assistant drafter's recurrent hidden
        # is HF's ``model_outputs.hidden_states[-1]``. For the gemma4_unified text
        # model that output is the POST-norm ``last_hidden_state`` (the model only
        # returns ``last_hidden_state``; there is no recorded pre-norm tuple), so
        # the drafter seed is captured AFTER ``self.norm``.
        post_norm_hidden = ttnn.clone(hidden_states) if (is_decode and return_hidden) else None

        # Traced prefill returns post-norm hidden states and runs the lm_head
        # OUTSIDE the trace, on just the last-token tile (see
        # process_logits_after_prefill_trace). The lm_head over the full padded
        # sequence (262k vocab) dwarfs the entire model body — ~40x the body at
        # 4k tokens — so baking it into the trace makes traced prefill far
        # SLOWER than non-traced. The last-token slice can't be baked into the
        # trace (the index varies per prompt), so the whole lm_head is deferred
        # to host-side post-processing on a 32-row slice of these hidden states.
        if not is_decode and getattr(self, "_prefill_trace_mode", False):
            return hidden_states

        # Speculative decoding: logits and the returned drafter seed both come
        # from the post-final-norm hidden, matching the target model's
        # ``last_hidden_state`` used by the assistant candidate generator.
        # lm_head deallocates its input.
        if is_decode and return_hidden:
            # Spec-decode reads full-vocab logits on host — never keep sharded.
            logits = self._apply_lm_head(hidden_states, is_decode=True, keep_sharded_for_sampling=False)
            return logits, post_norm_hidden

        # Slice to the last token tile before lm_head when caller only wants
        # next-token logits (prefill). Keeps the 262k-vocab matmul output at
        # 32 rows instead of seq_len rows — without this, prefill at seq_len
        # >= 4k OOMs DRAM on smaller WH SKUs (lm_head logits = seq_len * vocab
        # * 2B; at seq=4096 that's 2 GiB, doesn't fit in DRAM with weights).
        if get_last_token != -1:
            # get_last_token is an absolute position, so gather the CP-sharded
            # sequence before selecting the globally addressed last-token tile.
            hidden_states = self._cp_gather_prefill_sequence(hidden_states)
            tile_start = (int(get_last_token) // 32) * 32
            hidden_states = ttnn.slice(
                hidden_states,
                (0, 0, tile_start, 0),
                (1, 1, tile_start + 32, hidden_states.shape[-1]),
            )

        logits = self._apply_lm_head(
            hidden_states,
            is_decode=is_decode,
            keep_sharded_for_sampling=bool(keep_sharded_for_sampling and is_decode),
        )
        if not is_decode:
            # After lm_head only — mid-forward / pre-lm_head flush corrupts token-0 on TP.
            self._flush_deferred_bounded_fills_if_needed()
        return logits

    def _flush_deferred_bounded_fills_if_needed(self):
        """Commit stashed bounded ring K/V after lm_head (or batched-hidden return)."""
        if getattr(self, "bounded_sliding_kv_cache", False):
            flush_deferred_bounded_fills(self.layers)

    def _cp_block_pool_override(self, current_override, paged_attention_config, bounded_sliding_kv_cache):
        """Shrink the paged block pool to this rank's share under context parallelism.

        Each CP rank computes the K/V for the tokens it owns, so that slice *is* the
        part of the cache it should hold. Keeping the cache sharded along the CP axis
        means the fill needs no gather and no per-device write offset: a rank's pool
        is its shard, so local block indices suffice (which is why the page table can
        stay a replicated identity — see ``_identity_page_table``). Per-chunk offsets
        are already carried by ``chunk_page_table``, which does not vary by rank.

        The win is capacity and write bandwidth, both 1/cp: at long context the KV
        cache is what bounds how much context fits per device.

        The global position of a token is implied by (rank, local block, offset) —
        that permutation has to be undone when the cache is handed to the decode
        side; see ``export_paged_kv_cache_natural_order``.

        Refuses the bounded-sliding combination rather than silently mis-sizing it: a
        bounded pool holds one sliding window, and how a global window maps onto
        per-rank shards is a separate design question.
        """
        from models.demos.gemma4.tt.ccl import cp_degree

        cp = cp_degree(self.mesh_config)
        if cp <= 1 or paged_attention_config is None:
            return current_override
        if bounded_sliding_kv_cache:
            raise NotImplementedError(
                "bounded_sliding_kv_cache with context parallelism is not supported: the bounded pool "
                "holds one global sliding window, and its mapping onto per-rank sequence shards is "
                "undefined here. Run with GEMMA4_DISABLE_CP=1, or use an unbounded cache."
            )
        base_blocks = current_override if current_override is not None else paged_attention_config.max_num_blocks
        if base_blocks % cp != 0:
            raise ValueError(
                f"paged block pool {base_blocks} is not divisible by the CP degree {cp}; "
                f"each rank must own a whole number of blocks"
            )
        return base_blocks // cp

    def _cp_gather_prefill_sequence(self, hidden_states):
        """Gather a CP-sharded prefill sequence back to full length.

        Required before any slice by ABSOLUTE position. Those indices are mesh-wide
        scalars, but under context parallelism each rank holds a different span of
        the sequence, so the same scalar would select different (wrong) tokens on
        every rank — the position wanted actually lives on exactly one of them.

        Cheap in the place it is used: ~44 MB at 4k x 5376 bf16, and the lm_head
        that follows only ever sees the 32-row slice. Gathering the *logits*
        instead would be ~4.3 GB at a 262k vocab, which is why the head is fed a
        slice in the first place.

        Deliberately does not deallocate its input: one call site hands us a trace
        output the caller still owns.
        """
        from models.demos.gemma4.tt.ccl import cp_degree

        if cp_degree(self.mesh_config) <= 1:
            return hidden_states
        return ttnn.all_gather(
            hidden_states,
            dim=2,
            cluster_axis=self.mesh_config.sp_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _apply_lm_head(self, hidden_states, is_decode=False, keep_sharded_for_sampling=False, deallocate_input=True):
        """Project post-norm hidden states to vocab logits, softcap, all-gather.

        Factored out of ``__call__`` so traced prefill can defer it (the trace
        returns post-norm hidden states and this runs on a 32-row last-token
        slice outside the trace; see ``process_logits_after_prefill_trace``).

        - lm_head is column-parallel on the vocab dim when TP > 1.
        - Decode + prefill last-token both feed an M=32-row tile here (decode
          batch <=32 pads to a tile; prefill is sliced to 32 above), so the
          1D-mcast program config from ``_get_lm_head_program_config`` is shared
          across both paths. ttnn.linear's default heuristic picks a generic
          config that doesn't account for the 1024-N-tile width of the 262k-vocab
          shard; pinning an explicit MatmulMultiCoreReuseMultiCast1DProgramConfig
          keeps the split across the full compute grid (8x8 WH / 8x10 BH)
          deterministic.
        - Softcapping (``tanh(logits/cap)*cap``) is element-wise and works on the
          sharded vocab. ttnn.mul/ttnn.tanh are not in-place, so the results are
          captured — dropping them silently no-ops the cap and tanks PCC vs HF.
        - The sharded vocab is all-gathered back to full width unless
          ``keep_sharded_for_sampling`` (on-device sampling consumes shards).
          Host-sample decode must gather — skipping solely because
          ``self.sampling`` exists truncates argmax to ~vocab/TP and causes
          thought-loop / garbage generations on TP meshes.
        """
        # Bracket the lm_head matmul + softcap with a Tracy signpost so the
        # op_perf_results.py --signpost gemma4_lm_head filter sums just this
        # region (issue #44953 — measure LM head dispatch share of decode step).
        # Gated on is_decode so prefill last-token calls don't mix into the
        # decode region totals.
        if is_decode:
            signpost(header=LM_HEAD_SIGNPOST)
        if self.lm_head_weight is not None:
            lm_head_pc = _get_lm_head_program_config(
                self.mesh_device,
                m=hidden_states.shape[2],
                k=self.hidden_size,
                n=self.lm_head_weight.shape[-1],
            )
            logits = ttnn.linear(hidden_states, self.lm_head_weight, program_config=lm_head_pc)
            # ``deallocate_input=False`` is required when the caller owns a
            # *persistent* buffer that outlives this call — notably the batched
            # prefill-sampling trace, whose input is written by
            # copy_host_to_device_tensor on every replay. Freeing it inside the
            # capture leaves that copy writing into dead memory (segfault).
            if deallocate_input:
                hidden_states.deallocate(True)
        else:
            logits = hidden_states

        if self.final_logit_softcapping and self.final_logit_softcapping > 0:
            cap = self.final_logit_softcapping
            logits = ttnn.mul(logits, 1.0 / cap)
            logits = ttnn.tanh(logits)
            logits = ttnn.mul(logits, cap)
        if is_decode:
            signpost(header=LM_HEAD_SIGNPOST)

        if self.mesh_config is not None and self.mesh_config.tp > 1 and self.lm_head_weight is not None:
            if keep_sharded_for_sampling:
                pass  # On-device sampling module consumes TP-sharded logits.
            else:
                from models.demos.gemma4.tt.ccl import ccl_allgather

                logits = ccl_allgather(logits, self.mesh_config, self.ccl_manager)

        return logits

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
            from models.demos.gemma4.tt.ccl import ccl_allgather

            embeds = ccl_allgather(embeds, self.mesh_config, self.ccl_manager)
        return embeds

    def raw_embed(self, tokens):
        """Token embedding table lookup without the sqrt(hidden) scale.

        This helper exposes the raw table for diagnostics/compatibility. The
        it-assistant drafter path intentionally uses ``embed_tokens()``, matching
        HF ``get_input_embeddings()(ids)`` where Gemma4 applies the embedding
        scale inside the module.
        """
        if self.embedding_weight is None:
            raise RuntimeError("Embedding weights not loaded")
        embeds = ttnn.embedding(tokens, self.embedding_weight, dtype=ttnn.bfloat16)
        if self.mesh_config is not None and self.mesh_config.tp > 1:
            embeds = ttnn.unsqueeze_to_4D(embeds)
            from models.demos.gemma4.tt.ccl import ccl_allgather

            embeds = ccl_allgather(embeds, self.mesh_config, self.ccl_manager)
        return embeds

    def get_shared_kv_caches(self):
        """Return the target KV caches the it-assistant drafter cross-attends to.

        ``{layer_type: [k_cache, v_cache]}`` for the last full-attention and last
        sliding-attention layer — the EAGLE/MTP ``shared_kv_states`` contract.
        """
        return {lt: self.tt_kv_cache[idx] for lt, idx in self.last_kv_layer_by_type.items()}

    def ttnn_verify_forward(
        self, x, current_pos, current_pos_cache=None, page_table=None, kv_cache=None, page_tables_per_layer=None
    ):
        """Multi-token speculative *verify* forward (batch holds the candidates).

        The K candidate tokens occupy the batch dimension at consecutive
        positions ``current_pos = [p+1, ..., p+K]`` with the user's page-table
        row replicated K times. This reuses the ordinary batched-decode path:
        ``paged_update_cache`` writes all K tokens' KV before SDPA, so the
        per-position ``paged_scaled_dot_product_attention_decode`` (with the
        per-batch ``cur_pos`` and sliding window) yields exactly-correct causal +
        sliding-window verify attention — token p+i attends to [0..p+i] (full) or
        the last window (sliding). Rejected positions are simply overwritten on
        the next iteration (KV rollback = position bookkeeping at batch=1).

        Args:
            x: [1, K] uint32 candidate token ids (or precomputed [1,1,K,hidden] embeds).
            current_pos: [1,32] uint32 padded positions (first K = p+1..p+K).
            page_table: [K, num_blocks] int32 (the user's row replicated K times).
            kv_cache: optional KV cache override (defaults to self.tt_kv_cache).

        Returns:
            (logits, hidden) — logits [1,1,K,vocab] from the post-norm hidden;
            ``hidden`` is the post-final-norm hidden [1,1,K,hidden], the
            it-assistant drafter's recurrent seed.
        """
        if x.dtype in (ttnn.uint32, ttnn.int32):
            input_embeds = self.embed_tokens(x)
            if len(input_embeds.shape) == 3:
                input_embeds = ttnn.unsqueeze_to_4D(input_embeds)
            input_embeds = ttnn.to_layout(input_embeds, ttnn.TILE_LAYOUT)
        else:
            input_embeds = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        token_index = None if self.rope_caches_2d else 0
        if page_tables_per_layer is None:
            page_tables_per_layer = getattr(self, "_active_page_tables_per_layer", None)
        page_tables_per_layer = self._page_tables_to_ttnn(page_tables_per_layer)

        return self(
            hidden_states=input_embeds,
            position_idx=current_pos,
            page_table=page_table,
            kv_caches=kv_cache,
            is_decode=True,
            token_index=token_index,
            position_idx_cache=current_pos_cache if current_pos_cache is not None else current_pos,
            page_tables_per_layer=page_tables_per_layer,
            return_hidden=True,
            # Default True (race-safe). A timing/experiment harness can set
            # `_verify_seq_kv_write=False` to measure the cost of the per-candidate
            # serialized KV-write loop (KV is corrupted when False — timing only).
            sequential_kv_write=getattr(self, "_verify_seq_kv_write", True),
        )

    def ttnn_packed_verify_forward(
        self,
        x,
        position_idx,
        attn_mask_full,
        attn_mask_sliding,
        packed_p,
        page_table=None,
        kv_cache=None,
        kv_write_idxs=None,
        embed_idx_full=None,
        embed_idx_sliding=None,
        hot_pt=None,
    ):
        """Packed-query speculative verify — all P candidates in ONE batch=1 pass.

        Unlike ``ttnn_verify_forward`` (candidates in the batch dim, K+1
        pseudo-users, sequential per-candidate KV writes), this packs the P =
        K+1 positions into the query-heads dim: one QKV projection / norm /
        RoPE over P rows, ONE non-causal SDPA per layer with an additive mask
        that bakes in each packed row's causal upper bound (and the sliding
        window on sliding layers), and a loop-free staging KV write (one
        paged_fill_cache per K/V) when staging is provided.

        Args:
            x: [1, P] uint32 token ids ``[anchor, d1..dK]``.
            position_idx: [1, P] uint32 positions (p..p+K), used for RoPE
                gathers; also reused row-wise for the KV-write fallback.
            attn_mask_full / attn_mask_sliding: [1, 1, H_local*P, S_k] bf16
                TILE additive masks (S_k a multiple of 64).
            packed_p: P.
            kv_write_idxs: optional list of P int32 [1] tensors (per-position
                fallback writes when staging isn't wired).
            embed_idx_full / embed_idx_sliding: [1, nkv_local*S2] uint32 merge
                gather indices (loop-free staging path; nkv differs per type).
            hot_pt: [1, PV_HOT_BLOCKS] int32 physical fill pages (-1 = skip).

        Returns:
            (logits [1,1,P,vocab], hidden [1,1,P,hidden]) — same contract as
            ``ttnn_verify_forward``.
        """
        input_embeds = self.embed_tokens(x)
        if len(input_embeds.shape) == 3:
            input_embeds = ttnn.unsqueeze_to_4D(input_embeds)
        input_embeds = ttnn.to_layout(input_embeds, ttnn.TILE_LAYOUT)

        # Pre-gather RoPE once per layer type (identical for all layers of a
        # type — saves 2 embedding gathers per layer).
        rope_packed = {}
        for lt, (cos_2d, sin_2d) in self.rope_caches_2d.items():
            cos_bp = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, cos_2d, layout=ttnn.TILE_LAYOUT))
            sin_bp = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, sin_2d, layout=ttnn.TILE_LAYOUT))
            rope_packed[lt] = (cos_bp, sin_bp)

        packed = {
            "packed_p": packed_p,
            "position_idx": position_idx,
            "kv_write_idxs": kv_write_idxs,
            "attn_mask_full": attn_mask_full,
            "attn_mask_sliding": attn_mask_sliding,
            "rope_packed": rope_packed,
            "embed_idx_full": embed_idx_full,
            "embed_idx_sliding": embed_idx_sliding,
            "hot_pt": hot_pt,
        }

        out = self(
            hidden_states=input_embeds,
            position_idx=position_idx,
            page_table=page_table,
            kv_caches=kv_cache,
            is_decode=True,
            token_index=None if self.rope_caches_2d else 0,
            return_hidden=True,
            packed=packed,
        )
        for cos_bp, sin_bp in rope_packed.values():
            cos_bp.deallocate(True)
            sin_bp.deallocate(True)
        return out

    def compute_host_pli(self, token_id):
        """Compute per-layer input (PLI) on CPU for a single decode token.

        Main token embeddings are looked up on device via ``embed_tokens``;
        this path only builds the E2B/E4B PLI tensor still computed on host.

        Returns:
            pli_combined: torch.Tensor [1, 1, n_layers, pli_size] bfloat16, or None
        """
        import torch.nn.functional as F

        if not self.hidden_size_per_layer_input or not self.per_layer_input_weights:
            return None

        token_tensor = torch.tensor([[token_id]], dtype=torch.long)
        embeds = F.embedding(token_tensor, self._embed_weight_cpu).float() * self.embed_scale
        pli_list = self._compute_per_layer_inputs(token_tensor.int(), embeds)
        if pli_list is None:
            return None
        return torch.stack(pli_list, dim=2)  # [1, 1, n_layers, pli_size]

    def compute_host_embeddings(self, token_id):
        """Host token embedding + PLI (legacy fallback).

        Decode should use ``embed_tokens`` on device plus ``compute_host_pli``.
        Kept for callers/tests that still compare against the old host path.

        Returns:
            (embeds, pli_combined) where:
            - embeds: torch.Tensor [1, 1, 1, hidden_size] bfloat16
            - pli_combined: torch.Tensor [1, 1, n_layers, pli_size] bfloat16, or None
        """
        import torch.nn.functional as F

        token_tensor = torch.tensor([[token_id]], dtype=torch.long)
        embeds = F.embedding(token_tensor, self._embed_weight_cpu).float() * self.embed_scale
        pli_combined = self.compute_host_pli(token_id)
        embeds = embeds.reshape(1, 1, 1, self.hidden_size).to(torch.bfloat16)
        return embeds, pli_combined

    # ── Generator-compatible interface ────────────────────────────────────

    def _replicate_to_mesh_mapper(self):
        """ReplicateTensorToMesh on multi-device meshes; None on single device."""
        is_mesh = hasattr(self.mesh_device, "shape")
        if is_mesh and self.mesh_device.get_num_devices() > 1:
            return ttnn.ReplicateTensorToMesh(self.mesh_device)
        return None

    def _page_table_torch_to_ttnn(self, page_table_torch):
        """Build a page-table device tensor from a torch tensor.

        Prefill is usually batch=1; decode warmup/runtime pass the full
        ``max_batch_size`` rows. Preserve the host batch dim so
        ``paged_update_cache`` sees ``page_table.shape[0] == input.shape[1]``.
        """
        pt = page_table_torch if page_table_torch.dim() > 1 else page_table_torch.unsqueeze(0)
        return ttnn.from_torch(
            pt,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._replicate_to_mesh_mapper(),
        )

    @staticmethod
    def _host_page_tables_batch(page_tables_per_layer) -> int | None:
        for pt in page_tables_per_layer or []:
            if pt is not None and isinstance(pt, torch.Tensor):
                return int(pt.shape[0]) if pt.dim() > 1 else 1
        return None

    def _page_tables_to_ttnn(self, page_tables_per_layer):
        """Lazy-allocate persistent device tensors for per-layer page tables.

        Buffers are stored **per host batch** so B=1 and B=max decode traces
        each bind stable addresses without cross-batch realloc (which used to
        orphan Metal decode traces and hang B=32 after sequential prefill).

        Within a batch key: never shrink; pad host up to the device shape in
        :meth:`update_persistent_per_layer_page_tables`. Grow only when host
        is strictly larger (and invalidate traces).

        Layers in upstream's HMA tensor-sharing layout that point at the
        same DRAM buffer still get their own page table object here — the
        per-layer block IDs are what differs across them, even when the
        underlying KV tensor is shared.
        """
        if page_tables_per_layer is None:
            return None
        by_batch = getattr(self, "_persistent_pt_by_batch", None)
        if by_batch is None:
            by_batch = {}
            self._persistent_pt_by_batch = by_batch
        batch_key = self._host_page_tables_batch(page_tables_per_layer)
        if batch_key is None:
            return None
        persistent = by_batch.get(batch_key)
        n = len(page_tables_per_layer)
        needs_alloc = persistent is None or len(persistent) != n
        needs_grow = False
        if not needs_alloc and persistent is not None:
            for i, pt in enumerate(page_tables_per_layer):
                if pt is None or isinstance(pt, ttnn.Tensor) or persistent[i] is None:
                    continue
                try:
                    host_b = int(pt.shape[0]) if pt.dim() > 1 else 1
                    host_w = int(pt.shape[-1]) if pt.dim() > 1 else int(pt.shape[0])
                    dev_b = int(persistent[i].shape[0])
                    dev_w = int(persistent[i].shape[-1])
                except (TypeError, IndexError, AttributeError):
                    continue
                if host_b > dev_b or host_w > dev_w:
                    needs_grow = True
                    break
        if needs_alloc or needs_grow:
            if needs_grow:
                # Growing after decode-trace capture would leave execute_trace
                # reading stale addresses; force a recapture on the next decode.
                self._invalidate_decode_traces_after_page_table_realloc = True
            persistent = []
            for pt in page_tables_per_layer:
                if pt is None:
                    persistent.append(None)
                    continue
                if isinstance(pt, ttnn.Tensor):
                    persistent.append(pt)
                    continue
                persistent.append(self._page_table_torch_to_ttnn(pt))
            by_batch[batch_key] = persistent
        self._persistent_per_layer_page_tables = persistent
        return persistent

    @staticmethod
    def _pad_page_table_host_to_shape(pt_host, target_b, target_w):
        """Pad a host page table with 0 up to ``(target_b, target_w)``."""
        pt_host = pt_host if pt_host.dim() > 1 else pt_host.unsqueeze(0)
        host_b, host_w = int(pt_host.shape[0]), int(pt_host.shape[-1])
        if host_b == target_b and host_w == target_w:
            return pt_host
        if host_b > target_b or host_w > target_w:
            # Caller should have grown the device buffer already.
            return pt_host[:target_b, :target_w].contiguous()
        out = torch.zeros((target_b, target_w), dtype=torch.int32)
        out[:host_b, :host_w] = pt_host.to(dtype=torch.int32)
        return out

    def update_persistent_per_layer_page_tables(self, page_tables_per_layer):
        """Update the content of persistent per-layer page-table device
        tensors in place.

        Trace replay reads block IDs from stable device addresses, so we
        ``copy_host_to_device`` rather than reallocate. Called by the
        vLLM hybrid bridge before each forward (out-of-trace) so the
        next traced call observes the new block IDs.

        Prefill often passes B < max_batch (e.g. sequential B=31); pad the
        host table to the captured device shape so we never reallocate and
        orphan decode-trace addresses. Skip the H2D when the host table is
        unchanged from the last update for this batch key (decode steps
        within a KV block).
        """
        if page_tables_per_layer is None:
            return
        batch_key = self._host_page_tables_batch(page_tables_per_layer)
        persistent = self._page_tables_to_ttnn(page_tables_per_layer)
        if persistent is None:
            return
        last_by_batch = getattr(self, "_last_host_pt_by_batch", None)
        if last_by_batch is None:
            last_by_batch = {}
            self._last_host_pt_by_batch = last_by_batch
        last_hosts = last_by_batch.get(batch_key)
        # Sliding layers often share one remapped host table; copy each unique
        # (padded) host tensor once and fan out to every persistent buffer.
        host_cache = {}
        new_last = [None] * len(page_tables_per_layer)
        for i, pt in enumerate(page_tables_per_layer):
            if pt is None or persistent[i] is None or isinstance(pt, ttnn.Tensor):
                continue
            pt_host = pt if pt.dim() > 1 else pt.unsqueeze(0)
            try:
                target_b = int(persistent[i].shape[0])
                target_w = int(persistent[i].shape[-1])
            except (TypeError, IndexError, AttributeError):
                target_b = int(pt_host.shape[0])
                target_w = int(pt_host.shape[-1])
            pt_padded = self._pad_page_table_host_to_shape(pt_host, target_b, target_w)
            if (
                last_hosts is not None
                and i < len(last_hosts)
                and last_hosts[i] is not None
                and torch.equal(last_hosts[i], pt_padded)
            ):
                new_last[i] = last_hosts[i]
                continue
            new_last[i] = pt_padded.detach().clone() if pt_padded is not None else None
            # Cache key includes target shape so B=1 and B=32 pads don't collide.
            key = (id(pt), target_b, target_w)
            host_pt = host_cache.get(key)
            if host_pt is None:
                host_pt = ttnn.from_torch(
                    pt_padded,
                    device=None,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=self._replicate_to_mesh_mapper(),
                )
                host_cache[key] = host_pt
            ttnn.copy_host_to_device_tensor(host_pt, persistent[i])
        last_by_batch[batch_key] = new_last

    def prepare_inputs_prefill(
        self,
        tokens,
        start_pos=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        trace_enabled=False,
        last_token_idx=None,
        global_user_id=None,
        batch_size=1,
        user_id=0,
        batched_prefill=False,
        **kwargs,
    ):
        """Build prefill device inputs and cache the host-side state needed
        for per-layer inputs.

        Returns a 6-tuple matching
        ``models/tt_transformers/tt/model.py:prepare_inputs_prefill``:
        ``(tt_input, None, None, tt_page_table, tt_chunk_page_table,
        tt_chunk_start_idx)``. ``tt_input`` is host-staged token IDs when
        ``trace_enabled`` (so the trace owns the embed step) and tile-laid
        embeddings otherwise. The two ``None`` slots are placeholders for
        ``rot_mats_global``/``rot_mats_local`` — Gemma4 computes RoPE
        internally from layer state. ``tt_chunk_start_idx`` is a device
        scalar when tracing multi-chunk / APC (so RoPE + chunked SDPA can
        refresh the absolute start via ``copy_host_to_device``); otherwise
        ``None`` and the generator passes a Python int into
        ``ttnn_prefill_forward``.
        """
        import torch.nn.functional as F

        del start_pos, last_token_idx, global_user_id, user_id, batched_prefill, kwargs

        device = None if trace_enabled else self.mesh_device
        mesh_mapper = self._replicate_to_mesh_mapper()

        tokens_torch = tokens.to(torch.long)
        if batch_size > 1:
            assert tokens_torch.dim() == 2, "batched prefill tokens must be [batch, seq_len]"
            per_user_seq_len = tokens_torch.shape[-1]
            tokens_for_embed = tokens_torch.reshape(1, 1, 1, -1)
        else:
            per_user_seq_len = tokens_torch.shape[-1]
            # Match test_full_model / vLLM parity: [1, seq_len] token rows, not
            # [1, 1, 1, seq_len]. The flattened layout is for batched-prefill streams.
            tokens_for_embed = tokens_torch

        tt_tokens = ttnn.from_torch(
            tokens_for_embed,
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )

        tt_page_table = None
        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mesh_mapper,
            )

        tt_chunk_page_table = None
        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mesh_mapper,
            )

        # Device scalar for traced multi-chunk / APC: refresh absolute start via
        # copy_host_to_device (RoPE + chunked SDPA flexible offset). Skip for
        # cold single-chunk (start=0, no chunk_page_table) so existing 4k traces
        # stay unchanged.
        tt_chunk_start_idx = None
        if trace_enabled and chunk_start_idx is not None:
            start_i = int(chunk_start_idx) if not isinstance(chunk_start_idx, ttnn.Tensor) else None
            need_tensor = chunk_page_table is not None or (start_i is not None and start_i > 0)
            if isinstance(chunk_start_idx, ttnn.Tensor):
                tt_chunk_start_idx = chunk_start_idx
            elif need_tensor:
                tt_chunk_start_idx = ttnn.from_torch(
                    torch.tensor([int(chunk_start_idx)], dtype=torch.int32),
                    device=device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=mesh_mapper,
                )

        self._prefill_input_ids_torch = tokens_torch
        self._prefill_batch_size = batch_size
        self._prefill_seq_len_per_user = per_user_seq_len
        if self._embed_weight_cpu is not None:
            self._prefill_embeds_torch = F.embedding(tokens_torch, self._embed_weight_cpu).float() * self.embed_scale
        else:
            self._prefill_embeds_torch = None

        if trace_enabled:
            return tt_tokens, None, None, tt_page_table, tt_chunk_page_table, tt_chunk_start_idx

        tt_embeds = self.embed_tokens(tt_tokens)
        if batch_size > 1:
            if len(tt_embeds.shape) == 3:
                tt_embeds = ttnn.unsqueeze_to_4D(tt_embeds)
        else:
            tt_embeds = ttnn.reshape(tt_embeds, (1, 1, per_user_seq_len, self.hidden_size))
        tt_embeds = ttnn.to_layout(tt_embeds, ttnn.TILE_LAYOUT)

        return tt_embeds, None, None, tt_page_table, tt_chunk_page_table, None

    def prepare_prefill_inputs_trace(self, tokens, **kwargs):
        return self.prepare_inputs_prefill(tokens, trace_enabled=True, **kwargs)

    def _reshape_prefill_embeds(self, tt_embeds, seq_len):
        if len(tt_embeds.shape) == 3:
            return ttnn.reshape(tt_embeds, (1, 1, seq_len, self.hidden_size))
        if tt_embeds.shape[2] != seq_len:
            return ttnn.reshape(tt_embeds, (1, 1, seq_len, self.hidden_size))
        return tt_embeds

    def transform_and_embed_prefill_inputs_device(
        self, tokens, tt_page_table, tt_chunk_page_table, tt_chunk_start_idx=None
    ):
        """Inside-trace input transform: lookup embeddings and tile-lay them.

        Called when the captured trace owns the embed step (so the input
        tensor is the raw token tensor staged by ``prepare_inputs_prefill``
        with ``trace_enabled=True``).

        ``tt_chunk_start_idx`` is threaded through unchanged so the return
        tuple lines up with ``Generator``'s traced-prefill unpack
        (``transformed_inputs[3]`` → ``ttnn_prefill_forward(chunk_start_idx=...)``).
        """
        if len(tokens.shape) == 4 and tokens.shape[1] == 1 and tokens.shape[2] == 1:
            seq_len = tokens.shape[3]
            tokens = ttnn.reshape(tokens, (1, seq_len))
        else:
            seq_len = tokens.shape[-1]
        tt_embeds = self.embed_tokens(tokens)
        tt_embeds = self._reshape_prefill_embeds(tt_embeds, seq_len)
        tt_embeds = ttnn.to_layout(tt_embeds, ttnn.TILE_LAYOUT)
        return tt_embeds, tt_page_table, tt_chunk_page_table, tt_chunk_start_idx

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
        input_ids_torch=None,
        embeds_torch=None,
        pli_device_tensors=None,
        page_tables_per_layer=None,
        valid_seq_lens=None,
        **kwargs,
    ):
        """Prefill forward — Generator-compatible signature.

        Generator-irrelevant kwargs (``rot_mats_*``) are accepted and discarded —
        the model computes RoPE internally. ``chunk_start_idx`` /
        ``chunk_page_table`` drive generator-level multi-chunk prefill (eager
        Python int or traced device tensor). ``input_ids_torch``/``embeds_torch``
        may be passed directly by callers that compute them inline (text demos,
        unit tests); the Generator path stashes them on ``self`` during
        ``prepare_inputs_prefill`` and they're picked up here when the explicit
        kwargs are None.

        ``page_tables_per_layer`` likewise comes via a stash
        (``_active_page_tables_per_layer``) when running under the vLLM
        hybrid bridge — Generator's prefill internals don't thread the
        kwarg, so the bridge attaches it to the model object before
        invoking us. When None, falls back to legacy single-page-table
        behavior.

        ``get_last_token`` is passed down so the last-token slice happens
        *before* lm_head — slicing after would still allocate full-seq
        logits first. ``valid_seq_lens`` is the per-slot real token count for
        batched prefill KV fill (hetero prompts in one pad bucket).
        """
        del rot_mats_global, rot_mats_local, kwargs
        if input_ids_torch is None:
            input_ids_torch = self._prefill_input_ids_torch
        if embeds_torch is None:
            embeds_torch = self._prefill_embeds_torch
        if page_tables_per_layer is None:
            page_tables_per_layer = getattr(self, "_active_page_tables_per_layer", None)
        page_tables_per_layer = self._page_tables_to_ttnn(page_tables_per_layer)
        return self(
            hidden_states=x,
            position_idx=None,
            page_table=page_table,
            kv_caches=kv_cache,
            is_decode=False,
            input_ids_torch=input_ids_torch,
            embeds_torch=embeds_torch,
            pli_device_tensors=pli_device_tensors,
            get_last_token=get_last_token,
            page_tables_per_layer=page_tables_per_layer,
            batch_size=batch_size,
            user_id=user_id,
            chunk_start_idx=chunk_start_idx,
            chunk_page_table=chunk_page_table,
            valid_seq_lens=valid_seq_lens,
        )

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

    def process_logits_after_prefill_trace(self, hidden_states, last_token_idx):
        """Deferred lm_head for traced prefill.

        The trace returns post-norm hidden states ``[1,1,seq,hidden]`` when
        ``_prefill_trace_mode`` is set (lm_head skipped inside the trace).
        Slice the 32-row tile containing ``last_token_idx`` and run lm_head +
        softcap on those rows only.

        If the last dim is already vocab-sized (legacy / batched path that ran
        lm_head inside the trace), only slice and return.

        Under CP the trace returns a sequence-sharded tensor, so it is gathered
        first: ``last_token_idx`` is absolute and a mesh-wide scalar cannot address
        a per-rank span.
        """
        hidden_states = self._cp_gather_prefill_sequence(hidden_states)
        get_last_token = (last_token_idx // 32) * 32
        sliced = ttnn.slice(
            hidden_states,
            (0, 0, get_last_token, 0),
            (1, 1, get_last_token + 32, hidden_states.shape[-1]),
        )
        if sliced.shape[-1] == self.hidden_size:
            logits = self._apply_lm_head(sliced, is_decode=False)
        else:
            logits = sliced
        # Trace deferred lm_head: commit bounded ring fills after logits.
        self._flush_deferred_bounded_fills_if_needed()
        return logits

    def extract_last_tokens_batched_prefill(
        self, hidden_states, last_token_idx_list, padded_batch, prefill_seq_len, target_batch=None
    ):
        """Extract each user's last-token hidden from batched prefill output.

        Generator reshapes deferred post-norm hidden to
        ``[padded_batch, 1, prefill_seq_len, H]`` then calls this before
        on-device prefill sampling (``GEMMA4_HOST_SAMPLE=0``).

        Gemma4 residuals are full-width (replicated) after embed all-gather /
        layer all-reduces — unlike tt_transformers column-sharded activations —
        so host gather uses one device shard and re-uploads with
        ``ReplicateTensorToMesh``. Returns ``[1, 1, target_batch, H]``.
        """
        del prefill_seq_len  # layout already [B,1,S,H]; S unused after reshape
        active_indices = [lt for lt in last_token_idx_list if lt > 0]
        all_same = len(set(active_indices)) <= 1

        if all_same and active_indices:
            common_last = active_indices[0]
            get_last = (common_last // 32) * 32
            R = common_last % 32
            block = ttnn.slice(
                hidden_states,
                (0, 0, get_last, 0),
                (padded_batch, 1, get_last + 32, hidden_states.shape[-1]),
            )
        else:
            block = hidden_states
            R = None

        host_tensors = [ttnn.to_torch(dt) for dt in ttnn.get_device_tensors(block)]
        if host_tensors[0].shape[-1] == self.hidden_size:
            host_full = host_tensors[0]
        else:
            host_full = torch.cat(host_tensors, dim=-1)

        if R is not None:
            combined = host_full[:, :, R : R + 1, :].reshape(1, 1, padded_batch, -1).contiguous()
        else:
            rows = []
            for slot in range(padded_batch):
                lt_idx = int(last_token_idx_list[slot])
                rows.append(host_full[slot : slot + 1, :, lt_idx : lt_idx + 1, :])
            combined = torch.cat(rows, dim=0).reshape(1, 1, padded_batch, -1).contiguous()

        target_batch = padded_batch if target_batch is None else int(target_batch)
        if target_batch < padded_batch:
            raise ValueError(f"target_batch {target_batch} must be >= padded_batch {padded_batch}")
        if target_batch > padded_batch:
            padded_combined = torch.zeros(
                1,
                1,
                target_batch,
                combined.shape[-1],
                dtype=combined.dtype,
            )
            padded_combined[:, :, :padded_batch, :] = combined
            combined = padded_combined

        return ttnn.from_torch(
            combined,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._replicate_to_mesh_mapper(),
        )

    def _apply_norm_and_lm_head(self, x, deallocate_input=True):
        """Batched-prefill sampling: final norm already applied; run lm_head.

        Called by ``Generator`` after :meth:`extract_last_tokens_batched_prefill`.
        Keep logits TP-sharded for on-device sampling.

        ``deallocate_input=False`` when ``x`` is a persistent trace-input buffer
        that must survive the capture (see
        ``Gemma4Generator._capture_trace_prefill_sampling``).
        """
        return self._apply_lm_head(
            x, is_decode=False, keep_sharded_for_sampling=True, deallocate_input=deallocate_input
        )

    def switch_mode(self, mode):
        """Generator compatibility — no prefetcher to reinitialize."""

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """Create host tensors for one decode step (token IDs + optional PLI).

        Called by Generator._capture_decode_trace_text and _decode_forward_trace_text.
        Returns tuple of host ttnn tensors that copy_host_to_device will transfer.

        Index 0 is a uint32 token tensor (not precomputed embeddings). Embedding
        lookup runs on device inside ``ttnn_decode_forward`` via ``embed_tokens``.

        Args:
            tokens: torch.Tensor [batch] of token IDs
            current_pos: torch.Tensor [batch] of current positions
            page_table: optional torch.Tensor [batch, max_blocks] page table
        """
        import torch.nn.functional as F

        is_mesh = hasattr(self.mesh_device, "shape")
        replicate = (
            ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh and self.mesh_device.get_num_devices() > 1 else None
        )

        tok_flat = tokens.reshape(-1)
        pos_flat = current_pos.reshape(-1)
        batch = tok_flat.shape[0]

        # Stage token IDs (not embeddings): embed_tokens runs on device in
        # ttnn_decode_forward. Non-PLI models pad to sampling width [1,1,1,32] so
        # ``ttnn.sampling(output_tensor=...)`` can write the next token into this
        # buffer for async continuity. PLI models keep compact [1, batch] and
        # restage from host every step (always_refresh=True).
        # int64 (not int32) source: ttnn downcasts int64 to uint32 host-side, so the
        # C++ to_dtype path is skipped. An int32->uint32 conversion would instead query
        # tile metadata on a row-major host buffer and emit the #18536 warning.
        tok_i64 = tok_flat.to(torch.int64)
        if self._tt_vllm_always_refresh_decode_trace_inputs:
            tok_host = tok_i64.reshape(1, batch)
        else:
            pad_w = self._DECODE_TOKEN_FEEDBACK_WIDTH
            if batch > pad_w:
                raise ValueError(f"Decode batch {batch} exceeds token feedback width {pad_w}")
            if batch < pad_w:
                tok_i64 = F.pad(tok_i64, (0, pad_w - batch), "constant", 0)
            tok_host = tok_i64.reshape(1, 1, 1, pad_w)
        tokens_tt = ttnn.from_torch(
            tok_host,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )

        # Position: [1, 32] uint32 padded — per-user positions in the first
        # `batch` entries. The decode RoPE embedding lookup gathers one cos/sin
        # row per user, so different users can sit at different positions.
        # int64 source for the uint32 tensor (see tokens above): avoids the int32->uint32
        # host conversion that triggers the #18536 row-major get_tile() warning.
        #
        # Inactive decode rows (vLLM pad) use position -1 so paged_update / SDPA
        # skip them (kernel treats -1 as UINT32_MAX). RoPE embedding cannot take
        # that sentinel — clamp negatives to 0 for the uint32 lookup only; the
        # int32 cache/SDPA tensor below keeps the real -1 skip markers.
        pos_i64 = pos_flat.to(torch.int64).clone()
        pos_rope = pos_i64.clone()
        pos_rope[pos_rope < 0] = 0
        pos_rope = pos_rope.reshape(1, batch)
        pos_padded = F.pad(pos_rope, (0, 32 - batch), "constant", 0) if batch < 32 else pos_rope
        pos_tt = ttnn.from_torch(pos_padded, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=replicate)

        # int32 positions [batch] for KV cache update + SDPA (per user).
        pos_int32_tt = ttnn.from_torch(
            pos_i64.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=replicate
        )

        # Page table [batch, max_blocks] — one row per user.
        page_table_tt = None
        if page_table is not None:
            pt = page_table if page_table.dim() > 1 else page_table.unsqueeze(0)
            page_table_tt = ttnn.from_torch(pt, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=replicate)

        # PLI (E2B/E4B per-layer inputs). 31B has none. Batched PLI would need
        # per-user stacking + model-side per-user slicing — not yet wired up.
        pli_tt = None
        if self.hidden_size_per_layer_input and self.per_layer_input_weights:
            if batch != 1:
                raise NotImplementedError("Batched decode with per-layer inputs (E2B/E4B) is not yet supported")
            _, pli = self.compute_host_embeddings(int(tok_flat[0].item()))
            if pli is not None:
                pli_tt = ttnn.from_torch(
                    pli.to(torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
                )

        return (tokens_tt, pos_tt, pos_int32_tt, page_table_tt, pli_tt)

    def prepare_inputs_decode(self, tokens, current_pos, page_table=None):
        """Wrapper: prepare_decode_inputs_host + copy to device."""
        from models.tt_transformers.tt.common import copy_host_to_device

        host_inputs = self.prepare_decode_inputs_host(tokens, current_pos, page_table)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
        self.bind_decode_trace_inputs(device_inputs)
        return device_inputs

    def bind_decode_trace_inputs(self, device_inputs):
        """Stash extra (>4) device inputs on ``self`` so
        ``ttnn_decode_forward`` can pick them up.

        ``Generator``'s decode paths only thread the first four
        elements of ``prepare_inputs_decode``'s return tuple through
        the call signature; anything beyond that — Gemma4's
        host-precomputed per-layer-input (PLI) at index 4 — has to
        reach the model via a side channel. ``Generator`` calls this
        hook in both the no-trace path (through this wrapper) and at
        trace-capture time (so traced ops bind against
        ``trace_inputs_decode[i][4]`` rather than the compile-run
        buffer); see :meth:`Generator._capture_decode_trace_text`.
        """
        if len(device_inputs) > 4:
            self._decode_pli_combined = device_inputs[4]

    def ttnn_decode_forward(
        self,
        x,
        current_pos,
        rot_mat_idxs=None,
        page_table=None,
        kv_cache=None,
        on_device_logits=False,
        pli_combined=None,
        page_tables_per_layer=None,
    ):
        """Decode forward — matches tt_transformers Generator interface.

        x is a uint32 token tensor from prepare_decode_inputs_host (ROW_MAJOR).
        Generator calls: prepare_decode_inputs_host → copy_host_to_device → ttnn_decode_forward.

        Args:
            x: [1,1,1,1] or [1,1] uint32 ROW_MAJOR device tensor (decode token id).
            current_pos: [1,32] uint32 position tensor for RoPE embedding lookup.
            rot_mat_idxs: Unused (RoPE computed internally from current_pos).
            page_table: Optional paged attention table.
            kv_cache: Optional KV cache override.
            on_device_logits: If True, return logits in on-device sampling layout.
            pli_combined: Optional [1,1,n_layers,pli_size] device tensor of host-precomputed
                per-layer inputs (E2B/E4B). Required for Gemma3n-style models in decode.
            page_tables_per_layer: Optional list of per-layer page tables. Falls back to
                ``self._active_page_tables_per_layer`` (set by the vLLM hybrid bridge,
                since ``Generator``'s decode path doesn't thread the kwarg).
        """
        # Two input conventions are accepted:
        #   * uint32/int32 token-id tensor → run embed_tokens on device. This is
        #     the batched-decode path (one device embedding op handles all B
        #     users; the host-embedding path is hardcoded single-token).
        #   * bf16 pre-computed embedding → use directly (legacy / unit tests).
        # Active batch comes from the int32 KV-position tensor (exact B). The
        # token buffer may be sampling-width padded ([1,1,1,32]) for feedback.
        active_batch = None
        if rot_mat_idxs is not None:
            active_batch = int(rot_mat_idxs.shape[-1])
        if x.dtype in (ttnn.uint32, ttnn.int32):
            x_embed = x
            if active_batch is not None and len(x.shape) == 4 and int(x.shape[-1]) > active_batch:
                x_embed = ttnn.slice(x, [0, 0, 0, 0], [1, 1, 1, active_batch])
            if len(x_embed.shape) == 4:
                # embed_tokens expects a compact [1, B] (or [B]) id tensor.
                x_embed = ttnn.reshape(x_embed, (1, int(x_embed.shape[-1])))
            input_embeds = self.embed_tokens(x_embed)
            if len(input_embeds.shape) == 3:
                input_embeds = ttnn.unsqueeze_to_4D(input_embeds)
            input_embeds = ttnn.to_layout(input_embeds, ttnn.TILE_LAYOUT)
        else:
            input_embeds = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # RoPE: always use internal 2D caches with on-device embedding lookup
        token_index = None if self.rope_caches_2d else 0

        position_idx_cache = rot_mat_idxs  # Generator passes pos_int32 as rot_mat_idxs

        if page_tables_per_layer is None:
            page_tables_per_layer = getattr(self, "_active_page_tables_per_layer", None)
        page_tables_per_layer = self._page_tables_to_ttnn(page_tables_per_layer)

        # ``Generator``'s decode path slices ``prepare_inputs_decode``'s
        # return tuple to its first 4 elements before calling here, so
        # the PLI tensor produced by ``prepare_decode_inputs_host`` for
        # E2B/E4B per-layer inputs is dropped on the way in. Fall back
        # to the cached value the host-prep step stashed on ``self``.
        if pli_combined is None:
            pli_combined = self._decode_pli_combined

        logits = self(
            hidden_states=input_embeds,
            position_idx=current_pos,
            page_table=page_table,
            kv_caches=kv_cache,
            is_decode=True,
            token_index=token_index,
            position_idx_cache=position_idx_cache,
            pli_combined=ttnn.to_layout(pli_combined, ttnn.TILE_LAYOUT) if pli_combined is not None else None,
            page_tables_per_layer=page_tables_per_layer,
            # Only skip vocab all-gather when this step feeds on-device sampling.
            # Host-sample decode must gather full 262k vocab (see _apply_lm_head).
            keep_sharded_for_sampling=on_device_logits,
        )

        if on_device_logits:
            assert self.sampling is not None, (
                "decode forward got on_device_logits=True but no on-device sampling "
                "module exists (self.sampling is None)."
            )
            # Advance device positions for the next decode step (async-safe).
            # Mirror tt_transformers Transformer._increment_decode_positions_device.
            # ``rot_mat_idxs`` is Gemma4's int32 cache/SDPA position buffer (vLLM
            # pads inactive decode rows with -1). Without skip_negative, those
            # rows leave the skip sentinel (-1→0→1…) and paged_update can touch
            # KV. Page-table pad is 0 (null block); skip is the position sentinel.
            if not self._tt_vllm_always_refresh_decode_trace_inputs:
                if current_pos is not None:
                    ttnn.plus_one(current_pos, skip_negative_entries=True)
                if rot_mat_idxs is not None:
                    ttnn.plus_one(rot_mat_idxs, skip_negative_entries=True)
            batch_dim = logits.shape[2]
            if batch_dim < 32:
                logits = ttnn.pad(logits, padding=[(0, 0), (0, 0), (0, 32 - batch_dim), (0, 0)], value=0.0)
            return logits

        return logits, None

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False, is_log_probs=False):
        """Read decode logits or sampled tokens to host.

        Under TP, decode logits are already all-gathered across devices
        inside the model forward, so a single device tensor contains the
        full vocab.
        """
        if is_tokens or is_log_probs:
            if self.mesh_config is not None and self.mesh_config.tp > 1:
                torch_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
            else:
                torch_out = ttnn.to_torch(tt_out)
            return torch_out.reshape(-1)[:B]

        if self.mesh_config is not None and self.mesh_config.tp > 1:
            torch_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
        else:
            torch_out = ttnn.to_torch(tt_out)
        return torch_out[:, :, :B, : self.vocab_size].view(B, S, -1)


def _apply_gemma4_single_untilize_override(tt_sampling) -> None:
    """Keep Gemma4's wide-vocab argmax on the single-untilize path (Blackhole).

    Upstream ``TTSampling`` (#53167) untilizes wide logit rows in
    ``TOPK_MAX_WIDTH`` (64Ki) chunks and rebuilds the row with
    ``ttnn.concat(..., dim=3)``. The chunking avoids a wide-row untilize clash,
    but the concat re-materializes the same full-width row, and for Gemma4's
    262144 vocab that needs a ~4MB circular-buffer page against Blackhole's
    ~1.43MB per-core L1::

        TT_FATAL: ttnn.concat: required CB page size (4194304 B)
                  exceeds per-core L1 capacity (1461376 B)

    That aborts on-device sampling at init, forcing the host path (a full
    262144-vocab logits readback every decode step) and costing ~36% decode
    throughput (12B/P150x8 batch-32: 32.5 -> 19.2 tok/s/user).

    ``_untilize_chunk_count`` is a ``@staticmethod`` invoked as
    ``self._untilize_chunk_count(...)``, so an instance attribute shadows it.
    Scoping the override to Gemma4's own ``TTSampling`` instance leaves the
    shared ``models/common/sampling/tt_sampling.py`` untouched for every other
    model.

    Measured good on Blackhole (12B P150x8, batch-1 and batch-32, coherent
    output) and now on Wormhole too: on a real WH T3K the upstream chunk+concat
    path *hard-fails* rather than degrading, because Gemma4's 262144-vocab row
    needs the same ~4MB CB page against WH's ~1.33MB per-core L1::

        TT_FATAL: ttnn.concat: required CB page size (4194304 B)
                  exceeds per-core L1 capacity (1393472 B)

    That aborts prefill warmup, so text_demo_v2 batch-1 / batch-8 / batch-32 all
    fail outright on WH. With the single-untilize override the same three cases
    pass with coherent per-user output at 24.9 / 20.5 / 16.2 tok/s (12B, T3K), so
    the wide-row clash the chunking guards against does not reproduce here.
    Enabled on both arches; override with ``GEMMA4_SAMPLING_SINGLE_UNTILIZE``
    (1 = force on, 0 = force off) to fall back to upstream chunking.
    """
    env = os.environ.get("GEMMA4_SAMPLING_SINGLE_UNTILIZE")
    if env is not None:
        enable = env.lower() in ("1", "true", "yes")
    else:
        enable = True
    if not enable or tt_sampling is None:
        return
    if not hasattr(type(tt_sampling), "_untilize_chunk_count"):
        # Upstream dropped/renamed the hook - leave stock behaviour alone.
        return
    # Plain function, not staticmethod(): instance attributes bypass the
    # descriptor protocol, so this is called unbound as f(width). Wrapping in
    # staticmethod() only works on py>=3.10 where those objects became directly
    # callable; this form has no version dependency.
    tt_sampling._untilize_chunk_count = lambda width: 1
    logger.info("Gemma4 sampling: single-untilize argmax path (avoids wide-row ttnn.concat L1 overflow)")
