# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 prefill runtime (single-rank and pipeline-parallel).

Mirrors the DeepSeek ``TtPrefillRuntime`` contract so the model-agnostic prefill engine
(``models/demos/common/prefill/runners/prefill_runner.py``) can drive M3 through
``MiniMaxM3PrefillAdapter``: build model -> compile(kv_cache) -> prefill_chunk(chunk, kv_cache) per
chunk. The runtime is STATELESS w.r.t. the KV cache — the engine allocates it (via the adapter's
``allocate_kv_cache``) and passes it into every call that touches it. A runtime owns one rank's layer
slice (first_layer_idx / is_first_rank / is_last_rank); the config defaults make a single-rank runtime
own the whole model. For pipeline-parallel prefill the engine ships the cross-rank hidden state over the
D2D socket: a non-first rank receives an activation (not tokens) and a non-last rank returns its hidden
state instead of filling the LM head. KV-chunk-table migration is supported (single-rank) via
``build_kv_chunk_table`` (a multi-config table; see ``tt/runners/kv_chunk_table.py``).

Input convention (shared with the engine's H2D socket): ``make_forward_input`` returns the forward's
token IDs as an SP-sharded uint32 tensor — the SAME per-chip layout the request-mode socket delivers.
``prefill_forward`` embeds that token tensor on device, so the standalone (trace) and request (socket)
paths feed identical tensors into the forward and share one code path. The embedding lives here (M3
embeds in ``prepare_inputs_prefill``, separate from the model forward), unlike DeepSeek which embeds
inside ``forward``.

The cache holds ``num_users * num_layers`` rows (slot-major: cache_row = slot_id * num_layers + layer_idx); each
``prefill_forward`` call fills every PrefillChunk into its slot_id at its absolute KV offset, running the
cache-read attention path against the already-written prefix. ``prefill_chunk`` is the one-chunk form.

Reference: models/demos/deepseek_v3_d_p/tt/tt_prefill_runtime.py (interface) and
tests/galaxy_prefill_kv_pcc.py (cache readback + RoPE swizzle convention).
"""

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.common.utils import block_cyclic_reorder
from models.demos.common.prefill.chunk_layout import pack_forward_tokens
from models.demos.common.prefill.prefill_chunk import PrefillChunk, validate_chunks
from models.demos.minimax_m3.utils.general_utils import sparse_attention_freq


@dataclass
class TtPrefillRuntimeConfig:
    num_layers: int  # layers built/cached by this runtime (== model total for single-rank)
    max_seq_len: int  # per-user KV-cache length in tokens; must be a multiple of chunk_size
    mesh_shape: tuple = (8, 4)  # (SP rows, TP cols) on the Blackhole galaxy
    chunk_size: int = 5120  # tokens per chunk, the KV / RoPE block-cyclic period; one-shot sets this == max_seq_len
    # Tokens per forward: forward_tokens_budget // chunk_size chunks (prefill_forward). None = chunk_size.
    forward_tokens_budget: Optional[int] = None
    num_users: int = 1  # independent cache slots (user-major batch)
    sp_axis: int = 0
    tp_axis: int = 1
    # Topology of the legacy CCLs (high_bw_all_gather derives its own from the fabric); see tt/ccl.py.
    topology: ttnn.Topology = ttnn.Topology.Linear
    use_ep_moe: bool = True
    expert_weight_dtype: ttnn.DataType = ttnn.bfloat4_b
    cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    weight_cache_path: Optional[Path] = None
    # Pipeline-parallel rank slicing. first_layer_idx is the GLOBAL index of this rank's first layer;
    # is_first_rank gates the embedding + token input, is_last_rank marks the final stage. The defaults
    # make a single-rank runtime own the whole model.
    first_layer_idx: int = 0
    # Explicit global layer indices to build, overriding the contiguous run. Profiling only: lets a
    # 2-layer run cover one dense and one sparse layer ([0, 3]) instead of the four it takes to reach
    # the first sparse one. Requires a complete tilized weight cache.
    layer_indices: Optional[list] = None
    is_first_rank: bool = True
    is_last_rank: bool = True
    # Emb-axis sharding of the cross-rank D2D hidden state (must match the runner's D2D_MAPPER_CONFIG and
    # this model's residual layout): True => emb TP-sharded, False => emb replicated across TP. M3's SP
    # residual is emb-replicated, so its adapter passes False.
    pipeline_activation_emb_tp_sharded: bool = True
    # The runner picks the per-layer ack transport from this: traced runs use the host callback, untraced
    # ones the D2H service. M3 has no traced path, so it stays False.
    use_trace: bool = False
    # MoE layers: run the shared expert on its own sub-device concurrently with dispatch (the runner's
    # PrefillRunParams.overlap_shared_expert_with_dispatch). Off: it runs before the MoE on the full grid.
    overlap_shared_expert: bool = True

    def __post_init__(self):
        if self.forward_tokens_budget is None:
            self.forward_tokens_budget = self.chunk_size

    @property
    def forward_chunks(self) -> int:
        return self.forward_tokens_budget // self.chunk_size

    @property
    def sp_factor(self) -> int:
        return self.mesh_shape[self.sp_axis]

    @property
    def tp_factor(self) -> int:
        return self.mesh_shape[self.tp_axis]


class TtPrefillRuntime:
    """Single-rank prefill execution lifecycle: build model -> compile(kv_cache) ->
    prefill_chunk(chunk, kv_cache). Owns the SP=8 × TP=4 ``Model``; the KV cache is engine-owned and
    passed into every call."""

    def __init__(self, mesh_device, hf_config, state_dict: dict, config: TtPrefillRuntimeConfig):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.config = config

        assert (
            config.max_seq_len % config.chunk_size == 0
        ), f"max_seq_len ({config.max_seq_len}) must be a multiple of chunk_size ({config.chunk_size})"
        budget = config.forward_tokens_budget
        assert (
            budget >= config.chunk_size and budget % config.chunk_size == 0
        ), f"forward_tokens_budget ({budget}) must be a positive multiple of chunk_size ({config.chunk_size})"
        # The KV cache, its slot addressing and gather_layer are all sized by config.num_layers, while
        # the model builds one layer per entry of layer_indices. If those disagree, a layer addresses
        # past the per-user cache stride and silently corrupts another slot.
        assert config.layer_indices is None or len(config.layer_indices) == config.num_layers, (
            f"layer_indices has {len(config.layer_indices)} entries but num_layers is "
            f"{config.num_layers}; they size the model and the KV cache respectively and must match"
        )

        self.model_built = False
        self.compiled = False
        self._on_layer_complete = None
        # Per-layer completion sink for pipelined prefill, registered via set_layer_completion_sink().
        self._layer_completion_sink = None

        # The Model builds `hf_config.num_hidden_layers` decoder layers, but the KV cache (and gather /
        # PCC) is sized to `config.num_layers`. Pin them equal so a partial-model run (PREFILL_NUM_LAYERS
        # < 60) builds exactly the layers the cache holds — otherwise the model runs all 60 layers and
        # writes past the cache's per-user layer stride. A full run (num_layers == 60) is a no-op.
        self.hf_config.num_hidden_layers = config.num_layers

        self._build_model(state_dict)
        # RoPE is built once here (whole-cache indexed rope). The KV cache is engine-owned — the adapter
        # allocates it and passes it into compile / prefill_chunk / gather_layer — so it is NOT built here.
        self._build_indexed_rope()

    def _build_model(self, state_dict: dict) -> None:
        from models.demos.minimax_m3.config import MeshConfig
        from models.demos.minimax_m3.utils.general_utils import get_default_num_links

        from .ccl import CCLManager
        from .model import Model

        rows, cols = self.config.mesh_shape
        logger.info(
            f"Building TtPrefillRuntime model: num_layers={self.config.num_layers} "
            f"first_layer_idx={self.config.first_layer_idx} is_first_rank={self.config.is_first_rank} "
            f"is_last_rank={self.config.is_last_rank} max_seq_len={self.config.max_seq_len} "
            f"chunk_size={self.config.chunk_size} forward_tokens_budget={self.config.forward_tokens_budget} "
            f"num_users={self.config.num_users} "
            f"mesh_shape={self.config.mesh_shape}"
        )
        mesh_config = MeshConfig((rows, cols), tp=cols)
        ccl = CCLManager(
            self.mesh_device, num_links=get_default_num_links(self.mesh_device), topology=self.config.topology
        )
        self.model = Model(
            mesh_device=self.mesh_device,
            hf_config=self.hf_config,
            state_dict=state_dict,
            ccl_manager=ccl,
            mesh_config=mesh_config,
            tensor_cache_path=self.config.weight_cache_path,
            max_local_batch_size=1,
            sequence_parallel=True,
            use_ep_moe=self.config.use_ep_moe,
            ep_seq_len_per_chip=self.config.forward_tokens_budget // self.config.sp_factor,
            expert_weight_dtype=self.config.expert_weight_dtype,
            first_layer_idx=self.config.first_layer_idx,
            layer_indices=self.config.layer_indices,
            is_first_rank=self.config.is_first_rank,
            is_last_rank=self.config.is_last_rank,
            overlap_shared_expert=self.config.overlap_shared_expert,
        )
        self.model_built = True

    # --- RoPE: whole-cache cos/sin for the INDEXED on-device rope (rotary_embedding_indexed), built ONCE
    # and reused for every chunk. Replaces the old per-chunk to_torch/slice/from_torch reshard (a D2H2D
    # host roundtrip each prefill call). The cos/sin cover every cache position, block-cyclic-reordered
    # (keyed by the per-chip chunk) then SP-sharded, so device c's contiguous shard holds -- in
    # local-cache-row order -- the rope for every global position it carries; the indexed op then derives
    # each chunk's start row on-device from kv_actual_global (the same block-cyclic math the KV-cache
    # writer uses). Mirrors DeepSeek RotarySetup.get_rope_tensors_indexed.
    def _build_indexed_rope(self) -> None:
        mesh = self.mesh_device
        sp = self.config.sp_factor
        cache_seq = self.config.max_seq_len  # cache capacity; % chunk_size == 0 (asserted in __init__)
        chunk_local = self.config.chunk_size // sp
        rdims = [None, None]
        rdims[self.config.sp_axis] = 2  # seq dim across SP rows
        mapper = ttnn.ShardTensor2dMesh(mesh, dims=tuple(rdims), mesh_shape=mesh.shape)

        def build(dev_mat):
            # one-time D2H of the model's replicated cos/sin (device-0 copy), sliced to the cache capacity
            full = ttnn.to_torch(ttnn.get_device_tensors(dev_mat)[0])
            assert full.shape[2] >= cache_seq, (
                f"KV capacity {cache_seq} exceeds the model's RoPE table ({full.shape[2]} positions = "
                f"max_position_embeddings); the indexed rope would be silently truncated"
            )
            full = full[:, :, :cache_seq, :]
            bc = block_cyclic_reorder(full, chunk_local, sp, seq_dim=2)
            return ttnn.from_torch(bc, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper)

        rs = self.model.rope_setup
        self.rope_indexed = [build(rs.cos_matrix_prefill), build(rs.sin_matrix_prefill)]

    def reconfigure_capacity(self, max_seq_len: int) -> None:
        """Re-target the resident model at a different per-user KV-cache capacity WITHOUT rebuilding it.

        Everything sized by ``max_seq_len`` lives outside the weights: the engine-owned KV cache (the
        caller re-allocates it via ``allocate_kv_caches`` and passes the new handle into ``compile`` /
        ``prefill_chunk``), the indexed RoPE tables (rebuilt here) and the config the asserts / compile
        sweep / ``gather_layer`` un-rotation read (swapped here). The CCL scratch buffers are keyed by
        shape, so a new capacity just adds an entry. ``forward_tokens_budget`` stays fixed: it is baked into the
        MoE dispatch buffers at build time. Call ``compile(new_kv_cache)`` afterwards — the JIT buckets
        for the new cache shapes are not warm.

        Why: on a 1D fabric the MSA index_k gather moves close to the whole slot capacity per chunk
        (msa._gather_tp_sharded_index_k), so a run in an over-sized cache pays for capacity it never reads.
        Fitting the cache to each run keeps a resident-model sweep faithful to a standalone run.
        """
        assert self.model_built
        assert (
            max_seq_len % self.config.chunk_size == 0
        ), f"max_seq_len ({max_seq_len}) must be a multiple of chunk_size ({self.config.chunk_size})"
        if max_seq_len == self.config.max_seq_len:
            return
        logger.info(f"TtPrefillRuntime.reconfigure_capacity: max_seq_len {self.config.max_seq_len} -> {max_seq_len}")
        old_config, old_rope = self.config, self.rope_indexed
        self.config = replace(self.config, max_seq_len=max_seq_len)
        try:
            self._build_indexed_rope()  # builds the NEW tables first: a failure leaves the old capacity intact
        except Exception:
            self.config, self.rope_indexed = old_config, old_rope
            raise
        for t in old_rope:
            ttnn.deallocate(t)
        # The CCL scratch (ring-gather / high_bw-gather) is keyed by shape and the cache-read entries are
        # capacity-sized: drop them so re-targets do not accumulate dead DRAM (~0.7 GB/chip per capacity at 1M).
        self.model.ccl_manager.release_scratch_buffers()
        self.compiled = False

    def release_sub_device_managers(self) -> None:
        """Remove the MoE overlap sub-device manager before the mesh is closed. Idempotent."""
        if self.model_built:
            self.model.release_sub_device_managers()

    def release_trace(self) -> None:
        """The prefill runner's pre-close hook. M3 has no trace; this releases the sub-device managers."""
        self.release_sub_device_managers()

    def make_placeholder_activation(self) -> ttnn.Tensor:
        """Zero hidden-state activation matching the decoder-layer-boundary residual and the D2D receiver
        backing: global ``[1, 1, forward_tokens_budget, hidden_size]`` bf16 TILE DRAM, sequence sharded across the SP
        rows and the embedding sharded (or replicated) across the TP cols per
        ``pipeline_activation_emb_tp_sharded`` — the same layout as the runner's D2D_MAPPER_CONFIG. Warm-up
        stand-in for a non-first pipeline rank, which receives the real activation over the D2D socket at
        run time."""
        cfg = self.config
        dims = [None, None]
        dims[cfg.sp_axis] = 2  # seq across SP rows
        if cfg.pipeline_activation_emb_tp_sharded:
            dims[cfg.tp_axis] = 3  # emb across TP cols (else replicated on tp -> left None)
        zeros = torch.zeros(1, 1, cfg.forward_tokens_budget, self.hf_config.hidden_size, dtype=torch.bfloat16)
        return ttnn.from_torch(
            zeros,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=tuple(dims), mesh_shape=cfg.mesh_shape),
        )

    def make_chunk_input(self, token_ids: list, actual_start: int = 0) -> ttnn.Tensor:
        """One chunk's input for ``prefill_chunk``: ``make_forward_input`` with a single chunk."""
        assert self.config.forward_chunks == 1, "a packed runtime takes make_forward_input / prefill_forward"
        return self.make_forward_input([token_ids], [actual_start])

    def make_forward_input(self, chunk_tokens: list, starts: list) -> ttnn.Tensor:
        """Build one forward's device input for ``prefill_forward``. On the first rank: the chunks' token IDs
        as an SP-sharded uint32 ROW_MAJOR DRAM tensor of per-chip shape ``(1, 1, forward_tokens_budget // sp)``,
        replicated across the TP cols. ``chunk_tokens`` holds one natural-order chunk_size token list per
        chunk (pad the tail; an empty chunk is all pad) and ``starts`` the chunks' actual_start. Chip r receives,
        in chunk order, each chunk's tokens at ``rotated_chunk_positions(start)[r]`` — the contiguous slice
        ``[r*s_local : (r+1)*s_local]`` for a chunk-aligned start, rotated for a mid-slab one (a multi-turn
        resume) so the KV writer and indexed RoPE see each token at its true position. This is the SAME
        per-chip layout the request-mode H2D socket delivers (the engine's H2D connector applies the same
        reshuffle), so both paths feed one code path; ``prefill_forward`` embeds it on device.

        On a non-first pipeline rank the input is a hidden-state activation (received over the D2D socket at
        run time), not token IDs — return a placeholder activation of the right spec for warm-up."""
        if not self.config.is_first_rank:
            return self.make_placeholder_activation()
        cfg = self.config
        n = cfg.forward_chunks
        assert len(chunk_tokens) == n, f"{len(chunk_tokens)} chunks for a forward of {n}"
        for token_ids in chunk_tokens:
            assert len(token_ids) == cfg.chunk_size, (
                f"chunk input must be exactly chunk_size={cfg.chunk_size} tokens (pad the tail), "
                f"got {len(token_ids)}"
            )
        sp = cfg.sp_factor
        tok = torch.tensor(pack_forward_tokens(chunk_tokens, starts, sp), dtype=torch.int32)
        tok = tok.reshape(sp, 1, cfg.forward_tokens_budget // sp)
        return ttnn.from_torch(
            tok,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=cfg.mesh_shape, dims=(cfg.sp_axis, None)),
        )

    def _embed_tokens(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """Embed the SP-sharded tokens into the bf16 hidden state the layers consume, delegating to the
        model's TtParallelEmbedding. Returns [1, 1, s_local, emb_dim] full hidden TP-replicated, or
        [1, 1, s_local, emb_dim/tp] under M3_SHARDED_RESIDUAL — whichever the residual-stream contract
        is (tt/residual.py); bf16 (not bf8) preserves dynamic range. The sharding (2D vocab+hidden by
        default, or 1D hidden-only) and its CCL live in the module (tt/parallel_embedding.py)."""
        x = self.model.embedding(tokens)
        if len(x.shape) == 3:
            x = ttnn.unsqueeze_to_4D(x)
        return x

    def compile(self, kv_cache) -> None:
        """Warm the program cache so no served forward pays a first-run JIT.

        For a fixed cache capacity a chunk runs one of exactly THREE program sets per layer, so three warm-up
        forwards cover the served loop: (i) the FIRST chunk (start == 0: no-cache attention variants,
        exact-size gathers), followed by cache-read chunks when forward_tokens_budget > chunk_size, (ii) a full
        CACHE-READ chunk (start > 0), (iii) a RAGGED chunk (end < start + chunk_size: the MoE padding-config
        variant). (ii) and (iii) fill the rest of the forward with empty chunks (zero rows). Every
        per-chunk offset — kv_actual / cached_len, logical_n, kv_len, chunk_start_idx, gathered_dim_size,
        slot — is a runtime argument excluded from the op hashes (update_padded_kv_cache, ring_joint SDPA with
        kv_actual_isl, indexer_score_msa, topk_large_indices valid_length, sparse_sdpa_msa,
        high_bw_all_gather), so sweeping every KV-length bucket (the previous behaviour: max_seq_len /
        chunk_size chunks, 196 at 1M) re-ran the same programs and still missed variant (iii). What IS hashed
        is the cache capacity itself (gather buffer shapes, sparse_sdpa T), which is why reconfigure_capacity()
        invalidates this warm-up. The warm-up writes slots the real run overwrites."""
        assert self.model_built
        cfg = self.config
        chunk, n_chunks = cfg.chunk_size, cfg.forward_chunks
        per_slot = cfg.max_seq_len // chunk
        cache_read_start = chunk if per_slot >= 2 else 0
        empty = PrefillChunk(slot_id=0, actual_start=0, actual_end=0)

        def fill(k):  # k-th chunk of filling slots 0, 1, ... in order; empty once every slot is full
            slot_id, i = divmod(k, per_slot)
            return PrefillChunk(slot_id, i * chunk, (i + 1) * chunk) if slot_id < cfg.num_users else empty

        plan = [
            [fill(k) for k in range(n_chunks)],
            [PrefillChunk(0, cache_read_start, cache_read_start + chunk)] + [empty] * (n_chunks - 1),
            [PrefillChunk(0, cache_read_start, cache_read_start + chunk // 2)] + [empty] * (n_chunks - 1),
        ]
        logger.info(
            f"TtPrefillRuntime.compile() — warming first / cache-read / ragged forwards ({n_chunks} x {chunk}-token "
            f"chunks, capacity {cfg.max_seq_len})"
        )
        t0 = time.perf_counter()
        for chunks in plan:
            inp = self.make_forward_input([[0] * chunk] * n_chunks, [c.actual_start for c in chunks])
            self.prefill_forward(inp, kv_cache, chunks)
        ttnn.synchronize_device(self.mesh_device)
        warmup_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"[prefill timing] task_id=WARMUP variants={len(plan)} runtime.compile() = {warmup_ms:.2f} ms")
        self.compiled = True

    def prefill_chunk(
        self,
        input_tensor: ttnn.Tensor,
        kv_cache,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        request_id: int = 0,
        d2h_service=None,
        record_dev=None,
        *,
        skip_lm_head: bool = True,
        get_last_token: int = -1,
        metadata_msg=None,
    ):
        """Prefill ONE chunk ``[actual_start, actual_end)`` into slot ``slot_id``: ``prefill_forward`` with a
        single chunk (forward_tokens_budget == chunk_size). See prefill_forward for the arguments."""
        assert self.config.forward_chunks == 1, "a packed runtime takes prefill_forward with forward_chunks chunks"
        return self.prefill_forward(
            input_tensor,
            kv_cache,
            [PrefillChunk(slot_id, actual_start, actual_end)],
            request_id,
            d2h_service,
            record_dev,
            skip_lm_head=skip_lm_head,
            get_last_token=get_last_token,
            metadata_msg=metadata_msg,
        )

    def prefill_forward(
        self,
        input_tensor: ttnn.Tensor,
        kv_cache,
        chunks: list,
        request_id: int = 0,
        d2h_service=None,
        record_dev=None,
        *,
        skip_lm_head: bool = True,
        get_last_token: int = -1,
        metadata_msg=None,
    ):
        """Prefill ONE forward of ``forward_tokens_budget // chunk_size`` ``chunks`` (common/prefill/prefill_chunk.py) into the
        engine-owned ``kv_cache``. Norms, projections and the MoE run once on all rows; the attention core
        (RoPE, KV write, SP gathers, SDPA) runs per non-empty chunk against its slot_id. With ``skip_lm_head``
        (the default) returns None — single-rank is headless, the populated cache is the output, read back
        via ``gather_layer``; otherwise returns the LM-head logits (one-chunk forwards only).

        Each chunk's [actual_start, actual_end) is the absolute KV-position range of its real tokens:
        actual_start is the cache write offset (the valid prefix already cached) and actual_end is past the last
        real token (the tail of a request's final chunk may be pad, so actual_end < actual_start + chunk_size).
        The chunk occupies physical positions [actual_start, actual_start + chunk_size); causality makes the pad
        tail inert. A slot's chunks are consecutive in pack order (only the last may be partial), so each
        chunk's KV is written before the next one reads it; across forwards, call in order. Empty chunks
        (actual_start == actual_end) fill the tail of a partial batch and touch no slot. If a LayerAck channel
        is registered, the model bumps it per layer.

        On a non-last pipeline rank this returns the slice's output hidden-state activation (which the
        engine ships to the next rank over the D2D socket); the last/single rank keeps its headless
        cache-fill contract (returns None under skip_lm_head).

        Args:
            input_tensor: on the first rank, this forward's SP-sharded uint32 token tensor (make_forward_input
                or the H2D socket), embedded on device here then deallocated. On a non-first rank, the
                hidden-state activation handed over the D2D socket — used directly (the first decoder layer
                consumes/frees it), NOT embedded.
            kv_cache: the engine-owned MiniMaxKVCache (from the adapter's allocate_kv_cache); the chunks'
                KV is written into it. The same object is passed on every call.
            chunks: the forward's PrefillChunk list, in pack order (validate_chunks).
            request_id: the runner's forward index, part of the runner's call contract; unused here.
            d2h_service: the device-side per-layer ack transport. When None, acks go through the host sink
                (set_layer_completion_sink).
            record_dev: the forward's metadata tensor, unused here.
            metadata_msg: the forward's raw socket metadata tensor, sent as the D2H ack record.
        """
        assert d2h_service is None or (
            metadata_msg is not None and len(chunks) == 1
        ), "the D2H ack needs metadata_msg and one chunk per forward (the record names one chunk)"
        cfg = self.config
        assert self.model_built, "build the model before prefill_forward()"
        # A chunk may start mid-slab (multi-turn continuation resumes at a tile boundary): the input rotation
        # (make_forward_input / the engine's H2D reshuffle), the KV writer, indexed rope, ring-joint SDPA, the MSA
        # indexer / sparse_sdpa_msa and the MoE padding config all derive this device's rotated positions from
        # the chunk's actual_start, on the writer's tile-row grid.
        validate_chunks(
            chunks,
            chunk_size=cfg.chunk_size,
            forward_chunks=cfg.forward_chunks,
            num_users=cfg.num_users,
            max_seq_len=cfg.max_seq_len,
        )
        assert skip_lm_head or len(chunks) == 1, "a packed forward is headless (KV only)"

        # First rank embeds the SP-sharded tokens. On a non-first rank the input is already the upstream
        # hidden state, fed straight in — the first decoder layer frees it, so don't deallocate it here.
        if cfg.is_first_rank:
            x_embd = self._embed_tokens(input_tensor)
            ttnn.deallocate(input_tensor)
        else:
            x_embd = input_tensor

        # Whole-cache indexed rope built once (self.rope_indexed); the indexed op picks each chunk's rows
        # on-device from kv_actual_global (= the chunk's actual_start, threaded via indexed_rope=True). No per-chunk
        # host reshard, and the tensors are persistent (do NOT deallocate them here). The KV cache is
        # engine-owned and passed in. If a LayerAck channel is registered, the model bumps it once per layer
        # via on_layer_complete.
        # on_layer_complete runs once the layer is enqueued; the ack must not reach the host before the
        # layer's cache writes have executed.
        if d2h_service is not None:

            def on_layer_complete(layer_idx: int) -> None:
                ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2h_service, metadata=metadata_msg)

        elif self._layer_completion_sink is not None:
            # Pipelined mode: bind the forward's chunks per call so the synchronous callback reads no mutable state.
            sink = self._layer_completion_sink

            def on_layer_complete(layer_idx: int) -> None:
                ttnn.synchronize_device(self.mesh_device)
                # The model reports a rank-local layer index. Empty chunks write no KV, so they report nothing.
                global_layer = cfg.first_layer_idx + layer_idx
                for c in chunks:
                    if not c.is_empty:
                        sink.layers_completed(global_layer, global_layer + 1, c.slot_id, c.actual_start, c.actual_end)

        else:
            on_layer_complete = self._on_layer_complete
        out = self.model.prefill_forward(
            x_embd,
            rot_mats_global=self.rope_indexed,
            kv_cache=kv_cache,
            get_last_token=get_last_token,
            skip_lm_head=skip_lm_head,  # default: cache-fill only (skip final norm + lm_head)
            indexed_rope=True,
            on_layer_complete=on_layer_complete,
            # Each chunk's real-token range drives its attention core and the MoE padding config. Without
            # the ranges the padded tails are routed and dispatched as if real: wasted dispatch work that
            # also eats per-expert dispatch-buffer capacity.
            chunks=chunks,
        )
        if not cfg.is_last_rank:
            # Middle rank: hand the slice's output hidden state to the next rank.
            return out
        if skip_lm_head:
            if out is not None:
                out.deallocate(True)
            return None
        return out  # logits [1,1,chunk_local,vocab_shard], SP-sharded on seq / TP-sharded on vocab

    def set_layer_completion_sink(self, sink) -> None:
        """Register the completion sink for pipelined prefill, fired per layer for each non-empty chunk of the
        prefill_forward() call."""
        assert self.compiled, "Call compile() before set_layer_completion_sink()"
        self._layer_completion_sink = sink

    def set_d2h_ack_service(self, service) -> None:
        """The runner calls this only for traced runtimes; this one takes the service per forward in prefill_forward."""
        raise NotImplementedError("MiniMax-M3 prefill does not trace")

    def gather_layer(self, kv_cache, slot_id: int, layer_idx: int, n_tokens: int):
        """Read one layer's device cache back to NATURAL token order (un-rotating the block-cyclic SP
        layout). Returns (k, v, index_k) torch tensors in DEVICE convention (K / index_k are Meta-RoPE
        swizzled over the rotary slice — the caller reconciles vs the HF golden). Shapes:
        k,v -> [1, num_kv_heads, n_tokens, head_dim]; index_k -> [1, 1, n_tokens, head_dim] (zeros on
        dense layers, which carry no index_k). Optional bring-up hook — never used in production
        serving.

        Thin wrapper over ``read_slot_kv`` + ``naturalize_kv_block``. A caller that walks EVERY layer
        should read the slot once and un-rotate each layer itself instead of calling this per layer.
        """
        from models.demos.minimax_m3.tt.runners.prefill_kv_validation import naturalize_kv_block

        k_blk, v_blk, ik_blk = self.read_slot_kv(kv_cache, slot_id)
        cfg = self.config
        return (
            naturalize_kv_block(k_blk[layer_idx], n_tokens, cfg.sp_factor, cfg.chunk_size, cfg.max_seq_len).unsqueeze(
                0
            ),
            naturalize_kv_block(v_blk[layer_idx], n_tokens, cfg.sp_factor, cfg.chunk_size, cfg.max_seq_len).unsqueeze(
                0
            ),
            naturalize_kv_block(ik_blk[layer_idx], n_tokens, cfg.sp_factor, cfg.chunk_size, cfg.max_seq_len).unsqueeze(
                0
            ),
        )

    def kv_migration_stages(self, kv_cache, first_layer_idx=None, num_my_layers=None):
        """One ``KvCacheStage`` per migratable device cache, in the order ``build_kv_chunk_table``
        consumes their gathered layouts: k, v, index_k. All three share one layer-index space (index_k
        allocates a slot for every layer, zeros on dense ones), so every stage carries the same range."""
        from models.demos.common.prefill.runners.migration import KvCacheStage

        first_layer_idx = self.config.first_layer_idx if first_layer_idx is None else int(first_layer_idx)
        num_my_layers = self.config.num_layers if num_my_layers is None else int(num_my_layers)
        return [
            KvCacheStage(int(t.buffer_address()), first_layer_idx, num_my_layers)
            for t in (kv_cache.k, kv_cache.v, kv_cache.index_k)
        ]

    def build_kv_chunk_table(
        self,
        kv_cache,
        path: str,
        *,
        first_layer_idx: int = 0,
        num_my_layers: Optional[int] = None,
        stage_layouts=None,
    ) -> str:
        """Build + serialize M3's multi-config KV chunk address table (k_h0..N, v_h0..N, index_k) to
        ``path`` and return it. This issues no comms; the runner then either publishes the table to the
        migration worker (PREFILL_ENABLE_MIGRATION=1) or leaves it on disk.

        Multi-rank (pipeline-parallel): this rank owns layers [first_layer_idx, first_layer_idx +
        num_my_layers). The runner all-gathers one layout per ``kv_migration_stages`` entry (k, v,
        index_k) and passes them as ``stage_layouts`` so ONLY rank 0 builds the table spanning every
        stage; with stage_layouts None the table covers config.num_layers == the full model from this
        rank's own mesh."""
        from models.demos.minimax_m3.tt.runners.kv_chunk_table import build_and_serialize_kv_chunk_table

        c = self.config
        return build_and_serialize_kv_chunk_table(
            mesh_device=self.mesh_device,
            kv_cache=kv_cache,
            seq_len=c.max_seq_len,
            num_layers=c.num_layers,
            mesh_shape=c.mesh_shape,
            sp_axis=c.sp_axis,
            num_users=c.num_users,
            chunk_size=c.chunk_size,
            num_kv_heads=self.hf_config.num_key_value_heads,
            head_dim=self.hf_config.head_dim,
            path=path,
            stage_layouts=stage_layouts,
            index_k_layers=self.msa_layer_ids(),
        )

    def msa_layer_ids(self) -> set[int]:
        """Global ids of the MSA (block-sparse) layers — the only layers that write an index_k (layers 3-59 on
        M3, every layer when sparse_attention_freq is all 1). Read through the same helper as ``Layer``, so the table can't disagree with the model.
        """
        ids = {i for i, f in enumerate(sparse_attention_freq(self.hf_config) or []) if f}
        if not ids:
            logger.warning(
                "[migration] the config has no MSA layers (sparse_attention_freq missing or use_sparse_attention "
                "off): the KV chunk table publishes no index_k rows"
            )
        return ids

    def read_slot_kv(self, kv_cache, slot_id: int, n_tokens: int | None = None):
        """Read one slot's KV cache from device to host: ``[k, v, index_k]``, one host tensor per cache
        tensor, each ``[num_layers, heads(or 1), seq_read, head_dim]`` (index_k collapsed to one TP
        replica), in the raw on-device (block-cyclic) layout — not un-rotated to natural token order.

        ``n_tokens`` bounds the read to the first ``ceil(n_tokens / chunk_size)`` chunks: the block-cyclic
        writer puts slab k of every chip at local rows ``[k*chunk_local, (k+1)*chunk_local)``, so the first
        ``k`` chunks live in the first ``k*chunk_local`` rows of EVERY chip and the composed block is exactly
        the layout of a ``k*chunk_size`` cache — un-rotate it with ``naturalize_kv_block(...,
        max_seq_len=k*chunk_size)`` (see ``read_seq_len``). Without the bound a 1M-capacity slot is ~120 GB
        of fp32 per cache on host. ``None`` reads the whole capacity.

        One device slice + one mesh compose per cache. DRAM_MEMORY_CONFIG on the slice is required —
        the cache is ND-sharded ROUND_ROBIN_1D, and slicing into another ND-shard miscomputes the DRAM
        core on host read-back.

        A TP-deduped index_k (``kv_cache.index_k_tp_axis``) is returned in the SAME layout as the replicated
        one: its per-column stripes are re-interleaved on host, so callers need no special case.
        """
        start = slot_id * self.config.num_layers
        end = start + self.config.num_layers
        seq_read = self.read_seq_len(n_tokens)
        rows = seq_read // self.config.sp_factor  # per-chip rows holding the first seq_read positions
        composer = ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(2, 1), mesh_shape=self.mesh_device.shape)

        def _slot_block(tensor, *, collapse_tp: bool = False):
            """This slot's packed-cache rows, gathered to host in one mesh compose.

            ``ConcatMesh2dToTensor`` ``dims=(2, 1)`` so SP concatenates on seq and TP on heads — one
            ``to_torch`` instead of per-chip ``get_device_tensors``. ``collapse_tp`` keeps a single TP
            replica (index_k is replicated across cols). Returns
            ``[end - start, heads(or 1), seq_cache, head_dim]`` in on-device (block-cyclic) seq order.
            """
            s = list(tensor.shape)
            sl = ttnn.slice(
                tensor,
                [start, 0, 0, 0],
                [end, s[1], min(rows, s[2]), s[3]],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            host = ttnn.to_torch(sl, mesh_composer=composer).float()
            ttnn.deallocate(sl)
            return host[:, :1] if collapse_tp else host

        def _tp_sharded_slot_block(tensor):
            """A TP-deduped cache's slot, re-interleaved to the TP-replicated layout ``[L, 1, sp*rows, hd]``.

            Chip (s, t) holds stripe t (``chunk_local/tp`` rows) of every slab n at local rows
            ``n*stripe + i``; the compose stacks the cols on dim 1, so ``host[:, t, s*n_slabs*stripe +
            n*stripe + i]`` moves to ``s*n_slabs*chunk_local + n*chunk_local + t*stripe + i``."""
            sp, tp = self.config.sp_factor, self.mesh_device.shape[kv_cache.index_k_tp_axis]
            chunk_local = self.config.chunk_size // sp
            stripe = chunk_local // tp
            assert rows % chunk_local == 0, f"read of {rows} rows/chip is not a whole number of {chunk_local}-row slabs"
            n_slabs = rows // chunk_local
            s = list(tensor.shape)
            assert n_slabs * stripe <= s[2], f"read of {n_slabs} slabs past the {s[2]}-row index_k cache"
            sl = ttnn.slice(
                tensor,
                [start, 0, 0, 0],
                [end, s[1], n_slabs * stripe, s[3]],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            host = ttnn.to_torch(sl, mesh_composer=composer).float()  # [L, tp, sp*n_slabs*stripe, hd]
            ttnn.deallocate(sl)
            n_l, hd = host.shape[0], host.shape[3]
            host = host.reshape(n_l, tp, sp, n_slabs, stripe, hd).permute(0, 2, 3, 1, 4, 5)
            return host.reshape(n_l, 1, sp * n_slabs * chunk_local, hd)

        if kv_cache.index_k_tp_axis is None:
            index_k = _slot_block(kv_cache.index_k, collapse_tp=True)
        else:
            index_k = _tp_sharded_slot_block(kv_cache.index_k)
        return [
            _slot_block(kv_cache.k),
            _slot_block(kv_cache.v),
            index_k,
        ]

    def read_seq_len(self, n_tokens: int | None) -> int:
        """Cache length (tokens) that ``read_slot_kv(n_tokens=...)`` returns: the first ``ceil(n_tokens /
        chunk_size)`` chunks, i.e. the ``max_seq_len`` to pass to ``naturalize_kv_block`` for that block."""
        if n_tokens is None:
            return self.config.max_seq_len
        chunk = self.config.chunk_size
        return min(self.config.max_seq_len, -(-n_tokens // chunk) * chunk)

    def kv_cache_pcc_check(
        self,
        kv_cache,
        *,
        slot_id: int,
        n_chunks: int,
        trace_dir=None,
        first_layer_idx: int = 0,
        real_len=None,
        pt_path_override=None,
    ) -> float:
        """Optional bring-up hook (never called in production serving). PCC the populated engine-owned
        ``kv_cache`` for ``slot_id`` against the golden trace; returns the min per-layer PCC and asserts
        on failure (unless PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1). Thin forwarder into the model's
        validation module so the PCC logic lives in one place. ``real_len`` caps the compared extent to the
        real (non-pad) tokens; ``pt_path_override`` (per-slot .pt golden) is unsupported for M3 (trace-dir
        goldens only) and rejected if set."""
        if pt_path_override is not None:
            raise NotImplementedError(
                "MiniMax-M3 kv_cache_pcc_check has no per-slot .pt golden path; use PREFILL_TRACE_DIR"
            )
        from models.demos.minimax_m3.tt.runners.prefill_kv_validation import kv_cache_pcc_check

        return kv_cache_pcc_check(
            self,
            kv_cache,
            slot_id=slot_id,
            n_chunks=n_chunks,
            trace_dir=trace_dir,
            first_layer_idx=first_layer_idx,
            real_len=real_len,
        )
