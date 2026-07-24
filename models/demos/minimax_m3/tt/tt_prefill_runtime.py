# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 single-rank prefill runtime.

Mirrors the DeepSeek ``TtPrefillRuntime`` contract so the model-agnostic prefill engine
(``models/demos/common/prefill/runners/prefill_runner.py``) can drive M3 through
``MiniMaxM3PrefillAdapter``: build model -> compile(kv_cache) -> prefill_chunk(chunk, kv_cache) per
chunk. The runtime is STATELESS w.r.t. the KV cache — the engine allocates it (via the adapter's
``allocate_kv_cache``) and passes it into every call that touches it. Only single-rank prefill is
wired (no pipeline / D2D). KV-chunk-table migration is supported via ``build_kv_chunk_table`` (a
multi-config table; see ``tt/runners/kv_chunk_table.py``).

Input convention (shared with the engine's H2D socket): ``make_chunk_input`` returns the chunk's
token IDs as an SP-sharded uint32 tensor — the SAME per-chip layout the request-mode socket delivers.
``prefill_chunk`` embeds that token tensor on device, so the standalone (trace) and request (socket)
paths feed identical tensors into the forward and share one code path. The embedding lives here (M3
embeds in ``prepare_inputs_prefill``, separate from the model forward), unlike DeepSeek which embeds
inside ``forward``.

The cache holds ``num_users * num_layers`` slots (user-major batch); each ``prefill_chunk`` call fills
``slot_id``'s layers for one chunk at the given absolute KV offset, running the cache-read attention
path against the already-written prefix.

Reference: models/demos/deepseek_v3_d_p/tt/tt_prefill_runtime.py (interface) and
tests/galaxy_prefill_kv_pcc.py (cache readback + RoPE swizzle convention).
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder, blockcyclic_positions


@dataclass
class TtPrefillRuntimeConfig:
    num_layers: int  # layers built/cached by this runtime (== model total for single-rank)
    max_seq_len: int  # per-user KV-cache length in tokens; must be a multiple of chunk_size
    mesh_shape: tuple = (8, 4)  # (SP rows, TP cols) on the Blackhole galaxy
    chunk_size: int = 5120  # tokens per prefill_chunk() call; one-shot sets this == max_seq_len
    num_users: int = 1  # independent cache slots (user-major batch)
    sp_axis: int = 0
    tp_axis: int = 1
    topology: ttnn.Topology = ttnn.Topology.Linear
    use_ep_moe: bool = True
    expert_weight_dtype: ttnn.DataType = ttnn.bfloat4_b
    cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    weight_cache_path: Optional[Path] = None
    # Pipeline-parallel rank slicing (single-rank defaults own the whole model). M3 only wires
    # single-rank prefill today — these exist for engine-contract parity (the engine reads
    # is_first_rank / is_last_rank / first_layer_idx to drive the chunk schedule) and are asserted
    # single-rank below.
    first_layer_idx: int = 0
    is_first_rank: bool = True
    is_last_rank: bool = True

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
            config.is_first_rank and config.is_last_rank
        ), "MiniMax-M3 prefill wires single-rank only (no pipeline); is_first_rank and is_last_rank must be True"
        assert (
            config.max_seq_len % config.chunk_size == 0
        ), f"max_seq_len ({config.max_seq_len}) must be a multiple of chunk_size ({config.chunk_size})"

        self.model_built = False
        self.compiled = False
        # Per-layer LayerAck callback, registered via set_layer_ack_channel() after compile.
        self._on_layer_complete = None

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
            f"max_seq_len={self.config.max_seq_len} chunk_size={self.config.chunk_size} "
            f"num_users={self.config.num_users} mesh_shape={self.config.mesh_shape}"
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
            ep_seq_len_per_chip=self.config.chunk_size // self.config.sp_factor,
            expert_weight_dtype=self.config.expert_weight_dtype,
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
            full = ttnn.to_torch(ttnn.get_device_tensors(dev_mat)[0])[:, :, :cache_seq, :]
            bc = block_cyclic_reorder(full, chunk_local, sp, seq_dim=2)
            return ttnn.from_torch(bc, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper)

        rs = self.model.rope_setup
        self.rope_indexed = [build(rs.cos_matrix_prefill), build(rs.sin_matrix_prefill)]

    def make_chunk_input(self, token_ids: list) -> ttnn.Tensor:
        """Build one chunk's device input for ``prefill_chunk``: the chunk's token IDs as an SP-sharded
        uint32 ROW_MAJOR DRAM tensor of per-chip shape ``(1, 1, chunk_size // sp)`` — row r holds the
        contiguous token slice ``[r*s_local : (r+1)*s_local]``, replicated across the TP cols. This is the
        SAME per-chip layout the request-mode H2D socket delivers, so both paths feed one code path;
        ``prefill_chunk`` embeds it on device. (M3 uses a contiguous — non-balanced — SP shard, matching
        ``prepare_inputs_prefill`` and the block-cyclic layout ``_build_indexed_rope`` assumes.)"""
        assert len(token_ids) == self.config.chunk_size, (
            f"chunk input must be exactly chunk_size={self.config.chunk_size} tokens (pad the tail), "
            f"got {len(token_ids)}"
        )
        sp = self.config.sp_factor
        s_local = self.config.chunk_size // sp
        tok = torch.tensor(token_ids, dtype=torch.int32).reshape(sp, 1, s_local)
        return ttnn.from_torch(
            tok,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=self.config.mesh_shape, dims=(self.config.sp_axis, None)
            ),
        )

    def _embed_tokens(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """Embed the SP-sharded token tensor into the bf16 hidden state each layer consumes. The
        embedding weight is replicated across the mesh, so each chip embeds its own seq shard (rows) with
        the full embedding dim (cols); this reproduces ``prepare_inputs_prefill``'s SP embedding starting
        from an already-on-device token tensor. bf16 (not bf8) keeps the residual stream's dynamic range."""
        x = ttnn.embedding(tokens, self.model.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        if len(x.shape) == 3:
            x = ttnn.unsqueeze_to_4D(x)
        return x

    def compile(self, kv_cache) -> None:
        """Warm up one zero-token chunk so the per-chunk loop hits no first-run cost (JIT-compiles all
        ops). The engine passes the cache it owns; the warm-up writes slot 0 and is harmless."""
        assert self.model_built
        chunk = self.config.chunk_size
        logger.info(f"TtPrefillRuntime.compile() — warming up one {chunk}-token chunk")
        t0 = time.perf_counter()
        tt_input = self.make_chunk_input([0] * chunk)
        self.prefill_chunk(tt_input, kv_cache, slot_id=0, actual_start=0, actual_end=chunk)
        ttnn.synchronize_device(self.mesh_device)
        warmup_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            f"[prefill timing] task_id=WARMUP num_tokens={chunk} runtime.prefill_chunk(chunk) = {warmup_ms:.2f} ms"
        )
        self.compiled = True

    def prefill_chunk(
        self,
        input_tensor: ttnn.Tensor,
        kv_cache,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        *,
        skip_lm_head: bool = True,
        get_last_token: int = -1,
    ):
        """Prefill ONE chunk into user ``slot_id``'s slice of the engine-owned ``kv_cache``. With
        ``skip_lm_head`` (the default) returns None — single-rank is headless, the populated cache is the
        output, read back via ``gather_layer``; otherwise returns the LM-head logits. Drives the
        cache-read attention path for actual_start > 0.

        [actual_start, actual_end) is the absolute KV-position range of this chunk's real tokens:
        actual_start is the cache write offset (the valid prefix already cached) and actual_end is past
        the last real token (the tail of the final chunk may be pad, so actual_end < actual_start +
        chunk_size). The chunk occupies physical positions [actual_start, actual_start + chunk_size);
        causality makes the pad tail inert. Call once per chunk, in order — a chunk's KV must be written
        before the next chunk reads it. If a LayerAck channel is registered, the model bumps it per layer.

        Args:
            input_tensor: this chunk's SP-sharded uint32 token tensor (make_chunk_input, or the H2D
                socket). Embedded on device here, then deallocated.
            kv_cache: the engine-owned MiniMaxKVCache (from the adapter's allocate_kv_cache); this
                chunk's KV is written into it. The same object is passed on every call.
            slot_id: cache user slot to fill, in [0, num_users).
            actual_start: absolute KV pos of the chunk's first token (the cache write offset).
            actual_end: absolute KV pos past the chunk's last real (non-pad) token.
        """
        assert self.model_built, "build the model before prefill_chunk()"
        assert 0 <= slot_id < self.config.num_users, f"slot_id {slot_id} out of range [0, {self.config.num_users})"
        assert (
            actual_start + self.config.chunk_size <= self.config.max_seq_len
        ), f"chunk at actual_start={actual_start} exceeds per-user cache {self.config.max_seq_len}"
        assert (
            actual_start < actual_end <= actual_start + self.config.chunk_size
        ), f"[actual_start={actual_start}, actual_end={actual_end}) not within one chunk of {self.config.chunk_size}"

        # Tokens are embedded on device here: make_chunk_input (standalone) and the H2D socket (request)
        # both deliver raw SP-sharded token ids, so both paths share one code path.
        x_embd = self._embed_tokens(input_tensor)
        ttnn.deallocate(input_tensor)

        # Whole-cache indexed rope built once (self.rope_indexed); the indexed op picks this chunk's rows
        # on-device from kv_actual_global (= actual_start, threaded via cached_len + indexed_rope=True). No
        # per-chunk host reshard, and the tensors are persistent (do NOT deallocate them here). The KV
        # cache is engine-owned and passed in. If a LayerAck channel is registered, the model bumps it
        # once per layer via on_layer_complete.
        out = self.model.prefill_forward(
            x_embd,
            rot_mats_global=self.rope_indexed,
            kv_cache=kv_cache,
            cached_len=actual_start,  # valid prefix already in the cache before this chunk
            user_id=slot_id,
            get_last_token=get_last_token,
            skip_lm_head=skip_lm_head,  # default: cache-fill only (skip final norm + lm_head)
            indexed_rope=True,
            on_layer_complete=self._on_layer_complete,
        )
        if skip_lm_head:
            if out is not None:
                out.deallocate(True)
            return None
        return out  # logits [1,1,chunk_local,vocab_shard], SP-sharded on seq / TP-sharded on vocab

    def set_layer_ack_channel(self, layer_ack_channel) -> None:
        """Register the per-layer LayerAck channel (engine-created + owned). ``prefill_chunk`` bumps it
        once per layer (``inject(1)``); the scheduler reads the delta. The ack carries no payload. Called
        by the engine in single-rank request mode."""
        assert self.compiled, "Call compile() before set_layer_ack_channel()"

        def on_layer_complete(layer_idx: int) -> None:
            layer_ack_channel.inject(1)

        self._on_layer_complete = on_layer_complete

    def gather_layer(self, kv_cache, slot_id: int, layer_idx: int, n_tokens: int):
        """Read one layer's device cache back to NATURAL token order (un-rotating the block-cyclic SP
        layout). Returns (k, v, index_k) torch tensors in DEVICE convention (K / index_k are Meta-RoPE
        swizzled over the rotary slice — the caller reconciles vs the HF golden). Shapes:
        k,v -> [1, num_kv_heads, n_tokens, head_dim]; index_k -> [1, 1, n_tokens, head_dim] (zeros on
        dense layers, which carry no index_k). Optional bring-up hook — never used in production
        serving."""
        sp = self.config.sp_factor
        cols = self.config.tp_factor  # K/V head c on col c; index_k replicated -> read col 0
        nkv = self.hf_config.num_key_value_heads
        slot = slot_id * self.config.num_layers + layer_idx
        # shard-row -> natural global position (inverse of the update_padded_kv_cache writer).
        p = blockcyclic_positions(sp, self.config.chunk_size, self.config.max_seq_len)

        def gather(cache_tensor, col):
            dts = ttnn.get_device_tensors(cache_tensor)
            dev = torch.cat([ttnn.to_torch(dts[r * cols + col])[slot, 0].float() for r in range(sp)], dim=0)
            nat = torch.empty_like(dev)
            nat[p] = dev
            return nat[:n_tokens]

        k = torch.stack([gather(kv_cache.k, c) for c in range(nkv)], dim=0).unsqueeze(0)
        v = torch.stack([gather(kv_cache.v, c) for c in range(nkv)], dim=0).unsqueeze(0)
        index_k = gather(kv_cache.index_k, 0).unsqueeze(0).unsqueeze(0)
        return k, v, index_k

    def build_kv_chunk_table(self, kv_cache, path: str) -> str:
        """Build + serialize M3's multi-config KV chunk address table (k_h0..N, v_h0..N, index_k) to
        ``path`` and return it. The engine then PUBLISHES it to the migration worker (this issues no
        comms). Called by the runner when PREFILL_ENABLE_MIGRATION=1 / PREFILL_MOCK_MIGRATION=1.
        Single-rank only (asserted in __init__)."""
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
        )

    def read_slot_kv(self, kv_cache, slot: int):
        """Read one slot's KV cache from device to host: ``[k, v, index_k]``, one host tensor per cache
        tensor, each ``[num_layers, heads(or 1), seq_cache, head_dim]`` (index_k collapsed to one TP
        replica), in the raw on-device (block-cyclic) layout — not un-rotated to natural token order.
        DRAM_MEMORY_CONFIG on the slice is REQUIRED — the cache is ND-sharded ROUND_ROBIN_1D, and slicing
        into another ND-shard miscomputes the DRAM core on host read-back."""
        mesh_device = self.mesh_device
        num_layers = self.config.num_layers

        def _block(tensor, collapse_tp: bool):
            s = list(tensor.shape)
            sl = ttnn.slice(
                tensor,
                [slot * num_layers, 0, 0, 0],
                [(slot + 1) * num_layers, s[1], s[2], s[3]],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            block = ttnn.to_torch(
                sl, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
            ).float()  # [num_layers, nkv (or cols), seq_cache, head_dim]
            ttnn.deallocate(sl)
            return block[:, :1] if collapse_tp else block

        return [_block(kv_cache.k, False), _block(kv_cache.v, False), _block(kv_cache.index_k, True)]

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
