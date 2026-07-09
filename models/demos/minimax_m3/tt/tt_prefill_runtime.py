# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 single-rank prefill runtime.

Mirrors the DeepSeek ``TtPrefillRuntime`` lifecycle: build model -> allocate KV cache ->
compile -> prefill(chunk). The runtime is mode-agnostic — it only knows ``chunk_size`` and
``max_seq_len`` and fills the external ``MiniMaxKVCache`` one chunk at a time. The caller
(see tests/galaxy_prefill_kv_pcc.py) decides one-shot vs chunked, drives the chunk loop, and
does the golden KV-cache PCC comparison / throughput measurement. No env vars are read here.

The cache holds ``num_users * num_layers`` slots (user-major batch); each ``prefill()`` call
fills ``slot_id``'s layers for one chunk at the given absolute KV offset, running the cache-read
attention path against the already-written prefix.

Reference: models/demos/deepseek_v3_d_p/tt/tt_prefill_runtime.py (interface) and
tests/unit/test_kv_cache_write_vs_ref.py (cache readback + RoPE swizzle convention).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder, blockcyclic_positions

from .attention import allocate_kv_caches


@dataclass
class TtPrefillRuntimeConfig:
    num_layers: int  # layers built/cached by this runtime (== model total for single-rank)
    max_seq_len: int  # per-user KV-cache length in tokens; must be a multiple of chunk_size
    mesh_shape: tuple = (8, 4)  # (SP rows, TP cols) on the Blackhole galaxy
    chunk_size: int = 5120  # tokens per prefill() call; one-shot sets this == max_seq_len
    num_users: int = 1  # independent cache slots (user-major batch)
    sp_axis: int = 0
    tp_axis: int = 1
    topology: ttnn.Topology = ttnn.Topology.Linear
    use_ep_moe: bool = True
    expert_weight_dtype: ttnn.DataType = ttnn.bfloat4_b
    cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    weight_cache_path: Optional[Path] = None

    @property
    def sp_factor(self) -> int:
        return self.mesh_shape[self.sp_axis]

    @property
    def tp_factor(self) -> int:
        return self.mesh_shape[self.tp_axis]


class TtPrefillRuntime:
    """Single-rank prefill execution lifecycle: build model -> allocate KV cache -> compile ->
    prefill(chunk). Owns the SP=8 × TP=4 ``Model`` and the external ``MiniMaxKVCache``."""

    def __init__(self, mesh_device, hf_config, state_dict: dict, config: TtPrefillRuntimeConfig):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.config = config

        assert (
            config.max_seq_len % config.chunk_size == 0
        ), f"max_seq_len ({config.max_seq_len}) must be a multiple of chunk_size ({config.chunk_size})"

        self.model_built = False
        self.kv_cache_allocated = False
        self.compiled = False

        self._build_model(state_dict)
        self._allocate_kv_cache()
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

    def _allocate_kv_cache(self) -> None:
        # ONE cache holding num_users * num_layers slots (user-major); each (user, layer) slot is filled
        # per chunk by update_padded_kv_cache. K/V heads shard on the TP cols, index_k is replicated.
        self.kv_cache = allocate_kv_caches(
            self.mesh_device,
            num_layers=self.config.num_layers,
            max_seq_len=self.config.max_seq_len,
            sp_axis=self.config.sp_axis,
            num_users=self.config.num_users,
            head_dim=self.hf_config.head_dim,
            cache_dtype=self.config.cache_dtype,
        )
        self.kv_cache_allocated = True

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
        """Embed + SP-shard one chunk's token ids -> the model input tensor (consumed by prefill())."""
        assert len(token_ids) == self.config.chunk_size, (
            f"chunk input must be exactly chunk_size={self.config.chunk_size} tokens (pad the tail), "
            f"got {len(token_ids)}"
        )
        chunk_tok = torch.tensor(token_ids, dtype=torch.int32).reshape(1, len(token_ids))
        x_embd, _, _ = self.model.prepare_inputs_prefill(chunk_tok)  # we override its RoPE in prefill()
        return x_embd

    def compile(self) -> None:
        """Warm up the kernels by running one zero-token chunk through prefill (JIT-compiles all ops)."""
        assert self.model_built and self.kv_cache_allocated
        chunk = self.config.chunk_size
        logger.info(f"TtPrefillRuntime.compile() — warming up one {chunk}-token chunk")
        tt_input = self.make_chunk_input([0] * chunk)
        self.prefill(tt_input, slot_id=0, actual_start=0, actual_end=chunk)
        ttnn.synchronize_device(self.mesh_device)
        self.compiled = True

    def prefill(
        self,
        input_tensor: ttnn.Tensor,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        *,
        skip_lm_head: bool = True,
        get_last_token: int = -1,
    ):
        """Prefill ONE chunk into user ``slot_id``'s KV cache. Returns None — the populated cache is the
        output (read back via gather_layer). Drives the cache-read attention path for actual_start > 0.

        [actual_start, actual_end) is the absolute KV-position range of this chunk's real tokens:
        actual_start is the cache write offset (the valid prefix already cached) and actual_end is past
        the last real token (the tail of the final chunk may be pad, so actual_end < actual_start +
        chunk_size). The chunk occupies physical positions [actual_start, actual_start + chunk_size);
        causality makes the pad tail inert. Call once per chunk, in order — a chunk's KV must be written
        before the next chunk reads it.

        Args:
            input_tensor: this chunk's embedded, SP-sharded model input (make_chunk_input). Deallocated here.
            slot_id: cache user slot to fill, in [0, num_users).
            actual_start: absolute KV pos of the chunk's first token (the cache write offset).
            actual_end: absolute KV pos past the chunk's last real (non-pad) token.
        """
        assert self.model_built and self.kv_cache_allocated, "build the model and KV cache before prefill()"
        assert 0 <= slot_id < self.config.num_users, f"slot_id {slot_id} out of range [0, {self.config.num_users})"
        assert (
            actual_start + self.config.chunk_size <= self.config.max_seq_len
        ), f"chunk at actual_start={actual_start} exceeds per-user cache {self.config.max_seq_len}"
        assert (
            actual_start < actual_end <= actual_start + self.config.chunk_size
        ), f"[actual_start={actual_start}, actual_end={actual_end}) not within one chunk of {self.config.chunk_size}"

        # Whole-cache indexed rope built once (self.rope_indexed); the indexed op picks this chunk's rows
        # on-device from kv_actual_global (= actual_start, threaded via cached_len + indexed_rope=True). No
        # per-chunk host reshard, and the tensors are persistent (do NOT deallocate them here).
        out = self.model.prefill_forward(
            input_tensor,
            rot_mats_global=self.rope_indexed,
            kv_cache=self.kv_cache,
            cached_len=actual_start,  # valid prefix already in the cache before this chunk
            user_id=slot_id,
            get_last_token=get_last_token,
            skip_lm_head=skip_lm_head,  # default: cache-fill only (skip final norm + lm_head)
            indexed_rope=True,
        )
        ttnn.deallocate(input_tensor)
        if skip_lm_head:
            out.deallocate(True)
            return None
        return out  # logits [1,1,chunk_local,vocab_shard], SP-sharded on seq / TP-sharded on vocab

    def gather_layer(self, slot_id: int, layer_idx: int, n_tokens: int):
        """Read one layer's device cache back to NATURAL token order (un-rotating the block-cyclic SP
        layout). Returns (k, v, index_k) torch tensors in DEVICE convention (K / index_k are Meta-RoPE
        swizzled over the rotary slice — the caller reconciles vs the HF golden). Shapes:
        k,v -> [1, num_kv_heads, n_tokens, head_dim]; index_k -> [1, 1, n_tokens, head_dim] (zeros on
        dense layers, which carry no index_k)."""
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

        k = torch.stack([gather(self.kv_cache.k, c) for c in range(nkv)], dim=0).unsqueeze(0)
        v = torch.stack([gather(self.kv_cache.v, c) for c in range(nkv)], dim=0).unsqueeze(0)
        index_k = gather(self.kv_cache.index_k, 0).unsqueeze(0).unsqueeze(0)
        return k, v, index_k
