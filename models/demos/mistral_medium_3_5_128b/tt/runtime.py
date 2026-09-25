# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 prefill runtime on the 8x4 Blackhole Galaxy (SP=8 x TP=4).

Contract from ``minimax_m3/tt/tt_prefill_runtime.py`` (the ``common/prefill`` engine's runtime shape):
build model -> ``compile(kv_cache)`` -> ``prefill_chunk(make_chunk_input(ids), kv_cache, slot, start, end)``
per chunk, in order. The runtime is stateless w.r.t. the KV cache: the caller allocates it
(``allocate_kv_cache``) and passes it into every call.

``chunk_size`` is the block-cyclic period of the KV cache and the width of every chunk. One-shot prefill
is the ``chunk_size == max_seq_len`` case (a single chunk over the whole sequence). A chunk occupies
positions ``[actual_start, actual_start + chunk_size)``; ``actual_end`` marks its last real token (the
tail may be padding, inert under causality). Out-of-contract ranges fail loudly here rather than as a
scrambled cache read deep in attention.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from loguru import logger

import ttnn

from .ccl import CCLManager, MeshConfig
from .fabric import ccl_topology
from .kv_cache import allocate_kv_cache, read_slot_kv
from .model import Model
from .rope import RopeSetup


@dataclass
class PrefillRuntimeConfig:
    num_layers: int
    max_seq_len: int  # per-user KV capacity in tokens; a multiple of chunk_size
    chunk_size: int  # tokens per prefill_chunk call (== max_seq_len for one-shot)
    dtypes: dict = field(default_factory=dict)  # ttnn dtypes resolved from the spec (config.resolve_dataformats)
    cache_dtype: object = ttnn.bfloat8_b  # spec dataformats.kv_cache
    num_users: int = 1
    first_layer_idx: int = 0
    with_lm_head: bool = False
    embed_shard_vocab: bool = False
    weight_cache_path: Optional[str] = None
    pad_token_id: int = 11


class PrefillRuntime:
    def __init__(self, mesh_device, cfg, weights, config: PrefillRuntimeConfig, on_layer_built=None):
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.config = config
        self.mesh_config = MeshConfig(mesh_device.shape)
        sp = self.mesh_config.sp
        assert (
            config.max_seq_len % config.chunk_size == 0
        ), f"max_seq_len ({config.max_seq_len}) must be a multiple of chunk_size ({config.chunk_size})"
        assert config.chunk_size % (ttnn.TILE_SIZE * sp) == 0, (
            f"chunk_size ({config.chunk_size}) must be a multiple of 32 * sp ({32 * sp}): it is the KV "
            f"table's block-cyclic period, a misaligned value corrupts addresses silently"
        )
        self.ccl_manager = CCLManager(mesh_device, topology=ccl_topology())
        t0 = time.perf_counter()
        self.model = Model(
            mesh_device,
            self.mesh_config,
            self.ccl_manager,
            cfg,
            weights,
            dtypes=config.dtypes,
            num_layers=config.num_layers,
            first_layer_idx=config.first_layer_idx,
            with_lm_head=config.with_lm_head,
            embed_shard_vocab=config.embed_shard_vocab,
            tensor_cache_path=config.weight_cache_path,
            on_layer_built=on_layer_built,
        )
        self.build_seconds = time.perf_counter() - t0
        self.rope = RopeSetup(
            mesh_device, self.mesh_config, cfg, max_seq_len=config.max_seq_len, chunk_size=config.chunk_size
        )
        self.compiled = False
        logger.info(
            f"PrefillRuntime: {config.num_layers} layers from {config.first_layer_idx}, max_seq_len "
            f"{config.max_seq_len}, chunk {config.chunk_size}, mesh {tuple(mesh_device.shape)}, built in "
            f"{self.build_seconds:.1f}s"
        )

    def allocate_kv_cache(self):
        return allocate_kv_cache(
            self.mesh_device,
            self.mesh_config,
            num_layers=self.config.num_layers,
            max_seq_len=self.config.max_seq_len,
            num_users=self.config.num_users,
            num_local_kv_heads=self.cfg.num_key_value_heads // self.mesh_config.tp,
            head_dim=self.cfg.head_dim,
            cache_dtype=self.config.cache_dtype,
        )

    def make_chunk_input(self, token_ids):
        """One chunk's token ids (exactly ``chunk_size``; pad the tail) -> the SP-sharded uint32 device
        tensor ``prefill_chunk`` embeds (row r holds the contiguous slice of the chunk)."""
        assert (
            len(token_ids) == self.config.chunk_size
        ), f"chunk input must be exactly chunk_size={self.config.chunk_size} tokens (pad the tail), got {len(token_ids)}"
        return Model.token_tensor(token_ids, self.mesh_device, self.mesh_config)

    def _check_chunk(self, slot_id, actual_start, actual_end):
        c = self.config
        assert 0 <= slot_id < c.num_users, f"slot_id {slot_id} out of range [0, {c.num_users})"
        assert (
            actual_start % c.chunk_size == 0
        ), f"actual_start={actual_start} must be a multiple of chunk_size={c.chunk_size} (block-cyclic KV period)"
        assert (
            actual_start + c.chunk_size <= c.max_seq_len
        ), f"chunk at actual_start={actual_start} exceeds the per-user KV capacity {c.max_seq_len}"
        assert (
            actual_start < actual_end <= actual_start + c.chunk_size
        ), f"[actual_start={actual_start}, actual_end={actual_end}) is not a non-empty range within one chunk of {c.chunk_size}"

    def prefill_chunk(
        self,
        input_tensor,
        kv_cache,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        *,
        skip_lm_head=True,
        on_layer_complete=None,
    ):
        """Prefill ONE chunk into user ``slot_id``'s KV slots at absolute offset ``actual_start``. Headless
        by default (returns None; the populated cache is the output); else returns the chunk's logits."""
        self._check_chunk(slot_id, actual_start, actual_end)
        x = self.model.embed(input_tensor)
        ttnn.deallocate(input_tensor)
        x = self.model.forward_layers(
            x,
            self.rope,
            kv_cache=kv_cache,
            user_id=slot_id,
            cached_len=actual_start,
            on_layer_complete=on_layer_complete,
        )
        if skip_lm_head:
            x.deallocate(True)
            return None
        normed, logits = self.model.head(x)
        x.deallocate(True)
        normed.deallocate(True)
        return logits

    def prefill(self, token_ids, kv_cache, slot_id: int = 0):
        """Chunk loop over a whole prompt (tail padded with pad_token_id). Returns the number of chunks."""
        c = self.config
        n = len(token_ids)
        n_chunks = -(-n // c.chunk_size)
        padded = list(token_ids) + [c.pad_token_id] * (n_chunks * c.chunk_size - n)
        for i in range(n_chunks):
            start = i * c.chunk_size
            self.prefill_chunk(
                self.make_chunk_input(padded[start : start + c.chunk_size]),
                kv_cache,
                slot_id,
                actual_start=start,
                actual_end=min(start + c.chunk_size, n),
            )
        ttnn.synchronize_device(self.mesh_device)
        return n_chunks

    def compile(self, kv_cache):
        """Warm every KV-length bucket the served loop can reach (each chunk start), writing slot 0,
        which a real prefill then overwrites."""
        t0 = time.perf_counter()
        chunk = self.config.chunk_size
        for start in range(0, self.config.max_seq_len - chunk + 1, chunk):
            self.prefill_chunk(
                self.make_chunk_input([self.config.pad_token_id] * chunk), kv_cache, 0, start, start + chunk
            )
        ttnn.synchronize_device(self.mesh_device)
        self.compiled = True
        logger.info(f"PrefillRuntime.compile: {time.perf_counter() - t0:.1f}s")

    def read_slot_kv(self, kv_cache, slot_id: int = 0):
        """Host ``(k, v)`` for one user: ``[num_layers, num_kv_heads, max_seq_len, D]`` in block-cyclic order."""
        return read_slot_kv(self.mesh_device, kv_cache, slot_id)

    @staticmethod
    def pad_to(n: int, multiple: int) -> int:
        return -(-n // multiple) * multiple

    def last_token_logits(self, logits, n_tokens: int):
        """Host logits ``[vocab]`` of absolute position ``n_tokens - 1`` from a chunk's per-chip logits."""
        c = self.config
        pos = (n_tokens - 1) % c.chunk_size
        s_local = c.chunk_size // self.mesh_config.sp
        row, off = divmod(pos, s_local)
        tp = self.mesh_config.tp
        per = ttnn.get_device_tensors(logits)
        return torch.cat([ttnn.to_torch(per[row * tp + col])[0, 0, off].float() for col in range(tp)], dim=-1)
