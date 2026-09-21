# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chunked-prefill runtime: ``compile`` / ``make_chunk_input`` / ``prefill_chunk``.

The engine hands this a chunk of token ids and an absolute KV range and expects the caches to be
correct afterwards. Everything model-specific about *how* that happens lives in the modules; what
lives here is the chunk contract and the assertions that make a violation of it loud.

**A pad tail is not inert in this model.** For a pure attention model the convention is that the
last chunk may carry ``actual_end < actual_start + chunk_size`` and causality makes the padding
harmless. That is false for the 48 Gated DeltaNet layers: their scan is recurrent, so pad tokens
after the real tail are folded into the carried recurrent state and every later chunk inherits
the damage. So a short chunk is accepted **only as the final one**, and a call after it is a hard
error rather than a slow corruption.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from loguru import logger

import ttnn

from ..config import MeshConfig
from ..reference.config import Qwen35TextConfig
from .caches import PrefillCaches
from .model import Qwen35Model


@dataclass(frozen=True)
class PrefillRuntimeConfig:
    chunk_size: int
    max_seq_len: int
    num_users: int = 1

    def validate(self, mesh_config: MeshConfig) -> None:
        align = ttnn.TILE_SIZE * mesh_config.sp
        assert self.chunk_size % align == 0, (
            f"chunk_size {self.chunk_size} must be a multiple of TILE_SIZE*sp ({align}): it is the "
            f"block-cyclic addressing period of the KV table, and a misaligned value corrupts "
            f"addresses silently rather than failing"
        )
        assert self.max_seq_len % align == 0, f"max_seq_len {self.max_seq_len} % {align} != 0"
        assert (
            self.max_seq_len % self.chunk_size == 0
        ), f"max_seq_len {self.max_seq_len} must be a whole number of {self.chunk_size}-token chunks"


class TtPrefillRuntime:
    def __init__(
        self,
        mesh_device,
        model: Qwen35Model,
        cfg: Qwen35TextConfig,
        *,
        mesh_config: MeshConfig,
        config: PrefillRuntimeConfig,
    ) -> None:
        config.validate(mesh_config)
        self.mesh_device = mesh_device
        self.model = model
        self.cfg = cfg
        self.mesh_config = mesh_config
        self.config = config
        self.compiled = False
        self._final_chunk_start: Optional[int] = None

    # --- input -------------------------------------------------------------------------
    def make_chunk_input(self, token_ids: list[int] | torch.Tensor) -> torch.Tensor:
        """One chunk's token ids, padded to exactly ``chunk_size``.

        Kept on the host: the SP shard and the device upload happen inside
        ``Qwen35Model.shard_tokens``, so the token layout and the RoPE table's SP shard are
        derived in one place and cannot drift apart.
        """
        ids = torch.as_tensor(token_ids, dtype=torch.int64).reshape(-1)
        assert (
            ids.numel() <= self.config.chunk_size
        ), f"chunk input is {ids.numel()} tokens, longer than chunk_size={self.config.chunk_size}"
        if ids.numel() < self.config.chunk_size:
            ids = torch.nn.functional.pad(ids, (0, self.config.chunk_size - ids.numel()))
        return ids.reshape(1, -1)

    # --- warm-up -----------------------------------------------------------------------
    def compile(self, caches: PrefillCaches) -> None:
        """Warm every KV-length bucket the served loop can reach, so no served chunk pays a
        first-run JIT. Each warm-up writes user slot 0, which the real run overwrites.

        The GDN states are reset afterwards: a warm-up leaves a recurrent state behind, and a real
        run starting from it would be wrong in a way no shape check catches.
        """
        chunk = self.config.chunk_size
        starts = list(range(0, self.config.max_seq_len - chunk + 1, chunk))
        logger.info(f"TtPrefillRuntime.compile() — warming {len(starts)} KV-length buckets of {chunk} tokens")
        t0 = time.perf_counter()
        warm = self.make_chunk_input([0] * chunk)
        for start in starts:
            self.prefill_chunk(warm, caches, user_id=0, actual_start=start, actual_end=start + chunk)
            self._final_chunk_start = None
        ttnn.synchronize_device(self.mesh_device)
        self.reset(caches)
        logger.info(f"TtPrefillRuntime.compile() done in {(time.perf_counter() - t0) * 1000:.0f} ms")
        self.compiled = True

    def reset(self, caches: PrefillCaches) -> None:
        """Zero the Gated DeltaNet states and forget the chunk sequence — call between prompts.

        The KV cache needs no reset: ``actual_start`` decides where a write lands and causality
        masks anything past the valid prefix. The GDN recurrent state has no such masking, so a
        stale one silently seeds the next prompt.
        """
        from .caches import allocate_gdn_state

        for layer_idx, state in caches.gdn.items():
            state.conv_state.deallocate(True)
            state.recurrent.deallocate(True)
            fresh = allocate_gdn_state(self.mesh_device, self.cfg, mesh_config=self.mesh_config)
            state.conv_state, state.recurrent, state.seeded = fresh.conv_state, fresh.recurrent, False
        self._final_chunk_start = None

    # --- the chunk ---------------------------------------------------------------------
    def prefill_chunk(
        self,
        token_ids: torch.Tensor,
        caches: PrefillCaches,
        *,
        user_id: int = 0,
        actual_start: int,
        actual_end: int,
        skip_lm_head: bool = True,
        on_layer_complete: Optional[Callable[[int], None]] = None,
    ) -> ttnn.Tensor:
        """Prefill ONE chunk into user ``user_id``'s slice of the caches.

        ``[actual_start, actual_end)`` is the absolute KV-position range of this chunk's **real**
        tokens: ``actual_start`` is the cache write offset (the valid prefix already cached) and
        ``actual_end`` is one past the last real token. The chunk physically occupies
        ``[actual_start, actual_start + chunk_size)``.

        Call once per chunk, in order: a chunk's KV must be written before the next chunk reads it,
        and each Gated DeltaNet layer's state must be advanced exactly once per chunk.
        """
        chunk = self.config.chunk_size
        assert 0 <= user_id < self.config.num_users, f"user_id {user_id} out of [0, {self.config.num_users})"
        assert actual_start % chunk == 0, (
            f"actual_start={actual_start} is not a multiple of chunk_size={chunk}; the KV table is "
            f"block-cyclic with that period, so an off-period write lands at the wrong addresses"
        )
        assert (
            actual_start + chunk <= self.config.max_seq_len
        ), f"chunk at actual_start={actual_start} runs past the per-user cache {self.config.max_seq_len}"
        assert (
            actual_start < actual_end <= actual_start + chunk
        ), f"[actual_start={actual_start}, actual_end={actual_end}) is not within one {chunk}-token chunk"
        assert self._final_chunk_start is None, (
            f"the chunk starting at {self._final_chunk_start} was short (padded), so it was the final "
            f"one; a further chunk would continue from a Gated DeltaNet state that absorbed the pad "
            f"tokens. Call reset() and start the prompt again."
        )
        if actual_end < actual_start + chunk:
            # Legal, but only once: see the module docstring on why a pad tail is not inert here.
            self._final_chunk_start = actual_start

        ids = self.make_chunk_input(token_ids)
        return self.model.prefill_chunk(
            ids,
            start_pos=actual_start,
            caches=caches,
            user_id=user_id,
            skip_lm_head=skip_lm_head,
            on_layer_complete=on_layer_complete,
        )

    def prefill_sequence(
        self,
        token_ids: torch.Tensor,
        caches: PrefillCaches,
        *,
        user_id: int = 0,
        skip_lm_head: bool = True,
        on_layer_complete: Optional[Callable[[int], None]] = None,
    ) -> list[ttnn.Tensor]:
        """Drive a whole prompt through as many chunks as it needs. Returns each chunk's output."""
        ids = torch.as_tensor(token_ids, dtype=torch.int64).reshape(-1)
        total = ids.numel()
        assert total <= self.config.max_seq_len, f"prompt of {total} exceeds max_seq_len"
        chunk = self.config.chunk_size
        outputs = []
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            outputs.append(
                self.prefill_chunk(
                    ids[start:end],
                    caches,
                    user_id=user_id,
                    actual_start=start,
                    actual_end=end,
                    skip_lm_head=skip_lm_head,
                    on_layer_complete=on_layer_complete,
                )
            )
        return outputs
