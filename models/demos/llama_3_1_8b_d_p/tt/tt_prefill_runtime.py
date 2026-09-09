# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The chunked-prefill runtime: `compile`, `make_chunk_input`, `prefill_chunk`.

Borrowed structurally from `minimax_m3/tt/tt_prefill_runtime.py`, measured at the same mesh and the
same 5120-token chunk. The cache-read op it drives is already implemented and tested in D3
(`tt/attention/dense_sp.py`); what is new here is the bookkeeping around it.

## The chunk contract, and why it is asserted rather than documented

`[actual_start, actual_end)` is the absolute KV-position range of this chunk's REAL tokens:

* `actual_start` is the cache write offset — the valid prefix already cached.
* `actual_end` is one past the last real token. The tail of a final chunk may be padding, so
  `actual_end` can be less than `actual_start + chunk_size`.

The chunk always OCCUPIES physical positions `[actual_start, actual_start + chunk_size)`; only the
real-token count differs. Every one of these is checked on entry, because an out-of-contract range
does not raise anywhere downstream — it writes real values to wrong cache addresses, and the damage
first appears as a KV PCC drop many layers later.

## Rope

`indexed_rope=True` threads `actual_start` into `rotary_embedding_indexed`, which derives each
chip's block-cyclic start row ON DEVICE from `kv_actual_global` plus the device's SP coordinate —
the same arithmetic the KV-cache writer uses. The tables it consumes are the WHOLE-cache
block-cyclic SP-sharded cos/sin, built once. The alternative (a host-side per-chunk slice, which
`tests/galaxy_prefill_kv_pcc.py` uses) is equivalent and simpler to verify, but re-shards on every
chunk.
"""

from dataclasses import dataclass

import torch
from loguru import logger

import ttnn

from .attention.kv_cache import cache_capacity


@dataclass
class TtPrefillRuntimeConfig:
    """The resolved knobs one prefill runtime is built with."""

    max_seq_len: int
    chunk_size: int
    num_layers: int
    num_users: int = 1
    use_trace: bool = False
    """Read UNCONDITIONALLY by the common/prefill engine, so it must exist even when False. It is a
    ttnn command-buffer capture for replay — a perf mechanism, unrelated to the golden trace."""

    def __post_init__(self):
        sp_period = 32  # times sp, checked against the live mesh in the runtime
        assert self.chunk_size % sp_period == 0, (
            f"chunk_size {self.chunk_size} must be a multiple of {sp_period}; it is the block-cyclic "
            "addressing period of the KV table, and a misaligned value corrupts addresses silently"
        )
        assert self.chunk_size <= self.max_seq_len

    @property
    def capacity(self) -> int:
        """Cache capacity: `max_seq_len` rounded UP to a whole chunk."""
        return cache_capacity(self.max_seq_len, self.chunk_size)


class TtPrefillRuntime:
    """Drives a :class:`~.model.Model` over a sequence, one chunk at a time."""

    def __init__(self, model, mesh_device, mesh_config, config: TtPrefillRuntimeConfig):
        self.model = model
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.config = config
        sp = mesh_config.sp
        assert config.chunk_size % (32 * sp) == 0, (
            f"chunk_size {config.chunk_size} must be a multiple of 32*sp ({32 * sp})"
        )
        self.head_dim = model.hf_config.hidden_size // model.hf_config.num_attention_heads
        # Whole-cache Meta-order cos/sin, built once. Indexed rope derives each chunk's start row
        # on device from these, so there is no per-chunk host reshard.
        self._rope_mats = None

    # -- inputs ---------------------------------------------------------------

    def make_chunk_input(self, token_ids: list) -> ttnn.Tensor:
        """Token ids -> a `[1, 1, chunk_size]` uint32 tensor, SP-sharded across the mesh rows.

        Short inputs are padded to `chunk_size`; the padding is excluded from the cache write by
        `actual_end`, not by masking here.
        """
        chunk = self.config.chunk_size
        assert len(token_ids) <= chunk, f"{len(token_ids)} tokens exceeds chunk_size {chunk}"
        padded = list(token_ids) + [0] * (chunk - len(token_ids))
        dims = [None, None]
        dims[self.mesh_config.sp_axis] = 2
        return ttnn.from_torch(
            torch.tensor(padded, dtype=torch.int32).reshape(1, 1, chunk),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=dims
            ),
        )

    def rope_mats(self):
        """The whole-cache, block-cyclic, SP-sharded Meta-order cos/sin. Built once."""
        if self._rope_mats is not None:
            return self._rope_mats
        capacity = self.config.capacity
        hd = self.head_dim
        cos = ttnn.to_torch(ttnn.get_device_tensors(self.model.rope_setup.cos_matrix)[0]).reshape(-1, hd)
        sin = ttnn.to_torch(ttnn.get_device_tensors(self.model.rope_setup.sin_matrix)[0]).reshape(-1, hd)
        assert cos.shape[0] >= capacity, (
            f"rope tables cover {cos.shape[0]} positions but the cache holds {capacity}; build the "
            "RotarySetup at the cache capacity, not at max_seq_len"
        )
        dims = [None, None]
        dims[self.mesh_config.sp_axis] = 2

        def to_mesh(t):
            return ttnn.from_torch(
                t[:capacity].reshape(1, 1, capacity, hd),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=dims
                ),
            )

        self._rope_mats = (to_mesh(cos), to_mesh(sin))
        return self._rope_mats

    # -- execution ------------------------------------------------------------

    def compile(self, kv_cache) -> None:
        """Warm the program cache with one dummy chunk at each distinct code path.

        Two are needed, not one: chunk 0 may take the no-cache ring path while later chunks take the
        cache-read path, and they are different programs.
        """
        chunk = self.config.chunk_size
        dummy = self.make_chunk_input([0] * chunk)
        for start in (0, chunk) if self.config.capacity > chunk else (0,):
            if start + chunk > self.config.capacity:
                break
            logger.info(f"compiling prefill chunk at actual_start={start}")
            self.prefill_chunk(dummy, kv_cache, slot_id=0, actual_start=start, actual_end=start + chunk)

    def prefill_chunk(
        self,
        chunk_input,
        kv_cache,
        *,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        metadata_msg=None,
    ):
        """Run one chunk, writing its K/V into `kv_cache` and attending the prefix before it.

        Args:
            chunk_input: from :meth:`make_chunk_input`.
            slot_id: the user slot. Cache addressing is `slot_id * num_layers + layer_idx`.
            actual_start: absolute KV position of the chunk's first token — the cache write offset.
            actual_end: absolute KV position one past the chunk's last REAL (non-pad) token.
            metadata_msg: accepted for the engine's calling convention; unused here.

        Returns the model output for this chunk (the hidden state; logits are not needed to fill a
        KV cache and are skipped).
        """
        chunk = self.config.chunk_size
        capacity = self.config.capacity

        # The contract. Each of these fails loudly here or silently corrupts addresses downstream.
        assert actual_start >= 0, f"actual_start={actual_start} must be non-negative"
        assert actual_start % chunk == 0, (
            f"actual_start={actual_start} is not chunk-aligned (chunk_size={chunk}); the cache is "
            "block-cyclic with that period, so a misaligned offset writes to wrong addresses"
        )
        assert actual_start < actual_end, f"empty chunk: [{actual_start}, {actual_end})"
        assert actual_end <= actual_start + chunk, (
            f"[{actual_start}, {actual_end}) spans more than one chunk of {chunk}; one call writes at most one"
        )
        assert actual_start + chunk <= capacity, (
            f"chunk at actual_start={actual_start} would run past the cache capacity {capacity}"
        )
        assert slot_id < self.config.num_users, f"slot_id={slot_id} out of range for {self.config.num_users} users"
        assert chunk_input.shape[-1] * self.mesh_config.sp == chunk or chunk_input.shape[-1] == chunk // self.mesh_config.sp, (
            f"chunk input carries {chunk_input.shape[-1]} tokens per device, expected {chunk // self.mesh_config.sp}"
        )

        return self.model.forward(
            chunk_input,
            rope_mats=self.rope_mats(),
            kv_cache=kv_cache,
            slot_idx=slot_id,
            cached_len=actual_start,
            logical_n=actual_end,
            indexed_rope=True,
            return_logits=False,
        )
