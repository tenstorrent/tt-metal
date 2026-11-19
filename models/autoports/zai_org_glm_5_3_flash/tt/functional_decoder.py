# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional TTNN decoder layer for zai-org/GLM-5.3-Flash (`glm5_next`), single 1x1 Blackhole mesh.

One class covers the three layer kinds of the checkpoint:
    dense_kda   layers 0-2       KDA attention + dense SwiGLU (12288)
    sparse_kda  31 layers        KDA attention + MoE (288 routed top-8 + shared)
    sparse_dsa  11 layers        MLA/DSA attention + MoE

Contract
--------
Residual stream (mHC, hc_mult = 4): `[1, 1, T, 16384]` float32, stream i in columns [4096 i, 4096 (i + 1)); T = tokens
of one user (prefill) or users (decode).

Construction: `from_weight_cache(hf_config=, layer_idx=, mesh_device=, cache_dir=, manifest_path=)` (pre-packed
bfp4/bfp8/bf16 tile cache) or `from_state_dict(state_dict, hf_config=, layer_idx=, mesh_device=)` (one layer's host
state dict in the reference/HF post-conversion names). Both apply the same dtype map at construction time.

State: `allocate_cache(max_batch=, max_seq_len=)` -> `DecoderCache` with `max_batch` user slots. KDA layers: per slot a
recurrent state [1, 64, 128, 128] fp32 + the last 3 pre-conv q|k|v rows [1, 3, 24576] bf16 (persistent buffers, only
updated with in-place copies). MLA layers: paged latent cache
[slots * blocks_per_seq, 1, 64, 512] bf16; `make_page_table(cache, permutation=None)` binds the caller's page table
[slots, blocks_per_seq] (host copy kept for the prefill chunk tables).

Prefill (one user, chunked):
    begin_prefill_chunk(cache, user_slot=, chunk_start=, n_valid=)   host boundary: writes the per-chunk device
        scalars/tables (KDA valid end; MLA chunk start + chunk/user page tables); chunk_start == 0 starts a fresh
        request (the slot's KDA state is zeroed on device), chunk_start > 0 continues from the cached prefix
        (MLA: multiple of the 64-token page).
    prefill_chunk(x, cache=, user_slot=) -> y                       device-only pass on one physical chunk
        x [1, 1, P, 16384] fp32, P = round_up(n_valid, align) <= 2048 (align: KDA 32, MLA 64); rows >= n_valid are
        padding and are masked: KDA
        padded rows are identity recurrence steps excluded from the conv history, MLA padded latents land in the user's
        own not-yet-valid page slots (the cache reserves max_seq_len + 63 tokens) and are invisible to valid queries,
        MoE padded rows only cost compute. Returns all P rows. Every program of the pass depends only on P;
        `prepare_prefill` compiles them all before any decode trace exists.
    prefill_forward(streams, cache=, user_slot=, start_pos=) -> torch [1, seq_len, 16384]
        host orchestration of the above for a whole prompt of any length seq_len >= 1 (chunks of <= 2048 tokens).

Decode (B = max_batch users, one token each): `decode_forward(streams [1, 1, B, 16384], cache=, page_table=, cur_pos=)`
with `cur_pos` [B] int32 device tensor (absolute position of the token; -1 = inactive slot, whose state is left
bit-identical: idle users and users between the chunks of their own prefill). `update_page_table` rewrites the bound
page table in place at a request boundary. Device-only; all state lives in persistent
buffers updated in place, so `capture_decode_trace` records it once and `DecodeTrace.replay` re-executes it after the
caller refreshes `streams` / `cur_pos` (`check_decode_positions` validates positions at that boundary).

Context: KDA kinds have no positional limit (constant-size state). The MLA/DSA kind runs the DSA dense-exact regime
kv_len <= 2051 (`indexer.check_dense_exact`), the regime in which DSA top-k selects every visible token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.config import LayerSpec, round_up
from models.autoports.zai_org_glm_5_3_flash.tt.decoder_layer import DecoderLayer
from models.autoports.zai_org_glm_5_3_flash.tt.indexer import check_dense_exact
from models.autoports.zai_org_glm_5_3_flash.tt.kda import KDAStateCache
from models.autoports.zai_org_glm_5_3_flash.tt.weights import TileCacheSource, TorchStateDictSource
from models.common.lightweightmodule import LightweightModule

PREFILL_CHUNK = 2048
# Physical chunk granularity per attention kind: KDA pads to the 32-token recurrence chunk, so a physical chunk never
# contains a fully padded recurrence chunk (prepare_chunk_recurrence leaves those outputs unspecified); MLA pads to the
# 64-token attention q/k chunk (= page size).
PREFILL_ALIGN = {"kda": 32, "mla": 64}
BLOCK_SIZE = 64
CHUNK_TABLE_WIDTH = PREFILL_CHUNK // BLOCK_SIZE
HC_WIDTH = 4 * 4096
DRAM = ttnn.DRAM_MEMORY_CONFIG


@dataclass
class DecoderCache:
    """Per-layer request state for `max_batch` user slots."""

    max_batch: int
    max_seq_len: int
    kda: KDAStateCache | None = None
    latent: ttnn.Tensor | None = None
    blocks_per_seq: int = 0
    page_table_host: torch.Tensor | None = None
    # persistent per-chunk prefill inputs (refreshed by begin_prefill_chunk)
    chunk_table: ttnn.Tensor | None = None
    user_table: ttnn.Tensor | None = None
    _zero_recurrent: ttnn.Tensor | None = field(default=None, repr=False)
    _zero_conv: ttnn.Tensor | None = field(default=None, repr=False)


def bucket_for(n: int, align: int) -> int:
    """Physical length of a chunk with n valid tokens."""
    if not 1 <= n <= PREFILL_CHUNK:
        raise ValueError(f"chunk of {n} tokens outside [1, {PREFILL_CHUNK}]")
    return round_up(n, align)


def plan_prefill_chunks(seq_len: int, align: int) -> list[tuple[int, int, int]]:
    """[(offset in the new input, valid tokens, physical length)]; every chunk but the last is full."""
    if seq_len < 1:
        raise ValueError("prefill needs at least one token")
    return [
        (off, min(PREFILL_CHUNK, seq_len - off), bucket_for(min(PREFILL_CHUNK, seq_len - off), align))
        for off in range(0, seq_len, PREFILL_CHUNK)
    ]


class FunctionalDecoder(LightweightModule):
    def __init__(self, layer: DecoderLayer, hf_config, mesh_device, *, block_size: int = BLOCK_SIZE):
        self.layer = layer
        self.cfg = hf_config
        self.spec: LayerSpec = layer.spec
        self.layer_idx = layer.layer_idx
        self.device = mesh_device
        self.block_size = block_size
        self.is_kda = self.spec.attention == "kda"
        self.prefill_align = PREFILL_ALIGN[self.spec.attention]
        self.prefill_buckets = tuple(range(self.prefill_align, PREFILL_CHUNK + 1, self.prefill_align))
        u32 = lambda v: ttnn.from_torch(
            torch.tensor([v], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=DRAM,
        )
        self._valid_end = u32(PREFILL_CHUNK)  # KDA actual_end of the current chunk
        self._valid_end_f = ttnn.from_torch(
            torch.full((1, 1, 1, 1), float(PREFILL_CHUNK)),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=DRAM,
        )
        self._chunk_start = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=DRAM,
        )

    # ------------------------------------------------------------------------------------------------------- loading
    @classmethod
    def from_weight_cache(
        cls,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        cache_dir: str | None = None,
        manifest_path: str | None = None,
        block_size: int = BLOCK_SIZE,
        source=None,
    ):
        source = source or TileCacheSource(cache_dir, manifest_path)
        layer = DecoderLayer(hf_config, layer_idx, mesh_device, source, block_size=block_size)
        return cls(layer, hf_config, mesh_device, block_size=block_size)

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx: int, mesh_device, block_size: int = BLOCK_SIZE):
        source = TorchStateDictSource({layer_idx: state_dict}, hf_config)
        return cls.from_weight_cache(
            hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device, block_size=block_size, source=source
        )

    # --------------------------------------------------------------------------------------------------------- state
    def allocate_cache(self, *, max_batch: int, max_seq_len: int) -> DecoderCache:
        if max_seq_len > self.cfg.max_position_embeddings:
            raise ValueError(f"max_seq_len {max_seq_len} > advertised context {self.cfg.max_position_embeddings}")
        cache = DecoderCache(max_batch=max_batch, max_seq_len=max_seq_len)
        if self.is_kda:
            cache.kda = self.layer.attn.allocate_state(max_batch)
            cache._zero_recurrent = ttnn.zeros(
                tuple(cache.kda.recurrent[0].shape),
                dtype=cache.kda.recurrent[0].dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=DRAM,
            )
            cache._zero_conv = ttnn.zeros(
                tuple(cache.kda.conv[0].shape),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=DRAM,
            )
        else:
            check_dense_exact(self.layer_idx, max_seq_len)
            # a chunk's padded tail (< prefill_align rows) may end past max_seq_len: give it page slots of its own;
            # page-table rows are a multiple of 8 int32 entries (32-byte sticks, chunked SDPA requirement)
            blocks = round_up(max_seq_len + self.prefill_align - 1, self.block_size) // self.block_size
            cache.blocks_per_seq = round_up(blocks, 8)
            cache.latent = self.layer.attn.allocate_cache(max_batch * cache.blocks_per_seq)
            i32 = lambda shape: ttnn.from_torch(
                torch.zeros(shape, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=DRAM,
            )
            cache.chunk_table = i32((1, CHUNK_TABLE_WIDTH))
            cache.user_table = i32((1, cache.blocks_per_seq))
        return cache

    def make_page_table(self, cache: DecoderCache, permutation: torch.Tensor | None = None) -> ttnn.Tensor:
        """Request-boundary helper: page table [slots, blocks_per_seq] int32 on device (optional block permutation)."""
        n = cache.max_batch * cache.blocks_per_seq
        perm = torch.arange(n) if permutation is None else permutation
        cache.page_table_host = perm.reshape(cache.max_batch, cache.blocks_per_seq).to(torch.int32)
        return ttnn.from_torch(
            cache.page_table_host,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=DRAM,
        )

    def update_page_table(self, cache: DecoderCache, page_table: ttnn.Tensor, permutation: torch.Tensor) -> None:
        """Request-boundary helper: rewrite the bound page table in place (same device buffer, so a captured decode
        trace sees the new mapping) and the host copy used by prefill. `permutation` [slots * blocks_per_seq] maps
        (slot, logical block) to physical blocks; blocks of a live request must keep their mapping."""
        n = cache.max_batch * cache.blocks_per_seq
        perm = permutation.reshape(-1).to(torch.int32)
        if perm.numel() != n or sorted(perm.tolist()) != list(range(n)):
            raise ValueError(f"page table must be a permutation of the {n} physical blocks")
        cache.page_table_host = perm.reshape(cache.max_batch, cache.blocks_per_seq)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(cache.page_table_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), page_table
        )

    def snapshot_state(self, cache: DecoderCache):
        """Device-side copy of the mutable per-user KDA state (S + conv history); the MLA latent cache is only written
        at cur_pos, which a replayed decode rewrites with the same value. Used around warm-up / capture."""
        if self.is_kda:
            return [ttnn.clone(t) for t in cache.kda.recurrent], [ttnn.clone(t) for t in cache.kda.conv]
        return None

    def restore_state(self, cache: DecoderCache, snap):
        if self.is_kda:
            for src, dst in zip(snap[0] + snap[1], cache.kda.recurrent + cache.kda.conv):
                ttnn.copy(src, dst)

    def reset_slot(self, cache: DecoderCache, slot: int):
        """Fresh request in `slot`: zero its KDA state (MLA needs nothing: attention never reads past cur_pos)."""
        if self.is_kda:
            ttnn.copy(cache._zero_recurrent, cache.kda.recurrent[slot])
            ttnn.copy(cache._zero_conv, cache.kda.conv[slot])

    # ------------------------------------------------------------------------------------------------------- prefill
    def begin_prefill_chunk(self, cache: DecoderCache, *, user_slot: int, chunk_start: int, n_valid: int):
        """Host boundary before `prefill_chunk`: validate and write the chunk's device inputs."""
        if not 0 <= user_slot < cache.max_batch:
            raise ValueError(f"user_slot {user_slot} outside [0, {cache.max_batch})")
        if not 1 <= n_valid <= PREFILL_CHUNK or chunk_start < 0 or chunk_start + n_valid > cache.max_seq_len:
            raise ValueError(
                f"chunk [{chunk_start}, {chunk_start + n_valid}) invalid for max_seq_len {cache.max_seq_len}"
            )
        if self.is_kda:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(torch.tensor([n_valid], dtype=torch.int32), dtype=ttnn.uint32), self._valid_end
            )
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(torch.full((1, 1, 1, 1), float(n_valid)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
                self._valid_end_f,
            )
        else:
            if chunk_start % self.block_size:
                raise ValueError(f"MLA chunk start {chunk_start} must be a multiple of {self.block_size}")
            check_dense_exact(self.layer_idx, chunk_start + n_valid)
            if cache.page_table_host is None:
                raise ValueError("bind a page table with make_page_table before prefill")
            row = cache.page_table_host[user_slot]
            b0 = chunk_start // self.block_size
            blocks = row[b0 : b0 + CHUNK_TABLE_WIDTH]
            blocks = torch.cat([blocks, blocks[-1:].expand(CHUNK_TABLE_WIDTH - blocks.numel())])  # unused tail entries
            put = lambda t, dst: ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(t.reshape(tuple(dst.shape)).to(torch.int32), dtype=ttnn.int32), dst
            )
            put(blocks, cache.chunk_table)
            put(row, cache.user_table)
            put(torch.tensor([chunk_start]), self._chunk_start)
        if chunk_start == 0:
            self.reset_slot(cache, user_slot)

    def prefill_chunk(self, x: ttnn.Tensor, *, cache: DecoderCache, user_slot: int = 0) -> ttnn.Tensor:
        """Device-only pass over one physical chunk x [1, 1, P, 16384] fp32 -> [1, 1, P, 16384] fp32."""
        if self.is_kda:
            attn = lambda h: self.layer.attn.prefill(h, cache.kda, user_slot, self._valid_end, self._valid_end_f)
        else:
            attn = lambda h: self.layer.attn.prefill(
                h, cache.latent, cache.chunk_table, cache.user_table, self._chunk_start
            )
        return self.layer.forward(x, attn, decode=False)

    def prefill_forward(
        self, streams: torch.Tensor, *, cache: DecoderCache, user_slot: int = 0, start_pos: int = 0
    ) -> torch.Tensor:
        """Host orchestration of a whole prompt: streams (torch, [1, seq_len, 4, 4096] or [1, 1, seq_len, 16384]) ->
        torch [1, seq_len, 16384] fp32. Per chunk: pad on host, transfer, `begin_prefill_chunk`, `prefill_chunk`,
        read back the valid rows."""
        rows = streams.reshape(-1, HC_WIDTH).float()
        seq_len = rows.shape[0]
        if start_pos + seq_len > cache.max_seq_len:
            raise ValueError(f"prefill end {start_pos + seq_len} > allocated max_seq_len {cache.max_seq_len}")
        outs = []
        for off, n, phys in plan_prefill_chunks(seq_len, self.prefill_align):
            chunk = torch.zeros(phys, HC_WIDTH)
            chunk[:n] = rows[off : off + n]
            x = ttnn.from_torch(
                chunk.reshape(1, 1, phys, HC_WIDTH),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=DRAM,
            )
            self.begin_prefill_chunk(cache, user_slot=user_slot, chunk_start=start_pos + off, n_valid=n)
            y = self.prefill_chunk(x, cache=cache, user_slot=user_slot)
            outs.append(ttnn.to_torch(y).float().reshape(phys, HC_WIDTH)[:n])
            ttnn.deallocate(x)
            ttnn.deallocate(y)
        return torch.cat(outs).reshape(1, seq_len, HC_WIDTH)

    def prepare_prefill(self, cache: DecoderCache):
        """Compile every prefill program before any decode trace is captured: one chunk of each reachable physical
        bucket. Overwrites slot 0's state/pages (call at setup, before requests)."""
        for p in self.prefill_buckets:
            if p > bucket_for(min(cache.max_seq_len, PREFILL_CHUNK), self.prefill_align):
                continue  # not reachable with this cache
            n = min(p, cache.max_seq_len)
            x = ttnn.zeros(
                (1, 1, p, HC_WIDTH), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=DRAM
            )
            self.begin_prefill_chunk(cache, user_slot=0, chunk_start=0, n_valid=n)
            ttnn.deallocate(self.prefill_chunk(x, cache=cache, user_slot=0))
            ttnn.deallocate(x)
        self.reset_slot(cache, 0)
        ttnn.synchronize_device(self.device)

    # -------------------------------------------------------------------------------------------------------- decode
    def decode_forward(
        self,
        streams: ttnn.Tensor,
        *,
        cache: DecoderCache,
        page_table: ttnn.Tensor | None = None,
        cur_pos: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """streams [1, 1, B, 16384] fp32 (B = cache.max_batch users) -> [1, 1, B, 16384] fp32. Device-only.
        cur_pos[u] == -1 marks slot u inactive (idle, or between the chunks of its own prefill): its KDA state / conv
        history stay bit-identical and MLA skips its cache write and attention; its output row is don't-care."""
        if cur_pos is None:
            raise ValueError("decode_forward needs cur_pos [B] int32 (absolute position per slot, -1 = inactive)")
        if self.is_kda:
            active = self._active_mask(cur_pos)
            attn = lambda h: self.layer.attn.decode(h, cache.kda, active)
        else:
            attn = lambda h: self.layer.attn.decode(h, cache.latent, page_table, cur_pos)
        return self.layer.forward(streams, attn, decode=True)

    def _active_mask(self, cur_pos: ttnn.Tensor) -> ttnn.Tensor:
        """cur_pos [B] int32 -> [B,1,1,1] fp32, 1.0 where cur_pos >= 0 (device ops: refreshed with cur_pos per replay)."""
        b = cur_pos.shape[0]
        cp = ttnn.to_layout(ttnn.reshape(cur_pos, (b, 1)), ttnn.TILE_LAYOUT)
        active = ttnn.typecast(ttnn.ge(ttnn.typecast(cp, ttnn.float32), 0.0), ttnn.float32)
        return ttnn.reshape(active, (b, 1, 1, 1))

    def check_decode_positions(self, positions, cache: DecoderCache | None = None):
        """Host-side validation of the positions a caller is about to write into `cur_pos` (request boundary);
        -1 marks an inactive slot."""
        live = [p for p in positions if p != -1]
        if cache is not None:
            if len(positions) != cache.max_batch:
                raise ValueError(f"{len(positions)} positions for {cache.max_batch} decode slots")
            if min(positions) < -1 or (live and max(live) >= cache.max_seq_len):
                raise ValueError(f"decode positions {positions} outside [0, {cache.max_seq_len}) (or -1 = inactive)")
        if not self.is_kda and live:
            check_dense_exact(self.layer_idx, max(live) + 1)

    def capture_decode_trace(self, streams: ttnn.Tensor, *, cache: DecoderCache, page_table=None, cur_pos=None):
        """Record decode over persistent inputs (streams / cur_pos / page_table / state buffers). The caller warms the
        program cache first (one eager decode_forward with the same tensors) and restores the state it keeps."""
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        out = self.decode_forward(streams, cache=cache, page_table=page_table, cur_pos=cur_pos)
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        return DecodeTrace(self.device, trace_id, streams, out, cur_pos, page_table)


@dataclass
class DecodeTrace:
    device: object
    trace_id: int
    streams_in: ttnn.Tensor
    streams_out: ttnn.Tensor
    cur_pos: ttnn.Tensor | None
    page_table: ttnn.Tensor | None

    def replay(self, blocking: bool = False) -> ttnn.Tensor:
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=blocking)
        return self.streams_out

    def release(self):
        ttnn.release_trace(self.device, self.trace_id)
