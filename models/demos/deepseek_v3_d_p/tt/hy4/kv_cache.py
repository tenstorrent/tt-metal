# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview context-parallel KV cache: the MLA latent and index-key caches, block-cyclic over 32 chips (#65886).

The cache is the GLM-5.3 sparse-MLA pair the adapter allocates (``GLM53Adapter.allocate_kv_cache``), striped over
SP x TP (``tp_axis`` set, "KV dedup"). This module pins its geometry for Hy4 and gives every reader one definition of
it: prefill (written by ``update_padded_kv_cache``), the migration table (``populate_kv_chunk_address_table_*``),
and the decode indexer / sparse decode (#65888 / #65889), which must read the same rows without reshuffling.

Where token ``t`` lives (``KvStripe``)::

    B = block = 128 tokens, N = sp * tp = 32 chips, P = period = B * N = 4096 tokens
    chip(t)      = (t // B) % N                    linear chip, SP-outer / TP-inner: chip = sp_coord * tp + tp_coord
    row(t)       = (t // P) * B + t % B            row in that chip's slab of the (user, layer) slot
    slot index   = user * num_layers + layer       KVPE: every layer; index-K: full-indexer rank (21 layers)
    mesh coord   = coord[sp_axis] = chip // tp, coord[tp_axis] = chip % tp

The map depends on the position only, not on how the prompt was chunked: ``update_padded_kv_cache`` writes each
chip's rows at the offsets this map gives for any tile-aligned chunk start (its reader inverts the producer's
rotation of a mid-slab chunk, ``common/prefill/chunk_layout.py``), so a resumed or prefix-hit prefill and a
single-shot prefill leave identical caches. The block is the op's per-chip written window, ``chunk / (sp * tp)``,
which is why the prefill chunk size is pinned to ``P``: any other chunk on 32 chips gives a different block (5120
gives 160) and a cache decode could not address by the 128-token rule.

On-device formats (unchanged from GLM-5.3, measured by ``test_hy4_kv_cache_device.py``):

  * KVPE: ``MlaKvCacheFormat.SCALED_FP8``, ROW_MAJOR ``fp8_e4m3`` storage, one 656-byte packed row per token
    ``[512 FP8 latent | 4 FP32 scales, one per 128 latent values | 64 BF16 rope (interleaved pairs)]``, each row
    padded to 704 bytes of DRAM (64-byte page alignment).
  * index keys: ``bfloat8_b`` TILE, 128 wide (``[rope 64 interleaved | nope 64]``, Hadamard-rotated), 136 bytes
    per token, one slot per full-indexer layer.
  * both ND-sharded in 32-token shards round-robin over the chip's DRAM banks (``init_kvpe_cache``), slot-major.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch

# The block every Hy4 reader assumes (#65884 / #65886). The DRAM bank shard is 32 tokens, so a block is 4 shards.
HY4_KV_BLOCK = 128
HY4_KV_CHIPS = 32
TILE = 32

# Bytes per token per layer of the on-device formats. The packed scaled-FP8 row is 656 bytes
# (MlaKvCacheGeometry.packed_row_bytes for 512 + 64), but each ROW_MAJOR page is padded to the 64-byte DRAM alignment,
# so a row takes 704 bytes of DRAM (measured: test_hy4_kv_cache_device.py). bfloat8_b: 1088 bytes per 32x32 tile, one
# mantissa byte per value plus one shared exponent byte per 16, so 136 bytes per 128-wide row, no padding.
KVPE_SCALED_FP8_ROW_BYTES = 512 + 4 * 4 + 64 * 2
DRAM_ALIGNMENT = 64
KVPE_SCALED_FP8_DRAM_ROW_BYTES = -(-KVPE_SCALED_FP8_ROW_BYTES // DRAM_ALIGNMENT) * DRAM_ALIGNMENT
INDEX_K_BFP8_ROW_BYTES = 128 * 1088 // 1024


@dataclass(frozen=True)
class KvStripe:
    """Block-cyclic placement of one (user, layer) KV slot over an ``sp x tp`` mesh. See the module docstring."""

    sp: int
    tp: int
    sp_axis: int = 0
    block: int = HY4_KV_BLOCK

    def __post_init__(self) -> None:
        if self.sp_axis not in (0, 1):
            raise ValueError(f"sp_axis must be 0 or 1, got {self.sp_axis}")
        if self.sp < 1 or self.tp < 1:
            raise ValueError(f"sp and tp must be positive, got sp={self.sp} tp={self.tp}")
        if self.block % TILE:
            raise ValueError(f"block {self.block} must be a multiple of {TILE} (the cache writes whole tiles)")

    # --- construction -------------------------------------------------------------------------------------------
    @classmethod
    def from_mesh(cls, mesh_shape: Sequence[int], sp_axis: int, chunk_size: int) -> "KvStripe":
        """The stripe ``update_padded_kv_cache`` produces for a prefill chunk of ``chunk_size`` tokens on this mesh:
        block = chunk / (sp * tp)."""
        sp, tp = int(mesh_shape[sp_axis]), int(mesh_shape[1 - sp_axis])
        n = sp * tp
        if chunk_size % (n * TILE):
            raise ValueError(f"chunk {chunk_size} does not give every one of {n} chips whole tiles")
        return cls(sp=sp, tp=tp, sp_axis=sp_axis, block=chunk_size // n)

    @classmethod
    def from_params(cls, params) -> "KvStripe":
        """From the runner's ``PrefillRunParams`` (or a runtime config with the same fields)."""
        return cls.from_mesh(tuple(params.mesh_shape), params.sp_axis, params.chunk_size)

    # --- geometry -----------------------------------------------------------------------------------------------
    @property
    def tp_axis(self) -> int:
        return 1 - self.sp_axis

    @property
    def num_chips(self) -> int:
        return self.sp * self.tp

    @property
    def period(self) -> int:
        """Tokens per full cycle over every chip; the prefill chunk size that writes this stripe."""
        return self.block * self.num_chips

    @property
    def mesh_shape(self) -> tuple[int, int]:
        shape = [0, 0]
        shape[self.sp_axis], shape[self.tp_axis] = self.sp, self.tp
        return tuple(shape)

    def rows_per_chip(self, seq_len: int) -> int:
        if seq_len % self.period:
            raise ValueError(f"cache length {seq_len} is not a whole number of {self.period}-token periods")
        return seq_len // self.num_chips

    # --- token -> storage ---------------------------------------------------------------------------------------
    def chip(self, pos):
        """Linear chip holding position(s) ``pos`` (int or integer tensor)."""
        return (pos // self.block) % self.num_chips

    def row(self, pos):
        """Row within that chip's slot slab."""
        return (pos // self.period) * self.block + pos % self.block

    def position(self, chip, row):
        """Inverse of (chip, row): the global token position stored there."""
        return (row // self.block) * self.period + chip * self.block + row % self.block

    def coord(self, chip: int) -> tuple[int, int]:
        """Physical mesh coordinate of linear chip ``chip``: SP position ``chip // tp`` on ``sp_axis``."""
        if not 0 <= chip < self.num_chips:
            raise ValueError(f"chip {chip} outside [0, {self.num_chips})")
        c = [0, 0]
        c[self.sp_axis], c[self.tp_axis] = divmod(chip, self.tp)
        return tuple(c)

    def chip_of_coord(self, coord: Sequence[int]) -> int:
        return int(coord[self.sp_axis]) * self.tp + int(coord[self.tp_axis])

    def locate(self, pos: int) -> tuple[tuple[int, int], int]:
        """(mesh coordinate, row) of one position: where decode appends token ``pos``."""
        return self.coord(int(self.chip(pos))), int(self.row(pos))

    # --- what a chip holds --------------------------------------------------------------------------------------
    def populated_rows(self, chip: int, kv_len: int) -> int:
        """Rows of ``chip`` holding positions < ``kv_len``; always a prefix of its slab (rows increase with
        position). This is the per-chip length the decode indexer scores and sparse decode may select from."""
        full, rem = divmod(kv_len, self.period)
        return full * self.block + min(max(rem - chip * self.block, 0), self.block)

    def local_positions(self, chip: int, num_rows: int) -> torch.Tensor:
        """Global positions of rows ``[0, num_rows)`` of ``chip``: maps a chip-local top-k row back to a token."""
        return self.position(chip, torch.arange(num_rows, dtype=torch.int64))

    def chunk_positions(self, chunk_start: int) -> list[torch.Tensor]:
        """Positions each chip receives from the prefill chunk ``[chunk_start, chunk_start + period)``: the block
        rows the chunk fills on that chip, in row order. Chunk starts must be tile aligned (``check_resume``)."""
        self.check_resume(chunk_start)
        out = []
        for c in range(self.num_chips):
            first = self.populated_rows(c, chunk_start)
            out.append(self.local_positions(c, first + self.block)[first:])
        return out

    # --- host stripe / de-stripe --------------------------------------------------------------------------------
    def stripe(self, rows: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """``[S, ...]`` token-order rows -> ``[num_chips, seq_len / num_chips, ...]`` (zero padded to seq_len)."""
        seq_len = seq_len or -(-rows.shape[0] // self.period) * self.period
        if rows.shape[0] > seq_len:
            raise ValueError(f"{rows.shape[0]} rows do not fit a {seq_len}-token cache")
        full = rows.new_zeros((seq_len, *rows.shape[1:]))
        full[: rows.shape[0]] = rows
        r = self.rows_per_chip(seq_len)
        # [cycles, chips, block, ...] -> [chips, cycles, block, ...] -> [chips, rows, ...]
        x = full.reshape(seq_len // self.period, self.num_chips, self.block, *rows.shape[1:]).transpose(0, 1)
        return x.reshape(self.num_chips, r, *rows.shape[1:])

    def destripe(self, per_chip: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """``[num_chips, rows, ...]`` chip-order slabs -> ``[length, ...]`` token order. Inverse of ``stripe``."""
        n, r = per_chip.shape[0], per_chip.shape[1]
        if n != self.num_chips or r % self.block:
            raise ValueError(f"expected [{self.num_chips}, k*{self.block}, ...], got {list(per_chip.shape)}")
        x = per_chip.reshape(n, r // self.block, self.block, *per_chip.shape[2:]).transpose(0, 1)
        out = x.reshape(n * r, *per_chip.shape[2:])
        return out if length is None else out[:length]

    # --- contracts ----------------------------------------------------------------------------------------------
    def check_chunk_size(self, chunk_size: int) -> None:
        """A prefill chunk of ``chunk_size`` writes this stripe only when chunk / (sp * tp) == block."""
        if chunk_size != self.period:
            raise ValueError(
                f"the Hy4 KV cache stripes {self.block}-token blocks over {self.num_chips} chips, so the prefill "
                f"chunk must be {self.period} tokens (update_padded_kv_cache writes chunk / {self.num_chips} rows "
                f"per chip); PREFILL_CHUNK_SIZE={chunk_size} would write {chunk_size // self.num_chips}-token "
                "blocks that decode cannot address"
            )

    @staticmethod
    def check_resume(start: int) -> None:
        """A chunk (prefill resume, prefix-cache hit) may start at any tile-aligned position: the write op takes a
        tile-aligned ``kv_actual_global``. Block alignment is not required."""
        if start < 0 or start % TILE:
            raise ValueError(f"a prefill chunk must start at a multiple of {TILE} tokens, got {start}")

    def descriptor(self, **extra) -> dict:
        """JSON-able layout record for decode and serving (vLLM plugin, KV handoff)."""
        return {
            "kind": "block_cyclic",
            "block": self.block,
            "num_chips": self.num_chips,
            "period": self.period,
            "sp": self.sp,
            "tp": self.tp,
            "sp_axis": self.sp_axis,
            "tp_axis": self.tp_axis,
            "chip_order": [list(self.coord(c)) for c in range(self.num_chips)],
            "chip": "(pos // block) % num_chips",
            "row": "(pos // period) * block + pos % block",
            "slot": "user * num_layers + layer",
            **extra,
        }


def hy4_stripe(mesh_shape: Sequence[int], sp_axis: int, chunk_size: int) -> KvStripe:
    """The Hy4 stripe on a 32-chip mesh, refusing a chunk size that would not write 128-token blocks."""
    stripe = KvStripe.from_mesh(mesh_shape, sp_axis, chunk_size)
    if stripe.num_chips == HY4_KV_CHIPS:
        KvStripe(sp=stripe.sp, tp=stripe.tp, sp_axis=sp_axis).check_chunk_size(chunk_size)
    return stripe


# --- index-key layers ------------------------------------------------------------------------------------------
def full_indexer_layers(indexer_types: Sequence[str]) -> list[int]:
    return [i for i, t in enumerate(indexer_types) if t == "full"]


def index_slot(indexer_types: Sequence[str], layer: int) -> int:
    """Index-key cache slot of a full-indexer layer: its rank among the full layers (``full_indexer_rank``)."""
    if indexer_types[layer] != "full":
        raise ValueError(f"layer {layer} is a shared indexer layer and owns no index-key slot")
    return sum(1 for t in indexer_types[:layer] if t == "full")


# --- capacity --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class KvBudget:
    """Per-token KV bytes of the Hy4 cache pair and what fits in a chip's free DRAM."""

    num_layers: int = 78
    num_index_layers: int = 21
    kvpe_row_bytes: int = KVPE_SCALED_FP8_DRAM_ROW_BYTES
    index_row_bytes: int = INDEX_K_BFP8_ROW_BYTES
    num_chips: int = HY4_KV_CHIPS

    @property
    def values_per_token(self) -> int:
        return self.num_layers * 576 + self.num_index_layers * 128

    @property
    def bytes_per_token(self) -> int:
        return self.num_layers * self.kvpe_row_bytes + self.num_index_layers * self.index_row_bytes

    def tokens_per_chip(self, free_bytes_per_chip: int) -> int:
        return free_bytes_per_chip // self.bytes_per_token

    def tokens(self, free_bytes_per_chip: int) -> int:
        """Whole-mesh tokens: each chip holds 1/num_chips of every token."""
        return self.tokens_per_chip(free_bytes_per_chip) * self.num_chips

    def users(self, free_bytes_per_chip: int, context: int) -> int:
        return self.tokens(free_bytes_per_chip) // context


def bank_aligned_max_seq_len(stripe: KvStripe, max_tokens: int, num_dram_banks: int = 8) -> int:
    """Largest cache length <= ``max_tokens`` that ``init_kvpe_cache(bank_aligned=True)`` accepts: whole periods
    and every slot starting on DRAM bank 0 (rows per chip a multiple of 32 * banks)."""
    quantum = math.lcm(stripe.period, stripe.num_chips * TILE * num_dram_banks)
    return max_tokens // quantum * quantum


def describe_layers(indexer_types: Iterable[str]) -> dict:
    types = list(indexer_types)
    return {
        "num_layers": len(types),
        "full_indexer_layers": full_indexer_layers(types),
        "index_slot_of_layer": {layer: index_slot(types, layer) for layer in full_indexer_layers(types)},
    }


# --- device read-back ------------------------------------------------------------------------------------------
def read_chip_slabs(storage, mesh_device, stripe: KvStripe) -> torch.Tensor:
    """Every chip's slab of a striped cache, in linear chip order: ``[num_chips, slots, rows, row_width]``.

    Reads each device by its physical mesh coordinate (``ConcatMesh2dToTensor`` over dims 1 and 2), so it does not
    depend on the cache's declared topology, which lists coordinates row-major and is not the stripe order when SP
    is on mesh axis 1 (``init_kvpe_cache``). The rows are physical: a scaled-FP8 KVPE cache returns its packed bytes
    (decode them with ``MlaKvCache.unpack_host``)."""
    import ttnn

    shape = tuple(mesh_device.shape)
    if (shape[0], shape[1]) != stripe.mesh_shape:
        raise ValueError(f"stripe is for mesh {stripe.mesh_shape}, the device mesh is {shape}")
    host = ttnn.to_torch(storage, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 2), mesh_shape=shape))
    rows = host.shape[2] // shape[1]  # host: [slots, mesh rows, mesh cols * rows, width]
    out = []
    for chip in range(stripe.num_chips):
        r, c = stripe.coord(chip)
        out.append(host[:, r, c * rows : (c + 1) * rows])
    return torch.stack(out)


def read_slot(storage, mesh_device, stripe: KvStripe, slot: int, length: Optional[int] = None) -> torch.Tensor:
    """One (user, layer) slot of a striped cache in token order: ``[length, row_width]`` physical rows."""
    slabs = read_chip_slabs(storage, mesh_device, stripe)[:, slot]
    return stripe.destripe(slabs, length)


# --- distributed top-k over the stripe (the decode / prefill indexer's selection, #65887 / #65888) -------------
def merge_local_topk(
    local_values: torch.Tensor, local_rows: torch.Tensor, stripe: KvStripe, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Host reference of the gather + merge: every chip's local top-k ``[chips, Q, k_local]`` (scores and chip-local
    rows) -> the global top-k ``(values, positions)`` ``[Q, k]``. -inf entries (fewer than k valid keys) map to -1.

    The global top-k is contained in the union of the local top-k whenever ``k_local >= min(k, rows held)``, since a
    key ranked below k on its own chip has k better keys on that chip alone."""
    chips, q, kl = local_values.shape
    chip = torch.arange(chips).view(chips, 1, 1).expand(chips, q, kl)
    pos = stripe.position(chip, local_rows.long())
    values = local_values.permute(1, 0, 2).reshape(q, chips * kl)
    pos = pos.permute(1, 0, 2).reshape(q, chips * kl)
    v, j = values.topk(k, dim=-1)
    p = pos.gather(1, j).masked_fill(torch.isneginf(v), -1)
    return v, p


def stripe_key_axis(t: torch.Tensor, stripe: KvStripe) -> torch.Tensor:
    """``[Q, S]`` (a score matrix over S keys) -> ``[mesh rows, mesh cols, Q, S / 32]``: each chip's own key columns,
    in its row order, at its mesh coordinate. ``ttnn.ShardTensor2dMesh(dims=(0, 1))`` places it."""
    slabs = stripe.stripe(t.transpose(0, 1)).transpose(1, 2)  # [chips, Q, rows]
    rows, cols = stripe.mesh_shape
    out = slabs.new_empty((rows, cols, *slabs.shape[1:]))
    for chip in range(stripe.num_chips):
        r, c = stripe.coord(chip)
        out[r, c] = slabs[chip]
    return out


def device_topk_merge(local_scores, local_positions, k: int, k_local: int, stripe: KvStripe, topology=None):
    """Distributed top-k on device: each chip keeps a local top-``k_local`` of its own key rows, the 32 local lists
    are all-gathered (TP axis, then SP axis) and one more top-k picks the global ``k``.

    ``local_scores``: per chip ``[1, 1, Q, rows]`` (bf16 TILE, -inf where a key is invalid); ``local_positions``: the
    global position of each of those keys, same shape, uint32 TILE (``stripe.local_positions``). Both top-ks carry the
    positions as their labels (``indices_tensor``), so the result is positions directly and does not depend on the
    gather order. Returns ``(values, positions)``, ``[1, 1, Q, k]`` on every chip. Needs a fabric for the gathers."""
    import ttnn

    topology = topology or ttnn.Topology.Linear
    v, p = ttnn.topk(local_scores, k=k_local, dim=-1, largest=True, sorted=True, indices_tensor=local_positions)
    for axis in (stripe.tp_axis, stripe.sp_axis):
        gv = ttnn.all_gather(v, dim=3, cluster_axis=axis, topology=topology)
        gp = ttnn.all_gather(p, dim=3, cluster_axis=axis, topology=topology)
        ttnn.deallocate(v)
        ttnn.deallocate(p)
        v, p = gv, gp
    mv, mp = ttnn.topk(v, k=k, dim=-1, largest=True, sorted=True, indices_tensor=p)
    ttnn.deallocate(v)
    ttnn.deallocate(p)
    return mv, mp
