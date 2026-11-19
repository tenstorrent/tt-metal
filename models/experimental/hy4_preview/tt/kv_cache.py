# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview's KV cache on the SC1 layout: context parallel, block-cyclic over every chip of the
mesh (tenstorrent/tt-metal#65886).

What is cached. Per token, every layer stores one MLA latent, (k_pass 512 | k_rope 64) = 576 values
shared by all 64 heads (reference/attention.py Hy4Attention.latent), and the 21 full-indexer layers
store one 128-wide index key more. Nothing is stored per head, so the cache cannot be split by head;
it is split by sequence. tt/kv_layout.py BlockCyclicLayout is the contract: blocks of 128 positions
dealt round-robin to the chips, chips counted row-major.

Storage. One tensor per layer and kind, per chip ``[users, 1, slots, width]`` in TILE layout (the
stripe ``[users, slots, width]`` with the unit head dim the ttnn cache ops want), BFP8 or BF16.
Row ``s`` of user ``u`` on chip ``c`` holds position ``layout.position(c, s)`` of that user's
context. A chip holds 1/32 of every context and no chip holds a copy of another's rows.

Writing from prefill (``write_prefill``), on the device with no host round trip. The latent of a
prompt arrives whole on every chip, ``[1, 1, T, width]``. It is padded to whole rounds (a round is
one block per chip, 32 x 128 = 4096 positions), viewed as ``[1, rounds, 4096, width]`` and cut by
ttnn.mesh_partition on the 4096 dim: partition slice ``c`` of a round is block ``c`` of that round,
which is exactly the block chip ``c`` owns. Each chip is left with ``[1, 1, rounds * 128, width]``,
its own rows in slot order, and one fill_cache_for_user_ puts them at the first round's slot.
mesh_partition is a per-chip slice, so prefill costs no fabric traffic. A write may continue a
context (a cached prefix, the next chunk of a long prompt): when its start is not on a round
boundary, the rows of the first round that precede it are blended back from the cache before the
write. A start inside a tile (not a multiple of 32) moves every row of the chunk to another place
in its tile, so the chunk's tiles are rebuilt: untilize, pad in row major, tilize. That costs one
pass over the chunk; a chunk that starts on a tile row never leaves TILE layout.
``write_prefill_host`` is the same write with the striping done on the host (kv_layout.split
arithmetic) and one upload per chip, from any start; tests use it as the second path.

Writing from decode (``write_decode``). One token per user, appended at that user's position, must
change one row on one chip. ttnn.experimental.paged_update_cache takes its row indices as a device
tensor and skips a user whose index is -1, and on a mesh that tensor is per chip: the chip that owns
position ``p`` gets ``layout.slot(p)`` and the other 31 get -1 (``append_indices``). The op runs on
every chip and writes on one. It patches the row inside its 32-row tile strip and repacks the strip
in the cache dtype; BFP8 shares an exponent only inside a row, so the other rows of the strip come
back bit-identical. The new latent arrives as it does in prefill, whole on every chip,
``[1, 1, users, width]``.

Reading.
* ``stripe`` is the storage itself, for the indexer: every chip scores its own rows.
* ``slot_position`` (per chip ``[1, 1, 1, slots]`` fp32, the position each slot holds) and
  ``filled`` say which slots hold a position below a context length, on the device.
* ``locate`` turns selected positions (the indexer's merged top-k, ``[1, 1, S, K]`` for S queries,
  the same on every chip) into the rows this chip holds, and ``gather`` reads those rows: every
  chip returns ``K`` rows per query plus a 0/1 ``mine`` mask, 1 on exactly one chip per position.
  Sparse attention attends per chip over the rows flagged 1 and merges the partial outputs.
  ``gather`` is ttnn.embedding with the stripe as its table, and that op wants a row-major bf16
  table: given the TILE storage it first untilizes ALL of it (users x slots rows, to bf16) into a
  temporary, on every call. So a gather returns K rows but costs a pass over the whole stripe and a
  transient bf16 copy of it, per layer and per step; it is not O(K). Reading K rows out of tiled
  BFP8 storage without that pass needs an op ttnn does not have yet. Measured warm on the 8x4 mesh
  for K = 2048 (tests/test_kv_cache.py): 0.1 to 0.2 ms per call on a 4,096-row stripe (a 128K
  context) and 0.25 to 0.5 ms on the 32,768 rows of a 1M context.

Users. The leading storage dim: every user has an own stripe of ``slots`` rows on every chip, and
decode writes all users in one call. The cache keeps no lengths: the engine owns each user's
context length and hands positions in.

Not here: vLLM block paging. This storage is a fixed ``users x slots`` rectangle per chip, and the
user index is the only ownership it knows. The vLLM plugin owns KV as blocks: it hands every request
a page table row, virtual block -> physical block of one pool. The mapping that needs no change to
its allocator is a vLLM block of one round (block_size 4096 tokens = 128 rows on each of the 32
chips): physical block ``b`` is then rows ``[b * 128, b * 128 + 128)`` of a per-chip pool
``[num_blocks, 1, 128, width]``, the same page table serves every chip, and position -> (chip, slot)
inside a request stays kv_layout's. The writes become paged_fill_cache and paged_update_cache with
that table (both take one; the per-chip -1 index of ``append_indices`` carries over). The reads are
what changes: a request's stripe is no longer contiguous, so ``stripe`` has to gather its blocks,
and ``locate`` has to send a slot through the page table (row = page_table[slot // 128] * 128 +
slot % 128).

The layout arithmetic (KvCacheConfig, prefill_plan, host_stripes, append_index_table,
selection_table, locate_rows) is tested on the host; ttnn is only touched by TtHy4KvCache.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from models.experimental.hy4_preview.tt.kv_layout import BlockCyclicLayout

try:
    import ttnn
except ImportError:  # the layout arithmetic below runs without it
    ttnn = None

# config.json indexer_types: the layers marked "full" run their own indexer and cache an index key.
HY4_INDEXER_LAYERS = (0, 1) + tuple(range(5, 78, 4))

LATENT = "latent"
INDEX_KEY = "index_key"

# Stored bytes per value. BFP8 packs a 32x32 tile as 1024 mantissa bytes + 64 exponent bytes.
BYTES_PER_VALUE = {"bf16": 2.0, "bfp8": 1088 / 1024}


@dataclass(frozen=True)
class KvCacheConfig:
    """Cache geometry. Defaults are Hy4-preview's 78 decoder layers on 32 chips."""

    num_layers: int = 78
    indexer_layers: tuple = HY4_INDEXER_LAYERS
    latent_dim: int = 576  # k_pass 512 | k_rope 64
    index_key_dim: int = 128
    max_users: int = 1
    max_context: int = 4096  # positions per user; storage is rounded up to whole rounds
    dtype: str = "bfp8"  # "bfp8" or "bf16"
    layout: BlockCyclicLayout = field(default_factory=BlockCyclicLayout)

    def __post_init__(self):
        assert self.dtype in BYTES_PER_VALUE, f"dtype must be one of {sorted(BYTES_PER_VALUE)}, got {self.dtype}"
        for what, size in (
            ("latent_dim", self.latent_dim),
            ("index_key_dim", self.index_key_dim),
            ("block", self.layout.block),
        ):
            assert size % 32 == 0, f"{what} = {size} is not a whole number of tiles"
        assert all(0 <= layer < self.num_layers for layer in self.indexer_layers), self.indexer_layers

    @property
    def slots(self) -> int:
        """Rows of one user's stripe on one chip."""
        return self.layout.slots(self.max_context)

    @property
    def capacity(self) -> int:
        """Positions one user can hold (max_context rounded up to whole rounds)."""
        return self.slots * self.layout.num_chips

    def width(self, kind: str) -> int:
        return {LATENT: self.latent_dim, INDEX_KEY: self.index_key_dim}[kind]

    def kinds(self, layer: int) -> tuple:
        return (LATENT, INDEX_KEY) if layer in self.indexer_layers else (LATENT,)

    @property
    def values_per_token(self) -> int:
        return self.num_layers * self.latent_dim + len(self.indexer_layers) * self.index_key_dim

    @property
    def bytes_per_token(self) -> float:
        return self.values_per_token * BYTES_PER_VALUE[self.dtype]

    def stripe_bytes(self, kind: str) -> int:
        """Bytes one layer's storage of ``kind`` takes on one chip."""
        return int(self.max_users * self.slots * self.width(kind) * BYTES_PER_VALUE[self.dtype])

    def bytes_per_chip(self, layers=None) -> int:
        """Bytes the storage of ``layers`` (default: all) takes on one chip."""
        layers = range(self.num_layers) if layers is None else layers
        return sum(self.stripe_bytes(kind) for layer in layers for kind in self.kinds(layer))


@dataclass(frozen=True)
class PrefillPlan:
    """How ``length`` positions starting at ``start`` map onto whole rounds of the stripes."""

    start: int
    length: int
    lead: int  # rows between the first round's first position and ``start``
    tail: int  # rows between the last position and the end of its round
    rounds: int
    first_slot: int  # the slot, on every chip, where the first round begins
    slots: int  # rows written per chip: rounds * block


def prefill_plan(layout: BlockCyclicLayout, start: int, length: int, capacity: int | None = None) -> PrefillPlan:
    round_len = layout.num_chips * layout.block
    assert start >= 0 and length > 0, (start, length)
    assert capacity is None or start + length <= capacity, f"positions [{start}, {start + length}) exceed {capacity}"
    lead = start % round_len
    rounds = -(-(lead + length) // round_len)
    return PrefillPlan(
        start=start,
        length=length,
        lead=lead,
        tail=rounds * round_len - lead - length,
        rounds=rounds,
        first_slot=start // round_len * layout.block,
        slots=rounds * layout.block,
    )


def host_stripes(layout: BlockCyclicLayout, x: torch.Tensor, start: int = 0) -> torch.Tensor:
    """Stripe ``x`` ([T, width], positions ``start ..``) the way write_prefill does on the device:
    [num_chips, plan.slots, width], row ``s`` of chip ``c`` being slot ``plan.first_slot + s``, zero
    where the slot's position is outside ``[start, start + T)``."""
    plan = prefill_plan(layout, start, x.shape[0])
    positions = start + torch.arange(x.shape[0])
    out = x.new_zeros(layout.num_chips, plan.slots, *x.shape[1:])
    out[layout.chip(positions), layout.slot(positions) - plan.first_slot] = x
    return out


def prefill_keep_mask(layout: BlockCyclicLayout, plan: PrefillPlan) -> torch.Tensor:
    """[num_chips, block] bool over the first round a prefill touches: True where the slot holds a
    position before ``plan.start``, which the write must leave as it is."""
    chips = torch.arange(layout.num_chips)[:, None]
    slots = plan.first_slot + torch.arange(layout.block)[None, :]
    return layout.position(chips, slots) < plan.start


def append_index_table(layout: BlockCyclicLayout, positions, capacity: int | None = None) -> torch.Tensor:
    """Decode write indices, [num_chips, users] int32: user ``u`` appending at ``positions[u]`` has
    that position's slot on the chip that owns it and -1 (skip) on every other chip. A position of
    None or -1 skips the user on every chip."""
    table = torch.full((layout.num_chips, len(positions)), -1, dtype=torch.int32)
    for user, position in enumerate(positions):
        if position is None or position < 0:
            continue
        assert capacity is None or position < capacity, f"user {user}: position {position} exceeds {capacity}"
        table[layout.chip(position), user] = layout.slot(position)
    return table


def selection_table(layout: BlockCyclicLayout, positions: torch.Tensor):
    """Selected positions [..., K] -> (slot [num_chips, ..., K], mine [num_chips, ..., K] bool): on
    the chip that owns a position, its slot and True; elsewhere slot 0 and False."""
    chips = torch.arange(layout.num_chips).reshape(-1, *([1] * positions.dim()))
    mine = layout.chip(positions)[None] == chips
    return torch.where(mine, layout.slot(positions)[None], 0), mine


def _is_power_of_two(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def locate_rows(ops, positions, chip_id, layout: BlockCyclicLayout, row_offset: int = 0):
    """kv_layout's chip and slot of ``positions``, in fp32 elementwise ops: (rows, mine).

    ``mine`` is 1 where ``chip_id`` owns the position; ``rows`` is its slot there (plus
    ``row_offset``) and ``row_offset`` alone elsewhere. ``ops`` supplies floor, multiply, subtract,
    add and eq: ttnn on the device, and the host test runs these same lines on torch fp32 tensors.
    Exact while the block and the chip count are powers of two (the divisions are exact scalings)
    and positions and rows stay below 2**24 (every intermediate is an integer fp32 holds).
    """
    block, chips = layout.block, layout.num_chips
    assert _is_power_of_two(block) and _is_power_of_two(chips), "locate needs a power-of-two block and chip count"
    blocks = ops.floor(ops.multiply(positions, 1.0 / block))
    rounds = ops.floor(ops.multiply(blocks, 1.0 / chips))
    chip = ops.subtract(blocks, ops.multiply(rounds, float(chips)))
    mine = ops.eq(chip, chip_id)
    # slot = round * block + position % block
    slot = ops.add(ops.subtract(positions, ops.multiply(blocks, float(block))), ops.multiply(rounds, float(block)))
    rows = ops.multiply(slot, mine)
    if row_offset:
        rows = ops.add(rows, float(row_offset))
    return rows, mine


class TtHy4KvCache:
    """The cache of ``layers`` (default: every layer of ``cfg``) on ``mesh_device``.

    ``latent[layer]`` and, on indexer layers, ``index_key[layer]`` are the storage tensors, per chip
    ``[max_users, 1, cfg.slots, width]``. ``zero_init=False`` leaves new storage uninitialised
    (``filled`` is what says which rows mean anything).
    """

    def __init__(self, mesh_device, cfg: KvCacheConfig, layers=None, zero_init: bool = True):
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.layout = cfg.layout
        num_chips = mesh_device.get_num_devices()
        assert self.layout.num_chips == num_chips, f"layout is for {self.layout.num_chips} chips, mesh has {num_chips}"
        self.dtype = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[cfg.dtype]
        self.layers = tuple(range(cfg.num_layers) if layers is None else layers)

        self.latent = {}
        self.index_key = {}
        for layer in self.layers:
            for kind in cfg.kinds(layer):
                shape = (cfg.max_users, 1, cfg.slots, cfg.width(kind))
                self._storages(kind)[layer] = self._allocate(shape, zero_init)

        # The position each slot holds, per chip [1, 1, 1, slots]. fp32 is exact up to 2**24 positions.
        chips = torch.arange(num_chips)[:, None]
        slot_position = self.layout.position(chips, torch.arange(cfg.slots)[None, :])
        assert cfg.capacity <= 2**24, f"{cfg.capacity} positions do not fit fp32 exactly"
        assert cfg.max_users * cfg.slots <= 2**24, f"{cfg.max_users * cfg.slots} storage rows do not fit fp32 exactly"
        self.slot_position = self._per_chip(slot_position.float().reshape(num_chips, 1, 1, 1, cfg.slots), ttnn.float32)
        self.chip_id = self._per_chip(chips.float().reshape(num_chips, 1, 1, 1, 1), ttnn.float32)
        self._token_memory_configs = {}

    # ----------------------------------------------------------------------------------------
    # storage
    # ----------------------------------------------------------------------------------------
    def _storages(self, kind):
        return {LATENT: self.latent, INDEX_KEY: self.index_key}[kind]

    def _storage(self, layer, kind):
        storages = self._storages(kind)
        assert layer in storages, f"layer {layer} has no {kind} storage in this cache"
        return storages[layer]

    def _allocate(self, shape, zero_init):
        if zero_init:
            return ttnn.zeros(
                shape,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), self.dtype, ttnn.TILE_LAYOUT, self.mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )

    def _per_chip(self, t: torch.Tensor, dtype, layout=None):
        """Upload ``t[c]`` to chip ``c`` (row-major): a tensor that differs from chip to chip."""
        layout = ttnn.TILE_LAYOUT if layout is None else layout
        shards = [ttnn.from_torch(t[c].contiguous(), dtype=dtype, layout=layout) for c in range(t.shape[0])]
        return ttnn.to_device(ttnn.from_host_shards(shards, self.mesh_device.shape), self.mesh_device)

    def deallocate(self):
        for storages in (self.latent, self.index_key):
            for tensor in storages.values():
                tensor.deallocate(True)
            storages.clear()
        self.slot_position.deallocate(True)
        self.chip_id.deallocate(True)

    # ----------------------------------------------------------------------------------------
    # write: prefill
    # ----------------------------------------------------------------------------------------
    @staticmethod
    def pack(k_pass, k_rope):
        """(k_pass [1,1,T,512], k_rope [1,1,T,64]), as TtHy4Attention.latent returns them, -> the
        cache entry [1,1,T,576]."""
        return ttnn.concat([k_pass, k_rope], dim=-1)

    def write_prefill(self, layer: int, user: int, latent=None, index_key=None, start: int = 0):
        """Store a prompt's rows for one user, striped on the device.

        latent [1,1,T,latent_dim] and / or index_key [1,1,T,index_key_dim]: bf16 or fp32, whole and
        identical on every chip, row ``t`` being position ``start + t``. Rows of the touched rounds
        past ``start + T`` are zeroed (a context is append-only, nothing valid lives there). A
        ``start`` on a tile row (a multiple of 32) stays in TILE layout; any other start rebuilds the
        chunk's tiles through row major (see the module docstring).
        """
        tile = ttnn.TILE_SIZE
        for kind, x in ((LATENT, latent), (INDEX_KEY, index_key)):
            if x is None:
                continue
            width = self.cfg.width(kind)
            assert tuple(x.shape)[:2] == (1, 1) and x.shape[-1] == width, f"{kind}: got {tuple(x.shape)}"
            plan = prefill_plan(self.layout, start, x.shape[-2], self.cfg.capacity)
            if x.dtype != ttnn.bfloat16:
                x = ttnn.typecast(x, ttnn.bfloat16)
            # [lead zeros | x | tail zeros], whole rounds. ``shift`` is where the first row lands inside
            # its tile; the rest of the lead is whole tiles of zeros.
            shift = plan.lead % tile
            if shift:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
                x = ttnn.pad(x, [(0, 0), (0, 0), (shift, plan.tail), (0, 0)], 0.0)
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            elif plan.tail:
                x = ttnn.pad(x, [(0, 0), (0, 0), (0, plan.tail), (0, 0)], 0.0)
            if plan.lead - shift:
                # a concat of tile-aligned pieces: no row changes its place inside a tile
                lead = ttnn.zeros(
                    (1, 1, plan.lead - shift, width),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                )
                x = ttnn.concat([lead, x], dim=2)
            # [1, rounds, 4096, width]: slice c of the 4096 dim is the block chip c owns in that round
            x = ttnn.reshape(x, [1, plan.rounds, self.layout.num_chips * self.layout.block, width])
            x = ttnn.mesh_partition(x, 2)
            x = ttnn.reshape(x, [1, 1, plan.slots, width])
            self._store(self._storage(layer, kind), user, x, plan)

    def write_prefill_host(
        self, layer: int, user: int, latent: torch.Tensor = None, index_key: torch.Tensor = None, start: int = 0
    ):
        """The same write from host tensors ([T, width] torch), striped on the host: one upload per chip."""
        for kind, x in ((LATENT, latent), (INDEX_KEY, index_key)):
            if x is None:
                continue
            assert x.dim() == 2 and x.shape[-1] == self.cfg.width(kind), f"{kind}: got {tuple(x.shape)}"
            plan = prefill_plan(self.layout, start, x.shape[0], self.cfg.capacity)
            stripes = host_stripes(self.layout, x.float(), start)  # [num_chips, plan.slots, width]
            # With nothing to blend the rows are packed in the cache dtype on the host.
            dtype = ttnn.bfloat16 if plan.lead else self.dtype
            self._store(self._storage(layer, kind), user, self._per_chip(stripes[:, None, None], dtype), plan)

    def _store(self, storage, user: int, x, plan: PrefillPlan):
        """x: per chip [1, 1, plan.slots, width] bf16, this chip's rows of the plan's rounds."""
        assert 0 <= user < self.cfg.max_users, f"user {user} of {self.cfg.max_users}"
        width, block = x.shape[-1], self.layout.block
        if plan.lead:
            # The first round also holds positions before ``start``: keep what the cache has there.
            keep = prefill_keep_mask(self.layout, plan).float().reshape(self.layout.num_chips, 1, 1, block, 1)
            old = ttnn.slice(storage, [user, 0, plan.first_slot, 0], [user + 1, 1, plan.first_slot + block, width])
            if old.dtype != ttnn.bfloat16:
                old = ttnn.typecast(old, ttnn.bfloat16)
            head = ttnn.add(
                ttnn.multiply(
                    ttnn.slice(x, [0, 0, 0, 0], [1, 1, block, width]), self._per_chip(1.0 - keep, ttnn.bfloat16)
                ),
                ttnn.multiply(old, self._per_chip(keep, ttnn.bfloat16)),
            )
            if plan.rounds == 1:
                x = head
            else:
                x = ttnn.concat([head, ttnn.slice(x, [0, 0, block, 0], [1, 1, plan.slots, width])], dim=2)
        if x.dtype != self.dtype:
            x = ttnn.typecast(x, self.dtype)
        ttnn.kv_cache.fill_cache_for_user_(storage, x, user, update_idx=plan.first_slot)

    # ----------------------------------------------------------------------------------------
    # write: decode
    # ----------------------------------------------------------------------------------------
    def append_indices(self, positions):
        """positions: one per user (None or -1: no token for that user this step). Returns the
        per-chip int32 [max_users] row indices write_decode takes; one tensor serves every layer of
        the step."""
        assert len(positions) == self.cfg.max_users, f"{len(positions)} positions for {self.cfg.max_users} users"
        table = append_index_table(self.layout, positions, self.cfg.capacity)
        return self._per_chip(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    def _token_memory_config(self, width):
        if width not in self._token_memory_configs:
            grid = self.mesh_device.compute_with_storage_grid_size()
            assert self.cfg.max_users <= grid.x * grid.y, f"{self.cfg.max_users} users need one core each"
            self._token_memory_configs[width] = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, width),
                core_grid=ttnn.num_cores_to_corerangeset(self.cfg.max_users, grid, row_wise=True),
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            )
        return self._token_memory_configs[width]

    def write_decode(self, layer: int, indices, latent=None, index_key=None):
        """Append one token per user.

        latent [1,1,max_users,latent_dim] and / or index_key [1,1,max_users,index_key_dim]: bf16 or
        fp32, whole and identical on every chip, row ``u`` being user ``u``'s new token. ``indices``
        is append_indices(positions). Only the chip that owns a user's position changes.
        """
        users = self.cfg.max_users
        for kind, x in ((LATENT, latent), (INDEX_KEY, index_key)):
            if x is None:
                continue
            width = self.cfg.width(kind)
            assert tuple(x.shape) == (1, 1, users, width), f"{kind}: got {tuple(x.shape)}"
            x = ttnn.reshape(x, [1, users, 1, width])  # one user per shard
            x = ttnn.to_memory_config(x, self._token_memory_config(width))
            ttnn.experimental.paged_update_cache(self._storage(layer, kind), x, update_idxs_tensor=indices)

    # ----------------------------------------------------------------------------------------
    # read
    # ----------------------------------------------------------------------------------------
    def stripe(self, layer: int, kind: str = LATENT, user: int | None = None, length: int | None = None):
        """This chip's rows, in the cache dtype.

        With no arguments past ``kind`` it is the storage itself, per chip [max_users, 1, slots, width]
        (not a copy: do not deallocate it). ``user`` narrows it to [1, 1, ., width]; ``length`` to the
        ``layout.slots(length)`` rows a context of that many positions can reach.
        """
        storage = self._storage(layer, kind)
        if user is None and length is None:
            return storage
        users = (0, self.cfg.max_users) if user is None else (user, user + 1)
        rows = self.cfg.slots if length is None else self.layout.slots(length)
        return ttnn.slice(storage, [users[0], 0, 0, 0], [users[1], 1, rows, self.cfg.width(kind)])

    def filled(self, lengths):
        """Which slots hold a position below a context length: fp32 0/1, per chip.

        lengths: an int -> [1, 1, 1, slots]; a list with one length per row -> [n, 1, 1, slots] (one
        row per user, shaped like ``stripe``); or an fp32 device tensor that broadcasts against
        [1, 1, 1, slots], e.g. [1, 1, S, 1] of ``query position + 1`` for a causal [1, 1, S, slots].
        """
        if isinstance(lengths, int):
            return ttnn.lt(self.slot_position, float(lengths))
        if isinstance(lengths, (list, tuple)):
            lengths = ttnn.from_torch(
                torch.tensor(lengths, dtype=torch.float32).reshape(-1, 1, 1, 1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
            )
        return ttnn.lt(self.slot_position, lengths)

    def locate(self, positions, user: int = 0):
        """Selected positions -> where this chip holds them.

        positions: fp32 [1, 1, S, K], the same on every chip: K picks for each of S queries (the
        merged top-k of the indexer; S = 1 at decode). Every entry must be a position inside the
        cache's capacity, also the entries the indexer flags as not valid.
        Returns (rows, mine): ``rows`` uint32 row major [S, K], the storage row of ``user`` each
        position has on this chip, and ``mine`` fp32 0/1 [1, 1, S, K], 1 on the one chip that owns
        the position. Where ``mine`` is 0 ``rows`` points at the user's slot 0, so ``gather`` stays in
        range and the caller masks the row. One result serves every layer that shares the selection.

        The arithmetic is locate_rows: kv_layout's, exact in fp32.
        """
        assert 0 <= user < self.cfg.max_users, f"user {user} of {self.cfg.max_users}"
        assert tuple(positions.shape)[:2] == (1, 1), f"positions: got {tuple(positions.shape)}"
        rows, mine = locate_rows(ttnn, positions, self.chip_id, self.layout, user * self.cfg.slots)
        rows = ttnn.typecast(rows, ttnn.uint32)
        rows = ttnn.reshape(ttnn.to_layout(rows, ttnn.ROW_MAJOR_LAYOUT), [positions.shape[-2], positions.shape[-1]])
        return rows, mine

    def gather(self, layer: int, rows, kind: str = LATENT):
        """rows: uint32 [S, K] storage rows per chip (from ``locate``). Returns bf16 [1, S, K, width]:
        on each chip, the cached rows at those indices.

        Costs a pass over the whole storage, not K rows: ttnn.embedding untilizes its TILE table to
        a row-major bf16 temporary (users x slots x width) before it reads (see the module docstring).
        """
        storage = self._storage(layer, kind)
        width = self.cfg.width(kind)
        table = ttnn.reshape(storage, [1, 1, self.cfg.max_users * self.cfg.slots, width])
        out = ttnn.embedding(rows, table, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(out, [1, rows.shape[-2], rows.shape[-1], width])

    # ----------------------------------------------------------------------------------------
    # host readback (tests, KV handoff)
    # ----------------------------------------------------------------------------------------
    def stripes_to_torch(self, layer: int, kind: str = LATENT, user: int | None = None) -> torch.Tensor:
        """[num_chips, max_users, slots, width] fp32 (or [num_chips, slots, width] for one user)."""
        shards = [ttnn.to_torch(s).float()[:, 0] for s in ttnn.get_device_tensors(self.stripe(layer, kind, user))]
        stripes = torch.stack(shards)
        return stripes if user is None else stripes[:, 0]

    def to_torch(self, layer: int, user: int, length: int, kind: str = LATENT) -> torch.Tensor:
        """The first ``length`` positions of ``user`` in natural order, [length, width] fp32."""
        return self.layout.merge(self.stripes_to_torch(layer, kind, user), length)
