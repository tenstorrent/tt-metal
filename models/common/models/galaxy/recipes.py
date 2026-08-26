# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Topology-owned WH Galaxy `(8, 4)` geometry, core sets, and tensor placements.

Everything here is model neutral. A dense 2D transformer supplies its
dimensions through :class:`GalaxyDenseGeometry` and receives the resolved
placement, program, and subdevice recipes that the hardware-qualified
Milestone A module tests established. Precision, provider conversion, and
checkpoint policy stay in the model packages.

The recipes are deliberately explicit: every memory config, program config,
and core range is resolved on the host before a module hot path runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import ttnn

GALAXY_MESH_SHAPE = (8, 4)
GALAXY_ROWS, GALAXY_COLUMNS = GALAXY_MESH_SHAPE
GALAXY_DEVICE_COUNT = GALAXY_ROWS * GALAXY_COLUMNS
GALAXY_PHYSICAL_BATCH = 32
GALAXY_USERS_PER_COLUMN = GALAXY_PHYSICAL_BATCH // GALAXY_COLUMNS
GALAXY_ARCHITECTURE = ttnn.device.Arch.WORMHOLE_B0
TILE = ttnn.TILE_SIZE

# The qualified decode ring uses 24 gather-in0 matmul cores plus one hop core.
RING_CORE_COUNT = 24
RING_ALIGNMENT = TILE * RING_CORE_COUNT

_RING_CORE_COORDS = (
    (6, 6),
    (6, 7),
    (6, 9),
    (6, 0),
    (6, 1),
    (6, 2),
    (6, 4),
    (6, 5),
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 9),
    (5, 0),
    (5, 1),
    (5, 2),
    (5, 4),
    (1, 4),
    (1, 5),
    (1, 9),
    (1, 0),
    (2, 0),
    (2, 4),
    (2, 5),
    (2, 9),
)
_RING_RECEIVER_COORDS = (
    (1, 9),
    (2, 9),
    (1, 0),
    (2, 0),
    (1, 4),
    (2, 4),
    (1, 5),
    (2, 5),
    (5, 0),
    (6, 0),
    (5, 9),
    (6, 9),
    (5, 1),
    (6, 1),
    (5, 7),
    (6, 7),
    (5, 6),
    (6, 6),
    (5, 2),
    (6, 2),
    (5, 4),
    (6, 4),
    (5, 5),
    (6, 5),
)
_HOP_CORE_COORDS = ((3, 6),)
_WORKER_CORE_RANGES = ((1, 0, 3, 9), (5, 0, 6, 9))
_TOPK_CORE_RANGES = ((1, 0, 3, 9),)
_PREFETCH_SENDER_COORDS = (
    (0, 9),
    (0, 0),
    (0, 4),
    (0, 5),
    (4, 0),
    (4, 9),
    (4, 1),
    (4, 7),
    (4, 6),
    (4, 2),
    (4, 4),
    (4, 5),
)


def core_points(coords: tuple[tuple[int, int], ...]) -> ttnn.CoreRangeSet:
    """Return one single-core range per coordinate, preserving ring order."""

    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*coord), ttnn.CoreCoord(*coord)) for coord in coords])


def core_ranges(*ranges: tuple[int, int, int, int]) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, y0, x1, y1 in ranges]
    )


def ring_cores() -> ttnn.CoreRangeSet:
    return core_points(_RING_CORE_COORDS)


def ring_receiver_cores() -> ttnn.CoreRangeSet:
    return core_points(_RING_RECEIVER_COORDS)


def ring_hop_cores() -> ttnn.CoreRangeSet:
    return core_points(_HOP_CORE_COORDS)


def worker_cores() -> ttnn.CoreRangeSet:
    """Return the decode worker subdevice envelope shared by every collective."""

    return core_ranges(*_WORKER_CORE_RANGES)


def topk_cores() -> ttnn.CoreRangeSet:
    return core_ranges(*_TOPK_CORE_RANGES)


def prefetch_sender_cores() -> tuple[ttnn.CoreCoord, ...]:
    return tuple(ttnn.CoreCoord(x, y) for x, y in _PREFETCH_SENDER_COORDS)


def validate_galaxy_mesh(name: str, mesh_device: Any) -> None:
    """Fail closed unless the mesh is exactly a 32-device Wormhole `(8, 4)`."""

    shape = tuple(mesh_device.shape)
    if shape != GALAXY_MESH_SHAPE:
        raise ValueError(f"{name} requires logical mesh shape {GALAXY_MESH_SHAPE}, got {shape}")
    if mesh_device.get_num_devices() != GALAXY_DEVICE_COUNT:
        raise ValueError(f"{name} requires exactly {GALAXY_DEVICE_COUNT} devices")
    if mesh_device.arch() != GALAXY_ARCHITECTURE:
        raise ValueError(f"{name} supports Wormhole only, got {mesh_device.arch()}")


def pad_ring_width(value: int) -> int:
    """Pad a per-device width to the ring matmul's 24-core tile alignment."""

    return math.ceil(value / RING_ALIGNMENT) * RING_ALIGNMENT


def pad_tiles(value: int) -> int:
    return math.ceil(value / TILE) * TILE


def galaxy_padded_vocab_size(vocab_size: int) -> int:
    """Return the minimal vocabulary width LMHead2D and Sampling2D both accept."""

    multiple = GALAXY_ROWS * TILE
    return math.ceil(vocab_size / multiple) * multiple


def width_sharded_memory_config(width: int, cores: ttnn.CoreRangeSet, *, rows: int = TILE) -> ttnn.MemoryConfig:
    """Width-shard `width` columns evenly over `cores`."""

    count = cores.num_cores()
    if width % count:
        raise ValueError(f"width {width} is not divisible by {count} cores")
    return ttnn.create_sharded_memory_config(
        shape=(rows, width // count),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def height_sharded_memory_config(cores: ttnn.CoreRangeSet, width: int, *, rows: int = TILE) -> ttnn.MemoryConfig:
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, (rows, width), ttnn.ShardOrientation.ROW_MAJOR),
    )


def dram_sharded_weight_memory_config(mesh_device: Any, local_k: int, local_n: int) -> ttnn.MemoryConfig:
    """Return the DRAM width-sharded weight placement used by ring matmuls."""

    dram_width = mesh_device.dram_grid_size().x
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_width - 1, 0))})
    padded_n = pad_ring_width(local_n)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, (local_k, padded_n // dram_width), ttnn.ShardOrientation.ROW_MAJOR),
    )


def ring_matmul_program_config(local_k: int, padded_local_n: int) -> Any:
    """Return the qualified 24-core gather-in0 ring matmul program config."""

    out_block_w = padded_local_n // RING_CORE_COUNT // TILE
    out_subblock_w = min(8, out_block_w)
    while out_block_w % out_subblock_w:
        out_subblock_w -= 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 3),
        in0_block_w=local_k // RING_CORE_COUNT // TILE,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=ring_hop_cores(),
        num_global_cb_receivers=2,
        untilize_out=False,
    )


def dense_matmul_worker_rectangle(height: int) -> ttnn.CoreRangeSet:
    """Return the widest worker rectangle a 2D mcast matmul may anchor in.

    ``MatmulMultiCoreReuseMultiCastProgramConfig`` lays its work grid out as one
    rectangle from ``allowed_worker_cores.bounding_box().start_coord``, so the
    cores it can legally use are the largest rectangle that starts at the worker
    envelope's origin and never leaves it. On WH Galaxy that is three columns
    wide - ``worker_cores()`` is ``x=1..3`` and ``x=5..6``, split by the ``x=4``
    prefetch sender column - and the width is searched for rather than named so
    a change to the envelope moves this with it.
    """

    workers = worker_cores()
    start = workers.bounding_box().start
    width = 0
    while True:
        candidate = ttnn.CoreRangeSet({ttnn.CoreRange(start, ttnn.CoreCoord(start.x + width, start.y + height - 1))})
        if candidate.subtract(workers).num_cores():
            break
        width += 1
    if width == 0:
        raise ValueError(f"no worker rectangle of height {height} exists at {start}")
    return ttnn.CoreRangeSet({ttnn.CoreRange(start, ttnn.CoreCoord(start.x + width - 1, start.y + height - 1))})


def dense_matmul_program_config(rows: int, local_k: int, local_n: int) -> Any:
    """Return the qualified interleaved multicast matmul program config.

    The work grid is confined to ``worker_cores()`` with ``allowed_worker_cores``.
    Without it the op anchors at ``(0, 0)`` and spans the full seven-column
    compute grid, which reaches the ``x=0`` and ``x=4`` prefetch sender columns
    and the eight cores that belong to no sub-device at all, and the decode QKV
    matmul then aborts with

        TT_FATAL ... Illegal kernel placement for
                     bmm_large_block_zm_fused_bias_activation,
                     Kernels cannot be placed on dispatch cores!

    This is Milestone A limitation L3. Milestone A recorded it against exactly
    this ``(7, 1)`` grid and called it terminal; the Milestone B recipes moved
    the *MLP* to the ring/``gather_in0`` form but left both attention matmuls on
    this dense config, so L3 was still live on first silicon. ``ttnn`` grew
    ``allowed_worker_cores`` for this, deprecating
    ``compute_with_storage_grid_size``, and warns when a config that supports the
    field leaves it unset - so this is a config fix, not a new mechanism.

    Cost: three columns instead of seven, so these two matmuls get 3/7 of the
    cores they would otherwise use. That is a correctness-first trade. Moving
    attention to the 24-core ring form, as the MLP already is, is the
    performance answer and is not this job's scope.
    """

    grid_y = min(4, max(1, math.ceil(rows / TILE)))
    allowed = dense_matmul_worker_rectangle(grid_y)
    grid_x = allowed.bounding_box().end.x - allowed.bounding_box().start.x + 1
    m_tiles = max(1, math.ceil(rows / TILE))
    n_tiles = local_n // TILE
    k_tiles = local_k // TILE
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=math.gcd(k_tiles, 8),
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=math.ceil(m_tiles / grid_y),
        per_core_N=math.ceil(n_tiles / grid_x),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
        allowed_worker_cores=allowed,
    )


def sdpa_program_config(
    sequence_length: int, *, decode: bool, sub_core_grids: ttnn.CoreRangeSet | None = None
) -> ttnn.SDPAProgramConfig:
    """Return the qualified decode/prefill SDPA geometry."""

    if decode:
        q_chunk_size = k_chunk_size = 0
    elif sequence_length < 2048:
        q_chunk_size = k_chunk_size = 64
    else:
        q_chunk_size, k_chunk_size = 256, 512
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 4) if decode else (7, 10),
        sub_core_grids=sub_core_grids,
        exp_approx_mode=False,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )


def chunked_sdpa_program_config(
    *, sub_core_grids: ttnn.CoreRangeSet | None = None, chunk_alignment: int = 128
) -> ttnn.SDPAProgramConfig:
    """Return the prefix-cached/chunked prefill SDPA geometry.

    Chunked SDPA reads the paged KV cache, so its chunks are sized by the
    chunk alignment the page table is built around rather than by the request's
    sequence length.
    """

    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(7, 10),
        sub_core_grids=sub_core_grids,
        exp_approx_mode=False,
        q_chunk_size=chunk_alignment,
        k_chunk_size=chunk_alignment,
    )


def compute_kernel_config(
    *,
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2,
    math_approx_mode: bool = False,
    fp32_dest_acc_en: bool = False,
    packer_l1_acc: bool = False,
) -> ttnn.WormholeComputeKernelConfig:
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=math_approx_mode,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )


@dataclass(frozen=True)
class GalaxyDenseGeometry:
    """Static geometry of one dense 2D transformer on WH Galaxy `(8, 4)`.

    Hidden state is sharded over the four mesh columns, attention heads and the
    MLP hidden dimension over the eight mesh rows, and the 32 physical users
    over the four columns as eight users per column.
    """

    dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    max_seq_len: int
    max_batch_size: int = GALAXY_PHYSICAL_BATCH
    prefill_sequence_lengths: tuple[int, ...] = (128,)
    #: Per-row lengths served by the physical-batch-32 concatenated prefill.
    #: Each entry costs one more set of collective resources, keyed by its
    #: ``32 * length`` token count, so the default is empty.
    batched_prefill_sequence_lengths: tuple[int, ...] = ()
    chunk_alignment: int = 128

    def __post_init__(self) -> None:
        object.__setattr__(self, "prefill_sequence_lengths", tuple(sorted(set(self.prefill_sequence_lengths))))
        object.__setattr__(
            self, "batched_prefill_sequence_lengths", tuple(sorted(set(self.batched_prefill_sequence_lengths)))
        )
        if self.max_batch_size != GALAXY_PHYSICAL_BATCH:
            raise ValueError("Galaxy dense geometry requires physical batch 32")
        if min(self.dim, self.hidden_dim, self.n_heads, self.n_kv_heads, self.head_dim, self.vocab_size) <= 0:
            raise ValueError("Galaxy dense geometry dimensions must be positive")
        if self.dim % (GALAXY_COLUMNS * TILE) or self.dim % GALAXY_ROWS:
            raise ValueError(f"dim {self.dim} must shard over 4 columns (tile aligned) and 8 rows")
        if self.local_dim % (GALAXY_ROWS * TILE):
            raise ValueError(
                f"column-local dim {self.local_dim} must fill the two-wide distributed norm grid "
                f"(multiple of {GALAXY_ROWS * TILE})"
            )
        if self.hidden_dim % (GALAXY_ROWS * TILE):
            raise ValueError(f"hidden_dim {self.hidden_dim} must shard over 8 rows with tile alignment")
        if self.n_heads % GALAXY_ROWS or self.n_kv_heads % GALAXY_ROWS:
            raise ValueError("n_heads and n_kv_heads must shard over the 8 mesh rows")
        if self.head_dim % TILE:
            raise ValueError(f"head_dim {self.head_dim} must be tile aligned")
        if self.attention_dim % (GALAXY_ROWS * TILE):
            raise ValueError("n_heads * head_dim must shard over 8 rows with tile alignment")
        if self.max_seq_len <= 0 or self.max_seq_len % self.chunk_alignment:
            raise ValueError(f"max_seq_len {self.max_seq_len} must be a positive multiple of {self.chunk_alignment}")
        if not self.prefill_sequence_lengths:
            raise ValueError("at least one prefill sequence length is required")
        for length in self.prefill_sequence_lengths:
            if length <= 0 or length % self.chunk_alignment:
                raise ValueError(f"prefill length {length} must be a positive multiple of {self.chunk_alignment}")
            if length > self.max_seq_len:
                raise ValueError(f"prefill length {length} exceeds max_seq_len {self.max_seq_len}")
        for length in self.batched_prefill_sequence_lengths:
            if length <= 0 or length % self.chunk_alignment:
                raise ValueError(
                    f"batched prefill length {length} must be a positive multiple of {self.chunk_alignment}"
                )
            if length > self.max_seq_len:
                raise ValueError(f"batched prefill length {length} exceeds max_seq_len {self.max_seq_len}")

    # Hidden-state partitioning

    @property
    def local_dim(self) -> int:
        """Column-local hidden width, i.e. the residual-stream shard width."""

        return self.dim // GALAXY_COLUMNS

    @property
    def row_dim(self) -> int:
        return self.dim // GALAXY_ROWS

    @property
    def local_hidden_dim(self) -> int:
        return self.hidden_dim // GALAXY_ROWS

    # Attention partitioning

    @property
    def attention_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def local_attention_dim(self) -> int:
        return self.attention_dim // GALAXY_ROWS

    @property
    def qkv_size(self) -> int:
        return self.head_dim * (self.n_heads + 2 * self.n_kv_heads)

    @property
    def local_qkv_size(self) -> int:
        return self.qkv_size // GALAXY_ROWS

    @property
    def local_heads(self) -> int:
        return self.n_heads // GALAXY_ROWS

    @property
    def local_kv_heads(self) -> int:
        return self.n_kv_heads // GALAXY_ROWS

    @property
    def users_per_column(self) -> int:
        return GALAXY_USERS_PER_COLUMN

    # Vocabulary partitioning

    @property
    def padded_vocab_size(self) -> int:
        return galaxy_padded_vocab_size(self.vocab_size)

    @property
    def local_padded_vocab_size(self) -> int:
        return self.padded_vocab_size // GALAXY_ROWS

    # Derived collective geometry

    @property
    def decode_reduce_scatter_width(self) -> int:
        """Logical width of the MLP W1/W3 axis-1 reduce-scatter result.

        The ring matmul emits a 24-core aligned padded width. When that padding
        is not a whole number of shards the collective scatters the padded
        width; otherwise it scatters the logical width. This is the width the
        resource key is derived from, so it must match what TTNN reports for
        the scattered tensor.
        """

        padded = pad_ring_width(self.local_hidden_dim)
        shard = padded // RING_CORE_COUNT
        scattered = padded if self.local_hidden_dim % shard else self.local_hidden_dim
        return scattered // GALAXY_COLUMNS

    @property
    def decode_reduce_scatter_padded_width(self) -> int:
        """Physical width of the scattered W1/W3 placement, padded for the ring."""

        return pad_ring_width(self.local_hidden_dim) // GALAXY_COLUMNS

    @property
    def batched_prefill_token_counts(self) -> tuple[int, ...]:
        """Return the total token count each batched prefill recipe processes.

        Concatenated prefill runs the physical batch as one long token stream,
        so every collective it issues has the geometry of a single-row prefill
        of ``32 * per_row_length`` tokens.
        """

        return tuple(GALAXY_PHYSICAL_BATCH * length for length in self.batched_prefill_sequence_lengths)

    @property
    def collective_prefill_token_counts(self) -> tuple[int, ...]:
        """Return every token count that needs prefill collective resources."""

        return tuple(sorted(set(self.prefill_sequence_lengths) | set(self.batched_prefill_token_counts)))

    def prefill_leading_shape(self, sequence_length: int) -> tuple[int, int, int]:
        """Return the leading MLP prefill shape after its long-sequence reshape."""

        cutoff = 1024
        if sequence_length >= cutoff:
            if sequence_length % cutoff:
                raise ValueError(f"prefill length {sequence_length} must be a multiple of {cutoff}")
            return (1, sequence_length // cutoff, cutoff)
        return (1, 1, sequence_length)


@dataclass(frozen=True)
class GalaxyDecodePlacements:
    """Resolved decode placements shared by every module in one layer."""

    residual_memcfg: ttnn.MemoryConfig
    attention_input_memcfg: ttnn.MemoryConfig
    attention_qkv_output_memcfg: ttnn.MemoryConfig
    attention_heads_memcfg: ttnn.MemoryConfig
    attention_kv_memcfg: ttnn.MemoryConfig
    attention_sdpa_output_memcfg: ttnn.MemoryConfig
    attention_gather_users_memcfg: ttnn.MemoryConfig
    attention_concat_memcfg: ttnn.MemoryConfig
    attention_wo_output_memcfg: ttnn.MemoryConfig
    attention_qkv_collective_input_memcfg: ttnn.MemoryConfig
    attention_qkv_reduced_memcfg: ttnn.MemoryConfig
    attention_qkv_scratch_memcfg: ttnn.MemoryConfig
    attention_qkv_program_config: Any
    attention_wo_program_config: Any
    attention_sdpa_program_config: Any
    attention_sdpa_cores: ttnn.CoreRangeSet
    mlp_input_memcfg: ttnn.MemoryConfig
    mlp_w2_input_memcfg: ttnn.MemoryConfig
    mlp_w1_w3_output_memcfg: ttnn.MemoryConfig
    mlp_w2_output_memcfg: ttnn.MemoryConfig
    mlp_w1_w3_program_config: Any
    mlp_w2_program_config: Any
    mlp_reduce_scatter_memcfg: ttnn.MemoryConfig
    all_reduce_buffer_memcfg: ttnn.MemoryConfig
    worker_cores: ttnn.CoreRangeSet
    ring_cores: ttnn.CoreRangeSet


@dataclass(frozen=True)
class GalaxyPrefillPlacements:
    """Resolved prefill placements. Prefill runs interleaved in DRAM."""

    residual_memcfg: ttnn.MemoryConfig
    activation_memcfg: ttnn.MemoryConfig
    sdpa_cores: ttnn.CoreRangeSet
    attention_program_configs: dict[int, Any]
    attention_wo_program_configs: dict[int, Any]
    attention_sdpa_program_configs: dict[int, Any]
    # Concatenated physical-batch-32 prefill. Keyed by per-row length: the
    # projections see 32 rows of tokens at once, while SDPA still runs one
    # causal sequence per row.
    batched_attention_program_configs: dict[int, Any] = field(default_factory=dict)
    batched_attention_wo_program_configs: dict[int, Any] = field(default_factory=dict)
    batched_attention_sdpa_program_configs: dict[int, Any] = field(default_factory=dict)
    #: Prefix-cached/chunked prefill reads the paged cache, so one chunk-aligned
    #: SDPA geometry serves every sequence length.
    chunked_sdpa_program_config: Any = None


def _subgrid_cores(count: int, *, row_wise: bool) -> ttnn.CoreRangeSet:
    return ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(1, 0), count, worker_cores(), row_wise=row_wise
    )


def distributed_norm_decode_memory_config(geometry: GalaxyDenseGeometry) -> ttnn.MemoryConfig:
    """Return the residual-stream placement, identical to RMSNorm2D's default.

    ``RMSNorm2D`` resolves its decode input, residual, and output placement to
    a two-wide width-sharded grid whose origin is ``x=2``; the canonical
    column-dispatch Galaxy layout reserves ``x=0..1``. The attention and MLP
    decode outputs are placed here so the fused residual norm consumes them
    without a relocation.

    The fused-statistics buffer belongs on the *first core of this grid*, not on
    a core of its own: ``fused_rms_minimal`` creates its stats circular buffer
    there and binds it to the stats tensor's L1 address (Milestone A defect D1,
    enforced by ``RMSNorm2D._require_fused_stats_placement``). Derive it with
    :func:`distributed_norm_stats_memory_config` rather than naming a core.

    The grid keeps four width tiles per norm core, so the shard is always 128
    columns wide and the grid is ``local_dim / 256`` rows tall.
    """

    grid_height = geometry.local_dim // (GALAXY_ROWS * TILE)
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, grid_height - 1))})
    return width_sharded_memory_config(geometry.local_dim, cores)


def distributed_norm_stats_memory_config(residual_memcfg: ttnn.MemoryConfig) -> ttnn.MemoryConfig:
    """Return the fused-statistics placement implied by a norm-input placement.

    The origin is read off `residual_memcfg` rather than named, because the two
    are not independent: `fused_rms_minimal` builds its stats circular buffer on
    the first core of the norm input shard grid and binds it to this tensor's L1
    address, so any other placement makes the kernel reduce unrelated L1. That
    was Milestone A defect D1, and `RMSNorm2D._require_fused_stats_placement`
    rejects a mismatch outright. Deriving it here is what keeps the persistent
    collective buffer this recipe sizes and the placement `RMSNorm2D` resolves
    for itself from ever disagreeing.
    """

    shard_spec = getattr(residual_memcfg, "shard_spec", None)
    if shard_spec is None:
        raise ValueError("distributed norm decode input must be L1 width-sharded to place fused statistics")
    origin = shard_spec.grid.bounding_box().start
    return ttnn.create_sharded_memory_config(
        shape=(1, 1, TILE, TILE * GALAXY_COLUMNS),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(origin, origin)}),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )


def resolve_galaxy_decode_placements(geometry: GalaxyDenseGeometry, mesh_device: Any) -> GalaxyDecodePlacements:
    """Resolve every decode placement the qualified Milestone A recipes use."""

    validate_galaxy_mesh("Galaxy decode placements", mesh_device)
    workers = worker_cores()
    ring = ring_cores()
    receivers = ring_receiver_cores()

    qkv_core_count = geometry.local_qkv_size // geometry.head_dim
    qkv_cores = _subgrid_cores(qkv_core_count, row_wise=False)
    head_cores = _subgrid_cores(GALAXY_PHYSICAL_BATCH, row_wise=False)
    sdpa_cores = _subgrid_cores(GALAXY_PHYSICAL_BATCH, row_wise=True)
    gather_user_cores = _subgrid_cores(GALAXY_PHYSICAL_BATCH, row_wise=True)
    kv_cores = _subgrid_cores(geometry.users_per_column, row_wise=False)
    sdpa_output_cores = _subgrid_cores(geometry.users_per_column, row_wise=True)

    padded_local_dim = pad_ring_width(geometry.local_dim)
    padded_local_hidden = pad_ring_width(geometry.local_hidden_dim)
    # Placements carry the padded physical width; resource keys carry the
    # logical width TTNN reports for the same tensor.
    reduce_scatter_cores = _subgrid_cores(geometry.decode_reduce_scatter_padded_width // TILE, row_wise=True)

    return GalaxyDecodePlacements(
        residual_memcfg=distributed_norm_decode_memory_config(geometry),
        # The qualified Attention2D decode recipe consumes an interleaved DRAM
        # activation and reduces with the fused create-QKV-heads collective.
        attention_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        attention_qkv_output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        attention_heads_memcfg=height_sharded_memory_config(head_cores, geometry.head_dim),
        attention_kv_memcfg=height_sharded_memory_config(kv_cores, geometry.head_dim),
        attention_sdpa_output_memcfg=height_sharded_memory_config(sdpa_output_cores, geometry.head_dim),
        attention_gather_users_memcfg=height_sharded_memory_config(gather_user_cores, geometry.head_dim),
        attention_concat_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        attention_wo_output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        attention_qkv_collective_input_memcfg=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ring,
                (TILE, pad_ring_width(geometry.local_qkv_size) // RING_CORE_COUNT),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        # The reduced fused QKV projection is one head-dimension column per
        # local head core.
        attention_qkv_reduced_memcfg=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(qkv_cores, (TILE, geometry.head_dim), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        # The fused collective gathers all four column shards of the local fused
        # QKV projection into one scratch tensor, so each of the local head
        # cores owns four head-dimension columns.
        attention_qkv_scratch_memcfg=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(qkv_cores, (TILE, GALAXY_COLUMNS * geometry.head_dim), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        attention_qkv_program_config=dense_matmul_program_config(
            GALAXY_PHYSICAL_BATCH, geometry.local_dim, geometry.local_qkv_size
        ),
        attention_wo_program_config=dense_matmul_program_config(
            GALAXY_PHYSICAL_BATCH, geometry.local_attention_dim, geometry.local_dim
        ),
        attention_sdpa_program_config=sdpa_program_config(geometry.max_seq_len, decode=True, sub_core_grids=sdpa_cores),
        attention_sdpa_cores=sdpa_cores,
        mlp_input_memcfg=width_sharded_memory_config(padded_local_dim, ring),
        mlp_w2_input_memcfg=width_sharded_memory_config(padded_local_hidden, ring),
        mlp_w1_w3_output_memcfg=width_sharded_memory_config(padded_local_hidden, receivers),
        mlp_w2_output_memcfg=width_sharded_memory_config(padded_local_dim, receivers),
        mlp_w1_w3_program_config=ring_matmul_program_config(geometry.local_dim, padded_local_hidden),
        mlp_w2_program_config=ring_matmul_program_config(geometry.local_hidden_dim, padded_local_dim),
        mlp_reduce_scatter_memcfg=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(reduce_scatter_cores, (TILE, TILE), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        all_reduce_buffer_memcfg=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(workers, (TILE, 1024), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        worker_cores=workers,
        ring_cores=ring,
    )


def resolve_galaxy_prefill_placements(geometry: GalaxyDenseGeometry, mesh_device: Any) -> GalaxyPrefillPlacements:
    """Resolve the interleaved DRAM prefill recipes for every sequence length."""

    validate_galaxy_mesh("Galaxy prefill placements", mesh_device)
    workers = worker_cores()
    return GalaxyPrefillPlacements(
        residual_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        activation_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        sdpa_cores=workers,
        attention_program_configs={
            length: dense_matmul_program_config(length, geometry.local_dim, geometry.local_qkv_size)
            for length in geometry.prefill_sequence_lengths
        },
        attention_wo_program_configs={
            length: dense_matmul_program_config(length, geometry.local_attention_dim, geometry.local_dim)
            for length in geometry.prefill_sequence_lengths
        },
        attention_sdpa_program_configs={
            length: sdpa_program_config(length, decode=False, sub_core_grids=workers)
            for length in geometry.prefill_sequence_lengths
        },
        batched_attention_program_configs={
            length: dense_matmul_program_config(
                GALAXY_PHYSICAL_BATCH * length, geometry.local_dim, geometry.local_qkv_size
            )
            for length in geometry.batched_prefill_sequence_lengths
        },
        batched_attention_wo_program_configs={
            length: dense_matmul_program_config(
                GALAXY_PHYSICAL_BATCH * length, geometry.local_attention_dim, geometry.local_dim
            )
            for length in geometry.batched_prefill_sequence_lengths
        },
        batched_attention_sdpa_program_configs={
            length: sdpa_program_config(length, decode=False, sub_core_grids=workers)
            for length in geometry.batched_prefill_sequence_lengths
        },
        chunked_sdpa_program_config=chunked_sdpa_program_config(
            sub_core_grids=workers, chunk_alignment=geometry.chunk_alignment
        ),
    )


def galaxy_prefill_mode_plan_cores(mesh_device: Any) -> ttnn.CoreRangeSet:
    """Return the full compute grid used by the single prefill subdevice."""

    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})


def sampling_core_grids() -> tuple[ttnn.CoreRangeSet, ttnn.CoreRangeSet, ttnn.CoreCoord]:
    """Return the qualified Sampling2D `(sub_core_grids, topk grid, start core)`."""

    return worker_cores(), topk_cores(), ttnn.CoreCoord(1, 0)


def rope_core_grids(mesh_device: Any, *, use_qk_fused: bool) -> tuple[Any, ttnn.CoreRangeSet]:
    """Return the RotarySetup2D `(core_grid, batch_grid)` for Galaxy decode.

    ``batch_grid`` carries the decode cos/sin shards, so it must lie inside the
    worker sub-device like every other decode placement here. Taking the first
    ``rows`` cores of the *whole* compute grid instead puts shards on ``x=0`` and
    ``x=4`` - the two prefetch sender columns - and on a core outside every
    sub-device. ``ttnn.embedding`` then builds the decode tables on cores the
    loaded decode sub-device manager does not own and aborts with

        TT_FATAL ... Kernel group cores do not match sub device cores
                     for programmable core type TENSIX

    which is the same class of defect as the fused-norm statistics placement
    (Milestone A D1/C1): a grid named independently of the partition that has to
    contain it. ``_subgrid_cores`` is the qualified helper the attention KV, SDPA
    and reduce-scatter placements already use; it anchors at the first worker
    core ``(1, 0)`` and never leaves ``worker_cores()``.
    """

    validate_galaxy_mesh("Galaxy RoPE", mesh_device)
    core_grid = mesh_device.compute_with_storage_grid_size()
    rows = GALAXY_USERS_PER_COLUMN * (2 if use_qk_fused else 1)
    batch_grid = _subgrid_cores(rows, row_wise=True)
    return core_grid, batch_grid
