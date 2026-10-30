# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""DRAM-sharded decode matmul configuration shared by the device modules.

Every decode ``ttnn.linear`` here multiplies one padded tile row of activations by a resident weight; the DRAM-sharded
program streams each weight shard from its own bank into worker cores next to it. Upload the weight with
:func:`dram_sharded_weight_memory_config`, move the activation to the config from :func:`dram_sharded_matmul_configs`,
and call ``ttnn.linear`` with the returned program config and ``memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG``;
both calls of one linear take the same ``num_workers_per_dram_bank``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping

import ttnn


def _dram_bank_count(mesh_device) -> int:
    grid = mesh_device.dram_grid_size()
    if grid.y != 1:
        raise ValueError(f"DRAM weight sharding expects an Nx1 bank grid, got {grid.x}x{grid.y}")
    return grid.x


# Readers per DRAM bank for the projections in ``TWO_WORKER_PROJECTIONS`` (Blackhole admits two; bitwise equal to one).
# A bank's shard holds whole tiles per reader, which widens only the GDN input projection (544 -> 576 columns per bank).
DEFAULT_WORKERS_PER_DRAM_BANK = 2
WORKERS_PER_DRAM_BANK = (1, 2)
# (K, N) -> activation storage cores of the decode linears that run two readers per bank: the GDN input (2560 x 4160)
# and output (1536 x 2560) projections, the QSA query-gate (2560 x 3072) and output (1536 x 2560) projections, the
# two LM-head chunk widths (2560 x 8192 and 7040).  The K/V/index projections, the router, the gated-residual linears
# and the shared expert keep one reader.
TWO_WORKER_PROJECTIONS: dict[tuple[int, int], int] = {
    (2560, 4160): 8,
    (1536, 2560): 16,
    (2560, 3072): 8,
    (2560, 8192): 40,
    (2560, 7040): 40,
}
# Decode linears, by (K, N), that run tt-metal's multi-core DRAM-sharded pipeline (``use_multicore_pipeline``) on
# PIPELINE_WORKERS_PER_DRAM_BANK cores per bank: the LM-head chunks and the final mixer's up projection. The per-layer
# projections keep the single-reader program, because on the pipeline their served greedy streams change.
PIPELINE_WORKERS_PER_DRAM_BANK = 3
PIPELINE_PROJECTIONS: frozenset[tuple[int, int]] = frozenset({(2560, 8192), (2560, 7040), (320, 2560)})


def validate_decode_dram_workers(value) -> int:
    if type(value) is not int or value not in WORKERS_PER_DRAM_BANK:
        raise ValueError(f"decode DRAM workers per bank must be one of {WORKERS_PER_DRAM_BANK}, got {value!r}")
    return value


# ``bf8`` (default) | ``bf16`` | ``bf4``: the dtype of the resident dense decode matmul weights (GDN, QSA, shared
# expert, LM head, final mixer, MTP fc); ``QWEN38_DENSE_WEIGHT_DTYPE_<M>`` overrides one module. Those matmuls run
# HiFi4 / HiFi2 / LoFi for BF16 / BFP8 / BFP4 with fp32 accumulation. The embedding, router, norms, GDN conv taps /
# dt_bias / A_log, PLE tables and GR mixers stay BF16 (the fused gr_read uses BF16 circular buffers). Converted
# tensorbins carry the dtype tag in their name, so caches of different dtypes never collide.
DENSE_DTYPE_ENV = "QWEN38_DENSE_WEIGHT_DTYPE"
DEFAULT_DENSE_DTYPE_NAME = "bf8"
DENSE_MODULES = ("gdn", "qsa", "shared_expert", "lm_head", "final_mixer", "mtp")
DENSE_DTYPE_NAMES: dict[str, Any] = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b}
DENSE_DTYPE_TAGS: dict[Any, str] = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bf8b", ttnn.bfloat4_b: "bf4b"}
DENSE_MATH_FIDELITY_NAMES: dict[Any, str] = {ttnn.bfloat16: "HiFi4", ttnn.bfloat8_b: "HiFi2", ttnn.bfloat4_b: "LoFi"}
# One 32x32 tile: 1024 bf16 values; 1024 one-byte mantissas + 64 shared exponents; 512 bytes of 4-bit mantissas + 64.
DENSE_TILE_BYTES: dict[Any, int] = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
# Dtypes whose one- and two-reader programs were checked bitwise equal on device for every TWO_WORKER_PROJECTIONS shape
# (tools/qualify_dense_two_readers.py); any other dtype runs one reader everywhere.
TWO_READER_QUALIFIED_DTYPES: frozenset = frozenset({ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b})


def parse_dense_weight_dtype(raw: str, *, source: str = DENSE_DTYPE_ENV):
    key = raw.strip().lower()
    if key not in DENSE_DTYPE_NAMES:
        raise ValueError(f"{source} must be one of {tuple(DENSE_DTYPE_NAMES)}, got {raw!r}")
    return DENSE_DTYPE_NAMES[key]


def dense_dtype_tag(dtype) -> str:
    """``bf16`` / ``bf8b`` / ``bf4b``; any other dtype is refused."""

    if dtype not in DENSE_DTYPE_TAGS:
        raise ValueError(f"dense weight dtype must be one of {tuple(DENSE_DTYPE_TAGS)}, got {dtype!r}")
    return DENSE_DTYPE_TAGS[dtype]


def dense_math_fidelity_name(dtype) -> str:
    """The ``ttnn.MathFidelity`` member a matmul on ``dtype`` weights runs with (HiFi4 / HiFi2 / LoFi)."""

    dense_dtype_tag(dtype)
    return DENSE_MATH_FIDELITY_NAMES[dtype]


def dense_weight_name(name: str, dtype) -> str:
    """The cache file name of a converted matmul weight: ``name`` itself for BF16 (the production tensorbins), else
    ``name`` plus the dtype tag, so a bf8 / bf4 tensorbin never shadows or overwrites a bf16 one."""

    tag = dense_dtype_tag(dtype)
    return name if dtype == ttnn.bfloat16 else f"{name}.{tag}"


@dataclass(frozen=True)
class DenseWeightPlan:
    """Per module (DENSE_MODULES) the resident dense matmul weights' dtype."""

    dtypes: Mapping[str, Any]

    def __post_init__(self) -> None:
        if tuple(self.dtypes) != DENSE_MODULES:
            raise ValueError(f"dense weight plan must name {DENSE_MODULES} in order, got {tuple(self.dtypes)}")
        for dtype in self.dtypes.values():
            dense_dtype_tag(dtype)

    def dtype(self, module: str):
        return self.dtypes[module]

    def tag(self, module: str) -> str:
        return dense_dtype_tag(self.dtypes[module])

    @property
    def name(self) -> str:
        """``bf16`` / ``bf8`` / ``bf4`` when every module agrees, else ``mixed``."""

        names = {next(n for n, d in DENSE_DTYPE_NAMES.items() if d == dtype) for dtype in self.dtypes.values()}
        return names.pop() if len(names) == 1 else "mixed"

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "modules": {
                module: {"dtype": self.tag(module), "math_fidelity": dense_math_fidelity_name(dtype)}
                for module, dtype in self.dtypes.items()
            },
            "kept_bf16": [
                "embedding table",
                "router mlp.gate",
                "norms",
                "gdn conv taps / dt_bias / A_log",
                "ple",
                "gr mixers (the fused gr_read program streams BF16 tiles)",
            ],
        }


def default_dense_weight_plan(environ: Mapping[str, str] | None = None) -> DenseWeightPlan:
    """``QWEN38_DENSE_WEIGHT_DTYPE`` (bf8 when unset) per module, ``QWEN38_DENSE_WEIGHT_DTYPE_<MODULE>`` over it."""

    env = os.environ if environ is None else environ
    base = parse_dense_weight_dtype(env.get(DENSE_DTYPE_ENV, "").strip() or DEFAULT_DENSE_DTYPE_NAME)
    dtypes = {}
    for module in DENSE_MODULES:
        key = f"{DENSE_DTYPE_ENV}_{module.upper()}"
        raw = env.get(key, "").strip()
        dtypes[module] = parse_dense_weight_dtype(raw, source=key) if raw else base
    return DenseWeightPlan(dtypes)


def mesh_dram_bank_worker_signatures(mesh_device) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    """Per mesh coordinate, the DRAM bank -> worker core order the DRAM-sharded matmul reads in1 with (NOC 0)."""

    rows, columns = (int(extent) for extent in mesh_device.shape)
    signatures: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for row in range(rows):
        for column in range(columns):
            assignment = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(
                ttnn.NOC.RISCV_0_default, ttnn.MeshCoordinate(row, column)
            )
            signatures[(row, column)] = tuple(
                (int(assignment[bank].x), int(assignment[bank].y)) for bank in sorted(assignment)
            )
    return signatures


def mesh_device_geometries(mesh_device) -> tuple[tuple[int, int], ...]:
    """Per mesh coordinate (row-major): (DRAM bank count, max y + 1 of the optimal bank reader cores).

    Harvesting moves columns, not rows, so dies harvested in different columns share this geometry; tt-metal's own
    geometry check refuses differing columns at the first program build."""

    signatures = mesh_dram_bank_worker_signatures(mesh_device)
    out = []
    for coordinate in sorted(signatures):
        cores = signatures[coordinate]
        out.append((len(cores), max(y for _, y in cores) + 1))
    return tuple(out)


def decode_dram_workers_placement(mesh_device) -> str:
    """Where the two-reader programs' shared bank -> worker placement comes from: ``"identical"`` when every device
    reports the same optimal assignment, else ``"reference (r, c); other optimal readers at [...]"`` (tt-metal places
    the reference device's assignment on every device; the named devices read those banks from a further worker)."""

    signatures = mesh_dram_bank_worker_signatures(mesh_device)
    if len(set(signatures.values())) == 1:
        return "identical"
    reference_coordinate = min(signatures)
    differing = sorted(
        coordinate for coordinate, signature in signatures.items() if signature != signatures[reference_coordinate]
    )
    return f"reference {reference_coordinate}; other optimal readers at {differing}"


def qualify_decode_dram_workers(mesh_device, requested: int) -> tuple[int, str | None]:
    """The readers per DRAM bank this mesh admits: ``requested``, or 1 with the reason.

    tt-metal places the reference device's bank -> worker assignment on every device; readers address shards by bank
    id, so this is only a locality choice and stays bitwise. One reader remains when a device's bank count or worker-row
    extent differs from the reference's, or the bank count is not the eight ``TWO_WORKER_PROJECTIONS`` was qualified on.
    """

    validate_decode_dram_workers(requested)
    if requested == 1:
        return 1, None
    banks = _dram_bank_count(mesh_device)
    if banks != 8:
        return 1, (
            f"one reader per DRAM bank: the two-reader projections were qualified on 8 DRAM banks, this mesh has {banks}"
        )
    geometries = mesh_device_geometries(mesh_device)
    if len(set(geometries)) > 1:
        differing = sorted(index for index, geometry in enumerate(geometries) if geometry != geometries[0])
        return 1, (
            f"one reader per DRAM bank: the shared two-reader placement cannot be validated, the devices at mesh indices "
            f"{differing} have another DRAM bank count or worker-row extent than device 0 ({geometries[0]} vs "
            f"{[geometries[index] for index in differing]})"
        )
    return requested, None


def bank_tiles(mesh_device, k: int, n: int, num_workers_per_dram_bank: int = 1) -> int:
    """Weight tiles per DRAM bank: ``n`` over the banks, padded to whole tiles per reader."""

    validate_decode_dram_workers(num_workers_per_dram_bank)
    banks = _dram_bank_count(mesh_device)
    if num_workers_per_dram_bank != 1 and (banks != 8 or (k, n) not in TWO_WORKER_PROJECTIONS):
        raise ValueError(f"two readers per DRAM bank are not qualified for banks={banks}, K={k}, N={n}")
    return math.ceil(n / (ttnn.TILE_SIZE * banks * num_workers_per_dram_bank)) * num_workers_per_dram_bank


def dram_sharded_weight_memory_config(mesh_device, k: int, n: int, *, num_workers_per_dram_bank: int = 1):
    """WIDTH_SHARDED DRAM layout for one local ``[1, 1, k, n]`` linear weight."""

    banks = _dram_bank_count(mesh_device)
    bank_width = bank_tiles(mesh_device, k, n, num_workers_per_dram_bank) * ttnn.TILE_SIZE
    bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    shard_spec = ttnn.ShardSpec(bank_grid, (k, bank_width), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def weight_layout_tag(mesh_device, k: int, n: int, *, num_workers_per_dram_bank: int) -> str:
    """``""`` when the bank shard is the one-reader layout, else ``_bank<columns>``: the cache-file suffix of a weight
    whose padding changed, so a tensorbin of the narrower layout is never loaded into the wider allocation
    (``ttnn.as_tensor`` loads a cached tensorbin in the layout it was written in)."""

    tiles = bank_tiles(mesh_device, k, n, num_workers_per_dram_bank)
    return "" if tiles == bank_tiles(mesh_device, k, n) else f"_bank{tiles * ttnn.TILE_SIZE}"


def validate_dram_sharded_weight(tensor, mesh_device, k: int, n: int, *, num_workers_per_dram_bank: int) -> None:
    """Every device member of ``tensor`` holds the bank layout its decode linear runs on (a stale tensorbin fails)."""

    expected = dram_sharded_weight_memory_config(mesh_device, k, n, num_workers_per_dram_bank=num_workers_per_dram_bank)
    members = ttnn.get_device_tensors(tensor)
    if not members:
        raise ValueError(f"decode weight K={k}, N={n} has no device members")
    for member in members:
        if member.memory_config() != expected:
            raise ValueError(
                f"decode weight K={k}, N={n} is not in the {num_workers_per_dram_bank}-reader DRAM bank layout "
                f"(a stale tensorbin?): {member.memory_config()} != {expected}"
            )


def dram_sharded_matmul_configs(mesh_device, k: int, n: int, *, num_cores: int, num_workers_per_dram_bank: int = 1):
    """Activation memory config plus program config for one decode linear.

    ``num_cores`` is the L1 storage grid of the width-sharded activation; it only needs to split ``k`` into whole tiles
    per core. ``per_core_N`` is the activation grid's share of ``n`` with one reader, the bank's tiles with two.
    """

    tiles = bank_tiles(mesh_device, k, n, num_workers_per_dram_bank)
    if num_workers_per_dram_bank != 1 and num_cores != TWO_WORKER_PROJECTIONS[(k, n)]:
        raise ValueError(
            f"two readers per bank for K={k}, N={n} were qualified with {TWO_WORKER_PROJECTIONS[(k, n)]} activation "
            f"storage cores, got {num_cores}"
        )
    if num_cores <= 8:
        storage_grid = ttnn.CoreGrid(x=num_cores, y=1)
    elif num_cores % 8 == 0 and num_cores <= 64:
        storage_grid = ttnn.CoreGrid(x=8, y=num_cores // 8)
    else:
        raise ValueError(f"decode matmul storage grid must be rectangular over 8 columns, got {num_cores} cores")
    if k % (ttnn.TILE_SIZE * num_cores):
        raise ValueError(f"K={k} does not split into whole tiles over {num_cores} storage cores")
    k_tiles_per_core = k // (ttnn.TILE_SIZE * num_cores)
    in0_block_w = next(width for width in range(8, 0, -1) if k_tiles_per_core % width == 0)
    activation_memory_config = ttnn.create_sharded_memory_config(
        (ttnn.TILE_SIZE, k // num_cores),
        storage_grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    if (k, n) in PIPELINE_PROJECTIONS:
        workers = {"num_workers_per_dram_bank": PIPELINE_WORKERS_PER_DRAM_BANK, "use_multicore_pipeline": True}
    else:
        # one reader leaves the field at its default (1)
        workers = {} if num_workers_per_dram_bank == 1 else {"num_workers_per_dram_bank": num_workers_per_dram_bank}
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=1,
        per_core_N=math.ceil(n / (ttnn.TILE_SIZE * num_cores)) if num_workers_per_dram_bank == 1 else tiles,
        fused_activation=None,
        **workers,
    )
    return activation_memory_config, program_config


def _largest_divisor(value: int, cap: int) -> int:
    return next(divisor for divisor in range(min(value, cap), 0, -1) if value % divisor == 0)


def prefill_matmul_program_config(mesh_device, rows: int, k: int, n: int):
    """The 2D-multicast program config of a ``[rows, k] x [k, n]`` prefill-slab linear on interleaved operands.

    The grid width keeps the widest output subblock (the automatic config is slower on the K = 2560 linears);
    ``in0_block_w`` is the largest divisor of the K tiles up to 8; the output subblock is one tile high and up to four
    wide (fp32 accumulation halves the destination registers).
    """

    grid = mesh_device.compute_with_storage_grid_size()
    m_tiles, k_tiles, n_tiles = rows // ttnn.TILE_SIZE, k // ttnn.TILE_SIZE, math.ceil(n / ttnn.TILE_SIZE)
    if rows % ttnn.TILE_SIZE or k % ttnn.TILE_SIZE:
        raise ValueError(f"prefill linear needs whole row and K tiles, got rows={rows} k={k}")
    best_cols, best_key = 1, None
    for cols in range(1, min(int(grid.x), n_tiles) + 1):
        key = (_largest_divisor(math.ceil(n_tiles / cols), 4), cols)
        if best_key is None or key > best_key:
            best_key, best_cols = key, cols
    grid_rows = min(int(grid.y), m_tiles)
    per_core_n = math.ceil(n_tiles / best_cols)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(best_cols, grid_rows),
        in0_block_w=_largest_divisor(k_tiles, 8),
        out_subblock_h=1,
        out_subblock_w=_largest_divisor(per_core_n, 4),
        per_core_M=math.ceil(m_tiles / grid_rows),
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def prefill_linear(activation, weight, program_config, *, compute_kernel_config, dtype=None):
    """One prefill-slab linear on an interleaved copy of the resident DRAM-width-sharded ``weight``, released after.

    The 2D-multicast program admits a sharded in1 but reads it wrong, hence the copy."""

    weight_interleaved = ttnn.to_memory_config(weight, ttnn.DRAM_MEMORY_CONFIG)
    try:
        return ttnn.linear(
            activation,
            weight_interleaved,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            dtype=dtype,
        )
    finally:
        ttnn.deallocate(weight_interleaved)


def dram_sharded_row_tiles(rows, activation_memory_config) -> list:
    """The four 32-row tiles of an interleaved ``[1,1,128,W]`` activation, each moved into a decode linear's
    activation shard (``None``: left interleaved). The DRAM-sharded program admits one row tile per call; the slice
    bounds are literal (captured bodies carry no host-int shape op). The caller deallocates the tiles."""

    width = rows.shape[3]
    if tuple(int(value) for value in rows.shape) != (1, 1, 4 * ttnn.TILE_SIZE, width):
        raise ValueError(f"row tiles need a [1,1,128,W] activation, got {list(rows.shape)}")
    tiles = []
    for tile in (
        ttnn.slice(rows, (0, 0, 0, 0), (1, 1, ttnn.TILE_SIZE, width), memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ttnn.slice(
            rows, (0, 0, ttnn.TILE_SIZE, 0), (1, 1, 2 * ttnn.TILE_SIZE, width), memory_config=ttnn.DRAM_MEMORY_CONFIG
        ),
        ttnn.slice(
            rows,
            (0, 0, 2 * ttnn.TILE_SIZE, 0),
            (1, 1, 3 * ttnn.TILE_SIZE, width),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        ttnn.slice(
            rows,
            (0, 0, 3 * ttnn.TILE_SIZE, 0),
            (1, 1, 4 * ttnn.TILE_SIZE, width),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    ):
        if activation_memory_config is None:
            tiles.append(tile)
            continue
        sharded = ttnn.to_memory_config(tile, activation_memory_config)
        ttnn.deallocate(tile)
        if tuple(int(value) for value in sharded.shape) != (1, 1, ttnn.TILE_SIZE, width):
            raise RuntimeError(f"row tile in the activation shard is {list(sharded.shape)}, expected [1,1,32,{width}]")
        tiles.append(sharded)
    return tiles
