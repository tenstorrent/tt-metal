# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""DRAM-sharded decode matmul configuration shared by the device modules.

Every converted decode ``ttnn.linear`` in this model multiplies one padded
tile row of activations (M == one tile) by a resident weight.  The default
interleaved program serializes one small in1 DRAM read per K block regardless
of N or grid, so kernel time scales with K blocks alone; the DRAM-sharded
program config (the tt_transformers decode pattern) instead streams each
weight shard from its own DRAM bank into worker cores next to that bank.  The
operands, dtypes, and per-element sequential K accumulation under the
caller's compute kernel config are unchanged.

Usage: upload the weight with :func:`dram_sharded_weight_memory_config`, move
the activation to the activation config returned by
:func:`dram_sharded_matmul_configs`, and call ``ttnn.linear`` with the
returned program config and ``memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG``.
The readers per DRAM bank
(``num_workers_per_dram_bank``: the builder's ``decode_dram_workers_per_bank``,
``default_decode_dram_workers``) go to both calls of one linear.
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


# The DRAM-sharded program reads each weight bank with one worker core; on Blackhole it also admits two readers per
# bank (``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig.num_workers_per_dram_bank``, in1 on NOC_0).  Two
# readers serve the projections in ``TWO_WORKER_PROJECTIONS`` by default (2026-09-16, 4x p150: bitwise at rows 1..32
# and at the model level; kernel us per call GDN in-proj 85 -> 67, GDN/QSA out 34 -> 28, QSA query-gate 59 -> 43,
# LM-head chunks 163/185 -> 93/107; 40.27 -> 38.67 ms per decode step); ``QWEN38_DRAM_WORKERS=1`` keeps one reader
# per bank everywhere (the one-reader caches keep their identity).  A bank's shard holds a whole number of tiles per
# reader, which widens only the GDN input projection (4160 columns: 17 -> 18 tiles per bank, 544 -> 576 columns).
WORKERS_ENV = "QWEN38_DRAM_WORKERS"
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


def validate_decode_dram_workers(value) -> int:
    if type(value) is not int or value not in WORKERS_PER_DRAM_BANK:
        raise ValueError(f"decode DRAM workers per bank must be one of {WORKERS_PER_DRAM_BANK}, got {value!r}")
    return value


def default_decode_dram_workers(environ: Mapping[str, str] | None = None) -> int:
    """``QWEN38_DRAM_WORKERS`` (1 or 2) when set, else the serving default of two readers per bank."""

    raw = (os.environ if environ is None else environ).get(WORKERS_ENV, "").strip()
    if not raw:
        return DEFAULT_WORKERS_PER_DRAM_BANK
    if raw not in {str(value) for value in WORKERS_PER_DRAM_BANK}:
        raise ValueError(f"{WORKERS_ENV} must be one of {WORKERS_PER_DRAM_BANK}, got {raw!r}")
    return int(raw)


# -- dense weight dtype (QWEN38_DENSE_WEIGHT_DTYPE) -------------------------------------------------------------------
# ``bf8`` (the default: HiFi2 matmuls, about 1 GB less per die and 3 % off the decode step at a KL of 0.007 against
# the CPU oracle on the 2026-09-24 A/B) | ``bf16`` (the previous production format, byte for byte) | ``bf4``: the dtype of the resident dense
# matmul weights the decode linears stream from DRAM -- the GDN projections (qkvzab, out), the QSA projections (qg,
# k, v, out, index_q, index_k), the shared expert (gate, up, down, the scalar gate and the fused [gate|up|scalar]
# row), the LM-head chunks, the final mixer (down, up) and the MTP input fc linears.  ``QWEN38_DENSE_WEIGHT_DTYPE_<M>``
# (M in GDN, QSA, SHARED_EXPERT, LM_HEAD, FINAL_MIXER, MTP) overrides one module.  The matmuls whose weights changed
# run the compute fidelity tech_reports/LLMs/llms.md gives for the weight format (HiFi4 for BF16, HiFi2 for BFP8,
# LoFi for BFP4; fp32 accumulation, no approximation, no packer L1 accumulation, as before); every other program keeps
# its config.  Kept BF16 regardless: the embedding table, the router (mlp.gate), every norm, the GDN conv taps /
# dt_bias / A_log (the GDN gating projections in_proj_a / in_proj_b are packed into qkvzab and follow its dtype), the
# PLE tables, and the hyper-connection (GR) mixers, whose default-on fused read program (ttnn/fused/gr_read) streams
# BF16 tiles through BF16 circular buffers.  A converted tensorbin carries the dtype tag in its name (``.bf8b`` /
# ``.bf4b``; the GDN's ``.bf16`` / ``.bf8b`` scheme), so the bf16 caches are never read or written by another dtype.
DENSE_DTYPE_ENV = "QWEN38_DENSE_WEIGHT_DTYPE"
DEFAULT_DENSE_DTYPE_NAME = "bf8"
DENSE_MODULES = ("gdn", "qsa", "shared_expert", "lm_head", "final_mixer", "mtp")
DENSE_DTYPE_NAMES: dict[str, Any] = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b}
DENSE_DTYPE_TAGS: dict[Any, str] = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bf8b", ttnn.bfloat4_b: "bf4b"}
DENSE_MATH_FIDELITY_NAMES: dict[Any, str] = {ttnn.bfloat16: "HiFi4", ttnn.bfloat8_b: "HiFi2", ttnn.bfloat4_b: "LoFi"}
# One 32x32 tile: 1024 bf16 values; 1024 one-byte mantissas + 64 shared exponents; 512 bytes of 4-bit mantissas + 64.
DENSE_TILE_BYTES: dict[Any, int] = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
# The two-reader bank layouts are whole tiles per reader (bank_tiles), so they do not depend on the tile's bytes; a
# dtype enters this set once one reader and two readers were checked bitwise equal on the device for every shape of
# TWO_WORKER_PROJECTIONS (tools/qualify_dense_two_readers.py).  A dtype outside it runs one reader everywhere (the
# builder records the reason in decode_dram_workers_fallback).  Qualified 2026-09-24 on a QuietBox p150b (eight
# banks): bf16, bf8b (HiFi2) and bf4b (LoFi) bitwise equal one vs two readers on all five shapes at 32 rows.
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
    """``QWEN38_DENSE_WEIGHT_DTYPE`` (bf8 when unset) for every module, ``QWEN38_DENSE_WEIGHT_DTYPE_<MODULE>`` over it."""

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
            assignment = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment_at_mesh_coordinate(
                mesh_device, ttnn.NOC.RISCV_0_default, ttnn.MeshCoordinate(row, column)
            )
            signatures[(row, column)] = tuple((int(core.x), int(core.y)) for core in assignment)
    return signatures


def mesh_device_geometries(mesh_device) -> tuple[tuple[int, int], ...]:
    """Per mesh coordinate (row-major), (DRAM bank count, the row extent max y + 1 of the device's optimal bank reader
    cores): the geometry one shared DRAM-sharded placement must find on every device, read from the per-coordinate
    bank -> worker assignment Python has (a MeshDevice exposes no per-device handle).  Harvesting moves columns, not
    rows, so dies harvested in different columns share this geometry; a device whose worker grid differs in its rows
    or bank count is refused here, one whose columns differ is refused by tt-metal's geometry check at the first
    program build."""

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

    One DRAM-sharded matmul program is placed on every device of the mesh (tt-metal ``get_dram_bank_reader_assignments``):
    the reference (first) device's optimal bank -> worker assignment is the placement everywhere.  The DRAM-sharded
    readers address their shard by bank id, so any worker core can read any bank and the assignment is a locality
    choice.  The shared placement is valid when every device has the reference device's worker grid and DRAM grid; a
    mesh of dies harvested in different columns (a QuietBox 2: one die serves banks 4-7 from another worker column,
    2026-09-29) then runs two readers with the odd dies reading those banks from a worker one or more NoC hops further
    (``decode_dram_workers_placement`` names them, the READY record carries it, tt-metal warns once per program build
    with the largest hop deviation; the acceptance replay is bitwise the one-reader stream, the rows of record in
    NUMERICS).  One reader per bank remains for the cases the placement cannot be validated: a device whose DRAM bank
    count or worker-row extent differs from the reference device's (another die geometry; a worker grid differing in
    its columns is refused by tt-metal at the first program build), and a mesh whose bank count is not the eight the
    two-reader table (``TWO_WORKER_PROJECTIONS``) was qualified on (a seven-bank Blackhole ring).
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

    ``num_cores`` is the L1 storage grid holding the width-sharded activation;
    the kernel pins its compute workers to the DRAM banks holding the weight
    (``num_workers_per_dram_bank`` of them per bank), so the grid only needs to
    split ``k`` into whole tiles per core (which also fixes the in0 block width
    to the largest tile divisor).  ``per_core_N`` is the output storage width:
    the activation grid's share of ``n`` with one reader, the bank's tiles with
    two (the configuration the two-reader form was qualified with).
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
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=1,
        per_core_N=math.ceil(n / (ttnn.TILE_SIZE * num_cores)) if num_workers_per_dram_bank == 1 else tiles,
        fused_activation=None,
        # the one-reader constructor call stays the call of 2026-09-04 (the field's own default is 1)
        **({} if num_workers_per_dram_bank == 1 else {"num_workers_per_dram_bank": num_workers_per_dram_bank}),
    )
    return activation_memory_config, program_config


def _largest_divisor(value: int, cap: int) -> int:
    return next(divisor for divisor in range(min(value, cap), 0, -1) if value % divisor == 0)


def prefill_matmul_program_config(mesh_device, rows: int, k: int, n: int):
    """The 2D-multicast program config of a ``[rows, k] x [k, n]`` prefill-slab linear on interleaved operands.

    The grid width is the one that keeps the widest output subblock (the runtime's automatic config is 2-2.5x slower
    on the K = 2560 linears), ``in0_block_w`` the largest divisor of the K tiles up to 8, the output subblock one tile
    high and up to four wide (fp32 accumulation halves the destination registers), ``per_core_M`` the row tiles over
    the grid's rows.  Measured 2026-09-09 on 4x p150: 10-12x the per-tile DRAM-sharded form at 2048 rows, within one
    bf16 ULP of it.
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
    """One prefill-slab linear: the resident DRAM-width-sharded ``weight`` copied interleaved (the 2D-multicast
    program admits a sharded in1 but reads it wrong: 96 % of the outputs, measured 2026-09-09), the matmul on the
    interleaved activation into an interleaved DRAM output, the copy released.  The caller's compute config and
    output dtype are the decode linear's."""

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
    """The four 32-row tiles of an interleaved ``[1,1,128,W]`` activation (the long prefill chunk), each moved into
    a decode linear's activation shard (``None``: left interleaved, for the per-tile routed expert stream): the
    DRAM-sharded program admits one row tile per call (``per_core_M == 1``), so the 128-row activation runs four
    calls of the same program on the same rows.  The slice bounds are literal (the captured bodies carry no
    host-int-dependent shape op).  The caller deallocates the tiles."""

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
