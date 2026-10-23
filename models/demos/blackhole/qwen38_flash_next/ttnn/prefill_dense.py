# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The prefill slab's dense linears behind four switches: weight dtype, math fidelity, grid, fp32 accumulation.

Inside a slab (``--prefill-slab ROWS``) every dense linear of a layer runs as one 2D-multicast matmul over the rows
on a per-slab interleaved copy of its DRAM-width-sharded decode weight (``decode_matmul.prefill_linear``), in the
weight's format (``QWEN38_DENSE_WEIGHT_DTYPE``, bf8 by default) with the module's compute config for it (HiFi4 for
bf16, HiFi2 for bf8, LoFi for bf4; fp32 accumulation).  The switches below, read once when the modules are built,
move those linears and nothing else.  The dtype, fidelity and accumulation defaults keep the slab's arithmetic; the
grid's default is ``wide``
(measured on the 4-chip line: a device column identical to ``today`` at every scored position and the same
completions, the dense family 92.6 -> 75.8 ms per 2048-row slab), and ``GRID=today`` restores the earlier slab
bitwise:

    QWEN38_PREFILL_DENSE_DTYPE=bf16|bf8          (bf16)  bf8: a resident DRAM-interleaved bfloat8_b copy of each weight,
                                                          prefill only (no per-slab copy; the decode weights keep their format)
    QWEN38_PREFILL_DENSE_FIDELITY=hifi4|hifi2|lofi (hifi4)  hifi4: the module's own compute config (the fidelity of
                                                          its weight format); hifi2 / lofi: that fidelity instead
    QWEN38_PREFILL_DENSE_GRID=today|wide         (wide)  wide: more columns for the narrow-N shapes, and the pair-grouped
                                                          K/V and the shared expert's gate/up as one linear each on
                                                          resident copies in the weights' format (the same tiles the
                                                          separate linears read; 58 MB per device as bf8, 110 MB as
                                                          bf16); today: the grids of
                                                          ``decode_matmul.prefill_matmul_program_config``, no resident
    QWEN38_PREFILL_DENSE_FP32_ACC=1|0            (1)     fp32 accumulation across the K blocks

The switches act on a build that runs a slab: the builder is told the slab's rows before the target is built
(``Qwen38TTNNBuilder.enable_prefill_slab``, called by the chat chain, the device profile tool and the full-model
gate when ``--prefill-slab`` names one) and only then builds the resident copies behind the DRAM admission.  A
build without a slab holds no slab linear, so it allocates nothing and admits nothing: it is today's build.

The **exact set** never reads the switches: the MoE router (its logits pick the ten experts), the QSA ``index_q`` and
``index_k`` projections (their outputs pick the attended blocks and persist in the compressed index cache).  Their call
sites ask for today's program config with ``exact=True`` and pass no resident weight, so no switch value can move them.
The decode path is untouched: the DRAM-sharded decode linears, their weights and ``decode_matmul`` are as before.

Arms, expected gains, the quality readouts (the acceptance gate never reaches a slab body: only prompts of 2048 tokens
and more do) and the measurement plan: the prefill dense design note (a development document, not shipped).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    MESH_SHAPE,
    is_slab_rows,
    replicate_tensor_2d_mesh_mapper,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.decode_matmul import prefill_matmul_program_config

DTYPE_ENV = "QWEN38_PREFILL_DENSE_DTYPE"
FIDELITY_ENV = "QWEN38_PREFILL_DENSE_FIDELITY"
GRID_ENV = "QWEN38_PREFILL_DENSE_GRID"
FP32_ACC_ENV = "QWEN38_PREFILL_DENSE_FP32_ACC"
DTYPES = ("bf16", "bf8")
FIDELITIES = ("hifi4", "hifi2", "lofi")
GRIDS = ("today", "wide")
FP32_ACCS = ("1", "0")
DEFAULTS = {DTYPE_ENV: "bf16", FIDELITY_ENV: "hifi4", GRID_ENV: "wide", FP32_ACC_ENV: "1"}
CHOICES = {DTYPE_ENV: DTYPES, FIDELITY_ENV: FIDELITIES, GRID_ENV: GRIDS, FP32_ACC_ENV: FP32_ACCS}
SWITCHES = (DTYPE_ENV, FIDELITY_ENV, GRID_ENV, FP32_ACC_ENV)

# Bytes of one 32 x 32 tile: bf16; bfloat8_b (1024 mantissa bytes + 64 shared exponents); bfloat4_b (512 + 64).
TILE_BYTES = {
    "bf16": 2 * ttnn.TILE_SIZE * ttnn.TILE_SIZE,
    "bf8": ttnn.TILE_SIZE * ttnn.TILE_SIZE + 64,
    "bf4": ttnn.TILE_SIZE * ttnn.TILE_SIZE // 2 + 64,
}
# A resident weight's format: the policy's bf8 copies, or the module's dense weight format for the fused siblings
# (decode_matmul's QWEN38_DENSE_WEIGHT_DTYPE, whose ttnn dtypes these names stand for).
SPEC_DTYPES = ("bf16", "bf8", "bf4")
WEIGHT_DTYPE_NAMES = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bf8", ttnn.bfloat4_b: "bf4"}
TTNN_DTYPES = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b}
CACHE_TAGS = {"bf16": "bf16", "bf8": "bf8b", "bf4": "bf4b"}


def weight_dtype_name(dtype) -> str:
    """The spec-dtype name of a module's dense weight dtype (ttnn.bfloat16 / bfloat8_b / bfloat4_b)."""

    if dtype not in WEIGHT_DTYPE_NAMES:
        raise ValueError(f"dense weight dtype must be one of {tuple(WEIGHT_DTYPE_NAMES)}, got {dtype!r}")
    return WEIGHT_DTYPE_NAMES[dtype]


# The DRAM the builder keeps free beyond the resident prefill weights and the context-scaled state (the traces, the
# slab state, the transients inside a layer).
DRAM_MARGIN_BYTES = 512 << 20
# The 2048-row slab's DRAM working set per device beyond the resident prefill weights, the context state and the
# margin: what a slab process allocates after the build over those (the chunk states, the tensors alive through a
# slab, the layer transients and the prefill traces are the margin's contents; the allocator's fragmentation and the
# rest of the process are the excess).  MEASURED as a band on the 4-chip line at a 32k context (2026-09-25; the free
# bytes read after the build, before the context state is allocated): the slab ran with 251,272,000 B free per bank
# (the wide plan resident; 2,010,176,000 B per device) and OOM'd in its first layer with 141,002,560 B free per bank
# (the bf8 + wide plan under the device profiler's 64,000-program reservation; 1,128,020,480 B per device), so the
# post-build need lies in (1,128,020,480, 2,010,176,000] B per device.  The conservative reading is the band's upper
# end, the only value the slab is known to fit: the lower end is the OOM itself, and the slab8k derivation's 145 MB
# (the QSA sparse_sdpa stage's peak alone) plus the context state and the margin sum to 1,119 MB per device, which
# would have admitted the run that OOM'd.  The band holds the context state (436,797,440 B at 32k: 12 QSA generic
# states and the RoPE tables) and the margin, which the admission adds on their own, so the term is the band's upper
# end less those two: at 32k the admission asks for exactly the measured pass point after the plan (251.27 MB per
# bank), and the context state's growth at larger contexts adds on top of it.
SLAB_WORKING_SET_BYTES = 2_010_176_000 - 436_797_440 - DRAM_MARGIN_BYTES  # 1,036,507,648 B (988.5 MiB)
MEASURED_SLAB_ROWS = 2048


def slab_working_set_bytes(rows: int) -> int:
    """The working-set term for a slab of ``rows``: the measured 2048-row band up to 2048 rows (its constant parts, the
    traces and the fragmentation, are not separable from the rows-linear ones), scaled by the rows above 2048 (the slab
    state, the tensors alive through a slab and the layer transients are rows-linear)."""

    if not is_slab_rows(rows):
        raise ValueError(f"a prefill slab takes a multiple of 128 rows in 256..4096, got {rows!r}")
    return SLAB_WORKING_SET_BYTES * max(rows, MEASURED_SLAB_ROWS) // MEASURED_SLAB_ROWS


# The dense linears of the slab path by module, local [K, N] per device (the shapes the modules' decode weights have:
# gr.FLAT_LOCAL_WIDTH x gr.PARTIAL_WIDTH, gdn.HIDDEN_SIZE x gdn.PROJECTION_WIDTH_PER_DEVICE, ...; the static test pins
# them to the modules' constants).  ``shard_dim`` is the mesh shard axis of the [1, 1, K_total, N_total] host block
# (3: the columns over the devices, 2: the rows; None: replicated), ``columns`` the whole-tile column slices of a
# fused sibling.  Layer counts: 36 GDN, 12 QSA, 48 gated residuals per block (attn, mlp), 48 MoE.
HIDDEN = 2560
GR_PARTIAL_WIDTH = 384
GDN_PROJECTION_WIDTH = 4160
GDN_VALUE_WIDTH = 1536
QSA_LOCAL_QUERY_WIDTH = 1536
QSA_HEAD_DIM = 256
MOE_LOCAL_INTERMEDIATE = 160
KV_COLUMNS = (("k", 0, QSA_HEAD_DIM), ("v", QSA_HEAD_DIM, QSA_HEAD_DIM))
SHARED_COLUMNS = (
    ("shared_gate", 0, MOE_LOCAL_INTERMEDIATE),
    ("shared_up", MOE_LOCAL_INTERMEDIATE, MOE_LOCAL_INTERMEDIATE),
)
MODULES = ("gr", "gdn", "qsa", "moe")
LAYER_COUNTS = {"gr": 96, "gdn": 36, "qsa": 12, "moe": 48}
# The exact set, named so the static test can assert it carries no policy: module -> the weights.
EXACT_LINEARS = {"moe": ("router",), "qsa": ("index_q", "index_k")}


@dataclass(frozen=True)
class ResidentSpec:
    """One resident prefill weight: its name in the module's table, local K x N, mesh shard axis, dtype tag and, for a
    fused sibling, the (name, first column, width) slices its output is cut into."""

    name: str
    k: int
    n: int
    shard_dim: int | None
    dtype: str
    columns: tuple[tuple[str, int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.dtype not in SPEC_DTYPES:
            raise ValueError(f"resident prefill weight dtype must be one of {SPEC_DTYPES}, got {self.dtype!r}")

    @property
    def tiles(self) -> int:
        return math.ceil(self.k / ttnn.TILE_SIZE) * math.ceil(self.n / ttnn.TILE_SIZE)

    @property
    def bytes(self) -> int:
        return self.tiles * TILE_BYTES[self.dtype]

    @property
    def ttnn_dtype(self):
        return TTNN_DTYPES[self.dtype]

    @property
    def cache_name(self) -> str:
        """A new tensorbin name per layout and dtype: no cached tensorbin of another layout can be loaded into it."""

        return f"{self.name}_prefill_interleaved.{CACHE_TAGS[self.dtype]}"


@dataclass(frozen=True)
class Qwen38PrefillDensePolicy:
    """The four switches as one value.  ``from_environ`` reads them (unknown values raise); the default instance is
    the wide grid on today's arithmetic (bf16 weights copied per slab, HiFi4, fp32 accumulation); ``grid="today"``
    is the earlier slab, bitwise."""

    dtype: str = DEFAULTS[DTYPE_ENV]
    fidelity: str = DEFAULTS[FIDELITY_ENV]
    grid: str = DEFAULTS[GRID_ENV]
    fp32_acc: bool = True

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES or self.fidelity not in FIDELITIES or self.grid not in GRIDS:
            raise ValueError(
                f"prefill dense policy takes dtype {DTYPES}, fidelity {FIDELITIES}, grid {GRIDS}; got "
                f"{self.dtype!r} {self.fidelity!r} {self.grid!r}"
            )
        if type(self.fp32_acc) is not bool:
            raise ValueError(f"prefill dense fp32_acc must be a bool, got {self.fp32_acc!r}")

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> "Qwen38PrefillDensePolicy":
        environ = os.environ if environ is None else environ
        values = {}
        for variable in SWITCHES:
            raw = environ.get(variable, "").strip().lower() or DEFAULTS[variable]
            if raw not in CHOICES[variable]:
                raise ValueError(f"{variable} must be one of {CHOICES[variable]}, got {environ.get(variable)!r}")
            values[variable] = raw
        return cls(
            dtype=values[DTYPE_ENV],
            fidelity=values[FIDELITY_ENV],
            grid=values[GRID_ENV],
            fp32_acc=values[FP32_ACC_ENV] == "1",
        )

    @property
    def is_default(self) -> bool:
        return self == Qwen38PrefillDensePolicy()

    @property
    def resident_weights(self) -> bool:
        """Whether the builder allocates resident prefill weights: the bfloat8_b copies (dtype bf8) and/or the fused
        siblings (grid wide)."""

        return self.dtype == "bf8" or self.grid == "wide"

    @property
    def fused_siblings(self) -> bool:
        return self.grid == "wide"

    @property
    def math_fidelity(self):
        return {"hifi4": ttnn.MathFidelity.HiFi4, "hifi2": ttnn.MathFidelity.HiFi2, "lofi": ttnn.MathFidelity.LoFi}[
            self.fidelity
        ]

    @property
    def module_compute_config(self) -> bool:
        """True when the policy keeps the modules' own compute config (the fidelity of their dense weight format, fp32
        accumulation): nothing is rebuilt."""

        return self.fidelity == "hifi4" and self.fp32_acc

    def describe(self) -> str:
        return f"dtype={self.dtype} fidelity={self.fidelity} grid={self.grid} fp32_acc={int(self.fp32_acc)}" + (
            " (default)" if self.is_default else ""
        )

    def resident_plan(self, module: str, weight_dtype: str = "bf16") -> tuple[ResidentSpec, ...]:
        """The resident prefill weights one module of ``module`` holds under this policy: empty under grid today with
        dtype bf16; the bfloat8_b copies of its non-exact linears under dtype bf8; the fused [k | v] and [gate | up]
        siblings under grid wide, the default, in ``weight_dtype`` (the module's dense weight format,
        QWEN38_DENSE_WEIGHT_DTYPE: the fused linear then reads the same tiles the separate linears read) or in bf8
        under dtype bf8.  The exact set is never in the plan."""

        if module not in MODULES:
            raise ValueError(f"prefill dense module must be one of {MODULES}, got {module!r}")
        if weight_dtype not in SPEC_DTYPES:
            raise ValueError(f"the module's dense weight dtype must be one of {SPEC_DTYPES}, got {weight_dtype!r}")
        if not self.resident_weights:
            return ()
        bf8 = self.dtype == "bf8"
        sibling_dtype = "bf8" if bf8 else weight_dtype
        specs: list[ResidentSpec] = []
        if module == "gr" and bf8:
            specs.append(ResidentSpec("down_inject", HIDDEN, GR_PARTIAL_WIDTH, 2, self.dtype))
            specs.append(ResidentSpec("up", GR_PARTIAL_WIDTH, HIDDEN, 3, self.dtype))
        elif module == "gdn" and bf8:
            specs.append(ResidentSpec("qkvzab", HIDDEN, GDN_PROJECTION_WIDTH, 3, self.dtype))
            specs.append(ResidentSpec("out", GDN_VALUE_WIDTH, HIDDEN, 2, self.dtype))
        elif module == "qsa":
            if bf8:
                specs.append(ResidentSpec("qg", HIDDEN, 2 * QSA_LOCAL_QUERY_WIDTH, 3, self.dtype))
                specs.append(ResidentSpec("out", QSA_LOCAL_QUERY_WIDTH, HIDDEN, 2, self.dtype))
            if self.fused_siblings:
                specs.append(ResidentSpec("kv", HIDDEN, 2 * QSA_HEAD_DIM, 3, sibling_dtype, KV_COLUMNS))
            elif bf8:
                specs.append(ResidentSpec("k", HIDDEN, QSA_HEAD_DIM, 3, self.dtype))
                specs.append(ResidentSpec("v", HIDDEN, QSA_HEAD_DIM, 3, self.dtype))
        elif module == "moe":
            if self.fused_siblings:
                specs.append(
                    ResidentSpec("shared_gate_up", HIDDEN, 2 * MOE_LOCAL_INTERMEDIATE, 3, sibling_dtype, SHARED_COLUMNS)
                )
            elif bf8:
                specs.append(ResidentSpec("shared_gate", HIDDEN, MOE_LOCAL_INTERMEDIATE, 3, self.dtype))
                specs.append(ResidentSpec("shared_up", HIDDEN, MOE_LOCAL_INTERMEDIATE, 3, self.dtype))
            if bf8:
                specs.append(ResidentSpec("shared_down", MOE_LOCAL_INTERMEDIATE, HIDDEN, 2, self.dtype))
                specs.append(ResidentSpec("shared_scalar_gate", HIDDEN, 1, None, self.dtype))
        return tuple(specs)

    def plan_bytes_per_device(self, weight_dtypes: Mapping[str, str] | None = None) -> int:
        """Padded-tile bytes of every resident prefill weight of the 48 layers per device (DERIVED); ``weight_dtypes``
        names the modules' dense weight formats for the fused siblings (bf16 unless named)."""

        weight_dtypes = weight_dtypes or {}
        return sum(
            LAYER_COUNTS[module]
            * sum(spec.bytes for spec in self.resident_plan(module, weight_dtypes.get(module, "bf16")))
            for module in MODULES
        )

    def compute_config(self, mesh_device, module_default):
        """The module's own compute config when the policy keeps its fidelity and accumulation (the same object:
        bitwise), else a new one with the policy's fidelity and accumulation, the other fields the module's."""

        if self.module_compute_config:
            return module_default
        return ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=self.math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=self.fp32_acc,
            packer_l1_acc=False,
        )

    def program_config(self, mesh_device, rows: int, k: int, n: int):
        """Today's 2D-multicast config (grid today) or the wide one."""

        if self.grid == "today":
            return prefill_matmul_program_config(mesh_device, rows, k, n)
        return wide_prefill_matmul_program_config(mesh_device, rows, k, n)


def wide_prefill_matmul_program_config(mesh_device, rows: int, k: int, n: int):
    """The 2D-multicast config that runs the most columns for a ``[rows, k] x [k, n]`` slab linear.

    Today's rule (``prefill_matmul_program_config``) picks the widest output subblock first and leaves the narrow-N
    linears on 10-40 of the 110 cores.  This one ranks the column count the program factory runs
    (``ceil(N_tiles / per_core_N)``) first, among the configs whose output subblock is at least two tiles wide (a
    one-tile subblock stalls the matmul), then the wider subblock; a one-tile N falls back to today's rule.  The K
    block, the row split and every other field are today's, so the per-column arithmetic is the same and only the
    block order over the cores changes.
    """

    grid = mesh_device.compute_with_storage_grid_size()
    m_tiles, k_tiles, n_tiles = rows // ttnn.TILE_SIZE, k // ttnn.TILE_SIZE, math.ceil(n / ttnn.TILE_SIZE)
    if rows % ttnn.TILE_SIZE or k % ttnn.TILE_SIZE:
        raise ValueError(f"prefill linear needs whole row and K tiles, got rows={rows} k={k}")
    best = None
    for cols in range(1, min(int(grid.x), n_tiles) + 1):
        per_core_n = math.ceil(n_tiles / cols)
        running = math.ceil(n_tiles / per_core_n)
        subblock = decode_matmul._largest_divisor(per_core_n, 4)
        if subblock < 2:
            continue
        key = (running, subblock)
        if best is None or key > best[0]:
            best = (key, running, per_core_n, subblock)
    if best is None:
        return prefill_matmul_program_config(mesh_device, rows, k, n)
    _key, running, per_core_n, subblock = best
    grid_rows = min(int(grid.y), m_tiles)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(running, grid_rows),
        in0_block_w=decode_matmul._largest_divisor(k_tiles, 8),
        out_subblock_h=1,
        out_subblock_w=subblock,
        per_core_M=math.ceil(m_tiles / grid_rows),
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def prefill_linear(activation, weight, program_config, *, compute_kernel_config, dtype=None, resident_weight=None):
    """``decode_matmul.prefill_linear`` (the per-slab interleaved copy of ``weight``, released after the matmul) unless
    the policy holds a resident prefill weight for this linear: then the matmul reads ``resident_weight`` in place and
    nothing is copied or released.  Without a resident weight the call is the unchanged decode_matmul helper: the
    per-slab copy, bitwise the earlier slab."""

    if resident_weight is None:
        return decode_matmul.prefill_linear(
            activation, weight, program_config, compute_kernel_config=compute_kernel_config, dtype=dtype
        )
    return ttnn.linear(
        activation,
        resident_weight,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        dtype=dtype,
    )


class Qwen38TTNNPrefillDenseWeights:
    """The resident prefill weights of one module: name -> (device tensor, spec).  Empty under grid today with dtype
    bf16, and in every module but the QSA and the MoE under the default policy."""

    def __init__(self, entries: Mapping[str, tuple[Any, ResidentSpec]] | None = None) -> None:
        self._entries: dict[str, tuple[Any, ResidentSpec]] = dict(entries or {})

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def items(self):
        return self._entries.items()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._entries)

    @property
    def bytes(self) -> int:
        return sum(spec.bytes for _tensor, spec in self._entries.values())

    def tensor(self, name: str):
        return self._entries[name][0]

    def spec(self, name: str) -> ResidentSpec:
        return self._entries[name][1]

    def deallocate(self) -> None:
        for name, (tensor, _spec) in list(self._entries.items()):
            ttnn.deallocate(tensor)
            del self._entries[name]


class Qwen38TTNNPrefillDense:
    """One module's view of the policy: the compute config for its slab linears, the wide program configs (cached
    per shape), the resident prefill weights (attached by the builder after every other resident allocation) and the
    fused-sibling linear.  Construction touches no device: ``mesh_device`` is read when a config is first asked for."""

    def __init__(self, policy: Qwen38PrefillDensePolicy, mesh_device) -> None:
        if not isinstance(policy, Qwen38PrefillDensePolicy):
            raise TypeError(f"prefill dense policy must be a Qwen38PrefillDensePolicy, got {policy!r}")
        self.policy = policy
        self.mesh_device = mesh_device
        self.weights = Qwen38TTNNPrefillDenseWeights()
        self._program_configs: dict[tuple[int, int, int], Any] = {}
        self._compute_configs: list[tuple[Any, Any]] = []

    @classmethod
    def resolve(cls, given: "Qwen38TTNNPrefillDense | None", mesh_device) -> "Qwen38TTNNPrefillDense":
        """The module's ``prefill_dense`` argument, or a fresh one from the environment (the switches read once)."""

        if given is None:
            return cls(Qwen38PrefillDensePolicy.from_environ(), mesh_device)
        if not isinstance(given, Qwen38TTNNPrefillDense):
            raise TypeError(f"prefill_dense must be a Qwen38TTNNPrefillDense or None, got {given!r}")
        return given

    def compute_config(self, module_default):
        """``module_default`` itself under the default fidelity and accumulation (bitwise), else the policy's config
        built once per module default."""

        if self.policy.module_compute_config:
            return module_default
        for default, derived in self._compute_configs:
            if default is module_default:
                return derived
        derived = self.policy.compute_config(self.mesh_device, module_default)
        self._compute_configs.append((module_default, derived))
        return derived

    def program_config(self, rows: int, k: int, n: int):
        """The policy's 2D-multicast config for one shape, built once."""

        key = (rows, k, n)
        if key not in self._program_configs:
            self._program_configs[key] = self.policy.program_config(self.mesh_device, rows, k, n)
        return self._program_configs[key]

    def resident(self, name: str):
        """The resident prefill weight ``name`` holds, or None (today's per-slab copy)."""

        return self.weights.tensor(name) if name in self.weights else None

    def attach(self, weights: Qwen38TTNNPrefillDenseWeights) -> None:
        for name, entry in weights.items():
            if name in self.weights:
                raise ValueError(f"prefill dense weight {name!r} is attached twice")
            self.weights._entries[name] = entry

    def fused_linear(self, activation, name: str, rows: int, *, compute_kernel_config):
        """One linear on the fused sibling ``name`` over the slab rows: the ``[1, 1, rows, N]`` interleaved DRAM output
        the caller cuts into the spec's whole-tile column slices with its own literal bounds (a captured body admits no
        host-integer shape op) and hands to :meth:`retag_sharded`."""

        weight, spec = self.weights.tensor(name), self.weights.spec(name)
        if not spec.columns:
            raise ValueError(f"prefill dense weight {name!r} is not a fused sibling")
        return ttnn.linear(
            activation,
            weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.program_config(rows, spec.k, spec.n),
            compute_kernel_config=self.compute_config(compute_kernel_config),
        )

    def retag_sharded(self, *parts, reference, shard_dim: int) -> None:
        """Mesh metadata of a fused sibling's column slices: ``reference``'s distribution and coordinates, sharded on
        ``shard_dim`` (the slices are per-device values already; no bytes move)."""

        topology = reference.tensor_topology()
        for part in parts:
            part.update_tensor_topology(
                ttnn.TensorTopology(
                    topology.distribution_shape(),
                    [ttnn.PlacementReplicate(), ttnn.PlacementShard(shard_dim)],
                    topology.mesh_coords(),
                )
            )

    def deallocate(self) -> None:
        self.weights.deallocate()


def admit_prefill_dense_dram(
    mesh_device,
    *,
    plan_bytes: int,
    context_state_bytes: int,
    slab_working_set: int = SLAB_WORKING_SET_BYTES,
    margin: int = DRAM_MARGIN_BYTES,
) -> dict:
    """Refuse the resident prefill weights when the free DRAM per device (the allocator's view, after every other
    resident allocation) would not hold them plus what the slab process allocates after the build: the context-scaled
    state (the QSA caches and RoPE tables the model allocates next), the 2048-row slab's working set (the measured
    band, ``SLAB_WORKING_SET_BYTES``) and the margin; returns the numbers it admitted on."""

    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    banks = int(view.num_banks)
    free = int(view.total_bytes_free_per_bank) * banks
    needed = plan_bytes + context_state_bytes + slab_working_set + margin
    numbers = {
        "free_bytes_per_device": free,
        "plan_bytes_per_device": plan_bytes,
        "context_state_bytes_per_device": context_state_bytes,
        "slab_working_set_bytes_per_device": slab_working_set,
        "margin_bytes": margin,
        "needed_bytes_per_device": needed,
        "num_banks": banks,
    }
    if free < needed:

        def mib(value: int) -> str:
            return f"{value / 2**20:.0f} MiB"

        raise ValueError(
            "QWEN38_PREFILL_DENSE_DTYPE=bf8 / GRID=wide refused: the resident prefill weights need "
            f"{mib(plan_bytes)} per device plus {mib(context_state_bytes)} of context state, the 2048-row slab's "
            f"{mib(slab_working_set)} working set and a {mib(margin)} margin = {mib(needed)}, but {mib(free)} are free "
            f"after the resident build: {(free - plan_bytes) / banks / 1e6:.1f} MB per bank would remain after the "
            "weights, and the slab is measured to run at 251.3 MB per bank and to fail at 141.0; use a smaller "
            "context, GRID=today with DTYPE=bf16, or a process that reserves less DRAM (the device profiler's program "
            "count)"
        )
    return numbers


def _upload(mesh_device, host, spec: ResidentSpec, cache_dir: Path):
    """One resident DRAM-interleaved TILE tensor in the spec's dtype, cached under the spec's new tensorbin name."""

    import torch

    if spec.shard_dim is None:
        mapper = replicate_tensor_2d_mesh_mapper(mesh_device)
        expected = (1, 1, spec.k, spec.n)
    else:
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, spec.shard_dim))
        expected = (
            1,
            1,
            spec.k * (MESH_SHAPE[1] if spec.shard_dim == 2 else 1),
            spec.n * (MESH_SHAPE[1] if spec.shard_dim == 3 else 1),
        )
    if tuple(host.shape) != expected:
        raise RuntimeError(f"prefill dense {spec.name}: host block {tuple(host.shape)} is not {expected}")
    tensor = ttnn.as_tensor(
        host.to(torch.bfloat16).contiguous(),
        dtype=spec.ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
        cache_file_name=cache_dir / spec.cache_name,
    )
    local = tuple(int(value) for value in tensor.shape)
    if local != (1, 1, spec.k, spec.n):
        raise RuntimeError(f"prefill dense {spec.name}: local block {local} is not {(1, 1, spec.k, spec.n)}")
    return tensor


def _host_blocks(module: str, checkpoint, placement, *, layer_index: int, block: str | None) -> dict:
    """The [1, 1, K_total, N_total] host blocks of a module's non-exact linears, from the same tt/ sources and packers
    the decode loaders use (the decode loaders themselves are not touched)."""

    import torch

    if module == "gr":
        from models.demos.blackhole.qwen38_flash_next.tt.gr import Qwen38GatedResidualWeights
        from models.demos.blackhole.qwen38_flash_next.ttnn import gr

        source = Qwen38GatedResidualWeights.from_checkpoint(checkpoint, placement, layer_index=layer_index, block=block)
        prepared = gr._prepare_host_weights(source)
        return {"down_inject": prepared["down_inject"], "up": prepared["up"]}
    if module == "gdn":
        from models.demos.blackhole.qwen38_flash_next.tt.gdn import Qwen38GDNWeights
        from models.demos.blackhole.qwen38_flash_next.ttnn import gdn

        source = Qwen38GDNWeights.from_checkpoint(checkpoint, layer_index)
        shards = tuple(source.device_shard(device) for device in range(gdn.TP_SIZE))
        out = torch.cat([shard.out.transpose(0, 1).contiguous() for shard in shards], dim=0)
        return {
            "qkvzab": gdn.pack_projection_columns(shards),
            "out": out.reshape(1, 1, gdn.VALUE_WIDTH, gdn.HIDDEN_SIZE),
        }
    if module == "qsa":
        from models.demos.blackhole.qwen38_flash_next.ttnn import qsa

        prefix = f"model.language_model.layers.{layer_index}.self_attn."
        qg = checkpoint.tensor(prefix + "q_proj.weight").transpose(0, 1)
        k = qsa._expanded_pair_kv(checkpoint.tensor(prefix + "k_proj.weight")).transpose(0, 1)
        v = qsa._expanded_pair_kv(checkpoint.tensor(prefix + "v_proj.weight")).transpose(0, 1)
        out = checkpoint.tensor(prefix + "o_proj.weight").transpose(0, 1)
        kv = torch.cat(
            [
                torch.cat(
                    [k[:, d * QSA_HEAD_DIM : (d + 1) * QSA_HEAD_DIM], v[:, d * QSA_HEAD_DIM : (d + 1) * QSA_HEAD_DIM]],
                    dim=1,
                )
                for d in range(MESH_SHAPE[1])
            ],
            dim=1,
        )
        return {
            "qg": qg.reshape(1, 1, HIDDEN, 2 * qsa.QUERY_WIDTH),
            "k": k.reshape(1, 1, HIDDEN, MESH_SHAPE[1] * QSA_HEAD_DIM),
            "v": v.reshape(1, 1, HIDDEN, MESH_SHAPE[1] * QSA_HEAD_DIM),
            "kv": kv.reshape(1, 1, HIDDEN, MESH_SHAPE[1] * 2 * QSA_HEAD_DIM),
            "out": out.reshape(1, 1, qsa.QUERY_WIDTH, HIDDEN),
        }
    if module == "moe":
        from models.demos.blackhole.qwen38_flash_next.tt.moe import Qwen38MoEWeights
        from models.demos.blackhole.qwen38_flash_next.ttnn import moe

        source = Qwen38MoEWeights(checkpoint, placement, layer_index=layer_index)
        gate = source.shared_gate_proj.transpose(0, 1)
        up = source.shared_up_proj.transpose(0, 1)
        local = MOE_LOCAL_INTERMEDIATE
        gate_up = torch.cat(
            [
                torch.cat([gate[:, d * local : (d + 1) * local], up[:, d * local : (d + 1) * local]], dim=1)
                for d in range(MESH_SHAPE[1])
            ],
            dim=1,
        )
        return {
            "shared_gate": gate.reshape(1, 1, HIDDEN, moe.INTERMEDIATE_SIZE),
            "shared_up": up.reshape(1, 1, HIDDEN, moe.INTERMEDIATE_SIZE),
            "shared_gate_up": gate_up.reshape(1, 1, HIDDEN, MESH_SHAPE[1] * 2 * local),
            "shared_down": source.shared_down_proj.transpose(0, 1).reshape(1, 1, moe.INTERMEDIATE_SIZE, HIDDEN),
            "shared_scalar_gate": source.shared_scalar_gate.transpose(0, 1).reshape(1, 1, HIDDEN, 1),
        }
    raise ValueError(f"prefill dense module must be one of {MODULES}, got {module!r}")


def _cache_dir(
    module: str, checkpoint, mesh_contract, cache_root, *, layer_index: int, tt_metal_sha: str, block: str | None
) -> Path:
    """The module's own per-layer cache directory (the decode weights' tensorbins live there under other names)."""

    if module == "gr":
        from models.demos.blackhole.qwen38_flash_next.ttnn import gr

        return gr._cache_dir(cache_root, checkpoint, mesh_contract, tt_metal_sha, "backbone", layer_index, block)
    if module == "gdn":
        from models.demos.blackhole.qwen38_flash_next.ttnn import gdn

        return gdn._cache_directory(cache_root, checkpoint, mesh_contract, layer_index, tt_metal_sha)
    if module == "qsa":
        from models.demos.blackhole.qwen38_flash_next.ttnn import qsa

        return qsa._cache_dir(cache_root, checkpoint, mesh_contract, tt_metal_sha, "backbone", layer_index)
    from models.demos.blackhole.qwen38_flash_next.ttnn import moe

    return moe._layer_cache_dir(cache_root, checkpoint, mesh_contract, tt_metal_sha, "backbone", layer_index)


def build_prefill_dense_weights(
    policy: Qwen38PrefillDensePolicy,
    module: str,
    mesh_device,
    mesh_contract,
    checkpoint,
    placement,
    cache_root,
    *,
    layer_index: int,
    tt_metal_sha: str,
    block: str | None = None,
    weight_dtype: str = "bf16",
) -> Qwen38TTNNPrefillDenseWeights:
    """The resident prefill weights of one backbone module under ``policy`` (its fused siblings in ``weight_dtype``,
    the module's dense weight format): nothing is read or allocated when the plan is empty (grid today with dtype
    bf16, or a module the policy leaves alone: the gated residual and the GDN hold no fused sibling)."""

    plan = policy.resident_plan(module, weight_dtype)
    if not plan:
        return Qwen38TTNNPrefillDenseWeights()
    if module == "gr" and block not in ("attn", "mlp"):
        raise ValueError(f"the gated residual's prefill weights need block 'attn' or 'mlp', got {block!r}")
    mesh_contract.validate_mesh(mesh_device)
    hosts = _host_blocks(module, checkpoint, placement, layer_index=layer_index, block=block)
    cache_dir = _cache_dir(
        module, checkpoint, mesh_contract, cache_root, layer_index=layer_index, tt_metal_sha=tt_metal_sha, block=block
    )
    entries = {}
    for spec in plan:
        entries[spec.name] = (_upload(mesh_device, hosts[spec.name], spec, cache_dir), spec)
    return Qwen38TTNNPrefillDenseWeights(entries)
