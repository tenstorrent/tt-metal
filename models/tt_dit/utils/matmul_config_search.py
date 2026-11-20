# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Find one `MinimalMatmulConfig` per matmul shape of a model that serves every row count.

A model whose rows follow its input size (image or video tokens, reference images) runs each of
its matmuls at many M. One block config and one pinned grid orientation per matmul shape compile
one program for all of them. This tool times configs on one chip and picks the configs of all the
step's matmuls together, so that the step's matmul time at its worst input is as close as possible
to what the fastest measured config of every call at its rows would give.

    python -m models.tt_dit.utils.matmul_config_search SPEC DB.tsv

The specs of the tt_dit models' presets are in `matmul_search_specs/` next to this file, for example
`models/tt_dit/utils/matmul_search_specs/spec_wan_bh_2x4.json`.

DB.tsv holds every measurement ever taken, of any spec, machine or model. A search uses the points
that match its spec and measures only the ones it is missing, so running it again with a narrower
spec, for example a smaller range of input sizes, costs only the points that are new for it.

SPEC is a JSON file describing one denoising step on one machine::

    {
      "arch": "blackhole",
      "grid": [11, 10],
      "variables": {"M": {"start": 32, "stop": 10240, "step": 32}, "refs": [0, 1, 2]},
      "constants": {"T": 256, "R": 4096},
      "l1_limit": 1497344,
      "matmuls": [
        {"name": "addcmul", "K": 3072, "N": 1536, "op": "addcmul", "calls": [["M", 5], ["T", 5]]},
        {"name": "split qkv", "K": 3072, "N": 4608, "op": "split",
         "calls": [["M", 5], ["T", 5], ["M + refs * min(M, R) + T", 20]]},
        {"name": "ff out", "K": 4608, "N": 3072, "min_rows": 129, "calls": [["M", 5], ["T", 5]]}
      ]
    }

- `arch` (`wormhole_b0` or `blackhole`) is the machine the spec is for; the search refuses to run
  on another one.
- `grid` is the core grid the model passes to the op.
- `variables` are the inputs the step depends on, each a range (`start`, `stop`, `step`) or a list
  of values. The step is evaluated at every combination of their values.
- `calls` lists each call per step as (rows, count). Rows are an expression of the variables and
  `constants` (`+ - * //`, `min`, `max` and integers), padded up to whole tiles.
- `min_rows` (default 1) drops calls with fewer rows, such as those that run as a 1D matmul.
- `op` is `plain` (default, `minimal_matmul`), `split` (`minimal_matmul_split`, `chunks` default 3)
  or `addcmul` (`dit_minimal_matmul_addcmul_fused`). What the model fuses into the matmul is timed
  with it:
  - `bias` (default false) adds a bias of width N.
  - `activation` (plain and split) fuses `gelu`, `gelu_fast` or `gelu_tanh`, as `Linear` does for
    its `activation_fn` of that name.
  - `fuse_swiglu` (plain) fuses the SwiGLU of a packed [gate | up] weight.
  - `concat` (plain) is the K of the two inputs of a fused concat, `[K_prefix, K_suffix]`, which
    must add up to K: `minimal_matmul([prefix, suffix], weight)`.
- `math_fidelity`, `math_approx_mode`, `input_dtype`, `weight_dtype` and `output_dtype` set how a
  matmul is computed, as the model's layer does. Given on the spec, they are the defaults of its
  matmuls; the defaults of those are HiFi2, false and bfloat16.
- `l1_limit` (optional) is the lowest L1 address the model keeps occupied while the matmul runs: its
  `l1_small` region, CCL semaphores and the like. L1 from there up is reserved before timing, so a
  config whose static circular buffers would clash with the model's buffers fails here as it does
  in the model and is never picked. It must be a multiple of 32.
- `fp32_dest_acc_en` and `packer_l1_acc` (default true) are the compute kernel settings of all
  matmuls.

Matmuls of the same kind (op, K, N, fusions, fidelity and dtypes) share their measurements, and
matmuls with the same K and N share one config, as the model looks configs up by grid, K and N. The
search times configs in both grid orientations (`transpose_core_grid` False and True):

1. a coarse grid of block sizes at the ends of each kind's rows and at the powers of two between,
2. a local search: the neighbours of the best configs of each orientation so far, at the same row
   counts, repeated until none is new or for at most MAX_LOCAL_STEPS steps,
3. the best `--top` configs of each orientation at every row count the step uses, then the
   selection (below), then the neighbours of the selected configs at the sampled rows. Any of
   those that ranks among the best is timed at every row count and the selection is repeated,
   until the selected configs have no untried neighbours.

A config compiles once and is then reused at every M, so most of the time goes to compiling the
candidates. The measurements run in worker subprocesses of bounded size, because the device
profiler drops programs after about 87,000 in one process. One worker runs on each PCIe chip at
once, or on each chip given with `--devices`. A new config is measured on one chip only, so that it
compiles once, and the row counts of configs already measured are dealt to the workers whole. The
chips must be of one type, since a config's times then come from any of them. A chip behind
another over Ethernet, such as the second chip of an n300, cannot have a worker of its own. A
worker that dies or hangs on a batch of configs is replaced, and the batch is retried one config at
a time; a config that kills a worker on its own is recorded as failed. Reset the device if workers
keep failing.

The selection picks one config and orientation per shape by coordinate descent, with moves of one
and then of two shapes at a time. It minimises the step's extra matmul time at the worst input
against every call on the fastest config measured at every row count for its rows, then the mean
over the inputs, then the sum of each matmul's own worst loss, which decides for matmuls too small
to move the step time. Times are device kernel durations from the device profiler, or with
`--metric host` the host wall time per op over back-to-back trace executions. Use the host metric
on Wormhole: its profiler converts cycles at a clock that is wrong while the chip throttles.
"""

from __future__ import annotations

import argparse
import ast
import csv
import fcntl
import io
import itertools
import json
import math
import operator
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

MODULE = "models.tt_dit.utils.matmul_config_search"

TILE = 32
MAX_BLOCK = 16
L1_ALIGNMENT = 32
REPS = 5
# A trace repeats the op until it covers this much device time, at most MAX_REPEATS times, so the
# host time per op is not dominated by the dispatch of each trace execution.
HOST_WINDOW_NS = 400_000
MAX_REPEATS = 32
# Programs per worker process, well below where the profiler starts dropping programs.
PROGRAMS_PER_PROCESS = 50_000
# Configs compiled per batch, and programs between profiler reads (the device keeps at most 1,000).
CONFIGS_PER_BATCH = 48
PROGRAMS_PER_READ = 800
# The coarse pass and the local search time a kind at the ends of its rows and at the row counts
# nearest to TILE times each power of two in between. Searches over different ranges share those.
SAMPLE_BASE = 2
# The local search refines this many of the best configs of each orientation per step.
SEEDS_PER_LAYOUT = 4
MAX_LOCAL_STEPS = 4
# Rounds of timing the neighbours of the selected configs.
MAX_SELECTION_ROUNDS = 6
# A worker whose results file has not grown for this long is taken to hang and is stopped.
WATCHDOG_SECONDS = 900
# Worker failures in a row, without any measurement in between, before the search gives up.
MAX_FAILURES = 3

WORKER_ENV = {
    "TT_METAL_DEVICE_PROFILER": "1",
    "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
    "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
    "TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY": "1",
    "TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES": "1",
}

# The mesh graph descriptor of one Blackhole chip, relative to the tt-metal checkout.
SINGLE_BLACKHOLE_MESH_GRAPH = (
    Path(__file__).resolve().parents[3] / "tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"
)

ARCHES = ("wormhole_b0", "blackhole")
ACTIVATIONS = ("gelu", "gelu_fast", "gelu_tanh")
FIDELITIES = ("LoFi", "HiFi2", "HiFi3", "HiFi4")
DTYPES = ("bfloat16", "float32", "bfloat8_b", "bfloat4_b")


# --- spec ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    m_block: int
    k_block: int
    n_block: int
    subblock_h: int
    subblock_w: int
    transpose: bool

    def __str__(self) -> str:
        layout = "M > N" if self.transpose else "M <= N"
        return f"{self.m_block},{self.k_block},{self.n_block},{self.subblock_h},{self.subblock_w} ({layout})"

    def as_tuple(self) -> tuple[int, int, int, int, int, bool]:
        return (self.m_block, self.k_block, self.n_block, self.subblock_h, self.subblock_w, self.transpose)


@dataclass(frozen=True, order=True)
class Kind:
    """What a measurement depends on besides the rows, the config and the settings. Matmuls of the
    same kind share measurements."""

    op: str
    K: int
    N: int
    chunks: int = 1
    bias: bool = False
    activation: str = ""
    fuse_swiglu: bool = False
    concat_k: int = 0  # K of the first input of a fused concat, 0 without one
    math_fidelity: str = "HiFi2"
    math_approx_mode: bool = False
    input_dtype: str = "bfloat16"
    weight_dtype: str = "bfloat16"
    output_dtype: str = "bfloat16"

    @classmethod
    def from_strings(cls, v: dict[str, str]) -> Kind:
        return cls(
            op=v["op"],
            K=int(v["K"]),
            N=int(v["N"]),
            chunks=int(v["chunks"]),
            bias=v["bias"] == "True",
            activation=v["activation"],
            fuse_swiglu=v["fuse_swiglu"] == "True",
            concat_k=int(v["concat_k"]),
            math_fidelity=v["math_fidelity"],
            math_approx_mode=v["math_approx_mode"] == "True",
            input_dtype=v["input_dtype"],
            weight_dtype=v["weight_dtype"],
            output_dtype=v["output_dtype"],
        )

    def __str__(self) -> str:
        fused = [
            f"chunks {self.chunks}" if self.op == "split" else "",
            "bias" if self.bias else "",
            self.activation,
            "swiglu" if self.fuse_swiglu else "",
            f"concat {self.concat_k}+{self.K - self.concat_k}" if self.concat_k else "",
            self.math_fidelity if self.math_fidelity != "HiFi2" else "",
            "approx" if self.math_approx_mode else "",
        ]
        dtypes = (self.input_dtype, self.weight_dtype, self.output_dtype)
        if dtypes != ("bfloat16",) * 3:
            fused.append("/".join(dtypes))
        return " ".join([self.op, f"{self.K}x{self.N}", *(f for f in fused if f)])


KIND_FIELDS = tuple(f.name for f in fields(Kind))
# How a measurement was taken, besides its kind. Points that differ from the spec in any are ignored.
SETTINGS = ("arch", "grid_x", "grid_y", "l1_limit", "fp32_dest_acc_en", "packer_l1_acc")
DB_FIELDS: tuple[str, ...] = (
    *KIND_FIELDS,
    "M",
    "M_block",
    "K_block",
    "N_block",
    "subblock_h",
    "subblock_w",
    "transpose",
    *SETTINGS,
    "device_ns",
    "host_ns",
    "repeats",
    "error",
)


@dataclass(frozen=True)
class Matmul:
    name: str
    kind: Kind
    calls: tuple[tuple[str, int], ...]
    min_rows: int = 1

    @property
    def K(self) -> int:
        return self.kind.K

    @property
    def N(self) -> int:
        return self.kind.N

    @property
    def fuse_swiglu(self) -> bool:
        return self.kind.fuse_swiglu


# Settings of a matmul that the spec can also give, as the default of all its matmuls.
_KIND_DEFAULTS = ("math_fidelity", "math_approx_mode", "input_dtype", "weight_dtype", "output_dtype")
_MATMUL_KEYS = {
    "name",
    "K",
    "N",
    "calls",
    "op",
    "chunks",
    "bias",
    "activation",
    "fuse_swiglu",
    "concat",
    "min_rows",
    *_KIND_DEFAULTS,
}
_SPEC_KEYS = {
    "arch",
    "grid",
    "variables",
    "constants",
    "l1_limit",
    "fp32_dest_acc_en",
    "packer_l1_acc",
    "matmuls",
    *_KIND_DEFAULTS,
}

Point = tuple[tuple[str, int], ...]  # one value per variable, as (name, value) pairs


@dataclass(frozen=True)
class Spec:
    arch: str
    grid: tuple[int, int]
    points: tuple[Point, ...]
    matmuls: tuple[Matmul, ...]
    constants: dict[str, int] = field(default_factory=dict)
    l1_limit: int | None = None
    fp32_dest_acc_en: bool = True
    packer_l1_acc: bool = True

    @classmethod
    def load(cls, path: str | Path) -> Spec:
        raw = json.loads(Path(path).read_text())
        _check_keys("spec", raw, _SPEC_KEYS, required={"arch", "grid", "variables", "matmuls"})
        if raw["arch"] not in ARCHES:
            msg = f"arch must be one of {', '.join(ARCHES)}, got {raw['arch']!r}"
            raise ValueError(msg)
        defaults = {key: raw[key] for key in _KIND_DEFAULTS if key in raw}
        matmuls = []
        for m in raw["matmuls"]:
            _check_keys(f"matmul {m.get('name')!r}", m, _MATMUL_KEYS, required={"name", "K", "N", "calls"})
            matmuls.append(
                Matmul(
                    name=str(m["name"]),
                    kind=_parse_kind({**defaults, **m}),
                    calls=tuple((str(rows), int(count)) for rows, count in m["calls"]),
                    min_rows=max(1, int(m.get("min_rows", 1))),
                )
            )
        names = [m.name for m in matmuls]
        if len(set(names)) != len(names):
            msg = f"matmul names must be unique, got {names}"
            raise ValueError(msg)
        l1_limit = raw.get("l1_limit")
        if l1_limit is not None and int(l1_limit) % L1_ALIGNMENT:
            msg = f"l1_limit {l1_limit} must be a multiple of the L1 alignment, {L1_ALIGNMENT}"
            raise ValueError(msg)
        spec = cls(
            arch=raw["arch"],
            grid=(int(raw["grid"][0]), int(raw["grid"][1])),
            points=_points(raw["variables"]),
            matmuls=tuple(matmuls),
            constants={str(k): int(v) for k, v in raw.get("constants", {}).items()},
            l1_limit=int(l1_limit) if l1_limit is not None else None,
            fp32_dest_acc_en=bool(raw.get("fp32_dest_acc_en", True)),
            packer_l1_acc=bool(raw.get("packer_l1_acc", True)),
        )
        for m in spec.matmuls:
            if not spec.needed_rows(m):
                msg = f"{m.name}: no call has at least min_rows ({m.min_rows}) rows at any input"
                raise ValueError(msg)
        return spec

    @property
    def max_subblock_area(self) -> int:
        """Destination register tiles: 4 with fp32 accumulation, 8 without."""
        return 4 if self.fp32_dest_acc_en else 8

    def settings(self) -> dict[str, str]:
        """The SETTINGS columns as they are written to the database."""
        return {
            "arch": self.arch,
            "grid_x": str(self.grid[0]),
            "grid_y": str(self.grid[1]),
            "l1_limit": str(self.l1_limit) if self.l1_limit is not None else "",
            "fp32_dest_acc_en": str(self.fp32_dest_acc_en),
            "packer_l1_acc": str(self.packer_l1_acc),
        }

    def calls(self, matmul: Matmul, point: Point) -> list[tuple[int, int]]:
        """(rows, count) of `matmul`'s calls in the step at `point`, rows padded to tiles."""
        names = {**self.constants, **dict(point)}
        out = []
        for expr, count in matmul.calls:
            rows = _evaluate(expr, names)
            if rows >= matmul.min_rows:
                out.append((math.ceil(rows / TILE) * TILE, count))
        return out

    def needed_rows(self, matmul: Matmul) -> list[int]:
        return sorted({rows for p in self.points for rows, _ in self.calls(matmul, p)})


def _points(variables: dict) -> tuple[Point, ...]:
    if not variables:
        msg = "variables must name at least one input of the step"
        raise ValueError(msg)
    values = {}
    for name, v in variables.items():
        if isinstance(v, dict):
            _check_keys(f"variable {name!r}", v, {"start", "stop", "step"}, required={"start", "stop", "step"})
            values[name] = list(range(int(v["start"]), int(v["stop"]) + 1, int(v["step"])))
        else:
            values[name] = sorted({int(x) for x in v})
        if not values[name]:
            msg = f"variable {name!r} has no values"
            raise ValueError(msg)
    names = list(values)
    return tuple(tuple(zip(names, combo, strict=True)) for combo in itertools.product(*values.values()))


def _parse_kind(m: dict) -> Kind:
    name, op = m["name"], m.get("op", "plain")
    only = {"chunks": ("split",), "activation": ("plain", "split"), "fuse_swiglu": ("plain",), "concat": ("plain",)}
    if op not in ("plain", "split", "addcmul"):
        msg = f"{name}: op must be plain, split or addcmul, got {op!r}"
        raise ValueError(msg)
    for key, ops in only.items():
        if m.get(key) and op not in ops:
            msg = f"{name}: {key} applies to op {' or '.join(ops)} only"
            raise ValueError(msg)
    activation = str(m.get("activation") or "")
    if activation and activation not in ACTIVATIONS:
        msg = f"{name}: activation must be one of {', '.join(ACTIVATIONS)}, got {activation!r}"
        raise ValueError(msg)
    if activation and m.get("fuse_swiglu"):
        msg = f"{name}: activation and fuse_swiglu exclude each other"
        raise ValueError(msg)
    K = int(m["K"])
    concat = [int(k) for k in m.get("concat") or ()]
    if concat and (len(concat) != 2 or sum(concat) != K or any(k % TILE for k in concat)):
        msg = f"{name}: concat must be two multiples of {TILE} that add up to K ({K}), got {concat}"
        raise ValueError(msg)
    fidelity = str(m.get("math_fidelity", "HiFi2"))
    if fidelity not in FIDELITIES:
        msg = f"{name}: math_fidelity must be one of {', '.join(FIDELITIES)}, got {fidelity!r}"
        raise ValueError(msg)
    dtypes = {key: str(m.get(key, "bfloat16")) for key in ("input_dtype", "weight_dtype", "output_dtype")}
    for key, value in dtypes.items():
        if value not in DTYPES:
            msg = f"{name}: {key} must be one of {', '.join(DTYPES)}, got {value!r}"
            raise ValueError(msg)
    return Kind(
        op=op,
        K=K,
        N=int(m["N"]),
        chunks=int(m.get("chunks", 3 if op == "split" else 1)),
        bias=bool(m.get("bias", False)),
        activation=activation,
        fuse_swiglu=bool(m.get("fuse_swiglu", False)),
        concat_k=concat[0] if concat else 0,
        math_fidelity=fidelity,
        math_approx_mode=bool(m.get("math_approx_mode", False)),
        **dtypes,
    )


def _check_keys(what: str, raw: dict, allowed: set[str], *, required: set[str]) -> None:
    unknown, missing = set(raw) - allowed, required - set(raw)
    if unknown or missing:
        msg = f"{what}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}"
        raise ValueError(msg)


_OPERATORS: dict[type[ast.operator], Callable[[int, int], int]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
}
_FUNCTIONS: dict[str, Callable[..., int]] = {"min": min, "max": max}


def _evaluate(expr: str, names: dict[str, int]) -> int:
    def walk(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and node.args
            and not node.keywords
        ):
            return _FUNCTIONS[node.func.id](*(walk(a) for a in node.args))
        msg = f"rows expression {expr!r}: only + - * //, min, max, integers and {sorted(names)} are allowed"
        raise ValueError(msg)

    return walk(ast.parse(expr, mode="eval"))


# --- candidates ---------------------------------------------------------------------------------


def _subblocks(m_block: int, n_block: int, max_area: int) -> list[tuple[int, int]]:
    """The largest-area subblocks that divide the block."""
    fits = [
        (h, w)
        for h in range(1, max_area + 1)
        for w in range(1, max_area + 1)
        if h * w <= max_area and m_block % h == 0 and n_block % w == 0
    ]
    area = max(h * w for h, w in fits)
    return [(h, w) for h, w in fits if h * w == area]


def _n_blocks(matmul: Matmul) -> list[int]:
    """N blocks up to MAX_BLOCK and the N tiles, even ones only with the fused SwiGLU."""
    n_tiles = math.ceil(matmul.N / TILE)
    return [n for n in range(1, min(MAX_BLOCK, n_tiles) + 1) if not (matmul.fuse_swiglu and n % 2)]


def _k_blocks(matmul: Matmul) -> list[int]:
    """K blocks that divide the K tiles, from 2 up (K_block 1 only when K is one tile)."""
    k_tiles = math.ceil(matmul.K / TILE)
    divisors = [k for k in range(1, min(MAX_BLOCK, k_tiles) + 1) if k_tiles % k == 0]
    return [k for k in divisors if k > 1] or divisors


def _nearest(values: list[int], target: int) -> int:
    return min(values, key=lambda v: (abs(v - target), v))


def coarse_candidates(matmul: Matmul, max_area: int) -> list[Config]:
    """A coarse grid over the block sizes in both orientations, for the local search to refine."""
    n_blocks, k_blocks = _n_blocks(matmul), _k_blocks(matmul)
    ms = (4, 8, 12, 16)
    ks = sorted({_nearest(k_blocks, 4), _nearest(k_blocks, 8)})
    ns = sorted({_nearest(n_blocks, n) for n in (4, 8, 12, 16)})
    out = []
    for m, k, n in itertools.product(ms, ks, ns):
        h, w = _subblocks(m, n, max_area)[0]
        out.extend(Config(m, k, n, h, w, t) for t in (False, True))
    return out


def neighbours(config: Config, matmul: Matmul, max_area: int) -> Iterator[Config]:
    """The config's other largest-area subblocks, and configs one or two steps away in M_block or
    N_block or one step away in K_block, in the same orientation."""
    n_blocks, k_blocks = _n_blocks(matmul), _k_blocks(matmul)
    c = config

    def steps(values: list[int], value: int, reach: int) -> list[int]:
        i = values.index(_nearest(values, value))
        return [values[j] for j in range(i - reach, i + reach + 1) if j != i and 0 <= j < len(values)]

    for h, w in _subblocks(c.m_block, c.n_block, max_area):
        yield Config(c.m_block, c.k_block, c.n_block, h, w, c.transpose)
    blocks = [(m, c.k_block, c.n_block) for m in steps(list(range(1, MAX_BLOCK + 1)), c.m_block, 2)]
    blocks += [(c.m_block, k, c.n_block) for k in steps(k_blocks, c.k_block, 1)]
    blocks += [(c.m_block, c.k_block, n) for n in steps(n_blocks, c.n_block, 2)]
    for m, k, n in blocks:
        h, w = _subblocks(m, n, max_area)[0]
        yield Config(m, k, n, h, w, c.transpose)


# --- database -----------------------------------------------------------------------------------


Key = tuple[Kind, int, Config]  # (matmul kind, rows, config)


def _parse_row(row: dict[str, str]) -> Key:
    config = Config(
        int(row["M_block"]),
        int(row["K_block"]),
        int(row["N_block"]),
        int(row["subblock_h"]),
        int(row["subblock_w"]),
        row["transpose"] == "True",
    )
    return Kind.from_strings(row), int(row["M"]), config


class Results:
    """The points of the database taken with the spec's settings, keyed by (kind, rows, config)."""

    def __init__(self, path: Path, spec: Spec) -> None:
        self.done: set[Key] = set()
        self.failed: set[tuple[Kind, Config]] = set()
        self.device: dict[Key, float] = {}
        self.host: dict[Key, float] = {}
        if not path.exists():
            return
        settings = spec.settings()
        with path.open(newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if None in row or None in row.values():
                    continue  # a row cut short by a killed worker, or run into the next one
                try:
                    if any(row[s] != settings[s] for s in SETTINGS):
                        continue
                    key = _parse_row(row)
                    times = None if row["error"] else (float(row["device_ns"]), float(row["host_ns"]))
                except (KeyError, TypeError, ValueError):
                    continue  # a row cut short by a killed worker
                self.done.add(key)
                if times is None:
                    self.failed.add((key[0], key[2]))
                else:
                    self.device[key], self.host[key] = times

    def times(self, metric: str) -> dict[Key, float]:
        return self.host if metric == "host" else self.device


def _append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Append rows in one write, flushed to disk, with the header if the file is new or empty.

    The file is locked while writing, so workers on other devices never interleave their rows.
    """
    with path.open("a+", newline="") as f:  # readable, to check the last byte
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, DB_FIELDS, delimiter="\t", lineterminator="\n")
            size = os.fstat(f.fileno()).st_size
            if size == 0:
                writer.writeheader()
            elif os.pread(f.fileno(), 1, size - 1) != b"\n":
                buffer.write("\n")  # ends a row cut short by a killed worker, which is then skipped
            writer.writerows([{k: str(v).replace("\t", " ").replace("\n", " ") for k, v in r.items()} for r in rows])
            f.write(buffer.getvalue())
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _row(spec: Spec, kind: Kind, rows: int, config: Config, **values: object) -> dict[str, object]:
    c = config
    return {
        **asdict(kind),
        "M": rows,
        "M_block": c.m_block,
        "K_block": c.k_block,
        "N_block": c.n_block,
        "subblock_h": c.subblock_h,
        "subblock_w": c.subblock_w,
        "transpose": c.transpose,
        **spec.settings(),
        **values,
    }


# --- measurement --------------------------------------------------------------------------------


@dataclass
class Job:
    """Configs of one matmul kind to time at one row count."""

    kind: Kind
    rows: int
    configs: list[Config]
    one_at_a_time: bool = False

    def to_json(self) -> dict:
        return {**asdict(self), "configs": [c.as_tuple() for c in self.configs]}

    @classmethod
    def from_json(cls, raw: dict) -> Job:
        return cls(
            kind=Kind(**raw["kind"]),
            rows=int(raw["rows"]),
            configs=[Config(*c) for c in raw["configs"]],
            one_at_a_time=bool(raw["one_at_a_time"]),
        )


def _jobs(results: Results, wanted: Iterable[Key], suspects: set[Key]) -> list[Job]:
    """The wanted points not yet measured, skipping configs that failed at any row count (a program
    does not depend on M), grouped by matmul kind and rows. Suspects run one at a time."""
    groups: dict[tuple[Kind, int, bool], list[Config]] = defaultdict(list)
    seen = set()
    for key in wanted:
        kind, rows, config = key
        if key in results.done or key in seen or (kind, config) in results.failed:
            continue
        seen.add(key)
        groups[(kind, rows, key in suspects)].append(config)
    return [Job(kind, rows, configs, alone) for (kind, rows, alone), configs in sorted(groups.items())]


def _split_jobs(jobs: list[Job], n: int, compiled: set[tuple[Kind, Config]]) -> list[list[Job]]:
    """The jobs split into `n` shares.

    A config not measured before is in one share only, so that it is compiled once; a kind's new
    configs are dealt round robin. A config measured before is in the kernel cache on disk, so
    its points are dealt by job instead: a job keeps all of them, and every job costs a worker the
    same to set up (its inputs and a profiler read) however few configs it times.
    """
    share_of: dict[tuple[Kind, Config], int] = {}
    dealt = 0  # carried across kinds, so that no share takes every kind's leftover
    for kind in sorted({j.kind for j in jobs}):
        new = sorted(
            {c for j in jobs if j.kind == kind for c in j.configs if (kind, c) not in compiled}, key=Config.as_tuple
        )
        share_of.update({(kind, c): (dealt + i) % n for i, c in enumerate(new)})
        dealt += len(new)
    shares: list[list[Job]] = [[] for _ in range(n)]
    sizes = [0] * n
    for j in jobs:
        old = [c for c in j.configs if (j.kind, c) not in share_of]
        if old:
            i = sizes.index(min(sizes))  # the share with the fewest points so far
            shares[i].append(Job(j.kind, j.rows, old, j.one_at_a_time))
            sizes[i] += len(old)
        for i in range(n):
            configs = [c for c in j.configs if share_of.get((j.kind, c)) == i]
            if configs:
                shares[i].append(Job(j.kind, j.rows, configs, j.one_at_a_time))
                sizes[i] += len(configs)
    return shares


def run_jobs(spec_path: Path, spec: Spec, db: Path, wanted: list[Key], devices: list[int]) -> None:
    """Measure the `wanted` points that are not in the database yet, in worker subprocesses, one
    per device at a time."""
    inflights = {d: db.with_name(f"{db.name}.inflight.{d}.json") for d in devices}
    suspects: set[Key] = set()
    # Rounds in a row in which a device's worker measured nothing.
    failures = dict.fromkeys(devices, 0)
    while True:
        results = Results(db, spec)
        jobs = _jobs(results, wanted, suspects)
        pending = sum(len(j.configs) for j in jobs)
        if not pending:
            return
        print(f"{pending} points to measure", flush=True)
        for inflight in inflights.values():
            inflight.unlink(missing_ok=True)  # left by a driver that was stopped
        compiled = {(kind, config) for kind, _, config in results.done}
        shares = dict(zip(devices, _split_jobs(jobs, len(devices), compiled), strict=True))
        shares = {d: share for d, share in shares.items() if share}
        job_files = {}
        try:
            for d, share in shares.items():
                with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                    json.dump([j.to_json() for j in share], f)
                job_files[d] = f.name
            _run_workers(
                {d: [str(spec_path), job_files[d], str(db), str(inflights[d]), str(d)] for d in shares},
                {d: inflights[d] for d in shares},
                spec.arch,
                alone=len(devices) == 1,
            )
        finally:
            for name in job_files.values():
                os.unlink(name)
        done = Results(db, spec).done
        for d, share in shares.items():
            if any((j.kind, j.rows, c) in done for j in share for c in j.configs):
                failures[d] = 0
            else:
                failures[d] += 1
            inflight = inflights[d]
            if not inflight.exists():
                continue
            batch = Job.from_json(json.loads(inflight.read_text()))
            inflight.unlink()
            keys = [(batch.kind, batch.rows, c) for c in batch.configs if (batch.kind, batch.rows, c) not in done]
            if not keys:
                continue  # the worker was stopped after writing the batch's rows
            if len(keys) == 1:
                print(f"recording {batch.configs[0]} as failed: the worker died or hung on it alone", flush=True)
                _append_rows(
                    db, [_row(spec, *keys[0], device_ns="", host_ns="", repeats="", error="worker died or hung")]
                )
                # Not progress: a chip left wedged by a hang would otherwise fail one config after
                # another without ever reaching MAX_FAILURES.
            else:
                print(
                    f"the worker on device {d} died or hung in a batch of {len(keys)}; retrying them one at a time",
                    flush=True,
                )
                suspects.update(keys)
        stuck = [d for d, n in failures.items() if n >= MAX_FAILURES]
        if stuck:
            msg = (
                f"the workers on device {', '.join(map(str, stuck))} failed {MAX_FAILURES} rounds in a row without"
                " measuring anything; check or reset the device, or leave it out of --devices"
            )
            raise RuntimeError(msg)


def _run_workers(args: dict[int, list[str]], inflights: dict[int, Path], arch: str, *, alone: bool) -> None:
    """Run one worker per device at once.

    With several devices, each worker sees only its own chip (`TT_VISIBLE_DEVICES`), since a
    process opening one chip of several starts and locks all it sees. A single Blackhole chip of a
    multi-chip card is then a custom cluster to tt-metal, which wants a mesh graph descriptor, the
    one of a single chip. A lone worker sees every chip and opens its own among them.

    A worker writes its in-flight file before each batch and removes it after, so a worker whose
    file has not changed for WATCHDOG_SECONDS hangs and is stopped.
    """
    env = {"TT_METAL_OPERATION_TIMEOUT_SECONDS": "120", **os.environ, **WORKER_ENV}
    if arch == "blackhole" and not alone:
        env.setdefault("TT_MESH_GRAPH_DESC_PATH", str(SINGLE_BLACKHOLE_MESH_GRAPH))

    def state(path: Path) -> int | None:
        try:
            return path.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    procs: dict[int, subprocess.Popen] = {}
    try:
        for d, a in args.items():
            # Its own session, so a Ctrl-C reaches only the driver, which then stops the worker in order.
            procs[d] = subprocess.Popen(
                [sys.executable, "-m", MODULE, "_worker", *a],
                env=env if alone else {**env, "TT_VISIBLE_DEVICES": str(d)},
                start_new_session=True,
            )
        last = {d: (state(inflights[d]), time.monotonic()) for d in procs}
        running = set(procs)
        while running:
            for d in sorted(running):
                if procs[d].poll() is not None:
                    running.discard(d)
                    continue
                s = state(inflights[d])
                if s != last[d][0]:
                    last[d] = (s, time.monotonic())
                elif time.monotonic() - last[d][1] > WATCHDOG_SECONDS:
                    print(f"no progress on device {d} for {WATCHDOG_SECONDS} s, stopping its worker", flush=True)
                    _stop(procs[d])
                    running.discard(d)
            time.sleep(5)
    finally:
        # A worker holds its device; it must not outlive the driver or a hang.
        _stop(*procs.values())


def _stop(*procs: subprocess.Popen) -> None:
    """Terminate the processes, all at once, and kill those still running after 60 s."""
    running = [p for p in procs if p.poll() is None]
    for p in running:
        p.terminate()
    deadline = time.monotonic() + 60
    for p in running:
        try:
            p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def _worker(spec_path: Path, jobs_path: Path, db: Path, inflight: Path, device_id: int) -> None:
    import ttnn

    spec = Spec.load(spec_path)
    jobs = [Job.from_json(j) for j in json.loads(jobs_path.read_text())]
    # The driver hides the other chips from a worker, unless it runs alone (see _run_workers).
    ids = [] if "TT_VISIBLE_DEVICES" in os.environ else [device_id]
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=ids, trace_region_size=8 << 20)
    try:
        arch = device.arch().name.lower()
        if arch != spec.arch:
            msg = f"the spec is for {spec.arch}, but the device is {arch}"
            raise RuntimeError(msg)
        compute_configs = {}
        grid = ttnn.CoreCoord(*spec.grid)
        reserve = _reserve_l1(ttnn, device, spec.l1_limit)
        programs, t0 = 0, time.time()

        def tensor(dtype: str, *shape: int) -> ttnn.Tensor:
            # On the device: a host tensor of 16k rows takes most of a second to make and copy.
            return ttnn.rand(
                list(shape), device=device, dtype=getattr(ttnn, dtype), layout=ttnn.TILE_LAYOUT, low=-1.0, high=1.0
            )

        weight, weight_key = None, None
        for job in jobs:
            kind = job.kind
            K, N = kind.K, kind.N
            if weight is None or weight_key != (K, N, kind.weight_dtype):
                if weight is not None:
                    weight.deallocate()
                weight, weight_key = tensor(kind.weight_dtype, K, N), (K, N, kind.weight_dtype)
            x = kind.input_dtype
            if kind.concat_k:
                inputs = [tensor(x, 1, 1, job.rows, kind.concat_k), tensor(x, 1, 1, job.rows, K - kind.concat_k)]
            else:
                inputs = [tensor(x, 1, 1, job.rows, K)]
            bias = tensor(kind.weight_dtype, 1, N) if kind.bias else None
            addcmul = (tensor(x, 1, 1, job.rows, N), tensor(x, 1, 1, 1, N)) if kind.op == "addcmul" else ()
            _read_programs(ttnn, device)  # drops the programs that made the tensors from the profiler
            settings = (kind.math_fidelity, kind.math_approx_mode)
            if settings not in compute_configs:
                compute_configs[settings] = ttnn.init_device_compute_kernel_config(
                    device.arch(),
                    math_fidelity=getattr(ttnn.MathFidelity, kind.math_fidelity),
                    math_approx_mode=kind.math_approx_mode,
                    fp32_dest_acc_en=spec.fp32_dest_acc_en,
                    packer_l1_acc=spec.packer_l1_acc,
                )
            run = _matmul_runner(
                ttnn,
                kind,
                inputs,
                weight,
                bias,
                addcmul,
                grid=grid,
                compute=compute_configs[settings],
                dtype=getattr(ttnn, kind.output_dtype),
            )
            size = 1 if job.one_at_a_time else CONFIGS_PER_BATCH
            for start in range(0, len(job.configs), size):
                batch = job.configs[start : start + size]
                if programs + len(batch) * (1 + REPS * MAX_REPEATS) > PROGRAMS_PER_PROCESS:
                    return  # the driver starts a new worker for the rest
                inflight.write_text(json.dumps(Job(job.kind, job.rows, batch, job.one_at_a_time).to_json()))
                measured, ran = _time_batch(ttnn, device, run, batch)
                programs += ran
                _append_rows(
                    db,
                    [
                        _row(spec, job.kind, job.rows, c, device_ns=d, host_ns=h, repeats=r, error=e)
                        for c, (d, h, r, e) in zip(batch, measured, strict=True)
                    ],
                )
                inflight.unlink()
            for t in (*inputs, *addcmul, *([bias] if bias is not None else [])):
                t.deallocate()
            print(f"{kind}, {job.rows} rows: {len(job.configs)} configs, {time.time() - t0:.0f} s", flush=True)
        if reserve is not None:
            reserve.deallocate()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # a second SIGTERM must not skip closing the device
        ttnn.close_mesh_device(device)


def _matmul_runner(
    ttnn, kind: Kind, inputs, weight, bias, addcmul, *, grid, compute, dtype
) -> Callable[[Config], None]:
    """A function running the matmul once with a config and freeing its outputs."""
    # As `Linear` maps its `activation_fn` to the op's fused activation.
    activation = {
        "": None,
        "gelu": (ttnn.UnaryOpType.GELU, False),
        "gelu_fast": (ttnn.UnaryOpType.GELU, True),
        "gelu_tanh": ttnn.UnaryOpType.GELU_TANH,
    }[kind.activation]
    x = inputs if kind.concat_k else inputs[0]

    def run(c: Config) -> None:
        config = ttnn.MinimalMatmulConfig(
            M_block_size=c.m_block,
            K_block_size=c.k_block,
            N_block_size=c.n_block,
            subblock_h=c.subblock_h,
            subblock_w=c.subblock_w,
            compute_with_storage_grid_size=grid,
            transpose_core_grid=c.transpose,
        )
        if kind.op == "addcmul":
            outputs = [
                ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                    x,
                    weight,
                    1.0,
                    *addcmul,
                    bias_tensor=bias,
                    config=config,
                    dtype=dtype,
                    compute_kernel_config=compute,
                )
            ]
        elif kind.op == "split":
            outputs = ttnn.experimental.minimal_matmul_split(
                x,
                weight,
                chunks=kind.chunks,
                dim=-1,
                bias_tensor=bias,
                fused_activation=activation,
                config=config,
                dtype=dtype,
                compute_kernel_config=compute,
            )
        else:
            outputs = [
                ttnn.experimental.minimal_matmul(
                    input_tensor=x,
                    weight_tensor=weight,
                    bias_tensor=bias,
                    fused_activation=activation,
                    config=config,
                    dtype=dtype,
                    compute_kernel_config=compute,
                    fuse_swiglu=kind.fuse_swiglu,
                )
            ]
        for o in outputs:
            o.deallocate()

    return run


def _reserve_l1(ttnn, device, l1_limit: int | None):
    """Occupy L1 from `l1_limit` up on every core, as the model does while the matmul runs."""
    if l1_limit is None:
        return None
    view = ttnn.device.get_memory_view(device, ttnn.BufferType.L1)

    def allocate(size: int):
        # One page of `size` bytes per bank. L1 is allocated top-down, so it ends at the top.
        shape = ttnn.Shape([view.num_banks, size // 4])
        return ttnn.allocate_tensor_on_device(shape, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)

    probe = allocate(L1_ALIGNMENT)
    top = probe.buffer_address() + L1_ALIGNMENT
    probe.deallocate()
    if l1_limit >= top:
        msg = f"l1_limit {l1_limit} is not below the top of the allocatable L1, {top}"
        raise ValueError(msg)
    reserve = allocate(top - l1_limit)
    if reserve.buffer_address() != l1_limit:
        msg = f"the reserved L1 starts at {reserve.buffer_address()}, not at l1_limit {l1_limit}"
        raise RuntimeError(msg)
    return reserve


def _time_batch(ttnn, device, run, batch: list[Config]) -> tuple[list[tuple[str, str, str, str]], int]:
    """(device ns, host ns, repeats, error) per config, and the number of programs run.

    Each config runs once untraced, which compiles it, then as a trace that repeats the op enough
    times to cover HOST_WINDOW_NS of device time, executed REPS times. The device time is the
    fastest traced run. The host time is the wall time per op over the trace executions, which the
    repeats keep from being dominated by the dispatch of each execution.
    """
    errors: dict[Config, str] = {}
    valid = []
    for c in batch:
        try:
            run(c)
            valid.append(c)
        except Exception as e:  # noqa: BLE001 - an invalid config is recorded, not fatal
            errors[c] = _error_text(e)
    untraced = _read_programs(ttnn, device)
    if len(untraced) == len(valid):
        repeats = {
            c: max(1, min(MAX_REPEATS, math.ceil(HOST_WINDOW_NS / max(d, 1))))
            for c, d in zip(valid, untraced, strict=True)
        }
    else:
        print(f"warning: {len(untraced)} untraced programs for {len(valid)} configs; tracing each op once", flush=True)
        repeats = dict.fromkeys(valid, 1)

    host: dict[Config, float] = {}
    traced: list[Config] = []
    durations: list[float] = []
    pending = 0
    for c in valid:
        if pending + REPS * repeats[c] > PROGRAMS_PER_READ:
            durations += _read_programs(ttnn, device)
            pending = 0
        trace = None
        try:
            trace = ttnn.begin_trace_capture(device, cq_id=0)
            for _ in range(repeats[c]):
                run(c)
            ttnn.end_trace_capture(device, trace, cq_id=0)
            ttnn.synchronize_device(device)
            start = time.perf_counter()
            for _ in range(REPS):
                ttnn.execute_trace(device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            host[c] = (time.perf_counter() - start) / (REPS * repeats[c]) * 1e9
            traced.append(c)
            pending += REPS * repeats[c]
        except Exception as e:  # noqa: BLE001 - recorded, and the trace is released
            errors[c] = f"trace: {_error_text(e)}"
            # Only the programs of the configs traced since the last read count; the failed one's
            # partial runs, if any, come after them.
            durations += _read_programs(ttnn, device)[:pending]
            pending = 0
        finally:
            if trace is not None:
                ttnn.release_trace(device, trace)
    durations += _read_programs(ttnn, device)
    expected = sum(REPS * repeats[c] for c in traced)
    if len(durations) != expected:
        mismatch = f"the profiler returned {len(durations)} programs for {expected}"
        errors.update(dict.fromkeys(traced, mismatch))
        traced = []

    out, i = [], 0
    for c in batch:
        if c in traced:
            n = REPS * repeats[c]
            out.append((f"{min(durations[i : i + n]):.0f}", f"{host[c]:.0f}", str(repeats[c]), ""))
            i += n
        else:
            out.append(("", "", "", errors[c]))
    return out, len(batch) + sum(REPS * repeats[c] for c in valid)


def _error_text(e: Exception) -> str:
    """The message of a TT_THROW or TT_FATAL (the line after `info:`), else the first line."""
    lines = [line.strip() for line in str(e).splitlines() if line.strip()]
    if "info:" in lines and lines.index("info:") + 1 < len(lines):
        return lines[lines.index("info:") + 1][:300]
    return (lines[0] if lines else type(e).__name__)[:300]


def _read_programs(ttnn, device) -> list[float]:
    """Kernel durations (ns) of the programs run since the previous read, in execution order."""
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    programs = [p for ps in ttnn.get_latest_programs_perf_data().values() for p in ps]
    programs.sort(key=lambda p: p.program_execution_uid.runtime_id)
    return [p.program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration for p in programs]


# --- selection ----------------------------------------------------------------------------------


@dataclass
class Selection:
    configs: dict[str, Config]
    worst: float  # extra step matmul time at the worst point
    worst_point: Point
    mean: float  # mean over the points
    own_losses: dict[str, tuple[float, float]]  # per matmul, its worst and mean loss over its rows


def _total(arrays: list[np.ndarray]) -> np.ndarray:
    return np.sum(np.stack(arrays), axis=0)


def select(spec: Spec, results: Results, metric: str) -> Selection:
    """One config per matmul shape (K, N), minimising the worst extra step matmul time over the
    points, then the mean, then the sum of the matmuls' own worst losses."""
    times = results.times(metric)
    eligible: dict[str, set[Config]] = {}
    for m in spec.matmuls:
        rows = spec.needed_rows(m)
        configs = {c for (kind, _, c) in times if kind == m.kind}
        eligible[m.name] = {c for c in configs if all((m.kind, r, c) in times for r in rows)}
        if not eligible[m.name]:
            msg = f"{m.name}: no config is measured at every row count it needs"
            raise ValueError(msg)
    # The fastest per matmul and rows among the configs measured at every row count of the matmul,
    # so the baseline is the same set of configs at every point and not tighter at the rows the
    # search sampled. Per matmul, as two matmuls of one kind can need different rows.
    fastest = {
        (m.name, r): min(times[(m.kind, r, c)] for c in eligible[m.name])
        for m in spec.matmuls
        for r in spec.needed_rows(m)
    }
    calls = {m.name: [spec.calls(m, p) for p in spec.points] for m in spec.matmuls}
    floor = _total(
        [np.array([sum(n * fastest[(m.name, r)] for r, n in cs) for cs in calls[m.name]]) for m in spec.matmuls]
    )
    # Inputs at which no call reaches its matmul's min_rows have no matmul time to compare.
    timed = np.flatnonzero(floor > 0)
    if not len(timed):
        msg = "no input of the spec runs any of its matmuls"
        raise ValueError(msg)
    floor = floor[timed]
    own = {
        m.name: {
            c: [times[(m.kind, r, c)] / fastest[(m.name, r)] - 1 for r in spec.needed_rows(m)] for c in eligible[m.name]
        }
        for m in spec.matmuls
    }

    # The model looks configs up by (grid, K, N), so matmuls of one shape share a config.
    groups: dict[tuple[int, int], list[Matmul]] = defaultdict(list)
    for m in spec.matmuls:
        groups[(m.K, m.N)].append(m)
    shared = {}
    for shape, ms in groups.items():
        shared[shape] = sorted(set.intersection(*(eligible[m.name] for m in ms)), key=str)
        if not shared[shape]:
            msg = f"{' / '.join(m.name for m in ms)}: no config is measured at every row count of each"
            raise ValueError(msg)
    # Per shape and config, its matmuls' time in the step at every point, and their own worst losses.
    step_time = {
        shape: {
            c: _total([np.array([sum(n * times[(m.kind, r, c)] for r, n in cs) for cs in calls[m.name]]) for m in ms])[
                timed
            ]
            for c in shared[shape]
        }
        for shape, ms in groups.items()
    }
    own_worst = {
        shape: {c: sum(max(own[m.name][c]) for m in ms) for c in shared[shape]} for shape, ms in groups.items()
    }

    def score(choice: dict[tuple[int, int], Config]) -> tuple[float, float, float]:
        extra = _total([step_time[shape][c] for shape, c in choice.items()]) / floor - 1
        return (
            round(float(extra.max()), 4),
            round(float(extra.mean()), 5),
            sum(own_worst[shape][c] for shape, c in choice.items()),
        )

    shapes = list(groups)
    choice = {shape: min(shared[shape], key=lambda c, shape=shape: own_worst[shape][c]) for shape in shapes}
    current = score(choice)
    # Moves of one shape's config until none improves, then of two; an improving pair goes back to
    # single moves.
    moves = [[(shape,) for shape in shapes], list(itertools.combinations(shapes, 2))]
    level = 0
    while level < len(moves):
        improved = False
        for group in moves[level]:
            for configs in itertools.product(*(shared[shape] for shape in group)):
                trial = {**choice, **dict(zip(group, configs, strict=True))}
                s = score(trial)
                if s < current:
                    choice, current, improved = trial, s, True
        level = 0 if improved else level + 1

    extra = _total([step_time[shape][c] for shape, c in choice.items()]) / floor - 1
    configs = {m.name: choice[(m.K, m.N)] for m in spec.matmuls}
    return Selection(
        configs=configs,
        worst=float(extra.max()),
        worst_point=spec.points[int(timed[extra.argmax()])],
        mean=float(extra.mean()),
        own_losses={n: (max(own[n][c]), sum(own[n][c]) / len(own[n][c])) for n, c in configs.items()},
    )


def report(spec: Spec, metric: str, selection: Selection) -> str:
    worst_at = ", ".join(f"{name} = {value}" for name, value in selection.worst_point)
    lines = [
        f"{spec.arch}, grid {spec.grid[0]}x{spec.grid[1]}, {metric} time, {len(spec.points)} inputs",
        (
            "extra step matmul time against every call on the fastest config measured at every row count: "
            f"worst {selection.worst:.1%} ({worst_at}), mean {selection.mean:.1%}"
        ),
        "",
        "| matmul | kind | config | worst own loss | mean own loss |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for m in spec.matmuls:
        worst, mean = selection.own_losses[m.name]
        lines.append(f"| {m.name} | {m.kind} | {selection.configs[m.name]} | {worst:.1%} | {mean:.1%} |")
    lines += ["", "(grid_x, grid_y, K, N): (M_block, K_block, N_block, (subblock_h, subblock_w), transpose_core_grid)"]
    shapes: dict[tuple[int, int], list[str]] = defaultdict(list)
    for m in spec.matmuls:
        shapes[(m.K, m.N)].append(m.name)
    for (K, N), names in shapes.items():
        c = selection.configs[names[0]]
        lines.append(
            f"({spec.grid[0]}, {spec.grid[1]}, {K}, {N}): "
            f"({c.m_block}, {c.k_block}, {c.n_block}, ({c.subblock_h}, {c.subblock_w}), {c.transpose}),  # {', '.join(names)}"
        )
    return "\n".join(lines)


# --- search -------------------------------------------------------------------------------------


def _sample_rows(rows: list[int]) -> list[int]:
    """The ends of `rows`, and the row count of `rows` nearest to each TILE * SAMPLE_BASE ** i
    between them, so that searches over different ranges sample the same row counts."""
    anchors = []
    value = TILE
    while value <= rows[-1]:
        if value >= rows[0]:
            anchors.append(value)
        value *= SAMPLE_BASE
    return sorted({rows[0], rows[-1], *(min(rows, key=lambda r, a=a: (abs(r - a), r)) for a in anchors)})


def _best_per_layout(
    results: Results, metric: str, kind: Kind, rows: list[int], configs: Iterable[Config], count: int
) -> list[Config]:
    """The `count` configs of each orientation, among `configs` measured at all `rows`, with the
    lowest worst time over `rows` against the fastest measured at each."""
    times = results.times(metric)
    fastest: dict[int, float] = {}
    for (k, r, _), t in times.items():
        if k == kind and r in rows:
            fastest[r] = min(fastest.get(r, math.inf), t)
    ranked = []
    for c in set(configs):
        ts = [times.get((kind, r, c)) for r in rows]
        if all(t is not None for t in ts):
            ranked.append((max(t / fastest[r] for t, r in zip(ts, rows, strict=True) if t is not None), str(c), c))
    ranked.sort()
    return [c for layout in (False, True) for _, _, c in [x for x in ranked if x[2].transpose == layout][:count]]


def search(spec_path: Path, db: Path, metric: str, top: int, devices: list[int]) -> Selection:
    """Measure what the spec still needs and select its configs."""
    spec = Spec.load(spec_path)
    area = spec.max_subblock_area
    by_kind: dict[Kind, list[Matmul]] = defaultdict(list)
    for m in spec.matmuls:
        by_kind[m.kind].append(m)
    needed = {kind: sorted({r for m in ms for r in spec.needed_rows(m)}) for kind, ms in by_kind.items()}
    sample = {kind: _sample_rows(rows) for kind, rows in needed.items()}
    model = {kind: ms[0] for kind, ms in by_kind.items()}  # K, N and SwiGLU, for the candidates
    for kind, ms in by_kind.items():
        print(f"{' / '.join(m.name for m in ms)}: {kind}, {len(needed[kind])} row counts", flush=True)

    def measure(wanted: list[Key]) -> Results:
        run_jobs(spec_path, spec, db, wanted, devices)
        return Results(db, spec)

    # Every config measured for a kind so far, in any search, is a candidate.
    results = Results(db, spec)
    tried = {kind: set(coarse_candidates(model[kind], area)) for kind in by_kind}
    for k, _, c in results.done:
        if k in tried:
            tried[k].add(c)
    print("coarse pass", flush=True)
    results = measure([(k, r, c) for k in by_kind for r in sample[k] for c in tried[k]])

    for step in range(1, MAX_LOCAL_STEPS + 1):
        wanted = []
        for kind in by_kind:
            seeds = _best_per_layout(results, metric, kind, sample[kind], tried[kind], SEEDS_PER_LAYOUT)
            near = {n for c in seeds for n in neighbours(c, model[kind], area)} - tried[kind]
            tried[kind] |= near
            wanted += [(kind, r, c) for r in sample[kind] for c in near]
        if not wanted:
            break
        print(f"local step {step}: {len(wanted)} points", flush=True)
        results = measure(wanted)

    selected: set[tuple[Kind, Config]] = set()
    selection = None
    for round_ in range(1, MAX_SELECTION_ROUNDS + 1):
        # Matmuls of one shape share a config, so each kind of a shape is timed at every row count
        # with the finalists of all of the shape's kinds, and with the configs selected so far.
        final: dict[tuple[int, int], set[Config]] = defaultdict(set)
        for kind in by_kind:
            final[(kind.K, kind.N)].update(_best_per_layout(results, metric, kind, sample[kind], tried[kind], top))
        for kind, c in selected:
            final[(kind.K, kind.N)].add(c)
        print(f"selection round {round_}: finalists at every row count", flush=True)
        results = measure([(k, r, c) for k in by_kind for r in needed[k] for c in final[(k.K, k.N)]])
        selection = select(spec, results, metric)
        selected = {(m.kind, selection.configs[m.name]) for m in spec.matmuls}
        print(f"worst {selection.worst:.1%}, mean {selection.mean:.1%}", flush=True)
        if round_ == MAX_SELECTION_ROUNDS:
            break  # no round left to use the neighbours in

        # The selected configs' neighbours, at the sampled rows, to see whether any ranks among the
        # finalists next round.
        wanted = []
        for kind, c in selected:
            near = set(neighbours(c, model[kind], area)) - tried[kind]
            tried[kind] |= near
            wanted += [(kind, r, n) for r in sample[kind] for n in near]
        if not wanted:
            break
        print(f"{len(wanted)} points for the neighbours of the selected configs", flush=True)
        results = measure(wanted)
    assert selection is not None  # MAX_SELECTION_ROUNDS >= 1
    return selection


# --- command line -------------------------------------------------------------------------------


def _all_devices() -> list[int]:
    """The PCIe ids of the chips of this machine, or chip 0 alone if they cannot be listed."""
    try:
        ids = sorted(int(name) for name in os.listdir("/dev/tenstorrent") if name.isdigit())
    except OSError:
        ids = []
    return ids or [0]


def _device_ids(text: str) -> list[int]:
    ids = [int(d) for d in text.split(",")]
    if len(set(ids)) != len(ids):
        msg = f"device ids repeat: {text}"
        raise argparse.ArgumentTypeError(msg)
    return ids


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    # SIGTERM unwinds like Ctrl-C: the driver stops its worker, the worker closes the device.
    signal.signal(signal.SIGTERM, lambda signum, _frame: sys.exit(128 + signum))
    if args[:1] == ["_worker"]:
        spec, jobs, db, inflight, device_id = args[1:6]
        _worker(Path(spec), Path(jobs), Path(db), Path(inflight), int(device_id))
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", type=Path)
    parser.add_argument("db", type=Path, help="TSV database of every point measured, shared by all specs")
    parser.add_argument(
        "--metric",
        choices=("device", "host"),
        default="device",
        help="device kernel time (default) or host wall time; use host on Wormhole",
    )
    parser.add_argument("--top", type=int, default=8, help="finalists per orientation and kind (default 8)")
    parser.add_argument(
        "--devices",
        type=_device_ids,
        help="comma-separated PCIe device ids (as TT_VISIBLE_DEVICES takes them) to measure on, one worker each"
        " (default: every chip in /dev/tenstorrent)",
    )
    parsed = parser.parse_args(args)
    devices = parsed.devices if parsed.devices is not None else _all_devices()
    print(f"measuring on device {', '.join(map(str, devices))}", flush=True)
    selection = search(parsed.spec, parsed.db, parsed.metric, parsed.top, devices)
    print(report(Spec.load(parsed.spec), parsed.metric, selection))


if __name__ == "__main__":
    main()
