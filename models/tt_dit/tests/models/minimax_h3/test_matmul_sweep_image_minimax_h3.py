# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Model-faithful sweep of worker grid / orientation / blocking for the four MiniMax-H3-Image DiT matmuls.

The pins in `agmm_config.py` / `mmrs_config.py` were swept for the H3 *video* model (~4736 rows/device). This sweeps
them at the image sizes actually served, through the model's own modules: one dynamic-length
`MiniMaxH3TransformerBlock` built exactly as the image pipeline builds it (4x8 ring, TP=4 on axis 0, SP=8 on axis 1,
2 links, persistent CCL buffers, HiFi2 matmul config), random weights loaded into the four linears, and each layer
called with the kwargs the block passes -- only the frozen config the block would take from its pin is replaced:

  qkv  attn.to_qkv                 all-gather matmul, K=5376 N=5376/device, chunks=3
  out  attn.to_out                 all-gather matmul, K=7168 N=1344/device, fused gated residual (addcmul)
  ff1  ff.ff1                      all-gather matmul, K=5376 N=7168/device, fused SwiGLU
  ff2  ff.ff2.forward_fused_addcmul  fused matmul + reduce-scatter + addcmul, K=3584/device N=5376

The dynamic-length path compiles ONE program per length class and runs it at every row count in that class, so a
candidate is scored over the class's served row counts with traffic weights (sum of weight * time, lower is better);
a candidate missing any of them is disqualified for the class.

Stages (separate jobs, resumable -- finished (candidate, M) rows already in the CSV are skipped):
  A  per layer: orientation x worker grid, each with the current pin's blocking and 3 heuristic blockings that fill
     the grid at the class's reference M (2112 short, 9216 long).
  B  the full blocking space on the best H3_SWEEP_TOP_GRIDS (2) grids of stage A (or H3_SWEEP_GRIDS), then the
     alternative num_workers_per_link values on the top 3 blockings of each grid.
The current pin is measured in every run, so the comparison to today is under the same conditions.

Env:
  H3_SWEEP_LAYER   qkv|out|ff1|ff2, comma list, or all (default all)
  H3_SWEEP_STAGE   A|B (default A)
  H3_SWEEP_CLASS   short|long (default short)
  H3_SWEEP_OUT     results CSV (default /tmp/h3_image_matmul_sweep.csv); appended to, read back to resume
  H3_SWEEP_REPS    timed calls per (candidate, M) (default 10)
  H3_SWEEP_ORIENT  AGMM orientations for stage A: T|NT|both (default both)
  H3_SWEEP_GRIDS   stage-B grids instead of stage A's best, e.g. "NT:11x10,T:12x9" (ff2: "12x8,12x7")
  H3_SWEEP_TOP_GRIDS  stage-B grid count taken from stage A (default 2)
  H3_SWEEP_MS      override the class's row counts, e.g. "2112:0.55,544:0.15" (bare M = weight 1)
  H3_SWEEP_L1_KB   L1 budget of the circular-buffer estimate that prunes blockings before they run (default 1400)
  H3_SWEEP_CLEAR_PROGRAM_CACHE=1  clear the program cache after each M (bounds host memory on very long runs)

  pytest models/tt_dit/tests/models/minimax_h3/test_matmul_sweep_image_minimax_h3.py::test_matmul_sweep -s
  pytest models/tt_dit/tests/models/minimax_h3/test_matmul_sweep_image_minimax_h3.py::test_matmul_sweep_plan -s
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.agmm_config import AGMM_PINS, AgmmMode, resolve_agmm_mode
from ....models.transformers.minimax_h3.length_classes import length_class
from ....models.transformers.minimax_h3.mmrs_config import pinned_mmrs_config
from ....models.transformers.minimax_h3.transformer_block_minimax_h3 import MiniMaxH3TransformerBlock
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.device_constants import compute_grid_key
from ....utils.matmul import FusedMMRSConfig, agmm_worker_grid
from ....utils.tensor import from_torch
from .common import MESH_4X8_RING

# ------------------------------------------------------------------ served sizes and model geometry

# Per-device rows -> traffic weight. Short: 4K t2i (typical prompt / 4096-token prompt), 4K edit (1-2 refs / ~4 refs),
# 2K t2i, 2K edit or long prompt. Long: equal weights.
CLASS_MS: dict[str, dict[int, float]] = {
    "short": {2112: 0.55, 2560: 0.05, 2976: 0.15, 4608: 0.05, 544: 0.15, 1056: 0.05},
    "long": {7168: 1 / 3, 9216: 1 / 3, 14336: 1 / 3},
}
# The M the stage-A heuristic blockings are fitted to: the class's dominant size.
REFERENCE_M = {"short": 2112, "long": 9216}

# The 4x8 Blackhole preset the image pipeline resolves (`pipeline_minimax_h3._PRESETS`, (4, 8)): TP on axis 0, SP on
# axis 1, 2 links, ring. Restated so this test does not import the pipeline (and with it Qwen / the VAE).
TP_AXIS, SP_AXIS, NUM_LINKS = 0, 1, 2

# MiniMaxH3Transformer3DModel's defaults, overridden by MINIMAX_H3_IMAGE_MODEL_PATH/transformer/config.json if set.
DEFAULT_DIMS = {
    "hidden_size": 5376,
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "ffn_dim": 14336,
    "time_embed_dim": 2688,
    "rope_freq_dim": 16,
    "norm_eps": 1e-5,
    "qk_norm_eps": 1e-5,
}

TILE = 32
LAYERS = ("qkv", "out", "ff1", "ff2")
CSV_COLUMNS = [
    "layer",
    "class",
    "stage",
    "M",
    "orientation",
    "grid_x",
    "grid_y",
    "workers_per_link",
    "M_block",
    "K_block",
    "N_block",
    "ok",
    "us",
    "error",
]
MAX_MN_BLOCK = 16  # larger blocks starve the pipelining (and blow the CBs); as in sweep_mm_block_sizes.py
MIN_BLOCK = 2  # subblock is fixed at 2x2, so M_block and N_block are even and >= 2


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _even_up(x: int) -> int:
    return x + (x & 1)


@dataclass(frozen=True)
class LayerSpec:
    """One matmul. AGMM: `k` is the full (gathered) K, `n` the per-device output width (ff1: the packed gate|up
    width). MMRS: `k` is the per-device K, `n` the full pre-scatter width. `k_tiles` is what K_block must divide."""

    name: str
    k: int
    n: int
    k_tiles: int
    mmrs: bool = False
    swiglu: bool = False
    addcmul: bool = False


def layer_specs(dims: dict, tp: int) -> dict[str, LayerSpec]:
    hidden, ffn = dims["hidden_size"], dims["ffn_dim"]
    inner = dims["num_attention_heads"] * dims["attention_head_dim"]
    return {
        "qkv": LayerSpec("qkv", hidden, 3 * inner // tp, hidden // TILE // tp),
        "out": LayerSpec("out", inner, hidden // tp, inner // TILE // tp, addcmul=True),
        "ff1": LayerSpec("ff1", hidden, 2 * ffn // tp, hidden // TILE // tp, swiglu=True),
        "ff2": LayerSpec("ff2", ffn // tp, hidden, ffn // tp // TILE, mmrs=True, addcmul=True),
    }


# ------------------------------------------------------------------ candidates


@dataclass(frozen=True)
class Cand:
    """One frozen program. `orient`: "T" / "NT" (AGMM transpose_core_grid) or "MMRS". `workers`: AGMM
    num_workers_per_link (in0 senders per link), MMRS reduce-scatter workers per link (passed explicitly; the
    FusedMMRSConfig formula's value for the stage-A / pin candidates)."""

    orient: str
    gx: int
    gy: int
    workers: int
    mb: int
    kb: int
    nb: int

    @property
    def grid(self) -> tuple[str, int, int]:
        return (self.orient, self.gx, self.gy)

    def label(self) -> str:
        return f"{self.orient} {self.gx}x{self.gy} wpl={self.workers} blk=({self.mb},{self.kb},{self.nb})"


def per_core_tiles(spec: LayerSpec, orient: str, gx: int, gy: int, m: int) -> tuple[int, int]:
    """(M tiles, N tiles) one core walks, as the op splits them. AGMM transposed: M over grid.x, N over grid.y;
    non-transposed the other way round. MMRS never transposes: M over grid.y. SwiGLU splits whole gate/up pairs."""
    if orient == "T":
        m_axis, n_axis = gx, gy
    else:
        m_axis, n_axis = gy, gx
    n_tiles = spec.n // TILE
    n_pc = 2 * _cdiv(n_tiles // 2, n_axis) if spec.swiglu else _cdiv(n_tiles, n_axis)
    return _cdiv(_cdiv(m, TILE), m_axis), n_pc


def l1_estimate_kb(spec: LayerSpec, mb: int, kb: int, nb: int, n_pc: int, m_pc_max: int | None = None) -> int:
    """Circular-buffer footprint (KB): double-buffered in0 / in1 / out bf16, an fp32 accumulator, the bias row, the
    addcmul operands of a fused AGMM epilogue, and for MMRS the 2-block rolling L1 window of the MM output -- which
    FusedMMRSConfig.get_params only uses when a core walks >= 2 M blocks (at the class's largest M, `m_pc_max`
    M tiles per core; unknown counts as windowed)."""
    kbytes = 2 * (2 * mb * kb + 2 * kb * nb + 2 * mb * nb + nb) + 4 * mb * nb
    if spec.addcmul and not spec.mmrs:
        kbytes += 2 * (mb * nb + nb)
    if spec.mmrs and (m_pc_max is None or _cdiv(m_pc_max, mb) >= 2):
        kbytes += 2 * (2 * mb * n_pc)
    return kbytes


def l1_budget_kb() -> int:
    return int(os.environ.get("H3_SWEEP_L1_KB", "1400"))


def block_ok(spec: LayerSpec, mb: int, kb: int, nb: int, n_pc: int, m_pc_max: int | None = None) -> bool:
    """The asserted constraints (agmm_config / mmrs_config): subblock 2x2 -> M_block, N_block even (which also keeps
    SwiGLU gate/up pairs whole); K_block divides the per-device K tiles (ring delivery has no tail block); plus the
    L1 estimate."""
    return (
        mb >= MIN_BLOCK
        and nb >= MIN_BLOCK
        and mb % 2 == 0
        and nb % 2 == 0
        and 0 < kb <= spec.k_tiles
        and spec.k_tiles % kb == 0
        and l1_estimate_kb(spec, mb, kb, nb, n_pc, m_pc_max) <= l1_budget_kb()
    )


def k_divisors(spec: LayerSpec) -> list[int]:
    """K_block candidates: divisors of the per-device K tiles, 2 up to 28 (AGMM: 42 -> 2..21, 56 -> 2..28) or
    16 (MMRS, 112 local K tiles)."""
    cap = 16 if spec.mmrs else 28
    return [d for d in range(2, min(cap, spec.k_tiles) + 1) if spec.k_tiles % d == 0]


def split_even(total: int, parts: int) -> int:
    """The even block size that walks `total` tiles in `parts` blocks (>= 2)."""
    return max(MIN_BLOCK, _even_up(_cdiv(total, parts)))


def min_parts(total: int) -> int:
    """Fewest blocks a core needs for `total` tiles under the MAX_MN_BLOCK cap."""
    return max(1, _cdiv(total, MAX_MN_BLOCK))


def agmm_workers_options(in0_axis: int, num_links: int) -> list[int]:
    """num_workers_per_link values the op accepts: the in0 sender axis must form exactly `num_links` groups."""
    return [w for w in range(1, in0_axis + 1) if _cdiv(in0_axis, w) == num_links]


def agmm_default_workers(in0_axis: int, num_links: int) -> int:
    return _cdiv(in0_axis, num_links)  # what agmm_pins() picks


def mmrs_default_workers(full: tuple[int, int], gy: int, num_links: int) -> int:
    """FusedMMRSConfig.get_params' formula: the reduce-scatter takes the rows the matmul grid leaves free."""
    return (full[1] - gy) * full[0] // (2 * num_links) - 1


def agmm_grids(full: tuple[int, int], orient: str) -> list[tuple[str, int, int]]:
    """Stage-A AGMM worker grids. The op puts its in0 muxes on the last ROW when transposed, and when not transposed
    on the last COLUMN if the grid leaves it free (grid.x < device x), else on the last row -- so a transposed grid
    keeps y <= device y - 1, a non-transposed one keeps x <= device x - 1 or y <= device y - 1. Each axis shrinks
    by up to 4 from there (agmm_worker_grid's pick is the largest)."""
    fx, fy = full
    grids = []
    if orient in ("T", "both"):
        for gx in range(fx, fx - 5, -1):
            for gy in range(fy - 1, fy - 5, -1):
                grids.append(("T", gx, gy))
    if orient in ("NT", "both"):
        for gy in range(fy, fy - 5, -1):
            for gx in range(fx, fx - 5, -1):
                if gx == fx and gy == fy:
                    continue  # no free row or column for the muxes
                grids.append(("NT", gx, gy))
    return grids


def mmrs_grids(full: tuple[int, int], num_links: int) -> list[tuple[str, int, int]]:
    """Stage-A MMRS matmul grids: the reduce-scatter workers take the rows below, so the matmul's height trades
    against RS bandwidth (mmrs_config's 12x7 / 12x8 / 12x9 sweep); also 12x6 and two narrower widths."""
    fx, fy = full
    return [
        ("MMRS", gx, gy)
        for gx in (fx, fx - 1, fx - 2)
        for gy in range(fy - 4, fy)
        if gy >= 2 and mmrs_default_workers(full, gy, num_links) >= 1
    ]


def fit_blocking(
    spec: LayerSpec, mb: int, kb_pref: int, nb: int, n_pc: int, m_pc_max: int
) -> tuple[int, int, int] | None:
    """(mb, kb, nb) with K_block the largest divisor <= kb_pref that fits L1; failing that shrink N, then M."""
    ks = sorted({d for d in [*k_divisors(spec), spec.k_tiles] if d <= max(kb_pref, 2)}, reverse=True)
    while True:
        for kb in ks:
            if block_ok(spec, mb, kb, nb, n_pc, m_pc_max):
                return mb, kb, nb
        if nb > MIN_BLOCK:
            nb = max(MIN_BLOCK, _even_up(nb // 2))
        elif mb > MIN_BLOCK:
            mb = max(MIN_BLOCK, _even_up(mb // 2))
        else:
            return None


def max_m_per_core(spec: LayerSpec, orient: str, gx: int, gy: int, ms) -> int:
    return max(per_core_tiles(spec, orient, gx, gy, m)[0] for m in ms)


def heuristic_blockings(
    spec: LayerSpec, orient: str, gx: int, gy: int, ref_m: int, kb_pref: int, m_pc_max: int
) -> list[tuple[int, int, int]]:
    """Three blockings that fill the grid at `ref_m`: one M block per core x the fewest N blocks, two M blocks per
    core (pipelining) x the fewest N blocks, one M block x one more N block. The block count along each axis is what
    matches ceil(M_tiles / M_block) / ceil(N_tiles / N_block) to the grid axes; K_block follows the pin."""
    m_pc, n_pc = per_core_tiles(spec, orient, gx, gy, ref_m)
    pm, pn = min_parts(m_pc), min_parts(n_pc)
    out = []
    for m_parts, n_parts in ((pm, pn), (pm + 1, pn), (pm, pn + 1)):
        fitted = fit_blocking(spec, split_even(m_pc, m_parts), kb_pref, split_even(n_pc, n_parts), n_pc, m_pc_max)
        if fitted is not None and fitted not in out:
            out.append(fitted)
    return out


def default_workers(spec: LayerSpec, orient: str, gx: int, gy: int, full: tuple[int, int], num_links: int) -> int:
    if spec.mmrs:
        return mmrs_default_workers(full, gy, num_links)
    return agmm_default_workers(gx if orient == "T" else gy, num_links)


def stage_a_candidates(
    spec: LayerSpec, cls: str, ms, pin: Cand, full: tuple[int, int], num_links: int, orient: str
) -> list[Cand]:
    grids = mmrs_grids(full, num_links) if spec.mmrs else agmm_grids(full, orient)
    cands = [pin]
    for o, gx, gy in grids:
        workers = default_workers(spec, o, gx, gy, full, num_links)
        _, n_pc = per_core_tiles(spec, o, gx, gy, REFERENCE_M[cls])
        m_pc_max = max_m_per_core(spec, o, gx, gy, ms)
        blockings = []
        if block_ok(spec, pin.mb, pin.kb, pin.nb, n_pc, m_pc_max):
            blockings.append((pin.mb, pin.kb, pin.nb))
        blockings += heuristic_blockings(spec, o, gx, gy, REFERENCE_M[cls], pin.kb, m_pc_max)
        for mb, kb, nb in blockings:
            c = Cand(o, gx, gy, workers, mb, kb, nb)
            if c not in cands:
                cands.append(c)
    return cands


def stage_b_blockings(spec: LayerSpec, orient: str, gx: int, gy: int, ms) -> list[tuple[int, int, int]]:
    """The full blocking space of one grid: every even M_block up to the class's largest per-core M (cap 16); the
    even N_blocks that split the per-core N into the fewest 1..4 blocks (a larger N_block at the same block count
    only adds padding); every K_block divisor; pruned by the L1 estimate."""
    m_pc_max = max_m_per_core(spec, orient, gx, gy, ms)
    _, n_pc = per_core_tiles(spec, orient, gx, gy, max(ms))
    m_blocks = list(range(MIN_BLOCK, min(MAX_MN_BLOCK, _even_up(m_pc_max)) + 1, 2))
    n_blocks = sorted({min(MAX_MN_BLOCK, split_even(n_pc, p)) for p in range(min_parts(n_pc), min_parts(n_pc) + 4)})
    return [
        (mb, kb, nb)
        for mb in m_blocks
        for kb in k_divisors(spec)
        for nb in n_blocks
        if block_ok(spec, mb, kb, nb, n_pc, m_pc_max)
    ]


def workers_variants(spec: LayerSpec, c: Cand, full: tuple[int, int], num_links: int) -> list[int]:
    """Alternative workers-per-link values for a blocking. AGMM: every value the in0 axis accepts. MMRS: fewer RS
    workers than the formula (more can not fit the RS rows)."""
    if spec.mmrs:
        d = mmrs_default_workers(full, c.gy, num_links)
        return sorted({w for w in (d - 1, d - 2, _cdiv(d, 2), 1) if 1 <= w < d})
    in0 = c.gx if c.orient == "T" else c.gy
    return [w for w in agmm_workers_options(in0, num_links) if w != c.workers]


def agmm_pin(spec: LayerSpec, cls: str, full_grid, num_links: int) -> Cand:
    """The current (video, `profile=None`) pin of an AGMM layer, as `agmm_config.agmm_pins` builds it: the layer's
    pin for the resolved AGMM mode on this compute grid (else its ACCURATE pin), the worker grid of
    `agmm_worker_grid`, in0 senders per link, the class's blocking. The sweep holds the subblock at 2x2."""
    pins_by_mode = AGMM_PINS[compute_grid_key(full_grid)]
    layer = (
        pins_by_mode.get(resolve_agmm_mode(), {}).get((spec.k, spec.n))
        or pins_by_mode[AgmmMode.ACCURATE][(spec.k, spec.n)]
    )
    if layer.subblock != (2, 2):
        logger.warning(f"AGMM pin for {(spec.k, spec.n)} has subblock {layer.subblock}; the sweep holds 2x2")
    transposed = layer.transpose_core_grid
    grid = agmm_worker_grid(full_grid, transposed, num_links)
    in0 = grid.x if transposed else grid.y
    mb, kb, nb = layer.blockings[cls]
    return Cand("T" if transposed else "NT", grid.x, grid.y, math.ceil(in0 / num_links), mb, kb, nb)


def mmrs_pin(spec: LayerSpec, full_grid, num_links: int) -> Cand:
    """The current ff2 fused MM+RS pin (`mmrs_config.pinned_mmrs_config`), RS workers resolved by its formula."""
    cfg = pinned_mmrs_config(spec.k, spec.n, full_grid)
    if cfg is None:
        pytest.skip(f"no pinned MMRS config for (K, N)={(spec.k, spec.n)} on grid {full_grid}")
    if (cfg.subblock_h, cfg.subblock_w, cfg.chunk_width_in_mm_blocks) != (2, 2, 1):
        logger.warning(f"MMRS pin {cfg} is not subblock 2x2 / chunk 1; the sweep holds those fixed at 2x2 / 1")
    g = cfg.compute_with_storage_grid_size
    full = (full_grid.x, full_grid.y)
    workers = cfg.num_workers_per_link or mmrs_default_workers(full, g.y, num_links)
    return Cand("MMRS", g.x, g.y, workers, cfg.M_block_size, cfg.K_block_size, cfg.N_block_size)


# ------------------------------------------------------------------ results CSV (resumable)


def _row_key(layer: str, cls: str, stage: str, m: int, c: Cand) -> tuple:
    return (layer, cls, stage, int(m), c.orient, c.gx, c.gy, c.workers, c.mb, c.kb, c.nb)


def _row_cand(r: dict) -> Cand:
    return Cand(
        r["orientation"],
        int(r["grid_x"]),
        int(r["grid_y"]),
        int(r["workers_per_link"]),
        int(r["M_block"]),
        int(r["K_block"]),
        int(r["N_block"]),
    )


class Results:
    """Appends one CSV row per (candidate, M) as it finishes; reads the file back to resume. Before each measurement
    the candidate is written to `<csv>.inflight`, so a run that dies or hangs on it records it as failed on restart
    and never retries it."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.inflight = path + ".inflight"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        if os.path.exists(path):
            with open(path, newline="") as f:
                self.rows = list(csv.DictReader(f))
        else:
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()
        self.done = {self._key_of(r) for r in self.rows}
        if os.path.exists(self.inflight):
            with open(self.inflight) as f:
                pending = json.load(f)
            if self._key_of(pending) not in self.done:
                logger.warning(f"recording the candidate the last run died on as failed: {pending}")
                self.add({**pending, "ok": 0, "us": "", "error": "no result: the run died or hung on it"})
            if os.path.exists(self.inflight):
                os.remove(self.inflight)

    @staticmethod
    def _key_of(r: dict) -> tuple:
        return (r["layer"], r["class"], r["stage"], int(r["M"])) + tuple(
            int(r[k]) if k != "orientation" else r[k]
            for k in ("orientation", "grid_x", "grid_y", "workers_per_link", "M_block", "K_block", "N_block")
        )

    @staticmethod
    def base_row(layer: str, cls: str, stage: str, m: int, c: Cand) -> dict:
        return {
            "layer": layer,
            "class": cls,
            "stage": stage,
            "M": m,
            "orientation": c.orient,
            "grid_x": c.gx,
            "grid_y": c.gy,
            "workers_per_link": c.workers,
            "M_block": c.mb,
            "K_block": c.kb,
            "N_block": c.nb,
        }

    def begin(self, row: dict) -> None:
        with open(self.inflight, "w") as f:
            json.dump(row, f)

    def add(self, row: dict) -> None:
        row = {k: row.get(k, "") for k in CSV_COLUMNS}
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)
        if os.path.exists(self.inflight):
            os.remove(self.inflight)
        self.rows.append({k: str(v) for k, v in row.items()})
        self.done.add(self._key_of(row))

    def select(self, layer: str, cls: str, stages=None) -> list[dict]:
        return [
            r for r in self.rows if r["layer"] == layer and r["class"] == cls and (not stages or r["stage"] in stages)
        ]

    def failed(self, layer: str, cls: str, stage: str) -> set[Cand]:
        return {_row_cand(r) for r in self.select(layer, cls, {stage}) if str(r["ok"]) != "1"}


def weighted_scores(rows: list[dict], weights: dict[int, float]) -> dict[Cand, tuple[float, dict[int, float]]]:
    """Candidate -> (sum of weight * median time over the class's M, per-M times); only candidates with every M."""
    per: dict[Cand, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if str(r["ok"]) == "1" and int(r["M"]) in weights:
            per[_row_cand(r)][int(r["M"])].append(float(r["us"]))
    out = {}
    for c, by_m in per.items():
        if all(m in by_m for m in weights):
            times = {m: statistics.median(by_m[m]) for m in weights}
            out[c] = (sum(w * times[m] for m, w in weights.items()), times)
    return out


# ------------------------------------------------------------------ env


def _env_list(name: str, default: str, allowed) -> list[str]:
    raw = os.environ.get(name, default).strip()
    values = list(allowed) if raw == "all" else [v.strip() for v in raw.split(",") if v.strip()]
    bad = [v for v in values if v not in allowed]
    if bad:
        raise ValueError(f"{name}={raw!r}: {bad} not in {list(allowed)}")
    return values


def class_weights(cls: str) -> dict[int, float]:
    raw = os.environ.get("H3_SWEEP_MS")
    if not raw:
        return dict(CLASS_MS[cls])
    weights = {}
    for item in raw.split(","):
        m, _, w = item.strip().partition(":")
        weights[int(m)] = float(w) if w else 1.0
    for m in weights:
        if m % TILE or length_class(m).name != cls:
            raise ValueError(f"H3_SWEEP_MS row count {m} is not a tile multiple in class {cls}")
    return weights


def stage_b_grids(spec: LayerSpec, cls: str, results: Results, weights: dict[int, float]) -> list[tuple[str, int, int]]:
    raw = os.environ.get("H3_SWEEP_GRIDS")
    if raw:
        grids = []
        for item in raw.split(","):
            orient, _, xy = item.strip().rpartition(":")
            orient = "MMRS" if spec.mmrs else orient.upper()
            if orient not in ("T", "NT", "MMRS"):
                raise ValueError(f"H3_SWEEP_GRIDS item {item!r}: AGMM grids are 'T:<x>x<y>' or 'NT:<x>x<y>'")
            gx, gy = (int(v) for v in xy.lower().split("x"))
            grids.append((orient, gx, gy))
        return grids
    scores = weighted_scores(results.select(spec.name, cls, {"A"}), weights)
    best: dict[tuple, float] = {}
    for c, (score, _) in scores.items():
        best[c.grid] = min(score, best.get(c.grid, math.inf))
    top = int(os.environ.get("H3_SWEEP_TOP_GRIDS", "2"))
    grids = sorted(best, key=best.get)[:top]
    if not grids:
        pytest.skip(f"stage B for {spec.name}/{cls} needs stage-A results in {results.path} (or set H3_SWEEP_GRIDS)")
    for g in grids:
        logger.info(f"stage B {spec.name}/{cls}: stage-A grid {g} best score {best[g]:.1f} us")
    return grids


# ------------------------------------------------------------------ device harness


def _model_dims() -> dict:
    dims = dict(DEFAULT_DIMS)
    path = os.environ.get("MINIMAX_H3_IMAGE_MODEL_PATH")
    if path:
        config_path = Path(path) / "transformer" / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            dims.update({k: config[k] for k in dims if k in config})
            logger.info(f"model dims from {config_path}")
    return dims


def _random_device_tensor(mesh_device: ttnn.MeshDevice, shape: tuple[int, ...]) -> ttnn.Tensor:
    """A per-device-shaped bf16 tile tensor, the same values on every device (content does not matter to timing, and
    replicating a per-device shape keeps the host tensor small at the long class's sizes)."""
    return from_torch(torch.randn(shape, dtype=torch.bfloat16), device=mesh_device)


def _randomize_weight(linear) -> None:
    param = linear.weight
    param.load_torch_tensor(torch.randn(param.total_shape, dtype=torch.bfloat16) * 0.02)


class Harness:
    """One dynamic-length block built the way MiniMaxH3ImagePipeline builds its DiT blocks, and per-layer callers that
    reproduce the block's calls with a candidate's frozen config in place of the pin."""

    def __init__(self, mesh_device: ttnn.MeshDevice, dims: dict, capacity_rows: int) -> None:
        self.mesh_device = mesh_device
        shape = tuple(mesh_device.shape)
        self.tp, self.sp = shape[TP_AXIS], shape[SP_AXIS]
        # pipeline_minimax_h3_image.MiniMaxH3ImagePipeline.__init__: one CCLManager for the DiT, the DiT parallel
        # config, persistent CCL buffers on (the image pipeline's default), no FSDP.
        self.ccl_manager = CCLManager(mesh_device=mesh_device, num_links=NUM_LINKS, topology=ttnn.Topology.Ring)
        self.parallel_config = DiTParallelConfig(
            tensor_parallel=ParallelFactor(mesh_axis=TP_AXIS, factor=self.tp),
            sequence_parallel=ParallelFactor(mesh_axis=SP_AXIS, factor=self.sp),
            cfg_parallel=None,
        )
        # transformer_minimax_h3.MiniMaxH3Transformer3DModel's block construction (dynamic length: norm_row_capacity
        # is the K/V gather capacity over SP).
        self.block = MiniMaxH3TransformerBlock(
            hidden_size=dims["hidden_size"],
            num_heads=dims["num_attention_heads"],
            head_dim=dims["attention_head_dim"],
            rotary_dim=2 * 3 * dims["rope_freq_dim"],
            ffn_dim=dims["ffn_dim"],
            time_embed_dim=dims["time_embed_dim"],
            norm_eps=dims["norm_eps"],
            qk_norm_eps=dims["qk_norm_eps"],
            mesh_device=mesh_device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            is_fsdp=False,
            kv_gather_capacity=capacity_rows * self.sp,
            use_persistent_ccl_buffers=True,
            dynamic_length=True,
            norm_row_capacity=capacity_rows,
        )
        attn = self.block.attn
        assert attn.use_fused_agmm and self.block.use_fused_agmm, "the AGMM layers need a ring with TP > 1"
        assert self.block._mmrs_pin is not None, "ff2 has no pinned fused MM+RS config on this grid"
        self.full_grid = mesh_device.compute_with_storage_grid_size()
        self.full = (self.full_grid.x, self.full_grid.y)
        self.specs = layer_specs(dims, self.tp)
        self.linears = {
            "qkv": attn.to_qkv,
            "out": attn.to_out,
            "ff1": self.block.ff.ff1,
            "ff2": self.block.ff.ff2,
        }
        self._loaded: set[str] = set()

    def pin(self, layer: str, cls: str) -> Cand:
        spec = self.specs[layer]
        if spec.mmrs:
            return mmrs_pin(spec, self.full_grid, NUM_LINKS)
        c = agmm_pin(spec, cls, self.full_grid, NUM_LINKS)
        # Cross-check against the pin the block actually resolved.
        pins = {"qkv": self.block.attn._qkv_pins, "out": self.block.attn._out_pins, "ff1": self.block._ff1_pins}[layer]
        live = pins[cls]
        if live.num_workers_per_link != c.workers or bool(live.transpose_core_grid) != (c.orient == "T"):
            logger.warning(f"{layer}/{cls}: table-derived pin {c.label()} differs from the block's {live}")
        return c

    def load(self, layer: str) -> None:
        if layer not in self._loaded:
            _randomize_weight(self.linears[layer])
            self._loaded.add(layer)

    def runner(self, layer: str, m: int):
        """Allocate this layer's inputs at `m` per-device rows; return cand -> zero-arg call of the layer."""
        spec, block, attn = self.specs[layer], self.block, self.block.attn
        linear = self.linears[layer]
        rnd = lambda width: _random_device_tensor(self.mesh_device, (1, 1, m, width))  # noqa: E731

        if spec.mmrs:
            # ff2's input is ff1's output ([1, 1, M, ffn/tp]); the gated residual and the gate are the
            # TP-fractured hidden width, as forward_fused_addcmul receives them from the block.
            x, residual, gate = rnd(spec.k), rnd(spec.n // self.tp), rnd(spec.n // self.tp)

            def prepare(c: Cand):
                config = FusedMMRSConfig(ttnn.CoreCoord(c.gx, c.gy), c.mb, c.kb, c.nb, 2, 2, None, 1, c.workers)
                return lambda: linear.forward_fused_addcmul(
                    x,
                    residual,
                    gate,
                    compute_kernel_config=block.mm_compute_kernel_config,
                    use_persistent_buffer=block.use_persistent_ccl_buffers,
                    mmrs_config=config,
                )

            return prepare

        x = rnd(spec.k // self.tp)  # the TP-fractured activation the all-gather matmul gathers
        extra = {}
        if spec.addcmul:  # to_out folds `residual + out * gate` into its epilogue
            extra = {"addcmul_a": rnd(spec.n), "addcmul_b": rnd(spec.n)}
        # The model runs each AGMM with its resolved pin's compute config (the pin's Dst accumulation), per class.
        live_pins = {"qkv": attn._qkv_pins, "out": attn._out_pins, "ff1": block._ff1_pins}[layer]
        kernel_config = live_pins[length_class(m).name].compute_kernel_config
        persistent = attn.use_persistent_ccl_buffers if layer in ("qkv", "out") else block.use_persistent_ccl_buffers

        def prepare(c: Cand):
            config = ttnn.MinimalMatmulConfig(
                M_block_size=c.mb,
                K_block_size=c.kb,
                N_block_size=c.nb,
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(c.gx, c.gy),
            )
            return lambda: linear(
                x,
                compute_kernel_config=kernel_config,
                parallel_config=self.parallel_config,
                force_transpose=False,
                use_persistent_buffer=persistent,
                matmul_config=config,
                num_workers_per_link=c.workers,
                transpose_core_grid=c.orient == "T",
                **extra,
            )

        return prepare


def _time_call(fn, mesh_device: ttnn.MeshDevice, reps: int) -> tuple[float, float]:
    """(us per call, host-enqueue share). One untimed call compiles and allocates the persistent buffers; then `reps`
    back-to-back calls and one device sync, as the eager denoise loop runs them. Only one output is held at a time."""
    out = fn()
    del out
    ttnn.synchronize_device(mesh_device)
    start = time.perf_counter()
    for _ in range(reps):
        out = fn()
    enqueued = time.perf_counter() - start
    ttnn.synchronize_device(mesh_device)
    total = time.perf_counter() - start
    del out
    return total / reps * 1e6, enqueued / total


def _measure(
    harness: Harness,
    results: Results,
    layer: str,
    cls: str,
    stage: str,
    cands: list[Cand],
    weights: dict[int, float],
    reps: int,
) -> None:
    """Every (candidate, M) not already in the CSV, M-outer so each M's inputs and CCL buffers are allocated once. A
    candidate that fails at one M is disqualified for the class and not run at the rest."""
    mesh_device = harness.mesh_device
    failed = results.failed(layer, cls, stage)
    clear_cache = os.environ.get("H3_SWEEP_CLEAR_PROGRAM_CACHE") == "1"
    for m in sorted(weights, key=weights.get, reverse=True):  # the heaviest M first
        todo = [c for c in cands if c not in failed and _row_key(layer, cls, stage, m, c) not in results.done]
        if not todo:
            continue
        logger.info(f"{layer}/{cls}/stage {stage}: M={m}, {len(todo)} candidates")
        mark = time.perf_counter()
        with harness.ccl_manager.transient_ping_pong_buffers():
            prepare = harness.runner(layer, m)
            for i, c in enumerate(todo):
                row = Results.base_row(layer, cls, stage, m, c)
                results.begin(row)
                try:
                    us, host_share = _time_call(prepare(c), mesh_device, reps)
                    results.add({**row, "ok": 1, "us": f"{us:.2f}", "error": ""})
                    flag = "  [host-enqueue bound]" if host_share > 0.9 else ""
                    logger.info(f"  [{i + 1}/{len(todo)}] M={m} {c.label()}: {us:.1f} us{flag}")
                except Exception as err:  # noqa: BLE001 -- an invalid candidate is a result, not a test failure
                    msg = " ".join(str(err).split())[:300]
                    results.add({**row, "ok": 0, "us": "", "error": msg})
                    failed.add(c)
                    logger.warning(f"  [{i + 1}/{len(todo)}] M={m} {c.label()}: FAILED {msg[:160]}")
                    try:
                        ttnn.synchronize_device(mesh_device)
                    except Exception as sync_err:  # noqa: BLE001
                        logger.error(f"device sync after the failure also failed: {sync_err}")
            del prepare
        if clear_cache and hasattr(mesh_device, "clear_program_cache"):
            mesh_device.clear_program_cache()
        logger.info(f"{layer}/{cls}/stage {stage}: M={m} done in {time.perf_counter() - mark:.0f} s")


def _report(results: Results, layer: str, cls: str, stage: str, weights: dict[int, float], pin: Cand) -> None:
    """`RESULT sweep` lines: top 5 by weighted score and the pin, over this stage's rows (same run conditions as the
    pin) and over every stage in the CSV."""
    for scope, stages in ((f"stage{stage}", {stage}), ("all", None)):
        scores = weighted_scores(results.select(layer, cls, stages), weights)
        if not scores:
            logger.info(f"RESULT sweep layer={layer} class={cls} scope={scope}: no candidate measured at every M")
            continue
        pin_score = scores.get(pin, (math.nan, {}))[0]
        ranked = sorted(scores.items(), key=lambda kv: kv[1][0])
        for rank, (c, (score, times)) in enumerate(ranked[:5], start=1):
            per_m = " ".join(f"{m}:{times[m]:.0f}" for m in sorted(times))
            delta = (score / pin_score - 1) * 100 if pin_score == pin_score else math.nan
            logger.info(
                f"RESULT sweep layer={layer} class={cls} scope={scope} rank={rank} score_us={score:.1f} "
                f"vs_pin={delta:+.1f}% {c.label()} per_M_us=[{per_m}]"
            )
        pin_times = scores.get(pin, (None, {}))[1]
        per_m = " ".join(f"{m}:{pin_times[m]:.0f}" for m in sorted(pin_times))
        logger.info(
            f"RESULT sweep layer={layer} class={cls} scope={scope} PIN score_us={pin_score:.1f} {pin.label()} "
            f"per_M_us=[{per_m}] (of {len(scores)} complete candidates)"
        )


def _plan(spec: LayerSpec, cls: str, ms, pin: Cand, full: tuple[int, int], orient: str) -> list[Cand]:
    return stage_a_candidates(spec, cls, ms, pin, full, NUM_LINKS, "both" if spec.mmrs else orient)


@pytest.mark.timeout(0)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_matmul_sweep(mesh_device: ttnn.MeshDevice) -> None:
    layers = _env_list("H3_SWEEP_LAYER", "all", LAYERS)
    (stage,) = _env_list("H3_SWEEP_STAGE", "A", ("A", "B"))
    (cls,) = _env_list("H3_SWEEP_CLASS", "short", tuple(CLASS_MS))
    (orient,) = _env_list("H3_SWEEP_ORIENT", "both", ("T", "NT", "both"))
    reps = int(os.environ.get("H3_SWEEP_REPS", "10"))
    results = Results(os.environ.get("H3_SWEEP_OUT", "/tmp/h3_image_matmul_sweep.csv"))
    weights = class_weights(cls)

    dims = _model_dims()
    capacity_rows = max(max(ms) for ms in CLASS_MS.values())
    harness = Harness(mesh_device, dims, capacity_rows)
    logger.info(
        f"matmul sweep: layers {layers}, stage {stage}, class {cls} M->weight {weights}, reps {reps}, "
        f"device grid {harness.full}, CSV {results.path}"
    )

    for layer in layers:
        spec = harness.specs[layer]
        pin = harness.pin(layer, cls)
        harness.load(layer)
        logger.info(f"{layer}: {spec}, current pin {pin.label()}")

        if stage == "A":
            cands = _plan(spec, cls, list(weights), pin, harness.full, orient)
            logger.info(f"{layer}/{cls} stage A: {len(cands)} candidates x {len(weights)} M")
            _measure(harness, results, layer, cls, stage, cands, weights, reps)
        else:
            grids = stage_b_grids(spec, cls, results, weights)
            cands = [pin]
            for o, gx, gy in grids:
                workers = default_workers(spec, o, gx, gy, harness.full, NUM_LINKS)
                for mb, kb, nb in stage_b_blockings(spec, o, gx, gy, list(weights)):
                    c = Cand(o, gx, gy, workers, mb, kb, nb)
                    if c not in cands:
                        cands.append(c)
            logger.info(f"{layer}/{cls} stage B: {len(cands)} blockings over grids {grids} x {len(weights)} M")
            _measure(harness, results, layer, cls, stage, cands, weights, reps)

            # The workers-per-link axis, on each grid's 3 best blockings.
            scores = weighted_scores(results.select(layer, cls, {"B"}), weights)
            extra = []
            for grid in grids:
                on_grid = sorted((c for c in scores if c.grid == grid), key=lambda c: scores[c][0])[:3]
                for c in on_grid:
                    for w in workers_variants(spec, c, harness.full, NUM_LINKS):
                        v = Cand(c.orient, c.gx, c.gy, w, c.mb, c.kb, c.nb)
                        if v not in cands and v not in extra:
                            extra.append(v)
            logger.info(f"{layer}/{cls} stage B: {len(extra)} workers-per-link variants")
            _measure(harness, results, layer, cls, stage, extra, weights, reps)

        _report(results, layer, cls, stage, weights, pin)


def test_matmul_sweep_plan() -> None:
    """No device: the stage-A candidate counts per layer and class for a 12x10 Blackhole grid, to size the jobs."""
    full_grid = ttnn.CoreCoord(12, 10)
    specs = layer_specs(_model_dims(), 4)
    for cls in CLASS_MS:
        for layer, spec in specs.items():
            pin = mmrs_pin(spec, full_grid, NUM_LINKS) if spec.mmrs else agmm_pin(spec, cls, full_grid, NUM_LINKS)
            cands = _plan(spec, cls, list(CLASS_MS[cls]), pin, (12, 10), "both")
            b = stage_b_blockings(spec, pin.orient, pin.gx, pin.gy, list(CLASS_MS[cls]))
            logger.info(
                f"PLAN {layer}/{cls}: pin {pin.label()}; stage A {len(cands)} candidates x {len(CLASS_MS[cls])} M "
                f"= {len(cands) * len(CLASS_MS[cls])} runs; stage B ~{len(b)} blockings per grid on the pin's grid"
            )
