"""
Verdict — given a sharded memory footprint and a box, decide if it fits
and how tight the fit is.

This replaces the single binary FITS/no judgement of the old script with
a four-tier classifier:

  COMFORT      headroom >= 25% of usable-per-chip   "safe to grow batch/seq"
  ROOM         headroom 10-25%                        "fits with normal margin"
  TIGHT        headroom 0-10%                         "fits but on the edge"
  NO_FIT       headroom < 0                           "doesn't fit"

The tightness classifier is the user-facing answer to the legitimate
question "should I trust a +3 GB-headroom FITS verdict?".  TIGHT verdicts
should be confirmed with a hardware smoke test before committing to a port.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import List, Optional

from .architecture import MemoryModel
from .hardware import Box
from .parallelism import ParallelConfig, ShardedMemory, shard


class Tightness(enum.Enum):
    COMFORT = "FITS (comfortable)"
    ROOM = "FITS (room)"
    TIGHT = "FITS (tight)"
    NO_FIT = "no"

    @classmethod
    def classify(cls, headroom_gb: float, usable_gb: float) -> "Tightness":
        if headroom_gb < 0:
            return cls.NO_FIT
        ratio = headroom_gb / max(usable_gb, 1e-9)
        if ratio >= 0.25:
            return cls.COMFORT
        if ratio >= 0.10:
            return cls.ROOM
        return cls.TIGHT


@dataclass
class FitRow:
    """One row of the per-(box, dtype, mesh) fit table."""

    box: Box
    dtype: str
    mesh_shape: tuple
    parallel: ParallelConfig
    sharded: ShardedMemory

    usable_per_chip_gb: float
    per_chip_gb: float
    headroom_gb: float
    tightness: Tightness

    @property
    def fits(self) -> bool:
        return self.tightness != Tightness.NO_FIT


@dataclass
class FitVerdict:
    """Aggregated planner output for one model."""

    rows: List[FitRow]
    best: Optional[FitRow]
    notes: List[str]


def evaluate_one(
    model: MemoryModel,
    box: Box,
    dtype: str,
    mesh_shape: tuple,
    parallel: ParallelConfig,
    batch: int,
    seq: int,
    kv_dtype_bytes: float,
) -> FitRow:
    sharded_mem = shard(model, dtype, batch, seq, kv_dtype_bytes, parallel)

    usable_gb = box.usable_per_chip_gb(parallel.tp)
    per_chip_gb = sharded_mem.total_bytes / 1e9
    headroom_gb = usable_gb - per_chip_gb
    tightness = Tightness.classify(headroom_gb, usable_gb)

    return FitRow(
        box=box,
        dtype=dtype,
        mesh_shape=mesh_shape,
        parallel=parallel,
        sharded=sharded_mem,
        usable_per_chip_gb=usable_gb,
        per_chip_gb=per_chip_gb,
        headroom_gb=headroom_gb,
        tightness=tightness,
    )


_TIGHTNESS_RANK = {
    Tightness.COMFORT: 0,
    Tightness.ROOM: 1,
    Tightness.TIGHT: 2,
    Tightness.NO_FIT: 3,
}


def _row_score(row: FitRow) -> tuple:
    """
    Sort key for recommending the "best" row.

    Policy: pick the **smallest box that comfortably fits**.  Concretely:
      1. Must fit at all.
      2. Prefer COMFORT > ROOM > TIGHT.
      3. Within the same tightness category, prefer fewer chips
         (don't commit a Galaxy for a 2B model just because Galaxy has
         the largest absolute headroom).
      4. Tiebreak on smaller total HBM.

    This matches the "smallest sufficient hardware" deployment philosophy
    used by vLLM's `--gpu-memory-utilization` and TRT-LLM's TP picker.
    """
    return (
        _TIGHTNESS_RANK[row.tightness],
        row.box.chips,
        row.box.total_hbm_gb,
        -row.headroom_gb,
    )


def pick_best(rows: List[FitRow]) -> Optional[FitRow]:
    fitting = [r for r in rows if r.fits]
    if not fitting:
        return None
    return min(fitting, key=_row_score)


def evaluate_all(
    model: MemoryModel,
    boxes: List[Box],
    dtypes: List[str],
    batch: int,
    seq: int,
    kv_dtype_bytes: float = 2.0,
    all_meshes: bool = False,
    explore_pp: bool = False,
) -> FitVerdict:
    """
    Build the full fit table for every (box, dtype, mesh, parallelism) combination.

    Args:
      all_meshes: when False, only the largest-chip-count canonical mesh
                  per box is reported.
      explore_pp: when True, also enumerate TP×PP combinations whose
                  product is the mesh chip count (e.g. T3K can do TP=8
                  or TP=4×PP=2 or TP=2×PP=4 or TP=1×PP=8).  When False,
                  only pure-TP configs.
    """
    from .parallelism import enumerate_meshes

    rows: List[FitRow] = []
    notes: List[str] = []

    for box in boxes:
        meshes = list(enumerate_meshes(box, explore_pp=explore_pp))
        if not all_meshes:
            best_chips = max(p.chips for _, p in meshes)
            meshes = [(s, p) for s, p in meshes if p.chips == best_chips]

        for dtype in dtypes:
            for shape, pcfg in meshes:
                rows.append(
                    evaluate_one(
                        model=model,
                        box=box,
                        dtype=dtype,
                        mesh_shape=shape,
                        parallel=pcfg,
                        batch=batch,
                        seq=seq,
                        kv_dtype_bytes=kv_dtype_bytes,
                    )
                )

    return FitVerdict(rows=rows, best=pick_best(rows), notes=notes)
