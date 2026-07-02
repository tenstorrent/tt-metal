# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Memory-layout sweep for the TextEncoder ``EmbeddingsDeviceOperation``.

The TextEncoder begins with ``ttnn.embedding(ids, table) -> [1, 1, B*T, C]`` (see
``tt_text_encoder.py``: ids are reshaped to ``[1, 1, B*T]`` so the whole CNN stack runs batch-
flattened). In the Tracy report (``textencoder/latest.log``) this is

    EmbeddingsDeviceOperation (in0:dram_interleaved)   4.05 us   1 op   3 cores   UINT32, BF16 => BF16

i.e. it runs on only **3 cores** because the production output is height-shardable at most 3 ways
(``B*T = 96`` rows = 3 tiles; height sharding caps at one tile-row per core). The embedding is a
pure data-movement *gather* — it has **no compute_kernel_config / math-fidelity knob** — so the only
real lever is the memory layout, which sets how many cores the gather is spread across.

HARD CONSTRAINTS lifted from ``embedding_device_operation.cpp`` (validate()):
  - input indices MUST be INTERLEAVED (DRAM or L1) — "Embedding does not support sharded inputs".
  - weight table MUST be INTERLEAVED (DRAM or L1) — "Embedding does not support sharded weights".
  - only the OUTPUT is shardable. For a TILE (tilized) output, HEIGHT / WIDTH / BLOCK sharding are
    all permitted as long as shard_h % 32 == 0, shard_w % 32 == 0, (B*T) % shard_h == 0 and
    C % shard_w == 0. (ROW_MAJOR output supports HEIGHT sharding only — not swept here because the
    downstream conv consumes the TILE output, and a row-major win would need a re-tilize.)

So at ``B*T = 96`` the interesting question is whether WIDTH-sharding the output over ``C = 512``
(16 tiles -> up to 16 cores) or BLOCK-sharding it (up to 3x16 = 48 cores) beats the 3-core
height-sharded / interleaved baseline by spreading the gather writes across more L1s.

This sweep enumerates the full cross product of
  - input placement   : dram_interleaved | l1_interleaved
  - weight placement  : dram_interleaved | l1_interleaved
  - output placement  : dram_interleaved | l1_interleaved | l1_height | l1_width | l1_block (TILE)
measures true device-kernel time (profiler read-back, 1 warmup + 1 timed run) and PCC vs a torch
``Embedding`` gather, then prints the fastest PCC-passing config plus the best of each output family.

REQUIRED ENV (test skips otherwise):
  - ``TT_METAL_DEVICE_PROFILER=1``
  - ``TT_METAL_PROFILER_MID_RUN_DUMP=1``

Run:
    export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
    pytest -s models/experimental/kokoro/perf/test_embedding_text_encoder_perf_sweep.py -v
    # override shape:  KOKORO_EMB_N=96 KOKORO_EMB_C=512 KOKORO_EMB_VOCAB=178
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc

from .test_matmul_decoder_perf_sweep import _device_kernel_us, _drain_profiler

TILE = 32
_PCC_TARGET = 0.999  # a gather is exact at bf16; anything below this is a real correctness bug


@dataclass(frozen=True)
class EmbCase:
    label: str
    in_mem: str  # "dram" | "l1"   (indices — always interleaved)
    wt_mem: str  # "dram" | "l1"   (table   — always interleaved)
    out_mem: str  # "dram" | "l1" | "height" | "width" | "block"
    grid_x: int = 0
    grid_y: int = 0

    @property
    def num_cores(self) -> int:
        if self.out_mem in ("dram", "l1"):
            return 0  # interleaved: core count is op-internal, not pinned by a shard grid
        return self.grid_x * self.grid_y


def _rect(num_cores: int, gx_max: int, gy_max: int) -> Optional[tuple[int, int]]:
    """Widest core-grid rectangle whose area is exactly ``num_cores`` and fits the live grid."""
    for y in range(1, gy_max + 1):
        if num_cores % y:
            continue
        x = num_cores // y
        if x <= gx_max:
            return x, y
    return None


def _interleaved_mem(kind: str) -> ttnn.MemoryConfig:
    return ttnn.DRAM_MEMORY_CONFIG if kind == "dram" else ttnn.L1_MEMORY_CONFIG


def _make_cases(n: int, c: int, gx_max: int, gy_max: int) -> list[EmbCase]:
    """All (input, weight, output) placements feasible for an ``[1, 1, N, C]`` tiled embedding."""
    n_tiles, c_tiles = n // TILE, c // TILE

    # ---- output-placement options (the dominant axis) -------------------------------------------
    out_opts: list[tuple[str, int, int]] = [  # (out_mem, grid_x, grid_y)
        ("dram", 0, 0),
        ("l1", 0, 0),
    ]
    # HEIGHT: split the N rows. shards = cores; each core gets n_tiles/cores tile-rows.
    for cores in range(2, n_tiles + 1):
        if n_tiles % cores:
            continue
        g = _rect(cores, gx_max, gy_max)
        if g:
            out_opts.append(("height", g[0], g[1]))
    # WIDTH: split the C columns. shards = cores; each core gets c_tiles/cores tile-cols.
    for cores in range(2, c_tiles + 1):
        if c_tiles % cores:
            continue
        g = _rect(cores, gx_max, gy_max)
        if g:
            out_opts.append(("width", g[0], g[1]))
    # BLOCK: gy splits N rows, gx splits C cols (both > 1 → genuine 2D block).
    for gy in range(2, n_tiles + 1):
        if n_tiles % gy or gy > gy_max:
            continue
        for gx in range(2, c_tiles + 1):
            if c_tiles % gx or gx > gx_max:
                continue
            out_opts.append(("block", gx, gy))

    cases: list[EmbCase] = []
    for in_mem in ("dram", "l1"):
        for wt_mem in ("dram", "l1"):
            for out_mem, gx, gy in out_opts:
                if out_mem in ("dram", "l1"):
                    label = f"in:{in_mem}|wt:{wt_mem}|out:{out_mem}"
                else:
                    label = f"in:{in_mem}|wt:{wt_mem}|out:{out_mem}_{gx}x{gy}({gx * gy}c)"
                cases.append(EmbCase(label, in_mem, wt_mem, out_mem, gx, gy))
    return cases


def _sharded_out_mem(case: EmbCase, n: int, c: int) -> ttnn.MemoryConfig:
    strategy = {
        "height": ttnn.ShardStrategy.HEIGHT,
        "width": ttnn.ShardStrategy.WIDTH,
        "block": ttnn.ShardStrategy.BLOCK,
    }[case.out_mem]
    return ttnn.create_sharded_memory_config(
        shape=[1, 1, n, c],
        core_grid=ttnn.CoreGrid(y=case.grid_y, x=case.grid_x),
        strategy=strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


@dataclass
class EmbResult:
    case: EmbCase
    dev_us: float
    pcc: float
    pcc_pass: bool
    err: Optional[str] = None


def _shape() -> tuple[int, int, int]:
    n = int(os.getenv("KOKORO_EMB_N", "96"))  # B*T (default B=2, T=48 — matches LN sweep)
    c = int(os.getenv("KOKORO_EMB_C", "512"))  # channels = hidden_dim
    vocab = int(os.getenv("KOKORO_EMB_VOCAB", "178"))  # n_token
    return n, c, vocab


def test_text_encoder_embedding_perf_sweep(device):
    if os.getenv("TT_METAL_DEVICE_PROFILER") is None:
        pytest.skip("device-time sweep needs a profiler build + TT_METAL_DEVICE_PROFILER=1")
    if os.getenv("TT_METAL_PROFILER_MID_RUN_DUMP") is None:
        pytest.skip("set TT_METAL_PROFILER_MID_RUN_DUMP=1 so ReadDeviceProfiler flushes mid-run")

    N, C, VOCAB = _shape()
    assert N % TILE == 0 and C % TILE == 0, f"shape N={N} C={C} must be tile-aligned for a tiled output"

    grid = device.compute_with_storage_grid_size()
    gx_max, gy_max = grid.x, grid.y
    cases = _make_cases(N, C, gx_max, gy_max)

    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB, (1, 1, N), dtype=torch.int32)
    table = torch.randn((VOCAB, C), dtype=torch.bfloat16)
    # Reference gather: [1, 1, N] -> [1, 1, N, C]
    ref = torch.nn.functional.embedding(ids.long().squeeze(0), table.float())  # [1, N, C]
    ref = ref.unsqueeze(0)  # [1, 1, N, C]

    results: list[EmbResult] = []
    for case in cases:
        ids_t = wt_t = warm = timed = None
        try:
            ids_t = ttnn.from_torch(
                ids,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=_interleaved_mem(case.in_mem),
            )
            # Table stored ROW_MAJOR (matches production: ttnn.embedding row-gathers in row-major).
            wt_t = ttnn.from_torch(
                table,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=_interleaved_mem(case.wt_mem),
            )

            if case.out_mem in ("dram", "l1"):
                out_mem = _interleaved_mem(case.out_mem)
            else:
                out_mem = _sharded_out_mem(case, N, C)

            def _run():
                return ttnn.embedding(ids_t, wt_t, layout=ttnn.TILE_LAYOUT, memory_config=out_mem)

            warm = _run()
            ttnn.synchronize_device(device)
            ttnn.deallocate(warm)
            warm = None
            _drain_profiler(device)
            ttnn.synchronize_device(device)

            timed = _run()
            ttnn.synchronize_device(device)
            dev_us = _device_kernel_us(device)
            if dev_us is None:
                raise RuntimeError("profiler produced no device data — is this a profiler build?")

            out = ttnn.to_torch(timed).float()
            pcc_pass, pcc = comp_pcc(ref, out, _PCC_TARGET)
            results.append(EmbResult(case, dev_us, float(pcc), bool(pcc_pass)))
        except Exception as e:
            results.append(EmbResult(case, float("inf"), 0.0, False, str(e).strip().splitlines()[0][:90]))
        finally:
            for t in (timed, warm, ids_t, wt_t):
                if t is not None:
                    try:
                        ttnn.deallocate(t)
                    except Exception:
                        pass
            ttnn.synchronize_device(device)

    passing = [r for r in results if r.pcc_pass]
    best = min(passing, key=lambda r: r.dev_us) if passing else None
    # Production baseline: indices + table in DRAM, interleaved DRAM TILE output.
    baseline = next(
        (r for r in results if r.case.in_mem == "dram" and r.case.wt_mem == "dram" and r.case.out_mem == "dram"),
        None,
    )

    logger.info(f"=== TextEncoder Embedding sweep  N={N} C={C} vocab={VOCAB}  grid={gx_max}x{gy_max} ===")
    logger.info(f"{'config':>44} {'cores':>5} {'dev_us':>9} {'PCC':>8} {'result':>7}  note")
    for r in sorted(results, key=lambda r: r.dev_us):
        c = r.case
        cores = c.num_cores if c.num_cores else "-"
        if r.err:
            logger.info(f"{c.label:>44} {str(cores):>5} {'-':>9} {'-':>8} {'ERROR':>7}  {r.err}")
            continue
        tag = "best" if r is best else ("baseline" if r is baseline else "")
        logger.info(
            f"{c.label:>44} {str(cores):>5} {r.dev_us:>9.2f} {r.pcc:>8.4f} "
            f"{('PASS' if r.pcc_pass else 'FAIL'):>7}  {tag}"
        )

    # Best PCC-passing config per output family — makes the strategy comparison explicit.
    logger.info("--- best PCC-pass per output strategy ---")
    for fam in ("dram", "l1", "height", "width", "block"):
        fam_pass = [r for r in passing if r.case.out_mem == fam]
        if fam_pass:
            bf = min(fam_pass, key=lambda r: r.dev_us)
            logger.info(f"  out:{fam:>6} -> {bf.dev_us:>7.2f}us  ({bf.case.label})")
        else:
            logger.info(f"  out:{fam:>6} -> (no PCC-passing config)")

    if best is not None and baseline is not None and baseline.dev_us != float("inf"):
        logger.info(
            f"FASTEST PCC-PASS: {best.case.label} -> {best.dev_us:.2f}us (PCC={best.pcc:.4f}) | "
            f"baseline {baseline.case.label}={baseline.dev_us:.2f}us | "
            f"speedup={baseline.dev_us / best.dev_us:.2f}x"
        )

    assert passing, f"No embedding config reached PCC>={_PCC_TARGET} — sweep harness broken."
