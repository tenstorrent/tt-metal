# SPDX-License-Identifier: Apache-2.0

"""Sweep QK and PV subblock shapes for the retained q256 encoder SDPA.

Uses captured real layer-23 activations and the exact in-model precision/buffer
contract. Results are written to ``.auto/encoder_sdpa_subblock_sweep.csv``.

Run:
    TT_VISIBLE_DEVICES=0 python -m pytest \
        models/demos/wormhole/bge_m3/tests/perf/encoder_sdpa_subblock_sweep.py -sq
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.bge_m3.tt.custom_ops.encoder_sdpa import EncoderSDPAConfig
from models.demos.wormhole.bge_m3.tt.custom_ops.encoder_sdpa.op import build_encoder_sdpa_descriptor

CAPTURE_DIR = Path(".auto/qkv_capture")
CSV_PATH = Path(".auto/encoder_sdpa_subblock_sweep.csv")
TRACE_ITERATIONS = 12
PCC_GATE = 0.999

# q256 => Q block is 8 tiles; k2048 => K block is 64 tiles; DH is 2 tiles.
# Every shape below divides its parent block and occupies at most DEST8.
QK_SUBBLOCKS = ((2, 4), (4, 2), (1, 8), (8, 1), (2, 2), (1, 4), (4, 1))
OUT_SUBBLOCKS = ((4, 2), (8, 1), (2, 2), (1, 2), (2, 1))
BASELINE = ((2, 4), (4, 2))


def _load_capture(name: str) -> torch.Tensor:
    # The capture contains global B12. One N300 chip owns six examples in DP2.
    array = np.load(CAPTURE_DIR / f"{name}.npy", mmap_mode="r")
    return torch.from_numpy(np.array(array[:6], copy=True)).to(torch.bfloat16)


def _config(qk_subblock, out_subblock) -> EncoderSDPAConfig:
    return EncoderSDPAConfig(
        batch=6,
        q_chunk_size=256,
        k_chunk_size=2048,
        q_buffer_depth=2,
        k_buffer_depth=1,
        v_buffer_depth=1,
        fp32_dest_acc_en=False,
        direct_concat_heads=True,
        reuse_prev_max_for_exp=True,
        qk_subblock=qk_subblock,
        out_subblock=out_subblock,
    )


def _run_once(build):
    ttnn.generic_op(build.io_tensors, build.descriptor)


def _trace_bench(device, build):
    _run_once(build)
    ttnn.synchronize_device(device)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    _run_once(build)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    samples = []
    for _ in range(TRACE_ITERATIONS):
        start = time.perf_counter()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        samples.append((time.perf_counter() - start) * 1e3)
    actual = ttnn.to_torch(build.output).float()
    ttnn.release_trace(device, trace_id)
    samples.sort()
    return actual, samples[0], samples[len(samples) // 2]


@pytest.mark.parametrize("device_params", [{"trace_region_size": 20_000_000}], indirect=True)
def test_encoder_sdpa_subblock_sweep(device):
    q = ttnn.from_torch(
        _load_capture("q"),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    k = ttnn.from_torch(
        _load_capture("k"),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    v = ttnn.from_torch(
        _load_capture("v"),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    baseline_build = build_encoder_sdpa_descriptor(q, k, v, config=_config(*BASELINE))
    expected, baseline_min, baseline_median = _trace_bench(device, baseline_build)
    logger.info(
        f"SDPA_SUBBLOCK baseline qk={BASELINE[0]} out={BASELINE[1]} "
        f"min={baseline_min:.3f} median={baseline_median:.3f} ms"
    )

    rows = []
    for qk_subblock in QK_SUBBLOCKS:
        for out_subblock in OUT_SUBBLOCKS:
            row = {
                "qk_subblock_h": qk_subblock[0],
                "qk_subblock_w": qk_subblock[1],
                "out_subblock_h": out_subblock[0],
                "out_subblock_w": out_subblock[1],
                "status": "error",
                "pcc": "",
                "trace_min_ms": "",
                "trace_median_ms": "",
                "median_delta_ms": "",
                "error": "",
            }
            try:
                if (qk_subblock, out_subblock) == BASELINE:
                    actual = expected
                    trace_min = baseline_min
                    trace_median = baseline_median
                else:
                    build = build_encoder_sdpa_descriptor(q, k, v, config=_config(qk_subblock, out_subblock))
                    actual, trace_min, trace_median = _trace_bench(device, build)
                passed, pcc = comp_pcc(expected, actual, PCC_GATE)
                row.update(
                    status="pass" if passed else "pcc_fail",
                    pcc=float(pcc),
                    trace_min_ms=trace_min,
                    trace_median_ms=trace_median,
                    median_delta_ms=trace_median - baseline_median,
                )
            except Exception as error:
                row["error"] = " ".join(str(error).split())[:500]
            rows.append(row)
            logger.info(
                f"SDPA_SUBBLOCK qk={qk_subblock} out={out_subblock} status={row['status']} "
                f"median={row['trace_median_ms']} delta={row['median_delta_ms']} pcc={row['pcc']}"
            )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    passing = sorted(
        (row for row in rows if row["status"] == "pass"),
        key=lambda row: float(row["trace_median_ms"]),
    )
    assert passing, "no valid SDPA subblock configuration"
    logger.info(
        "SDPA_SUBBLOCK_BEST: "
        + ", ".join(
            f"qk={row['qk_subblock_h']}x{row['qk_subblock_w']}"
            f"/out={row['out_subblock_h']}x{row['out_subblock_w']}"
            f"={float(row['trace_median_ms']):.3f}ms"
            for row in passing[:8]
        )
    )
    logger.info(f"SDPA subblock sweep CSV: {CSV_PATH}")
