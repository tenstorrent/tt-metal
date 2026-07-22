# SPDX-License-Identifier: Apache-2.0

"""Sweep the model-local BGE QKV projection/scatter block configuration.

The sweep deliberately covers both BF16 input (encoder layer 0) and BF8 input
(layers 1-23).  A configuration is retainable only if it fits and passes on
both; the BF8-only descriptor probe does not expose layer-0 L1 failures.

Run:
    TT_VISIBLE_DEVICES=0 python -m pytest \
        models/demos/wormhole/bge_m3/tests/perf/qkv_scatter_config_sweep.py -sq

Results are written to ``.auto/qkv_scatter_config_sweep.csv`` and ranked by
traced median host wall time within each input dtype.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.bge_m3.tt.custom_ops.fused_qkv_heads.op import bge_qkv_heads_headsplit
from models.demos.wormhole.bge_m3.tt.custom_ops.qkv_scatter_matmul import QKVScatterConfig, bge_qkv_scatter_matmul

CSV_PATH = Path(".auto/qkv_scatter_config_sweep.csv")
PCC_GATE = 0.999
TRACE_ITERATIONS = 15

# Curated around the known m16/k8/n4 optimum.  Each axis is varied independently,
# then the most promising larger-M/larger-K combinations are included.  All
# entries obey the fixed 8x8 grid and 8-tile DST constraints.
CONFIGS = (
    QKVScatterConfig(8, 8, 4, 4, 2),
    QKVScatterConfig(12, 8, 4, 4, 2),
    QKVScatterConfig(16, 8, 4, 4, 2),  # retained default
    QKVScatterConfig(24, 8, 4, 2, 4),
    QKVScatterConfig(32, 8, 4, 4, 2),
    QKVScatterConfig(48, 8, 4, 4, 2),
    QKVScatterConfig(16, 4, 4, 4, 2),
    QKVScatterConfig(16, 16, 4, 4, 2),
    QKVScatterConfig(16, 8, 2, 4, 2),
    QKVScatterConfig(16, 8, 4, 8, 1),
    QKVScatterConfig(16, 8, 4, 2, 4),
    QKVScatterConfig(16, 8, 4, 2, 2),
    QKVScatterConfig(24, 16, 4, 2, 4),
    QKVScatterConfig(32, 16, 4, 4, 2),
)


def _label(config: QKVScatterConfig) -> str:
    return (
        f"m{config.M_block_size}k{config.K_block_size}n{config.N_block_size}"
        f"_sb{config.subblock_h}x{config.subblock_w}"
    )


def _make_reference(x, w, b):
    config = ttnn.MinimalMatmulConfig(
        M_block_size=16,
        K_block_size=8,
        N_block_size=4,
        subblock_h=4,
        subblock_w=2,
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        x.device().arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    qkv = ttnn.experimental.minimal_matmul(
        input_tensor=x,
        weight_tensor=w,
        bias_tensor=b,
        fused_activation=None,
        config=config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=compute_config,
    )
    outputs = bge_qkv_heads_headsplit(
        qkv,
        num_heads=16,
        head_groups=4,
        out_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        k_out_dtype=ttnn.bfloat4_b,
        v_out_dtype=ttnn.bfloat4_b,
    )
    ttnn.deallocate(qkv)
    return tuple(ttnn.to_torch(t) for t in outputs)


def _trace_trial(device, fn):
    warm_outputs = fn()
    ttnn.synchronize_device(device)
    for tensor in warm_outputs:
        ttnn.deallocate(tensor)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    captured = fn()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    samples = []
    for _ in range(TRACE_ITERATIONS):
        start = time.perf_counter()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        samples.append((time.perf_counter() - start) * 1e3)
    actual = tuple(ttnn.to_torch(t) for t in captured)
    ttnn.release_trace(device, trace_id)
    for tensor in captured:
        ttnn.deallocate(tensor)
    samples.sort()
    return actual, samples[0], samples[len(samples) // 2]


def _row(input_dtype: str, config: QKVScatterConfig) -> dict:
    return {
        "input_dtype": input_dtype,
        "config": _label(config),
        **asdict(config),
        "status": "error",
        "q_pcc": "",
        "k_pcc": "",
        "v_pcc": "",
        "trace_min_ms": "",
        "trace_median_ms": "",
        "error": "",
    }


@pytest.mark.parametrize("device_params", [{"trace_region_size": 10_000_000}], indirect=True)
def test_qkv_scatter_config_sweep(device):
    torch.manual_seed(0)
    weight_pt = torch.randn((1024, 3072), dtype=torch.bfloat16)
    bias_pt = torch.randn((1, 3072), dtype=torch.bfloat16)
    w = ttnn.from_torch(weight_pt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(bias_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    rows = []
    for dtype_name, input_dtype in (("bfloat16", ttnn.bfloat16), ("bfloat8_b", ttnn.bfloat8_b)):
        torch.manual_seed(1)
        x = ttnn.from_torch(
            torch.randn((6, 1, 8192, 1024), dtype=torch.bfloat16),
            dtype=input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        expected = _make_reference(x, w, b)
        ttnn.synchronize_device(device)

        for config in CONFIGS:
            row = _row(dtype_name, config)
            try:
                actual, trace_min, trace_median = _trace_trial(
                    device,
                    lambda config=config: bge_qkv_scatter_matmul(
                        x,
                        w,
                        bias_tensor=b,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        dtype=ttnn.bfloat8_b,
                        config=config,
                    ),
                )
                pccs = []
                for expected_tensor, actual_tensor in zip(expected, actual):
                    passed, pcc = comp_pcc(expected_tensor, actual_tensor, PCC_GATE)
                    pccs.append(float(pcc))
                    if not passed:
                        raise ValueError(f"PCC {pcc} is below {PCC_GATE}")
                row.update(
                    status="pass",
                    q_pcc=pccs[0],
                    k_pcc=pccs[1],
                    v_pcc=pccs[2],
                    trace_min_ms=trace_min,
                    trace_median_ms=trace_median,
                )
            except Exception as error:
                row["error"] = " ".join(str(error).split())[:500]
                logger.warning(f"QKV_SWEEP {dtype_name} {_label(config)} ERROR: {row['error']}")
            rows.append(row)
            logger.info(
                f"QKV_SWEEP {dtype_name} {_label(config)} status={row['status']} "
                f"median_ms={row['trace_median_ms']} q/k/v_pcc={row['q_pcc']}/{row['k_pcc']}/{row['v_pcc']}"
            )

        for tensor in expected:
            del tensor
        ttnn.deallocate(x)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    default_label = _label(QKVScatterConfig())
    for dtype_name in ("bfloat16", "bfloat8_b"):
        default_rows = [r for r in rows if r["input_dtype"] == dtype_name and r["config"] == default_label]
        assert (
            len(default_rows) == 1 and default_rows[0]["status"] == "pass"
        ), f"retained default failed for {dtype_name}: {default_rows}"
        passing = sorted(
            (r for r in rows if r["input_dtype"] == dtype_name and r["status"] == "pass"),
            key=lambda r: float(r["trace_median_ms"]),
        )
        logger.info(
            f"QKV_SWEEP_BEST {dtype_name}: "
            + ", ".join(f"{r['config']}={float(r['trace_median_ms']):.3f}ms" for r in passing[:5])
        )

    logger.info(f"QKV sweep CSV: {CSV_PATH}")
