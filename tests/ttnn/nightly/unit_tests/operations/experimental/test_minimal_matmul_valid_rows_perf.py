# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import os
import pytest
import torch
import ttnn
from loguru import logger

from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

from tests.ttnn.nightly.unit_tests.operations.experimental.test_minimal_matmul import (
    _uint32_scalar,
    _valid_rows_setup,
    perf_model,
    post_process_ops_log,
)

M_CAP, K, N, HEAD_DIM = 4096, 576, 2048, 128
BATCH_EXTENT, NUM_LAYERS, LAYER_IDX = 4, 2, 1
ITERS = 30
ROWS = [512, 1024, 2048, 4096]
MODES = ["dynamic", "dyn_nohead", "dyn_noslot", "dyn_tightcap", "static"]


def _static_setup(device, M):
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16)
    b = torch.randn(K, N, dtype=torch.bfloat16)
    return (
        ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("rows", ROWS)
def test_run_valid_rows_perf(device, mode, rows):
    cap = rows if mode == "dyn_tightcap" else M_CAP
    _, _, tt_cache, tt_weight, config, compute_config = _valid_rows_setup(device, BATCH_EXTENT, cap, K, N)
    if mode.startswith("dyn"):
        tt_valid = _uint32_scalar(device, rows)
        tt_slot = _uint32_scalar(device, 0)
        kwargs = dict(valid_rows_tensor=tt_valid, kv_num_layers=NUM_LAYERS, kv_layer_idx=LAYER_IDX)
        if mode != "dyn_noslot":
            kwargs["slot_tensor"] = tt_slot
        if mode != "dyn_nohead":
            kwargs["out_head_dim"] = HEAD_DIM
        act = tt_cache
    else:
        act, tt_weight = _static_setup(device, rows)
        kwargs = {}
    for _ in range(ITERS):
        out = ttnn.experimental.minimal_matmul(
            act, tt_weight, config=config, compute_kernel_config=compute_config, **kwargs
        )
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)


def test_valid_rows_perf_table():
    results = {}
    for mode in MODES:
        for rows in ROWS:
            subdir = f"mm_valid_rows_{mode}_{rows}"
            cmd = (
                "pytest tests/ttnn/nightly/unit_tests/operations/experimental/"
                f"test_minimal_matmul_valid_rows_perf.py::test_run_valid_rows_perf"
                f"[rows={rows}-mode={mode}]"
            )
            run_device_profiler(cmd, subdir, device_analysis_types=["device_kernel_duration"])
            r = post_process_ops_log(
                subdir,
                float_columns=["CORE COUNT", "DEVICE KERNEL DURATION [ns]"],
                sum_vals=False,
                has_signposts=False,
            )
            d = r["DEVICE KERNEL DURATION [ns]"]
            cc = int(r["CORE COUNT"].max())
            results[(mode, rows)] = (float(d.min()), float(d.mean()), len(d), cc)

    print("")
    print(f"| shape M x {K} x {N} | mode | cores | min us | mean us | ideal us | math util % |")
    print("|---|---|---|---|---|---|---|")
    for rows in ROWS:
        for mode in MODES:
            mn, mean, n, cc = results[(mode, rows)]
            ideal = perf_model(rows, K, N, cc, 2)
            print(
                f"| M={rows} | {mode:11s} | {cc} | {mn/1000:.2f} | {mean/1000:.2f} | "
                f"{ideal/1000:.2f} | {100*ideal/mn:.1f} |"
            )
    print("")
    for rows in ROWS:
        dyn = results[("dynamic", rows)][0]
        sta = results[("static", rows)][0]
        print(f"M={rows}: dynamic {dyn/1000:.2f} us vs static {sta/1000:.2f} us -> overhead {100*(dyn/sta-1):+.1f}%")


CAPS = [512, 1024, 2048, 4096, 8192]
FIXED_ROWS = 512
GRID = (11, 10)


def _runtime_m_blocks(m_cap, valid_rows, mbs, grid=GRID):
    m_cap_t, n_t, valid_t = m_cap // 32, N // 32, math.ceil(valid_rows / 32)
    m_cores = grid[0] if m_cap_t > n_t else grid[1]
    return math.ceil(math.ceil(valid_t / m_cores) / mbs)


@pytest.mark.parametrize("cap", CAPS)
def test_run_cap_sweep(device, cap):
    _, _, tt_cache, tt_weight, config, compute_config = _valid_rows_setup(device, BATCH_EXTENT, cap, K, N)
    tt_valid = _uint32_scalar(device, FIXED_ROWS)
    tt_slot = _uint32_scalar(device, 0)
    for _ in range(ITERS):
        out = ttnn.experimental.minimal_matmul(
            tt_cache,
            tt_weight,
            config=config,
            compute_kernel_config=compute_config,
            valid_rows_tensor=tt_valid,
            slot_tensor=tt_slot,
            kv_num_layers=NUM_LAYERS,
            kv_layer_idx=LAYER_IDX,
            out_head_dim=HEAD_DIM,
        )
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)


def test_cap_sweep_table():
    print("")
    print(f"| M_cap (valid_rows={FIXED_ROWS}) | cores | min us | mean us |")
    print("|---|---|---|---|")
    for cap in CAPS:
        subdir = f"mm_cap_{cap}"
        cmd = (
            "pytest tests/ttnn/nightly/unit_tests/operations/experimental/"
            f"test_minimal_matmul_valid_rows_perf.py::test_run_cap_sweep[cap={cap}]"
        )
        run_device_profiler(cmd, subdir, device_analysis_types=["device_kernel_duration"])
        r = post_process_ops_log(
            subdir, float_columns=["CORE COUNT", "DEVICE KERNEL DURATION [ns]"], sum_vals=False, has_signposts=False
        )
        d = r["DEVICE KERNEL DURATION [ns]"]
        print(f"| {cap} | {int(r['CORE COUNT'].max())} | {d.min()/1000:.2f} | {d.mean()/1000:.2f} |")


@pytest.mark.parametrize("mbs", [8, 14, 16])
def test_run_mblock_sweep(device, mbs):
    _, _, tt_cache, tt_weight, _cfg, compute_config = _valid_rows_setup(device, BATCH_EXTENT, 4096, K, N)
    config = ttnn.MinimalMatmulConfig(
        M_block_size=mbs,
        K_block_size=9,
        N_block_size=8,
        subblock_h=2,
        subblock_w=2,
        compute_with_storage_grid_size=ttnn.CoreCoord(*GRID),
    )
    tt_valid = _uint32_scalar(device, FIXED_ROWS)
    tt_slot = _uint32_scalar(device, 0)
    for _ in range(ITERS):
        out = ttnn.experimental.minimal_matmul(
            tt_cache,
            tt_weight,
            config=config,
            compute_kernel_config=compute_config,
            valid_rows_tensor=tt_valid,
            slot_tensor=tt_slot,
            kv_num_layers=NUM_LAYERS,
            kv_layer_idx=LAYER_IDX,
            out_head_dim=HEAD_DIM,
        )
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)


def test_mblock_sweep_table():
    print("")
    print(f"| M_cap=4096 valid_rows={FIXED_ROWS} M_block_size | runtime M blocks/core | min us | mean us |")
    print("|---|---|---|---|")
    for mbs in [8, 14, 16]:
        subdir = f"mm_mbs_{mbs}"
        cmd = (
            "pytest tests/ttnn/nightly/unit_tests/operations/experimental/"
            f"test_minimal_matmul_valid_rows_perf.py::test_run_mblock_sweep[mbs={mbs}]"
        )
        run_device_profiler(cmd, subdir, device_analysis_types=["device_kernel_duration"])
        r = post_process_ops_log(
            subdir, float_columns=["CORE COUNT", "DEVICE KERNEL DURATION [ns]"], sum_vals=False, has_signposts=False
        )
        d = r["DEVICE KERNEL DURATION [ns]"]
        print(f"| {mbs} | {_runtime_m_blocks(4096, FIXED_ROWS, mbs)} | {d.min()/1000:.2f} | {d.mean()/1000:.2f} |")
