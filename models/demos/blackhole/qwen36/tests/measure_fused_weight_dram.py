# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure the per-device DRAM cost of the fused decode weights (MLP w13 + attention wqkv_all).

Loads the 27B TP model and reports total DRAM allocated per device (num_banks ×
total_bytes_allocated_per_bank). Run twice — once with the fused weights built (default) and
once with QWEN_SKIP_FUSED_WEIGHTS=1 — and the delta is the DRAM these duplicate fused weights add
(they coexist with the separate prefill weights). No forward is run (load + query only).

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B QWEN_BENCH_B=8 \
      pytest models/demos/blackhole/qwen36/tests/measure_fused_weight_dram.py -v -s
    (repeat with QWEN_SKIP_FUSED_WEIGHTS=1)
"""
import os

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


def _dram_allocated_bytes(dev):
    v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    return int(v.num_banks) * int(v.total_bytes_allocated_per_bank)


@parametrize_mesh_tp()
def test_measure_fused_weight_dram(mesh_device, ensure_gc):
    from loguru import logger

    os.environ.setdefault("HF_MODEL", model_path())
    B = int(os.environ.get("QWEN_BENCH_B", "8"))
    skip = os.environ.get("QWEN_SKIP_FUSED_WEIGHTS") == "1"

    _ = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=2048)

    # Per-device DRAM allocated after load.
    try:
        devs = list(mesh_device.get_devices())
    except Exception:
        devs = [mesh_device]
    per_dev = [_dram_allocated_bytes(d) for d in devs]
    gb = [b / (1024**3) for b in per_dev]
    tag = "SKIP_FUSED" if skip else "WITH_FUSED"
    logger.info(f"FUSED_WEIGHT_DRAM [{tag}] per-device GB: " + ", ".join(f"{x:.3f}" for x in gb))
    logger.info(f"FUSED_WEIGHT_DRAM [{tag}] dev0={gb[0]:.3f} GB  mean={sum(gb)/len(gb):.3f} GB  ndev={len(devs)}")
