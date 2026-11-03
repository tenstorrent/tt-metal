# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verifier probe (not collected: no test_ prefix): per-chip breakdown of ONE call of the perf-focus shape.

Port cores (TRANSPORT) vs compute cores (attention), per chip, from the device profiler's kernel zones:
  op span, port-core kernel span (fence -> drain), compute-core span (min / median / max), and the compute tail
  after the last port finishes. Run:
    TT_METAL_DEVICE_PROFILER=1 scripts/run_safe_pytest.sh tests/.../probes/focus_breakdown.py -s
  FOCUS_FP32=1 (default; fp32_dest_acc_en=False is not SUPPORTED yet), FOCUS_HEADS=40, FOCUS_N=8192.
"""
import os

import pandas as pd
import pytest
import torch
import ttnn

from eval.golden_tests.ring_joint_scaled_dot_product_attention.conftest import _fabric_mesh_cache, mesh_device  # noqa
from ttnn.operations.ring_joint_scaled_dot_product_attention import ring_joint_scaled_dot_product_attention
from ttnn.operations.ring_joint_scaled_dot_product_attention import (
    ring_joint_scaled_dot_product_attention_program_descriptor as pdmod,
)

H = int(os.environ.get("FOCUS_HEADS", "40"))
N = int(os.environ.get("FOCUS_N", "8192"))
LINKS = int(os.environ.get("FOCUS_LINKS", "2"))


@pytest.mark.parametrize("mesh_shape,fabric_config", [((1, 4), ttnn.FabricConfig.FABRIC_2D_TORUS_XY)])
def test_focus_breakdown(mesh_shape, fabric_config, mesh_device):
    G = 4
    shp = (1, H, N, 128)
    mk = lambda: ttnn.from_torch(  # noqa: E731
        torch.randn((1, H, G * N, 128), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=mesh_shape),
    )
    q, k, v = mk(), mk(), mk()
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)
    call = lambda: ring_joint_scaled_dot_product_attention(  # noqa: E731
        q,
        k,
        v,
        logical_n=G * N - 8,
        cluster_axis=1,
        topology=ttnn.Topology.Ring,
        num_links=LINKS,
        compute_kernel_config=cfg,
    )
    call()
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)
    call()
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)
    chips, ring, G_, nl = pdmod.plan(mesh_device, cluster_axis=1, topology=ttnn.Topology.Ring, num_links=LINKS)
    log = "generated/profiler/.logs/profile_log_device.csv"
    df = pd.read_csv(log, skiprows=1, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    print("COLUMNS", list(df.columns))
    df = df[df["zone name"].astype(str).str.contains("KERNEL")]
    run_col = "run host ID" if "run host ID" in df.columns else None
    if run_col:
        last = df.groupby("PCIe slot")[run_col].transform("max")
        df = df[df[run_col] == last]
    freq_ghz = 1.35
    for dev, d in df.groupby("PCIe slot"):
        span = d.groupby(["core_x", "core_y"])["time[cycles since reset]"].agg(["min", "max"])
        t0 = span["min"].min()
        rel = (span - t0) / freq_ghz / 1e3  # us
        rel["dur"] = rel["max"] - rel["min"]
        print(f"CHIP {dev}: op span {rel['max'].max():.0f} us, cores {len(rel)}")
        print(
            "  longest 6 cores (us):", rel.sort_values("dur").tail(6)[["min", "max", "dur"]].round(0).to_dict("index")
        )
        print(
            "  shortest 6 cores (us):", rel.sort_values("dur").head(6)[["min", "max", "dur"]].round(0).to_dict("index")
        )
        print("  compute end quantiles (us):", rel["max"].quantile([0.0, 0.5, 1.0]).round(0).tolist())
    for coord, ch in chips.items():
        print("PORTS", coord, {k: (c.x, c.y) for k, c in ch["ports"].items()})
