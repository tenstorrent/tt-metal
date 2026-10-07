# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""In-process A/B of matmul_reduce_scatter variants: one mesh open per fabric payload, every case x variant in it,
device time read in-process (no tracy post-processing).

Run with TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1, under
scripts/run_safe_pytest.sh --no-precompile.

MMRS_AB_VARIANTS: ';'-separated variants, each 'name:ENV=V,ENV=V' (env applied before every call of that variant).
MMRS_AB_CASES (default focus,glm,mimo), MMRS_AB_PAYLOADS (default 14400,4352), MMRS_AB_ROUNDS (default 2: variants
interleaved per round), MMRS_AB_CALLS (measured calls per variant per round, default 5, after 2 warm-up calls).
Reported: per (payload, case, variant) the median over all measured calls of the slowest chip's kernel time."""

import os
import statistics

import pytest
import torch
import ttnn

from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter_program_descriptor as _pd

CASES = {
    "focus": ((1, 1, 640, 2048), (2048, 7168), 1, -1, ttnn.bfloat8_b, False),
    "glm": ((1, 1, 640, 4096), (4096, 6144), 1, -1, ttnn.bfloat8_b, False),
    "mimo": ((1, 1, 2048, 2048), (2048, 4096), 1, -2, ttnn.bfloat8_b, True),
    "smallk": ((1, 1, 640, 512), (512, 7168), 1, -1, ttnn.bfloat8_b, False),
    "r2": ((1, 1, 2048, 4096), (4096, 4096), 1, -1, ttnn.bfloat8_b, True),
}


def _variants():
    out = []
    for spec in os.environ.get("MMRS_AB_VARIANTS", "base:").split(";"):
        name, _, envs = spec.partition(":")
        out.append((name, dict(kv.split("=", 1) for kv in envs.split(",") if kv)))
    return out


def _router_params(payload):
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = payload
    return {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "fabric_router_config": cfg}


def _system_mesh_shape():
    shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    return tuple(shape[i] for i in range(shape.dims()))


def _to_mesh(stacked, mesh_device, dtype):
    rows = stacked.shape[0]
    glob = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        glob,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    )


def _slowest_chip_us(mesh_device, f):
    ttnn.ReadDeviceProfiler(mesh_device)
    f()
    ttnn.ReadDeviceProfiler(mesh_device)
    data = ttnn.get_latest_programs_perf_data()
    return max(
        sum(p.program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration for p in progs) / 1e3
        for progs in data.values()
    )


PAYLOADS = [int(p) for p in os.environ.get("MMRS_AB_PAYLOADS", "14400,4352").split(",")]


@pytest.mark.parametrize("mesh_device", [_system_mesh_shape()], indirect=True)
@pytest.mark.parametrize("device_params", [_router_params(p) for p in PAYLOADS], indirect=True, ids=lambda p: "")
def test_ab(mesh_device, device_params):
    payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
    variants = _variants()
    rounds = int(os.environ.get("MMRS_AB_ROUNDS", "2"))
    calls = int(os.environ.get("MMRS_AB_CALLS", "5"))
    base_env = {k: os.environ.get(k) for _, env in variants for k in env}
    pd_defaults = {k: getattr(_pd, k[3:]) for k in base_env if k.startswith("pd.")}
    rows, cols = tuple(mesh_device.shape)
    for case in os.environ.get("MMRS_AB_CASES", "focus,glm,mimo").split(","):
        a_shape, w_shape, axis, sd, wdt, fp32 = CASES[case]
        torch.manual_seed(0)
        a = _to_mesh(torch.randn((rows, cols, *a_shape)).to(torch.bfloat16), mesh_device, ttnn.bfloat16)
        w = _to_mesh(torch.randn((rows, cols, *w_shape)) * w_shape[0] ** -0.5, mesh_device, wdt)
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32)

        def call():
            return matmul_reduce_scatter(
                a, w, cluster_axis=axis, scatter_dim=sd, num_links=2, compute_kernel_config=cfg
            )

        times = {name: [] for name, _ in variants}
        ref = None
        for _ in range(rounds):
            for name, env in variants:
                for k, v in base_env.items():  # back to the defaults, then this variant's overrides
                    if k.startswith("pd."):
                        setattr(_pd, k[3:], pd_defaults[k])
                    else:
                        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
                for k, v in env.items():  # "pd.NAME=V": a descriptor-module knob (read per call / in the plan key)
                    if k.startswith("pd."):
                        setattr(_pd, k[3:], int(v))
                    else:
                        os.environ[k] = v
                mesh_device.disable_and_clear_program_cache()
                mesh_device.enable_program_cache()
                out = call()
                got = torch.cat([ttnn.to_torch(t).float().flatten() for t in ttnn.get_device_tensors(out)])
                if ref is None:
                    ref = got
                else:  # every variant computes the same sum (same precision contract)
                    pcc = torch.corrcoef(torch.stack([ref, got]))[0, 1].item()
                    assert pcc > 0.9999, (case, name, pcc)
                call()
                times[name] += [_slowest_chip_us(mesh_device, call) for _ in range(calls)]
        for k, v in base_env.items():
            if k.startswith("pd."):
                setattr(_pd, k[3:], pd_defaults[k])
            else:
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        for name, _ in variants:
            t = times[name]
            print(
                f"AB payload={payload} case={case:6s} {name:10s} median {statistics.median(t):7.1f} us"
                f"  (min {min(t):.1f}, max {max(t):.1f}, n={len(t)})"
            )
