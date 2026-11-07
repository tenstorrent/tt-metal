# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""FPU utilization of the Kimi-K3 KDA projection matmuls on the Galaxy mesh (a measurement tool, not a gate).

Each case runs replicated on all 32 chips; the slowest chip's realtime-profiler duration, in its own AI clock cycles,
is the case's cost (median of KDA_UTIL_RUNS single profiled calls after a warm-up). Utilization is the useful work,
M x K x N tiles at the fidelity's cycles per tile, spread over the case's cores, against those cycles:

    util = Mt * Kt * Nt * cycles_per_tile / (cols * rows) / cycles

so per-core padding (N not a multiple of the grid width) counts against it.

KDA_UTIL_CASES overrides the default cases with a JSON list of {"name", "M", "K", "N", "cols", "rows", "k_block",
"sub_h", "sub_w", "fidelity"?, "wdtype"?, "in0"?: "DRAM" | "L1", "out"?: "L1" | "DRAM", "per_core_N"?}. Results go to
stdout as KDA_UTIL= lines.
"""

from __future__ import annotations

import json
import os
import statistics
import time

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP

pytestmark = [run_for_blackhole(), pytest.mark.timeout(1800)]

_T = ttnn.TILE_SIZE
_CYCLES_PER_TILE = {"LoFi": 16, "HiFi2": 32, "HiFi3": 48, "HiFi4": 64}
_RUNS = int(os.environ.get("KDA_UTIL_RUNS", "5"))
_SETTLE_S = 0.3
_TIMEOUT_S = 10.0
_L1_SMALL_SIZE = 1152

# Galaxy SP8xTP4, one chip: 640 rows, K = hidden, N = q + k + v + decay rank + output gate + beta (24 heads, padded
# to a tile) = 389 tiles.
_HEADS = KimiK3Config.KDA_NUM_HEADS // 4
_INPUT_N = (
    3 * _HEADS * KimiK3Config.KDA_HEAD_DIM + KimiK3Config.KDA_HEAD_DIM + _HEADS * KimiK3Config.KDA_HEAD_DIM
) + _T
_DEFAULT = dict(M=PREFILL_CHUNK_TOKENS_PER_CHIP, K=KimiK3Config.EMB_SIZE, N=_INPUT_N, cols=11, rows=10)
_DEFAULT_CASES = [
    {**_DEFAULT, "name": "kb4_1x4", "k_block": 4, "sub_h": 1, "sub_w": 4},
    {**_DEFAULT, "name": "kb4_2x2", "k_block": 4, "sub_h": 2, "sub_w": 2},
    {**_DEFAULT, "name": "kb4_1x3", "k_block": 4, "sub_h": 1, "sub_w": 3},
    {**_DEFAULT, "name": "kb8_1x4", "k_block": 8, "sub_h": 1, "sub_w": 4},
    {**_DEFAULT, "name": "kb8_2x2", "k_block": 8, "sub_h": 2, "sub_w": 2},
]
_CASES = json.loads(os.environ["KDA_UTIL_CASES"]) if "KDA_UTIL_CASES" in os.environ else _DEFAULT_CASES


def _program_config(case: dict) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    per_core_m = -(-(case["M"] // _T) // case["rows"])
    per_core_n = case.get("per_core_N", -(-(-(-case["N"] // _T)) // case["cols"]))
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(case["cols"], case["rows"]),
        in0_block_w=case["k_block"],
        out_subblock_h=case["sub_h"],
        out_subblock_w=case["sub_w"],
        out_block_h=case.get("out_block_h", per_core_m),
        out_block_w=case.get("out_block_w", per_core_n),
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def _profiled_cycles(mesh_device: ttnn.MeshDevice, run) -> dict[int, int]:
    """Run ``run()`` once under the realtime profiler; return each chip's program cycles (one program expected)."""
    cycles: dict[int, int] = {}
    dropped = [0]

    def collect(batch):
        dropped[0] += int(batch.dropped)
        for record in batch.records:
            start, end = int(record.start_timestamp), int(record.end_timestamp)
            if record.runtime_id and end > start:
                chip = int(record.chip_id)
                cycles[chip] = cycles.get(chip, 0) + (end - start)

    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect)
    try:
        out = run()
        ttnn.synchronize_device(mesh_device)
        deadline, last, changed = time.monotonic() + _TIMEOUT_S, -1, time.monotonic()
        while time.monotonic() < deadline:
            if len(cycles) != last:
                last, changed = len(cycles), time.monotonic()
            elif cycles and time.monotonic() - changed > _SETTLE_S:
                break
            time.sleep(0.02)
    finally:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)
    ttnn.deallocate(out)
    assert cycles, "no realtime-profiler records"
    assert not dropped[0], f"{dropped[0]} realtime-profiler records dropped"
    return cycles


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        )
    ],
    indirect=True,
)
def test_projection_utilization(mesh_device: ttnn.MeshDevice, device_params: dict) -> None:
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    for case in _CASES:
        torch.manual_seed(0)
        a_torch = torch.randn(1, 1, case["M"], case["K"]) * 0.1
        w_torch = torch.randn(1, 1, case["K"], case["N"]) * 0.02
        a = ttnn.from_torch(
            a_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG if case.get("in0") == "L1" else ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        w = ttnn.from_torch(
            w_torch,
            dtype=getattr(ttnn, case.get("wdtype", "bfloat8_b")),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        fidelity = case.get("fidelity", "HiFi2")
        compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, fidelity),
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            dst_full_sync_en=case.get("dst_full_sync", False),
        )
        memory = ttnn.DRAM_MEMORY_CONFIG if case.get("out") == "DRAM" else ttnn.L1_MEMORY_CONFIG
        config = _program_config(case)

        def run():
            return ttnn.linear(a, w, program_config=config, memory_config=memory, compute_kernel_config=compute)

        try:
            first = run()
            ttnn.synchronize_device(mesh_device)
        except Exception as error:  # a config that does not fit is a result, not a failure
            print("KDA_UTIL=" + json.dumps({"name": case["name"], "error": str(error).splitlines()[0][:300]}))
            ttnn.deallocate(a)
            ttnn.deallocate(w)
            continue
        expected = torch.matmul(a_torch.float(), ttnn.to_torch(ttnn.get_device_tensors(w)[0]).float())
        got = ttnn.to_torch(ttnn.get_device_tensors(first)[0]).float()
        pcc = torch.corrcoef(torch.stack([expected.flatten(), got.flatten()]))[0, 1].item()
        ttnn.deallocate(first)

        if os.environ.get("KDA_UTIL_DEVICE_PROFILER"):
            # Under the device profiler (run_safe_pytest.sh --profile): a few plain calls, read from its op report.
            for _ in range(_RUNS):
                ttnn.deallocate(run())
            ttnn.synchronize_device(mesh_device)
            ttnn.ReadDeviceProfiler(mesh_device)
            print("KDA_UTIL=" + json.dumps({"name": case["name"], "pcc": round(float(pcc), 6)}), flush=True)
            ttnn.deallocate(a)
            ttnn.deallocate(w)
            continue
        _profiled_cycles(mesh_device, run)  # first profiled call discarded
        samples = [_profiled_cycles(mesh_device, run) for _ in range(_RUNS)]
        slowest = statistics.median(max(s.values()) for s in samples)
        chips = len(samples[0])
        cores = case["cols"] * case["rows"]
        ideal = (case["M"] // _T) * (case["K"] // _T) * (-(-case["N"] // _T)) * _CYCLES_PER_TILE[fidelity] / cores
        result = {
            "name": case["name"],
            "cores": cores,
            "per_core_N": config.per_core_N,
            "k_block": case["k_block"],
            "subblock": [case["sub_h"], case["sub_w"]],
            "fidelity": fidelity,
            "chips": chips,
            "slowest_cycles": int(slowest),
            "chip_cycles_min_med": [
                int(statistics.median(min(s.values()) for s in samples)),
                int(statistics.median(statistics.median(s.values()) for s in samples)),
            ],
            "out_block_w": config.out_block_w,
            "util_pct": round(100.0 * ideal / slowest, 2),
            "pcc": round(float(pcc), 6),
        }
        print("KDA_UTIL=" + json.dumps(result), flush=True)
        ttnn.deallocate(a)
        ttnn.deallocate(w)
