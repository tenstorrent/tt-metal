# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device time of ttnn.linear program configs on the Galaxy mesh (a measurement tool, not a gate).

KDA_MM_CASES is a JSON list of cases: {"name", "M", "K", "N", "kind": "2d" | "1d", "cols", "rows", "k_block",
"sub_h", "sub_w", "per_core_M"?, "per_core_N"?, "fidelity": "HiFi2", "wdtype": "bfloat8_b", "out": "L1" | "DRAM",
"in0": "DRAM" | "L1", "out_block_w"?, "out_block_h"?, "activation"?}. "activation" fuses a unary op into the matmul's
last K block: an op name ("sigmoid", the FP32-accurate RC sigmoid) or [name, params...] (e.g. ["sigmoid", 1, 0] for
the row-vector mode). out_block_w / out_block_h below per_core_N / per_core_M shrink the in1 / in0 circular buffers,
which lets larger k_blocks fit L1 (in1 holds 2 x k_block x out_block_w tiles). Each case runs replicated on all 32 chips; the slowest chip's realtime-profiler duration is
reported (median of KDA_MM_RUNS runs). Results go to stdout as KDA_MM= lines and to KDA_MM_OUT as JSON.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import _L1_SMALL_SIZE
from models.demos.deepseek_v3_d_p.tt.kda.config import KDA_WORKER_L1_SIZE
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

_CASES = json.loads(os.environ.get("KDA_MM_CASES", "[]"))
_RUNS = int(os.environ.get("KDA_MM_RUNS", "5"))
_OUT = Path(os.environ.get("KDA_MM_OUT", "generated/kda_mm.json"))
_T = 32


def _activation(case: dict) -> ttnn.UnaryWithParam | None:
    spec = case.get("activation")
    if spec is None:
        return None
    name, *params = [spec] if isinstance(spec, str) else spec
    return ttnn.UnaryWithParam(getattr(ttnn.UnaryOpType, name.upper()), *[float(p) for p in params])


def _program_config(case: dict):
    m_tiles, n_tiles = case["M"] // _T, case["N"] // _T
    grid = ttnn.CoreCoord(case["cols"], case["rows"])
    if case.get("kind", "2d") == "1d":
        cores = case["cols"] * case["rows"]
        per_core_m = case.get("per_core_M", m_tiles)
        per_core_n = case.get("per_core_N", -(-n_tiles // cores))
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=case["k_block"],
            out_subblock_h=case["sub_h"],
            out_subblock_w=case["sub_w"],
            out_block_h=case.get("out_block_h", per_core_m),
            out_block_w=case.get("out_block_w", per_core_n),
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=_activation(case),
            mcast_in0=True,
        )
    per_core_m = case.get("per_core_M", -(-m_tiles // case["rows"]))
    per_core_n = case.get("per_core_N", -(-n_tiles // case["cols"]))
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=case["k_block"],
        out_subblock_h=case["sub_h"],
        out_subblock_w=case["sub_w"],
        out_block_h=case.get("out_block_h", per_core_m),
        out_block_w=case.get("out_block_w", per_core_n),
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=case.get("transpose_mcast", False),
        fused_activation=_activation(case),
        fuse_batch=True,
    )


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=_L1_SMALL_SIZE, worker_l1_size=KDA_WORKER_L1_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        ),
    ],
    indirect=True,
)
def test_projection_microbench(mesh_device: ttnn.MeshDevice, device_params: dict) -> None:
    """Time each KDA_MM_CASES config of ttnn.linear."""
    if not _CASES:
        pytest.skip("KDA_MM_CASES is empty")
    require_realtime_profiler("the projection microbenchmark")
    results = []
    tensors = {}
    for case in _CASES:
        key = (case["M"], case["K"], case["N"], case.get("wdtype", "bfloat8_b"), case.get("in0", "DRAM"))
        if key not in tensors:
            for old in tensors.values():
                for t in old:
                    ttnn.deallocate(t)
            tensors.clear()
            replicate = ttnn.ReplicateTensorToMesh(mesh_device)
            a = ttnn.from_torch(
                torch.randn(1, 1, case["M"], case["K"]) * 0.1,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.L1_MEMORY_CONFIG if case.get("in0") == "L1" else ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
            )
            w = ttnn.from_torch(
                torch.randn(1, 1, case["K"], case["N"]) * 0.02,
                dtype=getattr(ttnn, case.get("wdtype", "bfloat8_b")),
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
            )
            tensors[key] = (a, w)
        a, w = tensors[key]
        compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, case.get("fidelity", "HiFi2")),
            fp32_dest_acc_en=case.get("fp32_acc", True),
            packer_l1_acc=case.get("packer_l1_acc", True),
        )
        memory = ttnn.L1_MEMORY_CONFIG if case.get("out", "L1") == "L1" else ttnn.DRAM_MEMORY_CONFIG
        try:
            config = _program_config(case)

            def run():
                return ttnn.linear(a, w, program_config=config, memory_config=memory, compute_kernel_config=compute)

            ttnn.deallocate(run())  # compile
            durations = []
            for _ in range(_RUNS):
                out, programs = profile_realtime_program_merged(mesh_device, run)
                ttnn.deallocate(out)
                durations.append(max(p["duration_ns"] for p in programs.values()) / 1e3)
            entry = {**case, "us": statistics.median(durations), "runs_us": durations}
        except Exception as error:  # a config the op rejects, or one that does not fit L1
            entry = {**case, "error": str(error).splitlines()[0][:300]}
        results.append(entry)
        print("KDA_MM=" + json.dumps(entry))
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(results))
