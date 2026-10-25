# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Issue cycles per interleaved page read in a data-movement kernel (a measurement tool, not a gate).

Runs kernels/page_read_issue.cpp on one worker core of every chip. Each variant issues KDA_PRI_PAGES consecutive page
reads of an interleaved tensor (FP32 tiles, L1 or DRAM), KDA_PRI_REPS times, and records the cycles to issue them and
the cycles until the read barrier. Prints one KDA_PRI= line per (buffer, variant) with the median over chips and reps.
It also checks the pages() iterator's stepped addresses against get_noc_addr for many starts and strides.
"""

from __future__ import annotations

import json
import os
import statistics

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import _L1_SMALL_SIZE
from models.demos.deepseek_v3_d_p.tt.kda.config import KDA_WORKER_L1_SIZE

_PAGES = int(os.environ.get("KDA_PRI_PAGES", "16"))
_REPS = int(os.environ.get("KDA_PRI_REPS", "32"))
_CORE = tuple(int(x) for x in os.environ.get("KDA_PRI_CORE", "0,0").split(","))
_KERNEL = os.environ.get("KDA_PRI_KERNEL", "models/demos/deepseek_v3_d_p/tests/kda/perf/kernels/page_read_issue.cpp")
_VARIANTS = [
    "pages_iter_default",
    "page_id_default",
    "pages_iter_one_packet",
    "incremental_one_packet",
    "incremental_stateful",
    "floor_same_core_with_state",
    "addr_only_pages_iter",
    "addr_only_incremental",
]
_TOTAL_PAGES = 1024
_PAGE = 4096


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
@pytest.mark.parametrize("buffer", ["l1", "dram"])
def test_page_read_issue(mesh_device: ttnn.MeshDevice, device_params: dict, buffer: str) -> None:
    assert len(_VARIANTS) * _REPS * 2 <= 1023
    memory = ttnn.L1_MEMORY_CONFIG if buffer == "l1" else ttnn.DRAM_MEMORY_CONFIG
    src = ttnn.from_torch(
        torch.randn(1, 1, 32 * _TOTAL_PAGES, 32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out = ttnn.from_torch(
        torch.zeros(1, 1, 1, 1024, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*_CORE), ttnn.CoreCoord(*_CORE))])
    ct = [_PAGES, _REPS]
    ct += ttnn.TensorAccessorArgs(src).get_compile_time_args()
    ct += ttnn.TensorAccessorArgs(out).get_compile_time_args()

    def cb(index, page, pages, fmt):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
        )

    rt = ttnn.RuntimeArgs()
    rt[_CORE[0]][_CORE[1]] = [src.buffer_address(), out.buffer_address(), _TOTAL_PAGES]
    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=_KERNEL,
                core_ranges=core,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ],
        semaphores=[],
        cbs=[cb(0, _PAGE, _PAGES, ttnn.float32), cb(1, 4096, 1, ttnn.uint32)],
    )
    ttnn.generic_op([src, out], program)
    ttnn.synchronize_device(mesh_device)
    chips = [ttnn.to_torch(t).flatten().to(torch.int64) & 0xFFFFFFFF for t in ttnn.get_device_tensors(out)]
    for words in chips:
        assert int(words[1021]) > 0
        assert (
            int(words[1022]) == 0
        ), f"{int(words[1022])} of {int(words[1021])} iterator addresses differ from get_noc_addr"
    print(f"KDA_PRI_ADDR_CHECK={int(chips[0][1021])} pages per chip, 0 mismatches")
    for v, name in enumerate(_VARIANTS):
        issue, done = [], []
        for words in chips:
            for r in range(1, _REPS):  # rep 0 warms the instruction cache
                issue.append(int(words[(v * _REPS + r) * 2]))
                done.append(int(words[(v * _REPS + r) * 2 + 1]))
        entry = {
            "buffer": buffer,
            "variant": name,
            "pages": _PAGES,
            "issue_cycles_per_read": statistics.median(issue) / _PAGES,
            "complete_cycles_per_read": statistics.median(done) / _PAGES,
            "issue_min": min(issue) / _PAGES,
            "issue_max": max(issue) / _PAGES,
        }
        print("KDA_PRI=" + json.dumps(entry))
