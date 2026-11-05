# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DRAM accounting of the vision tower (4x8 mesh, MiB per bank): weights (_load), persistent input buffers (_alloc_inputs), after warmup(), after capture_traces() (+ trace region use).
The server end-of-build shows +90 MiB/bank for 'vision init + warm' while the weights are about 12: this test splits it. Prints VDR lines."""
import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.vision import DSV41Vision, VisionArgs
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager


def dram(md):
    ttnn.synchronize_device(md)
    mv = ttnn.get_memory_view(md, ttnn.BufferType.DRAM)
    try:
        tv = ttnn.get_memory_view(md, ttnn.BufferType.TRACE)
        tr = tv.total_bytes_allocated_per_bank / 2**20
    except Exception:  # noqa: BLE001
        tr = float("nan")
    return mv.total_bytes_allocated_per_bank / 2**20, tr


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 400_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_vision_dram(mesh_device):
    md = mesh_device
    marks = {}
    orig_load, orig_alloc = DSV41Vision._load, DSV41Vision._alloc_inputs

    def load(self):
        a = dram(md)[0]
        orig_load(self)
        marks["weights"] = dram(md)[0] - a

    def alloc(self):
        a = dram(md)[0]
        orig_alloc(self)
        marks["inputs"] = dram(md)[0] - a

    DSV41Vision._load, DSV41Vision._alloc_inputs = load, alloc
    try:
        d0, _ = dram(md)
        vision = DSV41Vision(
            md,
            mesh_4x8(),
            CCLManager(md, num_links=2, topology=ttnn.Topology.Ring),
            args=VisionArgs(),
            shards=_Shards(),
        )
        d1, _ = dram(md)
        print(
            f"VDR construct: total {d1 - d0:.1f} MiB/bank = weights {marks['weights']:.1f} + inputs {marks['inputs']:.1f} + other {d1 - d0 - marks['weights'] - marks['inputs']:.1f}",
            flush=True,
        )
        vision.warmup()
        d2, _ = dram(md)
        print(f"VDR after warmup(): +{d2 - d1:.1f} MiB/bank (cumulative {d2 - d0:.1f})", flush=True)
        vision.capture_traces()
        d3, t3 = dram(md)
        print(
            f"VDR after capture_traces(): +{d3 - d2:.1f} MiB/bank DRAM (cumulative {d3 - d0:.1f}), trace region used {t3:.1f} MiB/bank",
            flush=True,
        )
    finally:
        DSV41Vision._load, DSV41Vision._alloc_inputs = orig_load, orig_alloc
