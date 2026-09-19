# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""What the vocoder's T-halo exchanges cost on the 4x8 mesh (factor 8 over axis 1, one batch item per device as with the
batch shard): `_t_neighbor_pad` alone at every per-device band shape for the activation's replicate halo (5 sticks) and the
convs' zero halos (p = 1..25), plus the fused activation end to end for the same shapes. ~253 halo exchanges per decode.
    pytest models/tt_dit/tests/models/minimax_h3/tools/halo_probe.py -s
"""

import time

import pytest
import torch
from loguru import logger

import ttnn

from .....layers.audio_aa_snake import FusedActivation1d
from .....layers.audio_ops import _t_neighbor_pad
from .....parallel.config import ParallelFactor
from .....parallel.manager import CCLManager
from .....pipelines.minimax_h3.pipeline_minimax_h3 import resolve_mesh_preset

MESH = [
    pytest.param(
        (4, 8),
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True, "l1_small_size": 65536},
        id="mesh4x8",
    )
]
# (channels, pack, rows per device): bands 0-4 unpacked, 5-6 packed, act_post (repacked two per row by the module).
SHAPES = [(512, 1, 375), (256, 1, 1875), (128, 1, 3750), (64, 1, 7500), (32, 1, 15000), (16, 2, 15000), (8, 4, 15000), (8, 1, 60000)]
CONV_HALOS = (1, 3, 5, 9, 15, 25)


def _timed(mesh_device, fn, n=10):
    out = fn()
    ttnn.synchronize_device(mesh_device)
    best = float("inf")
    for _ in range(n):
        ttnn.synchronize_device(mesh_device)
        mark = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        best = min(best, time.perf_counter() - mark)
    return out, best * 1e3


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH, indirect=["mesh_device", "device_params"])
def test_halo_cost(mesh_device):
    torch.manual_seed(0)
    mesh_rows, mesh_cols = tuple(mesh_device.shape)
    pc = ParallelFactor(factor=mesh_cols, mesh_axis=1)
    preset = resolve_mesh_preset((mesh_rows, mesh_cols), required=False)
    ccl = CCLManager(mesh_device, num_links=preset.get("num_links", 1), topology=preset.get("topology", ttnn.Topology.Linear))
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(mesh_rows, mesh_cols), dims=(None, 1))
    total_halo_ms = 0.0
    for channels, pack, rows in SHAPES:
        x = ttnn.from_torch(
            torch.randn(1, rows * mesh_cols, pack * channels) * 0.5,
            device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32, mesh_mapper=mapper,
        )
        pad_act = -(-5 // pack)
        _, ms_rep = _timed(mesh_device, lambda: _t_neighbor_pad(x, pad_left=pad_act, pad_right=pad_act, parallel_config=pc, ccl_manager=ccl, padding_mode="replicate"))
        zeros = {}
        for p in CONV_HALOS:
            pr = -(-p // pack)
            _, zeros[p] = _timed(mesh_device, lambda: _t_neighbor_pad(x, pad_left=pr, pad_right=pr, parallel_config=pc, ccl_manager=ccl, padding_mode="zeros"))
        fused = FusedActivation1d(channels=channels, mesh_device=mesh_device, dtype=ttnn.float32, parallel_config=pc, ccl_manager=ccl)
        fused.load_torch_state_dict({"act.alpha": torch.randn(channels) * 0.3, "act.beta": torch.randn(channels) * 0.3})
        _, ms_fused = _timed(mesh_device, lambda: fused(x))
        zeros_txt = " ".join(f"p{p}:{ms:.3f}" for p, ms in zeros.items())
        logger.info(
            f"HALO c={channels} k={pack} rows/device={rows}: replicate halo(5) {ms_rep:.3f} ms; zero halos {zeros_txt} ms; "
            f"fused activation incl. its halo {ms_fused:.3f} ms (kernel share ~{ms_fused - ms_rep:.3f} ms)"
        )
        total_halo_ms += ms_rep
        ttnn.deallocate(x)
    logger.info(f"HALO_TOTAL one replicate halo per shape summed: {total_halo_ms:.2f} ms (the decode runs ~253 halo exchanges)")
