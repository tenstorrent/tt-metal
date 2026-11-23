# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1b pins: WIDTH_SHARDED at large R x k — on-chip row-owner pre-combine of the k shard members (the stats
table holds one record per (chip, row, stat)) + row-chunked per-row CBs.

Each case goes through the golden harness's own builder (helpers.run_all_gather_rms_norm): a WIDTH_SHARDED input with
the auto_shard_config L1 shard spec, a WIDTH_SHARDED output for the gathered shape, helpers.TOLERANCES.

  2048x384  RMS / LN ..... R = 64, k = 12: table was 393 KB (> STATS_TABLE_CAP), several row chunks per member
  640x1792  RMS .......... k = 56: two pre-combine gather tiles, one row per owner
  640x32    LN ........... k = 1, R = 20: the per-row LN CBs only fit row-chunked
  2x1x128x128 LN ......... several rows per owner, several chunks, aligned
  2x1x128x100 LN ......... the same with a W tail (masked tail member) and the R_U staged landing
"""

from __future__ import annotations

import pytest
import ttnn

from eval.golden_tests.all_gather_rms_norm.helpers import run_all_gather_rms_norm
from eval.golden_tests.all_gather_rms_norm.topology import ring_mock_meshes, system_mesh_shape, wrap_axes


def _device_params():
    params = {"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}
    try:
        cfg = ttnn._ttnn.fabric.FabricRouterConfig()
        cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
        params["fabric_router_config"] = cfg
    except Exception:  # noqa: BLE001 — fabric default payload (correctness is unaffected)
        pass
    return params


def _ring_ok(mesh_shape, cluster_axis):
    if (tuple(mesh_shape), cluster_axis) in [(tuple(s), a) for s, a in ring_mock_meshes()]:
        return True
    return system_mesh_shape()[cluster_axis] == mesh_shape[cluster_axis] and cluster_axis in wrap_axes()


BF16, F32 = ttnn.bfloat16, ttnn.float32

# (shape, norm_type, dtype, fp32_dest_acc_en)
CASES = [
    ((1, 1, 2048, 384), "rms_norm", BF16, True),
    ((1, 1, 2048, 384), "layer_norm", BF16, True),
    ((1, 1, 640, 1792), "rms_norm", BF16, True),
    ((1, 1, 640, 32), "layer_norm", BF16, True),
    ((2, 1, 128, 128), "layer_norm", BF16, True),
    ((2, 1, 128, 100), "layer_norm", BF16, True),
    ((1, 1, 2048, 384), "rms_norm", BF16, False),
]


def _id(c):
    shape, norm, dtype, f32 = c
    return f"{'x'.join(map(str, shape))}-{norm}-{dtype.name}-fp32dest{int(f32)}"


@pytest.mark.parametrize("device_params", [pytest.param(_device_params(), id="torus_xy")], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4")], indirect=True)
@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_width_precombine(mesh_device, case):
    shape, norm_type, dtype, fp32_dest = case
    if not _ring_ok(tuple(mesh_device.shape), 1):
        pytest.skip("no ring along cluster_axis 1 on this cluster")
    ln = norm_type == "layer_norm"
    run_all_gather_rms_norm(
        (shape,),
        mesh_device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=fp32_dest,
        gamma_mode="gamma",
        gamma_dtype=BF16,
        gamma_layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        cluster_axis=1,
        topology=ttnn.Topology.Ring,
        num_links=2,
        norm_type=norm_type,
        beta_mode="beta" if ln else "no_beta",
    )
