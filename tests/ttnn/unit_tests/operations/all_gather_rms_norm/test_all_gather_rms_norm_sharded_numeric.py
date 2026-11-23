# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 pins: HEIGHT / WIDTH_SHARDED input (zero-copy cb_x, sharded output landing) and fp32_dest_acc_en=False.

Each case goes through the golden harness's own builder (eval.golden_tests.all_gather_rms_norm.helpers
run_all_gather_rms_norm): a sharded input gets the auto_shard_config L1 shard spec of the per-device shape and a
sharded output of the same strategy for the gathered shape; accuracy is gated with helpers.TOLERANCES.

Paths pinned (op_design.md / changelog Refinement 1):
  HEIGHT, one tile-row per shard ...... cb_out_block backed on the own slice of the output shard (out_in_place)
  HEIGHT, W tail ...................... the caller's shard keeps its padding: mask block x DestReuse in compute
  WIDTH, k = #shards, P > 32 (LN) ...... LN combine over several gather tiles (exact ReduceTile)
  WIDTH, W tail on a 1-tile member with several tile-rows (masked sums, crt > 1)
  WIDTH, ragged last shard at R = 1
  16-bit DEST, interleaved and sharded, RMS and LN
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


HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED
IL = ttnn.TensorMemoryLayout.INTERLEAVED
BF16, BF8, F32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32

# (shape, memory_layout, norm_type, dtype, fp32_dest_acc_en)
CASES = [
    ((1, 1, 640, 384), HS, "rms_norm", BF16, True),  # 20 one-row shards, out_in_place
    ((1, 1, 640, 384), HS, "layer_norm", BF16, True),
    ((1, 1, 640, 1536), HS, "rms_norm", F32, True),  # needs out_in_place to fit L1
    ((1, 1, 32, 50), HS, "layer_norm", BF16, True),  # W tail masked, shard not zeroed
    ((1, 1, 640, 384), WS, "rms_norm", BF16, True),  # k = 12, P = 48
    ((1, 1, 32, 384), WS, "layer_norm", BF16, True),  # LN P = 48: two gather tiles
    ((2, 1, 128, 100), WS, "layer_norm", BF16, True),  # 1-tile tail member, crt = 8
    ((2, 1, 128, 100), WS, "rms_norm", BF16, True),
    ((1, 1, 32, 8192), WS, "rms_norm", BF16, True),  # ragged last shard, R = 1, P = 344
    ((1, 1, 640, 384), IL, "rms_norm", BF16, False),  # 16-bit DEST
    ((1, 1, 640, 384), IL, "layer_norm", BF8, False),
    ((1, 1, 64, 1000), IL, "layer_norm", BF16, False),  # 16-bit DEST + W tail (interleaved)
    ((1, 1, 640, 384), HS, "layer_norm", BF16, False),  # 16-bit DEST + sharded
]


def _id(c):
    shape, ml, norm, dtype, f32 = c
    return f"{'x'.join(map(str, shape))}-{ml.name}-{norm}-{dtype.name}-fp32dest{int(f32)}"


@pytest.mark.parametrize("device_params", [pytest.param(_device_params(), id="torus_xy")], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4")], indirect=True)
@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_sharded_and_16bit_dest(mesh_device, case):
    shape, memory_layout, norm_type, dtype, fp32_dest = case
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
        memory_layout=memory_layout,
        cluster_axis=1,
        topology=ttnn.Topology.Ring,
        num_links=2,
        norm_type=norm_type,
        beta_mode="beta" if ln else "no_beta",
    )
