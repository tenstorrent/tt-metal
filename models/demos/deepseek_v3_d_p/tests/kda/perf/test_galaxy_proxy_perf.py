# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Galaxy SP8xTP4 K3 KDA (T=5120) per-chip work on an 8-chip LoudBox.

Galaxy chips each own 640 rows and 96/4 = 24 heads. Only the output reduce-scatter couples TP
ranks, so Galaxy's op list is the two parts below; merge their profiles with
``galaxy_proxy_merge.py``.

* ``test_galaxy_sp``: the full layer as SP8xTP1 with 24 heads on an (8,1) ring.
* ``test_galaxy_tp``: the [640, 7168] BF16 reduce-scatter over 4 chips.

Run each under ``scripts/run_safe_pytest.sh --profile``; the measured pass follows the
``galaxy_proxy`` signpost. Use a LoudBox with Galaxy's 12x10 grid.
"""

from __future__ import annotations

import statistics
import time

import pytest
import torch
from tracy import signpost

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import _PCC_THRESHOLD, _SEQUENCE
from models.demos.deepseek_v3_d_p.tests.kda.reference_cache import load_or_compute_cpu_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    check_kimi_k3_accuracy,
    deallocate_state,
    make_kimi_k3_device_case,
    make_synthetic_kimi_k3_test_case,
)
from models.tt_transformers.tt.ccl import TT_CCL
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = [run_for_blackhole(), pytest.mark.perf, pytest.mark.timeout(1800)]

GALAXY_SP, GALAXY_TP = 8, 4
LOCAL_ROWS = _SEQUENCE // GALAXY_SP
LOCAL_HEADS = KimiK3Config.KDA_NUM_HEADS // GALAXY_TP


def _trace_wall_ms(mesh_device: ttnn.MeshDevice, run, repetitions: int = 10) -> float:
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    run()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(repetitions):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        samples.append((time.perf_counter() - start) * 1e3 / repetitions)
    ttnn.release_trace(mesh_device, trace_id)
    return statistics.median(samples)


def _profile_one_pass(mesh_device: ttnn.MeshDevice, run) -> None:
    ttnn.ReadDeviceProfiler(mesh_device)
    signpost("galaxy_proxy")
    run()
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [((8, 1), fabric_1d_device_params(fabric_config=ttnn.FabricConfig.FABRIC_1D_RING))],
    indirect=True,
)
def test_galaxy_sp(mesh_device: ttnn.MeshDevice, device_params: dict) -> None:
    case = make_synthetic_kimi_k3_test_case(sequence=_SEQUENCE, num_heads=LOCAL_HEADS)
    layer, hidden = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=1, cache_weights=False)
    assert layer.active_seq_len_local == LOCAL_ROWS and layer.config.num_heads == LOCAL_HEADS
    state = layer.allocate_state(batch_size=1)
    actual_start = make_actual_start(mesh_device, 0)

    output, next_state = layer.forward(hidden, state, actual_start)
    golden_output, golden_state, _ = load_or_compute_cpu_reference(case)
    check_kimi_k3_accuracy(
        "Galaxy SP part",
        case,
        golden_output,
        golden_state,
        next_state,
        output,
        mesh_device,
        1,
        pcc_threshold=_PCC_THRESHOLD,
    )
    ttnn.deallocate(output)
    deallocate_state(next_state)

    def run() -> None:
        output, next_state = layer.forward(hidden, state, actual_start)
        ttnn.deallocate(output)
        deallocate_state(next_state)

    print(f"GALAXY_SP_TRACE_WALL_MS={_trace_wall_ms(mesh_device, run):.4f}")
    _profile_one_pass(mesh_device, run)
    deallocate_state(state)


@pytest.mark.parametrize("mesh_device,device_params", [((2, 4), fabric_1d_device_params())], indirect=True)
def test_galaxy_tp(mesh_device: ttnn.MeshDevice, device_params: dict) -> None:
    axis = 1
    ccl = TT_CCL(mesh_device)
    host = torch.randn(mesh_device.get_num_devices(), LOCAL_ROWS, KimiK3Config.EMB_SIZE, dtype=torch.bfloat16)
    partials = ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    def reduce_scatter() -> ttnn.Tensor:
        return ttnn.experimental.reduce_scatter_minimal_async(
            partials,
            dim=-1,
            multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(axis),
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(axis),
            num_links=ccl.get_num_links(axis),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            cluster_axis=axis,
        )

    output = reduce_scatter()
    shards = [ttnn.to_torch(shard).float().reshape(LOCAL_ROWS, -1) for shard in ttnn.get_device_tensors(output)]
    width = KimiK3Config.EMB_SIZE // GALAXY_TP
    for chip, shard in enumerate(shards):
        row, col = divmod(chip, GALAXY_TP)
        expected = host[row * GALAXY_TP : (row + 1) * GALAXY_TP].float().sum(0)[:, col * width : (col + 1) * width]
        torch.testing.assert_close(shard, expected, rtol=0.05, atol=0.5)
    ttnn.deallocate(output)

    def run() -> None:
        ttnn.deallocate(reduce_scatter())

    print(f"GALAXY_TP_TRACE_WALL_MS={_trace_wall_ms(mesh_device, run):.4f}")
    _profile_one_pass(mesh_device, run)
