# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Multi-card MoE combine tests on a 1x4 line (FABRIC_1D, Linear topology, cluster_axis=1).

The fabric transport of ``selective_reduce_combine`` (Fabric Mux V2 between the worker cores and the fabric
routers) is otherwise covered only by tests/nightly/tg/ccl/moe, which needs a Galaxy host (Ring topology,
1x8 / 1x16 meshes, 2-4 links). These cases run the same goldens and validators on any four devices wired as a
line, on the two paths that build the combine program:

  - standalone ``ttnn.experimental.selective_reduce_combine``: exact match (``torch.equal``) against the 6U
    reference, two op calls (program-cache miss, then hit), a hidden size that splits a token row over four
    data-parallel cores and one that splits it over two;
  - fused ``ttnn.experimental.moe_compute`` FullCcl (tilize + matmul + activation + fabric combine) through the
    6U runner: PCC / allclose on every output, two iterations, once with one link and once with the link count
    left to the op (every link the fabric reports along the axis).

Skipped when fewer than four devices are available or the fabric could not be enabled. The 6U helpers are reused
verbatim (no golden, layout or validator logic lives here). Developed on a Blackhole p150 1x4 line.
"""

import pytest
import ttnn
from loguru import logger

from ttnn.operations.ccl import MoEActivationFunction, Topology
from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim, effective_matmul_ring_size

from tests.nightly.tg.ccl.moe import test_selective_combine_6U as combine_6u
from tests.nightly.tg.ccl.moe.test_moe_compute_6U import _run_moe_compute_impl

MESH_1X4 = [((1, 4), (1, 4))]

DEVICE_PARAMS_FABRIC_1D = {
    "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    "trace_region_size": 750000,
}


def _skip_unless_line_of_four_with_fabric(mesh_device):
    if mesh_device.get_num_devices() < 4:
        pytest.skip("needs four devices opened as a 1x4 line")
    if ttnn.get_fabric_config() == ttnn.FabricConfig.DISABLED:
        pytest.skip("needs the fabric enabled (FABRIC_1D)")


def _core_range_set(start, end):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*start), ttnn.CoreCoord(*end))])


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_FABRIC_1D], ids=["fabric_1d"], indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", MESH_1X4, ids=["1x4"], indirect=["mesh_device"])
@pytest.mark.parametrize("num_links", [1], ids=["links1"])
@pytest.mark.parametrize(
    "hidden_size, select_experts_k",
    [(7168, 8), (7168, 2), (4096, 8), (4096, 2)],
    ids=["h7168-k8", "h7168-k2", "h4096-k8", "h4096-k2"],
)
def test_selective_reduce_combine_linear_1x4(mesh_device, mesh_shape, hidden_size, select_experts_k, num_links):
    """Standalone combine over a 1x4 Linear axis, exact match against the 6U reference, cache miss and hit."""
    _skip_unless_line_of_four_with_fabric(mesh_device)

    batch, seq, cluster_axis = 64, 1, 1
    experts_per_device = 4  # the 6U generator requires select_experts_k < experts
    experts = experts_per_device * mesh_shape[cluster_axis]
    token_parallel_core_dim = 4
    # The 6U host layout cuts a token row into at most one fabric packet per core; the op must be called with the
    # same split and a worker grid sized by it (bf16 7168 = 4 cores, 4096 = 2 cores).
    data_parallel_core_dim = combine_6u.effective_data_parallel_core_dim(hidden_size, 4)
    worker_cores = _core_range_set((0, 0), (token_parallel_core_dim - 1, data_parallel_core_dim - 1))
    mux_cores = _core_range_set((4, 0), (5, 7))
    assert worker_cores.num_cores() == token_parallel_core_dim * data_parallel_core_dim

    for it in range(2):  # program-cache miss, then hit
        combine_6u._run_test(
            batch,
            experts,
            select_experts_k,
            hidden_size,
            seq,
            cluster_axis,
            worker_cores,
            data_parallel_core_dim,
            token_parallel_core_dim,
            num_links,
            mux_cores,
            mesh_device,
            num_test_iters=1,
            trace_mode=False,
            scheme="random",
            topology=Topology.Linear,
            # The op derives the per-token metadata stride from the buffer alignment; the generator pads each entry
            # to 16 B, which is the L1 alignment (DRAM on Blackhole is 64 B).
            metadata_memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        logger.info(
            f"selective_reduce_combine 1x4 Linear iter {it} exact match PASSED: batch={batch} k={select_experts_k} "
            f"hidden={hidden_size} experts={experts} tp={token_parallel_core_dim} dp={data_parallel_core_dim} "
            f"num_links={num_links}"
        )


# id, intermediate size N, hidden size, selected experts k, experts per device, tokens per device.
# Hidden 2048 with 32 tokens per device puts the combine on the critical path (many short rows); hidden 7168 with
# 8 tokens per device is the wide-row case (for configurations such as DeepSeek-V3-class experts).
_FUSED_SHAPES = [
    ("h2048_n512_k8_e4_t32", 512, 2048, 8, 4, 32),
    ("h7168_n2048_k8_e2_t8", 2048, 7168, 8, 2, 8),
]


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_FABRIC_1D], ids=["fabric_1d"], indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", MESH_1X4, ids=["1x4"], indirect=["mesh_device"])
@pytest.mark.parametrize("num_links", [1, None], ids=["links1", "links_auto"])
@pytest.mark.parametrize("shape", _FUSED_SHAPES, ids=[s[0] for s in _FUSED_SHAPES])
def test_moe_compute_fullccl_linear_1x4(mesh_device, mesh_shape, num_links, shape):
    """Fused moe_compute with the fabric combine over a 1x4 Linear axis; num_links None = every link the fabric
    reports along the axis (the op's default), so a line with two links per hop also runs the multi-link path."""
    _skip_unless_line_of_four_with_fabric(mesh_device)

    _, N, hidden_size, selected_experts_k, experts_per_device, tokens_per_device = shape
    ring_n = effective_matmul_ring_size(mesh_device)
    _run_moe_compute_impl(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        cluster_axis=1,
        experts_per_device=experts_per_device,
        tokens_per_device=tokens_per_device,
        selected_experts_k=selected_experts_k,
        num_layers=1,
        num_iterations=2,
        N=N,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size, matmul_ring_size=ring_n),
        dtype=ttnn.bfloat16,
        enable_trace=False,
        activation_type=MoEActivationFunction.SILU,
        has_bias=False,
        topology=Topology.Linear,
        num_links=num_links,
        mux_core_range=((1, 1), (3, 3)),
    )
