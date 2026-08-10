# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test the op-allocated stats scratch in ttnn.fused_rms_minimal (the rms_allgather op).

Covers cross-device E(x^2) averaging, program cache hits and trace replay against a freshly
allocated scratch, and rejection of an undersized caller-provided buffer. Requires a TG (8x4).
"""

import torch
import pytest
from loguru import logger
import ttnn

from models.common.utility_functions import skip_for_blackhole
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_and_get_pcc


def get_torch_rms(x, gamma, eps):
    """fp32 RMSNorm over the last dim."""
    x = x.to(torch.float32)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma.to(torch.float32)


def get_torch_rms_per_device(x, gamma, eps, num_devices):
    """RMSNorm where each device's slice is normalized by only its own E(x^2), i.e. the buggy output."""
    x = x.to(torch.float32)
    per = x.shape[-1] // num_devices
    out = torch.empty_like(x)
    for d in range(num_devices):
        sl = slice(d * per, (d + 1) * per)
        xd = x[..., sl]
        out[..., sl] = xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps)
    return out * gamma.to(torch.float32)


def build_heterogeneous_inputs(mesh_device, num_devices, hidden_size, seq_len, input_shard_grid):
    """Heterogeneous input, gamma, memory/program configs and semaphore for one fused_rms_minimal call."""
    ccl_sub_device_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 7))})
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

    torch.manual_seed(1234)

    total_cores = input_shard_grid.num_cores() * num_devices
    shard_width_per_core = ttnn.core.roundup(hidden_size // total_cores, ttnn.TILE_SIZE)
    shard_height = ttnn.core.roundup(seq_len, ttnn.TILE_SIZE)

    # Device d owns hidden columns [d*per : (d+1)*per] (dims=(3, None), cluster_axis=0). Scale each
    # device's slice by (d + 1) so E_d(x^2) ~ (d + 1)^2.
    x_torch = torch.randn((1, 1, seq_len, hidden_size))
    hidden_per_device = hidden_size // num_devices
    for d in range(num_devices):
        x_torch[..., d * hidden_per_device : (d + 1) * hidden_per_device] *= d + 1

    gamma_torch = torch.randn((1, 1, 1, hidden_size))

    input_memory_config = ttnn.create_sharded_memory_config(
        shape=(shard_height, shard_width_per_core),
        core_grid=input_shard_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    layer_norm_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(4, 8),
        subblock_w=1,
        block_h=1,
        block_w=shard_width_per_core // ttnn.TILE_SIZE,
        inplace=False,
    )

    input_tensor = ttnn.as_tensor(
        x_torch,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device=mesh_device, dims=(3, None), mesh_shape=list(ttnn.MeshShape(num_devices, 1))
        ),
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_memory_config,
    )
    gamma_tensor = ttnn.as_tensor(
        gamma_torch.reshape([1, 1, hidden_size // 32, 32]),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device=mesh_device, dims=(2, None), mesh_shape=list(ttnn.MeshShape(num_devices, 1))
        ),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return {
        "x_torch": x_torch,
        "gamma_torch": gamma_torch,
        "input_tensor": input_tensor,
        "gamma_tensor": gamma_tensor,
        "layer_norm_config": layer_norm_config,
        "output_memory_config": input_memory_config,
        "semaphore": ttnn.create_global_semaphore(mesh_device, input_shard_grid, 0),
    }


@skip_for_blackhole("This is a wormhole test")
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("num_iters", [3])
def test_rms_heterogeneous_internal_stats(mesh_device, topology, num_iters, function_level_defaults):
    """Omitting stats gives a correctly sized scratch, on the first call and on program cache hits."""
    num_devices = 8  # cluster_axis=0 -> 8 rows on the 8x4 mesh
    hidden_size = 896 * num_devices
    seq_len = 32
    epsilon = 1e-6
    input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))})

    ctx = build_heterogeneous_inputs(mesh_device, num_devices, hidden_size, seq_len, input_shard_grid)

    golden = get_torch_rms(ctx["x_torch"], ctx["gamma_torch"], epsilon)
    golden_per_device = get_torch_rms_per_device(ctx["x_torch"], ctx["gamma_torch"], epsilon, num_devices)

    outputs = []
    cache_entries = []
    for i in range(num_iters):
        outputs.append(
            ttnn.fused_rms_minimal(
                ctx["input_tensor"],
                ctx["layer_norm_config"],
                0,  # cluster_axis
                mesh_device,
                ctx["semaphore"],
                topology=topology,
                memory_config=ctx["output_memory_config"],
                epsilon=epsilon,
                dtype=ttnn.bfloat16,
                weight=ctx["gamma_tensor"],
                residual_input_tensor=None,
                stats=None,  # op allocates its own stats scratch
                use_noc1_only=False,
            )
        )
        cache_entries.append(mesh_device.num_program_cache_entries())
    ttnn.synchronize_device(mesh_device)

    # Assert the calls were cache hits, otherwise the loop could be recompiling each time and never
    # exercise override_runtime_arguments repointing the freshly allocated scratch.
    logger.info(f"program cache entries per iteration: {cache_entries}")
    assert (
        cache_entries[1:] == cache_entries[:-1]
    ), f"expected program cache hits after the first call, but entry count grew: {cache_entries}"

    for i, tt_out in enumerate(outputs):
        tt_out_torch = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=(num_devices, 1)),
        )[0].unsqueeze(0)

        passing, output, pcc = comp_and_get_pcc(tt_out_torch, golden, 0.999)
        _, _, pcc_per_device = comp_and_get_pcc(tt_out_torch, golden_per_device, 0.0)
        logger.info(f"[iter {i}] PCC vs full-hidden golden: {pcc}, vs per-device golden: {pcc_per_device}")

        assert passing, f"iter {i}: {output}"
        # The two goldens are far apart on 64x-heterogeneous data, so tracking the full-hidden one
        # shows all devices were averaged. There is no Python-visible device count to assert on.
        assert pcc > pcc_per_device + 0.05, (
            f"iter {i}: output is not clearly closer to the full-hidden golden ({pcc}) than to the "
            f"single-device reference ({pcc_per_device}); cross-device averaging may have collapsed"
        )

    mesh_device.reset_sub_device_stall_group()


@skip_for_blackhole("This is a wormhole test")
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 23887872, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("num_iters", [3])
def test_rms_internal_stats_trace(mesh_device, topology, num_iters, function_level_defaults):
    """The op-allocated scratch survives trace capture and replay.

    Checks the output after execute_trace, unlike the existing trace tests, so a stale cb_stats or
    writer address would fail here rather than pass silently.
    """
    num_devices = 8
    hidden_size = 896 * num_devices
    seq_len = 32
    epsilon = 1e-6
    input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))})

    ctx = build_heterogeneous_inputs(mesh_device, num_devices, hidden_size, seq_len, input_shard_grid)

    def run_op():
        return ttnn.fused_rms_minimal(
            ctx["input_tensor"],
            ctx["layer_norm_config"],
            0,  # cluster_axis
            mesh_device,
            ctx["semaphore"],
            topology=topology,
            memory_config=ctx["output_memory_config"],
            epsilon=epsilon,
            dtype=ttnn.bfloat16,
            weight=ctx["gamma_tensor"],
            residual_input_tensor=None,
            stats=None,  # op allocates its own stats scratch
            use_noc1_only=False,
        )

    # Compile outside the trace so capture only records device commands.
    run_op().deallocate(True)
    ttnn.synchronize_device(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(num_iters - 1):
        run_op().deallocate(True)
    tt_out = run_op()  # keep the last output alive so it can be read back after replay
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ttnn.release_trace(mesh_device, trace_id)

    tt_out_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=(num_devices, 1)),
    )[0].unsqueeze(0)

    golden = get_torch_rms(ctx["x_torch"], ctx["gamma_torch"], epsilon)
    golden_per_device = get_torch_rms_per_device(ctx["x_torch"], ctx["gamma_torch"], epsilon, num_devices)

    passing, output, pcc = comp_and_get_pcc(tt_out_torch, golden, 0.999)
    _, _, pcc_per_device = comp_and_get_pcc(tt_out_torch, golden_per_device, 0.0)
    logger.info(f"[trace] PCC vs full-hidden golden: {pcc}, vs per-device golden: {pcc_per_device}")

    assert passing, output
    assert pcc > pcc_per_device + 0.05, (
        f"traced output is not clearly closer to the full-hidden golden ({pcc}) than to the "
        f"single-device reference ({pcc_per_device}); cross-device averaging may have collapsed"
    )

    mesh_device.reset_sub_device_stall_group()


@skip_for_blackhole("This is a wormhole test")
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_rms_undersized_stats_rejected(mesh_device, topology, function_level_defaults, expect_error):
    """A stats buffer narrower than the cluster-axis device count is rejected, not silently degraded."""
    num_devices = 8
    hidden_size = 896 * num_devices
    seq_len = 32
    epsilon = 1e-6
    input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))})

    ctx = build_heterogeneous_inputs(mesh_device, num_devices, hidden_size, seq_len, input_shard_grid)

    # Sharding dim 3 across devices leaves each device a single 1-tile shard.
    undersized_stats = ttnn.from_torch(
        torch.zeros([1, 1, 32, num_devices], dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.create_sharded_memory_config(
            shape=(32, 32),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(3, None), mesh_shape=list(ttnn.MeshShape(num_devices, 1))
        ),
    )

    with expect_error(RuntimeError, "stats buffer of at least ring_size"):
        ttnn.fused_rms_minimal(
            ctx["input_tensor"],
            ctx["layer_norm_config"],
            0,
            mesh_device,
            ctx["semaphore"],
            topology=topology,
            memory_config=ctx["output_memory_config"],
            epsilon=epsilon,
            dtype=ttnn.bfloat16,
            weight=ctx["gamma_tensor"],
            residual_input_tensor=None,
            stats=undersized_stats,
            use_noc1_only=False,
        )

    mesh_device.reset_sub_device_stall_group()
