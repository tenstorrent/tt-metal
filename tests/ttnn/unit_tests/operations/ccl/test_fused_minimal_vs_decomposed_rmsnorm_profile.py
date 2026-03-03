# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkProfiler
from models.common.utility_functions import skip_for_blackhole
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def _get_submesh(mesh_device, num_devices, cluster_axis):
    submesh_shape = ttnn.MeshShape(num_devices, 1) if cluster_axis == 0 else ttnn.MeshShape(1, num_devices)
    test_mesh = mesh_device.create_submesh(submesh_shape)
    assert test_mesh.get_num_devices() == num_devices
    return test_mesh


def _build_inputs(mesh_device, seq_len=1, hidden_dim=8192, epsilon=1e-5, batch_size=32, cluster_axis=1):
    torch.manual_seed(0)
    num_devices = mesh_device.get_num_devices()
    token_count = batch_size * seq_len
    torch_input_bsh = torch.randn((batch_size, seq_len, hidden_dim), dtype=torch.float32)
    torch_input_2d = torch_input_bsh.reshape(token_count, hidden_dim)
    torch_input_4d = torch_input_bsh.reshape(1, 1, token_count, hidden_dim)
    torch_weight = torch.randn((1, hidden_dim), dtype=torch.bfloat16)

    cluster_axis_size = mesh_device.shape[cluster_axis]
    scale = torch.tensor([[1.0 / (hidden_dim * cluster_axis_size)]], dtype=torch.float32)
    eps = torch.tensor([[epsilon]], dtype=torch.float32)

    stats = torch.sum(torch_input_2d * torch_input_2d, dim=-1, keepdim=True)
    inv_rms = torch.rsqrt((stats * scale) + eps)
    expected = torch_weight * (torch_input_2d * inv_rms).to(torch.bfloat16)

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "token_count": token_count,
        "hidden_dim": hidden_dim,
        "epsilon": epsilon,
        "num_devices": num_devices,
        "cluster_axis": cluster_axis,
        "torch_input_bsh": torch_input_bsh,
        "torch_input_4d": torch_input_4d,
        "torch_weight": torch_weight,
        "scale": scale,
        "eps": eps,
        "expected": expected,
    }


def _prepare_decomposed_state(mesh_device, inputs):
    mesh_shape = (
        ttnn.MeshShape(inputs["num_devices"], 1)
        if inputs["cluster_axis"] == 0
        else ttnn.MeshShape(1, inputs["num_devices"])
    )
    per_chip_hidden = inputs["hidden_dim"] // inputs["num_devices"]
    input_shard_dims = (3, None) if inputs["cluster_axis"] == 0 else (None, 3)
    weight_shard_dims = (1, None) if inputs["cluster_axis"] == 0 else (None, 1)
    tt_input = ttnn.from_torch(
        inputs["torch_input_4d"],
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device=mesh_device, dims=input_shard_dims, mesh_shape=list(mesh_shape)),
    )
    tt_weight = ttnn.from_torch(
        inputs["torch_weight"],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device=mesh_device, dims=weight_shard_dims, mesh_shape=list(mesh_shape)
        ),
    )
    tt_scale = ttnn.from_torch(
        inputs["scale"],
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_eps = ttnn.from_torch(
        inputs["eps"],
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return {
        "per_chip_hidden": per_chip_hidden,
        "tt_input": tt_input,
        "tt_weight": tt_weight,
        "tt_scale": tt_scale,
        "tt_eps": tt_eps,
    }


def _run_decomposed(mesh_device, inputs, log_shapes=False, state=None):
    state = state if state is not None else _prepare_decomposed_state(mesh_device, inputs)
    per_chip_hidden = state["per_chip_hidden"]
    tt_input = state["tt_input"]
    tt_weight = state["tt_weight"]
    tt_scale = state["tt_scale"]
    tt_eps = state["tt_eps"]

    if log_shapes:
        logger.info("[decomp] tt_input shape={}", tuple(tt_input.shape))
        logger.info("[decomp] tt_weight shape={}", tuple(tt_weight.shape))
        logger.info("[decomp] tt_scale shape={}", tuple(tt_scale.shape))
        logger.info("[decomp] tt_eps shape={}", tuple(tt_eps.shape))

    # Based on the provided decomposition sequence, in local per-chip hidden space.
    tt_input_3d = ttnn.reshape(tt_input, (inputs["token_count"], 1, per_chip_hidden))
    if log_shapes:
        logger.info(
            "[decomp][reshape] input4d={} -> input3d=({}, 1, {})",
            tuple(tt_input.shape),
            inputs["token_count"],
            per_chip_hidden,
        )
        logger.info("[decomp] tt_input_3d shape={}", tuple(tt_input_3d.shape))
    tt_pow2 = ttnn.pow(tt_input_3d, 2.0)
    if log_shapes:
        logger.info("[decomp] tt_pow2 shape={}", tuple(tt_pow2.shape))
    tt_sum = ttnn.sum(tt_pow2, dim=2, keepdim=False)
    if log_shapes:
        logger.info("[decomp] tt_sum shape={}", tuple(tt_sum.shape))
    tt_sum_4d = ttnn.reshape(tt_sum, (1, 1, inputs["token_count"], 1))
    if log_shapes:
        logger.info(
            "[decomp][reshape] sum2d={} -> sum4d=(1, 1, {}, 1)",
            tuple(tt_sum.shape),
            inputs["token_count"],
        )
        logger.info("[decomp] tt_sum_4d shape={}", tuple(tt_sum_4d.shape))
    cluster_axis = inputs["cluster_axis"]
    tt_ar = ttnn.all_reduce(tt_sum_4d, cluster_axis=cluster_axis, topology=ttnn.Topology.Linear)
    if log_shapes:
        logger.info("[decomp] tt_ar(all_reduce) shape={}", tuple(tt_ar.shape))
    tt_stats_bsx1 = ttnn.reshape(tt_ar, (inputs["batch_size"], inputs["seq_len"], 1))
    if log_shapes:
        logger.info(
            "[decomp][reshape] allreduce4d={} -> stats3d=({}, {}, 1)",
            tuple(tt_ar.shape),
            inputs["batch_size"],
            inputs["seq_len"],
        )
        logger.info("[decomp] tt_stats_bsx1 shape={}", tuple(tt_stats_bsx1.shape))
    tt_scaled = ttnn.multiply(tt_stats_bsx1, tt_scale, dtype=ttnn.float32)
    if log_shapes:
        logger.info("[decomp] tt_scaled shape={}", tuple(tt_scaled.shape))
    tt_shifted = ttnn.add(tt_scaled, tt_eps, dtype=ttnn.float32)
    if log_shapes:
        logger.info("[decomp] tt_shifted shape={}", tuple(tt_shifted.shape))
    tt_inv_rms = ttnn.rsqrt(tt_shifted)
    if log_shapes:
        logger.info("[decomp] tt_inv_rms shape={}", tuple(tt_inv_rms.shape))
    tt_norm = ttnn.multiply(tt_input_3d, tt_inv_rms, dtype=ttnn.float32)
    if log_shapes:
        logger.info("[decomp] tt_norm shape={}", tuple(tt_norm.shape))
    tt_norm_bf16 = ttnn.typecast(tt_norm, dtype=ttnn.bfloat16)
    if log_shapes:
        logger.info("[decomp] tt_norm_bf16 shape={}", tuple(tt_norm_bf16.shape))
    tt_weight_b1h = ttnn.reshape(tt_weight, (1, 1, per_chip_hidden))
    if log_shapes:
        logger.info(
            "[decomp][reshape] weight2d={} -> weight3d=(1, 1, {})",
            tuple(tt_weight.shape),
            per_chip_hidden,
        )
        logger.info("[decomp] tt_weight_b1h shape={}", tuple(tt_weight_b1h.shape))
    tt_out = ttnn.multiply(tt_weight_b1h, tt_norm_bf16, dtype=ttnn.bfloat16)
    if log_shapes:
        logger.info("[decomp] tt_out shape={}", tuple(tt_out.shape))
    return tt_out


def _prepare_fused_state(mesh_device, inputs):
    mesh_shape = (
        ttnn.MeshShape(inputs["num_devices"], 1)
        if inputs["cluster_axis"] == 0
        else ttnn.MeshShape(1, inputs["num_devices"])
    )
    input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    num_cores = input_shard_grid.num_cores()

    total_cores = num_cores * inputs["num_devices"]
    padded_dim_per_core = int(torch.ceil(torch.tensor(inputs["hidden_dim"] / total_cores / 32)).item() * 32)
    padded_dim = padded_dim_per_core * total_cores
    assert padded_dim == inputs["hidden_dim"]
    size_per_device = padded_dim // inputs["num_devices"]

    input_memcfg = ttnn.create_sharded_memory_config(
        shape=(inputs["batch_size"], padded_dim_per_core),
        core_grid=input_shard_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    layer_norm_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(8, 1),
        subblock_w=1,
        block_h=1,
        block_w=(size_per_device // num_cores) // 32,
        inplace=False,
    )
    input_shard_dims = (3, None) if inputs["cluster_axis"] == 0 else (None, 3)
    weight_shard_dims = (2, None) if inputs["cluster_axis"] == 0 else (None, 2)
    tt_input = ttnn.as_tensor(
        inputs["torch_input_4d"],
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device=mesh_device, dims=input_shard_dims, mesh_shape=list(mesh_shape)),
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_memcfg,
    )
    # Use the original sharded 4D input directly for fused RMS.
    tt_weight = ttnn.as_tensor(
        inputs["torch_weight"].reshape(1, 1, padded_dim // 32, 32),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device=mesh_device, dims=weight_shard_dims, mesh_shape=list(mesh_shape)
        ),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_stats = ttnn.from_torch(
        torch.ones((1, 1, inputs["batch_size"], inputs["num_devices"] * 32), dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.create_sharded_memory_config(
            shape=(inputs["batch_size"], inputs["num_devices"] * 32),
            core_grid=input_shard_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device=mesh_device, dims=input_shard_dims, mesh_shape=list(mesh_shape)),
    )
    ccl_sem = ttnn.create_global_semaphore(mesh_device, input_shard_grid, 0)
    return {
        "size_per_device": size_per_device,
        "input_memcfg": input_memcfg,
        "layer_norm_config": layer_norm_config,
        "tt_input": tt_input,
        "tt_weight": tt_weight,
        "tt_stats": tt_stats,
        "ccl_sem": ccl_sem,
    }


def _run_fused(mesh_device, inputs, log_shapes=False, return_local_out=True, state=None):
    state = state if state is not None else _prepare_fused_state(mesh_device, inputs)
    size_per_device = state["size_per_device"]
    input_memcfg = state["input_memcfg"]
    layer_norm_config = state["layer_norm_config"]
    tt_input = state["tt_input"]
    tt_weight = state["tt_weight"]
    tt_stats = state["tt_stats"]
    ccl_sem = state["ccl_sem"]

    if log_shapes:
        logger.info(
            "[fused][reshape] input4d={} (used directly)",
            tuple(tt_input.shape),
        )
    tt_out = ttnn.fused_rms_minimal(
        tt_input,
        layer_norm_config,
        inputs["cluster_axis"],
        mesh_device,
        ccl_sem,
        topology=ttnn.Topology.Linear,
        memory_config=input_memcfg,
        epsilon=inputs["epsilon"],
        dtype=ttnn.bfloat16,
        weight=tt_weight,
        residual_input_tensor=None,
        stats=tt_stats,
        use_noc1_only=False,
    )
    if not return_local_out:
        return tt_out

    ttnn.synchronize_device(mesh_device)
    local_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).reshape(inputs["token_count"], size_per_device)
    if log_shapes:
        logger.info(
            "[fused][reshape] tt_out4d={} -> local_out2d=({}, {})",
            tuple(tt_out.shape),
            inputs["token_count"],
            size_per_device,
        )
        logger.info("[fused] output local shape={} (target 2D end shape)", tuple(local_out.shape))
    return local_out


def _profile_trace(mesh_device, label, num_iters, run_once, profiler):
    if num_iters <= 0:
        profiler.start(label)
        profiler.end(label)
        return

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(num_iters):
        run_once()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    profiler.start(label)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    profiler.end(label)
    ttnn.release_trace(mesh_device, trace_id)


def _profile_no_trace(mesh_device, label, num_iters, run_once, profiler):
    if num_iters <= 0:
        profiler.start(label)
        profiler.end(label)
        return

    profiler.start(label)
    for _ in range(num_iters):
        run_once()
    ttnn.synchronize_device(mesh_device)
    profiler.end(label)


def run_fused_minimal_vs_decomposed_rmsnorm_impl(
    mesh_device,
    cluster_axis,
    num_devices,
    elements_per_batch,
    seq_len=1,
    batch_size=32,
    num_warmup=100,
    num_iters=200,
    profiler=BenchmarkProfiler(),
):
    test_mesh = _get_submesh(mesh_device, num_devices, cluster_axis)
    inputs = _build_inputs(
        test_mesh, seq_len=seq_len, hidden_dim=elements_per_batch, batch_size=batch_size, cluster_axis=cluster_axis
    )
    decomp_state = _prepare_decomposed_state(test_mesh, inputs)
    fused_state = _prepare_fused_state(test_mesh, inputs)

    tt_decomposed = _run_decomposed(test_mesh, inputs, log_shapes=True, state=decomp_state)
    ttnn.synchronize_device(test_mesh)
    # Compare local per-chip outputs from both paths.
    per_chip_hidden = inputs["hidden_dim"] // inputs["num_devices"]
    decomposed_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_decomposed)[0]).reshape(
        inputs["token_count"], per_chip_hidden
    )
    fused_torch = _run_fused(test_mesh, inputs, log_shapes=True, state=fused_state)
    expected_local = inputs["expected"].reshape(inputs["token_count"], inputs["hidden_dim"])[:, :per_chip_hidden]

    passing, output = comp_pcc(decomposed_torch, fused_torch, 0.999)
    assert passing, f"decomposed vs fused mismatch: {output}"
    passing, output = comp_pcc(fused_torch, expected_local, 0.999)
    assert passing, f"fused vs torch mismatch: {output}"

    logger.info("Profiling with TT trace replay for fused/decomposed loops")
    _profile_trace(
        test_mesh,
        "fused-warmup",
        num_warmup,
        lambda: _run_fused(test_mesh, inputs, return_local_out=False, state=fused_state),
        profiler,
    )
    _profile_trace(
        test_mesh,
        "fused",
        num_iters,
        lambda: _run_fused(test_mesh, inputs, return_local_out=False, state=fused_state),
        profiler,
    )
    fused_s = profiler.get_duration("fused")

    _profile_trace(
        test_mesh,
        "decomposed-warmup",
        num_warmup,
        lambda: _run_decomposed(test_mesh, inputs, state=decomp_state),
        profiler,
    )
    _profile_trace(
        test_mesh,
        "decomposed",
        num_iters,
        lambda: _run_decomposed(test_mesh, inputs, state=decomp_state),
        profiler,
    )
    decomposed_s = profiler.get_duration("decomposed")

    logger.info(
        "RMSNorm profile (warmup={}, iters={}): decomposed={:.6f}s ({:.6f}s warmup), fused={:.6f}s ({:.6f}s warmup), speedup={:.3f}x",
        num_warmup,
        num_iters,
        decomposed_s,
        profiler.get_duration("decomposed-warmup"),
        fused_s,
        profiler.get_duration("fused-warmup"),
        decomposed_s / fused_s if fused_s > 0 else float("inf"),
    )

    logger.info("Profiling without TT trace for fused/decomposed loops")
    _profile_no_trace(
        test_mesh,
        "fused-no-trace-warmup",
        num_warmup,
        lambda: _run_fused(test_mesh, inputs, return_local_out=False, state=fused_state),
        profiler,
    )
    _profile_no_trace(
        test_mesh,
        "fused-no-trace",
        num_iters,
        lambda: _run_fused(test_mesh, inputs, return_local_out=False, state=fused_state),
        profiler,
    )
    fused_no_trace_s = profiler.get_duration("fused-no-trace")

    _profile_no_trace(
        test_mesh,
        "decomposed-no-trace-warmup",
        num_warmup,
        lambda: _run_decomposed(test_mesh, inputs, state=decomp_state),
        profiler,
    )
    _profile_no_trace(
        test_mesh,
        "decomposed-no-trace",
        num_iters,
        lambda: _run_decomposed(test_mesh, inputs, state=decomp_state),
        profiler,
    )
    decomposed_no_trace_s = profiler.get_duration("decomposed-no-trace")

    logger.info(
        "RMSNorm no-trace profile (warmup={}, iters={}): decomposed={:.6f}s ({:.6f}s warmup), fused={:.6f}s ({:.6f}s warmup), speedup={:.3f}x",
        num_warmup,
        num_iters,
        decomposed_no_trace_s,
        profiler.get_duration("decomposed-no-trace-warmup"),
        fused_no_trace_s,
        profiler.get_duration("fused-no-trace-warmup"),
        decomposed_no_trace_s / fused_no_trace_s if fused_no_trace_s > 0 else float("inf"),
    )

    logger.info(
        "Trace vs no-trace comparison (iters={}): fused no-trace/trace={:.3f}x, decomposed no-trace/trace={:.3f}x",
        num_iters,
        fused_no_trace_s / fused_s if fused_s > 0 else float("inf"),
        decomposed_no_trace_s / decomposed_s if decomposed_s > 0 else float("inf"),
    )
    return decomposed_s, fused_s


@skip_for_blackhole("This is a wormhole test")
@pytest.mark.parametrize("cluster_axis, num_devices, elements_per_batch, seq_len, batch_size", [(0, 2, 8192, 1, 32)])
@pytest.mark.parametrize("num_warmup, num_iters", [(100, 200)])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4_grid")], indirect=True)
def test_fused_minimal_and_decomposed_rmsnorm_match(
    mesh_device,
    cluster_axis,
    num_devices,
    elements_per_batch,
    seq_len,
    batch_size,
    num_warmup,
    num_iters,
    function_level_defaults,
):
    _ = function_level_defaults
    run_fused_minimal_vs_decomposed_rmsnorm_impl(
        mesh_device,
        cluster_axis,
        num_devices,
        elements_per_batch,
        seq_len=seq_len,
        batch_size=batch_size,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )


@skip_for_blackhole("This is a wormhole test")
@pytest.mark.parametrize(
    "cluster_axis, num_devices, elements_per_batch, seq_len, batch_size",
    [(0, 2, 8192, 1, 32), (1, 4, 8192, 1, 32)],
)
@pytest.mark.parametrize("num_warmup, num_iters", [(100, 200)])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4_grid")], indirect=True)
def test_profile_fused_minimal_vs_decomposed_rmsnorm(
    mesh_device,
    cluster_axis,
    num_devices,
    elements_per_batch,
    seq_len,
    batch_size,
    num_warmup,
    num_iters,
    function_level_defaults,
):
    _ = function_level_defaults
    decomposed_s, fused_s = run_fused_minimal_vs_decomposed_rmsnorm_impl(
        mesh_device,
        cluster_axis,
        num_devices,
        elements_per_batch,
        seq_len=seq_len,
        batch_size=batch_size,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )
    assert decomposed_s > 0
    assert fused_s > 0
