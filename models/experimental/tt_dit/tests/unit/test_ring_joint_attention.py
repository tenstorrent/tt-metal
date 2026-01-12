# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger
import pytest
import torch.nn.functional as F
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)

from tests.tt_eager.python_api_testing.unit_testing.misc.test_scaled_dot_product_attention import fa_rand

from models.experimental.tt_dit.utils.padding import get_padded_vision_seq_len

from tracy.process_model_log import run_device_profiler, post_process_ops_log


def torch_sdpa(q, k, v, joint_q, joint_k, joint_v, num_devices):
    scale = k.size(-1) ** -0.5
    seq_len = k.size(2)
    slice_seq_len = seq_len // num_devices
    out = None
    lse = None
    lse_list = []
    Q = torch.cat([q, joint_q], dim=2)
    for ring_id in range(num_devices):
        k_slice = k[:, :, ring_id * slice_seq_len : (ring_id + 1) * slice_seq_len, :]
        v_slice = v[:, :, ring_id * slice_seq_len : (ring_id + 1) * slice_seq_len, :]
        if ring_id == num_devices - 1:
            k_slice = torch.cat([k_slice, joint_k], dim=2)
            v_slice = torch.cat([v_slice, joint_v], dim=2)
        attn_weights = torch.matmul(Q, k_slice.transpose(-2, -1)) * scale
        cur_max, _ = torch.max(attn_weights, dim=-1, keepdim=True)
        attn_weights = torch.exp(attn_weights - cur_max)
        cur_sum = torch.sum(attn_weights, dim=-1, keepdim=True)
        cur_out = torch.matmul(attn_weights, v_slice)
        cur_out = cur_out / cur_sum
        cur_lse = cur_max + torch.log(cur_sum)
        if ring_id == 0:
            out = cur_out
            lse = cur_lse
        else:
            sig = F.sigmoid(cur_lse - lse)
            out = out - sig * (out - cur_out)
            lse = lse - F.logsigmoid(lse - cur_lse)
        lse_list.append(lse)

    return out, lse_list


def create_global_semaphores(mesh_device, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]
    return ccl_semaphore_handles


def create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor):
    submesh_shape = [0, 0]
    submesh_shape[rp_axis] = rp_factor
    submesh_shape[up_axis] = up_factor
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(submesh_shape[0], submesh_shape[1]))
    return submesh_device


def run_ring_joint_sdpa(
    submesh,
    b,
    nh,
    base_seq_len,
    padded_seq_len,
    joint_seq_len,
    d,
    q_chunk_size,
    k_chunk_size,
    dtype,
    n_iters,
    trace_enabled,
    num_links,
    rp_axis,
    up_axis,
    all_gather_topology,
    skip_check,
    pcc_threshold,
    max_mse=None,
):
    full_compute_grid = submesh.compute_with_storage_grid_size()
    sdpa_compute_grid = (full_compute_grid.x, full_compute_grid.y - 1)
    ccl_core_grid_offset = (0, full_compute_grid.y - 1)

    # Basic CCL setup
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_compute_grid.x - 1, full_compute_grid.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    sub_device_manager = submesh.create_sub_device_manager([worker_sub_device], 0)
    submesh.load_sub_device_manager(sub_device_manager)
    submesh.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [create_global_semaphores(submesh, ccl_sub_device_crs, 0) for _ in range(n_iters)]

    kv_shard_dims = [None, None]
    kv_shard_dims[rp_axis] = None  # Output of AllGather is not sharded on RP axis
    kv_shard_dims[up_axis] = 1  # UP shards on heads dim1

    # Create persistent output buffers
    ag_output_shape = (b, nh, padded_seq_len, d)

    persistent_output_buffers = [
        [
            ttnn.from_torch(
                torch.zeros(ag_output_shape),
                device=submesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=kv_shard_dims),
            )
            for _ in range(2)  # Num inputs K, V
        ]
        for _ in range(n_iters)
    ]

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=sdpa_compute_grid,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nh, base_seq_len, d)
    K = fa_rand(b, nh, base_seq_len, d)
    V = fa_rand(b, nh, base_seq_len, d)

    padded_Q = torch.cat([Q, torch.zeros(b, nh, padded_seq_len - base_seq_len, d)], dim=2)
    padded_K = torch.cat([K, torch.zeros(b, nh, padded_seq_len - base_seq_len, d)], dim=2)
    padded_V = torch.cat([V, torch.zeros(b, nh, padded_seq_len - base_seq_len, d)], dim=2)

    joint_Q = fa_rand(b, nh, joint_seq_len, d)
    joint_K = fa_rand(b, nh, joint_seq_len, d)
    joint_V = fa_rand(b, nh, joint_seq_len, d)

    # Print shapes of all inputs along with input names
    logger.debug(f"Q: {Q.shape}")
    logger.debug(f"K: {K.shape}")
    logger.debug(f"V: {V.shape}")
    logger.debug(f"padded_Q: {padded_Q.shape}")
    logger.debug(f"padded_K: {padded_K.shape}")
    logger.debug(f"padded_V: {padded_V.shape}")

    sdpa_input_shard_dims = [None, None]
    sdpa_input_shard_dims[rp_axis] = 2  # sequence dim
    sdpa_input_shard_dims[up_axis] = 1  # head dim

    # Joint input only sharded on head dim
    sdpa_joint_shard_dims = [None, None]
    sdpa_joint_shard_dims[up_axis] = 1  # head dim

    tt_Q = ttnn.from_torch(
        padded_Q,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_input_shard_dims),
    )
    tt_K = ttnn.from_torch(
        padded_K,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_input_shard_dims),
    )
    tt_V = ttnn.from_torch(
        padded_V,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_input_shard_dims),
    )
    tt_joint_Q = ttnn.from_torch(
        joint_Q,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_joint_shard_dims),
    )
    tt_joint_K = ttnn.from_torch(
        joint_K,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_joint_shard_dims),
    )
    tt_joint_V = ttnn.from_torch(
        joint_V,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_joint_shard_dims),
    )

    logger.debug(f"tt_Q: {tt_Q.shape}")
    logger.debug(f"tt_joint_Q: {tt_joint_Q.shape}")

    tt_out_list = []
    tt_joint_out_list = []

    def run_iters(tt_out_list, tt_joint_out_list):
        for i in range(n_iters):
            tt_out, tt_joint_out, tt_lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                tt_Q,
                tt_K,
                tt_V,
                tt_joint_Q,
                tt_joint_K,
                tt_joint_V,
                persistent_output_buffer_k=persistent_output_buffers[i][0],
                persistent_output_buffer_v=persistent_output_buffers[i][1],
                joint_strategy="rear",
                logical_n=base_seq_len,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                num_links=num_links,
                cluster_axis=rp_axis,
                mesh_device=submesh,
                topology=all_gather_topology,
                subdevice_id=worker_sub_device_id,
                ccl_core_grid_offset=ccl_core_grid_offset,
            )
            tt_out_list.append(tt_out)
            tt_joint_out_list.append(tt_joint_out)

    if trace_enabled:
        logger.info("Compile run")
        run_iters([], [])
        logger.info("Capture trace")
        trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
        run_iters(tt_out_list, tt_joint_out_list)
        ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
        ttnn.synchronize_device(submesh)
        logger.info("Execute trace")
        ttnn.execute_trace(submesh, trace_id, blocking=False)
        ttnn.release_trace(submesh, trace_id)
        ttnn.synchronize_device(submesh)

    else:
        logger.info("Run without trace")
        run_iters(tt_out_list, tt_joint_out_list)

    if not skip_check:
        pt_Q = torch.cat([Q, joint_Q], dim=2)
        pt_K = torch.cat([K, joint_K], dim=2)
        pt_V = torch.cat([V, joint_V], dim=2)
        gt = torch.nn.functional.scaled_dot_product_attention(pt_Q, pt_K, pt_V, is_causal=False)
        gt_out = gt[:, :, :base_seq_len, :]
        gt_joint_out = gt[:, :, base_seq_len:, :]

        for i in range(n_iters):
            tt_out = ttnn.to_torch(
                tt_out_list[i],
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_input_shard_dims
                ),
            )
            joint_shard_dims = [None, None]
            joint_shard_dims[up_axis] = 1
            joint_shard_dims[rp_axis] = 0  # Concat replicas on sequence length into batch
            tt_joint_out = ttnn.to_torch(
                tt_joint_out_list[i],
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    submesh, mesh_shape=tuple(submesh.shape), dims=joint_shard_dims
                ),
            )
            # Slice out any tile-padding
            tt_out = tt_out[:, :, :base_seq_len, :]
            tt_joint_out = tt_joint_out[:, :, :joint_seq_len, :]
            logger.debug(f"tt_out: {tt_out.shape}")
            logger.debug(f"tt_joint_out: {tt_joint_out.shape}")

            passing = True
            out_pass, out_pcc = comp_pcc(tt_out, gt_out, pcc_threshold)
            logger.debug("spatial")
            logger.debug(f"{out_pcc}")
            mse = ((gt_out - tt_out) ** 2).mean()
            logger.debug(f"mse: {mse}")
            if max_mse is not None and mse > max_mse:
                passing = False
            passing = passing and out_pass

            if joint_seq_len > 0:
                logger.debug("prompt")
                for joint_replica_id in range(tt_joint_out.shape[0]):
                    joint_replica_out = tt_joint_out[joint_replica_id, :, :, :]
                    out_pass, out_pcc = comp_pcc(joint_replica_out, gt_joint_out, pcc_threshold)
                    logger.debug(f"{out_pcc}")
                    mse = ((gt_joint_out - joint_replica_out) ** 2).mean()
                    logger.debug(f"mse: {mse}")
                    if max_mse is not None and mse > max_mse:
                        passing = False
                    passing = passing and out_pass

            assert passing


def run_test_ring_joint_sdpa(
    mesh_device,
    model_input_shape,
    parallel_config,
    q_chunk_size,
    k_chunk_size,
    n_iters,
    trace_enabled,
    num_links,
    all_gather_topology,
    skip_check,
    dtype,
    pcc_threshold=0.994,
    max_mse=None,
):
    b, nh, base_seq_len, joint_seq_len, d = model_input_shape
    rp_axis, rp_factor, up_axis, up_factor = parallel_config
    import math

    if nh % up_factor != 0:
        orig_nh = nh
        nh = math.ceil(nh / up_factor) * up_factor
        logger.info(f"Rounding up nh from {orig_nh} to {nh} so that it divides evenly by up_factor={up_factor}.")
    mesh_device_shape = list(mesh_device.shape)
    assert mesh_device_shape[rp_axis] >= rp_factor and mesh_device_shape[up_axis] >= up_factor

    submesh = create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor)

    padded_seq_len = get_padded_vision_seq_len(base_seq_len, mesh_device_shape[rp_axis])

    logger.debug(f"RP axis: {rp_axis} factor: {rp_factor}, UP axis: {up_axis} factor: {up_factor}")
    logger.debug(f"submesh: {submesh.shape}")

    run_ring_joint_sdpa(
        submesh,
        b,
        nh,
        base_seq_len,
        padded_seq_len,
        joint_seq_len,
        d,
        q_chunk_size,
        k_chunk_size,
        dtype,
        n_iters,
        trace_enabled,
        num_links,
        rp_axis,
        up_axis,
        all_gather_topology,
        skip_check,
        pcc_threshold,
        max_mse=max_mse,
    )


benchmark_model_input_shapes = {
    "wan_14b_720p": (1, 40, 75600, 0, 128),
    "wan_14b_480p": (1, 40, 32760, 0, 128),
    "mochi": (1, 24, 44520, 118, 128),
    "flux": (1, 24, 4096, 512, 128),
    "sd35": (1, 38, 4096, 333, 64),
}

parallel_config_map = {
    "wh_glx": {
        "wan_14b_720p": (0, 8, 1, 4),
        "wan_14b_480p": (0, 8, 1, 4),
        "mochi": (0, 8, 1, 4),
        "flux": (0, 8, 1, 4),
        "sd35": (0, 4, 1, 4),
    },
    "wh_t3k": {
        "wan_14b_720p": (0, 2, 1, 4),
        "wan_14b_480p": (0, 2, 1, 4),
        "mochi": (0, 2, 1, 4),
        "flux": (0, 2, 1, 4),
        "sd35": (0, 2, 1, 2),
    },
    "bh_glx": {
        "wan_14b_720p": (0, 8, 1, 4),
        "wan_14b_480p": (0, 8, 1, 4),
        "mochi": (0, 8, 1, 4),
        "flux": (0, 8, 1, 4),
        "sd35": (0, 4, 1, 4),
    },
    "bh_qb_ge": {
        "wan_14b_720p": (0, 2, 1, 2),
        "wan_14b_480p": (0, 2, 1, 2),
        "mochi": (0, 2, 1, 2),
        "flux": (0, 2, 1, 2),
        "sd35": (0, 2, 1, 2),
    },
}

mesh_device_map = {
    "wh_glx": [(8, 4), 4],
    "wh_t3k": [(2, 4), 1],
    "bh_glx": [(8, 4), 2],
    "bh_qb_ge": [(2, 2), 2],
}

all_parallel_configs = list(set(config for configs in parallel_config_map.values() for config in configs.values()))


def get_parallel_config_id(rp_factor, up_factor):
    return f"{rp_factor}rpx{up_factor}up"


all_parallel_config_ids = [
    get_parallel_config_id(rp_factor, up_factor) for rp_axis, rp_factor, up_axis, up_factor in all_parallel_configs
]


@pytest.mark.parametrize(
    "model_input_shape",
    benchmark_model_input_shapes.values(),
    ids=benchmark_model_input_shapes.keys(),
)
@pytest.mark.parametrize("parallel_config", all_parallel_configs, ids=all_parallel_config_ids)
@pytest.mark.parametrize("q_chunk_size", [64, 128, 256], ids=["q64", "q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [64, 128, 256, 512], ids=["k64", "k128", "k256", "k512"])
@pytest.mark.parametrize(
    "n_iters, trace_enabled, skip_check",
    [(1, False, False), (1, False, True)],
    ids=["no_trace_check", "no_trace_no_check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {"worker_l1_size": 1344544, "trace_region_size": 1000000, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=[
        "line",
    ],
)
@pytest.mark.parametrize(
    "mesh_device, num_links",
    mesh_device_map.values(),
    ids=mesh_device_map.keys(),
    indirect=["mesh_device"],
)
def test_ring_joint_sdpa(
    mesh_device,
    model_input_shape,
    parallel_config,
    q_chunk_size,
    k_chunk_size,
    n_iters,
    trace_enabled,
    num_links,
    all_gather_topology,
    skip_check,
    reset_seeds,
):
    dtype = ttnn.bfloat16

    run_test_ring_joint_sdpa(
        mesh_device,
        model_input_shape,
        parallel_config,
        q_chunk_size,
        k_chunk_size,
        n_iters,
        trace_enabled,
        num_links,
        all_gather_topology,
        skip_check,
        dtype,
    )


@pytest.mark.parametrize(
    "mesh_device_id",
    mesh_device_map.keys(),
    ids=mesh_device_map.keys(),
)
def test_ring_joint_sdpa_perf_table(mesh_device_id):
    results = []
    for model_input_id, model_input_shape in benchmark_model_input_shapes.items():
        parallel_config = parallel_config_map[mesh_device_id][model_input_id]
        rp_axis, rp_factor, up_axis, up_factor = parallel_config
        parallel_name = get_parallel_config_id(rp_factor, up_factor)
        k_expr = f"{model_input_id} and {parallel_name} and {mesh_device_id} and no_trace_no_check"
        command = f"-m 'pytest models/experimental/tt_dit/tests/models/wan2_2/test_ring_joint_attention.py::test_ring_joint_sdpa -k \"{k_expr}\"'"

        run_device_profiler(
            command,
            "ring_joint_sdpa",
            check_test_return_code=False,
            device_analysis_types=["device_kernel_duration"],
            is_command_binary_exe=True,
        )
        r = post_process_ops_log("ring_joint_sdpa", sum_vals=False, has_signposts=False)
        attrs = r["ATTRIBUTES"].tolist()
        durations = r["DEVICE KERNEL DURATION [ns]"].tolist()
        result = sorted(zip(durations, attrs), key=lambda x: x[0])[0]
        best_duration, best_attrs = result
        results.append([model_input_shape, model_input_id, parallel_name, best_duration, best_attrs])

    header = "| model_input_id | model_input_shape | parallel_name | padded seq | qchunk, kchunk | measured perf (ms) |"
    sep = "|---:|---:|---:|---:|---:|---:|"
    print(header)
    print(sep)
    for result in results:
        model_input_shape, model_input_id, parallel_name, duration, attrs = result
        q_chunk = attrs.split("q_chunk_size=")[1].split(";")[0]
        k_chunk = attrs.split("k_chunk_size=")[1].split(";")[0]
        new_seqlen = get_padded_vision_seq_len(int(model_input_shape[2]), rp_factor)
        print(
            f"| {model_input_id} | {model_input_shape} | {parallel_name} | {new_seqlen} | {q_chunk}, {k_chunk} | {duration / 1e6:.3f} |"
        )


model_input_shapes = [
    # original smoke cases
    (1, 24, 4096, 512, 128),  # padded-divisible spatial, joint > 0
    (1, 38, 4096, 333, 64),  # many heads, smaller head dim, uneven joint
    (1, 24, 4224, 128, 128),  # N not divisible by chunk, moderate joint
    (1, 2, 3072, 0, 128),  # small head count, no joint
    (1, 2, 4000, 2, 128),  # tiny joint, near-multiple-of-chunk
    # additional stress cases
    (1, 24, 8192, 0, 128),  # long sequence, no joint
    (1, 24, 8200, 64, 128),  # long, non-multiple N, small joint
    (1, 16, 1024, 256, 128),  # mid length, significant joint
    (1, 16, 1056, 128, 64),  # mid length, smaller head dim
    (1, 8, 2048, 0, 256),  # wider head dim
    (1, 8, 2176, 128, 128),  # mid length, non-multiple, modest joint
    (1, 4, 512, 64, 128),  # short length with joint
    (1, 4, 4096, 128, 128),
    (1, 2, 256, 16, 64),  # minimal heads/dim
]

model_input_ids = [
    "wan_14b_720p",
    "wan_14b_480p",
    "wan_5b_720p",
    "mochi",
    "flux",
    "long_no_joint",
    "long_unaligned_joint",
    "mid_joint",
    "mid_small_d",
    "wide_d",
    "mid_unaligned_joint",
    "short_joint",
    "batch2",
    "tiny_head",
]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "b, nh, base_seq_len, joint_seq_len, d",
    model_input_shapes,
    ids=model_input_ids,
)
@pytest.mark.parametrize("q_chunk_size", [32, 64, 128, 256], ids=["q32", "q64", "q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [32, 64, 128, 256], ids=["k32", "k64", "k128", "k256"])
@pytest.mark.parametrize(
    "n_iters, trace_enabled, skip_check",
    [
        (1, False, False),
    ],
    ids=["no_trace"],
)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {"worker_l1_size": 1344544, "trace_region_size": 1000000, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=[
        "line",
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(2, 4)],
    ids=["2x4"],
    indirect=True,
)
@pytest.mark.parametrize(
    "rp_axis, rp_factor, up_axis, up_factor",
    [
        [1, 4, 0, 2],
    ],
    ids=[
        "4rpx2up",
    ],
)
def test_ring_joint_sdpa_shapes(
    mesh_device,
    b,
    nh,
    base_seq_len,
    joint_seq_len,
    d,
    q_chunk_size,
    k_chunk_size,
    dtype,
    n_iters,
    trace_enabled,
    num_links,
    rp_axis,
    rp_factor,
    up_axis,
    up_factor,
    all_gather_topology,
    skip_check,
    reset_seeds,
):
    mesh_device_shape = list(mesh_device.shape)
    assert mesh_device_shape[rp_axis] >= rp_factor and mesh_device_shape[up_axis] >= up_factor

    submesh = create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor)

    padded_seq_len = get_padded_vision_seq_len(base_seq_len, mesh_device_shape[rp_axis])

    logger.debug(f"RP axis: {rp_axis} factor: {rp_factor}, UP axis: {up_axis} factor: {up_factor}")
    logger.debug(f"submesh: {submesh.shape}")

    run_ring_joint_sdpa(
        submesh,
        b,
        nh,
        base_seq_len,
        padded_seq_len,
        joint_seq_len,
        d,
        q_chunk_size,
        k_chunk_size,
        dtype,
        n_iters,
        trace_enabled,
        num_links,
        rp_axis,
        up_axis,
        all_gather_topology,
        skip_check,
        0.999,
    )


wh_t3k_unit_test_params = pytest.mark.parametrize(
    "input_shape, parallel_config, chunk_sizes, expected_correctness",
    [
        [
            benchmark_model_input_shapes["wan_14b_720p"],
            parallel_config_map["wh_t3k"]["wan_14b_720p"],
            (256, 256),
            (0.9994, 7.5e-5),
        ],
        [
            benchmark_model_input_shapes["wan_14b_480p"],
            parallel_config_map["wh_t3k"]["wan_14b_480p"],
            (256, 256),
            (0.9996, 5e-5),
        ],
        [benchmark_model_input_shapes["mochi"], parallel_config_map["wh_t3k"]["mochi"], (128, 512), (0.9995, 6e-5)],
        [benchmark_model_input_shapes["flux"], parallel_config_map["wh_t3k"]["flux"], (128, 512), (0.9997, 2.2e-5)],
        [benchmark_model_input_shapes["sd35"], parallel_config_map["wh_t3k"]["sd35"], (256, 512), (0.9997, 3.5e-5)],
    ],
    ids=[
        "wan_14b_720p",
        "wan_14b_480p",
        "mochi",
        "flux",
        "sd35",
    ],
)


@wh_t3k_unit_test_params
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {"worker_l1_size": 1344544, "trace_region_size": 1000000, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=[
        "line",
    ],
)
@pytest.mark.parametrize("mesh_device, num_links", [mesh_device_map["wh_t3k"]], ids=["2x4"], indirect=["mesh_device"])
def test_ring_joint_sdpa_dit_wh_t3k(
    mesh_device,
    input_shape,
    parallel_config,
    chunk_sizes,
    expected_correctness,
    num_links,
    all_gather_topology,
    reset_seeds,
):
    dtype = ttnn.bfloat16
    n_iters = 1
    trace_enabled = False
    skip_check = False
    pcc_threshold, max_mse = expected_correctness
    q_chunk_size, k_chunk_size = chunk_sizes

    run_test_ring_joint_sdpa(
        mesh_device,
        input_shape,
        parallel_config,
        q_chunk_size,
        k_chunk_size,
        n_iters,
        trace_enabled,
        num_links,
        all_gather_topology,
        skip_check,
        dtype,
        pcc_threshold=pcc_threshold,
        max_mse=max_mse,
    )


bh_qb_ge_unit_test_params = pytest.mark.parametrize(
    "input_shape, parallel_config, chunk_sizes, expected_correctness",
    [
        [
            benchmark_model_input_shapes["wan_14b_720p"],
            parallel_config_map["bh_qb_ge"]["wan_14b_720p"],
            (128, 512),
            (0.9994, 7e-5),
        ],
        [
            benchmark_model_input_shapes["wan_14b_480p"],
            parallel_config_map["bh_qb_ge"]["wan_14b_480p"],
            (128, 512),
            (0.9996, 5e-5),
        ],
        [benchmark_model_input_shapes["mochi"], parallel_config_map["bh_qb_ge"]["mochi"], (128, 512), (0.9995, 6e-5)],
        [benchmark_model_input_shapes["flux"], parallel_config_map["bh_qb_ge"]["flux"], (128, 512), (0.9997, 2.2e-5)],
        [benchmark_model_input_shapes["sd35"], parallel_config_map["bh_qb_ge"]["sd35"], (256, 512), (0.9997, 3.5e-5)],
    ],
    ids=[
        "wan_14b_720p",
        "wan_14b_480p",
        "mochi",
        "flux",
        "sd35",
    ],
)


@bh_qb_ge_unit_test_params
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {"worker_l1_size": 1344544, "trace_region_size": 1000000, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=[
        "line",
    ],
)
@pytest.mark.parametrize("mesh_device, num_links", [mesh_device_map["bh_qb_ge"]], ids=["2x2"], indirect=["mesh_device"])
def test_ring_joint_sdpa_dit_bh_qb_ge(
    mesh_device,
    input_shape,
    parallel_config,
    chunk_sizes,
    expected_correctness,
    num_links,
    all_gather_topology,
    reset_seeds,
):
    dtype = ttnn.bfloat16
    n_iters = 1
    trace_enabled = False
    skip_check = False
    pcc_threshold, max_mse = expected_correctness
    q_chunk_size, k_chunk_size = chunk_sizes

    run_test_ring_joint_sdpa(
        mesh_device,
        input_shape,
        parallel_config,
        q_chunk_size,
        k_chunk_size,
        n_iters,
        trace_enabled,
        num_links,
        all_gather_topology,
        skip_check,
        dtype,
        pcc_threshold=pcc_threshold,
        max_mse=max_mse,
    )


wh_glx_unit_test_params = pytest.mark.parametrize(
    "input_shape, parallel_config, chunk_sizes, expected_correctness",
    [
        [
            benchmark_model_input_shapes["wan_14b_720p"],
            parallel_config_map["wh_glx"]["wan_14b_720p"],
            (256, 256),
            (0.9993, 8e-5),
        ],
        [
            benchmark_model_input_shapes["wan_14b_480p"],
            parallel_config_map["wh_glx"]["wan_14b_480p"],
            (128, 512),
            (0.9995, 6e-5),
        ],
        [benchmark_model_input_shapes["mochi"], parallel_config_map["wh_glx"]["mochi"], (128, 512), (0.9994, 7e-5)],
        [benchmark_model_input_shapes["flux"], parallel_config_map["wh_glx"]["flux"], (128, 256), (0.9997, 3e-5)],
        [benchmark_model_input_shapes["sd35"], parallel_config_map["wh_glx"]["sd35"], (256, 512), (0.9997, 4e-5)],
    ],
    ids=[
        "wan_14b_720p",
        "wan_14b_480p",
        "mochi",
        "flux",
        "sd35",
    ],
)


@wh_glx_unit_test_params
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {"worker_l1_size": 1344544, "trace_region_size": 1000000, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=[
        "line",
    ],
)
@pytest.mark.parametrize("mesh_device, num_links", [mesh_device_map["wh_glx"]], ids=["8x4"], indirect=["mesh_device"])
def test_ring_joint_sdpa_dit_wh_glx(
    mesh_device,
    input_shape,
    parallel_config,
    chunk_sizes,
    expected_correctness,
    num_links,
    all_gather_topology,
    reset_seeds,
):
    dtype = ttnn.bfloat16
    n_iters = 1
    trace_enabled = False
    skip_check = False
    pcc_threshold, max_mse = expected_correctness
    q_chunk_size, k_chunk_size = chunk_sizes

    run_test_ring_joint_sdpa(
        mesh_device,
        input_shape,
        parallel_config,
        q_chunk_size,
        k_chunk_size,
        n_iters,
        trace_enabled,
        num_links,
        all_gather_topology,
        skip_check,
        dtype,
        pcc_threshold=pcc_threshold,
        max_mse=max_mse,
    )


bh_glx_unit_test_params = pytest.mark.parametrize(
    "input_shape, parallel_config, chunk_sizes, expected_correctness",
    [
        [
            benchmark_model_input_shapes["wan_14b_720p"],
            parallel_config_map["bh_glx"]["wan_14b_720p"],
            (128, 512),
            (0.9993, 8e-5),
        ],
        [
            benchmark_model_input_shapes["wan_14b_480p"],
            parallel_config_map["bh_glx"]["wan_14b_480p"],
            (256, 256),
            (0.9995, 6e-5),
        ],
        [benchmark_model_input_shapes["mochi"], parallel_config_map["bh_glx"]["mochi"], (128, 512), (0.9994, 7e-5)],
        [benchmark_model_input_shapes["flux"], parallel_config_map["bh_glx"]["flux"], (64, 512), (0.9997, 3e-5)],
        [benchmark_model_input_shapes["sd35"], parallel_config_map["bh_glx"]["sd35"], (128, 512), (0.9997, 4e-5)],
    ],
    ids=[
        "wan_14b_720p",
        "wan_14b_480p",
        "mochi",
        "flux",
        "sd35",
    ],
)


@bh_glx_unit_test_params
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {"worker_l1_size": 1344544, "trace_region_size": 1000000, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=[
        "line",
    ],
)
@pytest.mark.parametrize("mesh_device, num_links", [mesh_device_map["bh_glx"]], ids=["8x4"], indirect=["mesh_device"])
def test_ring_joint_sdpa_dit_bh_glx(
    mesh_device,
    input_shape,
    parallel_config,
    chunk_sizes,
    expected_correctness,
    num_links,
    all_gather_topology,
    reset_seeds,
):
    dtype = ttnn.bfloat16
    n_iters = 1
    trace_enabled = False
    skip_check = False
    pcc_threshold, max_mse = expected_correctness
    q_chunk_size, k_chunk_size = chunk_sizes

    run_test_ring_joint_sdpa(
        mesh_device,
        input_shape,
        parallel_config,
        q_chunk_size,
        k_chunk_size,
        n_iters,
        trace_enabled,
        num_links,
        all_gather_topology,
        skip_check,
        dtype,
        pcc_threshold=pcc_threshold,
        max_mse=max_mse,
    )
