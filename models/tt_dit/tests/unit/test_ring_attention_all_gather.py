# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import time
import torch
from loguru import logger

import ttnn
from models.tt_dit.utils.padding import get_padded_vision_seq_len
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal


def create_global_semaphores(mesh_device, cores, initial_value):
    return [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]


def run_ring_attention_all_gather(
    submesh,
    ag_output_shape,
    ag_num_inputs,
    rp_axis,
    rp_factor,
    up_axis,
    up_factor,
    num_links,
    dtype,
    n_iters,
    trace_enabled,
    all_gather_topology,
    skip_check,
    ccl_core_grid_offset=None,
    use_column_major_ccl=False,
):
    torch.manual_seed(0)

    sequence_dim = 2
    head_dim = 1

    compute_grid_size = submesh.compute_with_storage_grid_size()
    if ccl_core_grid_offset is None:
        ccl_core_grid_offset = ttnn.CoreCoord(0, 0)
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    sub_device_manager = submesh.create_sub_device_manager([worker_sub_device], 0)
    submesh.load_sub_device_manager(sub_device_manager)
    submesh.set_sub_device_stall_group(sub_device_stall_group)

    ccl_semaphore_handles = [create_global_semaphores(submesh, ccl_sub_device_crs, 0) for _ in range(n_iters)]

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    # Persistent output buffers: sharded on heads (up_axis) only, NOT on sequence (gathered)
    output_shard_dims = [None, None]
    output_shard_dims[up_axis] = 1
    persistent_output_buffers = [
        [
            ttnn.from_torch(
                torch.zeros(ag_output_shape),
                device=submesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=mem_config,
                mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=output_shard_dims),
            )
            for _ in range(ag_num_inputs)
        ]
        for _ in range(n_iters)
    ]

    # Input tensors: sharded on sequence (rp_axis) and heads (up_axis)
    input_shard_dims = [None, None]
    input_shard_dims[rp_axis] = sequence_dim
    input_shard_dims[up_axis] = head_dim

    gt_tensors = []
    input_tensors = []
    for i in range(n_iters):
        iter_gts = []
        iter_inputs = []
        for j in range(ag_num_inputs):
            gt = torch.rand(ag_output_shape).bfloat16()
            iter_gts.append(gt)
            tt_input = ttnn.from_torch(
                gt,
                device=submesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=input_shard_dims),
                memory_config=mem_config,
            )
            iter_inputs.append(tt_input)
        gt_tensors.append(iter_gts)
        input_tensors.append(iter_inputs)

    tt_out_list = []

    def run_op(i):
        return ttnn.experimental.ring_attention_all_gather_async(
            input_tensors[i],
            persistent_output_buffer=persistent_output_buffers[i],
            dim=sequence_dim,
            multi_device_global_semaphore=ccl_semaphore_handles[i],
            cluster_axis=rp_axis,
            mesh_device=submesh,
            num_links=num_links,
            memory_config=mem_config,
            topology=all_gather_topology,
            subdevice_id=worker_sub_device_id,
            ccl_core_grid_offset=ccl_core_grid_offset,
            use_column_major_ccl=use_column_major_ccl,
        )

    def run_iters():
        for i in range(n_iters):
            tt_out = run_op(i)
            tt_out_list.append(tt_out)

    if trace_enabled:
        logger.info("Compile run")
        run_op(0)
        ttnn.synchronize_device(submesh, sub_device_ids=sub_device_stall_group)

        logger.info("Capture trace")
        trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
        run_iters()
        ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
        ttnn.synchronize_device(submesh, sub_device_ids=sub_device_stall_group)

        logger.info("Execute trace")
        ttnn.execute_trace(submesh, trace_id, blocking=False)
        ttnn.release_trace(submesh, trace_id)
        ttnn.synchronize_device(submesh, sub_device_ids=sub_device_stall_group)
    else:
        logger.info("Compile run")
        run_op(0)
        ttnn.synchronize_device(submesh, sub_device_ids=sub_device_stall_group)

        logger.info("Run without trace")
        start_time = time.time()
        run_iters()
        ttnn.synchronize_device(submesh, sub_device_ids=sub_device_stall_group)
        end_time = time.time()
        logger.info(f"E2E execution time: {end_time - start_time:.6f} seconds")
        logger.info(f"Time per iter: {(end_time - start_time) / n_iters * 1e3:.3f} ms")

    if not skip_check:
        output_concat_dims = [None, None]
        output_concat_dims[rp_axis] = sequence_dim
        output_concat_dims[up_axis] = head_dim

        seq_len = ag_output_shape[sequence_dim]
        seq_len_per_device = seq_len // submesh.shape[rp_axis]

        for i in range(n_iters):
            tt_outs = tt_out_list[i]
            gts = gt_tensors[i if not trace_enabled else 0]
            for j in range(ag_num_inputs):
                tt_out = ttnn.to_torch(
                    tt_outs[j],
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        submesh, mesh_shape=tuple(submesh.shape), dims=output_concat_dims
                    ),
                )
                tt_chunks = torch.chunk(tt_out, submesh.shape[rp_axis], dim=sequence_dim)
                for ring_idx, tt_chunk in enumerate(tt_chunks):
                    # AG does not write local slice to output; zero it out before comparison
                    tt_check = tt_chunk.clone()
                    torch.narrow(tt_check, sequence_dim, ring_idx * seq_len_per_device, seq_len_per_device).zero_()
                    gt_check = gts[j].clone()
                    torch.narrow(gt_check, sequence_dim, ring_idx * seq_len_per_device, seq_len_per_device).zero_()
                    eq, output = comp_equal(tt_check, gt_check)
                    assert eq, f"FAILED: iter {i}, input {j}, ring {ring_idx}: {output}"

    submesh.reset_sub_device_stall_group()
    submesh.clear_loaded_sub_device_manager()


def run_test_ring_attention_all_gather(
    mesh_device,
    model_input_shape,
    parallel_config,
    n_iters,
    trace_enabled,
    num_links,
    all_gather_topology,
    skip_check,
    dtype,
    ccl_core_grid_offset=None,
    use_column_major_ccl=False,
):
    b, nh, base_seq_len, _joint_seq_len, d = model_input_shape
    rp_axis, rp_factor, up_axis, up_factor = parallel_config

    submesh_shape = [0, 0]
    submesh_shape[rp_axis] = rp_factor
    submesh_shape[up_axis] = up_factor
    submesh = mesh_device.create_submesh(ttnn.MeshShape(submesh_shape[0], submesh_shape[1]))

    padded_seq_len = get_padded_vision_seq_len(base_seq_len, rp_factor)
    ag_output_shape = (b, nh, padded_seq_len, d)
    ag_num_inputs = 2  # K and V

    logger.info(f"AG output shape: {ag_output_shape}, inputs: {ag_num_inputs}")
    logger.info(f"RP axis: {rp_axis} factor: {rp_factor}, UP axis: {up_axis} factor: {up_factor}")
    logger.info(f"Per-device input: ({b}, {nh // up_factor}, {padded_seq_len // rp_factor}, {d})")
    logger.info(f"Per-device output: ({b}, {nh // up_factor}, {padded_seq_len}, {d})")

    run_ring_attention_all_gather(
        submesh,
        ag_output_shape,
        ag_num_inputs,
        rp_axis,
        rp_factor,
        up_axis,
        up_factor,
        num_links,
        dtype,
        n_iters,
        trace_enabled,
        all_gather_topology,
        skip_check,
        ccl_core_grid_offset=ccl_core_grid_offset,
        use_column_major_ccl=use_column_major_ccl,
    )


benchmark_model_input_shapes = {
    "wan_14b_720p": (1, 40, 75600, 0, 128),
    "wan_quad_14b_720p": (1, 40, 18944, 0, 128),
}

parallel_config_map = {
    "bh_glx": {
        "wan_14b_720p": (0, 8, 1, 4),
        "wan_quad_14b_720p": (0, 8, 1, 4),
    },
}

mesh_device_map = {
    "bh_glx": [(8, 4), 2],
}


@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "input_shape",
    benchmark_model_input_shapes.values(),
    ids=benchmark_model_input_shapes.keys(),
)
@pytest.mark.parametrize(
    "parallel_config",
    list(set(c for configs in parallel_config_map.values() for c in configs.values())),
    ids=[f"{rp}rpx{up}up" for _, rp, _, up in set(c for configs in parallel_config_map.values() for c in configs.values())],
)
@pytest.mark.parametrize(
    "n_iters, trace_enabled, skip_check",
    [
        (1, False, False),
        (5, False, True),
        (5, True, True),
    ],
    ids=["check", "no_trace_perf", "trace_perf"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {"worker_l1_size": 1344544, "trace_region_size": 10000000, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
            ttnn.Topology.Linear,
        ),
        (
            {"worker_l1_size": 1344544, "trace_region_size": 10000000, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=["linear", "ring"],
)
@pytest.mark.parametrize(
    "mesh_device, num_links",
    mesh_device_map.values(),
    ids=mesh_device_map.keys(),
    indirect=["mesh_device"],
)
def test_ring_attention_all_gather_bh_glx(
    mesh_device,
    input_shape,
    parallel_config,
    n_iters,
    trace_enabled,
    num_links,
    all_gather_topology,
    skip_check,
    reset_seeds,
):
    run_test_ring_attention_all_gather(
        mesh_device,
        input_shape,
        parallel_config,
        n_iters,
        trace_enabled,
        num_links,
        all_gather_topology,
        skip_check,
        dtype=ttnn.bfloat16,
    )


@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "mesh_device_id",
    mesh_device_map.keys(),
    ids=mesh_device_map.keys(),
)
def test_ring_attention_all_gather_perf_table(mesh_device_id):
    from collections import defaultdict
    from tracy.process_model_log import post_process_ops_log, run_device_profiler

    LINK_BW_BYTES_PER_NS = 25  # 200 Gbps = 25 GB/s = 25 bytes/ns
    AG_NUM_INPUTS = 2  # K and V
    ELEMENT_SIZE = 2  # bfloat16

    _, num_links = mesh_device_map[mesh_device_id]

    topologies = {"linear": "linear", "ring": "ring"}

    results = []
    for topology_name, topology_id in topologies.items():
        for model_input_id, model_input_shape in benchmark_model_input_shapes.items():
            parallel_config = parallel_config_map[mesh_device_id][model_input_id]
            rp_axis, rp_factor, up_axis, up_factor = parallel_config
            parallel_name = f"{rp_factor}rpx{up_factor}up"
            k_expr = f"{model_input_id} and {parallel_name} and {mesh_device_id} and no_trace_perf and {topology_id}"
            command = f"-m 'pytest models/tt_dit/tests/unit/test_ring_attention_all_gather.py::test_ring_attention_all_gather_bh_glx -k \"{k_expr}\"'"

            run_device_profiler(
                command,
                "ring_attention_all_gather",
                check_test_return_code=False,
                device_analysis_types=["device_kernel_duration"],
                is_command_binary_exe=True,
            )
            r = post_process_ops_log("ring_attention_all_gather", sum_vals=False, has_signposts=False)

            invocation_groups = defaultdict(list)
            for _, row in r.iterrows():
                dur_str = str(row["DEVICE KERNEL DURATION [ns]"]).strip()
                if not dur_str:
                    continue
                base = int(row["GLOBAL CALL COUNT"]) - int(row["DEVICE ID"])
                invocation_groups[base].append(row)

            iter_maxes = []
            for base in sorted(invocation_groups):
                chunk_rows = invocation_groups[base]
                max_dur = max(int(row["DEVICE KERNEL DURATION [ns]"]) for row in chunk_rows)
                iter_maxes.append(max_dur)

            best_iter_max_ns = min(iter_maxes) if iter_maxes else 0

            b, nh, base_seq_len, _joint_seq_len, d = model_input_shape
            padded_seq_len = get_padded_vision_seq_len(base_seq_len, rp_factor)
            local_shard_bytes = b * (nh // up_factor) * (padded_seq_len // rp_factor) * d * ELEMENT_SIZE

            if topology_name == "ring":
                data_per_link = local_shard_bytes * AG_NUM_INPUTS * (rp_factor - 1) / 2 / num_links
            else:
                data_per_link = local_shard_bytes * AG_NUM_INPUTS * (rp_factor - 1) / num_links

            best_case_ns = data_per_link / LINK_BW_BYTES_PER_NS
            bw_util = (best_case_ns / best_iter_max_ns * 100) if best_iter_max_ns > 0 else 0

            results.append([
                model_input_id, model_input_shape, parallel_name, padded_seq_len,
                topology_name, best_iter_max_ns, data_per_link, best_case_ns, bw_util,
            ])

    results.sort(key=lambda x: x[5])

    header = (
        "| model_input_id | model_input_shape | parallel | padded seq | topology |"
        " kernel (ms) | data/link (MB) | ideal (ms) | BW util (%) |"
    )
    sep = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    print(header)
    print(sep)
    for row in results:
        (model_input_id, model_input_shape, parallel_name, padded_seq_len,
         topology_name, best_iter_max_ns, data_per_link, best_case_ns, bw_util) = row
        print(
            f"| {model_input_id} | {model_input_shape} | {parallel_name} | {padded_seq_len}"
            f" | {topology_name} | {best_iter_max_ns / 1e6:.3f} | {data_per_link / 1e6:.2f}"
            f" | {best_case_ns / 1e6:.3f} | {bw_util:.1f} |"
        )
