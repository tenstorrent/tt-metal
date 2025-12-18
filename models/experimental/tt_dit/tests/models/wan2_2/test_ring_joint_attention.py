# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
import pytest
from tests.nightly.t3000.ccl.test_ring_joint_attention import run_ring_joint_sdpa, create_ring_joint_sdpa_submesh
from models.experimental.tt_dit.utils.padding import get_padded_vision_seq_len

from tracy.process_model_log import run_device_profiler, post_process_ops_log

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
        0.994,
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


@pytest.mark.parametrize(
    "input_shape, parallel_config, chunk_sizes, expected_correctness",
    # input_shape: (b, nh, base_seq_len, joint_seq_len, d)
    # parallel_config: (rp_axis, rp_factor, up_axis, up_factor)
    # chunk_sizes: (q_chunk_size, k_chunk_size)
    # expected_corretness: (min_pcc, max_mse)
    [
        [(1, 40, 32760, 0, 128), (0, 2, 1, 4), (256, 256), (0.9996, 5e-5)],
        [(1, 24, 44520, 118, 128), (0, 2, 1, 4), (128, 512), (0.9995, 6e-5)],
        [(1, 24, 4096, 512, 128), (0, 2, 1, 4), (128, 512), (0.9997, 2.2e-5)],
        [(1, 38, 4096, 333, 64), (0, 2, 1, 2), (256, 512), (0.9997, 3.5e-5)],
    ],
    ids=[
        "wan_14b_480p_2x4",
        "mochi_2x4",
        "flux_2x4",
        "sd35_2x2",
    ],
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
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
def test_ring_joint_sdpa_dit_t3k(
    mesh_device,
    input_shape,
    parallel_config,
    chunk_sizes,
    expected_correctness,
    all_gather_topology,
    reset_seeds,
):
    b, nh, base_seq_len, joint_seq_len, d = input_shape
    rp_axis, rp_factor, up_axis, up_factor = parallel_config
    q_chunk_size, k_chunk_size = chunk_sizes

    # contants
    dtype = ttnn.bfloat16
    n_iters = 1
    trace_enabled = False
    skip_check = False
    num_links = 1
    pcc_threshold, max_mse = expected_correctness

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
