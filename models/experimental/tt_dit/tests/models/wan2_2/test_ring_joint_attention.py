# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
import pytest
from tests.nightly.t3000.ccl.test_ring_joint_attention import run_ring_joint_sdpa, create_ring_joint_sdpa_submesh
from models.experimental.tt_dit.utils.padding import get_padded_vision_seq_len

from tracy.process_model_log import run_device_profiler, post_process_ops_log


model_input_shapes = [
    (1, 40, 75600, 0, 128),
    (1, 40, 32760, 0, 128),
    (1, 40, 27280, 0, 128),
    (1, 24, 44520, 118, 128),
    (1, 24, 4096, 512, 128),
    (1, 38, 4096, 333, 64),
    (1, 24, 4224, 128, 128),
]

model_input_ids = ["wan_14b_720p", "wan_14b_480p", "wan_5b_720p", "mochi", "flux", "sd35", "uneven"]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "b, nh, base_seq_len, joint_seq_len, d",
    model_input_shapes,
    ids=model_input_ids,
)
@pytest.mark.parametrize("q_chunk_size", [32, 64, 128, 256], ids=["q32", "q64", "q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [32, 64, 128, 256], ids=["k32", "k64", "k128", "k256"])
@pytest.mark.parametrize(
    "n_iters, trace_enabled, skip_check", [(1, False, False), (10, True, True)], ids=["no_trace", "yes_trace"]
)
@pytest.mark.parametrize("num_links", [1, 2, 4], ids=["1link", "2link", "4link"])
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
    [(8, 4), (2, 4)],
    ids=["8x4", "2x4"],
    indirect=True,
)
@pytest.mark.parametrize(
    "rp_axis, rp_factor, up_axis, up_factor",
    [
        [0, 8, 1, 4],  # 8x4 RP x UP
        [0, 8, 1, 1],
        [1, 4, 0, 2],
    ],
    ids=[
        "8rpx4up",
        "8rpx1up",
        "4rpx2up",
    ],
)
def test_ring_joint_sdpa(
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
    "mesh_device_shape",
    [(8, 4), (2, 4)],
    ids=["8x4", "2x4"],
)
@pytest.mark.parametrize(
    "rp_axis, rp_factor, up_axis, up_factor",
    [
        [0, 8, 1, 4],  # 8x4 RP x UP
        [0, 8, 1, 1],
        [1, 4, 0, 2],
    ],
    ids=[
        "8rpx4up",
        "8rpx1up",
        "4rpx2up",
    ],
)
@pytest.mark.parametrize("num_links", [1, 2, 4], ids=["1link", "2link", "4link"])
def test_ring_joint_sdpa_perf_table(mesh_device_shape, rp_axis, rp_factor, up_axis, up_factor, num_links):
    results = []
    for model_input_shape, model_input_id in zip(model_input_shapes, model_input_ids):
        b, nh, base_seq_len, joint_seq_len, d = model_input_shape

        parallel_config = f"{rp_factor}rpx{up_factor}up"
        mesh_name = f"{mesh_device_shape[0]}x{mesh_device_shape[1]}"
        k_expr = f"{model_input_id} and bf16 and {parallel_config} and {mesh_name} and {num_links}link and no_trace"
        command = f"-m 'pytest models/experimental/tt_dit/tests/models/wan2_2/test_ring_joint_attention.py::test_ring_joint_sdpa -k \"{k_expr}\"'"

        run_device_profiler(
            command,
            "ring_joint_sdpa",
            device_analysis_types=["device_kernel_duration"],
            is_command_binary_exe=True,
        )
        r = post_process_ops_log("ring_joint_sdpa", sum_vals=False, has_signposts=False)
        attrs = r["ATTRIBUTES"].tolist()
        durations = r["DEVICE KERNEL DURATION [ns]"].tolist()
        result = sorted(zip(durations, attrs), key=lambda x: x[0])[0]
        best_duration, best_attrs = result
        results.append([model_input_shape, best_duration, best_attrs])

    header = "| model_input_shape | padded seq | qchunk, kchunk | measured perf (ms) |"
    sep = "|---:|---:|---:|---:|"
    print(header)
    print(sep)
    for result in results:
        model_input_shape, duration, attrs = result
        q_chunk = attrs.split("q_chunk_size=")[1].split(";")[0]
        k_chunk = attrs.split("k_chunk_size=")[1].split(";")[0]
        new_seqlen = get_padded_vision_seq_len(int(model_input_shape[2]), max(int(q_chunk), int(k_chunk)), rp_factor)
        print(f"| {model_input_shape} | {new_seqlen} | {q_chunk}, {k_chunk} | {duration / 1e6:.3f} |")
