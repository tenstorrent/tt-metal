# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import ttnn
from models.common.utility_functions import is_blackhole
from tests.nightly.t3000.ccl.test_neighbor_pad_async import run_neighbor_pad_1d_impl, run_neighbor_pad_2d_impl


@pytest.mark.timeout(200)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 3),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "input_shape, halo_shard_dim, other_shard_dim, layout, input_dtype, padding_left, padding_right, padding_mode, cluster_axis, num_links, skip_for_ci_env, use_persistent_output_buffer",
    [
        ([1, 3, 23 * 4, 20 * 8, 32], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 3, False, False),
        ([3, 25 * 4, 20 * 8, 32], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4, False, False),
        ([1, 3, 23 * 4, 20 * 8, 384], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 3, False, False),
        ([3, 25 * 4, 20 * 8, 384], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4, False, False),
        ([1, 2, 46 * 4, 40 * 8, 384], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 2, False, False),
        ([2, 48 * 4, 40 * 8, 384], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4, False, False),
        ([1, 4, 46 * 4, 40 * 8, 192], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 4, True, False),
        ([4, 48 * 4, 40 * 8, 192], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4, True, False),
        ([1, 4, 46 * 4, 40 * 8, 384], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 4, True, False),
        ([4, 48 * 4, 40 * 8, 384], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4, True, False),
        ([1, 4, 92 * 4, 80 * 8, 384], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 4, True, False),
        ([4, 94 * 4, 80 * 8, 384], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4, True, False),
        ([1, 6, 92 * 4, 80 * 8, 192], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 4, True, False),
        ([6, 94 * 4, 80 * 8, 192], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4, True, False),
        ([1, 4, 184 * 4, 160 * 8, 192], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 4, True, False),
        ([4, 186 * 4, 160 * 8, 192], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4, True, False),
        ([1, 6, 184 * 4, 160 * 8, 96], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 4, True, False),
        ([6, 186 * 4, 160 * 8, 96], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4, True, False),
        ([28, 5, 106, 32], 0, 2, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 0, "replicate", 1, 1, False, False),
        ([28, 5, 106, 32], 2, 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 2, "replicate", 1, 1, False, False),
        ([28, 5, 106, 32], 2, 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 2, "zeros", 1, 1, False, False),
        ([28, 60, 106, 768], 0, 2, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 0, "replicate", 1, 1, True, False),
        ([82, 120, 212, 512], 0, 2, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 0, "replicate", 1, 1, True, False),
        ([1, 3, 23 * 4, 20 * 8, 32], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 3, False, True),
    ],
    ids=[
        "Wan_shape_1",
        "Wan_shape_2",
        "Wan_shape_3",
        "Wan_shape_4",
        "Wan_shape_5",
        "Wan_shape_6",
        "Wan_shape_7",
        "Wan_shape_8",
        "Wan_shape_9",
        "Wan_shape_10",
        "Wan_shape_11",
        "Wan_shape_12",
        "Wan_shape_13",
        "Wan_shape_14",
        "Wan_shape_15",
        "Wan_shape_16",
        "Wan_shape_17",
        "Wan_shape_18",
        "replicate_T_dim",
        "replicate_W_dim",
        "zeros_W_dim",
        "mochi_shape_1",
        "mochi_shape_2",
        "persistent_buffer",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_output",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "device_params, neighbor_pad_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_neighbor_pad_async_1d(
    mesh_device,
    input_shape,
    halo_shard_dim,
    other_shard_dim,
    padding_left,
    padding_right,
    padding_mode,
    cluster_axis,
    num_links,
    input_dtype,
    layout,
    mem_config_input,
    mem_config_output,
    enable_trace,
    neighbor_pad_topology,
    num_iters,
    is_ci_env,
    skip_for_ci_env,
    use_persistent_output_buffer,
):
    if is_ci_env:
        if skip_for_ci_env:
            pytest.skip("Skipping certain shapes in CI to reduce pipeline time")

    if is_blackhole() and num_links > 2:
        pytest.skip("Skipping num_links > 2 on BH (only 2 ethernet channels available)")

    run_neighbor_pad_1d_impl(
        mesh_device,
        input_shape=input_shape,
        halo_shard_dim=halo_shard_dim,
        other_shard_dim=other_shard_dim,
        padding_left=padding_left,
        padding_right=padding_right,
        padding_mode=padding_mode,
        cluster_axis=cluster_axis,
        num_links=num_links,
        input_dtype=input_dtype,
        layout=layout,
        mem_config_input=mem_config_input,
        mem_config_output=mem_config_output,
        enable_trace=enable_trace,
        neighbor_pad_topology=neighbor_pad_topology,
        num_iters=num_iters,
        use_persistent_output_buffer=use_persistent_output_buffer,
    )


# ---------------------------------------------------------------------------
# 2D tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "mesh_device, num_links",
    [
        [(4, 8), 1],
        [(4, 8), 4],
        [(4, 8), 1],
        [(4, 8), 2],
    ],
    ids=[
        "wh_4x8_1link",
        "wh_4x8_4link",
        "bh_4x8_1link",
        "bh_4x8_2link",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, use_persistent_output_buffer",
    [
        # 5D: [B, T, H, W, C] — H along axis 0, W along axis 1
        ([1, 2, 8, 16, 32], 2, 3, 0, 1, 1, 1, False),
        ([1, 3, 12, 16, 32], 2, 3, 0, 1, 1, 1, False),
        # VAE conv_0 shape (full H=90, W=160)
        ([1, 3, 92, 160, 32], 2, 3, 0, 1, 1, 1, False),
        # Flipped axes: H along axis 1, W along axis 0
        ([1, 2, 16, 8, 32], 2, 3, 1, 0, 1, 1, False),
        # 4D tensor [B, H, W, C]
        ([2, 8, 16, 32], 1, 2, 0, 1, 1, 1, False),
        # Larger channel dim
        ([1, 2, 8, 16, 384], 2, 3, 0, 1, 1, 1, False),
        # Padding > 1
        ([1, 2, 8, 16, 32], 2, 3, 0, 1, 2, 2, False),
        # Persistent output buffer
        ([1, 2, 8, 16, 32], 2, 3, 0, 1, 1, 1, True),
    ],
    ids=[
        "small_5d_h0w1",
        "medium_5d_h0w1",
        "vae_conv0_h0w1",
        "small_5d_h1w0",
        "small_4d_h0w1",
        "small_5d_largeC",
        "small_5d_pad2",
        "small_5d_persistent",
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_neighbor_pad_async_2d(
    mesh_device,
    input_shape,
    h_dim,
    w_dim,
    h_axis,
    w_axis,
    pH,
    pW,
    num_links,
    use_persistent_output_buffer,
    device_params,
):
    run_neighbor_pad_2d_impl(
        mesh_device,
        input_shape=list(input_shape),
        h_dim=h_dim,
        w_dim=w_dim,
        h_axis=h_axis,
        w_axis=w_axis,
        pH=pH,
        pW=pW,
        padding_mode="zeros",
        num_links=num_links,
        input_dtype=ttnn.bfloat16,
        topology=ttnn.Topology.Linear,
        use_persistent_output_buffer=use_persistent_output_buffer,
    )
