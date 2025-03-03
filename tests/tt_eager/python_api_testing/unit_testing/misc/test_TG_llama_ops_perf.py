# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch
import pytest
from loguru import logger

from models.utility_functions import (
    skip_for_wormhole_b0,
    comp_allclose_and_pcc,
    comp_pcc,
    comp_allclose,
)

from models.utility_functions import tt2torch_tensor, get_devices_for_t3000, skip_for_grayskull
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from tt_metal.tools.profiler.process_model_log import get_samples_per_s
from tests.ttnn.unit_tests.operations.test_distributed_layernorm_sharded import (
    create_input_and_weight_tensors,
    create_tt_tensors,
    create_output_memory_config,
    compute_reference_output,
    compute_pre_allgather_stats,
    compute_post_allgather_output,
)
from tests.tt_eager.python_api_testing.unit_testing.misc.test_scaled_dot_product_attention_decode import (
    run_test_sdpa_decode_single_iter,
)


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.parametrize("is_rmsnorm", [True])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize(("min_pcc", "max_atol"), ((0.9997, 0.45),))
@pytest.mark.parametrize("input_width", [2048])
@pytest.mark.parametrize("num_devices", [1])
@pytest.mark.parametrize("input_df", [ttnn.bfloat16])
@pytest.mark.parametrize("weights_df", [ttnn.bfloat16])
@pytest.mark.parametrize(("mean", "std"), ([0, 1],))
@pytest.mark.parametrize(
    "core_grid, grid_offset, output_core_grid",
    [
        ((2, 8), ttnn.CoreCoord(1, 0), (2, 8)),
    ],
)
def test_llama_tg_LayerNorm(
    mesh_device,
    use_program_cache,
    input_width,
    num_devices,
    is_rmsnorm,
    input_df,
    weights_df,
    seed,
    eps,
    mean,
    std,
    min_pcc,
    max_atol,
    core_grid,
    grid_offset,
    output_core_grid,
):
    device = mesh_device.get_devices()[0]
    # Create input and weight tensors
    torch_input_tensor, torch_weight, torch_input_chunks, torch_weight_chunks = create_input_and_weight_tensors(
        input_width, num_devices, seed, mean, std
    )

    if output_core_grid is None:
        output_core_grid = core_grid
    out_memory_config = create_output_memory_config(output_core_grid, torch_input_chunks[0].shape)

    # Compute reference output
    torch_output_tensor = compute_reference_output(torch_input_tensor, torch_weight, is_rmsnorm, eps)
    torch_output_chunks = torch.chunk(torch_output_tensor, num_devices, dim=-1)

    # Simulate multi-device pre-allgather computation
    tt_pre_allgather_outputs = []
    for d in range(num_devices):
        tt_input_tensor = create_tt_tensors(
            torch_input_chunks[d], device, input_df, core_grid, input_width, grid_offset=grid_offset
        )
        tt_pre_allgather_output = compute_pre_allgather_stats(tt_input_tensor, core_grid, input_width, is_rmsnorm)
        tt_pre_allgather_outputs.append(tt_pre_allgather_output)

    # Extract and concatenate statistics from pre-allgather outputs
    tt_stats_list = []
    for tt_pre_allgather_output in tt_pre_allgather_outputs:
        tt_pre_allgather_output = ttnn.to_memory_config(tt_pre_allgather_output, memory_config=ttnn.L1_MEMORY_CONFIG)
        tt_stats_list.append(tt_pre_allgather_output)

    tt_global_stats = ttnn.concat(tt_stats_list, -1)
    # shard to 1 core
    tt_stats_sharded_config = ttnn.create_sharded_memory_config(
        shape=(32, tt_global_stats.padded_shape[-1]),
        core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(grid_offset, grid_offset)]),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )
    tt_global_stats = ttnn.to_memory_config(tt_global_stats, memory_config=tt_stats_sharded_config)

    # Simulate multi-device post-allgather computation
    tt_output_chunks = []
    for d in range(num_devices):
        tt_input_tensor = create_tt_tensors(
            torch_input_chunks[d], device, input_df, core_grid, input_width, grid_offset=grid_offset
        )
        tt_weights = create_tt_tensors(
            torch_weight_chunks[d], device, weights_df, core_grid, input_width, is_weight=True
        )
        tt_output_tensor = compute_post_allgather_output(
            tt_input_tensor,
            tt_weights,
            tt_global_stats,
            eps,
            is_rmsnorm,
            core_grid,
            input_width,
            input_df,
            out_memory_config,
        )

        tt_output_chunks.append(ttnn.to_torch(tt_output_tensor).to(torch.bfloat16))

    # Concatenate output chunks
    tt_output_torch = torch.cat(tt_output_chunks, dim=-1)

    # Compare results
    _, pcc_out = comp_pcc(torch_output_tensor, tt_output_torch, pcc=min_pcc)
    all_close_passing = torch.allclose(torch_output_tensor, tt_output_torch, atol=max_atol, equal_nan=False)
    atol_delta = torch.max(torch.abs(torch_output_tensor - tt_output_torch)).item()

    assert pcc_out >= min_pcc, f"PCC test failed: {pcc_out} (threshold: {min_pcc})"
    assert atol_delta <= max_atol, f"Max Atol exceeded: {atol_delta} (allowed: {max_atol})"


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "bfp8_cache_bf16_act",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    ([8, 8, 1, 256, 128, (8, 4)],),  # Llama2-70B
)
@pytest.mark.parametrize(
    "start_core, sub_core_grids",
    [
        (
            ttnn.CoreCoord(1, 0),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
                ]
            ),
        ),
    ],
)
def test_llama_tg_ScaledDotProductAttentionDecode(
    mesh_device, use_program_cache, b, nh, nkv, s, d, dtype, grid_size, q_dtype, start_core, sub_core_grids
):
    device = mesh_device.get_devices()[0]
    run_test_sdpa_decode_single_iter(
        device,
        b,
        nh,
        nkv,
        s,
        d,
        dtype,
        grid_size,
        q_dtype,
        sharded_in=True,
        sharded_out=True,
        start_core=start_core,
        sub_core_grids=sub_core_grids,
        override_q_chunk_size=256,
        override_k_chunk_size=256,
    )
    assert device.num_program_cache_entries() == 1


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    ("op_name", "expected_kernel_duration_us"),
    [
        ("LayerNorm", 13),
        ("ScaledDotProductAttentionDecode", 20),
    ],
)
def test_llama_tg_ops_perf_device(op_name, expected_kernel_duration_us):
    batch = 32
    test = "llama-distributed-ln"
    subdir = "llama-unit-tests"
    margin = 0.03
    num_iterations = 1

    command = (
        f"pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_TG_llama_ops_perf.py::test_llama_tg_{op_name}"
    )
    cols = ["DEVICE KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"
    expected_perf_cols = {inference_time_key: expected_kernel_duration_us * 1e3}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch, op_name, has_signposts=False)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"llama-tg-{op_name}",
        batch_size=batch,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
