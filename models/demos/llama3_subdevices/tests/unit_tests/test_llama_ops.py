# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch
import pytest

from models.utility_functions import comp_pcc

from models.utility_functions import skip_for_blackhole
from tests.ttnn.unit_tests.operations.test_distributed_layernorm_sharded import (
    create_input_and_weight_tensors,
    create_tt_tensors,
    create_output_memory_config,
    compute_reference_output,
    compute_pre_allgather_stats,
    compute_post_allgather_output,
)
from tests.tt_eager.python_api_testing.unit_testing.misc.test_scaled_dot_product_attention_decode import (
    run_test_sdpa_decode_paged_attention_single_iter,
)
from tests.tt_eager.python_api_testing.unit_testing.misc.test_nlp_create_qkv_heads_decode import (
    run_test_create_min_width_shard,
)
from tests.tt_eager.python_api_testing.unit_testing.misc.test_nlp_concat_heads_decode import run_test_concat_head
from tests.ttnn.unit_tests.operations.test_paged_fused_update_cache import run_test_paged_fused_update_cache_decode
from tests.tt_eager.python_api_testing.unit_testing.misc.test_rotary_embedding_llama import (
    run_test_rotary_embedding_llama,
)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
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
    device,
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
    ([8, 8, 1, 4096, 128, (8, 4)],),  # Llama2-70B
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
    device, use_program_cache, b, nh, nkv, s, d, dtype, grid_size, q_dtype, start_core, sub_core_grids
):
    run_test_sdpa_decode_paged_attention_single_iter(
        device,
        b,
        nh,
        nkv,
        s,
        d,
        dtype,
        grid_size,
        q_dtype,
        cur_pos=127,
        block_size=32,
        q_chunk_size=0,
        k_chunk_size=0,
        sharded_in=True,
        sharded_out=True,
        start_core=start_core,
        sub_core_grids=sub_core_grids,
    )
    assert device.num_program_cache_entries() == 1


@skip_for_blackhole("Requires eth connected devices to run, see #12349")
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.parametrize("batch, batch_offset, slice_size", ((32, 0, 8),))
@pytest.mark.parametrize(
    "n_local_heads, n_local_kv_heads, head_dim",
    ((8, 1, 128),),
)
@pytest.mark.parametrize("overlap_coregrid", (False,))
@pytest.mark.parametrize(
    "sub_core_grids",
    (
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
            }
        ),
    ),
)
def test_llama_tg_NLPCreateHeadsDecodeDeviceOperation(
    device,
    batch,
    batch_offset,
    slice_size,
    n_local_heads,
    n_local_kv_heads,
    head_dim,
    overlap_coregrid,
    use_program_cache,
    sub_core_grids,
):
    batch_offset_tensor = torch.tensor([batch_offset], dtype=torch.int32)
    # convert to tt tensor
    batch_offset_tensor_tt = ttnn.from_torch(batch_offset_tensor, device=device, layout=ttnn.TILE_LAYOUT)

    torch.manual_seed(0)
    run_test_create_min_width_shard(
        device=device,
        batch=batch,
        n_local_heads=n_local_heads,
        n_local_kv_heads=n_local_kv_heads,
        head_dim=head_dim,
        overlap_coregrid=overlap_coregrid,
        sub_core_grids=sub_core_grids,
        batch_offset=batch_offset_tensor_tt,
        slice_size=slice_size,
    )


@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.parametrize(
    "n_local_heads, padded_local_heads, head_dim, batch_size, sub_core_grids",
    (
        (
            8,
            32,
            128,
            32,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 0)),
                ]
            ),
        ),
    ),
)
def test_llama_tg_NLPConcatHeadsDecodeDeviceOperation(
    device,
    n_local_heads,
    padded_local_heads,
    head_dim,
    batch_size,
    sub_core_grids,
    use_program_cache,
):
    torch.manual_seed(0)

    run_test_concat_head(device, n_local_heads, padded_local_heads, head_dim, batch_size, sub_core_grids)


@pytest.mark.parametrize("paged_update", [True])
@pytest.mark.parametrize("block_size", [64], ids=["block64"])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [8])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("cache_idx", [127])
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("pcc", [0.9995])
def test_llama_tg_PagedUpdateCacheDeviceOperation(
    device,
    paged_update,
    cache_idx,
    block_size,
    head_dim,
    max_seq_len,
    num_users,
    num_heads,
    input_dtype,
    cache_dtype,
    use_program_cache,
    pcc,
):
    run_test_paged_fused_update_cache_decode(
        paged_update,
        cache_idx,
        block_size,
        head_dim,
        max_seq_len,
        num_users,
        num_heads,
        input_dtype,
        cache_dtype,
        device,
        pcc,
    )


@skip_for_blackhole("Requires eth connected devices to run, only single chip BH available. See #12349")
@pytest.mark.parametrize("batch, seq_len", ((8, 1),))
@pytest.mark.parametrize(
    "n_heads, n_kv_heads, head_dim",
    ((8, 1, 128),),
)
@pytest.mark.parametrize("datatype", (ttnn.bfloat16,))
@pytest.mark.parametrize("pcc", (0.9997,))
def test_llama_tg_RotaryEmbeddingLlamaFusedQK(
    batch,
    seq_len,
    n_heads,
    n_kv_heads,
    head_dim,
    datatype,
    pcc,
    device,
):
    run_test_rotary_embedding_llama(
        device, batch, seq_len, pcc, n_heads, n_kv_heads, head_dim, 1, datatype, fuse_qk=True
    )
