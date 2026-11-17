# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Smoke tests for the Quasar copies of the conv / pool / experimental ops (adaptive_pool, cnn, conv1d, conv3d,
conv_transpose2d, deepseek_prefill, grid_sample, high_bw_all_gather, indexer_score, kda,
ring_attention_all_gather_async, rotate, topk_large_indices, upsample) under ttnn.experimental.quasar.

test_quasar_op_copies_are_bound needs no device: it catches a copy that compiles but was never registered with Python.
The device tests run a few of the copies whose host code was rewired onto other Quasar ops (upsample -> quasar
untilize kernel; adaptive pools -> quasar pool_generic) against a torch reference.
"""

import functools

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

_QUASAR = "ttnn.experimental.quasar."
_DS = _QUASAR + "deepseek_prefill."
_KDA = _QUASAR + "kda."

QUASAR_OP_COPIES = (
    [
        _QUASAR + name
        for name in (
            "adaptive_avg_pool2d",
            "adaptive_max_pool2d",
            "conv1d",
            "conv3d",
            "conv_transpose2d",
            "convert_to_chw",
            "convert_to_hwc",
            "grid_sample",
            "high_bw_all_gather",
            "indexer_score_dsa",
            "indexer_score_msa",
            "prepare_conv3d_weights",
            "ring_attention_all_gather_async",
            "ring_indexer_score_dsa",
            "rotate",
            "topk_large_indices",
            "upsample",
        )
    ]
    + [
        _DS + name
        for name in (
            "attn_res_block_mix",
            "attn_res_block_stats",
            "attn_res_gather_softmax",
            "attn_res_weighted_reduce_nc",
            "combine",
            "combine_fabric2d",
            "compressor_state_exchange",
            "csa_compressor",
            "dispatch",
            "dispatch_fabric2d",
            "dispatch_fabric2d_buffers",
            "extract",
            "hybrid_routed_expert_moe",
            "hybrid_routed_expert_moe_channel_semaphore_count",
            "hybrid_routed_expert_moe_forwarding_buffer",
            "inbound_socket_service_sync",
            "insert",
            "masked_bincount",
            "mhc_split_sinkhorn",
            "moe_fused_swiglu",
            "moe_grouped_topk",
            "moe_hash_gate",
            "moe_padding_config",
            "offset_cumsum",
            "outbound_socket_service_sync",
            "pack_scaled_fp8_kv_cache",
            "per_token_cast_back",
            "per_token_cast_to_fp8",
            "post_combine_reduce",
            "rotary_embedding_indexed",
            "routed_expert_ffn",
            "unified_routed_expert_moe",
            "update_padded_kv_cache",
            "zero_padded_kv_cache",
        )
    ]
    + [
        _KDA + name
        for name in (
            "affine_exclusive_scan",
            "chain_affine_transforms",
            "entry_correction",
            "indexed_entry_copy",
            "prepare_chunk_recurrence",
            "qkv_causal_conv1d_silu",
            "qkv_causal_conv1d_silu_with_carry",
            "recurrent_chunk_scan",
            "relay_affine_prefix",
            "select_final_carry",
            "sigmoid_gated_rms_norm",
            "summarize_chunk_recurrence",
        )
    ]
)


@pytest.mark.parametrize("qualified_name", QUASAR_OP_COPIES)
def test_quasar_op_copies_are_bound(qualified_name):
    path = qualified_name.split(".")[1:]  # drop the leading "ttnn"
    op = functools.reduce(getattr, path, ttnn)
    assert callable(op), f"{qualified_name} is bound but not callable"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
# Widths are multiples of 32 so TILE_LAYOUT inputs have no padding; the TILE case runs the quasar untilize kernel.
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["row_major", "tile"])
@pytest.mark.parametrize("batch_size, channels, height, width", [(1, 64, 8, 32), (2, 32, 4, 32)])
@pytest.mark.parametrize("scale_h, scale_w", [(2, 2), (3, 1)])
def test_quasar_upsample_nearest(device, layout, batch_size, channels, height, width, scale_h, scale_w):
    torch.manual_seed(0)
    torch_input_nhwc = torch.rand((batch_size, height, width, channels), dtype=torch.bfloat16)
    expected = torch_input_nhwc.repeat_interleave(scale_h, dim=1).repeat_interleave(scale_w, dim=2)

    tt_input = ttnn.from_torch(torch_input_nhwc, device=device, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_output = ttnn.experimental.quasar.upsample(tt_input, (scale_h, scale_w))
    actual = ttnn.to_torch(tt_output)

    assert_with_pcc(expected, actual, 0.9999)


# 17x17 -> 4x3 becomes a 5x7 avg pool with stride (4, 6). On Wormhole the existing Quasar avg_pool2d (pool_generic)
# returns garbage for any window of more than 16 elements with stride > 1, because it sums device memory that was
# never cleared (the original ttnn.avg_pool2d does not). On a freshly opened device that memory happens to be zero,
# so the case passes; after earlier tests it fails. Not strict for that reason.
_LARGE_WINDOW_AVG_POOL_XFAIL = pytest.mark.xfail(
    reason="Quasar avg_pool2d reads uncleared memory for >16-element windows with stride > 1", strict=False
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "pool_type, batch_size, channels, in_h, in_w, out_h, out_w",
    [
        ("avg", 1, 64, 14, 14, 7, 7),
        ("max", 1, 64, 14, 14, 7, 7),
        ("avg", 1, 128, 16, 16, 1, 1),
        ("max", 1, 128, 16, 16, 1, 1),
        pytest.param("avg", 1, 32, 17, 17, 4, 3, marks=_LARGE_WINDOW_AVG_POOL_XFAIL),
        ("max", 1, 32, 17, 17, 4, 3),
    ],
)
def test_quasar_adaptive_pool2d(device, pool_type, batch_size, channels, in_h, in_w, out_h, out_w):
    torch.manual_seed(0)
    # ttnn pools take a flattened [1, 1, N*H*W, C] (NHWC) input.
    torch_nchw = torch.randn((batch_size, channels, in_h, in_w), dtype=torch.bfloat16)
    torch_flat = torch_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch_size * in_h * in_w, channels)

    torch_pool = (
        torch.nn.functional.adaptive_avg_pool2d if pool_type == "avg" else torch.nn.functional.adaptive_max_pool2d
    )
    expected = torch_pool(torch_nchw.float(), (out_h, out_w)).permute(0, 2, 3, 1).reshape(1, 1, -1, channels)

    tt_input = ttnn.from_torch(torch_flat, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_op = (
        ttnn.experimental.quasar.adaptive_avg_pool2d
        if pool_type == "avg"
        else ttnn.experimental.quasar.adaptive_max_pool2d
    )
    tt_output = tt_op(
        input_tensor=tt_input,
        batch_size=batch_size,
        input_h=in_h,
        input_w=in_w,
        channels=channels,
        output_size=[out_h, out_w],
    )
    actual = ttnn.to_torch(tt_output).float().reshape(1, 1, -1, channels)

    assert_with_pcc(expected, actual, 0.999)
