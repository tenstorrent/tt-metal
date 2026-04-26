# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import models.demos.deepseek_v4_flash.ttnn_attention_projection as attention_projection
import ttnn
from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.real_paged_sdpa_decode_trace_smoke import run_paged_sdpa_decode_trace_smoke
from models.demos.deepseek_v4_flash.real_traceable_decode_smoke import (
    TRACEABLE_DECODE_ATTENTION_LEGACY_BLEND_MODE,
    TRACEABLE_DECODE_ATTENTION_QK_SOFTMAX_MODE,
    TRACEABLE_DECODE_ATTENTION_READ_FIXED_SLICE,
    TRACEABLE_DECODE_ATTENTION_READ_PAGED_SDPA_DECODE,
    TRACEABLE_DECODE_ATTENTION_READ_SELECTED_ROWS_COMPRESSED_KV,
    TRACEABLE_DECODE_ATTENTION_READ_SELECTED_ROWS_DENSE,
    TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR,
    TRACEABLE_DECODE_CACHE_UPDATE_HOST_SCALAR,
    TRACEABLE_DECODE_COMPRESSOR_STATEFUL_RATIO4_PROBE,
    TRACEABLE_DECODE_INDEXER_COMPRESSOR_STATEFUL_RATIO4_PROBE,
    TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR,
    TRACEABLE_DECODE_ROPE_POSITION_STATIC,
    TRACEABLE_DECODE_SPARSE_INDEXER_TRACE_TOPK_ATTENTION,
    TraceableDecodeHostFallbackError,
    TraceableDecodeHostGuard,
    _device_selected_cache_rows_by_step_from_outputs,
    run_traceable_decode_subpath_smoke,
)
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def _trace_replay_paged_update_cache_two_positions(device_id: int = 0) -> dict[str, object]:
    device = ttnn.open_device(device_id=int(device_id), num_command_queues=1, trace_region_size=32 * 1024 * 1024)
    trace_id = None
    try:
        cache = ttnn.from_torch(
            torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_input = _to_sharded_paged_update_input(
            torch.full((1, 1, 1, 32), 11.0, dtype=torch.bfloat16),
            device=device,
        )
        update_idxs = ttnn.from_torch(
            torch.tensor([4], dtype=torch.int32),
            device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.experimental.paged_update_cache(cache, tt_input, update_idxs_tensor=update_idxs)
        ttnn.synchronize_device(device)

        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        ttnn.experimental.paged_update_cache(cache, tt_input, update_idxs_tensor=update_idxs)
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        ttnn.synchronize_device(device)

        replay_steps = [(4, 21.0), (5, 31.0)]
        for position, value in replay_steps:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.full((1, 1, 1, 32), value, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                tt_input,
            )
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.tensor([position], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                update_idxs,
            )
            ttnn.synchronize_device(device)
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(device)

        got = ttnn.to_torch(cache).float()
        return {
            "trace_count": 1,
            "positions": [position for position, _ in replay_steps],
            "row4": got[0, 0, 4, :].clone(),
            "row5": got[0, 0, 5, :].clone(),
            "row6": got[0, 0, 6, :].clone(),
        }
    finally:
        if trace_id is not None:
            ttnn.release_trace(device, trace_id)
        ttnn.close_device(device)


def _to_sharded_paged_update_input(torch_input: torch.Tensor, *, device):
    host = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    shard_grid = ttnn.num_cores_to_corerangeset(1, device.compute_with_storage_grid_size(), row_wise=True)
    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        [host.volume() // host.padded_shape[-1], host.padded_shape[-1]],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        input_shard_spec,
    )
    return host.to(device, input_memory_config)


def test_traceable_decode_guard_rejects_ttnn_to_torch_inside_region() -> None:
    with TraceableDecodeHostGuard() as guard:
        assert "ttnn.to_torch" in guard.guarded_labels
        with pytest.raises(TraceableDecodeHostFallbackError, match="ttnn.to_torch"):
            ttnn.to_torch(object())


def test_traceable_decode_guard_rejects_known_attention_host_helper() -> None:
    with TraceableDecodeHostGuard():
        with pytest.raises(TraceableDecodeHostFallbackError, match="grouped_output_projection_a"):
            attention_projection.grouped_output_projection_a(
                torch.zeros(1, 1, 4, dtype=torch.bfloat16),
                torch.zeros(4, 4, dtype=torch.bfloat16),
                o_groups=1,
            )


def test_paged_update_cache_tensor_index_replays_one_trace_across_positions() -> None:
    if os.environ.get("DSV4_FLASH_TRACEABLE_DECODE", "0") != "1":
        pytest.skip("Set DSV4_FLASH_TRACEABLE_DECODE=1 to run the paged_update_cache trace replay primitive")

    result = _trace_replay_paged_update_cache_two_positions(device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")))

    assert result["trace_count"] == 1
    assert result["positions"] == [4, 5]
    assert torch.equal(result["row4"], torch.full((32,), 21.0))
    assert torch.equal(result["row5"], torch.full((32,), 31.0))
    assert torch.equal(result["row6"], torch.zeros((32,)))


def test_cpu_traceable_decode_subpath_reports_inventory_and_limitations(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = run_traceable_decode_subpath_smoke(
        snapshot,
        layer=3,
        seq_len=4,
        max_bytes=20 * 1024,
        routed_topk_prefix=1,
        cpu_only=True,
    )

    assert result["mode"] == "cpu-reference"
    assert result["passed"] is True
    assert result["decode_step_count"] == 1
    assert result["positions"] == [4]
    assert result["positions_used"] == [4]
    assert result["one_trace_capture_replayed_across_positions"] is False
    assert result["trace_capture"]["attempted"] is False
    assert result["trace_capture_attempted"] is False
    assert result["trace_capture_passed"] is False
    assert result["trace_capture"]["single_capture_replayed_across_positions"] is False
    assert result["trace_capture"]["one_trace_capture_replayed_across_positions"] is False
    assert result["trace_capture"]["cache_update_index_dynamic"] is False
    assert result["trace_capture"]["ttnn_to_torch_guarded"] is True
    assert result["guard_status"]["ttnn_to_torch_guarded"] is True
    assert result["host_boundaries_inside_trace"] == []
    assert result["cache_update"]["name"] == "compressed_kv_projection_cache_append"
    assert result["cache_update"]["cache_update_api"] == TRACEABLE_DECODE_CACHE_UPDATE_HOST_SCALAR
    assert result["cache_update"]["update_index_source"] == "host_scalar"
    assert result["cache_update"]["cache_len"] == 64
    assert result["cache_update"]["update_index"] == 4
    assert result["cache_update"]["update_indices"] == [4]
    assert result["cache_update"]["updated_rows"] == [4]
    assert result["cache_update"]["per_step_updated_rows"] == [[4]]
    assert result["cache_update"]["dynamic_update_index_in_trace"] is False
    assert result["cache_update"]["cache_read_window_dynamic"] is False
    assert result["attention_read_api"] == TRACEABLE_DECODE_ATTENTION_READ_FIXED_SLICE
    assert result["cache_update"]["attention_read_api"] == TRACEABLE_DECODE_ATTENTION_READ_FIXED_SLICE
    assert result["cache_update"]["rope_position_dynamic"] is False
    assert result["rope_position_api"] == TRACEABLE_DECODE_ROPE_POSITION_STATIC
    assert result["rope_position_dynamic"] is False
    assert result["rope_positions"] == [4]
    assert result["position_dependent_decode_inventory"]["dynamic_rope_position"]["status"] == "available"
    assert result["position_dependent_decode_inventory"]["dynamic_cache_read_current_position"]["status"] == (
        "available_not_integrated"
    )
    assert result["cache_update"]["single_capture_replay_across_positions"] is False
    assert result["cache_update"]["device_resident_inside_trace"] is True
    assert result["multi_position_replay"]["single_capture_replayed_across_positions"] is False
    assert result["multi_position_replay"]["cache_update_index_dynamic"] is False
    assert result["cache_read_window_dynamic"] is False
    assert result["rope_position_dynamic"] is False
    assert result["traceable_decode_scope"]["not_full_forward"] is True
    assert result["traceable_decode_scope"]["inside_trace"] == [
        "ttnn.rms_norm(attn_norm)",
        "TtAttentionProjection.project_q_rank",
        "TtAttentionProjection.project_q_from_rank",
        "ttnn.linear(wkv)",
        "ttnn.rms_norm(kv_norm)",
        "ttnn.to_memory_config(kv_update_height_sharded)",
        "ttnn.update_cache(kv_projection_cache)",
        "ttnn.slice(kv_cache_fixed_window)",
        "ttnn.reshape(q_output_to_q_heads_token_major)",
        "ttnn.transpose(q_heads_token_major_to_heads)",
        "ttnn.rms_norm(q_heads)",
        "ttnn.slice(q_nope/q_rope)",
        "ttnn.experimental.rotary_embedding_llama(q_rope)",
        "ttnn.concat(q_nope,q_rope_rotated)",
        "ttnn.slice(kv_cache_window_to_k_nope/k_rope)",
        "ttnn.experimental.rotary_embedding_llama(k_rope)",
        "ttnn.concat(k_nope,k_rope_rotated)",
        "ttnn.repeat(k_cache_to_attention_heads)",
        "ttnn.repeat(v_cache_window_to_attention_heads)",
        "ttnn.transpose(k_heads_to_k_heads_transposed)",
        "ttnn.matmul(q_heads,k_heads_transposed)",
        "ttnn.mul(qk_scores,1/sqrt(head_dim))",
        "ttnn.softmax(qk_scores)",
        "ttnn.matmul(attention_probs,value_heads)",
        "ttnn.transpose(context_heads_to_token_major)",
        "ttnn.reshape(context_heads_to_attention_output)",
        "TtAttentionProjection.project_output",
        "ttnn.slice(attention_output_group_0..N)",
        "ttnn.linear(grouped_wo_a_group_0..N)",
        "ttnn.concat(grouped_wo_a_rank)",
        "ttnn.linear(wo_b)",
        "ttnn.add(hidden,attention_projected)",
        "ttnn.rms_norm(ffn_norm)",
        "ttnn.linear(router_gate)",
        "ttnn.softplus(router_logits)",
        "ttnn.sqrt(router_softplus)",
        "ttnn.add(router_scores,router_bias)",
        "ttnn.topk(router_selection_scores)",
        "ttnn.gather(router_scores,router_topk_indices)",
        "ttnn.sum(router_topk_route_scores)",
        "ttnn.div(router_topk_route_scores,router_weight_sum)",
        "ttnn.mul(router_route_weights,routed_scaling_factor)",
        "ttnn.slice(router_route_weights_topk_prefix)",
        "ttnn.mul(router_selected_route_weights,decode_row_mask)",
        "TtRoutedExpertMLP(selected_topk_prefix)",
        "ttnn.mul(routed_hidden,device_router_route_weight)",
        "ttnn.add(routed_expert_outputs)",
        "TtSharedExpertMLP",
        "ttnn.add(shared_output,routed_output)",
        "ttnn.add(post_attention_residual,combined_ffn_output)",
    ]
    assert result["traceability_flags"]["router_gate_matmul_in_trace"] is True
    assert result["traceability_flags"]["router_scoring_in_trace"] is True
    assert result["traceability_flags"]["router_topk_in_trace"] is True
    assert result["traceability_flags"]["router_route_weights_in_trace"] is True
    assert result["traceability_flags"]["router_expert_dispatch_dynamic_in_trace"] is False
    assert result["router_trace"]["mode"] == "device_gate_scoring_topk_route_weights_static_dispatch"
    assert result["router_trace"]["topk_in_trace"] is True
    assert result["router_trace"]["route_weights_in_trace"] is True
    assert result["router_trace"]["expert_dispatch"] == "static_preflight"
    assert result["router_trace"]["selected_expert_ids"] == [3]
    assert result["router_trace"]["expected_full_topk_expert_ids"] == [3, 2]
    assert (
        "dynamic MoE expert dispatch; selected expert modules are statically instantiated from a host preflight plan"
        in result["traceable_decode_scope"]["excluded_from_trace"]
    )
    assert result["selected_routing"]["selection_boundary"] == "host_preflight_static_dispatch_device_router_weights"
    assert result["selected_routing"]["topk_prefix_limit"] == 1
    assert result["selected_routing"]["topk_prefix_is_full"] is False
    assert result["selected_routing"]["full_topk_mode"] is False
    assert result["selected_routing"]["selected_expert_ids"] == [3]
    assert result["selected_routing"]["executed_expert_ids"] == [3]
    assert result["selected_routing"]["route_weights_device_resident_inside_trace"] is True
    assert result["routed_expert_execution"]["loaded_expert_ids"] == [3]
    assert result["routed_expert_execution"]["executed_expert_ids"] == [3]
    assert result["routed_expert_execution"]["full_topk_executed"] is False
    assert (
        "DeepSeek sparse attention-sink/indexer semantics; fixed-window dense softmax uses contiguous cache rows"
        in result["traceable_decode_scope"]["excluded_from_trace"]
    )
    assert result["attention_path"]["mode"] == TRACEABLE_DECODE_ATTENTION_QK_SOFTMAX_MODE
    assert result["attention_path"]["attention_read_api"] == TRACEABLE_DECODE_ATTENTION_READ_FIXED_SLICE
    assert result["attention_path"]["host_provided_attention_output"] is False
    assert result["attention_path"]["cache_window"] == {
        "start": 4,
        "end_exclusive": 8,
        "length": 4,
        "rows": [4, 8],
        "logical_decode_row": 4,
        "updated_row_is_first_window_row": True,
        "static_padding_rows": 3,
    }
    assert result["attention_path"]["cache_expand"]["repeat_factor"] == 4
    assert result["attention_path"]["cache_expand"]["repeat_axis"] == "attention_heads"
    assert result["attention_path"]["kv_source"]["key_value_share_same_cache_slice"] is True
    assert result["attention_path"]["kv_source"]["kv_split_in_trace"] is True
    assert result["attention_path"]["kv_source"]["explicit_kv_tensors_in_trace"] is True
    assert result["attention_path"]["kv_source"]["key_value_identical_in_trace"] is False
    assert result["attention_path"]["kv_source"]["true_kv_split_in_trace"] is False
    assert result["attention_path"]["rope"]["q_rope_split_in_trace"] is True
    assert result["attention_path"]["rope"]["k_rope_split_in_trace"] is True
    assert result["attention_path"]["rope"]["q_rope_rotation_in_trace"] is True
    assert result["attention_path"]["rope"]["k_rope_rotation_in_trace"] is True
    assert result["attention_path"]["rope"]["rope_in_trace"] is True
    assert result["traceability_flags"]["kv_split_in_trace"] is True
    assert result["traceability_flags"]["true_kv_split_in_trace"] is False
    assert result["traceability_flags"]["rope_in_trace"] is True
    assert result["traceability_flags"]["attention_sink_in_trace"] is False
    assert result["attention_sink_status"] == "loaded_not_integrated_in_fixed_slice_attention"
    assert result["selected_cache_rows_topk_shape"] == [1, 1, 9]
    assert result["sparse_attention"]["status"] == "real_indexer_rows_materialized_fixed_slice_attention_blocked"
    assert result["sparse_attention"]["compress_ratio"] == 4
    assert result["sparse_attention"]["sparse_indexer_status"] == (
        "real_indexer_topk_materialized_outside_trace_not_consumed_by_attention"
    )
    assert result["sparse_attention"]["selected_cache_rows"]["derived_from_real_indexer"] is True
    assert result["sparse_attention"]["selected_cache_rows"]["selected_rows_consumed_by_attention"] is False
    assert result["sparse_attention"]["selected_cache_rows"]["per_step"][0]["compressed_rows"] == [8]
    assert result["sparse_attention"]["selected_cache_rows"]["per_step"][0]["compressed_rows_source"] == (
        "real_learned_indexer_topk_host_preflight"
    )
    assert result["sparse_attention"]["selected_cache_rows"]["per_step"][0]["runtime_rows"] == [0, 1, 2, 3, 4, 8]
    assert result["sparse_attention"]["selected_cache_rows"]["per_step"][0]["rows_drive_attention_in_trace"] is False
    assert result["sparse_attention"]["attention_sink_status"] == "loaded_not_integrated_in_fixed_slice_attention"
    assert result["sparse_attention"]["attention_sink"]["in_trace"] is False
    assert result["sparse_attention"]["dynamic_cache_write"] is False
    assert result["sparse_attention"]["dynamic_rope"] is False
    assert result["sparse_attention"]["dynamic_current_position"] is False
    assert result["attention_path"]["sparse_compressed_tokens"]["contributed"] is True
    assert result["attention_path"]["compressed_token_contribution"]["contributed"] is True
    assert result["attention_path"]["softmax"]["qk_scores_in_trace"] is True
    assert result["attention_path"]["softmax"]["fixed_window_softmax_in_trace"] is True
    assert result["attention_path"]["softmax"]["value_reduction_in_trace"] is True
    assert result["attention_path"]["context"]["produced_in_trace"] is True
    assert result["loaded_tensor_groups"]["attention_query"]["count"] == 4
    assert result["loaded_tensor_groups"]["attention_output"]["count"] == 2
    assert result["loaded_tensor_groups"]["attention_output"]["canonical_keys"] == [
        "layers.3.attn.wo_a.weight",
        "layers.3.attn.wo_b.weight",
    ]
    assert result["loaded_tensor_groups"]["kv_projection"]["count"] == 2
    assert result["loaded_tensor_groups"]["attention_sink"]["canonical_keys"] == ["layers.3.attn.attn_sink"]
    assert result["loaded_tensor_groups"]["ffn_norm"]["canonical_keys"] == ["layers.3.ffn_norm.weight"]
    assert result["loaded_tensor_groups"]["router_selector"]["count"] == 2
    assert result["loaded_tensor_groups"]["shared_expert"]["count"] == 6
    assert result["loaded_tensor_groups"]["routed_experts"]["count"] == 6
    assert result["payload_bytes"]["attention_output"] == 5120
    assert result["payload_bytes"]["kv_projection"] == 544
    assert result["payload_bytes"]["attention_sink"] == 16
    assert result["payload_bytes"]["router_selector"] == 272
    assert result["payload_bytes"]["routed_experts"] == 1920
    assert result["payload_bytes"]["total"] == 16396
    assert result["decoded_tensors"]["wq_a"]["shape"] == [16, 32]
    assert result["decoded_tensors"]["wq_b"]["shape"] == [32, 16]
    assert result["decoded_tensors"]["wo_a"]["shape"] == [64, 8]
    assert result["decoded_tensors"]["wo_b"]["shape"] == [32, 64]
    assert result["decoded_tensors"]["wkv"]["shape"] == [8, 32]
    assert result["decoded_tensors"]["kv_norm"]["shape"] == [8]
    assert result["decoded_tensors"]["attn_sink"]["shape"] == [4]
    assert result["decoded_tensors"]["router_gate"]["shape"] == [4, 32]
    assert result["decoded_tensors"]["routed_experts"]["3"]["w1"]["shape"] == [32, 32]
    assert result["decoded_tensors"]["shared_w1"]["shape"] == [32, 32]
    assert result["inputs"]["kv_cache_initial"]["shape"] == [1, 1, 64, 8]
    assert result["reference"]["kv_output"]["shape"] == [1, 1, 4, 8]
    assert result["reference"]["kv_cache"]["shape"] == [1, 1, 64, 8]
    assert result["reference"]["attention_cache_window"]["shape"] == [1, 1, 4, 8]
    assert result["reference"]["rope_cos"]["shape"] == [1, 1, 4, 4]
    assert result["reference"]["rope_sin"]["shape"] == [1, 1, 4, 4]
    assert result["reference"]["attention_q_heads_pre_norm"]["shape"] == [1, 4, 4, 8]
    assert result["reference"]["attention_q_heads_norm"]["shape"] == [1, 4, 4, 8]
    assert result["reference"]["attention_q_nope"]["shape"] == [1, 4, 4, 4]
    assert result["reference"]["attention_q_rope"]["shape"] == [1, 4, 4, 4]
    assert result["reference"]["attention_q_rope_rotated"]["shape"] == [1, 4, 4, 4]
    assert result["reference"]["attention_q_heads"]["shape"] == [1, 4, 4, 8]
    assert result["reference"]["attention_key_cache_nope"]["shape"] == [1, 1, 4, 4]
    assert result["reference"]["attention_key_cache_rope"]["shape"] == [1, 1, 4, 4]
    assert result["reference"]["attention_key_cache_rope_rotated"]["shape"] == [1, 1, 4, 4]
    assert result["reference"]["attention_key_cache"]["shape"] == [1, 1, 4, 8]
    assert result["reference"]["attention_key_heads"]["shape"] == [1, 4, 4, 8]
    assert result["reference"]["attention_value_heads"]["shape"] == [1, 4, 4, 8]
    assert result["reference"]["qk_scores"]["shape"] == [1, 4, 4, 4]
    assert result["reference"]["attention_probs"]["shape"] == [1, 4, 4, 4]
    assert result["reference"]["attention_context_heads"]["shape"] == [1, 4, 4, 8]
    assert result["reference"]["attention_output"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["attention_projected"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["post_attention_residual"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["router_logits"]["shape"] == [1, 1, 4, 4]
    assert result["reference"]["router_topk_indices"]["shape"] == [1, 1, 4, 2]
    assert result["reference"]["router_route_weights"]["shape"] == [1, 1, 4, 2]
    assert result["reference"]["router_selected_route_weights_masked"]["shape"] == [1, 1, 4, 1]
    assert result["reference"]["router_decode_topk_indices"]["shape"] == [1, 1, 1, 2]
    assert result["reference"]["router_decode_route_weights"]["shape"] == [1, 1, 1, 2]
    assert result["reference"]["routed_output"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["combined_ffn_output"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["residual_output"]["shape"] == [1, 1, 4, 32]
    assert "router_static_dispatch_preflight" in result["host_boundaries_outside_trace"]
    assert "router_decode_row_mask_host_to_device" in result["host_boundaries_outside_trace"]
    assert "kv_cache_seed_host_to_device" in result["host_boundaries_outside_trace"]
    assert "rope_table_host_to_device" in result["host_boundaries_outside_trace"]
    assert "attention_output_host_to_device" not in result["host_boundaries_outside_trace"]
    assert result["accuracy"]["cpu_reference"]["passed"] is True
    assert result["accuracy_by_step"][0]["accuracy"]["cpu_reference"]["passed"] is True
    assert result["decode_steps_detail"][0]["position"] == 4
    assert result["decode_steps_detail"][0]["cache_rows_updated"] == [4]
    assert result["decode_steps_detail"][0]["preflight_topk_expert_ids"] == [3, 2]
    assert result["decode_steps_detail"][0]["selected_expert_ids"] == [3]
    assert len(result["decode_steps_detail"][0]["replay_topk_route_weights"]) == 2


def test_cpu_traceable_decode_subpath_reports_two_static_cache_positions(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = run_traceable_decode_subpath_smoke(
        snapshot,
        layer=3,
        seq_len=4,
        max_bytes=24 * 1024,
        routed_topk_prefix=1,
        decode_steps=2,
        cpu_only=True,
    )

    assert result["passed"] is True
    assert result["decode_step_count"] == 2
    assert result["positions"] == [4, 5]
    assert result["positions_used"] == [4, 5]
    assert result["cache_update_api"] == TRACEABLE_DECODE_CACHE_UPDATE_HOST_SCALAR
    assert result["update_index_source"] == "host_scalar"
    assert result["cache_update"]["update_indices"] == [4, 5]
    assert result["cache_update"]["updated_rows"] == [4, 5]
    assert result["cache_update"]["per_step_updated_rows"] == [[4], [5]]
    assert result["cache_update"]["update_index_kind"] == "static_host_argument_per_trace_capture"
    assert result["cache_update"]["dynamic_update_index_in_trace"] is False
    assert result["cache_read_window_dynamic"] is False
    assert result["rope_position_dynamic"] is False
    assert result["rope_positions"] == [4, 5]
    assert result["multi_position_replay"]["carried_device_kv_cache_state"] is True
    assert result["multi_position_replay"]["single_capture_replayed_across_positions"] is False
    assert result["multi_position_replay"]["recaptured_per_position"] is True
    assert result["multi_position_replay"]["cache_update_index_dynamic"] is False
    assert len(result["decode_steps_detail"]) == 2
    assert [step["position"] for step in result["decode_steps_detail"]] == [4, 5]
    assert [step["cache_rows_updated"] for step in result["decode_steps_detail"]] == [[4], [5]]
    assert all(step["attention_path_stayed_in_trace"] for step in result["decode_steps_detail"])
    assert all(step["router_weights_stayed_in_trace"] for step in result["decode_steps_detail"])
    assert all(not step["router_expert_dispatch_dynamic_in_trace"] for step in result["decode_steps_detail"])
    assert all(len(step["preflight_topk_expert_ids"]) == 2 for step in result["decode_steps_detail"])
    assert all(len(step["replay_topk_route_weights"]) == 2 for step in result["decode_steps_detail"])
    assert result["reference_by_step"][1]["position"] == 5
    assert result["replay_reference_by_step"][1]["position"] == 5
    assert result["accuracy_by_step"][1]["accuracy"]["cpu_reference"]["passed"] is True


def test_cpu_traceable_decode_subpath_can_report_dynamic_rope_position_probe(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = run_traceable_decode_subpath_smoke(
        snapshot,
        layer=3,
        seq_len=4,
        max_bytes=24 * 1024,
        routed_topk_prefix=1,
        decode_steps=2,
        cpu_only=True,
        cache_update_api=TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR,
        rope_position_api=TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR,
    )

    assert result["passed"] is True
    assert result["cache_update_api"] == TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR
    assert result["rope_position_api"] == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR
    assert result["cache_update"]["dynamic_update_index_in_trace"] is True
    assert result["cache_read_window_dynamic"] is False
    assert result["attention_read_api"] == TRACEABLE_DECODE_ATTENTION_READ_FIXED_SLICE
    assert result["rope_position_dynamic"] is True
    assert result["rope_position_status"] == "replay_mutable_device_tensor_embedding"
    assert result["rope_positions"] == [4, 5]
    assert result["attention_path_by_step"][0]["cache_window"]["start"] == 4
    assert result["attention_path_by_step"][1]["cache_window"]["start"] == 4
    assert result["attention_path_by_step"][0]["rope"]["position_rows"] == [4, 8]
    assert result["attention_path_by_step"][1]["rope"]["position_rows"] == [5, 9]
    assert result["attention_path_by_step"][1]["rope"]["position_dynamic"] is True
    assert result["decode_steps_detail"][1]["cache_window_rows"] == [4, 8]
    assert result["decode_steps_detail"][1]["rope_position_rows"] == [5, 9]
    assert result["decode_steps_detail"][1]["rope_position_dynamic"] is True
    assert result["position_dependent_decode_inventory"]["dynamic_rope_position"]["status"] == "used"
    assert result["position_dependent_decode_inventory"]["dynamic_cache_read_current_position"]["status"] == (
        "available_not_integrated"
    )


def test_cpu_traceable_decode_subpath_can_report_paged_sdpa_read_probe(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = run_traceable_decode_subpath_smoke(
        snapshot,
        layer=3,
        seq_len=4,
        max_bytes=24 * 1024,
        routed_topk_prefix=1,
        decode_steps=2,
        cpu_only=True,
        attention_read_api=TRACEABLE_DECODE_ATTENTION_READ_PAGED_SDPA_DECODE,
        cache_update_api=TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR,
        rope_position_api=TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR,
    )

    assert result["passed"] is True
    assert result["attention_read_api"] == TRACEABLE_DECODE_ATTENTION_READ_PAGED_SDPA_DECODE
    assert result["cache_update"]["dynamic_update_index_in_trace"] is True
    assert result["rope_position_dynamic"] is True
    assert result["cache_read_window_dynamic"] is True
    assert result["dynamic_cache_read_current_position"] is True
    assert result["position_dependent_decode_inventory"]["dynamic_cache_read_current_position"]["status"] == "used"
    assert result["multi_position_replay"]["current_position_dynamic"] is True
    assert result["traceability_flags"]["dynamic_cache_read_current_position_in_trace"] is True
    assert result["attention_path"]["softmax"]["paged_sdpa_decode_in_trace"] is True
    assert result["attention_path"]["softmax"]["fixed_window_softmax_in_trace"] is False
    assert result["attention_path"]["qk_scores"]["inside_paged_sdpa_kernel"] is True
    assert result["attention_path"]["kv_source"]["key_source"] == "paged_deepseek_v4_cache_ready_kv_projection_cache"
    assert result["attention_path"]["kv_source"]["v_channel_kind"] == "fused_cache_ready_kv_reuse"
    assert result["attention_path"]["kv_source"]["true_separate_v_channel_in_trace"] is False
    assert result["attention_path"]["rope"]["kv_cache_ready_rope_rotation_before_cache_write"] is True
    assert result["attention_path"]["rope"]["attention_output_inverse_rope_in_trace"] is True
    assert result["traceability_flags"]["kv_cache_ready_rope_in_trace"] is True
    assert result["traceability_flags"]["attention_output_inverse_rope_in_trace"] is True
    assert result["traceability_flags"]["compressed_kv_cache_in_trace"] is False
    assert result["traceability_flags"]["sparse_indexer_in_trace"] is False
    assert result["traceability_flags"]["attention_sink_in_trace"] is True
    assert result["attention_sink_status"] == "used_in_paged_sdpa_decode"
    assert result["selected_cache_rows_topk_shape"] == [1, 1, 9]
    assert result["sparse_attention"]["status"] == "real_indexer_rows_materialized_attention_read_blocked"
    assert result["sparse_attention"]["sparse_indexer_status"] == (
        "real_indexer_topk_materialized_outside_trace_not_consumed_by_attention"
    )
    assert result["sparse_attention"]["selected_cache_rows"]["derived_from_real_indexer"] is True
    assert result["sparse_attention"]["selected_cache_rows"]["selected_rows_consumed_by_attention"] is False
    assert result["sparse_attention"]["selected_cache_rows"]["per_step"][0]["compressed_rows"] == [8]
    assert result["sparse_attention"]["selected_cache_rows"]["per_step"][0]["runtime_rows"] == [0, 1, 2, 3, 4, 8]
    assert result["sparse_attention"]["selected_cache_rows"]["per_step"][0]["paged_sdpa_dense_rows"] == {
        "start": 0,
        "end_exclusive": 5,
        "count": 5,
    }
    assert result["sparse_attention"]["selected_cache_rows"]["per_step"][0]["rows_drive_attention_in_trace"] is False
    assert result["sparse_attention"]["selected_row_attention_blocker"]["selected_rows_consumed_by_attention"] is False
    assert result["sparse_attention"]["selected_row_attention_blocker"]["concrete_shapes"]["hf_selected_rows"] == [
        0,
        1,
        2,
        3,
        4,
        8,
    ]
    assert result["sparse_attention"]["attention_sink_status"] == "used_in_paged_sdpa_decode"
    assert result["sparse_attention"]["attention_sink"]["in_trace"] is True
    assert result["sparse_attention"]["dynamic_cache_write"] is True
    assert result["sparse_attention"]["dynamic_rope"] is True
    assert result["sparse_attention"]["dynamic_current_position"] is True
    assert result["attention_path"]["softmax"]["attention_sink_softmax_in_trace"] is True
    assert result["attention_path"]["attention_sink"]["status"] == "used_in_paged_sdpa_decode"
    assert (
        result["deepseek_attention_reference_inventory"]["v_channel_layout"][
            "true_separate_v_projection_in_hf_reference"
        ]
        is False
    )
    assert result["deepseek_attention_reference_inventory"]["v_channel_layout"]["post_attention_inverse_rope"] is True
    assert result["deepseek_attention_reference_inventory"]["compressed_kv_layout"]["trace_status"] == "not_integrated"
    assert result["deepseek_attention_reference_inventory"]["attention_sink_layout"]["trace_status"] == (
        "integrated_in_paged_sdpa_decode"
    )
    assert result["attention_path_by_step"][0]["cache_window"]["rows"] == [0, 5]
    assert result["attention_path_by_step"][1]["cache_window"]["rows"] == [0, 6]
    assert result["decode_steps_detail"][1]["cache_rows_read"]["count"] == 6
    assert result["decode_steps_detail"][1]["cache_pages_read"]["logical_pages"] == [0]
    assert result["decode_steps_detail"][1]["kv_source"] == (
        "paged_cache_ready_fused_kv_projection_reused_for_explicit_k_and_v"
    )
    assert result["decode_steps_detail"][1]["true_separate_v_channel"] is False
    assert result["reference"]["kv_cache_ready"]["shape"] == [1, 1, 4, 8]
    assert result["reference"]["k_cache_unpaged"]["shape"] == [4, 1, 64, 8]
    assert result["reference"]["v_cache_unpaged"]["shape"] == [4, 1, 64, 8]
    assert result["reference"]["paged_k_cache"]["shape"] == [8, 1, 32, 8]
    assert result["reference"]["paged_v_cache"]["shape"] == [8, 1, 32, 8]
    assert result["reference"]["paged_attention_output_heads_rotary"]["shape"] == [1, 4, 4, 8]
    assert result["reference"]["attention_output_rope_unrotated"]["shape"] == [1, 4, 4, 4]
    assert result["reference"]["cur_pos_tensor"]["shape"] == [4]
    assert (
        "ttnn.transformer.paged_scaled_dot_product_attention_decode(q,k_cache,v_cache,page_table,cur_pos_tensor)"
        in result["traceable_decode_scope"]["inside_trace"]
    )
    assert (
        "ttnn.transformer.paged_scaled_dot_product_attention_decode(attention_sink)"
        in result["traceable_decode_scope"]["inside_trace"]
    )
    assert (
        "ttnn.experimental.rotary_embedding_llama(kv_update_rope)" in result["traceable_decode_scope"]["inside_trace"]
    )
    assert (
        "ttnn.experimental.rotary_embedding_llama(paged_sdpa_output_rope,inverse)"
        in result["traceable_decode_scope"]["inside_trace"]
    )


def test_cpu_traceable_decode_subpath_can_report_selected_rows_dense_attention(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = run_traceable_decode_subpath_smoke(
        snapshot,
        layer=3,
        seq_len=4,
        max_bytes=24 * 1024,
        routed_topk_prefix=1,
        cpu_only=True,
        attention_read_api=TRACEABLE_DECODE_ATTENTION_READ_SELECTED_ROWS_DENSE,
    )

    assert result["passed"] is True
    assert result["attention_read_api"] == TRACEABLE_DECODE_ATTENTION_READ_SELECTED_ROWS_DENSE
    assert result["sparse_attention"]["status"] == "real_indexer_rows_consumed_by_selected_rows_dense_attention"
    assert result["sparse_attention"]["sparse_indexer_status"] == (
        "real_indexer_topk_materialized_outside_trace_consumed_by_selected_rows_dense_attention"
    )
    assert result["sparse_attention"]["selected_cache_rows"]["derived_from_real_indexer"] is True
    assert result["sparse_attention"]["selected_cache_rows"]["selected_rows_consumed_by_attention"] is True
    assert result["sparse_attention"]["selected_cache_rows"]["selected_row_ids_source"] == "static_host_preflight"
    assert result["sparse_attention"]["selected_cache_rows"]["first_step_runtime_rows"] == [0, 1, 2, 3, 4, 8]
    assert result["sparse_attention"]["selected_cache_rows"]["compact_window_shape"] == [1, 1, 6, 8]
    assert result["sparse_attention"]["selected_cache_rows"]["per_step"][0]["rows_drive_attention_in_trace"] is True
    assert result["sparse_attention"]["selected_row_attention_blocker"] is None
    assert result["sparse_attention"]["selected_row_attention_proof"]["selected_rows_consumed_by_attention"] is True
    assert result["sparse_attention"]["attention_sink_status"] == "used_in_selected_rows_dense_attention"
    assert result["sparse_attention"]["attention_sink"]["in_trace"] is True
    assert result["sparse_attention"]["k_v_source"] == "selected_rows_from_projected_kv_cache_reuse"
    assert result["attention_path"]["selected_rows"]["consumed_by_attention"] is True
    assert result["attention_path"]["selected_rows"]["ids"] == [0, 1, 2, 3, 4, 8]
    assert result["attention_path"]["selected_rows"]["compact_window_shape"] == [1, 1, 6, 8]
    assert result["attention_path"]["softmax"]["selected_rows_dense_attention_in_trace"] is True
    assert result["attention_path"]["softmax"]["attention_sink_softmax_in_trace"] is True
    assert result["attention_path"]["qk_scores"]["shape"] == [1, 4, 4, 6]
    assert result["traceability_flags"]["selected_rows_consumed_by_attention"] is True
    assert result["traceability_flags"]["selected_row_compaction_in_trace"] is True
    assert result["traceability_flags"]["selected_row_ids_source"] == "static_host_preflight"
    assert result["traceability_flags"]["attention_sink_in_trace"] is True
    assert result["reference"]["selected_attention_cache_window"]["shape"] == [1, 1, 6, 8]
    assert result["reference"]["qk_scores"]["shape"] == [1, 4, 4, 6]
    assert result["reference"]["attention_scores_with_sink"]["shape"] == [1, 4, 4, 7]
    assert result["reference"]["attention_probs"]["shape"] == [1, 4, 4, 6]
    assert result["decode_steps_detail"][0]["selected_cache_rows"] == [0, 1, 2, 3, 4, 8]
    assert result["decode_steps_detail"][0]["selected_rows_consumed_by_attention"] is True
    assert result["decode_steps_detail"][0]["compact_window_shape"] == [1, 1, 6, 8]
    assert "ttnn.embedding(selected_row_idxs,kv_cache_table)" in result["traceable_decode_scope"]["inside_trace"]
    assert "ttnn.concat(selected_qk_scores,attention_sink_logits)" in result["traceable_decode_scope"]["inside_trace"]


def test_cpu_traceable_decode_subpath_can_report_selected_rows_compressed_kv_attention(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = run_traceable_decode_subpath_smoke(
        snapshot,
        layer=3,
        seq_len=4,
        max_bytes=64 * 1024,
        routed_topk_prefix=1,
        cpu_only=True,
        attention_read_api=TRACEABLE_DECODE_ATTENTION_READ_SELECTED_ROWS_COMPRESSED_KV,
    )

    assert result["passed"] is True
    assert result["attention_read_api"] == TRACEABLE_DECODE_ATTENTION_READ_SELECTED_ROWS_COMPRESSED_KV
    assert result["sparse_attention"]["status"] == (
        "real_indexer_rows_consumed_by_selected_rows_compressed_kv_attention"
    )
    assert result["sparse_attention"]["sparse_indexer_status"] == (
        "real_indexer_topk_materialized_outside_trace_consumed_by_selected_rows_compressed_kv_attention"
    )
    assert result["sparse_attention"]["attention_sink_status"] == ("used_in_selected_rows_compressed_kv_attention")
    assert result["sparse_attention"]["k_v_source"] == "selected_rows_from_compressed_kv_sparse_cache_reuse"
    assert result["sparse_attention"]["selected_cache_rows"]["selected_rows_consumed_by_attention"] is True
    assert result["sparse_attention"]["selected_cache_rows"]["selected_row_ids_source"] == "static_host_preflight"
    assert result["sparse_attention"]["selected_cache_rows"]["first_step_runtime_rows"] == [0, 1, 2, 3, 4, 8]
    assert result["sparse_attention"]["selected_cache_rows"]["compact_window_shape"] == [1, 1, 6, 8]
    assert result["sparse_attention"]["selected_row_attention_blocker"] is None
    assert result["sparse_attention"]["selected_row_attention_proof"]["kv_source"] == (
        "selected_rows_from_compressed_kv_sparse_cache"
    )
    assert result["sparse_attention"]["compressed_kv_cache"]["enabled"] is True
    assert result["sparse_attention"]["compressed_kv_cache"]["compress_ratio"] == 4
    assert result["sparse_attention"]["compressed_kv_cache"]["sparse_cache_shape"] == [1, 1, 10, 8]
    assert result["sparse_attention"]["compressed_kv_cache"]["compressed_cache_shape"] == [1, 1, 2, 8]
    assert result["sparse_attention"]["compressed_kv_cache"]["decode_update_rows"] == [9]
    assert result["traceability_flags"]["compressed_kv_cache_in_trace"] is True
    assert result["attention_path"]["selected_rows"]["consumed_by_attention"] is True
    assert result["attention_path"]["selected_rows"]["ids"] == [0, 1, 2, 3, 4, 8]
    assert result["attention_path"]["selected_rows"]["compact_window_shape"] == [1, 1, 6, 8]
    assert result["attention_path"]["kv_source"]["key_source"] == "static_selected_rows_compressed_kv_sparse_cache"
    assert result["attention_path"]["softmax"]["selected_rows_compressed_kv_attention_in_trace"] is True
    assert result["reference"]["selected_attention_cache_window"]["shape"] == [1, 1, 6, 8]
    assert result["reference"]["sparse_kv_cache"]["shape"] == [1, 1, 10, 8]
    assert result["reference"]["compressed_kv_cache"]["shape"] == [1, 1, 2, 8]
    assert result["decode_steps_detail"][0]["selected_cache_rows"] == [0, 1, 2, 3, 4, 8]
    assert result["decode_steps_detail"][0]["kv_source"] == (
        "selected_rows_compressed_kv_sparse_cache_reused_for_explicit_k_and_v"
    )
    assert result["decode_steps_detail"][0]["compressed_kv_status"] == (
        "selected_rows_use_real_compressor_seeded_sparse_cache_with_decode_sized_update"
    )
    assert "ttnn.embedding(selected_row_idxs,sparse_kv_cache_table)" in result["traceable_decode_scope"]["inside_trace"]
    assert "ttnn.update_cache(sparse_kv_cache_compressed_row)" in result["traceable_decode_scope"]["inside_trace"]


def test_cpu_traceable_decode_boundary_materializes_indexer_compressor_row(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = run_traceable_decode_subpath_smoke(
        snapshot,
        layer=3,
        seq_len=8,
        cache_update_index=11,
        max_bytes=96 * 1024,
        routed_topk_prefix=1,
        cpu_only=True,
        attention_read_api=TRACEABLE_DECODE_ATTENTION_READ_SELECTED_ROWS_COMPRESSED_KV,
        sparse_indexer_mode=TRACEABLE_DECODE_SPARSE_INDEXER_TRACE_TOPK_ATTENTION,
        compressor_mode=TRACEABLE_DECODE_COMPRESSOR_STATEFUL_RATIO4_PROBE,
        indexer_compressor_mode=TRACEABLE_DECODE_INDEXER_COMPRESSOR_STATEFUL_RATIO4_PROBE,
    )

    assert result["passed"] is True
    assert result["indexer_compressor_mode"] == TRACEABLE_DECODE_INDEXER_COMPRESSOR_STATEFUL_RATIO4_PROBE
    assert result["sparse_indexer_trace"]["indexer_kv_state_in_trace"] is True
    assert result["sparse_indexer_trace"]["indexer_score_state_in_trace"] is True
    assert result["sparse_indexer_trace"]["boundary_row_materialized_for_position"] is True
    assert result["sparse_indexer_trace"]["boundary_row_consumed_by_attention"] is True
    assert result["sparse_indexer_trace"]["indexer_boundary_write_status"] == (
        "static_boundary_position_writes_one_indexer_compressed_row"
    )
    assert result["sparse_indexer_trace"]["indexer_kv_cache_shape"] == [1, 1, 3, 8]
    assert result["attention_path"]["selected_rows"]["ids_source"] == (
        "static_sliding_window_rows_plus_device_indexer_compressed_rows"
    )
    assert result["attention_path"]["selected_rows"]["device_selected_row_ids_drive_compaction"] is True
    assert result["attention_path"]["selected_rows"]["consumed_by_attention"] is True
    assert result["sparse_attention"]["selected_cache_rows"]["per_step"][0]["compressed_cache_length"] == 3
    assert (
        result["sparse_attention"]["selected_cache_rows"]["per_step"][0]["real_indexer_selection"][
            "indexer_boundary_row_materialized"
        ]
        is True
    )
    assert "ttnn.update_cache(indexer_kv_cache_boundary_row)" in result["traceable_decode_scope"]["inside_trace"]


def test_cpu_traceable_decode_carried_cache_multistep_boundary_variants(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=5, num_routed_experts=4)

    result = run_traceable_decode_subpath_smoke(
        snapshot,
        layer=4,
        seq_len=32,
        cache_len=96,
        cache_update_index=32,
        decode_steps=8,
        max_bytes=192 * 1024,
        routed_topk_prefix=1,
        cpu_only=True,
        attention_read_api=TRACEABLE_DECODE_ATTENTION_READ_SELECTED_ROWS_COMPRESSED_KV,
        sparse_indexer_mode=TRACEABLE_DECODE_SPARSE_INDEXER_TRACE_TOPK_ATTENTION,
        compressor_mode=TRACEABLE_DECODE_COMPRESSOR_STATEFUL_RATIO4_PROBE,
        indexer_compressor_mode=TRACEABLE_DECODE_INDEXER_COMPRESSOR_STATEFUL_RATIO4_PROBE,
    )

    assert result["passed"] is True
    assert result["decode_step_count"] == 8
    assert result["positions"] == list(range(32, 40))
    assert result["multi_position_replay"]["cache_state_carried_on_device"] is True
    assert result["multi_position_replay"]["carried_device_sparse_kv_cache_state"] is True
    assert result["multi_position_replay"]["carried_device_compressor_state"] is True
    assert result["multi_position_replay"]["carried_device_indexer_compressor_state"] is True
    assert result["multi_position_replay"]["state_rebuilt_on_host_between_steps"] is False
    assert result["host_boundaries_inside_trace"] == []
    assert [item["is_compression_boundary"] for item in result["per_step_boundary_status"]] == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    assert [item["trace_variant_executed"] for item in result["per_step_trace_variants"]] == [
        "non_boundary_trace",
        "non_boundary_trace",
        "non_boundary_trace",
        "boundary_trace",
        "non_boundary_trace",
        "non_boundary_trace",
        "non_boundary_trace",
        "boundary_trace",
    ]
    assert result["decode_steps_detail"][3]["compressor_boundary_write"] is True
    assert result["decode_steps_detail"][7]["indexer_compressor_boundary_write"] is True
    assert "dynamic sparse-indexer boundary predicate across arbitrary positions" in result["remaining_limitations"]


def test_trace_topk_attention_reference_rows_follow_device_topk_order(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    config = DeepSeekV4FlashConfig.from_model_path(snapshot)
    sliding_window = int(config.sliding_window)

    rows = _device_selected_cache_rows_by_step_from_outputs(
        [
            {
                "sparse_indexer_topk_indices_uint32": torch.tensor(
                    [[[[2, 0, 1]]]],
                    dtype=torch.uint32,
                )
            }
        ],
        selected_cache_rows_by_step={0: (0, 1, 2, sliding_window, sliding_window + 1, sliding_window + 2)},
        config=config,
        attention_read_api=TRACEABLE_DECODE_ATTENTION_READ_SELECTED_ROWS_COMPRESSED_KV,
        sparse_indexer_mode=TRACEABLE_DECODE_SPARSE_INDEXER_TRACE_TOPK_ATTENTION,
        compressed_kv_cache_rows=3,
    )

    assert rows == {0: (0, 1, 2, sliding_window + 2, sliding_window, sliding_window + 1)}


def test_cpu_traceable_decode_subpath_can_report_legacy_attention_mode(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = run_traceable_decode_subpath_smoke(
        snapshot,
        layer=3,
        seq_len=4,
        max_bytes=20 * 1024,
        routed_topk_prefix=1,
        cpu_only=True,
        attention_mode=TRACEABLE_DECODE_ATTENTION_LEGACY_BLEND_MODE,
    )

    assert result["passed"] is True
    assert result["attention_path"]["mode"] == TRACEABLE_DECODE_ATTENTION_LEGACY_BLEND_MODE
    assert result["attention_path"]["softmax"]["qk_scores_in_trace"] is False
    assert result["attention_path"]["softmax"]["fixed_window_softmax_in_trace"] is False
    assert result["attention_path"]["cache_expand"]["repeat_axis"] == "attention_width"
    assert result["reference"]["expanded_attention_cache"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["attention_output"]["shape"] == [1, 1, 4, 32]
    assert (
        "QK scoring, softmax, and value reduction in legacy q+kv blend mode"
        in result["traceable_decode_scope"]["excluded_from_trace"]
    )


def test_cpu_traceable_decode_subpath_cli_outputs_json(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.real_traceable_decode_smoke",
            "--snapshot-dir",
            str(snapshot),
            "--layer",
            "3",
            "--seq-len",
            "4",
            "--max-bytes",
            str(24 * 1024),
            "--cpu-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["mode"] == "cpu-reference"
    assert payload["cache_update_api"] == TRACEABLE_DECODE_CACHE_UPDATE_HOST_SCALAR
    assert payload["update_index_source"] == "host_scalar"
    assert payload["trace_capture"]["attempted"] is False
    assert payload["trace_capture_attempted"] is False
    assert payload["trace_capture"]["ttnn_to_torch_guarded"] is True
    assert payload["guard_status"]["ttnn_to_torch_guarded"] is True
    assert payload["host_boundaries_inside_trace"] == []
    assert payload["traceable_decode_scope"]["not_full_forward"] is True
    assert payload["cache_update"]["update_index"] == 4
    assert payload["cache_update"]["cache_update_api"] == TRACEABLE_DECODE_CACHE_UPDATE_HOST_SCALAR
    assert payload["attention_path"]["mode"] == TRACEABLE_DECODE_ATTENTION_QK_SOFTMAX_MODE
    assert payload["attention_path"]["softmax"]["qk_scores_in_trace"] is True
    assert payload["attention_path"]["context"]["produced_in_trace"] is True
    assert payload["selected_routing"]["full_topk"] == 2
    assert payload["selected_routing"]["topk_prefix_limit"] == 2
    assert payload["selected_routing"]["topk_prefix_is_full"] is True
    assert payload["selected_routing"]["full_topk_mode"] is True
    assert payload["router_trace"]["mode"] == "device_gate_scoring_topk_route_weights_static_dispatch"
    assert payload["router_trace"]["gate_matmul_in_trace"] is True
    assert payload["router_trace"]["scoring_in_trace"] is True
    assert payload["router_trace"]["topk_in_trace"] is True
    assert payload["router_trace"]["route_weights_in_trace"] is True
    assert payload["router_trace"]["expert_dispatch"] == "static_preflight"
    assert payload["attention_path"]["host_provided_attention_output"] is False
    assert payload["attention_path"]["kv_source"]["kv_split_in_trace"] is True
    assert payload["attention_path"]["kv_source"]["true_kv_split_in_trace"] is False
    assert payload["attention_path"]["rope"]["rope_in_trace"] is True
    assert payload["traceability_flags"]["rope_in_trace"] is True
    assert payload["selected_routing"]["selected_expert_ids"] == [3, 2]
    assert payload["selected_routing"]["executed_expert_ids"] == [3, 2]
    assert payload["routed_expert_execution"]["loaded_expert_ids"] == [3, 2]
    assert payload["routed_expert_execution"]["executed_expert_ids"] == [3, 2]
    assert payload["routed_expert_execution"]["full_topk_executed"] is True
    assert payload["payload_bytes"]["attention_output"] == 5120
    assert payload["payload_bytes"]["routed_experts"] == 3840
    assert payload["payload_bytes"]["attention_sink"] == 16
    assert payload["payload_bytes"]["total"] == 18316
    assert payload["decoded_tensors"]["wo_a"]["shape"] == [64, 8]
    assert payload["decoded_tensors"]["routed_experts"]["3"]["w2"]["shape"] == [32, 32]
    assert payload["decoded_tensors"]["routed_experts"]["2"]["w2"]["shape"] == [32, 32]
    assert payload["reference"]["qk_scores"]["shape"] == [1, 4, 4, 4]
    assert payload["reference"]["attention_probs"]["shape"] == [1, 4, 4, 4]
    assert payload["reference"]["attention_output"]["shape"] == [1, 1, 4, 32]
    assert payload["reference"]["attention_projected"]["shape"] == [1, 1, 4, 32]
    assert payload["reference"]["router_logits"]["shape"] == [1, 1, 4, 4]
    assert payload["reference"]["router_route_weights"]["shape"] == [1, 1, 4, 2]


def test_traceable_decode_subpath_gated_galaxy_trace_replay() -> None:
    if os.environ.get("DSV4_FLASH_TRACEABLE_DECODE", "0") != "1":
        pytest.skip("Set DSV4_FLASH_TRACEABLE_DECODE=1 to run the Galaxy trace capture/replay smoke")

    snapshot = Path(os.environ.get("DSV4_FLASH_REAL_SNAPSHOT_DIR", str(REAL_SNAPSHOT_DIR)))
    if not snapshot.is_dir():
        pytest.fail(f"Real DeepSeek V4 Flash snapshot is missing: {snapshot}")

    decode_steps = int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_STEPS", "2"))
    result = run_traceable_decode_subpath_smoke(
        snapshot,
        layer=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_LAYER", "3")),
        seq_len=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_SEQ_LEN", "32")),
        cache_len=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_CACHE_LEN", str(96))),
        decode_steps=decode_steps,
        device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
        trace_region_size=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_TRACE_REGION_SIZE", str(64 * 1024 * 1024))),
        cache_update_api=TRACEABLE_DECODE_CACHE_UPDATE_HOST_SCALAR,
    )

    assert result["passed"], json.dumps(result["accuracy"], indent=2, sort_keys=True)
    assert result["mode"] == "ttnn-trace"
    assert result["trace_capture"]["attempted"] is True
    assert result["trace_capture"]["capture_passed"] is True
    assert result["trace_capture"]["execute_replay_passed"] is True
    assert result["decode_step_count"] == decode_steps
    assert len(result["positions_used"]) == result["decode_step_count"]
    assert result["trace_capture"]["capture_count"] == result["decode_step_count"]
    assert result["cache_update_api"] == TRACEABLE_DECODE_CACHE_UPDATE_HOST_SCALAR
    assert result["update_index_source"] == "host_scalar"
    assert result["trace_capture"]["single_capture_replayed_across_positions"] is False
    assert result["trace_capture"]["recaptured_per_position"] is (decode_steps > 1)
    assert result["trace_capture"]["cache_update_index_dynamic"] is False
    assert result["cache_update"]["dynamic_update_index_in_trace"] is False
    assert result["multi_position_replay"]["carried_device_kv_cache_state"] is True
    assert result["multi_position_replay"]["single_capture_replayed_across_positions"] is False
    assert result["multi_position_replay"]["recaptured_per_position"] is (decode_steps > 1)
    assert result["trace_capture_attempted"] is True
    assert result["trace_capture_passed"] is True
    assert result["trace_execute_replay_passed"] is True
    assert result["trace_capture"]["ttnn_to_torch_guarded"] is True
    assert result["guard_status"]["ttnn_to_torch_guarded"] is True
    assert result["host_boundaries_inside_trace"] == []
    assert result["cache_update"]["device_resident_inside_trace"] is True
    assert "ttnn.linear(grouped_wo_a_group_0..N)" in result["trace_capture"]["traced_operations"]
    assert "ttnn.experimental.rotary_embedding_llama(q_rope)" in result["trace_capture"]["traced_operations"]
    assert "ttnn.experimental.rotary_embedding_llama(k_rope)" in result["trace_capture"]["traced_operations"]
    assert "ttnn.matmul(q_heads,k_heads_transposed)" in result["trace_capture"]["traced_operations"]
    assert "ttnn.softmax(qk_scores)" in result["trace_capture"]["traced_operations"]
    assert "ttnn.matmul(attention_probs,value_heads)" in result["trace_capture"]["traced_operations"]
    assert "ttnn.linear(router_gate)" in result["trace_capture"]["traced_operations"]
    assert "ttnn.topk(router_selection_scores)" in result["trace_capture"]["traced_operations"]
    assert "ttnn.mul(router_route_weights,routed_scaling_factor)" in result["trace_capture"]["traced_operations"]
    assert "ttnn.linear(routed_w1_selected_topk_prefix)" in result["trace_capture"]["traced_operations"]
    assert result["selected_routing"]["topk_prefix_limit"] == result["selected_routing"]["full_topk"]
    assert result["selected_routing"]["topk_prefix_is_full"] is True
    assert result["routed_expert_execution"]["full_topk_executed"] is True
    assert result["selected_routing"]["route_weights_device_resident_inside_trace"] is True
    assert result["traceability_flags"]["router_gate_matmul_in_trace"] is True
    assert result["traceability_flags"]["router_scoring_in_trace"] is True
    assert result["traceability_flags"]["router_topk_in_trace"] is True
    assert result["traceability_flags"]["router_route_weights_in_trace"] is True
    assert result["traceability_flags"]["router_expert_dispatch_dynamic_in_trace"] is False
    assert (
        "DeepSeek sparse attention-sink/indexer semantics; fixed-window dense softmax uses contiguous cache rows"
        in result["traceable_decode_scope"]["excluded_from_trace"]
    )
    assert result["attention_path"]["host_provided_attention_output"] is False
    assert result["attention_path"]["mode"] == TRACEABLE_DECODE_ATTENTION_QK_SOFTMAX_MODE
    assert result["attention_path"]["kv_source"]["kv_split_in_trace"] is True
    assert result["attention_path"]["kv_source"]["true_kv_split_in_trace"] is False
    assert result["attention_path"]["rope"]["rope_in_trace"] is True
    assert result["attention_path"]["softmax"]["qk_scores_in_trace"] is True
    assert result["attention_path"]["softmax"]["fixed_window_softmax_in_trace"] is True
    assert result["attention_path"]["softmax"]["value_reduction_in_trace"] is True
    assert result["accuracy"]["attention_output"]["passed"] is True
    assert result["accuracy"]["attention_projected"]["passed"] is True
    assert result["accuracy"]["post_attention_residual"]["passed"] is True
    assert result["accuracy"]["router_decode_route_weights"]["passed"] is True
    assert result["accuracy"]["router_decode_topk_indices"]["required_for_pass"] is False
    assert result["router_trace"]["topk_indices_accuracy_required_for_pass"] is False
    assert result["accuracy"]["routed_output"]["passed"] is True
    assert result["accuracy"]["combined_ffn_output"]["passed"] is True
    assert result["accuracy"]["residual_output"]["passed"] is True
    assert result["accuracy"]["kv_cache"]["passed"] is True
    assert len(result["accuracy_by_step"]) == result["decode_step_count"]
    for step in result["accuracy_by_step"]:
        assert step["accuracy"]["attention_output"]["passed"] is True
        assert step["accuracy"]["attention_projected"]["passed"] is True
        assert step["accuracy"]["combined_ffn_output"]["passed"] is True
        assert step["accuracy"]["residual_output"]["passed"] is True
        assert step["accuracy"]["kv_cache"]["passed"] is True


def test_traceable_decode_subpath_gated_galaxy_paged_cache_write_single_capture_replay() -> None:
    if os.environ.get("DSV4_FLASH_TRACEABLE_DECODE", "0") != "1":
        pytest.skip("Set DSV4_FLASH_TRACEABLE_DECODE=1 to run the Galaxy paged cache-write replay smoke")

    snapshot = Path(os.environ.get("DSV4_FLASH_REAL_SNAPSHOT_DIR", str(REAL_SNAPSHOT_DIR)))
    if not snapshot.is_dir():
        pytest.fail(f"Real DeepSeek V4 Flash snapshot is missing: {snapshot}")

    decode_steps = int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_STEPS", "2"))
    result = run_traceable_decode_subpath_smoke(
        snapshot,
        layer=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_LAYER", "3")),
        seq_len=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_SEQ_LEN", "32")),
        cache_len=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_CACHE_LEN", str(96))),
        decode_steps=decode_steps,
        device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
        trace_region_size=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_TRACE_REGION_SIZE", str(64 * 1024 * 1024))),
        cache_update_api=TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR,
    )

    assert result["passed"], json.dumps(result["accuracy_by_step"], indent=2, sort_keys=True)
    assert result["cache_update_api"] == TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR
    assert result["update_index_source"] == "device_tensor"
    assert result["one_trace_capture_replayed_across_positions"] is (decode_steps > 1)
    assert result["trace_capture"]["capture_count"] == 1
    assert result["trace_capture"]["single_capture_replayed_across_positions"] is (decode_steps > 1)
    assert result["trace_capture"]["recaptured_per_position"] is False
    assert result["trace_capture"]["cache_update_index_dynamic"] is True
    assert result["cache_update"]["dynamic_update_index_in_trace"] is True
    assert result["cache_update"]["single_capture_replay_across_positions"] is (decode_steps > 1)
    assert result["cache_update"]["per_step_updated_rows"] == [[position] for position in result["positions_used"]]
    assert result["cache_read_window_dynamic"] is False
    assert result["rope_position_dynamic"] is False
    assert result["attention_path_by_step"][0]["cache_window"]["start"] == result["positions_used"][0]
    assert all(
        item["cache_window"]["start"] == result["positions_used"][0] for item in result["attention_path_by_step"]
    )
    assert all(item["rope"]["position_dynamic"] is False for item in result["attention_path_by_step"])
    assert "cache_update_index_host_to_device" in result["host_boundaries_outside_trace"]
    assert "trace_recapture_per_position" not in result["host_boundaries_outside_trace"]
    assert result["host_boundaries_inside_trace"] == []
    assert "ttnn.experimental.paged_update_cache(kv_projection_cache,update_idxs_tensor)" in result["ttnn_ops"]
    assert len(result["accuracy_by_step"]) == decode_steps
    for step in result["accuracy_by_step"]:
        assert step["accuracy"]["kv_cache"]["passed"] is True
        assert step["accuracy"]["attention_cache_window"]["passed"] is True
        assert step["accuracy"]["attention_output"]["passed"] is True
        assert step["accuracy"]["combined_ffn_output"]["passed"] is True
        assert step["accuracy"]["residual_output"]["passed"] is True


def test_traceable_decode_subpath_gated_galaxy_dynamic_rope_position_single_capture_replay() -> None:
    if os.environ.get("DSV4_FLASH_TRACEABLE_DECODE", "0") != "1":
        pytest.skip("Set DSV4_FLASH_TRACEABLE_DECODE=1 to run the Galaxy dynamic RoPE replay smoke")

    snapshot = Path(os.environ.get("DSV4_FLASH_REAL_SNAPSHOT_DIR", str(REAL_SNAPSHOT_DIR)))
    if not snapshot.is_dir():
        pytest.fail(f"Real DeepSeek V4 Flash snapshot is missing: {snapshot}")

    decode_steps = int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_STEPS", "2"))
    result = run_traceable_decode_subpath_smoke(
        snapshot,
        layer=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_LAYER", "3")),
        seq_len=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_SEQ_LEN", "32")),
        cache_len=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_CACHE_LEN", str(96))),
        decode_steps=decode_steps,
        device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
        trace_region_size=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_TRACE_REGION_SIZE", str(64 * 1024 * 1024))),
        cache_update_api=TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR,
        rope_position_api=TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR,
    )

    assert result["passed"], json.dumps(result["accuracy_by_step"], indent=2, sort_keys=True)
    assert result["one_trace_capture_replayed_across_positions"] is (decode_steps > 1)
    assert result["trace_capture"]["capture_count"] == 1
    assert result["cache_update"]["dynamic_update_index_in_trace"] is True
    assert result["cache_read_window_dynamic"] is False
    assert result["rope_position_api"] == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR
    assert result["rope_position_dynamic"] is True
    assert result["trace_capture"]["rope_position_dynamic"] is True
    assert result["trace_capture"]["rope_position_status"] == "replay_mutable_device_tensor_embedding"
    assert result["position_dependent_decode_inventory"]["dynamic_rope_position"]["status"] == "used"
    assert result["position_dependent_decode_inventory"]["dynamic_cache_read_current_position"]["status"] == (
        "available_not_integrated"
    )
    assert "rope_position_index_host_to_device" in result["host_boundaries_outside_trace"]
    assert "ttnn.embedding(rope_position_idxs,rope_cos_table)" in result["trace_capture"]["traced_operations"]
    assert "ttnn.embedding(rope_position_idxs,rope_sin_table)" in result["trace_capture"]["traced_operations"]
    assert len(result["accuracy_by_step"]) == decode_steps
    for step, position in zip(result["decode_steps_detail"], result["positions_used"]):
        assert step["cache_window_rows"][0] == result["positions_used"][0]
        assert step["rope_position_rows"][0] == position
        assert step["rope_position_dynamic"] is True
    for step in result["accuracy_by_step"]:
        assert step["accuracy"]["rope_cos"]["passed"] is True
        assert step["accuracy"]["rope_sin"]["passed"] is True
        assert step["accuracy"]["attention_output"]["passed"] is True
        assert step["accuracy"]["combined_ffn_output"]["passed"] is True
        assert step["accuracy"]["residual_output"]["passed"] is True


def test_traceable_decode_subpath_gated_galaxy_paged_sdpa_read_single_capture_replay() -> None:
    if os.environ.get("DSV4_FLASH_TRACEABLE_DECODE", "0") != "1":
        pytest.skip("Set DSV4_FLASH_TRACEABLE_DECODE=1 to run the Galaxy paged SDPA read replay smoke")

    snapshot = Path(os.environ.get("DSV4_FLASH_REAL_SNAPSHOT_DIR", str(REAL_SNAPSHOT_DIR)))
    if not snapshot.is_dir():
        pytest.fail(f"Real DeepSeek V4 Flash snapshot is missing: {snapshot}")

    decode_steps = int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_STEPS", "2"))
    result = run_traceable_decode_subpath_smoke(
        snapshot,
        layer=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_LAYER", "3")),
        seq_len=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_SEQ_LEN", "32")),
        cache_len=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_CACHE_LEN", str(96))),
        decode_steps=decode_steps,
        device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
        trace_region_size=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_TRACE_REGION_SIZE", str(64 * 1024 * 1024))),
        attention_read_api=TRACEABLE_DECODE_ATTENTION_READ_PAGED_SDPA_DECODE,
        cache_update_api=TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR,
        rope_position_api=TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR,
    )

    assert result["passed"], json.dumps(result["accuracy_by_step"], indent=2, sort_keys=True)
    assert result["attention_read_api"] == TRACEABLE_DECODE_ATTENTION_READ_PAGED_SDPA_DECODE
    assert result["one_trace_capture_replayed_across_positions"] is (decode_steps > 1)
    assert result["trace_capture"]["capture_count"] == 1
    assert result["trace_capture"]["cache_update_index_dynamic"] is True
    assert result["trace_capture"]["rope_position_dynamic"] is True
    assert result["trace_capture"]["dynamic_cache_read_current_position"] is True
    assert result["trace_capture"]["cur_pos_tensor_dynamic"] is True
    assert result["cache_read_window_dynamic"] is True
    assert result["dynamic_cache_read_current_position"] is True
    assert result["position_dependent_decode_inventory"]["dynamic_cache_read_current_position"]["status"] == "used"
    assert result["attention_path"]["kv_source"]["v_channel_kind"] == "fused_cache_ready_kv_reuse"
    assert result["attention_path"]["kv_source"]["true_separate_v_channel_in_trace"] is False
    assert result["attention_path"]["rope"]["kv_cache_ready_rope_rotation_before_cache_write"] is True
    assert result["attention_path"]["rope"]["attention_output_inverse_rope_in_trace"] is True
    assert result["traceability_flags"]["kv_cache_ready_rope_in_trace"] is True
    assert result["traceability_flags"]["attention_output_inverse_rope_in_trace"] is True
    assert "paged_sdpa_cur_pos_host_to_device" in result["host_boundaries_outside_trace"]
    assert "ttnn.slice(kv_cache_fixed_window)" not in result["trace_capture"]["traced_operations"]
    assert (
        "ttnn.transformer.paged_scaled_dot_product_attention_decode(q,k_cache,v_cache,page_table,cur_pos_tensor)"
        in result["trace_capture"]["traced_operations"]
    )
    assert len(result["cache_rows_pages_read_per_step"]) == decode_steps
    assert result["cache_rows_pages_read_per_step"][0]["rows"]["count"] == result["positions_used"][0] + 1
    assert len(result["accuracy_by_step"]) == decode_steps
    for step in result["accuracy_by_step"]:
        assert step["accuracy"]["paged_k_cache"]["passed"] is True
        assert step["accuracy"]["paged_v_cache"]["passed"] is True
        assert step["accuracy"]["attention_output"]["passed"] is True
        assert step["accuracy"]["combined_ffn_output"]["passed"] is True
        assert step["accuracy"]["residual_output"]["passed"] is True


def test_traceable_decode_subpath_gated_galaxy_selected_rows_dense_attention() -> None:
    if os.environ.get("DSV4_FLASH_TRACEABLE_DECODE", "0") != "1":
        pytest.skip("Set DSV4_FLASH_TRACEABLE_DECODE=1 to run the Galaxy selected-row attention smoke")

    snapshot = Path(os.environ.get("DSV4_FLASH_REAL_SNAPSHOT_DIR", str(REAL_SNAPSHOT_DIR)))
    if not snapshot.is_dir():
        pytest.fail(f"Real DeepSeek V4 Flash snapshot is missing: {snapshot}")

    result = run_traceable_decode_subpath_smoke(
        snapshot,
        layer=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_LAYER", "3")),
        seq_len=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_SEQ_LEN", "32")),
        cache_len=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_CACHE_LEN", str(96))),
        decode_steps=1,
        device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
        trace_region_size=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_TRACE_REGION_SIZE", str(64 * 1024 * 1024))),
        attention_read_api=TRACEABLE_DECODE_ATTENTION_READ_SELECTED_ROWS_DENSE,
        cache_update_api=TRACEABLE_DECODE_CACHE_UPDATE_HOST_SCALAR,
        rope_position_api=TRACEABLE_DECODE_ROPE_POSITION_STATIC,
    )

    assert result["passed"], json.dumps(result["accuracy_by_step"], indent=2, sort_keys=True)
    assert result["attention_read_api"] == TRACEABLE_DECODE_ATTENTION_READ_SELECTED_ROWS_DENSE
    assert result["sparse_attention"]["selected_cache_rows"]["selected_rows_consumed_by_attention"] is True
    assert result["trace_capture"]["selected_row_compaction_in_trace"] is True
    assert result["traceability_flags"]["selected_row_ids_source"] == "static_host_preflight"
    assert result["attention_path"]["softmax"]["selected_rows_dense_attention_in_trace"] is True
    assert "ttnn.embedding(selected_row_idxs,kv_cache_table)" in result["trace_capture"]["traced_operations"]
    assert result["accuracy"]["selected_attention_cache_window"]["passed"] is True
    assert result["accuracy"]["attention_output"]["passed"] is True
    assert result["accuracy"]["combined_ffn_output"]["passed"] is True
    assert result["accuracy"]["residual_output"]["passed"] is True


def test_paged_sdpa_decode_gated_galaxy_current_position_single_capture_replay() -> None:
    if os.environ.get("DSV4_FLASH_TRACEABLE_DECODE", "0") != "1":
        pytest.skip("Set DSV4_FLASH_TRACEABLE_DECODE=1 to run the paged SDPA decode trace replay proof")

    if os.environ.get("IRD_NUM_PCIE_CHIPS") == "32":
        os.environ.setdefault("TT_VISIBLE_DEVICES", "0")

    result = run_paged_sdpa_decode_trace_smoke(device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")))

    assert result["passed"], json.dumps(result["accuracy_by_step"], indent=2, sort_keys=True)
    assert result["attention_read_api"] == TRACEABLE_DECODE_ATTENTION_READ_PAGED_SDPA_DECODE
    assert result["attention_read_api_kind"] == "paged_sdpa_decode"
    assert result["one_trace_capture_replayed_across_positions"] is True
    assert result["trace_capture"]["capture_count"] == 1
    assert result["trace_capture"]["single_capture_replayed_across_positions"] is True
    assert result["trace_capture"]["cur_pos_tensor_dynamic"] is True
    assert result["trace_capture"]["q_input_dynamic"] is True
    assert result["trace_capture"]["cache_read_current_position_dynamic"] is True
    assert result["trace_capture"]["host_boundaries_inside_trace"] == []
    assert result["dynamic_cache_read_current_position"]["status"] == "proved"
    assert result["dynamic_cache_read_current_position"]["dynamic"] is True
    assert result["dynamic_cache_write_position"]["status"] == "not_in_scope"
    assert result["dynamic_rope_position"]["status"] == "not_in_scope"
    assert result["attention_path"]["simplified_dense_paged_attention_stepping_stone"] is True
    assert result["attention_path"]["true_deepseek_sparse_indexer_semantics"] is False
    assert result["attention_path"]["production_autoregressive_decode"] is False
    assert result["cache_rows_pages_read_per_step"][0]["rows"]["count"] == 32
    assert result["cache_rows_pages_read_per_step"][1]["rows"]["count"] == 96
    assert result["cache_rows_pages_read_per_step"][0]["pages"]["logical_pages"] == [0]
    assert result["cache_rows_pages_read_per_step"][1]["pages"]["logical_pages"] == [0, 1]
    assert result["output_difference"]["passed"] is True
    assert result["output_difference"]["max_abs"] > 1.0
    assert len(result["accuracy_by_step"]) == 2
    for step in result["accuracy_by_step"]:
        assert step["accuracy"]["attention_output"]["passed"] is True
