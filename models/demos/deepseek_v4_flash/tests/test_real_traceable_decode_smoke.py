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
from models.demos.deepseek_v4_flash.real_traceable_decode_smoke import (
    TRACEABLE_DECODE_ATTENTION_LEGACY_BLEND_MODE,
    TRACEABLE_DECODE_ATTENTION_QK_SOFTMAX_MODE,
    TraceableDecodeHostFallbackError,
    TraceableDecodeHostGuard,
    run_traceable_decode_subpath_smoke,
)
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


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


def test_cpu_traceable_decode_subpath_reports_inventory_and_limitations(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = run_traceable_decode_subpath_smoke(
        snapshot,
        layer=3,
        seq_len=4,
        max_bytes=16 * 1024,
        routed_topk_prefix=1,
        cpu_only=True,
    )

    assert result["mode"] == "cpu-reference"
    assert result["passed"] is True
    assert result["trace_capture"]["attempted"] is False
    assert result["trace_capture_attempted"] is False
    assert result["trace_capture_passed"] is False
    assert result["trace_capture"]["ttnn_to_torch_guarded"] is True
    assert result["guard_status"]["ttnn_to_torch_guarded"] is True
    assert result["host_boundaries_inside_trace"] == []
    assert result["cache_update"]["name"] == "compressed_kv_projection_cache_append"
    assert result["cache_update"]["cache_len"] == 64
    assert result["cache_update"]["update_index"] == 4
    assert result["cache_update"]["device_resident_inside_trace"] is True
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
        "ttnn.repeat(kv_cache_window_to_attention_heads)",
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
        "TtRoutedExpertMLP(selected_topk_prefix)",
        "ttnn.mul(routed_hidden,preselected_route_weight)",
        "ttnn.add(routed_expert_outputs)",
        "TtSharedExpertMLP",
        "ttnn.add(shared_output,routed_output)",
        "ttnn.add(post_attention_residual,combined_ffn_output)",
    ]
    assert (
        "router scoring/top-k/hash selection; selected expert ids and route weights are precomputed on host"
        in result["traceable_decode_scope"]["excluded_from_trace"]
    )
    assert result["selected_routing"]["selection_boundary"] == "host_pretrace_router_topk"
    assert result["selected_routing"]["topk_prefix_limit"] == 1
    assert result["selected_routing"]["topk_prefix_is_full"] is False
    assert result["selected_routing"]["full_topk_mode"] is False
    assert result["selected_routing"]["selected_expert_ids"] == [2]
    assert result["selected_routing"]["executed_expert_ids"] == [2]
    assert result["selected_routing"]["route_weights_device_resident_inside_trace"] is True
    assert result["routed_expert_execution"]["loaded_expert_ids"] == [2]
    assert result["routed_expert_execution"]["executed_expert_ids"] == [2]
    assert result["routed_expert_execution"]["full_topk_executed"] is False
    assert (
        "DeepSeek sparse attention-sink/indexer semantics; fixed-window dense softmax uses contiguous cache rows"
        in result["traceable_decode_scope"]["excluded_from_trace"]
    )
    assert result["attention_path"]["mode"] == TRACEABLE_DECODE_ATTENTION_QK_SOFTMAX_MODE
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
    assert result["attention_path"]["kv_source"]["true_kv_split_in_trace"] is False
    assert result["attention_path"]["rope"]["q_rope_split_in_trace"] is False
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
    assert result["loaded_tensor_groups"]["ffn_norm"]["canonical_keys"] == ["layers.3.ffn_norm.weight"]
    assert result["loaded_tensor_groups"]["router_selector"]["count"] == 2
    assert result["loaded_tensor_groups"]["shared_expert"]["count"] == 6
    assert result["loaded_tensor_groups"]["routed_experts"]["count"] == 6
    assert result["payload_bytes"]["attention_output"] == 5120
    assert result["payload_bytes"]["kv_projection"] == 544
    assert result["payload_bytes"]["router_selector"] == 272
    assert result["payload_bytes"]["routed_experts"] == 1920
    assert result["payload_bytes"]["total"] == 16380
    assert result["decoded_tensors"]["wq_a"]["shape"] == [16, 32]
    assert result["decoded_tensors"]["wq_b"]["shape"] == [32, 16]
    assert result["decoded_tensors"]["wo_a"]["shape"] == [64, 8]
    assert result["decoded_tensors"]["wo_b"]["shape"] == [32, 64]
    assert result["decoded_tensors"]["wkv"]["shape"] == [8, 32]
    assert result["decoded_tensors"]["kv_norm"]["shape"] == [8]
    assert result["decoded_tensors"]["router_gate"]["shape"] == [4, 32]
    assert result["decoded_tensors"]["routed_experts"]["2"]["w1"]["shape"] == [32, 32]
    assert result["decoded_tensors"]["shared_w1"]["shape"] == [32, 32]
    assert result["inputs"]["kv_cache_initial"]["shape"] == [1, 1, 64, 8]
    assert result["reference"]["kv_output"]["shape"] == [1, 1, 4, 8]
    assert result["reference"]["kv_cache"]["shape"] == [1, 1, 64, 8]
    assert result["reference"]["attention_cache_window"]["shape"] == [1, 1, 4, 8]
    assert result["reference"]["attention_q_heads"]["shape"] == [1, 4, 4, 8]
    assert result["reference"]["attention_key_heads"]["shape"] == [1, 4, 4, 8]
    assert result["reference"]["attention_value_heads"]["shape"] == [1, 4, 4, 8]
    assert result["reference"]["qk_scores"]["shape"] == [1, 4, 4, 4]
    assert result["reference"]["attention_probs"]["shape"] == [1, 4, 4, 4]
    assert result["reference"]["attention_context_heads"]["shape"] == [1, 4, 4, 8]
    assert result["reference"]["attention_output"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["attention_projected"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["post_attention_residual"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["routed_output"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["combined_ffn_output"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["residual_output"]["shape"] == [1, 1, 4, 32]
    assert "router_topk_pretrace" in result["host_boundaries_outside_trace"]
    assert "route_weight_host_to_device" in result["host_boundaries_outside_trace"]
    assert "kv_cache_seed_host_to_device" in result["host_boundaries_outside_trace"]
    assert "attention_output_host_to_device" not in result["host_boundaries_outside_trace"]
    assert result["accuracy"]["cpu_reference"]["passed"] is True


def test_cpu_traceable_decode_subpath_can_report_legacy_attention_mode(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = run_traceable_decode_subpath_smoke(
        snapshot,
        layer=3,
        seq_len=4,
        max_bytes=16 * 1024,
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
    assert payload["trace_capture"]["attempted"] is False
    assert payload["trace_capture_attempted"] is False
    assert payload["trace_capture"]["ttnn_to_torch_guarded"] is True
    assert payload["guard_status"]["ttnn_to_torch_guarded"] is True
    assert payload["host_boundaries_inside_trace"] == []
    assert payload["traceable_decode_scope"]["not_full_forward"] is True
    assert payload["cache_update"]["update_index"] == 4
    assert payload["attention_path"]["mode"] == TRACEABLE_DECODE_ATTENTION_QK_SOFTMAX_MODE
    assert payload["attention_path"]["softmax"]["qk_scores_in_trace"] is True
    assert payload["attention_path"]["context"]["produced_in_trace"] is True
    assert payload["selected_routing"]["full_topk"] == 2
    assert payload["selected_routing"]["topk_prefix_limit"] == 2
    assert payload["selected_routing"]["topk_prefix_is_full"] is True
    assert payload["selected_routing"]["full_topk_mode"] is True
    assert payload["attention_path"]["host_provided_attention_output"] is False
    assert payload["selected_routing"]["selected_expert_ids"] == [2, 3]
    assert payload["selected_routing"]["executed_expert_ids"] == [2, 3]
    assert payload["routed_expert_execution"]["loaded_expert_ids"] == [2, 3]
    assert payload["routed_expert_execution"]["executed_expert_ids"] == [2, 3]
    assert payload["routed_expert_execution"]["full_topk_executed"] is True
    assert payload["payload_bytes"]["attention_output"] == 5120
    assert payload["payload_bytes"]["routed_experts"] == 3840
    assert payload["payload_bytes"]["total"] == 18300
    assert payload["decoded_tensors"]["wo_a"]["shape"] == [64, 8]
    assert payload["decoded_tensors"]["routed_experts"]["3"]["w2"]["shape"] == [32, 32]
    assert payload["decoded_tensors"]["routed_experts"]["2"]["w2"]["shape"] == [32, 32]
    assert payload["reference"]["qk_scores"]["shape"] == [1, 4, 4, 4]
    assert payload["reference"]["attention_probs"]["shape"] == [1, 4, 4, 4]
    assert payload["reference"]["attention_output"]["shape"] == [1, 1, 4, 32]
    assert payload["reference"]["attention_projected"]["shape"] == [1, 1, 4, 32]


def test_traceable_decode_subpath_gated_galaxy_trace_replay() -> None:
    if os.environ.get("DSV4_FLASH_TRACEABLE_DECODE", "0") != "1":
        pytest.skip("Set DSV4_FLASH_TRACEABLE_DECODE=1 to run the Galaxy trace capture/replay smoke")

    snapshot = Path(os.environ.get("DSV4_FLASH_REAL_SNAPSHOT_DIR", str(REAL_SNAPSHOT_DIR)))
    if not snapshot.is_dir():
        pytest.fail(f"Real DeepSeek V4 Flash snapshot is missing: {snapshot}")

    result = run_traceable_decode_subpath_smoke(
        snapshot,
        layer=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_LAYER", "3")),
        seq_len=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_SEQ_LEN", "32")),
        device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
        trace_region_size=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_TRACE_REGION_SIZE", str(64 * 1024 * 1024))),
    )

    assert result["passed"], json.dumps(result["accuracy"], indent=2, sort_keys=True)
    assert result["mode"] == "ttnn-trace"
    assert result["trace_capture"]["attempted"] is True
    assert result["trace_capture"]["capture_passed"] is True
    assert result["trace_capture"]["execute_replay_passed"] is True
    assert result["trace_capture_attempted"] is True
    assert result["trace_capture_passed"] is True
    assert result["trace_execute_replay_passed"] is True
    assert result["trace_capture"]["ttnn_to_torch_guarded"] is True
    assert result["guard_status"]["ttnn_to_torch_guarded"] is True
    assert result["host_boundaries_inside_trace"] == []
    assert result["cache_update"]["device_resident_inside_trace"] is True
    assert "ttnn.linear(grouped_wo_a_group_0..N)" in result["trace_capture"]["traced_operations"]
    assert "ttnn.matmul(q_heads,k_heads_transposed)" in result["trace_capture"]["traced_operations"]
    assert "ttnn.softmax(qk_scores)" in result["trace_capture"]["traced_operations"]
    assert "ttnn.matmul(attention_probs,value_heads)" in result["trace_capture"]["traced_operations"]
    assert "ttnn.linear(routed_w1_selected_topk_prefix)" in result["trace_capture"]["traced_operations"]
    assert result["selected_routing"]["topk_prefix_limit"] == result["selected_routing"]["full_topk"]
    assert result["selected_routing"]["topk_prefix_is_full"] is True
    assert result["routed_expert_execution"]["full_topk_executed"] is True
    assert result["selected_routing"]["route_weights_device_resident_inside_trace"] is True
    assert (
        "DeepSeek sparse attention-sink/indexer semantics; fixed-window dense softmax uses contiguous cache rows"
        in result["traceable_decode_scope"]["excluded_from_trace"]
    )
    assert result["attention_path"]["host_provided_attention_output"] is False
    assert result["attention_path"]["mode"] == TRACEABLE_DECODE_ATTENTION_QK_SOFTMAX_MODE
    assert result["attention_path"]["softmax"]["qk_scores_in_trace"] is True
    assert result["attention_path"]["softmax"]["fixed_window_softmax_in_trace"] is True
    assert result["attention_path"]["softmax"]["value_reduction_in_trace"] is True
    assert result["accuracy"]["attention_output"]["passed"] is True
    assert result["accuracy"]["attention_projected"]["passed"] is True
    assert result["accuracy"]["post_attention_residual"]["passed"] is True
    assert result["accuracy"]["routed_output"]["passed"] is True
    assert result["accuracy"]["combined_ffn_output"]["passed"] is True
    assert result["accuracy"]["residual_output"]["passed"] is True
    assert result["accuracy"]["kv_cache"]["passed"] is True
