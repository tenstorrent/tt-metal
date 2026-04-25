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
        "TtAttentionProjection.project_output",
        "ttnn.slice(attention_output_group_0..N)",
        "ttnn.linear(grouped_wo_a_group_0..N)",
        "ttnn.concat(grouped_wo_a_rank)",
        "ttnn.linear(wo_b)",
        "ttnn.add(hidden,attention_projected)",
        "ttnn.rms_norm(ffn_norm)",
        "TtSharedExpertMLP",
        "ttnn.add(post_attention_residual,shared_output)",
    ]
    assert "router scoring/top-k/hash selection" in result["traceable_decode_scope"]["excluded_from_trace"]
    assert (
        "real sparse-attention output production; deterministic attention tensor is uploaded before trace"
        in result["traceable_decode_scope"]["excluded_from_trace"]
    )
    assert result["loaded_tensor_groups"]["attention_query"]["count"] == 4
    assert result["loaded_tensor_groups"]["attention_output"]["count"] == 2
    assert result["loaded_tensor_groups"]["attention_output"]["canonical_keys"] == [
        "layers.3.attn.wo_a.weight",
        "layers.3.attn.wo_b.weight",
    ]
    assert result["loaded_tensor_groups"]["kv_projection"]["count"] == 2
    assert result["loaded_tensor_groups"]["ffn_norm"]["canonical_keys"] == ["layers.3.ffn_norm.weight"]
    assert result["loaded_tensor_groups"]["shared_expert"]["count"] == 6
    assert result["payload_bytes"]["attention_output"] == 5120
    assert result["payload_bytes"]["kv_projection"] == 544
    assert result["payload_bytes"]["total"] == 14188
    assert result["decoded_tensors"]["wq_a"]["shape"] == [16, 32]
    assert result["decoded_tensors"]["wq_b"]["shape"] == [32, 16]
    assert result["decoded_tensors"]["wo_a"]["shape"] == [64, 8]
    assert result["decoded_tensors"]["wo_b"]["shape"] == [32, 64]
    assert result["decoded_tensors"]["wkv"]["shape"] == [8, 32]
    assert result["decoded_tensors"]["kv_norm"]["shape"] == [8]
    assert result["decoded_tensors"]["shared_w1"]["shape"] == [32, 32]
    assert result["inputs"]["capture_attention_output"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["kv_output"]["shape"] == [1, 1, 4, 8]
    assert result["reference"]["kv_cache"]["shape"] == [1, 1, 64, 8]
    assert result["reference"]["attention_projected"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["post_attention_residual"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["residual_output"]["shape"] == [1, 1, 4, 32]
    assert "attention_output_host_to_device" in result["host_boundaries_outside_trace"]
    assert result["accuracy"]["cpu_reference"]["passed"] is True


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
            str(16 * 1024),
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
    assert payload["payload_bytes"]["attention_output"] == 5120
    assert payload["payload_bytes"]["total"] == 14188
    assert payload["decoded_tensors"]["wo_a"]["shape"] == [64, 8]
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
    assert (
        "real sparse-attention output production; deterministic attention tensor is uploaded before trace"
        in result["traceable_decode_scope"]["excluded_from_trace"]
    )
    assert result["accuracy"]["attention_projected"]["passed"] is True
    assert result["accuracy"]["post_attention_residual"]["passed"] is True
    assert result["accuracy"]["kv_cache"]["passed"] is True
