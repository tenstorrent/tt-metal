# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.real_checkpoint_loader import RealCheckpointTensorIndex
from models.demos.deepseek_v4_flash.real_prefill_attention_runtime_smoke import (
    layer_prefill_attention_runtime_keys,
    run_real_prefill_attention_runtime_smoke,
)
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def test_layer_prefill_attention_runtime_selector_loads_layer3_runtime_slice_without_indexer(
    tmp_path: Path,
) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )
    config = DeepSeekV4FlashConfig.from_model_path(snapshot)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot)

    keys = layer_prefill_attention_runtime_keys(index, config=config, layer=3)

    assert keys == [
        "layers.3.attn_norm.weight",
        "layers.3.attn.q_norm.weight",
        "layers.3.attn.wq_a.weight",
        "layers.3.attn.wq_b.weight",
        "layers.3.attn.kv_norm.weight",
        "layers.3.attn.wkv.weight",
        "layers.3.attn.attn_sink",
        "layers.3.attn.compressor.ape",
        "layers.3.attn.compressor.wkv.weight",
        "layers.3.attn.compressor.wgate.weight",
        "layers.3.attn.compressor.norm.weight",
        "layers.3.attn.wo_a.weight",
        "layers.3.attn.wo_b.weight",
    ]
    assert all(".attn.indexer." not in key for key in keys)


def test_cpu_real_prefill_attention_runtime_smoke_selects_references_and_reports_boundaries(
    tmp_path: Path,
) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )

    result = run_real_prefill_attention_runtime_smoke(
        snapshot,
        layer=3,
        seq_len=4,
        max_bytes=64 * 1024,
        cpu_only=True,
    )

    loaded_keys = [item["canonical_key"] for item in result["loaded_tensors"]]
    assert loaded_keys == [
        "layers.3.attn_norm.weight",
        "layers.3.attn.q_norm.weight",
        "layers.3.attn.wq_a.weight",
        "layers.3.attn.wq_b.weight",
        "layers.3.attn.kv_norm.weight",
        "layers.3.attn.wkv.weight",
        "layers.3.attn.attn_sink",
        "layers.3.attn.compressor.ape",
        "layers.3.attn.compressor.wkv.weight",
        "layers.3.attn.compressor.wgate.weight",
        "layers.3.attn.compressor.norm.weight",
        "layers.3.attn.wo_a.weight",
        "layers.3.attn.wo_b.weight",
    ]
    assert result["mode"] == "cpu-reference"
    assert result["model"]["compress_ratio"] == 128
    assert result["output_shapes"]["q_prefill"] == [1, 4, 4, 8]
    assert result["output_shapes"]["kv_cache_ready"] == [1, 4, 8]
    assert result["output_shapes"]["window_topk_idxs"] == [1, 4, 4]
    assert result["output_shapes"]["compress_topk_idxs"] == [1, 4, 0]
    assert result["output_shapes"]["runtime_topk_idxs"] == [1, 4, 4]
    assert result["output_shapes"]["attention_output_rotary"] == [1, 4, 4, 8]
    assert result["output_shapes"]["attention_output_projected"] == [1, 1, 4, 32]
    assert result["sparse_attention_inputs"]["compressor_tensors_loaded"] is True
    assert result["sparse_attention_inputs"]["compressor_executed"] is False
    assert result["sparse_attention_inputs"]["indexer_expected_for_layer"] is False
    assert result["sparse_attention_inputs"]["indexer_tensors_loaded"] is False
    assert {boundary["name"] for boundary in result["host_boundaries"]} >= {
        "cache_prep_readback",
        "topk_host",
        "sparse_attention_host_fallback",
        "inverse_rope_host",
        "grouped_wo_a_host",
    }
    assert result["payload_bytes"]["output_projection"] == 5120
    assert result["ttnn_ops"] == []
    assert result["passed"] is True


def test_cpu_real_prefill_attention_runtime_smoke_refuses_budget_overruns(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )

    with pytest.raises(ValueError, match="tensor budget"):
        run_real_prefill_attention_runtime_smoke(
            snapshot,
            layer=3,
            seq_len=4,
            max_tensors=12,
            max_bytes=64 * 1024,
            cpu_only=True,
        )
    with pytest.raises(ValueError, match="byte budget"):
        run_real_prefill_attention_runtime_smoke(
            snapshot,
            layer=3,
            seq_len=4,
            max_tensors=13,
            max_bytes=20239,
            cpu_only=True,
        )


def test_cpu_real_prefill_attention_runtime_smoke_cli_outputs_json(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.real_prefill_attention_runtime_smoke",
            "--snapshot-dir",
            str(snapshot),
            "--layer",
            "3",
            "--seq-len",
            "4",
            "--max-bytes",
            str(64 * 1024),
            "--cpu-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["mode"] == "cpu-reference"
    assert payload["layer"] == 3
    assert payload["payload_bytes"]["total"] == 20240
    assert payload["runtime_scope"]["path"] == (
        "attn_norm -> real Q/KV cache prep -> sliding-window sparse attention -> "
        "inverse RoPE -> grouped wo_a -> wo_b"
    )
    assert payload["output_shapes"]["attention_output_projected"] == [1, 1, 4, 32]
    assert payload["host_boundaries"][-2]["name"] == "grouped_wo_a_host"


def test_real_prefill_attention_runtime_smoke_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_REAL_PREFILL_ATTENTION_RUNTIME_SMOKE", "0") == "1"
    snapshot = Path(os.environ.get("DSV4_FLASH_REAL_SNAPSHOT_DIR", str(REAL_SNAPSHOT_DIR)))
    if not snapshot.is_dir():
        if required:
            pytest.fail(f"Real DeepSeek V4 Flash snapshot is missing: {snapshot}")
        pytest.skip(f"Real DeepSeek V4 Flash snapshot is missing: {snapshot}")

    available, reason = _available_ttnn_devices()
    if available < 1:
        if required:
            pytest.fail(reason)
        pytest.skip(reason)

    result = run_real_prefill_attention_runtime_smoke(
        snapshot,
        layer=int(os.environ.get("DSV4_FLASH_REAL_PREFILL_ATTENTION_RUNTIME_LAYER", "3")),
        seq_len=32,
        device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
    )

    assert result["passed"], json.dumps(result["accuracy"], indent=2, sort_keys=True)


def _available_ttnn_devices() -> tuple[int, str]:
    try:
        import ttnn
    except Exception as exc:
        return 0, f"Unable to import ttnn: {exc}"

    try:
        return int(ttnn.GetNumAvailableDevices()), "No TTNN devices available"
    except Exception as exc:
        return 0, f"Unable to query TTNN devices: {exc}"
