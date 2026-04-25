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
import torch

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.real_checkpoint_loader import RealCheckpointTensorIndex
from models.demos.deepseek_v4_flash.real_prefill_cache_prep_smoke import (
    apply_deepseek_v4_rotary,
    build_prefill_cache_prep_from_projected,
    layer_prefill_cache_prep_keys,
    precompute_deepseek_v4_rope_frequencies,
    run_real_prefill_cache_prep_smoke,
)
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def test_layer_prefill_cache_prep_selector_reuses_query_and_kv_projection_paths(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )
    index = RealCheckpointTensorIndex.from_snapshot(snapshot)

    keys = layer_prefill_cache_prep_keys(index, layer=3)

    assert keys == [
        "layers.3.attn_norm.weight",
        "layers.3.attn.q_norm.weight",
        "layers.3.attn.wq_a.weight",
        "layers.3.attn.wq_b.weight",
        "layers.3.attn.kv_norm.weight",
        "layers.3.attn.wkv.weight",
    ]


def test_cpu_real_prefill_cache_prep_smoke_selects_decodes_and_references_one_layer(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )

    result = run_real_prefill_cache_prep_smoke(snapshot, layer=3, seq_len=4, max_bytes=4096, cpu_only=True)

    loaded_keys = [item["canonical_key"] for item in result["loaded_tensors"]]
    assert loaded_keys == [
        "layers.3.attn_norm.weight",
        "layers.3.attn.q_norm.weight",
        "layers.3.attn.wq_a.weight",
        "layers.3.attn.wq_b.weight",
        "layers.3.attn.kv_norm.weight",
        "layers.3.attn.wkv.weight",
    ]
    assert result["selected_source_keys"] == loaded_keys
    assert result["mode"] == "cpu-reference"
    assert result["payload_bytes"] == {
        "attn_norm": 128,
        "q_norm": 64,
        "kv_norm": 32,
        "wq_a_weight": 1024,
        "wq_a_scale": 0,
        "wq_b_weight": 1024,
        "wq_b_scale": 0,
        "wkv_weight": 512,
        "wkv_scale": 0,
        "norms": 224,
        "q_low_rank": 1024,
        "q_output": 1024,
        "kv_projection": 512,
        "weights": 2560,
        "scales": 0,
        "total": 2784,
    }
    assert result["model"]["compress_ratio"] == 128
    assert result["model"]["kv_nope_head_dim"] == 4
    assert result["model"]["qk_rope_head_dim"] == 4
    assert result["rope"]["applied"] is True
    assert result["rope"]["base"] == 160000.0
    assert result["output_shapes"]["q_output"] == [1, 1, 4, 32]
    assert result["output_shapes"]["q_prefill"] == [1, 4, 4, 8]
    assert result["output_shapes"]["kv_cache_ready"] == [1, 4, 8]
    assert result["output_shapes"]["window_topk_idxs"] == [1, 4, 4]
    assert result["cache_prep"]["split_boundary"] == {
        "kv_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
    }
    assert (
        result["cache_prep"]["compressed_cache"]
        == "not materialized in this slice; seq_len is below layer compress_ratio"
    )
    assert {boundary["name"] for boundary in result["host_boundaries"]} >= {
        "projection_fp8_decode_to_bf16",
        "activation_host_to_device",
        "projection_and_split_readback",
        "rope_cache_prep_host",
    }
    assert result["ttnn_ops"] == []
    assert result["passed"] is True


def test_cpu_real_prefill_cache_prep_smoke_refuses_budget_overruns(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )

    with pytest.raises(ValueError, match="tensor budget"):
        run_real_prefill_cache_prep_smoke(
            snapshot,
            layer=3,
            seq_len=4,
            max_tensors=5,
            max_bytes=4096,
            cpu_only=True,
        )
    with pytest.raises(ValueError, match="byte budget"):
        run_real_prefill_cache_prep_smoke(
            snapshot,
            layer=3,
            seq_len=4,
            max_tensors=6,
            max_bytes=2783,
            cpu_only=True,
        )


def test_prefill_cache_prep_rope_matches_manual_complex_rotation(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )
    config = DeepSeekV4FlashConfig.from_model_path(snapshot)
    q_output = torch.arange(1 * 1 * 4 * 32, dtype=torch.float32).reshape(1, 1, 4, 32).to(torch.bfloat16)
    kv_output = torch.arange(1 * 1 * 4 * 8, dtype=torch.float32).reshape(1, 1, 4, 8).to(torch.bfloat16)

    result = build_prefill_cache_prep_from_projected(q_output, kv_output, config=config, layer=3)

    freqs = precompute_deepseek_v4_rope_frequencies(config, layer=3, seq_len=4)
    manual_kv_nope, manual_kv_rope = kv_output[:, 0].split([4, 4], dim=-1)
    manual_kv = torch.cat([manual_kv_nope, apply_deepseek_v4_rotary(manual_kv_rope.contiguous(), freqs)], dim=-1)
    torch.testing.assert_close(result["kv_cache_ready"], manual_kv)
    assert result["window_topk_idxs"].tolist() == [[[0, -1, -1, -1], [0, 1, -1, -1], [0, 1, 2, -1], [0, 1, 2, 3]]]


def test_cpu_real_prefill_cache_prep_smoke_cli_outputs_json(tmp_path: Path) -> None:
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
            "models.demos.deepseek_v4_flash.real_prefill_cache_prep_smoke",
            "--snapshot-dir",
            str(snapshot),
            "--layer",
            "3",
            "--seq-len",
            "4",
            "--max-bytes",
            "4096",
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
    assert payload["payload_bytes"]["total"] == 2784
    assert payload["projection_scope"]["path"] == (
        "attention_norm -> real Q projection + real K/V projection -> q/kv reshape -> split -> RoPE"
    )
    assert payload["output_shapes"]["q_prefill"] == [1, 4, 4, 8]
    assert payload["output_shapes"]["kv_cache_ready"] == [1, 4, 8]
    assert payload["host_boundaries"][-1]["name"] == "rope_cache_prep_host"


def test_real_prefill_cache_prep_smoke_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_REAL_PREFILL_CACHE_PREP_SMOKE", "0") == "1"
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

    result = run_real_prefill_cache_prep_smoke(
        snapshot,
        layer=int(os.environ.get("DSV4_FLASH_REAL_PREFILL_CACHE_PREP_LAYER", "3")),
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
