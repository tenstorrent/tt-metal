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

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.real_checkpoint_loader import RealCheckpointTensorIndex
from models.demos.deepseek_v4_flash.real_decode_logits_smoke import (
    layer_decode_logits_keys,
    load_decode_logits_weights,
    run_real_decode_logits_smoke,
)
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint
from models.demos.deepseek_v4_flash.tests.test_real_decode_decoder_layer_smoke import _available_ttnn_devices

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def test_layer_decode_logits_selector_loads_decoder_attention_final_norm_and_head(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )
    config = DeepSeekV4FlashConfig.from_model_path(snapshot)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot)

    keys = layer_decode_logits_keys(index, config=config, layer=3)
    weights = load_decode_logits_weights(
        index,
        config=config,
        vocab_mode="slice",
        vocab_start=8,
        vocab_size=16,
        already_loaded_metadata=[],
        max_tensors=2,
        max_bytes=4096,
    )

    assert keys[:13] == [
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
    assert keys[-2:] == ["norm.weight", "head.weight"]
    assert weights.vocab_mode == "slice"
    assert weights.vocab_start == 8
    assert weights.head_weight.shape == (16, 32)
    assert [item.canonical_key for item in weights.metadata] == ["norm.weight", "head.weight"]
    assert [item.nbytes for item in weights.metadata] == [128, 1024]


def test_cpu_real_decode_logits_smoke_projects_decoder_hidden_to_logits(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )

    result = run_real_decode_logits_smoke(
        snapshot,
        layer=3,
        prefill_seq_len=4,
        top_k=3,
        max_bytes=128 * 1024,
        cpu_only=True,
    )

    assert result["mode"] == "cpu-reference"
    assert result["passed"] is True
    assert result["vocab"] == {
        "mode": "full",
        "is_sliced": False,
        "vocab_start": 0,
        "vocab_size": 64,
        "full_vocab_size": 64,
        "deterministic_slice": None,
    }
    assert result["output_shapes"]["post_ffn_residual"] == [1, 1, 1, 32]
    assert result["output_shapes"]["final_norm"] == [1, 1, 1, 32]
    assert result["output_shapes"]["logits"] == [1, 1, 1, 64]
    assert result["final_norm_lm_head"]["loaded_keys"] == {
        "final_norm": "norm.weight",
        "lm_head": "head.weight",
    }
    assert result["final_norm_lm_head"]["payload_bytes"] == {
        "final_norm": 128,
        "lm_head": 4096,
        "total": 4224,
    }
    assert result["payload_bytes"]["total"] == 32940
    assert result["reference"]["final_norm"]["shape"] == [1, 1, 1, 32]
    assert result["reference"]["logits"]["shape"] == [1, 1, 1, 64]
    assert len(result["reference"]["top_k"]) == 3
    assert "ttnn.rms_norm(final_norm)" not in result["ttnn_ops"]
    assert {boundary["name"] for boundary in result["host_boundaries"]} >= {
        "decoder_layer_output_readback",
        "logits_readback",
    }
    assert result["accuracy"]["cpu_reference"]["passed"] is True


def test_cpu_real_decode_logits_smoke_refuses_budget_overruns(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )

    with pytest.raises(ValueError, match="byte budget"):
        run_real_decode_logits_smoke(
            snapshot,
            layer=3,
            prefill_seq_len=4,
            max_bytes=32939,
            cpu_only=True,
        )


def test_cpu_real_decode_logits_smoke_cli_outputs_json(tmp_path: Path) -> None:
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
            "models.demos.deepseek_v4_flash.real_decode_logits_smoke",
            "--snapshot-dir",
            str(snapshot),
            "--layer",
            "3",
            "--prefill-seq-len",
            "4",
            "--vocab-mode",
            "slice",
            "--vocab-start",
            "8",
            "--vocab-size",
            "16",
            "--top-k",
            "2",
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
    assert payload["vocab"]["mode"] == "slice"
    assert payload["vocab"]["deterministic_slice"] == "[8, 24)"
    assert payload["output_shapes"]["logits"] == [1, 1, 1, 16]
    assert payload["final_norm_lm_head"]["lm_head_shape_loaded"] == [16, 32]
    assert payload["payload_bytes"]["final_norm_lm_head"]["total"] == 1152
    assert payload["reference"]["top_k"][0]["id"] >= 8
    assert payload["reference"]["top_k"][0]["id"] < 24
    assert payload["ttnn_ops"] == []
    assert payload["host_boundaries"][-1]["name"] == "lm_head_vocab_slice"
    assert payload["accuracy"]["cpu_reference"]["passed"] is True


def test_real_decode_logits_smoke_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_REAL_DECODE_LOGITS_SMOKE", "0") == "1"
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

    vocab_mode = os.environ.get("DSV4_FLASH_REAL_DECODE_LOGITS_VOCAB_MODE", "full")
    vocab_size_env = os.environ.get("DSV4_FLASH_REAL_DECODE_LOGITS_VOCAB_SIZE")
    result = run_real_decode_logits_smoke(
        snapshot,
        layer=int(os.environ.get("DSV4_FLASH_REAL_DECODE_LOGITS_LAYER", "3")),
        prefill_seq_len=int(os.environ.get("DSV4_FLASH_REAL_DECODE_LOGITS_PREFILL_SEQ_LEN", "32")),
        vocab_mode=vocab_mode,  # type: ignore[arg-type]
        vocab_start=int(os.environ.get("DSV4_FLASH_REAL_DECODE_LOGITS_VOCAB_START", "0")),
        vocab_size=None if vocab_size_env is None else int(vocab_size_env),
        device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
    )

    assert result["passed"], json.dumps(result["accuracy"], indent=2, sort_keys=True)
    assert result["current_position"] == 32
    assert result["output_shapes"]["final_norm"] == [1, 1, 1, result["model"]["hidden_size"]]
    assert result["output_shapes"]["logits"][-1] == result["vocab"]["vocab_size"]
    assert result["final_norm_lm_head"]["loaded_keys"]["final_norm"] == "norm.weight"
    assert result["final_norm_lm_head"]["loaded_keys"]["lm_head"] in {"head.weight", "embed.weight"}
