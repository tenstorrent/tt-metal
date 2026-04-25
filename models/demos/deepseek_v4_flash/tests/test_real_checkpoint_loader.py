# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from models.demos.deepseek_v4_flash.converter import MODEL_INDEX_FILENAME
from models.demos.deepseek_v4_flash.real_checkpoint_loader import (
    SLICE_ARTIFACT_FILENAME,
    SLICE_MANIFEST_FILENAME,
    RealCheckpointTensorIndex,
    layer_router_norm_keys,
)


def test_real_tensor_index_reads_selected_metadata_without_payloads(tmp_path: Path) -> None:
    snapshot = _write_fake_router_norm_snapshot(tmp_path / "snapshot", layer=3, hash_router=False)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot)

    keys = layer_router_norm_keys(index, layer=3)
    metadata = {item.canonical_key: item for item in index.metadata_for_keys(keys)}

    assert keys == [
        "layers.3.attn_norm.weight",
        "layers.3.ffn_norm.weight",
        "layers.3.ffn.gate.weight",
        "layers.3.ffn.gate.bias",
    ]
    assert metadata["layers.3.attn_norm.weight"].source_key == "model.layers.3.input_layernorm.weight"
    assert metadata["layers.3.attn_norm.weight"].dtype == "BF16"
    assert metadata["layers.3.attn_norm.weight"].shape == (4,)
    assert metadata["layers.3.attn_norm.weight"].nbytes == 8
    assert metadata["layers.3.ffn.gate.weight"].shape == (3, 4)
    assert metadata["layers.3.ffn.gate.weight"].nbytes == 24
    assert metadata["layers.3.ffn.gate.bias"].dtype == "F32"
    assert metadata["layers.3.ffn.gate.bias"].nbytes == 12


def test_layer_router_norm_selector_accepts_hash_router_sidecar(tmp_path: Path) -> None:
    snapshot = _write_fake_router_norm_snapshot(tmp_path / "snapshot", layer=0, hash_router=True)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot)

    assert layer_router_norm_keys(index, layer=0)[-1] == "layers.0.ffn.gate.tid2eid"


def test_selective_loader_refuses_budget_overruns(tmp_path: Path) -> None:
    snapshot = _write_fake_router_norm_snapshot(tmp_path / "snapshot", layer=3, hash_router=False)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot)
    keys = layer_router_norm_keys(index, layer=3)

    with pytest.raises(ValueError, match="tensor budget"):
        index.load_tensors(keys, max_tensors=3, max_bytes=1024)
    with pytest.raises(ValueError, match="byte budget"):
        index.load_tensors(keys, max_tensors=4, max_bytes=51)


def test_selective_loader_materializes_only_requested_payloads(tmp_path: Path) -> None:
    snapshot = _write_fake_router_norm_snapshot(tmp_path / "snapshot", layer=3, hash_router=False)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot)

    tensors, metadata = index.load_tensors(["layers.3.ffn.gate.bias"], max_tensors=1, max_bytes=12)

    assert list(tensors) == ["layers.3.ffn.gate.bias"]
    assert [item.canonical_key for item in metadata] == ["layers.3.ffn.gate.bias"]
    torch.testing.assert_close(tensors["layers.3.ffn.gate.bias"], torch.tensor([0.5, 1.5, 2.5]))


def test_layer_router_norm_cli_writes_tiny_materialization(tmp_path: Path) -> None:
    snapshot = _write_fake_router_norm_snapshot(tmp_path / "snapshot", layer=3, hash_router=False)
    output_dir = tmp_path / "slice"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.real_checkpoint_loader",
            "--snapshot-dir",
            str(snapshot),
            "--output-dir",
            str(output_dir),
            "--layer",
            "3",
            "--max-bytes",
            "128",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    cli_result = json.loads(result.stdout)
    assert cli_result["output_dir"] == str(output_dir.resolve())
    artifact_path = output_dir / SLICE_ARTIFACT_FILENAME
    manifest_path = output_dir / SLICE_MANIFEST_FILENAME
    assert artifact_path.is_file()
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["selector"] == "layer-router-norms"
    assert manifest["layer"] == 3
    assert manifest["budget"]["selected_tensors"] == 4
    assert manifest["budget"]["selected_payload_bytes"] == 52
    assert manifest["artifact"] == SLICE_ARTIFACT_FILENAME

    with safe_open(artifact_path, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == {
            "layers.3.attn_norm.weight",
            "layers.3.ffn_norm.weight",
            "layers.3.ffn.gate.weight",
            "layers.3.ffn.gate.bias",
        }
        torch.testing.assert_close(handle.get_tensor("layers.3.ffn.gate.bias"), torch.tensor([0.5, 1.5, 2.5]))


def _write_fake_router_norm_snapshot(snapshot_dir: Path, *, layer: int, hash_router: bool) -> Path:
    snapshot_dir.mkdir(parents=True)
    shard_name = "model-00001-of-00001.safetensors"
    prefix = f"model.layers.{layer}"
    canonical_prefix = f"layers.{layer}"
    tensors = {
        f"{prefix}.input_layernorm.weight": torch.arange(1, 5, dtype=torch.float32).to(torch.bfloat16),
        f"{prefix}.post_attention_layernorm.weight": torch.arange(5, 9, dtype=torch.float32).to(torch.bfloat16),
        f"{prefix}.mlp.gate.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.bfloat16),
    }
    if hash_router:
        tensors[f"{canonical_prefix}.ffn.gate.tid2eid"] = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    else:
        tensors[f"{prefix}.mlp.gate.e_score_correction_bias"] = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)

    save_file(tensors, str(snapshot_dir / shard_name))
    index = {
        "metadata": {"total_size": sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())},
        "weight_map": {key: shard_name for key in sorted(tensors)},
    }
    (snapshot_dir / MODEL_INDEX_FILENAME).write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return snapshot_dir
