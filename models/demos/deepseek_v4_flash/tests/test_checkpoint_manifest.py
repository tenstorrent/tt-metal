# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from models.demos.deepseek_v4_flash.checkpoint_manifest import (
    _expected_weight_names,
    build_checkpoint_manifest,
    build_preprocessing_plan,
    build_snapshot_status,
)
from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.synthetic import tiny_config_dict, tiny_inference_config_dict


def test_checkpoint_manifest_parses_fake_index_only_snapshot(tmp_path: Path) -> None:
    snapshot = _write_fake_snapshot(tmp_path)

    manifest = build_checkpoint_manifest(snapshot)

    json.dumps(manifest, sort_keys=True)
    assert manifest["source"]["checkpoint_kind"] == "tiny_synthetic"
    assert manifest["source"]["is_tiny_synthetic"] is True
    assert manifest["source"]["is_real_hf_snapshot"] is False
    assert manifest["files"]["tokenizer_files"] == ["tokenizer.json", "tokenizer_config.json", "generation_config.json"]
    assert manifest["files"]["shard_count"] == 2
    assert manifest["weights"]["indexed_total_size_bytes"] == 123456
    assert manifest["weights"]["validation"]["status"] == "ok"
    assert manifest["weights"]["validation"]["missing_required_count"] == 0
    assert manifest["weights"]["validation"]["unexpected_count"] == 0
    assert manifest["dimensions"]["num_hidden_layers"] == 3
    assert manifest["dimensions"]["n_routed_experts"] == 4
    assert manifest["mla_indexer"]["compress_ratio_counts"] == {"0": 1, "4": 1, "128": 1}
    assert manifest["weights"]["coverage"]["counts"]["routed_expert_weights"] == 3 * 4 * 3
    assert manifest["weights"]["coverage"]["counts"]["routed_expert_scales"] == 3 * 4 * 3
    assert manifest["weights"]["coverage"]["counts"]["indexer"] == 3
    assert manifest["weights"]["coverage"]["counts"]["indexer_compressor"] == 4
    assert manifest["weights"]["observed"]["layer_count"] == 3
    assert manifest["weights"]["observed"]["routed_expert_count"] == 4


def test_snapshot_status_reports_metadata_only_missing_shards(tmp_path: Path) -> None:
    snapshot = _write_fake_snapshot(tmp_path, shard_contents={})

    status = build_snapshot_status(snapshot)

    json.dumps(status, sort_keys=True)
    assert status["manifest_dry_run_can_proceed"] is True
    assert status["manifest_dry_run"]["blockers"] == []
    assert status["manifest_dry_run"]["warnings"]
    assert status["weights"]["tensor_index_available"] is True
    assert status["weights"]["expected_shard_count"] == 2
    assert status["weights"]["present_shard_count"] == 0
    assert status["weights"]["missing_shard_count"] == 2
    assert status["weights"]["missing_shard_names"] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert status["weights"]["present_bytes"] == 0
    assert status["weights"]["indexed_bytes"] == 123456
    assert "config.json" in {file["path"] for file in status["files"]["non_weight_files"]}
    assert "model.safetensors.index.json" in {file["path"] for file in status["files"]["non_weight_files"]}


def test_snapshot_status_reports_partially_materialized_snapshot(tmp_path: Path) -> None:
    snapshot = _write_fake_snapshot(tmp_path, shard_contents={"model-00001-of-00002.safetensors": b"abc"})

    status = build_snapshot_status(snapshot)

    assert status["manifest_dry_run_can_proceed"] is True
    assert status["weights"]["expected_shard_count"] == 2
    assert status["weights"]["present_shard_count"] == 1
    assert status["weights"]["present_shard_names"] == ["model-00001-of-00002.safetensors"]
    assert status["weights"]["missing_shard_names"] == ["model-00002-of-00002.safetensors"]
    assert status["weights"]["present_bytes"] == 3


def test_snapshot_status_reports_complete_fake_snapshot_bytes(tmp_path: Path) -> None:
    snapshot = _write_fake_snapshot(
        tmp_path,
        shard_contents={
            "model-00001-of-00002.safetensors": b"",
            "model-00002-of-00002.safetensors": b"payload",
        },
    )

    status = build_snapshot_status(snapshot)

    assert status["manifest_dry_run_can_proceed"] is True
    assert status["manifest_dry_run"]["warnings"] == []
    assert status["weights"]["expected_shard_count"] == 2
    assert status["weights"]["present_shard_count"] == 2
    assert status["weights"]["missing_shard_names"] == []
    assert status["weights"]["present_bytes"] == len(b"payload")


def test_checkpoint_manifest_reports_unknown_weight_patterns(tmp_path: Path) -> None:
    snapshot = _write_fake_snapshot(tmp_path, extra_weight_names=("layers.0.not_a_real_family.weight",))

    manifest = build_checkpoint_manifest(snapshot)

    assert manifest["weights"]["validation"]["status"] == "warning"
    assert manifest["weights"]["validation"]["missing_required_count"] == 0
    assert manifest["weights"]["validation"]["unexpected_count"] == 1
    assert manifest["weights"]["validation"]["unexpected_examples"] == ["layers.0.not_a_real_family.weight"]


def test_preprocessing_plan_is_json_serializable_and_marks_future_work(tmp_path: Path) -> None:
    manifest = build_checkpoint_manifest(_write_fake_snapshot(tmp_path))

    plan = build_preprocessing_plan(manifest, "galaxy")

    json.dumps(plan, sort_keys=True)
    actions = {action["family"]: action for action in plan["actions"]}
    assert plan["dry_run"] is True
    assert plan["topology"]["name"] == "galaxy"
    assert plan["topology"]["mesh_shape_hint"] == [8, 4]
    assert actions["config_tokenizer"]["action"] == "copy"
    assert actions["routed_experts"]["support"] == "planned_placeholder"
    assert "FP4" in actions["routed_experts"]["source_format"]
    assert "BFP4" in actions["routed_experts"]["planned_format"]
    assert actions["mtp_next_token_prediction"]["support"] == "unsupported"
    assert actions["tensor_caches"]["support"] == "unsupported"
    assert any("DeepSeek FP4 source" in note for note in plan["format_notes"])


def test_checkpoint_manifest_cli_dry_run_on_fake_snapshot(tmp_path: Path) -> None:
    snapshot = _write_fake_snapshot(tmp_path)
    repo_root = Path(__file__).resolve().parents[4]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.checkpoint_manifest",
            "--snapshot-dir",
            str(snapshot),
            "--topology",
            "galaxy",
            "--dry-run",
        ],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(result.stdout)
    assert output["manifest"]["source"]["checkpoint_kind"] == "tiny_synthetic"
    assert output["preprocessing_plan"]["topology"]["name"] == "galaxy"
    assert output["preprocessing_plan"]["dry_run"] is True


def test_checkpoint_manifest_cli_status_on_fake_snapshot(tmp_path: Path) -> None:
    snapshot = _write_fake_snapshot(tmp_path, shard_contents={"model-00002-of-00002.safetensors": b"present"})
    repo_root = Path(__file__).resolve().parents[4]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.checkpoint_manifest",
            "--snapshot-dir",
            str(snapshot),
            "--topology",
            "galaxy",
            "--status",
        ],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(result.stdout)
    assert set(output) == {"snapshot_status"}
    status = output["snapshot_status"]
    assert status["manifest_dry_run_can_proceed"] is True
    assert status["weights"]["expected_shard_count"] == 2
    assert status["weights"]["present_shard_count"] == 1
    assert status["weights"]["missing_shard_names"] == ["model-00001-of-00002.safetensors"]


def _write_fake_snapshot(
    tmp_path: Path,
    *,
    extra_weight_names: tuple[str, ...] = (),
    shard_contents: dict[str, bytes] | None = None,
) -> Path:
    snapshot = tmp_path / "fake_hf_snapshot"
    snapshot.mkdir()
    (snapshot / "inference").mkdir()

    config = tiny_config_dict(num_hidden_layers=3, num_routed_experts=4, compress_ratios=(0, 4, 128))
    inference_config = tiny_inference_config_dict(config)
    _write_json(snapshot / "config.json", config)
    _write_json(snapshot / "inference" / "config.json", inference_config)
    _write_json(snapshot / "tokenizer.json", {"version": "1.0"})
    _write_json(snapshot / "tokenizer_config.json", {"model_max_length": 128})
    _write_json(snapshot / "generation_config.json", {"bos_token_id": 0, "eos_token_id": 1})

    checkpoint_config = DeepSeekV4FlashConfig.from_hf_configs(config, inference_config)
    weight_names = sorted(_expected_weight_names(checkpoint_config) | set(extra_weight_names))
    shard_names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    if shard_contents is None:
        shard_contents = {shard_name: b"" for shard_name in shard_names}
    for shard_name, contents in shard_contents.items():
        (snapshot / shard_name).write_bytes(contents)

    weight_map = {name: shard_names[index % len(shard_names)] for index, name in enumerate(weight_names)}
    _write_json(
        snapshot / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 123456},
            "weight_map": weight_map,
        },
    )
    return snapshot


def _write_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")
