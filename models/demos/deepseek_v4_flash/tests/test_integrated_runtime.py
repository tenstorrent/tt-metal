# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from models.demos.deepseek_v4_flash import cpu_reference
from models.demos.deepseek_v4_flash import runtime as runtime_module
from models.demos.deepseek_v4_flash.runtime import (
    DeepSeekRuntimeBlocked,
    HardwareMeshProbeResult,
    HostBoundaryViolation,
    RuntimeHostBoundaryGuard,
    TtDeepSeekV4FlashRuntime,
    runtime_summary_from_exception,
)
from models.demos.deepseek_v4_flash.converter import convert_hf_checkpoint
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint


def test_integrated_runtime_preflight_is_fail_closed_with_actionable_blockers(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=5,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128, 4],
    )

    runtime = TtDeepSeekV4FlashRuntime.from_hf_snapshot(snapshot)
    report = runtime.preflight("generate", context={"prompt_length": 4, "requested_new_tokens": 2})
    payload = report.to_mapping()

    json.dumps(payload, sort_keys=True)
    assert payload["model_id"] == "deepseek-ai/DeepSeek-V4-Flash"
    assert payload["phase"] == "generate"
    assert payload["preprocessed_dir"] == str((tmp_path / "hf_tt_preprocessed").resolve())
    assert payload["blocked"] is True
    assert payload["batch_size"] == 1
    assert payload["device_topology"] == "Blackhole Loudbox 2x4 (8 devices)"
    assert payload["tracing_enabled"] is True
    assert payload["configured_max_position_embeddings"] == 1024
    assert payload["max_seq_len_supported"] is None
    assert payload["estimated_max_seq_len_if_device_path_ready"] is None
    assert payload["context"] == {"prompt_length": 4, "requested_new_tokens": 2}

    codes = {blocker["code"] for blocker in payload["blockers"]}
    assert not any(code.startswith("checkpoint.") for code in codes)
    assert not any(code.startswith("tokenizer.") for code in codes)
    assert "runtime.full_model_device_path" in codes
    assert "runtime.full_weight_materialization" in codes
    assert "prefill.device_only_attention" in codes
    assert "prefill.device_resident_cache_seed" in codes
    assert "decode.paged_sparse_attention" in codes
    assert "decode.device_resident_cache" in codes
    assert "decode.traced_moe_routing_dispatch" in codes
    assert "decode.traced_token_selection" in codes
    assert all(blocker["next_action"] for blocker in payload["blockers"])


def test_integrated_runtime_methods_raise_structured_blocker_reports(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=3, compress_ratios=[0, 0, 4])
    runtime = TtDeepSeekV4FlashRuntime(snapshot)

    with pytest.raises(DeepSeekRuntimeBlocked) as exc_info:
        runtime.prefill(torch.tensor([[0, 1, 2, 3]], dtype=torch.int64))

    payload = runtime_summary_from_exception(exc_info.value)
    assert payload["phase"] == "prefill"
    assert payload["blocked"] is True
    assert payload["context"] == {"prompt_length": 4}
    codes = {blocker["code"] for blocker in payload["blockers"]}
    assert "prefill.device_only_attention" in codes
    assert "decode.paged_sparse_attention" not in codes

    with pytest.raises(DeepSeekRuntimeBlocked) as generate_exc:
        runtime.generate([0, 1, 2, 3], max_new_tokens=2)
    generate_payload = generate_exc.value.report.to_mapping()
    assert generate_payload["phase"] == "generate"
    assert generate_payload["context"]["requested_new_tokens"] == 2


def test_integrated_runtime_validates_token_ids_before_reporting_runtime_blockers(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=3, compress_ratios=[0, 0, 4])
    runtime = TtDeepSeekV4FlashRuntime(snapshot)

    with pytest.raises(ValueError, match="outside vocab_size=64"):
        runtime.prefill([0, 64])

    with pytest.raises(ValueError, match="batch size 1"):
        runtime.prefill(torch.zeros((2, 4), dtype=torch.int64))

    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        runtime.generate([0, 1], max_new_tokens=0)


def test_integrated_runtime_reports_missing_real_checkpoint_assets(tmp_path: Path) -> None:
    snapshot = tmp_path / "missing"
    runtime = TtDeepSeekV4FlashRuntime(snapshot)

    report = runtime.preflight("generate")
    codes = [blocker.code for blocker in report.blockers]

    assert codes[0] == "checkpoint.snapshot_missing"
    assert "runtime.full_model_device_path" in codes


def test_integrated_runtime_validates_complete_snapshot_assets(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=1)
    (snapshot / "generation_config.json").unlink()
    (snapshot / "encoding" / "encoding_dsv4.py").unlink()
    (snapshot / "model-00001-of-00001.safetensors").unlink()

    runtime = TtDeepSeekV4FlashRuntime(snapshot)
    report = runtime.preflight("generate")
    codes = {blocker.code for blocker in report.blockers}

    assert "checkpoint.weight_shards_missing" in codes
    assert "tokenizer.assets_missing" in codes
    assert "tokenizer.encoding_missing" in codes


def test_integrated_runtime_recognizes_complete_tt_preprocessed_weights(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=1)
    preprocessed = convert_hf_checkpoint(snapshot, tmp_path / "tt_preprocessed")

    runtime = TtDeepSeekV4FlashRuntime(snapshot, preprocessed_dir=preprocessed)
    report = runtime.preflight("generate")
    codes = {blocker.code for blocker in report.blockers}

    assert "runtime.full_weight_materialization" not in codes
    assert "runtime.device_weight_ownership" in codes
    assert not any(code.startswith("weights.") for code in codes)
    assert report.estimated_max_seq_len_if_device_path_ready == 1024


def test_integrated_runtime_hardware_preflight_reports_mesh_open_blocker(tmp_path: Path, monkeypatch) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=1)

    def fake_probe(*, mesh_shape, visible_devices, timeout_s):
        assert mesh_shape == (2, 4)
        assert visible_devices == "0,1,2,3,4,5,6,7"
        assert timeout_s == 12.0
        return HardwareMeshProbeResult(
            target_topology="Blackhole Loudbox 2x4 (8 devices)",
            visible_devices=visible_devices,
            mesh_graph_desc_path="/repo/tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto",
            opened=False,
            available_devices=8,
            returncode=0,
            timed_out=False,
            error="Graph specified in MGD could not fit in the discovered physical topology for mesh 0.",
            evidence=("TopologyMapper mapping start", "Graph specified in MGD could not fit"),
        )

    monkeypatch.setattr(runtime_module, "_run_ttnn_mesh_open_probe", fake_probe)

    runtime = TtDeepSeekV4FlashRuntime(snapshot)
    report = runtime.preflight("generate", check_hardware=True, hardware_timeout_s=12.0)
    codes = {blocker.code for blocker in report.blockers}

    assert "hardware.mesh_open_failed" in codes
    assert report.context["hardware_preflight"]["opened"] is False
    assert report.context["hardware_preflight"]["available_devices"] == 8
    assert report.context["hardware_preflight"]["visible_devices"] == "0,1,2,3,4,5,6,7"


def test_integrated_runtime_rejects_partial_tt_preprocessed_weights(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=1)
    preprocessed = convert_hf_checkpoint(snapshot, tmp_path / "tt_preprocessed")
    manifest = load_tt_manifest(preprocessed)
    (preprocessed / manifest["artifacts"]["non_expert_safetensors"][0]).unlink()

    runtime = TtDeepSeekV4FlashRuntime(snapshot, preprocessed_dir=preprocessed)
    report = runtime.preflight("generate")
    codes = {blocker.code for blocker in report.blockers}

    assert "weights.tt_artifacts_missing" in codes
    assert "runtime.full_weight_materialization" in codes
    assert "runtime.device_weight_ownership" not in codes


def test_runtime_host_boundary_guard_blocks_known_fallback_helpers() -> None:
    with RuntimeHostBoundaryGuard() as guard:
        assert "cpu_reference.sparse_attention" in guard.guarded_labels
        with pytest.raises(HostBoundaryViolation, match="cpu_reference.sparse_attention"):
            cpu_reference.sparse_attention(None, None, None, None, 1.0)

    with pytest.raises(AttributeError):
        # The guard restored the original helper; calling it with invalid inputs now reaches the real function.
        cpu_reference.sparse_attention(None, None, None, None, 1.0)
