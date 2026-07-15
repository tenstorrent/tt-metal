# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import inspect
from pathlib import Path

import pytest

import ttnn
from models.autoports.google_gemma_4_31b.tests.run_full_model_qualitative import (
    _model_config_from_environment,
    _run_benchmark_only,
)
from models.autoports.google_gemma_4_31b.tt.generator import build_generator
from models.autoports.google_gemma_4_31b.tt.model import Gemma4FullModelConfig
from models.autoports.google_gemma_4_31b.tt.precision import load_precision_config

MODEL_DIR = Path("models/autoports/google_gemma_4_31b")
BASELINE = MODEL_DIR / "doc/datatype_sweep/configs/baseline_bfp8attn_bfp4mlp_lofi_bf16lm.json"
SELECTED = MODEL_DIR / "doc/datatype_sweep/selected_precision_config.json"


def test_baseline_precision_config_resolves_every_runtime_field():
    policy = load_precision_config(BASELINE)
    summary = policy.summary()
    assert summary["config_id"] == "baseline_bfp8attn_bfp4mlp_lofi_bf16lm"
    assert summary["weight_groups"] == {
        "attention_prefill": "bfloat8_b",
        "attention_qkv": "bfloat8_b",
        "attention_output": "bfloat8_b",
        "mlp_gate_up": "bfloat4_b",
        "mlp_down": "bfloat4_b",
        "lm_head": "bfloat16",
    }
    assert summary["compute_fidelities"]["attention_qkv"] == "LoFi"
    assert summary["compute_fidelities"]["mlp_down"] == "LoFi"
    assert summary["ccl_dtype"] == {"prefill": "bfloat16", "decode": "bfloat8_b"}
    assert summary["kv_cache_dtype"] == "bfloat8_b"


def test_full_model_config_consumes_resolved_precision_policy():
    config = Gemma4FullModelConfig.from_precision_config(BASELINE)
    assert config.precision_config_id == "baseline_bfp8attn_bfp4mlp_lofi_bf16lm"
    assert config.decoder_optimization_policy.resolved_attention_qkv_weight_dtype == ttnn.bfloat8_b
    assert config.decoder_optimization_policy.mlp_gate_up_weight_dtype == ttnn.bfloat4_b
    assert config.decoder_optimization_policy.kv_cache_dtype == ttnn.bfloat8_b
    assert config.activation_dtype == ttnn.bfloat16
    assert config.residual_dtype == ttnn.bfloat16
    assert config.prefill_ccl_dtype == ttnn.bfloat16
    assert config.decode_ccl_dtype == ttnn.bfloat8_b
    assert config.lm_head_weight_dtype == ttnn.bfloat16
    assert config.logits_dtype == ttnn.bfloat16
    assert config.sampling_dtype == ttnn.float32


@pytest.mark.parametrize("path", sorted((MODEL_DIR / "doc/datatype_sweep/configs").glob("*.json")))
def test_every_sweep_candidate_resolves(path):
    resolved = load_precision_config(path)
    assert resolved.config_id == path.stem
    assert len(resolved.summary()["weight_groups"]) == 6


def test_normal_construction_and_token_out_harness_consume_precision_artifact(monkeypatch):
    source = inspect.getsource(build_generator)
    assert "GEMMA4_31B_PRECISION_CONFIG" in source
    assert "doc/datatype_sweep/selected_precision_config.json" in source
    monkeypatch.setenv("GEMMA4_31B_PRECISION_CONFIG", str(BASELINE))
    config = _model_config_from_environment()
    assert config.precision_config_id == "baseline_bfp8attn_bfp4mlp_lofi_bf16lm"
    assert config.precision_config_path == str(BASELINE.resolve())


def test_selected_precision_is_the_normal_default_when_stage_artifact_exists(monkeypatch):
    if not SELECTED.exists():
        pytest.skip("Stage 08 selected precision artifact has not been written yet")
    monkeypatch.delenv("GEMMA4_31B_PRECISION_CONFIG", raising=False)
    selected = load_precision_config(SELECTED)
    config = _model_config_from_environment()
    assert config.precision_config_id == selected.config_id
    assert config.precision_config_path == str(SELECTED.resolve())
    benchmark_source = inspect.getsource(_run_benchmark_only)
    assert '"runtime_precision": runtime_precision' in benchmark_source
