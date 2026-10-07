# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual plugin adapter and the worker-side qualification handoff."""

import json
from types import SimpleNamespace

from models.demos.qwen38_27b_qb2.demo.galaxy_serving import qualified_runtime_environment, verify_worker_precision
from models.demos.qwen38_27b_qb2.tt import generator_vllm
from models.demos.qwen38_27b_qb2.tt.precision import load_precision, precision_fingerprint


def candidate(tmp_path):
    policy = dict(load_precision("baseline"), decode_attention="accurate_full_tile")
    path = tmp_path / "precision.json"
    path.write_text(json.dumps(policy))
    return path, policy


def test_vllm_adapter_uses_the_qualified_environment_artifact(tmp_path, monkeypatch):
    path, policy = candidate(tmp_path)
    monkeypatch.setenv("QWEN_PRECISION_CONFIG", str(path))
    runtime = qualified_runtime_environment(policy)
    monkeypatch.setenv("QWEN_EXPECTED_PRECISION_SHA256", runtime["QWEN_EXPECTED_PRECISION_SHA256"])
    calls = []

    def build(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model=SimpleNamespace(precision=kwargs["precision_config"]))

    monkeypatch.setattr(generator_vllm, "build_generator", build)
    model = generator_vllm.Qwen38ForCausalLM.initialize_vllm_model(None, SimpleNamespace(shape=(1, 4)), 16, 262144)
    assert calls[0]["precision_config"] == policy
    assert model.generator.model.precision == policy


def test_wrong_override_fails_before_weights_or_model_allocation(tmp_path, monkeypatch, expect_error):
    path, policy = candidate(tmp_path)
    monkeypatch.setenv("QWEN_PRECISION_CONFIG", str(path))
    monkeypatch.setenv("QWEN_EXPECTED_PRECISION_SHA256", precision_fingerprint(policy))

    def must_not_build(*args, **kwargs):
        raise AssertionError("Must reject wrong precision before building the model")

    monkeypatch.setattr(generator_vllm, "build_generator", must_not_build)
    with expect_error(ValueError, "refusing weight loading"):
        generator_vllm.Qwen38ForCausalLM.initialize_vllm_model(
            None, SimpleNamespace(shape=(1, 4)), 16, 262144, precision_config="baseline"
        )


def test_missing_override_cannot_fall_back_to_default_under_candidate_receipt(tmp_path, monkeypatch, expect_error):
    _, policy = candidate(tmp_path)
    monkeypatch.delenv("QWEN_PRECISION_CONFIG", raising=False)
    monkeypatch.setenv("QWEN_EXPECTED_PRECISION_SHA256", precision_fingerprint(policy))
    with expect_error(ValueError, "refusing weight loading"):
        generator_vllm.Qwen38ForCausalLM.initialize_vllm_model(None, SimpleNamespace(shape=(1, 4)), 16, 262144)


def test_all_actual_worker_policies_must_match_before_evaluation(expect_error):
    policy = {"decode_attention": "accurate_full_tile"}
    rows = [f"(Worker pid={100 + i}) INFO Qwen3.8 vLLM precision: {policy}" for i in range(8)]
    assert len(verify_worker_precision("\n".join(rows), policy)) == 8
    assert len(verify_worker_precision(rows[0], policy, require_all=False)) == 1
    with expect_error(ValueError, "all eight workers"):
        verify_worker_precision("\n".join([rows[0]] * 8), policy)
    with expect_error(ValueError, "differs"):
        verify_worker_precision("\n".join(rows).replace("accurate_full_tile", "native", 1), policy, require_all=False)
