"""Reject ambiguous or unimplemented policies before device construction."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("precision", Path(__file__).parents[1] / "tt/precision.py")
precision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precision)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["weight_groups"].pop("down"),
        lambda p: p["compute_fidelities"].update(attention="HiFi2"),
        lambda p: p.update(residual_dtype="bfloat8_b"),
        lambda p: p["ccl_dtype"].update(decode="float32"),
        lambda p: p["layer_exceptions"].update({"60": {"mlp": "bfp4"}}),
        lambda p: p.update(max_seq_len=32768),
    ],
)
def test_reject_unconstructible_policy(mutate):
    policy = copy.deepcopy(precision.BASELINE)
    mutate(policy)
    with pytest.raises(ValueError):  # allow-pytest.raises: Host-only schema test runs without TTNN fixtures.
        precision.load_precision(policy)


def test_policy_does_not_mutate_caller():
    policy = precision.load_precision(precision.BASELINE)
    policy["weight_groups"]["down"] = "bf16"
    assert precision.BASELINE["weight_groups"]["down"] == "bfp4"


def test_selected_default_survives_source_only_publication(monkeypatch):
    def unavailable_documentation(*args, **kwargs):
        raise FileNotFoundError("Documentation is not shipped to serving CI")

    monkeypatch.setattr(Path, "read_text", unavailable_documentation)
    policy = precision.load_precision()
    assert policy["config_id"] == "head_bfp4"
    assert policy["weight_groups"]["lm_head"] == "bfp4"
    assert policy["compute_fidelities"]["lm_head"] == "LoFi"
    assert policy["max_seq_len"] == 262144
    policy["weight_groups"]["lm_head"] = "bf16"
    assert precision.load_precision()["weight_groups"]["lm_head"] == "bfp4"


def test_serving_experiment_is_pinned_and_does_not_change_default():
    import json

    before = precision.load_precision()
    policy = precision.load_serving_precision("decoder_bfp8", precision.EXPERIMENTAL_DECODER_BFP8_SHA256)
    expected = json.loads((Path(__file__).parents[1] / "doc/datatype_sweep/configs/decoder_bfp8.json").read_text())
    assert policy == expected
    assert precision.load_serving_precision() == before
    assert all(value == "bfp8" for value in policy["weight_groups"].values())
    assert policy["max_seq_len"] == before["max_seq_len"] == 262144
    policy["weight_groups"]["attention"] = "bf16"
    assert precision.load_serving_precision("decoder_bfp8", precision.EXPERIMENTAL_DECODER_BFP8_SHA256) == expected


@pytest.mark.parametrize(
    "selector,digest",
    [("other", None), ("", None), ("decoder_bfp8", None), ("decoder_bfp8", "0" * 64), (None, "0" * 64)],
)
def test_serving_experiment_rejects_ambiguous_request(selector, digest):
    with pytest.raises(ValueError):  # allow-pytest.raises: Host-only schema test runs without TTNN fixtures.
        precision.load_serving_precision(selector, digest)


def test_serving_experiment_detects_policy_content_drift(monkeypatch):
    changed = copy.deepcopy(precision.SELECTED)
    changed["activation_dtype"] = "bfloat16"
    monkeypatch.setattr(precision, "SELECTED", changed)
    with pytest.raises(ValueError, match="contents changed"):  # allow-pytest.raises: Host-only schema test.
        precision.load_serving_precision("decoder_bfp8", precision.EXPERIMENTAL_DECODER_BFP8_SHA256)


@pytest.mark.parametrize("experimental", [False, True])
def test_actual_adapter_initializer_passes_policy_before_construction(monkeypatch, experimental):
    # Execute the actual initializer body with a fake model/mesh, avoiding TT imports/devices.
    import ast
    import os
    from types import SimpleNamespace

    tree = ast.parse((Path(__file__).parents[1] / "tt/generator_vllm.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Gemma4ForCausalLM")
    method = copy.deepcopy(
        next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "initialize_vllm_model")
    )
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    calls = []

    def construct(mesh, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(precision_config=copy.deepcopy(kwargs["precision_config"]), layers=[])

    monkeypatch.delenv("GEMMA4_EXPERIMENTAL_PRECISION", raising=False)
    monkeypatch.delenv("GEMMA4_EXPERIMENTAL_PRECISION_SHA256", raising=False)
    monkeypatch.delenv("GEMMA4_BENCHMARK_PORT", raising=False)
    monkeypatch.delenv("GEMMA4_VLLM_REDUCED", raising=False)
    if experimental:
        monkeypatch.setenv("GEMMA4_EXPERIMENTAL_PRECISION", "decoder_bfp8")
        monkeypatch.setenv("GEMMA4_EXPERIMENTAL_PRECISION_SHA256", precision.EXPERIMENTAL_DECODER_BFP8_SHA256)
    namespace = {
        "os": os,
        "Gemma4Model": construct,
        "load_serving_precision": precision.load_serving_precision,
        "json": __import__("json"),
        "__file__": "actual_adapter_host_fixture",
    }
    exec(compile(module, "actual_adapter_initializer", "exec"), namespace)
    initialize = namespace["initialize_vllm_model"]
    adapter = initialize(type("HostAdapter", (), {}), None, SimpleNamespace(shape=(1, 4)), 32, 262144)
    assert adapter.model.precision_config["config_id"] == ("decoder_bfp8" if experimental else "head_bfp4")
    assert adapter.batch == 32 and adapter.context == 262144
    assert calls[0]["layer_indices"] is None
    monkeypatch.setenv("GEMMA4_EXPERIMENTAL_PRECISION", "unknown")
    with pytest.raises(ValueError, match="Unknown"):  # allow-pytest.raises: Host-only initializer fixture.
        initialize(type("HostAdapter", (), {}), None, SimpleNamespace(shape=(1, 4)), 32, 262144)
    assert len(calls) == 1
