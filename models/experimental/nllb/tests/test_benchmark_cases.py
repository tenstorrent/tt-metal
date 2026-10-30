# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Shared workload parity, model routing, and fail-closed configuration binding."""
import hashlib
import json
from pathlib import Path

import pytest
from models.experimental.nllb.benchmarks.benchmark_paired import load_case
from models.experimental.nllb.tt.nllb_validation import validate_config

ROOT = Path(__file__).resolve().parents[1] / "benchmarks"
ORIGINAL_INPUT_SHA256 = {
    "short_b1": "c7d1a9dad0653a85984977e8d3399b5a7a7146b06711f8f3d53b3b32fc48f77e",
    "short_b2": "ef0d81c0a4fd745fd7af294106f6369605e03c8e9acec704d943407b36e6e5ce",
    "long_b1": "10a6c02d6ad8a168f11f0832223fc5f2a07db57bfd916dffe969807cd8c10ef6",
    "long_b2": "5710df9abd8916907dca071fc0d9c236abc657b981dfaa977c7752a31e00ebf0",
    "source256_b1": "b7e63b10e76628c1fa4cb49e19a8f36a77ef16743218931d291d4b710d95c1a5",
    "source256_b4": "20112c74dd3f824c29d58b07c07557ffac734f452f68768a27adb689c15802af",
    "natural_long_b2": "4f94116127f866584afb9de5c41d1156d9fa6f76547e4fdfc7343e4e0e751a27",
}


@pytest.mark.parametrize("name", sorted(ORIGINAL_INPUT_SHA256))
def test_shared_inputs_preserve_original_workload(name):
    data = json.loads((ROOT / "cases" / (name + ".json")).read_text())
    data.pop("config_sha256_by_model")
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == ORIGINAL_INPUT_SHA256[name]


@pytest.mark.parametrize("model", ["600m", "1.3b-distilled", "3.3b"])
@pytest.mark.parametrize("name", ["short_b1", "short_b2", "long_b1", "long_b2", "source256_b1", "source256_b4"])
def test_shared_workload_accepts_each_bound_model(model, name):
    config_path = ROOT / "configs" / (model + ".json")
    config = validate_config(json.loads(config_path.read_text()))
    actual, ids, mask, target, cap, note = load_case(ROOT / "cases" / (name + ".json"), config_path, config)
    assert actual == name and ids.shape == mask.shape
    assert ids.shape[0] == int(name[-1]) and target == 256057
    assert cap == (40 if name.startswith("long") else 8)
    if name.startswith("source256"):
        assert ids.shape[1] == 256 and "synthetic" in note


@pytest.mark.parametrize("fault", ["wrong_config", "empty", "wrong_type", "bad_hash", "unknown_model"])
def test_configuration_binding_rejects_mismatch(tmp_path, fault):
    config_path = ROOT / "configs/600m.json"
    config = validate_config(json.loads(config_path.read_text()))
    data = json.loads((ROOT / "cases/short_b1.json").read_text())
    if fault == "wrong_config":
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(dict(config, d_model=2048)))
    else:
        data["config_sha256_by_model"] = {
            "empty": {},
            "wrong_type": [],
            "bad_hash": {"600m": "g" * 64},
            "unknown_model": {"unknown": "0" * 64},
        }[fault]
    case = tmp_path / "case.json"
    case.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="configuration hash"):  # allow-pytest.raises: CPU-only check.
        load_case(case, config_path, config)


def test_natural_case_does_not_claim_larger_model_validation():
    config_path = ROOT / "configs/1.3b-distilled.json"
    with pytest.raises(ValueError, match="configuration hash"):  # allow-pytest.raises: CPU-only check.
        load_case(ROOT / "cases/natural_long_b2.json", config_path, json.loads(config_path.read_text()))
