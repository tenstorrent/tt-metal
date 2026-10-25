"""Host checks for the source-only serving deployment boundary."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tt" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    for key in ("QWEN38_CHECKPOINT", "HF_MODEL", "MODEL_WEIGHTS_DIR"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "weights"
    path.mkdir()
    for name in ("config.json", "model.safetensors.index.json"):
        (path / name).write_text("{}")
    return path


@pytest.mark.parametrize("source", ["QWEN38_CHECKPOINT", "HF_MODEL", "MODEL_WEIGHTS_DIR", "config", "legacy"])
def test_prepared_checkpoint(source, checkpoint, monkeypatch, tmp_path):
    cfg = SimpleNamespace()
    if source == "config":
        cfg.name_or_path = str(checkpoint)
    elif source == "legacy":
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".cache").mkdir()
        (tmp_path / ".cache/qwen38-full").symlink_to(checkpoint)
    else:
        monkeypatch.setenv(source, str(checkpoint))
    assert load("checkpoint_path").serving_checkpoint(cfg) == checkpoint


def test_explicit_override_and_invalid_source(checkpoint, monkeypatch):
    monkeypatch.setenv("HF_MODEL", "Qwen/Qwen3.8-27B")
    monkeypatch.setenv("QWEN38_CHECKPOINT", str(checkpoint))
    resolve = load("checkpoint_path").serving_checkpoint
    assert resolve(SimpleNamespace()) == checkpoint
    monkeypatch.setenv("QWEN38_CHECKPOINT", "")
    with pytest.raises(FileNotFoundError, match="HF_MODEL"):
        resolve(SimpleNamespace())


def test_selected_policy_matches_benchmark():
    policy = load("precision").load_precision()
    assert policy == json.loads((ROOT / "tt/selected_precision_config.json").read_text())
    assert policy["config_id"] == "head_bfp8_hifi2"
    assert policy["weight_groups"]["head"] == "bfloat8_b"
    assert policy["compute_fidelities"]["head"] == "HiFi2"
