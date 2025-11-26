# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import safetensors.torch
import torch

from models.demos.deepseek_v3.utils.config_helpers import get_state_dicts, sub_state_dict
from models.demos.deepseek_v3.utils.test_utils import load_state_dict


def _write_index(model_dir: Path, weight_map: dict[str, str]) -> None:
    index = {"metadata": {}, "weight_map": weight_map}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))


def test_lazy_state_dict_access_single_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create two safetensors files with disjoint keys
    file1 = model_dir / "model-00001-of-00002.safetensors"
    file2 = model_dir / "model-00002-of-00002.safetensors"
    t1 = torch.randn(2, 3, dtype=torch.bfloat16)
    t2 = torch.randn(4, 5, dtype=torch.bfloat16)
    safetensors.torch.save_file({"w1": t1}, str(file1))
    safetensors.torch.save_file({"w2": t2}, str(file2))
    _write_index(model_dir, {"w1": file1.name, "w2": file2.name})

    # Count safetensors.safe_open calls
    lsd = importlib.import_module("models.demos.deepseek_v3.utils.lazy_state_dict")
    open_counts: dict[str, int] = {}
    original_safe_open = lsd.safe_open

    def counting_safe_open(path, *args, **kwargs):
        name = Path(path).name
        open_counts[name] = open_counts.get(name, 0) + 1
        return original_safe_open(path, *args, **kwargs)

    monkeypatch.setattr(lsd, "safe_open", counting_safe_open, raising=True)

    # Lazily load and access only one key
    state = load_state_dict(model_dir, "")
    v1 = state["w1"]
    assert torch.equal(v1, t1)
    assert open_counts.get(file1.name, 0) == 1
    assert open_counts.get(file2.name, 0) == 0


def test_lazy_sub_state_dict_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    file1 = model_dir / "model-00001-of-00002.safetensors"
    file2 = model_dir / "model-00002-of-00002.safetensors"
    t_0_foo = torch.randn(2, 2)
    t_0_bar = torch.randn(1, 3)
    t_1_foo = torch.randn(3, 1)
    safetensors.torch.save_file(
        {"model.layers.0.foo": t_0_foo, "model.layers.0.bar": t_0_bar},
        str(file1),
    )
    safetensors.torch.save_file({"model.layers.1.foo": t_1_foo}, str(file2))
    _write_index(
        model_dir,
        {
            "model.layers.0.foo": file1.name,
            "model.layers.0.bar": file1.name,
            "model.layers.1.foo": file2.name,
        },
    )

    lsd = importlib.import_module("models.demos.deepseek_v3.utils.lazy_state_dict")
    open_counts: dict[str, int] = {}
    original_safe_open = lsd.safe_open

    def counting_safe_open(path, *args, **kwargs):
        name = Path(path).name
        open_counts[name] = open_counts.get(name, 0) + 1
        return original_safe_open(path, *args, **kwargs)

    monkeypatch.setattr(lsd, "safe_open", counting_safe_open, raising=True)

    state = load_state_dict(model_dir, "")
    sub = sub_state_dict(state, "model.layers.0.")
    keys = sorted(list(sub.keys()))
    assert keys == ["bar", "foo"]

    # Access one value; only file1 is opened
    v = sub["foo"]
    assert torch.equal(v, t_0_foo)
    assert open_counts.get(file1.name, 0) == 1
    assert open_counts.get(file2.name, 0) == 0


def test_get_state_dicts_integration(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    file1 = model_dir / "model-00001-of-00001.safetensors"
    key_full = "model.layers.0.k"
    t = torch.randn(2, 2, dtype=torch.float32)
    safetensors.torch.save_file({key_full: t}, str(file1))
    _write_index(model_dir, {key_full: file1.name})

    state = load_state_dict(model_dir, "")
    sub = sub_state_dict(state, "model.layers.0.")

    out = get_state_dicts((sub,), "k", shape=(2, 2), dtype=torch.float32)
    assert out.shape == (1, 2, 2)
    assert torch.equal(out[0], t)
