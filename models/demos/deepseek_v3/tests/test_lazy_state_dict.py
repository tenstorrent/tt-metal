# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
import safetensors.torch
import torch

from models.demos.deepseek_v3.utils import lazy_state_dict as lsd
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


def test_lazy_cache_does_not_collide_across_views(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create keys with identical suffix under different prefixes
    file1 = model_dir / "model-00001-of-00002.safetensors"
    file2 = model_dir / "model-00002-of-00002.safetensors"
    key_mlp = "model.layers.3.mlp.gate_proj.weight"
    key_exp = "model.layers.3.mlp.experts.0.gate_proj.weight"
    t_mlp = torch.randn(32, 32, dtype=torch.bfloat16)
    t_exp = torch.randn(9, 5, dtype=torch.bfloat16)
    safetensors.torch.save_file({key_mlp: t_mlp}, str(file1))
    safetensors.torch.save_file({key_exp: t_exp}, str(file2))
    _write_index(model_dir, {key_mlp: file1.name, key_exp: file2.name})

    # Load lazily and create two different sub-views
    state = load_state_dict(model_dir, "")
    sub_mlp = sub_state_dict(state, "model.layers.3.mlp.")
    sub_exp = sub_state_dict(state, "model.layers.3.mlp.experts.0.")

    # Access the MLP key first, caching it
    v_mlp = sub_mlp["gate_proj.weight"]
    assert v_mlp.shape == t_mlp.shape

    # Then access the expert key; should not collide with MLP cache
    v_exp = sub_exp["gate_proj.weight"]
    assert v_exp.shape == t_exp.shape


def test_layer_filtering_in_view(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create safetensors with multiple layers and a non-layer key
    file1 = model_dir / "model-00001-of-00002.safetensors"
    file2 = model_dir / "model-00002-of-00002.safetensors"
    keys_file1 = {
        "model.layers.0.foo": torch.randn(1),
        "model.layers.1.foo": torch.randn(1),
        "lm_head.weight": torch.randn(2, 2),
    }
    keys_file2 = {
        "model.layers.2.foo": torch.randn(1),
    }
    safetensors.torch.save_file(keys_file1, str(file1))
    safetensors.torch.save_file(keys_file2, str(file2))
    _write_index(
        model_dir,
        {
            **{k: file1.name for k in keys_file1.keys()},
            **{k: file2.name for k in keys_file2.keys()},
        },
    )

    state = load_state_dict(model_dir, "")

    # Create a view that filters layers >= 2
    view = state.view_with_prefix("", num_layers=2)
    view_keys = set(view.keys())
    assert "model.layers.0.foo" in view_keys
    assert "model.layers.1.foo" in view_keys
    assert "lm_head.weight" in view_keys  # non-layer key should always pass
    assert "model.layers.2.foo" not in view_keys

    # Accessible keys load successfully
    _ = view["model.layers.0.foo"]
    _ = view["lm_head.weight"]

    # Filtered-out key should raise
    try:
        _ = view["model.layers.2.foo"]
        assert False, "Expected KeyError for filtered-out layer key"
    except KeyError:
        pass


def test_contains_and_len_reflect_filtering(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    file1 = model_dir / "a.safetensors"
    file2 = model_dir / "b.safetensors"
    safetensors.torch.save_file(
        {
            "model.layers.0.k": torch.randn(1),
            "model.layers.1.k": torch.randn(1),
            "lm_head.weight": torch.randn(2, 2),
        },
        str(file1),
    )
    safetensors.torch.save_file({"model.layers.2.k": torch.randn(1)}, str(file2))
    _write_index(
        model_dir,
        {
            "model.layers.0.k": file1.name,
            "model.layers.1.k": file1.name,
            "lm_head.weight": file1.name,
            "model.layers.2.k": file2.name,
        },
    )

    state = load_state_dict(model_dir, "")
    view = state.view_with_prefix("", num_layers=2)

    # __contains__
    assert "model.layers.0.k" in view
    assert "model.layers.1.k" in view
    assert "lm_head.weight" in view
    assert "model.layers.2.k" not in view

    # __len__
    assert len(view) == 3


def test_view_without_num_layers_includes_all(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    file1 = model_dir / "a.safetensors"
    safetensors.torch.save_file(
        {
            "model.layers.0.k": torch.randn(1),
            "model.layers.2.k": torch.randn(1),
        },
        str(file1),
    )
    _write_index(
        model_dir,
        {
            "model.layers.0.k": file1.name,
            "model.layers.2.k": file1.name,
        },
    )
    state = load_state_dict(model_dir, "")
    view = state.view_with_prefix("")  # no filtering
    keys = set(view.keys())
    assert "model.layers.0.k" in keys
    assert "model.layers.2.k" in keys


def test_cache_reuse_across_views_single_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    file1 = model_dir / "a.safetensors"
    full_key = "model.layers.0.w"
    t = torch.randn(3, 3)
    safetensors.torch.save_file({full_key: t}, str(file1))
    _write_index(model_dir, {full_key: file1.name})

    open_counts: dict[str, int] = {}
    original_safe_open = lsd.safe_open

    def counting_safe_open(path, *args, **kwargs):
        name = Path(path).name
        open_counts[name] = open_counts.get(name, 0) + 1
        return original_safe_open(path, *args, **kwargs)

    monkeypatch.setattr(lsd, "safe_open", counting_safe_open, raising=True)

    state = load_state_dict(model_dir, "")
    # Access via base (full key)
    _ = state[full_key]
    # Access via prefixed view
    sub = sub_state_dict(state, "model.layers.0.")
    _ = sub["w"]
    # Views must share underlying structures (no copying of cache or index)
    assert sub._cache is state._cache
    assert sub._full_to_file is state._full_to_file
    assert open_counts.get(file1.name, 0) == 1


def test_missing_file_raises(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Index points to a non-existent file
    missing = "missing.safetensors"
    key = "model.layers.0.q"
    _write_index(model_dir, {key: missing})

    state = load_state_dict(model_dir, "")
    with pytest.raises(KeyError):
        _ = state[key]


def test_handle_cache_reuses_shard_and_close_releases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    shard = model_dir / "a.safetensors"
    t1 = torch.randn(2, 2)
    t2 = torch.randn(3, 1)
    safetensors.torch.save_file({"w1": t1, "w2": t2}, str(shard))
    _write_index(model_dir, {"w1": shard.name, "w2": shard.name})

    # Count opens of the shard file
    open_counts: dict[str, int] = {}
    original_safe_open = lsd.safe_open

    def counting_safe_open(path, *args, **kwargs):
        name = Path(path).name
        open_counts[name] = open_counts.get(name, 0) + 1
        return original_safe_open(path, *args, **kwargs)

    monkeypatch.setattr(lsd, "safe_open", counting_safe_open, raising=True)

    state = load_state_dict(model_dir, "")

    # Access two keys in the same shard -> single open due to handle cache
    v1 = state["w1"]
    v2 = state["w2"]
    assert torch.equal(v1, t1)
    assert torch.equal(v2, t2)
    assert open_counts.get(shard.name, 0) == 1

    # Close handles, then access again -> reopens exactly once
    state.close()
    v1b = state["w1"]
    assert torch.equal(v1b, t1)
    assert open_counts.get(shard.name, 0) == 2


def test_non_numeric_layer_segment_not_filtered(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    file1 = model_dir / "a.safetensors"
    keys = {
        "model.layers.foo.bar": torch.randn(1),
        "model.layers.2.x": torch.randn(1),
        "misc.key": torch.randn(1),
    }
    safetensors.torch.save_file(keys, str(file1))
    _write_index(
        model_dir,
        {k: file1.name for k in keys.keys()},
    )

    state = load_state_dict(model_dir, "")
    view = state.view_with_prefix("", num_layers=2)
    keys_in_view = set(view.keys())

    # Non-numeric layer segment should not be filtered
    assert "model.layers.foo.bar" in keys_in_view

    # Numeric segment equal/above threshold should be filtered
    assert "model.layers.2.x" not in keys_in_view

    # Non-layer keys should always pass
    assert "misc.key" in keys_in_view

    # Access works for non-numeric segment
    _ = view["model.layers.foo.bar"]


def test_view_inherits_parent_layer_filter(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    file1 = model_dir / "a.safetensors"
    keys = {
        "model.layers.0.k": torch.randn(1),
        "model.layers.1.k": torch.randn(1),
        "model.layers.2.k": torch.randn(1),
    }
    safetensors.torch.save_file(keys, str(file1))
    _write_index(model_dir, {k: file1.name for k in keys.keys()})

    state = load_state_dict(model_dir, "")
    parent = state.view_with_prefix("", num_layers=2)  # allow layers 0,1 only
    child = parent.view_with_prefix("model.layers.")  # no num_layers argument -> inherit parent's

    child_keys = set(child.keys())
    assert "0.k" in child_keys
    assert "1.k" in child_keys
    assert "2.k" not in child_keys  # inherited filter still hides layer 2


def test_view_override_parent_layer_filter(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    file1 = model_dir / "a.safetensors"
    keys = {
        "model.layers.0.k": torch.randn(1),
        "model.layers.1.k": torch.randn(1),
        "model.layers.3.k": torch.randn(1),
    }
    safetensors.torch.save_file(keys, str(file1))
    _write_index(model_dir, {k: file1.name for k in keys.keys()})

    state = load_state_dict(model_dir, "")
    parent = state.view_with_prefix("", num_layers=2)  # allows 0,1

    # Tighter override
    child_tight = parent.view_with_prefix("model.layers.", num_layers=1)  # allows only 0
    tight_keys = set(child_tight.keys())
    assert "0.k" in tight_keys
    assert "1.k" not in tight_keys
    assert "3.k" not in tight_keys

    # Looser override
    child_loose = parent.view_with_prefix("model.layers.", num_layers=5)  # allows 0..4
    loose_keys = set(child_loose.keys())
    assert "0.k" in loose_keys
    assert "1.k" in loose_keys
    assert "3.k" in loose_keys
