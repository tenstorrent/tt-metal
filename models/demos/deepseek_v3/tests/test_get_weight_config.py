#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.config_dataclass import SavedWeight
from models.demos.deepseek_v3.utils.config_helpers import TENSOR_CACHE_EXTENSION
from models.demos.deepseek_v3.utils.weight_config import (
    WeightConfigEncoder,
    get_weight_config,
    normalize_weight_config_paths,
    validate_weight_config_paths,
)


@dataclass(frozen=True)
class _FakeMeshDevice:
    # `get_weight_config()` only uses `.shape` for path construction and passes the object through.
    shape: tuple[int, int]


def _make_hf_config(num_hidden_layers: int = 2) -> PretrainedConfig:
    hf_config = PretrainedConfig()
    hf_config.num_hidden_layers = num_hidden_layers
    return hf_config


def test_get_weight_config_cache_hit_skips_convert(tmp_path: Path) -> None:
    call_count = {"n": 0}

    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, weight_cache_path: Path, mesh_device):
            call_count["n"] += 1
            (weight_cache_path / "weights").mkdir(parents=True, exist_ok=True)
            rel_path = Path("weights") / f"w0{TENSOR_CACHE_EXTENSION}"
            (weight_cache_path / rel_path).write_bytes(b"unit-test")
            return {"w0": SavedWeight(path=rel_path, memory_config=ttnn.DRAM_MEMORY_CONFIG)}

    mesh_device = _FakeMeshDevice(shape=(4, 8))
    hf_config = _make_hf_config(num_hidden_layers=3)
    base_cache = tmp_path / "weight_cache"

    # First call: cache miss -> convert_weights is invoked and config.json is written.
    cfg0 = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=hf_config,
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=base_cache,
        mesh_device=mesh_device,
        force_recalculate=False,
    )
    assert call_count["n"] == 1
    assert isinstance(cfg0, dict) and "w0" in cfg0
    assert isinstance(cfg0["w0"], SavedWeight)
    assert cfg0["w0"].path.is_absolute()
    assert cfg0["w0"].path.exists()

    # Second call: cache hit -> convert_weights is skipped.
    cfg1 = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=hf_config,
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=base_cache,
        mesh_device=mesh_device,
        force_recalculate=False,
    )
    assert call_count["n"] == 1
    assert isinstance(cfg1["w0"], SavedWeight)
    assert cfg1["w0"].path.is_absolute()
    assert cfg1["w0"].path.exists()


def test_get_weight_config_force_recalculate_bypasses_cache(tmp_path: Path) -> None:
    call_count = {"n": 0}

    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, weight_cache_path: Path, mesh_device):
            call_count["n"] += 1
            (weight_cache_path / "weights").mkdir(parents=True, exist_ok=True)
            rel_path = Path("weights") / f"w{call_count['n']}{TENSOR_CACHE_EXTENSION}"
            (weight_cache_path / rel_path).write_bytes(f"unit-test-{call_count['n']}".encode("utf-8"))
            return {f"w{call_count['n']}": SavedWeight(path=rel_path, memory_config=ttnn.DRAM_MEMORY_CONFIG)}

    mesh_device = _FakeMeshDevice(shape=(1, 2))
    hf_config = _make_hf_config(num_hidden_layers=1)
    base_cache = tmp_path / "weight_cache"

    cfg0 = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=hf_config,
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=base_cache,
        mesh_device=mesh_device,
        force_recalculate=False,
    )
    assert call_count["n"] == 1

    cfg1 = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=hf_config,
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=base_cache,
        mesh_device=mesh_device,
        force_recalculate=True,
    )
    assert call_count["n"] == 2

    # When force_recalculate=True, we should get the newly produced config/weight path.
    assert cfg0 != cfg1


def test_get_weight_config_cache_miss_when_config_missing(tmp_path: Path) -> None:
    """Test that cache miss occurs when config.json doesn't exist."""
    call_count = {"n": 0}

    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, weight_cache_path: Path, mesh_device):
            call_count["n"] += 1
            (weight_cache_path / "weights").mkdir(parents=True, exist_ok=True)
            rel_path = Path("weights") / f"w{TENSOR_CACHE_EXTENSION}"
            (weight_cache_path / rel_path).write_bytes(b"unit-test")
            return {"w": SavedWeight(path=rel_path, memory_config=ttnn.DRAM_MEMORY_CONFIG)}

    mesh_device = _FakeMeshDevice(shape=(2, 4))
    hf_config = _make_hf_config(num_hidden_layers=5)
    base_cache = tmp_path / "weight_cache"

    # No cache exists, should call convert_weights
    cfg = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=hf_config,
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=base_cache,
        mesh_device=mesh_device,
        force_recalculate=False,
    )
    assert call_count["n"] == 1
    assert cfg["w"].path.is_absolute()
    assert cfg["w"].path.exists()


def test_get_weight_config_cache_invalidation_missing_weight_file(tmp_path: Path) -> None:
    """Test that cache is invalidated when weight file is missing."""
    call_count = {"n": 0}

    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, weight_cache_path: Path, mesh_device):
            call_count["n"] += 1
            (weight_cache_path / "weights").mkdir(parents=True, exist_ok=True)
            rel_path = Path("weights") / f"w{TENSOR_CACHE_EXTENSION}"
            (weight_cache_path / rel_path).write_bytes(b"unit-test")
            return {"w": SavedWeight(path=rel_path, memory_config=ttnn.DRAM_MEMORY_CONFIG)}

    mesh_device = _FakeMeshDevice(shape=(1, 1))
    hf_config = _make_hf_config(num_hidden_layers=1)
    base_cache = tmp_path / "weight_cache"

    # First call: create cache
    cfg0 = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=hf_config,
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=base_cache,
        mesh_device=mesh_device,
        force_recalculate=False,
    )
    assert call_count["n"] == 1

    # Delete the weight file but keep config.json
    # config.json has relative paths, so we should resolve them relative to weight_cache_path
    # This simulates what validate_weight_config_paths does
    weight_cache_path = (
        base_cache / f"{hf_config.num_hidden_layers}_layers" / f"mesh_{mesh_device.shape[0]}x{mesh_device.shape[1]}"
    )
    config_path = weight_cache_path / "config.json"

    # Read the config.json to get the relative path (as it's actually stored)
    with config_path.open() as f:
        saved_config = json.load(f)
    rel_path_str = saved_config["w"]["path"]
    assert not Path(rel_path_str).is_absolute(), "Config should store relative paths"

    # Resolve relative path to get the actual file location
    weight_file = weight_cache_path / rel_path_str
    assert weight_file.exists(), f"Weight file should exist: {weight_file}"
    weight_file.unlink()
    assert not weight_file.exists()

    # Second call: cache validation should fail, triggering recalculation
    cfg1 = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=hf_config,
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=base_cache,
        mesh_device=mesh_device,
        force_recalculate=False,
    )
    assert call_count["n"] == 2
    assert cfg1["w"].path.exists()


def test_get_weight_config_cache_invalidation_wrong_suffix(tmp_path: Path) -> None:
    """Test that cache is invalidated when weight file has wrong suffix."""
    call_count = {"n": 0}

    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, weight_cache_path: Path, mesh_device):
            call_count["n"] += 1
            (weight_cache_path / "weights").mkdir(parents=True, exist_ok=True)
            rel_path = Path("weights") / f"w{TENSOR_CACHE_EXTENSION}"
            (weight_cache_path / rel_path).write_bytes(b"unit-test")
            return {"w": SavedWeight(path=rel_path, memory_config=ttnn.DRAM_MEMORY_CONFIG)}

    mesh_device = _FakeMeshDevice(shape=(2, 2))
    hf_config = _make_hf_config(num_hidden_layers=2)
    base_cache = tmp_path / "weight_cache"

    # First call: create cache
    cfg0 = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=hf_config,
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=base_cache,
        mesh_device=mesh_device,
        force_recalculate=False,
    )
    assert call_count["n"] == 1

    # Corrupt config.json to have wrong suffix
    # Create a SavedWeight with wrong suffix - this will fail validation
    weight_cache_path = (
        base_cache / f"{hf_config.num_hidden_layers}_layers" / f"mesh_{mesh_device.shape[0]}x{mesh_device.shape[1]}"
    )
    config_path = weight_cache_path / "config.json"
    bad_weight = SavedWeight(path=Path("weights/w.bad"), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    bad_config = {"w": bad_weight}
    with config_path.open("w") as f:
        json.dump(bad_config, f, cls=WeightConfigEncoder)

    # Second call: cache validation should fail, triggering recalculation
    cfg1 = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=hf_config,
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=base_cache,
        mesh_device=mesh_device,
        force_recalculate=False,
    )
    assert call_count["n"] == 2


def test_get_weight_config_path_construction(tmp_path: Path) -> None:
    """Test that weight_cache_path is correctly constructed with layers and mesh shape."""

    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, weight_cache_path: Path, mesh_device):
            # Verify the path structure
            expected_suffix = f"{hf_config.num_hidden_layers}_layers/mesh_{mesh_device.shape[0]}x{mesh_device.shape[1]}"
            assert str(weight_cache_path).endswith(expected_suffix)
            (weight_cache_path / "weights").mkdir(parents=True, exist_ok=True)
            rel_path = Path("weights") / f"w{TENSOR_CACHE_EXTENSION}"
            (weight_cache_path / rel_path).write_bytes(b"unit-test")
            return {"w": SavedWeight(path=rel_path, memory_config=ttnn.DRAM_MEMORY_CONFIG)}

    mesh_device = _FakeMeshDevice(shape=(8, 16))
    hf_config = _make_hf_config(num_hidden_layers=12)
    base_cache = tmp_path / "weight_cache"

    cfg = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=hf_config,
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=base_cache,
        mesh_device=mesh_device,
        force_recalculate=False,
    )
    assert cfg["w"].path.is_absolute()


def test_get_weight_config_rejects_absolute_paths_from_module(tmp_path: Path) -> None:
    """Test that get_weight_config rejects absolute paths returned by ModuleClass.convert_weights."""

    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, weight_cache_path: Path, mesh_device):
            # Return an absolute path (incorrect behavior)
            (weight_cache_path / "weights").mkdir(parents=True, exist_ok=True)
            abs_path = weight_cache_path / "weights" / f"w{TENSOR_CACHE_EXTENSION}"
            abs_path.write_bytes(b"unit-test")
            # Return absolute path instead of relative
            return {"w": SavedWeight(path=abs_path, memory_config=ttnn.DRAM_MEMORY_CONFIG)}

    mesh_device = _FakeMeshDevice(shape=(2, 2))
    hf_config = _make_hf_config(num_hidden_layers=2)
    base_cache = tmp_path / "weight_cache"

    # get_weight_config should raise ValueError when convert_weights returns absolute paths
    with pytest.raises(ValueError, match="absolute path.*Only relative paths are allowed"):
        get_weight_config(
            ModuleClass=FakeModule,
            hf_config=hf_config,
            state_dicts=({"dummy": torch.empty(1)},),
            weight_cache_path=base_cache,
            mesh_device=mesh_device,
            force_recalculate=False,
        )


def test_get_weight_config_error_missing_weight_cache_path() -> None:
    """Test that ValueError is raised when weight_cache_path is None."""
    mesh_device = _FakeMeshDevice(shape=(1, 1))
    hf_config = _make_hf_config()

    with pytest.raises(ValueError, match="weight_cache_path must be provided"):
        get_weight_config(
            ModuleClass=None,  # Won't get this far
            hf_config=hf_config,
            weight_cache_path=None,
            mesh_device=mesh_device,
        )


def test_get_weight_config_error_missing_mesh_device() -> None:
    """Test that ValueError is raised when mesh_device is None."""
    hf_config = _make_hf_config()

    with pytest.raises(ValueError, match="mesh_device must be provided"):
        get_weight_config(
            ModuleClass=None,  # Won't get this far
            hf_config=hf_config,
            weight_cache_path=Path("/tmp"),
            mesh_device=None,
        )


def test_validate_weight_config_paths_success(tmp_path: Path) -> None:
    """Test successful validation of weight config paths."""
    root_path = tmp_path / "weights"
    root_path.mkdir(parents=True)

    weight_file1 = root_path / f"w1{TENSOR_CACHE_EXTENSION}"
    weight_file2 = root_path / f"w2{TENSOR_CACHE_EXTENSION}"
    weight_file1.write_bytes(b"test1")
    weight_file2.write_bytes(b"test2")

    weight_config = {
        "layer1": SavedWeight(path=Path(f"w1{TENSOR_CACHE_EXTENSION}"), memory_config=ttnn.DRAM_MEMORY_CONFIG),
        "layer2": SavedWeight(path=Path(f"w2{TENSOR_CACHE_EXTENSION}"), memory_config=ttnn.DRAM_MEMORY_CONFIG),
    }

    # Should not raise - both paths are relative
    validate_weight_config_paths(root_path, weight_config)


def test_validate_weight_config_paths_missing_file(tmp_path: Path) -> None:
    """Test validation raises ValueError when weight file is missing."""
    root_path = tmp_path / "weights"
    root_path.mkdir(parents=True)

    weight_config = {
        "layer1": SavedWeight(path=Path(f"missing{TENSOR_CACHE_EXTENSION}"), memory_config=ttnn.DRAM_MEMORY_CONFIG),
    }

    with pytest.raises(ValueError, match="references missing file"):
        validate_weight_config_paths(root_path, weight_config)


def test_validate_weight_config_paths_wrong_suffix(tmp_path: Path) -> None:
    """Test validation raises ValueError when weight file has wrong suffix."""
    root_path = tmp_path / "weights"
    root_path.mkdir(parents=True)

    weight_file = root_path / "w.bad"
    weight_file.write_bytes(b"test")

    weight_config = {
        "layer1": SavedWeight(path=Path("w.bad"), memory_config=ttnn.DRAM_MEMORY_CONFIG),
    }

    with pytest.raises(ValueError, match="invalid suffix"):
        validate_weight_config_paths(root_path, weight_config)


def test_validate_weight_config_paths_rejects_absolute_paths(tmp_path: Path) -> None:
    """Test validation raises ValueError when SavedWeight has absolute path."""
    root_path = tmp_path / "weights"
    root_path.mkdir(parents=True)

    abs_path = tmp_path / "other" / f"w{TENSOR_CACHE_EXTENSION}"
    abs_path.parent.mkdir(parents=True)
    abs_path.write_bytes(b"test")

    weight_config = {
        "layer1": SavedWeight(path=abs_path, memory_config=ttnn.DRAM_MEMORY_CONFIG),
    }

    with pytest.raises(ValueError, match="absolute path.*Only relative paths are allowed"):
        validate_weight_config_paths(root_path, weight_config)


def test_validate_weight_config_paths_nested_structures(tmp_path: Path) -> None:
    """Test validation works with nested dicts, lists, and tuples."""
    root_path = tmp_path / "weights"
    root_path.mkdir(parents=True)

    weight_file1 = root_path / f"w1{TENSOR_CACHE_EXTENSION}"
    weight_file2 = root_path / f"w2{TENSOR_CACHE_EXTENSION}"
    weight_file3 = root_path / f"w3{TENSOR_CACHE_EXTENSION}"
    weight_file1.write_bytes(b"test1")
    weight_file2.write_bytes(b"test2")
    weight_file3.write_bytes(b"test3")

    weight_config = {
        "dict_nested": {
            "w1": SavedWeight(path=Path(f"w1{TENSOR_CACHE_EXTENSION}"), memory_config=ttnn.DRAM_MEMORY_CONFIG),
        },
        "list_nested": [
            SavedWeight(path=Path(f"w2{TENSOR_CACHE_EXTENSION}"), memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ],
        "tuple_nested": (SavedWeight(path=Path(f"w3{TENSOR_CACHE_EXTENSION}"), memory_config=ttnn.DRAM_MEMORY_CONFIG),),
        "none_value": None,
    }

    # Should not raise
    validate_weight_config_paths(root_path, weight_config)


def test_validate_weight_config_paths_error_path_prefix(tmp_path: Path) -> None:
    """Test that error messages include path prefix for nested structures."""
    root_path = tmp_path / "weights"
    root_path.mkdir(parents=True)

    weight_config = {
        "outer": {
            "inner": SavedWeight(path=Path(f"missing{TENSOR_CACHE_EXTENSION}"), memory_config=ttnn.DRAM_MEMORY_CONFIG),
        },
    }

    with pytest.raises(ValueError, match="outer.inner"):
        validate_weight_config_paths(root_path, weight_config)


def test_normalize_weight_config_paths_relative_paths(tmp_path: Path) -> None:
    """Test normalization converts relative paths to absolute."""
    root_path = tmp_path / "weights"
    root_path.mkdir(parents=True)

    rel_path = Path(f"w{TENSOR_CACHE_EXTENSION}")
    weight_config = {
        "w": SavedWeight(path=rel_path, memory_config=ttnn.DRAM_MEMORY_CONFIG),
    }

    normalized = normalize_weight_config_paths(root_path, weight_config)

    assert normalized["w"].path.is_absolute()
    assert normalized["w"].path == root_path / rel_path
    # Original should be unchanged (no mutation)
    assert not weight_config["w"].path.is_absolute()


def test_normalize_weight_config_paths_absolute_paths(tmp_path: Path) -> None:
    """Test normalization preserves absolute paths."""
    root_path = tmp_path / "weights"
    root_path.mkdir(parents=True)

    abs_path = tmp_path / "other" / f"w{TENSOR_CACHE_EXTENSION}"
    abs_path.parent.mkdir(parents=True)
    abs_path.write_bytes(b"test")

    weight_config = {
        "w": SavedWeight(path=abs_path, memory_config=ttnn.DRAM_MEMORY_CONFIG),
    }

    normalized = normalize_weight_config_paths(root_path, weight_config)

    assert normalized["w"].path.is_absolute()
    assert normalized["w"].path == abs_path


def test_normalize_weight_config_paths_nested_structures(tmp_path: Path) -> None:
    """Test normalization works with nested dicts, lists, and tuples."""
    root_path = tmp_path / "weights"
    root_path.mkdir(parents=True)

    rel_path = Path(f"w{TENSOR_CACHE_EXTENSION}")
    weight_config = {
        "dict_nested": {
            "w": SavedWeight(path=rel_path, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        },
        "list_nested": [
            SavedWeight(path=rel_path, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ],
        "tuple_nested": (SavedWeight(path=rel_path, memory_config=ttnn.DRAM_MEMORY_CONFIG),),
        "none_value": None,
    }

    normalized = normalize_weight_config_paths(root_path, weight_config)

    assert normalized["dict_nested"]["w"].path.is_absolute()
    assert normalized["list_nested"][0].path.is_absolute()
    assert normalized["tuple_nested"][0].path.is_absolute()
    assert normalized["none_value"] is None
    # Verify tuple type is preserved
    assert isinstance(normalized["tuple_nested"], tuple)
    assert isinstance(normalized["list_nested"], list)


def test_get_weight_config_returns_normalized_paths(tmp_path: Path) -> None:
    """Test that get_weight_config returns config with absolute paths even though config.json has relative."""

    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, weight_cache_path: Path, mesh_device):
            (weight_cache_path / "weights").mkdir(parents=True, exist_ok=True)
            rel_path = Path("weights") / f"w{TENSOR_CACHE_EXTENSION}"
            (weight_cache_path / rel_path).write_bytes(b"unit-test")
            return {"w": SavedWeight(path=rel_path, memory_config=ttnn.DRAM_MEMORY_CONFIG)}

    mesh_device = _FakeMeshDevice(shape=(1, 1))
    hf_config = _make_hf_config(num_hidden_layers=1)
    base_cache = tmp_path / "weight_cache"

    cfg = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=hf_config,
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=base_cache,
        mesh_device=mesh_device,
        force_recalculate=False,
    )

    # Returned config should have absolute paths
    assert cfg["w"].path.is_absolute()

    # But config.json should have relative paths (check by reading it)
    weight_cache_path = (
        base_cache / f"{hf_config.num_hidden_layers}_layers" / f"mesh_{mesh_device.shape[0]}x{mesh_device.shape[1]}"
    )
    config_path = weight_cache_path / "config.json"
    with config_path.open() as f:
        saved_config = json.load(f)
    # The saved path should be relative (as a string)
    assert "weights" in saved_config["w"]["path"]
    assert not Path(saved_config["w"]["path"]).is_absolute()
