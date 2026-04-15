#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
    get_weight_config,
    normalize_weight_config_paths,
    validate_weight_config_paths,
)

pytestmark = pytest.mark.t3k_compat


@dataclass(frozen=True)
class _FakeMeshDevice:
    shape: tuple[int, int]


def _make_hf_config(num_hidden_layers: int = 2) -> PretrainedConfig:
    hf_config = PretrainedConfig()
    hf_config.num_hidden_layers = num_hidden_layers
    return hf_config


def test_get_weight_config_direct_path_runs_convert_on_every_call(tmp_path: Path) -> None:
    call_count = {"n": 0}

    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, output_path: Path, mesh_device):
            call_count["n"] += 1
            return {"w0": {"call": call_count["n"], "output_path": str(output_path)}}

    mesh_device = _FakeMeshDevice(shape=(4, 8))
    hf_config = _make_hf_config(num_hidden_layers=3)
    base_cache = tmp_path / "weight_cache"

    cfg0 = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=hf_config,
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=base_cache,
        mesh_device=mesh_device,
        force_recalculate=False,
    )
    cfg1 = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=hf_config,
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=base_cache,
        mesh_device=mesh_device,
        force_recalculate=False,
    )

    compatibility_output_path = base_cache / "3_layers" / "mesh_4x8"
    assert call_count["n"] == 2
    assert cfg0["w0"]["call"] == 1
    assert cfg1["w0"]["call"] == 2
    assert not (compatibility_output_path / "config.json").exists()


def test_get_weight_config_path_construction_for_direct_weights(tmp_path: Path) -> None:
    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, output_path: Path, mesh_device):
            assert output_path == tmp_path / "weight_cache" / "5_layers" / "mesh_2x4"
            return {"ok": True}

    cfg = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=_make_hf_config(num_hidden_layers=5),
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=tmp_path / "weight_cache",
        mesh_device=_FakeMeshDevice(shape=(2, 4)),
    )

    assert cfg == {"ok": True}


def test_get_weight_config_allows_missing_weight_cache_path_for_direct_weights() -> None:
    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, output_path: Path, mesh_device):
            assert output_path == Path("generated/deepseek_v3") / "2_layers" / "mesh_1x1"
            return {"ok": True}

    cfg = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=_make_hf_config(),
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=None,
        mesh_device=_FakeMeshDevice(shape=(1, 1)),
    )

    assert cfg == {"ok": True}


def test_get_weight_config_loads_legacy_cache_when_requested(tmp_path: Path) -> None:
    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, output_path: Path, mesh_device):
            raise AssertionError("convert_weights should not be called when use_weight_cache=True")

    mesh_device = _FakeMeshDevice(shape=(4, 8))
    hf_config = _make_hf_config(num_hidden_layers=3)
    cache_root = tmp_path / "weight_cache"
    output_path = cache_root / "3_layers" / "mesh_4x8"
    weight_path = output_path / "weights" / f"w{TENSOR_CACHE_EXTENSION}"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_bytes(b"unit-test")
    (output_path / "config.json").write_text(
        json.dumps(
            {
                "w": {
                    "path": f"weights/w{TENSOR_CACHE_EXTENSION}",
                    "memory_config": json.loads(ttnn.DRAM_MEMORY_CONFIG.to_json()),
                }
            }
        ),
        encoding="utf-8",
    )

    cfg = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=hf_config,
        state_dicts=None,
        weight_cache_path=cache_root,
        mesh_device=mesh_device,
        use_weight_cache=True,
    )

    assert isinstance(cfg["w"], SavedWeight)
    assert cfg["w"].path == weight_path


def test_get_weight_config_use_weight_cache_requires_existing_cache(tmp_path: Path) -> None:
    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, output_path: Path, mesh_device):
            raise AssertionError("convert_weights should not be called when use_weight_cache=True")

    with pytest.raises(FileNotFoundError, match="Requested DeepSeek weight cache"):
        get_weight_config(
            ModuleClass=FakeModule,
            hf_config=_make_hf_config(num_hidden_layers=3),
            state_dicts=None,
            weight_cache_path=tmp_path / "weight_cache",
            mesh_device=_FakeMeshDevice(shape=(4, 8)),
            use_weight_cache=True,
        )


def test_get_weight_config_legacy_saved_weights_are_validated_and_normalized(tmp_path: Path) -> None:
    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, output_path: Path, mesh_device):
            rel_path = Path("weights") / f"w{TENSOR_CACHE_EXTENSION}"
            (output_path / rel_path).parent.mkdir(parents=True, exist_ok=True)
            (output_path / rel_path).write_bytes(b"unit-test")
            return {"w": SavedWeight(path=rel_path, memory_config=ttnn.DRAM_MEMORY_CONFIG)}

    cfg = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=_make_hf_config(num_hidden_layers=61),
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=tmp_path / "weight_cache",
        mesh_device=_FakeMeshDevice(shape=(8, 8)),
        cache_subdir_name="61_layers_mtp",
    )

    assert isinstance(cfg["w"], SavedWeight)
    assert cfg["w"].path.is_absolute()
    assert cfg["w"].path.exists()


def test_get_weight_config_rejects_absolute_paths_from_module(tmp_path: Path) -> None:
    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, output_path: Path, mesh_device):
            abs_path = output_path / "weights" / f"w{TENSOR_CACHE_EXTENSION}"
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_bytes(b"unit-test")
            return {"w": SavedWeight(path=abs_path, memory_config=ttnn.DRAM_MEMORY_CONFIG)}

    with pytest.raises(ValueError, match="absolute path"):
        get_weight_config(
            ModuleClass=FakeModule,
            hf_config=_make_hf_config(num_hidden_layers=3),
            state_dicts=({"dummy": torch.empty(1)},),
            weight_cache_path=tmp_path / "weight_cache",
            mesh_device=_FakeMeshDevice(shape=(4, 8)),
        )


def test_get_weight_config_with_relative_base_cache_path_resolves_output_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, output_path: Path, mesh_device):
            assert output_path.is_absolute()
            rel_path = Path("weights") / f"w{TENSOR_CACHE_EXTENSION}"
            (output_path / rel_path).parent.mkdir(parents=True, exist_ok=True)
            (output_path / rel_path).write_bytes(b"unit-test")
            return {"w": SavedWeight(path=rel_path, memory_config=ttnn.DRAM_MEMORY_CONFIG)}

    monkeypatch.chdir(tmp_path)

    cfg = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=_make_hf_config(num_hidden_layers=61),
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=Path("relative_weight_cache"),
        mesh_device=_FakeMeshDevice(shape=(4, 8)),
    )

    assert cfg["w"].path.is_absolute()
    assert cfg["w"].path.exists()


def test_get_weight_config_error_missing_mesh_device() -> None:
    with pytest.raises(ValueError, match="mesh_device must be provided"):
        get_weight_config(
            ModuleClass=None,
            hf_config=_make_hf_config(),
            weight_cache_path=Path("/tmp"),
            mesh_device=None,
        )


def test_validate_weight_config_paths_success(tmp_path: Path) -> None:
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

    validate_weight_config_paths(root_path, weight_config)


def test_validate_weight_config_paths_missing_file(tmp_path: Path) -> None:
    root_path = tmp_path / "weights"
    root_path.mkdir(parents=True)

    weight_config = {
        "layer1": SavedWeight(path=Path(f"missing{TENSOR_CACHE_EXTENSION}"), memory_config=ttnn.DRAM_MEMORY_CONFIG),
    }

    with pytest.raises(ValueError, match="references missing file"):
        validate_weight_config_paths(root_path, weight_config)


def test_validate_weight_config_paths_wrong_suffix(tmp_path: Path) -> None:
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
    root_path = tmp_path / "weights"
    root_path.mkdir(parents=True)

    abs_path = tmp_path / "other" / f"w{TENSOR_CACHE_EXTENSION}"
    abs_path.parent.mkdir(parents=True)
    abs_path.write_bytes(b"test")

    weight_config = {
        "layer1": SavedWeight(path=abs_path, memory_config=ttnn.DRAM_MEMORY_CONFIG),
    }

    with pytest.raises(ValueError, match="absolute path"):
        validate_weight_config_paths(root_path, weight_config)


def test_validate_weight_config_paths_nested_structures(tmp_path: Path) -> None:
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
    }

    validate_weight_config_paths(root_path, weight_config)


def test_validate_weight_config_paths_error_path_prefix(tmp_path: Path) -> None:
    root_path = tmp_path / "weights"
    root_path.mkdir(parents=True)

    weight_config = {
        "outer": {
            "inner": SavedWeight(path=Path(f"missing{TENSOR_CACHE_EXTENSION}"), memory_config=ttnn.DRAM_MEMORY_CONFIG),
        }
    }

    with pytest.raises(ValueError, match="outer.inner"):
        validate_weight_config_paths(root_path, weight_config)


def test_normalize_weight_config_paths_relative_paths(tmp_path: Path) -> None:
    root_path = tmp_path / "weights"
    rel_path = Path(f"w{TENSOR_CACHE_EXTENSION}")
    weight_config = {
        "w": SavedWeight(path=rel_path, memory_config=ttnn.DRAM_MEMORY_CONFIG),
    }

    normalized = normalize_weight_config_paths(root_path, weight_config)

    assert normalized["w"].path == root_path / rel_path
    assert normalized["w"].memory_config == ttnn.DRAM_MEMORY_CONFIG


def test_normalize_weight_config_paths_absolute_paths(tmp_path: Path) -> None:
    abs_path = tmp_path / "other" / f"w{TENSOR_CACHE_EXTENSION}"
    weight_config = {
        "w": SavedWeight(path=abs_path, memory_config=ttnn.DRAM_MEMORY_CONFIG),
    }

    normalized = normalize_weight_config_paths(tmp_path / "weights", weight_config)

    assert normalized["w"].path == abs_path


def test_normalize_weight_config_paths_nested_structures(tmp_path: Path) -> None:
    root_path = tmp_path / "weights"
    rel_path = Path(f"w{TENSOR_CACHE_EXTENSION}")
    weight_config = {
        "dict_nested": {
            "w": SavedWeight(path=rel_path, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        },
        "list_nested": [
            SavedWeight(path=rel_path, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ],
        "tuple_nested": (SavedWeight(path=rel_path, memory_config=ttnn.DRAM_MEMORY_CONFIG),),
    }

    normalized = normalize_weight_config_paths(root_path, weight_config)

    assert normalized["dict_nested"]["w"].path == root_path / rel_path
    assert normalized["list_nested"][0].path == root_path / rel_path
    assert normalized["tuple_nested"][0].path == root_path / rel_path
