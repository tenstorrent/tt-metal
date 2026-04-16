#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import safetensors.torch
import torch
from transformers.configuration_utils import PretrainedConfig

import models.demos.deepseek_v3.utils.config_helpers as config_helpers_module
import models.demos.deepseek_v3.utils.test_utils as test_utils_module
import models.demos.deepseek_v3.utils.weight_config as weight_config_module
import ttnn
from models.demos.deepseek_v3.scripts.validate_weight_cache import repair_cache_directory, validate_cache_directory
from models.demos.deepseek_v3.utils.config_dataclass import SavedWeight
from models.demos.deepseek_v3.utils.config_helpers import TENSOR_CACHE_EXTENSION, sub_state_dict
from models.demos.deepseek_v3.utils.weight_config import (
    InvalidWeightCacheError,
    get_weight_config,
    normalize_weight_config_paths,
    validate_weight_config_paths,
    wrap_weight_cache_payload,
)

pytestmark = pytest.mark.t3k_compat


@dataclass(frozen=True)
class _FakeMeshDevice:
    shape: tuple[int, int]

    def get_num_devices(self) -> int:
        return self.shape[0] * self.shape[1]


def _make_hf_config(num_hidden_layers: int = 2) -> PretrainedConfig:
    hf_config = PretrainedConfig()
    hf_config.num_hidden_layers = num_hidden_layers
    return hf_config


def _write_index(model_dir: Path, weight_map: dict[str, str]) -> None:
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )


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


def test_get_weight_config_direct_path_handles_bspm_wrapped_stacked_lazy_state_dict(tmp_path: Path) -> None:
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _BsprnStateDict
    from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict

    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    shard = model_dir / "model-00001-of-00001.safetensors"
    tensors = {
        f"model.layers.3.mlp.experts_stacked.{proj_name}.weight": (
            torch.arange(2 * 32 * 32, dtype=torch.float32).reshape(2, 32, 32) + offset
        ).to(torch.bfloat16)
        for offset, proj_name in enumerate(("gate_proj", "down_proj", "up_proj"), start=1)
    }
    safetensors.torch.save_file(tensors, str(shard))
    _write_index(model_dir, {key: shard.name for key in tensors})

    wrapped_state_dict = _BsprnStateDict(
        LazyStateDict(model_dir),
        SimpleNamespace(first_k_dense_replace=3, n_routed_experts=2),
        tmp_path / "results",
        "unused",
        "B",
        3.5,
        _codes_cache={3: np.array([[[0], [1], [1]], [[1], [1], [1]]], dtype=np.int32)},
        _load_bspm=lambda _path, expected_n_experts=None: {"codes": np.zeros((2, 3, 1), dtype=np.int32)},
        _qdq=lambda tensor, mant_bits: tensor + mant_bits,
        _require_complete=True,
    )

    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, output_path: Path, mesh_device):
            mlp_state = sub_state_dict(state_dicts[0], "model.layers.3.mlp.")
            stacked_view = mlp_state.view_with_prefix("experts_stacked.")
            gate_weight = stacked_view["gate_proj.weight"]
            assert torch.equal(gate_weight[0], tensors["model.layers.3.mlp.experts_stacked.gate_proj.weight"][0] + 7)
            assert torch.equal(gate_weight[1], tensors["model.layers.3.mlp.experts_stacked.gate_proj.weight"][1])
            return {"ok": True}

    cfg = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=_make_hf_config(num_hidden_layers=5),
        state_dicts=(wrapped_state_dict,),
        weight_cache_path=tmp_path / "weight_cache",
        mesh_device=_FakeMeshDevice(shape=(1, 1)),
    )

    assert cfg == {"ok": True}


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


def test_get_weight_config_emit_weight_cache_force_recalculate_clears_existing_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    emitted_weight_config = {"w0": "direct"}
    seen: dict[str, object] = {}

    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, output_path: Path, mesh_device):
            return emitted_weight_config

    output_path = tmp_path / "weight_cache" / "3_layers" / "mesh_4x8"
    stale_file = output_path / "stale" / f"leftover{TENSOR_CACHE_EXTENSION}"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_bytes(b"stale")

    def fake_save_weight_config(root_path: Path, weight_config):
        assert not stale_file.exists()
        seen["root_path"] = root_path
        seen["weight_config"] = weight_config
        return {"saved": True}

    def fake_deallocate_weight_config_tensors(weight_config):
        seen["deallocated"] = weight_config

    monkeypatch.setattr(weight_config_module, "save_weight_config", fake_save_weight_config)
    monkeypatch.setattr(weight_config_module, "deallocate_weight_config_tensors", fake_deallocate_weight_config_tensors)

    cfg = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=_make_hf_config(num_hidden_layers=3),
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=tmp_path / "weight_cache",
        mesh_device=_FakeMeshDevice(shape=(4, 8)),
        force_recalculate=True,
        emit_weight_cache=True,
    )

    assert cfg == {"saved": True}
    assert seen["root_path"] == output_path
    assert seen["weight_config"] is emitted_weight_config
    assert seen["deallocated"] is emitted_weight_config


def test_get_weight_config_emit_weight_cache_streams_saved_weights_from_shard_and_save(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _FakeTTTensor:
        def __init__(self, memory_config):
            self._memory_config = memory_config

        def memory_config(self):
            return self._memory_config

    dumped_paths: list[Path] = []
    deallocated_tensors: list[object] = []

    def fake_shard_device_impl(**kwargs):
        return _FakeTTTensor(kwargs["memory_config"])

    def fake_dump_tensor(path: Path, tensor):
        dumped_paths.append(Path(path))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"unit-test")

    def fake_deallocate(tensor):
        deallocated_tensors.append(tensor)

    monkeypatch.setattr(config_helpers_module, "_shard_device_impl", fake_shard_device_impl)
    monkeypatch.setattr(config_helpers_module.ttnn, "dump_tensor", fake_dump_tensor)
    monkeypatch.setattr(config_helpers_module.ttnn, "deallocate", fake_deallocate)

    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, output_path: Path, mesh_device):
            return {
                "w": config_helpers_module.shard_and_save(
                    output_path / "weights" / "w",
                    torch.ones((1, 1, 32, 32), dtype=torch.bfloat16),
                    shard_dims=(None, None),
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            }

    cfg = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=_make_hf_config(num_hidden_layers=3),
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=tmp_path / "weight_cache",
        mesh_device=_FakeMeshDevice(shape=(4, 8)),
        force_recalculate=True,
        emit_weight_cache=True,
    )

    output_path = tmp_path / "weight_cache" / "3_layers" / "mesh_4x8"
    config_payload = json.loads((output_path / "config.json").read_text(encoding="utf-8"))

    assert dumped_paths == [output_path / "weights" / f"w{TENSOR_CACHE_EXTENSION}"]
    assert len(deallocated_tensors) == 1
    assert weight_config_module.WEIGHT_CACHE_METADATA_KEY in config_payload
    assert isinstance(cfg["w"], SavedWeight)
    assert cfg["w"].path == output_path / "weights" / f"w{TENSOR_CACHE_EXTENSION}"


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
            wrap_weight_cache_payload(
                {
                    "w": {
                        "path": f"weights/w{TENSOR_CACHE_EXTENSION}",
                        "memory_config": json.loads(ttnn.DRAM_MEMORY_CONFIG.to_json()),
                    }
                }
            )
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


def test_get_weight_config_use_weight_cache_rejects_invalid_cache(tmp_path: Path) -> None:
    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, output_path: Path, mesh_device):
            raise AssertionError("convert_weights should not be called when use_weight_cache=True")

    cache_root = tmp_path / "weight_cache"
    output_path = cache_root / "3_layers" / "mesh_4x8"
    output_path.mkdir(parents=True)
    (output_path / "config.json").write_text(
        json.dumps(
            wrap_weight_cache_payload(
                {
                    "w": {
                        "path": f"weights/missing{TENSOR_CACHE_EXTENSION}",
                        "memory_config": json.loads(ttnn.DRAM_MEMORY_CONFIG.to_json()),
                    }
                }
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(InvalidWeightCacheError, match="Requested DeepSeek weight cache was invalid"):
        get_weight_config(
            ModuleClass=FakeModule,
            hf_config=_make_hf_config(num_hidden_layers=3),
            state_dicts=None,
            weight_cache_path=cache_root,
            mesh_device=_FakeMeshDevice(shape=(4, 8)),
            use_weight_cache=True,
        )


def test_get_weight_config_use_weight_cache_rejects_unversioned_legacy_cache(tmp_path: Path) -> None:
    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, output_path: Path, mesh_device):
            raise AssertionError("convert_weights should not be called when use_weight_cache=True")

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

    with pytest.raises(InvalidWeightCacheError, match="missing deepseek_weight_cache_metadata"):
        get_weight_config(
            ModuleClass=FakeModule,
            hf_config=_make_hf_config(num_hidden_layers=3),
            state_dicts=None,
            weight_cache_path=cache_root,
            mesh_device=_FakeMeshDevice(shape=(4, 8)),
            use_weight_cache=True,
        )


def test_get_weight_config_use_weight_cache_rejects_outdated_wrapped_cache_version(tmp_path: Path) -> None:
    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, output_path: Path, mesh_device):
            raise AssertionError("convert_weights should not be called when use_weight_cache=True")

    cache_root = tmp_path / "weight_cache"
    output_path = cache_root / "3_layers" / "mesh_4x8"
    weight_path = output_path / "weights" / f"w{TENSOR_CACHE_EXTENSION}"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_bytes(b"unit-test")
    (output_path / "config.json").write_text(
        json.dumps(
            {
                weight_config_module.WEIGHT_CACHE_METADATA_KEY: {
                    weight_config_module.WEIGHT_CACHE_VERSION_KEY: weight_config_module.WEIGHT_CACHE_FORMAT_VERSION - 1,
                },
                weight_config_module.WEIGHT_CACHE_PAYLOAD_KEY: {
                    "w": {
                        "path": f"weights/w{TENSOR_CACHE_EXTENSION}",
                        "memory_config": json.loads(ttnn.DRAM_MEMORY_CONFIG.to_json()),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InvalidWeightCacheError, match="unsupported format version"):
        get_weight_config(
            ModuleClass=FakeModule,
            hf_config=_make_hf_config(num_hidden_layers=3),
            state_dicts=None,
            weight_cache_path=cache_root,
            mesh_device=_FakeMeshDevice(shape=(4, 8)),
            use_weight_cache=True,
        )


def test_validate_weight_cache_directory_rejects_unversioned_legacy_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "weight_cache" / "3_layers" / "mesh_4x8"
    weight_path = cache_dir / "weights" / f"w{TENSOR_CACHE_EXTENSION}"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_bytes(b"unit-test")
    (cache_dir / "config.json").write_text(
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

    result = validate_cache_directory(cache_dir)

    assert result["valid"] is False
    assert any("missing deepseek_weight_cache_metadata" in error for error in result["errors"])


def test_repair_weight_cache_directory_rejects_unversioned_legacy_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "weight_cache" / "3_layers" / "mesh_4x8"
    weight_path = cache_dir / "weights" / f"w{TENSOR_CACHE_EXTENSION}"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_bytes(b"unit-test")
    (cache_dir / "config.json").write_text(
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

    result = repair_cache_directory(cache_dir)

    assert result["repaired"] is False
    assert any("missing deepseek_weight_cache_metadata" in error for error in result["errors"])


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


def test_get_weight_config_emit_weight_cache_writes_manifest_for_saved_weights(tmp_path: Path) -> None:
    class FakeModule:
        @staticmethod
        def convert_weights(hf_config, state_dicts, output_path: Path, mesh_device):
            rel_path = Path("weights") / f"w{TENSOR_CACHE_EXTENSION}"
            (output_path / rel_path).parent.mkdir(parents=True, exist_ok=True)
            (output_path / rel_path).write_bytes(b"unit-test")
            return {"w": SavedWeight(path=rel_path, memory_config=ttnn.DRAM_MEMORY_CONFIG)}

    output_cfg = get_weight_config(
        ModuleClass=FakeModule,
        hf_config=_make_hf_config(num_hidden_layers=3),
        state_dicts=({"dummy": torch.empty(1)},),
        weight_cache_path=tmp_path / "weight_cache",
        mesh_device=_FakeMeshDevice(shape=(4, 8)),
        force_recalculate=True,
        emit_weight_cache=True,
    )

    output_path = tmp_path / "weight_cache" / "3_layers" / "mesh_4x8"
    config_payload = json.loads((output_path / "config.json").read_text(encoding="utf-8"))

    assert weight_config_module.WEIGHT_CACHE_METADATA_KEY in config_payload
    assert isinstance(output_cfg["w"], SavedWeight)
    assert output_cfg["w"].path == output_path / "weights" / f"w{TENSOR_CACHE_EXTENSION}"


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


def test_get_test_weight_config_defaults_to_direct_conversion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    direct_config = {"direct": True}

    def fake_get_weight_config(
        ModuleClass,
        hf_config,
        state_dicts,
        weight_cache_path: Path,
        mesh_device,
        force_recalculate=False,
    ):
        calls.append(
            {
                "weight_cache_path": weight_cache_path,
                "force_recalculate": force_recalculate,
            }
        )
        return direct_config

    monkeypatch.setattr(test_utils_module, "get_weight_config", fake_get_weight_config)

    cfg = test_utils_module.get_test_weight_config(
        ModuleClass=type("FakeModule", (), {}),
        hf_config=_make_hf_config(num_hidden_layers=3),
        state_dicts=({"dummy": torch.empty(1)},),
        cache_path=tmp_path / "weight_cache",
        mesh_device=_FakeMeshDevice(shape=(4, 8)),
        force_recalculate=False,
        test_name="test_bspm_demo",
        layer_id="uniform_5layers",
    )

    assert cfg is direct_config
    expected_path = tmp_path / "weight_cache" / "tests_cache" / "test_bspm_demo/FakeModule/real/uniform_5layers"
    assert calls == [
        {
            "weight_cache_path": expected_path,
            "force_recalculate": False,
        }
    ]


def test_shard_device_impl_uses_replicate_mapper_for_fully_replicated_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class _FakeTTTensor:
        def __init__(self, shape, dtype, layout, memory_config):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.layout = layout
            self._memory_config = memory_config

        def memory_config(self):
            return self._memory_config

        def reshape(self, new_shape):
            self.shape = tuple(new_shape)
            return self

    def fake_replicate(mesh_device):
        return ("replicate", mesh_device.shape)

    def fake_shard_1d(mesh_device, dim):
        return ("shard_1d", mesh_device.shape, dim)

    def fake_shard_2d(mesh_device, *, mesh_shape, dims):
        return ("shard_2d", mesh_shape, dims)

    def fake_from_torch(tensor, **kwargs):
        captured["mesh_mapper"] = kwargs["mesh_mapper"]
        return _FakeTTTensor(tensor.shape, kwargs["dtype"], kwargs["layout"], kwargs["memory_config"])

    monkeypatch.setattr(config_helpers_module.ttnn, "ReplicateTensorToMesh", fake_replicate)
    monkeypatch.setattr(config_helpers_module.ttnn, "ShardTensorToMesh", fake_shard_1d)
    monkeypatch.setattr(config_helpers_module.ttnn, "ShardTensor2dMesh", fake_shard_2d)
    monkeypatch.setattr(config_helpers_module.ttnn, "from_torch", fake_from_torch)

    result = config_helpers_module._shard_device_impl(
        path=tmp_path / "replicated_weight",
        tensor=torch.randn(1, 1, 32, 32, dtype=torch.bfloat16),
        shard_dims=(None, None),
        mesh_device=_FakeMeshDevice(shape=(4, 8)),
        remove_dims=(False, False),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    assert captured["mesh_mapper"] == ("replicate", (4, 8))
    assert result.shape == (1, 1, 32, 32)


def test_get_test_weight_config_includes_cache_identity_in_cache_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []
    direct_config = {"direct": True}

    def fake_get_weight_config(
        ModuleClass,
        hf_config,
        state_dicts,
        weight_cache_path: Path,
        mesh_device,
        force_recalculate=False,
    ):
        calls.append(
            {
                "weight_cache_path": weight_cache_path,
                "force_recalculate": force_recalculate,
            }
        )
        return direct_config

    monkeypatch.setattr(test_utils_module, "get_weight_config", fake_get_weight_config)

    cfg = test_utils_module.get_test_weight_config(
        ModuleClass=type("FakeModule", (), {}),
        hf_config=_make_hf_config(num_hidden_layers=3),
        state_dicts=({"dummy": torch.empty(1)},),
        cache_path=tmp_path / "weight_cache",
        mesh_device=_FakeMeshDevice(shape=(4, 8)),
        force_recalculate=False,
        test_name="test_bspm_demo",
        layer_id="uniform_5layers",
        cache_identity=Path("/tmp/stacked cache/dequantized"),
    )

    assert cfg is direct_config
    cache_identity_digest = hashlib.sha256(b"/tmp/stacked cache/dequantized").hexdigest()[:16]
    expected_path = (
        tmp_path
        / "weight_cache"
        / "tests_cache"
        / f"test_bspm_demo/FakeModule/real/uniform_5layers/__tmp__stacked_cache__dequantized__{cache_identity_digest}"
    )
    assert calls == [
        {
            "weight_cache_path": expected_path,
            "force_recalculate": False,
        }
    ]


def test_get_test_weight_config_cache_identity_uses_digest_to_avoid_collisions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[Path] = []

    def fake_get_weight_config(
        ModuleClass,
        hf_config,
        state_dicts,
        weight_cache_path: Path,
        mesh_device,
        force_recalculate=False,
    ):
        calls.append(weight_cache_path)
        return {"direct": True}

    monkeypatch.setattr(test_utils_module, "get_weight_config", fake_get_weight_config)

    kwargs = dict(
        ModuleClass=type("FakeModule", (), {}),
        hf_config=_make_hf_config(num_hidden_layers=3),
        state_dicts=({"dummy": torch.empty(1)},),
        cache_path=tmp_path / "weight_cache",
        mesh_device=_FakeMeshDevice(shape=(4, 8)),
        force_recalculate=False,
        test_name="test_bspm_demo",
        layer_id="uniform_5layers",
    )
    test_utils_module.get_test_weight_config(cache_identity=Path("/tmp/stacked cache/dequantized"), **kwargs)
    test_utils_module.get_test_weight_config(cache_identity=Path("/tmp/stacked_cache/dequantized"), **kwargs)

    assert len(calls) == 2
    assert calls[0] != calls[1]
    assert calls[0].name.startswith("__tmp__stacked_cache__dequantized__")
    assert calls[1].name.startswith("__tmp__stacked_cache__dequantized__")


def test_get_test_weight_config_prefers_legacy_cache_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    loaded_config = {"loaded": True}

    def fake_get_weight_config(
        *,
        weight_cache_path: Path,
        force_recalculate=False,
        emit_weight_cache=False,
        use_weight_cache=False,
        **kwargs,
    ):
        calls.append(
            {
                "weight_cache_path": weight_cache_path,
                "force_recalculate": force_recalculate,
                "emit_weight_cache": emit_weight_cache,
                "use_weight_cache": use_weight_cache,
            }
        )
        return loaded_config

    monkeypatch.setattr(test_utils_module, "get_weight_config", fake_get_weight_config)

    cfg = test_utils_module.get_test_weight_config(
        ModuleClass=type("FakeModule", (), {}),
        hf_config=_make_hf_config(num_hidden_layers=3),
        state_dicts=({"dummy": torch.empty(1)},),
        cache_path=tmp_path / "weight_cache",
        mesh_device=_FakeMeshDevice(shape=(4, 8)),
        force_recalculate=False,
        test_name="test_bspm_demo",
        layer_id="uniform_5layers",
        prefer_legacy_weight_cache=True,
    )

    expected_path = tmp_path / "weight_cache" / "tests_cache" / "test_bspm_demo/FakeModule/real/uniform_5layers"
    assert cfg is loaded_config
    assert calls == [
        {
            "weight_cache_path": expected_path,
            "force_recalculate": False,
            "emit_weight_cache": False,
            "use_weight_cache": True,
        },
    ]


def test_get_test_weight_config_rebuilds_legacy_cache_on_cache_miss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []
    emitted_config = {"emitted": True}

    def fake_get_weight_config(
        *,
        weight_cache_path: Path,
        force_recalculate=False,
        emit_weight_cache=False,
        use_weight_cache=False,
        **kwargs,
    ):
        calls.append(
            {
                "weight_cache_path": weight_cache_path,
                "force_recalculate": force_recalculate,
                "emit_weight_cache": emit_weight_cache,
                "use_weight_cache": use_weight_cache,
            }
        )
        if use_weight_cache:
            raise FileNotFoundError("Requested DeepSeek weight cache was not found")
        return emitted_config

    monkeypatch.setattr(test_utils_module, "get_weight_config", fake_get_weight_config)

    cfg = test_utils_module.get_test_weight_config(
        ModuleClass=type("FakeModule", (), {}),
        hf_config=_make_hf_config(num_hidden_layers=3),
        state_dicts=({"dummy": torch.empty(1)},),
        cache_path=tmp_path / "weight_cache",
        mesh_device=_FakeMeshDevice(shape=(4, 8)),
        force_recalculate=False,
        test_name="test_bspm_demo",
        layer_id="uniform_5layers",
        prefer_legacy_weight_cache=True,
    )

    expected_path = tmp_path / "weight_cache" / "tests_cache" / "test_bspm_demo/FakeModule/real/uniform_5layers"
    assert cfg is emitted_config
    assert calls == [
        {
            "weight_cache_path": expected_path,
            "force_recalculate": False,
            "emit_weight_cache": False,
            "use_weight_cache": True,
        },
        {
            "weight_cache_path": expected_path,
            "force_recalculate": False,
            "emit_weight_cache": True,
            "use_weight_cache": False,
        },
    ]


def test_get_test_weight_config_propagates_invalid_legacy_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_get_weight_config(
        *,
        weight_cache_path: Path,
        force_recalculate=False,
        emit_weight_cache=False,
        use_weight_cache=False,
        **kwargs,
    ):
        calls.append(
            {
                "weight_cache_path": weight_cache_path,
                "force_recalculate": force_recalculate,
                "emit_weight_cache": emit_weight_cache,
                "use_weight_cache": use_weight_cache,
            }
        )
        if use_weight_cache:
            raise InvalidWeightCacheError("Requested DeepSeek weight cache was invalid")
        raise AssertionError("Invalid legacy cache should not trigger regeneration")

    monkeypatch.setattr(test_utils_module, "get_weight_config", fake_get_weight_config)

    with pytest.raises(InvalidWeightCacheError, match="Requested DeepSeek weight cache was invalid"):
        test_utils_module.get_test_weight_config(
            ModuleClass=type("FakeModule", (), {}),
            hf_config=_make_hf_config(num_hidden_layers=3),
            state_dicts=({"dummy": torch.empty(1)},),
            cache_path=tmp_path / "weight_cache",
            mesh_device=_FakeMeshDevice(shape=(4, 8)),
            force_recalculate=False,
            test_name="test_bspm_demo",
            layer_id="uniform_5layers",
            prefer_legacy_weight_cache=True,
        )

    expected_path = tmp_path / "weight_cache" / "tests_cache" / "test_bspm_demo/FakeModule/real/uniform_5layers"
    assert calls == [
        {
            "weight_cache_path": expected_path,
            "force_recalculate": False,
            "emit_weight_cache": False,
            "use_weight_cache": True,
        }
    ]


def test_get_test_weight_config_force_recalculate_rebuilds_legacy_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []
    emitted_config = {"emitted": True}

    def fake_get_weight_config(
        *,
        weight_cache_path: Path,
        force_recalculate=False,
        emit_weight_cache=False,
        use_weight_cache=False,
        **kwargs,
    ):
        calls.append(
            {
                "weight_cache_path": weight_cache_path,
                "force_recalculate": force_recalculate,
                "emit_weight_cache": emit_weight_cache,
                "use_weight_cache": use_weight_cache,
            }
        )
        return emitted_config

    monkeypatch.setattr(test_utils_module, "get_weight_config", fake_get_weight_config)

    cfg = test_utils_module.get_test_weight_config(
        ModuleClass=type("FakeModule", (), {}),
        hf_config=_make_hf_config(num_hidden_layers=3),
        state_dicts=({"dummy": torch.empty(1)},),
        cache_path=tmp_path / "weight_cache",
        mesh_device=_FakeMeshDevice(shape=(4, 8)),
        force_recalculate=True,
        test_name="test_bspm_demo",
        layer_id="uniform_5layers",
        prefer_legacy_weight_cache=True,
    )

    assert cfg is emitted_config
    expected_path = tmp_path / "weight_cache" / "tests_cache" / "test_bspm_demo/FakeModule/real/uniform_5layers"
    assert calls == [
        {
            "weight_cache_path": expected_path,
            "force_recalculate": True,
            "emit_weight_cache": True,
            "use_weight_cache": False,
        }
    ]


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
