# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""`verify_saved_model`: an intact cache verifies, and every way a cache can be bad returns False rather
than raising, so the run that just built the weights can go on while the cache is left unmarked."""

from __future__ import annotations

from typing import NamedTuple

import pytest
import torch

import ttnn
from models.tt_dit.layers.module import Module, Parameter
from models.tt_dit.utils import cache as cache_module
from models.tt_dit.utils.cache import nonfinite_tensorbins, verify_saved_model


class TinyModel(Module):
    def __init__(self, *, device: ttnn.MeshDevice) -> None:
        super().__init__()
        self.weight = Parameter(total_shape=[32, 64], device=device)
        self.bias = Parameter(total_shape=[64], device=device)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(x, self.weight.data, bias=self.bias.data)


def _saved_model(mesh_device: ttnn.MeshDevice, tmp_path):
    model = TinyModel(device=mesh_device)
    model.load_torch_state_dict({"weight": torch.randn([32, 64]), "bias": torch.randn([64])})
    model.save(tmp_path)
    return model


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_intact_cache_verifies(mesh_device, tmp_path) -> None:
    model = _saved_model(mesh_device, tmp_path)
    assert verify_saved_model(model, tmp_path) is True


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_missing_tensorbin_fails_verification(mesh_device, tmp_path) -> None:
    model = _saved_model(mesh_device, tmp_path)
    (tmp_path / "bias.tensorbin").unlink()
    assert verify_saved_model(model, tmp_path) is False


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_truncated_tensorbin_fails_verification_without_raising(mesh_device, tmp_path) -> None:
    model = _saved_model(mesh_device, tmp_path)
    path = tmp_path / "weight.tensorbin"
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    assert verify_saved_model(model, tmp_path) is False


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_garbage_tensorbin_fails_verification_without_raising(mesh_device, tmp_path) -> None:
    model = _saved_model(mesh_device, tmp_path)
    (tmp_path / "weight.tensorbin").write_bytes(b"not a tensor")
    assert verify_saved_model(model, tmp_path) is False


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_altered_values_fail_verification(mesh_device, tmp_path) -> None:
    model = _saved_model(mesh_device, tmp_path)
    other = TinyModel(device=mesh_device)
    other.load_torch_state_dict({"weight": torch.randn([32, 64]), "bias": torch.randn([64])})
    other.save(tmp_path)  # same shapes, different values, overwrites both files
    assert verify_saved_model(model, tmp_path) is False
    # and the resident weights were left alone
    assert verify_saved_model(other, tmp_path) is True


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_nonfinite_values_fail_verification(mesh_device, tmp_path) -> None:
    """A cache that reads back exactly can still be garbage: weights prepped on a faulty device (#208)."""
    model = TinyModel(device=mesh_device)
    weight = torch.randn([32, 64])
    weight[3, 5] = float("nan")
    model.load_torch_state_dict({"weight": weight, "bias": torch.randn([64])})
    model.save(tmp_path)
    assert verify_saved_model(model, tmp_path) is False
    assert nonfinite_tensorbins(model, tmp_path) == ["weight"]


class _NoParallel(NamedTuple):
    tensor_parallel: None = None


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_load_model_rebuilds_a_nonfinite_cache(mesh_device, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TT_DIT_CACHE_VERIFY", "1")
    monkeypatch.setattr(cache_module, "_scanned_cache_dirs", set())
    good = {"weight": torch.randn([32, 64]), "bias": torch.randn([64])}
    args = dict(
        model_name="tiny", subfolder="m", parallel_config=_NoParallel(), mesh_shape=(1, 1), mesh_device=mesh_device
    )
    cache_module.load_model(TinyModel(device=mesh_device), **args, get_torch_state_dict=lambda: good)
    cache_dir = cache_module.model_cache_dir(**args)
    assert (cache_dir / cache_module.CACHE_DICT_FILE).is_file()

    poisoned = TinyModel(device=mesh_device)
    poisoned.load_torch_state_dict({"weight": torch.full([32, 64], float("inf")), "bias": good["bias"]})
    poisoned.save(cache_dir)  # same shapes and sizes: the manifest check alone serves it

    model = TinyModel(device=mesh_device)
    cache_module.load_model(model, **args, get_torch_state_dict=lambda: good)
    assert len(list(cache_dir.parent.glob(f"{cache_dir.name}.bad-nonfinite-*"))) == 1
    assert nonfinite_tensorbins(model, cache_dir) == []
    assert torch.allclose(
        ttnn.to_torch(ttnn.get_device_tensors(model.weight.data)[0]).float(), good["weight"], atol=0.1
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_a_clean_scan_is_stamped_and_skipped_until_a_tensorbin_changes(mesh_device, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TT_DIT_CACHE_VERIFY", "1")
    model = _saved_model(mesh_device, tmp_path)
    scans = []
    real = cache_module.nonfinite_tensorbins
    monkeypatch.setattr(cache_module, "nonfinite_tensorbins", lambda m, d: scans.append(d) or real(m, d))

    for _ in range(2):  # a new process each time: the in-process set is empty again
        monkeypatch.setattr(cache_module, "_scanned_cache_dirs", set())
        assert not cache_module._set_aside_if_nonfinite(model, tmp_path)
    assert len(scans) == 1 and (tmp_path / cache_module.SCAN_OK_FILE).is_file()

    poisoned = TinyModel(device=mesh_device)
    poisoned.load_torch_state_dict({"weight": torch.full([32, 64], float("nan")), "bias": torch.randn([64])})
    poisoned.save(tmp_path)  # same sizes, new mtimes: the stamp no longer matches
    monkeypatch.setattr(cache_module, "_scanned_cache_dirs", set())
    assert cache_module._set_aside_if_nonfinite(model, tmp_path)
    assert len(scans) == 2
