# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""`verify_saved_model`: an intact cache verifies, and every way a cache can be bad returns False rather
than raising, so the run that just built the weights can go on while the cache is left unmarked."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.tt_dit.layers.module import Module, Parameter
from models.tt_dit.utils.cache import verify_saved_model


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
