# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from models.experimental.ace_step_v1_5.torch_ref.safetensors_loader import load_safetensors_state_dict


def test_load_safetensors_state_dict_prefix_stripping(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    tensors = {
        "decoder.w": torch.randn(4, 3, dtype=torch.float32),
        "decoder.b": torch.randn(4, dtype=torch.float32),
        "other.w": torch.randn(2, 2, dtype=torch.float32),
    }
    path = tmp_path / "ckpt.safetensors"
    safetensors_torch.save_file(tensors, str(path))

    sd = load_safetensors_state_dict(str(path), prefix="decoder.").tensors
    assert set(sd.keys()) == {"w", "b"}
    assert isinstance(sd["w"], torch.Tensor) and tuple(sd["w"].shape) == (4, 3)
    assert isinstance(sd["b"], torch.Tensor) and tuple(sd["b"].shape) == (4,)


def test_load_safetensors_state_dict_preserves_bf16_torch(tmp_path):
    """BF16 weights stay as CPU torch tensors (no eager float32 numpy copy)."""
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    tensors = {
        "decoder.w": torch.randn(4, 3, dtype=torch.bfloat16),
        "decoder.b": torch.randn(4, dtype=torch.bfloat16),
    }
    path = tmp_path / "bf16.safetensors"
    safetensors_torch.save_file(tensors, str(path))

    sd = load_safetensors_state_dict(str(path), prefix="decoder.").tensors
    assert sd["w"].dtype == torch.bfloat16
    assert sd["b"].dtype == torch.bfloat16
