# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-free SigLIP2 MLP regression tests."""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc


@pytest.fixture(autouse=True)
def require_staged_checkpoint(tmp_path, monkeypatch):
    # Override the parent checkpoint fixture. Package initializers resolve a
    # checkpoint path even though this test supplies every weight explicitly.
    # An empty index satisfies that import-time lookup without downloading data.
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    monkeypatch.setenv("HUNYUAN_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("HY_SKIP_WEIGHT_DOWNLOAD", "1")


@pytest.mark.parametrize("seq_len", [64, 128])
def test_mlp_random_weights_tanh_gelu(device, monkeypatch, seq_len):
    """Check the production MLP and its activation without a checkpoint download."""
    from models.experimental.hunyuan_image_3_0.ref.model_config import VIT_CONFIG
    from models.experimental.hunyuan_image_3_0.ttnn.vision.siglip2 import HunyuanTtSiglip2MLP

    torch.manual_seed(63673)
    fc1 = torch.nn.Linear(VIT_CONFIG["hidden_size"], VIT_CONFIG["intermediate_size"]).bfloat16()
    fc2 = torch.nn.Linear(VIT_CONFIG["intermediate_size"], VIT_CONFIG["hidden_size"]).bfloat16()
    reference = torch.nn.Sequential(fc1, torch.nn.GELU(approximate="tanh"), fc2).eval()
    with torch.no_grad():
        fc1.weight.normal_(0, 1.5 / VIT_CONFIG["hidden_size"] ** 0.5)
        fc2.weight.normal_(0, 1 / VIT_CONFIG["intermediate_size"] ** 0.5)
    state = {
        f"encoder.layers.0.mlp.{name}.{key}": value
        for name, linear in (("fc1", fc1), ("fc2", fc2))
        for key, value in linear.state_dict().items()
    }
    module = HunyuanTtSiglip2MLP(device, state, layer_idx=0)
    hidden = torch.randn(1, seq_len, VIT_CONFIG["hidden_size"]).bfloat16()
    activation_errors = []
    gelu = ttnn.gelu

    def capture_gelu(value, **kwargs):
        expected = torch.nn.functional.gelu(ttnn.to_torch(value), approximate="tanh").float()
        result = gelu(value, **kwargs)
        activation_errors.append((ttnn.to_torch(result).float() - expected).abs().mean().item())
        return result

    monkeypatch.setattr(ttnn, "gelu", capture_gelu)
    with torch.no_grad():
        expected = reference(hidden).float()
    input_tensor = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(module(input_tensor)).reshape_as(expected).float()
    passing, block_pcc = comp_pcc(expected, actual, 0.999)
    assert passing, block_pcc
    print(
        f"SigLIP2 MLP mean_abs={(actual - expected).abs().mean().item():.8f}, "
        f"max_abs={(actual - expected).abs().max().item():.8f}, PCC={block_pcc}, GELU mean_abs={activation_errors}"
    )
    # PCC alone misses the biased LUT error. Bound the activation error separately
    # against tanh-GELU evaluated on the actual BF16 matmul output.
    assert len(activation_errors) == 1
    assert activation_errors[0] < 0.002
