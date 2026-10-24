# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

from ...experimental.lora.promote import promote_to_lora
from ...layers.linear import Linear
from ...utils.tensor import bf16_tensor
from ...utils.test import mesh_device_config_to_string


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).pow(2).mean().sqrt().item()


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=mesh_device_config_to_string, indirect=True)
def test_promoted_fuse_delta_survives_eviction(mesh_device: ttnn.MeshDevice) -> None:
    """Delta merges exactly once across resident reapply and evict -> reload -> reapply."""
    torch.manual_seed(0)
    K, N, rank, scale = 512, 1024, 8, 0.02
    torch_linear = torch.nn.Linear(K, N, bias=True).to(torch.bfloat16)
    A, B = torch.randn(rank, K), torch.randn(N, rank)

    layer = Linear(K, N, bias=True, mesh_device=mesh_device)
    layer.load_torch_state_dict(torch_linear.state_dict())
    assert promote_to_lora(layer) == 1
    layer.bind_active(layer.register_lora(A, B, scale=scale))

    x = torch.randn(1, 1, 64, K, dtype=torch.bfloat16)
    base = torch_linear(x).float()
    fused = (x.float() @ (torch_linear.weight.float() + scale * B @ A).T) + torch_linear.bias.float()
    doubled = (x.float() @ (torch_linear.weight.float() + 2 * scale * B @ A).T) + torch_linear.bias.float()
    margin = _rmse(base, fused)

    def forward() -> torch.Tensor:
        out = layer(bf16_tensor(x, device=mesh_device))
        return ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()

    assert _rmse(forward(), fused) < 0.1 * margin

    layer.reapply_after_load()
    assert _rmse(forward(), fused) < 0.1 * _rmse(fused, doubled), "resident reapply merged the delta twice"

    layer.deallocate_weights()
    assert not layer._delta_applied
    layer.load_torch_state_dict(torch_linear.state_dict())
    layer.reapply_after_load()
    assert _rmse(forward(), fused) < 0.1 * margin, "reloaded base weights lost the delta"
