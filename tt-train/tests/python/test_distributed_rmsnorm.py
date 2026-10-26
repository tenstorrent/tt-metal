# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Forward and backward of ``ttml.ops.distributed.rmsnorm`` against torch.

The hidden dim of the input and of gamma is sharded over the ``"tp"`` axis. The torch reference runs in fp32 on the
same bf16-rounded inputs the devices hold, so the comparison measures only device arithmetic.

Mesh: ``[1, 2]`` with axes ``("dp", "tp")``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ttnn
import ttml

pytestmark = pytest.mark.requires_device

EPSILON = 1e-5


def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _tp_sharded(values: np.ndarray):
    tensor = ttml.autograd.Tensor.from_numpy(
        values, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, ttml.mesh().axis_mapper("tp", tdim=3)
    )
    tensor.set_requires_grad(True)
    return tensor


def _read_tp_sharded(tensor) -> np.ndarray:
    """Concatenates the last-dim shards along the 'tp' axis; the 'dp' axis has size 1."""
    mesh = ttml.mesh()
    composer_dims = [3 if axis == mesh.axis_index("tp") else 0 for axis in range(len(mesh.shape))]
    composer = ttnn.create_mesh_composer(_device(), ttnn.MeshComposerConfig(composer_dims))
    return tensor.to_numpy(ttnn.DataType.FLOAT32, composer=composer)


def _bf16(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(values).to(torch.bfloat16).to(torch.float32)


@pytest.mark.parametrize(
    "batch, seq_len, hidden",
    [
        (2, 64, 256),
        # Non-tile-aligned sequence length and a wider row.
        (1, 33, 2048),
    ],
)
def test_matches_torch(tp_mesh, batch, seq_len, hidden):
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, (batch, 1, seq_len, hidden)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, (1, 1, 1, hidden)).astype(np.float32)
    dy = rng.uniform(-1.0, 1.0, (batch, 1, seq_len, hidden)).astype(np.float32)

    x_ref = _bf16(x).requires_grad_(True)
    gamma_ref = _bf16(gamma).requires_grad_(True)
    out_ref = gamma_ref * x_ref / torch.sqrt(torch.mean(x_ref * x_ref, dim=-1, keepdim=True) + EPSILON)
    out_ref.backward(_bf16(dy))

    tt_x = _tp_sharded(x)
    tt_gamma = _tp_sharded(gamma)
    out = ttml.ops.distributed.rmsnorm(tt_x, tt_gamma, EPSILON, cluster_axis=tp_mesh.axis_index("tp"))
    out.set_grad_from_tensor(_tp_sharded(dy))
    out.backward(False)
    assert tt_x.is_grad_initialized()
    assert tt_gamma.is_grad_initialized()

    np.testing.assert_allclose(_read_tp_sharded(out), out_ref.detach().numpy(), rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(_read_tp_sharded(tt_x.get_grad_tensor()), x_ref.grad.numpy(), rtol=2e-2, atol=2e-2)
    # dgamma sums batch * seq_len per-row errors.
    dgamma_atol = 1e-2 * np.sqrt(batch * seq_len)
    np.testing.assert_allclose(
        _read_tp_sharded(tt_gamma.get_grad_tensor()), gamma_ref.grad.numpy(), rtol=2e-2, atol=dgamma_atol
    )
    ctx.reset_graph()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
