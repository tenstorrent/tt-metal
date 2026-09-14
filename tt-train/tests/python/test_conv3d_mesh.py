# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``ttml.ops.conv.conv3d`` on a ``[1, 2]`` device mesh with replicated tensors.

Kept apart from ``test_conv3d.py``: the module-scoped ``tp_mesh`` fixture keeps the mesh open for the whole module,
and the single-device tests there open a plain device.
"""

import numpy as np
import pytest
import torch

import ttnn
import ttml

pytestmark = pytest.mark.requires_device

REL_OF_MAX = 1e-2
ABS_FLOOR = 2e-2
MIN_PCC = 0.999


def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _replicated_mapper():
    placements = [ttnn.PlacementReplicate() for _ in ttml.mesh().shape]
    return ttnn.create_mesh_mapper(_device(), ttnn.MeshMapperConfig(placements))


def _read_replicated(t) -> np.ndarray:
    """One host copy (fp32) of a tensor that is identical on every device."""
    mesh = ttml.mesh()
    n_axes = len(mesh.shape)
    composer = ttnn.create_mesh_composer(_device(), ttnn.MeshComposerConfig(list(range(n_axes))))
    full = t.to_numpy(ttnn.DataType.FLOAT32, composer=composer)
    slicer = [slice(None)] * full.ndim
    for axis in range(n_axes):
        slicer[axis] = slice(0, full.shape[axis] // mesh.shape[axis])
    return full[tuple(slicer)]


def _assert_close(actual, expected, what):
    tolerance = ABS_FLOOR + REL_OF_MAX * np.abs(expected).max()
    max_err = np.abs(actual - expected).max()
    assert max_err <= tolerance, f"{what}: max abs err {max_err} > {tolerance}"
    pcc = np.corrcoef(actual.ravel(), expected.ravel())[0, 1]
    assert pcc > MIN_PCC, f"{what}: pcc {pcc} <= {MIN_PCC}"


def _bf16(array):
    return torch.from_numpy(array).to(torch.bfloat16).to(torch.float32)


def _replicated(array, requires_grad):
    tensor = ttml.autograd.Tensor.from_numpy(
        np.ascontiguousarray(array), ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, _replicated_mapper()
    )
    tensor.set_requires_grad(requires_grad)
    return tensor


def test_conv3d_replicated_on_mesh(tp_mesh):
    """Forward and all three gradients on replicated tensors match the single-device torch reference."""
    torch.manual_seed(0)
    np.random.seed(0)
    N, C_in, C_out, spatial, kernel, padding = 2, 32, 40, (4, 5, 6), (3, 3, 3), (1, 1, 1)

    x_torch = _bf16(np.random.uniform(-1, 1, (N, C_in, *spatial)).astype(np.float32)).requires_grad_(True)
    w_torch = _bf16(np.random.uniform(-0.5, 0.5, (C_out, C_in, *kernel)).astype(np.float32)).requires_grad_(True)
    b_torch = _bf16(np.random.uniform(-1, 1, (C_out,)).astype(np.float32)).requires_grad_(True)
    out_torch = torch.nn.functional.conv3d(x_torch, w_torch, b_torch, padding=padding)
    grad_out = _bf16(np.random.uniform(-1, 1, tuple(out_torch.shape)).astype(np.float32))
    out_torch.backward(grad_out)

    x = _replicated(x_torch.detach().numpy().transpose(0, 2, 3, 4, 1), True)
    w = _replicated(w_torch.detach().numpy(), True)
    b = _replicated(b_torch.detach().numpy().reshape(1, 1, 1, C_out), True)

    out = ttml.ops.conv.conv3d(x, w, b, padding=padding)
    _assert_close(_read_replicated(out), out_torch.detach().numpy().transpose(0, 2, 3, 4, 1), "forward")

    out.set_grad_from_tensor(_replicated(grad_out.numpy().transpose(0, 2, 3, 4, 1), False))
    out.backward(False)
    _assert_close(_read_replicated(x.get_grad_tensor()), x_torch.grad.numpy().transpose(0, 2, 3, 4, 1), "grad_input")
    _assert_close(_read_replicated(w.get_grad_tensor()), w_torch.grad.numpy(), "grad_weight")
    _assert_close(_read_replicated(b.get_grad_tensor()).reshape(C_out), b_torch.grad.numpy(), "grad_bias")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
