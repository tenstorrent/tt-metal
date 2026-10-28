# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gradients of the tensor-parallel LoRA layers against a NumPy oracle.

``LoraColumnParallelLinear`` keeps lora_A replicated and shards lora_B by output column, so every
device back-propagates only its own columns into ``h = x @ lora_A``. Unless ``d(h)`` is all-reduced,
each device's lora_A grad is a partial sum: the copies drift apart after the first update and the
copy that gets saved is not the adapter that was trained. B is nonzero here because at B = 0 every
lora_A grad is zero and the bug cannot show.

Mesh: ``[1, 2]`` with axes ``("dp", "tp")``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ttnn
import ttml
from ttml.modules import (
    ColumnParallelLinear,
    LoraColumnParallelLinear,
    LoraConfig,
    LoraRowParallelLinear,
    RowParallelLinear,
)

pytestmark = pytest.mark.requires_device

SEQ = 32
RANK = 32
ALPHA = 2 * RANK


def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _bf16(a: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(torch.bfloat16).float().numpy()


def _replicated_mapper():
    placements = [ttnn.PlacementReplicate() for _ in ttml.mesh().shape]
    return ttnn.create_mesh_mapper(_device(), ttnn.MeshMapperConfig(placements))


def _tp_mapper(tdim: int):
    return ttml.mesh().axis_mapper("tp", tdim=tdim)


def _compose(t, tp_dim: int) -> np.ndarray:
    """Concatenate the per-device tensors along ``tp_dim`` on the tp axis (the other axes have size 1)."""
    mesh = ttml.mesh()
    dims = [tp_dim if axis == mesh.axis_index("tp") else 0 for axis in range(len(mesh.shape))]
    composer = ttnn.create_mesh_composer(_device(), ttnn.MeshComposerConfig(dims))
    return t.to_numpy(ttnn.DataType.FLOAT32, composer=composer)


def _per_device(t) -> list[np.ndarray]:
    """Every TP device's own copy of a 4-D tensor whose dim 1 has size 1."""
    full = _compose(t, 1)
    return [full[:, i : i + 1] for i in range(full.shape[1])]


def _upload(arr: np.ndarray, mapper, requires_grad: bool):
    t = ttml.autograd.Tensor.from_numpy(arr, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
    t.set_requires_grad(requires_grad)
    return t


def _set(param, arr: np.ndarray, mapper) -> None:
    param.tensor.set_value(_upload(arr, mapper, False).get_value())


def _assert_close(actual: np.ndarray, expected: np.ndarray, what: str) -> None:
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-2 * np.abs(expected).max(), err_msg=what)


def _oracle(x, w, a, b, g, scaling):
    """Forward and grads of ``sum(g * (x W^T + scaling * (x A^T) B^T))`` on 2-D float64 arrays."""
    x, w, a, b, g = (m.astype(np.float64) for m in (x, w, a, b, g))
    h = x @ a.T
    out = x @ w.T + scaling * (h @ b.T)
    dh = scaling * (g @ b)
    return {"out": out, "dx": g @ w + dh @ a, "dA": dh.T @ x, "dB": scaling * (g.T @ h)}


class TestLoraTensorParallelGrads:
    @pytest.fixture(autouse=True)
    def _bind(self, tp_mesh):
        self.tp = tp_mesh.axis_size("tp")
        self.ctx = ttml.autograd.AutoContext.get_instance()
        self.ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
        yield
        self.ctx.reset_graph()

    def _weights(self, in_features: int, out_features: int, seed: int):
        rng = np.random.default_rng(seed)
        return {
            "x": _bf16(rng.standard_normal((SEQ, in_features))),
            "w": _bf16(rng.standard_normal((out_features, in_features)) / np.sqrt(in_features)),
            "a": _bf16(rng.standard_normal((RANK, in_features)) / np.sqrt(in_features)),
            "b": _bf16(rng.standard_normal((out_features, RANK)) / np.sqrt(RANK)),
            "g": _bf16(rng.standard_normal((SEQ, out_features))),
        }

    @pytest.mark.parametrize("gather_output", [False, True])
    def test_column_parallel_matches_oracle(self, gather_output):
        in_features, out_features = 64, 128
        m = self._weights(in_features, out_features, seed=1)
        base = ColumnParallelLinear(in_features, out_features, has_bias=False, gather_output=gather_output)
        lora = LoraColumnParallelLinear(base, LoraConfig(rank=RANK, alpha=ALPHA))
        _set(lora.weight, m["w"][None, None], _tp_mapper(tdim=2))
        _set(lora.lora_A, m["a"][None, None], _replicated_mapper())
        _set(lora.lora_B, m["b"][None, None], _tp_mapper(tdim=2))

        x = _upload(m["x"][None, None], _replicated_mapper(), True)
        out_mapper = _replicated_mapper() if gather_output else _tp_mapper(tdim=3)
        out = lora(x)
        ttml.ops.binary.mul(out, _upload(m["g"][None, None], out_mapper, False)).backward(False)

        ref = _oracle(m["x"], m["w"], m["a"], m["b"], m["g"], lora.scaling)
        got_out = _per_device(out)[0] if gather_output else _compose(out, 3)
        _assert_close(got_out[0, 0], ref["out"], "forward")
        for i, dx in enumerate(_per_device(x.get_grad_tensor())):
            _assert_close(dx[0, 0], ref["dx"], f"input grad on device {i}")
        _assert_close(_compose(lora.lora_B.tensor.get_grad_tensor(), 2)[0, 0], ref["dB"], "lora_B grad")

        copies = _per_device(lora.lora_A.tensor.get_grad_tensor())
        assert len(copies) == self.tp
        for i, da in enumerate(copies):
            _assert_close(da[0, 0], ref["dA"], f"lora_A grad on device {i}")
        for i, da in enumerate(copies[1:], start=1):
            np.testing.assert_allclose(da, copies[0], rtol=1e-2, atol=1e-3, err_msg=f"lora_A grad copy {i} vs 0")

    def test_row_parallel_matches_oracle(self):
        in_features, out_features = 128, 64
        m = self._weights(in_features, out_features, seed=2)
        base = RowParallelLinear(in_features, out_features, has_bias=False, input_is_parallel=True)
        lora = LoraRowParallelLinear(base, LoraConfig(rank=RANK, alpha=ALPHA))
        _set(lora.weight, m["w"][None, None], _tp_mapper(tdim=3))
        _set(lora.lora_A, m["a"][None, None], _tp_mapper(tdim=3))
        _set(lora.lora_B, m["b"][None, None], _replicated_mapper())

        x = _upload(m["x"][None, None], _tp_mapper(tdim=3), True)
        out = lora(x)
        ttml.ops.binary.mul(out, _upload(m["g"][None, None], _replicated_mapper(), False)).backward(False)

        ref = _oracle(m["x"], m["w"], m["a"], m["b"], m["g"], lora.scaling)
        _assert_close(_per_device(out)[0][0, 0], ref["out"], "forward")
        _assert_close(_compose(x.get_grad_tensor(), 3)[0, 0], ref["dx"], "input grad")
        _assert_close(_compose(lora.lora_A.tensor.get_grad_tensor(), 3)[0, 0], ref["dA"], "lora_A grad")
        for i, db in enumerate(_per_device(lora.lora_B.tensor.get_grad_tensor())):
            _assert_close(db[0, 0], ref["dB"], f"lora_B grad on device {i}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
