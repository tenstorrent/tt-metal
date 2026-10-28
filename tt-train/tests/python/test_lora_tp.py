# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Column-parallel LoRA gradients on a ``[1, 2]`` tensor-parallel mesh.

``LoraColumnParallelLinear`` keeps ``lora_A`` replicated and ``lora_B`` sharded on its output rows. The grad that
reaches ``lora_A`` on device ``d`` only carries the output columns of ``d``'s ``lora_B`` shard, so without a sum
across TP each copy is updated with a different partial grad and the copies drift apart after the first step.
These tests compare every grad against a numpy reference and check the copies stay identical over optimizer steps.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttml
import ttnn
from ttml.modules import ColumnParallelLinear, LoraConfig
from ttml.modules.lora import LoraColumnParallelLinear

pytestmark = pytest.mark.requires_device

DIM = 64  # 32 output rows per device after the 2-way split
SEQ = 32
RANK = 32
NATIVE = ttml.autograd.PreferredPrecision.NATIVE


def _device_slices(value) -> list:
    """Each device's slice of a ttnn tensor as a float32 numpy array, in mesh order, stacked on dim 0 so no
    topology label is trusted."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
    stacked = ttnn.to_torch(value, mesh_composer=composer).float().numpy()
    return np.split(stacked, device.get_num_devices(), axis=0)


def _assert_identical_across_devices(value, what: str) -> np.ndarray:
    slices = _device_slices(value)
    for dev, other in enumerate(slices[1:], start=1):
        assert np.array_equal(slices[0], other), f"{what}: device {dev} differs from device 0"
    return slices[0]


def _assert_close(got: np.ndarray, ref: np.ndarray, what: str) -> None:
    np.testing.assert_allclose(got, ref, rtol=5e-2, atol=2e-2 * np.abs(ref).max(), err_msg=what)


def _set_sharded(param, array: np.ndarray, tdim: int) -> None:
    mapper = ttml.mesh().axis_mapper("tp", tdim=tdim)
    param.tensor.set_value(
        ttml.autograd.Tensor.from_numpy(array, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper).get_value()
    )


def _lora_layer(gather_output: bool) -> LoraColumnParallelLinear:
    """A column-parallel LoRA layer with a non-zero ``lora_B`` (its zero init would make ``lora_A``'s grad zero)."""
    rng = np.random.default_rng(0)
    base = ColumnParallelLinear(DIM, DIM, has_bias=False, gather_output=gather_output)
    layer = LoraColumnParallelLinear(base, LoraConfig(rank=RANK, alpha=float(RANK)))
    _set_sharded(layer.lora_B, rng.uniform(-0.5, 0.5, (1, 1, DIM, RANK)).astype(np.float32), tdim=2)
    return layer


def _full_weight(param) -> np.ndarray:
    """Rows sharded on dim 2, concatenated back on the host."""
    return np.concatenate(_device_slices(param.tensor.get_value(NATIVE)), axis=2)[0, 0]


@pytest.mark.parametrize("gather_output", [True, False], ids=["gathered", "sharded_output"])
def test_column_parallel_grads_match_reference(tp_mesh, gather_output):
    layer = _lora_layer(gather_output)
    x_np = np.random.default_rng(1).standard_normal((1, 1, SEQ, DIM)).astype(np.float32)
    x = ttml.autograd.Tensor.from_numpy(x_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)
    x.set_requires_grad(True)

    loss = ttml.ops.unary.mean(layer(x))
    loss.backward(False)

    # Reference in float32 from the bf16 values actually on device. A gathered output is replicated, so its mean
    # spans every column; a sharded output's mean only spans the local columns, and backward starts from 1 on each
    # device, so the loss being differentiated is the sum of the per-device means.
    x_ref = _device_slices(x.get_value(NATIVE))[0][0, 0]
    W = _full_weight(layer.weight)
    A = _assert_identical_across_devices(layer.lora_A.tensor.get_value(NATIVE), "lora_A")[0, 0]
    B = _full_weight(layer.lora_B)
    out_per_mean = DIM if gather_output else DIM // tp_mesh.shape[1]
    dY = np.full((SEQ, DIM), 1.0 / (SEQ * out_per_mean), dtype=np.float32)
    h = x_ref @ A.T
    dh = layer.scaling * dY @ B
    ref_dA = dh.T @ x_ref
    ref_dB = layer.scaling * dY.T @ h
    ref_dx = dY @ W + dh @ A

    dA = _assert_identical_across_devices(layer.lora_A.tensor.get_grad(), "lora_A grad")[0, 0]
    _assert_close(dA, ref_dA, "lora_A grad")
    dB = np.concatenate(_device_slices(layer.lora_B.tensor.get_grad()), axis=2)[0, 0]
    _assert_close(dB, ref_dB, "lora_B grad")
    dx = _assert_identical_across_devices(x.get_grad(), "input grad")[0, 0]
    _assert_close(dx, ref_dx, "input grad")

    ttml.autograd.AutoContext.get_instance().reset_graph()


def test_column_parallel_lora_A_stays_replicated_over_steps(tp_mesh):
    layer = _lora_layer(gather_output=False)
    params = layer.parameters()
    cfg = ttml.optimizers.AdamWConfig.make(lr=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0)
    opt = ttml.optimizers.AdamW(params, cfg)
    A_init = _device_slices(layer.lora_A.tensor.get_value(NATIVE))[0]
    rng = np.random.default_rng(2)

    for step in range(3):
        x = ttml.autograd.Tensor.from_numpy(
            rng.standard_normal((1, 1, SEQ, DIM)).astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
        )
        opt.zero_grad()
        loss = ttml.ops.unary.mean(layer(x))
        loss.backward(False)
        opt.step()
        ttml.autograd.AutoContext.get_instance().reset_graph()
        A = _assert_identical_across_devices(layer.lora_A.tensor.get_value(NATIVE), f"lora_A after step {step + 1}")

    assert not np.array_equal(A, A_init), "lora_A should have been updated"
