# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek's auxiliary-loss-free expert bias under data parallelism.

``MoE.update_expert_bias`` moves each expert's routing bias by the sign of (mean count - its count), from the token
counts the layer accumulated since the last update. Under data parallelism each device routes only its slice of the
batch, so the update must use the counts of the whole batch: then every device makes the same move and the devices
keep one bias.

On the ``[2, 2]`` mesh with axes ``("dp", "fsdp")`` that ``train.py`` builds for HSDP, a ``SparseMoEEP`` at EP size 1
(no expert axis: every device holds every expert) runs forward on a batch sharded over all four devices, so the
devices count different tokens along both axes. After the update the bias is the one computed from the counts of the
whole batch on every device, and both buffers are still labelled replicated. The counts are integers, but their average
over the devices is not, and near the mean its fraction decides the sign of the update: the counts are kept in FP32.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttnn
import ttml
from tp_block import assert_identical_across_devices, device_slices
from ttml.models.deepseek.moe_sparse_ep import SparseMoEEP

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(1800)]

SEED = 2026
DIM = 64
SEQ = 32
N_EXPERTS = 8
NATIVE = ttml.autograd.PreferredPrecision.NATIVE  # the counts' own FP32, not the default BF16 copy


class _Cfg:
    """The MoE config fields ``SparseMoEEP`` reads."""

    dim = DIM
    moe_inter_dim = 64
    n_routed_experts = N_EXPERTS
    n_activated_experts = 2
    n_shared_experts = 0
    n_expert_groups = 1
    n_limited_groups = 1
    score_func = "sigmoid"
    route_scale = 1.0
    moe_type = "sparse_ep"
    moe_axis_name = None


def _sharded_placements(value) -> list:
    """The shard placements of a tensor's topology label (none for a replicated tensor)."""
    return [p for p in value.tensor_topology().placements() if isinstance(p, ttnn.PlacementShard)]


def test_expert_bias_update_uses_the_counts_of_the_whole_batch(hsdp_mesh):
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.set_seed(SEED)
    layer = SparseMoEEP(_Cfg())
    tokens = np.random.default_rng(SEED).standard_normal((hsdp_mesh.num_devices(), 1, SEQ, DIM), dtype=np.float32)
    batch_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(ctx.get_device(), 0, None)
    x = ttml.autograd.Tensor.from_numpy(tokens, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, batch_mapper)
    layer(x)
    ctx.reset_graph()
    counts = [copy.reshape(-1) for copy in device_slices(layer._token_counts.tensor.get_value(NATIVE))]
    # Precondition: devices 0 and 1 are neighbours along "fsdp", devices 0 and 2 along "dp".
    for other, axis in ((1, "fsdp"), (2, "dp")):
        assert not np.array_equal(counts[0], counts[other]), f"precondition: the {axis} neighbours routed alike"

    coeff = 0.001
    layer.update_expert_bias(coeff)

    bias = assert_identical_across_devices(layer._expert_bias.tensor.get_value(), "the expert bias").reshape(-1)
    # The update from the counts of the whole batch. The counts and their average (quarters here) are exact in FP32,
    # and so are the comparisons with the mean; bf16 holds the step of 0.001 to about 5e-7.
    whole = np.mean(counts, axis=0)
    expected = coeff * np.sign(whole.mean() - whole)
    expected -= expected.mean()
    np.testing.assert_allclose(bias, expected, atol=1e-5)
    assert not any(
        c.any() for c in device_slices(layer._token_counts.tensor.get_value(NATIVE))
    ), "token counts not reset"
    # Both buffers keep their replicated label, not the batch-sharded one of the activations the counts came from:
    # whatever reads one copy as the value (a checkpoint, a gather) reads the bias every device holds.
    for name in ("_expert_bias", "_token_counts"):
        shards = _sharded_placements(getattr(layer, name).tensor.get_value(NATIVE))
        assert not shards, f"{name} is labelled sharded: {shards}"


def test_the_update_compares_averages_bf16_cannot_hold(hsdp_mesh):
    """Each device's counts are integers, but their average is not, and near the mean its fraction decides the sign:
    against a mean of 300, averages of 300.5 and 299.5 must move the bias down and up, while BF16, two apart there,
    rounds both to the mean. Each device's counts are set directly, to even numbers BF16 holds exactly."""
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.set_seed(SEED)
    layer = SparseMoEEP(_Cfg())
    local = np.full((hsdp_mesh.num_devices(), 1, 1, N_EXPERTS), 300.0, dtype=np.float32)
    local[0, 0, 0] = [302, 298, 300, 302, 298, 300, 302, 298]
    stored = layer._token_counts.tensor.get_value(NATIVE)
    mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(ctx.get_device(), 0, None)
    counts = ttml.autograd.Tensor.from_numpy(local, ttnn.Layout.TILE, stored.dtype, mapper).get_value(NATIVE)
    counts.update_tensor_topology(stored.tensor_topology())
    layer._token_counts.tensor.set_value(counts)

    coeff = 0.001
    layer.update_expert_bias(coeff)

    bias = assert_identical_across_devices(layer._expert_bias.tensor.get_value(), "the expert bias").reshape(-1)
    whole = local.mean(axis=0).reshape(-1)  # 300.5, 299.5 and 300 around a mean of 300
    expected = coeff * np.sign(whole.mean() - whole)
    expected -= expected.mean()
    assert np.count_nonzero(expected) == 6, "precondition: six experts off the mean"
    np.testing.assert_allclose(bias, expected, atol=1e-5)
