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
whole batch on every device, and both buffers are still labelled replicated.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttnn
import ttml
from ttml.models.deepseek.moe_sparse_ep import SparseMoEEP

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(1800)]

SEED = 2026
DIM = 64
SEQ = 32
N_EXPERTS = 8


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


@pytest.fixture(scope="module")
def hsdp_mesh(fresh_device_mesh):
    """``[2, 2]`` with axes ``("dp", "fsdp")``: both axes data-parallel."""
    with fresh_device_mesh((2, 2), ("dp", "fsdp"), what="DeepSeek expert-bias tests", require_mgd=True) as mesh:
        yield mesh


def _copies(value) -> list[np.ndarray]:
    """Each device's copy of a ``[1, 1, 1, E]`` buffer, flattened, in row-major mesh order."""
    return [ttnn.to_torch(shard).float().numpy().reshape(-1) for shard in ttnn.get_device_tensors(value)]


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
    counts = _copies(layer._token_counts.tensor.get_value())
    # Precondition: devices 0 and 1 are neighbours along "fsdp", devices 0 and 2 along "dp".
    for other, axis in ((1, "fsdp"), (2, "dp")):
        assert not np.array_equal(counts[0], counts[other]), f"precondition: the {axis} neighbours routed alike"

    coeff = 0.001
    layer.update_expert_bias(coeff)

    bias = _copies(layer._expert_bias.tensor.get_value())
    for device, copy in enumerate(bias[1:], start=1):
        assert np.array_equal(copy, bias[0]), f"device {device}'s expert bias differs from device 0's: {bias}"
    # The update from the counts of the whole batch. Counts are small integers and their averages quarters, so the
    # device's comparisons with the mean are exact; bf16 holds the step of 0.001 to about 5e-7.
    whole = np.mean(counts, axis=0)
    expected = coeff * np.sign(whole.mean() - whole)
    expected -= expected.mean()
    np.testing.assert_allclose(bias[0], expected, atol=1e-5)
    assert not any(c.any() for c in _copies(layer._token_counts.tensor.get_value())), "token counts not reset"
    # Both buffers keep their replicated label, not the batch-sharded one of the activations the counts came from:
    # whatever reads one copy as the value (a checkpoint, a gather) reads the bias every device holds.
    for name in ("_expert_bias", "_token_counts"):
        shards = _sharded_placements(getattr(layer, name).tensor.get_value())
        assert not shards, f"{name} is labelled sharded: {shards}"
