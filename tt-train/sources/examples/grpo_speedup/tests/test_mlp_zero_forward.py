# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Zero-weights forward pytest test for ``MLP.update``.

After ``MLP.update`` zeros every projection, ``MLP.forward`` must
collapse to all zeros:

* gate = x @ w1  -> 0 because w1 = 0
* up   = x @ w3  -> 0 because w3 = 0
* act  = silu(gate) * up = silu(0) * 0 = 0
* out  = act @ w2 = 0 @ w2 = 0

Exercises every ``ttnn.copy`` path in ``MLP._update_w{1,2,3}`` without
needing real model weights -- uses ``dummy_weights=True``, no HF auth.
"""

from __future__ import annotations

import pytest
import torch

from _completer_utils import build_completer, teardown_completer

# Prefill seq_len for the synthetic MLP input. 128 stays well below
# args.prefill_len_cutoff (512 on WH / 1024 on BH) so MLP.forward does
# not take the chunked-prefill reshape branch.
SEQ_LEN = 128


@pytest.fixture(scope="module")
def completer_and_mlp():
    completer = build_completer(dummy_weights=True)
    try:
        yield completer, completer.model.layers[0].feed_forward
    finally:
        teardown_completer(completer)


def _w1_w3_mesh_mapper(mlp):
    """Mirror the mesh mapper used in ``MLP.__init__`` for ``self.w1`` and ``self.w3``."""
    import ttnn

    dims = (-1, -2) if mlp.args.is_galaxy else (-2, -1)
    return ttnn.ShardTensor2dMesh(mlp.mesh_device, dims=dims, mesh_shape=mlp.args.cluster_shape)


def _w2_mesh_mapper(mlp):
    """Mirror the mesh mapper used in ``MLP.__init__`` for ``self.w2``."""
    import ttnn

    dims = (-2, -1) if mlp.args.is_galaxy else (-1, -2)
    return ttnn.ShardTensor2dMesh(mlp.mesh_device, dims=dims, mesh_shape=mlp.args.cluster_shape)


def _zeros_like(mlp, template, mapper):
    """Build an all-zeros ``ttnn.Tensor`` matching ``template``."""
    import ttnn

    return ttnn.from_torch(
        torch.zeros(tuple(template.shape), dtype=torch.bfloat16),
        dtype=template.dtype,
        layout=template.layout,
        device=mlp.mesh_device,
        memory_config=template.memory_config(),
        mesh_mapper=mapper,
    )


def _build_random_mlp_input(completer):
    """Construct a synthetic ``(1, 1, SEQ_LEN, dim)`` MLP input.

    Random values exercise the "activations multiply against weights"
    path properly -- if w1/w3 are exactly zero the product is exactly
    zero regardless of the activation magnitudes.
    """
    import ttnn

    dim = completer.model_args.dim
    return ttnn.from_torch(
        torch.randn(1, 1, SEQ_LEN, dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=completer.mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(completer.mesh_device),
    )


def test_mlp_forward_is_zero_when_weights_are_zero(completer_and_mlp):
    """Zero all three MLP projections via ``MLP.update`` and check the
    forward output is elementwise zero."""
    import ttnn

    from models.tt_transformers.tt.common import Mode

    completer, mlp = completer_and_mlp

    mlp.update(
        w1=_zeros_like(mlp, mlp.w1, _w1_w3_mesh_mapper(mlp)),
        w2=_zeros_like(mlp, mlp.w2, _w2_mesh_mapper(mlp)),
        w3=_zeros_like(mlp, mlp.w3, _w1_w3_mesh_mapper(mlp)),
    )

    x = _build_random_mlp_input(completer)
    out = ttnn.to_torch(mlp.forward(x, Mode.PREFILL))

    assert torch.equal(out, torch.zeros_like(out)), (
        f"MLP.forward != 0 after zeroing w1/w2/w3: "
        f"max|out|={float(out.abs().max()):.6g}, mean|out|={float(out.abs().mean()):.6g}"
    )
