# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Zero-weights forward test for ``MLP.update`` (dummy weights).

Zeroing every projection must make ``MLP.forward`` collapse to all zeros,
exercising the ``ttnn.copy`` paths in ``MLP._update_w{1,2,3}``.
"""

from __future__ import annotations

import pytest
import torch

from _completer_utils import as_update_input, open_completer

# Stays below args.prefill_len_cutoff so MLP.forward skips the chunked-prefill branch.
SEQ_LEN = 128


@pytest.fixture(scope="module")
def completer_and_mlp():
    with open_completer(dummy_weights=True) as completer:
        model = completer.models[0]
        yield completer, model.layers[0].feed_forward


def _build_random_mlp_input(completer):
    """Construct a synthetic random ``(1, 1, SEQ_LEN, dim)`` MLP input."""
    import ttnn

    model = completer.models[0]
    dim = model.args.dim
    return ttnn.from_torch(
        torch.randn(1, 1, SEQ_LEN, dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=model.mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
    )


def test_mlp_forward_is_zero_when_weights_are_zero(completer_and_mlp):
    """Zero all three MLP projections via ``MLP.update`` and check the
    forward output is elementwise zero."""
    import ttnn

    from models.tt_transformers.tt.common import Mode

    completer, mlp = completer_and_mlp

    H = mlp.args.dim
    I = mlp.args.hidden_dim
    gate_hf = torch.zeros(I, H, dtype=torch.bfloat16)
    up_hf = torch.zeros(I, H, dtype=torch.bfloat16)
    down_hf = torch.zeros(H, I, dtype=torch.bfloat16)

    mlp.update(
        gate_proj=as_update_input(gate_hf, mlp.mesh_device),
        up_proj=as_update_input(up_hf, mlp.mesh_device),
        down_proj=as_update_input(down_hf, mlp.mesh_device),
    )

    x = _build_random_mlp_input(completer)
    out = ttnn.to_torch(mlp.forward(x, Mode.PREFILL))

    assert torch.equal(out, torch.zeros_like(out)), (
        f"MLP.forward != 0 after zeroing gate/up/down_proj: "
        f"max|out|={float(out.abs().max()):.6g}, mean|out|={float(out.abs().mean()):.6g}"
    )
