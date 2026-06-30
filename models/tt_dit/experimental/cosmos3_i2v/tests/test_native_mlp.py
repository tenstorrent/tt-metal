# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the native Cosmos3VLTextMLP (SwiGLU), TP variants.

Builds the torch reference (`Cosmos3VLTextMLP`) with random weights and
runs the same weights through `Cosmos3VLTextMLP` (native) on subsets of
a WH LoudBox mesh — (1,1), (1,2), (1,4), (1,8).

Test config picks intermediate_size = 512 so the fused gate+up output
(doubled to 1024) splits cleanly across tp ∈ {1, 2, 4, 8}: per chip
512/tp out features after activation, with 2 tile-sized chunks per
device for the SwiGLU interleave permutation.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.tensor import bf16_tensor
from models.tt_dit.utils.test import line_params


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((1, 1), line_params, id="1x1"),
        pytest.param((1, 2), line_params, id="1x2"),
        pytest.param((1, 4), line_params, id="1x4"),
        pytest.param((1, 8), line_params, id="1x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(300)
def test_native_mlp(mesh_device: ttnn.MeshDevice) -> None:
    from models.tt_dit.experimental.cosmos3_i2v.model.mlp import Cosmos3VLTextMLP as TTCosmos3VLTextMLP
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import (
        Cosmos3VLTextMLP as RefCosmos3VLTextMLP,
    )

    torch.manual_seed(42)

    mesh_shape = tuple(mesh_device.shape)
    tp_factor = mesh_shape[1]
    sp_factor = mesh_shape[0]

    hidden_size = 256
    intermediate_size = 512
    N = 128

    torch_mlp = RefCosmos3VLTextMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
    torch_mlp.eval()
    torch_mlp.to(dtype=torch.bfloat16)

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        sequence_parallel=ParallelFactor(sp_factor, 0),
        tensor_parallel=ParallelFactor(tp_factor, 1),
    )
    ccl_manager = (
        CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear) if tp_factor > 1 else None
    )

    tt_mlp = TTCosmos3VLTextMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    tt_mlp.load_torch_state_dict(torch_mlp.state_dict())

    x = torch.randn(N, hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        torch_out = torch_mlp(x)

    x_tt = bf16_tensor(x.reshape(1, 1, N, hidden_size), device=mesh_device)
    tt_out = tt_mlp(x_tt)

    tt_out_view = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).reshape(N, hidden_size)

    assert_quality(torch_out, tt_out_view, pcc=0.98)
