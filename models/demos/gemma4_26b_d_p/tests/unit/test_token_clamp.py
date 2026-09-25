# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt-d-gen pads chunk tails with 0xFFFFFFFF; the embedding must see a valid id there."""

import pytest
import torch

import ttnn
from models.demos.gemma4_26b_d_p.tt.model import clamp_pad_tokens

PAD = 0xFFFFFFFF


@pytest.mark.parametrize("mesh_device, device_params", [pytest.param((1, 1), {}, id="1x1")], indirect=True)
def test_clamp_pad_tokens(mesh_device, device_params):
    V = 262144
    ids = torch.randint(0, V, (1, 1, 1024), dtype=torch.int64)
    ids[..., 900:] = PAD
    t = ttnn.from_torch(ids.to(torch.uint32) if hasattr(torch, "uint32") else ids.to(torch.int64), device=mesh_device, dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    out = clamp_pad_tokens(t, V)
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).to(torch.int64)
    exp = ids.clone()
    exp[..., 900:] = 0
    assert torch.equal(got.reshape(exp.shape), exp), (got.flatten()[895:905], exp.flatten()[895:905])
