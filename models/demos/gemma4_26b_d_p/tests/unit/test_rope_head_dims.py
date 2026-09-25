# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""rotary_embedding_llama prefill across head dims (the compute kernel is shared with
rotary_embedding_indexed). Regression for the DEST-blocking fix: head_dim 512 = 16 tiles > DEST."""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_transformers.tt.common import get_rot_transformation_mat


@pytest.mark.parametrize("mesh_device, device_params", [pytest.param((1, 1), {}, id="1x1")], indirect=True)
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("fp32_acc", [False, True], ids=["bf16dst", "fp32dst"])
def test_rope_llama_head_dim(mesh_device, device_params, head_dim, fp32_acc):
    torch.manual_seed(0)
    H, S = 4, 256
    x = torch.randn(1, H, S, head_dim)
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    f = torch.outer(torch.arange(S).float(), inv)
    cos = torch.stack([f.cos(), f.cos()], -1).flatten(-2)[None, None]
    sin = torch.stack([f.sin(), f.sin()], -1).flatten(-2)[None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    ref = x * cos + torch.stack([-x2, x1], -1).flatten(-2) * sin
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    to = lambda t: ttnn.from_torch(t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=rep)
    kcfg = ttnn.init_device_compute_kernel_config(mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_acc)
    out = ttnn.experimental.rotary_embedding_llama(
        to(x), to(cos), to(sin), to(get_rot_transformation_mat()), is_decode_mode=False, compute_kernel_config=kcfg
    )
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    ok, pcc = comp_pcc(ref, got, 0.9995)
    logger.info(f"rope_llama D={head_dim} fp32_acc={fp32_acc}: {pcc}")
    assert ok, pcc
