# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The model's FINAL norm — the same RMSNorm, applied to the tail instance rather than a layer's.

Target mesh (8, 4), random weights. This is the recipe's "test_norm_vs_ref applied to the tail" row.

What is different about the tail instance, and why it deserves its own test rather than being
covered by `test_norm_vs_ref.py`: it is pinned to the SINGLE-PASS form even under a sharded
residual (`is_distributed=False`). Its output feeds the column-parallel LM head, which needs full
emb anyway, so a distributed norm would cost three ops plus a gather where one op plus a gather
does — and it keeps this norm's gain replicated.

That pinning is a decision made in `tt/model.py`, so it is checked on the model's own instance, not
on a norm the test constructs.
"""

import torch

import ttnn
from models.demos.llama_3_1_8b_d_p.reference.model import REF_DTYPE, RefRMSNorm
from models.demos.llama_3_1_8b_d_p.tt.rms_norm import RMSNorm

from ..test_factory import ACT_DTYPE, assert_pcc, parametrize_target_mesh

SEQ = 512


@parametrize_target_mesh()
def test_final_norm_vs_ref(mesh_device, device_params, config, hf_config, mesh_config, ccl_manager, topology_name, monkeypatch):
    """The tail norm, pinned single-pass, vs the torch reference — under a SHARDED residual."""
    monkeypatch.setenv("LLAMA31_8B_SHARDED_RESIDUAL", "1")

    torch.manual_seed(0)
    x = torch.randn(1, 1, SEQ, config.hidden_size, dtype=REF_DTYPE)
    weight = (1.0 + 0.1 * torch.randn(config.hidden_size)).to(REF_DTYPE)
    ref = RefRMSNorm(config.hidden_size, config.rms_norm_eps)
    with torch.no_grad():
        ref.weight.copy_(weight)
        golden = ref(x)

    norm = RMSNorm(
        mesh_device,
        hf_config,
        {"weight": weight},
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
        is_distributed=False,  # what tt/model.py pins for the tail
    )
    assert not norm.is_distributed, "the final norm must stay single-pass even under a sharded residual"

    # Its input arrives as FULL emb: the model gathers before the tail norm.
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ACT_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[2, None]),
    )
    out = norm.forward(tt_x)
    ttnn.synchronize_device(mesh_device)
    got = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 0), mesh_shape=tuple(mesh_device.shape))
    )[:1]
    assert_pcc("final_norm", golden, got.to(REF_DTYPE), topology_name)
