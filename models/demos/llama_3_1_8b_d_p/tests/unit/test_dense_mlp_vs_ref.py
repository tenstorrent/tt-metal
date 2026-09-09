# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The dense MLP at real dims (4096 -> 14336 -> 4096) vs the torch reference.

Target mesh (8, 4), random weights, identical on both sides. Structure follows
`minimax_m3/tests/unit/test_dense_mlp_vs_ref.py`.

This is where the column-parallel / row-parallel split and the closing TP collective are measured
together: `gate`/`up` shard the intermediate dim across TP, `down` shards the contraction dim so
every chip holds a partial sum, and the collective is what turns those partials back into the
answer. Both closing modes are covered, because which one runs depends on the residual scheme.

`down_proj` contracts over 14336, the deepest contraction in the layer — which is exactly where
bf16 destination accumulation costs PCC. The module pins `fp32_dest_acc_en=True`.
"""

import pytest
import torch

import ttnn
from models.demos.llama_3_1_8b_d_p.reference.model import REF_DTYPE, RefMLP
from models.demos.llama_3_1_8b_d_p.tt.dense_mlp import DenseMLP

from ..test_factory import ACT_DTYPE, WEIGHT_DTYPE, assert_pcc, parametrize_target_mesh

SEQ = 512


@parametrize_target_mesh()
@pytest.mark.parametrize("scatter_output", [False, True], ids=["all_reduce", "reduce_scatter"])
def test_dense_mlp_vs_ref(
    mesh_device, device_params, config, hf_config, mesh_config, ccl_manager, topology_name, scatter_output
):
    """MLP output vs the torch reference, for both closing collectives."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, SEQ, config.hidden_size, dtype=REF_DTYPE)
    ref = RefMLP(config)
    with torch.no_grad():
        golden = ref(x)

    state_dict = {
        "gate_proj.weight": ref.gate_proj.weight.detach().clone(),
        "up_proj.weight": ref.up_proj.weight.detach().clone(),
        "down_proj.weight": ref.down_proj.weight.detach().clone(),
    }
    mlp = DenseMLP(
        mesh_device,
        hf_config,
        state_dict,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
        weight_dtype=WEIGHT_DTYPE,
        scatter_output=scatter_output,
    )

    # Input is full emb, sequence-sharded on the SP rows and replicated across TP.
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ACT_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[2, None]),
    )
    tt_out = mlp(tt_x)
    ttnn.synchronize_device(mesh_device)

    if scatter_output:
        # emb/tp per TP column: concat the sequence over SP rows and the features over TP cols.
        got = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=tuple(mesh_device.shape)),
        )
    else:
        # full emb on every TP column: concat the sequence, take one column.
        got = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 0), mesh_shape=tuple(mesh_device.shape)),
        )[:1]

    assert_pcc(f"dense_mlp[{'reduce_scatter' if scatter_output else 'all_reduce'}]", golden, got, topology_name)


@parametrize_target_mesh()
def test_dense_mlp_rejects_non_silu(mesh_device, device_params, hf_config, mesh_config, ccl_manager):
    """A config whose activation is not silu must be rejected rather than silently run as SwiGLU."""
    hf_config.hidden_act = "gelu"
    with pytest.raises(AssertionError, match="silu"):
        DenseMLP(mesh_device, hf_config, {}, mesh_config=mesh_config, ccl_manager=ccl_manager)
