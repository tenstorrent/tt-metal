# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from ...layers.attention import Attention
from ...models.transformers.transformer_motif import convert_motif_attention_state
from ...parallel.config import DiTParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...reference.motif.modeling_dit import JointAttn as MotifAttentionReference
from ...utils.check import assert_quality
from ...utils.padding import PaddingConfig
from ...utils.tensor import bf16_tensor, to_torch


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links"),
    [
        pytest.param((1, 2), 0, 1, 1, id="1x2sp0tp1"),
        pytest.param((2, 1), 1, 0, 1, id="2x1sp1tp0"),
        pytest.param((2, 2), 0, 1, 1, id="2x2sp0tp1"),
        pytest.param((2, 2), 1, 0, 1, id="2x2sp1tp0"),
        pytest.param((2, 4), 0, 1, 1, id="2x4sp0tp1"),
        # pytest.param((2, 4), 1, 0, 1, id="2x4sp1tp0"),  # hangs
        pytest.param((4, 4), 0, 1, 4, id="4x4sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    (
        # model
        "query_dim",
        "added_kv_proj_dim",
        "head_dim",
        "num_heads",
        "out_dim",
        "context_pre_only",
        "pre_only",
        # inputs
        "batch_size",
        "spatial_seq_len",
        "prompt_seq_len",
    ),
    [
        pytest.param(1920, 1920, 64, 30, 1920, False, False, 2, 4100, 333, id="motif"),
        pytest.param(1920, 1920, 64, 30, 1920, True, False, 2, 4100, 333, id="motif_context_pre_only"),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_attention_motif(
    *,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    query_dim: int,
    added_kv_proj_dim: int,
    head_dim: int,
    num_heads: int,
    out_dim: int,
    context_pre_only: bool,
    pre_only: bool,
    batch_size: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
) -> None:
    torch.manual_seed(0)

    sp_factor = mesh_device.shape[sp_axis]
    tp_factor = mesh_device.shape[tp_axis]

    inner_dim = num_heads * head_dim

    class ReferenceAttnConfig:
        def __init__(self) -> None:
            self.hidden_dim = inner_dim
            self.num_attention_heads = num_heads
            self.attn_mode = "flash"

    torch_model = MotifAttentionReference(ReferenceAttnConfig())
    torch_model.eval()

    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=ttnn.Topology.Linear,
    )

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=0, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )

    if num_heads % tp_factor != 0:
        padding_config = PaddingConfig.from_tensor_parallel_factor(num_heads, head_dim, tp_factor)
    else:
        padding_config = None

    tt_model = Attention(
        query_dim=query_dim,
        head_dim=head_dim,
        heads=num_heads,
        out_dim=out_dim,
        added_kv_proj_dim=added_kv_proj_dim,
        context_pre_only=context_pre_only,
        context_head_scaling=True,
        pre_only=pre_only,
        eps=1e-6,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )

    converted_state_dict = dict(torch_model.state_dict())
    convert_motif_attention_state(
        converted_state_dict,
        x_weight=torch.eye(inner_dim),
        x_bias=torch.zeros([inner_dim]),
        c_weight=torch.eye(inner_dim),
        c_bias=torch.zeros([inner_dim]),
        is_last_block=context_pre_only,
    )
    tt_model.load_torch_state_dict(converted_state_dict)

    spatial = torch.randn((batch_size, spatial_seq_len, query_dim))
    prompt = torch.randn((batch_size, prompt_seq_len, query_dim))

    spatial_padded = tt_model.pad_spatial_sequence(spatial, sp_factor=sp_factor)

    tt_spatial = bf16_tensor(spatial_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
    tt_prompt = bf16_tensor(prompt, device=mesh_device) if prompt is not None else None

    with torch.no_grad():
        assert isinstance(spatial, torch.FloatTensor)
        assert isinstance(prompt, torch.FloatTensor)
        torch_spatial, torch_prompt = torch_model.forward(spatial, prompt)

    tt_spatial_out, tt_prompt_out = tt_model.forward(
        spatial=tt_spatial,
        prompt=tt_prompt,
        spatial_sequence_length=spatial_seq_len,
    )

    tt_spatial_torch = to_torch(tt_spatial_out, device=mesh_device, mesh_mapping={sp_axis: 1, tp_axis: 2})
    tt_spatial_torch = tt_spatial_torch[:, :spatial_seq_len]
    assert_quality(torch_spatial, tt_spatial_torch, pcc=0.996, relative_rmse=0.13)

    if not context_pre_only:
        tt_prompt_torch = to_torch(tt_prompt_out, device=mesh_device, mesh_mapping={tp_axis: 2})
        assert_quality(torch_prompt, tt_prompt_torch, pcc=0.996, relative_rmse=0.13)
