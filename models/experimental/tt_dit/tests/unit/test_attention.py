# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import diffusers.models.attention_processor
import pytest
import torch
import ttnn

from ...layers.attention import Attention
from ...parallel.config import DiTParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...utils.check import assert_quality
from ...utils.padding import PaddingConfig
from ...utils.tensor import bf16_tensor


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links"),
    [
        pytest.param((1, 2), 0, 1, 1, id="1x2sp0tp1"),
        # pytest.param((2, 1), 1, 0, 1, id="2x1sp1tp0"),
        # pytest.param((2, 2), 0, 1, 1, id="2x2sp0tp1"),
        # pytest.param((2, 2), 1, 0, 1, id="2x2sp1tp0"),
        # pytest.param((2, 4), 0, 1, 1, id="2x4sp0tp1"),
        # pytest.param((2, 4), 1, 0, 1, id="2x4sp1tp0"),
        # pytest.param((4, 4), 0, 1, 4, id="4x4sp0tp1"),
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
        "added_head_scaling",
        # inputs
        "batch_size",
        "spatial_seq_len",
        "prompt_seq_len",
        "use_rope",
    ),
    [
        pytest.param(3072, 3072, 128, 24, 3072, False, False, False, 1, 4096, 512, True, id="flux1_joint"),
        pytest.param(3072, 0, 128, 24, 3072, True, True, False, 1, 4096 + 512, 0, True, id="flux1_single"),
        pytest.param(1920, 1920, 64, 30, 1920, False, False, True, 2, 4100, 333, False, id="motif"),
        # pytest.param(1920, 1920, 64, 30, 1920, True, False, True, 2, 4100, 333, False, id="motif_context_pre_only"),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_attention_flux(
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
    added_head_scaling: bool,
    batch_size: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
    use_rope: bool,
) -> None:
    joint_attention = added_kv_proj_dim != 0

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    torch_model = diffusers.models.attention_processor.Attention(
        query_dim=query_dim,
        added_kv_proj_dim=added_kv_proj_dim if added_kv_proj_dim != 0 else None,
        dim_head=head_dim,
        heads=num_heads,
        out_dim=out_dim,
        context_pre_only=context_pre_only,
        pre_only=pre_only,
        bias=True,
        qk_norm="rms_norm",
        eps=1e-6,
        processor=diffusers.models.attention_processor.FluxAttnProcessor2_0(),
    )
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
        pre_only=pre_only,
        eps=1e-6,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
        # added_head_scaling=added_head_scaling, # TODO
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch.manual_seed(0)
    spatial_input = torch.randn((batch_size, spatial_seq_len, query_dim))
    prompt_input = torch.randn((batch_size, prompt_seq_len, query_dim)) if joint_attention else None
    rope_cos = torch.randn([spatial_seq_len + prompt_seq_len, 128]) if use_rope else None
    rope_sin = torch.randn([spatial_seq_len + prompt_seq_len, 128]) if use_rope else None

    tt_spatial = bf16_tensor(spatial_input, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
    tt_prompt = bf16_tensor(prompt_input, device=mesh_device) if prompt_input is not None else None
    tt_spatial_rope_cos = (
        bf16_tensor(rope_cos[prompt_seq_len:], device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
        if rope_cos is not None
        else None
    )
    tt_spatial_rope_sin = (
        bf16_tensor(rope_sin[prompt_seq_len:], device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
        if rope_sin is not None
        else None
    )
    tt_prompt_rope_cos = (
        bf16_tensor(rope_cos[:prompt_seq_len], device=mesh_device) if joint_attention and rope_cos is not None else None
    )
    tt_prompt_rope_sin = (
        bf16_tensor(rope_sin[:prompt_seq_len], device=mesh_device) if joint_attention and rope_sin is not None else None
    )

    with torch.no_grad():
        torch_output = torch_model.forward(
            spatial_input, prompt_input, image_rotary_emb=(rope_cos, rope_sin) if use_rope else None
        )
        torch_spatial, torch_prompt = torch_output if joint_attention else (torch_output, None)

    tt_spatial_out, tt_prompt_out = tt_model.forward(
        spatial=tt_spatial,
        prompt=tt_prompt,
        spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin) if use_rope else None,
        prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin) if joint_attention and use_rope else None,
        spatial_sequence_length=spatial_seq_len,
    )

    shard_dims = [None, None]
    shard_dims[sp_axis], shard_dims[tp_axis] = 1, 2
    tt_spatial_torch = ttnn.to_torch(
        tt_spatial_out,
        mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(shard_dims)),
    )
    assert_quality(torch_spatial, tt_spatial_torch, pcc=0.995, relative_rmse=0.12)

    if torch_prompt is not None:
        shard_dims = [None, None]
        shard_dims[sp_axis], shard_dims[tp_axis] = 0, 2
        tt_prompt_torch = ttnn.to_torch(
            tt_prompt_out,
            mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(shard_dims)),
        )[:batch_size]

        assert_quality(torch_prompt, tt_prompt_torch, pcc=0.9957, relative_rmse=0.12)
