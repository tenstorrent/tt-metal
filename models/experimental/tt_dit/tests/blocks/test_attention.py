# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import diffusers.models.attention_processor
import pytest
import torch
import ttnn

from ...blocks.attention import Attention
from ...models.transformers.transformer_motif import convert_motif_attention_state
from ...parallel.config import DiTParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...reference.motif.modeling_dit import JointAttn as MotifAttentionReference
from ...utils import tensor
from ...utils.check import assert_quality
from ...utils.padding import PaddingConfig
from ...utils.tensor import bf16_tensor


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis"),
    [
        pytest.param((2, 4), (1, 1), 0, 1, id="1x1sp0tp1"),
        pytest.param((2, 4), (1, 2), 0, 1, id="1x2sp0tp1"),
        pytest.param((2, 4), (1, 2), 1, 0, id="1x2sp1tp0"),
        pytest.param((2, 4), (2, 1), 0, 1, id="2x1sp0tp1"),
        pytest.param((2, 4), (2, 1), 1, 0, id="2x1sp1tp0"),
        pytest.param((2, 4), (2, 2), 0, 1, id="2x2sp0tp1"),
        pytest.param((2, 4), (2, 2), 1, 0, id="2x2sp1tp0"),
        pytest.param((2, 4), (2, 4), 0, 1, id="2x4sp0tp1"),
        pytest.param((2, 4), (2, 4), 1, 0, id="2x4sp1tp0"),
        pytest.param((4, 8), (4, 4), 0, 1, id="4x4sp0tp1"),
        pytest.param((4, 8), (4, 8), 0, 1, id="4x8sp0tp1"),
        pytest.param((4, 8), (4, 8), 1, 0, id="4x8sp1tp0"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("batch_size", "spatial_seq_len", "prompt_seq_len"),
    [
        (1, 4096, 512),
        # (1, 4096 + 512, 0),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_attention_flux(
    *,
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    batch_size: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
) -> None:
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    torch.manual_seed(0)

    query_dim = 3072
    added_kv_proj_dim = 3072
    head_dim = 128
    num_heads = 24
    out_dim = 3072
    context_pre_only = prompt_seq_len == 0
    pre_only = prompt_seq_len == 0
    batch_size = 1

    joint_attention = added_kv_proj_dim != 0

    sp_factor = mesh_device.shape[sp_axis]
    tp_factor = mesh_device.shape[tp_axis]

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
        processor=diffusers.models.attention_processor.FluxAttnProcessor2_0(),  # type: ignore[arg-type]
    )
    torch_model.eval()

    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=1,
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
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    spatial = torch.randn((batch_size, spatial_seq_len, query_dim))
    prompt = torch.randn((batch_size, prompt_seq_len, query_dim)) if joint_attention else None
    rope_cos = torch.randn([spatial_seq_len + prompt_seq_len, 128])
    rope_sin = torch.randn([spatial_seq_len + prompt_seq_len, 128])

    spatial_padded = tt_model.pad_spatial_sequence(spatial, sp_factor=sp_factor)
    spatial_rope_cos_padded = tt_model.pad_spatial_sequence(rope_cos[prompt_seq_len:], sp_factor=sp_factor)
    spatial_rope_sin_padded = tt_model.pad_spatial_sequence(rope_sin[prompt_seq_len:], sp_factor=sp_factor)

    tt_spatial = bf16_tensor(spatial_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
    tt_prompt = bf16_tensor(prompt, device=mesh_device) if prompt is not None else None
    tt_spatial_rope_cos = bf16_tensor(spatial_rope_cos_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
    tt_spatial_rope_sin = bf16_tensor(spatial_rope_sin_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
    tt_prompt_rope_cos = bf16_tensor(rope_cos[:prompt_seq_len], device=mesh_device) if joint_attention else None
    tt_prompt_rope_sin = bf16_tensor(rope_sin[:prompt_seq_len], device=mesh_device) if joint_attention else None

    with torch.no_grad():
        torch_output = torch_model.forward(spatial, prompt, image_rotary_emb=(rope_cos, rope_sin))
        torch_spatial, torch_prompt = torch_output if joint_attention else (torch_output, None)

    tt_spatial_out, tt_prompt_out = tt_model.forward(
        spatial=tt_spatial,
        prompt=tt_prompt,
        spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
        prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin) if joint_attention else None,
        spatial_sequence_length=spatial_seq_len,
    )

    tt_spatial_torch = tensor.to_torch(tt_spatial_out, mesh_axes=[None, sp_axis, tp_axis])
    assert_quality(torch_spatial, tt_spatial_torch, pcc=0.995, relative_rmse=0.13)

    if torch_prompt is not None:
        tt_prompt_torch = tensor.to_torch(tt_prompt_out, mesh_axes=[None, None, tp_axis])
        assert_quality(torch_prompt, tt_prompt_torch, pcc=0.995, relative_rmse=0.13)


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

    tt_spatial_torch = tensor.to_torch(tt_spatial_out, mesh_axes=[None, sp_axis, tp_axis])
    tt_spatial_torch = tt_spatial_torch[:, :spatial_seq_len]
    assert_quality(torch_spatial, tt_spatial_torch, pcc=0.996, relative_rmse=0.13)

    if not context_pre_only:
        tt_prompt_torch = tensor.to_torch(tt_prompt_out, mesh_axes=[None, None, tp_axis])
        assert_quality(torch_prompt, tt_prompt_torch, pcc=0.996, relative_rmse=0.13)
