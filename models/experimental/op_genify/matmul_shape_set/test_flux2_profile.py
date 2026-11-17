# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Profile the stock FLUX.2-dev DiT (tt_dit) for the #65148 matmul shape set.

Same TT model, mesh and parallel settings as the QB2 row of
`models/tt_dit/tests/models/flux2/test_performance_flux2.py` (2x2 mesh, SP on axis 0, TP on axis 1,
Linear, 2 links, FSDP, shard_prompt=True, 1024x1024, 512 prompt tokens). The FLUX.2-dev checkpoint is
gated, so the weights are random at the FLUX.2-dev config (diffusers' Flux2Transformer2DModel defaults:
6144 hidden = 48 heads x 128, joint_attention_dim 15360, 128 latent channels). Matmul shapes, dtypes,
memory configs and device time do not depend on the weight values.

One double and two single blocks: every matmul outside the blocks (time/guidance MLPs, modulation,
x_embedder, context_embedder, proj_out) runs at full size, every double block has the same shapes, and
the first single block has the shapes of single blocks 1-47 while the second is the final one, which
skips the prompt's MLP and proj_out (8 double + 48 single in the full model).

Run (needs a Tracy build):
    scripts/run_safe_pytest.sh --profile models/experimental/op_genify/matmul_shape_set/test_flux2_profile.py
"""

import os

import diffusers
import pytest
import torch
from tracy import signpost

import ttnn
from models.tt_dit.models.transformers.transformer_flux2 import Flux2Transformer
from models.tt_dit.parallel.config import DiTGParallelConfigNoCFG, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tests.models.flux2.device_params import line_params_flux2_transformer
from models.tt_dit.utils import cache, tensor

NUM_SINGLE = 2


def _prepare_ids(*, height=1, width=1, text_sequence_length=1):
    return torch.cartesian_prod(
        torch.arange(1), torch.arange(height), torch.arange(width), torch.arange(text_sequence_length)
    )


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, topology, num_links, fsdp, shard_prompt, device_params",
    [[(2, 2), 0, 1, ttnn.Topology.Linear, 2, True, True, line_params_flux2_transformer]],
    ids=["bh_2x2_linear_pipeline"],
    indirect=["mesh_device", "device_params"],
)
def test_flux2_matmul_profile(mesh_device, sp_axis, tp_axis, topology, num_links, fsdp, shard_prompt):
    # Never read or write a weight cache: these weights are random.
    os.environ.pop("TT_DIT_CACHE_DIR", None)
    batch_size, height, width, prompt_seq_len = 1, 1024, 1024, 512
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    torch.manual_seed(0)
    torch_model = (
        diffusers.Flux2Transformer2DModel(num_layers=1, num_single_layers=NUM_SINGLE).to(torch.bfloat16).eval()
    )
    config = torch_model.config

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTGParallelConfigNoCFG(
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )
    assert config.num_attention_heads % tp_factor == 0
    tt_model = Flux2Transformer(
        in_channels=torch_model.in_channels,
        num_layers=1,
        num_single_layers=NUM_SINGLE,
        attention_head_dim=config.attention_head_dim,
        num_attention_heads=config.num_attention_heads,
        joint_attention_dim=config.joint_attention_dim,
        out_channels=torch_model.out_channels,
        device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=None,
        is_fsdp=fsdp,
        shard_prompt=shard_prompt,
    )
    cache.load_model(
        tt_model,
        get_torch_state_dict=torch_model.state_dict,
        model_name="FLUX.2-dev-random-weights-profile",
        subfolder=f"transformer_L1_S{NUM_SINGLE}",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        mesh_device=mesh_device,
        is_fsdp=fsdp,
        create_cache=False,
    )

    spatial_seq_len = height * width // 16**2
    spatial = torch.randn([batch_size, spatial_seq_len, torch_model.in_channels])
    prompt = torch.randn([batch_size, prompt_seq_len, config.joint_attention_dim])
    text_ids = _prepare_ids(text_sequence_length=prompt_seq_len)
    image_ids = _prepare_ids(height=height // 16, width=width // 16)
    prompt_rope_cos, prompt_rope_sin = torch_model.pos_embed.forward(text_ids)
    spatial_rope_cos, spatial_rope_sin = torch_model.pos_embed.forward(image_ids)

    tt_spatial = tensor.from_torch(spatial, device=mesh_device, mesh_axes=[None, sp_axis, None])
    prompt_mesh_axes = [None, sp_axis, None] if shard_prompt else None
    prompt_rope_mesh_axes = [None, None, sp_axis, None] if shard_prompt else None
    tt_prompt = tensor.from_torch(prompt, device=mesh_device, mesh_axes=prompt_mesh_axes)
    tt_timestep = tensor.from_torch(torch.full([batch_size, 1], 500), dtype=ttnn.float32, device=mesh_device)
    tt_guidance = tensor.from_torch(torch.full([batch_size, 1], 3), device=mesh_device)

    def rope(t, axes):
        return tensor.from_torch(t.unsqueeze(0).unsqueeze(0), device=mesh_device, mesh_axes=axes)

    spatial_rope = (
        rope(spatial_rope_cos, [None, None, sp_axis, None]),
        rope(spatial_rope_sin, [None, None, sp_axis, None]),
    )
    prompt_rope = (rope(prompt_rope_cos, prompt_rope_mesh_axes), rope(prompt_rope_sin, prompt_rope_mesh_axes))
    combined_rope = (
        ttnn.concat([spatial_rope[0], prompt_rope[0]], dim=2),
        ttnn.concat([spatial_rope[1], prompt_rope[1]], dim=2),
    )

    def step(embedded_prompt):
        return tt_model.forward(
            spatial=tt_spatial,
            embedded_prompt=embedded_prompt,
            timestep=tt_timestep,
            guidance=tt_guidance,
            spatial_rope=spatial_rope,
            prompt_rope=prompt_rope,
            combined_rope=combined_rope,
            spatial_sequence_length=spatial_seq_len,
            prompt_sequence_length=prompt_seq_len,
        )

    # Warm-up: compile every program once.
    step(tt_model.context_embedder(tt_prompt))
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)

    # Once per image: the prompt projection.
    signpost("CONTEXT_EMBEDDER")
    embedded_prompt = tt_model.context_embedder(tt_prompt)
    ttnn.synchronize_device(mesh_device)
    signpost("CONTEXT_EMBEDDER_END")
    ttnn.ReadDeviceProfiler(mesh_device)

    # Once per denoising step (50 per image).
    signpost("DIT_STEP")
    out = step(embedded_prompt)
    ttnn.synchronize_device(mesh_device)
    signpost("DIT_STEP_END")
    ttnn.ReadDeviceProfiler(mesh_device)
    assert tuple(out.shape)[-1] == torch_model.out_channels
