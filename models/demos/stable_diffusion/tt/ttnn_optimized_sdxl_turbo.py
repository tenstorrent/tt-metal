# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from typing import Optional, Tuple, Union
from models.demos.stable_diffusion.tt.utils import Timesteps
from models.demos.stable_diffusion.tt.ttnn_resnetblock2d import ResnetBlock2D
from models.demos.stable_diffusion.tt.ttnn_downblock2d import down_block_2d
from models.demos.stable_diffusion.tt.resnetblock2d_utils import (
    get_inputs,
    run_conv,
    run_conv_with_split,
)

from models.demos.stable_diffusion.tt.ttnn_transformer2d import sd_transformer_2d
from models.demos.stable_diffusion.tt.ttnn_cross_attention_downblock2d import sd_cross_attention_down_blocks2d
from models.demos.stable_diffusion.tt.ttnn_cross_attention_upblock2d import sd_crossattnupblock2d, up_block_2d


def timestep_embedding(x, parameters, device):
    x = ttnn.linear(
        x,
        parameters.linear_1.weight,
        bias=parameters.linear_1.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        activation="silu",
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )

    x = ttnn.linear(
        x,
        parameters.linear_2.weight,
        bias=parameters.linear_2.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )

    return x


def get_aug_embed(emb, text_embeds, time_ids, parameters, device):
    time_ids = ttnn.to_torch(time_ids)

    time_ids = ttnn.from_torch(time_ids.flatten(), dtype=ttnn.bfloat16, device=device)
    time_embeds = Timesteps(time_ids, 256, True, 0, device=device, flag=False)
    time_embeds = ttnn.reshape(time_embeds, (text_embeds.shape[0], -1))
    text_embeds = ttnn.to_torch(text_embeds)
    time_embeds = ttnn.to_torch(time_embeds)
    add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
    add_embeds = ttnn.from_torch(add_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    aug_emb = timestep_embedding(add_embeds, parameters.add_embedding, device)
    return aug_emb


def get_time_embed(sample, time_step, num_channels, flip_sin_to_cos, downscale_freq_shift, scale, device):
    timesteps = time_step
    # timesteps = ttnn.to_torch(timesteps)
    # timesteps = timesteps.expand(sample.shape[0])
    # timesteps = ttnn.expand(timesteps, sample.shape[0])
    t_emb = Timesteps(timesteps, num_channels, flip_sin_to_cos, downscale_freq_shift, scale, device)

    return t_emb


def stable_diffusion_xl_turbo(
    config,
    sample,
    timestep,
    encoder_hidden_states,
    text_embeds,
    time_ids,
    block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
    transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
    attention_head_dim: Union[int, Tuple[int]] = 8,
    addition_time_embed_dim: Optional[int] = None,
    time_embedding_dim: Optional[int] = None,
    parameters=None,
    device=None,
):
    t_emb = get_time_embed(
        sample,
        timestep,
        num_channels=block_out_channels[0],
        flip_sin_to_cos=True,
        downscale_freq_shift=0,
        scale=1,
        device=device,
    )
    t_emb = ttnn.to_device(t_emb, device=device)
    emb = timestep_embedding(t_emb, parameters.time_embedding, device)
    aug_emb = get_aug_embed(
        emb=emb,
        text_embeds=text_embeds,
        time_ids=time_ids,
        parameters=parameters,
        device=device,
    )

    emb = emb + aug_emb if aug_emb is not None else emb
    N = sample.shape[0]
    C = sample.shape[1]
    H = sample.shape[2]
    W = sample.shape[3]
    sample = ttnn.permute(sample, (0, 2, 3, 1))
    sample = run_conv(
        device,
        output_channels=parameters.conv_in.bias.shape[-1],
        input_channels=C,
        input_height=H,
        input_width=W,
        filter_height=3,
        stride_h=1,
        pad_h=1,
        tt_input_tensor=sample,
        tt_weight_tensor=parameters.conv_in.weight,
        tt_bias_tensor=parameters.conv_in.bias,
    )
    down_block_res_samples = (sample,)
    add_downsample = True

    for index in range(0, 3):
        if index == 0:
            sample, output_states = down_block_2d(device, parameters.down_blocks[index], config, sample, emb)
        else:
            sample, output_states = sd_cross_attention_down_blocks2d(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                config=config,
                conv_shortcut=True,
                use_torch_conv=False,
                device=device,
                parameters=parameters.down_blocks[index],
                attention_head_dim=10 * index,
                num_layers=2,
                transformer_layers_per_block=transformer_layers_per_block,
                add_downsample=add_downsample,
            )
            add_downsample = False
        res = ()
        for k in output_states:
            res = res + (ttnn.to_memory_config(k, memory_config=ttnn.DRAM_MEMORY_CONFIG),)
        down_block_res_samples += res
        output_states = ()

    sample = ResnetBlock2D(config, sample, emb, parameters=parameters.mid_block.resnets[0], device=device)
    hidden_states = sd_transformer_2d(
        sample,
        encoder_hidden_states,
        num_layers=10,
        attention_head_dim=20,
        device=device,
        parameters=parameters.mid_block.attentions[0],
        config=config,
    )

    sample = ResnetBlock2D(config, hidden_states, emb, device=device, parameters=parameters.mid_block.resnets[1])

    for i in range(3):
        res_samples = down_block_res_samples[-3:]
        down_block_res_samples = down_block_res_samples[:-3]
        print("cross attention upblock index :", i)
        if i < 2:
            sample = sd_crossattnupblock2d(
                device,
                sample,
                res_samples,
                emb,
                encoder_hidden_states,
                parameters.up_blocks[i],
                config,
                num_layers=10,
                attention_head_dim=20,
            )
        else:
            sample = up_block_2d(
                device,
                parameters.up_blocks[i],
                config,
                sample,
                emb,
                res_samples,
            )

    N = sample.shape[0]
    C = sample.shape[1]
    H = sample.shape[2]
    W = sample.shape[3]

    grid_size = ttnn.CoreGrid(y=4, x=8)

    sample = ttnn.to_torch(sample).to(torch.float)
    torch_weight = ttnn.to_torch(parameters.conv_norm_out.weight).to(torch.float)
    torch_bias = ttnn.to_torch(parameters.conv_norm_out.bias).to(torch.float)
    sample = (
        torch.nn.functional.group_norm(sample, 32, weight=torch_weight, bias=torch_bias)
        .permute(0, 2, 3, 1)
        .view(N, 1, W * H, C)
    )
    sample = ttnn.from_torch(sample, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    """
    OOM
    sample = ttnn.permute(sample, (0, 2, 3, 1))
    sample = ttnn.reshape(sample, (N, 1, W * H, C))
    sample = ttnn.to_layout(sample, layout=ttnn.ROW_MAJOR_LAYOUT)
    input_mask_tensor = get_mask_tensor(C, 32, grid_size.y, device)
    gamma_t, beta_t = get_weights(
        parameters.conv_norm_out.weight, parameters.conv_norm_out.bias, C, grid_size.y, device
    )

    # shard config

    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR, False)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    sample = ttnn.to_memory_config(sample, sharded_mem_config)
    sample = ttnn.group_norm(
        sample,
        num_groups=32,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        epsilon=1e-05,
        memory_config=sharded_mem_config,
        core_grid=grid_size,
    )
    """
    sample, memory_config = get_inputs(device, sample, grid_size)
    sample = ttnn.silu(sample, memory_config=memory_config)
    sample = ttnn.reshape(sample, (N, H, W, C))
    sample = ttnn.sharded_to_interleaved(sample)
    sample = ttnn.permute(sample, (0, 3, 1, 2))
    sample = run_conv_with_split(
        device,
        sample,
        batch_size=sample.shape[0],
        parameters=parameters.conv_out,
        kernel_size=3,
        stride=1,
        pad=1,
        weights_dtype=ttnn.bfloat8_b,
        split_factor=4,
        ttnn_weight=parameters.conv_out.weight,
        ttnn_bias=parameters.conv_out.bias,
    )

    return sample
