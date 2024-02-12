# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import tt_lib
import torch
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor


def apply_rotary_emb(xq, xk, bcast_freq_xq, bcast_freq_xk, device, mem_config):
    t_xq = ttnn.to_torch(ttnn.from_device(xq))
    t_xk = ttnn.to_torch(ttnn.from_device(xk))

    xq_real = torch_to_tt_tensor_rm(t_xq[..., :, :, ::2], device)
    xq_img = torch_to_tt_tensor_rm(t_xq[..., :, :, 1::2], device)

    xq = tt_lib.tensor.complex_tensor(xq_real, xq_img)

    xq_real.deallocate()
    xq_img.deallocate()

    xk_real = torch_to_tt_tensor_rm(t_xk[..., :, :, ::2], device)
    xk_img = torch_to_tt_tensor_rm(t_xk[..., :, :, 1::2], device)
    xk = tt_lib.tensor.complex_tensor(xk_real, xk_img)

    xk_real.deallocate()
    xk_img.deallocate()

    xq_out = tt_lib.tensor.complex_mul(xq, bcast_freq_xq, output_mem_config=mem_config)

    xk_out = tt_lib.tensor.complex_mul(xk, bcast_freq_xk, output_mem_config=mem_config)

    xq_out = tt_lib.tensor.concat([xq_out.real, xq_out.imag], -1, mem_config)
    xk_out = tt_lib.tensor.concat([xk_out.real, xk_out.imag], -1, mem_config)
    xq, xk = tt_to_torch_tensor(xq_out).to(torch.float32), tt_to_torch_tensor(xk_out).to(torch.float32)

    xq_out.deallocate()
    xk_out.deallocate()
    # FIXME: move this operation to on-device - should be easy.

    shapes = xq.shape
    dindex = shapes[3] // 2
    xq_out = torch.empty(xq.shape)
    # for col in range(dindex):
    #    xq_out[:,:,:,2*col] = xq[:,:,:,col]
    #    xq_out[:,:,:,2*col+1] = xq[:,:,:,col+dindex]
    xq_out[:, :, :, ::2] = xq[:, :, :, :dindex]
    xq_out[:, :, :, 1::2] = xq[:, :, :, dindex:]

    shapes = xk.shape
    dindex = shapes[3] // 2
    xk_out = torch.empty(xk.shape)
    xk_out[:, :, :, ::2] = xk[:, :, :, :dindex]
    xk_out[:, :, :, 1::2] = xk[:, :, :, dindex:]

    return xq_out, xk_out


def repeat_kv(key, values, repeats, device):
    dim = 2
    keys = ttnn.to_layout(ttnn.to_device(ttnn.from_torch(key, dtype=ttnn.bfloat16), device), layout=ttnn.TILE_LAYOUT)
    values = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(values, dtype=ttnn.bfloat16), device), layout=ttnn.TILE_LAYOUT
    )
    keys = ttnn.repeat_interleave(keys, repeats, dim)
    values = ttnn.repeat_interleave(values, repeats, dim)
    return keys, values


def attention(config, x, bcast_freq_xq, bcast_freq_xk, positions, mask, seqlen, parameters, device, mem_config):
    bsz, _, _ = x.shape
    xq = x @ parameters.wq.weight
    xk = x @ parameters.wk.weight
    xv = x @ parameters.wv.weight

    xq = ttnn.to_layout(xq, ttnn.ROW_MAJOR_LAYOUT)
    xk = ttnn.to_layout(xk, ttnn.ROW_MAJOR_LAYOUT)
    xv = ttnn.to_layout(xv, ttnn.ROW_MAJOR_LAYOUT)

    xq = xq[:, :seqlen, :]
    xk = xk[:, :seqlen, :]
    xv = xv[:, :seqlen, :]

    xq = ttnn.reshape(xq, (bsz, seqlen, config.n_heads, config.head_dim))
    xk = ttnn.reshape(xk, (bsz, seqlen, config.n_kv_heads, config.head_dim))
    xv = ttnn.reshape(xv, (bsz, seqlen, config.n_kv_heads, config.head_dim))

    xq, xk = apply_rotary_emb(xq, xk, bcast_freq_xq, bcast_freq_xk, device, mem_config)

    positions = ttnn.to_torch(ttnn.from_device(positions))

    scatter_pos = (positions[-config.sliding_window :] % config.sliding_window)[None, :, None, None]
    scatter_pos = scatter_pos.to(torch.int64)
    scatter_pos = scatter_pos.repeat(bsz, 1, config.n_kv_heads, config.head_dim)

    cache_k = tt_lib.tensor.empty(
        [config.max_batch_size, config.sliding_window, config.n_kv_heads, config.head_dim],
        layout=tt_lib.tensor.Layout.ROW_MAJOR,
        device=device,
        output_mem_config=config.out_mem_config,
    )
    cache_k = tt_to_torch_tensor(cache_k).to(torch.float32)
    cache_v = tt_lib.tensor.empty(
        [config.max_batch_size, config.sliding_window, config.n_kv_heads, config.head_dim],
        layout=tt_lib.tensor.Layout.ROW_MAJOR,
        device=device,
        output_mem_config=config.out_mem_config,
    )
    cache_v = tt_to_torch_tensor(cache_v).to(torch.float32)
    cache_k[:bsz].scatter_(dim=1, index=scatter_pos, src=xk[:, -config.sliding_window :])
    xv = ttnn.to_torch(xv).to(torch.float32)
    cache_v[:bsz].scatter_(dim=1, index=scatter_pos, src=xv[:, -config.sliding_window :])

    if positions.shape[0] > 1:
        key, value = repeat_kv(xk, xv, config.n_heads // config.n_kv_heads, device)
    else:
        curr_pos = int(positions[-1].item() + 1)
        key, value = repeat_kv(
            cache_k[:bsz, :curr_pos, ...], cache_v[:bsz, :curr_pos, ...], config.n_heads // config.n_kv_heads, device
        )

    xq = ttnn.to_layout(ttnn.to_device(ttnn.from_torch(xq, dtype=ttnn.bfloat16), device), layout=ttnn.TILE_LAYOUT)
    query = ttnn.permute(xq, (0, 2, 1, 3))

    key = ttnn.permute(ttnn.to_device(key, device), (0, 2, 3, 1))

    value = ttnn.permute(ttnn.to_device(value, device), (0, 2, 1, 3))

    scores = query @ key
    scores = scores * config.head_dim**-0.5
    scores = ttnn.to_device(scores, device)

    if mask is not None:
        scores = ttnn.permute(scores, (2, 3, 0, 1))
        mask = ttnn.to_torch(mask).unsqueeze(0).unsqueeze(0)
        mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16)
        mask = ttnn.to_layout(mask, layout=ttnn.TILE_LAYOUT)
        mask = ttnn.to_device(mask, device)
        value1 = scores.shape[-1]
        value2 = scores.shape[-2]
        value3 = scores.shape[1]
        value4 = scores.shape[0]

        mask = ttnn.pad(
            mask,
            padding=(
                (0, value4 - mask.shape[0]),
                (0, value3 - mask.shape[1]),
                (0, value2 - mask.shape[-2]),
                (0, value1 - mask.shape[-1]),
            ),
            value=0,
        )

        scores = ttnn.add(scores, mask)
        scores = ttnn.permute(scores, (2, 3, 0, 1))

    scores = ttnn.softmax(scores, dim=-1)
    output = scores @ value
    output = ttnn.permute(output, (0, 2, 1, 3))
    output = ttnn.to_torch(output)
    output = torch_to_tt_tensor_rm(output, device, put_on_device=True)
    output = tt_lib.tensor.reshape(output, 1, bsz, seqlen, -1)
    output = tt_to_torch_tensor(output)
    output = ttnn.to_device(
        ttnn.to_layout(ttnn.from_torch(output, dtype=ttnn.bfloat16), layout=ttnn.TILE_LAYOUT), device
    )
    output = output @ parameters.wo.weight
    return output
