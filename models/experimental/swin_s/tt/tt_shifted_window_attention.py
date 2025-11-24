# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn


def roll(tensor, shifts, dims):
    if isinstance(shifts, int):
        shifts = (shifts,)
    if isinstance(dims, int):
        dims = (dims,)

    assert len(shifts) == len(dims), "shifts and dims must have the same length"
    result = tensor
    shape = result.shape

    num_dims = len(shape)
    if num_dims == 1:
        shift = shifts[0] % shape[0]
        if shift == 0:
            return result
        left_part = ttnn.slice(
            result,
            slice_start=[shape[0] - shift],
            slice_end=[shape[0]],
            slice_step=[1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        right_part = ttnn.slice(
            result, slice_start=[0], slice_end=[shape[0] - shift], slice_step=[1], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        result = ttnn.concat([left_part, right_part], 0)
        return result

    for shift, dim in zip(shifts, dims):
        shift %= shape[dim]
        if shift == 0:
            continue
        start_left, end_left = [0] * (num_dims), list(shape)
        start_right, end_right = [0] * (num_dims), list(shape)
        start_left[dim] = shape[dim] - shift
        start_right[dim] = 0
        end_right[dim] = shape[dim] - shift

        left_part = ttnn.slice(
            result,
            slice_start=start_left,
            slice_end=end_left,
            slice_step=[1] * (num_dims),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        right_part = ttnn.slice(
            result,
            slice_start=start_right,
            slice_end=end_right,
            slice_step=[1] * (num_dims),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        result = ttnn.concat([left_part, right_part], dim, memory_config=ttnn.L1_MEMORY_CONFIG)
    return result


class TtShiftedWindowAttention(nn.Module):
    def __init__(
        self,
        parameters,
        device,
        dim,
        window_size,
        shift_size,
        num_heads,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_mask=None,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.parameters = parameters
        self.device = device
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attn_mask = attn_mask
        self.core_grid = self.device.compute_with_storage_grid_size()

    def forward(self, input_tensor):
        relative_position_bias = self.parameters[
            "relative_position_bias"
        ]  # relative position bias is taken from torch since it won't differ from input

        B, H, W, C = input_tensor.shape
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_values = (B, H + pad_r, W + pad_b, C)
        input_tensor = ttnn.pad(input_tensor, pad_values, [0, 0, 0, 0], 0)
        _, pad_H, pad_W, _ = input_tensor.shape

        self.shift_size = self.shift_size.copy()
        # If window size is larger than feature size, there is no need to shift window
        if self.window_size[0] >= pad_H:
            self.shift_size[0] = 0
        if self.window_size[1] >= pad_W:
            self.shift_size[1] = 0

        # cyclic shift
        if sum(self.shift_size) > 0:
            input_tensor = roll(input_tensor, (-self.shift_size[0], -self.shift_size[1]), [1, 2])

        # partition windows
        num_windows = (pad_H // self.window_size[0]) * (pad_W // self.window_size[1])

        input_tensor = ttnn.reshape(
            input_tensor,
            (
                (
                    B,
                    pad_H // self.window_size[0],
                    self.window_size[0],
                    pad_W // self.window_size[1],
                    self.window_size[1],
                    C,
                )
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        input_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2, 4, 5), memory_config=ttnn.L1_MEMORY_CONFIG)

        input_tensor = ttnn.reshape(
            input_tensor,
            (B * num_windows, self.window_size[0] * self.window_size[1], C),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        qkv_weight = self.parameters.qkv.weight
        qkv_bias = self.parameters.qkv.bias

        input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        if input_tensor.shape[0] == 361:
            shape = ttnn.pad_to_tile_shape([1, 384, 49, 96])
            mem_config = ttnn.create_sharded_memory_config_(
                shape=shape,
                core_grid=ttnn.CoreGrid(y=self.core_grid.y, x=self.core_grid.x),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            )

            input_tensor = ttnn.unsqueeze(input_tensor, dim=0)
            input_tensor = ttnn.interleaved_to_sharded(input_tensor, mem_config)
        elif input_tensor.shape[0] == 100:
            shape = ttnn.pad_to_tile_shape([1, 128, 49, 192])
            mem_config = ttnn.create_sharded_memory_config_(
                shape=shape,
                core_grid=ttnn.CoreGrid(y=self.core_grid.y, x=self.core_grid.x),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            )
            input_tensor = ttnn.unsqueeze(input_tensor, dim=0)
            input_tensor = ttnn.interleaved_to_sharded(input_tensor, mem_config)
        elif input_tensor.shape[0] == 25:
            shape = ttnn.pad_to_tile_shape([1, 32, 49, 384])
            mem_config = ttnn.create_sharded_memory_config_(
                shape=shape,
                core_grid=ttnn.CoreGrid(y=self.core_grid.y, x=self.core_grid.x),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            )
        elif input_tensor.shape[0] == 9:
            shape = ttnn.pad_to_tile_shape([1, 9, 49, 768])
            mem_config = ttnn.create_sharded_memory_config_(
                shape=shape,
                core_grid=ttnn.CoreGrid(y=self.core_grid.y, x=self.core_grid.x),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            )
        qkv = ttnn.linear(
            input_tensor,
            qkv_weight,
            bias=qkv_bias,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        input_tensor = ttnn.squeeze(input_tensor, dim=0)
        qkv = ttnn.to_layout(qkv, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        qkv = ttnn.reshape(qkv, (input_tensor.shape[0], input_tensor.shape[1], 3, self.num_heads, C // self.num_heads))

        qkv = ttnn.permute(qkv, (2, 0, 3, 1, 4), memory_config=ttnn.L1_MEMORY_CONFIG)
        channel = input_tensor.shape[1]
        ttnn.deallocate(input_tensor)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        q = ttnn.squeeze(q, 0)
        k = ttnn.squeeze(k, 0)
        v = ttnn.squeeze(v, 0)
        q = q * (C // self.num_heads) ** -0.5
        k = ttnn.permute(k, (0, 1, 3, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        attn = ttnn.matmul(
            q,
            k,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        # add relative position bias
        attn = ttnn.add(attn, relative_position_bias, memory_config=ttnn.L1_MEMORY_CONFIG)

        if sum(self.shift_size) > 0:
            attn = attn + self.attn_mask
            attn = ttnn.to_layout(attn, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
            attn = ttnn.reshape(attn, (-1, self.num_heads, channel, channel))
            attn = ttnn.to_layout(attn, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        attn = ttnn.softmax(attn, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        input_tensor = ttnn.matmul(
            attn,
            v,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(v)
        ttnn.deallocate(attn)
        input_tensor = ttnn.permute(input_tensor, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)

        input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        input_tensor = ttnn.reshape(input_tensor, (input_tensor.shape[0], input_tensor.shape[1], C))

        proj_weight = self.parameters.proj.weight
        proj_bias = self.parameters.proj.bias

        input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        if input_tensor.shape[0] == 361:
            shape = ttnn.pad_to_tile_shape([1, 384, 49, 96])
            mem_config = ttnn.create_sharded_memory_config_(
                shape=shape,
                core_grid=ttnn.CoreGrid(y=self.core_grid.y, x=self.core_grid.x),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            )
            input_tensor = ttnn.unsqueeze(input_tensor, dim=0)
            input_tensor = ttnn.interleaved_to_sharded(input_tensor, mem_config)
        elif input_tensor.shape[0] == 100:
            shape = ttnn.pad_to_tile_shape([1, 128, 49, 192])
            mem_config = ttnn.create_sharded_memory_config_(
                shape=shape,
                core_grid=ttnn.CoreGrid(y=self.core_grid.y, x=self.core_grid.x),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            )
            input_tensor = ttnn.unsqueeze(input_tensor, dim=0)
            input_tensor = ttnn.interleaved_to_sharded(input_tensor, mem_config)
        elif input_tensor.shape[0] == 25:
            shape = ttnn.pad_to_tile_shape([1, 32, 49, 384])
            mem_config = ttnn.create_sharded_memory_config_(
                shape=shape,
                core_grid=ttnn.CoreGrid(y=self.core_grid.y, x=self.core_grid.x),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            )
        elif input_tensor.shape[0] == 9:
            shape = ttnn.pad_to_tile_shape([1, 9, 49, 768])
            mem_config = ttnn.create_sharded_memory_config_(
                shape=shape,
                core_grid=ttnn.CoreGrid(y=self.core_grid.y, x=self.core_grid.x),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            )
        output_tensor = ttnn.linear(
            input_tensor,
            proj_weight,
            bias=proj_bias,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        output_tensor = ttnn.squeeze(output_tensor, dim=0)
        # reverse windows
        output_tensor = ttnn.reshape(
            output_tensor,
            (
                B,
                pad_H // self.window_size[0],
                pad_W // self.window_size[1],
                self.window_size[0],
                self.window_size[1],
                C,
            ),
        )
        output_tensor = ttnn.permute(output_tensor, (0, 1, 3, 2, 4, 5), memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.reshape(output_tensor, (B, pad_H, pad_W, C))

        # reverse cyclic shift
        if sum(self.shift_size) > 0:
            output_tensor = roll(output_tensor, self.shift_size, [1, 2])

        # unpad features
        output_tensor = output_tensor[:, :H, :W, :]

        return output_tensor
