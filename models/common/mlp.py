# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import ttnn
from models.common.modules import LightweightModule, WeightSetting
from functools import lru_cache


class MLP(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        state_dict_prefix,
        weight_cache_path=None,
        activation=ttnn.UnaryOpType.SILU,
        w1w3_dtype=ttnn.bfloat8_b,  # bfloat4_b is often fine for decode
        fidelity=ttnn.MathFidelity.LoFi,  # consider HiFi2 if PCC is low
        output_dtype=ttnn.bfloat16,  # consider bfloat8_b if going into collective op
        max_rows=1024,
        per_core_k_at_max_rows="estimate",
        fp32_dest_acc_en=False,
    ):
        super().__init__(device)
        self.activation = activation
        self.output_dtype = output_dtype
        self.max_rows = max_rows

        if per_core_k_at_max_rows == "estimate":
            self.per_core_k_at_max_rows = {ttnn.bfloat4_b: 4, ttnn.bfloat8_b: 2, ttnn.bfloat16: 1}[w1w3_dtype]
        else:
            self.per_core_k_at_max_rows = per_core_k_at_max_rows

        self.weight_settings = {
            "w1": WeightSetting(
                state_dict_key=f"{state_dict_prefix}.w1.weight",
                dtype=w1w3_dtype,
                conversion_fn=lambda x: x.transpose(-2, -1),
            ),
            "w2": WeightSetting(
                state_dict_key=f"{state_dict_prefix}.w2.weight",
                dtype=ttnn.bfloat8_b,
                conversion_fn=lambda x: x.transpose(-2, -1),
            ),
            "w3": WeightSetting(
                state_dict_key=f"{state_dict_prefix}.w3.weight",
                dtype=w1w3_dtype,
                conversion_fn=lambda x: x.transpose(-2, -1),
            ),
        }

        self.load_weights(state_dict, weight_cache_path)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=True,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]

        # Matmul configs are per sequence length, relatively small structs so should be ok
        if self.max_rows and seq_len > self.max_rows:
            x = ttnn.reshape(x, [1, seq_len // self.max_rows, self.max_rows, -1])
            num_rows = self.max_rows
            pc = lambda w, act: matmul_2d_config(
                num_rows,
                w.shape[-2],
                w.shape[-1],
                act=act,
                is_fp32_accumulate=self.compute_kernel_config.fp32_dest_acc_en,
                override_per_core_k=self.per_core_k_at_max_rows,
                fuse_batch=False,
            )
        else:
            num_rows = seq_len
            pc = lambda w, act: matmul_1d_config(
                num_rows,
                w.shape[-2],
                w.shape[-1],
                act=act,
                is_fp32_accumulate=self.compute_kernel_config.fp32_dest_acc_en,
            )

        pc1 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=4,  # 32, #16,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=56,  # N / TILE_WIDTH / Grid_Size
            transpose_mcast=False,
            fused_activation=ttnn.UnaryOpType.SILU,
            fuse_batch=False,
        )

        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
            program_config=pc(self.w1, self.activation),
        )

        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
            program_config=pc(self.w3, None),
        )
        # x.deallocate(True)
        w2_in = ttnn.multiply(w1_out, w3_out)
        w3_out.deallocate(True)
        w1_out.deallocate(True)

        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            dtype=self.output_dtype,
            compute_kernel_config=self.compute_kernel_config,
            program_config=pc(self.w2, None),
        )
        w2_in.deallocate(True)

        if self.max_rows and seq_len > self.max_rows:
            w2_out = ttnn.reshape(w2_out, [1, 1, seq_len, -1])

        return w2_out


@lru_cache(maxsize=None)
def matmul_1d_config(
    m,
    k,
    n,
    grid=ttnn.CoreGrid(x=8, y=8),
    act=None,
    is_fp32_accumulate=False,
    override_per_core_k=None,
    override_subblock_w=None,
    override_subblock_h=None,
):
    tile_width = 32
    tile_height = 32

    if n // tile_width // grid.num_cores < 1:  # use fewer of cores if we have more tiles (N) than cores
        # assert (n // tile_width) % grid.x == 0
        grid_y = n // tile_width // grid.x
        grid = ttnn.CoreGrid(x=grid.x, y=grid_y)

    per_core_m = m // tile_height
    per_core_k = math.ceil(k / tile_width / grid.num_cores)
    per_core_n = math.ceil(n / tile_width / grid.num_cores)

    if is_fp32_accumulate:
        max_subblock_w_h = 4
    else:
        max_subblock_w_h = 8

    # find the largest value between 1 and 8 that is a factor of per_core_n
    # e.g. if per_core_n is 14, then out_subblock_w = 7
    out_subblock_w = max([i for i in range(1, max_subblock_w_h + 1) if per_core_n % i == 0])

    # find the largest value that is a factor of per_core_m such that
    # out_subblock_w * out_subblock_h <= 8
    out_subblock_h = max(
        [i for i in range(1, max_subblock_w_h + 1) if per_core_m % i == 0 and i * out_subblock_w <= max_subblock_w_h]
    )

    if override_per_core_k is not None:
        per_core_k = override_per_core_k

    if override_subblock_w is not None:
        out_subblock_w = override_subblock_w

    if override_subblock_h is not None:
        out_subblock_h = override_subblock_h

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=per_core_k,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=act,
        mcast_in0=True,
    )


@lru_cache(maxsize=None)
def matmul_2d_config(
    m,
    k,
    n,
    grid=ttnn.CoreGrid(x=8, y=8),
    act=None,
    is_fp32_accumulate=False,
    transpose_mcast=False,
    override_per_core_k=None,
    override_subblock_w=None,
    override_subblock_h=None,
    fuse_batch=True,
):
    tile_width = 32
    tile_height = 32

    if transpose_mcast:
        grid_x = grid.y
        grid_y = grid.x
    else:
        grid_x = grid.x
        grid_y = grid.y

    assert m % (tile_height * grid_y) == 0, f"m: {m} // 32 not divisible by grid.y: {grid_y}"
    # assert(k % (tile_height * grid_x) == 0), f"k: {k} // 32 not divisible by grid.x: {grid_x}"
    # assert(n % (tile_height * grid_x) == 0), f"n: {n} // 32 not divisible by grid.x: {grid_x}"

    per_core_m = m // tile_height // grid_y
    # per_core_k = k // tile_width // grid_x
    # per_core_n = n // tile_width // grid_x
    per_core_k = math.ceil(k / tile_width / grid_x)
    per_core_n = math.ceil(n / tile_width / grid_x)

    if is_fp32_accumulate:
        max_subblock_w_h = 4
    else:
        max_subblock_w_h = 8

    # find the largest value between 1 and 8 that is a factor of per_core_n
    # e.g. if per_core_n is 14, then out_subblock_w = 7
    out_subblock_w = max([i for i in range(1, max_subblock_w_h + 1) if per_core_n % i == 0])

    # find the largest value that is a factor of per_core_m such that
    # out_subblock_w * out_subblock_h <= 8
    out_subblock_h = max(
        [i for i in range(1, max_subblock_w_h + 1) if per_core_m % i == 0 and i * out_subblock_w <= max_subblock_w_h]
    )

    if per_core_m * per_core_n >= 512:
        max_per_core_k = 1
    elif per_core_m * per_core_n >= 128:
        max_per_core_k = 8
    else:
        max_per_core_k = 16

    if override_per_core_k is not None:
        per_core_k = override_per_core_k
    else:
        per_core_k = min(per_core_k, max_per_core_k)

    if override_subblock_w is not None:
        out_subblock_w = override_subblock_w

    if override_subblock_h is not None:
        out_subblock_h = override_subblock_h

    # print(
    #     f"per_core_m: {per_core_m}, per_core_k: {per_core_k}, per_core_n: {per_core_n}, out_subblock_h: {out_subblock_h}, out_subblock_w: {out_subblock_w}"
    # )

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=per_core_k,  # how much inner dim you take each time
        out_subblock_h=out_subblock_h,  # Must be divisible by per_core_M
        out_subblock_w=out_subblock_w,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4 for is_fp32_accumulate otherwise <= 8
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=transpose_mcast,
        fused_activation=act,
        fuse_batch=fuse_batch,
    )
