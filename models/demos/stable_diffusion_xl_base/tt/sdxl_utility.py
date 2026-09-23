# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


def to_channel_last_ttnn(torch_tensor, dtype, device, memory_config, layout):
    torch_tensor = torch.permute(torch_tensor, (0, 2, 3, 1))
    ttnn_tensor = ttnn.from_torch(torch_tensor, dtype, device=device, memory_config=memory_config, layout=layout)
    return ttnn_tensor


def from_channel_last_ttnn(ttnn_tensor, output_shape):
    torch_tensor = ttnn.to_torch(ttnn_tensor)
    torch_tensor = torch_tensor.reshape(output_shape)
    torch_tensor = torch.permute(torch_tensor, (0, 3, 1, 2))
    return torch_tensor


def prepare_gn_mask(device, C, G, num_cores):
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, G, num_cores, ttnn.DataType.BFLOAT8_B)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)
    return input_mask_tensor


def prepare_gn_mask_negative_mask(device, C, G, num_cores):
    input_mask_tensor = ttnn.create_group_norm_input_negative_mask(C, G, num_cores, ttnn.DataType.BFLOAT8_B)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)
    return input_mask_tensor


def prepare_gn_beta_gamma(device, weights, bias, num_cores):
    gamma = ttnn.create_group_norm_weight_bias_rm(weights, weights.shape[0], num_cores)
    beta = ttnn.create_group_norm_weight_bias_rm(bias, bias.shape[0], num_cores)
    tt_gamma = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_bias = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return tt_gamma, tt_bias


def prepare_linear_params(device, weights, bias, dtype):
    tt_weights = ttnn.from_torch(weights.movedim(-1, -2), dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = ttnn.from_torch(bias, dtype, device=device, layout=ttnn.TILE_LAYOUT) if bias is not None else None
    return tt_weights, tt_bias


def prepare_conv_params(
    weights,
    bias,
    dtype,
):
    dtype = ttnn.float32 if dtype == ttnn.bfloat8_b else dtype
    tt_weights = ttnn.from_torch(weights, dtype)
    tt_bias = ttnn.from_torch(bias, dtype) if bias is not None else None

    conv_params = {
        "input_channels": tt_weights.shape[1],
        "output_channels": tt_weights.shape[0],
        "kernel_size": (tt_weights.shape[2], tt_weights.shape[3]),
    }

    return tt_weights, tt_bias, conv_params


def _has_usable_divisor(n):
    """A per-core tile-row count conv2d can block: small, or with a divisor in [3, 26] (not prime like 47)."""
    return n <= 4 or any(n % d == 0 for d in range(3, 27))


def generated_gn_grid(HW, C, cols_hw=11, rows_c=10):
    """The full-grid UNet block shard: HW across the grid columns, C down the rows, COL_MAJOR.

    Returns (cols, rows, [shard_h, shard_w]). Columns start at 11 and drop while the padding exceeds one core's
    share (conv2d's rule) or the per-core tile-row count has no usable act-block divisor: 16384 rows -> 10 x 52
    tiles (11 would give 47, prime), 4096 -> 11 x 12, 1024 -> 11 x 3.
    """
    assert C % (rows_c * 32) == 0, f"C={C} is not a multiple of {rows_c * 32}"
    tiles = -(-HW // 32)
    cols = cols_hw
    while cols > 1:
        per_core = -(-tiles // cols)
        if per_core * cols - tiles < per_core and _has_usable_divisor(per_core):
            break
        cols -= 1
    per_core = -(-tiles // cols)
    return cols, rows_c, [per_core * 32, C // rows_c]


def generated_gn_sharded_memory_config(shape):
    *lead, C = [int(v) for v in shape]  # (N, 1, HW, C) or the (N, H, W, C) image form: rows = every leading dim
    HW = 1
    for v in lead:
        HW *= v
    gx, gy, shard = generated_gn_grid(HW, C)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    spec = ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.COL_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)


def prepare_generated_gn_beta_gamma(device, weights, bias):
    """gamma / beta as (1, 1, 1, C) bf16 ROW_MAJOR DRAM tensors (the generated op's affine contract)."""
    C = weights.shape[0]
    mk = lambda t: ttnn.from_torch(
        t.reshape(1, 1, 1, C).to(torch.float32),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return mk(weights), mk(bias)


def run_group_norm(
    hidden_states,
    groupnorm_config,
    groupnorm_memory_config,
    input_mask,
    negative_mask,
    gamma,
    beta,
    num_groups,
    eps,
    in_place="if_copied",
    activation=None,
    placement="sharded",
):
    """Apply GroupNorm through whichever implementation the model config selected.

    Generated path (ttnn.operations.groupnorm_sc_N_1_HW_C): placement "sharded" moves the activation onto the
    full-grid block shard (layout kept as is) if it is not there already; "dram" streams it DRAM interleaved in and
    out (VAE). in_place: True / False / "if_copied" (in place only when the tensor is our private resharded copy, so a
    caller that still needs the un-normalized input keeps it). activation "silu" is fused when the config sets
    fuse_silu; otherwise the caller applies it (see fused_silu()).
    Reference path: the caller has already placed the tensor for ttnn.group_norm.
    """
    if groupnorm_config.get("generated"):
        from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

        if activation == "silu" and not groupnorm_config.get("fuse_silu"):
            activation = None
        if placement == "dram":
            if hidden_states.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
            in_place = False
        else:
            mem_cfg = generated_gn_sharded_memory_config(hidden_states.shape)
            copied = hidden_states.memory_config() != mem_cfg
            if copied:
                hidden_states = ttnn.to_memory_config(hidden_states, mem_cfg)
            if in_place == "if_copied":
                in_place = copied
        return groupnorm_sc_N_1_HW_C(
            hidden_states, num_groups, gamma=gamma, beta=beta, eps=eps, in_place=bool(in_place), activation=activation
        )
    return ttnn.group_norm(
        hidden_states,
        num_groups=num_groups,
        input_mask=input_mask,
        negative_mask=negative_mask,
        weight=gamma,
        bias=beta,
        epsilon=eps,
        memory_config=hidden_states.memory_config(),
        **groupnorm_config,
    )


def fused_silu(groupnorm_config):
    """True when run_group_norm(..., activation="silu") already applied the SiLU (caller must skip its own)."""
    return bool(groupnorm_config.get("generated") and groupnorm_config.get("fuse_silu"))
