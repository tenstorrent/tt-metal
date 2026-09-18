# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

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


# ---------------------------------------------------------------------------------------------
# Generated GroupNorm (ttnn.operations.groupnorm_sc_N_1_HW_C, a Python generic_op) on the full grid.
# The reference ttnn.group_norm needs each core's channel range to be a whole number of groups, which
# pins SDXL to 8 columns (40 / 80 / ... channels per core). The generated op aggregates groups through a
# membership matmul, so 10 columns (32-aligned shard widths for every SDXL C) are legal, and its ROW_MAJOR
# direct view / TILE zero-copy paths are exact for 32-aligned shard heights. 10 columns x {10, 8} rows is
# also the geometry conv2d's own parallel-config heuristic picks, so the conv keeps the GN's shard.
# ---------------------------------------------------------------------------------------------
GENERATED_GN_COLS = 10


def _has_usable_divisor(n):
    """A per-core tile-row count conv2d can block: small, or with a divisor in [3, 26] (not prime like 47)."""
    return n <= 4 or any(n % d == 0 for d in range(3, 27))


def generated_gn_grid_transposed(HW, C, cols_hw=11, rows_c=10):
    """Transposed ("full transposed" UNet) block shard: HW across the grid columns (11), C down the rows (10),
    COL_MAJOR. Returns (cols, rows, [shard_h, shard_w]). Columns start at 11 and drop while the padding rule
    (padding < one core's share, conv2d's rule) fails or the per-core tile-row count has no usable divisor:
    16384 rows -> 512 tiles -> 47 per core over 11 (prime) -> 10 columns x 52 tiles; 4096 -> 12 x 11;
    1024 -> 3 x 11 (the transformer blocks' [96, 128] / [384, 64] shards)."""
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


def generated_gn_grid(HW, C, cols=GENERATED_GN_COLS, max_rows=10, allow_prime_rows=False):
    """(cols, rows, [shard_h, shard_w]) for a block shard of an (N, 1, HW, C) activation.

    rows = the largest count <= max_rows for which the per-core tile-row count leaves less padding than one
    core's share (conv2d's `find_closest_largest_divisor_with_num_padding` rule), shard_h a multiple of 32.
    """
    assert C % (cols * 32) == 0, f"C={C} is not a multiple of {cols * 32}"
    tiles = -(-HW // 32)
    # conv2d's act_block_h picks the largest divisor of the per-core tile-row count that is <= the config's
    # override; 128 tile rows over 10 grid rows leaves 13 per core (prime -> act_block_h collapses to 1 tile,
    # measured 1.6x slower convs). 8 rows give 16 tiles per core for HW=4096 and 4 for HW=1024.
    # allow_prime_rows: the resnet convs that consume the shard carry explicit 13-tile act blocks, so 10 rows are
    # fine there; the attention path keeps 8 rows because its matmul program configs assume [512, 64] on 10x8.
    if tiles <= 128 and not allow_prime_rows:
        max_rows = min(max_rows, 8)
    if tiles <= 32:
        # HW=1024 (32 tile rows): experiment knob — fewer rows = more rows per core for the weight-bound convs
        max_rows = min(max_rows, int(os.environ.get("SDXL_GN_ROWS_1024", "8")))
    rows = max_rows
    while rows > 1:
        per_core = -(-tiles // rows)
        if per_core * rows - tiles < per_core:
            break
        rows -= 1
    per_core = -(-tiles // rows)
    return cols, rows, [per_core * 32, C // cols]


def generated_gn_sharded_memory_config(
    shape, cols=GENERATED_GN_COLS, max_rows=10, allow_prime_rows=False, transposed=False
):
    *lead, C = [int(v) for v in shape]  # (N, 1, HW, C) or the (N, H, W, C) image form: rows = every leading dim
    N, HW = 1, 1
    for v in lead:
        HW *= v
    if transposed:
        gx, gy, shard = generated_gn_grid_transposed(N * HW, C)
        orientation = ttnn.ShardOrientation.COL_MAJOR
    else:
        gx, gy, shard = generated_gn_grid(N * HW, C, cols, max_rows, allow_prime_rows)
        orientation = ttnn.ShardOrientation.ROW_MAJOR
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    spec = ttnn.ShardSpec(grid, shard, orientation)
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
    allow_prime_rows=False,
    placement="sharded",
    transposed=False,
):
    """Apply GroupNorm through whichever implementation the model config selected.

    Generated path: shard (or reshard) the activation onto the full-grid block shard (layout kept as is) and
    run the op in place. Reference path: the caller has already sharded / DRAM-placed the tensor as before.
    Returns the normalized tensor (sharded like its input for the generated path).
    in_place: True / False / "if_copied" (generated path only; see below).
    activation: None or "silu" — generated path fuses it when the config sets fuse_silu (see fused_silu()).
    placement: "sharded" (L1 block shard on the full grid) or "dram" (DRAM interleaved in and out, VAE).
    transposed: COL_MAJOR 11x10 shard, HW across the columns and C down the rows (generated_gn_grid_transposed).
    """
    dump_dir = os.environ.get(
        "SDXL_GN_DUMP"
    )  # investigation aid: save (input, output, gamma, beta) of the first 64 GN calls
    if dump_dir:
        _gn_dump_state["n"] = _gn_dump_state.get("n", 0) + 1
        if _gn_dump_state["n"] <= 64:
            x_host = ttnn.to_torch(hidden_states)
            out = _run_group_norm_impl(
                hidden_states,
                groupnorm_config,
                groupnorm_memory_config,
                input_mask,
                negative_mask,
                gamma,
                beta,
                num_groups,
                eps,
                in_place,
                activation,
                allow_prime_rows,
                placement,
                transposed,
            )
            torch.save(
                {
                    "x": x_host,
                    "y": ttnn.to_torch(out),
                    "gamma": ttnn.to_torch(gamma),
                    "beta": ttnn.to_torch(beta),
                    "groups": num_groups,
                    "eps": eps,
                    "generated": bool(groupnorm_config.get("generated")),
                },
                f"{dump_dir}/gn_{_gn_dump_state['n']:03d}.pt",
            )
            return out
    return _run_group_norm_impl(
        hidden_states,
        groupnorm_config,
        groupnorm_memory_config,
        input_mask,
        negative_mask,
        gamma,
        beta,
        num_groups,
        eps,
        in_place,
        activation,
        allow_prime_rows,
        placement,
        transposed,
    )


_gn_dump_state = {}


def _run_group_norm_impl(
    hidden_states,
    groupnorm_config,
    groupnorm_memory_config,
    input_mask,
    negative_mask,
    gamma,
    beta,
    num_groups,
    eps,
    in_place,
    activation,
    allow_prime_rows,
    placement,
    transposed,
):
    if groupnorm_config.get("generated"):
        from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

        if placement == "dram":
            # VAE-style DRAM path: the activation does not fit L1; the op streams it from DRAM interleaved and
            # writes a fresh DRAM interleaved output (no I2S / S2I around the norm).
            if hidden_states.is_sharded():
                hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
            if activation == "silu" and not groupnorm_config.get("fuse_silu"):
                activation = None
            return groupnorm_sc_N_1_HW_C(
                hidden_states, num_groups, gamma=gamma, beta=beta, eps=eps, in_place=False, activation=activation
            )
        mem_cfg = generated_gn_sharded_memory_config(
            hidden_states.shape, allow_prime_rows=allow_prime_rows, transposed=transposed
        )
        copied = hidden_states.memory_config() != mem_cfg
        if copied:
            hidden_states = ttnn.to_memory_config(hidden_states, mem_cfg)
        # in_place="if_copied": normalize in place only when the tensor is our private copy; a caller that still
        # needs the un-normalized tensor (residual paths) and handed us an already-sharded input gets a fresh output.
        if in_place == "if_copied":
            in_place = copied
        if activation == "silu" and not groupnorm_config.get("fuse_silu"):
            activation = None  # caller applies its own SiLU (see fused_silu())
        out = groupnorm_sc_N_1_HW_C(
            hidden_states, num_groups, gamma=gamma, beta=beta, eps=eps, in_place=bool(in_place), activation=activation
        )
        if os.environ.get("SDXL_GN_SCALE_HACK"):  # investigation: mimic the reference GN's output shrinkage
            out = ttnn.mul_(out, float(os.environ["SDXL_GN_SCALE_HACK"]))
        return out
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
