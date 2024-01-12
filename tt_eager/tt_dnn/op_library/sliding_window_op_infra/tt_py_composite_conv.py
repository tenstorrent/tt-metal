# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_op import TTPyOp
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_untilize_with_halo import TTPyUntilizeWithHalo
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.untilize_with_halo_config_generation_and_validation import (
    trace_conv_to_generate_data_top_left_indices_and_pad_metadata,
    decompose_conv_into_shards_and_generate_tensor_metadata,
)
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.sliding_window_op_config_generation_and_validation import (
    generate_sliding_window_op_sharded_input_top_left_indices,
)
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.sliding_window_op_utils import (
    SlidingWindowOpParams,
    SlidingWindowOpParamsWithParallelConfig,
    get_hash_from_sliding_window_op_params,
)
from tt_lib.utils import _nearest_32, _nearest_y

import tt_lib as ttl
import torch
import math
import warnings


def find_closest_largest_divisor(num: int, start_divisor: int):
    divisor = start_divisor
    while num % divisor != 0:
        divisor = divisor - 1
    return divisor


def determine_largest_subblock_size(block_height, block_width):
    subblocks = [
        (2, 4),
        (4, 2),
        (1, 8),
        (8, 1),
        (1, 7),
        (7, 1),
        (2, 3),
        (3, 2),
        (1, 6),
        (6, 1),
        (1, 5),
        (5, 1),
        (2, 2),
        (1, 4),
        (4, 1),
        (1, 3),
        (3, 1),
        (1, 2),
        (2, 1),
        (1, 1),
    ]
    for subblock_height, subblock_width in subblocks:
        if block_height % subblock_height == 0 and block_width % subblock_width == 0:
            if subblock_width != block_width and subblock_height != 1:
                continue
            break
    return subblock_height, subblock_width


def compute_conv_output_height_width(input_height, input_width, sliding_window_op_params):
    stride_h = sliding_window_op_params.stride_h
    stride_w = sliding_window_op_params.stride_w
    pad_h = sliding_window_op_params.pad_h
    pad_w = sliding_window_op_params.pad_w
    filter_h = sliding_window_op_params.window_h
    filter_w = sliding_window_op_params.window_w
    output_height = ((int)((input_height - filter_h + 2 * pad_h) / stride_h)) + 1
    output_width = ((int)((input_width - filter_w + 2 * pad_w) / stride_w)) + 1
    return output_height, output_width


def find_closest_largest_divisor_with_num_padding(num: int, start_divisor: int):
    divisor = start_divisor
    padded_num = _nearest_y(num, divisor)
    while (padded_num - num) >= (int)(padded_num / divisor):
        divisor = divisor - 1
        padded_num = _nearest_y(num, divisor)
    return divisor


def determine_parallel_config(
    is_1d_systolic,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    sliding_window_op_params,
    config_override={},
):
    output_height, output_width = compute_conv_output_height_width(input_height, input_width, sliding_window_op_params)
    conv_out_2d_matrix_height = batch_size * output_height * output_width
    # pad height to 32
    conv_out_2d_matrix_height = _nearest_32(conv_out_2d_matrix_height)
    conv_out_2d_matrix_height_ntiles = (int)(conv_out_2d_matrix_height / 32)
    conv_out_2d_matrix_width_ntiles = (int)(_nearest_32(output_channels) / 32)
    # max grid size of grayskull is (12,9). TODO: pass this as an argument
    max_grid_size = {"x": 12, "y": 9}
    if is_1d_systolic:
        max_num_cores = max_grid_size["x"] * max_grid_size["y"]
        actual_num_cores = find_closest_largest_divisor(conv_out_2d_matrix_height_ntiles, max_num_cores)
        per_core_out_matrix_height_ntiles = (int)(conv_out_2d_matrix_height_ntiles / actual_num_cores)
        per_core_out_matrix_width_ntiles = conv_out_2d_matrix_width_ntiles
        grid_size = [
            max_grid_size["x"],
            max_grid_size["y"],
        ]  # for 1d systolic array, we don't need to provide actual grid size because tensor sharding deterimines that automatically
        num_cores_nhw = actual_num_cores
        grid_size = [
            max_grid_size["x"],
            math.ceil(num_cores_nhw / max_grid_size["x"]),
        ]  # for 1d systolic array, grid size is the tightest bound of num_cores_nhw as a rectangle (x,y)
    else:
        actual_num_cores_x = find_closest_largest_divisor_with_num_padding(
            conv_out_2d_matrix_height_ntiles, max_grid_size["x"]
        )
        actual_num_cores_y = find_closest_largest_divisor(conv_out_2d_matrix_width_ntiles, max_grid_size["y"])
        per_core_out_matrix_height_ntiles = math.ceil(conv_out_2d_matrix_height_ntiles / actual_num_cores_x)
        per_core_out_matrix_width_ntiles = (int)(conv_out_2d_matrix_width_ntiles / actual_num_cores_y)
        grid_size = [actual_num_cores_x, actual_num_cores_y]
        num_cores_nhw = actual_num_cores_x

    if "grid_size" in config_override:
        if config_override["grid_size"][0] != grid_size[0]:
            grid_size_x_override = config_override["grid_size"][0]
            warnings.warn(
                f"Overriding config: grid_size.x from {grid_size[0]} to user provided config={grid_size_x_override}"
            )
            grid_size[0] = grid_size_x_override
        if config_override["grid_size"][1] != grid_size[1]:
            grid_size_y_override = grid_size[1]
            warnings.warn(
                f"Overriding config: grid_size.y from {grid_size[1]} to user provided config={grid_size_y_override}"
            )
            grid_size[1] = grid_size_y_override
    if "num_cores_nhw" in config_override:
        num_cores_nhw_override = config_override["num_cores_nhw"]
        if num_cores_nhw_override != num_cores_nhw:
            warnings.warn(
                f"Overriding config: num_cores_nhw from {num_cores_nhw} to user provided config={num_cores_nhw_override}"
            )
            num_cores_nhw = num_cores_nhw_override
    if is_1d_systolic:
        assert (
            num_cores_nhw <= grid_size[0] * grid_size[1]
        ), "Error: For 1d systolic conv, num_cores_nhw must be <= grid size"
    else:
        assert (
            num_cores_nhw == grid_size[0]
        ), "Error: For 2d systolic conv, num_cores_nhw must be == # of cols in grid size"
    if "per_core_out_matrix_height" in config_override:
        per_core_out_matrix_height_override = config_override["per_core_out_matrix_height"]
        assert (
            per_core_out_matrix_height_override % 32 == 0
        ), "per_core_out_matrix_height must be divisible by 32 (tile height)"
        if (per_core_out_matrix_height_override // 32) != per_core_out_matrix_height_ntiles:
            warnings.warn(
                f"Overriding config: per_core_out_matrix_height from {per_core_out_matrix_height_ntiles * 32} to user provided config={per_core_out_matrix_height_override}"
            )
            per_core_out_matrix_height_ntiles = per_core_out_matrix_height_override // 32

    if "per_core_weight_matrix_width" in config_override:
        per_core_out_matrix_width_override = config_override["per_core_weight_matrix_width"]
        assert (
            per_core_out_matrix_width_override % 32 == 0
        ), "per_core_weight_matrix_width must be divisible by 32 (tile width)"
        if (per_core_out_matrix_width_override // 32) != per_core_out_matrix_width_ntiles:
            warnings.warn(
                f"Overriding config: per_core_weight_matrix_width from {per_core_out_matrix_width_ntiles * 32} to user provided config={per_core_out_matrix_width_override}"
            )
            per_core_out_matrix_width_ntiles = per_core_out_matrix_width_override // 32

    conv_parallelization_config = ttl.tensor.OptimizedConvParallelizationConfig(
        grid_size=grid_size,
        per_core_out_matrix_height_ntiles=per_core_out_matrix_height_ntiles,
        per_core_weight_matrix_width_ntiles=per_core_out_matrix_width_ntiles,
    )
    return conv_parallelization_config, num_cores_nhw


def determine_per_core_block_config(
    is_1d_systolic,
    grid_size,
    per_core_out_matrix_height_ntiles,
    per_core_out_matrix_width_ntiles,
    input_channels,
    sliding_window_op_params,
    config_override={},
):
    act_block_h_override = 0
    if "act_block_h" in config_override:
        act_block_h_override = config_override["act_block_h"]
        assert act_block_h_override % 32 == 0, "act_block_h must be divisible by 32 (tile height)"
    act_block_h_ntiles_override = act_block_h_override // 32
    act_block_h_ntiles = (
        act_block_h_ntiles_override if act_block_h_ntiles_override > 0 else per_core_out_matrix_height_ntiles
    )
    act_block_w_ntiles = (int)(
        ((input_channels * sliding_window_op_params.window_w) if is_1d_systolic else input_channels) / 32
    )
    if is_1d_systolic:
        act_c_num_blocks = 1
    else:
        act_c_num_blocks = grid_size.y
        assert (
            input_channels % act_c_num_blocks == 0
        ), "Cannot parallelize conv as a 2d systolic array. Input channels must be divisible by act_c_num_blocks."
    out_block_h_ntiles = per_core_out_matrix_height_ntiles
    assert out_block_h_ntiles % act_block_h_ntiles == 0, "act_block_h must evenly divide out_block_h"
    weight_block_w_ntiles = per_core_out_matrix_width_ntiles
    out_subblock_h_ntiles, out_subblock_w_ntiles = determine_largest_subblock_size(
        act_block_h_ntiles, weight_block_w_ntiles
    )

    if "act_block_w" in config_override:
        act_block_w_override = config_override["act_block_w"]
        assert act_block_w_override % 32 == 0, "act_block_w must be divisible by 32 (tile width)"
        if (act_block_w_override // 32) != act_block_w_ntiles:
            warnings.warn(
                f"Overriding config: act_block_w from {act_block_w_ntiles * 32} to user provided config={act_block_w_override}"
            )
            act_block_w_ntiles = act_block_w_override // 32
    if "act_c_num_blocks" in config_override:
        act_c_num_blocks_override = config_override["act_c_num_blocks"]
        if config_override["act_c_num_blocks"] != act_c_num_blocks:
            warnings.warn(
                f"Overriding config: act_c_num_blocks from {act_c_num_blocks} to user provided config={act_c_num_blocks_override}"
            )
            act_c_num_blocks = act_c_num_blocks_override
    if "weight_block_w" in config_override:
        weight_block_w_override = config_override["weight_block_w"]
        assert weight_block_w_override % 32 == 0, "weight_block_w must be divisible by 32 (tile width)"
        if (weight_block_w_override // 32) != weight_block_w_ntiles:
            warnings.warn(
                f"Overriding config: weight_block_w from {weight_block_w_ntiles * 32} to user provided config={weight_block_w_override}"
            )
            weight_block_w_ntiles = weight_block_w_override // 32
    if "out_block_h" in config_override:
        out_block_h_override = config_override["out_block_h"]
        assert out_block_h_override % 32 == 0, "out_block_h must be divisible by 32 (tile height)"
        if (out_block_h_override // 32) != out_block_h_ntiles:
            warnings.warn(
                f"Overriding config: out_block_h from {out_block_h_ntiles * 32} to user provided config={out_block_h_override}"
            )
            out_block_h_ntiles = out_block_h_override // 32
    if "out_subblock_h" in config_override:
        assert (
            "out_subblock_w" in config_override
        ), "out_subblock_w must also be provided as override config if out_subblock_h is provided"
        out_subblock_h_override = config_override["out_subblock_h"]
        assert out_subblock_h_override % 32 == 0, "out_subblock_h must be divisible by 32 (tile height)"
        out_subblock_w_override = config_override["out_subblock_w"]
        assert out_subblock_w_override % 32 == 0, "out_subblock_w must be divisible by 32 (tile width)"
        if (out_subblock_h_override // 32) != out_subblock_h_ntiles:
            warnings.warn(
                f"Overriding config: out_subblock_h from {out_block_h_ntiles * 32} to user provided config={out_subblock_h_override}"
            )
        if (out_subblock_w_override // 32) != out_subblock_w_ntiles:
            warnings.warn(
                f"Overriding config: out_subblock_w from {out_subblock_w_ntiles * 32} to user provided config={out_subblock_w_override}"
            )
    if "out_subblock_w" in config_override:
        assert (
            "out_subblock_h" in config_override
        ), "out_subblock_h must also be provided as override config if out_subblock_w is provided"

    conv_blocking_config = ttl.tensor.OptimizedConvBlockConfig(
        act_block_h_ntiles=act_block_h_ntiles,
        act_block_w_ntiles=act_block_w_ntiles,
        act_c_num_blocks=act_c_num_blocks,
        weight_block_w_ntiles=weight_block_w_ntiles,
        out_block_h_ntiles=out_block_h_ntiles,
        out_subblock_h_ntiles=out_subblock_h_ntiles,
        out_subblock_w_ntiles=out_subblock_w_ntiles,
    )
    return conv_blocking_config


class TTPyCompositeConv(TTPyOp):
    config_keys = [
        "num_cores_nhw",
        "grid_size",
        "per_core_out_matrix_height",
        "per_core_weight_matrix_width",
        "act_block_h",
        "act_block_w",
        "act_c_num_blocks",
        "weight_block_w",
        "out_block_h",
        "out_block_w",
        "out_subblock_h",
        "out_subblock_w",
    ]

    def __init__(
        self,
        sliding_window_op_params: Union[SlidingWindowOpParams, SlidingWindowOpParamsWithParallelConfig],
        weight: ttl.tensor.Tensor,  # should user send TT tensor as weight tensor
        output_channels,
        input_channels,
        device,
        is_1d_systolic,
        reader_patterns_cache,
        bias: ttl.tensor.Tensor = None,
        conv_blocking_and_parallelization_config_override=None,
        fuse_relu=False,
        weights_dtype=None,
        output_dtype=None,
        math_fidelity=None,
        move_utwh_output=False,
    ):
        if len(reader_patterns_cache) == 0:
            reader_patterns_cache["conv"] = {}
            reader_patterns_cache["halo"] = {}
        else:
            assert len(reader_patterns_cache) == 2
            assert "conv" in reader_patterns_cache and "halo" in reader_patterns_cache
        for key in reader_patterns_cache:
            assert (
                key == "conv" or key == "halo"
            ), f"reader_patterns_cache should have 1 of the following keys only - conv or halo. Found key - {key}"
        if conv_blocking_and_parallelization_config_override is None:
            conv_blocking_and_parallelization_config_override = {}
        for key in conv_blocking_and_parallelization_config_override:
            assert (
                key in TTPyCompositeConv.config_keys
            ), f"Error: unsupported config key: {key}. Supported config keys are: {TTPyCompositeConv.config_keys}"
        batch_size = sliding_window_op_params.batch_size
        input_height = sliding_window_op_params.input_h
        input_width = sliding_window_op_params.input_w
        output_height, output_width = compute_conv_output_height_width(
            input_height, input_width, sliding_window_op_params
        )
        self.conv_output_shape = [batch_size, output_height, output_width, output_channels]
        self.input_tensor_shape = [batch_size, input_height, input_width, input_channels]
        self.is_1d_systolic = is_1d_systolic
        self.device = device
        # determine conv op parallelization and blocking config
        self.opt_conv_parall_conf_auto, num_cores_nhw = determine_parallel_config(
            is_1d_systolic,
            batch_size,
            output_channels,
            input_channels,
            input_height,
            input_width,
            sliding_window_op_params,
            config_override=conv_blocking_and_parallelization_config_override,
        )

        self.opt_conv_block_conf_auto = determine_per_core_block_config(
            is_1d_systolic,
            self.opt_conv_parall_conf_auto.grid_size,
            self.opt_conv_parall_conf_auto.per_core_out_matrix_height_ntiles,
            self.opt_conv_parall_conf_auto.per_core_weight_matrix_width_ntiles,
            input_channels,
            sliding_window_op_params,
            config_override=conv_blocking_and_parallelization_config_override,
        )

        if not is_1d_systolic:  # 2D conv
            output_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1
            )
        else:
            output_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1
            )

        stride_h = sliding_window_op_params.stride_h
        stride_w = sliding_window_op_params.stride_w
        pad_h = sliding_window_op_params.pad_h
        pad_w = sliding_window_op_params.pad_w
        filter_height = sliding_window_op_params.window_h
        filter_width = sliding_window_op_params.window_w

        if isinstance(sliding_window_op_params, SlidingWindowOpParams):
            # populate parallelization params in sliding_window_op_params
            sliding_window_op_params = SlidingWindowOpParamsWithParallelConfig(
                stride_h=stride_h,
                stride_w=stride_w,
                pad_h=pad_h,
                pad_w=pad_w,
                window_h=filter_height,
                window_w=filter_width,
                batch_size=batch_size,
                input_h=input_height,
                input_w=input_width,
                num_cores_h=self.opt_conv_parall_conf_auto.grid_size.y,
                num_cores_w=self.opt_conv_parall_conf_auto.grid_size.x,
                num_cores_nhw=num_cores_nhw,
            )

        self.sliding_window_op_params = sliding_window_op_params
        self.move_utwh_output = move_utwh_output

        sliding_window_op_params_hash = get_hash_from_sliding_window_op_params(sliding_window_op_params)

        # TODO: consolidate conv_params and sliding_window_op_params
        # K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
        conv_params = [
            output_channels,
            input_channels,
            filter_height,
            filter_width,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            1,
            1,
        ]
        # set_op_configs populates reader_patterns_cache["conv"][sliding_window_op_params_hash] with conv_reader_indices sharded tensor
        self.set_op_configs(
            self.device,
            sliding_window_op_params_hash,
            sliding_window_op_params,
            conv_params,
            not is_1d_systolic,
            reader_patterns_cache["conv"],
        )
        assert sliding_window_op_params_hash in reader_patterns_cache["conv"]
        conv_reader_indices = reader_patterns_cache["conv"][sliding_window_op_params_hash]

        self.set_op_weights_biases(
            weight,
            conv_params,
            self.device,
            self.opt_conv_block_conf_auto.act_block_w_ntiles,
            self.opt_conv_block_conf_auto.weight_block_w_ntiles,
            self.opt_conv_parall_conf_auto,
            self.opt_conv_block_conf_auto,
            fuse_relu,
            output_mem_config,
            output_dtype,
            math_fidelity,
            conv_reader_indices,
            bias=bias,
            weights_dtype=weights_dtype,
        )

        # create untilize with halo op
        self.tt_py_untilize_with_halo_op = TTPyUntilizeWithHalo(
            device, self.sliding_window_op_params, reader_patterns_cache["halo"]
        )

    # override abstract methods from base class TTPyOp
    def set_op_configs(
        self,
        device,
        sliding_window_op_params_hash,
        sliding_window_op_params,
        conv_params,
        conv_is_2d,
        conv_reader_patterns_cache,
    ):
        # TODO: Need way of hashing sliding_window_op_params
        if sliding_window_op_params_hash not in conv_reader_patterns_cache:
            # TODO: Need to clean up sliding_window_op_params and conv_params (they are basically the same)
            stride_h = sliding_window_op_params.stride_h
            stride_w = sliding_window_op_params.stride_w
            pad_h = sliding_window_op_params.pad_h
            pad_w = sliding_window_op_params.pad_w
            filter_h = sliding_window_op_params.window_h
            filter_w = sliding_window_op_params.window_w
            batch_size = sliding_window_op_params.batch_size
            input_h = sliding_window_op_params.input_h
            input_w = sliding_window_op_params.input_w
            # TODO: Had to add this (should this be shard grid?)
            num_cores_w = sliding_window_op_params.num_cores_w
            num_cores_h = sliding_window_op_params.num_cores_h
            num_cores_nhw = sliding_window_op_params.num_cores_nhw

            input_nchw_shape = [batch_size, 1, input_h, input_w]
            conv_input_volume = batch_size * input_h * input_w
            conv_output_h = ((int)((input_h + (2 * pad_h) - filter_h) / stride_h)) + 1
            conv_output_w = ((int)((input_w + (2 * pad_w) - filter_w) / stride_w)) + 1
            conv_output_volume = batch_size * conv_output_h * conv_output_w

            input_size_to_shard_evenly = _nearest_y(conv_input_volume, num_cores_nhw * 32)
            untilize_with_halo_input_shard_height = (int)(input_size_to_shard_evenly / num_cores_nhw)
            output_size_to_shard_evenly = _nearest_y(conv_output_volume, num_cores_nhw * 32)
            conv_output_shard_height = (int)(output_size_to_shard_evenly / num_cores_nhw)

            input_padded_width = input_w + 2 * pad_w

            # TODO: We should remove C from input_nchw_shape since none of the specs depend on it
            # TODO: Pass sliding_window_op_params instead of conv_param?
            pad_metadata, data_top_left_indices = trace_conv_to_generate_data_top_left_indices_and_pad_metadata(
                conv_params, input_nchw_shape
            )

            req_conv_input_shard_start_end, tensor_metadata = decompose_conv_into_shards_and_generate_tensor_metadata(
                data_top_left_indices,
                pad_metadata,
                input_padded_width,
                conv_output_shard_height,
                untilize_with_halo_input_shard_height,
                num_cores_nhw,
                filter_h,
                filter_w,
            )

            sliding_window_op_sharded_input_top_left_indices = (
                generate_sliding_window_op_sharded_input_top_left_indices(
                    data_top_left_indices, req_conv_input_shard_start_end
                )
            )

            # Pad indices for last core if not equal to other cores
            indices_length_per_core = len(sliding_window_op_sharded_input_top_left_indices[0])
            sliding_window_op_sharded_input_top_left_indices[-1].extend(
                [0] * (indices_length_per_core - len(sliding_window_op_sharded_input_top_left_indices[-1]))
            )

            indices_torch_dtype = torch.int16
            indices_tt_dtype = ttl.tensor.DataType.UINT16
            # For 2d convs, each core in a column share the same specs
            if conv_is_2d:
                sliding_window_op_sharded_input_top_left_indices *= num_cores_h

            # Create sharded tensor on device for conv_reader_indices
            conv_reader_indices_torch_tensor = torch.tensor(
                [[sliding_window_op_sharded_input_top_left_indices]], dtype=indices_torch_dtype
            )

            conv_reader_indices_tt_tensor = ttl.tensor.Tensor(
                conv_reader_indices_torch_tensor,
                indices_tt_dtype,
            )
            shard_grid = ttl.tensor.CoreRangeSet(
                {
                    ttl.tensor.CoreRange(
                        ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(num_cores_w - 1, num_cores_h - 1)
                    )
                }
            )
            shard_orientation = ttl.tensor.ShardOrientation.ROW_MAJOR
            shard_halo = False
            shard_spec = ttl.tensor.ShardSpec(shard_grid, [1, conv_output_shard_height], shard_orientation, shard_halo)
            mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1)
            conv_reader_indices_sharded_tensor = conv_reader_indices_tt_tensor.to(device, mem_config, shard_spec)

            conv_reader_patterns_cache[sliding_window_op_params_hash] = conv_reader_indices_sharded_tensor

    # TODO: Maybe need to have this be more general to settting up conv
    def set_op_weights_biases(
        self,
        weight: ttl.tensor.Tensor,
        conv_params,
        device,
        weight_block_h_ntiles,
        weight_block_w_ntiles,
        opt_conv_parall_conf,
        opt_conv_block_conf,
        fuse_relu,
        output_mem_config,
        output_dtype,
        math_fidelity,
        conv_reader_indices,
        weights_dtype=None,
        bias=None,
    ):
        assert len(conv_params) == 10
        K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]

        assert dilation == 1 and groups == 1

        weights_shape = [K, C, R, S]
        weights_channels_padded_shape = [_nearest_32(K), _nearest_y(C, 16), R, S]
        if weights_dtype is None:
            weights_dtype = weight.dtype()
        weights_untiled_dtype = (
            weights_dtype if weights_dtype != ttl.tensor.DataType.BFLOAT8_B else ttl.tensor.DataType.FLOAT32
        )
        assert weight.layout() == ttl.tensor.Layout.ROW_MAJOR
        assert weight.dtype() == weights_untiled_dtype
        assert weight.shape() == weights_shape
        weight_untiled = weight.pad(weights_channels_padded_shape, (0, 0, 0, 0), 0)
        # for conv op, pad the weights to block shape
        weight_tiled_ = ttl.tensor.convert_conv_weight_tensor_to_tiled_layout(
            weight_untiled,
            weight_block_h_ntiles,
            weight_block_w_ntiles,
            output_dtype=weights_dtype,
        )
        weight_on_device = weight_tiled_.to(device)
        bias_on_device = None
        if bias is not None:
            bias_shape = [1, 1, 1, K]
            assert bias.layout() == ttl.tensor.Layout.ROW_MAJOR
            assert bias.dtype() == weights_untiled_dtype
            assert bias.shape() == bias_shape

            assert K % (weight_block_w_ntiles * 32) == 0
            bias_channels_padded_shape = [1, 1, 32, _nearest_32(K)]
            bias_untiled = bias.pad(bias_channels_padded_shape, (0, 0, 0, 0), 0)
            # TODO: what api to use to convert the datatype of tensor?? Converting to pytorch for now and creating another tensor with it
            bias_untiled = bias_untiled.to_torch()
            bias_ = ttl.tensor.Tensor(bias_untiled, weights_dtype).to(ttl.tensor.Layout.TILE)
            bias_on_device = bias_.to(device)

        def conv_(activation):
            return ttl.tensor.optimized_conv(
                activation,
                weight_on_device,
                bias_on_device,
                conv_reader_indices,
                [R, S, U, V, P_H, P_W],
                K,
                False,
                True,
                fuse_relu,
                math_fidelity,
                opt_conv_parall_conf,
                opt_conv_block_conf,
                0,
                output_mem_config=activation.memory_config() if output_mem_config is None else output_mem_config,
                output_dtype=output_dtype,
                input_tensor_shape=self.input_tensor_shape,
            )
            # assert(output.storage_type() == ttl.tensor.StorageType.DEVICE)

        def composite_conv(activation):
            # assert(activation.layout() == ttl.tensor.Layout.ROW_MAJOR)
            utwh_output = self.tt_py_untilize_with_halo_op(activation)
            activation.deallocate()
            return conv_(utwh_output)

        def composite_conv_with_move_utwh_output(activation):
            # assert(activation.layout() == ttl.tensor.Layout.ROW_MAJOR)
            utwh_output = self.tt_py_untilize_with_halo_op(activation)
            activation.deallocate()
            move_output = ttl.tensor.move_sharded(utwh_output)
            utwh_output.deallocate()
            return conv_(move_output)

        if self.move_utwh_output:
            self.conv = composite_conv_with_move_utwh_output
        else:
            self.conv = composite_conv

    def __call__(self, activation):
        return self.conv(activation)

    def get_parallelization_config(self):
        return self.opt_conv_parall_conf_auto

    def get_blocking_config(self):
        return self.opt_conv_block_conf_auto

    def get_num_cores_nhw(self):
        return self.sliding_window_op_params.num_cores_nhw

    # TODO: with this api, we get incorrect output
    def copy_input_to_device_with_sharded_api(self, conv_input: ttl.tensor.Tensor):
        assert conv_input.shape() == self.input_tensor_shape
        num_cores_nhw = self.sliding_window_op_params.num_cores_nhw
        num_cores_w = self.sliding_window_op_params.num_cores_w
        num_cores_h = self.sliding_window_op_params.num_cores_h
        input_channels = self.input_tensor_shape[3]
        act_c_num_blocks = self.opt_conv_block_conf_auto.act_c_num_blocks
        assert input_channels % act_c_num_blocks == 0
        input_size_to_shard_evenly = _nearest_y(
            self.input_tensor_shape[0] * self.input_tensor_shape[1] * self.input_tensor_shape[2], num_cores_nhw * 32
        )
        untilize_with_halo_input_shard_height = (int)(input_size_to_shard_evenly / num_cores_nhw)
        conv_input = conv_input.reshape(
            1,
            1,
            self.input_tensor_shape[0] * self.input_tensor_shape[1] * self.input_tensor_shape[2],
            self.input_tensor_shape[3],
        )
        conv_input = conv_input.pad([1, 1, input_size_to_shard_evenly, self.input_tensor_shape[3]], (0, 0, 0, 0), 0.0)
        if self.input_tensor_shape[0] >= 32:
            # Convert activation RM to tile layout
            conv_input = conv_input.to(ttl.tensor.Layout.TILE)

        if self.is_1d_systolic and num_cores_nhw % num_cores_w > 0:
            assert num_cores_h * num_cores_w > num_cores_nhw
            first_range_num_cores_h = num_cores_nhw // num_cores_w
            assert num_cores_nhw % num_cores_w < num_cores_w

            shard_grid = ttl.tensor.CoreRangeSet(
                {
                    ttl.tensor.CoreRange(
                        ttl.tensor.CoreCoord(0, 0),
                        ttl.tensor.CoreCoord(num_cores_w - 1, first_range_num_cores_h - 1),
                    ),
                    ttl.tensor.CoreRange(
                        ttl.tensor.CoreCoord(0, first_range_num_cores_h),
                        ttl.tensor.CoreCoord((num_cores_nhw % num_cores_w) - 1, first_range_num_cores_h),
                    ),
                }
            )
        else:
            if self.is_1d_systolic:
                assert num_cores_nhw == num_cores_h * num_cores_w
            shard_grid = ttl.tensor.CoreRangeSet(
                {
                    ttl.tensor.CoreRange(
                        ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(num_cores_w - 1, num_cores_h - 1)
                    )
                }
            )

        shard_orientation = (
            ttl.tensor.ShardOrientation.ROW_MAJOR if self.is_1d_systolic else ttl.tensor.ShardOrientation.COL_MAJOR
        )
        shard_halo = False
        shard_shape = [
            untilize_with_halo_input_shard_height,
            input_channels if self.is_1d_systolic else (int)(input_channels / act_c_num_blocks),
        ]
        shard_spec = ttl.tensor.ShardSpec(shard_grid, shard_shape, shard_orientation, shard_halo)
        mem_config = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED
            if self.is_1d_systolic
            else ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.BufferType.L1,
        )
        conv_input_on_device = conv_input.to(self.device, mem_config, shard_spec)
        return conv_input_on_device

    def copy_input_to_device(self, conv_input: ttl.tensor.Tensor):
        interleaved_mem_config = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
        )
        assert conv_input.shape() == self.input_tensor_shape
        num_cores_nhw = self.sliding_window_op_params.num_cores_nhw
        num_cores_w = self.sliding_window_op_params.num_cores_w
        num_cores_h = self.sliding_window_op_params.num_cores_h

        input_channels = self.input_tensor_shape[3]
        act_c_num_blocks = self.opt_conv_block_conf_auto.act_c_num_blocks
        grid_size = (num_cores_w, num_cores_h)
        assert input_channels % act_c_num_blocks == 0

        # Convert activation RM to tile layout
        conv_input_on_device = conv_input.reshape(
            1,
            1,
            self.input_tensor_shape[0] * self.input_tensor_shape[1] * self.input_tensor_shape[2],
            self.input_tensor_shape[3],
        ).to(self.device, interleaved_mem_config)
        if input_channels >= 32:
            input_padded_shape = ttl.tensor.pad_to_tile_shape(conv_input_on_device.shape(), False, False, True, True)
            if conv_input.shape() != input_padded_shape:
                conv_input_on_device = ttl.tensor.format_input_tensor(
                    conv_input_on_device,
                    self.device,
                    input_padded_shape,
                    0.0,
                    ttl.tensor.Layout.TILE,
                    interleaved_mem_config,
                )
            else:
                conv_input_on_device = ttl.tensor.tilize(
                    conv_input_on_device, interleaved_mem_config, use_multicore=True
                )

        input_size_to_shard_evenly = _nearest_y(
            self.input_tensor_shape[0] * self.input_tensor_shape[1] * self.input_tensor_shape[2], num_cores_nhw * 32
        )
        untilize_with_halo_input_shard_height = (int)(input_size_to_shard_evenly / num_cores_nhw)
        # Convert interleaved to sharded
        if act_c_num_blocks > 1:  # 2D conv
            assert input_channels % act_c_num_blocks == 0
            conv_input_on_device = ttl.tensor.interleaved_to_sharded(
                conv_input_on_device,
                grid_size,
                [
                    untilize_with_halo_input_shard_height,
                    (int)(input_channels / act_c_num_blocks),
                ],  # act_block_w_datums may include reads of multiple pixels in window
                ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
                ttl.tensor.ShardOrientation.COL_MAJOR,
            )
        else:
            conv_input_on_device = ttl.tensor.interleaved_to_sharded(
                conv_input_on_device,
                grid_size,
                [
                    untilize_with_halo_input_shard_height,
                    input_channels,
                ],  # act_block_w_datums may include reads of multiple pixels in window
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttl.tensor.ShardOrientation.ROW_MAJOR,
            )
        return conv_input_on_device

    def copy_output_from_device(self, conv_output_on_device):
        interleaved_mem_config = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
        )
        # Convert sharded output to tiled interleaved
        conv_output_on_device = ttl.tensor.sharded_to_interleaved(conv_output_on_device, interleaved_mem_config)

        # convert tiled output to RM
        assert conv_output_on_device.layout() == ttl.tensor.Layout.TILE
        if conv_output_on_device.shape() != conv_output_on_device.shape_without_padding():
            conv_output_on_device = ttl.tensor.format_output_tensor(
                conv_output_on_device,
                conv_output_on_device.shape_without_padding(),
                self.device,
                ttl.tensor.Layout.ROW_MAJOR,
                interleaved_mem_config,
            )
        else:
            conv_output_on_device = ttl.tensor.untilize(
                conv_output_on_device, interleaved_mem_config, use_multicore=True
            )

        conv_output_on_device = conv_output_on_device.reshape(
            self.conv_output_shape[0],
            self.conv_output_shape[1],
            self.conv_output_shape[2],
            self.conv_output_shape[3],
        )

        # Copy to host
        return conv_output_on_device.cpu()

    # TODO: with this api, we get TT_ASSERT @ tt_metal/impl/dispatch/command_queue.cpp:790: dev_page_id < num_pages and dev_page_id >= 0
    def copy_output_from_device_with_sharded_api(self, conv_output_on_device):
        conv_output = conv_output_on_device.cpu().to(ttl.tensor.Layout.ROW_MAJOR)

        conv_output = conv_output.reshape(
            self.conv_output_shape[0],
            self.conv_output_shape[1],
            self.conv_output_shape[2],
            self.conv_output_shape[3],
        )

        return conv_output
