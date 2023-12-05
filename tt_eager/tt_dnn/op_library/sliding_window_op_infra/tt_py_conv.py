from typing import List, Union
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_op import TTPyOp
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.untilize_with_halo_config_generation_and_validation import (
    trace_conv_to_generate_data_top_left_indices_and_pad_metadata,
    decompose_conv_into_shards_and_generate_tensor_metadata,
)
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.sliding_window_op_config_generation_and_validation import (
    generate_sliding_window_op_sharded_input_top_left_indices,
)

from tt_lib.utils import _nearest_32, _nearest_y

import tt_lib as ttl
import torch


def _get_hash_from_sliding_window_op_params(sliding_window_op_params):
    stride_h, stride_w = sliding_window_op_params[0]
    pad_h, pad_w = sliding_window_op_params[1]
    filter_h, filter_w = sliding_window_op_params[2]
    batch_size, input_h, input_w = sliding_window_op_params[3]
    num_cores_w, num_cores_h = sliding_window_op_params[4]
    num_cores_nhw = sliding_window_op_params[5]

    return f"{stride_h}_{stride_w}_{pad_h}_{pad_w}_{filter_h}_{filter_w}_{batch_size}_{input_h}_{input_w}_{num_cores_w}_{num_cores_h}_{num_cores_nhw}"


class TTPyConv(TTPyOp):
    # cache map for kernel configs corresponding to unique sliding window op params
    # sliding window op params: tuple(stride_hw: tuple(int, int), pad_hw: tuple(int, int), filter_hw: tuple(int, int), input_nhw: tuple(int, int, int), num_cores_nhw: int)
    static_kernel_configs_cache_map = {}

    def __init__(
        self,
        sliding_window_op_params,
        weight: List[Union[int, float]],
        conv_params,
        device,
        act_block_shape_hw,
        weight_block_shape_hw,
        outsubblock_shape_hw,
        out_block_shape_h,
        grid_size,
        per_core_out_matrix_h_ntiles,
        per_core_weight_matrix_w_ntiles,
        bias,
        fuse_relu=False,
        output_mem_config=None,
        input_tensor_shape=None,
        weights_dtype=None,
        output_dtype=None,
        math_fidelity=None,
        act_c_num_blocks=1,
    ):
        self.sliding_window_op_params = sliding_window_op_params
        sliding_window_op_params_hash = _get_hash_from_sliding_window_op_params(sliding_window_op_params)

        # set_op_configs populates static_kernel_configs_cache_map[sliding_window_op_params_hash] with conv_reader_indices sharded tensor
        conv_is_2d = act_c_num_blocks > 1
        self.set_op_configs(
            device,
            sliding_window_op_params_hash,
            sliding_window_op_params,
            conv_params,
            conv_is_2d,
        )
        conv_reader_indices = TTPyConv.static_kernel_configs_cache_map[sliding_window_op_params_hash]

        self.set_op_weights_biases(
            weight,
            conv_params,
            device,
            act_block_shape_hw,
            weight_block_shape_hw,
            outsubblock_shape_hw,
            out_block_shape_h,
            grid_size,
            per_core_out_matrix_h_ntiles,
            per_core_weight_matrix_w_ntiles,
            bias,
            fuse_relu,
            output_mem_config,
            input_tensor_shape,
            weights_dtype,
            output_dtype,
            math_fidelity,
            act_c_num_blocks,
            conv_reader_indices,
        )

    # override abstract methods from base class TTPyOp
    @classmethod
    def set_op_configs(
        cls,
        device,
        sliding_window_op_params_hash,
        sliding_window_op_params,
        conv_params,
        conv_is_2d,
    ):
        # TODO: Need way of hashing sliding_window_op_params
        if sliding_window_op_params_hash not in cls.static_kernel_configs_cache_map:
            # TODO: Need to clean up sliding_window_op_params and conv_params (they are basically the same)
            stride_h, stride_w = sliding_window_op_params[0]
            pad_h, pad_w = sliding_window_op_params[1]
            filter_h, filter_w = sliding_window_op_params[2]
            batch_size, input_h, input_w = sliding_window_op_params[3]
            num_cores_w, num_cores_h = sliding_window_op_params[4]  # TODO: Had to add this (should this be shard grid?)
            num_cores_nhw = sliding_window_op_params[5]

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

            cls.static_kernel_configs_cache_map[sliding_window_op_params_hash] = conv_reader_indices_sharded_tensor

    # TODO: Maybe need to have this be more general to settting up conv
    def set_op_weights_biases(
        self,
        weight: List[Union[int, float]],
        conv_params,
        device,
        act_block_shape_hw,
        weight_block_shape_hw,
        outsubblock_shape_hw,
        out_block_shape_h,
        grid_size,
        per_core_out_matrix_h_ntiles,
        per_core_weight_matrix_w_ntiles,
        bias,
        fuse_relu,
        output_mem_config,
        input_tensor_shape,
        weights_dtype,
        output_dtype,
        math_fidelity,
        act_c_num_blocks,
        conv_reader_indices,
    ):
        assert len(conv_params) == 10
        K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]

        assert len(act_block_shape_hw) == 2
        assert len(weight_block_shape_hw) == 2
        assert len(outsubblock_shape_hw) == 2
        assert act_block_shape_hw[1] == weight_block_shape_hw[0]
        assert act_block_shape_hw[0] % 32 == 0
        assert act_block_shape_hw[1] % 32 == 0

        out_block_h = (int)(out_block_shape_h / 32)
        act_block_h = (int)(act_block_shape_hw[0] / 32)
        act_block_w = (int)(act_block_shape_hw[1] / 32)
        weight_block_h = act_block_w
        weight_block_w = (int)(weight_block_shape_hw[1] / 32)
        out_subblock_h = (int)(outsubblock_shape_hw[0] / 32)
        out_subblock_w = (int)(outsubblock_shape_hw[1] / 32)
        assert out_subblock_h * out_subblock_w <= 8

        assert dilation == 1 and groups == 1

        weights_shape = [K, C, R, S]
        weights_channels_padded_shape = [_nearest_32(K), _nearest_y(C, 16), R, S]
        weights_untiled_dtype = (
            weights_dtype if weights_dtype != ttl.tensor.DataType.BFLOAT8_B else ttl.tensor.DataType.FLOAT32
        )
        weight_untiled = ttl.tensor.Tensor(
            weight, weights_shape, weights_untiled_dtype, ttl.tensor.Layout.ROW_MAJOR
        ).pad(weights_channels_padded_shape, (0, 0, 0, 0), 0)
        act_block_w_equals_input_channels_x_filter_width = act_block_shape_hw[1] == (C * S)
        # for conv op, pad the weights to block shape
        if act_block_w_equals_input_channels_x_filter_width:
            weight_tiled_ = ttl.tensor.convert_conv_weight_tensor_to_tiled_layout(
                weight_untiled,
                weight_block_h,
                weight_block_w,
                output_dtype=weights_dtype,
            )
        else:
            if R == 1 and S == 1:
                assert C % act_block_shape_hw[1] == 0
            else:
                assert act_block_shape_hw[1] == C
            weight_tiled_ = ttl.tensor.convert_conv_weight_tensor_to_tiled_layout(
                weight_untiled,
                weight_block_h,
                weight_block_w,
                output_dtype=weights_dtype,
            )
        weight_on_device = weight_tiled_.to(device)
        bias_shape = [1, 1, 1, K]
        assert K % (weight_block_w * 32) == 0
        bias_channels_padded_shape = [1, 1, 32, _nearest_32(K)]
        bias = (
            torch.nn.functional.pad(torch.Tensor(bias).reshape(bias_shape), (0, _nearest_32(K) - K, 0, 31))
            .flatten()
            .tolist()
        )
        bias_ = ttl.tensor.Tensor(bias, bias_channels_padded_shape, weights_dtype, ttl.tensor.Layout.ROW_MAJOR).to(
            ttl.tensor.Layout.TILE
        )
        bias_on_device = bias_.to(device)

        opt_conv_parall_conf = ttl.tensor.OptimizedConvParallelizationConfig(
            grid_size=grid_size,
            per_core_out_matrix_height_ntiles=per_core_out_matrix_h_ntiles,
            per_core_weight_matrix_width_ntiles=per_core_weight_matrix_w_ntiles,
        )
        opt_conv_block_conf = ttl.tensor.OptimizedConvBlockConfig(
            act_block_h_ntiles=act_block_h,
            act_block_w_ntiles=act_block_w,
            act_c_num_blocks=act_c_num_blocks,
            weight_block_w_ntiles=weight_block_w,
            out_block_h_ntiles=out_block_h,
            out_subblock_h_ntiles=out_subblock_h,
            out_subblock_w_ntiles=out_subblock_w,
        )

        def conv_(activation):
            # assert(activation.layout() == ttl.tensor.Layout.ROW_MAJOR)
            output = ttl.tensor.optimized_conv(
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
                input_tensor_shape=input_tensor_shape,
            )
            # assert(output.storage_type() == ttl.tensor.StorageType.DEVICE)
            return output

        self.conv = conv_

    def __call__(self, activation):
        return self.conv(activation)
