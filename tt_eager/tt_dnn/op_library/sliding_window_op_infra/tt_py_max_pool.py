# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_op import TTPyOp
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.untilize_with_halo_config_generation_and_validation import (
    trace_conv_to_generate_data_top_left_indices_and_pad_metadata,
    decompose_conv_into_shards_and_generate_tensor_metadata,
)
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.sliding_window_op_config_generation_and_validation import (
    generate_sliding_window_op_sharded_input_top_left_indices,
)
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.sliding_window_op_utils import (
    SlidingWindowOpParamsWithParallelConfig,
    get_hash_from_sliding_window_op_params,
)

# from tt_lib.utils import _nearest_32, _nearest_y

import tt_lib as ttl
import torch


class TTPyMaxPool(TTPyOp):
    def __init__(
        self,
        sliding_window_op_params: SlidingWindowOpParamsWithParallelConfig,
        device,
        grid_size,
        reader_patterns_cache,
        output_mem_config=None,
    ):
        self.sliding_window_op_params = sliding_window_op_params
        sliding_window_op_params_hash = get_hash_from_sliding_window_op_params(sliding_window_op_params)

        self.set_op_configs(
            device, sliding_window_op_params_hash, sliding_window_op_params, grid_size, reader_patterns_cache
        )
        assert sliding_window_op_params_hash in reader_patterns_cache
        reader_indices = reader_patterns_cache[sliding_window_op_params_hash]

        self.set_op_weights_biases(
            sliding_window_op_params,
            output_mem_config,
            reader_indices,
        )

    # override abstract methods from base class TTPyOp
    def set_op_configs(
        self, device, sliding_window_op_params_hash, sliding_window_op_params, grid_size, reader_patterns_cache
    ):
        if sliding_window_op_params_hash not in reader_patterns_cache:
            stride_h = sliding_window_op_params.stride_h
            stride_w = sliding_window_op_params.stride_w
            pad_h = sliding_window_op_params.pad_h
            pad_w = sliding_window_op_params.pad_w
            window_h = sliding_window_op_params.window_h
            window_w = sliding_window_op_params.window_w
            batch_size = sliding_window_op_params.batch_size
            input_h = sliding_window_op_params.input_h
            input_w = sliding_window_op_params.input_w
            ncores_nhw = sliding_window_op_params.num_cores_nhw

            input_nchw_shape = [batch_size, 1, input_h, input_w]
            input_volume = batch_size * input_h * input_w
            output_h = ((int)((input_h + (2 * pad_h) - window_h) / stride_h)) + 1
            output_w = ((int)((input_w + (2 * pad_w) - window_w) / stride_w)) + 1
            output_volume = batch_size * output_h * output_w

            # input_size_to_shard_evenly = _nearest_y(input_volume, ncores_nhw * 32)
            assert input_volume % ncores_nhw == 0
            input_shard_height = input_volume // ncores_nhw

            # output_size_to_shard_evenly = _nearest_y(output_volume, ncores_nhw * 32)
            assert output_volume % ncores_nhw == 0
            output_shard_height = output_volume // ncores_nhw

            input_padded_width = input_w + 2 * pad_w

            pad_metadata, data_top_left_indices = trace_conv_to_generate_data_top_left_indices_and_pad_metadata(
                (1, 1, window_h, window_w, stride_h, stride_w, pad_h, pad_w, 1, 1), input_nchw_shape
            )

            req_conv_input_shard_start_end, tensor_metadata = decompose_conv_into_shards_and_generate_tensor_metadata(
                data_top_left_indices,
                pad_metadata,
                input_padded_width,
                output_shard_height,
                input_shard_height,
                ncores_nhw,
                window_h,
                window_w,
            )

            sliding_window_op_sharded_input_top_left_indices = (
                generate_sliding_window_op_sharded_input_top_left_indices(
                    data_top_left_indices, req_conv_input_shard_start_end
                )
            )

            # for core_data in sliding_window_op_sharded_input_top_left_indices:
            #     print(f"READER IDX {len(core_data)}: {core_data}")

            # Pad indices for last core if not equal to other cores
            indices_length_per_core = len(sliding_window_op_sharded_input_top_left_indices[0])
            sliding_window_op_sharded_input_top_left_indices[-1].extend(
                [0] * (indices_length_per_core - len(sliding_window_op_sharded_input_top_left_indices[-1]))
            )

            indices_torch_dtype = torch.int16
            indices_tt_dtype = ttl.tensor.DataType.UINT16

            # Create sharded tensor on device for conv_reader_indices
            reader_indices_torch_tensor = torch.tensor(
                [[sliding_window_op_sharded_input_top_left_indices]], dtype=indices_torch_dtype
            )

            reader_indices_tt_tensor = ttl.tensor.Tensor(
                reader_indices_torch_tensor,
                indices_tt_dtype,
            )
            shard_grid = ttl.tensor.CoreRangeSet(
                {
                    ttl.tensor.CoreRange(
                        ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(grid_size[0] - 1, grid_size[1] - 1)
                    )
                }
            )
            shard_orientation = ttl.tensor.ShardOrientation.ROW_MAJOR
            shard_halo = False
            shard_spec = ttl.tensor.ShardSpec(shard_grid, [1, output_shard_height], shard_orientation, shard_halo)
            mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1)
            reader_indices_sharded_tensor = reader_indices_tt_tensor.to(device, mem_config, shard_spec)

            reader_patterns_cache[sliding_window_op_params_hash] = reader_indices_sharded_tensor

        return

    def set_op_weights_biases(self, op_params, output_mem_config, reader_indices):
        stride_h = op_params.stride_h
        stride_w = op_params.stride_w
        pad_h = op_params.pad_h
        pad_w = op_params.pad_w
        window_h = op_params.window_h
        window_w = op_params.window_w
        in_n = op_params.batch_size
        in_h = op_params.input_h
        in_w = op_params.input_w

        def max_pool_(activation):
            output = ttl.tensor.max_pool2d_v2(
                activation,
                reader_indices,
                in_n,
                in_h,
                in_w,
                window_h,
                window_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                output_mem_config=activation.memory_config() if output_mem_config is None else output_mem_config,
            )
            return output

        self.max_pool = max_pool_

    def __call__(self, activation):
        return self.max_pool(activation)
