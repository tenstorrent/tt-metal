# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

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
    SlidingWindowOpParamsWithParallelConfig,
    SlidingWindowOpParams,
    get_hash_from_sliding_window_op_params,
)

from typing import Union

from tt_lib.utils import _nearest_32
import tt_lib as ttl

import math
import torch

GS_GRID_SIZE = (12, 9)
WH_GRID_SIZE = (8, 8)


# def determine_parallel_config(swo_params: SlidingWindowOpParams):
#     dilation_h, dilation_w = 1, 1
#     out_h = (
#         math.floor(
#             (swo_params.input_h + 2 * swo_params.pad_h - (dilation_h * swo_params.window_h - 1) - 1)
#             / swo_params.stride_h
#         )
#         + 1
#     )
#     out_w = (
#         math.floor(
#             (swo_params.input_w + 2 * swo_params.pad_w - (dilation_w * swo_params.window_w - 1) - 1)
#             / swo_params.stride_w
#         )
#         + 1
#     )

#     ncores_nhw = 1
#     grid_size = (1, 1)
#     shard_grid = ttl.tensor.CoreRangeSet({ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(0, 0))})
#     out_hw = out_h * out_w
#     out_nhw = swo_params.batch_size * out_hw

#     ## NOTE: these should match the max_pool op code for now.
#     if out_nhw == 1024:
#         ncores_nhw = 32
#         grid_size = (12, 3)
#         shard_grid = ttl.tensor.CoreRangeSet(
#             {
#                 ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(11, 1)),
#                 ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 2), ttl.tensor.CoreCoord(7, 2)),
#             }
#         )
#     elif out_nhw == 2048 or out_nhw == 4096 or out_nhw == 8192 or out_nhw == 16384 or out_nhw == 32768:
#         ncores_nhw = 64
#         grid_size = (12, 6)
#         shard_grid = ttl.tensor.CoreRangeSet(
#             {
#                 ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(11, 4)),
#                 ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 5), ttl.tensor.CoreCoord(3, 5)),
#             }
#         )
#     elif (
#         out_nhw == 3136
#         or out_nhw == 6272
#         or out_nhw == 12544
#         or out_nhw == 25088
#         or out_nhw == 50176
#         or out_nhw == 62720
#     ):
#         ncores_nhw = 98
#         grid_size = (12, 9)
#         shard_grid = ttl.tensor.CoreRangeSet(
#             {
#                 ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(11, 7)),
#                 ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 8), ttl.tensor.CoreCoord(1, 8)),
#             }
#         )
#     else:
#         grid_size = GS_GRID_SIZE
#         ncores_nhw = grid_size[0] * grid_size[1]
#         while ncores_nhw > 0:
#             ## 1. each shard should be equal, 2. each shard should be multiple of 32 (this is needed only for bfp8_b datatype (TILE))
#             if out_nhw % ncores_nhw == 0 and out_nhw // ncores_nhw % 32 == 0:
#                 break
#             ncores_nhw -= 1
#         if ncores_nhw == 0:
#             assert False, f"Unsupported output shape for max_pool: {out_nhw}"
#         grid_size = (grid_size[0], math.ceil(ncores_nhw / grid_size[0]))  ## bounding box
#         if ncores_nhw < grid_size[0]:
#             shard_grid = ttl.tensor.CoreRangeSet(
#                 {ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(ncores_nhw - 1, 0))}
#             )
#         else:
#             if ncores_nhw % grid_size[0] == 0:
#                 shard_grid = ttl.tensor.CoreRangeSet(
#                     {
#                         ttl.tensor.CoreRange(
#                             ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(grid_size[0] - 1, grid_size[1] - 1)
#                         )
#                     }
#                 )
#             else:
#                 shard_grid = ttl.tensor.CoreRangeSet(
#                     {
#                         ttl.tensor.CoreRange(
#                             ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(grid_size[0] - 1, grid_size[1] - 2)
#                         ),
#                         ttl.tensor.CoreRange(
#                             ttl.tensor.CoreCoord(0, grid_size[1] - 1),
#                             ttl.tensor.CoreCoord(ncores_nhw % grid_size[0] - 1, grid_size[1] - 1),
#                         ),
#                     }
#                 )

#     return grid_size, shard_grid, ncores_nhw


def calculate_shard_grid(ncores_nhw, grid_size):
    shard_grid = None
    if ncores_nhw < grid_size[0]:
        shard_grid = ttl.tensor.CoreRangeSet(
            {ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(ncores_nhw - 1, 0))}
        )
    else:
        if ncores_nhw % grid_size[0] == 0:
            shard_grid = ttl.tensor.CoreRangeSet(
                {
                    ttl.tensor.CoreRange(
                        ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(grid_size[0] - 1, grid_size[1] - 1)
                    )
                }
            )
        else:
            shard_grid = ttl.tensor.CoreRangeSet(
                {
                    ttl.tensor.CoreRange(
                        ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(grid_size[0] - 1, grid_size[1] - 2)
                    ),
                    ttl.tensor.CoreRange(
                        ttl.tensor.CoreCoord(0, grid_size[1] - 1),
                        ttl.tensor.CoreCoord(ncores_nhw % grid_size[0] - 1, grid_size[1] - 1),
                    ),
                }
            )
    return shard_grid


class TTPyMaxPool(TTPyOp):
    def __init__(
        self,
        sliding_window_op_params: Union[SlidingWindowOpParams, SlidingWindowOpParamsWithParallelConfig],
        device,
        reader_patterns_cache,
        pad_val=0xF7FF,
        output_mem_config=None,
    ):
        if "max_pool" not in reader_patterns_cache:
            reader_patterns_cache["max_pool"] = {}
        if "halo" not in reader_patterns_cache:
            reader_patterns_cache["halo"] = {}

        for key in reader_patterns_cache:
            assert (
                key == "max_pool" or key == "halo" or key == "conv"
            ), f"reader_patterns_cache should have 1 of the following keys - 'conv', 'max_pool' or 'halo'. Found key - {key}"

        # if output_mem_config is not None:
        #     dtype = output_mem_config.dtype()

        from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_composite_conv import (
            determine_parallel_config as conv_determine_parallel_config,
        )

        # self.grid_size, self.shard_grid, self.ncores_nhw = determine_parallel_config(sliding_window_op_params)
        conv_parallel_config, self.ncores_nhw = conv_determine_parallel_config(
            True,
            sliding_window_op_params.batch_size,
            0,
            0,
            sliding_window_op_params.input_h,
            sliding_window_op_params.input_w,
            sliding_window_op_params,
            device,
        )
        self.grid_size = (conv_parallel_config.grid_size.x, conv_parallel_config.grid_size.y)
        self.shard_grid = calculate_shard_grid(self.ncores_nhw, self.grid_size)

        print(f"grid_size: {self.grid_size}, shard_grid: {self.shard_grid}, ncores_nhw: {self.ncores_nhw}")

        if isinstance(sliding_window_op_params, SlidingWindowOpParams):
            self.sliding_window_op_params = SlidingWindowOpParamsWithParallelConfig(
                stride_h=sliding_window_op_params.stride_h,
                stride_w=sliding_window_op_params.stride_w,
                pad_h=sliding_window_op_params.pad_h,
                pad_w=sliding_window_op_params.pad_w,
                window_h=sliding_window_op_params.window_h,
                window_w=sliding_window_op_params.window_w,
                batch_size=sliding_window_op_params.batch_size,
                input_h=sliding_window_op_params.input_h,
                input_w=sliding_window_op_params.input_w,
                num_cores_h=self.grid_size[1],
                num_cores_w=self.grid_size[0],
                num_cores_nhw=self.ncores_nhw,
            )
        else:
            self.sliding_window_op_params = sliding_window_op_params

        sliding_window_op_params_hash = get_hash_from_sliding_window_op_params(self.sliding_window_op_params)

        self.device = device

        self.set_op_configs(
            sliding_window_op_params_hash,
            reader_patterns_cache["max_pool"],
        )
        assert sliding_window_op_params_hash in reader_patterns_cache["max_pool"]
        reader_indices = reader_patterns_cache["max_pool"][sliding_window_op_params_hash]

        self.set_op_weights_biases(
            self.sliding_window_op_params,
            output_mem_config,
            reader_indices,
        )

        self.pad_val = pad_val
        self.untilize_with_halo = TTPyUntilizeWithHalo(
            self.device,
            self.sliding_window_op_params,
            reader_patterns_cache["halo"],
            pad_val=self.pad_val,
            is_out_tiled=False,
        )

    # override abstract methods from base class TTPyOp
    def set_op_configs(self, sliding_window_op_params_hash, reader_patterns_cache):
        if sliding_window_op_params_hash not in reader_patterns_cache:
            stride_h = self.sliding_window_op_params.stride_h
            stride_w = self.sliding_window_op_params.stride_w
            pad_h = self.sliding_window_op_params.pad_h
            pad_w = self.sliding_window_op_params.pad_w
            window_h = self.sliding_window_op_params.window_h
            window_w = self.sliding_window_op_params.window_w
            batch_size = self.sliding_window_op_params.batch_size
            input_h = self.sliding_window_op_params.input_h
            input_w = self.sliding_window_op_params.input_w

            ncores_h = self.sliding_window_op_params.num_cores_h
            ncores_w = self.sliding_window_op_params.num_cores_w
            ncores_nhw = self.sliding_window_op_params.num_cores_nhw

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
                    data_top_left_indices, req_conv_input_shard_start_end, pad_tile=True, pad_last_core=True
                )
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
            shard_orientation = ttl.tensor.ShardOrientation.ROW_MAJOR
            shard_halo = False
            shard_spec = ttl.tensor.ShardSpec(self.shard_grid, [1, output_shard_height], shard_orientation, shard_halo)
            mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, shard_spec
            )
            reader_indices_sharded_tensor = reader_indices_tt_tensor.to(self.device, mem_config)

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
            act_mem_config = activation.memory_config()
            haloed_act = self.untilize_with_halo(activation)
            activation.deallocate()
            output = ttl.tensor.max_pool2d_v2(
                haloed_act,
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
                output_mem_config=act_mem_config if output_mem_config is None else output_mem_config,
            )
            return output

        self.max_pool = max_pool_

    def __call__(self, activation):
        return self.max_pool(activation)

    def copy_input_to_device(self, input: ttl.tensor.Tensor):
        interleaved_mem_config = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1
        )
        in_shape = input.shape()
        in_c = in_shape[-1]
        in_n = self.sliding_window_op_params.batch_size
        in_h = self.sliding_window_op_params.input_h
        in_w = self.sliding_window_op_params.input_w
        assert in_c % 16 == 0, "Input channels should be multiple of 32. General case is TODO"

        ## this op expects input tensor as { N, 1, H * W, C } or { 1, 1, N * H * W, C }

        in_hw = in_h * in_w
        if input.dtype() == ttl.tensor.DataType.BFLOAT8_B:
            ## currently the case when the input is bfp8_b and height is not divible by tile height is not supported. TODO.
            assert in_hw % 32 == 0, "For BFP8_B datatype, input height * width should be multiple of 32"
            ## last two dims are multiple of tile size (padded if needed)
            in_hw_padded = _nearest_32(in_hw)
            assert in_hw_padded == in_shape[1] * in_shape[2] or in_hw_padded == in_shape[0] * in_shape[1] * in_shape[2]
            act_shape = (in_n, 1, in_hw_padded, in_c)
        else:
            act_shape = (in_n, 1, in_hw, in_c)

        act_reshaped = input.reshape(act_shape).to(self.device, interleaved_mem_config)

        # padded_shape = ttl.tensor.pad_to_tile_shape(act_reshaped.shape(), False, False, False, True)
        # act_reshaped = ttl.tensor.format_input_tensor(
        #             act_reshaped,
        #             self.device,
        #             padded_shape,
        #             0.0,
        #             ttl.tensor.Layout.ROW_MAJOR,
        #             interleaved_mem_config,
        #         )

        shard_shape = [in_n * in_hw // self.sliding_window_op_params.num_cores_nhw, in_c]
        act_sharded = ttl.tensor.interleaved_to_sharded(
            act_reshaped,
            self.grid_size,
            shard_shape,
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )
        act_reshaped.deallocate()
        return act_sharded

    def copy_output_from_device(self, output_d: ttl.tensor.Tensor):
        interleaved_mem_config = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
        )
        output_d = ttl.tensor.sharded_to_interleaved(output_d, interleaved_mem_config)
        return output_d.cpu()
