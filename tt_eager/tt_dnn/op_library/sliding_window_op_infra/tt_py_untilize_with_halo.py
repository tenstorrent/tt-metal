# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_op import TTPyOp
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.untilize_with_halo_config_generation_and_validation import (
    trace_conv_to_generate_data_top_left_indices_and_pad_metadata,
    decompose_conv_into_shards_and_generate_tensor_metadata,
    generate_untilize_with_halo_kernel_configs,
)
from tt_lib.utils import _nearest_y
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.sliding_window_op_utils import (
    SlidingWindowOpParamsWithParallelConfig,
    get_hash_from_sliding_window_op_params,
    get_sliding_window_op_output_shard_nhw_size,
)
import tt_lib as ttl
import torch
import struct


class TTPyUntilizeWithHalo(TTPyOp):
    def __init__(
        self,
        device,
        sliding_window_op_params: SlidingWindowOpParamsWithParallelConfig,
        halo_reader_patterns_cache,
        pad_val=0x0,
    ):
        self.sliding_window_op_params = sliding_window_op_params
        self.device = device
        sliding_window_op_params_hash = get_hash_from_sliding_window_op_params(sliding_window_op_params)
        self.set_op_configs(device, sliding_window_op_params_hash, sliding_window_op_params, halo_reader_patterns_cache)
        assert sliding_window_op_params_hash in halo_reader_patterns_cache
        utwh_kernel_configs = halo_reader_patterns_cache[sliding_window_op_params_hash]

        ncores_w = sliding_window_op_params.num_cores_w
        ncores_h = sliding_window_op_params.num_cores_h
        ncores_nhw = sliding_window_op_params.num_cores_nhw

        is_block_sharding = ncores_w == ncores_nhw
        out_mem_config = None
        if is_block_sharding:
            out_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1
            )
        else:
            out_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1
            )

        def utwh_(activation):
            # print("sliding_window_op_params=", self.sliding_window_op_params)
            return ttl.tensor.untilize_with_halo_v2(
                activation,
                utwh_kernel_configs["local_pad_tensor"],
                utwh_kernel_configs["ll_data_tensor"],
                utwh_kernel_configs["l_data_tensor"],
                utwh_kernel_configs["local_data_tensor"],
                utwh_kernel_configs["r_data_tensor"],
                utwh_kernel_configs["rr_data_tensor"],
                pad_val,
                self.sliding_window_op_params.num_cores_nhw,
                utwh_kernel_configs["max_out_nsticks_per_core"],
                utwh_kernel_configs["local_pad_nsegments_per_core"],
                utwh_kernel_configs["ll_data_nsegments_per_core"],
                utwh_kernel_configs["l_data_nsegments_per_core"],
                utwh_kernel_configs["local_data_nsegments_per_core"],
                utwh_kernel_configs["r_data_nsegments_per_core"],
                utwh_kernel_configs["rr_data_nsegments_per_core"],
                utwh_kernel_configs["local_data_src_start_offsets_per_core"],
                utwh_kernel_configs["ll_data_src_start_offsets_per_core"],
                utwh_kernel_configs["l_data_src_start_offsets_per_core"],
                utwh_kernel_configs["r_data_src_start_offsets_per_core"],
                utwh_kernel_configs["rr_data_src_start_offsets_per_core"],
                out_mem_config,
            )

        self.utwh = utwh_

    # override abstract methods from base class TTPyOp
    def set_op_configs(
        self, device, sliding_window_op_params_hash, sliding_window_op_params, halo_reader_patterns_cache
    ):
        if sliding_window_op_params_hash not in halo_reader_patterns_cache:
            stride_h = sliding_window_op_params.stride_h
            stride_w = sliding_window_op_params.stride_w
            pad_h = sliding_window_op_params.pad_h
            pad_w = sliding_window_op_params.pad_w
            window_h = sliding_window_op_params.window_h
            window_w = sliding_window_op_params.window_w
            input_n = sliding_window_op_params.batch_size
            input_h = sliding_window_op_params.input_h
            input_w = sliding_window_op_params.input_w
            # TODO: Had to add this (should this be shard grid?)
            num_cores_w = sliding_window_op_params.num_cores_w
            num_cores_h = sliding_window_op_params.num_cores_h
            num_cores_nhw = sliding_window_op_params.num_cores_nhw
            assert num_cores_nhw > 0
            # TODO: send input_nhw_shape to generate functions (no need for C)
            # output_channels, input_channels, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation, groups
            sliding_window_op_all_params = [1, 1, window_h, window_w, stride_h, stride_w, pad_h, pad_w, 1, 1]
            input_nchw_shape = [input_n, 1, input_h, input_w]
            pad_metadata, data_top_left_indices = trace_conv_to_generate_data_top_left_indices_and_pad_metadata(
                sliding_window_op_all_params, input_nchw_shape
            )
            sliding_window_output_shard_nhw_size = get_sliding_window_op_output_shard_nhw_size(
                num_cores_nhw, input_n, input_h, input_w, stride_h, stride_w, pad_h, pad_w, window_h, window_w
            )
            untilize_w_halo_input_nhw_size_to_shard_evenly = _nearest_y(input_n * input_h * input_w, num_cores_nhw * 32)
            untilize_with_halo_input_shard_nhw_size = (int)(
                untilize_w_halo_input_nhw_size_to_shard_evenly / num_cores_nhw
            )
            req_conv_input_shard_start_end, tensor_metadata = decompose_conv_into_shards_and_generate_tensor_metadata(
                data_top_left_indices,
                pad_metadata,
                input_w + (2 * pad_w),
                sliding_window_output_shard_nhw_size,
                untilize_with_halo_input_shard_nhw_size,
                num_cores_nhw,
                window_h,
                window_w,
            )

            (
                local_data,
                local_pad,
                ll_data,
                l_data,
                r_data,
                rr_data,
                src_start_idx,
                local_data_nsegments_per_core,
                local_pad_nsegments_per_core,
                ll_data_nsegments_per_core,
                l_data_nsegments_per_core,
                r_data_nsegments_per_core,
                rr_data_nsegments_per_core,
                max_out_nsticks_per_core,
            ) = generate_untilize_with_halo_kernel_configs(tensor_metadata, req_conv_input_shard_start_end)

            assert len(local_data) == num_cores_nhw
            # Flatten the configs per core and construct the sharded tensor
            local_data = [item for sublist in local_data for item in sublist]
            local_pad = [item for sublist in local_pad for item in sublist]
            ll_data = [item for sublist in ll_data for item in sublist]
            assert len(ll_data) == 0
            l_data = [item for sublist in l_data for item in sublist]
            r_data = [item for sublist in r_data for item in sublist]
            rr_data = [item for sublist in rr_data for item in sublist]

            # TODO: move this data structure these to generate function to validate the final data structure
            ll_data_src_start_offsets_per_core = [src_start_idx[core_id][0] for core_id in range(num_cores_nhw)]
            l_data_src_start_offsets_per_core = [src_start_idx[core_id][1] for core_id in range(num_cores_nhw)]
            local_data_src_start_offsets_per_core = [src_start_idx[core_id][2] for core_id in range(num_cores_nhw)]
            r_data_src_start_offsets_per_core = [src_start_idx[core_id][3] for core_id in range(num_cores_nhw)]
            # print(f'r_data_src_start_offset: {self.r_data_src_start_offsets_per_core}')
            rr_data_src_start_offsets_per_core = [src_start_idx[core_id][4] for core_id in range(num_cores_nhw)]
            height_sharded_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1
            )

            block_sharding = num_cores_nhw == num_cores_w
            if not block_sharding:
                assert num_cores_w == 12
                num_cores_height_excluding_remainder_last_row = num_cores_nhw // num_cores_w
                assert num_cores_h >= num_cores_height_excluding_remainder_last_row
                core_range_1 = ttl.tensor.CoreRange(
                    ttl.tensor.CoreCoord(0, 0),
                    ttl.tensor.CoreCoord(num_cores_w - 1, num_cores_height_excluding_remainder_last_row - 1),
                )
                num_cores_last = num_cores_nhw % num_cores_w
                core_range_2 = None
                if num_cores_last > 0:
                    assert num_cores_h == num_cores_height_excluding_remainder_last_row + 1
                    core_range_2 = ttl.tensor.CoreRange(
                        ttl.tensor.CoreCoord(0, num_cores_height_excluding_remainder_last_row),
                        ttl.tensor.CoreCoord(num_cores_last - 1, num_cores_height_excluding_remainder_last_row),
                    )
                    shard_grid = ttl.tensor.CoreRangeSet({core_range_1, core_range_2})
                else:
                    assert num_cores_h == num_cores_height_excluding_remainder_last_row
                    shard_grid = ttl.tensor.CoreRangeSet({core_range_1})
            else:
                core_range = ttl.tensor.CoreRange(
                    ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(num_cores_w - 1, num_cores_h - 1)
                )
                shard_grid = ttl.tensor.CoreRangeSet({core_range})

            def gen_config_tt_tensors_uint16(config_list_uint16: list, toprint=False):
                config_size = len(config_list_uint16)
                if config_size == 0:
                    # return dummy tensor
                    return ttl.tensor.Tensor(
                        [0, 0], [1, 1, 1, 2], ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, device
                    )
                assert config_size % 2 == 0  ## each config is a tuple of (start, size)
                assert config_size % num_cores_nhw == 0

                shard_config_size = config_size // num_cores_nhw
                config_shard_shape = [1, shard_config_size]

                if block_sharding:
                    config_list_uint16 *= num_cores_h
                    config_size *= num_cores_h

                config_tensor_shape = (
                    1,
                    1,
                    1,
                    config_size,
                )

                # print(f"config list size = {config_size}")

                torch_tensor = torch.tensor(config_list_uint16, dtype=torch.short).reshape(config_tensor_shape)
                shard_orientation = ttl.tensor.ShardOrientation.ROW_MAJOR
                shard_halo = False
                shard_spec = ttl.tensor.ShardSpec(shard_grid, config_shard_shape, shard_orientation, shard_halo)

                tt_tensor = ttl.tensor.Tensor(torch_tensor, ttl.tensor.DataType.UINT16).to(
                    device, height_sharded_mem_config, shard_spec
                )
                # ttl.device.DumpDeviceMemoryState(device)

                ## validate
                tt_tensor_cpu = tt_tensor.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().reshape(config_tensor_shape)
                assert all(torch_tensor.reshape(-1) == tt_tensor_cpu.reshape(-1))

                return tt_tensor

            # print("gen local data config tt tensor")
            local_data_tensor = gen_config_tt_tensors_uint16(local_data)
            # print("gen local pad config tt tensor")
            local_pad_tensor = gen_config_tt_tensors_uint16(local_pad)
            # print("gen ll data config tt tensor")
            ll_data_tensor = gen_config_tt_tensors_uint16(ll_data)
            # print("gen l data config tt tensor")
            l_data_tensor = gen_config_tt_tensors_uint16(l_data)
            # print("gen r data config tt tensor")
            r_data_tensor = gen_config_tt_tensors_uint16(r_data)
            # print("gen rr data config tt tensor")
            rr_data_tensor = gen_config_tt_tensors_uint16(rr_data)

            halo_reader_patterns_cache[sliding_window_op_params_hash] = {
                "local_pad_tensor": local_pad_tensor,
                "local_data_tensor": local_data_tensor,
                "ll_data_tensor": ll_data_tensor,
                "l_data_tensor": l_data_tensor,
                "r_data_tensor": r_data_tensor,
                "rr_data_tensor": rr_data_tensor,
                "max_out_nsticks_per_core": max_out_nsticks_per_core,
                "local_pad_nsegments_per_core": local_pad_nsegments_per_core,
                "local_data_nsegments_per_core": local_data_nsegments_per_core,
                "ll_data_nsegments_per_core": ll_data_nsegments_per_core,
                "l_data_nsegments_per_core": l_data_nsegments_per_core,
                "r_data_nsegments_per_core": r_data_nsegments_per_core,
                "rr_data_nsegments_per_core": rr_data_nsegments_per_core,
                "local_data_src_start_offsets_per_core": local_data_src_start_offsets_per_core,
                "ll_data_src_start_offsets_per_core": ll_data_src_start_offsets_per_core,
                "l_data_src_start_offsets_per_core": l_data_src_start_offsets_per_core,
                "r_data_src_start_offsets_per_core": r_data_src_start_offsets_per_core,
                "rr_data_src_start_offsets_per_core": rr_data_src_start_offsets_per_core,
            }

        return

    def __call__(self, activation):
        return self.utwh(activation)
