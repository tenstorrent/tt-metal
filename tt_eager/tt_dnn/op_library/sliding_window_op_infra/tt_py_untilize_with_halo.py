from typing import List
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_op import TTPyOp
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.untilize_with_halo_config_generation_and_validation import (
    trace_conv_to_generate_data_top_left_indices_and_pad_metadata,
    decompose_conv_into_shards_and_generate_tensor_metadata,
    generate_untilize_with_halo_kernel_configs,
)
from tt_lib.utils import _nearest_y
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.sliding_window_op_utils import (
    get_sliding_window_op_output_shard_nhw_size,
)
import tt_lib as ttl
import torch
import struct


class TTPyUntilizeWithHalo(TTPyOp):
    # cache map for kernel configs corresponding to unique sliding window op params
    # sliding window op params: tuple(stride_hw: tuple(int, int), pad_hw: tuple(int, int), window_hw: tuple(int, int), input_nhw: tuple(int, int, int), num_cores_nhw: int)
    static_kernel_configs_cache_map = {}
    # TODO: add config_tensors to member variables

    def __init__(self, device, sliding_window_op_params, shard_grid):
        self.sliding_window_op_params = sliding_window_op_params
        self.shard_grid = shard_grid
        self.device = device

    # override abstract methods from base class TTPyOp
    def set_op_configs(self):
        # TODO: nitika - clean up params data structure
        assert len(self.sliding_window_op_params) == 5
        stride_h, stride_w = self.sliding_window_op_params[0]
        pad_h, pad_w = self.sliding_window_op_params[1]
        window_h, window_w = self.sliding_window_op_params[2]
        input_n, input_h, input_w = self.sliding_window_op_params[3]
        num_cores_nhw = self.sliding_window_op_params[4]
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
        untilize_with_halo_input_shard_nhw_size = (int)(untilize_w_halo_input_nhw_size_to_shard_evenly / num_cores_nhw)
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
            self.local_data_nsegments_per_core,
            self.local_pad_nsegments_per_core,
            self.ll_data_nsegments_per_core,
            self.l_data_nsegments_per_core,
            self.r_data_nsegments_per_core,
            self.rr_data_nsegments_per_core,
            self.max_out_nsticks_per_core,
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
        self.ll_data_src_start_offsets_per_core = [src_start_idx[core_id][0] for core_id in range(num_cores_nhw)]
        self.l_data_src_start_offsets_per_core = [src_start_idx[core_id][1] for core_id in range(num_cores_nhw)]
        self.local_data_src_start_offsets_per_core = [src_start_idx[core_id][2] for core_id in range(num_cores_nhw)]
        self.r_data_src_start_offsets_per_core = [src_start_idx[core_id][3] for core_id in range(num_cores_nhw)]
        # print(f'r_data_src_start_offset: {self.r_data_src_start_offsets_per_core}')
        self.rr_data_src_start_offsets_per_core = [src_start_idx[core_id][4] for core_id in range(num_cores_nhw)]
        self.height_sharded_mem_config = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1
        )

        def gen_config_tt_tensors(config_list_uint16, toprint=False):
            if len(config_list_uint16) == 0:
                # return dummy tensor
                return ttl.tensor.Tensor(
                    [0.0, 0.0], [1, 1, 1, 2], ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, self.device
                )
            config_list = []
            for i in range(0, len(config_list_uint16), 2):
                pstr = struct.pack("HH", config_list_uint16[i], config_list_uint16[i + 1])
                packedint = struct.unpack("I", pstr)
                if toprint:
                    print(f"{config_list_uint16[i]},{config_list_uint16[i+1]} -> {packedint}")
                config_list.append(packedint)
            config_size = len(config_list)
            assert config_size % num_cores_nhw == 0
            shard_config_size = (int)(config_size / num_cores_nhw)
            config_shard_shape = [1, shard_config_size]  # = local_data_nsegments * 2
            config_tensor_shape = (
                1,
                1,
                1,
                config_size,  # = num_cores * local_data_nsegments * 2
            )
            torch_tensor = torch.tensor(config_list).reshape(config_tensor_shape)
            shard_orientation = ttl.tensor.ShardOrientation.ROW_MAJOR
            shard_halo = False
            shard_spec = ttl.tensor.ShardSpec(self.shard_grid, config_shard_shape, shard_orientation, shard_halo)

            # tt_tensor = ttl.tensor.Tensor(config_list, config_tensor_shape, ttl.tensor.DataType.UINT32, ttl.tensor.Layout.ROW_MAJOR).to(self.device, mem_config, shard_spec)
            tt_tensor = ttl.tensor.Tensor(torch_tensor, ttl.tensor.DataType.UINT32).to(
                self.device, self.height_sharded_mem_config, shard_spec
            )

            tt_tensor_cpu = tt_tensor.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
            if toprint:
                tt_tensor_cpu.print()
            torch_tensor_after_round_trip = tt_tensor_cpu.to_torch().reshape(config_tensor_shape)
            assert all(torch_tensor.reshape(-1) == torch_tensor_after_round_trip.reshape(-1))
            return tt_tensor

        print("gen local data config tt tensor")
        self.local_data_tensor = gen_config_tt_tensors(local_data)
        print("gen local pad config tt tensor")
        self.local_pad_tensor = gen_config_tt_tensors(local_pad)
        print("gen ll data config tt tensor")
        self.ll_data_tensor = gen_config_tt_tensors(ll_data)
        print("gen l data config tt tensor")
        self.l_data_tensor = gen_config_tt_tensors(l_data)
        print("gen r data config tt tensor")
        self.r_data_tensor = gen_config_tt_tensors(r_data, toprint=False)
        print("gen rr data config tt tensor")
        self.rr_data_tensor = gen_config_tt_tensors(rr_data)

        return

    def run_forward(self, x):
        return ttl.tensor.untilize_with_halo_v2(
            x,
            self.local_pad_tensor,
            self.ll_data_tensor,
            self.l_data_tensor,
            self.local_data_tensor,
            self.r_data_tensor,
            self.rr_data_tensor,
            0x0,  ## pad val
            # 0xF7FF,  ## pad_val
            self.sliding_window_op_params[4],
            self.max_out_nsticks_per_core,
            self.local_pad_nsegments_per_core,
            self.ll_data_nsegments_per_core,
            self.l_data_nsegments_per_core,
            self.local_data_nsegments_per_core,
            self.r_data_nsegments_per_core,
            self.rr_data_nsegments_per_core,
            self.local_data_src_start_offsets_per_core,
            self.ll_data_src_start_offsets_per_core,
            self.l_data_src_start_offsets_per_core,
            self.r_data_src_start_offsets_per_core,
            self.rr_data_src_start_offsets_per_core,
            self.height_sharded_mem_config,
        )
