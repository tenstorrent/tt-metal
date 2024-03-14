// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/tensor_utils.hpp"
#include "tensor/owned_buffer.hpp"
#include "tensor/owned_buffer_functions.hpp"

namespace tt {

namespace tt_metal {


    template <typename T>
    Tensor to_weight_special_padding_tile_layout(const Tensor& conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w, DataType output_dtype) {
        auto w_shape = conv_weight_tensor.get_legacy_shape();
        auto input_buffer = owned_buffer::get_as<T>(conv_weight_tensor);
        uint32_t in1_block_h_datums = in1_block_h * constants::TILE_HEIGHT;
        uint32_t in1_block_w_datums = in1_block_w * constants::TILE_WIDTH;
        auto weight_matrix_cols = w_shape[0];
        // width padding
        if(weight_matrix_cols%in1_block_w_datums != 0) {
            weight_matrix_cols = (uint32_t) std::ceil( (double) weight_matrix_cols / (double) in1_block_w_datums ) * in1_block_w_datums;
        }
        // height padding
        assert(in1_block_h_datums >= w_shape[1]*w_shape[3]);
        uint32_t block_height_padding = in1_block_h_datums - (w_shape[1]*w_shape[3]);
        auto weight_matrix_rows = ((w_shape[1]*w_shape[3]) + block_height_padding)*w_shape[2];
        Shape output_shape = {1, 1, weight_matrix_rows, weight_matrix_cols};
        auto output_buffer = owned_buffer::create<T>(compute_volume(output_shape));
        for(auto r = 0; r < w_shape[2]; r++) {
            for(auto s = 0; s < w_shape[3]; s++) {
                for(auto c = 0; c < w_shape[1]; c++) {
                    for(auto k = 0; k < w_shape[0]; k++) {
                        auto matrix_idx = k + c * weight_matrix_cols + s * w_shape[1] * weight_matrix_cols + r * ((w_shape[3] * w_shape[1]) + block_height_padding) * weight_matrix_cols;
			auto idx = k * w_shape[1] * w_shape[2] * w_shape[3] + c * w_shape[2] * w_shape[3] + r * w_shape[3] + s;
			output_buffer[matrix_idx] = input_buffer[idx];
                    }
                }
            }
        }
        if constexpr (std::is_same<T, float>::value) {
            if (output_dtype == DataType::BFLOAT8_B) {
                auto output_float_data = output_buffer.get();
                auto output_packed_data = pack_fp32_vec_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
                auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
                auto rm_tensor = Tensor(std::move(OwnedStorage{std::move(output_uint32_buffer)}), output_shape, output_dtype, Layout::ROW_MAJOR);
                return rm_tensor.to(Layout::TILE);
            }
            if (output_dtype == DataType::BFLOAT4_B) {
                auto output_float_data = output_buffer.get();
                auto output_packed_data = pack_fp32_vec_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
                auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
                auto rm_tensor = Tensor(std::move(OwnedStorage{std::move(output_uint32_buffer)}), output_shape, output_dtype, Layout::ROW_MAJOR);
                return rm_tensor.to(Layout::TILE);
            }
        } else {
            TT_ASSERT((output_dtype != DataType::BFLOAT8_B) || (output_dtype != DataType::BFLOAT4_B));
        }
        auto rm_tensor = Tensor(std::move(OwnedStorage{std::move(output_buffer)}), output_shape, output_dtype, Layout::ROW_MAJOR);
        return rm_tensor.to(Layout::TILE);
    }


    template <typename T>
    Tensor to_weight_tile_layout(const Tensor& conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w, DataType output_dtype) {
        auto w_shape = conv_weight_tensor.get_legacy_shape();
        auto input_buffer = owned_buffer::get_as<T>(conv_weight_tensor);
        auto weight_matrix_cols = w_shape[0];
        // width padding
        uint32_t in1_block_w_datums = in1_block_w * constants::TILE_WIDTH;
        if(weight_matrix_cols%in1_block_w_datums != 0) {
            weight_matrix_cols = (uint32_t) std::ceil( (double) weight_matrix_cols / (double) in1_block_w_datums ) * in1_block_w_datums;
        }
        // height padding
        auto weight_matrix_rows = w_shape[1]*w_shape[2]*w_shape[3];
        uint32_t in1_block_h_datums = in1_block_h * constants::TILE_HEIGHT;
        if (weight_matrix_rows % in1_block_h_datums != 0) {
            weight_matrix_rows = (uint32_t) std::ceil( (double) weight_matrix_rows / (double) in1_block_h_datums ) * in1_block_h_datums;
        }
        Shape output_shape = {1, 1, weight_matrix_rows, weight_matrix_cols};
        auto output_buffer = owned_buffer::create<T>(compute_volume(output_shape));
        for(auto r = 0; r < w_shape[2]; r++) {
            for(auto s = 0; s < w_shape[3]; s++) {
                for(auto c = 0; c < w_shape[1]; c++) {
                    for(auto k = 0; k < w_shape[0]; k++) {
                        auto matrix_idx = k + c * weight_matrix_cols + s * w_shape[1] * weight_matrix_cols + r * w_shape[3] * w_shape[1] * weight_matrix_cols;
                        auto idx = k * w_shape[1] * w_shape[2] * w_shape[3] + c * w_shape[2] * w_shape[3] + r * w_shape[3] + s;
                        output_buffer[matrix_idx] = input_buffer[idx];
                    }
                }
            }
        }
        if constexpr (std::is_same<T, float>::value) {
            if (output_dtype == DataType::BFLOAT8_B) {
                auto output_float_data = output_buffer.get();
                auto output_packed_data = pack_fp32_vec_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
                auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
                auto rm_tensor = Tensor(std::move(OwnedStorage{std::move(output_uint32_buffer)}), output_shape, output_dtype, Layout::ROW_MAJOR);
                return rm_tensor.to(Layout::TILE);
            }
            if (output_dtype == DataType::BFLOAT4_B) {
                auto output_float_data = output_buffer.get();
                auto output_packed_data = pack_fp32_vec_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
                auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
                auto rm_tensor = Tensor(std::move(OwnedStorage{std::move(output_uint32_buffer)}), output_shape, output_dtype, Layout::ROW_MAJOR);
                return rm_tensor.to(Layout::TILE);
            }
        } else {
            TT_ASSERT((output_dtype != DataType::BFLOAT8_B) || (output_dtype != DataType::BFLOAT4_B));
        }
        auto rm_tensor = Tensor(std::move(OwnedStorage{std::move(output_buffer)}), output_shape, output_dtype, Layout::ROW_MAJOR);
        return rm_tensor.to(Layout::TILE);
    }

    // Converts convolution weights to tilized 2d matrix layout.
    // Returns a new tensor with layout=Tile
    Tensor convert_conv_weight_tensor_to_tiled_layout(Tensor conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w, std::optional<DataType> output_dtype) {
        TT_ASSERT(conv_weight_tensor.get_layout() == Layout::ROW_MAJOR && "Convolution weights should be in row major layout for conversion to tilized layout.");
        const static std::map<DataType, std::function<Tensor(const Tensor &, uint32_t in1_block_h, uint32_t in1_block_w, DataType output_dtype)>> to_w_tile_layout_map = {
            {DataType::BFLOAT16, &to_weight_tile_layout<bfloat16>},
            {DataType::FLOAT32, &to_weight_tile_layout<float>},
            {DataType::UINT32, &to_weight_tile_layout<uint32_t>},
        };
        if (output_dtype.has_value()) {
            if (output_dtype == DataType::BFLOAT8_B || output_dtype == DataType::BFLOAT4_B) {
                TT_ASSERT(conv_weight_tensor.get_dtype() == DataType::FLOAT32);
            } else {
                TT_ASSERT(conv_weight_tensor.get_dtype() == conv_weight_tensor.get_dtype());
            }
        }
        return to_w_tile_layout_map.at(conv_weight_tensor.get_dtype())(conv_weight_tensor, in1_block_h, in1_block_w, output_dtype.value_or(conv_weight_tensor.get_dtype()));
    }

    // Converts convolution weights to tilized 2d matrix layout.
    // Returns a new tensor with layout=Tile
    Tensor convert_conv_weight_tensor_to_special_padding_tiled_layout(Tensor conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w, std::optional<DataType> output_dtype) {
        TT_ASSERT(conv_weight_tensor.get_layout() == Layout::ROW_MAJOR && "Convolution weights should be in row major layout for conversion to tilized layout.");
        const static std::map<DataType, std::function<Tensor(const Tensor &, uint32_t in1_block_h, uint32_t in1_block_w, DataType output_dtype)>> to_w_tile_layout_map = {
            {DataType::BFLOAT16, &to_weight_special_padding_tile_layout<bfloat16>},
            {DataType::FLOAT32, &to_weight_special_padding_tile_layout<float>},
            {DataType::UINT32, &to_weight_special_padding_tile_layout<uint32_t>}
        };
        if (output_dtype.has_value()) {
            if (output_dtype == DataType::BFLOAT8_B || output_dtype == DataType::BFLOAT4_B) {
                TT_ASSERT(conv_weight_tensor.get_dtype() == DataType::FLOAT32);
            } else {
                TT_ASSERT(conv_weight_tensor.get_dtype() == conv_weight_tensor.get_dtype());
            }
        }
        return to_w_tile_layout_map.at(conv_weight_tensor.get_dtype())(conv_weight_tensor, in1_block_h, in1_block_w, output_dtype.value_or(conv_weight_tensor.get_dtype()));
    }

const Shape infer_dims_for_reshape(int N, int C, int H, int W, uint32_t old_volume) {
    vector<int> ns{N, C, H, W};
    int neg_idx = -1;
    for (int i = 0; i < ns.size(); i++) {
        if (ns[i] == -1) {
            TT_ASSERT(neg_idx == -1, "Only one -1 is allowed in reshape");
            neg_idx = i;
        } else {
            TT_ASSERT(ns[i] > 0, "New shape entries can only have -1 or positive values");
        }
    }

    switch (neg_idx) {
        case 0:
            TT_ASSERT(old_volume % C*H*W == 0);
            N = old_volume/(C*H*W);
            break;
        case 1:
            TT_ASSERT(old_volume % N*H*W == 0);
            C = old_volume/(N*H*W);
            break;
        case 2:
            TT_ASSERT(old_volume % N*C*W == 0);
            H = old_volume/(N*C*W);
            break;
        case 3:
            TT_ASSERT(old_volume % N*C*H == 0);
            W = old_volume/(N*C*H);
            break;
        case -1: // In case where there is no negative value in ns
            TT_ASSERT(N*C*H*W == old_volume);
            break;
        default:
            TT_ASSERT(false && "Unexpected neg_idx in reshape!");
    }

    return {(uint32_t)N, (uint32_t)C, (uint32_t)H, (uint32_t)W};
}

  bool is_arch_gs(const tt::ARCH& arch) {
    return arch == tt::ARCH::GRAYSKULL;
  }

  bool is_arch_whb0(const tt::ARCH& arch) {
    return arch == tt::ARCH::WORMHOLE_B0;
  }

  bool is_cpu_tensor(const Tensor& tensor) {
      return tensor.storage_type() == StorageType::OWNED || tensor.storage_type() == StorageType::BORROWED;
  }

  bool is_device_tensor(const Tensor& tensor) { return tensor.storage_type() == StorageType::DEVICE; }

Tensor get_device_tensor(const Device* device, const Tensor& multi_device_tensor) {
    const auto& tensor_storage = std::get<MultiDeviceStorage>(multi_device_tensor.get_storage());
    for (const auto& device_buffer : tensor_storage.buffers) {
        if (device_buffer->device() == device) {
            return Tensor{
                DeviceStorage{device_buffer},
                multi_device_tensor.get_legacy_shape(),
                multi_device_tensor.get_dtype(),
                multi_device_tensor.get_layout()
            };
        }
    }
    TT_THROW("Device not found in multi-device tensor");
}

std::vector<Device*> get_devices(const Tensor& tensor) {
    std::vector<Device*> devices;
    if (tensor.storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE) {
        const auto& tensor_storage = std::get<tt::tt_metal::MultiDeviceStorage>(tensor.get_storage());
        for (int i = 0; i < tensor_storage.buffers.size(); ++i) {
            devices.push_back(tensor_storage.buffers[i]->device());
        }
        return devices;
    } else {
        TT_THROW("Tensor is not a multi-device tensor");
    }
}

}

}
