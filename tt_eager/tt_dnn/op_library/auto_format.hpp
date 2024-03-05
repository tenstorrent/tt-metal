// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_dnn/op_library/operation.hpp"

#include <optional>

#include "tt_metal/common/math.hpp"
using namespace tt::constants;

namespace tt {

namespace tt_metal {

struct FormatParams {
    Shape pad_shape;
    float pad_value;
    Layout target_layout;
};

class AutoFormat {
    private:
        inline static Device* device = nullptr;

        AutoFormat() {}
    public:
        static void SetDefaultDevice(Device * dev) { device = dev; }
        static Device * GetDefaultDevice() { return device; }


        static Shape pad_to_tile_shape(const Shape& unpadded_shape, bool pad_c=false, bool pad_n=false, bool pad_h=true, bool pad_w=true) {
            auto n = pad_n ? round_up(unpadded_shape[0], TILE_HEIGHT) : unpadded_shape[0];
            auto c = pad_c ? round_up(unpadded_shape[1], TILE_WIDTH) : unpadded_shape[1];
            auto h = pad_h ? round_up(unpadded_shape[2], TILE_HEIGHT) : unpadded_shape[2];
            auto w = pad_w ? round_up(unpadded_shape[3], TILE_WIDTH) : unpadded_shape[3];
            Shape padded_shape = {n, c, h, w};
            return padded_shape;
        }

        static Shape pad_to_rm_shape(const Shape& unpadded_shape) {
            Shape padded_shape = unpadded_shape;
            padded_shape[3] = round_up(unpadded_shape[3], 2);
            return padded_shape;
        }

        static Shape pad_to_legal_shape(const Shape& unpadded_shape, Layout layout) {
            Shape padded_shape = unpadded_shape;
            switch (layout) {
                case Layout::ROW_MAJOR: padded_shape = pad_to_rm_shape(unpadded_shape); break;
                case Layout::TILE: padded_shape = pad_to_tile_shape(unpadded_shape);
                default: break;
            }
            return padded_shape;
        }

        // TODO: These legal checks should probably be somewhere else like tensor class, since it is common logic not just for autoformat
        static bool legal_tile_shape(const Shape& shape) {
            return (shape[2] % TILE_HEIGHT == 0 && shape[3] % TILE_WIDTH == 0);
        }

        static bool legal_rm_shape(const Shape& shape) {
            return (shape[3] % 2 == 0);
        }

        static bool legal_device_shape(const Shape& shape, Layout layout) {
            switch (layout) {
                case Layout::ROW_MAJOR: return legal_rm_shape(shape);
                case Layout::TILE: return legal_tile_shape(shape);
                default: return true;
            }
        }


        static bool check_input_tensor_format(const Tensor &a, const Shape& shape, Layout target_layout = Layout::TILE) {
            if (a.get_layout() == target_layout && a.get_legacy_shape() == shape && a.storage_type() == StorageType::DEVICE) {
                return true;
            }
            return false;
        }

        static Tensor move_tensor_to_device(const Tensor &input, Device * device, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

        static Tensor move_tensor_to_mem_config(const Tensor &input, const MemoryConfig& mem_config);

        static Tensor format_input_tensor(const Tensor &input, Device * device, const Shape& padded_shape, float pad_value, Layout target_layout, std::optional<MemoryConfig> target_mem_config = std::nullopt);

        static Tensor format_output_tensor(const Tensor &output, const Shape& shape, Device* device, Layout target_layout, std::optional<MemoryConfig> target_mem_config = std::nullopt);
};


}
}
