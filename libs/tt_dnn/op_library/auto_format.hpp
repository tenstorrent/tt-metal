#pragma once
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

#include <optional>

using namespace tt::constants;

namespace tt {

namespace tt_metal {

class AutoFormat {
    private:
        inline static Device* device = nullptr;
        constexpr static MemoryConfig default_mem_config = {.interleaved = true};

        AutoFormat() {}
    public:
        static void SetDefaultDevice(Device * dev) { device = dev; }
        static Device * GetDefaultDevice() { return device; }


        static Shape pad_to_tile_shape(const Shape& unpadded_shape, bool pad_c=false, bool pad_n=false, bool pad_h=true, bool pad_w=true) {
            auto n = pad_n ? roundup(unpadded_shape[0], TILE_HEIGHT) : unpadded_shape[0];
            auto c = pad_c ? roundup(unpadded_shape[1], TILE_WIDTH) : unpadded_shape[1];
            auto h = pad_h ? roundup(unpadded_shape[2], TILE_HEIGHT) : unpadded_shape[2];
            auto w = pad_w ? roundup(unpadded_shape[3], TILE_WIDTH) : unpadded_shape[3];
            Shape padded_shape = {n, c, h, w};
            return padded_shape;
        }

        static bool check_input_tensor_format(const Tensor &a, const Shape& shape, Layout target_layout = Layout::TILE) {
            if (a.layout() == target_layout && a.shape() == shape && a.storage_type() == StorageType::DEVICE) {
                return true;
            }
            return false;
        }

        static Tensor move_tensor_to_device(const Tensor &input, Device * device, const std::optional<MemoryConfig>& mem_config = std::nullopt);

        static Tensor format_input_tensor(const Tensor &input, Device * device, const Shape& padded_shape, float pad_value=0, Layout target_layout = Layout::TILE);

        static Tensor format_output_tensor(const Tensor &output, const Shape& shape, Device* device, Layout target_layout = Layout::TILE);
};


}
}
