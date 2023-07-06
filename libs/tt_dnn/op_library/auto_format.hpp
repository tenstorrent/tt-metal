#pragma once
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

// TODO: To be merged into Op Base Class
class AutoFormat {
    private:
        inline static Device* device = nullptr;
        AutoFormat() {}
    public:
        static void SetDefaultDevice(Device * dev) { device = dev; }
        static Device * GetDefaultDevice() { return device; }


        static std::array<uint32_t, 4> pad_to_tile_shape(const std::array<uint32_t, 4>& unpadded_shape, bool pad_c=false, bool pad_n=false) {
            auto n = pad_n ? roundup(unpadded_shape[0], TILE_HEIGHT) : unpadded_shape[0];
            auto c = pad_c ? roundup(unpadded_shape[1], TILE_HEIGHT) : unpadded_shape[1];
            auto h = roundup(unpadded_shape[2], TILE_HEIGHT);
            auto w = roundup(unpadded_shape[3], TILE_WIDTH);
            std::array<uint32_t, 4> padded_shape = {n, c, h, w};
            return padded_shape;
        }

        static bool check_input_tensor_format(const Tensor &a, const std::array<uint32_t, 4>& shape, Layout target_layout = Layout::TILE) {
            if (a.layout() == target_layout && a.shape() == shape && a.storage_type() == StorageType::DEVICE) {
                return true;
            }
            return false;
        }

        static Tensor format_input_tensor(const Tensor &a, Device * device, const std::array<uint32_t, 4>& padded_shape, float pad_value=0, Layout target_layout = Layout::TILE);

        static Tensor format_output_tensor(const Tensor &output, const std::array<uint32_t, 4>& shape, Device* device, Layout target_layout = Layout::TILE);
};


}
}
