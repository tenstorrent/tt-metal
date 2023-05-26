#pragma once
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"
#include "constants.hpp"
#include "tt_dnn/op_library/tilize/tilize_op.hpp"
#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/pad/pad_op.hpp"
#include "tt_dnn/op_library/unpad/unpad_op.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

// TODO: To be merged into Op Base Class
class AutoPad {
    private:
        inline static Device * device = nullptr;
        AutoPad() {}
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

        static bool check_input_tensor_format(const Tensor &a, const std::array<uint32_t, 4>& shape) {
            if (a.layout() == Layout::TILE && a.shape() == shape && !a.on_host()) {
                return true;
            }
            return false;
        }

        static Tensor format_input_tensor(const Tensor &a, Device * device, const std::array<uint32_t, 4>& padded_shape, float pad_value=0) {
            if (a.layout() != Layout::TILE || a.shape() != padded_shape) {
                // ON DEVICE PADDING/CONVERSIONS
                if (!a.on_host()) {
                    if (a.layout() == Layout::ROW_MAJOR && a.shape()[3] % 2 == 0) {
                        auto out = tilize_with_val_padding(a, padded_shape, {0, 0, 0, 0}, pad_value);
                        return out;
                    } else if (a.layout() == Layout::TILE) {
                        auto out = pad(a, padded_shape, {0, 0, 0, 0}, pad_value);
                        return out;
                    }
                }

                // ON HOST PADDING/CONVERSIONS
                auto host = GetHost();
                auto input = a.to(host);
                if (a.shape()!= padded_shape) {
                    if (a.layout() != Layout::ROW_MAJOR) {
                        input = input.to(Layout::ROW_MAJOR);
                    }
                    input = input.pad(padded_shape, {0, 0, 0, 0}, pad_value);
                }
                if(input.layout() != Layout::TILE) {
                    input = input.to(Layout::TILE);
                }
                input = input.to(device);

                delete host;
                return input;

            } else if (a.on_host()) {
                return a.to(device);
            } else {
                return a;
            }
        }

        static void format_output_tensor(const Tensor &a, Tensor &output, const std::array<uint32_t, 4>& shape, Device * device) {
            bool no_unpad = output.shape() == shape;
            // Hack env variable to leave outputs on device if no unpadding needed
            if (std::getenv("TT_LEAVE_TILE_OUTPUT_ON_DEVICE") != nullptr) {
                if (no_unpad && output.layout() == Layout::TILE) {
                    return;
                }
            }

            // ON DEVICE UNPADDING/CONVERSIONS
            if (!a.on_host()) {
                if (!no_unpad) {
                    if (a.layout() == Layout::TILE && output.layout() == Layout::TILE && shape[2] % TILE_HEIGHT == 0 && shape[3] % TILE_WIDTH == 0) {
                        output = unpad(output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1});
                        return;
                    } else if (shape[3] % 2 == 0 && ((a.layout() == Layout::ROW_MAJOR && output.layout() == Layout::TILE) || (a.layout() == Layout::TILE && output.layout() == Layout::TILE))) {
                        output = untilize_with_unpadding(output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1});
                        return;
                    }
                }
                if (a.layout() != output.layout()) {
                    if (a.layout() == Layout::TILE) {
                        // If we weren't able to unpad in the previous if, then we need to unpad on host, which is in RM so we don't tilize
                        if (no_unpad) {
                            if (output.shape()[2] % TILE_HEIGHT == 0 && output.shape()[3] % TILE_WIDTH == 0) {
                                output = tilize(output);
                            }
                            return;
                        }
                    } else if (a.layout() == Layout::ROW_MAJOR && output.layout() == Layout::TILE) {
                        output = untilize(output);
                        // If we weren't able to unpad in the previous if, then we need to unpad on host
                        if (no_unpad) {
                            return;
                        }
                    }
                }
            }

            // ON HOST UNPADDING/CONVERSIONS
            auto host = GetHost();
            // Unpad output if necessary
            if (!no_unpad) {

                output = output.to(host);

                // Requires RM for unpad
                if (output.layout() != Layout::ROW_MAJOR) {
                    output = output.to(Layout::ROW_MAJOR);
                }
                output = output.unpad({0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1});
            }

            // Converts layout if necessary, result will always be on host
            if (a.layout() != output.layout()) {
                if (!output.on_host()) {
                    output = output.to(host);
                }

                // Default to RM layout if we can't match the input layout
                if (a.layout() == Layout::TILE && !(output.shape()[2] % TILE_HEIGHT == 0 && output.shape()[3] % TILE_WIDTH == 0)) {
                    if (output.layout() != Layout::ROW_MAJOR) {
                        output = output.to(Layout::ROW_MAJOR);
                    }
                } else {
                    output = output.to(a.layout());
                }
            }

            // Send output to device if a was on device
            if (!a.on_host() && output.on_host()) {
                if (!((output.layout() == Layout::ROW_MAJOR && output.shape()[3] % 2 != 0) ||
                    (output.layout() == Layout::CHANNELS_LAST && output.shape()[1] % 2 != 0))) {
                        output = output.to(device);
                    }
            // Send output to host if a was on host
            } else if (a.on_host() && !output.on_host()) {
                output = output.to(host);
            }

            delete host;

        }
};


}
}
