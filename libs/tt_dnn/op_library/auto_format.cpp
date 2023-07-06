#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/tilize/tilize_op.hpp"
#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/pad/pad_op.hpp"
#include "tt_dnn/op_library/unpad/unpad_op.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

// This function always returns a new tensor
Tensor AutoFormat::format_input_tensor(const Tensor &a, Device * device, const std::array<uint32_t, 4>& padded_shape, float pad_value, Layout target_layout) {
    bool pad_input = a.shape() != padded_shape;
    bool convert_layout = a.layout() != target_layout;
    if (!convert_layout && !pad_input) {
        if (a.storage_type() == StorageType::HOST) {
            return a.to(device);
        } else {
            // Should never hit this to avoid unnecessary copies
            log_warning("WARNING: Calling auto formatting on input tensor already in correct format, resulting in extraneous copy");
            return a;
        }
    }
    // ON DEVICE PADDING/CONVERSIONS
    if (a.storage_type() == StorageType::DEVICE) {
        if (convert_layout && !pad_input) {
            if (target_layout == Layout::TILE) {
                return tilize(a);
            } else if (target_layout == Layout::ROW_MAJOR && a.layout() == Layout::TILE) {
                return untilize(a);
            }
        } else if (!convert_layout && pad_input) {
            if (a.layout() == Layout::ROW_MAJOR || a.layout() == Layout::TILE) {
                return pad(a, padded_shape, {0, 0, 0, 0}, pad_value);
            }
        } else if (convert_layout && pad_input) {
            if (a.layout() == Layout::ROW_MAJOR) {
                return tilize_with_val_padding(a, padded_shape, {0, 0, 0, 0}, pad_value);
            }
        }
        // Fall back to host if on device is unsupported
        auto host = GetHost();
        auto out = format_input_tensor(a.to(host), device, padded_shape, pad_value, target_layout);
        delete host;
        return out;
    } else {
        if (pad_input) {
            if (a.layout() != Layout::ROW_MAJOR) {
                auto input = a.to(Layout::ROW_MAJOR);
                input = input.pad(padded_shape, {0, 0, 0, 0}, pad_value);
                if(input.layout() != target_layout) {
                    input = input.to(target_layout);
                }
                input = input.to(device);
                return input;
            } else {
                auto input = a.pad(padded_shape, {0, 0, 0, 0}, pad_value);
                if (convert_layout) {
                    input = input.to(target_layout);
                }
                input = input.to(device);
                return input;
            }
        }
        if(convert_layout) {
            auto input = a.to(target_layout);
            input = input.to(device);
            return input;
        }
    }

    // Should never hit this to avoid unnecessary copies
    log_warning("WARNING: Calling auto formatting on input tensor already in correct format, resulting in extraneous copy");
    return a;
}


Tensor AutoFormat::format_output_tensor(const Tensor &output, const std::array<uint32_t, 4>& shape, Device* device, Layout target_layout) {
    bool unpad_output = output.shape() != shape;
    bool convert_layout = output.layout() != target_layout;

    Tensor formatted_output = output;
    // Noop
    if (!unpad_output && !convert_layout) {
        return formatted_output;
    }

    // ON DEVICE UNPADDING/CONVERSIONS
    if (formatted_output.storage_type() == StorageType::DEVICE) {
        if (!unpad_output && convert_layout) {
            // If target layout is tile but shape does not support tile, we don't do any conversions
            if (target_layout == Layout::TILE) {
                if (formatted_output.shape()[2] % TILE_HEIGHT == 0 && formatted_output.shape()[3] % TILE_WIDTH == 0) {
                    formatted_output = tilize(formatted_output);
                    return formatted_output;
                } else {
                    return formatted_output;
                }
            } else if (target_layout == Layout::ROW_MAJOR && formatted_output.layout() == Layout::TILE && formatted_output.shape()[2] % 2 == 0) {
                formatted_output = untilize(formatted_output);
                return formatted_output;
            }
        } else if (unpad_output && !convert_layout) {
            // Output can be unpadded and target layout supports the shape
            if ((target_layout == Layout::TILE && shape[2] % TILE_HEIGHT == 0 && shape[3] % TILE_WIDTH == 0) ||
                    target_layout == Layout::ROW_MAJOR && shape[3] % 2 == 0) {
                formatted_output = unpad(formatted_output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1});
                return formatted_output;
            // Output is tile but shape cannot be tile
            } else if (target_layout == Layout::TILE && shape[3] % 2 == 0) {
                formatted_output = untilize_with_unpadding(formatted_output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1});
                return formatted_output;
            }
        } else if (unpad_output && convert_layout) {
            if (target_layout == Layout::ROW_MAJOR && formatted_output.layout() == Layout::TILE && shape[3] % 2 == 0) {
                formatted_output = untilize_with_unpadding(formatted_output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1});
                return formatted_output;
            }
        }
    }

    // ON HOST UNPADDING/CONVERSIONS
    auto host = GetHost();
    // Unpad formatted_output if necessary
    if (unpad_output) {
        if (formatted_output.storage_type() == StorageType::DEVICE) {
            formatted_output = formatted_output.to(host);
        }
        // Requires RM for unpad
        if (formatted_output.layout() != Layout::ROW_MAJOR) {
            formatted_output = formatted_output.to(Layout::ROW_MAJOR);
        }
        formatted_output = formatted_output.unpad({0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1});
    }

    // Converts layout if necessary, result will always be on host
    if (target_layout != formatted_output.layout()) {
        if (formatted_output.storage_type() == StorageType::DEVICE) {
            formatted_output = formatted_output.to(host);
        }
        // Default to RM layout if we can't match the input layout
        if (target_layout == Layout::TILE && (formatted_output.shape()[2] % TILE_HEIGHT != 0 || formatted_output.shape()[3] % TILE_WIDTH != 0)) {
            if (formatted_output.layout() != Layout::ROW_MAJOR) {
                formatted_output = formatted_output.to(Layout::ROW_MAJOR);
            }
        // We do not support CL <-> TILE conversions
        } else if (target_layout == Layout::CHANNELS_LAST && formatted_output.layout() == Layout::TILE ||
                    target_layout == Layout::TILE && formatted_output.layout() == Layout::CHANNELS_LAST) {
            // No-Op, leave formatted_output in CL
        } else {
            formatted_output = formatted_output.to(target_layout);
        }
    }

    // Send formatted_output to device if possible
    if (formatted_output.storage_type() == StorageType::HOST) {
        // Check that shape is supported on device
        if ((formatted_output.layout() == Layout::ROW_MAJOR && formatted_output.shape()[3] % 2 == 0) ||
            (formatted_output.layout() == Layout::CHANNELS_LAST && formatted_output.shape()[1] % 2 == 0) ||
            (formatted_output.layout() == Layout::TILE && formatted_output.shape()[2] % TILE_HEIGHT == 0 && formatted_output.shape()[3] % TILE_WIDTH == 0)) {
            formatted_output = formatted_output.to(device);
        }
    }

    delete host;

    return formatted_output;
}

}
}
