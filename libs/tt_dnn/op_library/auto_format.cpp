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

    Tensor formatted_input = a;

    // TODO: Profile if it is faster to put host tensor to device and then pad/convert if possible
    // Device side conversions
    if (formatted_input.storage_type() == StorageType::DEVICE) {
        if (convert_layout && !pad_input) {
            if (target_layout == Layout::TILE) {
                return tilize(formatted_input);
            } else if (target_layout == Layout::ROW_MAJOR && formatted_input.layout() == Layout::TILE) {
                return untilize(formatted_input);
            }
        } else if (!convert_layout && pad_input) {
            if (formatted_input.layout() == Layout::ROW_MAJOR || formatted_input.layout() == Layout::TILE) {
                return pad(formatted_input, padded_shape, {0, 0, 0, 0}, pad_value);
            }
        } else if (convert_layout && pad_input) {
            if (target_layout == Layout::TILE) {
                return tilize_with_val_padding(formatted_input, padded_shape, {0, 0, 0, 0}, pad_value);
            }  else if (formatted_input.layout() == Layout::TILE && target_layout == Layout::ROW_MAJOR) {
                formatted_input = untilize(formatted_input);
                return pad(formatted_input, padded_shape, {0, 0, 0, 0}, pad_value);
            }
        }
        // Fall back to host conversions
        auto host = GetHost();
        formatted_input = formatted_input.to(host);
        delete host;
    }

    // Host side conversions
    if (pad_input) {
        if (formatted_input.layout() != Layout::ROW_MAJOR) {
            formatted_input = formatted_input.to(Layout::ROW_MAJOR);
            convert_layout = formatted_input.layout() != target_layout;
        }
        formatted_input = formatted_input.pad(padded_shape, {0, 0, 0, 0}, pad_value);
    }

    if(convert_layout) {
        formatted_input = formatted_input.to(target_layout);
    }

    if (formatted_input.storage_type() == StorageType::HOST) {
        formatted_input = formatted_input.to(device);
    }

    return formatted_input;
}


Tensor AutoFormat::format_output_tensor(const Tensor &output, const std::array<uint32_t, 4>& shape, Device* device, Layout target_layout) {
    bool unpad_output = output.shape() != shape;
    bool convert_layout = output.layout() != target_layout;

    Tensor formatted_output = output;

    // Device side conversions
    if (formatted_output.storage_type() == StorageType::DEVICE) {
        if (!unpad_output && convert_layout) {
            // If target layout is tile but shape does not support tile, we don't do any conversions
            if (target_layout == Layout::TILE) {
                if (formatted_output.shape()[2] % TILE_HEIGHT == 0 && formatted_output.shape()[3] % TILE_WIDTH == 0) {
                    formatted_output = tilize(formatted_output);
                }
                return formatted_output;
            } else if (target_layout == Layout::ROW_MAJOR && formatted_output.layout() == Layout::TILE) {
                formatted_output = untilize(formatted_output);
                return formatted_output;
            }
        } else if (unpad_output && !convert_layout) {
            // Output can be unpadded and layout supports the shape
            if ((formatted_output.layout() == Layout::TILE && shape[2] % TILE_HEIGHT == 0 && shape[3] % TILE_WIDTH == 0) ||
                (formatted_output.layout() == Layout::ROW_MAJOR && shape[3] % 2 == 0)) {
                formatted_output = unpad(formatted_output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1});
                return formatted_output;
            // Output is tile but shape cannot be tile. We leave in RM
            } else if (formatted_output.layout() == Layout::TILE && shape[3] % 2 == 0) {
                formatted_output = untilize_with_unpadding(formatted_output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1});
                return formatted_output;
            }
        } else if (unpad_output && convert_layout) {
            if (formatted_output.layout() == Layout::TILE && target_layout == Layout::ROW_MAJOR && shape[3] % 2 == 0) {
                formatted_output = untilize_with_unpadding(formatted_output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1});
                return formatted_output;
            } else if (formatted_output.layout() == Layout::ROW_MAJOR && target_layout == Layout::TILE && shape[2] % TILE_HEIGHT == 0 && shape[3] % TILE_WIDTH == 0) {
                formatted_output = unpad(formatted_output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1});
                formatted_output = tilize(formatted_output);
                return formatted_output;
            }
        }
        // Fall back to host conversions
        auto host = GetHost();
        formatted_output = formatted_output.to(host);
        delete host;
    }

    // Host side conversions
    if (unpad_output) {
        // Requires RM for unpad
        if (formatted_output.layout() != Layout::ROW_MAJOR) {
            formatted_output = formatted_output.to(Layout::ROW_MAJOR);
            convert_layout = formatted_output.layout() != target_layout;
        }
        formatted_output = formatted_output.unpad({0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1});
    }

    if (convert_layout) {
        // Default to RM layout if we can't match the formatted_input layout
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
    // Check that shape is supported on device
    if (formatted_output.storage_type() == StorageType::HOST) {
        if ((formatted_output.layout() == Layout::ROW_MAJOR && formatted_output.shape()[3] % 2 == 0) ||
            (formatted_output.layout() == Layout::CHANNELS_LAST && formatted_output.shape()[1] % 2 == 0) ||
            (formatted_output.layout() == Layout::TILE)) {
            formatted_output = formatted_output.to(device);
        }
    }

    return formatted_output;
}

}
}
