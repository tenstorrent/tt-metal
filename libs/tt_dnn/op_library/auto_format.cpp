#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/tilize/tilize_op.hpp"
#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/pad/pad_op.hpp"
#include "tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_dnn/op_library/layout_conversion/layout_conversion_op.hpp"
#include "tt_dnn/op_library/data_transfer/data_transfer_op.hpp"
#include "tt_dnn/op_library/transpose/transpose_op.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

Tensor AutoFormat::move_tensor_to_device(const Tensor &input, Device * device, const std::optional<MemoryConfig>& mem_config) {
    if (input.storage_type() == StorageType::OWNED) {
        return data_transfer_to_device(input, device, mem_config.has_value() ? mem_config.value() : default_mem_config);
    } else {
        return input;
    }
}

Tensor convert_from_channels_last_tensor_on_device(Tensor channels_last_tensor, Layout target_layout, bool pad_t = false, Shape padded_shape = {}, float pad_value = 0) {
    TT_ASSERT(target_layout == Layout::ROW_MAJOR || target_layout == Layout::TILE);
    TT_ASSERT(channels_last_tensor.layout() == Layout::CHANNELS_LAST);
    // need to interpret channels last tensor as rm tensor to call transpose op
    Shape cl_as_rm_shape = {channels_last_tensor.shape()[0], channels_last_tensor.shape()[2], channels_last_tensor.shape()[3], channels_last_tensor.shape()[1]};
    auto from_cl_tensor = Tensor(channels_last_tensor.device_storage().value(), cl_as_rm_shape, channels_last_tensor.dtype(), Layout::ROW_MAJOR);
    if(pad_t) {
        Shape padded_shape_cl_as_rm = {padded_shape[0], padded_shape[2], padded_shape[3], padded_shape[1]};
        from_cl_tensor = pad(from_cl_tensor, padded_shape_cl_as_rm, {0, 0, 0, 0}, pad_value);
    }
    TT_ASSERT(from_cl_tensor.shape()[3]%2 == 0 && from_cl_tensor.shape()[2]%2== 0);
    auto transpose_1_output = transpose_wh(from_cl_tensor);
    TT_ASSERT(transpose_1_output.shape()[3]%2 == 0);
    auto transpose_2_output = transpose_hc(transpose_1_output);
    TT_ASSERT(transpose_2_output.layout() == Layout::TILE);

    if(transpose_2_output.layout() == Layout::ROW_MAJOR && target_layout == Layout::TILE) {
        if(pad_t) {
            TT_ASSERT(transpose_2_output.shape() == padded_shape);
        }
        transpose_2_output = tilize(transpose_2_output);
    }
    if(transpose_2_output.layout() == Layout::TILE && target_layout == Layout::ROW_MAJOR) {
        TT_ASSERT(!pad_t);
        transpose_2_output = untilize(transpose_2_output);
    }
    TT_ASSERT(transpose_2_output.layout() == target_layout);
    if(!pad_t) {
        TT_ASSERT(transpose_2_output.shape() == channels_last_tensor.shape());
    }
    else {
        TT_ASSERT(transpose_2_output.shape() == padded_shape);
    }
    return transpose_2_output;
}

Tensor convert_to_channels_last_tensor_on_device(Tensor row_major_or_tile_tensor) {
    TT_ASSERT(row_major_or_tile_tensor.layout() == Layout::ROW_MAJOR || row_major_or_tile_tensor.layout() == Layout::TILE);
    auto transpose_1_output = transpose_hc(row_major_or_tile_tensor);
    TT_ASSERT(transpose_1_output.storage_type() == StorageType::DEVICE);
    auto transpose_2_output = transpose_wh(transpose_1_output);
    TT_ASSERT(transpose_2_output.storage_type() == StorageType::DEVICE);
    if(transpose_2_output.layout() == Layout::TILE) {
        transpose_2_output = untilize(transpose_2_output);
    }
    TT_ASSERT(transpose_2_output.storage_type() == StorageType::DEVICE);
    TT_ASSERT(transpose_2_output.layout() == Layout::ROW_MAJOR);
    TT_ASSERT(transpose_2_output.volume() == row_major_or_tile_tensor.volume());
    // re-interpret the final row major tensor as channels last tensor
    Tensor channels_last_tensor = Tensor(transpose_2_output.device_storage().value(), row_major_or_tile_tensor.shape(), row_major_or_tile_tensor.dtype(), Layout::CHANNELS_LAST);
    return channels_last_tensor;
}

Tensor AutoFormat::format_input_tensor(const Tensor &input, Device * device, const std::array<uint32_t, 4>& padded_shape, float pad_value, Layout target_layout) {
    bool pad_input = input.shape() != padded_shape;
    bool convert_layout = input.layout() != target_layout;

    if (!pad_input && !convert_layout) {
        return AutoFormat::move_tensor_to_device(input, device);
    }

    MemoryConfig mem_config = default_mem_config;
    if (input.storage_type() == StorageType::DEVICE) {
        mem_config = input.memory_config();
    }

    Tensor formatted_input = input;
    auto shape = formatted_input.shape();

    // TODO: Profile if it is faster to put host tensor to device and then pad/convert if possible
    // Device side conversions
    if (formatted_input.storage_type() == StorageType::DEVICE) {
        if (convert_layout && !pad_input) {
            if (target_layout == Layout::TILE && formatted_input.layout() == Layout::ROW_MAJOR) {
                return tilize(formatted_input, mem_config);
            } else if (target_layout == Layout::ROW_MAJOR && formatted_input.layout() == Layout::TILE) {
                return untilize(formatted_input, mem_config);
            } else if (target_layout == Layout::CHANNELS_LAST && formatted_input.layout() == Layout::ROW_MAJOR) {
                return convert_to_channels_last_tensor_on_device(formatted_input);
            } else if (target_layout == Layout::CHANNELS_LAST && formatted_input.layout() == Layout::TILE) {
                return convert_to_channels_last_tensor_on_device(formatted_input);
            }
            else if (target_layout == Layout::ROW_MAJOR &&  formatted_input.layout() == Layout::CHANNELS_LAST && shape[3] % 2 == 0) {
                return convert_from_channels_last_tensor_on_device(formatted_input, target_layout);
            }
            else if (target_layout == Layout::TILE &&  formatted_input.layout() == Layout::CHANNELS_LAST) {
                return convert_from_channels_last_tensor_on_device(formatted_input, target_layout);
            }
        } else if (!convert_layout && pad_input) {
            if (formatted_input.layout() == Layout::ROW_MAJOR || formatted_input.layout() == Layout::TILE || formatted_input.layout() == Layout::CHANNELS_LAST) {
                return pad(formatted_input, padded_shape, {0, 0, 0, 0}, pad_value, mem_config);
            }
        } else if (convert_layout && pad_input) {
            if (formatted_input.layout() == Layout::ROW_MAJOR && target_layout == Layout::TILE) {
                return tilize_with_val_padding(formatted_input, padded_shape, {0, 0, 0, 0}, pad_value, mem_config);
            }  else if (formatted_input.layout() == Layout::TILE && target_layout == Layout::ROW_MAJOR) {
                formatted_input = untilize(formatted_input, mem_config);
                return pad(formatted_input, padded_shape, {0, 0, 0, 0}, pad_value, mem_config);
            } else if (formatted_input.layout() == Layout::ROW_MAJOR && target_layout == Layout::CHANNELS_LAST) {
                Tensor channels_last_tensor = convert_to_channels_last_tensor_on_device(formatted_input);
                return pad(channels_last_tensor, padded_shape, {0, 0, 0, 0}, pad_value);
            } else if (formatted_input.layout() == Layout::TILE && target_layout == Layout::CHANNELS_LAST) {
                Tensor channels_last_tensor = convert_to_channels_last_tensor_on_device(formatted_input);
                return pad(channels_last_tensor, padded_shape, {0, 0, 0, 0}, pad_value);
            }
            else if (formatted_input.layout() == Layout::CHANNELS_LAST && ((target_layout == Layout::ROW_MAJOR && shape[3] % 2 == 0) || target_layout == Layout::TILE)) {
                return convert_from_channels_last_tensor_on_device(formatted_input, target_layout, pad_input, padded_shape, pad_value);
            }
        }
        // Fall back to host conversions
        auto host = GetHost();
        formatted_input = data_transfer_to_host(formatted_input, host);
        delete host;
    }

    // Host side conversions
    if (pad_input) {
        if (formatted_input.layout() != Layout::ROW_MAJOR) {
            formatted_input = layout_conversion_on_host(formatted_input, Layout::ROW_MAJOR);
            convert_layout = formatted_input.layout() != target_layout;
        }
        formatted_input = pad_on_host(formatted_input, padded_shape, {0, 0, 0, 0}, pad_value);
    }

    if(convert_layout) {
        formatted_input = layout_conversion_on_host(formatted_input, target_layout);
    }

    return AutoFormat::move_tensor_to_device(formatted_input, device, mem_config);
}


Tensor AutoFormat::format_output_tensor(const Tensor &output, const std::array<uint32_t, 4>& shape, Device* device, Layout target_layout) {
    bool unpad_output = output.shape() != shape;
    bool convert_layout = output.layout() != target_layout;

    if (!unpad_output && !convert_layout) {
        return output;
    }
    MemoryConfig mem_config = default_mem_config;
    if (output.storage_type() == StorageType::DEVICE) {
        mem_config = output.memory_config();
    }

    Tensor formatted_output = output;
    // Device side conversions
    if (formatted_output.storage_type() == StorageType::DEVICE) {
        if (!unpad_output && convert_layout) {
            // If target layout is tile but shape does not support tile, we don't do any conversions
            if (target_layout == Layout::TILE && formatted_output.layout() == Layout::ROW_MAJOR) {
                if (formatted_output.shape()[2] % TILE_HEIGHT == 0 && formatted_output.shape()[3] % TILE_WIDTH == 0) {
                    formatted_output = tilize(formatted_output, mem_config);
                }
                return formatted_output;
            } else if (target_layout == Layout::ROW_MAJOR && formatted_output.layout() == Layout::TILE) {
                formatted_output = untilize(formatted_output, mem_config);
                return formatted_output;
            } else if ((target_layout == Layout::ROW_MAJOR || target_layout == Layout::TILE) && formatted_output.layout() == Layout::CHANNELS_LAST) {
                return convert_from_channels_last_tensor_on_device(formatted_output, target_layout);
            } else if (target_layout == Layout::CHANNELS_LAST && (formatted_output.layout() == Layout::ROW_MAJOR || formatted_output.layout() == Layout::TILE)) {
                return convert_to_channels_last_tensor_on_device(formatted_output);
            }

        } else if (unpad_output && !convert_layout) {
            // Output can be unpadded and layout supports the shape
            if ((formatted_output.layout() == Layout::TILE && shape[2] % TILE_HEIGHT == 0 && shape[3] % TILE_WIDTH == 0) ||
                (formatted_output.layout() == Layout::ROW_MAJOR && shape[3] % 2 == 0)) {
                formatted_output = unpad(formatted_output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1}, mem_config);
                return formatted_output;
            // Output is tile but shape cannot be tile. We leave in RM
            } else if (formatted_output.layout() == Layout::TILE && shape[3] % 2 == 0) {
                formatted_output = untilize_with_unpadding(formatted_output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1}, mem_config);
                return formatted_output;
            }
        } else if (unpad_output && convert_layout) {
            if (formatted_output.layout() == Layout::TILE && target_layout == Layout::ROW_MAJOR && shape[3] % 2 == 0) {
                formatted_output = untilize_with_unpadding(formatted_output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1}, mem_config);
                return formatted_output;
            } else if (formatted_output.layout() == Layout::ROW_MAJOR && target_layout == Layout::TILE && shape[2] % TILE_HEIGHT == 0 && shape[3] % TILE_WIDTH == 0) {
                formatted_output = unpad(formatted_output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1}, mem_config);
                formatted_output = tilize(formatted_output, mem_config);
                return formatted_output;
            } else if (formatted_output.layout() == Layout::CHANNELS_LAST && ((target_layout == Layout::TILE && shape[2] % TILE_HEIGHT == 0 && shape[3] % TILE_WIDTH == 0) || (target_layout == Layout::ROW_MAJOR && shape[3] % 2 == 0)) ) {
                // need to interpret channels last tensor as rm tensor to call transpose op
                auto tiled_tensor = convert_from_channels_last_tensor_on_device(formatted_output, target_layout);
                auto unpadded_output = unpad(tiled_tensor, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1});
                return unpadded_output;
            }
        }
        // Fall back to host conversions
        auto host = GetHost();
        formatted_output = data_transfer_to_host(formatted_output, host);
        delete host;
    }

    // Host side conversions
    if (unpad_output) {
        // Requires RM for unpad
        if (formatted_output.layout() != Layout::ROW_MAJOR) {
            formatted_output = layout_conversion_on_host(formatted_output, Layout::ROW_MAJOR);
            convert_layout = formatted_output.layout() != target_layout;
        }
        formatted_output = unpad_on_host(formatted_output, {0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1});
    }

    if (convert_layout) {
        // Default to RM layout if we can't match the formatted_input layout
        if (target_layout == Layout::TILE && (formatted_output.shape()[2] % TILE_HEIGHT != 0 || formatted_output.shape()[3] % TILE_WIDTH != 0)) {
            if (formatted_output.layout() != Layout::ROW_MAJOR) {
                formatted_output = layout_conversion_on_host(formatted_output, Layout::ROW_MAJOR);
            }
        // We do not support CL <-> TILE conversions
        } else if (target_layout == Layout::CHANNELS_LAST && formatted_output.layout() == Layout::TILE ||
                   target_layout == Layout::TILE && formatted_output.layout() == Layout::CHANNELS_LAST) {
            // No-Op, leave formatted_output in CL
        } else {
            formatted_output = layout_conversion_on_host(formatted_output, target_layout);
        }
    }

    // Send formatted_output to device if possible
    // Check that shape is supported on device
    if (formatted_output.storage_type() == StorageType::OWNED) {
        if ((formatted_output.layout() == Layout::ROW_MAJOR && formatted_output.shape()[3] % 2 == 0) ||
            (formatted_output.layout() == Layout::CHANNELS_LAST && formatted_output.shape()[1] % 2 == 0) ||
            (formatted_output.layout() == Layout::TILE)) {
            formatted_output = AutoFormat::move_tensor_to_device(formatted_output, device, mem_config);
        }
    }

    return formatted_output;
}

}
}
