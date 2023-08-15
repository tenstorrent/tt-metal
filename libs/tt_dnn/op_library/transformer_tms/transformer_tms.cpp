#include "tt_dnn/op_library/transformer_tms/transformer_tms.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

namespace tt {
namespace operations {
namespace primary {
namespace transformers {

void SplitFusedQKVAndSplitHeads::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto batch_size = input_tensor.shape()[0];
    // TODO: See issue #1744
    TT_ASSERT(batch_size >= 7 && batch_size <= 9, "Input batch size must be between 2 to 9 for bert large TM ops!");

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 || input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");

    TT_ASSERT((input_tensor.shape() == Shape({batch_size, 1, 384, 3072})), "Unsupported input shape");
}

std::vector<Shape> SplitFusedQKVAndSplitHeads::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto batch_size = input_tensor.shape()[0];
    return {Shape{batch_size, 16, 384, 64}, Shape{batch_size, 16, 64, 384}, Shape{batch_size, 16, 384, 64}};
}

std::vector<Tensor> SplitFusedQKVAndSplitHeads::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks SplitFusedQKVAndSplitHeads::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto device_compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    TT_ASSERT((this->compute_with_storage_grid_size.x <= device_compute_with_storage_grid_size.x && this->compute_with_storage_grid_size.y <= device_compute_with_storage_grid_size.y), "Unsupported grid shape");


    return multi_core_split_fused_qkv_and_split_heads(input_tensor, output_tensors, this->compute_with_storage_grid_size);
}

tt::stl::reflection::Attributes SplitFusedQKVAndSplitHeads::attributes() const {
    return {
        {"compute_with_storage_grid_size", this->compute_with_storage_grid_size.str()},
        {"output_mem_config", this->output_mem_config},
    };
}

void ConcatenateHeads::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto batch_size = input_tensor.shape()[0];
    // TODO: See issue #1744
    TT_ASSERT(batch_size >= 7 && batch_size <= 9, "Input batch size must be between 2 to 9 for bert large TM ops!");

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 || input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");

    TT_ASSERT((input_tensor.shape() == Shape({batch_size, 16, 384, 64})), "Unsupported input shape");
}

std::vector<Shape> ConcatenateHeads::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto batch_size = input_tensor.shape()[0];
    return {Shape{batch_size, 1, 384, 1024}};
}

std::vector<Tensor> ConcatenateHeads::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks ConcatenateHeads::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    const auto batch_size = input_tensor.shape()[0];

    auto device_compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    TT_ASSERT((this->compute_with_storage_grid_size.x <= device_compute_with_storage_grid_size.x && this->compute_with_storage_grid_size.y <= device_compute_with_storage_grid_size.y), "Unsupported grid shape");

    return multi_core_concat_heads(input_tensor, output_tensor, this->compute_with_storage_grid_size);
}

tt::stl::reflection::Attributes ConcatenateHeads::attributes() const {
    return {
        {"compute_with_storage_grid_size", this->compute_with_storage_grid_size.str()},
        {"output_mem_config", this->output_mem_config},
    };
}

}  // namespace transformers
}  // namespace primary
}  // namespace operations
}  // namespace tt
