#include "tt_dnn/op_library/bert_large_tms/bert_large_tms.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

namespace tt {

namespace tt_metal {

void BertLargeTM::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    switch (this->bert_large_tm_op_type) {
        case BertLargeTMOpType::CREATE_QKV_HEADS:
            TT_ASSERT((input_tensor.shape() == Shape({9, 1, 384, 3072})), "Unsupported input shape");
            break;
        case BertLargeTMOpType::SPLIT_FUSED_QKV:
            TT_ASSERT((input_tensor.shape() == Shape({9, 1, 384, 3072})), "Unsupported input shape");
            break;
        case BertLargeTMOpType::CREATE_Q_HEAD:
        case BertLargeTMOpType::CREATE_K_HEAD:
        case BertLargeTMOpType::CREATE_V_HEAD:
            TT_ASSERT((input_tensor.shape() == Shape({9, 1, 384, 1024})), "Unsupported input shape");
            break;
        case BertLargeTMOpType::CONCAT_HEADS:
            TT_ASSERT((input_tensor.shape() == Shape({9, 16, 384, 64})), "Unsupported input shape");
            break;
        default:
            TT_ASSERT(false, "Unknown bert large tm op in validate!");
    }
}

std::vector<Shape> BertLargeTM::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    std::vector<Shape> output_shape_vec;
    switch (this->bert_large_tm_op_type) {
        case BertLargeTMOpType::CREATE_QKV_HEADS:
            output_shape_vec = {(Shape) {9, 16, 384, 64}, (Shape) {9, 16, 64, 384}, (Shape) {9, 16, 384, 64}};
            break;
        case BertLargeTMOpType::SPLIT_FUSED_QKV:
            output_shape_vec = {(Shape) {9, 1, 384, 1024}, (Shape) {9, 1, 384, 1024}, (Shape) {9, 1, 384, 1024}};
            break;
        case BertLargeTMOpType::CREATE_Q_HEAD:
        case BertLargeTMOpType::CREATE_V_HEAD:
            output_shape_vec = {(Shape) {9, 16, 384, 64}};
            break;
        case BertLargeTMOpType::CREATE_K_HEAD:
            output_shape_vec = {(Shape) {9, 16, 64, 384}};
            break;
        case BertLargeTMOpType::CONCAT_HEADS:
            output_shape_vec = {(Shape) {9, 1, 384, 1024}};
            break;
        default:
            TT_ASSERT(false, "Unknown bert large tm op in compute_output_shapes!");
    }
    return output_shape_vec;
}

std::vector<Tensor> BertLargeTM::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return operation::generic_create_output_tensors(*this, input_tensors, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks BertLargeTM::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto device_compute_and_storage_grid_size = input_tensor.device()->compute_and_storage_grid_size();
    CoreCoord compute_and_storage_grid_size = {12, 9};
    TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");

    switch (this->bert_large_tm_op_type) {
        case BertLargeTMOpType::CREATE_QKV_HEADS:
            return  multi_core_create_qkv_heads_from_fused_qkv(input_tensor, output_tensors, compute_and_storage_grid_size);
        case BertLargeTMOpType::SPLIT_FUSED_QKV:
            return  multi_core_split_fused_qkv(input_tensor, output_tensors, compute_and_storage_grid_size);
        // Q and V heads use transpose_hw=false, while K head requires the additional transpose with transpose_hw=true.
        case BertLargeTMOpType::CREATE_Q_HEAD:
        case BertLargeTMOpType::CREATE_V_HEAD:
            return multi_core_create_qkv_heads(input_tensor, output_tensor, compute_and_storage_grid_size, /*transpose_hw=*/false);
        case BertLargeTMOpType::CREATE_K_HEAD:
            return multi_core_create_qkv_heads(input_tensor, output_tensor, compute_and_storage_grid_size, /*transpose_hw=*/true);
        case BertLargeTMOpType::CONCAT_HEADS:
            return multi_core_concat_heads(input_tensor, output_tensor, compute_and_storage_grid_size);
        default:
            TT_ASSERT(false, "Unknown bert large tm op in create_program!");
    }
    return {};
}

operation::Hash BertLargeTM::compute_program_hash(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    return fmt::format(
        "bert_large_tm_{}_{}_{}",
         magic_enum::enum_name(this->bert_large_tm_op_type),
         operation::hash_memory_config(this->output_mem_config),
         operation::hash_tensor(input_tensor)
    );
}

std::ostream& operator<<(std::ostream& os, const BertLargeTM& op) {
    os << boost::core::demangle(typeid(op).name());
    os << "{";
    os << ".bert_large_tm_op_type=" << magic_enum::enum_name(op.bert_large_tm_op_type);
    // TODO(arakhmati): add output_mem_config
    os << "}";
    return os;
}

} // namespace tt_metal

} // namespace tt
