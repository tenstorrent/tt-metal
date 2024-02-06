// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_dnn/op_library/transpose/transpose_op.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/reshape/reshape_op.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

#include <limits>

using namespace tt::constants;

namespace reduce_op_utils {

using namespace tt::tt_metal;

string dim_to_kernel_name(ReduceOpDim reduce_dim, ReduceOpMath reduce_op){
    string kernel_name;
    switch(reduce_dim){
        case ReduceOpDim::H: kernel_name = "tt_eager/tt_dnn/op_library/reduce/kernels/compute/reduce_h.cpp"; break;
        case ReduceOpDim::W: kernel_name = "tt_eager/tt_dnn/op_library/reduce/kernels/compute/reduce_w.cpp"; break;
        case ReduceOpDim::HW: kernel_name = "tt_eager/tt_dnn/op_library/reduce/kernels/compute/reduce_hw.cpp"; break;
        default: TT_FATAL(false && "Undefined dim");
    }
    return kernel_name;
}

std::map<string, string> get_defines(ReduceOpMath reduce_op, ReduceOpDim reduce_dim){
    std::map<string, string> defines;
    // TOOD(AP): need a sync with Reduce::Max from HLK headers
    bool do_max = reduce_op == ReduceOpMath::MAX;
    string reduce_dim_str;
    switch(reduce_dim) {
        case ReduceOpDim::W: reduce_dim_str = "ReduceDim::REDUCE_ROW"; break;
        case ReduceOpDim::H: reduce_dim_str = "ReduceDim::REDUCE_COL"; break;
        case ReduceOpDim::HW: reduce_dim_str = "ReduceDim::REDUCE_SCALAR"; break;
        default: TT_ASSERT(false && "Invalid reduce_op!");
    }
    defines["REDUCE_OP"] = (do_max ? "PoolType::MAX" : "PoolType::SUM" );
    defines["REDUCE_DIM"] = reduce_dim_str;
    return defines;
}

} // namespace reduce_op_utils
namespace tt {
namespace tt_metal {

void Reduce::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to reduce need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr , "Operands to reduce need to be allocated in buffers on device!");
    TT_FATAL((input_tensor.layout() == Layout::TILE), "Inputs to reduce must be tilized");
    if (this->dim == ReduceOpDim::H) {
        if (input_tensor.memory_config().is_sharded()) {
            TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED);
        } else {
            TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
        TT_FATAL(input_tensor.memory_config().memory_layout == this->output_mem_config.memory_layout);
    } else {
        TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
    }
}

std::vector<Shape> Reduce::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    auto output_shape = input_tensor.shape();
    auto padding = output_shape.padding();
    switch (this->dim){
        case ReduceOpDim::H:
            output_shape[2] = TILE_HEIGHT;
            padding[2] = Padding::PadDimension{0, 31};
            break;
        case ReduceOpDim::W:
            output_shape[3] = TILE_WIDTH;
            padding[3] = Padding::PadDimension{0, 31};
            break;
        case ReduceOpDim::HW:
            output_shape[2] = TILE_HEIGHT;
            output_shape[3] = TILE_WIDTH;
            padding[2] = Padding::PadDimension{0, 31};
            padding[3] = Padding::PadDimension{0, 31};
            break;

    }
    return {Shape(output_shape, padding)};
}

std::vector<Tensor> Reduce::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if(this->output_mem_config.is_sharded()){
        auto output_shape = this->compute_output_shapes(input_tensors).at(0);
        auto shard_spec = input_tensor.shard_spec().value();
        shard_spec.shape[0] = tt_metal::compute_volume(output_shape) / output_shape[-1];
        auto mem_config = this->output_mem_config;
        mem_config.shard_spec = shard_spec;
        return {create_sharded_device_tensor(output_shape, this->output_dtype, Layout::TILE, input_tensor.device(), mem_config)};
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Reduce::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);

    switch (parallelization_strategy){
        case ReduceOpParallelizationStrategy::MULTI_CORE_H:
            return {reduce_multi_core_h(input_tensor, output_tensor, this->math_op, this->dim, this->scaler)};
        case ReduceOpParallelizationStrategy::MULTI_CORE_W:
            return {reduce_multi_core_w(input_tensor, output_tensor, this->math_op, this->dim, this->scaler)};
        case ReduceOpParallelizationStrategy::SINGLE_CORE:
        default:
            return {reduce_single_core(input_tensor, output_tensor, this->math_op, this->dim, this->scaler)};
    }

}

ReduceOpParallelizationStrategy Reduce::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    uint32_t num_tiles = input_tensor.volume() / TILE_HW;
    auto shape = input_tensor.shape();
    uint32_t Wt = shape[3]/TILE_WIDTH;
    uint32_t Ht = shape[2]/TILE_HEIGHT;
    uint32_t NC = shape[1]*shape[0];
    if((NC * Wt > 1 || (input_tensor.storage_type() == StorageType::DEVICE && input_tensor.is_sharded())) and this->dim == ReduceOpDim::H){
        return ReduceOpParallelizationStrategy::MULTI_CORE_H;
    }else if(NC * Ht > 1 and this->dim == ReduceOpDim::W){
        return ReduceOpParallelizationStrategy::MULTI_CORE_W;
    }else if(num_tiles > 1 and this->dim == ReduceOpDim::HW){
        return ReduceOpParallelizationStrategy::MULTI_CORE_HW;
    }else{
        return ReduceOpParallelizationStrategy::SINGLE_CORE;
    }
}

//reduce min
//reduce min = - reduce_max( -x )
Tensor reduce_min(const Tensor &input_tensor, ReduceOpDim reduce_dim, float scaler = 1.0f, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    Tensor n_input_tensor = neg(input_tensor,output_mem_config);
    Tensor max_reduce = reduce(n_input_tensor,ReduceOpMath::MAX,reduce_dim,scaler,output_mem_config);
    Tensor min_tensor = neg(max_reduce,output_mem_config);
    return min_tensor;
}

Tensor reduce(const Tensor &input_tensor, ReduceOpMath reduce_math, ReduceOpDim reduce_dim, float scaler, const MemoryConfig& output_mem_config, const std::optional<DataType>& output_dtype) {
    if ( reduce_math == ReduceOpMath::MIN ) {
        return reduce_min(input_tensor,reduce_dim,scaler,output_mem_config);
    }

    auto parallelization_strategy = Reduce{reduce_math, reduce_dim, scaler, output_mem_config}.get_parallelization_strategy({input_tensor});
    auto is_multicore_hw = parallelization_strategy == ReduceOpParallelizationStrategy::MULTI_CORE_HW;
    float pad_value = reduce_math == ReduceOpMath::MAX ? -std::numeric_limits<float>::infinity() : 0;

    if (is_multicore_hw) {
        Device * device;

        // Get the device
        if (input_tensor.storage_type() != StorageType::DEVICE) {
            device = AutoFormat::GetDefaultDevice();
            TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
        } else {
            device = input_tensor.device();
        }
        auto input_tensor_pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.shape());
        auto formatted_input_tensor = input_tensor;
        if (!AutoFormat::check_input_tensor_format(input_tensor, input_tensor_pad_shape)) {
            formatted_input_tensor = AutoFormat::format_input_tensor(input_tensor, device, input_tensor_pad_shape, pad_value, Layout::TILE);
        }
        const Tensor output_tensor = operation::run_without_autoformat(Reduce{reduce_math, ReduceOpDim::W, 1.0, output_mem_config, output_dtype.value_or(input_tensor.dtype())}, {formatted_input_tensor}).at(0);
        return operation::run_without_autoformat(Reduce{reduce_math, ReduceOpDim::H, scaler, output_mem_config}, {output_tensor}).at(0);
    } else {
        return operation::run_with_autoformat(Reduce{reduce_math, reduce_dim, scaler, output_mem_config, output_dtype.value_or(input_tensor.dtype())}, {input_tensor}, {}, pad_value).at(0);
    }
}
Tensor mean_hw(const Tensor& input_tensor, const MemoryConfig& output_mem_config) {
    return mean(input_tensor,2,output_mem_config);
}
Tensor global_mean(const Tensor& input_tensor, const MemoryConfig& output_mem_config) {
    float inv_volume = 1.0f/input_tensor.volume();
    return mul_unary_sfpu( inv_volume, global_sum(input_tensor,output_mem_config), output_mem_config);
}
Tensor mean(const Tensor& input_tensor,uint aggregate_dims /* = 2 */, const MemoryConfig& output_mem_config) {
    tt::tt_metal::Shape shape = input_tensor.shape();

    TT_FATAL( aggregate_dims >= 2 && aggregate_dims <= 4, "mean aggregate dimensions should be [HW],[CHW] or [NCHW]");
    switch( aggregate_dims ) {
        case 4:
        case 3:
            {
                Tensor result = mean(reshape(input_tensor,1,1,shape[0]*shape[1]*shape[2],shape[3],output_mem_config),2,output_mem_config);
                return reshape(result,shape[0],shape[1],shape[2],shape[3],output_mem_config);
            }
        default:
            break;
    }

    float inv_scale_hw = 1.0f/(shape[3]*shape[2]);
    Tensor scaled_sum_hw = reduce(input_tensor,ReduceOpMath::SUM,ReduceOpDim::HW,inv_scale_hw,output_mem_config);
    return scaled_sum_hw;
}

template <ReduceOpMath OpKind>
Tensor reduce_on_dim(const Tensor &input_tensor, uint dim, const MemoryConfig& output_mem_config) {
    TT_FATAL( dim >= 0 && dim <= 3, "dimension have to be 0-3 only corresponding to N,C,H,W");
    constexpr float scaler1 = 1.0;

    if ( dim == 3 ) {
        return reduce(input_tensor, OpKind, ReduceOpDim::W, scaler1, output_mem_config);
    } else if ( dim == 2 ) {
        return reduce(input_tensor, OpKind, ReduceOpDim::H, scaler1, output_mem_config);
    }

    // Other sum dims will autoformat first before doing composite operations
    Device * device;

    // Get the device
    if (input_tensor.storage_type() != StorageType::DEVICE) {
        device = AutoFormat::GetDefaultDevice();
        TT_FATAL(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = input_tensor.device();
    }

    // @tt-aho TODO: Profile/determine which is more performant
    // reduce sum on h
    // 1 extra transpose wh and reduce sum on w
    // Fused tranpose cw and reduce sum on w
    if ( dim == 1 ) {
        // Pad before running the op to only pay cost of formatting once
        auto input_tensor_pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.shape(), true);
        auto out_shape = input_tensor.shape();
        out_shape[1] = 1;

        auto formatted_input_tensor = input_tensor;
        float pad_value = (OpKind == ReduceOpMath::MAX) ? -std::numeric_limits<float>::infinity() : (OpKind == ReduceOpMath::MIN) ? std::numeric_limits<float>::infinity() : 0;

        if (!AutoFormat::check_input_tensor_format(input_tensor, input_tensor_pad_shape)) {
            formatted_input_tensor = AutoFormat::format_input_tensor(input_tensor, device, input_tensor_pad_shape, pad_value, Layout::TILE);
        }
        Tensor output = transpose(formatted_input_tensor, 1, -2, output_mem_config);
        output = reduce_on_dim<OpKind>(output, 2, output_mem_config);
        output = transpose(output, 1, -2, output_mem_config);
        return AutoFormat::format_output_tensor(output, out_shape, device, Layout::TILE);
    } else {
        // Pad before running the op to only pay cost of formatting once
        auto input_tensor_pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.shape(), false, true);
        auto out_shape = input_tensor.shape();
        out_shape[0] = 1;

        auto formatted_input_tensor = input_tensor;
        if (!AutoFormat::check_input_tensor_format(input_tensor, input_tensor_pad_shape)) {
            formatted_input_tensor = AutoFormat::format_input_tensor(input_tensor, device, input_tensor_pad_shape, 0.0, Layout::TILE);
        }
        Tensor output = transpose(input_tensor, 0, -2, output_mem_config);
        output = reduce_on_dim<OpKind>(output, 2, output_mem_config);
        output = transpose(output, 0, -2, output_mem_config);
        return AutoFormat::format_output_tensor(output, out_shape, device, Layout::TILE);
    }
}


Tensor sum(const Tensor &input_tensor, uint dim, const MemoryConfig& output_mem_config) {
    TT_FATAL( dim >= 0 && dim <= 3, "dimension have to be 0-3 only corresponding to N,C,H,W");
    return reduce_on_dim<ReduceOpMath::SUM>(input_tensor, dim, output_mem_config);
}

Tensor min(const Tensor &input_tensor, uint dim, const MemoryConfig& output_mem_config) {
    TT_FATAL( dim >= 0 && dim <= 3, "dimension have to be 0-3 only corresponding to N,C,H,W");
    return reduce_on_dim<ReduceOpMath::MIN>(input_tensor, dim, output_mem_config);
}

Tensor max(const Tensor &input_tensor, uint dim, const MemoryConfig& output_mem_config) {
    TT_FATAL( dim >= 0 && dim <= 3, "dimension have to be 0-3 only corresponding to N,C,H,W");
    return reduce_on_dim<ReduceOpMath::MAX>(input_tensor, dim, output_mem_config);
}


using ReduceFnT = Tensor(*)(const Tensor&,unsigned int, const MemoryConfig&);
Tensor global_reduce(ReduceFnT f,const Tensor& val, const MemoryConfig& output_mem_config) {
    Tensor result = val;
    for(int rank = val.shape().rank()-1; rank >=0; rank--)
        result = f(result, rank, output_mem_config);
    return result;
}

Tensor global_sum(const Tensor& val, const MemoryConfig& output_mem_config) {
    return  global_reduce(sum,val,output_mem_config);
}

Tensor global_max(const Tensor& val, const MemoryConfig& output_mem_config) {
    return  global_reduce(max,val,output_mem_config);
}

Tensor global_min(const Tensor& val, const MemoryConfig& output_mem_config) {
    return  global_reduce(min,val,output_mem_config);
}

}  // namespace tt_metal

}  // namespace tt
