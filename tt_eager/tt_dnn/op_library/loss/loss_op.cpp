// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/loss/loss_op.hpp"

#include <optional>

#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_eager/tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_eager/tt_dnn/op_library/composite/composite_ops.hpp"

#include "ttnn/operations/eltwise/unary/device/unary_op.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

using ttnn::operations::unary::UnaryWithParam;
using ttnn::operations::unary::UnaryOpType;

Tensor lossfunction(
    const Tensor& ref,
    const Tensor& prediction,
    const LossFunction loss_kind,
    const LossReductionMode reduce_mode,
    const MemoryConfig& mem_config) {
    Tensor result(ref);
    std::vector<UnaryWithParam> fused_ops;
    switch(loss_kind) {
        case LossFunction::MAE:
            fused_ops.push_back(UnaryWithParam{UnaryOpType::ABS});
            break;
        case LossFunction::MSE:
            fused_ops.push_back(UnaryWithParam{UnaryOpType::SQUARE});
            break;
        default:
            TT_FATAL("unsupported loss function");
    }
    result = ttnn::subtract(ref, prediction, std::nullopt, std::nullopt, std::nullopt, fused_ops);
    switch( reduce_mode ) {
        case LossReductionMode::SUM:
            return tt::tt_metal::global_sum(result, mem_config);
        case LossReductionMode::MEAN:
            return tt::tt_metal::global_mean(result, mem_config);
        case LossReductionMode::NONE:
        default:
            break;
    }
    return result;
}

Tensor mseloss(
    const Tensor& ref,
    const Tensor& prediction,
    const LossReductionMode mode,
    const MemoryConfig& mem_config) {
    return lossfunction(ref,prediction,LossFunction::MSE,mode,mem_config);
}

Tensor maeloss(
    const Tensor& ref,
    const Tensor& prediction,
    const LossReductionMode mode,
    const MemoryConfig& mem_config) {
    return lossfunction(ref,prediction,LossFunction::MAE,mode,mem_config);
}

}  // namespace tt_metal

}  // namespace tt
