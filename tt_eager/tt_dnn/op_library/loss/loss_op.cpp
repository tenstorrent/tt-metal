// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/loss/loss_op.hpp"

#include <optional>

#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_eager/tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_eager/tt_dnn/op_library/composite/composite_ops.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

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
            fused_ops = {UnaryWithParam{UnaryOpType::ABS}};
            result = sub(ref,prediction, fused_ops);
            break;
        case LossFunction::MSE:
            fused_ops = {UnaryWithParam{UnaryOpType::SQUARE}};
            result = sub(ref,prediction, fused_ops);
            break;
        default:
            TT_FATAL("unsupported loss function");
    }
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
