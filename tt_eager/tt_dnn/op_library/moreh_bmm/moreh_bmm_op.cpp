// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_bmm/moreh_bmm_op.hpp"

#include "tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
namespace operations {
namespace primary {

inline void moreh_bmm_validate(const Tensor& input, const Tensor& mat2) {
    const auto& a_shape = input.get_legacy_shape();
    const auto& b_shape = mat2.get_legacy_shape();

    TT_ASSERT(a_shape[0] == 1, "input must be a 3D tensor");
    TT_ASSERT(b_shape[0] == 1, "mat2 must be a 3D tensor");
}

Tensor moreh_bmm_(const Tensor& input, const Tensor& mat2, const MemoryConfig& mem_config) {
    moreh_bmm_validate(input, mat2);
    return moreh_matmul(input, mat2, std::nullopt, false, false, mem_config);
}

Tensor moreh_bmm(const Tensor& input, const Tensor& mat2, const MemoryConfig& output_mem_config) {
    TT_ASSERT(
        input.storage_type() == StorageType::DEVICE && mat2.storage_type() == StorageType::DEVICE,
        "input tensors need to be on device");

    return moreh_bmm_(input, mat2, output_mem_config);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
