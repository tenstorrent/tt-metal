// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "cyclic_sdpa_fw_device_operation_types.hpp"
#include "cyclic_sdpa_fw_program_factory.hpp"
#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cyclic_sdpa_fw::device {

struct CyclicSDPAForwardDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::cyclic_sdpa_fw::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::cyclic_sdpa_fw::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::cyclic_sdpa_fw::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::cyclic_sdpa_fw::device::tensor_return_value_t;
    using program_factory_t = std::variant<CyclicSDPAForwardProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::cyclic_sdpa_fw::device

namespace ttnn::prim {

ttml::metal::ops::cyclic_sdpa_fw::device::CyclicSDPAForwardDeviceOperation::tensor_return_value_t
ttml_cyclic_sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    uint32_t rows_per_block_tiles = 1U,
    ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::Causal,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_intermediates = std::nullopt,
    uint32_t max_groups = 0U,
    uint32_t sequence_chunks = 1U,
    const std::vector<uint32_t>& row_chunks = {},
    const std::vector<uint32_t>& col_chunks = {},
    bool fast = false);

}  // namespace ttnn::prim
