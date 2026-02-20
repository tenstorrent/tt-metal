// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "fd_kernel.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/types/xy_pair.hpp>

namespace tt::tt_metal {

// PrefetchWriterKernel runs on NCRISC of the same Tensix core as PrefetchKernel (BRISC).
// It receives the identical set of compile-time defines as the reader so that both kernels
// share the same configuration space. Currently a stub intended for future write-path work.
class PrefetchWriterKernel : public FDKernel {
public:
    PrefetchWriterKernel(
        int node_id, ChipId device_id, ChipId servicing_device_id, uint8_t cq_id, noc_selection_t noc_selection);

    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
};

}  // namespace tt::tt_metal
