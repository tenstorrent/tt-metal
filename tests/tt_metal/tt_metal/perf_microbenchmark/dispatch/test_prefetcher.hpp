// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Graph
#include "tt_metal/impl/dispatch/topology.hpp"
#include "tt_metal/impl/dispatch/kernel_config/fd_kernel.hpp"
// Compile args struct
#include "tt_metal/impl/dispatch/kernel_config/dispatch.hpp"
#include "tt_metal/impl/dispatch/kernel_config/prefetch.hpp"
#include "tt_metal/impl/dispatch/kernel_config/mux.hpp"
#include "tt_metal/impl/dispatch/kernel_config/demux.hpp"

namespace tt::tt_metal::dispatch_test {

using namespace tt::tt_metal::dispatch;

static constexpr int x = -1;  // unset
constexpr NOC MY_NOC_INDEX = NOC::NOC_0;
constexpr NOC DISPATCH_UPSTREAM_NOC_INDEX = NOC::NOC_1;

static const std::vector<dispatch_kernel_node_t> default_topology = {
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {1, x, x, x}, NOC::NOC_0, NOC::NOC_0, NOC::NOC_0},
    {1, 0, 0, 0, DISPATCH_HD, {0, x, x, x}, {x, x, x, x}, NOC::NOC_0, NOC::NOC_1, NOC::NOC_0},
};

class TestPrefetcherMuxKernel : public FDKernel {
public:
    TestPrefetcherMuxKernel(
        int node_id, chip_id_t device_id, chip_id_t servicing_device_id, uint8_t cq_id, noc_selection_t noc_selection) :
        FDKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection) {}

    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    void ConfigureCore() override;

    mux_static_config_t static_config;
    mux_dependent_config_t dependent_config;
};

class TestPrefetcherDemuxKernel : public FDKernel {
public:
    TestPrefetcherDemuxKernel(
        int node_id, chip_id_t device_id, chip_id_t servicing_device_id, uint8_t cq_id, noc_selection_t noc_selection) :
        FDKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection) {}

    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    void ConfigureCore() override;

    demux_static_config_t static_config;
    demux_dependent_config_t dependent_config;
};

class TestPrefetcherDispatchKernel : public FDKernel {
public:
    TestPrefetcherDispatchKernel(
        int node_id,
        chip_id_t device_id,
        chip_id_t servicing_device_id,
        uint8_t cq_id,
        noc_selection_t noc_selection,
        bool h_variant,
        bool d_variant,
        uint32_t dev_hugepage_completion_buffer_base) :
        FDKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection) {
        static_config.is_h_variant = h_variant;
        static_config.is_d_variant = d_variant;
        static_config.completion_queue_base_addr = dev_hugepage_completion_buffer_base;
        if (h_variant && !dev_hugepage_completion_buffer_base) {
            TT_THROW("dev_hugepage_completion_buffer_base is required to be non zero for variants that talk to host");
        }
    }

    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    void ConfigureCore() override;

    dispatch_static_config_t static_config;
    dispatch_dependent_config_t dependent_config;
};

class TestPrefetcherPrefetchKernel : public FDKernel {
public:
    TestPrefetcherPrefetchKernel(
        int node_id,
        chip_id_t device_id,
        chip_id_t servicing_device_id,
        uint8_t cq_id,
        noc_selection_t noc_selection,
        bool h_variant,
        bool d_variant,
        uint32_t dev_hugepage_issue_base) :
        FDKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection) {
        static_config.is_h_variant = h_variant;
        static_config.is_d_variant = d_variant;
        if (h_variant && !dev_hugepage_issue_base) {
            TT_THROW("dev_hugepage_issue_base is required to be non-zero for prefetch variants that talk to host");
        }
        static_config.pcie_base = dev_hugepage_issue_base;
    }

    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    void ConfigureCore() override;

    prefetch_static_config_t static_config;
    prefetch_dependent_config_t dependent_config;
};

}  // namespace tt::tt_metal::dispatch_test
