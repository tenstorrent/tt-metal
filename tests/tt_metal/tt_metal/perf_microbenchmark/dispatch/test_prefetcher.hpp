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
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {1, x, x, x}, MY_NOC_INDEX, MY_NOC_INDEX, MY_NOC_INDEX},
    {1, 0, 0, 0, DISPATCH_HD, {0, x, x, x}, {x, x, x, x}, MY_NOC_INDEX, DISPATCH_UPSTREAM_NOC_INDEX, MY_NOC_INDEX},
};

static const std::vector<dispatch_kernel_node_t> split_prefetcher_topology = {
    {0, 0, 0, 0, PREFETCH_H, {x, x, x, x}, {1, x, x, x}, MY_NOC_INDEX, MY_NOC_INDEX, MY_NOC_INDEX},
    {1, 0, 0, 0, PREFETCH_D, {0, x, x, x}, {2, x, x, x}, MY_NOC_INDEX, MY_NOC_INDEX, MY_NOC_INDEX},
    {2, 0, 0, 0, DISPATCH_HD, {1, x, x, x}, {x, x, x, x}, MY_NOC_INDEX, DISPATCH_UPSTREAM_NOC_INDEX, MY_NOC_INDEX},
};

static const std::vector<dispatch_kernel_node_t> split_dispatcher_topology = {
    {0, 0, 0, 0, PREFETCH_HD, {x, x, x, x}, {1, x, x, x}, MY_NOC_INDEX, MY_NOC_INDEX, MY_NOC_INDEX},
    {1, 0, 0, 0, DISPATCH_D, {0, x, x, x}, {2, x, x, x}, MY_NOC_INDEX, DISPATCH_UPSTREAM_NOC_INDEX, MY_NOC_INDEX},
    {2, 0, 0, 0, DISPATCH_H, {1, x, x, x}, {0, x, x, x}, MY_NOC_INDEX, DISPATCH_UPSTREAM_NOC_INDEX, MY_NOC_INDEX},
};

static const std::vector<dispatch_kernel_node_t> split_dispatcher_prefetcher_topology = {
    {0, 0, 0, 0, PREFETCH_H, {x, x, x, x}, {1, x, x, x}, MY_NOC_INDEX, MY_NOC_INDEX, MY_NOC_INDEX},
    {1, 0, 0, 0, PREFETCH_D, {0, x, x, x}, {2, x, x, x}, MY_NOC_INDEX, MY_NOC_INDEX, MY_NOC_INDEX},
    {2, 0, 0, 0, DISPATCH_D, {1, x, x, x}, {3, x, x, x}, MY_NOC_INDEX, DISPATCH_UPSTREAM_NOC_INDEX, MY_NOC_INDEX},
    {3, 0, 0, 0, DISPATCH_H, {2, x, x, x}, {0, x, x, x}, MY_NOC_INDEX, DISPATCH_UPSTREAM_NOC_INDEX, MY_NOC_INDEX},
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
        if (h_variant) {
            if (!dev_hugepage_completion_buffer_base)
                TT_THROW("dev_hugepage_completion_buffer_base is required to be non zero for variants that talk to host");
            static_config.completion_queue_base_addr = dev_hugepage_completion_buffer_base;
        } else {
            static_config.completion_queue_base_addr = 0;
        }
    }

    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    void ConfigureCore() override;

    std::vector<uint32_t> CreateCompileArgs();

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

        // pcie is unused for prefetch_d
        if (h_variant) {
            if (!dev_hugepage_issue_base)
                TT_THROW("dev_hugepage_issue_base is required to be non-zero for prefetch variants that talk to host");
            // pcie base is set in ctor
            static_config.pcie_base = dev_hugepage_issue_base;
        } else {
            static_config.pcie_base = 0;
        }
    }

    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    void ConfigureCore() override;

    std::vector<uint32_t> CreateCompileArgs();

    prefetch_static_config_t static_config;
    prefetch_dependent_config_t dependent_config;
};

}  // namespace tt::tt_metal::dispatch_test
