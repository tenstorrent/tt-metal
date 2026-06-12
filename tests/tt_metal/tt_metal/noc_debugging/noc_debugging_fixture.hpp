// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <string>
#include <gtest/gtest.h>
#include "profiler_state_manager.hpp"
#include "tt_metal/tt_metal/common/mesh_dispatch_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <impl/context/metal_context.hpp>
#include <impl/debug/noc_debugging.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>

namespace tt::tt_metal {

class NOCDebuggingFixture : public MeshDispatchFixture {
public:
    static void SetUpTestSuite() {
#if !defined(TRACY_ENABLE)
        return;
#endif
        // NOC debugging requires profiler + NOC event infrastructure, which is
        // created during MetalContext::initialize_impl() only when profiler_enabled
        // is true at that time.  Setting the env var before create_shared_devices()
        // ensures profiler_state_manager_ is allocated and firmware is configured
        // for NOC event collection.
        const char* prev = getenv("TT_METAL_NOC_DEBUG_DUMP");
        had_prev_env_ = (prev != nullptr);
        if (had_prev_env_) {
            prev_env_value_ = prev;
        }
        setenv("TT_METAL_NOC_DEBUG_DUMP", "1", /*overwrite=*/1);
        MeshDispatchFixture::create_shared_devices();
    }

    static void TearDownTestSuite() {
#if !defined(TRACY_ENABLE)
        return;
#endif
        MeshDispatchFixture::destroy_shared_devices();
        if (had_prev_env_) {
            setenv("TT_METAL_NOC_DEBUG_DUMP", prev_env_value_.c_str(), 1);
        } else {
            unsetenv("TT_METAL_NOC_DEBUG_DUMP");
        }
    }

    bool has_write_barrier_issue(ChipId chip_id, CoreCoord virtual_core, int processor_id) const {
        auto& noc_debug_state = tt::tt_metal::MetalContext::instance().noc_debug_state();
        if (!noc_debug_state) {
            return false;
        }
        tt_cxy_pair core{chip_id, {virtual_core.x, virtual_core.y}};
        return noc_debug_state->get_issues(core, processor_id)
            .has_base_issue(NOCDebugIssueBaseType::WRITE_FLUSH_BARRIER);
    }

    bool has_read_barrier_issue(ChipId chip_id, CoreCoord virtual_core, int processor_id) const {
        auto& noc_debug_state = tt::tt_metal::MetalContext::instance().noc_debug_state();
        if (!noc_debug_state) {
            return false;
        }
        tt_cxy_pair core{chip_id, {virtual_core.x, virtual_core.y}};
        return noc_debug_state->get_issues(core, processor_id).has_base_issue(NOCDebugIssueBaseType::READ_BARRIER);
    }

    bool has_unflushed_semaphore_mcast_issue(ChipId chip_id, CoreCoord virtual_core, int processor_id) const {
        auto& noc_debug_state = tt::tt_metal::MetalContext::instance().noc_debug_state();
        if (!noc_debug_state) {
            return false;
        }
        tt_cxy_pair core{chip_id, {virtual_core.x, virtual_core.y}};
        return noc_debug_state->get_issues(core, processor_id)
            .has_issue(
                NOCDebugIssueType(NOCDebugIssueBaseType::UNFLUSHED_WRITE_AT_END, /*mcast=*/true, /*semaphore=*/true));
    }

    bool has_unflushed_write_mcast_issue(ChipId chip_id, CoreCoord virtual_core, int processor_id) const {
        auto& noc_debug_state = tt::tt_metal::MetalContext::instance().noc_debug_state();
        if (!noc_debug_state) {
            return false;
        }
        tt_cxy_pair core{chip_id, {virtual_core.x, virtual_core.y}};
        return noc_debug_state->get_issues(core, processor_id)
            .has_issue(
                NOCDebugIssueType(NOCDebugIssueBaseType::UNFLUSHED_WRITE_AT_END, /*mcast=*/true, /*semaphore=*/false));
    }

    bool has_write_to_locked_issue(ChipId chip_id, CoreCoord virtual_core, int processor_id) const {
        auto& noc_debug_state = tt::tt_metal::MetalContext::instance().noc_debug_state();
        if (!noc_debug_state) {
            return false;
        }
        tt_cxy_pair core{chip_id, {virtual_core.x, virtual_core.y}};
        const NOCDebugIssue& issue = noc_debug_state->get_issues(core, processor_id);
        return issue.has_base_issue(NOCDebugIssueBaseType::WRITE_TO_LOCKED_CORE_LOCAL_MEM) ||
               issue.has_base_issue(NOCDebugIssueBaseType::WRITE_TO_LOCKED_CB);
    }

    std::vector<NOCDebugIssueType> get_write_to_locked_issues(
        ChipId chip_id, CoreCoord virtual_core, int processor_id) const {
        auto& noc_debug_state = tt::tt_metal::MetalContext::instance().noc_debug_state();
        std::vector<NOCDebugIssueType> result;
        if (!noc_debug_state) {
            return result;
        }
        tt_cxy_pair core{chip_id, {virtual_core.x, virtual_core.y}};
        const NOCDebugIssue& issue = noc_debug_state->get_issues(core, processor_id);
        auto mem_issues = issue.get_issues_by_base(NOCDebugIssueBaseType::WRITE_TO_LOCKED_CORE_LOCAL_MEM);
        auto cb_issues = issue.get_issues_by_base(NOCDebugIssueBaseType::WRITE_TO_LOCKED_CB);
        result.insert(result.end(), mem_issues.begin(), mem_issues.end());
        result.insert(result.end(), cb_issues.begin(), cb_issues.end());
        return result;
    }

    template <typename T>
    void RunTestOnDevice(
        const std::function<void(T*, std::shared_ptr<distributed::MeshDevice>)>& run_function,
        const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
        // Reset NOC debug state before each device iteration
        if (auto& noc_debug_state = tt::tt_metal::MetalContext::instance().noc_debug_state()) {
            noc_debug_state->reset_state();
        }
        auto run_function_no_args = [this, run_function, mesh_device]() {
            run_function(static_cast<T*>(this), mesh_device);
        };
        MeshDispatchFixture::RunTestOnDevice(run_function_no_args, mesh_device);
    }

protected:
    static inline bool had_prev_env_{false};
    static inline std::string prev_env_value_{};

    void SetUp() override {
#if !defined(TRACY_ENABLE)
        GTEST_SKIP() << "NOC debugging tests require a Tracy-enabled build (build with ENABLE_TRACY=ON)";
#endif
        MeshDispatchFixture::SetUp();

        if (this->IsSlowDispatch()) {
            GTEST_SKIP() << "NOC debugging tests require fast dispatch mode";
        }

        if (tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled() ||
            tt::tt_metal::MetalContext::instance().rtoptions().get_feature_enabled(
                tt::llrt::RunTimeDebugFeatureDprint)) {
            GTEST_SKIP() << "NOC debugging tests require Watcher and DPRINT to be disabled";
        }

        if (auto& noc_debug_state = tt::tt_metal::MetalContext::instance().noc_debug_state()) {
            noc_debug_state->reset_state();
        }
    }

    void TearDown() override {
#if !defined(TRACY_ENABLE)
        return;
#endif
        if (auto& noc_debug_state = tt::tt_metal::MetalContext::instance().noc_debug_state()) {
            noc_debug_state->reset_state();
        }
        MeshDispatchFixture::TearDown();
    }
};

}  // namespace tt::tt_metal
