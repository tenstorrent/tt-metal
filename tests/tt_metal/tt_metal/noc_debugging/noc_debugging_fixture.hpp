// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>
#include "tt_metal/tt_metal/common/mesh_dispatch_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <impl/context/metal_context.hpp>
#include <impl/debug/noc_debugging.hpp>

namespace tt::tt_metal {

class NOCDebuggingFixture : public MeshDispatchFixture {
public:
    bool has_write_barrier_issue(ChipId chip_id, CoreCoord virtual_core, int processor_id) const {
        auto& noc_debug_state = tt::tt_metal::MetalContext::instance().noc_debug_state();
        if (!noc_debug_state) {
            return false;
        }
        tt_cxy_pair core{chip_id, {virtual_core.x, virtual_core.y}};
        return noc_debug_state->get_issues(core, processor_id).has_issue(NOCDebugIssueType::WRITE_FLUSH_BARRIER);
    }

    bool has_read_barrier_issue(ChipId chip_id, CoreCoord virtual_core, int processor_id) const {
        auto& noc_debug_state = tt::tt_metal::MetalContext::instance().noc_debug_state();
        if (!noc_debug_state) {
            return false;
        }
        tt_cxy_pair core{chip_id, {virtual_core.x, virtual_core.y}};
        return noc_debug_state->get_issues(core, processor_id).has_issue(NOCDebugIssueType::READ_BARRIER);
    }

    template <typename T>
    void RunTestOnDevice(
        const std::function<void(T*, std::shared_ptr<distributed::MeshDevice>)>& run_function,
        const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
        auto run_function_no_args = [this, run_function, mesh_device]() {
            run_function(static_cast<T*>(this), mesh_device);
        };
        MeshDispatchFixture::RunTestOnDevice(run_function_no_args, mesh_device);
    }

protected:
    bool previous_debug_dump_enabled_{};

    void SetUp() override {
        if (this->IsSlowDispatch()) {
            GTEST_SKIP() << "NOC debugging tests require fast dispatch mode";
        }

        previous_debug_dump_enabled_ =
            tt::tt_metal::MetalContext::instance().rtoptions().get_experimental_device_debug_dump_enabled();
        tt::tt_metal::MetalContext::instance().rtoptions().set_experimental_device_debug_dump_enabled(true);

        MeshDispatchFixture::SetUp();

        if (auto& noc_debug_state = tt::tt_metal::MetalContext::instance().noc_debug_state()) {
            noc_debug_state->reset_state();
        }
    }

    void TearDown() override {
        MeshDispatchFixture::TearDown();

        if (auto& noc_debug_state = tt::tt_metal::MetalContext::instance().noc_debug_state()) {
            noc_debug_state->reset_state();
        }

        tt::tt_metal::MetalContext::instance().rtoptions().set_experimental_device_debug_dump_enabled(
            previous_debug_dump_enabled_);
    }
};

}  // namespace tt::tt_metal
