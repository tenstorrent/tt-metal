// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_emule/device.hpp"
#include "umd/device/chip/sw_emule_chip.hpp"
#include "umd/device/cluster.hpp"

namespace tt::tt_metal {

TEST_F(UnitMeshFixture, OrdinaryShortTileInitializesRuntimeCbGeometryWithoutUnpackOverride) {
    constexpr uint8_t cb_id = 0;
    constexpr uint32_t page_size = 64;
    const CoreCoord logical_core{0, 0};

    Program program = CreateProgram();
    CircularBufferConfig cb_config(page_size, {{cb_id, tt::DataFormat::Float16_b}});
    cb_config.set_page_size(cb_id, page_size).set_tile_dims(cb_id, Tile({1, 32}));
    ASSERT_FALSE(cb_config.unpack_face_geometry()[cb_id].has_value());
    CreateCircularBuffer(program, logical_core, cb_config);

    CreateKernelFromString(
        program,
        "void kernel_main() {}",
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    LaunchProgram(this->device(), std::move(program), /*wait_until_cores_done=*/true);

    auto& cluster = MetalContext::instance().get_cluster();
    auto* emule_chip = dynamic_cast<tt::umd::SWEmuleChip*>(cluster.get_driver()->get_chip(this->device().build_id()));
    ASSERT_NE(emule_chip, nullptr);
    const auto physical_core = this->device().virtual_core_from_logical_core(logical_core, CoreType::WORKER);
    auto* emule_core = emule_chip->get_core(tt_xy_pair(physical_core.x, physical_core.y));
    ASSERT_NE(emule_core, nullptr);

    const auto& cb_state = emule_core->cb_sync_array()[cb_id];
    ASSERT_EQ(cb_state.page_size, page_size);
    EXPECT_EQ(cb_state.face_r_dim, 1u);
    EXPECT_EQ(cb_state.num_faces, 2u);
}

}  // namespace tt::tt_metal
