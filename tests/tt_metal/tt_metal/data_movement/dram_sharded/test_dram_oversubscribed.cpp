// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal {

namespace unit_tests::dm::dram_oversubscribed {

struct OverSubscribedConfig {
    uint32_t num_banks;
    uint32_t num_shards_per_bank;
    uint32_t pages_per_shard;
    uint32_t page_size_bytes;
};

bool run_verify(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const OverSubscribedConfig& cfg) {
    IDevice* device = mesh_device->impl().get_device(0);

    const uint32_t num_shards = cfg.num_banks * cfg.num_shards_per_bank;
    const uint32_t shard_volume_bytes = cfg.pages_per_shard * cfg.page_size_bytes;
    const uint32_t num_pages = num_shards * cfg.pages_per_shard;
    const uint32_t total_size_bytes = num_pages * cfg.page_size_bytes;

    // BufferDistributionSpec with num_shards > num_banks: round-robin places
    // shard s on bank (s % num_banks) at bank-local offset (s / num_banks) * V.
    CoreRange dram_bank_range({0, 0}, {cfg.num_banks - 1, 0});
    BufferDistributionSpec shard_spec(
        Shape{1, num_pages}, Shape{1, cfg.pages_per_shard}, corerange_to_cores(dram_bank_range));

    distributed::DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = cfg.page_size_bytes,
        .buffer_type = BufferType::DRAM,
        .sharding_args = BufferShardingArgs(shard_spec)};
    distributed::ReplicatedBufferConfig mesh_buffer_config{.size = total_size_bytes};
    auto mesh_buffer = distributed::MeshBuffer::create(mesh_buffer_config, per_device_buffer_config, mesh_device.get());

    // Host data: fill shard s entirely with uint32 marker = s.
    std::vector<uint32_t> packed_input(total_size_bytes / sizeof(uint32_t), 0);
    const uint32_t u32_per_shard = shard_volume_bytes / sizeof(uint32_t);
    for (uint32_t s = 0; s < num_shards; ++s) {
        for (uint32_t i = 0; i < u32_per_shard; ++i) {
            packed_input[s * u32_per_shard + i] = s;
        }
    }

    CoreCoord reader_core{0, 0};
    CoreRangeSet reader_set(CoreRange(reader_core, reader_core));

    constexpr uint32_t marker_bytes = 16;  // 4 uint32s per (bank, slab)

    Program program = CreateProgram();
    KernelHandle reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/dram_sharded/kernels/dram_oversubscribed_verifier.cpp",
        reader_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {cfg.num_banks, cfg.num_shards_per_bank, shard_volume_bytes, marker_bytes}});

    const uint32_t l1_addr = get_l1_address_and_size(mesh_device, reader_core).base_address;
    SetRuntimeArgs(program, reader_kernel, reader_set, {static_cast<uint32_t>(mesh_buffer->address()), l1_addr});

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, mesh_buffer, packed_input);

    distributed::MeshWorkload mesh_workload;
    std::vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    const uint32_t readback_bytes = cfg.num_banks * cfg.num_shards_per_bank * marker_bytes;
    std::vector<uint32_t> result;
    tt::tt_metal::detail::ReadFromDeviceL1(device, reader_core, l1_addr, readback_bytes, result);

    bool ok = true;
    const uint32_t u32_per_slot = marker_bytes / sizeof(uint32_t);
    for (uint32_t bank = 0; bank < cfg.num_banks; ++bank) {
        for (uint32_t slab = 0; slab < cfg.num_shards_per_bank; ++slab) {
            const uint32_t expected_shard = bank + slab * cfg.num_banks;
            for (uint32_t k = 0; k < u32_per_slot; ++k) {
                const uint32_t idx = (bank * cfg.num_shards_per_bank + slab) * u32_per_slot + k;
                if (result[idx] != expected_shard) {
                    log_error(
                        tt::LogTest,
                        "Mismatch at bank={} slab={} u32={}: got {}, expected {}",
                        bank,
                        slab,
                        k,
                        result[idx],
                        expected_shard);
                    ok = false;
                }
            }
        }
    }
    return ok;
}

}  // namespace unit_tests::dm::dram_oversubscribed

// Verifies that BufferDistributionSpec correctly places `num_shards > num_banks`
// shards on DRAM banks in round-robin order, and that a kernel can read each
// bank-local slab via get_noc_addr_from_bank_id() and observe the expected data.
// This is the foundational layer the receiver-contiguous DRAM-core prefetcher
// layout depends on.
TEST_F(GenericMeshDeviceFixture, DramOverSubscribedShardLayoutSmall) {
    auto mesh_device = get_mesh_device();
    unit_tests::dm::dram_oversubscribed::OverSubscribedConfig cfg{
        .num_banks = 4, .num_shards_per_bank = 4, .pages_per_shard = 4, .page_size_bytes = 1024};
    ASSERT_LE(cfg.num_banks, mesh_device->num_dram_channels());
    EXPECT_TRUE(unit_tests::dm::dram_oversubscribed::run_verify(mesh_device, cfg));
}

// Production-shape-like topology: 8 banks x 8 receivers per bank = 64 shards.
TEST_F(GenericMeshDeviceFixture, DramOverSubscribedShardLayoutRing64) {
    auto mesh_device = get_mesh_device();
    if (mesh_device->num_dram_channels() < 8) {
        GTEST_SKIP() << "Requires at least 8 DRAM channels";
    }
    unit_tests::dm::dram_oversubscribed::OverSubscribedConfig cfg{
        .num_banks = 8, .num_shards_per_bank = 8, .pages_per_shard = 2, .page_size_bytes = 1088};
    EXPECT_TRUE(unit_tests::dm::dram_oversubscribed::run_verify(mesh_device, cfg));
}

}  // namespace tt::tt_metal
