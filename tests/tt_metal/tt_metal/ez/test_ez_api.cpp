// SPDX-FileCopyrightText: (c) 2026 Olof Johansson <olof@lixom.net>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/ez/ez.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::ez;

class EzApiTest : public ::testing::Test {};

TEST_F(EzApiTest, HelloWorld) {
    // Verify that DeviceContext + ProgramBuilder can create and run a trivial compute kernel.
    DeviceContext ctx(0);
    auto program = ProgramBuilder(CoreCoord{0, 0})
                       .compute("tests/tt_metal/tt_metal/ez/kernels/void_compute.cpp")
                       .build();
    ctx.run(std::move(program));
}

TEST_F(EzApiTest, DramBufferRoundtrip) {
    // Verify write/read roundtrip through a DRAM tile buffer.
    DeviceContext ctx(0);
    constexpr uint32_t n_tiles = 4;
    auto buf = ctx.dram_tile_buffer(n_tiles);

    constexpr uint32_t elements = n_tiles * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    std::vector<bfloat16> src(elements);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : src) {
        v = bfloat16(dist(rng));
    }

    ctx.write(buf, src);
    ctx.finish();  // ensure write completes
    auto dst = ctx.read<bfloat16>(buf);

    ASSERT_EQ(dst.size(), src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        EXPECT_EQ(static_cast<float>(src[i]), static_cast<float>(dst[i])) << "Mismatch at index " << i;
    }
}

TEST_F(EzApiTest, EltwiseBinary) {
    // Full reader/writer/compute pipeline: add two tile buffers.
    DeviceContext ctx(0);
    constexpr uint32_t n_tiles = 16;
    constexpr uint32_t elements = n_tiles * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;

    auto src0 = ctx.dram_tile_buffer(n_tiles);
    auto src1 = ctx.dram_tile_buffer(n_tiles);
    auto dst = ctx.dram_tile_buffer(n_tiles);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<bfloat16> a_data(elements), b_data(elements);
    for (size_t i = 0; i < elements; ++i) {
        a_data[i] = bfloat16(dist(rng));
        b_data[i] = bfloat16(dist(rng));
    }

    ctx.write(src0, a_data);
    ctx.write(src1, b_data);

    constexpr CoreCoord core = {0, 0};
    auto program = ProgramBuilder(core)
                       .cb(tt::CBIndex::c_0)
                       .cb(tt::CBIndex::c_1)
                       .cb(tt::CBIndex::c_16)
                       .reader(
                           "tests/tt_metal/tt_metal/ez/kernels/read_tiles.cpp",
                           {src0, src1})
                       .runtime_args({src0->address(), src1->address(), n_tiles})
                       .compute("tests/tt_metal/tt_metal/ez/kernels/tiles_add.cpp")
                       .runtime_args({n_tiles})
                       .writer(
                           "tests/tt_metal/tt_metal/ez/kernels/write_tile.cpp",
                           {dst})
                       .runtime_args({dst->address(), n_tiles})
                       .build();

    ctx.run(std::move(program));
    auto result = ctx.read<bfloat16>(dst);

    ASSERT_EQ(result.size(), elements);
    constexpr float eps = 1e-2f;
    for (size_t i = 0; i < elements; ++i) {
        float expected = static_cast<float>(a_data[i]) + static_cast<float>(b_data[i]);
        float actual = static_cast<float>(result[i]);
        EXPECT_NEAR(expected, actual, eps) << "Mismatch at index " << i;
    }
}

TEST_F(EzApiTest, MultiCore) {
    // ProgramBuilder with CoreRange and per-core runtime args.
    DeviceContext ctx(0);
    CoreRange cores({0, 0}, {1, 0});

    auto builder = ProgramBuilder(cores);
    auto& k = builder.compute("tests/tt_metal/tt_metal/ez/kernels/void_compute.cpp");
    k.runtime_args_at(CoreCoord{0, 0}, {});
    k.runtime_args_at(CoreCoord{1, 0}, {});

    auto program = builder.build();
    ctx.run(std::move(program));
}

TEST_F(EzApiTest, PerCoreLambdaArgs) {
    // Verify lambda-based runtime_args iterates over all cores in a CoreRange.
    DeviceContext ctx(0);
    CoreRange cores({0, 0}, {1, 0});

    auto builder = ProgramBuilder(cores);
    builder.compute("tests/tt_metal/tt_metal/ez/kernels/void_compute.cpp")
        .runtime_args([](const CoreCoord& core) -> std::vector<uint32_t> {
            // Each core gets a unique arg based on its x coordinate.
            return {core.x};
        });

    auto program = builder.build();
    ctx.run(std::move(program));
}

TEST_F(EzApiTest, CoreOverride) {
    // Use .on() to place a kernel on a different core than the default.
    DeviceContext ctx(0);
    constexpr CoreCoord default_core = {0, 0};
    constexpr CoreCoord other_core = {1, 0};

    auto program = ProgramBuilder(default_core)
                       .compute("tests/tt_metal/tt_metal/ez/kernels/void_compute.cpp")
                       .on(other_core)
                       .compute("tests/tt_metal/tt_metal/ez/kernels/void_compute.cpp")
                       .build();

    ctx.run(std::move(program));
}

TEST_F(EzApiTest, ShardedL1Buffer) {
    // Verify sharded_l1_buffer creates a valid L1 sharded buffer.
    DeviceContext ctx(0);
    constexpr uint32_t n_tiles_x = 1;
    constexpr uint32_t n_tiles_y = 2;
    CoreRangeSet cores(CoreRange({0, 0}, {0, 1}));

    ShardConfig config{
        .cores = cores,
        .shard_shape = {tt::constants::TILE_HEIGHT, n_tiles_x * tt::constants::TILE_WIDTH},
        .tensor2d_shape_in_pages = {n_tiles_y, n_tiles_x},
        .layout = TensorMemoryLayout::HEIGHT_SHARDED,
    };
    auto buf = ctx.sharded_l1_buffer(config);
    ASSERT_NE(buf, nullptr);
    EXPECT_GT(buf->address(), 0u);
}

TEST_F(EzApiTest, L1BackedCircularBuffer) {
    // Verify cb() overload that backs a CB with an existing L1 buffer.
    DeviceContext ctx(0);
    constexpr uint32_t n_tiles = 4;
    auto ts = tt::tile_size(tt::DataFormat::Float16_b);
    auto l1_buf = ctx.l1_buffer(n_tiles * ts, ts);

    auto program = ProgramBuilder(CoreCoord{0, 0})
                       .cb(tt::CBIndex::c_0, l1_buf)
                       .compute("tests/tt_metal/tt_metal/ez/kernels/void_compute.cpp")
                       .build();
    ctx.run(std::move(program));
}

TEST_F(EzApiTest, WrapExistingDevice) {
    // DeviceContext wrapping an existing MeshDevice does NOT close it on destruction.
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    {
        DeviceContext ctx(mesh_device);
        // Use the context briefly.
        auto buf = ctx.dram_tile_buffer(1);
        // ctx goes out of scope here — should NOT close mesh_device.
    }
    // Device should still be usable.
    auto& cq = mesh_device->mesh_command_queue();
    distributed::Finish(cq);
    mesh_device->close();
}

TEST_F(EzApiTest, Semaphore) {
    // Verify semaphore creation and use via ProgramBuilder.
    DeviceContext ctx(0);
    CoreRange cores({0, 0}, {0, 1});

    auto builder = ProgramBuilder(cores);
    uint32_t sem_addr = builder.semaphore(0);

    builder.compute("tests/tt_metal/tt_metal/ez/kernels/void_compute.cpp")
        .runtime_args([&](const CoreCoord&) -> std::vector<uint32_t> { return {sem_addr}; });
    auto program = builder.build();
    ctx.run(std::move(program));
}

TEST_F(EzApiTest, PhysicalCore) {
    // Verify physical_core returns a valid physical coordinate.
    DeviceContext ctx(0);
    CoreCoord logical = {0, 0};
    CoreCoord physical = ctx.physical_core(logical);
    // Physical coords should be valid (non-negative) — exact values are device-dependent.
    EXPECT_GE(physical.x, 0u);
    EXPECT_GE(physical.y, 0u);
}

TEST_F(EzApiTest, NonBlockingLaunchAndFinish) {
    // Verify launch() + finish() works as an alternative to run().
    DeviceContext ctx(0);
    auto program = ProgramBuilder(CoreCoord{0, 0})
                       .compute("tests/tt_metal/tt_metal/ez/kernels/void_compute.cpp")
                       .build();
    ctx.launch(std::move(program));
    ctx.finish();
}

TEST_F(EzApiTest, PerKernelNamedArgs) {
    // Verify that named_args() called on a KernelRef applies to that kernel only.
    // The compute kernel reads CB indices from named compile-time args; if they
    // were routed to the wrong kernel (or lost), compilation would fail.
    DeviceContext ctx(0);
    constexpr uint32_t n_tiles = 4;
    constexpr uint32_t elements = n_tiles * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;

    auto src0 = ctx.dram_tile_buffer(n_tiles);
    auto src1 = ctx.dram_tile_buffer(n_tiles);
    auto dst = ctx.dram_tile_buffer(n_tiles);

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<bfloat16> a_data(elements), b_data(elements);
    for (size_t i = 0; i < elements; ++i) {
        a_data[i] = bfloat16(dist(rng));
        b_data[i] = bfloat16(dist(rng));
    }

    ctx.write(src0, a_data);
    ctx.write(src1, b_data);

    // named_args() is called on the KernelRef returned by .compute(), not on the builder.
    constexpr CoreCoord core = {0, 0};
    auto program = ProgramBuilder(core)
                       .cb(tt::CBIndex::c_0)
                       .cb(tt::CBIndex::c_1)
                       .cb(tt::CBIndex::c_16)
                       .reader(
                           "tests/tt_metal/tt_metal/ez/kernels/read_tiles.cpp",
                           {src0, src1})
                       .runtime_args({src0->address(), src1->address(), n_tiles})
                       .compute("tests/tt_metal/tt_metal/ez/kernels/tiles_add_named.cpp")
                       .named_args({{"cb_in0", (uint32_t)tt::CBIndex::c_0},
                                    {"cb_in1", (uint32_t)tt::CBIndex::c_1},
                                    {"cb_out0", (uint32_t)tt::CBIndex::c_16}})
                       .runtime_args({n_tiles})
                       .writer(
                           "tests/tt_metal/tt_metal/ez/kernels/write_tile.cpp",
                           {dst})
                       .runtime_args({dst->address(), n_tiles})
                       .build();

    ctx.run(std::move(program));
    auto result = ctx.read<bfloat16>(dst);

    ASSERT_EQ(result.size(), elements);
    constexpr float eps = 1e-2f;
    for (size_t i = 0; i < elements; ++i) {
        float expected = static_cast<float>(a_data[i]) + static_cast<float>(b_data[i]);
        float actual = static_cast<float>(result[i]);
        EXPECT_NEAR(expected, actual, eps) << "Mismatch at index " << i;
    }
}
