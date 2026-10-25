// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <sstream>
#include <string>

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/context/mesh_device_properties.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>

#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {
namespace {

// sfpu_nan() is a NaN, which never compares equal to itself, so the floats are compared bit for bit.
void ExpectSameProperties(
    const MetalDeviceProperties& expected, const MetalDeviceProperties& actual, const std::string& label) {
    EXPECT_EQ(actual.arch(), expected.arch()) << label;
    EXPECT_EQ(actual.arch_name(), expected.arch_name()) << label;
    EXPECT_EQ(actual.node_local_mem_size(), expected.node_local_mem_size()) << label;
    EXPECT_EQ(actual.dram_alignment(), expected.dram_alignment()) << label;
    EXPECT_EQ(actual.node_local_mem_alignment(), expected.node_local_mem_alignment()) << label;
    EXPECT_EQ(actual.num_dataflow_buffers(), expected.num_dataflow_buffers()) << label;
    EXPECT_EQ(actual.max_worker_node_local_mem_unreserved_size(), expected.max_worker_node_local_mem_unreserved_size())
        << label;
    EXPECT_EQ(std::bit_cast<uint32_t>(actual.sfpu_eps()), std::bit_cast<uint32_t>(expected.sfpu_eps())) << label;
    EXPECT_EQ(std::bit_cast<uint32_t>(actual.sfpu_nan()), std::bit_cast<uint32_t>(expected.sfpu_nan())) << label;
    EXPECT_EQ(std::bit_cast<uint32_t>(actual.sfpu_inf()), std::bit_cast<uint32_t>(expected.sfpu_inf())) << label;
}

// Checks the mesh-wide overload and the coordinate overload at every coordinate of the mesh against `expected`.
void ExpectMeshAnswersLike(
    const MetalDeviceProperties& expected, const distributed::MeshDevice& mesh, const std::string& label) {
    ExpectSameProperties(expected, experimental::get_device_properties(mesh), label + " mesh-wide");
    for (const auto& coord : distributed::MeshCoordinateRange(mesh.shape())) {
        std::ostringstream where;
        where << label << " at " << coord;
        ExpectSameProperties(expected, experimental::get_device_properties(mesh, coord), where.str());
    }
}

// The legacy default context keeps its cluster open until atexit and is reused as-is by later legacy opens, so it is
// released on entry (an earlier test may have left one bound to a different cluster) and on exit (so later tests in
// this binary can open their own cluster). The mock-mode flag is only read when the default context is created, so
// releasing a context after the flag changed is safe.
struct LegacyContextCleanup {
    explicit LegacyContextCleanup(bool disable_mock = false) : disable_mock_mode(disable_mock) {
        if (MetalContext::instance_exists()) {
            tt::tt_metal::detail::ReleaseOwnership();
        }
    }
    ~LegacyContextCleanup() {
        if (MetalContext::instance_exists()) {
            tt::tt_metal::detail::ReleaseOwnership();
        }
        if (disable_mock_mode) {
            experimental::disable_mock_mode();
        }
    }

    bool disable_mock_mode;
};

}  // namespace

TEST(MeshDeviceProperties, CPU_ExplicitEnvMock) {
    for (const tt::ARCH arch : {tt::ARCH::WORMHOLE_B0, tt::ARCH::BLACKHOLE}) {
        MetalEnv env({.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(arch, 2)});
        auto mesh = distributed::MeshDevice::open_mesh_device(
            env, distributed::MeshDeviceConfig(env.get_system_mesh().shape()));
        ASSERT_EQ(mesh->num_devices(), 2u);

        const auto env_props = env.get_device_properties();
        EXPECT_EQ(experimental::get_device_properties(*mesh).arch(), arch);
        ExpectMeshAnswersLike(env_props, *mesh, env_props.arch_name());
    }
}

TEST(MeshDeviceProperties, CPU_LegacyMock) {
    LegacyContextCleanup cleanup(/*disable_mock_mode=*/true);
    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, /*num_chips=*/2);

    auto mesh = distributed::MeshDevice::create(distributed::MeshDeviceConfig{distributed::MeshShape{1u, 2u}});
    ASSERT_NE(mesh, nullptr);

    const auto env_props = MetalContext::instance().get_env().get_device_properties();
    EXPECT_EQ(experimental::get_device_properties(*mesh).arch(), tt::ARCH::WORMHOLE_B0);
    ExpectMeshAnswersLike(env_props, *mesh, "legacy mock");

    mesh->close();
}

TEST(MeshDeviceProperties, CPU_OutOfRangeCoordinateThrows) {
    MetalEnv env({.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2)});
    auto mesh =
        distributed::MeshDevice::open_mesh_device(env, distributed::MeshDeviceConfig{distributed::MeshShape{1u, 2u}});
    ASSERT_EQ(mesh->shape(), distributed::MeshShape(1, 2));

    EXPECT_ANY_THROW(experimental::get_device_properties(*mesh, distributed::MeshCoordinate(0, 2)));
    EXPECT_ANY_THROW(experimental::get_device_properties(*mesh, distributed::MeshCoordinate(1, 0)));
    EXPECT_ANY_THROW(experimental::get_device_properties(*mesh, distributed::MeshCoordinate(0, 0, 0)));
    EXPECT_ANY_THROW(experimental::get_device_properties(*mesh, distributed::MeshCoordinate(0)));

    EXPECT_NO_THROW(experimental::get_device_properties(*mesh, distributed::MeshCoordinate(0, 1)));
}

TEST(MeshDeviceProperties, CPU_SubmeshAndUnitMesh) {
    {
        MetalEnv env({.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2)});
        const auto env_props = env.get_device_properties();
        auto parent = distributed::MeshDevice::open_mesh_device(
            env, distributed::MeshDeviceConfig{distributed::MeshShape{1u, 2u}});
        EXPECT_NO_THROW(experimental::get_device_properties(*parent, distributed::MeshCoordinate(0, 1)));

        auto submeshes = parent->create_submeshes(distributed::MeshShape(1, 1));
        ASSERT_EQ(submeshes.size(), 2u);
        for (const auto& submesh : submeshes) {
            ExpectMeshAnswersLike(env_props, *submesh, "submesh");
            // Coordinates are local to the submesh: {0, 1} exists in the parent but not in a 1x1 submesh.
            EXPECT_ANY_THROW(experimental::get_device_properties(*submesh, distributed::MeshCoordinate(0, 1)));
        }
    }

    {
        MetalEnv env({.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2)});
        const auto env_props = env.get_device_properties();
        const std::array<ChipId, 2> device_ids{0, 1};
        auto unit_meshes = distributed::MeshDevice::open_unit_meshes(env, device_ids);
        ASSERT_EQ(unit_meshes.size(), device_ids.size());
        for (const auto& [device_id, unit_mesh] : unit_meshes) {
            ExpectMeshAnswersLike(env_props, *unit_mesh, "unit mesh " + std::to_string(device_id));
            EXPECT_ANY_THROW(experimental::get_device_properties(*unit_mesh, distributed::MeshCoordinate(0, 1)));
        }
    }
}

TEST(MeshDeviceProperties, ExplicitEnv) {
    MetalEnv env;
    auto mesh =
        distributed::MeshDevice::open_mesh_device(env, distributed::MeshDeviceConfig(env.get_system_mesh().shape()));
    ASSERT_GT(mesh->num_devices(), 0u);
    ExpectMeshAnswersLike(env.get_device_properties(), *mesh, "silicon explicit env");
}

TEST(MeshDeviceProperties, Legacy) {
    LegacyContextCleanup cleanup;
    auto mesh =
        distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::SystemMesh::instance().shape()));
    ASSERT_GT(mesh->num_devices(), 0u);
    ExpectMeshAnswersLike(MetalContext::instance().get_env().get_device_properties(), *mesh, "silicon legacy");

    mesh->close();
}

}  // namespace tt::tt_metal
