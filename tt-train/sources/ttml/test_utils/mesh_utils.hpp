// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <optional>
#include <string_view>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/system_mesh.hpp>

#include "autograd/auto_context.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace ttml::test_utils {

// Global device capacity of the system mesh which spans all hosts in a multi-host run.
inline size_t system_mesh_size() {
    return tt::tt_metal::distributed::SystemMesh::instance().shape().mesh_size();
}

inline bool system_supports_mesh(const tt::tt_metal::distributed::MeshShape& shape) {
    return system_mesh_size() >= shape.mesh_size();
}

// The multi-device CI legs select suites by these prefixes (--gtest_filter="Mesh*" on QuietBox 2, "Galaxy*" on Galaxy)
// and the single-device legs skip them, so a multi-device suite named otherwise runs nowhere.
inline constexpr std::array<std::string_view, 2> kMultiDeviceSuitePrefixes = {"Mesh", "Galaxy"};

inline bool current_suite_has_multi_device_prefix() {
    const std::string_view suite = ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name();
    return std::ranges::any_of(
        kMultiDeviceSuitePrefixes, [&](std::string_view prefix) { return suite.starts_with(prefix); });
}

}  // namespace ttml::test_utils

// ASSERT_TRUE() and GTEST_SKIP() expand into return statements, so this has to be a macro.
// Skips if the system does not have enough devices for the mesh; any failure with the mesh open should fail the test
// elsewhere. A multi-device suite must be named for a multi-device CI leg to select it. That is asserted before the
// skip, so a misnamed suite fails on every runner, including single-device ones.
#define SKIP_UNLESS_MESH_SUPPORTED(shape)                                                                            \
    do {                                                                                                             \
        const auto& s = (shape);                                                                                     \
        ASSERT_TRUE(s.mesh_size() <= 1 || ttml::test_utils::current_suite_has_multi_device_prefix()) << fmt::format( \
            "Suite {} needs a {} mesh, so its name must start with one of [{}] for a multi-device "                  \
            "CI leg to select it",                                                                                   \
            ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name(),                              \
            s,                                                                                                       \
            fmt::join(ttml::test_utils::kMultiDeviceSuitePrefixes, ", "));                                           \
        if (!ttml::test_utils::system_supports_mesh(s)) {                                                            \
            GTEST_SKIP() << fmt::format(                                                                             \
                "Skipping test: a {} mesh needs {} devices, the system has {}",                                      \
                s,                                                                                                   \
                s.mesh_size(),                                                                                       \
                ttml::test_utils::system_mesh_size());                                                               \
        }                                                                                                            \
    } while (0)

namespace ttml::test_utils {

// Opens a fabric-enabled mesh of the given shape for each test, skipping if the system is too small.
class MeshFixture : public ::testing::Test {
protected:
    explicit MeshFixture(
        const tt::tt_metal::distributed::MeshShape& mesh_shape, std::optional<uint32_t> seed = std::nullopt) :
        m_mesh_shape(mesh_shape), m_seed(seed) {
    }

    void SetUp() override {
        SKIP_UNLESS_MESH_SUPPORTED(m_mesh_shape);

        ttml::ttnn_fixed::distributed::enable_fabric(static_cast<uint32_t>(m_mesh_shape.mesh_size()));
        ttml::autograd::ctx().open_device(m_mesh_shape);
        if (m_seed.has_value()) {
            ttml::autograd::ctx().set_seed(*m_seed);
        }
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }

    tt::tt_metal::distributed::MeshShape m_mesh_shape;
    std::optional<uint32_t> m_seed;
};

class Mesh1x2Fixture : public MeshFixture {
protected:
    explicit Mesh1x2Fixture(std::optional<uint32_t> seed = std::nullopt) :
        MeshFixture(tt::tt_metal::distributed::MeshShape(1, 2), seed) {
    }
};

// A 1x2 mesh opened once for the whole suite instead of once per test. Each test still starts on an empty program
// cache, as on a freshly opened mesh. GTEST_SKIP() in SetUpTestSuite does not skip the tests, so the suite opens the
// mesh only if the system can hold it, and SetUp skips each test otherwise.
class Mesh1x2SuiteFixture : public ::testing::Test {
protected:
    static tt::tt_metal::distributed::MeshShape mesh_shape() {
        return tt::tt_metal::distributed::MeshShape(1, 2);
    }

    static void SetUpTestSuite() {
        if (system_supports_mesh(mesh_shape())) {
            ttml::ttnn_fixed::distributed::enable_fabric(static_cast<uint32_t>(mesh_shape().mesh_size()));
            ttml::autograd::ctx().open_device(mesh_shape());
        }
    }

    static void TearDownTestSuite() {
        if (system_supports_mesh(mesh_shape())) {
            ttml::autograd::ctx().close_device();
        }
    }

    void SetUp() override {
        SKIP_UNLESS_MESH_SUPPORTED(mesh_shape());
        ttml::autograd::ctx().get_device().clear_program_cache();
    }
};

}  // namespace ttml::test_utils
