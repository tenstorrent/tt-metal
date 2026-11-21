// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_type_registry.hpp"

namespace tt::tt_fabric::manifest {
namespace {

TEST(ManifestTypeRegistry, NamesATypeWithoutItsNamespaces) {
    EXPECT_EQ(get_manifest_type_name<EDMStatus>(), "EDMStatus");
    EXPECT_EQ(get_manifest_type_name<connection_interface::ConnectionValue>(), "ConnectionValue");
    EXPECT_EQ(get_manifest_type_name<erisc::datamover::handshake::handshake_info_t>(), "handshake_info_t");
    EXPECT_EQ(get_manifest_type_name<FabricTelemetry>(), "FabricTelemetry");
}

TEST(ManifestTypeRegistry, NamesAPacketHeaderWithItsArgumentEvenWhenItIsTheDefault) {
    // The compiler drops an argument that equals the template's default: 36 and 0 are the defaults here.
    EXPECT_EQ(get_manifest_type_name<HybridMeshPacketHeaderT<36>>(), "HybridMeshPacketHeaderT<36>");
    EXPECT_EQ(get_manifest_type_name<HybridMeshPacketHeaderT<68>>(), "HybridMeshPacketHeaderT<68>");
    EXPECT_EQ(get_manifest_type_name<LowLatencyPacketHeaderT<0>>(), "LowLatencyPacketHeaderT<0>");
    EXPECT_EQ(get_manifest_type_name<LowLatencyPacketHeaderT<3>>(), "LowLatencyPacketHeaderT<3>");
}

// A manifest read back names its types as text, which must be a registered type's name exactly.
TEST(ManifestTypeRegistry, KnowsEachRegisteredNameAndNoOther) {
    // A registered type, by the name the writer gives it: no namespaces.
    EXPECT_TRUE(is_registered_type_name("EDMStatus"));
    // A registered template, with its argument written out even though 0 is the default.
    EXPECT_TRUE(is_registered_type_name("LowLatencyPacketHeaderT<0>"));
    // The same template as the compiler spells it with the default argument dropped. The writer never writes this
    // name, so a manifest holding it wasn't written by this build's writer.
    EXPECT_FALSE(is_registered_type_name("LowLatencyPacketHeaderT<>"));
    // A registered type with its namespaces: names match exactly, and the writer leaves namespaces out.
    EXPECT_FALSE(is_registered_type_name("tt::tt_fabric::EDMStatus"));
    // No type at all.
    EXPECT_FALSE(is_registered_type_name(""));
}

}  // namespace
}  // namespace tt::tt_fabric::manifest
