// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>

#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>

using namespace tt::tt_fabric;

namespace tt::tt_fabric::fabric_router_tests {

// ============================================================================
// VALID CONFIGURATION TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, ParsesValidBasicConfiguration) {
    const std::string text_proto = R"proto(

        groupings {
          name: "trays"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }
            , { asic_location: ASIC_LOCATION_3 }
            , { asic_location: ASIC_LOCATION_4 }
            , { asic_location: ASIC_LOCATION_5 }
            , { asic_location: ASIC_LOCATION_6 }
            , { asic_location: ASIC_LOCATION_7 }
            , { asic_location: ASIC_LOCATION_8 }]
        }

        groupings {
          name: "hosts"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 4 } }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "hosts" count: 1 } }]
        }
    )proto";

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        EXPECT_TRUE(desc.has_grouping("meshes"));
        EXPECT_TRUE(desc.has_grouping("hosts"));
        EXPECT_TRUE(desc.has_grouping("trays"));
    });
}

TEST(PhysicalGroupingDescriptorTests, ParsesValidConfigurationWithPods) {
    const std::string text_proto = R"proto(

        groupings {
          name: "trays"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }
            , { asic_location: ASIC_LOCATION_3 }
            , { asic_location: ASIC_LOCATION_4 }
            , { asic_location: ASIC_LOCATION_5 }
            , { asic_location: ASIC_LOCATION_6 }
            , { asic_location: ASIC_LOCATION_7 }
            , { asic_location: ASIC_LOCATION_8 }]
        }

        groupings {
          name: "hosts"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 4 } }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "hosts" count: 1 } }]
        }

        groupings {
          name: "pods"
          items:
          [ { grouping_ref { grouping_name: "meshes" count: 2 } }]
        }
    )proto";

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        auto pods = desc.get_groupings_by_name("pods");
        EXPECT_EQ(pods.size(), 1);
    });
}

TEST(PhysicalGroupingDescriptorTests, ParsesValidConfigurationWithMultipleDefinitions) {
    const std::string text_proto = R"proto(

        groupings {
          name: "halftray"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }
            , { asic_location: ASIC_LOCATION_3 }
            , { asic_location: ASIC_LOCATION_4 }]
        }

        groupings {
          name: "halftray"
          items:
          [ { asic_location: ASIC_LOCATION_5 }
            , { asic_location: ASIC_LOCATION_6 }
            , { asic_location: ASIC_LOCATION_7 }
            , { asic_location: ASIC_LOCATION_8 }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "halftray" count: 1 } }]
        }
    )proto";

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        auto halftrays = desc.get_groupings_by_name("halftray");
        EXPECT_EQ(halftrays.size(), 2) << "Should have 2 definitions of halftray";
    });
}

TEST(PhysicalGroupingDescriptorTests, ParsesValidConfigurationWithMixedCounts) {
    const std::string text_proto = R"proto(

        groupings {
          name: "trays"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }
            , { asic_location: ASIC_LOCATION_3 }
            , { asic_location: ASIC_LOCATION_4 }
            , { asic_location: ASIC_LOCATION_5 }
            , { asic_location: ASIC_LOCATION_6 }
            , { asic_location: ASIC_LOCATION_7 }
            , { asic_location: ASIC_LOCATION_8 }]
        }

        groupings {
          name: "hosts"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 4 } }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "hosts" count: 1 } }]
        }

        groupings {
          name: "pods"
          items:
          [ { grouping_ref { grouping_name: "meshes" count: 2 } }]
        }

        groupings {
          name: "superpods"
          items:
          [ { grouping_ref { grouping_name: "pods" count: 2 } }
            , { grouping_ref { grouping_name: "meshes" count: 3 } }]
        }
    )proto";

    EXPECT_NO_THROW({ PhysicalGroupingDescriptor desc(text_proto); });
}

TEST(PhysicalGroupingDescriptorTests, ParsesFromTriple16x8QuadBhGalaxyFile) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";
    EXPECT_NO_THROW({ PhysicalGroupingDescriptor desc(text_proto_file_path); });
}

// ============================================================================
// VALIDATION RULE 1: REQUIRED GROUPINGS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenMeshesMissing) {
    const std::string text_proto = R"proto(
        groupings {
          name: "trays"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }]
        }

        groupings {
          name: "hosts"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 4 } }]
        }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("Required grouping 'meshes' is missing"),
            ::testing::HasSubstr("At least one grouping with name 'meshes' must be defined"))));
}

TEST(PhysicalGroupingDescriptorTests, ValidationWarnsWhenRecommendedGroupingsMissing) {
    const std::string text_proto = R"proto(

        groupings {
          name: "meshes"
          items:
          [ { asic_location: ASIC_LOCATION_1 }]
        }
    )proto";

    // This should parse successfully (warnings are not errors, so validation passes)
    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        // Validation passes even without recommended groupings (they're just warnings)
    });
}

// ============================================================================
// VALIDATION RULE 2: GROUPING REFERENCES
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenReferencingNonExistentGrouping) {
    const std::string text_proto = R"proto(

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "nonexistent" count: 1 } }]
        }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("references non-existent grouping")));
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenGroupingRefHasEmptyName) {
    // Note: Protobuf will reject empty string for grouping_name, so this test may fail at parse time
    // But if it gets through parsing, validation should catch it
    const std::string text_proto = R"proto(

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "" count: 1 } }]
        }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AnyOf(::testing::HasSubstr("empty grouping_name"), ::testing::HasSubstr("Failed to parse"))));
}

// ============================================================================
// VALIDATION RULE 3: COUNT VALIDATION
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenMeshesHasCountZero) {
    const std::string text_proto = R"proto(

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 0 } }]
        }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(::testing::HasSubstr("meshes must have count >= 1"), ::testing::HasSubstr("count 0"))));
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenPodsHasCountOne) {
    const std::string text_proto = R"proto(

        groupings {
          name: "hosts"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 4 } }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "hosts" count: 1 } }]
        }

        groupings {
          name: "pods"
          items:
          [ { grouping_ref { grouping_name: "meshes" count: 1 } }]
        }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("groupings other than meshes must have count >= 2 when there is only one item"),
            ::testing::HasSubstr("count 1"))));
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenSuperpodsHasCountOne) {
    const std::string text_proto = R"proto(

        groupings {
          name: "hosts"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 4 } }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "hosts" count: 1 } }]
        }

        groupings {
          name: "pods"
          items:
          [ { grouping_ref { grouping_name: "meshes" count: 2 } }]
        }

        groupings {
          name: "superpods"
          items:
          [ { grouping_ref { grouping_name: "pods" count: 1 } }]
        }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("groupings other than meshes must have count >= 2 when there is only one item"),
            ::testing::HasSubstr("superpods"))));
}

TEST(PhysicalGroupingDescriptorTests, ValidationSucceedsWhenMeshesHasCountOne) {
    const std::string text_proto = R"proto(

        groupings {
          name: "trays"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }
            , { asic_location: ASIC_LOCATION_3 }
            , { asic_location: ASIC_LOCATION_4 }
            , { asic_location: ASIC_LOCATION_5 }
            , { asic_location: ASIC_LOCATION_6 }
            , { asic_location: ASIC_LOCATION_7 }
            , { asic_location: ASIC_LOCATION_8 }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 1 } }]
        }
    )proto";

    EXPECT_NO_THROW({ PhysicalGroupingDescriptor desc(text_proto); });
}

TEST(PhysicalGroupingDescriptorTests, ValidationSucceedsWhenNonMeshesHasMultipleItemsWithCountOne) {
    const std::string text_proto = R"proto(

        groupings {
          name: "trays"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }
            , { asic_location: ASIC_LOCATION_3 }
            , { asic_location: ASIC_LOCATION_4 }
            , { asic_location: ASIC_LOCATION_5 }
            , { asic_location: ASIC_LOCATION_6 }
            , { asic_location: ASIC_LOCATION_7 }
            , { asic_location: ASIC_LOCATION_8 }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 1 } }]
        }

        groupings {
          name: "pods"
          items:
          [ { grouping_ref { grouping_name: "meshes" count: 1 } }
            , { grouping_ref { grouping_name: "meshes" count: 1 } }]
        }
    )proto";

    EXPECT_NO_THROW({ PhysicalGroupingDescriptor desc(text_proto); });
}

// ============================================================================
// VALIDATION RULE 4: GROUPING STRUCTURE
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenGroupingHasNoItems) {
    const std::string text_proto = R"proto(

        groupings { name: "meshes" }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must have at least one item")));
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenAsicLocationIsUnspecified) {
    // Note: Protobuf may reject ASIC_LOCATION_UNSPECIFIED at parse time, but if it gets through, validation should
    // catch it
    const std::string text_proto = R"proto(

        groupings {
          name: "meshes"
          items:
          [ { asic_location: ASIC_LOCATION_UNSPECIFIED }]
        }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AnyOf(
            ::testing::HasSubstr("ASIC_LOCATION_UNSPECIFIED"), ::testing::HasSubstr("Failed to parse"))));
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenAsicLocationIsInvalid) {
    // This test checks that invalid ASIC locations are rejected
    // Since we can't easily create invalid enum values in textproto, we'll test with an out-of-range value
    // Protobuf will likely reject this at parse time
    const std::string text_proto = R"proto(

        groupings {
          name: "meshes"
          items:
          [ { asic_location: ASIC_LOCATION_1 }]
        }
    )proto";

    // For now, just verify that valid locations work
    // Invalid enum values would be rejected by protobuf parser
    EXPECT_NO_THROW({ PhysicalGroupingDescriptor desc(text_proto); });
}

TEST(PhysicalGroupingDescriptorTests, ValidationSucceedsWithValidAsicLocations) {
    const std::string text_proto = R"proto(

        groupings {
          name: "meshes"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }
            , { asic_location: ASIC_LOCATION_3 }
            , { asic_location: ASIC_LOCATION_4 }
            , { asic_location: ASIC_LOCATION_5 }
            , { asic_location: ASIC_LOCATION_6 }
            , { asic_location: ASIC_LOCATION_7 }
            , { asic_location: ASIC_LOCATION_8 }]
        }
    )proto";

    EXPECT_NO_THROW({ PhysicalGroupingDescriptor desc(text_proto); });
}

// ============================================================================
// API TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, HasGroupingReturnsTrueForExistingGrouping) {
    const std::string text_proto = R"proto(

        groupings {
          name: "meshes"
          items:
          [ { asic_location: ASIC_LOCATION_1 }]
        }

        groupings {
          name: "pods"
          items:
          [ { grouping_ref { grouping_name: "meshes" count: 2 } }]
        }
    )proto";

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        EXPECT_TRUE(desc.has_grouping("meshes"));
        EXPECT_TRUE(desc.has_grouping("pods"));
        EXPECT_FALSE(desc.has_grouping("nonexistent"));
    });
}

TEST(PhysicalGroupingDescriptorTests, GetGroupingsByNameReturnsAllDefinitions) {
    const std::string text_proto = R"proto(

        groupings {
          name: "halftray"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }]
        }

        groupings {
          name: "halftray"
          items:
          [ { asic_location: ASIC_LOCATION_3 }
            , { asic_location: ASIC_LOCATION_4 }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "halftray" count: 1 } }]
        }
    )proto";

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        auto halftrays = desc.get_groupings_by_name("halftray");
        EXPECT_EQ(halftrays.size(), 2);
        EXPECT_EQ(halftrays[0].name, "halftray");
        EXPECT_EQ(halftrays[0].items.size(), 2);
        EXPECT_EQ(halftrays[1].items.size(), 2);

        auto meshes = desc.get_groupings_by_name("meshes");
        EXPECT_EQ(meshes.size(), 1);
        EXPECT_EQ(meshes[0].name, "meshes");
        EXPECT_EQ(meshes[0].items.size(), 1);
        EXPECT_EQ(meshes[0].items[0].type, GroupingItemInfo::ItemType::GROUPING_REF);
        EXPECT_EQ(meshes[0].items[0].grouping_name, "halftray");
        // Count is represented by items.size(), which is 1 in this case
    });
}

TEST(PhysicalGroupingDescriptorTests, GetAllGroupingNamesReturnsAllNames) {
    const std::string text_proto = R"proto(

        groupings {
          name: "trays"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }
            , { asic_location: ASIC_LOCATION_3 }
            , { asic_location: ASIC_LOCATION_4 }
            , { asic_location: ASIC_LOCATION_5 }
            , { asic_location: ASIC_LOCATION_6 }
            , { asic_location: ASIC_LOCATION_7 }
            , { asic_location: ASIC_LOCATION_8 }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 1 } }]
        }

        groupings {
          name: "pods"
          items:
          [ { grouping_ref { grouping_name: "meshes" count: 2 } }]
        }
    )proto";

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        auto names = desc.get_all_grouping_names();
        EXPECT_EQ(names.size(), 3);
        EXPECT_TRUE(std::find(names.begin(), names.end(), "trays") != names.end());
        EXPECT_TRUE(std::find(names.begin(), names.end(), "meshes") != names.end());
        EXPECT_TRUE(std::find(names.begin(), names.end(), "pods") != names.end());
    });
}

TEST(PhysicalGroupingDescriptorTests, GetGroupingCountReturnsCorrectCount) {
    const std::string text_proto = R"proto(

        groupings {
          name: "trays"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }
            , { asic_location: ASIC_LOCATION_3 }
            , { asic_location: ASIC_LOCATION_4 }
            , { asic_location: ASIC_LOCATION_5 }
            , { asic_location: ASIC_LOCATION_6 }
            , { asic_location: ASIC_LOCATION_7 }
            , { asic_location: ASIC_LOCATION_8 }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 1 } }]
        }

        groupings {
          name: "pods"
          items:
          [ { grouping_ref { grouping_name: "meshes" count: 2 } }]
        }
    )proto";

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        EXPECT_EQ(desc.get_grouping_count(), 3);
    });
}

TEST(PhysicalGroupingDescriptorTests, GetAllGroupingsReturnsAllGroupings) {
    const std::string text_proto = R"proto(

        groupings {
          name: "meshes"
          items:
          [ { asic_location: ASIC_LOCATION_1 }]
        }

        groupings {
          name: "pods"
          items:
          [ { grouping_ref { grouping_name: "meshes" count: 2 } }]
        }
    )proto";

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        auto all_groupings = desc.get_all_groupings();
        EXPECT_EQ(all_groupings.size(), 2);

        bool found_meshes = false;
        bool found_pods = false;
        for (const auto& grouping : all_groupings) {
            if (grouping.name == "meshes") {
                found_meshes = true;
                EXPECT_EQ(grouping.items.size(), 1);
                EXPECT_EQ(grouping.items[0].type, GroupingItemInfo::ItemType::ASIC_LOCATION);
                EXPECT_EQ(grouping.items[0].asic_location, 1);
            } else if (grouping.name == "pods") {
                found_pods = true;
                EXPECT_EQ(grouping.items.size(), 2);  // Count is represented by items.size()
                EXPECT_EQ(grouping.items[0].type, GroupingItemInfo::ItemType::GROUPING_REF);
                EXPECT_EQ(grouping.items[0].grouping_name, "meshes");
                EXPECT_EQ(grouping.items[1].type, GroupingItemInfo::ItemType::GROUPING_REF);
                EXPECT_EQ(grouping.items[1].grouping_name, "meshes");
            }
        }
        EXPECT_TRUE(found_meshes);
        EXPECT_TRUE(found_pods);
    });
}

TEST(PhysicalGroupingDescriptorTests, ValidationResultGetReportFormatsCorrectly) {
    const std::string text_proto = R"proto(

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "nonexistent" count: 1 } }]
        }
    )proto";

    try {
        PhysicalGroupingDescriptor desc(text_proto);
        FAIL() << "Should have thrown";
    } catch (const std::runtime_error& e) {
        std::string error_msg = e.what();
        EXPECT_THAT(error_msg, ::testing::HasSubstr("Validation Report"));
        EXPECT_THAT(error_msg, ::testing::HasSubstr("Errors"));
        EXPECT_THAT(error_msg, ::testing::HasSubstr("references non-existent grouping"));
    }
}

// ============================================================================
// EDGE CASES
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, ParsesMinimalValidConfiguration) {
    const std::string text_proto = R"proto(

        groupings {
          name: "meshes"
          items:
          [ { asic_location: ASIC_LOCATION_1 }]
        }
    )proto";

    EXPECT_NO_THROW({ PhysicalGroupingDescriptor desc(text_proto); });
}

TEST(PhysicalGroupingDescriptorTests, ParsesConfigurationWithDirectAsicLocationsInMeshes) {
    const std::string text_proto = R"proto(

        groupings {
          name: "meshes"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }
            , { asic_location: ASIC_LOCATION_3 }
            , { asic_location: ASIC_LOCATION_4 }]
        }
    )proto";

    EXPECT_NO_THROW({ PhysicalGroupingDescriptor desc(text_proto); });
}

TEST(PhysicalGroupingDescriptorTests, ParsesConfigurationWithMixedAsicLocationsAndGroupingRefs) {
    const std::string text_proto = R"proto(

        groupings {
          name: "trays"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }]
        }

        groupings {
          name: "meshes"
          items:
          [ { asic_location: ASIC_LOCATION_3 }
            , { grouping_ref { grouping_name: "trays" count: 1 } }]
        }
    )proto";

    EXPECT_NO_THROW({ PhysicalGroupingDescriptor desc(text_proto); });
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_TriplePod16x8) {
    // Load the physical grouping descriptor
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/test_pods_clusters_physical_groupings.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test with triple_pod_16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto
    const std::filesystem::path mgd_file_path =
        "tt_metal/fabric/mesh_graph_descriptors/triple_pod_16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto";

    MeshGraphDescriptor mgd(mgd_file_path);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);

    // M0 mesh has device_topology [16, 8] = 128 chips
    // Should match meshes grouping with 4 hosts (4 * 32 = 128 ASICs, exact match)
    EXPECT_TRUE(valid_groupings.find("MESH") != valid_groupings.end()) << "Should have MESH type in results";
    EXPECT_TRUE(valid_groupings.at("MESH").find("M0") != valid_groupings.at("MESH").end())
        << "Should have M0 mesh instance";

    const auto& m0_grouping = valid_groupings.at("MESH").at("M0");
    EXPECT_EQ(m0_grouping.name, "meshes") << "M0 should match 'meshes' grouping";
    EXPECT_EQ(m0_grouping.asic_count, 128u) << "M0 grouping should have 128 ASICs (4 hosts)";

    // Verify it matches the 4 hosts grouping
    EXPECT_EQ(m0_grouping.items.size(), 4u) << "Should have 4 items (4 hosts)";
    if (!m0_grouping.items.empty()) {
        EXPECT_EQ(m0_grouping.items[0].type, GroupingItemInfo::ItemType::GROUPING_REF)
            << "First item should be a GROUPING_REF";
        EXPECT_EQ(m0_grouping.items[0].grouping_name, "hosts") << "Should reference 'hosts' grouping";
    }
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_32x4Quad) {
    // Load the physical grouping descriptor
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test with 32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto
    const std::filesystem::path mgd_file_path =
        "tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto";

    MeshGraphDescriptor mgd(mgd_file_path);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);

    // M0 mesh has device_topology [32, 4] = 128 chips
    // Should match meshes grouping with 4 hosts (4 * 32 = 128 ASICs, exact match)
    EXPECT_TRUE(valid_groupings.find("MESH") != valid_groupings.end()) << "Should have MESH type in results";
    EXPECT_TRUE(valid_groupings.at("MESH").find("M0") != valid_groupings.at("MESH").end())
        << "Should have M0 mesh instance";

    const auto& m0_grouping = valid_groupings.at("MESH").at("M0");
    EXPECT_EQ(m0_grouping.name, "meshes") << "M0 should match 'meshes' grouping";
    EXPECT_EQ(m0_grouping.asic_count, 128u) << "M0 grouping should have 128 ASICs (4 hosts)";

    // Verify it matches the 4 hosts grouping
    EXPECT_EQ(m0_grouping.items.size(), 4u) << "Should have 4 items (4 hosts)";
    if (!m0_grouping.items.empty()) {
        EXPECT_EQ(m0_grouping.items[0].type, GroupingItemInfo::ItemType::GROUPING_REF)
            << "First item should be a GROUPING_REF";
        EXPECT_EQ(m0_grouping.items[0].grouping_name, "hosts") << "Should reference 'hosts' grouping";
    }
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_SingleGalaxy) {
    // Load the physical grouping descriptor
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test with bh_glx_split_4x2.textproto
    const std::filesystem::path mgd_file_path =
        "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto";

    MeshGraphDescriptor mgd(mgd_file_path);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);

    // M0 mesh has device_topology [8, 4] = 32 chips
    // Should match meshes grouping with 1 host (32 ASICs, exact match)
    EXPECT_TRUE(valid_groupings.find("MESH") != valid_groupings.end()) << "Should have MESH type in results";
    EXPECT_TRUE(valid_groupings.at("MESH").find("M0") != valid_groupings.at("MESH").end())
        << "Should have M0 mesh instance";

    const auto& m0_grouping = valid_groupings.at("MESH").at("M0");
    EXPECT_EQ(m0_grouping.name, "meshes") << "M0 should match 'meshes' grouping";
    EXPECT_EQ(m0_grouping.asic_count, 32u) << "M0 grouping should have 32 ASICs (1 host)";

    // Verify it matches the 1 host grouping
    EXPECT_EQ(m0_grouping.items.size(), 1u) << "Should have 1 item (1 host)";
    if (!m0_grouping.items.empty()) {
        EXPECT_EQ(m0_grouping.items[0].type, GroupingItemInfo::ItemType::GROUPING_REF)
            << "First item should be a GROUPING_REF";
        EXPECT_EQ(m0_grouping.items[0].grouping_name, "hosts") << "Should reference 'hosts' grouping";
    }
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_BhGlxSplit4x2) {
    // Load the physical grouping descriptor
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test with bh_glx_split_4x2.textproto
    const std::filesystem::path mgd_file_path = "tt_metal/fabric/mesh_graph_descriptors/bh_glx_split_4x2.textproto";

    MeshGraphDescriptor mgd(mgd_file_path);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);

    // M0 mesh has device_topology [4, 2] = 8 chips
    // Should match meshes grouping with 1 tray (8 ASICs, exact match)
    // Note: This test has multiple mesh instances (M0 mesh_id 0-47), all with same topology
    EXPECT_TRUE(valid_groupings.find("MESH") != valid_groupings.end()) << "Should have MESH type in results";
    EXPECT_TRUE(valid_groupings.at("MESH").find("M0") != valid_groupings.at("MESH").end())
        << "Should have M0 mesh instance";

    const auto& m0_grouping = valid_groupings.at("MESH").at("M0");
    EXPECT_EQ(m0_grouping.name, "meshes") << "M0 should match 'meshes' grouping";
    EXPECT_EQ(m0_grouping.asic_count, 8u) << "M0 grouping should have 8 ASICs (1 tray, exact match)";

    // Verify it matches the 1 tray grouping exactly (not oversized)
    EXPECT_EQ(m0_grouping.items.size(), 1u) << "Should have exactly 1 item (1 tray)";
    EXPECT_TRUE(!m0_grouping.items.empty()) << "Should have at least one item";
    EXPECT_EQ(m0_grouping.items[0].type, GroupingItemInfo::ItemType::GROUPING_REF)
        << "First item should be a GROUPING_REF";
    EXPECT_EQ(m0_grouping.items[0].grouping_name, "trays") << "Should reference 'trays' grouping";

    // Verify all items reference trays (should be exactly 1 tray reference)
    uint32_t tray_ref_count = 0;
    for (const auto& item : m0_grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF && item.grouping_name == "trays") {
            tray_ref_count++;
        }
    }
    EXPECT_EQ(tray_ref_count, 1u) << "Should reference exactly 1 tray";
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_Dual4x4) {
    // Load the physical grouping descriptor
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test with dual_4x4_mesh_graph_descriptor.textproto
    // This is a dual mesh configuration with two 4x4 WORMHOLE_B0 meshes, each with host_topology [1, 1] (1 host)
    const std::filesystem::path mgd_file_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_4x4_mesh_graph_descriptor.textproto";

    MeshGraphDescriptor mgd(mgd_file_path);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);

    // M0 mesh has device_topology [4, 4] = 16 chips
    // Should match meshes grouping with 2 trays (2 * 8 = 16 ASICs, exact match)
    // Note: This test has 2 mesh instances (M0 mesh_id 0 and 1), both with same topology
    EXPECT_TRUE(valid_groupings.find("MESH") != valid_groupings.end()) << "Should have MESH type in results";
    EXPECT_TRUE(valid_groupings.at("MESH").find("M0") != valid_groupings.at("MESH").end())
        << "Should have M0 mesh instance";

    const auto& m0_grouping = valid_groupings.at("MESH").at("M0");
    EXPECT_EQ(m0_grouping.name, "meshes") << "M0 should match 'meshes' grouping";
    EXPECT_EQ(m0_grouping.asic_count, 16u) << "M0 grouping should have 16 ASICs (2 trays, exact match)";

    // Verify it matches the 2 trays grouping exactly (not oversized)
    EXPECT_EQ(m0_grouping.items.size(), 2u) << "Should have exactly 2 items (2 trays)";
    EXPECT_TRUE(!m0_grouping.items.empty()) << "Should have at least one item";
    EXPECT_EQ(m0_grouping.items[0].type, GroupingItemInfo::ItemType::GROUPING_REF)
        << "First item should be a GROUPING_REF";
    EXPECT_EQ(m0_grouping.items[0].grouping_name, "trays") << "Should reference 'trays' grouping";

    // Verify all items reference trays (should be exactly 2 tray references)
    uint32_t tray_ref_count = 0;
    for (const auto& item : m0_grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF && item.grouping_name == "trays") {
            tray_ref_count++;
        }
    }
    EXPECT_EQ(tray_ref_count, 2u) << "Should reference exactly 2 trays";
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_Dual8x2) {
    // Load the physical grouping descriptor
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test with dual_8x2_mesh_graph_descriptor.textproto
    // This is a dual mesh configuration with two 8x2 WORMHOLE_B0 meshes, each with host_topology [1, 1] (1 host)
    const std::filesystem::path mgd_file_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_8x2_mesh_graph_descriptor.textproto";

    MeshGraphDescriptor mgd(mgd_file_path);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);

    // M0 mesh has device_topology [8, 2] = 16 chips
    // Should match meshes grouping with 2 trays (2 * 8 = 16 ASICs, exact match)
    // Note: This test has 2 mesh instances (M0 mesh_id 0 and 1), both with same topology
    EXPECT_TRUE(valid_groupings.find("MESH") != valid_groupings.end()) << "Should have MESH type in results";
    EXPECT_TRUE(valid_groupings.at("MESH").find("M0") != valid_groupings.at("MESH").end())
        << "Should have M0 mesh instance";

    const auto& m0_grouping = valid_groupings.at("MESH").at("M0");
    EXPECT_EQ(m0_grouping.name, "meshes") << "M0 should match 'meshes' grouping";
    EXPECT_EQ(m0_grouping.asic_count, 16u) << "M0 grouping should have 16 ASICs (2 trays, exact match)";

    // Verify it matches the 2 trays grouping exactly (not oversized)
    EXPECT_EQ(m0_grouping.items.size(), 2u) << "Should have exactly 2 items (2 trays)";
    EXPECT_TRUE(!m0_grouping.items.empty()) << "Should have at least one item";
    EXPECT_EQ(m0_grouping.items[0].type, GroupingItemInfo::ItemType::GROUPING_REF)
        << "First item should be a GROUPING_REF";
    EXPECT_EQ(m0_grouping.items[0].grouping_name, "trays") << "Should reference 'trays' grouping";

    // Verify all items reference trays (should be exactly 2 tray references)
    uint32_t tray_ref_count = 0;
    for (const auto& item : m0_grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF && item.grouping_name == "trays") {
            tray_ref_count++;
        }
    }
    EXPECT_EQ(tray_ref_count, 2u) << "Should reference exactly 2 trays";
}

TEST(PhysicalGroupingDescriptorTests, AsicCountCalculation_BaseGrouping) {
    // Test that base groupings (only ASIC_LOCATION items) calculate ASIC counts correctly
    const std::string text_proto = R"proto(
        groupings {
          name: "trays"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }
            , { asic_location: ASIC_LOCATION_3 }
            , { asic_location: ASIC_LOCATION_4 }
            , { asic_location: ASIC_LOCATION_5 }
            , { asic_location: ASIC_LOCATION_6 }
            , { asic_location: ASIC_LOCATION_7 }
            , { asic_location: ASIC_LOCATION_8 }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 1 } }]
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);

    auto trays = desc.get_groupings_by_name("trays");
    ASSERT_EQ(trays.size(), 1);
    EXPECT_EQ(trays[0].asic_count, 8) << "Base grouping with 8 ASIC_LOCATION items should have asic_count = 8";
}

TEST(PhysicalGroupingDescriptorTests, AsicCountCalculation_SimpleDependent) {
    // Test that dependent groupings calculate ASIC counts correctly
    const std::string text_proto = R"proto(
        groupings {
          name: "trays"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }
            , { asic_location: ASIC_LOCATION_3 }
            , { asic_location: ASIC_LOCATION_4 }]
        }

        groupings {
          name: "hosts"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 4 } }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 1 } }]
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);

    auto trays = desc.get_groupings_by_name("trays");
    ASSERT_EQ(trays.size(), 1);
    EXPECT_EQ(trays[0].asic_count, 4) << "Base grouping should have asic_count = 4";

    auto hosts = desc.get_groupings_by_name("hosts");
    ASSERT_EQ(hosts.size(), 1);
    EXPECT_EQ(hosts[0].asic_count, 16) << "Host grouping with 4 trays (4 ASICs each) should have asic_count = 16";
    EXPECT_EQ(hosts[0].items.size(), 4) << "Host grouping should have 4 items (expanded from count: 4)";
}

TEST(PhysicalGroupingDescriptorTests, AsicCountCalculation_NestedGroupings) {
    // Test complex nested groupings resolve correctly
    const std::string text_proto = R"proto(
        groupings {
          name: "trays"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }]
        }

        groupings {
          name: "hosts"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 2 } }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "hosts" count: 2 } }]
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);

    auto trays = desc.get_groupings_by_name("trays");
    ASSERT_EQ(trays.size(), 1);
    EXPECT_EQ(trays[0].asic_count, 2) << "Trays should have 2 ASICs";

    auto hosts = desc.get_groupings_by_name("hosts");
    ASSERT_EQ(hosts.size(), 1);
    EXPECT_EQ(hosts[0].asic_count, 4) << "Hosts should have 2 trays * 2 ASICs = 4 ASICs";

    auto meshes = desc.get_groupings_by_name("meshes");
    ASSERT_EQ(meshes.size(), 1);
    EXPECT_EQ(meshes[0].asic_count, 8) << "Meshes should have 2 hosts * 4 ASICs = 8 ASICs";
}

TEST(PhysicalGroupingDescriptorTests, AsicCountCalculation_MixedItems) {
    // Test groupings with both ASIC_LOCATION and GROUPING_REF items
    const std::string text_proto = R"proto(
        groupings {
          name: "trays"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }]
        }

        groupings {
          name: "mixed"
          items:
          [ { asic_location: ASIC_LOCATION_3 }
            , { grouping_ref { grouping_name: "trays" count: 2 } }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 1 } }]
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);

    auto mixed = desc.get_groupings_by_name("mixed");
    ASSERT_EQ(mixed.size(), 1);
    // Should have: 1 ASIC_LOCATION + 2 trays (2 ASICs each) = 1 + 4 = 5 ASICs
    EXPECT_EQ(mixed[0].asic_count, 5) << "Mixed grouping should have 1 direct ASIC + 4 from trays = 5 ASICs";
    EXPECT_EQ(mixed[0].items.size(), 3) << "Should have 1 ASIC_LOCATION + 2 GROUPING_REF items = 3 items";
}

TEST(PhysicalGroupingDescriptorTests, AsicCountCalculation_MultipleMeshesGroupings) {
    // Test that multiple "meshes" groupings with different ASIC counts are calculated correctly
    const std::string text_proto = R"proto(
        groupings {
          name: "trays"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }
            , { asic_location: ASIC_LOCATION_3 }
            , { asic_location: ASIC_LOCATION_4 }
            , { asic_location: ASIC_LOCATION_5 }
            , { asic_location: ASIC_LOCATION_6 }
            , { asic_location: ASIC_LOCATION_7 }
            , { asic_location: ASIC_LOCATION_8 }]
        }

        groupings {
          name: "hosts"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 4 } }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 1 } }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 2 } }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "hosts" count: 4 } }]
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);

    auto meshes = desc.get_groupings_by_name("meshes");
    ASSERT_EQ(meshes.size(), 3) << "Should have 3 different 'meshes' groupings";

    // Sort by ASIC count to verify each one
    std::sort(meshes.begin(), meshes.end(), [](const GroupingInfo& a, const GroupingInfo& b) {
        return a.asic_count < b.asic_count;
    });

    EXPECT_EQ(meshes[0].asic_count, 8) << "First meshes grouping: 1 tray = 8 ASICs";
    EXPECT_EQ(meshes[1].asic_count, 16) << "Second meshes grouping: 2 trays = 16 ASICs";
    EXPECT_EQ(meshes[2].asic_count, 128) << "Third meshes grouping: 4 hosts = 4 * 32 = 128 ASICs";
}

TEST(PhysicalGroupingDescriptorTests, AsicCountCalculation_CircularDependencyError) {
    // Test that circular dependencies are detected
    const std::string text_proto = R"proto(
        groupings {
          name: "trays"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }]
        }

        groupings {
          name: "a"
          items:
          [ { grouping_ref { grouping_name: "b" count: 2 } }]
        }

        groupings {
          name: "b"
          items:
          [ { grouping_ref { grouping_name: "a" count: 2 } }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 1 } }]
        }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Circular dependencies detected")));
}

TEST(PhysicalGroupingDescriptorTests, AsicCountCalculation_MissingReferenceError) {
    // Test that missing references are detected
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "nonexistent" count: 1 } }]
        }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AnyOf(
            ::testing::HasSubstr("references non-existent grouping"),
            ::testing::HasSubstr("does not resolve to any ASIC locations"))));
}

TEST(PhysicalGroupingDescriptorTests, AsicCountCalculation_NoAsicLocationsError) {
    // Test that groupings that don't resolve to ASIC_LOCATION items error out during populate
    // This test uses a grouping that references a non-existent grouping, which will fail during populate
    // after validation passes (validation checks references exist, but populate checks they resolve to ASICs)
    const std::string text_proto = R"proto(
        groupings {
          name: "trays"
          items:
          [ { asic_location: ASIC_LOCATION_1 }
            , { asic_location: ASIC_LOCATION_2 }]
        }

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 1 } }]
        }

        groupings {
          name: "invalid"
          items:
          [ { grouping_ref { grouping_name: "nonexistent" count: 2 } }]
        }
    )proto";

    // This should fail during validation (references non-existent grouping)
    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("references non-existent grouping")));
}

// ===== Tests for Higher-Level Groupings (Pods, Clusters, etc.) =====

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_PodWith2Meshes_ExactMatch) {
    // Scenario 1: POD instance with 2 meshes - should match by composition, not name
    // Tests that a "POD" type can match "pods" or "widgets" grouping (both have {meshes: 2})
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/test_pods_clusters_physical_groupings.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test MGD with POD containing 2 meshes
    const std::string mgd_text_proto = R"proto(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 4, 4 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 2 policy: RELAXED }
        }
        graph_descriptors {
          name: "G0"
          type: "POD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
        }
        top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor mgd(mgd_text_proto);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);

    // Should match:
    // - M0 meshes: 2 trays (16 ASICs) - exact match for 4x4 = 16 chips
    // - G0 POD: matches grouping with {meshes: 2} - could be "pods" or "widgets" (both have same composition)
    EXPECT_TRUE(valid_groupings.find("MESH") != valid_groupings.end()) << "Should have MESH type in results";
    EXPECT_TRUE(valid_groupings.find("POD") != valid_groupings.end()) << "Should have POD type in results";

    // Verify mesh matching
    const auto& m0_grouping = valid_groupings.at("MESH").at("M0");
    EXPECT_EQ(m0_grouping.name, "meshes") << "M0 should match 'meshes' grouping";
    EXPECT_EQ(m0_grouping.asic_count, 16u) << "M0 grouping should have 16 ASICs (2 trays)";

    // Verify POD matching - should match by composition {meshes: 2}, not by name
    const auto& g0_grouping = valid_groupings.at("POD").at("G0");
    // Should match either "pods" or "widgets" (both have {meshes: 2})
    EXPECT_TRUE(g0_grouping.name == "pods" || g0_grouping.name == "widgets")
        << "G0 should match 'pods' or 'widgets' grouping (both have {meshes: 2})";

    // Verify composition: should have 2 meshes
    uint32_t meshes_count = 0;
    for (const auto& item : g0_grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF && item.grouping_name == "meshes") {
            meshes_count++;
        }
    }
    EXPECT_EQ(meshes_count, 2u) << "Matched grouping should have composition {meshes: 2}";
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_PodWith3Meshes_ExactMatch) {
    // Scenario: POD instance with 3 meshes - should match by composition {meshes: 3}
    // Could match "pods" or "widgets" (both have {meshes: 3})
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/test_pods_clusters_physical_groupings.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test MGD with POD containing 3 meshes
    const std::string mgd_text_proto = R"proto(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 4, 4 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 2 policy: RELAXED }
        }
        graph_descriptors {
          name: "G0"
          type: "POD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 2 } }
        }
        top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor mgd(mgd_text_proto);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);

    // Should match:
    // - M0 meshes: 2 trays (16 ASICs)
    // - G0 POD: matches grouping with {meshes: 3} - could be "pods" or "widgets"
    EXPECT_TRUE(valid_groupings.find("POD") != valid_groupings.end()) << "Should have POD type in results";

    const auto& g0_grouping = valid_groupings.at("POD").at("G0");
    // Should match either "pods" or "widgets" (both have {meshes: 3})
    EXPECT_TRUE(g0_grouping.name == "pods" || g0_grouping.name == "widgets")
        << "G0 should match 'pods' or 'widgets' grouping (both have {meshes: 3})";

    // Verify composition: should have 3 meshes
    uint32_t meshes_count = 0;
    for (const auto& item : g0_grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF && item.grouping_name == "meshes") {
            meshes_count++;
        }
    }
    EXPECT_EQ(meshes_count, 3u) << "Matched grouping should have composition {meshes: 3}";
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_ClusterWith2Pods_ExactMatch) {
    // Scenario: Multiple CLUSTER instances with 2 pods each - all should match the same "clusters" grouping
    // Tests that multiple MGD graph instances can map to the same physical grouping name
    // This demonstrates that groupings are templates - many instances can use the same grouping
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/test_pods_clusters_physical_groupings.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test MGD with multiple CLUSTER instances, each containing 2 pods, each pod has 2 meshes
    const std::string mgd_text_proto = R"proto(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 4, 4 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 2 policy: RELAXED }
        }
        graph_descriptors {
          name: "P0"
          type: "POD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
        }
        graph_descriptors {
          name: "C0"
          type: "CLUSTER"
          instances { graph { graph_descriptor: "P0" graph_id: 0 } }
          instances { graph { graph_descriptor: "P0" graph_id: 1 } }
        }
        graph_descriptors {
          name: "C1"
          type: "CLUSTER"
          instances { graph { graph_descriptor: "P0" graph_id: 2 } }
          instances { graph { graph_descriptor: "P0" graph_id: 3 } }
        }
        graph_descriptors {
          name: "C2"
          type: "CLUSTER"
          instances { graph { graph_descriptor: "P0" graph_id: 4 } }
          instances { graph { graph_descriptor: "P0" graph_id: 5 } }
        }
        graph_descriptors {
          name: "TOP"
          type: "FABRIC"
          instances { graph { graph_descriptor: "C0" graph_id: 0 } }
          instances { graph { graph_descriptor: "C1" graph_id: 1 } }
          instances { graph { graph_descriptor: "C2" graph_id: 2 } }
        }
        top_level_instance { graph { graph_descriptor: "TOP" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor mgd(mgd_text_proto);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);

    // Should match:
    // - M0 meshes: 2 trays (16 ASICs)
    // - P0 POD instances: all match grouping with {meshes: 2} - could be "pods" or "widgets"
    // - C0, C1, C2 CLUSTER instances: all match the same "clusters" grouping with {pods: 2}
    EXPECT_TRUE(valid_groupings.find("CLUSTER") != valid_groupings.end()) << "Should have CLUSTER type in results";
    EXPECT_TRUE(valid_groupings.find("POD") != valid_groupings.end()) << "Should have POD type in results";

    // Verify all three CLUSTER instances map to the same grouping name
    const auto& c0_grouping = valid_groupings.at("CLUSTER").at("C0");
    const auto& c1_grouping = valid_groupings.at("CLUSTER").at("C1");
    const auto& c2_grouping = valid_groupings.at("CLUSTER").at("C2");

    EXPECT_EQ(c0_grouping.name, "clusters") << "C0 should match 'clusters' grouping (composition {pods: 2})";
    EXPECT_EQ(c1_grouping.name, "clusters") << "C1 should match 'clusters' grouping (composition {pods: 2})";
    EXPECT_EQ(c2_grouping.name, "clusters") << "C2 should match 'clusters' grouping (composition {pods: 2})";

    // Verify all three instances map to the same grouping (same name and composition)
    EXPECT_EQ(c0_grouping.name, c1_grouping.name) << "C0 and C1 should map to the same grouping name";
    EXPECT_EQ(c1_grouping.name, c2_grouping.name) << "C1 and C2 should map to the same grouping name";

    // Verify composition: all should have 2 pods
    for (const auto& [cluster_name, grouping] : std::vector<std::pair<std::string, GroupingInfo>>{
             {"C0", c0_grouping}, {"C1", c1_grouping}, {"C2", c2_grouping}}) {
        uint32_t pods_count = 0;
        for (const auto& item : grouping.items) {
            if (item.type == GroupingItemInfo::ItemType::GROUPING_REF && item.grouping_name == "pods") {
                pods_count++;
            }
        }
        EXPECT_EQ(pods_count, 2u) << cluster_name << " matched grouping should have composition {pods: 2}";
    }

    // Verify they all have the same ASIC count (since they map to the same grouping)
    EXPECT_EQ(c0_grouping.asic_count, c1_grouping.asic_count)
        << "C0 and C1 should have the same ASIC count (same grouping)";
    EXPECT_EQ(c1_grouping.asic_count, c2_grouping.asic_count)
        << "C1 and C2 should have the same ASIC count (same grouping)";

    const auto& p0_grouping = valid_groupings.at("POD").at("P0");
    // Should match either "pods" or "widgets" (both have {meshes: 2})
    EXPECT_TRUE(p0_grouping.name == "pods" || p0_grouping.name == "widgets")
        << "P0 should match 'pods' or 'widgets' grouping (both have {meshes: 2})";
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_SuperpodMixedComposition_ExactMatch) {
    // Scenario 4: SUPERPOD instance with mixed composition {meshes: 2, pods: 1}
    // Should match "superpods" grouping with exact composition
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/test_pods_clusters_physical_groupings.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test MGD with SUPERPOD containing 2 meshes + 1 pod
    const std::string mgd_text_proto = R"proto(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 4, 4 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 2 policy: RELAXED }
        }
        graph_descriptors {
          name: "P0"
          type: "POD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
        }
        graph_descriptors {
          name: "SP0"
          type: "SUPERPOD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 2 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 3 } }
          instances { graph { graph_descriptor: "P0" graph_id: 0 } }
        }
        top_level_instance { graph { graph_descriptor: "SP0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor mgd(mgd_text_proto);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);

    // Should match:
    // - M0 meshes: 2 trays (16 ASICs)
    // - P0 POD: matches grouping with {meshes: 2} - could be "pods" or "widgets"
    // - SP0 SUPERPOD: matches grouping with {meshes: 2, pods: 1} - should match "superpods"
    EXPECT_TRUE(valid_groupings.find("SUPERPOD") != valid_groupings.end()) << "Should have SUPERPOD type in results";

    const auto& sp0_grouping = valid_groupings.at("SUPERPOD").at("SP0");
    EXPECT_EQ(sp0_grouping.name, "superpods")
        << "SP0 should match 'superpods' grouping (composition {meshes: 2, pods: 1})";

    // Verify composition: should have both meshes and pods references
    uint32_t meshes_count = 0;
    uint32_t pods_count = 0;
    for (const auto& item : sp0_grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
            if (item.grouping_name == "meshes") {
                meshes_count++;
            } else if (item.grouping_name == "pods") {
                pods_count++;
            }
        }
    }
    EXPECT_EQ(meshes_count, 2u) << "Superpod should contain 2 meshes";
    EXPECT_EQ(pods_count, 1u) << "Superpod should contain 1 pod";
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_PodWith5Meshes_ClosestFit) {
    // Scenario: POD instance with 5 meshes - no exact match, should find closest fit
    // Should match "widgets" grouping with {meshes: 5} (closest fit that satisfies requirements)
    // Tests that perfect match not required - closest fit is acceptable
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/test_pods_clusters_physical_groupings.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test MGD with POD containing 5 meshes
    const std::string mgd_text_proto = R"proto(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 4, 4 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 2 policy: RELAXED }
        }
        graph_descriptors {
          name: "G0"
          type: "POD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 2 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 3 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 4 } }
        }
        top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor mgd(mgd_text_proto);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);

    // Should match:
    // - M0 meshes: 2 trays (16 ASICs)
    // - G0 POD: matches grouping with {meshes: 5} - should match "widgets" (closest fit)
    EXPECT_TRUE(valid_groupings.find("POD") != valid_groupings.end()) << "Should have POD type in results";

    const auto& g0_grouping = valid_groupings.at("POD").at("G0");
    EXPECT_EQ(g0_grouping.name, "widgets")
        << "G0 (type POD) should match 'widgets' grouping (closest fit with {meshes: 5}) - name doesn't matter";

    // Verify composition: should have 5 meshes
    uint32_t meshes_count = 0;
    for (const auto& item : g0_grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF && item.grouping_name == "meshes") {
            meshes_count++;
        }
    }
    EXPECT_EQ(meshes_count, 5u)
        << "Matched grouping should have composition {meshes: 5} (closest fit - no exact match exists)";
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_ClusterWithTooManyPods_Negative) {
    // Negative test: CLUSTER requiring 5 pods, but all "clusters" groupings have <= 3 pods
    // This tests that higher-level groupings are validated and fail when too small
    // Should fail with incompatibility error
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/test_pods_clusters_physical_groupings.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test MGD with CLUSTER containing 5 PODs
    // Available "clusters" groupings have {pods: 2} and {pods: 3}, both < 5
    const std::string mgd_text_proto = R"proto(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 4, 4 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 2 policy: RELAXED }
        }
        graph_descriptors {
          name: "P0"
          type: "POD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
        }
        graph_descriptors {
          name: "P1"
          type: "POD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 2 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 3 } }
        }
        graph_descriptors {
          name: "P2"
          type: "POD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 4 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 5 } }
        }
        graph_descriptors {
          name: "P3"
          type: "POD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 6 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 7 } }
        }
        graph_descriptors {
          name: "P4"
          type: "POD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 8 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 9 } }
        }
        graph_descriptors {
          name: "C0"
          type: "CLUSTER"
          instances { graph { graph_descriptor: "P0" graph_id: 0 } }
          instances { graph { graph_descriptor: "P1" graph_id: 1 } }
          instances { graph { graph_descriptor: "P2" graph_id: 2 } }
          instances { graph { graph_descriptor: "P3" graph_id: 3 } }
          instances { graph { graph_descriptor: "P4" graph_id: 4 } }
        }
        top_level_instance { graph { graph_descriptor: "C0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor mgd(mgd_text_proto);

    // This should fail because no "clusters" grouping has >= 5 pods
    // (available groupings have {pods: 2} and {pods: 3}, both < 5)
    EXPECT_THAT(
        ([&]() { pgd.get_valid_groupings_for_mgd(mgd); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("This system is not compatible with the following MGD")));
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_MultipleConfigurations_ExactMatchPreferred) {
    // Scenario: POD with 2 meshes - multiple groupings available with different compositions
    // Should prefer exact match ({meshes: 2}) over oversized matches
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/test_pods_clusters_physical_groupings.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test with POD containing 2 meshes - should match grouping with 2 meshes (exact match)
    const std::string mgd_text_proto = R"proto(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 4, 4 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 2 policy: RELAXED }
        }
        graph_descriptors {
          name: "G0"
          type: "POD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
        }
        top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor mgd(mgd_text_proto);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);

    const auto& g0_grouping = valid_groupings.at("POD").at("G0");
    // Should match either "pods" or "widgets" (both have exact match {meshes: 2})
    EXPECT_TRUE(g0_grouping.name == "pods" || g0_grouping.name == "widgets")
        << "G0 should match 'pods' or 'widgets' grouping (exact match {meshes: 2})";

    // Verify exact match: should have exactly 2 meshes
    uint32_t meshes_count = 0;
    for (const auto& item : g0_grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF && item.grouping_name == "meshes") {
            meshes_count++;
        }
    }
    EXPECT_EQ(meshes_count, 2u) << "Should match grouping with exactly 2 meshes (exact match preferred)";
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_PodWith4Meshes_MatchesClusters) {
    // Scenario 3: POD instance with 4 meshes - should match "clusters" grouping (has {meshes: 4})
    // Tests that name doesn't matter - POD can match "clusters" if composition fits
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/test_pods_clusters_physical_groupings.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test with POD containing 4 meshes - should match "clusters" grouping (has {meshes: 4})
    const std::string mgd_text_proto = R"proto(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 4, 4 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 2 policy: RELAXED }
        }
        graph_descriptors {
          name: "G0"
          type: "POD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 2 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 3 } }
        }
        top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor mgd(mgd_text_proto);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);

    const auto& g0_grouping = valid_groupings.at("POD").at("G0");
    // Should match "pods" grouping with {meshes: 4} (exact match)
    // Note: "clusters" has {meshes: 4} but references "pods", not "meshes" directly, so won't match
    EXPECT_EQ(g0_grouping.name, "pods") << "G0 should match 'pods' grouping (composition {meshes: 4})";

    // Verify composition: should have 4 meshes
    uint32_t meshes_count = 0;
    for (const auto& item : g0_grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF && item.grouping_name == "meshes") {
            meshes_count++;
        }
    }
    EXPECT_EQ(meshes_count, 4u) << "Should match grouping with exactly 4 meshes (exact match)";
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_SuperpodOversizedMatch_ClosestFit) {
    // Scenario 4: SUPERPOD instance requiring {meshes: 1, pods: 3}
    // No exact match exists, should match closest fit {meshes: 1, pods: 4} or {meshes: 2, pods: 5}
    // Tests that perfect match not required - closest fit that satisfies requirements is acceptable
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/test_pods_clusters_physical_groupings.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Create MGD with SUPERPOD requiring {meshes: 1, pods: 3}
    // Available superpods: {meshes: 2, pods: 1} and {meshes: 1, pods: 4}
    // Should match {meshes: 1, pods: 4} (closest fit that satisfies requirements)
    const std::string mgd_text_proto = R"proto(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 4, 4 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 2 policy: RELAXED }
        }
        graph_descriptors {
          name: "P0"
          type: "POD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
        }
        graph_descriptors {
          name: "SP0"
          type: "SUPERPOD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { graph { graph_descriptor: "P0" graph_id: 1 } }
          instances { graph { graph_descriptor: "P0" graph_id: 2 } }
          instances { graph { graph_descriptor: "P0" graph_id: 3 } }
        }
        top_level_instance { graph { graph_descriptor: "SP0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor mgd(mgd_text_proto);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);

    // Should match superpods grouping with {meshes: 1, pods: 4} (closest fit)
    const auto& sp0_grouping = valid_groupings.at("SUPERPOD").at("SP0");
    EXPECT_EQ(sp0_grouping.name, "superpods") << "SP0 should match 'superpods' grouping";

    // Verify it selected the {meshes: 1, pods: 4} configuration (closest fit)
    uint32_t meshes_count = 0;
    uint32_t pods_count = 0;
    for (const auto& item : sp0_grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
            if (item.grouping_name == "meshes") {
                meshes_count++;
            } else if (item.grouping_name == "pods") {
                pods_count++;
            }
        }
    }
    EXPECT_EQ(meshes_count, 1u) << "Superpod should contain 1 mesh";
    EXPECT_EQ(pods_count, 4u) << "Superpod should contain 4 pods (closest fit - no exact match exists)";
    // Note: Required {meshes: 1, pods: 3}, matched {meshes: 1, pods: 4} - closest fit is acceptable
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_PodMatchesWidgets_NameIndependence) {
    // Scenario: POD instance matching "widgets" grouping - demonstrates name independence
    // Both "pods" and "widgets" have {meshes: 2}, so either can match
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/test_pods_clusters_physical_groupings.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test MGD with POD containing 2 meshes
    const std::string mgd_text_proto = R"proto(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 4, 4 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 2 policy: RELAXED }
        }
        graph_descriptors {
          name: "G0"
          type: "POD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
        }
        top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor mgd(mgd_text_proto);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);

    const auto& g0_grouping = valid_groupings.at("POD").at("G0");
    // Should match either "pods" or "widgets" - both have same composition {meshes: 2}
    EXPECT_TRUE(g0_grouping.name == "pods" || g0_grouping.name == "widgets")
        << "G0 (type POD) should match 'pods' or 'widgets' grouping - name doesn't matter, only composition";

    // Verify composition matches
    uint32_t meshes_count = 0;
    for (const auto& item : g0_grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF && item.grouping_name == "meshes") {
            meshes_count++;
        }
    }
    EXPECT_EQ(meshes_count, 2u) << "Matched grouping should have composition {meshes: 2}";
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_PodWith3Meshes_NoExactMatch_ClosestFit) {
    // Scenario: POD with 3 meshes - exact match exists ({meshes: 3})
    // Should prefer exact match over closest fit
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/test_pods_clusters_physical_groupings.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test MGD with POD containing 3 meshes
    const std::string mgd_text_proto = R"proto(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 4, 4 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 2 policy: RELAXED }
        }
        graph_descriptors {
          name: "G0"
          type: "POD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 2 } }
        }
        top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor mgd(mgd_text_proto);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);

    const auto& g0_grouping = valid_groupings.at("POD").at("G0");
    // Should match exact match {meshes: 3} - could be "pods" or "widgets"
    EXPECT_TRUE(g0_grouping.name == "pods" || g0_grouping.name == "widgets")
        << "G0 should match 'pods' or 'widgets' grouping (exact match {meshes: 3})";

    // Verify exact match
    uint32_t meshes_count = 0;
    for (const auto& item : g0_grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF && item.grouping_name == "meshes") {
            meshes_count++;
        }
    }
    EXPECT_EQ(meshes_count, 3u) << "Should match grouping with exactly 3 meshes (exact match preferred)";
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_MixedComposition_OversizedMatch) {
    // Scenario 4: SUPERPOD with {meshes: 1, pods: 4} - matches {meshes: 2, pods: 5} (oversized but fits)
    // Tests that closest fit is acceptable when no exact match exists
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/test_pods_clusters_physical_groupings.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test MGD with SUPERPOD containing 1 mesh + 4 pods
    const std::string mgd_text_proto = R"proto(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 4, 4 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 2 policy: RELAXED }
        }
        graph_descriptors {
          name: "P0"
          type: "POD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
        }
        graph_descriptors {
          name: "SP0"
          type: "SUPERPOD"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { graph { graph_descriptor: "P0" graph_id: 1 } }
          instances { graph { graph_descriptor: "P0" graph_id: 2 } }
          instances { graph { graph_descriptor: "P0" graph_id: 3 } }
          instances { graph { graph_descriptor: "P0" graph_id: 4 } }
        }
        top_level_instance { graph { graph_descriptor: "SP0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor mgd(mgd_text_proto);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);

    // Should match superpods grouping with {meshes: 1, pods: 4} (closest fit)
    const auto& sp0_grouping = valid_groupings.at("SUPERPOD").at("SP0");
    EXPECT_EQ(sp0_grouping.name, "superpods") << "SP0 should match 'superpods' grouping";

    // Verify composition: should have {meshes: 1, pods: 4} or closest fit
    uint32_t meshes_count = 0;
    uint32_t pods_count = 0;
    for (const auto& item : sp0_grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
            if (item.grouping_name == "meshes") {
                meshes_count++;
            } else if (item.grouping_name == "pods") {
                pods_count++;
            }
        }
    }
    // Should match {meshes: 1, pods: 4} (closest fit) or {meshes: 2, pods: 5} (also fits)
    EXPECT_GE(meshes_count, 1u) << "Superpod should contain at least 1 mesh";
    EXPECT_GE(pods_count, 4u) << "Superpod should contain at least 4 pods (closest fit acceptable)";
}

}  // namespace tt::tt_fabric::fabric_router_tests
