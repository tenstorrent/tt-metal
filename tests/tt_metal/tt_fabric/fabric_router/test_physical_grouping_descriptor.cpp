// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <filesystem>
#include <vector>
#include <string>

#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>

using namespace tt::tt_fabric;

namespace tt::tt_fabric::fabric_router_tests {

// Helper function to check validation errors contain expected messages
void check_validation_errors(
    const PhysicalGroupingDescriptor::ValidationResult& result,
    const std::vector<std::string>& expected_error_substrings) {
    EXPECT_FALSE(result.is_valid()) << "Expected validation to fail";
    EXPECT_EQ(result.errors.size(), expected_error_substrings.size())
        << "Expected " << expected_error_substrings.size() << " errors, got " << result.errors.size();

    for (const auto& expected_substring : expected_error_substrings) {
        bool found = false;
        for (const auto& error : result.errors) {
            if (error.find(expected_substring) != std::string::npos) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Expected error message containing: " << expected_substring;
    }
}

// ============================================================================
// VALID CONFIGURATION TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, ParsesValidBasicConfiguration) {
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
    )proto";

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        EXPECT_TRUE(desc.get_validation_result().is_valid());
        EXPECT_TRUE(desc.has_grouping("meshes"));
        EXPECT_TRUE(desc.has_grouping("hosts"));
        EXPECT_TRUE(desc.has_grouping("trays"));
    });
}

TEST(PhysicalGroupingDescriptorTests, ParsesValidConfigurationWithPods) {
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
    )proto";

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        EXPECT_TRUE(desc.get_validation_result().is_valid());
        auto pods = desc.get_groupings_by_name("pods");
        EXPECT_EQ(pods.size(), 1);
    });
}

TEST(PhysicalGroupingDescriptorTests, ParsesValidConfigurationWithMultipleDefinitions) {
    const std::string text_proto = R"proto(

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
        EXPECT_TRUE(desc.get_validation_result().is_valid());
        auto halftrays = desc.get_groupings_by_name("halftray");
        EXPECT_EQ(halftrays.size(), 2) << "Should have 2 definitions of halftray";
    });
}

TEST(PhysicalGroupingDescriptorTests, ParsesValidConfigurationWithMixedCounts) {
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
          [ { grouping_ref { grouping_name: "pods" count: 2 } }
            , { grouping_ref { grouping_name: "meshes" count: 3 } }]
        }
    )proto";

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        EXPECT_TRUE(desc.get_validation_result().is_valid());
    });
}

TEST(PhysicalGroupingDescriptorTests, ParsesFromTextProtoFile) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/3_pod_16x8_bh_galaxy_physical_groupings.textproto";
    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto_file_path);
        EXPECT_TRUE(desc.get_validation_result().is_valid());
    });
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
    )proto";

    // This should parse but have warnings
    PhysicalGroupingDescriptor desc(text_proto);
    const auto& result = desc.get_validation_result();
    EXPECT_TRUE(result.is_valid());
    EXPECT_FALSE(result.warnings.empty());

    bool has_trays_warning = false;
    bool has_hosts_warning = false;
    for (const auto& warning : result.warnings) {
        if (warning.find("trays") != std::string::npos) {
            has_trays_warning = true;
        }
        if (warning.find("hosts") != std::string::npos) {
            has_hosts_warning = true;
        }
    }
    EXPECT_TRUE(has_trays_warning || has_hosts_warning);
}

// ============================================================================
// VALIDATION RULE 2: GROUPING REFERENCES
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenReferencingNonExistentGrouping) {
    const std::string text_proto = R"proto(
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("references non-existent grouping")));
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenGroupingRefHasEmptyName) {
    const std::string text_proto = R"proto(
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("empty grouping_name")));
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
            ::testing::HasSubstr("groupings other than meshes must have count >= 2"),
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
            ::testing::HasSubstr("groupings other than meshes must have count >= 2"),
            ::testing::HasSubstr("superpods"))));
}

TEST(PhysicalGroupingDescriptorTests, ValidationSucceedsWhenMeshesHasCountOne) {
    const std::string text_proto = R"proto(

        groupings {
          name: "meshes"
          items:
          [ { grouping_ref { grouping_name: "trays" count: 1 } }]
        }
    )proto";

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        EXPECT_TRUE(desc.get_validation_result().is_valid());
    });
}

// ============================================================================
// VALIDATION RULE 4: GROUPING STRUCTURE
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenGroupingHasNoItems) {
    const std::string text_proto = R"proto(

        TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenItemHasNoAsicLocationOrGroupingRef
        ) { const std::string text_proto = R"proto(
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("must have either asic_location or grouping_ref")));
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenAsicLocationIsUnspecified) {
    const std::string text_proto = R"proto(
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("ASIC_LOCATION_UNSPECIFIED")));
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenAsicLocationIsInvalid) {
    const std::string text_proto = R"proto(
    )proto";

    // This should fail at protobuf parsing level, but let's test what happens
    // Note: Protobuf will reject invalid enum values, so this might fail earlier
    EXPECT_ANY_THROW({ PhysicalGroupingDescriptor desc(text_proto); });
}

TEST(PhysicalGroupingDescriptorTests, ValidationSucceedsWithValidAsicLocations) {
    const std::string text_proto = R"proto(
    )proto";

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        EXPECT_TRUE(desc.get_validation_result().is_valid());
    });
}

// ============================================================================
// API TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, HasGroupingReturnsTrueForExistingGrouping) {
    const std::string text_proto = R"proto(

        groupings {
          name: "pods"
          items:
          [ { grouping_ref { grouping_name: "meshes" count: 2 } }]
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    EXPECT_TRUE(desc.has_grouping("meshes"));
    EXPECT_TRUE(desc.has_grouping("pods"));
    EXPECT_FALSE(desc.has_grouping("nonexistent"));
}

TEST(PhysicalGroupingDescriptorTests, GetGroupingsByNameReturnsAllDefinitions) {
    const std::string text_proto = R"proto(

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
    EXPECT_EQ(meshes[0].items[0].type, PhysicalGroupingDescriptor::GroupingItemInfo::ItemType::GROUPING_REF);
    EXPECT_EQ(meshes[0].items[0].grouping_name, "halftray");
    EXPECT_EQ(meshes[0].items[0].count, 1);
}

TEST(PhysicalGroupingDescriptorTests, GetAllGroupingNamesReturnsAllNames) {
    const std::string text_proto = R"proto(

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

    PhysicalGroupingDescriptor desc(text_proto);
    auto names = desc.get_all_grouping_names();
    EXPECT_EQ(names.size(), 3);
    EXPECT_TRUE(std::find(names.begin(), names.end(), "trays") != names.end());
    EXPECT_TRUE(std::find(names.begin(), names.end(), "meshes") != names.end());
    EXPECT_TRUE(std::find(names.begin(), names.end(), "pods") != names.end());
}

TEST(PhysicalGroupingDescriptorTests, GetGroupingCountReturnsCorrectCount) {
    const std::string text_proto = R"proto(

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

    PhysicalGroupingDescriptor desc(text_proto);
    EXPECT_EQ(desc.get_grouping_count(), 3);
}

TEST(PhysicalGroupingDescriptorTests, GetAllGroupingsReturnsAllGroupings) {
    const std::string text_proto = R"proto(

        groupings {
          name: "pods"
          items:
          [ { grouping_ref { grouping_name: "meshes" count: 2 } }]
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto all_groupings = desc.get_all_groupings();
    EXPECT_EQ(all_groupings.size(), 2);

    bool found_meshes = false;
    bool found_pods = false;
    for (const auto& grouping : all_groupings) {
        if (grouping.name == "meshes") {
            found_meshes = true;
            EXPECT_EQ(grouping.items.size(), 1);
            EXPECT_EQ(grouping.items[0].type, PhysicalGroupingDescriptor::GroupingItemInfo::ItemType::ASIC_LOCATION);
            EXPECT_EQ(grouping.items[0].asic_location, 1);
        } else if (grouping.name == "pods") {
            found_pods = true;
            EXPECT_EQ(grouping.items.size(), 1);
            EXPECT_EQ(grouping.items[0].type, PhysicalGroupingDescriptor::GroupingItemInfo::ItemType::GROUPING_REF);
            EXPECT_EQ(grouping.items[0].grouping_name, "meshes");
            EXPECT_EQ(grouping.items[0].count, 2);
        }
    }
    EXPECT_TRUE(found_meshes);
    EXPECT_TRUE(found_pods);
}

TEST(PhysicalGroupingDescriptorTests, ValidationResultGetReportFormatsCorrectly) {
    const std::string text_proto = R"proto(
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
    )proto";

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        EXPECT_TRUE(desc.get_validation_result().is_valid());
    });
}

TEST(PhysicalGroupingDescriptorTests, ParsesConfigurationWithDirectAsicLocationsInMeshes) {
    const std::string text_proto = R"proto(
    )proto";

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        EXPECT_TRUE(desc.get_validation_result().is_valid());
    });
}

TEST(PhysicalGroupingDescriptorTests, ParsesConfigurationWithMixedAsicLocationsAndGroupingRefs) {
    const std::string text_proto = R"proto(

        groupings {
          name: "meshes"
          items:
          [ { asic_location: ASIC_LOCATION_3 }
            , { grouping_ref { grouping_name: "trays" count: 1 } }]
        }
    )proto";

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        EXPECT_TRUE(desc.get_validation_result().is_valid());
    });
}

}  // namespace tt::tt_fabric::fabric_router_tests
