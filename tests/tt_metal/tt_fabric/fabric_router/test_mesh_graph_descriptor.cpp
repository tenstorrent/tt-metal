// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <filesystem>
#include <vector>
#include <string>
#include <cstdio>

#include <tt-metalium/mesh_graph_descriptor.hpp>

namespace tt::tt_fabric::fabric_router_tests {

// Helper function to print unexpected error messages for debugging
void print_unexpected_errors(const std::string& stdout_output, const std::vector<std::string>& expected_errors) {
    printf("\n=== UNEXPECTED ERROR MESSAGES ===\n");
    printf("Captured stdout output:\n");
    printf("%s\n", stdout_output.c_str());
    
    printf("\nExpected error messages:\n");
    for (const auto& expected : expected_errors) {
        printf("- %s\n", expected.c_str());
    }
    
    printf("\nChecking if any expected errors were found:\n");
    bool found_any = false;
    for (const auto& expected : expected_errors) {
        if (stdout_output.find(expected) != std::string::npos) {
            printf("✓ FOUND: %s\n", expected.c_str());
            found_any = true;
        } else {
            printf("✗ NOT FOUND: %s\n", expected.c_str());
        }
    }
    
    if (!found_any) {
        printf("\n NONE of the expected error messages were found in the output!\n");
    }
    printf("================================\n\n");
}

// Helper function to check if all expected error messages are present in stdout
bool check_all_expected_errors_found(const std::string& stdout_output, const std::vector<std::string>& expected_errors) {
    for (const auto& expected : expected_errors) {
        if (stdout_output.find(expected) == std::string::npos) {
            return false;
        }
    }
    return true;
}

TEST(MeshGraphDescriptorTests, ParsesFromTextProtoString) {
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          dfsdadf: 3  # Allowing unknown fields for backwards compatibility
          arch: WORMHOLE_B0
          device_topology: {
            dims: [ 1, 4 ]
            dim_types: [ LINE, RING ]
          }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 4 ] }
          express_connections: { src: 0 dst: 1 }
          express_connections: { src: 1 dst: 2 }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" id: 0 } }
    )proto";

    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto));
}

TEST(MeshGraphDescriptorTests, ParsesFromTextProtoFile) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/mgd2_syntax_check_mesh_graph_descriptor.textproto";
    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto_file_path));
}

TEST(MeshGraphDescriptorTests, InvalidProtoNoMeshDescriptors) {
    std::string text_proto = R"proto(
        top_level_instance: { mesh: { mesh_descriptor: "M0" id: 0 } }
    )proto";

    // Capture stdout to check for log_error messages
    testing::internal::CaptureStdout();
    
    try {
        MeshGraphDescriptor desc(text_proto);
        FAIL() << "Expected std::runtime_error for no mesh descriptors";
            } catch (const std::runtime_error& e) {
            std::string stdout_output = testing::internal::GetCapturedStdout();
            EXPECT_THAT(e.what(), ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"));
            
            std::vector<std::string> expected_errors = {
                "MeshGraphDescriptor: There must be at least one mesh descriptor"
            };
            
            bool found_expected = check_all_expected_errors_found(stdout_output, expected_errors);
            
            if (!found_expected) {
                print_unexpected_errors(stdout_output, expected_errors);
            }
            
            EXPECT_TRUE(found_expected) << "Expected error message not found in output";
        }
}

TEST(MeshGraphDescriptorTests, InvalidProtoDimensionValidationFailures) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          # missing name - will cause protobuf parsing issues
          arch: WORMHOLE_B0
          device_topology: {
            dims: [ 1, 4, 3 ]  # 3D for WORMHOLE_B0 (max 2D)
            dim_types: [ LINE ]  # size mismatch with dims
          }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 2 ] }  # dimension mismatch with device topology
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" id: 0 } }
    )proto";

    // Capture stdout to check for log_error messages
    testing::internal::CaptureStdout();
    
    try {
        MeshGraphDescriptor desc(text_proto);
        FAIL() << "Expected std::runtime_error for multiple validation failures";
    } catch (const std::runtime_error& e) {
        std::string stdout_output = testing::internal::GetCapturedStdout();
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"));
        
        std::vector<std::string> expected_errors = {
            "MeshGraphDescriptor: Mesh descriptor name cannot be empty",
            "MeshGraphDescriptor: Device and host topology dimensions must be the same size",
            "MeshGraphDescriptor: Architecture::WORMHOLE_B0 architecture devices allow a maximum of 2 dimensions, but 3 were provided"
        };
        
        // Check that all expected validation errors are present in stdout
        bool all_found = check_all_expected_errors_found(stdout_output, expected_errors);
        
        if (!all_found) {
            print_unexpected_errors(stdout_output, expected_errors);
        }
        
        EXPECT_TRUE(all_found) << "Not all expected error messages were found in stdout";
    }
}

TEST(MeshGraphDescriptorTests, InvalidProtoMeshDescriptorValidation) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 3 ] }  # 3 devices total (0,1,2)
          channels: { count: 1 }
          host_topology: { dims: [ 1, 3 ] }
          express_connections: { src: 5 dst: 1 }   # src out of bounds
          express_connections: { src: 1 dst: 5 }   # dst out of bounds
        }

        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ -11, 3 ] }  # 3 devices total (0,1,2)
          channels: { count: 1 }
        }

        mesh_descriptors: {
          name: "M1"
          express_connections: { src: -1 dst: 1 }  # negative src
          express_connections: { src: 1 dst: -1 }  # negative dst
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    // Capture stdout to check for log_error messages
    testing::internal::CaptureStdout();
    
    try {
        MeshGraphDescriptor desc(text_proto);
        FAIL() << "Expected std::runtime_error for multiple express connection validation failures";
    } catch (const std::runtime_error& e) {
        std::string stdout_output = testing::internal::GetCapturedStdout();
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"));
        
        std::vector<std::string> expected_errors = {
            "MeshGraphDescriptor: Express connection source is out of bounds for mesh 'M0'",
            "MeshGraphDescriptor: Express connection destination is out of bounds for mesh 'M0'",
            "MeshGraphDescriptor: Express connection source is out of bounds for mesh 'M1'",
            "MeshGraphDescriptor: Express connection destination is out of bounds for mesh 'M1'",
            "MeshGraphDescriptor: Device topology dimensions must be positive",
            "MeshGraphDescriptor: Mesh descriptor 'M1' must have a valid architecture",
            "MeshGraphDescriptor: Mesh descriptor 'M1' must have device topology with dimensions",
            "MeshGraphDescriptor: Mesh descriptor 'M1' must have host topology with dimensions"
        };
        
        // Check that all expected express connection validation errors are present in stdout
        bool all_found = check_all_expected_errors_found(stdout_output, expected_errors);
        
        if (!all_found) {
            print_unexpected_errors(stdout_output, expected_errors);
        }
        
        EXPECT_TRUE(all_found) << "Not all expected error messages were found in stdout";
    }
}

TEST(MeshGraphDescriptorTests, InvalidProtoGraphDescriptorValidation) {
    std::string text_proto = R"proto(
        graph_descriptors: {
            name: "G0"
            connections: {
                nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
                nodes: { mesh: { mesh_descriptor: "M1" mesh_id: 0 } }
                channels: { count: -1 }
                directional: false
            }
            connections: {
                nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
                channels: { count: 4 }
                directional: false
            }
        }

        graph_descriptors: {
            name: "G0"
            instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
            instances: { mesh: { mesh_descriptor: "M1" mesh_id: 0 } }
            graph_topology: {
                layout_type: ALL_TO_ALL
                channels: { count: -1 }
            }
        }

        graph_descriptors: {
            name: "G1"
            instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
            instances: { mesh: { mesh_descriptor: "M1" mesh_id: 0 } }
        }

        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    // Capture stdout to check for log_error messages
    testing::internal::CaptureStdout();
    
    try {
        MeshGraphDescriptor desc(text_proto);
        FAIL() << "Expected std::runtime_error for multiple validation failures";
    } catch (const std::runtime_error& e) {
        std::string stdout_output = testing::internal::GetCapturedStdout();
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"));

        std::vector<std::string> expected_errors = {
            "MeshGraphDescriptor: There must be at least one mesh descriptor",
            "MeshGraphDescriptor: Graph descriptor name 'G0' is not unique",
            "MeshGraphDescriptor: Graph descriptor 'G0' must have at least one instance",
            "MeshGraphDescriptor: Connection in graph 'G0' channel count must be positive",
            "MeshGraphDescriptor: Graph Descriptor 'G0' channel count must be positive",
            "MeshGraphDescriptor: Connection in graph 'G0' must have at least two nodes",
            "MeshGraphDescriptor: Graph descriptor 'G1' must have either graph_topology or connections defined",
            "MeshGraphDescriptor: Graph descriptor 'G1' must have a type specified",
        };

        // Check that all expected validation errors are present in stdout
        bool all_found = check_all_expected_errors_found(stdout_output, expected_errors);

        if (!all_found) {
            print_unexpected_errors(stdout_output, expected_errors);
        }
        
        EXPECT_TRUE(all_found) << "Not all expected error messages were found in stdout";
    }
}



TEST(MeshGraphDescriptorTests, InvalidProtoInstanceValidation) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 3 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 3 ] }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M1" id: 0 } }
    )proto";

    // FIXME: Not done yet
    // EXPECT_THROW(MeshGraphDescriptor desc(text_proto), std::runtime_error);
}

}  // namespace tt::tt_fabric::fabric_router_tests
