// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cabling_generator/cabling_generator.hpp>
#include <factory_system_descriptor/utils.hpp>

namespace tt::scaleout_tools {

TEST(Cluster, TestFactorySystemDescriptor16LB) {
    // Create the cabling generator with file paths
    CablingGenerator cabling_generator(
        "tools/tests/scaleout/cabling_descriptors/16_n300_lb_cluster.textproto",
        "tools/tests/scaleout/deployment_descriptors/16_lb_deployment.textproto");

    cabling_generator.emit_factory_system_descriptor("fsd/factory_system_descriptor_16_n300_lb.textproto");

    // Validate the FSD against the discovered GSD using the common utility function
    validate_fsd_against_gsd(
        "fsd/factory_system_descriptor_16_n300_lb.textproto",
        "tools/tests/scaleout/global_system_descriptors/16_lb_physical_desc.yaml");
}

TEST(Cluster, TestFactorySystemDescriptor5LB) {
    // Create the cabling generator with file paths
    CablingGenerator cabling_generator(
        "tools/tests/scaleout/cabling_descriptors/5_n300_lb_superpod.textproto",
        "tools/tests/scaleout/deployment_descriptors/5_lb_deployment.textproto");

    // Generate the FSD (textproto format)
    cabling_generator.emit_factory_system_descriptor("fsd/factory_system_descriptor_5_n300_lb.textproto");

    // Validate the FSD against the discovered GSD using the common utility function
    validate_fsd_against_gsd(
        "fsd/factory_system_descriptor_5_n300_lb.textproto",
        "tools/tests/scaleout/global_system_descriptors/5_lb_physical_desc.yaml");
}

TEST(Cluster, TestFactorySystemDescriptor5WHGalaxyYTorus) {
    // Create the cabling generator with file paths
    CablingGenerator cabling_generator(
        "tools/tests/scaleout/cabling_descriptors/5_wh_galaxy_y_torus_superpod.textproto",
        "tools/tests/scaleout/deployment_descriptors/5_wh_galaxy_y_torus_deployment.textproto");

    // Generate the FSD (textproto format)
    cabling_generator.emit_factory_system_descriptor("fsd/factory_system_descriptor_5_wh_galaxy_y_torus.textproto");

    // Validate the FSD against the discovered GSD using the common utility function
    EXPECT_THROW(
        {
            try {
                validate_fsd_against_gsd(
                    "fsd/factory_system_descriptor_5_wh_galaxy_y_torus.textproto",
                    "tools/tests/scaleout/global_system_descriptors/"
                    "5_wh_galaxy_y_torus_physical_desc.yaml");
            } catch (const std::runtime_error& e) {
                std::cout << e.what() << std::endl;
                throw;
            }
        },
        std::runtime_error);
}

}  // namespace tt::scaleout_tools
