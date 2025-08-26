// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Cabling Generator
 *    Dependencies: protobuf
 *
 *    Description: Generates a cut sheet CSV file based on deployment and cabling specifications.
 *
 *    Usage: ./cabling_generator <deployment.textproto> <cabling.textproto>
 */

#include <fstream>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <string>
#include "deployment.pb.h"  // Generated from deployment.proto
#include "cabling.pb.h"     // Generated from cabling.proto
#include <google/protobuf/text_format.h>

typedef enum { CABLE_0_5, CABLE_1, CABLE_2_5, CABLE_3, CABLE_5, OPTICAL_CABLE } cable_length_t;

std::unordered_map<cable_length_t, std::string> cable_length_str = {
    {CABLE_0_5, "0.5m"},
    {CABLE_1, "1m"},
    {CABLE_2_5, "2.5m"},
    {CABLE_3, "3m"},
    {CABLE_5, "5m"},
    {OPTICAL_CABLE, "Optical"}};

cable_length_t calc_cable_length(int rack_0, int shelf_u_0, int rack_1, int shelf_u_1) {
    double standard_rack_w = 600.0;    // mm
    double standard_rack_u_h = 44.45;  // mm

    double rack_distance = fabs(rack_0 - rack_1) * standard_rack_w;
    double u_distance = (fabs(shelf_u_0 - shelf_u_1) + 5) * standard_rack_u_h;

    double cable_length = sqrt(rack_distance * rack_distance + u_distance * u_distance);

    if (cable_length <= 500.0) {
        return CABLE_0_5;
    } else if (cable_length <= 1000.0) {
        return CABLE_1;
    } else if (cable_length <= 2500.0) {
        return CABLE_2_5;
    } else if (cable_length <= 3000.0) {
        return CABLE_3;
    } else if (cable_length <= 5000.0) {
        return CABLE_5;
    } else {
        return OPTICAL_CABLE;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <deployment.textproto> <cabling.textproto>" << std::endl;
        return 1;
    }

    const std::string dep_fn = argv[1];
    std::ifstream d_input(dep_fn);
    if (!d_input) {
        std::cerr << "Failed to open file: " << dep_fn << std::endl;
        return 1;
    }
    // Create the Deployment protobuf object
    deployment::DeploymentDescriptor deployment;
    // Read the entire d_input file into a string
    std::string d_input_str((std::istreambuf_iterator<char>(d_input)), std::istreambuf_iterator<char>());

    // Parse the textproto into the protobuf object
    if (!google::protobuf::TextFormat::ParseFromString(d_input_str, &deployment)) {
        std::cerr << "Failed to parse deployment textproto." << std::endl;
        std::cerr << "May have misordered input parameters, please check ordering and try again." << std::endl;
        std::cerr << "Usage: " << argv[0] << " <deployment.textproto> <cabling.textproto>" << std::endl;
        return 1;
    }

    const std::string cable_fn = argv[2];
    std::ifstream c_input(cable_fn);
    if (!c_input) {
        std::cerr << "Failed to open output file: " << cable_fn << std::endl;
        return 1;
    }
    cabling::CablingDescriptor cabling;
    std::string c_input_str((std::istreambuf_iterator<char>(c_input)), std::istreambuf_iterator<char>());

    // Parse the textproto into the cabling protobuf object
    if (!google::protobuf::TextFormat::ParseFromString(c_input_str, &cabling)) {
        std::cerr << "Failed to parse cabling textproto." << std::endl;
        return 1;
    }

    // Create a text file to write output
    std::ofstream output_file("cut_sheet.csv");
    if (!output_file) {
        std::cerr << "Failed to create cut_sheet.csv for writing." << std::endl;
        return 1;
    }
    output_file.fill('0');
    output_file << "Source,,,,,,,Destination,,,,,,,Cable Length,Cable Type" << std::endl;
    output_file << "Hall,Aisle,Rack,U,Tray,Port,Label,Hall,Aisle,Rack,U,Tray,Port,Label,," << std::endl;

    uint32_t max_host = deployment.hosts_size();
    for (size_t i = 0; i < cabling.connections_size(); i++) {
        const auto& connection = cabling.connections(i);

        if (connection.ep_a().host() >= max_host || connection.ep_b().host() >= max_host) {
            std::cerr << "Invalid host index in connection " << i
                      << ": Please review cabling and deployment specifications." << std::endl;
            break;
        }
        std::stringstream label;
        label.fill('0');
        int32_t host_a = connection.ep_a().host();
        int32_t host_b = connection.ep_b().host();

        deployment::Host host_info_a = deployment.hosts(host_a);
        deployment::Host host_info_b = deployment.hosts(host_b);

        cable_length_t cable_l =
            calc_cable_length(host_info_a.rack(), host_info_a.shelf_u(), host_info_b.rack(), host_info_b.shelf_u());

        label << host_info_a.hall() << host_info_a.aisle() << std::setw(2) << host_info_a.rack() << "U" << std::setw(2)
              << host_info_a.shelf_u() << "-" << connection.ep_a().tray() << "-" << connection.ep_a().port();

        output_file << host_info_a.hall() << "," << host_info_a.aisle() << "," << std::setw(2) << host_info_a.rack()
                    << "," << host_info_a.shelf_u() << "," << connection.ep_a().tray() << ","
                    << connection.ep_a().port() << "," << label.str() << ",";
        label.str("");  // Clear the stringstream for reuse
        label << host_info_b.hall() << host_info_b.aisle() << std::setw(2) << host_info_b.rack() << "U" << std::setw(2)
              << host_info_b.shelf_u() << "-" << connection.ep_b().tray() << "-" << connection.ep_b().port();
        output_file << host_info_b.hall() << "," << host_info_b.aisle() << "," << std::setw(2) << host_info_b.rack()
                    << "," << host_info_b.shelf_u() << "," << connection.ep_b().tray() << ","
                    << connection.ep_b().port() << "," << label.str() << ",";
        output_file << cable_length_str.at(cable_l) << "," << ((cable_l == OPTICAL_CABLE) ? "Optical" : "QSFP_DD")
                    << std::endl;
    }

    d_input.close();
    c_input.close();
    output_file.close();

    return 0;
}
