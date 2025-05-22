// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <tuple>
#include <vector>

#include <umd/device/types/cluster_descriptor_types.h>

namespace tt::tt_fabric {

static const std::array<std::tuple<std::string, std::vector<std::vector<eth_coord_t>>>, 5>
    t3k_mesh_descriptor_chip_mappings = {
        std::tuple{
            "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml",
            std::vector<std::vector<eth_coord_t>>{
                {{0, 0, 0, 0, 0},
                 {0, 1, 0, 0, 0},
                 {0, 2, 0, 0, 0},
                 {0, 3, 0, 0, 0},
                 {0, 0, 1, 0, 0},
                 {0, 1, 1, 0, 0},
                 {0, 2, 1, 0, 0},
                 {0, 3, 1, 0, 0}}}},
        std::tuple{
            "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.yaml",
            std::vector<std::vector<eth_coord_t>>{
                {{0, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 1, 0, 0}},
                {{0, 2, 0, 0, 0}, {0, 3, 0, 0, 0}, {0, 2, 1, 0, 0}, {0, 3, 1, 0, 0}}}},
        std::tuple{
            "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x2_mesh_graph_descriptor.yaml",
            std::vector<std::vector<eth_coord_t>>{
                {{0, 0, 0, 0, 0}, {0, 1, 0, 0, 0}},
                {{0, 2, 0, 0, 0}, {0, 3, 0, 0, 0}},
                {{0, 0, 1, 0, 0}, {0, 1, 1, 0, 0}},
                {{0, 2, 1, 0, 0}, {0, 3, 1, 0, 0}}}},
        std::tuple{
            "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x1_mesh_graph_descriptor.yaml",
            std::vector<std::vector<eth_coord_t>>{
                {{0, 0, 0, 0, 0}},
                {{0, 1, 0, 0, 0}},
                {{0, 2, 0, 0, 0}},
                {{0, 3, 0, 0, 0}},
                {{0, 0, 1, 0, 0}},
                {{0, 1, 1, 0, 0}},
                {{0, 2, 1, 0, 0}},
                {{0, 3, 1, 0, 0}}}},
        std::tuple{
            "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_1x2_1x1_mesh_graph_descriptor.yaml",
            std::vector<std::vector<eth_coord_t>>{
                {{0, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 1, 0, 0}},
                {{0, 2, 0, 0, 0}, {0, 3, 0, 0, 0}},
                {{0, 2, 1, 0, 0}},
                {{0, 3, 1, 0, 0}}}}};

}  // namespace tt::tt_fabric
