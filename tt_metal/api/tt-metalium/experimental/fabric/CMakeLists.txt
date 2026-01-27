# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Experimental Fabric API Headers
target_sources(
    tt_metal
    PUBLIC
        FILE_SET api
        FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/control_plane.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/edm_fabric_counters.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/fabric.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/fabric_edm_types.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/fabric_types.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/mesh_graph.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/mesh_graph_descriptor.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/routing_table_generator.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/topology_mapper_utils.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/fabric_telemetry.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/fabric_telemetry_reader.hpp
)
