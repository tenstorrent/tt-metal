# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Experimental Fabric API Headers
set(TT_METAL_EXPERIMENTAL_FABRIC_API
    api/tt-metalium/experimental/fabric/control_plane.hpp
    api/tt-metalium/experimental/fabric/edm_fabric_counters.hpp
    api/tt-metalium/experimental/fabric/fabric.hpp
    api/tt-metalium/experimental/fabric/fabric_edm_types.hpp
    api/tt-metalium/experimental/fabric/fabric_types.hpp
    api/tt-metalium/experimental/fabric/mesh_graph.hpp
    api/tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp
    api/tt-metalium/experimental/fabric/routing_table_generator.hpp
    api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp
    api/tt-metalium/experimental/fabric/fabric_telemetry.hpp
    api/tt-metalium/experimental/fabric/fabric_telemetry_reader.hpp
    PARENT_SCOPE
)
