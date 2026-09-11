# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_l2cpu_mux)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# Host: N Tensix mux clients on chip A -> L2CPU-hosted mux (x280/fw_mux.c) -> fabric
# router -> chip B. Reuses the fabric_forward host helpers and the x280 boot tool from
# the sibling examples; build the firmware with x280/build_fw.sh.
add_executable(metal_example_l2cpu_mux)
target_sources(metal_example_l2cpu_mux PRIVATE l2cpu_mux.cpp)
target_link_libraries(metal_example_l2cpu_mux PUBLIC TT::Metalium)
