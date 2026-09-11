# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_l2cpu_fabric_forward)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# Host orchestrator: boots the x280 on each chip (via the sibling boot tool), resolves
# the fabric router connection with a Tensix setup kernel, and drives L2CPU -> fabric
# -> L2CPU / Tensix transfers. All L2CPU memory access goes through Tensix kernels.
add_executable(metal_example_l2cpu_fabric_forward)
target_sources(metal_example_l2cpu_fabric_forward PRIVATE l2cpu_fabric_forward.cpp)
target_link_libraries(metal_example_l2cpu_fabric_forward PUBLIC TT::Metalium)

# The x280 boot tool (metal_example_l2cpu_x280_boot) is built by the sibling
# l2cpu_noc_transfer example. Build the firmware this example loads with x280/build_fw.sh.
