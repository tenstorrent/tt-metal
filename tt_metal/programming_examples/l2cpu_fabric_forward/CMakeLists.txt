# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_l2cpu_fabric_forward)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# Host orchestrator: a Tensix producer stages a payload into the L2CPU (x280),
# which forwards it over FABRIC_1D to a Tensix receiver on a second chip.
add_executable(metal_example_l2cpu_fabric_forward)
target_sources(metal_example_l2cpu_fabric_forward PRIVATE l2cpu_fabric_forward.cpp)
target_link_libraries(metal_example_l2cpu_fabric_forward PUBLIC TT::Metalium)

# Raw UMD is used only to reach the passive L2CPU tile mailbox (same mechanism as
# the sibling x280_boot tool). It is only present in an in-tree metalium build (not
# the standalone find_package(TT-Metalium) path), so guard on it; the host compiles
# without it (the mailbox path becomes an inert HARDWARE-marked stub).
if(TARGET umd::tt-umd)
    target_link_libraries(metal_example_l2cpu_fabric_forward PUBLIC umd::tt-umd)
    target_compile_definitions(metal_example_l2cpu_fabric_forward PRIVATE FF_HAVE_UMD=1)
endif()

# NOTE: the x280 hart boot tool (metal_example_l2cpu_x280_boot) is reused from the
# sibling l2cpu_noc_transfer example — it is NOT redefined here. Build the fabric
# firmware it loads with x280/build_fw.sh (authored separately, plan Task 1).
