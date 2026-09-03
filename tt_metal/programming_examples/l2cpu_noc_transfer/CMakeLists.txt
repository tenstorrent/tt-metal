cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_l2cpu_noc_transfer)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# Stage 1: Tensix <-> L2CPU LIM round-trip (L2CPU is a passive NOC target).
add_executable(metal_example_l2cpu_noc_transfer)
target_sources(metal_example_l2cpu_noc_transfer PRIVATE l2cpu_noc_transfer.cpp)
target_link_libraries(metal_example_l2cpu_noc_transfer PUBLIC TT::Metalium)

# Stage 2: host driver for the x280 echo demo (Tensix polls; x280 writes back).
# Run AFTER x280/x280_boot has released the hart with x280/build/echo.bin.
add_executable(metal_example_l2cpu_x280_echo)
target_sources(metal_example_l2cpu_x280_echo PRIVATE x280/x280_echo_test.cpp)
target_link_libraries(metal_example_l2cpu_x280_echo PUBLIC TT::Metalium)

# x280 hart boot tool: raw UMD only (no tt-metal). Loads firmware into the L2CPU
# tile over the NOC, sets reset vectors, steps PLL4, releases the one-shot reset.
# Build the firmware it loads with x280/build_fw.sh (clang; no riscv64 GCC needed).
# Requires the raw UMD target, which is only present in an in-tree metalium build
# (not the standalone find_package(TT-Metalium) path), so guard on it.
if(TARGET umd::tt-umd)
    add_executable(metal_example_l2cpu_x280_boot)
    target_sources(metal_example_l2cpu_x280_boot PRIVATE x280/x280_boot.cpp)
    target_link_libraries(metal_example_l2cpu_x280_boot PUBLIC umd::tt-umd)
endif()
