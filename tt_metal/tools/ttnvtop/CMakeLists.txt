# ttnvtop: live per-core utilization monitor. Split into two binaries per
# PLAN.md §4: a headless collector that samples device state into
# /dev/shm/tt_device_*_util, and a lightweight viewer that renders those
# files into a TUI.
#
# The collector needs UMD (direct chip access, bypassing tt-metal Cluster so
# it does not take the CHIP_IN_USE mutex) plus the Wormhole device-shared
# headers for mailbox field offsets.
#
# The viewer needs nothing beyond libc and TT::STL. It is a pure SHM reader.

# ----- ttnvtop-collector (headless sampler) -----

add_executable(
    ttnvtop-collector
    collector/main.cpp
    collector/shm_publisher.cpp
)
target_link_libraries(
    ttnvtop-collector
    PRIVATE
        umd::tt-umd
        TT::STL
        rt # shm_open
)
target_include_directories(
    ttnvtop-collector
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/common
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        ${PROJECT_SOURCE_DIR}/tt_metal/hostdevcommon/api
        # Wormhole device headers — Phase 1 is WH-only; BH comes in Phase 3.
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/wormhole
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal
        .
)
set_target_properties(
    ttnvtop-collector
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)

# ----- ttnvtop (SHM viewer TUI) -----

add_executable(
    ttnvtop
    viewer/main.cpp
    collector/shm_publisher.cpp # for list_shm_files() only
)
target_link_libraries(
    ttnvtop
    PRIVATE
        TT::STL
        rt
)
target_include_directories(
    ttnvtop
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/common
        .
)
set_target_properties(
    ttnvtop
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)
