include(sources.cmake)

add_executable(ttnvtop ${TTNVTOP_SOURCES})

# ttnvtop bypasses tt-metal's Cluster entirely — it uses umd::TopologyDiscovery
# to open chips read-only, so it needs only UMD at runtime. It does however
# #include dev_msgs.h / dev_mem_map.h from tt-metal's device-shared headers to
# pick up the Wormhole mailbox layout.
target_link_libraries(
    ttnvtop
    PRIVATE
        umd::tt-umd
        TT::STL
)
target_include_directories(
    ttnvtop
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/common
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        ${PROJECT_SOURCE_DIR}/tt_metal/hostdevcommon/api
        # Wormhole-specific device headers (core_config.h, dev_mem_map.h, etc).
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/wormhole
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal
        .
)
set_target_properties(
    ttnvtop
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)
