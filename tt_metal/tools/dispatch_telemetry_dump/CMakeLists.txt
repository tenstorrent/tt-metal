include(sources.cmake)

add_executable(dispatch_telemetry_dump ${DISPATCH_TELEMETRY_DUMP_SOURCES})
target_link_libraries(
    dispatch_telemetry_dump
    PRIVATE
        Metalium::Metal
        TT::STL
        numa
)
target_include_directories(
    dispatch_telemetry_dump
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/common
        .
)
set_target_properties(
    dispatch_telemetry_dump
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)

tt_reuse_precompile_headers(dispatch_telemetry_dump TT::CommonPCH)
