include(sources.cmake)

add_executable(lightmetal_runner ${LIGHTMETAL_RUNNER_SOURCES})
target_link_libraries(
    lightmetal_runner
    PRIVATE
        Metalium::Metal
        TT::STL
        FlatBuffers::FlatBuffers
)
target_include_directories(
    lightmetal_runner
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/common
        .
)

set_target_properties(
    lightmetal_runner
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)

tt_reuse_precompile_headers(lightmetal_runner TT::CommonPCH)
