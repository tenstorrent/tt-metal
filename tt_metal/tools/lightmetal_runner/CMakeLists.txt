add_executable(lightmetal_runner ${CMAKE_CURRENT_SOURCE_DIR}/lightmetal_runner.cpp)
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
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(
    lightmetal_runner
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)

target_precompile_headers(lightmetal_runner REUSE_FROM TT::CommonPCH)
