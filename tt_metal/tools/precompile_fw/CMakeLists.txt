add_executable(precompile_fw ${CMAKE_CURRENT_SOURCE_DIR}/precompile_fw.cpp)
target_link_libraries(
    precompile_fw
    PRIVATE
        Metalium::Metal
        TT::STL
        yaml-cpp::yaml-cpp
)
target_include_directories(
    precompile_fw
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(
    precompile_fw
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)

target_precompile_headers(precompile_fw REUSE_FROM TT::CommonPCH)
