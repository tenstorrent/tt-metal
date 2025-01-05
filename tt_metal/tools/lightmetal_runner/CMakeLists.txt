add_executable(lightmetal_runner ${CMAKE_CURRENT_SOURCE_DIR}/lightmetal_runner.cpp)
target_link_libraries(lightmetal_runner PRIVATE tt_metal)
target_include_directories(
    lightmetal_runner
    PRIVATE
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/common
        ${PROJECT_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(
    lightmetal_runner
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)
