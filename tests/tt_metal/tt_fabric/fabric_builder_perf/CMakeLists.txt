# Included only from the existing Metal fabric test build.
if(ENABLE_TRACY)
    add_executable(fabric_builder_benchmark fabric_builder_benchmark.cpp)
    target_link_libraries(fabric_builder_benchmark PRIVATE tt_metal test_common_libs)
    target_include_directories(fabric_builder_benchmark PRIVATE
        ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tt_metal ${UMD_HOME})
    set_target_properties(fabric_builder_benchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric)
endif()
