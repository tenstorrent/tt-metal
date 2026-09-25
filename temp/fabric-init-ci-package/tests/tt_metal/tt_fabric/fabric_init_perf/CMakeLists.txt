# Included only from the existing Metal fabric test build.
if(ENABLE_TRACY)
    add_executable(fabric_init_benchmark fabric_init_benchmark.cpp)
    target_link_libraries(fabric_init_benchmark PRIVATE tt_metal test_common_libs)
    target_compile_definitions(fabric_init_benchmark PRIVATE FABRIC_INIT_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
    target_include_directories(fabric_init_benchmark PRIVATE
        ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tt_metal ${UMD_HOME})
    set_target_properties(fabric_init_benchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric)
endif()
