add_executable(test_get_programs_perf_data test_get_programs_perf_data.cpp)

target_link_libraries(
    test_get_programs_perf_data
    PUBLIC
        ttnn
        test_common_libs
        Python3::Python
)

set_target_properties(
    test_get_programs_perf_data
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/ttnn/tracy/
        # BUILD_RPATH for finding shared libraries when running from build tree or CI tar artifact
        # The executable is at build/test/ttnn/tracy/, needs to find:
        #   - libtt_metal.so in build/lib/
        #   - _ttnn.so in ttnn/ttnn/ (source tree)
        BUILD_RPATH
            "${PROJECT_BINARY_DIR}/lib;${PROJECT_SOURCE_DIR}/ttnn/ttnn"
)
