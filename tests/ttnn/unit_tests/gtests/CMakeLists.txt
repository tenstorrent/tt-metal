
set(TTNN_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mlir_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_binary_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_softmax_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_unary_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_matmul_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_repeat_interleave.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_async_runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multiprod_queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_cq_multi_dev.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reflect.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_to_and_from_json.cpp
)
set(TTNN_CCL_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_erisc_data_mover_with_workers.cpp
)

add_executable(unit_tests_ttnn ${TTNN_UNIT_TESTS_SRC})
add_executable(unit_tests_ttnn_ccl ${TTNN_CCL_UNIT_TESTS_SRC})
add_executable(test_multi_device ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_device.cpp)
add_executable(galaxy_unit_tests_ttnn ${CMAKE_CURRENT_SOURCE_DIR}/test_ccl_on_tg.cpp)

# Common function to set up target properties
function(setup_ttnn_test_target target_name)
    target_link_libraries(${target_name} PUBLIC test_common_libs ttnn tt_metal)
    target_include_directories(${target_name} PRIVATE
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(${target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/ttnn)
endfunction()

# Set up properties for both targets
setup_ttnn_test_target(unit_tests_ttnn)
setup_ttnn_test_target(unit_tests_ttnn_ccl)
setup_ttnn_test_target(test_multi_device)
setup_ttnn_test_target(galaxy_unit_tests_ttnn)

# setup ttnn_mlir_interface unit test
set(UNIT_TEST_MLIR_LIB unit_test_mlir_interface_lib)
add_executable(${UNIT_TEST_MLIR_LIB} ${CMAKE_CURRENT_SOURCE_DIR}/test_mlir_interface_lib.cpp)

target_link_libraries(${UNIT_TEST_MLIR_LIB} PRIVATE
    ttnn_mlir_interface
    gtest
    gtest_main
    ttnn # doesnt build without it - to figure out

    # tests/ttnn/unit_tests/gtests/CMakeFiles/unittest_mlir_interface_lib.dir/test_mlir_interface_lib.cpp.o: in function `testing::AssertionResult testing::internal::CmpHelperEQFailure<int, int>(char const*, char const*, int const&, int const&)':
    # test_mlir_interface_lib.cpp:(.text._ZN7testing8internal18CmpHelperEQFailureIiiEENS_15AssertionResultEPKcS4_RKT_RKT0_[_ZN7testing8internal18CmpHelperEQFailureIiiEENS_15AssertionResultEPKcS4_RKT_RKT0_]+0x88): undefined reference to `testing::internal::EqFailure(char const*, char const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, bool)'
    # clang++-17: error: linker command failed with exit code 1 (use -v to see invocation)
    # ninja: build stopped: subcommand failed.
)

get_target_property(MLIR_INTERFACE_PUBLIC_HEADER ttnn_mlir_interface PUBLIC_HEADER)
get_filename_component(MLIR_INTERFACE_PUBLIC_HEADER_INCLUDE ${MLIR_INTERFACE_PUBLIC_HEADER} DIRECTORY)

target_include_directories(${UNIT_TEST_MLIR_LIB} PRIVATE
    ${gtest_SOURCE_DIR}/include
    ${MLIR_INTERFACE_PUBLIC_HEADER_INCLUDE}
)

set_target_properties(${UNIT_TEST_MLIR_LIB} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/ttnn)
