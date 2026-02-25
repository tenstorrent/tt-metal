set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/ttnn)

add_library(test_common_utils STATIC)
target_sources(test_common_utils PUBLIC common_test_utils.cpp)
target_link_libraries(test_common_utils PRIVATE TTNN::CPP)

function(setup_ttnn_test_target target_name)
    target_link_libraries(
        ${target_name}
        PUBLIC
            test_common_libs
            TTNN::CPP
    )
    target_include_directories(
        ${target_name}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/tests
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
endfunction()

# unit_tests_ttnn

# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

add_library(unit_tests_ttnn_smoke OBJECT)
add_library(TTNN::Test::Smoke ALIAS unit_tests_ttnn_smoke)
TT_ENABLE_UNITY_BUILD(unit_tests_ttnn_smoke)
target_sources(unit_tests_ttnn_smoke PRIVATE ${UNIT_TESTS_TTNN_SMOKE_SOURCES})
target_include_directories(unit_tests_ttnn_smoke PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(
    unit_tests_ttnn_smoke
    PRIVATE
        test_common_libs
        test_common_utils
        TTNN::CPP
)

add_library(unit_tests_ttnn_basic OBJECT)
add_library(TTNN::Test::Basic ALIAS unit_tests_ttnn_basic)
TT_ENABLE_UNITY_BUILD(unit_tests_ttnn_basic)
target_sources(unit_tests_ttnn_basic PRIVATE ${UNIT_TESTS_TTNN_BASIC_SOURCES})
target_include_directories(unit_tests_ttnn_basic PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(
    unit_tests_ttnn_basic
    PRIVATE
        test_common_libs
        TTNN::CPP
)

add_executable(unit_tests_ttnn)
target_link_libraries(
    unit_tests_ttnn
    PRIVATE
        TTNN::Test::Smoke
        TTNN::Test::Basic
)

# unit_tests_ttnn_ccl

add_library(unit_tests_ttnn_ccl_lib OBJECT)
add_library(TTNN::Test::CCL ALIAS unit_tests_ttnn_ccl_lib)
target_sources(unit_tests_ttnn_ccl_lib PRIVATE ${UNIT_TESTS_TTNN_CCL_SOURCES})
target_include_directories(unit_tests_ttnn_ccl_lib PUBLIC ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(
    unit_tests_ttnn_ccl_lib
    PUBLIC
        test_common_libs
        TTNN::CPP
)

add_executable(unit_tests_ttnn_ccl)
target_link_libraries(unit_tests_ttnn_ccl PRIVATE TTNN::Test::CCL)

# unit_tests_ttnn_ccl_ops

add_library(unit_tests_ttnn_ccl_ops_lib OBJECT)
add_library(TTNN::Test::CCL::Ops ALIAS unit_tests_ttnn_ccl_ops_lib)
target_sources(unit_tests_ttnn_ccl_ops_lib PRIVATE ${UNIT_TESTS_TTNN_CCL_OPS_SOURCES})
target_include_directories(
    unit_tests_ttnn_ccl_ops_lib
    PUBLIC
        ${PROJECT_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    unit_tests_ttnn_ccl_ops_lib
    PUBLIC
        test_common_libs
        TTNN::CPP
)

add_executable(unit_tests_ttnn_ccl_ops)
target_link_libraries(unit_tests_ttnn_ccl_ops PRIVATE TTNN::Test::CCL::Ops)

# unit_tests_ttnn_ccl_multi_tensor

add_executable(unit_tests_ttnn_ccl_multi_tensor)
target_sources(unit_tests_ttnn_ccl_multi_tensor PRIVATE ${UNIT_TESTS_TTNN_CCL_MULTI_TENSOR_SOURCES})
setup_ttnn_test_target(unit_tests_ttnn_ccl_multi_tensor)

# unit_tests_ttnn_accessor

add_executable(unit_tests_ttnn_accessor)
target_sources(unit_tests_ttnn_accessor PRIVATE ${UNIT_TESTS_TTNN_ACCESSOR_SOURCES})
# needs access to "compile_time_args.h"
target_link_libraries(unit_tests_ttnn_accessor PRIVATE Metalium::Metal::Hardware)
setup_ttnn_test_target(unit_tests_ttnn_accessor)

# unit_tests_ttnn_tensor

add_library(unit_tests_ttnn_tensor_lib OBJECT)
add_library(TTNN::Test::Basic::Tensor ALIAS unit_tests_ttnn_tensor_lib)
target_sources(unit_tests_ttnn_tensor_lib PRIVATE ${UNIT_TESTS_TTNN_TENSOR_SOURCES})
setup_ttnn_test_target(unit_tests_ttnn_tensor_lib)
target_link_libraries(unit_tests_ttnn_tensor_lib PRIVATE xtensor)

add_executable(unit_tests_ttnn_tensor)
target_link_libraries(unit_tests_ttnn_tensor PRIVATE TTNN::Test::Basic::Tensor)

# test_ccl_multi_cq_multi_device

add_executable(test_ccl_multi_cq_multi_device)
target_sources(test_ccl_multi_cq_multi_device PRIVATE ${TEST_CCL_MULTI_CQ_MULTI_DEVICE_SOURCES})
setup_ttnn_test_target(test_ccl_multi_cq_multi_device)
target_link_libraries(
    test_ccl_multi_cq_multi_device
    PRIVATE
        Boost::asio
        Boost::lockfree
        test_common_utils
)

# unit_tests_ttnn_emitc

add_executable(unit_tests_ttnn_emitc)
target_sources(unit_tests_ttnn_emitc PRIVATE ${UNIT_TESTS_TTNN_EMITC_SOURCES})
setup_ttnn_test_target(unit_tests_ttnn_emitc)

add_subdirectory(multiprocess)
add_subdirectory(udm)
