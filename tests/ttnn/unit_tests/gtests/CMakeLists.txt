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

add_library(unit_tests_ttnn_smoke OBJECT)
add_library(TTNN::Test::Smoke ALIAS unit_tests_ttnn_smoke)
TT_ENABLE_UNITY_BUILD(unit_tests_ttnn_smoke)
target_sources(
    unit_tests_ttnn_smoke
    PRIVATE
        test_reflect.cpp
        test_to_and_from_json.cpp
        test_sliding_window_infra.cpp
        test_async_runtime.cpp
        test_conv2d.cpp
        test_multi_cq_multi_dev.cpp
        test_multiprod_queue.cpp
)
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
target_sources(
    unit_tests_ttnn_basic
    PRIVATE
        test_add.cpp
        test_broadcast_to.cpp
        test_generic_op.cpp
        test_graph_add.cpp
        test_graph_basic.cpp
        test_graph_capture_arguments_morehdot.cpp
        test_graph_capture_arguments_transpose.cpp
        test_graph_query_op_constraints.cpp
        test_graph_query_op_runtime.cpp
        test_op_runtime_offline.cpp
        test_launch_operation.cpp # TODO: Fix data race (TSan) then shift-left
        test_matmul_benchmark.cpp
)
target_include_directories(unit_tests_ttnn_basic PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(
    unit_tests_ttnn_basic
    PRIVATE
        test_common_libs
        TTNN::CPP
)

# Add mlp-op-perf dependency if BUILD_MLP_OP_PERF is enabled
if(BUILD_MLP_OP_PERF)
    message(STATUS "BUILD_MLP_OP_PERF is enabled")

    # Try multiple possible variable names for the mlp-op-perf source directory
    set(MLP_OP_PERF_DIR "")
    if(mlp_op_perf_SOURCE_DIR)
        set(MLP_OP_PERF_DIR ${mlp_op_perf_SOURCE_DIR})
        message(STATUS "Using mlp_op_perf_SOURCE_DIR: ${mlp_op_perf_SOURCE_DIR}")
    elseif(mlp-op-perf_SOURCE_DIR)
        set(MLP_OP_PERF_DIR ${mlp-op-perf_SOURCE_DIR})
        message(STATUS "Using mlp-op-perf_SOURCE_DIR: ${mlp-op-perf_SOURCE_DIR}")
    else()
        # Fallback to CPM cache directory - find interface.hpp directly
        file(GLOB_RECURSE INTERFACE_HPP_FILE "${CMAKE_SOURCE_DIR}/.cpmcache/mlp-op-perf/*/interface/interface.hpp")
        if(INTERFACE_HPP_FILE)
            list(GET INTERFACE_HPP_FILE 0 INTERFACE_FILE)
            get_filename_component(MLP_OP_PERF_DIR ${INTERFACE_FILE} DIRECTORY)
            get_filename_component(MLP_OP_PERF_DIR ${MLP_OP_PERF_DIR} DIRECTORY)
            message(STATUS "Found mlp-op-perf at: ${MLP_OP_PERF_DIR}")
        else()
            message(
                FATAL_ERROR
                "mlp-op-perf interface.hpp not found. Make sure BUILD_MLP_OP_PERF is enabled and the package is downloaded."
            )
        endif()
    endif()

    # Verify the interface directory exists
    if(NOT EXISTS "${MLP_OP_PERF_DIR}/interface/interface.hpp")
        message(FATAL_ERROR "interface.hpp not found at ${MLP_OP_PERF_DIR}/interface/interface.hpp")
    endif()

    message(STATUS "Adding include directory: ${MLP_OP_PERF_DIR}/interface")

    target_include_directories(
        unit_tests_ttnn_basic
        PRIVATE
            ${MLP_OP_PERF_DIR}/interface
            ${MLP_OP_PERF_DIR}/ops/include
    )
    target_compile_definitions(unit_tests_ttnn_basic PRIVATE BUILD_MLP_OP_PERF=1)

    # Link the mlp-op-perf ops library
    target_link_libraries(unit_tests_ttnn_basic PRIVATE ops_library)
endif()

add_library(unit_tests_ttnn_slow OBJECT)
add_library(TTNN::Test::Slow ALIAS unit_tests_ttnn_slow)
TT_ENABLE_UNITY_BUILD(unit_tests_ttnn_slow)
target_sources(
    unit_tests_ttnn_slow
    PRIVATE
        test_async_runtime.cpp # TODO: Fix memory leak (LSan) then shift-left
        test_conv2d.cpp # TODO: Fix misaligned memory load (UBSan) then shift-left
        test_multi_cq_multi_dev.cpp
        test_multiprod_queue.cpp
)
target_include_directories(unit_tests_ttnn_slow PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(
    unit_tests_ttnn_slow
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
target_sources(
    unit_tests_ttnn_ccl_lib
    PRIVATE
        ccl/test_ccl_commands.cpp
        ccl/test_ccl_helpers.cpp
        ccl/test_ccl_reduce_scatter_host_helpers.cpp
        ccl/test_ccl_tensor_slicers.cpp
        ccl/test_erisc_data_mover_with_workers.cpp
        ccl/test_fabric_erisc_data_mover_loopback_with_workers.cpp
        ccl/test_sharded_address_generators.cpp
        ccl/test_sharded_address_generators_new.cpp
)
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
target_sources(
    unit_tests_ttnn_ccl_ops_lib
    PRIVATE
        ccl/test_persistent_fabric_ccl_ops.cpp
        ccl/test_send_recv_ops.cpp
)
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
target_sources(unit_tests_ttnn_ccl_multi_tensor PRIVATE ccl/test_multi_tensor_ccl.cpp)
setup_ttnn_test_target(unit_tests_ttnn_ccl_multi_tensor)

# unit_tests_ttnn_accessor

add_executable(unit_tests_ttnn_accessor)
target_sources(
    unit_tests_ttnn_accessor
    PRIVATE
        accessor/common.cpp
        accessor/test_accessor_benchmarks.cpp
        accessor/test_tensor_accessor.cpp
        accessor/test_tensor_accessor_on_device.cpp
)
setup_ttnn_test_target(unit_tests_ttnn_accessor)

# unit_tests_ttnn_tensor

add_library(unit_tests_ttnn_tensor_lib OBJECT)
add_library(TTNN::Test::Basic::Tensor ALIAS unit_tests_ttnn_tensor_lib)
target_sources(
    unit_tests_ttnn_tensor_lib
    PRIVATE
        tensor/common_tensor_test_utils.cpp
        tensor/test_create_tensor.cpp
        tensor/test_create_tensor_multi_device.cpp
        tensor/test_create_tensor_with_layout.cpp
        tensor/test_distributed_tensor.cpp
        tensor/test_tensor_topology.cpp
        tensor/test_mesh_tensor.cpp
        tensor/test_partition.cpp
        tensor/test_tensor_layout.cpp
        tensor/test_tensor_nd_sharding.cpp
        tensor/test_tensor_serialization.cpp
        tensor/test_tensor_sharding.cpp
        tensor/test_vector_conversion.cpp
        tensor/test_xtensor_adapter.cpp
        tensor/test_xtensor_conversion.cpp
)
setup_ttnn_test_target(unit_tests_ttnn_tensor_lib)
target_link_libraries(unit_tests_ttnn_tensor_lib PRIVATE xtensor)

add_executable(unit_tests_ttnn_tensor)
target_link_libraries(unit_tests_ttnn_tensor PRIVATE TTNN::Test::Basic::Tensor)

# test_ccl_multi_cq_multi_device

add_executable(test_ccl_multi_cq_multi_device)
target_sources(
    test_ccl_multi_cq_multi_device
    PRIVATE
        multi_thread/test_ccl_multi_cq_multi_device.cpp
        multi_thread/test_utils.cpp
)
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
target_sources(unit_tests_ttnn_emitc PRIVATE emitc/test_sanity.cpp)
setup_ttnn_test_target(unit_tests_ttnn_emitc)

add_subdirectory(multiprocess)
