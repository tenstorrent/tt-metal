
set(TT_EAGER_TESTS_DTX
    dtx/collapse_transformations
    dtx/overlap
    dtx/tensor
    dtx/test_dtx_tilized_row_to_col_major
    dtx/test_dtx
    dtx/unit_tests
)
# test_dtx                           # <- SD only
# test_dtx_tilized_row_to_col_major  # <- SD only
# tensor_evaluate                    # <- not called in run_tt_eager.py

set(TT_EAGER_TESTS_OPS
    # ccl/test_all_gather_utils           # <- not called in run_tt_eager.py
    ops/test_average_pool
    ops/test_eltwise_binary_op
    ops/test_eltwise_unary_op
    ops/test_softmax_op
    ops/test_layernorm_op
    # ops/test_moreh_adam_op                  # <- not called in run_tt_eager.py
    # ops/test_moreh_matmul_op                # <- not called in run_tt_eager.py
    # ops/test_moreh_layernorm_op             # <- not called in run_tt_eager.py
    ops/test_multi_queue_api
    ops/test_transpose_op
    ops/test_transpose_wh_single_core
    ops/test_transpose_wh_multi_core
    ops/test_reduce_op
    ops/test_bcast_op
    ops/test_bmm_op
    # ops/test_pad_op                         # <- not called in run_tt_eager.py
    ops/test_tilize_op
    ops/test_tilize_zero_padding
    ops/test_tilize_op_channels_last
    ops/test_tilize_zero_padding_channels_last
    ops/test_sfpu                           # <- SD only, no WH
    ops/test_fold_op
)

set(TT_EAGER_TESTS_TENSORS
    tensors/test_copy_and_move
    tensors/test_host_device_loopback
    tensors/test_raw_host_memory_pointer
    # tensors/test_sharded_loopback           # <- not called in run_tt_eager.py
    tensors/test_async_tensor_apis
)

set(TT_EAGER_TESTS_INTEGRATION
    integration_tests/test_bert
)

set(EAGER_TEST_TARGETS "")

function(CREATE_EXEC TESTLIST)
    foreach (TEST_SRC ${TESTLIST})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME)
        get_filename_component(TEST_DIR ${TEST_SRC} DIRECTORY)
        set(TEST_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_SRC})
        # bc unit_tests and tensor are already taken as target names/executables
        if(${TEST_NAME} STREQUAL "unit_tests")
            set(TEST_TARGET "test_unit_tests")
        elseif(${TEST_NAME} STREQUAL "tensor")
            set(TEST_TARGET "test_tensor")
        else()
            set(TEST_TARGET ${TEST_NAME})
        endif()
        add_executable(${TEST_TARGET} ${TEST_SRC_PATH})

        target_link_libraries(${TEST_TARGET} PUBLIC tt_dnn tensor queue)
        target_link_libraries(${TEST_TARGET} PUBLIC ${COMMON_LIBS} stdc++fs yaml-cpp gtest gtest_main m)
        target_include_directories(${TEST_TARGET} PRIVATE
            ${UMD_HOME}
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/tt_metal
            ${CMAKE_SOURCE_DIR}/tt_metal/common
            ${CMAKE_SOURCE_DIR}/tests
            ${CMAKE_CURRENT_SOURCE_DIR}
        )
        target_compile_options(${TEST_TARGET} PUBLIC ${COMPILE_OPTIONS})
        set_target_properties(${TEST_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/tt_eager/${TEST_DIR})
        list(APPEND EAGER_TEST_TARGETS ${TEST_TARGET})
    endforeach()
    set(EAGER_TEST_TARGETS "${EAGER_TEST_TARGETS}" PARENT_SCOPE)   # set parent scope so variable is available outside of function
endfunction()

CREATE_EXEC("${TT_EAGER_TESTS_DTX}")
CREATE_EXEC("${TT_EAGER_TESTS_OPS}")
CREATE_EXEC("${TT_EAGER_TESTS_TENSORS}")
CREATE_EXEC("${TT_EAGER_TESTS_INTEGRATION}")

add_custom_target(eager_tests DEPENDS ${EAGER_TEST_TARGETS})

# function(CREATE_EXEC TEST_PATH)
#     STRING(REGEX REPLACE ".*/(.*)" "\\1" TEST_NAME ${TEST_PATH})
#     set(TEST_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_PATH})
#     # bc unit_tests and tensor are already taken as target names/executables
#     if(${TEST_NAME} STREQUAL "unit_tests")
#         set(TEST_TARGET "test_unit_tests")
#     elseif(${TEST_NAME} STREQUAL "tensor")
#         set(TEST_TARGET "test_tensor")
#     else()
#         set(TEST_TARGET ${TEST_NAME})
#     endif()
#     add_executable(${TEST_TARGET} ${TEST_SRC_PATH})

#     target_link_libraries(${TEST_TARGET} PUBLIC tt_dnn tt_metal_libs)
#     target_link_libraries(${TEST_TARGET} PUBLIC stdc++fs yaml-cpp gtest gtest_main m)
#     target_include_directories(${TEST_TARGET} PRIVATE
#         ${UMD_HOME}
#         ${CMAKE_SOURCE_DIR}
#         ${CMAKE_SOURCE_DIR}/tt_metal
#         ${CMAKE_SOURCE_DIR}/tt_metal/common
#         ${CMAKE_SOURCE_DIR}/tests
#         ${CMAKE_CURRENT_SOURCE_DIR}
#     )
#     target_compile_options(${TEST_TARGET} PUBLIC ${COMPILE_OPTIONS})
#     list(APPEND EAGER_TEST_TARGETS ${TEST_TARGET})
#     set(EAGER_TEST_TARGETS "${EAGER_TEST_TARGETS}" PARENT_SCOPE)   # set parent scope so variable is available outside of function

# endfunction()

# foreach(TEST ${TT_EAGER_TESTS_DTX})
#     message("TESTS: ${TEST}")
#     CREATE_EXEC(${TEST})
# endforeach()
