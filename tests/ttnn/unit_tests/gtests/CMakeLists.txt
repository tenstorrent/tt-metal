
set(TTNN_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_async_runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multiprod_queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_cq_multi_dev.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reflect.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_to_and_from_json.cpp
)
set(TTNN_CCL_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_erisc_data_mover_with_workers.cpp
)
set(TTNN_TENSOR_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_create_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_tensor_layout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_create_tensor_with_layout.cpp
)

add_executable(unit_tests_ttnn ${TTNN_UNIT_TESTS_SRC})
add_executable(unit_tests_ttnn_ccl ${TTNN_CCL_UNIT_TESTS_SRC})
add_executable(unit_tests_ttnn_tensor ${TTNN_TENSOR_UNIT_TESTS_SRC})
add_executable(test_multi_device ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_device.cpp)
add_executable(galaxy_unit_tests_ttnn ${CMAKE_CURRENT_SOURCE_DIR}/test_ccl_on_galaxy.cpp)

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
setup_ttnn_test_target(unit_tests_ttnn_tensor)
setup_ttnn_test_target(test_multi_device)
setup_ttnn_test_target(galaxy_unit_tests_ttnn)
