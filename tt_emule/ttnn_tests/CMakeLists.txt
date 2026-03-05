cmake_minimum_required(VERSION 3.16)
include(GoogleTest)

function(add_ttnn_emule_test TEST_NAME TEST_SOURCE)
    set(COPIED "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}.cpp")
    configure_file("${PROJECT_SOURCE_DIR}/${TEST_SOURCE}" "${COPIED}" COPYONLY)
    add_executable(${TEST_NAME} "${COPIED}")
    target_include_directories(
        ${TEST_NAME}
        PRIVATE
            $<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>
            ${PROJECT_SOURCE_DIR}/ttnn/api
            ${PROJECT_SOURCE_DIR}/ttnn/cpp
            ${PROJECT_SOURCE_DIR}/tests/ttnn/unit_tests/gtests # ttnn_test_fixtures.hpp
            ${PROJECT_SOURCE_DIR}/tests/tt_metal
            ${PROJECT_SOURCE_DIR}/tt_stl
    )
    target_link_options(${TEST_NAME} PRIVATE -rdynamic)
    target_link_libraries(
        ${TEST_NAME}
        PRIVATE
            ttnncpp
            GTest::gtest_main
            Threads::Threads
            ${CMAKE_DL_LIBS}
    )
    set_target_properties(
        ${TEST_NAME}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${CMAKE_BINARY_DIR}/test/tt_emule
    )
    gtest_discover_tests(${TEST_NAME} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endfunction()

add_ttnn_emule_test(test_ttnn_add_int
    tests/ttnn/unit_tests/gtests/test_add_int.cpp
)

add_ttnn_emule_test(test_ttnn_add
    tests/ttnn/unit_tests/gtests/test_add.cpp
)

add_ttnn_emule_test(test_ttnn_sub_int
    tests/ttnn/unit_tests/gtests/test_sub_int.cpp
)

add_ttnn_emule_test(test_ttnn_rsub_int
    tests/ttnn/unit_tests/gtests/test_rsub_int.cpp
)

add_ttnn_emule_test(test_ttnn_matmul
    tests/ttnn/unit_tests/gtests/test_matmul.cpp
)

add_ttnn_emule_test(test_ttnn_matmul_sweep
    tests/ttnn/unit_tests/gtests/test_matmul_sweep.cpp
)

add_ttnn_emule_test(test_ttnn_relational_int
    tests/ttnn/unit_tests/gtests/test_relational_int.cpp
)
