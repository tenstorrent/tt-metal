function(CREATE_EAGER_TEST_EXE TESTLIST)
    foreach(TEST_SRC ${TESTLIST})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        get_filename_component(TEST_DIR ${TEST_SRC} DIRECTORY)
        set(TEST_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_SRC})
        # unit_tests and tensor are already taken as target names/executables
        if(${TEST_NAME} STREQUAL "unit_tests")
            set(TEST_TARGET "test_unit_tests")
        elseif(${TEST_NAME} STREQUAL "tensor")
            set(TEST_TARGET "test_tensor")
        else()
            set(TEST_TARGET ${TEST_NAME})
        endif()
        add_executable(${TEST_TARGET} ${TEST_SRC_PATH})
        TT_ENABLE_UNITY_BUILD(${TEST_TARGET})

        target_link_libraries(
            ${TEST_TARGET}
            PUBLIC
                test_eager_common_libs
                ttnn
        )
        target_include_directories(
            ${TEST_TARGET}
            PRIVATE
                ${UMD_HOME}
                ${PROJECT_SOURCE_DIR}
                ${PROJECT_SOURCE_DIR}/tt_metal
                ${PROJECT_SOURCE_DIR}/ttnn/cpp
                ${PROJECT_SOURCE_DIR}/tt_metal/common
                ${PROJECT_SOURCE_DIR}/tests
                ${CMAKE_CURRENT_SOURCE_DIR}
        )
        set_target_properties(
            ${TEST_TARGET}
            PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY
                    ${PROJECT_BINARY_DIR}/test/tt_eager/${TEST_DIR}
        )
        list(APPEND EAGER_TEST_TARGETS ${TEST_TARGET})
    endforeach()
    set(EAGER_TEST_TARGETS "${EAGER_TEST_TARGETS}" PARENT_SCOPE)
endfunction()

function(CREATE_PGM_EXAMPLES_EXE TESTLIST SUBDIR)
    foreach(TEST_SRC ${TESTLIST})
        get_filename_component(TEST_TARGET ${TEST_SRC} NAME_WE)

        add_executable(${TEST_TARGET} ${TEST_SRC})
        target_link_libraries(
            ${TEST_TARGET}
            PUBLIC
                tt_metal
                m
                pthread
        )
        target_include_directories(
            ${TEST_TARGET}
            PRIVATE
                ${UMD_HOME}
                ${PROJECT_SOURCE_DIR}
                ${PROJECT_SOURCE_DIR}/tt_metal
                ${PROJECT_SOURCE_DIR}/tt_metal/common
                ${CMAKE_CURRENT_SOURCE_DIR}
        )
        set_target_properties(
            ${TEST_TARGET}
            PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY
                    ${PROJECT_BINARY_DIR}/programming_examples/${SUBDIR}
        )
        list(APPEND PROGRAMMING_EXAMPLES_TEST_TARGETS ${TEST_TARGET})
    endforeach()
    set(PROGRAMMING_EXAMPLES_TEST_TARGETS "${PROGRAMMING_EXAMPLES_TEST_TARGETS}" PARENT_SCOPE)
endfunction()
