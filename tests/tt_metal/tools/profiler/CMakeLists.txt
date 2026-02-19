set(PROFILER_TEST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/test_full_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_timestamped_events.cpp
)

foreach(TEST_SRC ${PROFILER_TEST_SRCS})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    set(TEST_TARGET profiler_${TEST_NAME})

    add_executable(${TEST_TARGET} ${TEST_SRC})
    target_link_libraries(${TEST_TARGET} PRIVATE tt_metal)

    target_include_directories(
        ${TEST_TARGET}
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set_target_properties(
        ${TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal/tools/profiler
            OUTPUT_NAME
                ${TEST_NAME}
    )
endforeach()
