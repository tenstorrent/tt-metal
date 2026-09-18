set(PROFILER_TEST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/test_full_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_timestamped_events.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_profiler_decode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_profiler_fetch.cpp
)

foreach(TEST_SRC ${PROFILER_TEST_SRCS})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    set(TEST_TARGET profiler_${TEST_NAME})

    add_executable(${TEST_TARGET} ${TEST_SRC})
    target_link_libraries(${TEST_TARGET} PRIVATE tt_metal)

    set_target_properties(
        ${TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal/tools/profiler
            OUTPUT_NAME
                ${TEST_NAME}
    )
endforeach()

# These tests create Ethernet kernels through the internal kernel API.
foreach(TEST_TARGET profiler_test_full_buffer profiler_test_timestamped_events)
    target_include_directories(
        ${TEST_TARGET}
        PRIVATE
            ${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/tt_metal
            ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
            ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
            ${PROJECT_SOURCE_DIR}/tt_metal/llrt
    )
endforeach()

# Streaming tests compile the internal decoder, which uses SIMDe directly.
foreach(TEST_TARGET profiler_test_streaming_profiler_decode profiler_test_streaming_profiler_fetch)
    target_link_libraries(${TEST_TARGET} PRIVATE simde::simde)
    target_include_directories(
        ${TEST_TARGET}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/tt_metal
            ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
    )
endforeach()
