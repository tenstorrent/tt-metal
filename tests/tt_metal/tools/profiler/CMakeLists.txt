set(PROFILER_TEST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/test_full_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_timestamped_events.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_active_eth_slow_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_profiler_callbacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_profiler_decode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_profiler_external.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_profiler_fetch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_profiler_records.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sync_events.cpp
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

# host_errors.cpp holds one record type a callback can't take for each HOST_ERROR value. Case N, compiled the way the
# records test is, passes only when it fails with exactly one error, the Nth message below.
set(_records_test profiler_test_streaming_profiler_records)
set(_id 0)
foreach(
    _message
    IN
    ITEMS
        "a std::tuple payload takes two or more arguments"
        "a std::tuple payload takes two or more arguments"
        "Variadic<T> is a whole payload"
        "Variadic<T> is a whole payload"
        "strings are not supported in a payload"
        "strings are not supported in a payload"
        "Timestamps in metadata are not supported"
        "metadata is a struct"
        "only aggregate structs, fixed-width scalars, enums, String, Timestamp and arrays of them are supported"
        "only aggregate structs, fixed-width scalars, enums, String, Timestamp and arrays of them are supported"
        "only aggregate structs, fixed-width scalars, enums, String, Timestamp and arrays of them are supported"
        "only aggregate structs, fixed-width scalars, enums, String, Timestamp and arrays of them are supported"
        "only aggregate structs, fixed-width scalars, enums, String, Timestamp and arrays of them are supported"
        "this type is not supported"
        "this type is not supported"
        "arrays of arrays are not supported"
        "empty arrays are not supported"
        "templates, function-local types and types in an anonymous namespace are not supported"
        "Variadic<T> is a whole payload"
        "enums without a fixed underlying type are not supported"
        "unions are not supported"
        "const and reference members are not supported"
        "types with a custom copy, move or destructor are not supported"
        "alignment beyond alignof(std::max_align_t) is not supported"
        "a Batch lists each record type once"
        "a callback can take at most 64 record types"
        "arrays of empty structs are not supported"
)
    math(EXPR _id "${_id} + 1")
    add_test(
        NAME streaming_profiler_host_error_${_id}
        COMMAND
            ${CMAKE_COMMAND} -DCOMPILER=${CMAKE_CXX_COMPILER}
            "-DFLAGS=-std=c++20;-I$<JOIN:$<TARGET_PROPERTY:${_records_test},INCLUDE_DIRECTORIES>,;-I>;-D$<JOIN:$<TARGET_PROPERTY:${_records_test},COMPILE_DEFINITIONS>,;-D>"
            -DMACRO=HOST_ERROR -DCASE=${_id} "-DMESSAGE=${_message}"
            -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/host_errors.cpp -P ${CMAKE_CURRENT_SOURCE_DIR}/check_one_error.cmake
    )
endforeach()
