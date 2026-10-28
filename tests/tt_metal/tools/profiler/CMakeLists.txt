set(PROFILER_TEST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/test_full_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_timestamped_events.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_active_eth_slow_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_profiler_callbacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_profiler_decode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_profiler_fetch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_profiler_records.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_profiler_semantics.cpp
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

# host_errors.cpp holds one record type a callback can't take for each HOST_ERROR value. Each case passes only when its
# compile fails with exactly one error, its rule's message, compiled the way the records test is.
set(_records_test profiler_test_streaming_profiler_records)
foreach(
    _case
    IN
    ITEMS
        "1|a std::tuple payload takes two or more arguments"
        "2|a std::tuple payload takes two or more arguments"
        "3|Variadic<T> is a whole payload"
        "4|Variadic<T> is a whole payload"
        "5|strings are not supported in a payload"
        "6|strings are not supported in a payload"
        "7|Timestamps in metadata are not supported"
        "8|metadata is a struct"
        "9|only aggregate structs, fixed-width scalars, enums, String, Timestamp and arrays of them are supported"
        "10|only aggregate structs, fixed-width scalars, enums, String, Timestamp and arrays of them are supported"
        "11|only aggregate structs, fixed-width scalars, enums, String, Timestamp and arrays of them are supported"
        "12|only aggregate structs, fixed-width scalars, enums, String, Timestamp and arrays of them are supported"
        "13|only aggregate structs, fixed-width scalars, enums, String, Timestamp and arrays of them are supported"
        "14|this type is not supported"
        "15|this type is not supported"
        "16|arrays of arrays are not supported"
        "17|empty arrays are not supported"
        "18|templates, function-local types and types in an anonymous namespace are not supported"
        "19|Variadic<T> is a whole payload"
        "20|enums without a fixed underlying type are not supported"
        "21|unions are not supported"
        "22|const and reference members are not supported"
        "23|types with a custom copy, move or destructor are not supported"
        "24|alignment beyond alignof(std::max_align_t) is not supported"
        "25|a Batch lists each record type once"
        "26|a callback can take at most 64 record types"
)
    string(FIND "${_case}" "|" _bar)
    string(SUBSTRING "${_case}" 0 ${_bar} _id)
    math(EXPR _bar "${_bar} + 1")
    string(SUBSTRING "${_case}" ${_bar} -1 _message)
    add_test(
        NAME streaming_profiler_host_error_${_id}
        COMMAND
            ${CMAKE_COMMAND} -DCOMPILER=${CMAKE_CXX_COMPILER}
            "-DFLAGS=-std=c++20;-I$<JOIN:$<TARGET_PROPERTY:${_records_test},INCLUDE_DIRECTORIES>,;-I>;-D$<JOIN:$<TARGET_PROPERTY:${_records_test},COMPILE_DEFINITIONS>,;-D>"
            -DMACRO=HOST_ERROR -DCASE=${_id} "-DMESSAGE=${_message}"
            -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/host_errors.cpp -P ${CMAKE_CURRENT_SOURCE_DIR}/check_one_error.cmake
    )
endforeach()
