set(DISTRIBUTED_TRCE_AND_EVENTS ${CMAKE_CURRENT_SOURCE_DIR}/distributed_trace_and_events.cpp)
add_executable(distributed_trace_and_events ${DISTRIBUTED_TRCE_AND_EVENTS})

target_link_libraries(
    distributed_trace_and_events
    PUBLIC
        tt_metal
        pthread
)

target_include_directories(distributed_trace_and_events PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(
    distributed_trace_and_events
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/programming_examples/distributed
)
