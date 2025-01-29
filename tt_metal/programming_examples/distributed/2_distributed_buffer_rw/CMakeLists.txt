set(DISTRIBUTED_BUFFER_RW_SRC ${CMAKE_CURRENT_SOURCE_DIR}/distributed_buffer_rw.cpp)
add_executable(distributed_buffer_rw ${DISTRIBUTED_BUFFER_RW_SRC})

target_link_libraries(
    distributed_buffer_rw
    PUBLIC
        tt_metal
        pthread
)

target_include_directories(distributed_buffer_rw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(
    distributed_buffer_rw
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/programming_examples/distributed
)
