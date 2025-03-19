set(DISTRIBUTED_PROGRAM_DISPATCH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/distributed_program_dispatch.cpp)
add_executable(distributed_program_dispatch ${DISTRIBUTED_PROGRAM_DISPATCH_SRC})

target_link_libraries(
    distributed_program_dispatch
    PUBLIC
        tt_metal
        pthread
)

target_include_directories(distributed_program_dispatch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(
    distributed_program_dispatch
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/programming_examples/distributed
)
