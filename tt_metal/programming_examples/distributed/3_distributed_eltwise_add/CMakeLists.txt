set(DISTRIBUTED_ELTWISE_ADD_SRC ${CMAKE_CURRENT_SOURCE_DIR}/distributed_eltwise_add.cpp)
add_executable(distributed_eltwise_add ${DISTRIBUTED_ELTWISE_ADD_SRC})

target_link_libraries(
    distributed_eltwise_add
    PUBLIC
        tt_metal
        pthread
)

target_include_directories(distributed_eltwise_add PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(
    distributed_eltwise_add
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/programming_examples/distributed
)
