set(MESH_CONSUMER_CHECK_SRC ${CMAKE_CURRENT_SOURCE_DIR}/mesh_consumer_check.cpp)
add_executable(mesh_consumer_check ${MESH_CONSUMER_CHECK_SRC})

target_link_libraries(
    mesh_consumer_check
    PUBLIC
        tt_metal
        pthread
)

target_include_directories(
    mesh_consumer_check
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(
    mesh_consumer_check
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/programming_examples/distributed
)
