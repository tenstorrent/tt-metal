set(H2D_SOCKET_TENSOR_PUSH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/h2d_socket_tensor_push.cpp)
add_executable(h2d_socket_tensor_push ${H2D_SOCKET_TENSOR_PUSH_SRC})

target_link_libraries(
    h2d_socket_tensor_push
    PUBLIC
        tt_metal
        pthread
)

target_include_directories(
    h2d_socket_tensor_push
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(
    h2d_socket_tensor_push
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/programming_examples/distributed
)
