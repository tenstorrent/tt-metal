set(DISTRIBUTED_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/distributed.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_device_view.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_buffer.cpp
)

add_library(distributed OBJECT ${DISTRIBUTED_SRC})
target_link_libraries(distributed PUBLIC common PRIVATE Metalium::Metal::Impl)
