set(DISTRIBUTED_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/distributed.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_handle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/system_mesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_device_view.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_workload.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_workload_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_command_queue.cpp
)

add_library(distributed OBJECT ${DISTRIBUTED_SRC})
target_link_libraries(distributed PUBLIC common PRIVATE Metalium::Metal::Impl)
