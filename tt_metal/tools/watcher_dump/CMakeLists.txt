
add_executable(watcher_dump ${CMAKE_CURRENT_SOURCE_DIR}/watcher_dump.cpp)
target_link_libraries(watcher_dump PUBLIC test_metal_common_libs tt_eager)
target_include_directories(watcher_dump PRIVATE
    ${UMD_HOME}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tt_metal
    ${CMAKE_SOURCE_DIR}/tt_metal/common
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(watcher_dump PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tools)
