
add_executable(watcher_dump ${CMAKE_CURRENT_SOURCE_DIR}/watcher_dump.cpp)
target_link_libraries(watcher_dump PRIVATE test_metal_common_libs)
target_include_directories(watcher_dump PRIVATE
    ${UMD_HOME}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tt_metal
    ${PROJECT_SOURCE_DIR}/tt_metal/common
    ${PROJECT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(watcher_dump PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/tools)
