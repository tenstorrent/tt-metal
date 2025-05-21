add_executable(watcher_dump ${CMAKE_CURRENT_SOURCE_DIR}/watcher_dump.cpp)
target_link_libraries(
    watcher_dump
    PRIVATE
        Metalium::Metal
        Metalium::Metal::LLRT
        TT::STL
)
target_include_directories(
    watcher_dump
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/common
        ${PROJECT_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(
    watcher_dump
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)

target_precompile_headers(watcher_dump REUSE_FROM TT::CommonPCH)
