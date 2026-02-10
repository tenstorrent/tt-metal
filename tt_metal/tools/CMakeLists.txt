add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/watcher_dump)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lightmetal_runner)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mem_bench)

# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

add_library(tools OBJECT ${TOOLS_SRC})
target_link_libraries(
    tools
    PUBLIC
        Metalium::Metal::Impl
        Metalium::Metal::LLRT
        Metalium::Metal
    PRIVATE
        TT::Metalium::HostDevCommon
)
