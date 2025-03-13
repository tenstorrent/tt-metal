add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/watcher_dump)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lightmetal_runner)

set(TOOLS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/memset.cpp)

add_library(tools OBJECT ${TOOLS_SRC})
target_link_libraries(
    tools
    PUBLIC
        Metalium::Metal::Impl
        Metalium::Metal::LLRT
    PRIVATE
        TT::Metalium::HostDevCommon
)
