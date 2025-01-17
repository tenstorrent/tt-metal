set(PROFILER_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_metal_profiler.cpp
)

add_library(profiler OBJECT ${PROFILER_SRC})
target_link_libraries(
    profiler
    PUBLIC
        common
    PRIVATE
        Tracy::TracyClient
        Taskflow::Taskflow
        TT::Metalium::HostDevCommon
)
