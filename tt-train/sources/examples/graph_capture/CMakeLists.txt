project(graph_capture)

set(SOURCES main.cpp)

add_executable(graph_capture ${SOURCES})
target_link_libraries(graph_capture PRIVATE ttml)

tt_reuse_precompile_headers(graph_capture ttml_pch)
