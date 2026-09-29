project(graph_capture)

set(SOURCES main.cpp)

add_executable(graph_capture ${SOURCES})
target_link_libraries(graph_capture PRIVATE ttml)

target_precompile_headers(graph_capture REUSE_FROM ttml_pch)
