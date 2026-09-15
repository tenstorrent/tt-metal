project(sample_app)

set(SOURCES main.cpp)

add_executable(sample_app ${SOURCES})
target_link_libraries(sample_app PRIVATE ttml)

target_precompile_headers(sample_app REUSE_FROM ttml_pch)
