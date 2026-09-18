project(linear_regression)

set(SOURCES main.cpp)
add_executable(linear_regression ${SOURCES})
target_link_libraries(linear_regression PRIVATE ttml)

target_precompile_headers(linear_regression REUSE_FROM ttml_pch)
