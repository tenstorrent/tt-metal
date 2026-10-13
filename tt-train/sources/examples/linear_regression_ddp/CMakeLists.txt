project(linear_regression_ddp)

set(SOURCES main.cpp)
add_executable(linear_regression_ddp ${SOURCES})
target_link_libraries(linear_regression_ddp PRIVATE ttml)

tt_reuse_precompile_headers(linear_regression_ddp ttml_pch)
