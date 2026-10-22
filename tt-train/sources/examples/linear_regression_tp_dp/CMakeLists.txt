project(linear_regression_tp_dp)

set(SOURCES main.cpp)
add_executable(linear_regression_tp_dp ${SOURCES})
target_link_libraries(linear_regression_tp_dp PRIVATE ttml)

tt_reuse_precompile_headers(linear_regression_tp_dp ttml_pch)
