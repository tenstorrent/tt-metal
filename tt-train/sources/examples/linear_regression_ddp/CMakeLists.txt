project(linear_regression_ddp)

set(SOURCES main.cpp)
add_executable(linear_regression_ddp ${SOURCES})
target_link_libraries(linear_regression_ddp PRIVATE ttml)
tt_train_set_executable_rpath(linear_regression_ddp TTNN)
