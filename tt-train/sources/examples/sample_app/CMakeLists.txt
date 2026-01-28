project(sample_app)

set(SOURCES main.cpp)

add_executable(sample_app ${SOURCES})
target_link_libraries(sample_app PRIVATE ttml)
tt_train_set_executable_rpath(sample_app TTNN)
