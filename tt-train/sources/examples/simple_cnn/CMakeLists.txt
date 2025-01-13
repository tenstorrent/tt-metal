project(simple_cnn)

set(SOURCES main.cpp)

add_executable(simple_cnn ${SOURCES})
target_link_libraries(simple_cnn PRIVATE ttml)
