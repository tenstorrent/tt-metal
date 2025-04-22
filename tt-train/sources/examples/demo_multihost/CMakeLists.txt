project(demo_multihost)

set(SOURCES main.cpp)

add_executable(demo_multihost ${SOURCES})
target_link_libraries(demo_multihost PRIVATE ttml)
