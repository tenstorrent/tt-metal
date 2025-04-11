project(demo_multihost)

set(SOURCES main.cpp)

add_executable(demo_multihost ${SOURCES})
target_link_libraries(demo_multihost PRIVATE ttml)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(demo_multihost PRIVATE ${MPI_CXX_LIBRARIES})
