
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/profiler)

set(TOOLS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/memset.cpp)

add_library(tools OBJECT ${TOOLS_SRC})
target_link_libraries(tools PUBLIC profiler llrt yaml-cpp stdc++fs m)
