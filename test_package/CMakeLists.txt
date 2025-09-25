cmake_minimum_required(VERSION 3.25)
project(test_package)

find_package(
    tt-metalium
    REQUIRED
    COMPONENTS
        tt-metalium::ttnn
        tt-metalium::tt-metalium
        tt-metalium::tt_stl
)

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE main.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC tt-metalium::ttnn)
target_compile_definitions(${PROJECT_NAME} PUBLIC "$<$<CXX_COMPILER_ID:GNU>:DISABLE_NAMESPACE_STATIC_ASSERT>")
