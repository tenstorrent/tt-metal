project(linear_regression)

set(SOURCES main.cpp)
add_executable(benchmark_simd_vs_non ${SOURCES})
find_package(Threads REQUIRED)
target_link_libraries(
    benchmark_simd_vs_non
    PRIVATE
        ttml
        Threads::Threads
)
