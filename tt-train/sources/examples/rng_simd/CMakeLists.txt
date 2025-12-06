project(rng_simd)

find_package(Threads REQUIRED)

# RNG SIMD benchmark
add_executable(rng_simd main.cpp)
target_link_libraries(
    rng_simd
    PRIVATE
        ttml
        Threads::Threads
)

# RNG distribution comparison tool
add_executable(rng_compare compare.cpp)
target_link_libraries(
    rng_compare
    PRIVATE
        ttml
        Threads::Threads
)

# RNG comprehensive benchmark
add_executable(rng_benchmark benchmark.cpp)
target_link_libraries(
    rng_benchmark
    PRIVATE
        ttml
        Threads::Threads
)
