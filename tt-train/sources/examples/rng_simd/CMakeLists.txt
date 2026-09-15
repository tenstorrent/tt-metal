project(rng_simd)

add_executable(rng_simd main.cpp)
find_package(Threads REQUIRED)
target_link_libraries(
    rng_simd
    PRIVATE
        ttml
        Threads::Threads
)

target_precompile_headers(rng_simd REUSE_FROM ttml_pch)
