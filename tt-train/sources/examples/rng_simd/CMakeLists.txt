project(rng_simd)

set(SOURCES main.cpp)
add_executable(rng_simd ${SOURCES})
find_package(Threads REQUIRED)
target_link_libraries(
    rng_simd
    PRIVATE
        ttml
        Threads::Threads
)
