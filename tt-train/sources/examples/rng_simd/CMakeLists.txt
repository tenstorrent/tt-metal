project(rng_simd)

add_executable(rng_simd main.cpp)
find_package(Threads REQUIRED)
target_link_libraries(
    rng_simd
    PRIVATE
        ttml
        Threads::Threads
)
tt_train_set_executable_rpath(rng_simd TTNN)
