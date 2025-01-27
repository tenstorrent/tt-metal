# Disable clang-tidy for all 3rd party code
set(CMAKE_C_CLANG_TIDY "")
set(CMAKE_CXX_CLANG_TIDY "")

add_subdirectory(wandb-cpp)
