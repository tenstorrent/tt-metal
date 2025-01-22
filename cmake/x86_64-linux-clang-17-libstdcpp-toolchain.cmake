set(CMAKE_SYSTEM_PROCESSOR "x86_64")

set(CMAKE_C_COMPILER clang-17 CACHE INTERNAL "C compiler")

set(CMAKE_CXX_COMPILER clang++-17 CACHE INTERNAL "C++ compiler")

# Use for configure time
set(ENABLE_LIBCXX FALSE CACHE INTERNAL "Using clang's libc++")
