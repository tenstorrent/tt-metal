set(CMAKE_SYSTEM_PROCESSOR "x86_64")

set(CMAKE_C_COMPILER clang-17 CACHE INTERNAL "C compiler")

set(CMAKE_CXX_COMPILER clang++-17 CACHE INTERNAL "C++ compiler")

set(CMAKE_CXX_FLAGS_INIT "-stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-lc++ -lc++abi")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-lc++ -lc++abi")

# Use for configure time
set(ENABLE_LIBCXX TRUE CACHE INTERNAL "Using clang's libc++")
