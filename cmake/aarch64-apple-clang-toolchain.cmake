set(CMAKE_SYSTEM_PROCESSOR "aarch64")
set(CMAKE_SYSTEM_NAME "Darwin")

# Apple Clang from Xcode Command Line Tools
set(CMAKE_C_COMPILER /usr/bin/clang CACHE INTERNAL "C compiler")
set(CMAKE_CXX_COMPILER /usr/bin/clang++ CACHE INTERNAL "C++ compiler")

# Use libc++ (Apple's default C++ stdlib)
set(ENABLE_LIBCXX TRUE CACHE INTERNAL "Using clang's libc++")

# Apple's ld (no mold/lld on macOS by default)
# Use ld.lld if available, else leave linker as default (Apple ld)
find_program(LLD ld.lld)
if(LLD)
    set(CMAKE_LINKER_TYPE LLD)
endif()
