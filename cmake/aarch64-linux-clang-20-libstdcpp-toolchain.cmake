set(CMAKE_SYSTEM_PROCESSOR "aarch64")

set(CMAKE_C_COMPILER clang-20 CACHE INTERNAL "C compiler")

set(CMAKE_CXX_COMPILER clang++-20 CACHE INTERNAL "C++ compiler")

# Use for configure time
set(ENABLE_LIBCXX FALSE CACHE INTERNAL "Using clang's libc++")

# Choose the fastest available linker
find_program(MOLD ld.mold)
if(MOLD)
    set(CMAKE_LINKER_TYPE MOLD)
else()
    find_program(LLD ld.lld-20)
    if(LLD)
        set(CMAKE_LINKER_TYPE LLD)
    endif()
endif()
