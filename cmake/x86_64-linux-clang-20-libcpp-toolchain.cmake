set(CMAKE_SYSTEM_PROCESSOR "x86_64")

set(CMAKE_C_COMPILER clang-20 CACHE INTERNAL "C compiler")

set(CMAKE_CXX_COMPILER clang++-20 CACHE INTERNAL "C++ compiler")

set(CMAKE_CXX_FLAGS_INIT "-stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-lc++ -lc++abi")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-lc++ -lc++abi")

# Use for configure time
set(ENABLE_LIBCXX TRUE CACHE INTERNAL "Using clang's libc++")

# Our build is super slow; put a band-aid on it by choosing a linker that can cope better.
# We really need to fix our code, though.
find_program(MOLD ld.mold)
if(MOLD)
    set(CMAKE_LINKER_TYPE MOLD)
else()
    find_program(LLD ld.lld-20)
    if(LLD)
        set(CMAKE_LINKER_TYPE LLD)
    endif()
endif()
