set(CMAKE_SYSTEM_PROCESSOR "x86_64")

set(CMAKE_C_COMPILER clang-20 CACHE INTERNAL "C compiler")

set(CMAKE_CXX_COMPILER clang++-20 CACHE INTERNAL "C++ compiler")

set(CMAKE_AR /usr/bin/llvm-ar-20 CACHE INTERNAL "Archiver")
set(CMAKE_RANLIB /usr/bin/llvm-ranlib-20 CACHE INTERNAL "Ranlib")
set(CMAKE_NM /usr/bin/llvm-nm-20 CACHE INTERNAL "NM")

# Use for configure time
set(ENABLE_LIBCXX FALSE CACHE INTERNAL "Using clang's libc++")

# Our build is super slow; put a band-aid on it by choosing a linker that can cope better.
# We really need to fix out code, though.
find_program(MOLD ld.mold)
if(MOLD)
    set(CMAKE_LINKER_TYPE MOLD)
else()
    find_program(LLD ld.lld-20)
    if(LLD)
        set(CMAKE_LINKER_TYPE LLD)
    endif()
endif()
