set(CMAKE_SYSTEM_PROCESSOR "x86_64")

set(CMAKE_C_COMPILER clang-18 CACHE INTERNAL "C compiler")

set(CMAKE_CXX_COMPILER clang++-18 CACHE INTERNAL "C++ compiler")

set(ENABLE_LIBCXX FALSE CACHE INTERNAL "Using clang's libc++")

find_program(MOLD ld.mold)
if(MOLD)
    set(CMAKE_LINKER_TYPE MOLD)
else()
    find_program(LLD ld.lld-18)
    if(LLD)
        set(CMAKE_LINKER_TYPE LLD)
    endif()
endif()
