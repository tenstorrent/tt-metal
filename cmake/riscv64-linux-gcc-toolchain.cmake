set(CMAKE_SYSTEM_PROCESSOR "riscv64")

# Unversioned names: install_dependencies.sh installs the distribution's default g++ on
# hosts without versioned packages, and CHECK_COMPILERS() already enforces the GCC >= 12
# minimum. Set -DCMAKE_CXX_COMPILER explicitly to pick a specific version.
set(CMAKE_C_COMPILER gcc CACHE INTERNAL "C compiler")

set(CMAKE_CXX_COMPILER g++ CACHE INTERNAL "C++ compiler")

# Use for configure time
set(ENABLE_LIBCXX FALSE CACHE INTERNAL "Using clang's libc++")

# Choose the fastest available linker
find_program(MOLD ld.mold)
if(MOLD)
    set(CMAKE_LINKER_TYPE MOLD)
else()
    find_program(LLD ld.lld)
    if(LLD)
        set(CMAKE_LINKER_TYPE LLD)
    endif()
endif()
