set(CMAKE_SYSTEM_PROCESSOR "x86_64")

set(CMAKE_C_COMPILER clang-20 CACHE INTERNAL "C compiler")

set(CMAKE_CXX_COMPILER clang++-20 CACHE INTERNAL "C++ compiler")

# Clang on Linux may leave C compiler archive tools unresolved (CMAKE_*_COMPILER_AR/RANLIB
# as *-NOTFOUND), which later breaks static library rules in nested C projects.
# Seed host archiver tools at toolchain load time so all project()/enable_language() calls
# (including nested Tracy projects) initialize archive rules with real binaries.
if(NOT CMAKE_AR OR "${CMAKE_AR}" MATCHES "NOTFOUND")
    find_program(
        _tt_toolchain_ar
        NAMES
            ar
            llvm-ar
            gcc-ar
    )
    if(_tt_toolchain_ar)
        set(CMAKE_AR "${_tt_toolchain_ar}" CACHE FILEPATH "Path to archiver for static libraries" FORCE)
    endif()
endif()
if(NOT CMAKE_RANLIB OR "${CMAKE_RANLIB}" MATCHES "NOTFOUND")
    find_program(
        _tt_toolchain_ranlib
        NAMES
            ranlib
            llvm-ranlib
            gcc-ranlib
    )
    if(_tt_toolchain_ranlib)
        set(CMAKE_RANLIB "${_tt_toolchain_ranlib}" CACHE FILEPATH "Path to ranlib for static libraries" FORCE)
    endif()
endif()
if(CMAKE_AR AND NOT "${CMAKE_AR}" MATCHES "NOTFOUND")
    set(CMAKE_C_COMPILER_AR "${CMAKE_AR}" CACHE FILEPATH "Archiver for C compiler" FORCE)
    set(CMAKE_CXX_COMPILER_AR "${CMAKE_AR}" CACHE FILEPATH "Archiver for CXX compiler" FORCE)
endif()
if(CMAKE_RANLIB AND NOT "${CMAKE_RANLIB}" MATCHES "NOTFOUND")
    set(CMAKE_C_COMPILER_RANLIB "${CMAKE_RANLIB}" CACHE FILEPATH "Ranlib for C compiler" FORCE)
    set(CMAKE_CXX_COMPILER_RANLIB "${CMAKE_RANLIB}" CACHE FILEPATH "Ranlib for CXX compiler" FORCE)
endif()
unset(_tt_toolchain_ar CACHE)
unset(_tt_toolchain_ranlib CACHE)

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
