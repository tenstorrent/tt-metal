# Vendored gimsatul v1.1.3 (parallel SAT solver) built as an in-process static
# library with a thin C shim (gim_shim.c). Replaces the historical subprocess +
# DIMACS-file integration in the fabric topology SAT solver.
#
# CRITICAL: everything here MUST compile with -DNDEBUG. struct ruler has an
# `#ifndef NDEBUG original` field, so a mismatched NDEBUG changes sizeof(struct
# ruler) and silently corrupts the ABI between the shim and the solver objects.

# This is vendored third-party C; do not run project linters over it.
set(CMAKE_C_CLANG_TIDY "")
set(CMAKE_C_CPPCHECK "")

# All solver translation units EXCEPT gimsatul.c (which carries main()).
file(GLOB GIMSATUL_SRCS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)
list(FILTER GIMSATUL_SRCS EXCLUDE REGEX "/gimsatul\\.c$")

add_library(gimsatul STATIC ${GIMSATUL_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/gim_shim.c)

# The shim includes gimsatul headers by bare name; solver .c files include their
# own siblings. Both live in src/. gim_shim.h lives one level up.
target_include_directories(
    gimsatul
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# -DNDEBUG (ABI, see above) + -O3 for solver speed. -w silences the vendored
# code's warnings so a parent -Werror scope cannot break the third-party build.
# Never let a unity build fold these translation units together (many share
# static symbol names).
target_compile_options(gimsatul PRIVATE -O3 -w -pthread)
target_compile_definitions(gimsatul PRIVATE NDEBUG)
set_target_properties(
    gimsatul
    PROPERTIES
        C_STANDARD 11
        UNITY_BUILD OFF
        POSITION_INDEPENDENT_CODE ON
)

find_package(Threads REQUIRED)
target_link_libraries(gimsatul PUBLIC Threads::Threads m)
