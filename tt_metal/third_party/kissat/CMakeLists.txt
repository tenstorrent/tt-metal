# Vendored kissat_extras (incremental Kissat fork, jix/kissat_extras) built as an in-process static
# library for the fabric deterministic SAT engine. Replaces CaDiCaL's role in the deterministic path.
# See tt_metal/fabric/KISSAT_SWAP_PLAN.md for the design/decision.
#
# Pattern mirrors the vendored gimsatul target: glob src/*.c (minus the CLI main), deterministic build.h,
# -O3 -w -DNDEBUG, static PIC lib, no unity build.

# This is vendored third-party C; do not run project linters over it.
set(CMAKE_C_CLANG_TIDY "")
set(CMAKE_C_CPPCHECK "")

# build.c #includes "build.h", which upstream generates at configure time (VERSION/COMPILER/ID/BUILD/DIR
# banner strings). That file is .gitignored, so a fresh checkout lacks it. Rather than depend on running
# kissat's ./configure (non-deterministic: embeds date/host), generate a deterministic build.h here. The
# macros only feed the version banner (silenced in library use), so fixed values are fine.
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION KISSAT_VERSION)
string(STRIP "${KISSAT_VERSION}" KISSAT_VERSION)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/build.h
    "#define VERSION \"${KISSAT_VERSION}\"\n"
    "#define COMPILER \"${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}\"\n"
    "#define ID \"tt-metal-vendored\"\n"
    "#define BUILD \"tt-metal vendored (cmake)\"\n"
    "#define DIR \"tt_metal/third_party/kissat\"\n")

# All solver translation units EXCEPT main.c (which carries the CLI main()). kitten.c's main() is guarded
# by STAND_ALONE_KITTEN (undefined here), so kitten.c is safely part of the library.
file(GLOB KISSAT_SRCS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)
list(FILTER KISSAT_SRCS EXCLUDE REGEX "/main\\.c$")

add_library(kissat STATIC ${KISSAT_SRCS})

# IMPORTANT: src must NOT be on kissat's OWN compile include path. kissat ships its own src/limits.h
# (an internal header); if src is a `-I` dir, `#include <limits.h>` in kissat's headers resolves to that
# instead of the system <limits.h>, breaking UINT_MAX (upstream's Makefile avoids this by compiling from
# within src/, so `<...>` uses the system path while `"..."` sibling includes are file-relative). We rely
# on the same: kissat's own .c files find siblings via file-relative quote-includes (no -I needed), and
# only build.h (in the binary dir) needs a -I. Consumers get src via INTERFACE to find kissat.h.
target_include_directories(
    kissat
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/src  # consumers (fabric) find kissat.h; NOT on kissat's own -I path
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}      # generated build.h (see above)
)

# -O3 for solver speed, -DNDEBUG for release semantics, -w to silence the vendored code's warnings so a
# parent -Werror scope cannot break the third-party build. No unity build (many static symbol collisions).
target_compile_options(kissat PRIVATE -O3 -w)
target_compile_definitions(kissat PRIVATE NDEBUG)
set_target_properties(
    kissat
    PROPERTIES
        C_STANDARD 11
        UNITY_BUILD OFF
        POSITION_INDEPENDENT_CODE ON
)

find_package(Threads REQUIRED)
target_link_libraries(kissat PUBLIC Threads::Threads m)
