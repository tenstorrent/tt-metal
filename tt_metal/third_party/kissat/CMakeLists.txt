# kissat_extras (incremental Kissat fork, jix/kissat_extras) built as an in-process static library for the
# fabric deterministic SAT engine. Replaces CaDiCaL's role in the deterministic min-host solver.
# See tt_metal/fabric/KISSAT_SWAP_PLAN.md for the design/decision.
#
# The source is FETCHED via CPM (DOWNLOAD_ONLY) rather than checked in, mirroring how CaDiCaL is handled in
# third_party/CMakeLists.txt: upstream ships no CMakeLists, so we download the sources and build them here
# (glob src/*.c minus the CLI main, deterministic build.h, -O3 -w -DNDEBUG, static PIC lib, no unity build).

# This is third-party C; do not run project linters over it.
set(CMAKE_C_CLANG_TIDY "")
set(CMAKE_C_CPPCHECK "")

# ── Fetch upstream source (no CMakeLists upstream → DOWNLOAD_ONLY, we build below) ────────────────────────
if(NOT DEFINED kissat_SOURCE_DIR)
    CPMAddPackage(
        NAME kissat
        GITHUB_REPOSITORY jix/kissat_extras
        GIT_TAG cacff0d5091bf32b6da6e49575ff9bfa2d564f86 # main @ 2022-08-13 (VERSION 0.1.0-dev)
        DOWNLOAD_ONLY YES
    )
endif()

# build.c #includes "build.h", which upstream generates at configure time (VERSION/COMPILER/ID/BUILD/DIR
# banner strings) via its ./configure — non-deterministic (embeds date/host) and not present after a plain
# source download. Generate a deterministic build.h here instead; the macros only feed the version banner
# (silenced in library use), so fixed values are fine. build.c finds it via the PRIVATE binary-dir -I below.
file(READ ${kissat_SOURCE_DIR}/VERSION KISSAT_VERSION)
string(STRIP "${KISSAT_VERSION}" KISSAT_VERSION)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/build.h
    "#define VERSION \"${KISSAT_VERSION}\"\n"
    "#define COMPILER \"${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}\"\n"
    "#define ID \"tt-metal-cpm\"\n"
    "#define BUILD \"tt-metal (cmake/cpm)\"\n"
    "#define DIR \"${kissat_SOURCE_DIR}\"\n")

# Clean public include for consumers: an extern-C wrapper around the upstream public kissat.h. Generated (not
# checked in) into a dir that contains ONLY this header, so kissat's src/ never lands on a consumer's -I path.
# kissat ships its own internal src/limits.h; exposing src/ would shadow the system <limits.h> in consumer
# TUs (UINT_MAX etc. become undeclared). The absolute #include below resolves the upstream header directly
# without adding src/ to the search path. Upstream kissat.h has no extern "C" guard, so we add one here.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/public/kissat.h
    "#ifndef TT_METAL_FABRIC_KISSAT_PUBLIC_H\n"
    "#define TT_METAL_FABRIC_KISSAT_PUBLIC_H\n"
    "#ifdef __cplusplus\n"
    "extern \"C\" {\n"
    "#endif\n"
    "#include \"${kissat_SOURCE_DIR}/src/kissat.h\"\n"
    "#ifdef __cplusplus\n"
    "}\n"
    "#endif\n"
    "#endif  // TT_METAL_FABRIC_KISSAT_PUBLIC_H\n")

# All solver translation units EXCEPT main.c (which carries the CLI main()). kitten.c's main() is guarded
# by STAND_ALONE_KITTEN (undefined here), so kitten.c is safely part of the library.
file(GLOB KISSAT_SRCS CONFIGURE_DEPENDS ${kissat_SOURCE_DIR}/src/*.c)
list(FILTER KISSAT_SRCS EXCLUDE REGEX "/main\\.c$")

add_library(kissat STATIC ${KISSAT_SRCS})

target_include_directories(
    kissat
    INTERFACE
        # Consumers see ONLY the generated extern-C wrapper, NOT kissat's src/ (see the wrapper note above).
        ${CMAKE_CURRENT_BINARY_DIR}/public
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

# kissat vendors its own copy of "kitten" (a sub-solver), and so does CaDiCaL. The fabric SAT engine keeps
# both backends linkable at once (CaDiCaL default, kissat opt-in via TT_TOPO_SAT_ENGINE), so both archives
# reach the same link (libtt_metal.so) and their duplicate kitten_* globals cause "multiple definition"
# errors. kissat is only ever called through its public C API, so we localize every non-kissat_* symbol in
# the built archive. See localize_symbols.cmake for the full rationale and mechanism.
find_program(TT_KISSAT_LD ld)
find_program(TT_KISSAT_OBJCOPY objcopy)
find_program(TT_KISSAT_NM nm)
find_program(TT_KISSAT_AR ar)
if(TT_KISSAT_LD AND TT_KISSAT_OBJCOPY AND TT_KISSAT_NM AND TT_KISSAT_AR)
    add_custom_command(
        TARGET kissat
        POST_BUILD
        COMMAND
            ${CMAKE_COMMAND} -DKISSAT_LIB=$<TARGET_FILE:kissat> -DTOOL_AR=${TT_KISSAT_AR}
            -DTOOL_LD=${TT_KISSAT_LD} -DTOOL_OBJCOPY=${TT_KISSAT_OBJCOPY} -DTOOL_NM=${TT_KISSAT_NM} -P
            ${CMAKE_CURRENT_SOURCE_DIR}/localize_symbols.cmake
        COMMENT "Localizing kissat non-public symbols (avoid kitten_* clash with CaDiCaL)"
        VERBATIM
    )
else()
    message(
        WARNING
        "kissat: ld/objcopy/nm/ar not all found; skipping symbol localization. Linking both kissat and "
        "CaDiCaL into one target may fail with duplicate kitten_* symbols."
    )
endif()
