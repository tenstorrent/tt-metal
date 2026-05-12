# Built as outlined in Tracy documentation (pg.12)
set(TRACY_HOME ${PROJECT_SOURCE_DIR}/tt_metal/third_party/tracy)

# Propagate ENABLE_TRACY to TRACY_ENABLE (Tracy build option)
# CMake options are propagated as PUBLIC compile definitions to tracy build
if(ENABLE_TRACY)
    set(TRACY_ENABLE ON)
else()
    set(TRACY_ENABLE OFF)
endif()

# Tracy will always be built with fallback timing enabled
# avoiding hard failures on systems without invariant TSC and using OS clocks when needed.
set(TRACY_TIMER_FALLBACK ON)

set(DEFAULT_COMPONENT_NAME ${CMAKE_INSTALL_DEFAULT_COMPONENT_NAME})
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME tracy)
set(_saved_clang_tidy "${CMAKE_CXX_CLANG_TIDY}")
set(CMAKE_CXX_CLANG_TIDY "")
add_subdirectory(${TRACY_HOME})
set(CMAKE_CXX_CLANG_TIDY "${_saved_clang_tidy}")
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME ${DEFAULT_COMPONENT_NAME})

set_target_properties(
    TracyClient
    PROPERTIES
        EXCLUDE_FROM_ALL
            TRUE
        LIBRARY_OUTPUT_DIRECTORY
            "${PROJECT_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY
            "${PROJECT_BINARY_DIR}/lib"
        OUTPUT_NAME
            "tracy"
)

if(ENABLE_TRACY)
    # This setup ensures that when someone links against TracyClient:
    # Their code preserves frame pointers (-fno-omit-frame-pointer) → backtraces work.
    # Their executable exports symbols (-rdynamic) → Tracy can map addresses to names.
    target_compile_options(
        TracyClient
        PUBLIC
            -fno-omit-frame-pointer
        PRIVATE
            "$<$<CXX_COMPILER_ID:Clang>:-Wno-conditional-uninitialized>" # FIXME: Fix this upstream
    )
    target_link_options(TracyClient PUBLIC -rdynamic)
endif()

# Our current fork of tracy does not have CMake support for these subdirectories
# Once we update, we can change this

# Build Tracy tools (csvexport and capture) using CMake
# These are CLI tools that don't need a file selector or GUI dependencies.
set(NO_FILESELECTOR ON)

# Tracy code has warnings that the parent project's -Werror would make fatal.
get_directory_property(_parent_compile_opts COMPILE_OPTIONS)
add_compile_options(-Wno-error)
add_subdirectory(${TRACY_HOME}/csvexport)
add_subdirectory(${TRACY_HOME}/capture)
set_property(
    DIRECTORY
    PROPERTY
        COMPILE_OPTIONS
            "${_parent_compile_opts}"
)
unset(NO_FILESELECTOR)

# Build Tracy profiler WASM project using Emscripten.
#
# Ninja runs each command in a clean environment (no shell profile). CI also splits
# `build_metal.sh --configure-only` from `cmake --build`, so `emcmake` is not on PATH
# at build time even though emsdk was sourced during configure. Resolve an absolute path
# at configure time (configure inherits PATH when run from build_metal.sh, or we probe
# the emsdk checkout next to the build directory).
set(_tt_tracy_emcmake "")
find_program(_tt_tracy_emcmake emcmake DOC "Emscripten emcmake (Tracy WASM)")
if(NOT _tt_tracy_emcmake AND CMAKE_HOST_UNIX)
    set(_tt_emsdk_env_sh "${CMAKE_BINARY_DIR}/emsdk/emsdk_env.sh")
    if(EXISTS "${_tt_emsdk_env_sh}")
        execute_process(
            COMMAND
                bash -c ". \"${_tt_emsdk_env_sh}\" && command -v emcmake"
            OUTPUT_VARIABLE _tt_tracy_emcmake
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _tt_tracy_emcmake_rc
            ERROR_QUIET
        )
        if(NOT _tt_tracy_emcmake_rc EQUAL 0)
            set(_tt_tracy_emcmake "")
        endif()
    endif()
endif()
if(NOT _tt_tracy_emcmake)
    message(
        FATAL_ERROR
        "Tracy WASM build requires 'emcmake' (Emscripten). "
        "Install and activate emsdk, or run ./build_metal.sh which installs it under "
        "<build-dir>/emsdk and puts emcmake on PATH during configure."
    )
endif()
get_filename_component(_tt_tracy_emcmake "${_tt_tracy_emcmake}" REALPATH)

add_custom_target(
    tracy_profiler_wasm
    ALL
    COMMAND
        ${CMAKE_COMMAND} -E echo "Building Tracy profiler WASM..."
    COMMAND
        "${_tt_tracy_emcmake}" cmake -DEMSCRIPTEN=ON -B ${CMAKE_BINARY_DIR}/profiler/build_wasm -S
        ${TRACY_HOME}/profiler
    COMMAND
        ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/profiler/build_wasm
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Building Tracy profiler WASM with Emscripten"
)
