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
# Out-of-tree only: keep all Tracy CMake state under ${CMAKE_BINARY_DIR}/profiler so removing
# the top-level build directory leaves submodule source trees clean for git.
set(_tt_profiler_binary_dir "${CMAKE_BINARY_DIR}/profiler")
set(_saved_clang_tidy "${CMAKE_CXX_CLANG_TIDY}")
set(CMAKE_CXX_CLANG_TIDY "")
add_subdirectory(${TRACY_HOME} ${_tt_profiler_binary_dir})
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

# Tracy CLI tools must run from CI artifacts without build/_deps. Capstone defaults to a
# static library when BUILD_SHARED_LIBS is OFF; tt-metal sets BUILD_SHARED_LIBS ON globally
# (cmake/project_options.cmake), which makes Tracy's vendored capstone a shared .so.
set(_tt_tracy_saved_build_shared_libs "${BUILD_SHARED_LIBS}")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Create shared libraries" FORCE)

# Tracy code has warnings that the parent project's -Werror would make fatal.
get_directory_property(_parent_compile_opts COMPILE_OPTIONS)
add_compile_options(-Wno-error)
add_subdirectory(${TRACY_HOME}/csvexport ${_tt_profiler_binary_dir}/csvexport)
add_subdirectory(${TRACY_HOME}/capture ${_tt_profiler_binary_dir}/capture)

set_property(
    DIRECTORY
    PROPERTY
        COMPILE_OPTIONS
            "${_parent_compile_opts}"
)
unset(NO_FILESELECTOR)

set(BUILD_SHARED_LIBS "${_tt_tracy_saved_build_shared_libs}" CACHE BOOL "Create shared libraries" FORCE)
unset(_tt_tracy_saved_build_shared_libs)

# Link Tracy CLI tools into build/tools/profiler/bin (not CMAKE_INSTALL_BINDIR / build/bin).
set(_tt_tracy_cli_output_dir "${CMAKE_BINARY_DIR}/tools/profiler/bin")
foreach(_tt_tracy_cli_target IN ITEMS tracy-capture tracy-csvexport tracy-capture-daemon)
    set_target_properties(
        ${_tt_tracy_cli_target}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                "${_tt_tracy_cli_output_dir}"
            EXCLUDE_FROM_INSTALL
                TRUE
    )
endforeach()
unset(_tt_tracy_cli_output_dir)

add_custom_target(tracy_profiler_cli_tools ALL)
add_dependencies(
    tracy_profiler_cli_tools
    tracy-capture
    tracy-csvexport
    tracy-capture-daemon
)

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
    # cibuildwheel / manylinux and other minimal images configure without Emscripten. Full
    # dev builds use ./build_metal.sh (emsdk + PATH) or a system emcmake on PATH at configure.
    message(
        WARNING
        "Tracy WASM viewer will not be built: 'emcmake' not found and "
        "${CMAKE_BINARY_DIR}/emsdk/emsdk_env.sh is missing or unusable. "
        "Host Tracy profiling still works; install emsdk or run build_metal.sh to enable the web UI."
    )
else()
    get_filename_component(_tt_tracy_emcmake "${_tt_tracy_emcmake}" REALPATH)

    add_custom_target(
        tracy_profiler_wasm
        ALL
        COMMAND
            ${CMAKE_COMMAND} -E echo "Building Tracy profiler WASM..."
        COMMAND
            "${_tt_tracy_emcmake}" cmake -DEMSCRIPTEN=ON -B ${_tt_profiler_binary_dir}/build_wasm -S
            ${TRACY_HOME}/profiler
        COMMAND
            ${CMAKE_COMMAND} --build ${_tt_profiler_binary_dir}/build_wasm
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Building Tracy profiler WASM with Emscripten"
    )
    add_dependencies(tracy_profiler_wasm tracy_profiler_cli_tools)
endif()
