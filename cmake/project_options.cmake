###########################################################################################
# Project Options
#   The following options and their defaults impact what artifacts get built
###########################################################################################
option(WITH_PYTHON_BINDINGS "Enables build of python bindings" ON)
option(ENABLE_CODE_TIMERS "Enable code timers" OFF)
option(ENABLE_TRACY "Enable Tracy Profiling" ON)
option(ENABLE_LIBCXX "Enable using libc++" OFF)
option(ENABLE_BUILD_TIME_TRACE "Enable build time trace (Clang only -ftime-trace)" OFF)
option(BUILD_SHARED_LIBS "Create shared libraries" ON)
option(BUILD_PROGRAMMING_EXAMPLES "Enables build of tt_metal programming examples" OFF)
option(TT_METAL_BUILD_TESTS "Enables build of tt_metal tests" OFF)
option(TTNN_BUILD_TESTS "Enables build of ttnn tests" OFF)
option(ENABLE_CCACHE "Build with compiler cache" FALSE)
option(TT_UNITY_BUILDS "Build with Unity builds" ON)
option(BUILD_TT_TRAIN "Enables build of tt-train" OFF)
option(ENABLE_TTNN_SHARED_SUBLIBS "Use shared libraries for ttnn to speed up incremental builds" OFF)
option(TT_ENABLE_LIGHT_METAL_TRACE "Enable Light Metal Trace" ON)
option(TT_ENABLE_LTO "Build Releases with Link-Time-Optimization (LTO)" OFF)
option(ENABLE_DISTRIBUTED "Enable multihost distributed compute support (OpenMPI)" ON)
option(TT_UMD_BUILD_SIMULATION "Force UMD to include its simulation harnessing" ON)
option(TT_INSTALL "Define installation rules" ON)
option(TT_USE_SYSTEM_SFPI "Use system path for SFPI. SFPI is used to compile firmware." OFF)
option(TT_METAL_USE_EMULE "Build with tt-emule software emulation (no hardware required)" OFF)
set(TT_EMULE_PATH "" CACHE PATH "Local path to tt-emule source (overrides CPM fetch from GitHub)")
option(TT_EMULE_ASAN "Build emulator with AddressSanitizer (requires TT_METAL_USE_EMULE=ON)" OFF)

if(TT_METAL_USE_EMULE)
    set(TT_UMD_BUILD_EMULE ON)
    if(TT_EMULE_PATH)
        set(CPM_tt_emule_SOURCE "${TT_EMULE_PATH}")
    endif()
endif()

if(TT_EMULE_ASAN)
    if(NOT TT_METAL_USE_EMULE)
        message(FATAL_ERROR "TT_EMULE_ASAN requires TT_METAL_USE_EMULE=ON")
    endif()

    # SWEmuleChip's ASan poison hooks call __emule_buffer_alloc/free, defined
    # in tt-emule. The main tt-metal libraries link tt-emule and resolve them,
    # but UMD's standalone tools (harvesting, system_health, etc.) link only
    # libtt-umd.so + spdlog and have no path to those symbols — link fails.
    # Force TT_UMD_BUILD_TOOLS=OFF for ASan builds; the diagnostic tools target
    # real silicon and aren't exercised by the emulation regression anyway.
    set(TT_UMD_BUILD_TOOLS OFF CACHE BOOL "Disabled for ASan builds (tools don't link tt-emule)" FORCE)
    # -shared-libasan is required because the JIT'd kernel .so uses the
    # dynamic libclang_rt.asan; the host must match or the loader rejects
    # the kernel with "incompatible ASan runtimes".
    add_compile_options(
        -fsanitize=address
        -shared-libasan
        -fno-omit-frame-pointer
        -g
    )
    add_link_options(
        -fsanitize=address
        -shared-libasan
    )

    # Resolve the clang-rt directory so the JIT command can rpath the
    # dynamic libasan into each kernel .so. Without this, dlopen of the
    # kernel .so fails with "libclang_rt.asan-x86_64.so: cannot open
    # shared object file" unless LD_LIBRARY_PATH is set externally.
    execute_process(
        COMMAND
            ${CMAKE_CXX_COMPILER} -print-file-name=libclang_rt.asan-x86_64.so
        OUTPUT_VARIABLE _ASAN_RT_LIB
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_ASAN_RT_LIB AND IS_ABSOLUTE "${_ASAN_RT_LIB}")
        get_filename_component(_ASAN_RT_DIR "${_ASAN_RT_LIB}" DIRECTORY)
        set(TT_EMULE_ASAN_RT_DIR "${_ASAN_RT_DIR}" CACHE INTERNAL "clang-rt asan dir")
        message(STATUS "TT_EMULE_ASAN: using clang-rt at ${TT_EMULE_ASAN_RT_DIR}")
    else()
        message(
            WARNING
            "TT_EMULE_ASAN: could not resolve libclang_rt.asan-x86_64.so via "
            "${CMAKE_CXX_COMPILER}. JIT kernels may fail to dlopen."
        )
    endif()
endif()

###########################################################################################

if(WITH_PYTHON_BINDINGS)
    message(STATUS "Building with Python Bindings: nanobind")
endif()

if(CMAKE_CXX_CLANG_TIDY AND TT_UNITY_BUILDS)
    # There should be a way to have clang-tidy handle Unity builds properly,
    # but it didn't work in my brief testing.  Worth investigating deeper later
    # as it may provide some speedups.
    message(WARNING "Disabling UNITY builds for clang-tidy scans")
    set(TT_UNITY_BUILDS OFF)
endif()

if(TT_UNITY_BUILDS)
    if(CMAKE_EXPORT_COMPILE_COMMANDS)
        message(STATUS "Disabling Unity builds because CMAKE_EXPORT_COMPILE_COMMANDS is ON")
        set(TT_UNITY_BUILDS OFF)
    endif()
    if(CMAKE_VERSION VERSION_LESS "3.20.0")
        message(STATUS "CMake 3.20 or newer is required for Unity builds, disabling")
        set(TT_UNITY_BUILDS OFF)
    endif()
endif()

if(TT_INSTALL AND NOT BUILD_SHARED_LIBS)
    message(FATAL_ERROR "Shared libs are required for installation rules.  Set TT_INSTALL=OFF or BUILD_SHARED_LIBS=ON.")
endif()
