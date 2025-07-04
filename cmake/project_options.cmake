###########################################################################################
# Project Options
#   The following options and their defaults impact what artifacts get built
###########################################################################################
option(WITH_PYTHON_BINDINGS "Enables build of python bindings" ON)
option(EXPERIMENTAL_NANOBIND_BINDINGS "Enables experimental build of python bindings with nanobind" OFF)
option(ENABLE_CODE_TIMERS "Enable code timers" OFF)
option(ENABLE_TRACY "Enable Tracy Profiling" OFF)
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
option(TT_ENABLE_LTO "Build Releases with Link-Time-Optimization (LTO)" ON)
option(ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)
option(ENABLE_DISTRIBUTED "Enable multihost distributed compute support (OpenMPI)" ON)
option(TT_UMD_BUILD_SIMULATION "Force UMD to include its simulation harnessing" ON)
option(TT_INSTALL "Define installation rules" ON)
option(TT_USE_SYSTEM_SFPI "Use system path for SFPI. SFPI is used to compile firmware." OFF)

###########################################################################################

if(WITH_PYTHON_BINDINGS)
    if(EXPERIMENTAL_NANOBIND_BINDINGS)
        set(PY_BINDING "nanobind")
    else()
        set(PY_BINDING "pybind")
    endif()
    message(STATUS "Python Binding Backend: ${PY_BINDING}")
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
