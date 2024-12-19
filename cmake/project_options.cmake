###########################################################################################
# Project Options
#   The following options and their defaults impact what artifacts get built
###########################################################################################
option(WITH_PYTHON_BINDINGS "Enables build of python bindings" ON)
option(ENABLE_CODE_TIMERS "Enable code timers" OFF)
option(ENABLE_TRACY "Enable Tracy Profiling" OFF)
option(ENABLE_LIBCXX "Enable using libc++" ON)
option(ENABLE_BUILD_TIME_TRACE "Enable build time trace (Clang only -ftime-trace)" OFF)
option(BUILD_SHARED_LIBS "Create shared libraries" ON)
option(ENABLE_ASAN "Enable build with AddressSanitizer" OFF)
option(ENABLE_MSAN "Enable build with MemorySanitizer" OFF)
option(ENABLE_TSAN "Enable build with ThreadSanitizer" OFF)
option(ENABLE_UBSAN "Enable build with UndefinedBehaviorSanitizer" OFF)
option(BUILD_PROGRAMMING_EXAMPLES "Enables build of tt_metal programming examples" OFF)
option(TT_METAL_BUILD_TESTS "Enables build of tt_metal tests" OFF)
option(TTNN_BUILD_TESTS "Enables build of ttnn tests" OFF)
option(ENABLE_CCACHE "Build with compiler cache" FALSE)
option(TT_UNITY_BUILDS "Build with Unity builds" ON)
option(BUILD_TT_TRAIN "Enables build of tt-train" OFF)
###########################################################################################

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
