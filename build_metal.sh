#!/bin/bash

set -eo pipefail

# Function to display help
show_help() {
    echo "Usage: $0 [-h] [-e] [-c] [-b build_type] [-t] [-a]"
    echo "  -h  Show this help message."
    echo "  -e  Enable CMAKE_EXPORT_COMPILE_COMMANDS."
    echo "  -c  Enable ccache for the build."
    echo "  -b  Set the build type. Default is Release. Other options are Debug, RelWithDebInfo, and CI."
    echo "  -t  Enable build time trace (clang only)."
    echo "  -a  Enable AddressSanitizer."
    echo "  -m  Enable MemorySanitizer."
    echo "  -s  Enable ThreadSanitizer."
    echo "  -u  Enable UndefinedBehaviorSanitizer."
    echo "  -p  Enable Tracy profiler."
}

# Parse CLI options
export_compile_commands="OFF"
enable_ccache="OFF"
enable_time_trace="OFF"
enable_asan="OFF"
enable_msan="OFF"
enable_tsan="OFF"
enable_ubsan="OFF"
build_type="Release"
enable_profiler="OFF"

while getopts "hectamsub:p" opt; do
    case ${opt} in
        h )
            show_help
            exit 0
            ;;
        e )
            export_compile_commands="ON"
            ;;
        c )
            enable_ccache="ON"
            ;;
        t )
            enable_time_trace="ON"
            ;;
        a )
            enable_asan="ON"
            ;;
        m )
            enable_msan="ON"
            ;;
        s )
            enable_tsan="ON"
            ;;
        u )
            enable_ubsan="ON"
            ;;
        b )
            build_type="$OPTARG"
            ;;
        p )
            enable_profiler="ON"
            ;;
        \? )
            show_help
            exit 1
            ;;
    esac
done

# Set the python environment directory if not already set
if [ -z "$PYTHON_ENV_DIR" ]; then
    PYTHON_ENV_DIR=$(pwd)/python_env
fi

# Debug output to verify parsed options
echo "Export compile commands: $export_compile_commands"
echo "Enable ccache: $enable_ccache"
echo "Build type: $build_type"
echo "Enable time trace: $enable_time_trace"
echo "Enable AddressSanitizer: $enable_asan"
echo "Enable MemorySanitizer: $enable_msan"
echo "Enable ThreadSanitizer: $enable_tsan"
echo "Enable UndefinedBehaviorSanitizer: $enable_ubsan"

build_dir="build_$build_type"

if [ "$enable_ccache" = "ON" ]; then
    cmake_args="$cmake_args -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
fi

if [ "$enable_time_trace" = "ON" ]; then
    cmake_args="$cmake_args -DENABLE_BUILD_TIME_TRACE=ON"
fi

if [ "$enable_asan" = "ON" ]; then
    cmake_args="$cmake_args -DENABLE_ASAN=ON"
fi

if [ "$enable_msan" = "ON" ]; then
    cmake_args="$cmake_args -DENABLE_MSAN=ON"
fi

if [ "$enable_tsan" = "ON" ]; then
    cmake_args="$cmake_args -DENABLE_TSAN=ON"
fi

if [ "$enable_ubsan" = "ON" ]; then
    cmake_args="$cmake_args -DENABLE_UBSAN=ON"
fi

if [ "$enable_profiler" = "ON" ]; then
    cmake_args="$cmake_args -DENABLE_TRACY=ON"
    build_dir="${build_dir}_tracy"
fi

# Create and link the build directory
mkdir -p $build_dir
ln -nsf $build_dir build

# Prepare cmake arguments
# -DCXX_INCLUDE_WHAT_YOU_USE=include-what-you-use
cmake_args="$cmake_args -B $build_dir -G Ninja -DCMAKE_BUILD_TYPE=$build_type -DCMAKE_EXPORT_COMPILE_COMMANDS=$export_compile_commands"

# Configure cmake
cmake $cmake_args

# Build libraries and cpp tests
echo "Building libraries and cpp tests"
cmake --build $build_dir --target tests      # <- Can also just run `ninja tests -C build`
cmake --build $build_dir --target install    # <- This is a general cmake way, can also just run `ninja install -C build`
