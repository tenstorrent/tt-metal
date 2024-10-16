#!/bin/bash

set -eo pipefail

# Function to display help
show_help() {
    echo "Usage: $0 [options]..."
    echo "  -h, --help                       Show this help message."
    echo "  -e, --export-compile-commands    Enable CMAKE_EXPORT_COMPILE_COMMANDS."
    echo "  -c, --enable-ccache              Enable ccache for the build."
    echo "  -b, --build-type build_type      Set the build type. Default is Release. Other options are Debug, RelWithDebInfo, and CI."
    echo "  -t, --trace                      Enable build time trace (clang only)."
    echo "  -a, --enable-asan                Enable AddressSanitizer."
    echo "  -m, --enable-msan                Enable MemorySanitizer."
    echo "  -s, --enable-tsan                Enable ThreadSanitizer."
    echo "  -u, --enable-ubsan               Enable UndefinedBehaviorSanitizer."
    echo "  -p, --enable-profiler            Enable Tracy profiler."
    echo "  --install-prefix                 Where to install build artifacts."
    echo "  --build-tests                    Build All Testcases."
    echo "  --build-ttnn-tests               Build ttnn Testcases."
    echo "  --build-metal-tests              Build metal Testcases."
    echo "  --build-umd-tests                Build umd Testcases."
    echo "  --build-programming-examples     Build programming examples."
    echo "  --release                        Set the build type as Release."
    echo "  --development                    Set the build type as RelWithDebInfo."
    echo "  --debug                          Set the build type as Debug."
    echo "  --clean                          Remove build workspaces."
}

clean() {
    echo "INFO: Removing build artifacts!"
    rm -rf build_Release* build_Debug* build_RelWithDebInfo* build built
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
build_tests="OFF"
build_ttnn_tests="OFF"
build_metal_tests="OFF"
build_umd_tests="OFF"
build_programming_examples="OFF"

declare -a cmake_args

OPTIONS=h,e,c,t,a,m,s,u,b:,p
LONGOPTIONS=help,export-compile-commands,enable-ccache,enable-time-trace,enable-asan,enable-msan,enable-tsan,enable-ubsan,build-type:,enable-profiler,install-prefix:,build-tests,build-ttnn-tests,build-metal-tests,build-umd-tests,build-programming-examples,release,development,debug,clean

# Parse the options
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
if [[ $? -ne 0 ]]; then
    # If getopt has errors
    echo "INFO: Failed to parse arguments!"
    show_help
    exit 1
fi

eval set -- "$PARSED"
while true; do
    case "$1" in
        -h|--help)
            show_help;exit 0;;
        -e|--export-compile-commands)
            export_compile_commands="ON";;
        -c|--enable-ccache)
            enable_ccache="ON";;
        -t|--enable-time-trace)
            enable_time_trace="ON";;
        -a|--enable-asan)
            enable_asan="ON";;
        -m|--enable-msan)
            enable_msan="ON";;
        -s|--enable-tsan)
            enable_tsan="ON";;
        -u|--enable-ubsan)
            enable_ubsan="ON";;
        -b|--build-type)
            build_type="$2";shift;;
        -p|--enable-profiler)
            enable_profiler="ON";;
        --install-prefix)
            install_prefix="$2";shift;;
        --build-tests)
            build_tests="ON";;
        --build-ttnn-tests)
            build_ttnn_tests="ON";;
        --build-metal-tests)
            build_metal_tests="ON";;
        --build-umd-tests)
            build_umd_tests="ON";;
        --build-programming-examples)
            build_programming_examples="ON";;
        --release)
            build_type="Release";;
        --development)
            build_type="RelWithDebInfo";;
        --debug)
            build_type="Debug";;
        --clean)
	    clean; exit 0;;
        --)
            shift;break;;
    esac
    shift
done

# Check if there are unrecognized positional arguments left
if [[ $# -gt 0 ]]; then
    echo "ERROR: Unrecognized positional argument(s): $@"
    show_help
    exit 1
fi

# Validate the build_type
VALID_BUILD_TYPES=("Release" "Debug" "RelWithDebInfo")
if [[ ! " ${VALID_BUILD_TYPES[@]} " =~ " ${build_type} " ]]; then
    echo "ERROR: Invalid build type '$build_type'. Allowed values are Release, Debug, RelWithDebInfo."
    show_help
    exit 1
fi

build_dir="build_$build_type"

if [ "$enable_profiler" = "ON" ]; then
    build_dir="${build_dir}_tracy"
fi

install_prefix_default=$build_dir
cmake_install_prefix=${install_prefix:="${install_prefix_default}"}

# Set the python environment directory if not already set
if [ -z "$PYTHON_ENV_DIR" ]; then
    PYTHON_ENV_DIR=$(pwd)/python_env
fi

# Debug output to verify parsed options
echo "INFO: Export compile commands: $export_compile_commands"
echo "INFO: Enable ccache: $enable_ccache"
echo "INFO: Build type: $build_type"
echo "INFO: Enable time trace: $enable_time_trace"
echo "INFO: Enable AddressSanitizer: $enable_asan"
echo "INFO: Enable MemorySanitizer: $enable_msan"
echo "INFO: Enable ThreadSanitizer: $enable_tsan"
echo "INFO: Enable UndefinedBehaviorSanitizer: $enable_ubsan"
echo "INFO: Build directory: $build_dir"
echo "INFO: Install Prefix: $cmake_install_prefix"
echo "INFO: Build tests: $build_tests"

# Prepare cmake arguments
cmake_args+=("-B" "$build_dir")
cmake_args+=("-G" "Ninja")
cmake_args+=("-DCMAKE_BUILD_TYPE=$build_type")
cmake_args+=("-DCMAKE_INSTALL_PREFIX=$cmake_install_prefix")

if [ "$enable_ccache" = "ON" ]; then
    cmake_args+=("-DCMAKE_C_COMPILER_LAUNCHER=ccache" "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache")
fi

if [ "$enable_time_trace" = "ON" ]; then
    cmake_args+=("-DENABLE_BUILD_TIME_TRACE=ON")
fi

if [ "$enable_asan" = "ON" ]; then
    cmake_args+=("-DENABLE_ASAN=ON")
fi

if [ "$enable_msan" = "ON" ]; then
    cmake_args+=("-DENABLE_MSAN=ON")
fi

if [ "$enable_tsan" = "ON" ]; then
    cmake_args+=("-DENABLE_TSAN=ON")
fi

if [ "$enable_ubsan" = "ON" ]; then
    cmake_args+=("-DENABLE_UBSAN=ON")
fi

if [ "$enable_profiler" = "ON" ]; then
    cmake_args+=("-DENABLE_TRACY=ON")
fi

if [ "$export_compile_commands" = "ON" ]; then
    cmake_args+=("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
fi

if [ "$build_tests" = "ON" ]; then
    cmake_args+=("-DTT_METAL_BUILD_TESTS=ON")
    cmake_args+=("-DTTNN_BUILD_TESTS=ON")
    cmake_args+=("-DTT_UMD_BUILD_TESTS=ON")
fi

if [ "$build_metal_tests" = "ON" ]; then
    cmake_args+=("-DTT_METAL_BUILD_TESTS=ON")
fi

if [ "$build_ttnn_tests" = "ON" ]; then
    cmake_args+=("-DTTNN_BUILD_TESTS=ON")
fi

if [ "$build_tt_umd_tests" = "ON" ]; then
    cmake_args+=("-DTT_UMD_BUILD_TESTS=ON")
fi

if [ "$build_programming_examples" = "ON" ]; then
    cmake_args+=("-DBUILD_PROGRAMMING_EXAMPLES=ON")
fi

# Create and link the build directory
mkdir -p $build_dir
ln -nsf $build_dir build

echo "INFO: Configuring Project"
echo "INFO: Running: cmake "${cmake_args[@]}""
cmake "${cmake_args[@]}"

# Build libraries and cpp tests
echo "INFO: Building Project"
cmake --build $build_dir --target install
