#!/bin/bash

set -eo pipefail

FLAVOR=`grep '^ID=' /etc/os-release | awk -F= '{print $2}' | tr -d '"'`
ARCH=`uname -m`

# VERSION_ID and BUILD_ID are standard within /etc/os-release but both optional
source /etc/os-release
VERSION="unknown-version"
if [[ -v "$VERSION_ID" ]]; then
    VERSION="${VERSION_ID}"
elif [[ -v "$BUILD_ID" ]]; then
    VERSION="${BUILD_ID}"
fi

# Function to display help
show_help() {
    echo "Usage: $0 [options]..."
    echo "  -h, --help                       Show this help message."
    echo "  -e, --export-compile-commands    Enable CMAKE_EXPORT_COMPILE_COMMANDS."
    echo "  -c, --enable-ccache              Enable ccache for the build."
    echo "  -b, --build-type build_type      Set the build type. Default is Release."
    echo "  -t, --enable-time-trace          Enable build time trace (clang only)."
    echo "  --disable-profiler               Disable Tracy profiler (enabled by default)."
    echo "  --install-prefix                 Where to install build artifacts."
    echo "  --build-dir                      Build directory."
    echo "  --build-tests                    Build All Testcases."
    echo "  --build-ttnn-tests               Build ttnn Testcases."
    echo "  --build-metal-tests              Build metal Testcases."
    echo "  --build-umd-tests                Build umd Testcases."
    echo "  --build-programming-examples     Build programming examples."
    echo "  --build-tt-train                 Build tt-train."
    echo "  --build-packages                 Build installation packages (.deb)"
    echo "  --build-telemetry                Build tt-telemetry server."
    echo "  --build-all                      Build all optional components."
    echo "  --release                        Set the build type as Release."
    echo "  --development                    Set the build type as RelWithDebInfo."
    echo "  --debug                          Set the build type as Debug."
    echo "  --clean                          Remove build workspaces."
    echo "  --build-static-libs              Build tt_metal (not ttnn) as a static lib (BUILD_SHARED_LIBS=OFF)"
    echo "  --disable-unity-builds           Disable Unity builds"
    echo "  --disable-light-metal-trace      Disable Light Metal tracing to binary."
    echo "  --cxx-compiler-path              Set path to C++ compiler."
    echo "  --c-compiler-path                Set path to C++ compiler."
    echo "  --cpm-source-cache               Set path to CPM Source Cache."
    echo "  --cpm-use-local-packages         Attempt to use locally installed dependencies."
    echo "  --ttnn-shared-sub-libs           Use shared libraries for ttnn."
    echo "  --toolchain-path                 Set path to CMake toolchain file."
    echo "  --configure-only                 Only configure the project, do not build."
    echo "  --without-distributed            Disable distributed compute support (OpenMPI dependency). Enabled by default."
    echo "  --without-python-bindings        Disable Python bindings (ttnncpp will be available as standalone library, otherwise ttnn will include the cpp backend and the python bindings), Enabled by default"
    echo "  --enable-fake-kernels-target     Enable fake kernels target, to enable generation of compile_commands.json for the kernels to enable IDE support."
    echo "  --enable-lto                     Enable Link Time Optimization (LTO) for Release/RelWithDebInfo builds."
}

clean() {
    echo "INFO: Removing build artifacts!"
    rm -rf build_Release* build_Debug* build_RelWithDebInfo* build_ASan* build_TSan* build_CodeCoverage* build_ASanCoverage build built .cpmcache
    rm -rf ~/.cache/tt-metal-cache /tmp/tt-metal-cache
    if [[ ! -z $TT_METAL_CACHE ]]; then
        echo "User has TT_METAL_CACHE set, please make sure you delete it in order to delete all artifacts!"
    fi
}

# Parse CLI options
export_compile_commands="OFF"
enable_ccache="OFF"
enable_time_trace="OFF"
build_type="Release"
disable_profiler="OFF"
build_dir=""
build_tests="OFF"
build_ttnn_tests="OFF"
build_metal_tests="OFF"
build_umd_tests="OFF"
build_programming_examples="OFF"
build_tt_train="OFF"
build_telemetry="OFF"
build_static_libs="OFF"
unity_builds="ON"
light_metal_trace="ON"
build_packages="OFF"
build_all="OFF"
cxx_compiler_path=""
cpm_source_cache=""
c_compiler_path=""
ttnn_shared_sub_libs="OFF"
toolchain_path="cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake"


configure_only="OFF"
enable_distributed="ON"
with_python_bindings="ON"
enable_fake_kernels_target="OFF"
enable_lto="OFF"

declare -a cmake_args

OPTIONS=h,e,c,t,a,m,s,u,b:
LONGOPTIONS="
help
build-all
export-compile-commands
enable-ccache
enable-time-trace
build-type:
disable-profiler
install-prefix:
build-dir:
build-tests
build-ttnn-tests
build-metal-tests
build-umd-tests
build-programming-examples
build-tt-train
build-packages
build-telemetry
build-static-libs
disable-unity-builds
disable-light-metal-trace
release
development
debug
clean
cxx-compiler-path:
cpm-source-cache:
cpm-use-local-packages
c-compiler-path:
ttnn-shared-sub-libs
toolchain-path:
configure-only
without-distributed
without-python-bindings
enable-fake-kernels-target
enable-lto
"

# Flatten LONGOPTIONS into a comma-separated string for getopt
LONGOPTIONS=$(echo "$LONGOPTIONS" | tr '\n' ',' | sed 's/,$//')

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
            export_compile_commands="ON";unity_builds="OFF";;
        -c|--enable-ccache)
            enable_ccache="ON";;
        -t|--enable-time-trace)
            enable_time_trace="ON";;
        --without-distributed)
            enable_distributed="OFF";;
	--build-dir)
            build_dir="$2";shift;;
        -b|--build-type)
            build_type="$2";shift;;
        --disable-profiler)
            disable_profiler="ON";;
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
        --build-tt-train)
            build_tt_train="ON";;
        --build-packages)
            build_packages="ON";;
        --build-telemetry)
            build_telemetry="ON";;
        --build-static-libs)
            build_static_libs="ON";;
        --build-all)
            build_all="ON";;
        --ttnn-shared-sub-libs)
            ttnn_shared_sub_libs="ON";;
        --configure-only)
            configure_only="ON";;
        --without-python-bindings)
            with_python_bindings="OFF";;
        --enable-fake-kernels-target)
            enable_fake_kernels_target="ON";;
        --enable-lto)
            enable_lto="ON";;
        --disable-unity-builds)
	    unity_builds="OFF";;
        --disable-light-metal-trace)
            light_metal_trace="OFF";;
        --cxx-compiler-path)
            cxx_compiler_path="$2";shift;;
        --cpm-source-cache)
            cpm_source_cache="$2";shift;;
        --c-compiler-path)
            c_compiler_path="$2";shift;;
        --toolchain-path)
            toolchain_path="$2";shift;;
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

# Determine Tracy default: enabled unless explicitly disabled
tracy_enabled="ON"
if [ "$disable_profiler" = "ON" ]; then
    tracy_enabled="OFF"
fi

# Validate the build_type
VALID_BUILD_TYPES=("Release" "Debug" "RelWithDebInfo" "ASan" "TSan" "CodeCoverage" "ASanCoverage")
if [[ ! " ${VALID_BUILD_TYPES[@]} " =~ " ${build_type} " ]]; then
    echo "ERROR: Invalid build type '$build_type'. Allowed values are ${VALID_BUILD_TYPES[*]}."
    show_help
    exit 1
fi

# Disable unity builds for CodeCoverage builds to get accurate per-file coverage
if [[ "$build_type" == "CodeCoverage" || "$build_type" == "ASanCoverage" ]]; then
    unity_builds="OFF"
fi

# If build-dir is not specified
# Use build_type to choose a default path
if [ "$build_dir" = "" ]; then
    build_dir="build_$build_type"
fi

# Create and link the build directory
mkdir -p $build_dir
ln -nsf $build_dir build

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
echo "INFO: Build directory: $build_dir"
echo "INFO: Install Prefix: $cmake_install_prefix"
echo "INFO: Build tests: $build_tests"
echo "INFO: Enable Unity builds: $unity_builds"
echo "INFO: TTNN Shared sub libs : $ttnn_shared_sub_libs"
echo "INFO: Enable Light Metal Trace: $light_metal_trace"
echo "INFO: Enable Distributed: $enable_distributed"
echo "INFO: With python bindings: $with_python_bindings"
echo "INFO: Enable Tracy: $tracy_enabled"
echo "INFO: Enable LTO: $enable_lto"

# Prepare cmake arguments
cmake_args+=("-B" "$build_dir")
cmake_args+=("-G" "Ninja")
cmake_args+=("-DCMAKE_BUILD_TYPE=$build_type")
cmake_args+=("-DCMAKE_INSTALL_PREFIX=$cmake_install_prefix")

if [ "$cxx_compiler_path" != "" ]; then
    echo "INFO: C++ compiler: $cxx_compiler_path"
    cmake_args+=("-DCMAKE_CXX_COMPILER=$cxx_compiler_path")
fi
if [ "$c_compiler_path" != "" ]; then
    echo "INFO: C compiler: $c_compiler_path"
    cmake_args+=("-DCMAKE_C_COMPILER=$c_compiler_path")
fi

if [ "$cpm_source_cache" != "" ]; then
    echo "INFO: CPM_SOURCE_CACHE: $cpm_source_cache"
    cmake_args+=("-DCPM_SOURCE_CACHE=$cpm_source_cache")
fi

if [ "$enable_ccache" = "ON" ]; then
    cmake_args+=("-DENABLE_CCACHE=TRUE")
fi

if [ "$enable_time_trace" = "ON" ]; then
    cmake_args+=("-DENABLE_BUILD_TIME_TRACE=ON")
fi

if [ "$disable_profiler" = "ON" ]; then
    cmake_args+=("-DENABLE_TRACY=OFF")
fi

if [ "$export_compile_commands" = "ON" ]; then
    cmake_args+=("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
else
    cmake_args+=("-DCMAKE_EXPORT_COMPILE_COMMANDS=OFF")
fi

if [ "$ttnn_shared_sub_libs" = "ON" ]; then
    cmake_args+=("-DENABLE_TTNN_SHARED_SUBLIBS=ON")
fi

if [ "$build_tests" = "ON" ]; then
    cmake_args+=("-DTT_METAL_BUILD_TESTS=ON")
    cmake_args+=("-DTTNN_BUILD_TESTS=ON")
fi

if [ "$build_metal_tests" = "ON" ]; then
    cmake_args+=("-DTT_METAL_BUILD_TESTS=ON")
fi

if [ "$build_ttnn_tests" = "ON" ]; then
    cmake_args+=("-DTTNN_BUILD_TESTS=ON")
fi

if [ "$build_umd_tests" = "ON" ]; then
    cmake_args+=("-DTT_UMD_BUILD_TESTS=ON")
fi

if [ "$build_programming_examples" = "ON" ]; then
    cmake_args+=("-DBUILD_PROGRAMMING_EXAMPLES=ON")
fi

if [ "$build_tt_train" = "ON" ]; then
    cmake_args+=("-DBUILD_TT_TRAIN=ON")
fi

if [ "$build_telemetry" = "ON" ]; then
    cmake_args+=("-DBUILD_TELEMETRY=ON")
fi

if [ "$build_static_libs" = "ON" ]; then
    cmake_args+=("-DBUILD_SHARED_LIBS=OFF")
    cmake_args+=("-DTT_INSTALL=OFF")
fi

if [ "$unity_builds" = "ON" ]; then
    cmake_args+=("-DTT_UNITY_BUILDS=ON")
else
    cmake_args+=("-DTT_UNITY_BUILDS=OFF")
fi

if [ "$light_metal_trace" = "ON" ]; then
    cmake_args+=("-DTT_ENABLE_LIGHT_METAL_TRACE=ON")
else
    cmake_args+=("-DTT_ENABLE_LIGHT_METAL_TRACE=OFF")
fi

if [ "$build_all" = "ON" ]; then
    cmake_args+=("-DTT_METAL_BUILD_TESTS=ON")
    cmake_args+=("-DTTNN_BUILD_TESTS=ON")
    cmake_args+=("-DBUILD_PROGRAMMING_EXAMPLES=ON")
    cmake_args+=("-DBUILD_TT_TRAIN=ON")
    cmake_args+=("-DBUILD_TELEMETRY=ON")
fi

if [ "$light_metal_trace" = "ON" ]; then
    cmake_args+=("-DTT_ENABLE_LIGHT_METAL_TRACE=ON")
else
    cmake_args+=("-DTT_ENABLE_LIGHT_METAL_TRACE=OFF")
fi

if [ "$with_python_bindings" = "ON" ]; then
    cmake_args+=("-DWITH_PYTHON_BINDINGS=ON")
    cmake_args+=("-DPython3_EXECUTABLE=$(which python3)")
    cmake_args+=("-DPython3_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])")")
    cmake_args+=("-DPython3_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') + '/libpython' + sysconfig.get_config_var('LDVERSION') + '.so')")")
else
    cmake_args+=("-DWITH_PYTHON_BINDINGS=OFF")
fi

if [ "$enable_distributed" = "ON" ]; then
    cmake_args+=("-DENABLE_DISTRIBUTED=ON")
else
    cmake_args+=("-DENABLE_DISTRIBUTED=OFF")
fi

if [ "$enable_fake_kernels_target" = "ON" ]; then
    cmake_args+=("-DENABLE_FAKE_KERNELS_TARGET=ON")
else
    cmake_args+=("-DENABLE_FAKE_KERNELS_TARGET=OFF")
fi

if [ "$enable_lto" = "ON" ]; then
    cmake_args+=("-DTT_ENABLE_LTO=ON")
fi

# toolchain and cxx_compiler settings would conflict with eachother
# only use toolchain if not setting cxx compiler directly
if [ "$cxx_compiler_path" == "" ]; then
    echo "INFO: CMAKE_TOOLCHAIN_FILE: $toolchain_path"
    cmake_args+=("-DCMAKE_TOOLCHAIN_FILE=${toolchain_path}")
fi

echo "INFO: Configuring Project"
echo "INFO: Running: cmake "${cmake_args[@]}""
cmake "${cmake_args[@]}"

if [ "$build_packages" == "ON" ];  then
  target="package"
else
  target="install"
fi

# Build libraries and cpp tests
if [ "$configure_only" = "OFF" ]; then
    echo "INFO: Building Project"
    cmake --build $build_dir --target $target
fi
