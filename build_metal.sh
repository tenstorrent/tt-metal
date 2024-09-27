#!/bin/bash
: '
TLDR: Follow the steps outlined below to build metal. For more in-depth information, keep reading

Steps:
    1. Create python_env (you only need to do this ONCE): ./create_venv.sh
        - The env will by default be created in $TT_METAL_HOME/python_env. If you want to change it, set the PYTHON_ENV_DIR.
        - This step is not dependent on any of the other steps; you only need to run it ONCE and in any order.
    2. Configure and generate build files: `cmake -B build -G Ninja`
        - The `-B` indicates where the build folder is; you can change the folder name to whatever you want.
        - The `-G Ninja` specifies to cmake to use the Ninja build system, which is faster and more reliable than make.
    3. Build metal: `ninja install -C build`
        - The general command would be: `cmake --build build --target install`
        - The `-C` indicates where to run the command; in this case, it will be your build folder(s).
        - We are targeting `install` since that will also just build src.
        - The install target will install pybinding .so (_C.so & _ttnn.so) into the src files, so pybinds can be used.
    4. Build cpp tests: `ninja tests -C build`
        - Building tests will also automatically build src.

Notes:
    - YOU ONLY NEED TO BUILD THE PYTHON_ENV ONCE!!!!! (unless you touch the dev python dependencies)
    - ALWAYS INSTALL, i.e., just run `ninja install -C build` as the new make build.
    - `cmake --build build --target install` is the EXACT same as running `ninja install -C build`. You would use the cmake command if you want to be agnostic of the build system (Ninja or Make).

Different configs: To change build type or use tracy, you have to change the configuration cmake step (step #2).
    - Changing build types: `cmake -B build -DCMAKE_BUILD_TYPE=<type> -G Ninja`
        - Valid CMAKE_BUILD_TYPE values: `Release`, `Debug`, `RelWithDebInfo`, `CI`
        - Release is the default if you do not set CMAKE_BUILD_TYPE.
    - Tracy: `cmake -B build -G Ninja -DENABLE_TRACY=ON`

Now you can have multiple build folders with different configs. If you want to switch, just run `ninja install -C <your_build_folder>` to install different pybinds.
    - Caveats:
        - At least one of these folders has to be named `build`, and if using tracy config, it has to be named `build`.. pending issue #8767.
        - They have to be built with the original build folder name, i.e., you cannot change the build folder name after building because it will mess up the RPATHs and linking.

Example:
    ./create_venv.sh
    cmake -B build -G Ninja && ninja -C build                       # <- Build in Release, inside folder called `build`.
    cmake -DCMAKE_BUILD_TYPE=Debug -B build_debug -G Ninja && ninja -C build_debug    # <- Build in Debug, inside folder called `build_debug`.
    source python_env/bin/activate                                  # <- You cannot run pytests yet since pybinds have not been installed.
    ninja install -C build                                          # <- Install Release pybinds.
    <run a pytest>                                                  # <- This test ran in Release config.
    ninja install -C build_debug                                    # <- Install Debug pybinds.
    <run a pytest>                                                  # <- This test ran in Debug config.

NOTE ON DEBUGGING!:
    GDB/LLDB is not stable right now. Recommend using GCC11 or higher for debugging or Clang-17 with GDB 14+.
'

set -eo pipefail

# Function to display help
show_help() {
    echo "Usage: $0 [options]..."
    echo "  -h, --help                       Show this help message."
    echo "  -b, --build-type build_type      Set the build type. Default is Release. Other options are Debug, RelWithDebInfo, and CI."
    echo "  -e, --export-compile-commands    Enable CMAKE_EXPORT_COMPILE_COMMANDS."
    echo "  -c, --enable-ccache              Enable ccache for the build."
    echo "  -a, --enable-asan                Enable AddressSanitizer."
    echo "  -m, --enable-msan                Enable MemorySanitizer."
    echo "  -s, --enable-tsan                Enable ThreadSanitizer."
    echo "  -u, --enable-ubsan               Enable UndefinedBehaviorSanitizer."
    echo "  -p, --enable-profiler            Enable Tracy profiler."
    echo "  -t, --trace                      Enable build time trace (clang only)."
    echo "  --build-tests                    Build Testcases."
    echo "  --install-prefix                 Where to install build artifacts."
    echo "  --cmake-options                  Supply additional options to cmake."
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

declare -a cmake_args

#!/bin/bash

OPTIONS=h,e,c,t,a,m,s,u,b:,p,build-tests,install-prefix:,cmake-options::
LONGOPTS=help,export_compile_commands,enable_ccache,enable_time_trace,enable_asan,enable_msan,enable_tsan,enable_ubsan,build-type:,enable_profiler,build-tests,install-prefix:,cmake-options::

# Parse the options
PARSED=$(getopt -n 'build_metal' --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
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
            show_help
            exit 0
            ;;
        -e|--export-compile-commands)
            export_compile_commands="ON";;
        -c|--enable-ccache)
            enable_ccache="ON";;
        -t|--enable-time-trace)
            enable_time_trace="ON"
            ;;
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
        --build-tests)
            build_tests="ON";;
        --install-prefix) shift; install_prefix=$1;;
        --cmake-options)
	    cmake_args+=($2);shift;;
        --)
            shift;break;;
    esac
    shift
done

build_dir="build_$build_type"


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
echo "INFO: Build Tests: $build_tests"

# Don't like this, but okay
if [ "$enable_profiler" = "ON" ]; then
    build_dir="${build_dir}_tracy"
fi

# Prepare cmake arguments
cmake_args+=("-B" "$build_dir")
cmake_args+=("-G" "Ninja")
cmake_args+=("-DCMAKE_BUILD_TYPE=$build_type")

if [ "$enable_ccache" = "ON" ]; then
    cmake_args+=("-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache")
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
    cmake_args+=("-DTT_METALLIUM_BUILD_TESTS=ON")
    cmake_args+=("-DTTNN_BUILD_TESTS=ON")
    cmake_args+=("-DTT_METAL_BUILD_TESTS=ON")
    cmake_args+=("-DTT_UMD_BUILD_TESTS=ON")
fi

install_prefix_default=$build_dir
cmake_install_prefix=${install_prefix:="${install_prefix_default}"}

cmake_args+=("-DCMAKE_INSTALL_PREFIX=$cmake_install_prefix")

echo "INFO: Build directory: $build_dir"
echo "INFO: Install Prefix: $cmake_install_prefix"

# Create and link the build directory
mkdir -p $build_dir
ln -nsf $build_dir build

# Configure cmake
echo "INFO: Configuring project"
echo "INFO: Running: cmake "${cmake_args[@]}""
cmake "${cmake_args[@]}"

# Build libraries and cpp tests
echo "INFO: Building project"
cmake --build $build_dir
cmake --build $build_dir --target install
