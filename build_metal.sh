#!/bin/bash
: '
TLDR: Follow the steps outlined below to build metal. For more in-depth information keep reading

Steps:
    1. Create python_env (you only need to do this ONCE): ./create_venv.sh
        - the env will default get created in $TT_METAL_HOME/python_env, if you want to change it set the PYTHON_ENV_DIR=
        - this step is not dependent on any of the other steps, you only need to run ONCE and in any order
    2. Configure and generate build files: `cmake -B build -G Ninja`
        - the `-B` indicates where the build folder is, you can change the folder name to whatever you want
        - the `-G Ninja` specifies to cmake to use the Ninja build system, which is faster and more reliable than make
    3. Build metal: `ninja install -C build`
        - the general command would be: `cmake --build build --target install`
        - the `-C` indicates where to run the command, in this case it will be your build folder(s)
        - we are targeting `install` since that will also just build src
        - the install target will install pybinding .so (_C.so & _ttnn.so) into the src files, so pybinds can be used
    4. Build cpp tests: `ninja tests -C build`
        - building tests will also automatically build src

Notes:
    - YOU ONLY NEED TO BUILD THE PYTHON_ENV ONCE!!!!! (unless you touch the dev python dependencies)
    - ALWAYS INSTALL, i.e just run `ninja install -C build` as the new make build
    - `cmake --build build --target install` is the EXACT same as running `ninja install -C build`, you would use the cmake command if you want to be
        agnostic of the build system (Ninja or Make)

Different configs: to change build type or use tracy, you have to change the configuration cmake step (step #2)
    - changing build types: `CONFIG=<type> cmake -B build -G Ninja`
        - valid build_types: `Release`, `Debug`, `RelWithDebInfo`
        - Release is the default if you do not set CONFIG
    - tracy: `cmake -B build -G Ninja -DENABLE_TRACY=ON`

Now you can have multiple build folders with different configs, if you want to switch just run `ninja install -C <your_build_folder` to install different pybinds
    - Caveats:
        - at least one of these folders have to be named `build`, and if using tracy config it has to be named `build`.. pending issue #8767
        - they have to be built with the original build folder name, i.e you can not change the build folder name after building bc it will mess up the RPATHs and linking

Example:
    ./create_venv.sh
    cmake -B build -G Ninja && ninja -C build                       # <- build in Release, inside folder called `build`
    CONFIG=Debug cmake -B build_debug -G Ninja && ninja -C build    # <- build in Debug, inside folder called `build_debug`
    source python_env/bin/activate                                  # <- you can not run pytests yet since pybinds have not been installed
    ninja install -C build                                          # <- install Release pybinds
    <run a pytest>                                                  # <- this test ran in Release config
    ninja install -C build_debug                                    # <- install Debug pybinds
    <run a pytest>                                                  # <- this test ran in Debug config

NOTE ON DEBUGGING!:
    GDB/LLDB is not stable right now. Recommend to use GCC11 or higher for debugging or Clang-17 with GDB 14+
'

set -eo pipefail

# Function to display help
show_help() {
    echo "Usage: $0 [-h] [-e]"
    echo "  -h  Show this help message."
    echo "  -e  Enable CMAKE_EXPORT_COMPILE_COMMANDS."
    echo "  -c  Enable ccache for the build."
}

# Parse CLI options
export_compile_commands="OFF"
enable_ccache="OFF"
while getopts "hec" opt; do
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
        \? )
            show_help
            exit 1
            ;;
    esac
done

if [ -z "$PYTHON_ENV_DIR" ]; then
    PYTHON_ENV_DIR=$(pwd)/python_env
fi

if [ -z "$CONFIG" ]; then
    echo "Build type defaulted to Release"
else
    VALID_CONFIGS="RelWithDebInfo Debug Release ci"
    if [[ $VALID_CONFIGS =~ (^|[[:space:]])"$CONFIG"($|[[:space:]]) ]]; then
        echo "CONFIG set to $CONFIG"
    else
        echo "Invalid config "$CONFIG" given.. Valid configs are: $VALID_CONFIGS"
        exit 1
    fi
fi

echo "Building tt-metal"
cmake_args="-B build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=$export_compile_commands"

if [ "$enable_ccache" = "ON" ]; then
    cmake_args="$cmake_args -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
fi

cmake $cmake_args
cmake --build build --target install    # <- this is a general cmake way, can also just run `ninja install -C build`

echo "Building cpp tests"
cmake --build build --target tests      # <- can also just run `ninja tests -C build`
