function count_loc {
    dir="$1"

    shift
    exclude_dirs="$*"
    exclude_option=""
    if [ -n "$exclude_dirs" ]; then
        exclude_option="--exclude-dir=$exclude_dirs"
    fi

    #echo "$exclude_option"
    echo -n "\t$dir \t$exclude_dirs \t"
    cloc --quiet "$dir" $exclude_option | awk '/SUM:/{print $5}'
}

# count_loc <dir> <exclude_dirs> # exclude_dirs comma separated
# code

# core
echo "Core code:"
count_loc tt_metal/build_kernels_for_riscv tests
count_loc tt_metal/common
count_loc meow_hash
count_loc tt_metal/hostdevcommon
count_loc tt_metal tests,op_library
count_loc pymetal
count_loc tt_metal/llrt tests
echo
echo

# kernels
echo "Kernels/OPs:"
count_loc kernels
count_loc tt_metal/op_library
echo
echo

# infra
echo "Infra + tools:"
count_loc git_hooks
count_loc reg_scripts
count_loc release
count_loc tools
echo
echo

# tests
echo "Tests:"
count_loc tt_metal/tests
count_loc tt_metal/build_kernels_for_riscv/tests
count_loc tt_metal/llrt/tests
count_loc tt_metal/programming_examples
count_loc python_api_testing
count_loc tensor # testing lib
echo
echo

# device
echo "Device:"
count_loc device
echo
echo

# LLK/firmware
echo "LLK/NOC/firmware for GS:"
count_loc src/ckernels/grayskull sfpi
count_loc src/firmware/riscv/common
count_loc src/firmware/riscv/grayskull
count_loc src/firmware/riscv/targets erisc
count_loc src/firmware/riscv/toolchain
echo
echo

# external libraries
# only one file so it breaks
echo "External libraries:"
count_loc tt_metal/utils/meow_hash
echo
echo

# other stuff
echo "third party:"
count_loc tt_metal/third_party
echo
echo

echo "unused LLK/FW:"
count_loc src/ckernels/wormhole
count_loc src/ckernels/wormhole_b0
count_loc src/firmware/riscv/wormhole
count_loc src/firmware/riscv/targets/erisc
echo
echo

echo "sandbox:"
count_loc sandbox
