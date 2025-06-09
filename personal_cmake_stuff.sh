#!/bin/bash

EXAMPLES_DIR="/usr/share/tt-metalium/examples"


set -euo pipefail

# List of examples to build
EXAMPLES_TO_BUILD=(
    "hello_world_compute_kernel"
    "hello_world_datamovement_kernel"
    "loopback"
    "matmul_multicore_reuse"
    "add_2_integers_in_riscv"
    "hello_world_datatypes_kernel"
    "add_2_integers_in_compute"
    "eltwise_binary"
    "matmul_multi_core"
    "eltwise_sfpu"
    "matmul_single_core"
    "pad_multi_core"
    "shard_data_rm"
    "contributed/vecadd"
)

BOOST_SPAN_BROKEN_EXAMPLES=(

)

# Loop through specified examples
for example in "${EXAMPLES_TO_BUILD[@]}"; do
    dir="$EXAMPLES_DIR/$example"
    if [ -d "$dir" ]; then
        echo "Building example: $example"
        cd "$dir"
        mkdir -p build
        cd build
        cmake -G Ninja ..
        ninja
        cd "/usr/share/tt-metalium/examples"  # Go back to original directory
    else
        echo "Warning: Example directory not found: $example"
    fi
done

echo "--------------------------------"
echo "TESTING ALL EXAMPLES"
echo "--------------------------------"


for example in "${EXAMPLES_TO_BUILD[@]}"; do
    dir="$EXAMPLES_DIR/$example"
    if [ -d "$dir" ]; then
        echo "Testing example: $example"
        cd "$dir"
        cd build
        ./metal_example_${example##*/}
        echo "****************************************************"
        echo "****************************************************"
        echo ""
        echo ""
        echo ""
        echo ""
        echo ""
        echo ""
        cd "/usr/share/tt-metalium/examples"  # Go back to original directory
    else
        echo "Warning: Example directory not found: $example"
    fi
done
