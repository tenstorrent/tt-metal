#!/bin/bash

EXAMPLES_DIR="/usr/share/tt-metalium/examples"



# List of examples to build
EXAMPLES_TO_BUILD=(
    "hello_world_compute_kernel"
    "hello_world_datamovement_kernel"
    "loopback"
    "matmul_multicore_reuse"
)

BOOST_SPAN_BROKEN_EXAMPLES=(
    "hello_world_datatypes_kernel"
    "add_2_integers_in_compute"
    "add_2_integers_in_riscv"
    "eltwise_binary"
    "matmul_multi_core"
    "eltwise_sfpu"
    "matmul_single_core"
    "pad"
    "sharding"
    "contributed/vecadd"
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