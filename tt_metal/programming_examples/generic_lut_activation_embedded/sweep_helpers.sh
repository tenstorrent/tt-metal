#!/bin/bash
# Minimal local helpers used by run_csv.sh.

GREEN=${GREEN:-$'\033[0;32m'}
NC=${NC:-$'\033[0m'}

init_arch_detection() {
    return 0
}

parse_shape() {
    local test_shape="$1"
    shape_name="${test_shape%%:*}"
    local shape_rest="${test_shape#*:}"
    shape_rows="${shape_rest%%:*}"
    shape_cols="${shape_rest#*:}"
    tile_count=$(( (shape_rows / 32) * (shape_cols / 32) ))
}

compute_kernel_exec_min() {
    local values="$1"
    python3 - "$values" <<'PY'
import sys
vals = [float(x) for x in sys.argv[1].split(",") if x]
print(min(vals) if vals else 0)
PY
}

get_hardware_output_dir() {
    local activation="$1"
    local work_dir="$2"
    local out="$work_dir/data/hardware_outputs/$activation"
    mkdir -p "$out"
    echo "$out"
}
