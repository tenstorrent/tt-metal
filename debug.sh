#!/bin/bash

# Path to the file
FILE="ttnn/cpp/ttnn/operations/experimental/bcast_to/bcast_to.cpp"

# Use awk to comment out all lines of TT_FATAL calls inside Tensor BcastTo::invoke
awk '
/Tensor BcastTo::invoke/ { in_func=1 }
in_func && /TT_FATAL\s*\(/ { in_tt_fatal=1 }
in_func && in_tt_fatal {
    sub(/^/, "// ")
    if (/\)\s*;/) in_tt_fatal=0
}
{ print }
in_func && /^}/ { in_func=0 }
' "$FILE" > "${FILE}.tmp" && mv "${FILE}.tmp" "$FILE"

cp ../test_problem.py ./tests/ttnn/unit_tests/operations/eltwise/test_problem.py

# Define file paths
FILES=(
    "tests/ttnn/unit_tests/operations/eltwise/test_binaryng_fp32.py"
    "tests/ttnn/unit_tests/operations/eltwise/test_broadcast_to.py"
    "tests/ttnn/unit_tests/operations/eltwise/test_complex.py"
)
TARGET="tests/ttnn/unit_tests/operations/eltwise/test_problem.py"

# Extract all import lines, remove duplicates, and sort
IMPORTS=$(grep -hE '^(import |from )' "${FILES[@]}" | sort -u)

# Get the rest of the target file, skipping its existing import lines
REST=$(awk '/^(import |from )/ {next} {print}' "$TARGET")

# Write the new file: imports first, then the rest
{
    echo "$IMPORTS"
    echo
    echo "$REST"
} > "${TARGET}.tmp" && mv "${TARGET}.tmp" "$TARGET"

git submodule sync; git submodule update --init  --recursive

cp ../exports.sh;

source exports.sh;

./build_metal.sh --clean
./build_metal.sh

tt-smi -r

source python_env/bin/activate;
pytest ./tests/ttnn/unit_tests/operations/eltwise/test_problem.py
