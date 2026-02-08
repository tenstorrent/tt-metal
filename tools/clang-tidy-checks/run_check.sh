#!/bin/bash
# Run the attribute protocol check on key operation files

set -euo pipefail

CHECK_SO="$(pwd)/build/libTTAttributeProtocolCheck.so"
COMPILE_DB="../../.build/clang-tidy/compile_commands.json"

if [ ! -f "$CHECK_SO" ]; then
    echo "Error: $CHECK_SO not found. Build the check first with: cmake --build build"
    exit 1
fi

if [ ! -f "$COMPILE_DB" ]; then
    echo "Error: $COMPILE_DB not found. Configure clang-tidy preset first: cmake --preset clang-tidy"
    exit 1
fi

# Key files to check
FILES=(
    "../../ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp"
    "../../ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp"
    "../../ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_device_operation.cpp"
    "../../ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.cpp"
    "../../ttnn/cpp/ttnn/operations/pool/rotate/device/rotate_device_operation.cpp"
    "../../ttnn/cpp/ttnn/operations/pool/generic/device/pool_op.cpp"
)

echo "Running attribute protocol check on ${#FILES[@]} files..."
echo ""

ERRORS=0
for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Warning: $file not found, skipping"
        continue
    fi

    echo "Checking: $file"
    if timeout 60 clang-tidy-20 \
        -load "$CHECK_SO" \
        -checks='-*,tt-attribute-protocol' \
        -p "$COMPILE_DB" \
        "$file" 2>&1 | grep -v "^$" | head -20; then
        echo "  ✓ No issues found"
    else
        ERRORS=$((ERRORS + 1))
    fi
    echo ""
done

if [ $ERRORS -eq 0 ]; then
    echo "All checks passed!"
    exit 0
else
    echo "Found issues in $ERRORS file(s)"
    exit 1
fi
