#/bin/bash

# Create .pth files the same way as create_venv.sh script
# Used separately when installing ttnn with tt-train from wheel
if [[ -z "${TT_METAL_RUNTIME_ROOT:-}" ]]; then
    echo "ERROR: TT_METAL_RUNTIME_ROOT is not set."
    exit 1
fi
SITE_PACKAGES=$(uv pip show ttnn | sed -n 's/^Location: //p')
if [[ -z "$SITE_PACKAGES" ]]; then
    echo "ERROR: Could not determine site-packages from ttnn. Is the wheel installed?"
    exit 1
fi
TTML_SRC_DIR="$TT_METAL_RUNTIME_ROOT/tt-train/sources/ttml"
TTML_BUILD_DIR="$TT_METAL_RUNTIME_ROOT/build/tt-train/sources/ttml"
echo "$TTML_SRC_DIR" > "$SITE_PACKAGES/ttml.pth"
echo "$TTML_BUILD_DIR" > "$SITE_PACKAGES/_ttml.pth"
