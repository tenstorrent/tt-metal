#/bin/bash

set -euo pipefail

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

# ttml module is pure-Python that is imported directly from the source directory.
TTML_SRC_DIR="$TT_METAL_RUNTIME_ROOT/tt-train/sources/ttml"
echo "$TTML_SRC_DIR" > "$SITE_PACKAGES/ttml.pth"

# _ttml module consists of the _ttml.so library which needs rebinding
# 1. Copy _ttml.so to site-packages/ttnn.libs
TTML_BUILD_DIR="$TT_METAL_RUNTIME_ROOT/build/tt-train/sources/ttml"
TTML_SO=$(ls "$TTML_BUILD_DIR"/_ttml*.so 2>/dev/null | head -1 || true)
if [[ -z "$TTML_SO" ]]; then
    echo "ERROR: could not find _ttml*.so under $TTML_BUILD_DIR"
    exit 1
fi
TTML_SO_NAME=$(basename "$TTML_SO")
TTNN_LIBS="$SITE_PACKAGES/ttnn.libs"
cp "$TTML_SO" "$TTNN_LIBS/"

# 2. Set '$ORIGIN' of _ttml.so to ttnn.libs directory
patchelf --set-rpath '$ORIGIN' "$TTNN_LIBS/$TTML_SO_NAME"

# 3. Rebind each NEEDED reference to the hashed libraries in ttnn.libs directory
for need in $(patchelf --print-needed "$TTNN_LIBS/$TTML_SO_NAME"); do
    base=${need%%.so*}
    match=$(ls "$TTNN_LIBS/${base}"-*.so* 2>/dev/null | head -1 || true)
    if [[ -n "$match" ]]; then
        patchelf --replace-needed "$need" "$(basename "$match")" "$TTNN_LIBS/$TTML_SO_NAME"
        echo "rebound $need -> $(basename "$match")"
    fi
done

# 4. Point _ttml.pth to ttnn.libs directory
echo "$TTNN_LIBS" > "$SITE_PACKAGES/_ttml.pth"
