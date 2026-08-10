#!/usr/bin/env bash
# Build the GUI-free Tracy context inspector.
#
# Why this is fiddly (learned the hard way):
#   * Linking the prebuilt csvexport server objects (csvexport/build/unix/obj/release/*.o)
#     segfaults on load — those objects were compiled with different flags than a fresh TU,
#     so the Tracy C++ struct layouts mismatch (ODR). We therefore compile the Tracy C++
#     server sources FROM SOURCE here with consistent flags, and only reuse the C objects
#     (zstd + getopt), which are ABI-stable.
#   * TracySort.hpp needs ppqsort.h (C++20 -> std::bit_width / std::ranges). Point -I at the
#     CPM-fetched ppqsort include and build with -std=c++20.
#   * TracyWorker.cpp uses capstone v5 (CS_ARCH_AARCH64). The system libcapstone is v4, so we
#     compile against the CPM-fetched capstone v5 HEADERS but link the system .so — the disasm
#     path isn't exercised during file load, and the basic cs_* symbols exist in both.
#
# Run from the tt-metal repo root (or set TT_METAL_ROOT). Produces ./tracy_ctx_inspect.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${TT_METAL_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TRACY="$ROOT/tt_metal/third_party/tracy"
PPQ="$(dirname "$(find "$ROOT/.cpmcache/ppqsort" -name ppqsort.h | head -1)")"
CAP="$(dirname "$(dirname "$(find "$ROOT/.cpmcache/capstone" -name capstone.h | head -1)")")"
CZ="$(find "$TRACY"/csvexport/build/unix/obj/release/zstd "$TRACY"/csvexport/build/unix/obj/release/getopt -name '*.o' | tr '\n' ' ')"
SRCS="public/common/TracySocket.cpp public/common/TracyStackFrames.cpp public/common/TracySystem.cpp \
public/common/tracy_lz4.cpp public/common/tracy_lz4hc.cpp server/TracyMemory.cpp server/TracyMmap.cpp \
server/TracyPrint.cpp server/TracyTaskDispatch.cpp server/TracyTextureCompression.cpp \
server/TracyThreadCompress.cpp server/TracyWorker.cpp"
OUT="${1:-$(pwd)/tracy_ctx_inspect}"
cd "$TRACY"
# shellcheck disable=SC2086
g++ -std=c++20 -O2 -DNDEBUG -pthread -I. -I"$CAP" -I"$CAP/capstone" -I"$PPQ" \
    "$SCRIPT_DIR/tracy_ctx_inspect.cpp" $SRCS $CZ -lcapstone -ltbb -lpthread -ldl -o "$OUT"
echo "built: $OUT"
