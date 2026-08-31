#!/usr/bin/env bash
# Build tracy_zone_csv: dumps every device (GPU) zone of a .tracy to CSV so nesting can be checked off-line.
#
# Does NOT link build_Release's libTracyServer.a -- that archive is thin-LTO LLVM bitcode and ld rejects it
# ("file format not recognized"). Compiles the server sources directly and takes zstd from the system.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ../../.. -- this tool lives at tools/drisc_drain/tracy_zone_csv/, so the repo root is THREE levels up.
# This was ../.. after an earlier move left the relative path one level too shallow, so ROOT silently
# resolved to tools/ and every find below missed. Keep this in step with any future move.
ROOT="${TT_METAL_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TRACY="$ROOT/tt_metal/third_party/tracy"
PPQ="$(dirname "$(find "$ROOT/.cpmcache/ppqsort" -name ppqsort.h | head -1)")"
CAP="$(dirname "$(dirname "$(find "$ROOT/.cpmcache/capstone" -name capstone.h | head -1)")")"
OUT="${1:-$ROOT/build/tools/drisc_drain/tracy_zone_csv}"
mkdir -p "$(dirname "$OUT")"
cd "$TRACY"
g++ -std=c++20 -O2 -DNDEBUG -pthread -I. -I"$CAP" -I"$CAP/capstone" -I"$PPQ" \
    "$SCRIPT_DIR/tracy_zone_csv.cpp" \
    public/common/TracySocket.cpp public/common/TracyStackFrames.cpp public/common/TracySystem.cpp \
    public/common/tracy_lz4.cpp public/common/tracy_lz4hc.cpp server/TracyMemory.cpp server/TracyMmap.cpp \
    server/TracyPrint.cpp server/TracyTaskDispatch.cpp server/TracyTextureCompression.cpp \
    server/TracyThreadCompress.cpp server/TracyWorker.cpp \
    -lzstd -lcapstone -ltbb -lpthread -ldl -o "$OUT"
echo "built: $OUT"
