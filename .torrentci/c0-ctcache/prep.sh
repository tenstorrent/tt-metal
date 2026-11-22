#!/bin/bash
# C0 (torrentci#3525): the clang-tidy job's cpm-cache and configure steps,
# verbatim from .torrentci/jobs/Analysis/70-clang-tidy.yaml, so the CPM
# source cache is restored or normalised exactly as the real job does.
set -euo pipefail
cat cmake/CPM.cmake third_party/CMakeLists.txt tt_metal/third_party/CMakeLists.txt \
  tt_metal/third_party/umd/third_party/CMakeLists.txt tt-train/cmake/dependencies.cmake \
  third_party/*.patch tt-train/cmake/*.patch .gitmodules | sha256sum | cut -d' ' -f1 > /tmp/cpm-deps.sha256
if [ ! -f .cpmcache/.deps.sha256 ]; then echo "CPM cache: miss"
elif cmp -s .cpmcache/.deps.sha256 /tmp/cpm-deps.sha256; then
  echo "CPM cache: hit, $(du -sh .cpmcache | cut -f1)"; touch /tmp/cpm-unchanged
else echo "CPM cache: dependencies changed, dropping it"; rm -rf .cpmcache; fi
mkdir -p .cpmcache && cp /tmp/cpm-deps.sha256 .cpmcache/.deps.sha256
mkdir -p .build/clang-tidy/fixes
configure() {
  cmake --preset clang-tidy-fix-parallel \
    -DCMAKE_CXX_CLANG_TIDY="$PWD/cmake/clang-tidy-export-fixes-wrapper.sh;$PWD/.build/clang-tidy/fixes;--warnings-as-errors=*" \
    -DCMAKE_C_CLANG_TIDY="$PWD/cmake/clang-tidy-export-fixes-wrapper.sh;$PWD/.build/clang-tidy/fixes;--warnings-as-errors=*" \
    -DTT_USE_SYSTEM_SFPI=ON >/tmp/c0-prep-configure.log 2>&1 || { tail -50 /tmp/c0-prep-configure.log; return 1; }
}
configure
if [ -f /tmp/cpm-unchanged ]; then exit 0; fi
find .cpmcache -name .git -prune -exec rm -rf {} +
find .cpmcache -type l -exec sh -c 'for l; do t=$(readlink -f "$l") && rm "$l" && cp -rL "$t" "$l" || true; done' _ {} +
find .cpmcache -type l -delete
find .cpmcache -type f -size +0 -printf '%S %p\n' | awk '$1 < 1 {sub(/^[^ ]+ /, ""); print}' \
  | while IFS= read -r f; do cp --sparse=never "$f" "$f.dense" && mv "$f.dense" "$f"; done
rm -rf .build/clang-tidy && mkdir -p .build/clang-tidy/fixes
configure
