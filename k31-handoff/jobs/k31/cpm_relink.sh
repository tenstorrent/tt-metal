#!/bin/bash
# cpm_relink.sh <tree>: configure-only loop. CPM keys its source cache by a hash of the package args, which carry
# this tree's absolute path, so a cache copied from another tree misses. Each round: configure with GitHub forced to
# fail at once, take the first package whose cache dir is missing, link it to the copy already in the cache, repeat.
set -u
T=$1; cd "$T" || exit 1
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.file:///nonexistent-offline/.insteadOf GIT_CONFIG_VALUE_0=https://github.com/
for round in $(seq 1 40); do
  rm -rf build_Release/CMakeCache.txt build_Release/_deps/*-subbuild 2>/dev/null
  out=$(timeout 2400 cmake -B build_Release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCPM_SOURCE_CACHE=$T/.cpmcache -DFETCHCONTENT_SOURCE_DIR_SFPI=$T/runtime/sfpi \
        -DTT_METAL_BUILD_TESTS=ON -DTTNN_BUILD_TESTS=ON -DWITH_PYTHON_BINDINGS=ON -DENABLE_DISTRIBUTED=ON \
        -DCMAKE_TOOLCHAIN_FILE=cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake 2>&1)
  rc=$?
  if [ $rc = 0 ]; then echo "configure ok after $round round(s)"; exit 0; fi
  # The last package CPM announced before the failure.
  line=$(grep -E "^-- CPM: Adding package .* to .*\.cpmcache/" <<<"$out" | tail -1)
  path=$(sed -E 's/.* to (.*)\)$/\1/' <<<"$line")
  if [ -z "$path" ] || [ -e "$path" ]; then echo "round $round: no missing cache dir to fix (rc=$rc)"; tail -30 <<<"$out"; exit 1; fi
  pkgdir=$(dirname "$path")
  src=$(find "$pkgdir" -mindepth 1 -maxdepth 1 -type d ! -name "$(basename "$path")" | head -1)
  [ -n "$src" ] || { echo "round $round: $pkgdir has no copy to link for $path"; tail -30 <<<"$out"; exit 1; }
  ln -sfn "$(basename "$src")" "$path"
  echo "round $round: linked $path -> $(basename "$src")"
done
echo "gave up after 40 rounds"; exit 1
