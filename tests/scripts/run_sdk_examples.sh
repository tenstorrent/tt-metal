#!/bin/bash
# Build and run SDK example binaries installed under /usr/share/<product>/examples/.
#
# Required env:
#   PRODUCT   — tt-metalium or tt-nn
# Optional env:
#   ASAN_BUILD — true to compile examples with ASan flags (default: false)

set -euo pipefail

PRODUCT="${PRODUCT:?PRODUCT must be set (tt-metalium or tt-nn)}"
ASAN_BUILD="${ASAN_BUILD:-false}"

FAILED_EXAMPLES=""

for example in "/usr/share/${PRODUCT}/examples"/*; do
  [ -e "$example" ] || continue

  example_name=$(basename "$example")

  # https://github.com/actions/toolkit/issues/1001
  echo "::group::${example_name} - build"
  cd "$(mktemp -d)"

  CMAKE_ARGS=(-G Ninja -S "$example" -B .)
  if [ "$ASAN_BUILD" = "true" ]; then
    SANITIZE_FLAGS="-fno-omit-frame-pointer -fsanitize=address,leak,undefined"
    CMAKE_ARGS+=(
      "-DCMAKE_C_FLAGS=${SANITIZE_FLAGS}"
      "-DCMAKE_CXX_FLAGS=${SANITIZE_FLAGS} -Wno-c++11-narrowing"
      "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,leak,undefined"
    )
  fi

  if ! LD_PRELOAD="" cmake "${CMAKE_ARGS[@]}" || ! LD_PRELOAD="" cmake --build .; then
    echo "::endgroup::"
    echo "::warning::${example_name} failed to compile — skipping"
    FAILED_EXAMPLES="${FAILED_EXAMPLES} ${example_name}(build)"
    continue
  fi
  echo "::endgroup::"

  exec_path=$(find . -maxdepth 2 -type f -executable -not -name "*.so")
  for exe in $exec_path; do
    exe_basename=$(basename "$exe")
    echo "::group::${exe_basename} - run"
    rc=0
    "$exe" || rc=$?
    if [ $rc -ne 0 ]; then
      echo "::error::${exe_basename} failed with exit code $rc"
      FAILED_EXAMPLES="${FAILED_EXAMPLES} ${exe_basename}(rc=$rc)"
    fi
    echo "::endgroup::"
  done
done

if [ -n "$FAILED_EXAMPLES" ]; then
  echo "::error::Failed examples:${FAILED_EXAMPLES}"
  exit 1
fi
